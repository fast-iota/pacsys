"""Tests for pacsys.cli.info -- acinfo / pacsys-info CLI tool."""

import contextlib
import io
import json
from datetime import datetime, timezone
from unittest import mock

from pacsys.digital_status import DigitalStatus, StatusBit
from pacsys.types import DeviceMeta, Reading, ValueType


def _make_reading(
    drf="M:OUTTMP",
    value=72.5,
    error_code=0,
    units="degF",
    name="M:OUTTMP",
    value_type=ValueType.SCALAR,
):
    meta = DeviceMeta(device_index=0, name=name, description="Outside temp", units=units)
    return Reading(
        drf=drf,
        value_type=value_type,
        value=value,
        error_code=error_code,
        timestamp=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        meta=meta,
    )


def _make_digital_status(device="M:OUTTMP"):
    return DigitalStatus(
        device=device,
        raw_value=0b10110011,
        bits=(
            StatusBit(position=0, name="On", value="Yes", is_set=True),
            StatusBit(position=1, name="Ready", value="Yes", is_set=True),
            StatusBit(position=2, name="Remote", value="No", is_set=False),
            StatusBit(position=3, name="Polarity", value="Minus", is_set=False),
            StatusBit(position=4, name="Ramp", value="Ramp", is_set=True),
            StatusBit(position=5, name="Bit5", value="Yes", is_set=True),
            StatusBit(position=6, name="Bit6", value="No", is_set=False),
            StatusBit(position=7, name="Bit7", value="Yes", is_set=True),
        ),
        on=True,
        ready=True,
        remote=False,
        positive=False,
        ramp=True,
    )


def _mock_device_factory(
    description="Outside temperature",
    reading=None,
    setting=None,
    analog_alarm=None,
    status=None,
    digital_status=None,
    digital_alarm=None,
    description_error=None,
):
    """Create a mock Device class that returns configured values."""
    if reading is None:
        reading = _make_reading()
    if setting is None:
        setting = _make_reading(value=72.5)
    if analog_alarm is None:
        analog_alarm = {"min": 30.0, "max": 110.0, "units": "degF"}
    if status is None:
        status = {"on": True, "ready": True, "remote": True, "positive": True, "ramp": False}
    if digital_status is None:
        digital_status = _make_digital_status()
    if digital_alarm is None:
        digital_alarm = {"nominal": 0b10110000, "mask": 0b11110000}

    class MockDevice:
        def __init__(self, name, backend=None):
            self.name = name
            self._backend = backend

        def description(self, **kw):
            if description_error:
                raise description_error
            return description

        def get(self, **kw):
            prop = kw.get("prop")
            if prop == "setting":
                return setting
            return reading

        def analog_alarm(self, **kw):
            return analog_alarm

        def status(self, **kw):
            return status

        def digital_status(self, **kw):
            return digital_status

        def digital_alarm(self, **kw):
            return digital_alarm

    return MockDevice


def _run(args, mock_device_cls, *, devdb_return=None):
    """Run acinfo main() with mocked backend and Device.

    If devdb_return is given, _get_devdb is mocked to return it.
    If None (default), _get_devdb returns None (no DevDB).
    """
    from pacsys.cli.info import main

    out = io.StringIO()
    err = io.StringIO()
    with mock.patch("pacsys.cli.info.make_backend") as mb:
        backend = mock.MagicMock()
        mb.return_value = backend
        with mock.patch("pacsys.cli.info.Device", mock_device_cls):
            with mock.patch("pacsys.cli.info._get_devdb", return_value=devdb_return):
                with mock.patch("sys.argv", ["acinfo", *args]):
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        try:
                            rc = main()
                        except SystemExit as e:
                            rc = e.code
    return rc, out.getvalue(), err.getvalue()


class TestBasicInfo:
    """Shows device name, description, and key sections."""

    def test_basic_info(self):
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory())
        assert rc == 0
        assert "M:OUTTMP" in out
        assert "Outside temperature" in out
        assert "Description" in out
        assert "Reading" in out
        assert "Setting" in out


class TestVerboseShowsBits:
    """Verbose mode shows per-bit digital status."""

    def test_verbose_shows_bits(self):
        rc, out, _ = _run(["-v", "M:OUTTMP"], _mock_device_factory())
        assert rc == 0
        assert "Bit 0" in out
        assert "Bit 1" in out
        assert "On" in out
        assert "Ready" in out


class TestCompactShowsBitfield:
    """Default mode shows binary bitfield string for digital status."""

    def test_compact_shows_bitfield(self):
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory())
        assert rc == 0
        # Bits 0-7: is_set = T,T,F,F,T,T,F,T -> MSB-first "10110011"
        assert "10110011" in out
        assert "(8 bits)" in out


class TestJsonOutput:
    """JSON output contains structured data with digital_status array."""

    def test_json_output(self):
        rc, out, _ = _run(["--format", "json", "M:OUTTMP"], _mock_device_factory())
        assert rc == 0
        data = json.loads(out.strip())
        assert data["device"] == "M:OUTTMP"
        assert data["description"] == "Outside temperature"
        assert "digital_status" in data
        assert isinstance(data["digital_status"], list)
        assert data["digital_status"][0]["bit"] == 0
        assert "label" in data["digital_status"][0]
        assert "reading" in data
        assert "setting" in data
        assert "status" in data


class TestConnectionError:
    """Connection error produces exit code 2."""

    def test_connection_error(self):
        from pacsys.cli.info import main

        out = io.StringIO()
        err = io.StringIO()
        with mock.patch("pacsys.cli.info.make_backend", side_effect=Exception("connection refused")):
            with mock.patch("sys.argv", ["acinfo", "M:OUTTMP"]):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    try:
                        rc = main()
                    except SystemExit as e:
                        rc = e.code
        assert rc == 2


class TestErrorInline:
    """When a section raises, show [ERROR] inline instead of aborting."""

    def test_error_inline(self):
        factory = _mock_device_factory(description_error=RuntimeError("device offline"))
        rc, out, _ = _run(["M:OUTTMP"], factory)
        assert rc == 1, "section error should produce exit code 1"
        assert "[ERROR]" in out
        assert "device offline" in out
        # Other sections should still appear
        assert "Reading" in out


class TestExitCodeOnError:
    """Exit code 1 when any section has an error."""

    def test_text_exit_code_on_section_error(self):
        factory = _mock_device_factory(description_error=RuntimeError("boom"))
        rc, out, _ = _run(["M:OUTTMP"], factory)
        assert rc == 1

    def test_json_exit_code_on_section_error(self):
        factory = _mock_device_factory(description_error=RuntimeError("boom"))
        rc, out, _ = _run(["--format", "json", "M:OUTTMP"], factory)
        assert rc == 1
        data = json.loads(out.strip())
        assert data["description"] == {"error": "boom"}

    def test_no_error_exit_code_zero(self):
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory())
        assert rc == 0


class TestMultipleDevices:
    """Two devices are separated by a blank line."""

    def test_multiple_devices(self):
        rc, out, _ = _run(["M:OUTTMP", "G:AMANDA"], _mock_device_factory())
        assert rc == 0
        assert "M:OUTTMP" in out
        assert "G:AMANDA" in out
        # Blank line between devices
        assert "\n\n" in out


def _make_mock_devdb(*, has_reading=True, has_setting=True, has_status=True):
    """Create a mock DevDB client that returns property availability."""
    from pacsys.devdb import DeviceInfoResult, PropertyInfo

    prop = PropertyInfo(
        primary_units="degF",
        common_units="degF",
        min_val=0.0,
        max_val=100.0,
        p_index=0,
        c_index=0,
        coeff=(1.0,),
        is_step_motor=False,
        is_destructive_read=False,
        is_fe_scaling=False,
        is_contr_setting=False,
        is_knobbable=False,
    )
    info = DeviceInfoResult(
        device_index=42,
        description="Test device",
        reading=prop if has_reading else None,
        setting=prop if has_setting else None,
        control=None,
        status_bits=(mock.MagicMock(),) if has_status else None,
        ext_status_bits=None,
    )
    devdb = mock.MagicMock()
    devdb.get_device_info.return_value = {"M:OUTTMP": info}
    return devdb


class TestDevDBPropertyFiltering:
    """DevDB-aware property filtering skips sections for unsupported properties."""

    def test_read_only_device_skips_setting(self):
        devdb = _make_mock_devdb(has_setting=False)
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory(), devdb_return=devdb)
        assert rc == 0
        assert "Reading" in out
        assert "Setting" not in out

    def test_no_status_skips_status_and_digital_status(self):
        devdb = _make_mock_devdb(has_status=False)
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory(), devdb_return=devdb)
        assert rc == 0
        assert "Status" not in out
        assert "Digital status" not in out

    def test_read_only_sensor_skips_setting_and_status(self):
        """DevDB filters reading/setting/status; alarms always attempted."""
        devdb = _make_mock_devdb(has_setting=False, has_status=False)
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory(), devdb_return=devdb)
        assert rc == 0
        assert "Description" in out
        assert "Reading" in out
        assert "Setting" not in out
        assert "Status" not in out
        assert "Digital status" not in out
        # Alarms are always attempted (DevDB can't reliably filter them)
        assert "Analog alarm" in out
        assert "Digital alarm" in out

    def test_json_omits_absent_properties(self):
        devdb = _make_mock_devdb(has_setting=False, has_status=False)
        rc, out, _ = _run(["--format", "json", "M:OUTTMP"], _mock_device_factory(), devdb_return=devdb)
        assert rc == 0
        data = json.loads(out.strip())
        assert "reading" in data
        assert "setting" not in data
        assert "status" not in data
        assert "digital_status" not in data

    def test_no_devdb_queries_everything(self):
        """Without DevDB, all sections appear (fallback to current behavior)."""
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory(), devdb_return=None)
        assert rc == 0
        assert "Reading" in out
        assert "Setting" in out
        assert "Status" in out
        assert "Digital status" in out
        assert "Analog alarm" in out
        assert "Digital alarm" in out

    def test_exit_code_zero_when_props_skipped(self):
        """Absent properties should NOT cause exit code 1."""
        devdb = _make_mock_devdb(has_setting=False, has_status=False)
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory(), devdb_return=devdb)
        assert rc == 0


def _mock_noprop_device_factory():
    """Device that raises DBM_NOPROP for setting, status, digital_status, digital_alarm."""
    from pacsys.errors import DeviceError

    reading = _make_reading()
    analog_alarm = {"min": 30.0, "max": 110.0, "units": "degF"}

    class MockDevice:
        def __init__(self, name, backend=None):
            self.name = name

        def description(self, **kw):
            return "Outdoor temperature"

        def get(self, **kw):
            if kw.get("prop") == "setting":
                raise DeviceError("M:OUTTMP.SETTING@I", 16, -13, "DBM_NOPROP")
            return reading

        def analog_alarm(self, **kw):
            return analog_alarm

        def status(self, **kw):
            raise DeviceError("M:OUTTMP.STATUS@I", 16, -13, "DBM_NOPROP")

        def digital_status(self, **kw):
            raise DeviceError("M:OUTTMP.STATUS.BIT_VALUE@I", 16, -13, "DBM_NOPROP")

        def digital_alarm(self, **kw):
            raise DeviceError("M:OUTTMP.DIGITAL@I", 16, -13, "DBM_NOPROP")

    return MockDevice


class TestNoPropSuppression:
    """DBM_NOPROP errors are hidden in non-verbose mode, shown in verbose mode."""

    def test_noprop_hidden_without_verbose(self):
        rc, out, _ = _run(["M:OUTTMP"], _mock_noprop_device_factory())
        assert rc == 0
        assert "Setting" not in out
        assert "Status" not in out
        assert "Digital status" not in out
        assert "Digital alarm" not in out
        assert "DBM_NOPROP" not in out
        # Supported sections still appear
        assert "Reading" in out
        assert "Analog alarm" in out

    def test_noprop_shown_with_verbose(self):
        rc, out, _ = _run(["-v", "M:OUTTMP"], _mock_noprop_device_factory())
        assert rc == 0
        assert "Setting" in out
        assert "Status" in out
        assert "Digital status" in out
        assert "Digital alarm" in out
        assert "DBM_NOPROP" in out

    def test_noprop_exit_code_zero(self):
        """DBM_NOPROP should not cause error exit code."""
        rc, _, _ = _run(["M:OUTTMP"], _mock_noprop_device_factory())
        assert rc == 0

    def test_noprop_json_omits_keys(self):
        """JSON output omits keys for DBM_NOPROP properties."""
        rc, out, _ = _run(["--format", "json", "M:OUTTMP"], _mock_noprop_device_factory())
        assert rc == 0
        data = json.loads(out.strip())
        assert "reading" in data
        assert "analog_alarm" in data
        assert "setting" not in data
        assert "status" not in data
        assert "digital_status" not in data
        assert "digital_alarm" not in data

    def test_real_error_still_shown(self):
        """Non-NOPROP errors still show in non-verbose mode."""
        factory = _mock_device_factory(description_error=RuntimeError("device offline"))
        rc, out, _ = _run(["M:OUTTMP"], factory)
        assert rc == 1
        assert "[ERROR]" in out
        assert "device offline" in out


class TestDeviceIndex:
    """Device index is shown on the name line."""

    def test_di_from_devdb(self):
        devdb = _make_mock_devdb()
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory(), devdb_return=devdb)
        assert rc == 0
        assert "M:OUTTMP (di=42)" in out

    def test_di_from_backend_reading(self):
        """When DevDB is unavailable, di comes from backend reading metadata."""
        rc, out, _ = _run(["M:OUTTMP"], _mock_device_factory(), devdb_return=None)
        # _make_reading sets device_index=0 in meta, so no di shown (0 is falsy)
        assert "di=" not in out

        # Now with a real device_index in meta
        reading = _make_reading()
        meta_with_di = reading.meta.__class__(
            device_index=12345, name="M:OUTTMP", description="Outside temp", units="degF"
        )
        reading_with_di = reading.__class__(
            drf=reading.drf,
            value_type=reading.value_type,
            value=reading.value,
            error_code=reading.error_code,
            timestamp=reading.timestamp,
            meta=meta_with_di,
        )
        factory = _mock_device_factory(reading=reading_with_di)
        rc, out, _ = _run(["M:OUTTMP"], factory, devdb_return=None)
        assert rc == 0
        assert "M:OUTTMP (di=12345)" in out

    def test_di_in_json_from_devdb(self):
        devdb = _make_mock_devdb()
        rc, out, _ = _run(["--format", "json", "M:OUTTMP"], _mock_device_factory(), devdb_return=devdb)
        data = json.loads(out.strip())
        assert data["device_index"] == 42

    def test_no_di_without_devdb_or_meta(self):
        """No di key in JSON when neither DevDB nor meta has it."""
        rc, out, _ = _run(["--format", "json", "M:OUTTMP"], _mock_device_factory(), devdb_return=None)
        data = json.loads(out.strip())
        assert "device_index" not in data
