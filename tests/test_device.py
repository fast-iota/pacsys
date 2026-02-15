"""
Tests for Device API.

Tests cover:
- Device creation with valid/invalid DRF strings
- Property accessors (drf, name, has_event, is_periodic)
- Immutability (with_event, with_range, with_backend return new Device)
- Typed device subclasses (ScalarDevice, ArrayDevice, TextDevice)
- Read/get delegation to backend
- Property-specific read methods (setting, status, analog_alarm, etc.)
- Write methods (write, control, on/off/reset shortcuts)
- Alarm setters (set_analog_alarm, set_digital_alarm)
- Verify flow (write+verify, check_first, control+verify)
- Field validation
"""

import pytest
from unittest import mock
import numpy as np

from pacsys.device import Device, ScalarDevice, ArrayDevice, TextDevice
from pacsys.errors import DeviceError
from pacsys.testing import FakeBackend
from pacsys.types import Reading, ValueType, WriteResult, BasicControl
from pacsys.verify import Verify


# Fixtures


@pytest.fixture
def fake():
    """FakeBackend with DRF validation - catches malformed DRF strings."""
    return FakeBackend()


@pytest.fixture
def mock_backend():
    """MagicMock backend - only for tests needing side_effect sequences."""
    backend = mock.MagicMock()
    backend.read.return_value = 72.5
    backend.get.return_value = Reading(
        drf="M:OUTTMP",
        value_type=ValueType.SCALAR,
        value=72.5,
        error_code=0,
    )
    backend.write.return_value = WriteResult(drf="M:OUTTMP.SETTING@N", error_code=0)
    return backend


# Device Creation Tests


class TestDeviceCreation:
    """Tests for Device creation and DRF validation."""

    def test_create_invalid_drf_raises_valueerror(self):
        with pytest.raises(ValueError):
            Device("X")  # Too short

    def test_create_device_with_invalid_event_raises_valueerror(self):
        with pytest.raises(ValueError):
            Device("M:OUTTMP@Z")  # Invalid event type


# Immutability Tests


class TestDeviceImmutability:
    """Tests for Device immutability - modification methods return new instances."""

    def test_with_event_returns_new_device(self):
        original = Device("M:OUTTMP")
        modified = original.with_event("p,1000")
        assert modified.is_periodic
        assert modified.has_event
        assert not original.has_event
        assert not original.is_periodic
        assert original is not modified

    def test_with_range_returns_new_device(self):
        original = Device("B:HS23T")
        modified = original.with_range(0, 10)
        assert modified.request.range is not None
        assert modified.request.range.low == 0
        assert modified.request.range.high == 10
        assert original.request.range is None
        assert original is not modified

    def test_with_backend_returns_new_device(self, fake):
        original = Device("M:OUTTMP")
        modified = original.with_backend(fake)
        assert modified._backend is fake
        assert original._backend is None
        assert original is not modified

    def test_chained_modifications(self, fake):
        dev = Device("M:OUTTMP").with_event("p,1000").with_backend(fake)
        assert dev.is_periodic
        assert dev._backend is fake

    def test_with_event_replaces_existing_event(self):
        dev = Device("M:OUTTMP@p,1000")
        modified = dev.with_event("E,0F")
        assert not modified.is_periodic
        assert "@E,0F" in modified.drf.upper() or "E,0F" in modified.drf


# Read/Get Delegation Tests


class TestDeviceReadOperations:
    """Tests for Device read and get operations - uses FakeBackend for DRF validation."""

    def test_read_delegates_to_backend(self, fake):
        fake.set_reading("M:OUTTMP.READING", 72.5)
        dev = Device("M:OUTTMP", backend=fake)
        result = dev.read()
        assert result == 72.5
        assert len(fake.reads) == 1

    def test_get_delegates_to_backend(self, fake):
        fake.set_reading("M:OUTTMP.READING", 72.5)
        dev = Device("M:OUTTMP", backend=fake)
        result = dev.get()
        assert result.value == 72.5
        assert len(fake.reads) == 1

    def test_get_with_prop_builds_drf(self, fake):
        fake.set_reading("M:OUTTMP.SETTING", 72.5)
        dev = Device("M:OUTTMP", backend=fake)
        dev.get(prop="setting")
        drf = fake.reads[-1]
        assert ".SETTING" in drf
        assert "@I" in drf

    def test_get_with_prop_and_field(self, fake):
        fake.set_reading("M:OUTTMP.STATUS.ON", True)
        dev = Device("M:OUTTMP", backend=fake)
        dev.get(prop="status", field="on")
        drf = fake.reads[-1]
        assert ".STATUS" in drf
        assert ".ON" in drf
        assert "@I" in drf

    def test_get_with_field_but_no_prop_raises(self, fake):
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(ValueError, match="field requires prop"):
            dev.get(field="raw")

    def test_get_with_prop_preserves_range(self, fake):
        fake.set_reading("M:OUTTMP.SETTING", list(range(20)), value_type=ValueType.SCALAR_ARRAY)
        dev = Device("M:OUTTMP[0:10]", backend=fake)
        dev.get(prop="setting")
        drf = fake.reads[-1]
        assert "[0:10]" in drf
        assert ".SETTING" in drf

    def test_get_with_invalid_prop_raises(self, fake):
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(KeyError):
            dev.get(prop="nonexistent")

    def test_read_with_timeout(self, fake):
        fake.set_reading("M:OUTTMP.READING", 72.5)
        dev = Device("M:OUTTMP", backend=fake)
        result = dev.read(timeout=5.0)
        assert result == 72.5
        assert "M:OUTTMP" in fake.reads[-1]

    def test_read_uses_reading_property_and_immediate_event(self, fake):
        fake.set_reading("M:OUTTMP.READING", 72.5)
        dev = Device("M:OUTTMP", backend=fake)
        dev.read()
        called_drf = fake.reads[-1]
        assert ".READING" in called_drf
        assert "@I" in called_drf


# ScalarDevice Tests


class TestScalarDevice:
    """Tests for ScalarDevice."""

    def test_scalar_device_returns_float(self, fake):
        fake.set_reading("M:OUTTMP.READING", 72.5)
        dev = ScalarDevice("M:OUTTMP", backend=fake)
        result = dev.read()
        assert isinstance(result, float)
        assert result == 72.5

    def test_scalar_device_converts_int(self, fake):
        fake.set_reading("M:OUTTMP.READING", 42)
        dev = ScalarDevice("M:OUTTMP", backend=fake)
        result = dev.read()
        assert isinstance(result, float)
        assert result == 42.0

    @pytest.mark.parametrize(
        "bad_value,vtype",
        [
            (np.array([1, 2, 3]), ValueType.SCALAR_ARRAY),
            ("text", ValueType.TEXT),
        ],
    )
    def test_scalar_device_raises_on_wrong_type(self, fake, bad_value, vtype):
        fake.set_reading("M:OUTTMP.READING", bad_value, value_type=vtype)
        dev = ScalarDevice("M:OUTTMP", backend=fake)
        with pytest.raises(TypeError, match="Expected scalar"):
            dev.read()


# ArrayDevice Tests


class TestArrayDevice:
    """Tests for ArrayDevice."""

    def test_array_device_returns_ndarray(self, fake):
        fake.set_reading("B:HS23T.READING", np.array([1.0, 2.0, 3.0]), value_type=ValueType.SCALAR_ARRAY)
        dev = ArrayDevice("B:HS23T[0:10]", backend=fake)
        result = dev.read()
        assert isinstance(result, np.ndarray)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_array_device_converts_list(self, fake):
        fake.set_reading("B:HS23T.READING", [1.0, 2.0, 3.0], value_type=ValueType.SCALAR_ARRAY)
        dev = ArrayDevice("B:HS23T[0:10]", backend=fake)
        result = dev.read()
        assert isinstance(result, np.ndarray)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_array_device_converts_tuple(self, fake):
        fake.set_reading("B:HS23T.READING", (1.0, 2.0, 3.0), value_type=ValueType.SCALAR_ARRAY)
        dev = ArrayDevice("B:HS23T[0:10]", backend=fake)
        result = dev.read()
        assert isinstance(result, np.ndarray)
        assert list(result) == [1.0, 2.0, 3.0]

    @pytest.mark.parametrize(
        "bad_value,vtype",
        [
            (72.5, ValueType.SCALAR),
            ("text", ValueType.TEXT),
        ],
    )
    def test_array_device_raises_on_wrong_type(self, fake, bad_value, vtype):
        fake.set_reading("B:HS23T.READING", bad_value, value_type=vtype)
        dev = ArrayDevice("B:HS23T", backend=fake)  # No range - tests type check, not range
        with pytest.raises(TypeError, match="Expected array"):
            dev.read()


# TextDevice Tests


class TestTextDevice:
    """Tests for TextDevice."""

    def test_text_device_returns_string(self, fake):
        fake.set_reading("M:OUTTMP.READING", "some text", value_type=ValueType.TEXT)
        dev = TextDevice("M:OUTTMP", backend=fake)
        result = dev.read()
        assert isinstance(result, str)
        assert result == "some text"

    @pytest.mark.parametrize(
        "bad_value,vtype",
        [
            (72.5, ValueType.SCALAR),
            (np.array([1, 2, 3]), ValueType.SCALAR_ARRAY),
        ],
    )
    def test_text_device_raises_on_wrong_type(self, fake, bad_value, vtype):
        fake.set_reading("M:OUTTMP.READING", bad_value, value_type=vtype)
        dev = TextDevice("M:OUTTMP", backend=fake)
        with pytest.raises(TypeError, match="Expected string"):
            dev.read()


# Subclass Preservation Tests


class TestSubclassPreservation:
    """Tests that modification methods preserve the Device subclass."""

    def test_scalar_device_with_event_returns_scalar_device(self):
        dev = ScalarDevice("M:OUTTMP")
        modified = dev.with_event("p,1000")
        assert isinstance(modified, ScalarDevice)

    def test_scalar_device_with_backend_returns_scalar_device(self, fake):
        dev = ScalarDevice("M:OUTTMP")
        modified = dev.with_backend(fake)
        assert isinstance(modified, ScalarDevice)

    def test_array_device_with_range_returns_array_device(self):
        dev = ArrayDevice("B:HS23T")
        modified = dev.with_range(0, 10)
        assert isinstance(modified, ArrayDevice)


# Edge Cases


class TestEdgeCases:
    """Tests for edge cases."""

    def test_with_range_single_element(self):
        dev = Device("B:HS23T")
        modified = dev.with_range(5)
        assert modified.request.range is not None
        assert modified.request.range.low == 5

    def test_device_with_lowercase_drf(self):
        dev = Device("m:outtmp")
        assert dev.name.upper() == "M:OUTTMP"


# ─── New tests for property-specific reads ─────────────────────────────


class TestDeviceReadMethods:
    """Tests for property-specific read methods - FakeBackend validates all DRFs."""

    def test_read_default_field(self, fake):
        fake.set_reading("M:OUTTMP.READING", 72.5)
        dev = Device("M:OUTTMP", backend=fake)
        dev.read()
        drf = fake.reads[-1]
        assert ".READING" in drf
        assert "@I" in drf

    def test_read_raw_field(self, fake):
        fake.set_reading("M:OUTTMP.READING.RAW", b"\x00\x01", value_type=ValueType.RAW)
        dev = Device("M:OUTTMP", backend=fake)
        dev.read(field="raw")
        drf = fake.reads[-1]
        assert ".READING" in drf
        assert ".RAW" in drf
        assert "@I" in drf

    def test_setting_default(self, fake):
        fake.set_reading("M:OUTTMP.SETTING", 72.5)
        dev = Device("M:OUTTMP", backend=fake)
        dev.setting()
        drf = fake.reads[-1]
        assert ".SETTING" in drf
        assert "@I" in drf

    def test_setting_raw_field(self, fake):
        fake.set_reading("M:OUTTMP.SETTING.RAW", b"\x00\x01", value_type=ValueType.RAW)
        dev = Device("M:OUTTMP", backend=fake)
        dev.setting(field="raw")
        drf = fake.reads[-1]
        assert ".SETTING" in drf
        assert ".RAW" in drf

    def test_status_default(self, fake):
        fake.set_reading("M:OUTTMP.STATUS", 0)
        dev = Device("M:OUTTMP", backend=fake)
        dev.status()
        drf = fake.reads[-1]
        assert ".STATUS" in drf
        assert "@I" in drf

    def test_status_on_field(self, fake):
        fake.set_reading("M:OUTTMP.STATUS.ON", True)
        dev = Device("M:OUTTMP", backend=fake)
        dev.status(field="on")
        drf = fake.reads[-1]
        assert ".STATUS" in drf
        assert ".ON" in drf

    def test_analog_alarm_default(self, fake):
        fake.set_reading("M:OUTTMP.ANALOG", {"minimum": 0.0, "maximum": 100.0}, value_type=ValueType.ANALOG_ALARM)
        dev = Device("M:OUTTMP", backend=fake)
        dev.analog_alarm()
        drf = fake.reads[-1]
        assert ".ANALOG" in drf
        assert "@I" in drf

    def test_analog_alarm_min_field(self, fake):
        fake.set_reading("M:OUTTMP.ANALOG.MIN", 0.0)
        dev = Device("M:OUTTMP", backend=fake)
        dev.analog_alarm(field="min")
        drf = fake.reads[-1]
        assert ".ANALOG" in drf
        assert ".MIN" in drf

    def test_digital_alarm_default(self, fake):
        fake.set_reading("M:OUTTMP.DIGITAL", {"nominal": 0, "mask": 0xFF}, value_type=ValueType.DIGITAL_ALARM)
        dev = Device("M:OUTTMP", backend=fake)
        dev.digital_alarm()
        drf = fake.reads[-1]
        assert ".DIGITAL" in drf
        assert "@I" in drf

    def test_description(self, fake):
        fake.set_reading("M:OUTTMP.DESCRIPTION", "Outside temperature", value_type=ValueType.TEXT)
        dev = Device("M:OUTTMP", backend=fake)
        result = dev.description()
        assert result == "Outside temperature"
        drf = fake.reads[-1]
        assert ".DESCRIPTION" in drf

    def test_read_preserves_range(self, fake):
        fake.set_reading("B:HS23T.READING", list(range(20)), value_type=ValueType.SCALAR_ARRAY)
        dev = Device("B:HS23T[0:10]", backend=fake)
        dev.read()
        drf = fake.reads[-1]
        assert "[0:10]" in drf
        assert ".READING" in drf

    def test_setting_preserves_range(self, fake):
        fake.set_reading("B:HS23T.SETTING", list(range(20)), value_type=ValueType.SCALAR_ARRAY)
        dev = Device("B:HS23T[0:10]", backend=fake)
        dev.setting()
        drf = fake.reads[-1]
        assert "[0:10]" in drf
        assert ".SETTING" in drf


class TestDeviceFieldValidation:
    """Tests that invalid fields raise ValueError (before backend call)."""

    def test_invalid_field_for_reading(self, fake):
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(ValueError, match="not allowed"):
            dev.read(field="on")  # ON is a STATUS field, not READING

    def test_invalid_field_for_status(self, fake):
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(ValueError, match="not allowed"):
            dev.status(field="primary")  # PRIMARY is a READING field, not STATUS

    def test_invalid_field_for_setting(self, fake):
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(ValueError, match="not allowed"):
            dev.setting(field="on")

    def test_unknown_field_name(self, fake):
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(ValueError):
            dev.read(field="nonexistent")


# ─── Write method tests ────────────────────────────────────────────────


class TestDeviceWriteMethods:
    """Tests for Device write methods - FakeBackend validates DRF and records writes."""

    def test_write_basic(self, fake):
        fake.set_reading("M:OUTTMP.SETTING", 0.0)
        dev = Device("M:OUTTMP", backend=fake)
        result = dev.write(72.5)
        assert result.success
        drf, value = fake.writes[-1]
        assert ".SETTING" in drf
        assert "@N" in drf
        assert value == 72.5

    def test_write_raw_field(self, fake):
        fake.set_reading("M:OUTTMP.SETTING.RAW", b"\x00", value_type=ValueType.RAW)
        dev = Device("M:OUTTMP", backend=fake)
        dev.write(100, field="raw")
        drf, _ = fake.writes[-1]
        assert ".SETTING" in drf
        assert ".RAW" in drf
        assert "@N" in drf

    def test_control_on(self, fake):
        dev = Device("Z:ACLTST", backend=fake)
        dev.control(BasicControl.ON)
        drf, value = fake.writes[-1]
        assert ".CONTROL" in drf
        assert "@N" in drf
        assert value == BasicControl.ON

    @pytest.mark.parametrize(
        "method,expected",
        [
            ("on", BasicControl.ON),
            ("off", BasicControl.OFF),
            ("reset", BasicControl.RESET),
            ("positive", BasicControl.POSITIVE),
            ("negative", BasicControl.NEGATIVE),
            ("ramp", BasicControl.RAMP),
            ("dc", BasicControl.DC),
            ("local", BasicControl.LOCAL),
            ("remote", BasicControl.REMOTE),
            ("trip", BasicControl.TRIP),
        ],
    )
    def test_control_shortcut(self, fake, method, expected):
        """Control shortcuts delegate to control() with correct command."""
        dev = Device("Z:ACLTST", backend=fake)
        getattr(dev, method)()
        assert fake.writes[-1][1] == expected

    def test_write_failed_returns_result(self, fake):
        fake.set_write_result("M:OUTTMP.SETTING", success=False, error_code=-1, message="Fail")
        dev = Device("M:OUTTMP", backend=fake)
        result = dev.write(72.5)
        assert not result.success
        assert result.error_code == -1


# ─── Alarm setter tests ────────────────────────────────────────────────


class TestDeviceAlarmSetters:
    """Tests for alarm setter methods."""

    def test_set_analog_alarm(self, fake):
        dev = Device("M:OUTTMP", backend=fake)
        settings = {"minimum": 40, "maximum": 80}
        dev.set_analog_alarm(settings)
        drf, value = fake.writes[-1]
        assert ".ANALOG" in drf
        assert "@N" in drf
        assert value == settings

    def test_set_digital_alarm(self, fake):
        dev = Device("M:OUTTMP", backend=fake)
        settings = {"nominal": 0x01, "mask": 0xFF}
        dev.set_digital_alarm(settings)
        drf, value = fake.writes[-1]
        assert ".DIGITAL" in drf
        assert "@N" in drf
        assert value == settings


# ─── Verify flow tests ─────────────────────────────────────────────────
# These tests need side_effect sequences and assert_not_called, so they
# use MagicMock - they test orchestration logic, not DRF construction.


class TestDeviceVerify:
    """Tests for write-and-verify flow."""

    def test_write_with_verify_true_reads_back(self, mock_backend):
        """verify=True triggers readback after write."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, retry_delay=0))
        assert result.verified is True
        assert result.readback == 72.5
        assert result.attempts == 1

    def test_write_verify_mismatch(self, mock_backend):
        """Verify fails when readback doesn't match."""
        mock_backend.read.return_value = 99.0
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, retry_delay=0, max_attempts=2))
        assert result.verified is False
        assert result.readback == 99.0
        assert result.attempts == 2

    def test_write_verify_with_tolerance(self, mock_backend):
        """Verify passes within tolerance."""
        mock_backend.read.return_value = 72.6
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, tolerance=0.2))
        assert result.verified is True

    def test_write_check_first_skip(self, mock_backend):
        """check_first skips write when value already matches."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(check_first=True, initial_delay=0))
        assert result.skipped is True
        assert result.verified is True
        mock_backend.write.assert_not_called()

    def test_write_check_first_proceeds_on_mismatch(self, mock_backend):
        """check_first proceeds to write when value differs."""
        # First read returns current (different), then readback returns written value
        mock_backend.read.side_effect = [50.0, 72.5]
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(check_first=True, initial_delay=0, retry_delay=0))
        mock_backend.write.assert_called_once()
        assert result.verified is True
        assert not result.skipped

    def test_write_no_verify_returns_plain_result(self, fake):
        """Without verify, no readback occurs."""
        dev = Device("M:OUTTMP", backend=fake)
        result = dev.write(72.5)
        assert result.verified is None
        assert result.readback is None
        assert result.skipped is False
        assert len(fake.reads) == 0

    def test_write_verify_false_no_readback(self, fake):
        """verify=False explicitly disables verification."""
        dev = Device("M:OUTTMP", backend=fake)
        result = dev.write(72.5, verify=False)
        assert result.verified is None
        assert len(fake.reads) == 0

    def test_control_with_verify(self, mock_backend):
        """Control commands verify via STATUS read."""
        mock_backend.read.return_value = True
        dev = Device("Z:ACLTST", backend=mock_backend)
        result = dev.on(verify=Verify(initial_delay=0, retry_delay=0))
        assert result.verified is True
        # Should have read STATUS.ON
        read_drf = mock_backend.read.call_args[0][0]
        assert ".STATUS" in read_drf
        assert ".ON" in read_drf

    def test_control_verify_check_first_skip(self, mock_backend):
        """Control check_first skips if status already matches."""
        mock_backend.read.return_value = True  # already on
        dev = Device("Z:ACLTST", backend=mock_backend)
        result = dev.on(verify=Verify(check_first=True, initial_delay=0))
        assert result.skipped is True
        mock_backend.write.assert_not_called()

    def test_control_verify_unmapped_command_raises(self, fake):
        """Verify raises ValueError for commands without a STATUS field mapping."""
        dev = Device("Z:ACLTST", backend=fake)
        with pytest.raises(ValueError, match="no STATUS field mapping"):
            dev.control(99, verify=True)

    def test_control_no_verify_unmapped_command_succeeds(self, fake):
        """Unmapped commands work fine without verify."""
        dev = Device("Z:ACLTST", backend=fake)
        result = dev.control(99)
        assert result.verified is None

    def test_write_failed_skips_verify(self, fake):
        """If the write itself fails, no verify is attempted."""
        fake.set_write_result("M:OUTTMP.SETTING", success=False, error_code=-1, message="Fail")
        dev = Device("M:OUTTMP", backend=fake)
        result = dev.write(72.5, verify=Verify(initial_delay=0))
        assert result.error_code == -1
        assert result.verified is None
        assert len(fake.reads) == 0

    def test_verify_context_always(self, mock_backend):
        """Verify context with always=True auto-verifies."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP", backend=mock_backend)
        with Verify(always=True, initial_delay=0, retry_delay=0):
            result = dev.write(72.5)
        assert result.verified is True

    def test_write_with_custom_readback_drf(self, mock_backend):
        """Verify uses custom readback DRF when specified."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(readback="M:OUTTMP.READING@I", initial_delay=0))
        read_drf = mock_backend.read.call_args[0][0]
        assert read_drf == "M:OUTTMP.READING@I"
        assert result.verified is True


class TestDeviceExtra:
    """Tests for DRF extra modifier preservation (e.g., <-FTP)."""

    def test_build_drf_preserves_ftp_extra_on_read(self, fake):
        """read() preserves <-FTP extra in the DRF passed to backend."""
        fake.set_reading("M:OUTTMP.READING<-FTP", 72.5)
        dev = Device("M:OUTTMP<-FTP", backend=fake)
        dev.read()
        drf = fake.reads[-1]
        assert drf.endswith("<-FTP")
        assert drf == "M:OUTTMP.READING@I<-FTP"

    def test_build_drf_preserves_ftp_extra_on_write(self, fake):
        """write() preserves <-FTP extra in the DRF passed to backend."""
        dev = Device("M:OUTTMP<-FTP", backend=fake)
        dev.write(72.5)
        drf = fake.writes[-1][0]
        assert drf.endswith("<-FTP")
        assert drf == "M:OUTTMP.SETTING@N<-FTP"

    def test_build_drf_preserves_ftp_extra_on_status(self, fake):
        """status() preserves <-FTP extra."""
        fake.set_reading("M:OUTTMP.STATUS.ON<-FTP", True)
        dev = Device("M:OUTTMP<-FTP", backend=fake)
        dev.status(field="on")
        drf = fake.reads[-1]
        assert drf == "M:OUTTMP.STATUS.ON@I<-FTP"

    def test_digital_status_preserves_ftp_extra(self, mock_backend):
        """digital_status() preserves <-FTP extra in all three DRFs."""
        from pacsys.types import Reading, ValueType

        mock_backend.get_many.return_value = [
            Reading(drf="M:OUTTMP.STATUS.BIT_VALUE@I<-FTP", value=0x02, value_type=ValueType.SCALAR),
            Reading(drf="M:OUTTMP.STATUS.BIT_NAMES@I<-FTP", value=["On", "Ready"], value_type=ValueType.TEXT_ARRAY),
            Reading(drf="M:OUTTMP.STATUS.BIT_VALUES@I<-FTP", value=["No", "Yes"], value_type=ValueType.TEXT_ARRAY),
        ]
        dev = Device("M:OUTTMP<-FTP", backend=mock_backend)
        dev.digital_status()
        drfs = mock_backend.get_many.call_args[0][0]
        for drf in drfs:
            assert drf.endswith("<-FTP"), f"{drf} missing <-FTP"

    def test_no_extra_omits_suffix(self, fake):
        """Devices without extra don't get a spurious suffix."""
        fake.set_reading("M:OUTTMP.READING", 72.5)
        dev = Device("M:OUTTMP", backend=fake)
        dev.read()
        drf = fake.reads[-1]
        assert "<-" not in drf


class TestVerifyReadbackError:
    """Tests for verify readback when DeviceError occurs."""

    def test_readback_device_error_returns_verified_false(self, mock_backend):
        """If readback raises DeviceError, result is verified=False, not exception."""
        mock_backend.read.side_effect = DeviceError("M:OUTTMP", 0, -6, "Timeout")
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, retry_delay=0, max_attempts=2))
        assert result.verified is False
        assert result.readback is None
        assert result.attempts == 2

    def test_readback_error_then_success(self, mock_backend):
        """If first readback fails but second succeeds, result is verified=True."""
        mock_backend.read.side_effect = [
            DeviceError("M:OUTTMP", 0, -6, "Timeout"),
            72.5,
        ]
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, retry_delay=0, max_attempts=3))
        assert result.verified is True
        assert result.readback == 72.5
        assert result.attempts == 2

    def test_control_readback_error_returns_verified_false(self, mock_backend):
        """Control verify handles DeviceError during STATUS readback."""
        mock_backend.read.side_effect = DeviceError("Z:ACLTST", 0, -6, "Timeout")
        dev = Device("Z:ACLTST", backend=mock_backend)
        result = dev.on(verify=Verify(initial_delay=0, retry_delay=0, max_attempts=1))
        assert result.verified is False


# ─── Subscribe tests ─────────────────────────────────────────────────


class TestDeviceSubscribe:
    """Tests for Device.subscribe() streaming method."""

    def test_subscribe_with_explicit_event(self, fake):
        """subscribe(event=...) builds correct DRF and delegates to backend."""
        dev = Device("M:OUTTMP", backend=fake)
        handle = dev.subscribe(event="p,1000")
        assert not handle.stopped
        handle.stop()

    def test_subscribe_drf_has_event(self, fake):
        """subscribe() passes the event string (not 'I') in the DRF."""
        dev = Device("M:OUTTMP", backend=fake)
        handle = dev.subscribe(event="p,1000")
        # FakeSubscriptionHandle stores the DRFs it was created with
        assert len(handle._drfs) == 1
        drf = next(iter(handle._drfs))
        assert "@p,1000" in drf
        assert ".READING" in drf
        handle.stop()

    def test_subscribe_with_device_event(self, fake):
        """subscribe() uses device's event when no event kwarg given."""
        dev = Device("M:OUTTMP@p,1000", backend=fake)
        handle = dev.subscribe()
        drf = next(iter(handle._drfs))
        assert "@p,1000" in drf
        handle.stop()

    def test_subscribe_with_event_from_with_event(self, fake):
        """subscribe() uses event from with_event()."""
        dev = Device("M:OUTTMP", backend=fake).with_event("p,500").with_backend(fake)
        handle = dev.subscribe()
        drf = next(iter(handle._drfs))
        assert "@p,500" in drf
        handle.stop()

    def test_subscribe_no_event_raises(self, fake):
        """subscribe() raises ValueError when no event available."""
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(ValueError, match="subscribe requires an event"):
            dev.subscribe()

    def test_subscribe_field_without_prop_raises(self, fake):
        """subscribe(field=...) without prop raises ValueError."""
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(ValueError, match="field requires prop"):
            dev.subscribe(field="raw", event="p,1000")

    def test_subscribe_with_prop(self, fake):
        """subscribe(prop='setting') builds DRF with .SETTING."""
        dev = Device("M:OUTTMP", backend=fake)
        handle = dev.subscribe(prop="setting", event="p,1000")
        drf = next(iter(handle._drfs))
        assert ".SETTING" in drf
        assert "@p,1000" in drf
        handle.stop()

    def test_subscribe_with_prop_and_field(self, fake):
        """subscribe(prop='status', field='on') builds correct DRF."""
        dev = Device("M:OUTTMP", backend=fake)
        handle = dev.subscribe(prop="status", field="on", event="p,1000")
        drf = next(iter(handle._drfs))
        assert ".STATUS" in drf
        assert ".ON" in drf
        assert "@p,1000" in drf
        handle.stop()

    def test_subscribe_event_kwarg_overrides_device_event(self, fake):
        """Explicit event= overrides the device's existing event."""
        dev = Device("M:OUTTMP@p,1000", backend=fake)
        handle = dev.subscribe(event="E,0F")
        drf = next(iter(handle._drfs))
        assert "@E,0F" in drf
        assert "p,1000" not in drf
        handle.stop()

    def test_subscribe_preserves_range(self, fake):
        """subscribe() preserves array range in DRF."""
        dev = Device("B:HS23T[0:10]", backend=fake)
        handle = dev.subscribe(event="p,1000")
        drf = next(iter(handle._drfs))
        assert "[0:10]" in drf
        handle.stop()

    def test_subscribe_preserves_extra(self, fake):
        """subscribe() preserves <-FTP extra modifier."""
        dev = Device("M:OUTTMP<-FTP", backend=fake)
        handle = dev.subscribe(event="p,1000")
        drf = next(iter(handle._drfs))
        assert drf.endswith("<-FTP")
        handle.stop()

    def test_subscribe_delivers_readings(self, fake):
        """subscribe() handle receives emitted readings."""
        dev = Device("M:OUTTMP", backend=fake)
        handle = dev.subscribe(event="p,1000")
        # The DRF used for subscription
        sub_drf = next(iter(handle._drfs))
        fake.emit_reading(sub_drf, 72.5)
        readings = list(handle.readings(timeout=1.0))
        assert len(readings) == 1
        assert readings[0][0].value == 72.5
        handle.stop()

    def test_subscribe_context_manager(self, fake):
        """subscribe() handle works as context manager."""
        dev = Device("M:OUTTMP", backend=fake)
        with dev.subscribe(event="p,1000") as handle:
            sub_drf = next(iter(handle._drfs))
            fake.emit_reading(sub_drf, 42.0)
            for reading, h in handle.readings(timeout=1.0):
                assert reading.value == 42.0
                break
        assert handle.stopped

    def test_subscribe_with_callback(self, fake):
        """subscribe() with callback receives readings via callback."""
        received = []
        dev = Device("M:OUTTMP", backend=fake)
        handle = dev.subscribe(event="p,1000", callback=lambda r, h: received.append(r))
        sub_drf = next(iter(handle._drfs))
        fake.emit_reading(sub_drf, 99.0)
        assert len(received) == 1
        assert received[0].value == 99.0
        handle.stop()

    def test_subscribe_invalid_event_raises(self, fake):
        """subscribe(event=...) with invalid event string raises ValueError."""
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(ValueError):
            dev.subscribe(event="Z")

    def test_subscribe_never_event_kwarg_raises(self, fake):
        """subscribe(event='N') raises ValueError — @N is write-only."""
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(ValueError, match="cannot use @N"):
            dev.subscribe(event="N")

    def test_subscribe_never_event_on_device_raises(self, fake):
        """subscribe() on a device with @N raises ValueError."""
        dev = Device("M:OUTTMP@N", backend=fake)
        with pytest.raises(ValueError, match="cannot use @N"):
            dev.subscribe()

    def test_subscribe_validates_event_string(self, fake):
        """subscribe() validates the event string via parse_event()."""
        dev = Device("M:OUTTMP", backend=fake)
        # Valid event should work
        handle = dev.subscribe(event="E,0F")
        handle.stop()
        # Malformed event should fail at parse time
        with pytest.raises(ValueError):
            dev.subscribe(event="@p,1000")  # leading @ is invalid

    def test_subscribe_non_callable_callback_raises(self, fake):
        """subscribe('p,1000') raises TypeError — catches positional misuse."""
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(TypeError, match="callback must be callable"):
            dev.subscribe("p,1000")

    def test_subscribe_non_callable_on_error_raises(self, fake):
        """subscribe(on_error=...) with non-callable raises TypeError."""
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(TypeError, match="on_error must be callable"):
            dev.subscribe(on_error="not_a_function", event="p,1000")

    def test_subscribe_callback_wrong_arity_raises(self, fake):
        """subscribe(lambda r: ...) raises — callback needs 2 args."""
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(TypeError, match="must accept 2 arguments"):
            dev.subscribe(lambda r: None, event="p,1000")

    def test_subscribe_on_error_wrong_arity_raises(self, fake):
        """on_error with 1 arg raises TypeError."""
        dev = Device("M:OUTTMP", backend=fake)
        with pytest.raises(TypeError, match="must accept 2 arguments"):
            dev.subscribe(on_error=lambda e: None, event="p,1000")

    def test_subscribe_callback_with_extra_args_ok(self, fake):
        """Callback with *args or extra defaulted params is fine."""
        dev = Device("M:OUTTMP", backend=fake)
        # *args — uninspectable arity, should pass
        handle = dev.subscribe(lambda *args: None, event="p,1000")
        handle.stop()
        # 3 params with default — has >= 2 positional, should pass
        handle = dev.subscribe(lambda r, h, extra=None: None, event="p,1000")
        handle.stop()
