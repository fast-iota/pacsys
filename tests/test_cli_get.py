"""Tests for pacsys.cli.get — acget / pacsys-get CLI tool."""

import contextlib
import io
import json
from datetime import datetime
from unittest import mock

import numpy as np

from pacsys.types import Reading, ValueType, DeviceMeta


def _make_reading(
    drf="M:OUTTMP",
    value=72.5,
    error_code=0,
    units="degF",
    msg=None,
    ts=None,
    name="M:OUTTMP",
):
    meta = DeviceMeta(device_index=0, name=name, description="Outside temp", units=units)
    return Reading(
        drf=drf,
        value_type=ValueType.SCALAR,
        value=value,
        error_code=error_code,
        message=msg,
        timestamp=ts or datetime(2025, 6, 15, 12, 0, 0),
        meta=meta,
    )


class TestSingleDevice:
    """Single device read calls backend.get() and prints the result."""

    @mock.patch("pacsys.cli.get.make_backend")
    def test_single_device(self, mock_mb):
        from pacsys.cli.get import main

        backend = mock.MagicMock()
        backend.get.return_value = _make_reading()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acget", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        backend.get.assert_called_once()
        output = buf.getvalue()
        assert "M:OUTTMP" in output
        assert "72.5" in output
        backend.close.assert_called_once()


class TestMultipleDevices:
    """Multiple devices use get_many() and print all results."""

    @mock.patch("pacsys.cli.get.make_backend")
    def test_multiple_devices(self, mock_mb):
        from pacsys.cli.get import main

        r1 = _make_reading(drf="M:OUTTMP", value=72.5, name="M:OUTTMP")
        r2 = _make_reading(drf="G:AMANDA", value=1.23, name="G:AMANDA", units="mm")
        backend = mock.MagicMock()
        backend.get_many.return_value = [r1, r2]
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acget", "M:OUTTMP", "G:AMANDA"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        backend.get_many.assert_called_once()
        output = buf.getvalue()
        assert "M:OUTTMP" in output
        assert "G:AMANDA" in output
        assert "72.5" in output
        assert "1.23" in output


class TestTerseOutput:
    """Terse mode prints bare values without device names or units."""

    @mock.patch("pacsys.cli.get.make_backend")
    def test_terse_output(self, mock_mb):
        from pacsys.cli.get import main

        backend = mock.MagicMock()
        backend.get.return_value = _make_reading()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acget", "-t", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        output = buf.getvalue().strip()
        assert "72.5" in output
        assert "M:OUTTMP" not in output
        assert "degF" not in output


class TestJsonOutput:
    """JSON output contains structured device/value fields."""

    @mock.patch("pacsys.cli.get.make_backend")
    def test_json_output(self, mock_mb):
        from pacsys.cli.get import main

        backend = mock.MagicMock()
        backend.get.return_value = _make_reading()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acget", "--format", "json", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        data = json.loads(buf.getvalue().strip())
        assert data["device"] == "M:OUTTMP"
        assert data["value"] == 72.5
        assert data["ok"] is True


class TestNumberFormat:
    """Number format spec (-f) applies to displayed value."""

    @mock.patch("pacsys.cli.get.make_backend")
    def test_number_format(self, mock_mb):
        from pacsys.cli.get import main

        backend = mock.MagicMock()
        backend.get.return_value = _make_reading()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acget", "-f", ".3f", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        assert "72.500" in buf.getvalue()


class TestDeviceErrorInline:
    """Error reading prints ERROR and returns exit code 1."""

    @mock.patch("pacsys.cli.get.make_backend")
    def test_device_error_inline(self, mock_mb):
        from pacsys.cli.get import main

        backend = mock.MagicMock()
        backend.get.return_value = _make_reading(value=None, error_code=-1, msg="DIO_NOATT")
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acget", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 1
        assert "ERROR" in buf.getvalue()


class TestConnectionError:
    """Connection error produces exit code 2."""

    def test_connection_error_exit_code(self):
        from pacsys.cli.get import main

        with mock.patch("pacsys.cli.get.make_backend", side_effect=Exception("connection refused")):
            with mock.patch("sys.argv", ["acget", "M:OUTTMP"]):
                buf = io.StringIO()
                err = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                        code = main()
                except SystemExit as e:
                    code = e.code
                assert code == 2


class TestRejectControlRead:
    """Reading CONTROL property is rejected — it is write-only."""

    def test_explicit_control_property(self):
        from pacsys.cli.get import main

        err = io.StringIO()
        with mock.patch("sys.argv", ["acget", "Z:ACLTST.CONTROL"]):
            with contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 2
        assert "CONTROL" in err.getvalue()

    def test_ampersand_qualifier(self):
        from pacsys.cli.get import main

        err = io.StringIO()
        with mock.patch("sys.argv", ["acget", "Z&ACLTST"]):
            with contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 2
        assert "CONTROL" in err.getvalue()

    @mock.patch("pacsys.cli.get.make_backend")
    def test_status_qualifier_allowed(self, mock_mb):
        """STATUS (|) is readable — only CONTROL is rejected."""
        from pacsys.cli.get import main

        backend = mock.MagicMock()
        backend.get.return_value = _make_reading()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acget", "Z|ACLTST"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0


class TestArrayWithRange:
    """Array range (-r) slices numpy array values."""

    @mock.patch("pacsys.cli.get.make_backend")
    def test_array_with_range(self, mock_mb):
        from pacsys.cli.get import main

        arr = np.arange(10, dtype=float)
        backend = mock.MagicMock()
        backend.get.return_value = _make_reading(value=arr, units=None)
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acget", "-r", "0:3", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        output = buf.getvalue()
        assert "0 1 2" in output
        # Elements beyond the slice should not appear
        assert "9" not in output
