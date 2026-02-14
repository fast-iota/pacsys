"""Tests for pacsys.cli.put -- acput / pacsys-put CLI tool."""

import contextlib
import io
import json
from unittest import mock

from pacsys.types import WriteResult


def _ok_result(drf="M:OUTTMP"):
    return WriteResult(drf=drf)


def _err_result(drf="M:OUTTMP"):
    return WriteResult(drf=drf, error_code=-1, message="DIO_NOATT")


class TestSingleWrite:
    """Single device+value pair calls backend.write()."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_single_write(self, mock_mb):
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.return_value = _ok_result()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "M:OUTTMP", "72.5"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        backend.write.assert_called_once_with("M:OUTTMP", 72.5, timeout=5.0)
        assert "ok" in buf.getvalue().lower()
        backend.close.assert_called_once()


class TestMultipleWrites:
    """Multiple device+value pairs call backend.write_many()."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_multiple_writes(self, mock_mb):
        from pacsys.cli.put import main

        r1 = _ok_result("M:OUTTMP")
        r2 = _ok_result("G:AMANDA")
        backend = mock.MagicMock()
        backend.write_many.return_value = [r1, r2]
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "M:OUTTMP", "72.5", "G:AMANDA", "1.0"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        backend.write_many.assert_called_once_with([("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)], timeout=5.0)
        output = buf.getvalue()
        assert "M:OUTTMP" in output
        assert "G:AMANDA" in output


class TestWriteError:
    """Error write result returns exit code 1."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_write_error(self, mock_mb):
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.return_value = _err_result()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "M:OUTTMP", "72.5"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 1
        assert "FAILED" in buf.getvalue()


class TestOddArgsError:
    """Odd number of positional args gives exit code 2."""

    def test_odd_args_error(self):
        from pacsys.cli.put import main

        err = io.StringIO()
        with mock.patch("sys.argv", ["acput", "M:OUTTMP"]):
            with contextlib.redirect_stderr(err):
                try:
                    rc = main()
                except SystemExit as e:
                    rc = e.code
        assert rc == 2


class TestJsonOutput:
    """JSON output produces valid JSON with ok field."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_json_output(self, mock_mb):
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.return_value = _ok_result()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "--format", "json", "M:OUTTMP", "72.5"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        data = json.loads(buf.getvalue().strip())
        assert data["ok"] is True


class TestArrayValue:
    """Comma-separated value is parsed as a list and passed to write."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_array_value(self, mock_mb):
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.return_value = _ok_result()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "M:OUTTMP", "1.0,2.0,3.0"]):
            with contextlib.redirect_stdout(buf):
                rc = main()

        assert rc == 0
        backend.write.assert_called_once_with("M:OUTTMP", [1.0, 2.0, 3.0], timeout=5.0)


class TestConnectionError:
    """Connection error from make_backend gives exit code 2."""

    def test_connection_error(self):
        from pacsys.cli.put import main

        with mock.patch("pacsys.cli.put.make_backend", side_effect=Exception("connection refused")):
            with mock.patch("sys.argv", ["acput", "M:OUTTMP", "72.5"]):
                err = io.StringIO()
                try:
                    with contextlib.redirect_stderr(err):
                        rc = main()
                except SystemExit as e:
                    rc = e.code
                assert rc == 2


class TestVerifyPath:
    """Tests for --verify / --tolerance / --retries path."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_verify_uses_device_write(self, mock_mb):
        """--verify uses Device().write() with Verify config."""
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        mock_mb.return_value = backend

        with mock.patch("pacsys.device.Device") as MockDevice:
            mock_dev = mock.MagicMock()
            mock_dev.write.return_value = WriteResult(drf="M:OUTTMP", verified=True, readback=72.5)
            MockDevice.return_value = mock_dev

            buf = io.StringIO()
            with mock.patch("sys.argv", ["acput", "--verify", "M:OUTTMP", "72.5"]):
                with contextlib.redirect_stdout(buf):
                    rc = main()

            assert rc == 0
            MockDevice.assert_called_once()
            mock_dev.write.assert_called_once()
            call_kwargs = mock_dev.write.call_args
            assert call_kwargs is not None

    @mock.patch("pacsys.cli.put.make_backend")
    def test_tolerance_implies_verify(self, mock_mb):
        """--tolerance without --verify still activates verification."""
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        mock_mb.return_value = backend

        with mock.patch("pacsys.device.Device") as MockDevice:
            mock_dev = mock.MagicMock()
            mock_dev.write.return_value = WriteResult(drf="M:OUTTMP", verified=True, readback=72.5)
            MockDevice.return_value = mock_dev

            buf = io.StringIO()
            with mock.patch("sys.argv", ["acput", "--tolerance", "0.5", "M:OUTTMP", "72.5"]):
                with contextlib.redirect_stdout(buf):
                    rc = main()

            assert rc == 0
            mock_dev.write.assert_called_once()
            backend.write.assert_not_called()
