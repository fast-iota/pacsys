"""Tests for pacsys.cli.put -- acput / pacsys-put CLI tool."""

import contextlib
import io
import json
from unittest import mock

import pytest

from pacsys.types import BasicControl, ValueType, WriteResult


def _ok_result(drf="M:OUTTMP"):
    return WriteResult(drf=drf)


def _err_result(drf="M:OUTTMP"):
    return WriteResult(drf=drf, error_code=-1, message="DIO_NOATT")


def _run_fake(argv, fake):
    from pacsys.cli.put import main

    out, err = io.StringIO(), io.StringIO()
    with (
        mock.patch("pacsys.cli.put.make_backend", return_value=fake) as mock_mb,
        mock.patch("pacsys.device.time.sleep"),
        mock.patch("sys.argv", ["acput", *argv]),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        rc = main()
    return rc, out.getvalue(), err.getvalue(), mock_mb


class TestSingleWrite:
    """Single device+value pair calls backend.write()."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_single_write(self, mock_mb):
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.return_value = _ok_result()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "M:OUTTMP", "72.5"]), contextlib.redirect_stdout(buf):
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
        with mock.patch("sys.argv", ["acput", "M:OUTTMP", "72.5", "G:AMANDA", "1.0"]), contextlib.redirect_stdout(buf):
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
        with mock.patch("sys.argv", ["acput", "M:OUTTMP", "72.5"]), contextlib.redirect_stdout(buf):
            rc = main()

        assert rc == 1
        assert "FAILED" in buf.getvalue()


class TestOddArgsError:
    """Odd number of positional args gives exit code 2."""

    def test_odd_args_error(self):
        from pacsys.cli.put import main

        err = io.StringIO()
        with mock.patch("sys.argv", ["acput", "M:OUTTMP"]), contextlib.redirect_stderr(err):
            try:
                rc = main()
            except SystemExit as e:
                rc = e.code
        assert rc == 2


class TestNonFiniteValueError:
    """Non-finite setpoints are rejected before any backend connection."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_non_finite_rejected(self, mock_mb):
        from pacsys.cli.put import main

        err = io.StringIO()
        with mock.patch("sys.argv", ["acput", "M:OUTTMP", "inf"]), contextlib.redirect_stderr(err):
            rc = main()
        assert rc == 2
        assert "non-finite" in err.getvalue()
        mock_mb.assert_not_called()


class TestMalformedDrfError:
    """Malformed DRFs are reported as usage errors, not tracebacks."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_malformed_drf_with_control_value_rejected(self, mock_mb):
        from pacsys.cli.put import main

        err = io.StringIO()
        with mock.patch("sys.argv", ["acput", "M:OUTTMP{bad}", "on"]), contextlib.redirect_stderr(err):
            rc = main()
        assert rc == 2
        assert "Invalid" in err.getvalue()
        mock_mb.assert_not_called()

    @mock.patch("pacsys.cli.put.make_backend")
    def test_malformed_drf_with_verify_rejected(self, mock_mb):
        from pacsys.cli.put import main

        err = io.StringIO()
        with mock.patch("sys.argv", ["acput", "--verify", "M:OUTTMP{bad}", "5"]):
            with contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 2
        assert "Invalid" in err.getvalue()
        mock_mb.assert_not_called()


class TestJsonOutput:
    """JSON output produces valid JSON with ok field."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_json_output(self, mock_mb):
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.return_value = _ok_result()
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "--format", "json", "M:OUTTMP", "72.5"]), contextlib.redirect_stdout(buf):
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
        with mock.patch("sys.argv", ["acput", "M:OUTTMP", "1.0,2.0,3.0"]), contextlib.redirect_stdout(buf):
            rc = main()

        assert rc == 0
        backend.write.assert_called_once_with("M:OUTTMP", [1.0, 2.0, 3.0], timeout=5.0)


class TestConnectionError:
    """Connection error from make_backend gives exit code 2."""

    def test_connection_error(self):
        from pacsys.cli.put import main

        with (
            mock.patch("pacsys.cli.put.make_backend", side_effect=Exception("connection refused")),
            mock.patch("sys.argv", ["acput", "M:OUTTMP", "72.5"]),
        ):
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err):
                    rc = main()
            except SystemExit as e:
                rc = e.code
            assert rc == 2

    def test_write_runtime_error_exit_code(self):
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.side_effect = RuntimeError("write failed")
        with (
            mock.patch("pacsys.cli.put.make_backend", return_value=backend),
            mock.patch("sys.argv", ["acput", "M:OUTTMP", "72.5"]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            rc = main()

        assert rc == 1
        backend.close.assert_called_once_with()


class TestControlWrite:
    """Control shorthand names are passed as BasicControl enum to backend."""

    @mock.patch("pacsys.cli.put.make_backend")
    def test_control_with_status_qualifier(self, mock_mb):
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.return_value = _ok_result("Z:ACLTST")
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "Z|ACLTST", "on"]), contextlib.redirect_stdout(buf):
            rc = main()

        assert rc == 0
        # The STATUS qualifier is preserved for backend control dispatch.
        backend.write.assert_called_once_with("Z|ACLTST", BasicControl.ON, timeout=5.0)

    @mock.patch("pacsys.cli.put.make_backend")
    def test_bare_drf_auto_targets_control(self, mock_mb):
        """acput Z:ACLTST on → DRF rewritten to Z:ACLTST.CONTROL."""
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.return_value = _ok_result("Z:ACLTST")
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "Z:ACLTST", "on"]), contextlib.redirect_stdout(buf):
            rc = main()

        assert rc == 0
        drf_arg = backend.write.call_args[0][0]
        assert "CONTROL" in drf_arg

    @mock.patch("pacsys.cli.put.make_backend")
    def test_control_on_epics_device_rejected(self, mock_mb):
        """Basic control has no EPICS analogue - usage error, no backend call."""
        from pacsys.cli.put import main

        err = io.StringIO()
        with mock.patch("sys.argv", ["acput", "SR:BPM:01:X", "on"]), contextlib.redirect_stderr(err):
            rc = main()

        assert rc != 0
        assert "ACNET" in err.getvalue()
        mock_mb.return_value.write.assert_not_called()


class TestVerifyPath:
    """Tests for --verify / --tolerance / --retries path."""

    def test_verify_failure_returns_device_error(self):
        backend = mock.MagicMock()
        backend.write.return_value = _ok_result("M:OUTTMP.SETTING@N")
        backend.read.return_value = 70.0
        rc, out, _, _ = _run_fake(["--verify", "M:OUTTMP", "72.5"], backend)

        assert rc == 1
        assert "verify FAILED" in out
        assert backend.read.call_count == 3  # default --retries
        backend.close.assert_called_once()

    @mock.patch("pacsys.cli.put.make_backend")
    def test_verify_control_never_writes_setting(self, mock_mb):
        """acput DEV reset --verify must go through control() (.CONTROL@N), never .SETTING@N."""
        from pacsys.cli.put import main

        backend = mock.MagicMock()
        backend.write.return_value = _ok_result("Z:ACLTST.CONTROL")
        backend.read.return_value = True  # STATUS.READY readback
        mock_mb.return_value = backend

        buf = io.StringIO()
        with mock.patch("sys.argv", ["acput", "--verify", "Z:ACLTST", "reset"]), contextlib.redirect_stdout(buf):
            rc = main()

        assert rc == 0
        write_drf, written_value = backend.write.call_args[0][:2]
        assert ".CONTROL" in write_drf
        assert "@N" in write_drf
        assert written_value == BasicControl.RESET
        all_drfs = [c[0][0] for c in backend.write.call_args_list] + [c[0][0] for c in backend.read.call_args_list]
        assert not any("SETTING" in d for d in all_drfs)

    def test_tolerance_implies_verify(self):
        """--tolerance without --verify still verifies, within that tolerance."""
        backend = mock.MagicMock()
        backend.write.return_value = _ok_result("M:OUTTMP.SETTING@N")
        backend.read.return_value = 72.9
        rc, out, _, _ = _run_fake(["--tolerance", "0.5", "M:OUTTMP", "72.5"], backend)

        assert rc == 0
        assert "verified" in out
        backend.write.assert_called_once_with("M:OUTTMP.SETTING@N", 72.5, timeout=5.0)
        backend.read.assert_called_once_with("M:OUTTMP.SETTING@I", 5.0)


class TestVerifyTarget:
    """--verify writes and reads back the DRF's own writable property/field/range."""

    @pytest.mark.parametrize(
        ("drf", "write_drf", "read_drf"),
        [
            ("Z:ACLTST.ANALOG.NOM", "Z:ACLTST.ANALOG.NOM@N", "Z:ACLTST.ANALOG.NOM@I"),
            ("Z@ACLTST.MAX", "Z:ACLTST.ANALOG.MAX@N", "Z:ACLTST.ANALOG.MAX@I"),
            ("Z$ACLTST.MASK", "Z:ACLTST.DIGITAL.MASK@N", "Z:ACLTST.DIGITAL.MASK@I"),
            ("M:OUTTMP", "M:OUTTMP.SETTING@N", "M:OUTTMP.SETTING@I"),
            ("M:OUTTMP.READING.PRIMARY", "M:OUTTMP.SETTING.PRIMARY@N", "M:OUTTMP.SETTING.PRIMARY@I"),
            ("B:HS23T[3]", "B:HS23T.SETTING[3]@N", "B:HS23T.SETTING[3]@I"),
            ("SR:BPM:01:X", "SR:BPM:01:X@N", "SR:BPM:01:X@I"),
        ],
    )
    def test_verify_preserves_target(self, drf, write_drf, read_drf):
        from pacsys.testing import FakeBackend

        fake = FakeBackend()
        fake.set_reading(
            "B:HS23T.SETTING", [0.0] * 5, value_type=ValueType.SCALAR_ARRAY
        )  # partial ranged writes need a seeded array
        rc, out, _, _ = _run_fake(["--format", "json", "--verify", drf, "5"], fake)

        assert rc == 0
        assert fake.writes == [(write_drf, 5.0)]
        assert fake.reads == [read_drf]
        data = json.loads(out)
        assert data["confirmed"] is True
        assert data["readback"] == 5.0

    def test_verify_alarm_mismatch_fails(self):
        backend = mock.MagicMock()
        backend.write.return_value = _ok_result("Z:ACLTST.ANALOG.NOM@N")
        backend.read.return_value = 1.0
        rc, out, _, _ = _run_fake(["--verify", "--retries", "2", "Z:ACLTST.ANALOG.NOM", "5"], backend)

        assert rc == 1
        assert "verify FAILED" in out
        backend.write.assert_called_once_with("Z:ACLTST.ANALOG.NOM@N", 5.0, timeout=5.0)
        assert [c.args[0] for c in backend.read.call_args_list] == ["Z:ACLTST.ANALOG.NOM@I"] * 2

    def test_verify_control_still_reads_status(self):
        from pacsys.testing import FakeBackend

        fake = FakeBackend()
        fake.set_reading("Z:ACLTST.STATUS.ON@I", True)
        rc, _, _, _ = _run_fake(["--verify", "Z:ACLTST", "on"], fake)

        assert rc == 0
        assert fake.writes == [("Z:ACLTST.CONTROL@N", BasicControl.ON)]
        assert fake.reads == ["Z:ACLTST.STATUS.ON@I"]

    @pytest.mark.parametrize(
        "drf",
        [
            "Z:ACLTST.ANALOG",  # whole alarm block readback is a dict, not a CLI value
            "Z$ACLTST",
            "Z:ACLTST.CONTROL",  # non-BasicControl value to CONTROL
            "Z|ACLTST",
            "Z:ACLTST.DESCRIPTION",
            "Z:ACLTST.BIT_STATUS",
            "M:OUTTMP.SETTING.RAW",  # RAW readback is bytes, which no CLI value can match
            "M:OUTTMP.READING.RAW",
            "Z:ACLTST.ANALOG.RAW",
            "Z:ACLTST.DIGITAL.RAW",
        ],
    )
    def test_unsupported_verify_rejected_before_any_write(self, drf):
        from pacsys.testing import FakeBackend

        fake = FakeBackend()
        rc, _, err, mock_mb = _run_fake(["--verify", "M:OUTTMP", "1", drf, "5"], fake)

        assert rc == 2
        assert "--verify" in err
        mock_mb.assert_not_called()
        assert fake.writes == []

    @pytest.mark.parametrize("tolerance", ["-1", "nan"])
    def test_malformed_tolerance_rejected_before_connect(self, tolerance):
        from pacsys.testing import FakeBackend

        fake = FakeBackend()
        rc, _, err, mock_mb = _run_fake([f"--tolerance={tolerance}", "M:OUTTMP", "1"], fake)

        assert rc == 2
        assert "tolerance" in err
        mock_mb.assert_not_called()
        assert fake.writes == []
