"""Tests for pacsys.cli.monitor -- acmonitor / pacsys-monitor CLI tool."""

import contextlib
import io
import json
from datetime import datetime
from unittest import mock

from pacsys.types import Reading, ValueType, DeviceMeta


def _make_reading(
    drf="M:OUTTMP@p,1000",
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


def _make_handle(readings_to_yield):
    """Create a mock SubscriptionHandle that yields predetermined readings then stops."""
    handle = mock.MagicMock()
    handle.stopped = False

    def fake_readings(timeout=None):
        for r in readings_to_yield:
            yield (r, handle)
        handle.stopped = True

    handle.readings = fake_readings
    handle.stop = mock.MagicMock()
    handle.__enter__ = mock.MagicMock(return_value=handle)
    handle.__exit__ = mock.MagicMock(return_value=False)
    return handle


class TestBasicMonitoring:
    """Basic streaming: 2 readings yielded, both appear in output."""

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_basic_monitoring(self, mock_mb):
        from pacsys.cli.monitor import main

        r1 = _make_reading(value=72.5)
        r2 = _make_reading(value=73.0)
        handle = _make_handle([r1, r2])

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 0
        output = buf.getvalue()
        assert "72.5" in output
        assert "73.0" in output
        assert "M:OUTTMP" in output


class TestJsonOutput:
    """JSON output produces valid JSON lines."""

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_json_output(self, mock_mb):
        from pacsys.cli.monitor import main

        r1 = _make_reading(value=72.5)
        handle = _make_handle([r1])

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "--format", "json", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 0
        data = json.loads(buf.getvalue().strip().splitlines()[0])
        assert data["device"] == "M:OUTTMP"
        assert data["value"] == 72.5
        assert data["ok"] is True


class TestTerseOutput:
    """Terse mode prints bare values without device names or units."""

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_terse_output(self, mock_mb):
        from pacsys.cli.monitor import main

        r1 = _make_reading(value=72.5)
        handle = _make_handle([r1])

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "-t", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 0
        output = buf.getvalue().strip()
        assert "72.5" in output
        assert "M:OUTTMP" not in output
        assert "degF" not in output


class TestCountLimitsOutput:
    """-n flag stops after N total readings."""

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_count_limits_output(self, mock_mb):
        from pacsys.cli.monitor import main

        readings = [_make_reading(value=float(i)) for i in range(10)]
        handle = _make_handle(readings)

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "-n", "3", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 0
        lines = [ln for ln in buf.getvalue().strip().splitlines() if ln.strip()]
        assert len(lines) == 3


class TestDefaultEventAppended:
    """subscribe() called with @p,1000 appended when no event in DRF."""

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_default_event_appended(self, mock_mb):
        from pacsys.cli.monitor import main

        handle = _make_handle([])

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                main()

        # subscribe should have been called with the event appended
        call_args = backend.subscribe.call_args
        drfs = call_args[0][0] if call_args[0] else call_args[1].get("drfs", call_args[1].get("devices"))
        assert any("@p,1000" in d.lower() for d in drfs)

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_explicit_event_not_modified(self, mock_mb):
        from pacsys.cli.monitor import main

        handle = _make_handle([])

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "M:OUTTMP@e,1d"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                main()

        call_args = backend.subscribe.call_args
        drfs = call_args[0][0] if call_args[0] else call_args[1].get("drfs", call_args[1].get("devices"))
        # Should NOT have @p,1000 appended since explicit event was given
        assert any("@e,1d" in d.lower() for d in drfs)
        assert not any("@p,1000" in d.lower() for d in drfs)


class TestConnectionError:
    """Connection error from make_backend gives exit code 2."""

    def test_connection_error(self):
        from pacsys.cli.monitor import main

        with mock.patch("pacsys.cli.monitor.make_backend", side_effect=Exception("connection refused")):
            with mock.patch("sys.argv", ["acmonitor", "M:OUTTMP"]):
                err = io.StringIO()
                try:
                    with contextlib.redirect_stderr(err):
                        rc = main()
                except SystemExit as e:
                    rc = e.code
                assert rc == 2
                assert "connection" in err.getvalue().lower()


class TestEpochTimestampFormat:
    """--timestamp-format epoch outputs unix epoch floats."""

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_epoch_text(self, mock_mb):
        from pacsys.cli.monitor import main

        ts = datetime(2025, 6, 15, 12, 0, 0)
        r1 = _make_reading(value=72.5, ts=ts)
        handle = _make_handle([r1])

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "-s", "epoch", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 0
        output = buf.getvalue()
        expected = f"{ts.timestamp():.3f}"
        assert expected in output

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_epoch_json(self, mock_mb):
        from pacsys.cli.monitor import main

        ts = datetime(2025, 6, 15, 12, 0, 0)
        r1 = _make_reading(value=72.5, ts=ts)
        handle = _make_handle([r1])

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "-s", "epoch", "--format", "json", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 0
        data = json.loads(buf.getvalue().strip().splitlines()[0])
        assert isinstance(data["timestamp"], float)


class TestRelativeTimestampFormat:
    """--timestamp-format relative outputs seconds since first reading."""

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_relative_text(self, mock_mb):
        from pacsys.cli.monitor import main

        ts1 = datetime(2025, 6, 15, 12, 0, 0)
        ts2 = datetime(2025, 6, 15, 12, 0, 5)
        r1 = _make_reading(value=72.5, ts=ts1)
        r2 = _make_reading(value=73.0, ts=ts2)
        handle = _make_handle([r1, r2])

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "-s", "relative", "M:OUTTMP"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 0
        lines = buf.getvalue().strip().splitlines()
        assert "0.000" in lines[0]  # first reading at t=0
        assert "5.000" in lines[1]  # second reading at t=5


class TestSummaryPrinted:
    """Summary line appears on stderr after streaming completes."""

    @mock.patch("pacsys.cli.monitor.make_backend")
    def test_summary_printed(self, mock_mb):
        from pacsys.cli.monitor import main

        r1 = _make_reading(value=72.5, name="M:OUTTMP")
        r2 = _make_reading(value=73.0, name="M:OUTTMP")
        r3 = _make_reading(drf="G:AMANDA@p,1000", value=1.23, name="G:AMANDA", units="mm")
        handle = _make_handle([r1, r2, r3])

        backend = mock.MagicMock()
        backend.subscribe.return_value = handle
        mock_mb.return_value = backend

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.argv", ["acmonitor", "M:OUTTMP", "G:AMANDA"]):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                rc = main()

        assert rc == 0
        summary = err.getvalue()
        assert "---" in summary
        assert "2 readings from M:OUTTMP" in summary
        assert "1 readings from G:AMANDA" in summary
