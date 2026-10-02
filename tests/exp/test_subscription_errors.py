"""Experiment consumers distinguish retry notifications from terminal stream errors."""

import importlib
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pacsys.backends.grpc_backend import GRPC_AVAILABLE, GRPCBackend, _GRPCSubscriptionHandle
from pacsys.exp import DataLogger, Monitor, read_fresh, watch
from pacsys.types import DispatchMode, Reading, ValueType

pytestmark = pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpc not installed")

DRF = "M:OUTTMP@p,1000"


@pytest.fixture
def subscription(monkeypatch):
    with GRPCBackend(dispatch_mode=DispatchMode.DIRECT) as backend:
        handles = []

        def subscribe(drfs, callback=None, on_error=None):
            handle = _GRPCSubscriptionHandle(backend, drfs, callback, on_error)
            handles.append(handle)
            return handle

        monkeypatch.setattr(backend, "subscribe", subscribe)
        yield backend, handles
        for handle in handles:
            handle.stop()


@pytest.mark.parametrize("helper", [watch, read_fresh])
@pytest.mark.parametrize("outcome", ["recovered", "timeout", "fatal", "queued_transient"])
def test_waiting_consumer_subscription_errors(helper, outcome, subscription, monkeypatch):
    backend, handles = subscription
    transient = ConnectionError("retrying")
    terminal = RuntimeError("terminal stream failure")
    reading = Reading(drf=DRF, value=72.5, value_type=ValueType.SCALAR, error_code=0)
    timeout = 0.01
    waits = []

    class DeliveryEvent(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            handle = handles[0]
            if outcome == "queued_transient":
                # The queued retry notification is delivered after the stream has failed.
                handle._signal_error(terminal)
                handle._dispatch_error(transient, fatal=False)
            elif outcome == "fatal":
                handle._dispatch_error(terminal, fatal=True)
            else:
                handle._dispatch_error(transient, fatal=False)
                handle._dispatch_error(transient, fatal=False)
                # An erroneously completed wait must exit before the later reading arrives.
                if not self.is_set() and outcome == "recovered":
                    handle._dispatch(reading)
            return super().wait(timeout)

    module = importlib.import_module(helper.__module__)
    monkeypatch.setattr(module, "threading", SimpleNamespace(Event=DeliveryEvent, Lock=threading.Lock))
    args = (DRF, lambda r: r.value > 70) if helper is watch else ([DRF],)
    try:
        if outcome == "recovered":
            result = helper(*args, timeout=timeout, backend=backend)
            assert (result if helper is watch else result[0].reading) is reading
        else:
            expected = TimeoutError if outcome == "timeout" else RuntimeError
            with pytest.raises(expected) as info:
                helper(*args, timeout=timeout, backend=backend)
            if outcome != "timeout":
                assert info.value is terminal
    finally:
        assert waits == [timeout]
        assert handles[0]._stop_requested
        assert handles[0].stopped


@pytest.mark.parametrize("fatal", [False, True])
def test_logger_subscription_errors(subscription, fatal, caplog):
    backend, handles = subscription
    writer = Mock()
    dl = DataLogger([DRF], writer=writer, flush_interval=999, backend=backend)
    dl.start()
    handle = handles[0]
    transient = ConnectionError("retrying")
    terminal = RuntimeError("terminal stream failure")
    reading = Reading(drf=DRF, value=72.5, value_type=ValueType.SCALAR, error_code=0)
    try:
        handle._dispatch_error(transient, fatal=False)
        assert dl.running
        assert not dl.failed
        assert dl.last_error is None
        assert "logging stopped" not in caplog.text
        handle._dispatch(reading)
        if fatal:
            handle._signal_error(terminal)
            handle._dispatch_error(transient, fatal=False)
            assert dl.failed
            assert dl.last_error is terminal
            assert str(terminal) in caplog.text
            assert str(transient) not in caplog.text
    finally:
        if fatal and handle.exc is not None:
            with pytest.raises(RuntimeError, match="subscription failed") as info:
                dl.stop()
            assert info.value.__cause__ is terminal
        else:
            dl.stop()
    writer.write_readings.assert_called_once_with([reading])
    writer.close.assert_called_once_with()
    assert not dl.running


def test_monitor_logs_only_terminal_errors(subscription, caplog):
    backend, handles = subscription
    transient = ConnectionError("retrying")
    terminal = RuntimeError("terminal stream failure")
    mon = Monitor([DRF], backend=backend)
    mon.start()
    handle = handles[0]
    try:
        handle._dispatch_error(transient, fatal=False)
        assert mon.running
        assert "go stale" not in caplog.text
        handle._signal_error(terminal)
        handle._dispatch_error(transient, fatal=False)
        assert str(terminal) in caplog.text
        assert str(transient) not in caplog.text
    finally:
        mon.stop()
