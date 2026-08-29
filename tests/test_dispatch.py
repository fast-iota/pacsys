"""Tests for CallbackDispatcher (pacsys.backends._dispatch)."""

import logging
import threading
import time

from pacsys.backends._dispatch import _SLOW_THRESHOLD, CallbackDispatcher
from pacsys.types import DispatchMode, Reading, ValueType


def _make_reading(drf: str = "M:OUTTMP", value: float = 72.5) -> Reading:
    return Reading(drf=drf, value_type=ValueType.SCALAR, value=value)


class _FakeHandle:
    """Minimal stand-in for SubscriptionHandle in tests."""

    stopped = False
    _stop_requested = False

    def __init__(self):
        self.dispatch_drops = 0

    def _note_dispatch_drop(self):
        self.dispatch_drops += 1


# ─── WORKER mode ─────────────────────────────────────────────────────────


class TestWorkerMode:
    def test_delivers_on_worker_thread(self):
        """Callback runs on a thread different from the caller."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        tid = []
        event = threading.Event()

        def cb(reading, handle):
            tid.append(threading.current_thread().ident)
            event.set()

        try:
            d.dispatch_reading(cb, _make_reading(), _FakeHandle())
            assert event.wait(2.0)
            assert tid[0] != threading.current_thread().ident
        finally:
            d.close()

    def test_preserves_ordering(self):
        """Readings dispatched in order are delivered in order."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        results = []
        done = threading.Event()
        count = 50

        def cb(reading, handle):
            results.append(reading.value)
            if len(results) == count:
                done.set()

        try:
            for i in range(count):
                d.dispatch_reading(cb, _make_reading(value=float(i)), _FakeHandle())
            assert done.wait(5.0)
            assert results == [float(i) for i in range(count)]
        finally:
            d.close()

    def test_lazy_worker_start(self):
        """No thread is created until first dispatch."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        assert d._thread is None
        event = threading.Event()

        def cb(reading, handle):
            event.set()

        try:
            d.dispatch_reading(cb, _make_reading(), _FakeHandle())
            assert event.wait(2.0)
            assert d._thread is not None
        finally:
            d.close()

    def test_close_joins_worker(self):
        """close() stops and joins the worker thread."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        event = threading.Event()

        def cb(reading, handle):
            event.set()

        d.dispatch_reading(cb, _make_reading(), _FakeHandle())
        event.wait(2.0)
        thread = d._thread
        assert thread is not None
        assert thread.is_alive()
        d.close()
        assert not thread.is_alive()

    def test_close_from_worker_callback(self):
        """close() called from inside a worker callback must not raise (self-join)."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        errors = []
        worker = []
        done = threading.Event()

        def cb(reading, handle):
            worker.append(threading.current_thread())
            try:
                d.close()
            except Exception as e:  # noqa: BLE001
                errors.append(e)
            finally:
                done.set()

        d.dispatch_reading(cb, _make_reading(), _FakeHandle())
        assert done.wait(2.0)
        assert errors == []
        # Self-close must not null the thread reference: a later cross-thread
        # close() still needs it to join the worker.
        assert d._thread is worker[0]
        d.close()
        assert d._thread is None
        assert not worker[0].is_alive()

    def test_callback_exception_doesnt_crash_worker(self):
        """A failing callback doesn't kill the worker - next dispatch still works."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        ok = threading.Event()

        def bad_cb(reading, handle):
            raise RuntimeError("boom")

        def good_cb(reading, handle):
            ok.set()

        try:
            d.dispatch_reading(bad_cb, _make_reading(), _FakeHandle())
            d.dispatch_reading(good_cb, _make_reading(), _FakeHandle())
            assert ok.wait(2.0)
        finally:
            d.close()

    def test_dispatch_after_close_is_silent(self):
        """Dispatching after close() silently drops instead of queueing to dead worker."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        called = []
        delivered = threading.Event()

        def cb(reading, handle):
            called.append(1)
            delivered.set()

        d.dispatch_reading(cb, _make_reading(), _FakeHandle())
        assert delivered.wait(2.0)
        d.close()
        # Should not raise or enqueue
        d.dispatch_reading(cb, _make_reading(), _FakeHandle())
        d.dispatch_error(lambda e, h: called.append(2), RuntimeError("x"), _FakeHandle())
        assert len(called) == 1  # only the pre-close dispatch

    def test_error_dispatch_worker(self):
        """dispatch_error works in WORKER mode."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        captured = []
        event = threading.Event()

        def on_error(exc, handle):
            captured.append(exc)
            event.set()

        try:
            d.dispatch_error(on_error, RuntimeError("fail"), _FakeHandle())
            assert event.wait(2.0)
            assert isinstance(captured[0], RuntimeError)
        finally:
            d.close()


# ─── DIRECT mode ─────────────────────────────────────────────────────────


class TestDirectMode:
    def test_delivers_on_same_thread(self):
        """Callback runs on the caller's thread."""
        d = CallbackDispatcher(DispatchMode.DIRECT)
        tid = []

        def cb(reading, handle):
            tid.append(threading.current_thread().ident)

        d.dispatch_reading(cb, _make_reading(), _FakeHandle())
        assert tid[0] == threading.current_thread().ident

    def test_slow_callback_warning(self, caplog):
        """Callback taking >50ms logs a warning."""
        d = CallbackDispatcher(DispatchMode.DIRECT)

        def slow_cb(reading, handle):
            time.sleep(_SLOW_THRESHOLD + 0.02)

        with caplog.at_level(logging.WARNING, logger="pacsys.backends._dispatch"):
            d.dispatch_reading(slow_cb, _make_reading(), _FakeHandle())

        assert any("Slow callback" in r.message for r in caplog.records)

    def test_warning_rate_limited(self, caplog):
        """Multiple slow callbacks within 10s produce only one warning."""
        d = CallbackDispatcher(DispatchMode.DIRECT)

        def slow_cb(reading, handle):
            time.sleep(_SLOW_THRESHOLD + 0.02)

        with caplog.at_level(logging.WARNING, logger="pacsys.backends._dispatch"):
            d.dispatch_reading(slow_cb, _make_reading(), _FakeHandle())
            d.dispatch_reading(slow_cb, _make_reading(), _FakeHandle())
            d.dispatch_reading(slow_cb, _make_reading(), _FakeHandle())

        slow_warnings = [r for r in caplog.records if "Slow callback" in r.message]
        assert len(slow_warnings) == 1

    def test_error_dispatch_direct(self):
        """dispatch_error works in DIRECT mode."""
        d = CallbackDispatcher(DispatchMode.DIRECT)
        captured = []

        def on_error(exc, handle):
            captured.append(exc)

        d.dispatch_error(on_error, RuntimeError("fail"), _FakeHandle())
        assert isinstance(captured[0], RuntimeError)

    def test_callback_exception_logged(self, caplog):
        """Callback exception is caught and logged, not re-raised."""
        d = CallbackDispatcher(DispatchMode.DIRECT)

        def bad_cb(reading, handle):
            raise ValueError("boom")

        with caplog.at_level(logging.ERROR, logger="pacsys.backends._dispatch"):
            d.dispatch_reading(bad_cb, _make_reading(), _FakeHandle())

        assert "Error in reading callback" in caplog.text
        assert "ValueError: boom" in caplog.text

    def test_close_is_noop(self):
        """close() on DIRECT dispatcher with no worker is safe."""
        d = CallbackDispatcher(DispatchMode.DIRECT)
        d.close()  # should not raise


class TestStopAndDropAccounting:
    @staticmethod
    def _deliver_then_stop(requested: bool) -> list[float]:
        d = CallbackDispatcher(DispatchMode.WORKER)
        handle = _FakeHandle()
        gate = threading.Event()
        delivered = []

        def cb(reading, h):
            gate.wait(1.0)  # hold the worker so later items pile up in the queue
            delivered.append(reading.value)

        try:
            d.dispatch_reading(cb, _make_reading(value=1.0), handle)
            time.sleep(0.05)  # worker is now inside cb(1.0)
            for v in (2.0, 3.0):
                d.dispatch_reading(cb, _make_reading(value=v), handle)
            handle.stopped = True
            handle._stop_requested = requested
            gate.set()
            time.sleep(0.1)
        finally:
            d.close()
        return delivered

    def test_queued_callbacks_skipped_after_stop(self):
        """Readings queued before stop() must not reach the callback afterwards."""
        assert self._deliver_then_stop(requested=True) == [1.0]

    def test_queued_callbacks_delivered_after_stream_end(self):
        """A stream ending on its own (stopped, but no stop() call) still delivers its queued tail."""
        assert self._deliver_then_stop(requested=False) == [1.0, 2.0, 3.0]

    def test_error_callback_delivered_after_stop(self):
        """A stream error stops the handle before its on_error is queued; it must still arrive."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        handle = _FakeHandle()
        handle.stopped = True
        got = threading.Event()
        try:
            d.dispatch_error(lambda exc, h: got.set(), RuntimeError("boom"), handle)
            assert got.wait(1.0)
        finally:
            d.close()

    def test_queue_full_counted_on_handle(self):
        d = CallbackDispatcher(DispatchMode.WORKER)
        handle = _FakeHandle()
        gate = threading.Event()
        try:
            d.dispatch_reading(lambda r, h: gate.wait(2.0), _make_reading(), handle)
            time.sleep(0.05)
            from pacsys.backends import _dispatch

            for _ in range(_dispatch._QUEUE_MAX_SIZE + 3):
                d.dispatch_reading(lambda r, h: None, _make_reading(), handle)
            assert handle.dispatch_drops == 3
        finally:
            gate.set()
            d.close()
