"""Tests for CallbackDispatcher (pacsys.backends._dispatch)."""

import logging
import threading
import time


from pacsys.backends._dispatch import CallbackDispatcher, _SLOW_THRESHOLD
from pacsys.types import DispatchMode, Reading, ValueType


def _make_reading(drf: str = "M:OUTTMP", value: float = 72.5) -> Reading:
    return Reading(drf=drf, value_type=ValueType.SCALAR, value=value)


class _FakeHandle:
    """Minimal stand-in for SubscriptionHandle in tests."""

    stopped = False


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
        N = 50

        def cb(reading, handle):
            results.append(reading.value)
            if len(results) == N:
                done.set()

        try:
            for i in range(N):
                d.dispatch_reading(cb, _make_reading(value=float(i)), _FakeHandle())
            assert done.wait(5.0)
            assert results == [float(i) for i in range(N)]
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
        assert thread is not None and thread.is_alive()
        d.close()
        assert not thread.is_alive()

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
            time.sleep(0.05)  # let worker process first item
            d.dispatch_reading(good_cb, _make_reading(), _FakeHandle())
            assert ok.wait(2.0)
        finally:
            d.close()

    def test_dispatch_after_close_is_silent(self):
        """Dispatching after close() silently drops instead of queueing to dead worker."""
        d = CallbackDispatcher(DispatchMode.WORKER)
        called = []

        def cb(reading, handle):
            called.append(1)

        d.dispatch_reading(cb, _make_reading(), _FakeHandle())
        time.sleep(0.05)
        d.close()
        # Should not raise or enqueue
        d.dispatch_reading(cb, _make_reading(), _FakeHandle())
        d.dispatch_error(lambda e, h: called.append(2), RuntimeError("x"), _FakeHandle())
        time.sleep(0.05)
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

        assert any("boom" in r.message for r in caplog.records)

    def test_close_is_noop(self):
        """close() on DIRECT dispatcher with no worker is safe."""
        d = CallbackDispatcher(DispatchMode.DIRECT)
        d.close()  # should not raise
