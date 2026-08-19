"""Tests for BufferedSubscriptionHandle (sync)."""

import logging
import threading
import time

import pytest

from pacsys.backends._subscription import BufferedSubscriptionHandle
from pacsys.types import Reading, ValueType


@pytest.fixture
def make_reading():
    def _make(val):
        return Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=val, error_code=0)

    return _make


@pytest.fixture
def handle():
    return BufferedSubscriptionHandle()


def _track_next_wait(condition: threading.Condition) -> threading.Event:
    waiting = threading.Event()
    original_wait = condition.wait

    def tracked_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    condition.wait = tracked_wait
    return waiting


# =============================================================================
# Core dispatch + iteration
# =============================================================================


class TestDispatchAndIterate:
    def test_dispatch_and_iterate(self, handle, make_reading):
        """Readings arrive in order; tuple contains the handle itself."""
        handle._dispatch(make_reading(1.0))
        handle._dispatch(make_reading(2.0))
        handle._signal_stop()

        results = []
        for reading, h in handle.readings():
            results.append(reading.value)
            assert h is handle
        assert results == [1.0, 2.0]

    def test_drains_before_stop(self, handle, make_reading):
        """All buffered readings are yielded before honoring stop."""
        handle._dispatch(make_reading(10.0))
        handle._dispatch(make_reading(20.0))
        handle._dispatch(make_reading(30.0))
        handle._signal_stop()

        results = [r.value for r, _ in handle.readings()]
        assert results == [10.0, 20.0, 30.0]

    def test_dispatch_after_stop_is_noop(self, handle, make_reading):
        """Dispatch after stop silently discards the reading."""
        handle._signal_stop()
        handle._dispatch(make_reading(99.0))

        results = list(handle.readings())
        assert results == []


# =============================================================================
# Error propagation
# =============================================================================


class TestErrorPropagation:
    def test_error_after_buffered_readings(self, handle, make_reading):
        """Buffered readings are drained, then the error is raised."""
        handle._dispatch(make_reading(1.0))
        handle._signal_error(RuntimeError("boom"))

        results = []
        with pytest.raises(RuntimeError, match="boom"):  # noqa: PT012 -- verifies buffered values first
            for reading, _ in handle.readings():
                results.append(reading.value)
        assert results == [1.0]

    def test_error_on_empty_buffer(self, handle):
        """Error with no buffered data raises immediately."""
        handle._signal_error(RuntimeError("immediate"))

        with pytest.raises(RuntimeError, match="immediate"):
            list(handle.readings())

    def test_first_error_wins(self, handle):
        """Only the first error is stored."""
        handle._signal_error(RuntimeError("first"))
        handle._signal_error(ValueError("second"))

        assert isinstance(handle.exc, RuntimeError)
        assert str(handle.exc) == "first"

    def test_drains_before_raising_error(self, handle, make_reading):
        """All buffered readings are yielded before the exception."""
        handle._dispatch(make_reading(1.0))
        handle._dispatch(make_reading(2.0))
        handle._signal_error(ValueError("late"))

        results = []
        with pytest.raises(ValueError, match="late"):  # noqa: PT012 -- verifies buffered values first
            for reading, _ in handle.readings():
                results.append(reading.value)
        assert results == [1.0, 2.0]


# =============================================================================
# Buffer overflow
# =============================================================================


class TestBufferOverflow:
    def test_overflow_drops_newest(self, handle, make_reading):
        """When buffer is full, newest readings are dropped (oldest survive)."""
        handle._maxsize = 3
        for i in range(5):
            handle._dispatch(make_reading(float(i)))
        handle._signal_stop()

        results = [r.value for r, _ in handle.readings()]
        assert results == [0.0, 1.0, 2.0]

    def test_overflow_drop_logging_throttled(self, handle, make_reading, caplog):
        """Drop warnings are throttled to once per 5s window."""
        handle._maxsize = 1
        handle._dispatch(make_reading(0.0))  # fills buffer

        with caplog.at_level(logging.WARNING, logger="pacsys.backends._subscription"):
            handle._dispatch(make_reading(1.0))  # dropped, logs warning
            handle._dispatch(make_reading(2.0))  # dropped, throttled (same 5s window)

        warnings = [r for r in caplog.records if "buffer full" in r.message.lower()]
        assert len(warnings) == 1


# =============================================================================
# Timeout behavior
# =============================================================================


class TestTimeout:
    def test_timeout_zero_nonblocking_drain(self, handle, make_reading):
        """timeout=0 drains buffered readings and returns immediately."""
        handle._dispatch(make_reading(1.0))
        handle._dispatch(make_reading(2.0))

        results = [r.value for r, _ in handle.readings(timeout=0)]
        assert results == [1.0, 2.0]

    def test_timeout_zero_empty_buffer_returns_immediately(self, handle):
        """timeout=0 with empty buffer returns with no values, no blocking."""
        t0 = time.monotonic()
        results = list(handle.readings(timeout=0))
        elapsed = time.monotonic() - t0
        assert results == []
        assert elapsed < 0.5

    def test_timeout_expires_with_no_data(self, handle):
        """timeout>0 returns after wall-clock expiry, yields nothing."""
        t0 = time.monotonic()
        results = list(handle.readings(timeout=0.1))
        elapsed = time.monotonic() - t0
        assert results == []
        assert elapsed >= 0.05  # waited at least some time
        assert elapsed < 2.0  # didn't block forever


# =============================================================================
# Stop semantics
# =============================================================================


class TestStopSemantics:
    def test_stopped_transitions(self, handle):
        """stopped is False after init, True after stop."""
        assert handle.stopped is False
        handle._signal_stop()
        assert handle.stopped is True

    def test_signal_stop_idempotent(self, handle):
        """Calling _signal_stop() twice is safe."""
        handle._signal_stop()
        handle._signal_stop()
        assert handle.stopped is True

    def test_signal_error_sets_stopped(self, handle):
        """Error signal also sets stopped."""
        handle._signal_error(RuntimeError("x"))
        assert handle.stopped is True


# =============================================================================
# Properties
# =============================================================================


class TestProperties:
    def test_ref_ids_defensive_copy(self, handle):
        """Mutating ref_ids return value doesn't affect the handle."""
        handle._ref_ids = [1, 2, 3]
        ids = handle.ref_ids
        ids.append(99)
        assert handle.ref_ids == [1, 2, 3]

    def test_exc_none_initially(self, handle):
        assert handle.exc is None

    def test_exc_after_error(self, handle):
        handle._signal_error(ValueError("v"))
        assert isinstance(handle.exc, ValueError)


# =============================================================================
# Callback mode guard
# =============================================================================


class TestCallbackModeGuard:
    def test_readings_raises_in_callback_mode(self, handle):
        """readings() raises RuntimeError when handle is in callback mode."""
        handle._is_callback_mode = True
        with pytest.raises(RuntimeError, match="callback"):
            list(handle.readings())


# =============================================================================
# Concurrency
# =============================================================================


class TestConcurrency:
    def test_concurrent_producer_consumer(self, handle, make_reading):
        """Consumer in a thread receives all readings dispatched by producer."""
        n = 20
        results = []
        waiting = _track_next_wait(handle._cond)

        def consume():
            for reading, _ in handle.readings(timeout=5.0):
                results.append(reading.value)

        consumer = threading.Thread(target=consume)
        consumer.start()
        assert waiting.wait(1.0)

        for i in range(n):
            handle._dispatch(make_reading(float(i)))
        handle._signal_stop()

        consumer.join(timeout=5.0)
        assert not consumer.is_alive()
        assert results == [float(i) for i in range(n)]

    def test_data_arriving_during_wait(self, handle, make_reading):
        """Reader blocked on empty buffer receives data when dispatched."""
        results = []
        waiting = _track_next_wait(handle._cond)

        def consume():
            for reading, _ in handle.readings(timeout=5.0):
                results.append(reading.value)

        consumer = threading.Thread(target=consume)
        consumer.start()

        assert waiting.wait(1.0)
        handle._dispatch(make_reading(42.0))
        handle._signal_stop()

        consumer.join(timeout=5.0)
        assert not consumer.is_alive()
        assert results == [42.0]

    def test_error_while_blocked(self, handle, make_reading):
        """Error signal wakes a consumer blocked on empty buffer."""
        results = []
        exc_caught = []
        waiting = _track_next_wait(handle._cond)

        def consume():
            try:
                for reading, _ in handle.readings(timeout=5.0):
                    results.append(reading.value)
            except RuntimeError as e:
                exc_caught.append(e)

        consumer = threading.Thread(target=consume)
        consumer.start()

        assert waiting.wait(1.0)
        handle._signal_error(RuntimeError("injected"))

        consumer.join(timeout=5.0)
        assert not consumer.is_alive()
        assert results == []
        assert len(exc_caught) == 1
        assert str(exc_caught[0]) == "injected"

    def test_stop_while_blocked(self, handle):
        """Stop signal wakes a consumer blocked on empty buffer."""
        results = []
        waiting = _track_next_wait(handle._cond)

        def consume():
            for reading, _ in handle.readings(timeout=5.0):
                results.append(reading.value)

        consumer = threading.Thread(target=consume)
        consumer.start()

        assert waiting.wait(1.0)
        handle._signal_stop()

        consumer.join(timeout=5.0)
        assert not consumer.is_alive()
        assert results == []
