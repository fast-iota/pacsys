"""
Tests for pacsys.types module.
"""

import threading
import time
from datetime import datetime

import pytest

from pacsys.testing import FakeBackend
from pacsys.types import (
    CombinedStream,
    ValueType,
    Reading,
)


class TestReading:
    """Tests for Reading dataclass."""

    @pytest.mark.parametrize(
        "error_code,is_success,is_warning,is_error",
        [
            (0, True, False, False),  # success
            (1, False, True, False),  # warning
            (100, False, True, False),  # warning (high)
            (-1, False, False, True),  # error
            (-42, False, False, True),  # error (other)
        ],
    )
    def test_status_flags(self, error_code, is_success, is_warning, is_error):
        """Reading status flags reflect error_code correctly."""
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, error_code=error_code)
        assert reading.is_success is is_success
        assert reading.is_warning is is_warning
        assert reading.is_error is is_error

    @pytest.mark.parametrize(
        "error_code,value,expected_ok",
        [
            (0, 72.5, True),  # success + value = ok
            (0, None, False),  # success + no value = not ok
            (1, 72.5, True),  # warning + value = ok
            (1, None, False),  # warning + no value = not ok
            (-1, 72.5, False),  # error + value = not ok
            (-1, None, False),  # error + no value = not ok
        ],
    )
    def test_ok_property(self, error_code, value, expected_ok):
        """Reading.ok requires non-negative error_code AND value is not None."""
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, error_code=error_code, value=value)
        assert reading.ok is expected_ok

    @pytest.mark.parametrize(
        "drf,expected_name",
        [
            ("M:OUTTMP", "M:OUTTMP"),  # simple
            ("M:OUTTMP@p,1000", "M:OUTTMP"),  # with event
            ("B:HS23T[0:10]", "B:HS23T"),  # with range
            ("B:HS23T[0:10]@p,1000", "B:HS23T"),  # with range and event
        ],
    )
    def test_name_from_drf(self, drf, expected_name):
        """Reading.name extracts device name from DRF."""
        reading = Reading(drf=drf, value_type=ValueType.SCALAR)
        assert reading.name == expected_name


class TestCombinedStream:
    """Tests for CombinedStream."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            CombinedStream([])

    def test_context_manager_stops_all(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])

        with CombinedStream([h1, h2]) as cs:
            assert not cs.stopped

        assert h1.stopped
        assert h2.stopped

    def test_stopped_property(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])
        cs = CombinedStream([h1, h2])

        assert not cs.stopped
        h1.stop()
        assert not cs.stopped
        h2.stop()
        assert cs.stopped

    def test_exc_property(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])
        cs = CombinedStream([h1, h2])

        assert cs.exc is None
        err = RuntimeError("boom")
        fake.emit_error(err)
        assert cs.exc is err

    # --- non-blocking mode (timeout=0) ---

    def test_nonblocking_empty(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        results = list(CombinedStream([h1]).readings(timeout=0))
        assert results == []
        h1.stop()

    def test_nonblocking_single_sub(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        fake.emit_reading("M:OUTTMP", 1.0)
        fake.emit_reading("M:OUTTMP", 2.0)

        results = list(CombinedStream([h1]).readings(timeout=0))
        assert len(results) == 2
        assert results[0][0].value == 1.0
        assert results[1][0].value == 2.0
        h1.stop()

    def test_nonblocking_sorts_by_timestamp(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])

        t_early = datetime(2025, 1, 1, 0, 0, 0)
        t_late = datetime(2025, 1, 1, 0, 0, 1)

        fake.emit_reading("G:AMANDA", 10.0, timestamp=t_late)
        fake.emit_reading("M:OUTTMP", 20.0, timestamp=t_early)

        results = list(CombinedStream([h1, h2]).readings(timeout=0))
        assert len(results) == 2
        assert results[0][0].value == 20.0  # earlier timestamp first
        assert results[1][0].value == 10.0
        h1.stop()
        h2.stop()

    def test_nonblocking_skips_stopped_subs(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])

        fake.emit_reading("M:OUTTMP", 1.0)
        fake.emit_reading("G:AMANDA", 2.0)
        h1.stop()

        results = list(CombinedStream([h1, h2]).readings(timeout=0))
        assert len(results) == 1
        assert results[0][0].value == 2.0
        h2.stop()

    def test_nonblocking_raises_on_error(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        err = RuntimeError("boom")
        fake.emit_error(err)

        with pytest.raises(RuntimeError, match="boom"):
            list(CombinedStream([h1]).readings(timeout=0))

    def test_nonblocking_returns_correct_handle(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])
        fake.emit_reading("M:OUTTMP", 1.0)
        fake.emit_reading("G:AMANDA", 2.0)

        results = list(CombinedStream([h1, h2]).readings(timeout=0))
        handles = {r[0].drf: r[1] for r in results}
        assert handles["M:OUTTMP"] is h1
        assert handles["G:AMANDA"] is h2
        h1.stop()
        h2.stop()

    # --- blocking mode ---

    def test_blocking_receives_readings(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])

        def emit_then_stop():
            fake.emit_reading("M:OUTTMP", 42.0)
            h1.stop()

        threading.Timer(0.05, emit_then_stop).start()

        results = list(CombinedStream([h1]).readings(timeout=2))
        assert len(results) == 1
        assert results[0][0].value == 42.0

    def test_blocking_multiple_subs(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])
        cs = CombinedStream([h1, h2])

        def emit_then_stop():
            fake.emit_reading("M:OUTTMP", 1.0)
            fake.emit_reading("G:AMANDA", 2.0)
            h1.stop()
            h2.stop()

        threading.Timer(0.05, emit_then_stop).start()

        results = list(cs.readings(timeout=2))
        values = sorted(r[0].value for r in results)
        assert values == [1.0, 2.0]

    def test_blocking_sorts_by_timestamp(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])

        t_early = datetime(2025, 1, 1, 0, 0, 0)
        t_late = datetime(2025, 1, 1, 0, 0, 1)

        # Emit both readings first so they're buffered in the queues,
        # then stop - this avoids timing-dependent batch boundaries.
        fake.emit_reading("G:AMANDA", 10.0, timestamp=t_late)
        fake.emit_reading("M:OUTTMP", 20.0, timestamp=t_early)

        def stop_later():
            time.sleep(0.05)
            h1.stop()
            h2.stop()

        threading.Thread(target=stop_later, daemon=True).start()

        results = list(CombinedStream([h1, h2]).readings(timeout=2))
        assert len(results) == 2
        assert results[0][0].value == 20.0  # earlier timestamp first
        assert results[1][0].value == 10.0

    def test_blocking_timeout_expires(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])

        results = list(CombinedStream([h1]).readings(timeout=0.2))
        assert results == []
        h1.stop()

    def test_blocking_error_propagation(self):
        """Error set before iteration starts is raised by feeder."""
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        fake.emit_error(RuntimeError("fail"))

        with pytest.raises(RuntimeError, match="fail"):
            list(CombinedStream([h1]).readings(timeout=2))

    def test_blocking_error_available_after_stop(self):
        """Error set mid-stream is accessible via .exc after readings end."""
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        err = RuntimeError("late")
        cs = CombinedStream([h1])

        def error_then_stop():
            fake.emit_error(err)

        threading.Timer(0.05, error_then_stop).start()

        list(cs.readings(timeout=2))
        assert cs.exc is err

    def test_blocking_none_timeout_stops_when_all_done(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])

        def emit_then_stop():
            fake.emit_reading("M:OUTTMP", 99.0)
            h1.stop()

        threading.Timer(0.05, emit_then_stop).start()

        results = list(CombinedStream([h1]).readings(timeout=None))
        assert len(results) == 1
        assert results[0][0].value == 99.0

    def test_stop_terminates_blocking_readings(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        cs = CombinedStream([h1])

        threading.Timer(0.1, cs.stop).start()

        results = list(cs.readings(timeout=5))
        assert results == []
