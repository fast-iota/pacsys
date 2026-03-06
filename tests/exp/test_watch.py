"""Tests for watch."""

import threading
import time

import pytest

from pacsys.exp._watch import watch
from pacsys.testing import FakeBackend


@pytest.fixture
def fake():
    return FakeBackend()


class TestWatch:
    def test_returns_triggering_reading(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 70.0)
            fake.emit_reading("M:OUTTMP@p,1000", 75.0)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        reading = watch(
            "M:OUTTMP@p,1000",
            lambda r: r.value > 72,
            timeout=1.0,
            backend=fake,
        )
        assert reading.value == 75.0

    def test_timeout_raises(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 70.0)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        with pytest.raises(TimeoutError, match="Condition not met"):
            watch(
                "M:OUTTMP@p,1000",
                lambda r: r.value > 100,
                timeout=0.1,
                backend=fake,
            )

    def test_condition_receives_reading_object(self, fake):
        """Predicate gets a Reading, not just value."""

        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        reading = watch(
            "M:OUTTMP@p,1000",
            lambda r: r.ok and r.value is not None,
            timeout=1.0,
            backend=fake,
        )
        assert reading.value == 72.5

    def test_first_reading_matches(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 100.0)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        reading = watch(
            "M:OUTTMP@p,1000",
            lambda r: r.value > 50,
            timeout=1.0,
            backend=fake,
        )
        assert reading.value == 100.0

    def test_predicate_exception_propagated(self, fake):
        """If the condition raises, watch re-raises it (not TimeoutError)."""

        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        with pytest.raises(TypeError):
            watch(
                "M:OUTTMP@p,1000",
                lambda r: r.value + "bad",  # TypeError
                timeout=1.0,
                backend=fake,
            )

    def test_stream_error_propagated(self, fake):
        """Stream errors are raised, not masked as TimeoutError."""

        def emitter():
            time.sleep(0.02)
            fake.emit_error(ConnectionError("lost connection"))

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        with pytest.raises(ConnectionError, match="lost connection"):
            watch(
                "M:OUTTMP@p,1000",
                lambda r: r.value > 72,
                timeout=1.0,
                backend=fake,
            )
