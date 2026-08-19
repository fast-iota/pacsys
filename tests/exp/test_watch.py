"""Tests for watch."""

import threading
from contextlib import contextmanager

import pytest

from pacsys.exp._watch import watch
from pacsys.testing import FakeBackend


@pytest.fixture
def fake():
    return FakeBackend()


@contextmanager
def _emit_after_subscription(fake, emit):
    subscribed = []
    waiting = threading.Event()

    def run():
        waiting.set()
        subscribed.append(fake.wait_for_subscription("M:OUTTMP@p,1000"))
        if subscribed[-1]:
            emit()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert waiting.wait(timeout=1.0)
    try:
        yield
    finally:
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert subscribed == [True]


class TestWatch:
    def test_returns_triggering_reading(self, fake):
        def emit():
            fake.emit_reading("M:OUTTMP@p,1000", 70.0)
            fake.emit_reading("M:OUTTMP@p,1000", 75.0)

        with _emit_after_subscription(fake, emit):
            reading = watch(
                "M:OUTTMP@p,1000",
                lambda r: r.value > 72,
                timeout=1.0,
                backend=fake,
            )
        assert reading.value == 75.0

    def test_timeout_raises(self, fake):
        def emit():
            fake.emit_reading("M:OUTTMP@p,1000", 70.0)

        with _emit_after_subscription(fake, emit), pytest.raises(TimeoutError, match="Condition not met"):
            watch("M:OUTTMP@p,1000", lambda r: r.value > 100, timeout=0.1, backend=fake)

    def test_condition_receives_reading_object(self, fake):
        """Predicate gets a Reading, not just value."""

        def emit():
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)

        with _emit_after_subscription(fake, emit):
            reading = watch(
                "M:OUTTMP@p,1000",
                lambda r: r.ok and r.value is not None,
                timeout=1.0,
                backend=fake,
            )
        assert reading.value == 72.5

    def test_first_reading_matches(self, fake):
        def emit():
            fake.emit_reading("M:OUTTMP@p,1000", 100.0)

        with _emit_after_subscription(fake, emit):
            reading = watch("M:OUTTMP@p,1000", lambda r: r.value > 50, timeout=1.0, backend=fake)
        assert reading.value == 100.0

    def test_predicate_exception_propagated(self, fake):
        """If the condition raises, watch re-raises it (not TimeoutError)."""

        def emit():
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)

        with _emit_after_subscription(fake, emit), pytest.raises(TypeError):
            watch(
                "M:OUTTMP@p,1000",
                lambda r: r.value + "bad",  # TypeError
                timeout=1.0,
                backend=fake,
            )

    def test_stream_error_propagated(self, fake):
        """Stream errors are raised, not masked as TimeoutError."""

        def emit():
            fake.emit_error(ConnectionError("lost connection"))

        with _emit_after_subscription(fake, emit), pytest.raises(ConnectionError, match="lost connection"):
            watch(
                "M:OUTTMP@p,1000",
                lambda r: r.value > 72,
                timeout=1.0,
                backend=fake,
            )
