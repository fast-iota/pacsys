"""Tests for read_fresh."""

import threading
import time

import pytest

from pacsys.exp._read_fresh import read_fresh
from pacsys.testing import FakeBackend


@pytest.fixture
def fake():
    return FakeBackend()


class TestReadFresh:
    def test_basic(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(["M:OUTTMP@p,1000"], timeout=1.0, backend=fake)
        assert len(results) == 1
        assert results[0].value == 72.5

    def test_multiple_channels(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)
            fake.emit_reading("G:AMANDA@p,1000", 1.5)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(
            ["M:OUTTMP@p,1000", "G:AMANDA@p,1000"],
            timeout=1.0,
            backend=fake,
        )
        assert len(results) == 2
        assert results[0].value == 72.5
        assert results[1].value == 1.5

    def test_timeout_raises(self, fake):
        with pytest.raises(TimeoutError, match="Timed out"):
            read_fresh(["M:OUTTMP@p,1000"], timeout=0.1, backend=fake)

    def test_default_event_applied(self, fake):
        """default_event is applied to DRFs without events."""

        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP.READING@p,1000", 72.5)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(
            ["M:OUTTMP"],
            default_event="p,1000",
            timeout=1.0,
            backend=fake,
        )
        assert results[0].value == 72.5

    def test_empty_devices_raises(self, fake):
        with pytest.raises(ValueError, match="devices cannot be empty"):
            read_fresh([], backend=fake)

    def test_preserves_order(self, fake):
        def emitter():
            time.sleep(0.02)
            # Emit in reverse order
            fake.emit_reading("B:DEV@p,1000", 2.0)
            fake.emit_reading("A:DEV@p,1000", 1.0)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(
            ["A:DEV@p,1000", "B:DEV@p,1000"],
            timeout=1.0,
            backend=fake,
        )
        assert results[0].value == 1.0
        assert results[1].value == 2.0

    def test_only_takes_first_reading_per_channel(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)
            fake.emit_reading("M:OUTTMP@p,1000", 99.0)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(["M:OUTTMP@p,1000"], timeout=1.0, backend=fake)
        assert results[0].value == 72.5
