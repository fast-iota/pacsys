"""Tests for Monitor live mode."""

import threading
import time

import pytest

from pacsys.exp._monitor import Monitor
from pacsys.testing import FakeBackend


@pytest.fixture
def fake():
    fb = FakeBackend()
    fb.set_reading("M:OUTTMP", 72.0)
    fb.set_reading("G:AMANDA", 1.0)
    return fb


class TestMonitorInit:
    def test_empty_devices_raises(self):
        with pytest.raises(ValueError, match="devices cannot be empty"):
            Monitor([])

    def test_accepts_drf_strings(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        assert not mon.running


class TestMonitorLiveMode:
    def test_start_stop(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        assert mon.running
        mon.stop()
        assert not mon.running

    def test_double_start_raises(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        with pytest.raises(RuntimeError, match="already running"):
            mon.start()
        mon.stop()

    def test_snapshot_returns_collected_readings(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        fake.emit_reading("M:OUTTMP@p,1000", 73.0)
        time.sleep(0.05)  # let callbacks deliver
        snap = mon.snapshot()
        mon.stop()
        assert len(snap.channels["M:OUTTMP@p,1000"].readings) == 2

    def test_flush_resets_buffers(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        time.sleep(0.05)
        result = mon.flush()
        assert len(result.channels["M:OUTTMP@p,1000"].readings) == 1
        # Buffer is now empty
        snap = mon.snapshot()
        assert len(snap.channels["M:OUTTMP@p,1000"].readings) == 0
        mon.stop()

    def test_context_manager(self, fake):
        with Monitor(["M:OUTTMP@p,1000"], backend=fake) as mon:
            fake.emit_reading("M:OUTTMP@p,1000", 72.0)
            time.sleep(0.05)
            snap = mon.snapshot()
        assert not mon.running
        assert len(snap.channels["M:OUTTMP@p,1000"].readings) == 1


class TestMonitorCollect:
    def test_collect_duration(self, fake):
        def emitter():
            for i in range(5):
                fake.emit_reading("M:OUTTMP@p,1000", float(i))
                time.sleep(0.02)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        result = Monitor(["M:OUTTMP@p,1000"], backend=fake).collect(duration=0.15)
        assert len(result.channels["M:OUTTMP@p,1000"].readings) >= 1

    def test_collect_count(self, fake):
        def emitter():
            for i in range(10):
                fake.emit_reading("M:OUTTMP@p,1000", float(i))
                time.sleep(0.01)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        result = Monitor(["M:OUTTMP@p,1000"], backend=fake).collect(count=3, timeout=2.0)
        assert len(result.channels["M:OUTTMP@p,1000"].readings) >= 3

    def test_collect_requires_exactly_one_arg(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        with pytest.raises(ValueError, match="Exactly one"):
            mon.collect()
        with pytest.raises(ValueError, match="Exactly one"):
            mon.collect(duration=1.0, count=10)


class TestMonitorBufferSize:
    def test_ring_buffer_evicts_oldest(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], buffer_size=3, backend=fake)
        mon.start()
        for i in range(5):
            fake.emit_reading("M:OUTTMP@p,1000", float(i))
        time.sleep(0.05)
        snap = mon.snapshot()
        mon.stop()
        vals = snap.values("M:OUTTMP@p,1000")
        assert vals == [2.0, 3.0, 4.0]
