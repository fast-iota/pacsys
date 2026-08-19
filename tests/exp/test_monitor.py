"""Tests for Monitor live mode."""

import threading
import time

import pytest

from pacsys.exp._monitor import ChannelHealth, Monitor
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

    def test_stale_after_zero_raises(self):
        with pytest.raises(ValueError, match="stale_after must be positive"):
            Monitor(["M:OUTTMP@p,1000"], stale_after=0)

    def test_stale_after_negative_raises(self):
        with pytest.raises(ValueError, match="stale_after must be positive"):
            Monitor(["M:OUTTMP@p,1000"], stale_after=-1.0)

    @pytest.mark.parametrize("size", [0, -1])
    def test_invalid_buffer_size_raises(self, size):
        with pytest.raises(ValueError, match="buffer_size must be >= 1"):
            Monitor(["M:OUTTMP@p,1000"], buffer_size=size)


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

    @pytest.mark.parametrize("count", [0, -3])
    def test_collect_nonpositive_count_raises(self, fake, count):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        with pytest.raises(ValueError, match="count must be >= 1"):
            mon.collect(count=count)

    def test_collect_count_exceeding_buffer_size_raises(self, fake):
        """count > buffer_size can never be satisfied -- must fail fast, not hang."""
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake, buffer_size=5)
        with pytest.raises(ValueError, match="cannot exceed buffer_size"):
            mon.collect(count=10)

    def test_collect_count_subscription_stop_raises(self, fake):
        """A subscription that stops without error must raise, not spin forever."""

        def stopper():
            time.sleep(0.1)
            fake.emit_reading("M:OUTTMP@p,1000", 1.0)
            time.sleep(0.05)
            with fake._lock:
                subs = list(fake._subscriptions)
            for sub in subs:
                sub.stop()

        t = threading.Thread(target=stopper, daemon=True)
        t.start()
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        start = time.monotonic()
        # Failure must not wait for the collection timeout.
        with pytest.raises(RuntimeError, match="Subscription stopped"):
            mon.collect(count=5, timeout=5.0)
        assert time.monotonic() - start < 3.0
        t.join(timeout=2.0)


class TestMonitorTags:
    def test_tags_start_at_zero(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        assert mon.tags == {"M:OUTTMP@p,1000": 0}
        mon.stop()

    def test_tags_increment_on_readings(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        fake.emit_reading("M:OUTTMP@p,1000", 73.0)
        time.sleep(0.05)
        assert mon.tags == {"M:OUTTMP@p,1000": 2}
        mon.stop()

    def test_has_new_detects_change(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        old = mon.tags
        assert not mon.has_new(old)
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        time.sleep(0.05)
        assert mon.has_new(old)
        mon.stop()


class TestMonitorRunToken:
    def test_reading_racing_subscribe_return_is_counted(self, fake):
        """A delivery arriving before start() assigns self._handle is counted."""
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        orig = fake.subscribe

        def racing_subscribe(drfs, callback=None, **kw):
            h = orig(drfs, callback=callback, **kw)
            fake.emit_reading("M:OUTTMP@p,1000", 42.0)
            deadline = time.monotonic() + 2.0
            while mon.tags["M:OUTTMP@p,1000"] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)  # let the dispatcher deliver before subscribe() returns
            return h

        fake.subscribe = racing_subscribe
        try:
            mon.start()
            assert mon.tags == {"M:OUTTMP@p,1000": 1}
            assert mon._latest["M:OUTTMP@p,1000"].value == 42.0
        finally:
            fake.subscribe = orig
            mon.stop()

    def test_late_delivery_from_old_run_dropped(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        old_token = mon._run_token
        mon.stop()
        mon.start()
        try:
            from pacsys.types import Reading, ValueType

            stale = Reading(drf="M:OUTTMP@p,1000", value_type=ValueType.SCALAR, value=1.0)
            mon._on_reading(stale, None, old_token)
            assert mon.tags == {"M:OUTTMP@p,1000": 0}
        finally:
            mon.stop()


class TestMonitorAwaitNext:
    def test_await_next_returns_reading(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()

        def emit_later():
            time.sleep(0.05)
            fake.emit_reading("M:OUTTMP@p,1000", 99.0)

        t = threading.Thread(target=emit_later, daemon=True)
        t.start()
        reading = mon.await_next("M:OUTTMP@p,1000", timeout=2.0)
        assert reading.value == 99.0
        mon.stop()

    def test_await_next_timeout_raises(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        with pytest.raises(TimeoutError):
            mon.await_next("M:OUTTMP@p,1000", timeout=0.05)
        mon.stop()

    def test_await_next_skips_already_buffered(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        fake.emit_reading("M:OUTTMP@p,1000", 1.0)
        time.sleep(0.05)

        def emit_later():
            time.sleep(0.05)
            fake.emit_reading("M:OUTTMP@p,1000", 2.0)

        t = threading.Thread(target=emit_later, daemon=True)
        t.start()
        reading = mon.await_next("M:OUTTMP@p,1000", timeout=2.0)
        assert reading.value == 2.0
        mon.stop()

    def test_await_next_not_running_raises(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        with pytest.raises(RuntimeError, match="not running"):
            mon.await_next("M:OUTTMP@p,1000", timeout=1.0)

    def test_await_next_across_restart(self, fake):
        """A waiter blocked across stop()/start() either gets the clean not-running
        error or the new run's reading - never the internal counter error or a hang,
        even when the new run reaches the old baseline count."""
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        for i in range(5):
            fake.emit_reading("M:OUTTMP@p,1000", float(i))
        time.sleep(0.05)
        result = {}

        def waiter():
            try:
                result["reading"] = mon.await_next("M:OUTTMP@p,1000", timeout=3.0)
            except RuntimeError as e:
                result["exc"] = e

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.1)
        mon.stop()
        mon.start()
        for i in range(5):  # exactly the old baseline count
            fake.emit_reading("M:OUTTMP@p,1000", 100.0 + i)
        t.join(timeout=4.0)
        assert not t.is_alive()
        mon.stop()
        if "exc" in result:
            assert "not running" in str(result["exc"])
        else:
            assert result["reading"].value >= 100.0


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


class TestChannelHealth:
    def test_gap_returns_inf_when_never_received(self):
        ch = ChannelHealth(drf="X:TEST@p,1000", last_reading=None, last_received_at=None, total_received=0, stale=False)
        assert ch.gap == float("inf")

    def test_gap_returns_elapsed_seconds(self):
        now = time.monotonic()
        ch = ChannelHealth(
            drf="X:TEST@p,1000", last_reading=None, last_received_at=now - 2.5, total_received=1, stale=False
        )
        assert ch.gap >= 2.5

    def test_frozen(self):
        ch = ChannelHealth(drf="X:TEST@p,1000", last_reading=None, last_received_at=None, total_received=0, stale=False)
        with pytest.raises(AttributeError):
            ch.stale = True  # type: ignore[misc]


class TestMonitorHealthOnDemand:
    def test_health_before_start(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        h = mon.health("M:OUTTMP@p,1000")
        assert isinstance(h, ChannelHealth)
        assert h.last_reading is None
        assert h.last_received_at is None
        assert h.total_received == 0
        assert h.gap == float("inf")

    def test_health_after_reading(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        time.sleep(0.05)
        h = mon.health("M:OUTTMP@p,1000")
        mon.stop()
        assert h.last_reading is not None
        assert h.last_reading.value == 72.0
        assert h.last_received_at is not None
        assert h.gap < 1.0
        assert h.total_received == 1

    def test_health_all_channels(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000", "G:AMANDA@p,1000"], backend=fake)
        mon.start()
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        time.sleep(0.05)
        result = mon.health()
        mon.stop()
        assert isinstance(result, dict)
        assert len(result) == 2
        assert result["M:OUTTMP@p,1000"].total_received == 1
        assert result["G:AMANDA@p,1000"].total_received == 0

    def test_health_unknown_channel_raises(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        with pytest.raises(KeyError, match="No channel"):
            mon.health("Z:FAKE@p,1000")

    def test_stale_false_when_no_threshold(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        h = mon.health("M:OUTTMP@p,1000")
        assert h.stale is False


class TestMonitorStaleness:
    def test_stale_after_threshold(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], stale_after=0.1, backend=fake)
        mon.start()
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        time.sleep(0.05)
        assert mon.health("M:OUTTMP@p,1000").stale is False
        time.sleep(0.15)  # exceed threshold
        assert mon.health("M:OUTTMP@p,1000").stale is True
        mon.stop()

    def test_grace_period_no_false_stale_at_startup(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], stale_after=0.5, backend=fake)
        mon.start()
        # Immediately after start, not stale (grace period)
        h = mon.health("M:OUTTMP@p,1000")
        assert h.stale is False
        mon.stop()

    def test_grace_period_expires_without_data(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], stale_after=0.1, backend=fake)
        mon.start()
        time.sleep(0.15)  # grace period expired, no data ever received
        h = mon.health("M:OUTTMP@p,1000")
        assert h.stale is True
        mon.stop()

    def test_flush_does_not_reset_health_state(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], stale_after=5.0, backend=fake)
        mon.start()
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        time.sleep(0.05)
        mon.flush()
        h = mon.health("M:OUTTMP@p,1000")
        assert h.total_received == 1  # counters survive flush
        assert h.last_received_at is not None  # timestamps survive flush
        mon.stop()


class TestMonitorRestart:
    def test_start_resets_health_state(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], stale_after=5.0, backend=fake)
        mon.start()
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        time.sleep(0.05)
        assert mon.health("M:OUTTMP@p,1000").total_received == 1
        mon.stop()
        mon.start()
        h = mon.health("M:OUTTMP@p,1000")
        assert h.total_received == 0
        assert h.last_reading is None
        assert h.last_received_at is None
        assert h.stale is False  # grace period active
        mon.stop()

    def test_restart_from_on_stale_callback(self, fake):
        """The documented auto-resubscribe pattern: on_stale restarts the monitor
        from the watchdog thread itself. Must leave one live watchdog, not zero or two."""
        restarted = threading.Event()

        def on_stale(drf, h):
            if not restarted.is_set():
                mon.stop()
                mon.start()
                restarted.set()

        mon = Monitor(["M:OUTTMP@p,1000"], stale_after=0.1, on_stale=on_stale, backend=fake)
        mon.start()
        assert restarted.wait(timeout=2.0)
        time.sleep(0.3)  # let the superseded watchdog exit
        assert mon.running
        watchdogs = [t for t in threading.enumerate() if t.name == "pacsys-watchdog" and t.is_alive()]
        assert len(watchdogs) == 1
        mon.stop()

    def test_restart_resets_stale_set(self, fake):
        stale_events = []
        mon = Monitor(
            ["M:OUTTMP@p,1000"],
            stale_after=0.1,
            on_stale=lambda drf, h: stale_events.append(drf),
            backend=fake,
        )
        mon.start()
        time.sleep(0.25)  # go stale
        assert len(stale_events) == 1
        mon.stop()
        stale_events.clear()
        mon.start()
        time.sleep(0.25)  # should go stale again (not suppressed)
        mon.stop()
        assert len(stale_events) == 1


class TestMonitorWatchdog:
    def test_on_stale_fires_once(self, fake):
        stale_events = []
        mon = Monitor(
            ["M:OUTTMP@p,1000"],
            stale_after=0.1,
            on_stale=lambda drf, h: stale_events.append((drf, h)),
            backend=fake,
        )
        mon.start()
        # No data — should go stale after grace period
        time.sleep(0.4)
        mon.stop()
        assert len(stale_events) == 1
        assert stale_events[0][0] == "M:OUTTMP@p,1000"
        assert stale_events[0][1].stale is True

    def test_on_stale_not_repeated(self, fake):
        stale_events = []
        mon = Monitor(
            ["M:OUTTMP@p,1000"],
            stale_after=0.1,
            on_stale=lambda drf, h: stale_events.append(drf),
            backend=fake,
        )
        mon.start()
        time.sleep(0.5)  # well past threshold, multiple watchdog cycles
        mon.stop()
        assert len(stale_events) == 1  # edge-triggered, not repeated

    def test_on_recover_fires_on_recovery(self, fake):
        stale_events = []
        recover_events = []
        mon = Monitor(
            ["M:OUTTMP@p,1000"],
            stale_after=0.1,
            on_stale=lambda drf, h: stale_events.append(drf),
            on_recover=lambda drf, h: recover_events.append(drf),
            backend=fake,
        )
        mon.start()
        time.sleep(0.25)  # go stale
        assert len(stale_events) == 1
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        time.sleep(0.15)  # watchdog detects recovery
        mon.stop()
        assert len(recover_events) == 1

    def test_no_on_recover_no_crash(self, fake):
        """Recovery with on_recover=None should not crash."""
        mon = Monitor(
            ["M:OUTTMP@p,1000"],
            stale_after=0.1,
            on_stale=lambda drf, h: None,
            backend=fake,
        )
        mon.start()
        time.sleep(0.25)  # go stale
        fake.emit_reading("M:OUTTMP@p,1000", 72.0)
        time.sleep(0.15)  # recover without on_recover
        mon.stop()  # should not raise

    def test_watchdog_thread_exits_on_stop(self, fake):
        mon = Monitor(
            ["M:OUTTMP@p,1000"],
            stale_after=0.1,
            backend=fake,
        )
        mon.start()
        assert mon._watchdog is not None
        assert mon._watchdog.is_alive()
        mon.stop()
        time.sleep(0.2)
        assert not mon._watchdog.is_alive()

    def test_no_watchdog_without_stale_after(self, fake):
        mon = Monitor(["M:OUTTMP@p,1000"], backend=fake)
        mon.start()
        assert mon._watchdog is None
        mon.stop()

    def test_callback_exception_does_not_crash_watchdog(self, fake):
        call_count = []

        def bad_callback(drf, h):
            call_count.append(1)
            raise RuntimeError("boom")

        mon = Monitor(
            ["M:OUTTMP@p,1000", "G:AMANDA@p,1000"],
            stale_after=0.1,
            on_stale=bad_callback,
            backend=fake,
        )
        mon.start()
        time.sleep(0.4)  # both channels should go stale
        mon.stop()
        # Both channels should have triggered despite exception
        assert len(call_count) == 2
