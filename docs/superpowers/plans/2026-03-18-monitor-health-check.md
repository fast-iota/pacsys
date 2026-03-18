# Monitor Health Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-channel health tracking and a watchdog thread to `Monitor` that detects stale channels and fires callbacks.

**Architecture:** Health metadata (`_received_at` monotonic timestamps) is always tracked in `_on_reading`. A `ChannelHealth` frozen dataclass exposes per-channel state via `mon.health(drf)`. An optional watchdog daemon thread (enabled by `stale_after=`) polls channels and fires edge-triggered `on_stale`/`on_recover` callbacks outside the lock.

**Tech Stack:** Python stdlib only (`threading`, `time`, `logging`, `dataclasses`). Tests use `FakeBackend` from `pacsys.testing`.

**Spec:** `docs/superpowers/specs/2026-03-18-monitor-health-check-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pacsys/exp/_monitor.py` | `ChannelHealth` dataclass + all Monitor modifications |
| `pacsys/exp/__init__.py` | Export `ChannelHealth` |
| `tests/exp/test_monitor.py` | All health check tests (appended to existing file) |

---

### Task 1: Add `ChannelHealth` dataclass

**Files:**
- Modify: `pacsys/exp/_monitor.py` (add class after `ChannelData`, before `MonitorResult`)
- Modify: `pacsys/exp/__init__.py` (add export)
- Test: `tests/exp/test_monitor.py`

- [ ] **Step 1: Write failing tests for `ChannelHealth`**

Append to `tests/exp/test_monitor.py`:

```python
from pacsys.exp._monitor import ChannelHealth


class TestChannelHealth:
    def test_gap_returns_inf_when_never_received(self):
        ch = ChannelHealth(drf="X:TEST@p,1000", last_reading=None, last_received_at=None, total_received=0, stale=False)
        assert ch.gap == float("inf")

    def test_gap_returns_elapsed_seconds(self):
        now = time.monotonic()
        ch = ChannelHealth(drf="X:TEST@p,1000", last_reading=None, last_received_at=now - 2.5, total_received=1, stale=False)
        assert ch.gap >= 2.5

    def test_frozen(self):
        ch = ChannelHealth(drf="X:TEST@p,1000", last_reading=None, last_received_at=None, total_received=0, stale=False)
        with pytest.raises(AttributeError):
            ch.stale = True  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/exp/test_monitor.py::TestChannelHealth -v -x 2>&1 | tail -10`
Expected: FAIL — `ChannelHealth` not defined

- [ ] **Step 3: Implement `ChannelHealth`**

Add to `pacsys/exp/_monitor.py`, after the `ChannelData` class (before `MonitorResult`):

```python
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelHealth:
    """Per-channel health snapshot."""

    drf: str
    last_reading: Reading | None
    last_received_at: float | None  # time.monotonic() value
    total_received: int
    stale: bool

    @property
    def gap(self) -> float:
        """Seconds since last reading, or inf if never received."""
        if self.last_received_at is None:
            return float("inf")
        return time.monotonic() - self.last_received_at
```

Add `import logging` to the imports at the top of the file. Add `logger = logging.getLogger(__name__)` after the `builtins_max = max` line.

- [ ] **Step 4: Export `ChannelHealth`**

In `pacsys/exp/__init__.py`, add `ChannelHealth` to the import from `_monitor` and to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/exp/test_monitor.py::TestChannelHealth -v -x 2>&1 | tail -10`
Expected: 3 PASSED

- [ ] **Step 6: Lint and commit**

```bash
ruff check --fix -q pacsys/exp/_monitor.py pacsys/exp/__init__.py tests/exp/test_monitor.py && ruff check pacsys/exp/_monitor.py pacsys/exp/__init__.py tests/exp/test_monitor.py
ruff format -q pacsys/exp/_monitor.py pacsys/exp/__init__.py tests/exp/test_monitor.py
git add pacsys/exp/_monitor.py pacsys/exp/__init__.py tests/exp/test_monitor.py
git commit -m "add ChannelHealth dataclass with gap property"
```

---

### Task 2: Add health tracking state to `Monitor.__init__` and `_on_reading`

**Files:**
- Modify: `pacsys/exp/_monitor.py:351-367` (`__init__`) and `pacsys/exp/_monitor.py:429-436` (`_on_reading`)
- Test: `tests/exp/test_monitor.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/exp/test_monitor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/exp/test_monitor.py::TestMonitorHealthOnDemand -v -x 2>&1 | tail -10`
Expected: FAIL — `Monitor` has no `health` method

- [ ] **Step 3: Implement health tracking and `health()` method**

Modify `Monitor.__init__` in `pacsys/exp/_monitor.py` to add new parameters and state. The full new `__init__` signature:

```python
def __init__(
    self,
    devices: list[DeviceSpec],
    buffer_size: int = 10_000,
    backend: Backend | None = None,
    stale_after: float | None = None,
    on_stale: Callable[[str, ChannelHealth], None] | None = None,
    on_recover: Callable[[str, ChannelHealth], None] | None = None,
):
```

Add after existing init state:

```python
    self._received_at: dict[str, float | None] = {drf: None for drf in self._drfs}
    self._stale_after = stale_after
    self._on_stale = on_stale
    self._on_recover = on_recover
    self._stale_set: set[str] = set()
    self._watchdog: threading.Thread | None = None
    self._started_mono: float | None = None
```

Add one line inside `_on_reading`, after `self._latest[drf] = reading`:

```python
                self._received_at[drf] = time.monotonic()
```

Add `_build_health` and `health` methods after `has_new`:

```python
def _build_health(self, drf: str, now: float) -> ChannelHealth:
    received_at = self._received_at[drf]
    if self._stale_after is None:
        stale = False
    elif received_at is not None:
        stale = (now - received_at) > self._stale_after
    elif self._started_mono is not None:
        # Grace period: not stale until stale_after elapsed since start
        stale = (now - self._started_mono) >= self._stale_after
    else:
        # Not started yet
        stale = False
    return ChannelHealth(
        drf=drf,
        last_reading=self._latest[drf],
        last_received_at=received_at,
        total_received=self._counters[drf],
        stale=stale,
    )

def health(self, drf: DeviceSpec | None = None) -> ChannelHealth | dict[str, ChannelHealth]:
    """Per-channel health snapshot."""
    now = time.monotonic()
    with self._lock:
        if drf is not None:
            key = resolve_drf(drf)
            if key not in self._received_at:
                raise KeyError(f"No channel {key!r}. Available: {list(self._received_at)}")
            return self._build_health(key, now)
        return {d: self._build_health(d, now) for d in self._drfs}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/exp/test_monitor.py::TestMonitorHealthOnDemand -v -x 2>&1 | tail -10`
Expected: 5 PASSED

- [ ] **Step 5: Run all existing tests to check for regressions**

Run: `python -m pytest tests/exp/test_monitor.py -v -x 2>&1 | tail -15`
Expected: All tests PASSED (existing + new)

- [ ] **Step 6: Lint and commit**

```bash
ruff check --fix -q pacsys/exp/_monitor.py tests/exp/test_monitor.py && ruff check pacsys/exp/_monitor.py tests/exp/test_monitor.py
ruff format -q pacsys/exp/_monitor.py tests/exp/test_monitor.py
git add pacsys/exp/_monitor.py tests/exp/test_monitor.py
git commit -m "add health tracking and on-demand health() method to Monitor"
```

---

### Task 3: Add staleness detection tests and grace period logic

**Files:**
- Test: `tests/exp/test_monitor.py`
- Modify: `pacsys/exp/_monitor.py` (already implemented in Task 2's `_build_health`, this task validates it)

- [ ] **Step 1: Write staleness and grace period tests**

Append to `tests/exp/test_monitor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/exp/test_monitor.py::TestMonitorStaleness -v -x 2>&1 | tail -10`
Expected: 4 PASSED

- [ ] **Step 3: Lint and commit**

```bash
ruff check --fix -q tests/exp/test_monitor.py && ruff check tests/exp/test_monitor.py
ruff format -q tests/exp/test_monitor.py
git add tests/exp/test_monitor.py
git commit -m "add staleness detection and grace period tests"
```

---

### Task 4: Add watchdog thread with `on_stale` and `on_recover` callbacks

**Files:**
- Modify: `pacsys/exp/_monitor.py:409-422` (`start` and `stop` methods)
- Add: `_watchdog_loop` method to `Monitor`
- Test: `tests/exp/test_monitor.py`

- [ ] **Step 1: Write failing watchdog tests**

Append to `tests/exp/test_monitor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/exp/test_monitor.py::TestMonitorWatchdog -v -x 2>&1 | tail -10`
Expected: FAIL — `_watchdog` attribute doesn't exist or watchdog not started

- [ ] **Step 3: Implement `_watchdog_loop` and modify `start`/`stop`**

Add `_watchdog_loop` method to `Monitor` class in `pacsys/exp/_monitor.py`, after `_build_health`:

```python
def _watchdog_loop(self) -> None:
    interval = max(0.1, builtins_min(self._stale_after / 2, 1.0))
    while self.running:
        now = time.monotonic()
        stale_events: list[tuple[str, ChannelHealth]] = []
        recover_events: list[tuple[str, ChannelHealth]] = []
        with self._lock:
            for drf in self._drfs:
                was_stale = drf in self._stale_set
                ch = self._build_health(drf, now)
                if ch.stale and not was_stale:
                    self._stale_set.add(drf)
                    stale_events.append((drf, ch))
                elif not ch.stale and was_stale:
                    self._stale_set.discard(drf)
                    recover_events.append((drf, ch))
        for drf, ch in stale_events:
            logger.warning("channel %s stale (%.1fs since last reading)", drf, ch.gap)
            if self._on_stale:
                try:
                    self._on_stale(drf, ch)
                except Exception:
                    logger.error("on_stale callback failed for %s", drf, exc_info=True)
        for drf, ch in recover_events:
            logger.info("channel %s recovered", drf)
            if self._on_recover:
                try:
                    self._on_recover(drf, ch)
                except Exception:
                    logger.error("on_recover callback failed for %s", drf, exc_info=True)
        time.sleep(interval)
```

Modify `start()` — add after `self._handle = backend.subscribe(...)`:

```python
        self._started_mono = time.monotonic()
        if self._stale_after is not None:
            self._watchdog = threading.Thread(
                target=self._watchdog_loop, daemon=True, name="pacsys-watchdog"
            )
            self._watchdog.start()
```

Modify `stop()` — add after the existing `self._lock.notify_all()` block:

```python
        if self._watchdog is not None:
            self._watchdog.join(timeout=2.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/exp/test_monitor.py::TestMonitorWatchdog -v -x 2>&1 | tail -10`
Expected: 7 PASSED

- [ ] **Step 5: Run ALL monitor tests for regressions**

Run: `python -m pytest tests/exp/test_monitor.py -v -x 2>&1 | tail -20`
Expected: All tests PASSED

- [ ] **Step 6: Lint and commit**

```bash
ruff check --fix -q pacsys/exp/_monitor.py tests/exp/test_monitor.py && ruff check pacsys/exp/_monitor.py tests/exp/test_monitor.py
ruff format -q pacsys/exp/_monitor.py tests/exp/test_monitor.py
git add pacsys/exp/_monitor.py tests/exp/test_monitor.py
git commit -m "add watchdog thread with on_stale/on_recover callbacks"
```

---

### Task 5: Final validation

**Files:**
- All modified files from Tasks 1-4

- [ ] **Step 1: Run full unit test suite**

Run: `python -m pytest tests/ -v -x 2>&1 | tail -30`
Expected: All PASSED, no regressions

- [ ] **Step 2: Type check**

Run: `ty check pacsys/exp/_monitor.py 2>&1 | tail -20`
Expected: No errors (warnings are OK)

- [ ] **Step 3: Lint entire module**

```bash
ruff check --fix -q pacsys/exp/ && ruff check pacsys/exp/
ruff format -q pacsys/exp/
```

- [ ] **Step 4: Update SPECIFICATION.md**

Add a section under the Monitor documentation describing:
- `ChannelHealth` dataclass and its fields
- `health()` method (single channel and all channels)
- `stale_after`, `on_stale`, `on_recover` parameters
- Watchdog behavior (edge-triggered, grace period, callback safety)

- [ ] **Step 5: Commit docs**

```bash
git add SPECIFICATION.md
git commit -m "document monitor health check in specification"
```
