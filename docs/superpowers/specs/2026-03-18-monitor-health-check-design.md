# Monitor Health Check Design

## Overview

Add a health check capability to the `Monitor` class that detects stale channels (no data received within a configurable window) and provides on-demand per-channel health inspection.

## Motivation

Long-running monitors collecting accelerator data need to detect when a channel silently stops delivering readings. Currently, the only way to notice is to manually call `await_next()` with a timeout or inspect `tags`. There is no proactive alerting.

## Design

### Approach: Built into Monitor, lazy watchdog thread

Health metadata tracking (`_received_at` timestamps) is always present in Monitor. A watchdog thread only spawns if the user provides `stale_after`. The `health()` method works regardless of whether the watchdog is running.

### New type: `ChannelHealth`

```python
@dataclass(frozen=True)
class ChannelHealth:
    drf: str
    last_reading: Reading | None
    last_received_at: float | None   # monotonic timestamp
    total_received: int
    stale: bool

    @property
    def gap(self) -> float:
        """Seconds since last reading, or inf if never received."""
        if self.last_received_at is None:
            return float("inf")
        return time.monotonic() - self.last_received_at
```

- `last_received_at` is a `time.monotonic()` value (not wall-clock) for accurate gap measurement
- `gap` is live-computed each time it's accessed
- `stale` is computed at construction time by `_build_health()`: `stale_after is not None and (received_at is None or (now - received_at) > stale_after)`, with the startup grace period adjustment applied. When `stale_after` is `None`, `stale` is always `False`.
- `total_received` mirrors the value from `_counters[drf]` (same source as `Monitor.tags`)

### Monitor `__init__` changes

New parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stale_after` | `float \| None` | `None` | Seconds of silence before a channel is stale. `None` disables watchdog. |
| `on_stale` | `Callable[[str, ChannelHealth], None] \| None` | `None` | Called once when a channel transitions to stale. |
| `on_recover` | `Callable[[str, ChannelHealth], None] \| None` | `None` | Called once when a stale channel receives data again. Optional. |

New internal state (all initialized in `__init__`):

- `_received_at: dict[str, float | None]` — per-channel monotonic timestamp, initialized to `{drf: None for drf in self._drfs}`, updated in `_on_reading`
- `_stale_set: set[str]` — channels currently in stale state (for edge detection), initialized empty
- `_watchdog: threading.Thread | None` — only created if `stale_after` is set, initialized to `None`
- `_started_mono: float | None` — monotonic time of `start()`, for startup grace period. Distinct from the existing `_started` (wall-clock `datetime`); this is used only for grace period calculation.

A module-level logger is added: `logger = logging.getLogger(__name__)`.

### `_on_reading` change

One additional line:

```python
self._received_at[drf] = time.monotonic()
```

### `health()` method

```python
def health(self, drf: DeviceSpec | None = None) -> ChannelHealth | dict[str, ChannelHealth]:
```

- `health("M:OUTTMP@p,1000")` — returns `ChannelHealth` for one channel
- `health()` — returns `dict[str, ChannelHealth]` for all channels
- Input is resolved via `resolve_drf()` before lookup, matching the pattern in `await_next()` and `MonitorResult._get_channel()`
- Works anytime after `__init__`, even before `start()` (returns `None`/0/stale values)
- Raises `KeyError` if the DRF is not a monitored channel

### Watchdog thread

Spawned in `start()` only if `stale_after` is set. Daemon thread.

**Check interval:** `max(0.1, min(stale_after / 2, 1.0))` — responsive without busy-looping. Floor of 100ms prevents CPU-intensive polling for very small `stale_after` values.

**Edge-triggered alerts:**
- Fires `on_stale` once when a channel transitions from healthy to stale
- Fires `on_recover` once when a channel transitions from stale to healthy (only if `on_recover` was provided)
- Always logs: `WARNING` for stale, `INFO` for recovery
- Does not fire repeatedly while a channel remains stale

**Callback safety:** Events are collected under the lock, but callbacks are fired after releasing it. This prevents deadlocks if the callback interacts with the Monitor. Callbacks are wrapped in `try/except` — exceptions are logged at `ERROR` level but do not crash the watchdog thread.

**Recovery detection:** Recovery is handled only by the watchdog thread, not by `_on_reading`. The `_on_reading` callback does not interact with `_stale_set`.

**Lifecycle:**
- Starts after `backend.subscribe()` in `start()`
- Exits naturally when `self.running` becomes `False`
- Joined in `stop()` with a short timeout

**Interaction with `flush()`:** `flush()` does not affect health tracking state (`_received_at`, `_stale_set`, `_started_mono`). Only `_buffers` and `_started` (wall-clock) are reset, as today.

### Startup grace period

When the Monitor starts, no channel has received data yet. Without grace, every channel would immediately appear stale.

Rule: a channel is not considered stale if `received_at is None and (now - _started_mono) < stale_after`. This means:
- If a channel never delivers anything, you get an alert after `stale_after` seconds from start
- If a channel delivers data then stops, you get an alert after `stale_after` seconds of silence
- No false stale alerts at startup

### Exports

`ChannelHealth` added to `pacsys/exp/__init__.py` and `__all__`.

## Usage examples

```python
# On-demand health check (no watchdog)
mon = Monitor(["M:OUTTMP@p,1000", "G:AMANDA@E,17"])
mon.start()
# ... later ...
h = mon.health("M:OUTTMP@p,1000")
print(h.gap, h.total_received, h.stale)

# With watchdog alerting
def alert(drf, health):
    print(f"STALE: {drf} — {health.gap:.1f}s since last reading")

mon = Monitor(
    ["M:OUTTMP@p,1000", "G:AMANDA@E,17"],
    stale_after=5.0,
    on_stale=alert,
)
with mon:
    time.sleep(60)  # watchdog fires alert callback if any channel goes silent

# With recovery notification
mon = Monitor(
    ["M:OUTTMP@p,1000"],
    stale_after=5.0,
    on_stale=lambda drf, h: print(f"STALE: {drf}"),
    on_recover=lambda drf, h: print(f"RECOVERED: {drf}"),
)
```

## Files changed

| File | Change |
|------|--------|
| `pacsys/exp/_monitor.py` | Add `ChannelHealth`, modify `Monitor.__init__`, `_on_reading`, `start`, `stop`; add `health()`, `_build_health()`, `_watchdog_loop()` |
| `pacsys/exp/__init__.py` | Export `ChannelHealth` |
| `tests/exp/test_monitor.py` | Tests for health check, watchdog, edge detection, grace period |

## Testing

- `health()` returns correct values before/after readings arrive
- `gap` grows over time when no readings arrive
- Watchdog fires `on_stale` exactly once per stale transition
- Watchdog fires `on_recover` only when provided and channel recovers
- Startup grace period prevents false stale alerts
- Watchdog thread exits cleanly on `stop()`
- `health()` works without watchdog (`stale_after=None`)
- `stale` is `False` when `stale_after` is `None` (no threshold means never stale)
- Callback exceptions are caught and logged, do not crash the watchdog
