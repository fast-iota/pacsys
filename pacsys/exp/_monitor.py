"""Monitor: subscribe to channels and collect readings with aggregation."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Iterator, Optional

from pacsys.types import DeviceSpec, Reading, SubscriptionHandle, Value, ValueType
from pacsys.exp._resolve import resolve_drf, resolve_backend

if TYPE_CHECKING:
    import numpy as np
    from pacsys.backends import Backend

# Alias builtins to avoid shadowing by methods
builtins_min = min
builtins_max = max

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelData:
    """Collected readings for a single channel."""

    drf: str
    readings: tuple[Reading, ...] = ()

    def __getitem__(self, index: int) -> Reading:
        return self.readings[index]

    def __len__(self) -> int:
        return len(self.readings)

    def __iter__(self) -> Iterator[Reading]:
        return iter(self.readings)

    def values(self) -> list[Value]:
        """Extract values from all ok readings."""
        return [r.value for r in self.readings if r.ok and r.value is not None]

    def timestamps(self) -> list[datetime]:
        """Extract timestamps from all readings with timestamps."""
        return [r.timestamp for r in self.readings if r.timestamp is not None]


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


@dataclass(frozen=True)
class MonitorResult:
    """Snapshot of collected monitoring data across channels."""

    channels: dict[str, ChannelData]
    started: Optional[datetime] = None
    stopped: Optional[datetime] = None

    def __getitem__(self, drf: DeviceSpec) -> ChannelData:
        return self._get_channel(drf)

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Iterator[str]:
        return iter(self.channels)

    def __contains__(self, drf: DeviceSpec) -> bool:
        try:
            self._get_channel(drf)
            return True
        except KeyError:
            return False

    @property
    def counts(self) -> dict[str, int]:
        """Number of readings per channel."""
        return {drf: len(ch.readings) for drf, ch in self.channels.items()}

    @property
    def elapsed(self) -> timedelta | None:
        """Time between started and stopped, or None if timestamps missing."""
        if self.started is None or self.stopped is None:
            return None
        return self.stopped - self.started

    def rate(self, drf: DeviceSpec | None = None) -> float | dict[str, float]:
        """Readings per second. Single channel if drf given, else dict."""
        el = self.elapsed
        if el is None or el.total_seconds() == 0:
            raise ValueError("Cannot compute rate: no elapsed time")
        secs = el.total_seconds()
        if drf is not None:
            return len(self._get_channel(drf).readings) / secs
        return {d: len(ch.readings) / secs for d, ch in self.channels.items()}

    def _resolve(self, drf: DeviceSpec) -> str:
        return resolve_drf(drf)

    def _get_channel(self, drf: DeviceSpec) -> ChannelData:
        key = self._resolve(drf)
        if key not in self.channels:
            raise KeyError(f"No channel {key!r}. Available: {list(self.channels)}")
        return self.channels[key]

    @staticmethod
    def _unwrap_arrays(vals: list) -> list | None:
        """Extract numeric arrays from values. Returns None if not array-valued."""
        import numpy as np

        if not vals:
            return None
        first = vals[0]
        if isinstance(first, np.ndarray):
            return vals
        if isinstance(first, dict) and "data" in first:
            return [v["data"] for v in vals]
        return None

    def _try_array_stack(self, drf: DeviceSpec):
        """If channel holds ndarray values, stack into 2D numpy array. Else return None."""
        import numpy as np

        arrays = self._unwrap_arrays(self._get_channel(drf).values())
        if arrays is None:
            return None
        return np.stack(arrays)  # type: ignore[arg-type]

    def _numeric_values(self, drf: DeviceSpec) -> list[float]:
        vals = self._get_channel(drf).values()
        nums = []
        for v in vals:
            if isinstance(v, bool) or type(v).__name__ == "bool_":
                raise TypeError(f"Cannot compute stats on non-numeric value {type(v).__name__} in {drf}")
            try:
                nums.append(float(v))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise TypeError(f"Cannot compute stats on non-numeric value {type(v).__name__} in {drf}")
        return nums

    def _mean_one(self, drf: DeviceSpec) -> float | np.ndarray:
        stacked = self._try_array_stack(drf)
        if stacked is not None:
            return stacked.mean(axis=0)
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        return sum(vals) / len(vals)

    def mean(self, drf: DeviceSpec | None = None) -> float | np.ndarray | dict[str, float | np.ndarray]:
        """Mean of values. Single channel if drf given, else dict of all."""
        if drf is not None:
            return self._mean_one(drf)
        return {d: self._mean_one(d) for d in self.channels}

    def _std_one(self, drf: DeviceSpec) -> float | np.ndarray:
        stacked = self._try_array_stack(drf)
        if stacked is not None:
            return stacked.std(axis=0)
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        m = sum(vals) / len(vals)
        variance = sum((v - m) ** 2 for v in vals) / len(vals)
        return variance**0.5

    def std(self, drf: DeviceSpec | None = None) -> float | np.ndarray | dict[str, float | np.ndarray]:
        """Standard deviation. Single channel if drf given, else dict."""
        if drf is not None:
            return self._std_one(drf)
        return {d: self._std_one(d) for d in self.channels}

    def _median_one(self, drf: DeviceSpec) -> float | np.ndarray:
        stacked = self._try_array_stack(drf)
        if stacked is not None:
            import numpy as np

            return np.median(stacked, axis=0)
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        s = sorted(vals)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    def median(self, drf: DeviceSpec | None = None) -> float | np.ndarray | dict[str, float | np.ndarray]:
        """Median of values. Single channel if drf given, else dict."""
        if drf is not None:
            return self._median_one(drf)
        return {d: self._median_one(d) for d in self.channels}

    def _min_one(self, drf: DeviceSpec) -> float | np.ndarray:
        stacked = self._try_array_stack(drf)
        if stacked is not None:
            return stacked.min(axis=0)
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        return builtins_min(vals)

    def min(self, drf: DeviceSpec | None = None) -> float | np.ndarray | dict[str, float | np.ndarray]:
        """Minimum value. Single channel if drf given, else dict."""
        if drf is not None:
            return self._min_one(drf)
        return {d: self._min_one(d) for d in self.channels}

    def _max_one(self, drf: DeviceSpec) -> float | np.ndarray:
        stacked = self._try_array_stack(drf)
        if stacked is not None:
            return stacked.max(axis=0)
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        return builtins_max(vals)

    def max(self, drf: DeviceSpec | None = None) -> float | np.ndarray | dict[str, float | np.ndarray]:
        """Maximum value. Single channel if drf given, else dict."""
        if drf is not None:
            return self._max_one(drf)
        return {d: self._max_one(d) for d in self.channels}

    def _last_one(self, n: int, drf: DeviceSpec) -> list[Value]:
        if n < 1:
            raise ValueError("n must be >= 1")
        return self._get_channel(drf).values()[-n:]

    def last(self, n: int, drf: DeviceSpec | None = None) -> list[Value] | dict[str, list[Value]]:
        """Last n values. Single channel if drf given, else dict."""
        if drf is not None:
            return self._last_one(n, drf)
        return {d: self._last_one(n, d) for d in self.channels}

    def values(self, drf: DeviceSpec) -> list[Value]:
        """All values for a channel."""
        return self._get_channel(drf).values()

    def timestamps(self, drf: DeviceSpec) -> list[datetime]:
        """All timestamps for a channel."""
        return self._get_channel(drf).timestamps()

    def slice(self, drf: DeviceSpec, start: datetime | None = None, end: datetime | None = None) -> ChannelData:
        """Filter a channel's readings by timestamp range (inclusive)."""
        ch = self._get_channel(drf)
        filtered = []
        for r in ch.readings:
            if r.timestamp is None:
                continue
            if start is not None and r.timestamp < start:
                continue
            if end is not None and r.timestamp > end:
                continue
            filtered.append(r)
        return ChannelData(drf=ch.drf, readings=tuple(filtered))

    def to_dict(self) -> dict[str, list[Reading]]:
        """Return {drf: [readings...]} for all channels."""
        return {drf: list(ch.readings) for drf, ch in self.channels.items()}

    _NUMPY_TYPES = frozenset({ValueType.SCALAR, ValueType.SCALAR_ARRAY, ValueType.TIMED_SCALAR_ARRAY})

    def to_numpy(self, drf: DeviceSpec) -> tuple:
        """Return (timestamps, values) as numpy arrays.

        Timestamps are float64 epoch seconds (UTC). Error readings are skipped.
        Only supports numeric channels (SCALAR, SCALAR_ARRAY, TIMED_SCALAR_ARRAY).
        Requires numpy.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for to_numpy(). Install with: pip install numpy")
        ch = self._get_channel(drf)
        ok = [r for r in ch.readings if r.ok and r.value is not None]
        if not ok:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        vtype = ok[0].value_type
        if vtype is not None and vtype not in self._NUMPY_TYPES:
            raise TypeError(
                f"to_numpy() requires numeric channels (SCALAR, SCALAR_ARRAY, TIMED_SCALAR_ARRAY), got {vtype.value}"
            )
        timestamps = np.array([r.timestamp.timestamp() if r.timestamp else 0.0 for r in ok], dtype=np.float64)
        arrays = self._unwrap_arrays([r.value for r in ok])
        if arrays is not None:
            values = np.stack(arrays).astype(np.float64)  # type: ignore[arg-type]
        else:
            values = np.array([float(r.value) for r in ok], dtype=np.float64)  # type: ignore[arg-type]
        return timestamps, values

    def to_dataframe(self, drf: DeviceSpec | None = None, *, relative: bool = False):
        """Convert to pandas DataFrame (requires pandas).

        If drf is given, returns a single-device DataFrame indexed by timestamp.
        If drf is None, returns a DataFrame with all channels.
        If relative is True, timestamps become seconds since self.started.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        if relative and self.started is None:
            raise ValueError("Cannot use relative=True: started timestamp is None")

        if drf is not None:
            ch = self._get_channel(drf)
            rows = []
            for r in ch.readings:
                if r.ok:
                    rows.append({"value": r.value, "units": r.units})
            if relative:
                index = [(r.timestamp - self.started).total_seconds() for r in ch.readings if r.ok]  # type: ignore[operator]
                df = pd.DataFrame(rows, index=index)  # type: ignore[arg-type]
                df.index.name = "elapsed_s"
            else:
                df = pd.DataFrame(rows, index=[r.timestamp for r in ch.readings if r.ok])  # type: ignore[arg-type]
                df.index.name = "timestamp"
            return df

        rows = []
        for ch_drf, ch in self.channels.items():
            for r in ch.readings:
                if r.ok:
                    row: dict = {
                        "drf": ch_drf,
                        "value": r.value,
                        "units": r.units,
                    }
                    if relative:
                        row["elapsed_s"] = (r.timestamp - self.started).total_seconds()  # type: ignore[operator]
                    else:
                        row["timestamp"] = r.timestamp
                    rows.append(row)
        return pd.DataFrame(rows)


class Monitor:
    """Subscribe to channels and collect readings into ring buffers.

    Usage:
        mon = Monitor(["M:OUTTMP@p,1000", "G:AMANDA@e,8f"])
        mon.start()
        snap = mon.snapshot()   # non-destructive peek
        result = mon.flush()    # swap out buffers, return old data
        mon.stop()

        # Or blocking:
        result = mon.collect(duration=10.0)
        result = mon.collect(count=100)
    """

    def __init__(
        self,
        devices: list[DeviceSpec],
        buffer_size: int = 10_000,
        backend: Backend | None = None,
        stale_after: float | None = None,
        on_stale: Callable[[str, ChannelHealth], None] | None = None,
        on_recover: Callable[[str, ChannelHealth], None] | None = None,
    ):
        if not devices:
            raise ValueError("devices cannot be empty")
        if stale_after is not None and stale_after <= 0:
            raise ValueError("stale_after must be positive")
        self._drfs = [resolve_drf(d) for d in devices]
        self._buffer_size = buffer_size
        self._backend = backend
        self._lock = threading.Condition(threading.Lock())
        self._buffers: dict[str, deque[Reading]] = {drf: deque(maxlen=buffer_size) for drf in self._drfs}
        self._counters: dict[str, int] = {drf: 0 for drf in self._drfs}
        self._latest: dict[str, Reading | None] = {drf: None for drf in self._drfs}
        self._started: datetime | None = None
        self._handle: SubscriptionHandle | None = None
        self._received_at: dict[str, float | None] = {drf: None for drf in self._drfs}
        self._stale_after = stale_after
        self._on_stale = on_stale
        self._on_recover = on_recover
        self._stale_set: set[str] = set()
        self._watchdog: threading.Thread | None = None
        self._started_mono: float | None = None

    @property
    def running(self) -> bool:
        return self._handle is not None and not self._handle.stopped

    @property
    def tags(self) -> dict[str, int]:
        """Per-channel event counters. Cheap way to check for new data."""
        with self._lock:
            return dict(self._counters)

    def has_new(self, old_tags: dict[str, int]) -> bool:
        """True if any channel has received readings since old_tags was captured."""
        with self._lock:
            return any(self._counters.get(drf, 0) > old_tags.get(drf, 0) for drf in self._counters)

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

    def _watchdog_loop(self) -> None:
        assert self._stale_after is not None
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

    def await_next(self, drf: DeviceSpec, timeout: float = 5.0) -> Reading:
        """Block until the next new reading arrives on a channel.

        Returns the Reading that arrived. Only considers readings
        arriving after this call (ignores already-buffered data).
        """
        if not self.running:
            raise RuntimeError("Monitor is not running")
        key = resolve_drf(drf)
        if key not in self._counters:
            raise KeyError(f"No channel {key!r}. Available: {list(self._counters)}")
        with self._lock:
            baseline = self._counters[key]
            deadline = time.monotonic() + timeout
            while self._counters[key] == baseline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No new reading on {key!r} within {timeout}s")
                if self._handle is not None and self._handle.exc is not None:
                    raise self._handle.exc
                if not self.running:
                    raise RuntimeError("Monitor is not running")
                self._lock.wait(timeout=builtins_min(remaining, 0.5))
            return self._latest[key]  # type: ignore[return-value]

    def start(self) -> None:
        """Start collecting readings in the background."""
        if self.running:
            raise RuntimeError("Monitor is already running")
        # Ensure old watchdog is dead before restarting
        if self._watchdog is not None:
            self._watchdog.join(timeout=2.0)
            self._watchdog = None
        # Reset all per-run state
        with self._lock:
            for drf in self._drfs:
                self._buffers[drf].clear()
                self._counters[drf] = 0
                self._latest[drf] = None
                self._received_at[drf] = None
            self._stale_set.clear()
        self._started = datetime.now(timezone.utc)
        self._started_mono = time.monotonic()
        backend = resolve_backend(self._backend)
        self._handle = backend.subscribe(self._drfs, callback=self._on_reading)
        if self._stale_after is not None:
            self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True, name="pacsys-watchdog")
            self._watchdog.start()

    def stop(self) -> None:
        """Stop collecting."""
        if self._handle is not None:
            self._handle.stop()
            with self._lock:
                self._lock.notify_all()
        if self._watchdog is not None and threading.current_thread() is not self._watchdog:
            self._watchdog.join(timeout=2.0)

    def __len__(self) -> int:
        """Total readings across all buffers."""
        with self._lock:
            return sum(len(buf) for buf in self._buffers.values())

    def _on_reading(self, reading: Reading, handle: SubscriptionHandle) -> None:
        with self._lock:
            if handle is not self._handle:
                return  # ignore late deliveries from old subscriptions
            drf = reading.drf
            if drf in self._buffers:
                self._buffers[drf].append(reading)
                self._counters[drf] += 1
                self._latest[drf] = reading
                self._received_at[drf] = time.monotonic()
                self._lock.notify_all()

    def snapshot(self) -> MonitorResult:
        """Non-destructive peek at current data."""
        with self._lock:
            channels = {drf: ChannelData(drf=drf, readings=tuple(buf)) for drf, buf in self._buffers.items()}
            started = self._started
        return MonitorResult(channels=channels, started=started, stopped=datetime.now(timezone.utc))

    def flush(self) -> MonitorResult:
        """Atomically swap out buffers and return old data."""
        with self._lock:
            old_buffers = self._buffers
            self._buffers = {drf: deque(maxlen=self._buffer_size) for drf in self._drfs}
            old_started = self._started
            self._started = datetime.now(timezone.utc)
        channels = {drf: ChannelData(drf=drf, readings=tuple(buf)) for drf, buf in old_buffers.items()}
        return MonitorResult(channels=channels, started=old_started, stopped=datetime.now(timezone.utc))

    def collect(
        self,
        *,
        duration: float | None = None,
        count: int | None = None,
        timeout: float | None = None,
    ) -> MonitorResult:
        """Blocking convenience: start, wait, stop, return result.

        Args:
            duration: Collect for this many seconds.
            count: Collect until each channel has at least this many readings.
            timeout: Max seconds to wait in count mode (default: no limit).
            Exactly one of duration/count must be given.
        """
        if (duration is None) == (count is None):
            raise ValueError("Exactly one of duration or count must be specified")

        # Clear stale data from previous runs
        with self._lock:
            for buf in self._buffers.values():
                buf.clear()
        self.start()
        try:
            if duration is not None:
                deadline = time.monotonic() + duration
                while time.monotonic() < deadline:
                    if self._handle is not None and self._handle.exc is not None:
                        raise self._handle.exc
                    remaining = deadline - time.monotonic()
                    time.sleep(min(0.1, max(0, remaining)))
            else:
                assert count is not None
                deadline = time.monotonic() + timeout if timeout is not None else None
                while True:
                    with self._lock:
                        if all(len(buf) >= count for buf in self._buffers.values()):
                            break
                    if self._handle is not None and self._handle.exc is not None:
                        raise self._handle.exc
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out after {timeout}s waiting for {count} readings per channel")
                    time.sleep(0.01)
        finally:
            self.stop()

        return self.snapshot()

    def wait_until(
        self,
        predicate: Callable[[MonitorResult], bool],
        timeout: float,
        poll: float = 0.1,
    ) -> MonitorResult:
        """Block until predicate(snapshot()) is True or timeout expires.

        Returns the snapshot that satisfied the predicate.
        Raises TimeoutError if timeout expires first.
        Monitor must already be running.
        """
        if not self.running:
            raise RuntimeError("Monitor is not running")
        deadline = time.monotonic() + timeout
        while True:
            snap = self.snapshot()
            if predicate(snap):
                return snap
            if self._handle is not None and self._handle.exc is not None:
                raise self._handle.exc
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Predicate not satisfied within {timeout}s")
            time.sleep(builtins_min(poll, max(0, deadline - time.monotonic())))

    def __enter__(self) -> Monitor:
        self.start()
        return self

    def __exit__(self, *args) -> bool:
        self.stop()
        return False
