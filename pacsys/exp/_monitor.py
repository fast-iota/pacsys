"""Monitor: subscribe to channels and collect readings with aggregation."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Optional

from pacsys.types import DeviceSpec, Reading, SubscriptionHandle, Value
from pacsys.exp._resolve import resolve_drf, resolve_backend

if TYPE_CHECKING:
    from pacsys.backends import Backend

# Alias builtins to avoid shadowing by methods
builtins_min = min
builtins_max = max


@dataclass(frozen=True)
class ChannelData:
    """Collected readings for a single channel."""

    drf: str
    readings: tuple[Reading, ...] = ()

    def values(self) -> list[Value]:
        """Extract values from all ok readings."""
        return [r.value for r in self.readings if r.ok and r.value is not None]

    def timestamps(self) -> list[datetime]:
        """Extract timestamps from all readings with timestamps."""
        return [r.timestamp for r in self.readings if r.timestamp is not None]


@dataclass(frozen=True)
class MonitorResult:
    """Snapshot of collected monitoring data across channels."""

    channels: dict[str, ChannelData]
    started: Optional[datetime] = None
    stopped: Optional[datetime] = None

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

    def _mean_one(self, drf: DeviceSpec) -> float:
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        return sum(vals) / len(vals)

    def mean(self, drf: DeviceSpec | None = None) -> float | dict[str, float]:
        """Mean of values. Single channel if drf given, else dict of all."""
        if drf is not None:
            return self._mean_one(drf)
        return {d: self._mean_one(d) for d in self.channels}

    def _std_one(self, drf: DeviceSpec) -> float:
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        m = sum(vals) / len(vals)
        variance = sum((v - m) ** 2 for v in vals) / len(vals)
        return variance**0.5

    def std(self, drf: DeviceSpec | None = None) -> float | dict[str, float]:
        """Standard deviation. Single channel if drf given, else dict."""
        if drf is not None:
            return self._std_one(drf)
        return {d: self._std_one(d) for d in self.channels}

    def _median_one(self, drf: DeviceSpec) -> float:
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        s = sorted(vals)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    def median(self, drf: DeviceSpec | None = None) -> float | dict[str, float]:
        """Median of values. Single channel if drf given, else dict."""
        if drf is not None:
            return self._median_one(drf)
        return {d: self._median_one(d) for d in self.channels}

    def _min_one(self, drf: DeviceSpec) -> float:
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        return builtins_min(vals)

    def min(self, drf: DeviceSpec | None = None) -> float | dict[str, float]:
        """Minimum value. Single channel if drf given, else dict."""
        if drf is not None:
            return self._min_one(drf)
        return {d: self._min_one(d) for d in self.channels}

    def _max_one(self, drf: DeviceSpec) -> float:
        vals = self._numeric_values(drf)
        if not vals:
            raise ValueError(f"No readings for {drf}")
        return builtins_max(vals)

    def max(self, drf: DeviceSpec | None = None) -> float | dict[str, float]:
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

    def to_numpy(self, drf: DeviceSpec) -> tuple:
        """Return (timestamps, values) as numpy arrays.

        Timestamps are float64 epoch seconds (UTC). Error readings are skipped.
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
        timestamps = np.array([r.timestamp.timestamp() if r.timestamp else 0.0 for r in ok], dtype=np.float64)
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
    ):
        if not devices:
            raise ValueError("devices cannot be empty")
        self._drfs = [resolve_drf(d) for d in devices]
        self._buffer_size = buffer_size
        self._backend = backend
        self._lock = threading.Condition(threading.Lock())
        self._buffers: dict[str, deque[Reading]] = {drf: deque(maxlen=buffer_size) for drf in self._drfs}
        self._counters: dict[str, int] = {drf: 0 for drf in self._drfs}
        self._latest: dict[str, Reading | None] = {drf: None for drf in self._drfs}
        self._started: datetime | None = None
        self._handle: SubscriptionHandle | None = None

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
        self._started = datetime.now(timezone.utc)
        backend = resolve_backend(self._backend)
        self._handle = backend.subscribe(self._drfs, callback=self._on_reading)

    def stop(self) -> None:
        """Stop collecting."""
        if self._handle is not None:
            self._handle.stop()
            with self._lock:
                self._lock.notify_all()

    def __len__(self) -> int:
        """Total readings across all buffers."""
        with self._lock:
            return sum(len(buf) for buf in self._buffers.values())

    def _on_reading(self, reading: Reading, handle: SubscriptionHandle) -> None:
        with self._lock:
            drf = reading.drf
            if drf in self._buffers:
                self._buffers[drf].append(reading)
                self._counters[drf] += 1
                self._latest[drf] = reading
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
