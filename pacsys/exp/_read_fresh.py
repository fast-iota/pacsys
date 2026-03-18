"""read_fresh: wait for fresh readings per channel via temporary subscription."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Iterator

from pacsys.types import DeviceSpec, Reading, Value
from pacsys.drf_utils import has_event, replace_event
from pacsys.exp._resolve import resolve_drf, resolve_backend

if TYPE_CHECKING:
    from pacsys.backends import Backend

# Alias builtins to avoid shadowing by methods
builtins_min = min
builtins_max = max


@dataclass(frozen=True)
class FreshResult:
    """Result of read_fresh for a single channel."""

    drf: str
    readings: tuple[Reading, ...]
    requested_count: int

    def __getitem__(self, index: int) -> Reading:
        return self.readings[index]

    def __len__(self) -> int:
        return len(self.readings)

    def __iter__(self) -> Iterator[Reading]:
        return iter(self.readings)

    @property
    def length(self) -> int:
        return len(self.values)

    @property
    def value(self) -> Value | None:
        return self.readings[-1].value

    @property
    def values(self) -> list[Value]:
        return [r.value for r in self.readings if r.ok and r.value is not None]

    @property
    def reading(self) -> Reading:
        return self.readings[-1]

    @property
    def timestamp(self) -> datetime | None:
        return self.readings[-1].timestamp

    @property
    def timestamps(self) -> list[datetime]:
        return [r.timestamp for r in self.readings if r.timestamp is not None]

    def _windowed_values(self, n: int | None) -> list[float]:
        """Extract numeric values with windowing."""
        all_vals = self.values
        if n is None:
            vals = all_vals[-self.requested_count :]
        elif n == -1:
            vals = all_vals
        else:
            if n < 1:
                raise ValueError(f"n must be >= 1 or -1 for all, got {n}")
            if len(all_vals) < n:
                raise ValueError(f"Only {len(all_vals)} values available, fewer than {n}")
            vals = all_vals[-n:]
        if not vals:
            raise ValueError(f"No values for {self.drf}")
        nums = []
        for v in vals:
            if isinstance(v, bool) or type(v).__name__ == "bool_":
                raise TypeError(f"Cannot compute stats on {type(v).__name__}")
            try:
                nums.append(float(v))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise TypeError(f"Cannot compute stats on {type(v).__name__}")
        return nums

    def mean(self, n: int | None = None) -> float:
        vals = self._windowed_values(n)
        return sum(vals) / len(vals)

    def median(self, n: int | None = None) -> float:
        vals = sorted(self._windowed_values(n))
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2

    def std(self, n: int | None = None) -> float:
        vals = self._windowed_values(n)
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    def min(self, n: int | None = None) -> float:
        return builtins_min(self._windowed_values(n))

    def max(self, n: int | None = None) -> float:
        return builtins_max(self._windowed_values(n))


def read_fresh(
    devices: list[DeviceSpec],
    *,
    count: int = 1,
    default_event: str | None = None,
    timeout: float = 5.0,
    backend: Backend | None = None,
) -> list[FreshResult]:
    """Wait for fresh readings per channel via temporary subscription.

    Args:
        devices: List of DRF strings or Device objects.
        count: Number of readings to collect per channel (>= 1).
        default_event: Event to apply to DRFs that lack one (e.g. "p,1000").
        timeout: Max seconds to wait for all channels to reach count.
        backend: Optional backend (uses global default if None).

    Returns:
        list[FreshResult] in same order as input.

    Raises:
        TimeoutError: If any channel doesn't reach count within timeout.
        ValueError: If devices is empty or count < 1.
    """
    if not devices:
        raise ValueError("devices cannot be empty")
    if count < 1:
        raise ValueError("count must be >= 1")

    drfs = []
    for d in devices:
        drf = resolve_drf(d)
        if default_event is not None and not has_event(drf):
            drf = replace_event(drf, default_event)
        drfs.append(drf)

    be = resolve_backend(backend)
    unique_drfs = list(dict.fromkeys(drfs))

    collected: dict[str, list[Reading]] = {drf: [] for drf in unique_drfs}
    channels_done = 0  # O(1) completion counter
    error_box: list[Exception] = []
    lock = threading.Lock()
    done = threading.Event()

    def on_reading(reading, handle):
        nonlocal channels_done
        with lock:
            drf = reading.drf
            if drf in collected:
                collected[drf].append(reading)
                if len(collected[drf]) == count:
                    channels_done += 1
                    if channels_done >= len(unique_drfs):
                        done.set()

    def on_error(exc, handle):
        error_box.append(exc)
        done.set()

    handle = None
    try:
        handle = be.subscribe(unique_drfs, callback=on_reading, on_error=on_error)
        if not done.wait(timeout=timeout):
            # Re-check under lock to avoid race at deadline boundary
            with lock:
                if channels_done >= len(unique_drfs):
                    return [FreshResult(drf=drf, readings=tuple(collected[drf]), requested_count=count) for drf in drfs]
                missing = [d for d in unique_drfs if len(collected[d]) < count]
            raise TimeoutError(f"Timed out waiting for {count} readings: {missing}")
        with lock:
            if channels_done >= len(unique_drfs):
                return [FreshResult(drf=drf, readings=tuple(collected[drf]), requested_count=count) for drf in drfs]
        if error_box:
            raise error_box[0]
        # done was set but neither completion nor error — should not happen
        raise RuntimeError("read_fresh: unexpected state")
    finally:
        if handle is not None:
            handle.stop()
