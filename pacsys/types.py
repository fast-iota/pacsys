"""
Core data types - Reading, WriteResult, SubscriptionHandle, CombinedStream.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Union, Optional, Callable, Iterator, TYPE_CHECKING
from enum import Enum, Flag, IntEnum, auto

if TYPE_CHECKING:
    import numpy as np
    from pacsys.device import Device

from pacsys.drf_utils import get_device_name as _get_device_name

# Value types supported by ACNET
# np.ndarray is only in the annotation at type-check time; at runtime the
# alias omits it (numpy is heavy to import and not needed for annotation eval).
if TYPE_CHECKING:
    Value = Union[float, int, str, bytes, np.ndarray, list, dict]
else:
    Value = Union[float, int, str, bytes, list, dict]

# Type alias for functions accepting DRF strings or Device objects
DeviceSpec = Union[str, "Device"]

# Callback type for streaming subscriptions - receives (reading, handle) pairs
ReadingCallback = Callable[["Reading", "SubscriptionHandle"], None]

# Callback type for error handling in streaming - receives (exception, handle) pairs
ErrorCallback = Callable[[Exception, "SubscriptionHandle"], None]


class DispatchMode(Enum):
    """How streaming callbacks are dispatched.

    WORKER: callbacks run on a dedicated worker thread (default, protects reactor)
    DIRECT: callbacks run inline on the reactor thread (50ms slow-callback warning)
    """

    WORKER = "worker"
    DIRECT = "direct"


class BackendCapability(Flag):
    """Capabilities supported by a backend."""

    READ = auto()
    WRITE = auto()
    STREAM = auto()
    AUTH_KERBEROS = auto()
    AUTH_JWT = auto()
    BATCH = auto()


class ValueType(Enum):
    """Type of value returned from a device read."""

    SCALAR = "scalar"
    SCALAR_ARRAY = "scalarArr"
    TIMED_SCALAR_ARRAY = "timedScalarArr"
    RAW = "raw"
    TEXT = "text"
    TEXT_ARRAY = "textArr"
    ANALOG_ALARM = "anaAlarm"
    DIGITAL_ALARM = "digAlarm"
    BASIC_STATUS = "basicStatus"


class BasicControl(IntEnum):
    """Control commands for device CONTROL property writes.

    Ordinals match the Java BasicControlDefs constants. Commands 0-6 are
    also defined in the DMQ DAQData.proto enum; commands 7-9 (LOCAL, REMOTE,
    TRIP) are sent as numeric values on DMQ since the proto enum lacks them.

    Each command toggles a status bit (see _CONTROL_STATUS_MAP in device.py).

    Usage::

        backend.write("Z|ACLTST", BasicControl.ON)
        backend.write("Z&ACLTST", BasicControl.OFF)
    """

    RESET = 0
    ON = 1
    OFF = 2
    POSITIVE = 3
    NEGATIVE = 4
    RAMP = 5
    DC = 6
    LOCAL = 7
    REMOTE = 8
    TRIP = 9


@dataclass(frozen=True)
class DeviceMeta:
    """Device metadata from DPM DeviceInfo."""

    device_index: int
    name: str
    description: str
    units: Optional[str] = None
    format_hint: Optional[int] = None


@dataclass(frozen=True)
class Reading:
    """A device reading with status and optional data.

    Status semantics (matches gRPC Status message):
    - facility_code: ACNET facility identifier (0=success, 1=ACNET, 16=DBM, 17=DPM)
    - error_code: 0=success, >0=warning, <0=error
    """

    drf: str
    value_type: ValueType
    facility_code: int = 0
    error_code: int = 0
    value: Optional[Value] = None
    message: Optional[str] = None
    timestamp: Optional[datetime] = None
    cycle: Optional[int] = None
    meta: Optional[DeviceMeta] = None

    @property
    def is_success(self) -> bool:
        """True if status indicates success (error_code == 0)."""
        return self.error_code == 0

    @property
    def is_warning(self) -> bool:
        """True if status indicates warning (error_code > 0)."""
        return self.error_code > 0

    @property
    def is_error(self) -> bool:
        """True if status indicates error (error_code < 0)."""
        return self.error_code < 0

    @property
    def ok(self) -> bool:
        """True if data is usable (success or warning with data)."""
        return self.error_code >= 0 and self.value is not None

    @property
    def name(self) -> str:
        """Device name extracted from DRF or metadata."""
        if self.meta:
            return self.meta.name
        return _get_device_name(self.drf)

    @property
    def units(self) -> Optional[str]:
        """Engineering units from metadata, or None if unavailable."""
        return self.meta.units if self.meta else None


@dataclass(frozen=True)
class WriteResult:
    """Result of a write operation, optionally with verification info."""

    drf: str
    facility_code: int = 0
    error_code: int = 0
    message: Optional[str] = None
    # Verification fields (None if verify not used)
    verified: Optional[bool] = None  # True=readback matched, False=failed, None=no verify
    readback: Optional[Value] = None  # Last readback value
    skipped: bool = False  # True if check_first found value already correct
    attempts: int = 0  # Number of readback attempts made

    @property
    def ok(self) -> bool:
        """True if write succeeded (error_code == 0)."""
        return self.error_code == 0

    @property
    def success(self) -> bool:
        """Alias for ok."""
        return self.ok


class SubscriptionHandle:
    """Handle for a streaming subscription.

    Provides access to subscription state and allows stopping the subscription.
    Each handle has its own queue for readings. Use readings() to iterate over
    readings from THIS subscription only.

    Usage:
        # Context manager (recommended) - auto-stops on exit
        with backend.subscribe(["M:OUTTMP@p,1000"]) as sub:
            for reading, handle in sub.readings(timeout=10):
                print(reading.value)
                if reading.value > 10:
                    sub.stop()

        # Manual control
        sub = backend.subscribe(["M:OUTTMP@p,1000"])
        for reading, handle in sub.readings(timeout=10):
            print(reading.value)
        sub.stop()
    """

    @property
    def ref_ids(self) -> list[int]:
        """Reference IDs for devices in this subscription."""
        raise NotImplementedError

    @property
    def stopped(self) -> bool:
        """True if this subscription has been stopped."""
        raise NotImplementedError

    @property
    def exc(self) -> Optional[Exception]:
        """Exception if an error occurred, else None."""
        raise NotImplementedError

    def readings(
        self,
        timeout: Optional[float] = None,
    ) -> Iterator[tuple["Reading", "SubscriptionHandle"]]:
        """Yield (reading, handle) pairs for THIS subscription.

        Args:
            timeout: Seconds to wait for next reading.
                    None = block forever (until stop() called)
                    0 = non-blocking (drain buffered readings only)

        Yields:
            (reading, handle) pairs
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Stop this subscription."""
        raise NotImplementedError

    def __enter__(self) -> "SubscriptionHandle":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager - stops the subscription."""
        self.stop()
        return False


class CombinedStream:
    """Combines multiple subscriptions into a single iterable stream.

    Yields readings from all subscriptions. Readings are sorted by timestamp within
    each batch of available data, but global ordering is not guaranteed if streams
    have different latencies. Stops when all subscriptions are stopped or timeout
    is reached.

    Usage:
        with backend.subscribe(["M:OUTTMP@p,1000"]) as sub1:
            with backend.subscribe(["G:AMANDA@P,500"]) as sub2:
                for reading, handle in CombinedStream([sub1, sub2]).readings(timeout=10):
                    print(f"{reading.name}: {reading.value}")
    """

    def __init__(self, subscriptions: list["SubscriptionHandle"]):
        if not subscriptions:
            raise ValueError("subscriptions cannot be empty")
        self._subscriptions = list(subscriptions)

    @property
    def stopped(self) -> bool:
        """True if all subscriptions have been stopped."""
        return all(sub.stopped for sub in self._subscriptions)

    @property
    def exc(self) -> Optional[Exception]:
        """First exception from any subscription, or None."""
        for sub in self._subscriptions:
            if sub.exc is not None:
                return sub.exc
        return None

    def readings(
        self,
        timeout: Optional[float] = None,
    ) -> Iterator[tuple["Reading", "SubscriptionHandle"]]:
        """Yield readings from all subscriptions.

        Readings are sorted by timestamp within each batch of available data,
        but global ordering is not guaranteed if streams have different latencies.

        Args:
            timeout: Total timeout for the combined stream in seconds.
                    None = block forever (until all stopped)
                    0 = non-blocking (drain buffered readings only)

        Yields:
            (reading, handle) pairs from any subscription

        Raises:
            Exception: If any subscription has an error and no on_error was provided
        """
        import heapq
        import queue as queue_mod
        import threading
        import time

        # Non-blocking mode: drain without spawning threads
        if timeout == 0:
            if self.exc is not None:
                raise self.exc
            heap: list[tuple[datetime, int, "Reading", "SubscriptionHandle"]] = []
            counter = 0
            for sub in self._subscriptions:
                if sub.stopped:
                    continue
                for reading, handle in sub.readings(timeout=0):
                    ts = reading.timestamp or datetime.min
                    heapq.heappush(heap, (ts, counter, reading, handle))
                    counter += 1
            while heap:
                _, _, reading, handle = heapq.heappop(heap)
                yield (reading, handle)
            return

        # Blocking mode: feeder threads push into a shared queue
        shared: queue_mod.Queue = queue_mod.Queue()
        stop_event = threading.Event()
        _sentinel = object()
        n_subs = len(self._subscriptions)

        def feeder(sub: "SubscriptionHandle") -> None:
            try:
                while not stop_event.is_set():
                    for reading, handle in sub.readings(timeout=0.5):
                        shared.put((reading, handle))
                        if stop_event.is_set():
                            return
                    if sub.stopped:
                        return
            except Exception as exc:
                shared.put(exc)
            finally:
                shared.put(_sentinel)

        threads = []
        for sub in self._subscriptions:
            t = threading.Thread(target=feeder, args=(sub,), daemon=True)
            t.start()
            threads.append(t)

        start_time = time.monotonic()
        counter = 0
        finished_count = 0

        try:
            while True:
                if self.exc is not None:
                    raise self.exc

                # Calculate how long to block
                if timeout is not None:
                    remaining = timeout - (time.monotonic() - start_time)
                    if remaining <= 0:
                        break
                    block_time = min(remaining, 0.5)
                else:
                    block_time = 0.5  # periodic check for exc/stopped

                # Block until at least one reading arrives
                try:
                    item = shared.get(timeout=block_time)
                except queue_mod.Empty:
                    if self.stopped:
                        break
                    continue

                if item is _sentinel:
                    finished_count += 1
                    if finished_count >= n_subs:
                        break
                    continue

                if isinstance(item, Exception):
                    raise item

                # Got first reading -- drain all currently available into a heap
                heap = []
                reading, handle = item
                ts = reading.timestamp or datetime.min
                heapq.heappush(heap, (ts, counter, reading, handle))
                counter += 1

                while True:
                    try:
                        item = shared.get_nowait()
                    except queue_mod.Empty:
                        break
                    if item is _sentinel:
                        finished_count += 1
                        continue
                    if isinstance(item, Exception):
                        raise item
                    reading, handle = item
                    ts = reading.timestamp or datetime.min
                    heapq.heappush(heap, (ts, counter, reading, handle))
                    counter += 1

                # Yield batch in timestamp order
                while heap:
                    _, _, reading, handle = heapq.heappop(heap)
                    yield (reading, handle)

                if finished_count >= n_subs:
                    break
        finally:
            stop_event.set()
            for t in threads:
                t.join(timeout=2.0)

    def stop(self) -> None:
        """Stop all subscriptions."""
        for sub in self._subscriptions:
            sub.stop()

    def __enter__(self) -> "CombinedStream":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager - stops all subscriptions."""
        self.stop()
        return False
