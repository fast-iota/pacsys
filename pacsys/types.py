"""
Core data types - Reading, WriteResult, SubscriptionHandle, CombinedStream.
"""

import base64
import inspect
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, Flag, IntEnum, auto
from typing import TYPE_CHECKING, Any, Union, cast

if TYPE_CHECKING:
    import numpy as np

    from pacsys.device import Device

from pacsys.drf_utils import get_device_name as _get_device_name

# Value types supported by ACNET
# np.ndarray is only in the annotation at type-check time; at runtime the
# alias omits it (numpy is heavy to import and not needed for annotation eval).
if TYPE_CHECKING:
    Value = float | int | str | bytes | np.ndarray | list | dict
else:
    Value = float | int | str | bytes | list | dict


def _loaded_numpy_types(*names: str) -> tuple[type, ...]:
    """Return requested numpy types without importing numpy."""
    np = sys.modules.get("numpy")
    if np is None:
        return ()
    return tuple(candidate for name in names if isinstance((candidate := getattr(np, name, None)), type))


def _value_to_json(value: object) -> object:
    """Convert a Value to a JSON-serializable Python object.

    Handles numpy arrays/scalars (→ lists/primitives), bytes (→ base64),
    and dicts with numpy array values (e.g. TIMED_SCALAR_ARRAY).
    """
    if value is None:
        return None
    if isinstance(value, _loaded_numpy_types("ndarray")):
        return cast("Any", value).tolist()
    if isinstance(value, _loaded_numpy_types("integer", "floating", "bool_")):
        return cast("Any", value).item()
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {k: _value_to_json(v) for k, v in value.items()}
    return value


def _value_from_json(
    value: object, value_type: "ValueType | None", dtype: "str | dict[str, str] | None" = None
) -> "Value | None":
    """Reconstruct a Value from its JSON representation, ValueType, and recorded dtype(s)."""
    if value is None:
        return None
    if value_type is None:
        return cast("Value", value)
    if value_type in (ValueType.SCALAR_ARRAY, ValueType.TIMED_SCALAR_ARRAY):
        import numpy as np

        if isinstance(value, dict):
            dtypes = dtype if isinstance(dtype, dict) else {}
            return {k: np.array(v, dtype=dtypes.get(k)) for k, v in value.items()}
        return np.array(value, dtype=dtype if isinstance(dtype, str) else None)
    if value_type == ValueType.RAW and isinstance(value, str):
        return base64.b64decode(value, validate=True)
    return cast("Value", value)


def _value_dtype(value: object) -> "str | dict[str, str] | None":
    """Dtype string(s) of ndarray content in a Value, recorded for exact round-trip.

    Uses ``dtype.str`` (e.g. ``'<f8'``, ``'<U2'``) — unlike ``dtype.name``, it is
    valid ``np.dtype()`` input for every dtype (``'str64'`` is not).
    """
    ndarray_types = _loaded_numpy_types("ndarray")
    if isinstance(value, ndarray_types):
        return cast("Any", value).dtype.str
    if isinstance(value, dict):
        dtypes = {k: cast("Any", v).dtype.str for k, v in value.items() if isinstance(v, ndarray_types)}
        return cast("dict[str, str]", dtypes) or None
    return None


def _infer_serialization_type(value: object) -> "ValueType | None":
    """Serialization tag for a bare Value (WriteResult.readback has no value_type).

    Only types whose JSON form is lossy need a tag: ndarray, ndarray-valued
    dicts (timed arrays), and bytes. Scalars, str, and lists round-trip as
    plain JSON and stay untagged.
    """
    ndarray_types = _loaded_numpy_types("ndarray")
    if isinstance(value, ndarray_types):
        return ValueType.SCALAR_ARRAY
    if (
        ndarray_types
        and isinstance(value, dict)
        and value
        and all(isinstance(v, ndarray_types) for v in value.values())
    ):
        return ValueType.TIMED_SCALAR_ARRAY
    if isinstance(value, (bytes, bytearray)):
        return ValueType.RAW
    return None


# Type alias for functions accepting DRF strings or Device objects
DeviceSpec = Union[str, "Device"]

# Write settings: list of (device, value) tuples or a dict mapping device -> value
WriteSettings = list[tuple[DeviceSpec, Value]] | dict[DeviceSpec, Value]

# Callback type for streaming subscriptions - receives (reading, handle) pairs
ReadingCallback = Callable[["Reading", "SubscriptionHandle"], None]

# Callback type for error handling in streaming - receives (exception, handle) pairs
ErrorCallback = Callable[[Exception, "SubscriptionHandle"], None]


def _validate_callback_signature(fn: Any, name: str, arguments: str) -> None:
    """Require that *fn* can be called with two positional arguments."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return
    try:
        sig.bind(object(), object())
    except TypeError as exc:
        raise TypeError(f"{name} must accept 2 arguments ({arguments}), but {fn!r} cannot: {exc}") from None


def _validate_callback(callback: object, on_error: object, *, event_hint: bool = False) -> None:
    """Validate subscription callback callability and arity."""
    if callback is not None:
        if not callable(callback):
            message = f"callback must be callable, got {type(callback).__name__}"
            if event_hint:
                message += f" — did you mean subscribe(event={callback!r})?"
            raise TypeError(message)
        _validate_callback_signature(callback, "callback", "reading, handle")
    if on_error is not None:
        if not callable(on_error):
            raise TypeError(f"on_error must be callable, got {type(on_error).__name__}")
        _validate_callback_signature(on_error, "on_error", "exception, handle")


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
    units: str | None = None
    format_hint: int | None = None

    def to_dict(self) -> dict:
        d: dict = {"device_index": self.device_index, "name": self.name, "description": self.description}
        if self.units is not None:
            d["units"] = self.units
        if self.format_hint is not None:
            d["format_hint"] = self.format_hint
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceMeta":
        return cls(
            device_index=d["device_index"],
            name=d["name"],
            description=d["description"],
            units=d.get("units"),
            format_hint=d.get("format_hint"),
        )


def _frozen_value(value: object) -> object:
    """Return value with ndarrays read-only, for stable hashing.

    Buffer-owning arrays are frozen in place (zero-copy: passing one into a
    Reading/WriteResult transfers ownership — later caller mutation raises).
    Views are replaced by read-only copies since freezing a view would leave
    its base buffer writable and the hash unstable.
    """
    if isinstance(value, dict):
        return {k: _frozen_value(v) for k, v in value.items()}
    # An ndarray (or subclass, from any module) implies numpy is already
    # imported — check sys.modules so numpy-less environments never pay for it
    ndarray_types = _loaded_numpy_types("ndarray")
    if not isinstance(value, ndarray_types):
        return value
    value = cast("Any", value)
    if value.base is not None:
        value = value.copy()
    value.flags.writeable = False
    # MaskedArray: the mask is a separate buffer that also feeds tobytes()/hash.
    # Freeze the stored _mask — the .mask property returns a fresh view per access
    mask = getattr(value, "_mask", None)
    if isinstance(mask, ndarray_types):
        mask = cast("Any", mask)
        if mask.ndim:
            mask.flags.writeable = False
    return value


def _value_hashable(value: object) -> object:
    """Convert a Value to something hashable for use in __hash__."""
    if value is None:
        return None
    if isinstance(value, _loaded_numpy_types("ndarray")):
        value = cast("Any", value)
        return (value.dtype.str, value.tobytes())
    if isinstance(value, (dict, list)):
        # Mutable after construction - a hash would drift and corrupt any set/dict holding it
        raise TypeError(f"unhashable value of type {type(value).__name__}")
    return value


def _values_equal(a: "Value | None", b: "Value | None") -> bool:
    """Compare two Value objects, handling numpy arrays correctly."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    ndarray_types = _loaded_numpy_types("ndarray")
    if isinstance(a, ndarray_types) or isinstance(b, ndarray_types):
        if not isinstance(a, ndarray_types) or not isinstance(b, ndarray_types):
            return False
        import numpy as np

        array_a = cast("Any", a)
        array_b = cast("Any", b)
        return (
            array_a.dtype == array_b.dtype
            # equal_nan for NaN/NaT-capable kinds only; it raises on str/object dtypes
            and np.array_equal(array_a, array_b, equal_nan=array_a.dtype.kind in "fcmM")
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    return a == b


@dataclass(frozen=True)
class Reading:
    """A device reading with status and optional data.

    Status semantics (matches gRPC Status message):
    - facility_code: ACNET facility identifier (0=success, 1=ACNET, 16=DBM, 17=DPM)
    - error_code: 0=success, >0=warning, <0=error
    """

    drf: str
    value_type: ValueType | None = None
    facility_code: int = 0
    error_code: int = 0
    value: Value | None = None
    message: str | None = None
    timestamp: datetime | None = None
    cycle: int | None = None
    meta: DeviceMeta | None = None

    def __post_init__(self) -> None:
        if self.value is not None and self.value_type is None:
            raise ValueError("value_type is required when value is set")
        object.__setattr__(self, "value", _frozen_value(self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Reading):
            return NotImplemented
        return (
            self.drf == other.drf
            and self.value_type == other.value_type
            and self.facility_code == other.facility_code
            and self.error_code == other.error_code
            and self.message == other.message
            and self.timestamp == other.timestamp
            and self.cycle == other.cycle
            and self.meta == other.meta
            and _values_equal(self.value, other.value)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.drf,
                self.value_type,
                self.facility_code,
                self.error_code,
                self.message,
                self.timestamp,
                self.cycle,
                self.meta,
                _value_hashable(self.value),
            )
        )

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
    def units(self) -> str | None:
        """Engineering units from metadata, or None if unavailable."""
        return self.meta.units if self.meta else None

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict. Round-trippable via ``Reading.from_dict()``."""
        d: dict = {
            "drf": self.drf,
            "facility_code": self.facility_code,
            "error_code": self.error_code,
        }
        if self.value_type is not None:
            d["value_type"] = self.value_type.value
        if self.value is not None:
            d["value"] = _value_to_json(self.value)
            if (dt := _value_dtype(self.value)) is not None:
                d["value_dtype"] = dt
        if self.message is not None:
            d["message"] = self.message
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp.isoformat()
        if self.cycle is not None:
            d["cycle"] = self.cycle
        if self.meta is not None:
            d["meta"] = self.meta.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Reading":
        """Deserialize from a dict produced by ``to_dict()``."""
        vt = ValueType(d["value_type"]) if "value_type" in d else None
        ts = datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else None
        meta = DeviceMeta.from_dict(d["meta"]) if "meta" in d else None
        return cls(
            drf=d["drf"],
            value_type=vt,
            facility_code=d.get("facility_code", 0),
            error_code=d.get("error_code", 0),
            value=_value_from_json(d.get("value"), vt, d.get("value_dtype")),
            message=d.get("message"),
            timestamp=ts,
            cycle=d.get("cycle"),
            meta=meta,
        )


@dataclass(frozen=True)
class WriteResult:
    """Result of a write operation, optionally with verification info."""

    drf: str
    facility_code: int = 0
    error_code: int = 0
    message: str | None = None
    verified: bool | None = None  # True=readback matched, False=failed, None=no verify
    readback: Value | None = None
    skipped: bool = False
    attempts: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "readback", _frozen_value(self.readback))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WriteResult):
            return NotImplemented
        return (
            self.drf == other.drf
            and self.facility_code == other.facility_code
            and self.error_code == other.error_code
            and self.message == other.message
            and self.verified == other.verified
            and self.skipped == other.skipped
            and self.attempts == other.attempts
            and _values_equal(self.readback, other.readback)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.drf,
                self.facility_code,
                self.error_code,
                self.message,
                self.verified,
                self.skipped,
                self.attempts,
                _value_hashable(self.readback),
            )
        )

    @property
    def ok(self) -> bool:
        """True if write succeeded (error_code == 0)."""
        return self.error_code == 0

    @property
    def success(self) -> bool:
        """Alias for ok."""
        return self.ok

    @property
    def confirmed(self) -> bool:
        """True if the write succeeded and requested verification did not fail."""
        return self.ok and self.verified is not False

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict. Round-trippable via ``WriteResult.from_dict()``."""
        d: dict = {
            "drf": self.drf,
            "facility_code": self.facility_code,
            "error_code": self.error_code,
        }
        if self.message is not None:
            d["message"] = self.message
        if self.verified is not None:
            d["verified"] = self.verified
        if self.readback is not None:
            d["readback"] = _value_to_json(self.readback)
            if (vt := _infer_serialization_type(self.readback)) is not None:
                d["readback_type"] = vt.value
                if (dt := _value_dtype(self.readback)) is not None:
                    d["readback_dtype"] = dt
        if self.skipped:
            d["skipped"] = self.skipped
        if self.attempts:
            d["attempts"] = self.attempts
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WriteResult":
        """Deserialize from a dict produced by ``to_dict()``."""
        rvt = ValueType(d["readback_type"]) if "readback_type" in d else None
        return cls(
            drf=d["drf"],
            facility_code=d.get("facility_code", 0),
            error_code=d.get("error_code", 0),
            message=d.get("message"),
            verified=d.get("verified"),
            readback=_value_from_json(d.get("readback"), rvt, d.get("readback_dtype")),
            skipped=d.get("skipped", False),
            attempts=d.get("attempts", 0),
        )


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
    def exc(self) -> Exception | None:
        """Exception if an error occurred, else None."""
        raise NotImplementedError

    @property
    def dropped(self) -> int:
        """Readings discarded because the buffer or callback queue was full (cumulative)."""
        return 0

    def _note_dispatch_drop(self) -> None:
        """Called by the callback dispatcher when it had to drop a reading for this handle."""

    def readings(
        self,
        timeout: float | None = None,
    ) -> Iterator[tuple["Reading", "SubscriptionHandle"]]:
        """Yield (reading, handle) pairs for THIS subscription.

        Args:
            timeout: Total wall-clock window in seconds.
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


def _reading_timestamp_key(reading: Reading) -> datetime:
    """Sort missing timestamps first and interpret naive timestamps as UTC."""
    timestamp = reading.timestamp
    if timestamp is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp


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
    def exc(self) -> Exception | None:
        """First exception from any subscription, or None."""
        for sub in self._subscriptions:
            if sub.exc is not None:
                return sub.exc
        return None

    def readings(
        self,
        timeout: float | None = None,
    ) -> Iterator[tuple["Reading", "SubscriptionHandle"]]:
        """Yield readings from all subscriptions.

        Readings are sorted by timestamp within each batch of available data,
        but global ordering is not guaranteed if streams have different latencies.
        Naive timestamps are interpreted as UTC; missing timestamps sort first.

        Delivery is destructive and at-most-once: readings are prefetched off the
        constituent handles for batching, and prefetched-but-unyielded readings are
        discarded (not requeued) if the iterator is abandoned early, the timeout
        expires, or a subscription reports an error. Errors raise promptly without
        draining buffers first.

        Args:
            timeout: Total timeout for the combined stream in seconds.
                    None = block forever (until all stopped)
                    0 = non-blocking (drain buffered readings only, including
                        buffers of already-stopped subscriptions)

        Yields:
            (reading, handle) pairs from any subscription

        Raises:
            Exception: If any subscription has an error and no on_error was provided
        """
        import heapq
        import queue as queue_mod
        import threading
        import time

        if timeout == 0:
            if self.exc is not None:
                raise self.exc
            heap: list[tuple[datetime, int, Reading, SubscriptionHandle]] = []
            counter = 0
            for sub in self._subscriptions:
                # Stopped handles may still hold buffered readings -- drain them too
                for reading, handle in sub.readings(timeout=0):
                    ts = _reading_timestamp_key(reading)
                    heapq.heappush(heap, (ts, counter, reading, handle))
                    counter += 1
            while heap:
                _, _, reading, handle = heapq.heappop(heap)
                yield (reading, handle)
            return

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
            except Exception as exc:  # noqa: BLE001
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

                if timeout is not None:
                    remaining = timeout - (time.monotonic() - start_time)
                    if remaining <= 0:
                        break
                    block_time = min(remaining, 0.5)
                else:
                    block_time = 0.5  # periodic check for exc/stopped

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
                ts = _reading_timestamp_key(reading)
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
                    ts = _reading_timestamp_key(reading)
                    heapq.heappush(heap, (ts, counter, reading, handle))
                    counter += 1

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
