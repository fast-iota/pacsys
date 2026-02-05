"""
Testing utilities - FakeBackend for unit tests without network.

See SPECIFICATION.md for available methods and pytest fixtures.
"""

from dataclasses import replace
from datetime import datetime
import queue
import threading
from typing import Any, Iterator, Callable

from pacsys.acnet.errors import ERR_NOPROP, ERR_OK, ERR_RETRY, FACILITY_ACNET, FACILITY_DBM
from pacsys.backends import Backend
from pacsys.drf3 import parse_request
from pacsys.drf3.field import DEFAULT_FIELD_FOR_PROPERTY
from pacsys.drf3.range import ARRAY_RANGE, BYTE_RANGE
from pacsys.types import (
    Value,
    Reading,
    WriteResult,
    BackendCapability,
    ValueType,
    DeviceMeta,
    SubscriptionHandle,
    ReadingCallback,
    ErrorCallback,
)
from pacsys.errors import DeviceError
from pacsys.drf_utils import get_device_name


def _normalize_drf(drf: str) -> str:
    """Normalize DRF to device-state key: device + property + field + extra.

    Strips events (acquisition mode) and ranges (resolved at read time)
    so that different access patterns (@I, @N, @p,1000) and different
    range requests all reference the same underlying device state.
    """
    try:
        req = parse_request(drf)
        out = req.device
        if req.property is not None:
            out += f".{req.property.name}"
        if req.field is not None:
            default = DEFAULT_FIELD_FOR_PROPERTY.get(req.property)
            if req.field != default:
                out += f".{req.field.name}"
        if req.extra is not None:
            out += f"<-{req.extra.name}"
        return out
    except ValueError:
        return drf


def _get_range(drf: str) -> ARRAY_RANGE | BYTE_RANGE | None:
    """Extract range from a DRF string."""
    try:
        return parse_request(drf).range
    except ValueError:
        return None


def _apply_range(drf: str, value: Any, rng: ARRAY_RANGE | BYTE_RANGE | None) -> Any:
    """Slice value according to a DRF range. Returns value unchanged if no range."""
    if rng is None:
        return value

    try:
        if isinstance(rng, ARRAY_RANGE):
            if rng.mode == "full":
                return value[:]
            if rng.mode == "single":
                return value[rng.low]
            low = rng.low if rng.low is not None else 0
            high = rng.high
            if high is not None:
                return value[low : high + 1]  # ACNET ranges are inclusive
            return value[low:]

        if isinstance(rng, BYTE_RANGE):
            if rng.mode == "full":
                return value[:]
            if rng.mode == "single":
                assert rng.offset is not None
                return value[rng.offset : rng.offset + 1]
            offset = rng.offset if rng.offset is not None else 0
            length = rng.length
            if length is not None:
                return value[offset : offset + length]
            return value[offset:]
    except (TypeError, IndexError) as e:
        raise DeviceError(drf, FACILITY_ACNET, ERR_RETRY, f"Cannot apply range to stored value: {e}")

    return value


def _write_range(existing: Any, value: Any, rng: ARRAY_RANGE | BYTE_RANGE) -> Any:
    """Slice-assign value into existing array/bytes at the given range.

    Returns a new object with the slice replaced (copies numpy arrays
    to keep stored Readings effectively immutable).
    """
    import numpy as np

    if isinstance(existing, np.ndarray):
        out = existing.copy()
    elif isinstance(existing, (list, bytearray)):
        out = existing.copy()
    elif isinstance(existing, bytes):
        out = bytearray(existing)
    else:
        raise TypeError(f"Cannot slice-assign into {type(existing).__name__}")

    if isinstance(rng, ARRAY_RANGE):
        if rng.mode == "full":
            out[:] = value
        elif rng.mode == "single":
            assert rng.low is not None
            out[rng.low] = value
        else:
            low = rng.low if rng.low is not None else 0
            high = rng.high
            if high is not None:
                out[low : high + 1] = value
            else:
                out[low:] = value
    elif isinstance(rng, BYTE_RANGE):
        if rng.mode == "full":
            out[:] = value
        elif rng.mode == "single":
            assert rng.offset is not None
            out[rng.offset : rng.offset + 1] = value
        else:
            offset = rng.offset if rng.offset is not None else 0
            length = rng.length
            if length is not None:
                out[offset : offset + length] = value
            else:
                out[offset:] = value

    if isinstance(existing, bytes):
        return bytes(out)
    return out


class FakeSubscriptionHandle(SubscriptionHandle):
    """Subscription handle for FakeBackend streaming tests.

    Supports both callback mode (readings pushed to callback) and
    iterator mode (readings yielded from readings() method).

    Example (iterator mode):
        fake = FakeBackend()
        with fake.subscribe(["M:OUTTMP@p,1000"]) as sub:
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)
            for reading, handle in sub.readings(timeout=1.0):
                assert reading.value == 72.5
                break

    Example (callback mode):
        readings = []
        def on_reading(reading, handle):
            readings.append(reading)

        fake = FakeBackend()
        sub = fake.subscribe(["M:OUTTMP@p,1000"], callback=on_reading)
        fake.emit_reading("M:OUTTMP@p,1000", 72.5)
        assert len(readings) == 1
        sub.stop()
    """

    def __init__(
        self,
        drfs: list[str],
        callback: ReadingCallback | None,
        on_error: ErrorCallback | None,
        remover: Callable[["FakeSubscriptionHandle"], None],
    ):
        self._drfs = set(drfs)
        self._callback = callback
        self._on_error = on_error
        self._remover = remover
        self._queue: queue.Queue[Reading | None] = queue.Queue()
        self._stopped = False
        self._exc: Exception | None = None
        self._ref_ids = list(range(len(drfs)))

    @property
    def ref_ids(self) -> list[int]:
        """Reference IDs for devices in this subscription."""
        return self._ref_ids

    @property
    def stopped(self) -> bool:
        """True if this subscription has been stopped."""
        return self._stopped

    @property
    def exc(self) -> Exception | None:
        """Exception if an error occurred, else None."""
        return self._exc

    def readings(
        self,
        timeout: float | None = None,
    ) -> Iterator[tuple[Reading, SubscriptionHandle]]:
        """Yield (reading, handle) pairs for this subscription.

        Args:
            timeout: Seconds to wait for next reading. None = block forever.
        """
        if self._exc is not None:
            raise self._exc

        while not self._stopped:
            try:
                reading = self._queue.get(block=True, timeout=timeout)
                if reading is None:
                    # Poison pill - stop signal
                    break
                yield (reading, self)
            except queue.Empty:
                # Timeout reached
                break

    def stop(self) -> None:
        """Stop this subscription."""
        if not self._stopped:
            self._stopped = True
            self._queue.put(None)  # Poison pill to unblock readings()
            self._remover(self)

    def _put_reading(self, reading: Reading) -> None:
        """Internal: deliver reading to callback or queue."""
        if self._stopped:
            return

        if self._callback:
            self._callback(reading, self)
        else:
            self._queue.put(reading)

    def _set_error(self, exc: Exception) -> None:
        """Internal: set error and notify callback."""
        self._exc = exc
        if self._on_error:
            self._on_error(exc, self)
        else:
            self._queue.put(None)  # Unblock iterator


class FakeBackend(Backend):
    """Fake backend that simulates device state without network access.

    Models devices as shared state: writes update the stored value,
    subsequent reads return it.  Keys are device identity (name + property +
    field) -- events and ranges are stripped so @I, @N, @p,1000 all hit the
    same state.  Range requests slice the stored value at read time.

    Example:
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.set_error("M:BADDEV", -42, "Device not found")

        # Use it like any other backend
        assert fake.read("M:OUTTMP") == 72.5
        reading = fake.get("M:BADDEV")
        assert reading.is_error

        # Writes update readable state
        fake.write("M:OUTTMP.SETTING@N", 80.0)
        assert fake.read("M:OUTTMP.SETTING@I") == 80.0

        # Inspect what happened
        assert fake.was_read("M:OUTTMP")
        assert fake.reads == ["M:OUTTMP", "M:BADDEV", "M:OUTTMP.SETTING@I"]
    """

    def __init__(self):
        """Create a fake backend with no configured readings."""
        self._readings: dict[str, Reading] = {}
        self._errors: dict[str, tuple[int, str]] = {}
        self._write_results: dict[str, WriteResult] = {}
        self._read_history: list[str] = []
        self._write_history: list[tuple[str, Value]] = []
        self._subscriptions: list[FakeSubscriptionHandle] = []
        self._lock = threading.Lock()
        self._closed = False

    # ─────────────────────────────────────────────────────────────────────
    # Configuration Methods (call these in test setup)
    # ─────────────────────────────────────────────────────────────────────

    def set_reading(
        self,
        drf: str,
        value: Value,
        value_type: ValueType = ValueType.SCALAR,
        units: str | None = None,
        description: str | None = None,
        timestamp: datetime | None = None,
        cycle: int = 0,
    ) -> None:
        """Pre-configure a successful reading for a device.

        Args:
            drf: Device request string
            value: The value to return when this device is read
            value_type: Type of the value (default: SCALAR)
            units: Optional units string
            description: Optional device description
            timestamp: Optional timestamp for the reading
            cycle: Cycle number (default: 0)
        """
        device_name = get_device_name(drf)

        meta = DeviceMeta(
            device_index=0,
            name=device_name,
            description=description or f"Test device {device_name}",
            units=units,
        )

        key = _normalize_drf(drf)
        self._readings[key] = Reading(
            drf=drf,
            value_type=value_type,
            value=value,
            error_code=ERR_OK,
            timestamp=timestamp or datetime.now(),
            cycle=cycle,
            meta=meta,
        )

        # Remove any error for this DRF
        self._errors.pop(key, None)

    def set_error(self, drf: str, error_code: int, message: str) -> None:
        """Pre-configure an error for a device.

        Args:
            drf: Device request string
            error_code: Negative error code
            message: Error message
        """
        key = _normalize_drf(drf)
        self._errors[key] = (error_code, message)
        # Remove any reading for this DRF
        self._readings.pop(key, None)

    def set_write_result(
        self,
        drf: str,
        success: bool = True,
        error_code: int | None = None,
        message: str | None = None,
    ) -> None:
        """Pre-configure the result of a write operation.

        Args:
            drf: Device request string
            success: If True, use error_code=ERR_OK (unless overridden)
            error_code: Status code to return (default: ERR_OK if success, ERR_RETRY if not)
            message: Optional message
        """
        if error_code is None:
            error_code = ERR_OK if success else ERR_RETRY

        self._write_results[_normalize_drf(drf)] = WriteResult(
            drf=drf,
            error_code=error_code,
            message=message,
        )

    def set_analog_alarm(self, drf: str, alarm_dict: dict) -> None:
        """Pre-configure an analog alarm structured reading.

        Args:
            drf: Device request string (e.g., "Z:TEST.ANALOG")
            alarm_dict: Dictionary with alarm fields:
                minimum, maximum, alarm_enable, alarm_status,
                abort, abort_inhibit, tries_needed, tries_now
        """
        self.set_reading(drf, alarm_dict, value_type=ValueType.ANALOG_ALARM)

    def set_digital_alarm(self, drf: str, alarm_dict: dict) -> None:
        """Pre-configure a digital alarm structured reading.

        Args:
            drf: Device request string (e.g., "Z:TEST.DIGITAL")
            alarm_dict: Dictionary with alarm fields:
                nominal, mask, alarm_enable, alarm_status,
                abort, abort_inhibit, tries_needed, tries_now
        """
        self.set_reading(drf, alarm_dict, value_type=ValueType.DIGITAL_ALARM)

    # ─────────────────────────────────────────────────────────────────────
    # Inspection Methods (call these in test assertions)
    # ─────────────────────────────────────────────────────────────────────

    @property
    def reads(self) -> list[str]:
        """Get list of DRF strings that were read (in order)."""
        return self._read_history.copy()

    @property
    def writes(self) -> list[tuple[str, Value]]:
        """Get list of (drf, value) tuples that were written (in order)."""
        return self._write_history.copy()

    def was_read(self, drf: str) -> bool:
        """Check if a specific device was read (normalizes DRF for comparison)."""
        key = _normalize_drf(drf)
        return any(_normalize_drf(d) == key for d in self._read_history)

    def was_written(self, drf: str) -> bool:
        """Check if a specific device was written (normalizes DRF for comparison)."""
        key = _normalize_drf(drf)
        return any(_normalize_drf(d) == key for d, _ in self._write_history)

    def get_written_value(self, drf: str) -> Value | None:
        """Get the value that was written to a device (last write, normalizes DRF)."""
        key = _normalize_drf(drf)
        for d, v in reversed(self._write_history):
            if _normalize_drf(d) == key:
                return v
        return None

    def reset(self) -> None:
        """Clear all configured readings, errors, recorded operations, and subscriptions."""
        self._readings.clear()
        self._errors.clear()
        self._write_results.clear()
        self._read_history.clear()
        self._write_history.clear()
        self.stop_streaming()

    def _device_known(self, drf: str) -> bool:
        """Check if any reading or error is configured for this device name."""
        name = get_device_name(drf)
        return any(k.split(".")[0] == name for k in self._readings) or any(
            k.split(".")[0] == name for k in self._errors
        )

    # ─────────────────────────────────────────────────────────────────────
    # Backend Interface Implementation
    # ─────────────────────────────────────────────────────────────────────

    @property
    def capabilities(self) -> BackendCapability:
        """FakeBackend supports everything."""
        return BackendCapability.READ | BackendCapability.WRITE | BackendCapability.STREAM | BackendCapability.BATCH

    @property
    def authenticated(self) -> bool:
        """FakeBackend is always 'authenticated'."""
        return True

    @property
    def principal(self) -> str | None:
        """Return a fake principal."""
        return "test-user@FNAL.GOV"

    def read(self, drf: str, timeout: float | None = None) -> Value:
        """Read a pre-configured value or raise error.

        Args:
            drf: Device request string
            timeout: Ignored (for interface compatibility)

        Returns:
            The pre-configured value (sliced if DRF includes a range)

        Raises:
            DeviceError: If an error was configured or no reading exists
        """
        self._read_history.append(drf)
        key = _normalize_drf(drf)

        # Check for configured error
        if key in self._errors:
            error_code, message = self._errors[key]
            raise DeviceError(drf, FACILITY_ACNET, error_code, message)

        # Check for configured reading
        if key in self._readings:
            reading = self._readings[key]
            if reading.is_error:
                raise DeviceError(
                    drf,
                    reading.facility_code,
                    reading.error_code,
                    reading.message,
                )
            return _apply_range(drf, reading.value, _get_range(drf))

        # No reading configured -- distinguish property-not-found from unknown device
        if self._device_known(drf):
            raise DeviceError(drf, FACILITY_DBM, ERR_NOPROP, f"No such property for {get_device_name(drf)}")
        raise DeviceError(drf, FACILITY_ACNET, ERR_RETRY, f"No reading configured for {drf}")

    def get(self, drf: str, timeout: float | None = None) -> Reading:
        """Get a pre-configured reading.

        Args:
            drf: Device request string
            timeout: Ignored (for interface compatibility)

        Returns:
            Reading object (may have is_error=True)
        """
        self._read_history.append(drf)
        key = _normalize_drf(drf)

        # Check for configured error
        if key in self._errors:
            error_code, message = self._errors[key]
            return Reading(
                drf=drf,
                value_type=ValueType.SCALAR,
                error_code=error_code,
                message=message,
            )

        # Check for configured reading
        if key in self._readings:
            reading = self._readings[key]
            rng = _get_range(drf)
            if rng is not None and reading.value is not None:
                return replace(reading, value=_apply_range(drf, reading.value, rng))
            return reading

        # No reading configured -- distinguish property-not-found from unknown device
        if self._device_known(drf):
            return Reading(
                drf=drf,
                value_type=ValueType.SCALAR,
                facility_code=FACILITY_DBM,
                error_code=ERR_NOPROP,
                message=f"No such property for {get_device_name(drf)}",
            )
        return Reading(
            drf=drf,
            value_type=ValueType.SCALAR,
            facility_code=FACILITY_ACNET,
            error_code=ERR_RETRY,
            message=f"No reading configured for {drf}",
        )

    def get_many(self, drfs: list[str], timeout: float | None = None) -> list[Reading]:
        """Get multiple pre-configured readings.

        Args:
            drfs: List of device request strings
            timeout: Ignored (for interface compatibility)

        Returns:
            List of Reading objects in same order as input
        """
        return [self.get(drf, timeout) for drf in drfs]

    def write(
        self,
        drf: str,
        value: Value,
        timeout: float | None = None,
    ) -> WriteResult:
        """Write a value and update device state.

        Successful writes update the stored reading so that subsequent
        reads of the same device+property return the written value.

        Args:
            drf: Device to write
            value: Value to write
            timeout: Ignored (for interface compatibility)

        Returns:
            Pre-configured WriteResult, or success by default
        """
        self._write_history.append((drf, value))
        key = _normalize_drf(drf)

        # Check for configured write result
        if key in self._write_results:
            result = self._write_results[key]
            if result.success:
                self._update_state(key, drf, value)
            return result

        # Default: write succeeds and updates state
        self._update_state(key, drf, value)
        return WriteResult(drf=drf, error_code=ERR_OK)

    def _update_state(self, key: str, drf: str, value: Value) -> None:
        """Update device state after a successful write.

        If the write DRF includes a range and an existing array is stored,
        performs a slice assignment instead of replacing the whole value.
        """
        # Clear any error -- device is now responsive
        self._errors.pop(key, None)

        rng = _get_range(drf)
        merged = value

        # Ranged write: slice-assign into existing stored value
        if rng is not None and key in self._readings:
            existing = self._readings[key].value
            merged = _write_range(existing, value, rng)

        if key in self._readings:
            old = self._readings[key]
            self._readings[key] = replace(old, value=merged, error_code=ERR_OK, timestamp=datetime.now())
        else:
            device_name = get_device_name(drf)
            self._readings[key] = Reading(
                drf=drf,
                value_type=ValueType.SCALAR,
                value=value,
                error_code=ERR_OK,
                timestamp=datetime.now(),
                meta=DeviceMeta(
                    device_index=0,
                    name=device_name,
                    description=f"Test device {device_name}",
                ),
            )

    def write_many(
        self,
        settings: list[tuple[str, Value]],
        timeout: float | None = None,
    ) -> list[WriteResult]:
        """Record multiple write operations.

        Args:
            settings: List of (drf, value) tuples
            timeout: Ignored (for interface compatibility)

        Returns:
            List of WriteResult objects
        """
        return [self.write(drf, value, timeout=timeout) for drf, value in settings]

    # ─────────────────────────────────────────────────────────────────────
    # Streaming Methods
    # ─────────────────────────────────────────────────────────────────────

    def subscribe(
        self,
        drfs: list[str],
        callback: ReadingCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> FakeSubscriptionHandle:
        """Subscribe to devices for streaming.

        Args:
            drfs: List of device request strings
            callback: Optional callback for push-mode (called on each reading)
            on_error: Optional error handler (called on connection errors)

        Returns:
            FakeSubscriptionHandle for managing subscription
        """
        handle = FakeSubscriptionHandle(drfs, callback, on_error, self._remove_subscription)
        with self._lock:
            self._subscriptions.append(handle)
        return handle

    def emit_reading(
        self,
        drf: str,
        value: Value,
        value_type: ValueType = ValueType.SCALAR,
        units: str | None = None,
        description: str | None = None,
        timestamp: datetime | None = None,
        cycle: int = 0,
    ) -> None:
        """Emit a reading to all matching subscriptions.

        Use this in tests to simulate incoming data from the server.

        Args:
            drf: Device request string (must match subscription DRF)
            value: The value to emit
            value_type: Type of the value (default: SCALAR)
            units: Optional units string
            description: Optional device description
            timestamp: Optional timestamp (default: now)
            cycle: Cycle number (default: 0)
        """
        device_name = get_device_name(drf)
        meta = DeviceMeta(
            device_index=0,
            name=device_name,
            description=description or f"Test device {device_name}",
            units=units,
        )
        reading = Reading(
            drf=drf,
            value_type=value_type,
            value=value,
            error_code=ERR_OK,
            timestamp=timestamp or datetime.now(),
            cycle=cycle,
            meta=meta,
        )

        # Deliver to matching subscriptions (copy list to avoid holding lock during callbacks)
        with self._lock:
            matching = [sub for sub in self._subscriptions if drf in sub._drfs]
        for sub in matching:
            sub._put_reading(reading)

    def emit_error(self, exception: Exception) -> None:
        """Emit an error to all active subscriptions.

        Use this in tests to simulate connection errors.

        Args:
            exception: The exception to deliver
        """
        with self._lock:
            subs = list(self._subscriptions)
        for sub in subs:
            sub._set_error(exception)

    def remove(self, handle: SubscriptionHandle) -> None:
        """Remove a subscription.

        Args:
            handle: The subscription handle to remove
        """
        handle.stop()

    def stop_streaming(self) -> None:
        """Stop all active subscriptions."""
        with self._lock:
            subs = list(self._subscriptions)
        for sub in subs:
            sub.stop()

    def _remove_subscription(self, handle: FakeSubscriptionHandle) -> None:
        """Internal: remove subscription from tracking."""
        with self._lock:
            if handle in self._subscriptions:
                self._subscriptions.remove(handle)

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close backend and stop all subscriptions."""
        self.stop_streaming()
        self._closed = True


# ─────────────────────────────────────────────────────────────────────────────
# Optional pytest integration
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pytest

    @pytest.fixture
    def fake_backend():
        """Pytest fixture providing a FakeBackend instance.

        Usage:
            def test_something(fake_backend):
                fake_backend.set_reading("M:OUTTMP", 72.5)
                # ... test code using fake_backend ...
        """
        return FakeBackend()

    @pytest.fixture
    def mock_pacsys(fake_backend, monkeypatch):
        """Pytest fixture that patches the global pacsys backend.

        This allows testing code that uses pacsys.read() etc. without
        manually patching.

        Usage:
            def test_something(mock_pacsys):
                mock_pacsys.set_reading("M:OUTTMP", 72.5)
                import pacsys
                assert pacsys.read("M:OUTTMP") == 72.5
        """
        import pacsys

        monkeypatch.setattr(pacsys, "_global_dpm_backend", fake_backend)
        monkeypatch.setattr(pacsys, "_backend_initialized", True)
        yield fake_backend

except ImportError:
    # pytest not installed, fixtures not available
    pass


__all__ = ["FakeBackend", "FakeSubscriptionHandle"]
