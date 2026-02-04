"""
Backend abstract base class. See SPECIFICATION.md for backend comparison.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pacsys.types import (
    Value,
    Reading,
    WriteResult,
    BackendCapability,
    SubscriptionHandle,
    ReadingCallback,
    ErrorCallback,
)


def timestamp_from_millis(millis: int) -> datetime:
    """Convert timestamp (milliseconds since Unix epoch) to datetime."""
    return datetime.fromtimestamp(millis / 1_000)


# Alarm dict key sets shared by DMQ and gRPC backends
ALARM_ANALOG_ONLY_KEYS = frozenset({"minimum", "maximum"})
ALARM_DIGITAL_ONLY_KEYS = frozenset({"nominal", "mask"})
ALARM_SHARED_KEYS = frozenset({"alarm_enable", "abort_inhibit", "tries_needed"})
ALARM_ANALOG_KEYS = ALARM_ANALOG_ONLY_KEYS | ALARM_SHARED_KEYS
ALARM_DIGITAL_KEYS = ALARM_DIGITAL_ONLY_KEYS | ALARM_SHARED_KEYS
ALARM_READONLY_KEYS = frozenset({"abort", "alarm_status", "tries_now"})


def validate_alarm_dict(d: dict) -> str:
    """Validate an alarm dict and return 'analog' or 'digital'.

    Raises ValueError on unknown keys, mixed types, or missing type-specific keys.
    """
    keys = set(d) - ALARM_READONLY_KEYS
    unknown = keys - ALARM_ANALOG_KEYS - ALARM_DIGITAL_KEYS
    if unknown:
        raise ValueError(f"Unknown alarm dict keys: {unknown}")
    has_analog = bool(keys & ALARM_ANALOG_ONLY_KEYS)
    has_digital = bool(keys & ALARM_DIGITAL_ONLY_KEYS)
    if has_analog and has_digital:
        raise ValueError("Cannot mix analog (minimum/maximum) and digital (nominal/mask) alarm keys")
    if not has_analog and not has_digital:
        raise ValueError(
            "Alarm dict must include at least one type-specific key: minimum/maximum (analog) or nominal/mask (digital)"
        )
    return "analog" if has_analog else "digital"


class Backend(ABC):
    """
    Abstract base class for backend instances.

    Each backend provides read/write access to ACNET devices through a specific
    protocol. Backends may support different capabilities (read, write, stream,
    authentication).

    Lifecycle:
        - Use as context manager (recommended): `with Backend() as b: ...`
        - Or manually call close() when done

    Thread Safety:
        One-shot operations (read, get, get_many) are thread-safe. Each call
        borrows a connection from an internal pool for its duration.
    """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapability:
        """
        Backend capabilities (READ, WRITE, STREAM, etc.).

        Returns:
            BackendCapability flags indicating supported operations
        """

    @property
    def authenticated(self) -> bool:
        """
        True if backend is configured for authenticated operations.

        Override in auth-capable backends (DPM with Kerberos, gRPC with JWT).
        """
        return False

    @property
    def principal(self) -> Optional[str]:
        """
        Kerberos principal or JWT subject if authenticated, else None.

        Override in auth-capable backends.
        """
        return None

    @abstractmethod
    def read(self, drf: str, timeout: Optional[float] = None) -> Value:
        """
        Read a single device value.

        Args:
            drf: Device request string (e.g., "M:OUTTMP", "B:HS23T[0:10]")
            timeout: Total timeout for operation in seconds (None=default)

        Returns:
            The device value (float, numpy array, string, bytes, etc.)

        Raises:
            ValueError: If DRF syntax is invalid
            DeviceError: If the read fails (no usable data)
        """

    @abstractmethod
    def get(self, drf: str, timeout: Optional[float] = None) -> Reading:
        """
        Read a single device with full metadata.

        Args:
            drf: Device request string
            timeout: Total timeout for operation in seconds (None=default)

        Returns:
            Reading object with value, status, timestamp, and metadata.
            Check reading.is_error for error status.

        Raises:
            ValueError: If DRF syntax is invalid
        """

    @abstractmethod
    def get_many(self, drfs: list[str], timeout: Optional[float] = None) -> list[Reading]:
        """
        Read multiple devices in a single batch.

        Args:
            drfs: List of device request strings
            timeout: Total timeout for entire batch (not per-device)

        Returns:
            List of Reading objects in same order as input. Individual
            devices that fail return Reading with is_error=True.

        Raises:
            ValueError: If any DRF syntax is invalid (before network I/O)
        """

    def write(
        self,
        drf: str,
        value: Value,
        timeout: Optional[float] = None,
    ) -> WriteResult:
        """
        Write a single device value.

        Args:
            drf: Device to set
            value: Value to write
            timeout: Total timeout for operation

        Returns:
            WriteResult with status

        Raises:
            NotImplementedError: If backend doesn't support writes
            AuthenticationError: If authentication is required but not configured
        """
        raise NotImplementedError("This backend does not support writes")

    def write_many(
        self,
        settings: list[tuple[str, Value]],
        timeout: Optional[float] = None,
    ) -> list[WriteResult]:
        """
        Write multiple device values.

        Args:
            settings: List of (drf, value) tuples
            timeout: Total timeout for entire batch

        Returns:
            List of WriteResult objects in same order as input

        Raises:
            NotImplementedError: If backend doesn't support writes
            AuthenticationError: If authentication is required but not configured
        """
        raise NotImplementedError("This backend does not support writes")

    # ─────────────────────────────────────────────────────────────────────────
    # Streaming Methods - use dedicated connection separate from pool
    # ─────────────────────────────────────────────────────────────────────────

    def subscribe(
        self,
        drfs: list[str],
        callback: Optional[ReadingCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ) -> SubscriptionHandle:
        """
        Subscribe to devices for streaming data.

        Creates a subscription that immediately starts receiving data.
        The handle can be used as a context manager for automatic cleanup.

        Args:
            drfs: List of device request strings (with events, e.g. "M:OUTTMP@p,1000")
            callback: Optional function called for each reading, receives (reading, handle).
                     If provided, readings are pushed to the callback on the receiver thread.
                     If None, use handle.readings() to iterate over readings.
            on_error: Optional function called when a connection error occurs,
                     receives (exception, handle). If not provided and callback is set,
                     errors are logged. If neither callback nor on_error, errors are
                     raised during iteration via handle.readings().

        Returns:
            SubscriptionHandle for managing this subscription

        Raises:
            NotImplementedError: If backend doesn't support streaming

        Example (callback mode):
            def on_reading(reading, handle):
                print(f"{reading.name}: {reading.value}")
                if reading.value > 100:
                    handle.stop()

            handle = backend.subscribe(["M:OUTTMP@p,1000"], callback=on_reading)
            time.sleep(10)
            handle.stop()

        Example (iterator mode):
            with backend.subscribe(["M:OUTTMP@p,1000"]) as sub:
                for reading, handle in sub.readings(timeout=10):
                    print(f"{reading.name}: {reading.value}")
                    if reading.value > 10:
                        sub.stop()

        Example (with error handler):
            def on_error(exc, handle):
                print(f"Connection error: {exc}")

            handle = backend.subscribe(
                ["M:OUTTMP@p,1000"],
                callback=on_reading,
                on_error=on_error,
            )
        """
        raise NotImplementedError("This backend does not support streaming")

    def remove(self, handle: SubscriptionHandle) -> None:
        """
        Remove a subscription.

        Stops the associated DPM list(s) or gRPC stream(s).

        Args:
            handle: SubscriptionHandle returned from subscribe()

        Raises:
            NotImplementedError: If backend doesn't support streaming
        """
        raise NotImplementedError("This backend does not support streaming")

    def stop_streaming(self) -> None:
        """
        Stop all streaming subscriptions and close the streaming connection.

        Safe to call multiple times. Does not affect the connection pool
        used for one-shot reads.

        Raises:
            NotImplementedError: If backend doesn't support streaming
        """
        raise NotImplementedError("This backend does not support streaming")

    @abstractmethod
    def close(self) -> None:
        """
        Close the backend and release resources.

        Safe to call multiple times. After close(), the backend cannot be reused.
        """

    def __enter__(self) -> "Backend":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager - closes the backend."""
        self.close()
        return False


__all__ = ["Backend", "SubscriptionHandle"]
