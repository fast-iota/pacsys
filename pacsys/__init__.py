"""
pacsys - Pure Python library for ACNET control system at Fermilab.

"""

import atexit
import importlib
import logging
import math
import os
import sys
import threading
import types as _stdlib_types
import weakref
from typing import TYPE_CHECKING, Optional

from pacsys.auth import Auth, JWTAuth, KerberosAuth
from pacsys.device import ArrayDevice, Device, ScalarDevice, TextDevice
from pacsys.drf3 import DataRequest
from pacsys.errors import ACLError, AuthenticationError, DeviceError, ReadError
from pacsys.types import (
    BackendCapability,
    BasicControl,
    CombinedStream,
    DeviceMeta,
    DeviceSpec,
    DispatchMode,
    ErrorCallback,
    Reading,
    ReadingCallback,
    SubscriptionHandle,
    Value,
    ValueType,
    WriteResult,
    WriteSettings,
)

if TYPE_CHECKING:
    from pacsys.backends import Backend
    from pacsys.backends.acl import ACLBackend
    from pacsys.backends.dmq import DMQBackend
    from pacsys.backends.dpm_http import DPMHTTPBackend
    from pacsys.backends.grpc_backend import GRPCBackend
    from pacsys.devdb import DevDBClient
    from pacsys.ssh import SSHClient, SSHHop
    from pacsys.supervised import SupervisedServer

__version__ = "0.2.3"

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Environment Variables (read at import)
# ─────────────────────────────────────────────────────────────────────────────


def _get_env_int(name: str, default: int | None = None) -> int | None:
    """Get environment variable as int."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got {val!r}") from None


def _get_env_float(name: str, default: float | None = None) -> float | None:
    """Get environment variable as float."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a number, got {val!r}") from None


_env_dpm_host = os.environ.get("PACSYS_DPM_HOST")
_env_dpm_port = _get_env_int("PACSYS_DPM_PORT")
_env_pool_size = _get_env_int("PACSYS_POOL_SIZE")
_env_timeout = _get_env_float("PACSYS_TIMEOUT")
_env_devdb_host = os.environ.get("PACSYS_DEVDB_HOST")
_env_devdb_port = _get_env_int("PACSYS_DEVDB_PORT")


# ─────────────────────────────────────────────────────────────────────────────
# Global Backend Management
# ─────────────────────────────────────────────────────────────────────────────

_global_lock = threading.Lock()

_global_backend: Optional["Backend"] = None

_backend_initialized = False

# Valid backend type names
_VALID_BACKENDS = {"dpm", "grpc", "dmq", "acl"}

# User-configured settings (set via configure())
_config_backend: str | None = None
_config_auth: Auth | None = None
_config_role: str | None = None
_config_dpm_host: str | None = None
_config_dpm_port: int | None = None
_config_pool_size: int | None = None
_config_timeout: float | None = None
_config_devdb_host: str | None = None
_config_devdb_port: int | None = None

# Global lazy-initialized DevDB client (None until first use)
_global_devdb: Optional["DevDBClient"] = None
_devdb_initialized = False

# All backends created via factory functions, tracked for atexit cleanup.
# WeakSet so backends closed+dereferenced via `with` get garbage collected.
_live_backends: weakref.WeakSet = weakref.WeakSet()
_live_backends_lock = threading.Lock()


def _track(backend):
    """Register a backend for atexit cleanup and return it."""
    with _live_backends_lock:
        _live_backends.add(backend)
    return backend


def _atexit_close_backends() -> None:
    """Close all backends at interpreter exit."""
    with _live_backends_lock:
        backends = list(_live_backends)
    for backend in backends:
        try:
            backend.close()
        except Exception:
            logger.debug("Error closing backend during atexit", exc_info=True)


atexit.register(_atexit_close_backends)


class _Unset:
    """Sentinel: distinguish 'not passed' from 'passed as None'."""


_UNSET = _Unset()


def _validate_config_host(value: object, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be a non-empty string or None")


def _validate_config_port(value: object, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535):
        raise ValueError(f"{name} must be an integer from 1 to 65535 or None")


def _validate_config_positive(value: object, name: str, *, integer: bool = False) -> None:
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if value is not None and (isinstance(value, bool) or not valid_type or not math.isfinite(value) or value <= 0):
        kind = "a positive integer" if integer else "a positive finite number"
        raise ValueError(f"{name} must be {kind} or None")


def configure(
    *,
    dpm_host: str | _Unset | None = _UNSET,
    dpm_port: int | _Unset | None = _UNSET,
    pool_size: int | _Unset | None = _UNSET,
    default_timeout: float | _Unset | None = _UNSET,
    devdb_host: str | _Unset | None = _UNSET,
    devdb_port: int | _Unset | None = _UNSET,
    backend: str | _Unset | None = _UNSET,
    auth: Auth | str | _Unset | None = _UNSET,
    role: str | _Unset | None = _UNSET,
) -> None:
    """Configure pacsys global settings.

    Can be called at any time. If a backend is already initialized, it will
    be automatically shut down before applying the new settings. Pass None
    to clear a previously set value (falls back to environment variable or
    default).

    Args:
        dpm_host: DPM proxy hostname (default: from PACSYS_DPM_HOST or acsys-proxy.fnal.gov)
        dpm_port: DPM proxy port (default: from PACSYS_DPM_PORT or 6802)
        pool_size: Connection pool size (default: from PACSYS_POOL_SIZE or 4)
        default_timeout: Default operation timeout in seconds (default: from PACSYS_TIMEOUT or 5.0)
        devdb_host: DevDB gRPC hostname (default: from PACSYS_DEVDB_HOST or ad-services.fnal.gov/services.devdb)
        devdb_port: DevDB gRPC port (default: from PACSYS_DEVDB_PORT or 6802)
        backend: Backend type - one of "dpm", "grpc", "dmq", "acl" (default: "dpm")
        auth: Authentication object (KerberosAuth or JWTAuth) for writes,
              or "krb" as shortcut for KerberosAuth()
        role: Role for authenticated operations (e.g., "testing")

    Raises:
        ValueError: If a supplied value or backend/auth combination is invalid.
    """
    global _config_dpm_host, _config_dpm_port, _config_pool_size, _config_timeout
    global _config_devdb_host, _config_devdb_port
    global _config_backend, _config_auth, _config_role

    if not isinstance(backend, _Unset) and backend is not None and backend not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend {backend!r}, must be one of {sorted(_VALID_BACKENDS)}")
    if isinstance(auth, str):
        if auth != "krb":
            raise ValueError("auth string must be 'krb'")
        normalized_auth = KerberosAuth()
    else:
        normalized_auth = auth

    if not isinstance(dpm_host, _Unset):
        _validate_config_host(dpm_host, "dpm_host")
    if not isinstance(dpm_port, _Unset):
        _validate_config_port(dpm_port, "dpm_port")
    if not isinstance(pool_size, _Unset):
        _validate_config_positive(pool_size, "pool_size", integer=True)
    if not isinstance(default_timeout, _Unset):
        _validate_config_positive(default_timeout, "default_timeout")
    if not isinstance(devdb_host, _Unset):
        _validate_config_host(devdb_host, "devdb_host")
    if not isinstance(devdb_port, _Unset):
        _validate_config_port(devdb_port, "devdb_port")

    with _global_lock:
        configured_backend = backend if not isinstance(backend, _Unset) else _config_backend
        configured_auth = normalized_auth if not isinstance(normalized_auth, _Unset) else _config_auth
        effective_backend = configured_backend or "dpm"
        if effective_backend == "dpm" and configured_auth is not None and not isinstance(configured_auth, KerberosAuth):
            raise ValueError("DPM backend auth must be KerberosAuth or None")
        if effective_backend == "grpc" and configured_auth is not None and not isinstance(configured_auth, JWTAuth):
            raise ValueError("gRPC backend auth must be JWTAuth or None")
        if effective_backend == "dmq" and not isinstance(configured_auth, KerberosAuth):
            raise ValueError("DMQ backend requires KerberosAuth")

        if _backend_initialized or _devdb_initialized:
            logger.debug("configure() called with active backend — auto-replacing")
            _shutdown_locked()

        if not isinstance(backend, _Unset):
            _config_backend = backend
        if not isinstance(auth, _Unset):
            assert not isinstance(normalized_auth, _Unset)
            _config_auth = normalized_auth
        if not isinstance(role, _Unset):
            _config_role = role
        if not isinstance(dpm_host, _Unset):
            _config_dpm_host = dpm_host
        if not isinstance(dpm_port, _Unset):
            _config_dpm_port = dpm_port
        if not isinstance(pool_size, _Unset):
            _config_pool_size = pool_size
        if not isinstance(default_timeout, _Unset):
            _config_timeout = default_timeout
        if not isinstance(devdb_host, _Unset):
            _config_devdb_host = devdb_host
        if not isinstance(devdb_port, _Unset):
            _config_devdb_port = devdb_port


def _shutdown_locked() -> None:
    """Close and reset global backend/devdb state. Caller must hold _global_lock."""
    global _global_backend, _backend_initialized
    global _global_devdb, _devdb_initialized

    if _global_backend is not None:
        _global_backend.close()
        _global_backend = None

    if _global_devdb is not None:
        _global_devdb.close()
        _global_devdb = None

    _backend_initialized = False
    _devdb_initialized = False


def shutdown() -> None:
    """Close and release the global lazy-initialized backend and DevDB client.

    The global backend is automatically closed on interpreter exit via atexit,
    so explicit shutdown() is only needed to reset state mid-process (e.g.,
    between tests or before re-configuring).

    After shutdown(), the next read/get call will re-initialize the backend
    using existing configuration from configure(). Configuration is preserved
    across shutdown/re-init cycles -- use configure() to change settings.

    Safe to call multiple times or when no backend is initialized.
    """
    with _global_lock:
        _shutdown_locked()


def _get_global_backend() -> "Backend":
    """Get or create the global backend (lazy initialization).

    Dispatches to the backend type set via configure(backend=...).
    Defaults to DPM HTTP if no backend type is configured.

    Returns:
        Backend instance

    Thread Safety:
        Thread-safe - uses lock for initialization.
    """
    global _global_backend, _backend_initialized

    if _global_backend is not None:
        return _global_backend

    with _global_lock:
        if _global_backend is not None:
            return _global_backend

        timeout = (
            _config_timeout if _config_timeout is not None else (_env_timeout if _env_timeout is not None else 5.0)
        )
        backend_type = _config_backend or "dpm"

        if backend_type == "dpm":
            _global_backend = _create_global_dpm(timeout)
        elif backend_type == "grpc":
            _global_backend = _create_global_grpc(timeout)
        elif backend_type == "dmq":
            _global_backend = _create_global_dmq(timeout)
        elif backend_type == "acl":
            _global_backend = _create_global_acl(timeout)
        else:
            raise ValueError(f"Unknown backend type {backend_type!r}")

        _backend_initialized = True
        return _global_backend


def _create_global_dpm(timeout: float) -> "DPMHTTPBackend":
    from pacsys.backends.dpm_http import DPMHTTPBackend

    host = (
        _config_dpm_host
        if _config_dpm_host is not None
        else (_env_dpm_host if _env_dpm_host is not None else "acsys-proxy.fnal.gov")
    )
    port = _config_dpm_port if _config_dpm_port is not None else (_env_dpm_port if _env_dpm_port is not None else 6802)
    pool_size = (
        _config_pool_size if _config_pool_size is not None else (_env_pool_size if _env_pool_size is not None else 4)
    )
    kwargs: dict = dict(host=host, port=port, pool_size=pool_size, timeout=timeout)
    kwargs["auth"] = _config_auth if _config_auth is not None else KerberosAuth(_lazy=True)
    if _config_role is not None:
        kwargs["role"] = _config_role
    return _track(DPMHTTPBackend(**kwargs))


def _create_global_grpc(timeout: float) -> "GRPCBackend":
    from pacsys.backends.grpc_backend import GRPCBackend

    kwargs: dict = dict(timeout=timeout)
    if _config_auth is not None:
        kwargs["auth"] = _config_auth
    return _track(GRPCBackend(**kwargs))


def _create_global_dmq(timeout: float) -> "DMQBackend":
    from pacsys.backends.dmq import DMQBackend

    kwargs: dict = dict(timeout=timeout)
    if _config_auth is not None:
        kwargs["auth"] = _config_auth
    return _track(DMQBackend(**kwargs))


def _create_global_acl(timeout: float) -> "ACLBackend":
    from pacsys.backends.acl import ACLBackend

    return _track(ACLBackend(timeout=timeout))


def _get_global_devdb() -> Optional["DevDBClient"]:
    """Get or create the global DevDB client if configured.

    Returns None if DevDB is not configured (no host in env or configure()).
    The global DevDB is opt-in -- only created if PACSYS_DEVDB_HOST is set
    or configure(devdb_host=...) was called.
    """
    global _global_devdb, _devdb_initialized

    if _devdb_initialized:
        return _global_devdb

    with _global_lock:
        if _devdb_initialized:
            return _global_devdb

        from pacsys.devdb import DEVDB_AVAILABLE, DevDBClient

        if not DEVDB_AVAILABLE:
            _devdb_initialized = True
            return None

        host = _config_devdb_host or _env_devdb_host or "ad-services.fnal.gov/services.devdb"
        port = _config_devdb_port or _env_devdb_port or 6802
        _global_devdb = DevDBClient(host=host, port=port)
        _devdb_initialized = True
        # DevDB clients share backend atexit cleanup.
        with _live_backends_lock:
            _live_backends.add(_global_devdb)
        return _global_devdb


# ─────────────────────────────────────────────────────────────────────────────
# DeviceSpec Resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_drf(device: DeviceSpec) -> str:
    """Convert DeviceSpec to DRF string.

    Args:
        device: DRF string or Device object

    Returns:
        DRF string

    Raises:
        TypeError: If device is neither str nor Device
    """
    if isinstance(device, str):
        return device
    if isinstance(device, Device):
        return device.drf
    raise TypeError(f"Expected str or Device, got {type(device).__name__}")


def _resolve_setting(device: DeviceSpec, value: Value) -> tuple[str, Value]:
    """Resolve a write target; BasicControl values are routed to CONTROL, never SETTING."""
    from pacsys.drf_utils import prepare_for_control

    drf = _resolve_drf(device)
    if isinstance(value, BasicControl):
        drf = prepare_for_control(drf)
    return drf, value


# ─────────────────────────────────────────────────────────────────────────────
# Simple API Functions
# ─────────────────────────────────────────────────────────────────────────────


def read(device: DeviceSpec, timeout: float | None = None) -> Value:
    """Read a single device value using the global DPM backend.

    Args:
        device: DRF string or Device object
        timeout: Total timeout for entire operation in seconds (default: 5.0)

    Returns:
        The device value (float, numpy array, string, etc.)

    Raises:
        ValueError: If DRF syntax is invalid
        DeviceError: If the read returns no usable data

    Note:
        Even if DRF specifies periodic event (@p,1000), only FIRST reading
        is returned. Use Session for continuous data.

    Thread Safety:
        Safe to call from multiple threads. Each call borrows a connection
        from the shared pool for the duration of the operation.
    """
    drf = _resolve_drf(device)
    backend = _get_global_backend()
    return backend.read(drf, timeout=timeout)


def get(device: DeviceSpec, timeout: float | None = None) -> Reading:
    """Read a single device with full metadata using the global DPM backend.

    Args:
        device: DRF string or Device object
        timeout: Total timeout for operation in seconds (default: 5.0)

    Returns:
        Reading object with value, status, timestamp, and metadata.
        Check ``reading.ok`` before using the value; ``is_error`` and
        ``is_warning`` classify nonzero statuses.

    Raises:
        ValueError: If DRF syntax is invalid

    Thread Safety:
        Safe to call from multiple threads.
    """
    drf = _resolve_drf(device)
    backend = _get_global_backend()
    return backend.get(drf, timeout=timeout)


def get_many(
    devices: list[DeviceSpec],
    timeout: float | None = None,
) -> list[Reading]:
    """Read multiple devices in a single batch using the global DPM backend.

    Args:
        devices: List of DRF strings or Device objects (can mix)
        timeout: Total timeout for entire batch in seconds (not per-device)

    Returns:
        List of Reading objects in same order as input.

    Raises:
        ReadError: On transport failure (timeout, connection drop).
            Partial results are available via ``exc.readings``.
        ValueError: If any DRF syntax is invalid (before network I/O)

    Thread Safety:
        Safe to call from multiple threads.
    """
    drfs = [_resolve_drf(d) for d in devices]
    backend = _get_global_backend()
    return backend.get_many(drfs, timeout=timeout)


def read_many(
    devices: list[DeviceSpec],
    timeout: float | None = None,
) -> list[Value]:
    """Read multiple device values in a single batch using the global backend.

    Args:
        devices: List of DRF strings or Device objects (can mix)
        timeout: Total timeout for entire batch in seconds (not per-device)

    Returns:
        List of values in same order as input (float, numpy array, string, etc.)

    Raises:
        ReadError: If any reading is unusable (ACNET error or missing value),
            or on transport failure. Partial results via ``exc.readings``.
        ValueError: If any DRF syntax is invalid (before network I/O)

    Thread Safety:
        Safe to call from multiple threads.
    """
    drfs = [_resolve_drf(d) for d in devices]
    backend = _get_global_backend()
    return backend.read_many(drfs, timeout=timeout)


def write(device: DeviceSpec, value: Value, timeout: float | None = None) -> WriteResult:
    """Write a single device value using the global backend.

    Args:
        device: DRF string or Device object
        value: Value to write. BasicControl values target the CONTROL property
            (a bare device name is retargeted; SETTING is never used).
        timeout: Total timeout in seconds

    Returns:
        WriteResult with status

    Raises:
        AuthenticationError: If no auth configured (use configure(auth=...))
        DeviceError: If the write fails

    Thread Safety:
        Safe to call from multiple threads.
    """
    drf, value = _resolve_setting(device, value)
    backend = _get_global_backend()
    return backend.write(drf, value, timeout=timeout)


def write_many(
    settings: WriteSettings,
    timeout: float | None = None,
) -> list[WriteResult]:
    """Write multiple device values in a single batch using the global backend.

    Args:
        settings: List of (device, value) tuples, or a dict mapping device -> value
        timeout: Total timeout for entire batch in seconds

    Returns:
        List of WriteResult objects in same order as input.

    Raises:
        AuthenticationError: If no auth configured (use configure(auth=...))

    Thread Safety:
        Safe to call from multiple threads.
    """
    items = settings.items() if isinstance(settings, dict) else settings
    resolved = [_resolve_setting(d, v) for d, v in items]
    backend = _get_global_backend()
    return backend.write_many(resolved, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming API Functions
# ─────────────────────────────────────────────────────────────────────────────


def subscribe(
    drfs: list[DeviceSpec],
    callback: ReadingCallback | None = None,
    on_error: ErrorCallback | None = None,
) -> SubscriptionHandle:
    """Subscribe to devices for streaming using the global DPM backend.

    Creates subscriptions that immediately start receiving data.
    The handle can be used as a context manager for automatic cleanup.

    Args:
        drfs: List of device request strings or Device objects (with events, e.g. "M:OUTTMP@p,1000")
        callback: Optional function called for each reading, receives (reading, handle).
                 If provided, readings are pushed to the callback on the receiver thread.
                 If None, use handle.readings() to iterate over readings.
        on_error: Optional function called when a connection error occurs,
                 receives (exception, handle). If not provided, errors are raised
                 during iteration or logged in callback mode.

    Returns:
        SubscriptionHandle for managing this subscription

    Example (callback mode):
        def on_reading(reading, handle):
            print(f"{reading.name}: {reading.value}")
            if reading.value > 100:
                handle.stop()

        handle = pacsys.subscribe(["M:OUTTMP@p,1000"], callback=on_reading)
        time.sleep(10)
        handle.stop()
        pacsys.shutdown()

    Example (iterator mode):
        with pacsys.subscribe(["M:OUTTMP@p,1000"]) as sub:
            for reading, handle in sub.readings(timeout=10):
                print(f"{reading.name}: {reading.value}")
                if reading.value > 10:
                    sub.stop()
        pacsys.shutdown()

    Example (with error handler):
        def on_error(exc, handle):
            print(f"Connection error: {exc}")

        handle = pacsys.subscribe(
            ["M:OUTTMP@p,1000"],
            callback=on_reading,
            on_error=on_error,
        )
    """
    resolved = [_resolve_drf(d) for d in drfs]
    backend = _get_global_backend()
    return backend.subscribe(resolved, callback=callback, on_error=on_error)


# ─────────────────────────────────────────────────────────────────────────────
# Backend Factory Functions
# ─────────────────────────────────────────────────────────────────────────────


def dpm(
    host: str | None = None,
    port: int | None = None,
    pool_size: int | None = None,
    timeout: float | None = None,
    auth: Auth | None = None,
    role: str | None = None,
    dispatch_mode: DispatchMode = DispatchMode.WORKER,
) -> "DPMHTTPBackend":
    """Create a DPM backend instance with its own connection pool.

    Delegates to dpm_http(). Each subscribe() call creates its own TCP
    connection, allowing independent subscriptions.

    Args:
        host: DPM proxy hostname (default: acsys-proxy.fnal.gov)
        port: DPM proxy port (default: 6802)
        pool_size: Connection pool size for one-shot reads (default: 4)
        timeout: Default operation timeout in seconds (default: 5.0)
        auth: Authentication object (KerberosAuth for writes)
        role: Optional role for authenticated operations (e.g., "testing")
        dispatch_mode: How streaming callbacks are dispatched (default: WORKER)

    Returns:
        DPMHTTPBackend instance (use as context manager or call close() when done)

    Example (read-only):
        with pacsys.dpm() as backend:
            temp = backend.read("M:OUTTMP")

    Example (authenticated writes):
        auth = KerberosAuth()
        with pacsys.dpm(auth=auth, role="testing") as backend:
            print(f"Authenticated as: {backend.principal}")
            result = backend.write("M:OUTTMP", 72.5)
    """
    return dpm_http(
        host=host, port=port, pool_size=pool_size, timeout=timeout, auth=auth, role=role, dispatch_mode=dispatch_mode
    )


def dpm_http(
    host: str | None = None,
    port: int | None = None,
    pool_size: int | None = None,
    timeout: float | None = None,
    auth: Auth | None = None,
    role: str | None = None,
    dispatch_mode: DispatchMode = DispatchMode.WORKER,
) -> "DPMHTTPBackend":
    """Create a DPM HTTP backend with independent streaming subscriptions.

    This backend uses the TCP/PC protocol to communicate with DPM. Each
    subscribe() call creates its own TCP connection, allowing truly
    independent subscriptions that can be started/stopped individually.

    Args:
        host: DPM proxy hostname (default: acsys-proxy.fnal.gov)
        port: DPM proxy port (default: 6802)
        pool_size: Connection pool size for reads (default: 4)
        timeout: Default operation timeout in seconds (default: 5.0)
        auth: Authentication object (KerberosAuth for writes)
        role: Role for authenticated operations (e.g., "testing")
        dispatch_mode: How streaming callbacks are dispatched (default: WORKER)

    Returns:
        DPMHTTPBackend instance

    Example (multiple independent subscriptions):
        with pacsys.dpm_http() as backend:
            sub1 = backend.subscribe(["M:OUTTMP@p,1000"])
            sub2 = backend.subscribe(["G:AMANDA@p,500"])

            # Stopping sub1 doesn't affect sub2
            sub1.stop()

            for reading, _ in sub2.readings(timeout=10):
                print(f"{reading.name}: {reading.value}")
    """
    from pacsys.backends.dpm_http import DPMHTTPBackend

    effective_host = (
        host if host is not None else (_env_dpm_host if _env_dpm_host is not None else "acsys-proxy.fnal.gov")
    )
    effective_port = port if port is not None else (_env_dpm_port if _env_dpm_port is not None else 6802)
    effective_pool_size = pool_size if pool_size is not None else (_env_pool_size if _env_pool_size is not None else 4)
    effective_timeout = timeout if timeout is not None else (_env_timeout if _env_timeout is not None else 5.0)

    return _track(
        DPMHTTPBackend(
            host=effective_host,
            port=effective_port,
            pool_size=effective_pool_size,
            timeout=effective_timeout,
            auth=auth,
            role=role,
            dispatch_mode=dispatch_mode,
        )
    )


def grpc(
    host: str | None = None,
    port: int | None = None,
    auth: Auth | None = None,
    timeout: float | None = None,
    dispatch_mode: DispatchMode = DispatchMode.WORKER,
) -> "GRPCBackend":
    """Create a gRPC backend instance.

    Uses the DAQ gRPC service for reads and writes. Writes require
    JWT authentication.

    Args:
        host: gRPC server hostname (env: PACSYS_GRPC_HOST, default: dce08.fnal.gov)
        port: gRPC server port (env: PACSYS_GRPC_PORT, default: 50051)
        auth: Authentication object (JWTAuth for writes). If None, tries PACSYS_JWT_TOKEN env.
        timeout: Default operation timeout in seconds (default: 5.0)
        dispatch_mode: How streaming callbacks are dispatched (default: WORKER)

    Returns:
        GRPCBackend instance (use as context manager or call close() when done)

    Raises:
        ImportError: If grpc package is not installed

    Example (read-only):
        with pacsys.grpc() as backend:
            temp = backend.read("M:OUTTMP")

    Example (with JWT):
        auth = JWTAuth(token="eyJ...")
        with pacsys.grpc(auth=auth) as backend:
            print(f"Authenticated as: {backend.principal}")
            result = backend.write("M:OUTTMP", 72.5)

    Example (token from environment):
        # export PACSYS_JWT_TOKEN="eyJ..."
        with pacsys.grpc() as backend:
            if backend.authenticated:
                print(f"Authenticated as: {backend.principal}")
    """
    from pacsys.backends.grpc_backend import GRPCBackend

    return _track(GRPCBackend(host=host, port=port, auth=auth, timeout=timeout, dispatch_mode=dispatch_mode))


def acl(
    base_url: str | None = None,
    timeout: float | None = None,
) -> "ACLBackend":
    """Create an ACL backend instance (read-only, no streaming, no auth).

    Args:
        base_url: ACL CGI base URL (default: https://www-bd.fnal.gov/cgi-bin/acl.pl)
        timeout: Default operation timeout in seconds

    Returns:
        ACLBackend instance

    Example:
        with pacsys.acl() as backend:
            temp = backend.read("M:OUTTMP")
            reading = backend.get("M:OUTTMP")
            readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
    """
    from pacsys.backends.acl import ACLBackend

    return _track(ACLBackend(base_url=base_url, timeout=timeout))


def dmq(
    host: str | None = None,
    port: int | None = None,
    timeout: float | None = None,
    auth: Auth | None = None,
    write_session_ttl: float | None = None,
    dispatch_mode: DispatchMode = DispatchMode.WORKER,
) -> "DMQBackend":
    """Create a DMQ backend instance (RabbitMQ/AMQP).

    Uses RabbitMQ to communicate with ACNET via the DMQ server.
    Requires Kerberos authentication for ALL operations (including reads).

    Args:
        host: RabbitMQ broker hostname (default: from PACSYS_DMQ_HOST or appsrv2.fnal.gov)
        port: RabbitMQ broker port (default: from PACSYS_DMQ_PORT or 5672)
        timeout: Default operation timeout in seconds (default: 10.0)
        auth: KerberosAuth required for all DMQ operations
        write_session_ttl: Idle timeout for write sessions in seconds (default: 600)
        dispatch_mode: How streaming callbacks are dispatched (default: WORKER)

    Returns:
        DMQBackend instance (use as context manager or call close() when done)

    Raises:
        AuthenticationError: If auth is not provided or not KerberosAuth
        ImportError: If pika or gssapi packages are not installed

    Example:
        auth = KerberosAuth()
        with pacsys.dmq(auth=auth) as backend:
            temp = backend.read("M:OUTTMP")
            result = backend.write("Z:ACLTST", 45.0)
    """
    from pacsys.backends.dmq import DMQBackend

    kwargs: dict = {}
    if host is not None:
        kwargs["host"] = host
    if port is not None:
        kwargs["port"] = port
    if timeout is not None:
        kwargs["timeout"] = timeout
    if auth is not None:
        kwargs["auth"] = auth
    if write_session_ttl is not None:
        kwargs["write_session_ttl"] = write_session_ttl
    kwargs["dispatch_mode"] = dispatch_mode

    return _track(DMQBackend(**kwargs))


def devdb(
    host: str | None = None,
    port: int | None = None,
    timeout: float | None = None,
    cache_ttl: float = 3600.0,
) -> "DevDBClient":
    """Create a DevDB client for device metadata queries.

    DevDB provides device information like scaling parameters, control commands,
    and status bit definitions from the master PostgreSQL database.

    Args:
        host: DevDB gRPC hostname (default: from PACSYS_DEVDB_HOST or ad-services.fnal.gov/services.devdb)
        port: DevDB gRPC port (default: from PACSYS_DEVDB_PORT or 6802)
        timeout: RPC timeout in seconds (default: 5.0)
        cache_ttl: Cache TTL in seconds (default: 3600.0)

    Returns:
        DevDBClient instance (use as context manager or call close() when done)

    Raises:
        ImportError: If grpc package is not available

    Example:
        with pacsys.devdb() as db:
            info = db.get_device_info(["Z:ACLTST"])
            print(info["Z:ACLTST"].description)
    """
    from pacsys.devdb import DevDBClient

    return _track(DevDBClient(host=host, port=port, timeout=timeout, cache_ttl=cache_ttl))


def ssh(
    hops: "str | SSHHop | list[str | SSHHop]",
    auth: Auth | None = None,
    connect_timeout: float = 10.0,
) -> "SSHClient":
    """Create an SSH client for remote command execution, tunneling, and SFTP.

    Supports multi-hop connections through jump hosts using Kerberos (GSSAPI),
    key-based, or password authentication.

    Args:
        hops: Target host(s). Accepts a hostname string, SSHHop, or list of either.
              Multiple hops create a chain (jump hosts).
        auth: Optional KerberosAuth for GSSAPI hops. If None and any hop uses
              gssapi auth, credentials are validated at construction time.
        connect_timeout: TCP connection timeout in seconds (default 10.0).

    Returns:
        SSHClient instance (use as context manager or call close() when done)

    Example (single hop):
        with pacsys.ssh("target.fnal.gov") as client:
            result = client.exec("hostname")
            print(result.stdout)

    Example (multi-hop with Kerberos):
        auth = KerberosAuth()
        with pacsys.ssh(["jump.fnal.gov", "target.fnal.gov"], auth=auth) as client:
            result = client.exec("ls /data")

    Example (port forwarding):
        with pacsys.ssh("jump.fnal.gov") as client:
            with client.forward(23456, "grpc-host.fnal.gov", 50051) as tunnel:
                # Use gRPC backend via tunnel
                with pacsys.grpc(port=tunnel.local_port) as backend:
                    value = backend.read("M:OUTTMP")
    """
    from pacsys.ssh import SSHClient as _SSHClient

    return _SSHClient(hops=hops, auth=auth, connect_timeout=connect_timeout)


def supervised(
    backend: "Backend",
    port: int = 50051,
    host: str = "[::]",
    policies: list | None = None,
) -> "SupervisedServer":
    """Create a supervised gRPC proxy server with logging and policy enforcement.

    Wraps any Backend and exposes it as a gRPC DAQ service, forwarding
    requests while enforcing access policies and logging all traffic.

    Args:
        backend: Backend instance to proxy requests to
        port: Port to listen on (default: 50051). Use 0 for OS-assigned.
        host: Host to bind (default: "[::]" for all interfaces)
        policies: Optional list of Policy instances for access control

    Returns:
        SupervisedServer instance (use as context manager or call start()/stop())

    Example:
        from pacsys.supervised import ReadOnlyPolicy

        with pacsys.dpm() as backend:
            with pacsys.supervised(backend, port=50051, policies=[ReadOnlyPolicy()]) as srv:
                srv.wait()  # Block until interrupted
    """
    from pacsys.supervised import SupervisedServer

    return SupervisedServer(backend=backend, port=port, host=host, policies=policies)


# ─────────────────────────────────────────────────────────────────────────────
# Submodule Shadowing Protection
# ─────────────────────────────────────────────────────────────────────────────
# Factory functions ssh(), devdb(), supervised() share names with submodule
# files. When Python imports a submodule, it does setattr(parent, child_name,
# module), overwriting the factory function. A custom module class with
# __setattr__ blocks this for protected names.


class _PacsysModule(_stdlib_types.ModuleType):
    _PROTECTED = frozenset({"ssh", "devdb", "supervised"})

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._PROTECTED and isinstance(value, _stdlib_types.ModuleType):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _PacsysModule


# ─────────────────────────────────────────────────────────────────────────────
# Lazy Imports
# ─────────────────────────────────────────────────────────────────────────────

_LAZY_IMPORTS: dict[str, str] = {
    # alarm_block
    "AlarmBlock": "pacsys.alarm_block",
    "AnalogAlarm": "pacsys.alarm_block",
    "DigitalAlarm": "pacsys.alarm_block",
    "AlarmFlags": "pacsys.alarm_block",
    "FTD": "pacsys.alarm_block",
    "LimitType": "pacsys.alarm_block",
    "DataType": "pacsys.alarm_block",
    "DataLength": "pacsys.alarm_block",
    # scaling
    "Scaler": "pacsys.scaling",
    "ScalingError": "pacsys.scaling",
    # digital_status
    "StatusBit": "pacsys.digital_status",
    "DigitalStatus": "pacsys.digital_status",
    # verify
    "Verify": "pacsys.verify",
    # ssh
    "SSHClient": "pacsys.ssh",
    "SSHHop": "pacsys.ssh",
    "CommandResult": "pacsys.ssh",
    "Tunnel": "pacsys.ssh",
    "SFTPSession": "pacsys.ssh",
    "RemoteProcess": "pacsys.ssh",
    "SSHError": "pacsys.ssh",
    "SSHConnectionError": "pacsys.ssh",
    "SSHCommandError": "pacsys.ssh",
    "SSHTimeoutError": "pacsys.ssh",
    # acl_session
    "ACLSession": "pacsys.acl_session",
    # devdb
    "DeviceInfoResult": "pacsys.devdb",
    "PropertyInfo": "pacsys.devdb",
    "StatusBitDef": "pacsys.devdb",
    "ExtStatusBitDef": "pacsys.devdb",
    "ControlCommandDef": "pacsys.devdb",
    # mcp
    "create_server": "pacsys.mcp",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    if name == "acnet":
        acnet = importlib.import_module("pacsys.acnet")
        globals()["acnet"] = acnet
        return acnet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Version
    "__version__",
    # DRF3 Parser
    "DataRequest",
    # Auth
    "Auth",
    "KerberosAuth",
    "JWTAuth",
    # Types
    "Value",
    "DeviceSpec",
    "ValueType",
    "BackendCapability",
    "DispatchMode",
    "DeviceMeta",
    "Reading",
    "WriteResult",
    "WriteSettings",
    "SubscriptionHandle",
    "CombinedStream",
    "ReadingCallback",
    "ErrorCallback",
    "BasicControl",
    # Errors
    "DeviceError",
    "AuthenticationError",
    "ACLError",
    "ReadError",
    # Device classes
    "Device",
    "ScalarDevice",
    "ArrayDevice",
    "TextDevice",
    # Alarm blocks
    "AlarmBlock",
    "AnalogAlarm",
    "DigitalAlarm",
    "AlarmFlags",
    "FTD",
    "LimitType",
    "DataType",
    "DataLength",
    # Digital status
    "StatusBit",
    "DigitalStatus",
    # Verify
    "Verify",
    # Scaling
    "Scaler",
    "ScalingError",
    # SSH
    "SSHClient",
    "SSHHop",
    "CommandResult",
    "Tunnel",
    "SFTPSession",
    "RemoteProcess",
    "SSHError",
    "SSHConnectionError",
    "SSHCommandError",
    "SSHTimeoutError",
    # ACL Session
    "ACLSession",
    # DevDB result types
    "DeviceInfoResult",
    "PropertyInfo",
    "StatusBitDef",
    "ExtStatusBitDef",
    "ControlCommandDef",
    # MCP server
    "create_server",
    # Simple API functions
    "read",
    "read_many",
    "get",
    "get_many",
    "write",
    "write_many",
    # Streaming API functions
    "subscribe",
    # Configuration
    "configure",
    "shutdown",
    # Backend factories
    "dpm",
    "dpm_http",
    "grpc",
    "dmq",
    "acl",
    "ssh",
    "devdb",
    "supervised",
    # Submodule
    "acnet",
    # Internal (for Device)
    "_get_global_backend",
    "_get_global_devdb",
]
