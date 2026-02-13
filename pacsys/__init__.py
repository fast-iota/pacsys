"""
pacsys - Pure Python library for ACNET control system at Fermilab.

"""

import atexit
import importlib
import logging
import os
import threading
import weakref
from typing import Optional, Union, TYPE_CHECKING

from pacsys.auth import Auth, KerberosAuth, JWTAuth
from pacsys.drf3 import DataRequest
from pacsys.types import (
    Value,
    DeviceSpec,
    ValueType,
    BackendCapability,
    DispatchMode,
    DeviceMeta,
    Reading,
    WriteResult,
    SubscriptionHandle,
    CombinedStream,
    ReadingCallback,
    ErrorCallback,
    BasicControl,
)
from pacsys.errors import DeviceError, AuthenticationError, ACLError, ReadError
from pacsys.device import Device, ScalarDevice, ArrayDevice, TextDevice

if TYPE_CHECKING:
    from pacsys.backends import Backend
    from pacsys.backends.dpm_http import DPMHTTPBackend
    from pacsys.backends.grpc_backend import GRPCBackend
    from pacsys.backends.acl import ACLBackend
    from pacsys.backends.dmq import DMQBackend
    from pacsys.devdb import DevDBClient
    from pacsys.ssh import SSHClient, SSHHop
    from pacsys.supervised import SupervisedServer

__version__ = "0.2.0"

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Environment Variables (read at import)
# ─────────────────────────────────────────────────────────────────────────────


def _get_env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Get environment variable as int."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got {val!r}")


def _get_env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """Get environment variable as float."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a number, got {val!r}")


# Read environment variables at import time
_env_dpm_host = os.environ.get("PACSYS_DPM_HOST")
_env_dpm_port = _get_env_int("PACSYS_DPM_PORT")
_env_pool_size = _get_env_int("PACSYS_POOL_SIZE")
_env_timeout = _get_env_float("PACSYS_TIMEOUT")
_env_devdb_host = os.environ.get("PACSYS_DEVDB_HOST")
_env_devdb_port = _get_env_int("PACSYS_DEVDB_PORT")


# ─────────────────────────────────────────────────────────────────────────────
# Global Backend Management
# ─────────────────────────────────────────────────────────────────────────────

# Thread-safe lock for global backend initialization
_global_lock = threading.Lock()

# Global lazy-initialized backend (None until first use)
_global_backend: Optional["Backend"] = None

# Flag to track if backend has been initialized
_backend_initialized = False

# Valid backend type names
_VALID_BACKENDS = {"dpm", "grpc", "dmq", "acl"}

# User-configured settings (set via configure())
_config_backend: Optional[str] = None
_config_auth: Optional[Auth] = None
_config_role: Optional[str] = None
_config_dpm_host: Optional[str] = None
_config_dpm_port: Optional[int] = None
_config_pool_size: Optional[int] = None
_config_timeout: Optional[float] = None
_config_devdb_host: Optional[str] = None
_config_devdb_port: Optional[int] = None

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


_UNSET = object()  # sentinel: distinguish "not passed" from "passed as None"


def configure(
    *,
    dpm_host: Optional[str] = _UNSET,
    dpm_port: Optional[int] = _UNSET,
    pool_size: Optional[int] = _UNSET,
    default_timeout: Optional[float] = _UNSET,
    devdb_host: Optional[str] = _UNSET,
    devdb_port: Optional[int] = _UNSET,
    backend: Optional[str] = _UNSET,
    auth: Optional[Auth] = _UNSET,
    role: Optional[str] = _UNSET,
) -> None:
    """Configure pacsys global settings.

    Must be called BEFORE any read/get operations. Pass None to clear
    a previously set value (falls back to environment variable or default).

    Args:
        dpm_host: DPM proxy hostname (default: from PACSYS_DPM_HOST or acsys-proxy.fnal.gov)
        dpm_port: DPM proxy port (default: from PACSYS_DPM_PORT or 6802)
        pool_size: Connection pool size (default: from PACSYS_POOL_SIZE or 4)
        default_timeout: Default operation timeout in seconds (default: from PACSYS_TIMEOUT or 5.0)
        devdb_host: DevDB gRPC hostname (default: from PACSYS_DEVDB_HOST or localhost)
        devdb_port: DevDB gRPC port (default: from PACSYS_DEVDB_PORT or 6802)
        backend: Backend type - one of "dpm", "grpc", "dmq", "acl" (default: "dpm")
        auth: Authentication object (KerberosAuth or JWTAuth) for writes
        role: Role for authenticated operations (e.g., "testing")

    Raises:
        RuntimeError: If called after any backend is initialized
        ValueError: If backend is not a valid backend type
    """
    global _config_dpm_host, _config_dpm_port, _config_pool_size, _config_timeout
    global _config_devdb_host, _config_devdb_port
    global _config_backend, _config_auth, _config_role

    with _global_lock:
        if _backend_initialized or _devdb_initialized:
            raise RuntimeError(
                "configure() must be called before any read/get operations. "
                "Call shutdown() first to close the backend, then configure() to change settings."
            )

        if backend is not _UNSET:
            if backend is not None and backend not in _VALID_BACKENDS:
                raise ValueError(f"Invalid backend {backend!r}, must be one of {sorted(_VALID_BACKENDS)}")
            _config_backend = backend
        if auth is not _UNSET:
            _config_auth = auth
        if role is not _UNSET:
            _config_role = role
        if dpm_host is not _UNSET:
            _config_dpm_host = dpm_host
        if dpm_port is not _UNSET:
            _config_dpm_port = dpm_port
        if pool_size is not _UNSET:
            _config_pool_size = pool_size
        if default_timeout is not _UNSET:
            _config_timeout = default_timeout
        if devdb_host is not _UNSET:
            _config_devdb_host = devdb_host
        if devdb_port is not _UNSET:
            _config_devdb_port = devdb_port


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
    global _global_backend, _backend_initialized
    global _global_devdb, _devdb_initialized

    with _global_lock:
        if _global_backend is not None:
            _global_backend.close()
            _global_backend = None

        if _global_devdb is not None:
            _global_devdb.close()
            _global_devdb = None

        _backend_initialized = False
        _devdb_initialized = False


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

    # Fast path: already initialized
    if _global_backend is not None:
        return _global_backend

    with _global_lock:
        # Double-check under lock
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
    if _config_auth is not None:
        kwargs["auth"] = _config_auth
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

        host = _config_devdb_host or _env_devdb_host
        if host is None:
            _devdb_initialized = True
            return None

        from pacsys.devdb import DevDBClient, DEVDB_AVAILABLE

        if not DEVDB_AVAILABLE:
            from pacsys.devdb import _import_error

            logger.warning("DevDB configured (host=%s) but gRPC not available: %s", host, _import_error)
            _devdb_initialized = True
            return None

        port = _config_devdb_port or _env_devdb_port or 6802
        _global_devdb = DevDBClient(host=host, port=port)
        _devdb_initialized = True
        # Track for atexit cleanup
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


# ─────────────────────────────────────────────────────────────────────────────
# Simple API Functions
# ─────────────────────────────────────────────────────────────────────────────


def read(device: DeviceSpec, timeout: Optional[float] = None) -> Value:
    """Read a single device value using the global DPM backend.

    Args:
        device: DRF string or Device object
        timeout: Total timeout for entire operation in seconds (default: 5.0)

    Returns:
        The device value (float, numpy array, string, etc.)

    Raises:
        ValueError: If DRF syntax is invalid
        DeviceError: If the read fails (status_code < 0)

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


def get(device: DeviceSpec, timeout: Optional[float] = None) -> Reading:
    """Read a single device with full metadata using the global DPM backend.

    Args:
        device: DRF string or Device object
        timeout: Total timeout for operation in seconds (default: 5.0)

    Returns:
        Reading object with value, status, timestamp, and metadata.
        Check reading.is_error for error status.

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
    timeout: Optional[float] = None,
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


# ─────────────────────────────────────────────────────────────────────────────
# Streaming API Functions
# ─────────────────────────────────────────────────────────────────────────────


def subscribe(
    drfs: list[DeviceSpec],
    callback: Optional[ReadingCallback] = None,
    on_error: Optional[ErrorCallback] = None,
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
    host: Optional[str] = None,
    port: Optional[int] = None,
    pool_size: Optional[int] = None,
    timeout: Optional[float] = None,
    auth: Optional[Auth] = None,
    role: Optional[str] = None,
    dispatch_mode: DispatchMode = DispatchMode.WORKER,
) -> "DPMHTTPBackend":
    """Create a DPM backend instance with its own connection pool.

    Alias for dpm_http(). Each subscribe() call creates its own TCP
    connection, allowing independent subscriptions.

    Args:
        host: DPM proxy hostname (default: acsys-proxy.fnal.gov)
        port: DPM proxy port (default: 6802)
        pool_size: Connection pool size (default: 4)
        timeout: Default operation timeout in seconds (default: 5.0)
        auth: Authentication object (KerberosAuth for writes)
        role: Role for authenticated operations (e.g., "testing")
              Required for write operations.
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
    host: Optional[str] = None,
    port: Optional[int] = None,
    pool_size: Optional[int] = None,
    timeout: Optional[float] = None,
    auth: Optional[Auth] = None,
    role: Optional[str] = None,
    dispatch_mode: DispatchMode = DispatchMode.WORKER,
) -> "DPMHTTPBackend":
    """Create a DPM HTTP backend with independent streaming subscriptions.

    This backend uses TCP/HTTP protocol to communicate with DPM. Each
    subscribe() call creates its own TCP connection, allowing truly
    independent subscriptions that can be started/stopped individually.

    Args:
        host: DPM proxy hostname (default: acsys-proxy.fnal.gov)
        port: DPM proxy port (default: 6802)
        pool_size: Connection pool size for reads (default: 4)
        timeout: Default operation timeout in seconds (default: 5.0)
        auth: Authentication object (KerberosAuth for writes)
        role: Role for authenticated operations (e.g., "testing")

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

    effective_host = host if host is not None else "acsys-proxy.fnal.gov"
    effective_port = port if port is not None else 6802
    effective_pool_size = pool_size if pool_size is not None else 4
    effective_timeout = timeout if timeout is not None else 5.0

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
    host: Optional[str] = None,
    port: Optional[int] = None,
    auth: Optional[Auth] = None,
    timeout: Optional[float] = None,
    dispatch_mode: DispatchMode = DispatchMode.WORKER,
) -> "GRPCBackend":
    """Create a gRPC backend instance.

    Uses the DAQ gRPC service for reads and writes. Writes require
    JWT authentication.

    Args:
        host: gRPC server hostname (default: localhost)
        port: gRPC server port (default: 23456)
        auth: Authentication object (JWTAuth for writes). If None, tries PACSYS_JWT_TOKEN env.
        timeout: Default operation timeout in seconds (default: 5.0)

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
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
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
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[float] = None,
    auth: Optional[Auth] = None,
    write_session_ttl: Optional[float] = None,
    dispatch_mode: DispatchMode = DispatchMode.WORKER,
) -> "DMQBackend":
    """Create a DMQ backend instance (RabbitMQ/AMQP).

    Uses RabbitMQ to communicate with ACNET via the DMQ server.
    Requires Kerberos authentication for ALL operations (including reads).

    Args:
        host: RabbitMQ broker hostname (default: from PACSYS_DMQ_HOST or appsrv2.fnal.gov)
        port: RabbitMQ broker port (default: from PACSYS_DMQ_PORT or 5672)
        timeout: Default operation timeout in seconds (default: 5.0)
        auth: KerberosAuth required for all DMQ operations
        write_session_ttl: Idle timeout for write sessions in seconds (default: 600)

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
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[float] = None,
    cache_ttl: float = 3600.0,
) -> "DevDBClient":
    """Create a DevDB client for device metadata queries.

    DevDB provides device information like scaling parameters, control commands,
    and status bit definitions from the master PostgreSQL database.

    Args:
        host: DevDB gRPC hostname (default: from PACSYS_DEVDB_HOST or localhost)
        port: DevDB gRPC port (default: from PACSYS_DEVDB_PORT or 6802)
        timeout: RPC timeout in seconds (default: 5.0)
        cache_ttl: Cache TTL in seconds (default: 3600.0)

    Returns:
        DevDBClient instance (use as context manager or call close() when done)

    Raises:
        ImportError: If grpc package is not available

    Example:
        with pacsys.devdb(host="localhost", port=45678) as db:
            info = db.get_device_info(["Z:ACLTST"])
            print(info["Z:ACLTST"].description)
    """
    from pacsys.devdb import DevDBClient

    return _track(DevDBClient(host=host, port=port, timeout=timeout, cache_ttl=cache_ttl))


def ssh(
    hops: "Union[str, SSHHop, list[str | SSHHop]]",
    auth: Optional[Auth] = None,
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
    policies: Optional[list] = None,
) -> "SupervisedServer":
    """Create a supervised gRPC proxy server with logging and policy enforcement.

    Wraps any Backend and exposes it as a gRPC DAQ service, forwarding
    requests while enforcing access policies and logging all traffic.

    Args:
        backend: Backend instance to proxy requests to
        port: Port to listen on (default: 50051). Use 0 for OS-assigned.
        host: Host to bind (default: "[::] " for all interfaces)
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
    # ramp
    "Ramp": "pacsys.ramp",
    "RampGroup": "pacsys.ramp",
    "BoosterHVRamp": "pacsys.ramp",
    "BoosterHVRampGroup": "pacsys.ramp",
    "read_ramps": "pacsys.ramp",
    "write_ramps": "pacsys.ramp",
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
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(mod, name)
        # Cache on module to avoid repeated __getattr__ calls
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
    # Ramp
    "Ramp",
    "RampGroup",
    "BoosterHVRamp",
    "BoosterHVRampGroup",
    "read_ramps",
    "write_ramps",
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
    # Simple API functions
    "read",
    "get",
    "get_many",
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

# Testing utilities (import explicitly when needed):
# from pacsys.testing import FakeBackend
