"""pacsys.aio - async API for pacsys backends.

Module-level convenience API mirrors pacsys.read(), pacsys.get(), etc.
for use in async code. Uses lazy-initialized global async backend.

    import pacsys.aio as aio

    aio.configure(backend="dpm")
    value = await aio.read("M:OUTTMP")
    await aio.shutdown()
"""

import asyncio
from typing import Optional

from pacsys.aio._backends import AsyncBackend
from pacsys.aio._device import AsyncDevice
from pacsys.aio._subscription import AsyncSubscriptionHandle

__all__ = [
    "AsyncBackend",
    "AsyncDevice",
    "AsyncSubscriptionHandle",
    "configure",
    "shutdown",
    "read",
    "read_many",
    "get",
    "get_many",
    "write",
    "write_many",
    "subscribe",
    "grpc",
    "dpm",
]


# ── Backend Factory Functions ─────────────────────────────────────────────


def grpc(host=None, port=None, auth=None, timeout=5.0):
    """Create an async gRPC backend."""
    from pacsys.aio._grpc import AsyncGRPCBackend

    return AsyncGRPCBackend(host=host, port=port, auth=auth, timeout=timeout)


def dpm(host=None, port=None, pool_size=None, timeout=None, auth=None, role=None):
    """Create an async DPM HTTP backend."""
    from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

    from pacsys import _env_dpm_host, _env_dpm_port, _env_pool_size, _env_timeout

    return AsyncDPMHTTPBackend(
        host=host if host is not None else (_env_dpm_host if _env_dpm_host is not None else "acsys-proxy.fnal.gov"),
        port=port if port is not None else (_env_dpm_port if _env_dpm_port is not None else 6802),
        pool_size=pool_size if pool_size is not None else (_env_pool_size if _env_pool_size is not None else 4),
        timeout=timeout if timeout is not None else (_env_timeout if _env_timeout is not None else 5.0),
        auth=auth,
        role=role,
    )


# ── Module-level Global Backend ──────────────────────────────────────────

_UNSET = object()
_VALID_ASYNC_BACKENDS = {"dpm", "grpc"}

_config_backend: Optional[str] = None
_config_auth = None
_config_role: Optional[str] = None
_config_host: Optional[str] = None
_config_port: Optional[int] = None
_config_pool_size: Optional[int] = None
_config_timeout: Optional[float] = None

_global_async_backend: AsyncBackend | None = None
_async_backend_initialized: bool = False


def configure(
    *,
    backend: Optional[str] = _UNSET,  # type: ignore[assignment]
    host: Optional[str] = _UNSET,  # type: ignore[assignment]
    port: Optional[int] = _UNSET,  # type: ignore[assignment]
    pool_size: Optional[int] = _UNSET,  # type: ignore[assignment]
    timeout: Optional[float] = _UNSET,  # type: ignore[assignment]
    auth=_UNSET,
    role: Optional[str] = _UNSET,  # type: ignore[assignment]
) -> None:
    """Configure async backend settings.

    Can be called at any time. If a backend is already initialized, it will
    be marked as closed and replaced on the next operation. Pass None to
    clear a previously set value (falls back to default).

    For a graceful close of the old backend (flushing connections), call
    ``await shutdown()`` before ``configure()``.

    Args:
        backend: Backend type - "dpm" or "grpc" (default: "dpm")
        host: Server hostname
        port: Server port
        pool_size: Connection pool size (DPM only, default: 4)
        timeout: Default timeout in seconds (default: 5.0)
        auth: Authentication object (KerberosAuth for DPM, JWTAuth for gRPC),
              or "krb" as shortcut for KerberosAuth()
        role: Role for authenticated operations (DPM only)

    Raises:
        ValueError: If backend is not a valid type
    """
    global _config_backend, _config_auth, _config_role
    global _config_host, _config_port, _config_pool_size, _config_timeout
    global _global_async_backend, _async_backend_initialized

    if _async_backend_initialized:
        old_backend = _global_async_backend
        _global_async_backend = None
        _async_backend_initialized = False
        if old_backend is not None:
            old_backend._closed = True
            # Schedule proper cleanup if event loop is running
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(old_backend.close())
            except RuntimeError:
                # No running loop -- force close synchronously
                old_backend._closed = True

    if backend is not _UNSET:
        if backend is not None and backend not in _VALID_ASYNC_BACKENDS:
            raise ValueError(f"Invalid backend {backend!r}, must be one of {sorted(_VALID_ASYNC_BACKENDS)}")
        _config_backend = backend
    if auth is not _UNSET:
        if auth == "krb":
            from pacsys.auth import KerberosAuth

            auth = KerberosAuth()
        _config_auth = auth
    if role is not _UNSET:
        _config_role = role
    if host is not _UNSET:
        _config_host = host
    if port is not _UNSET:
        _config_port = port
    if pool_size is not _UNSET:
        _config_pool_size = pool_size
    if timeout is not _UNSET:
        _config_timeout = timeout


async def shutdown() -> None:
    """Close and release the global async backend.

    After shutdown(), configure() can be called again and the next
    operation will re-initialize the backend.
    """
    global _global_async_backend, _async_backend_initialized

    if _global_async_backend is not None:
        await _global_async_backend.close()
        _global_async_backend = None
    _async_backend_initialized = False


def _get_global_async_backend() -> AsyncBackend:
    """Get or create the global async backend (lazy initialization).

    Backend creation is synchronous (no I/O at construction).
    Connection happens lazily on first operation.
    """
    global _global_async_backend, _async_backend_initialized

    if _global_async_backend is not None:
        return _global_async_backend

    timeout = _config_timeout if _config_timeout is not None else 5.0
    backend_type = _config_backend if _config_backend is not None else "dpm"

    if backend_type == "dpm":
        _global_async_backend = dpm(
            host=_config_host,
            port=_config_port,
            pool_size=_config_pool_size,
            timeout=timeout,
            auth=_config_auth,
            role=_config_role,
        )
    elif backend_type == "grpc":
        _global_async_backend = grpc(
            host=_config_host,
            port=_config_port,
            auth=_config_auth,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown backend type {backend_type!r}")

    _async_backend_initialized = True
    return _global_async_backend


# ── DRF Resolution ────────────────────────────────────────────────────────


def _resolve_drf(device) -> str:
    """Convert device spec to DRF string. Supports str, Device, and AsyncDevice."""
    if isinstance(device, str):
        return device
    from pacsys._device_base import _DeviceBase

    if isinstance(device, _DeviceBase):
        return device.drf
    raise TypeError(f"Expected str or device, got {type(device).__name__}")


# ── Simple API Functions ──────────────────────────────────────────────────


async def read(device, timeout: Optional[float] = None):
    """Read a single device value using the global async backend."""
    drf = _resolve_drf(device)
    backend = _get_global_async_backend()
    return await backend.read(drf, timeout=timeout)


async def get(device, timeout: Optional[float] = None):
    """Read a single device with full metadata."""
    drf = _resolve_drf(device)
    backend = _get_global_async_backend()
    return await backend.get(drf, timeout=timeout)


async def get_many(devices: list, timeout: Optional[float] = None):
    """Read multiple devices in a single batch."""
    drfs = [_resolve_drf(d) for d in devices]
    backend = _get_global_async_backend()
    return await backend.get_many(drfs, timeout=timeout)


async def read_many(devices: list, timeout: Optional[float] = None):
    """Read multiple device values in a single batch.

    Returns bare values. Raises ReadError if any device fails.
    """
    from pacsys.errors import ReadError

    drfs = [_resolve_drf(d) for d in devices]
    backend = _get_global_async_backend()
    readings = await backend.get_many(drfs, timeout=timeout)
    errors = [r for r in readings if not r.ok]
    if errors:
        failed = ", ".join(r.drf for r in errors)
        raise ReadError(readings, f"Device errors: {failed}")
    return [r.value for r in readings]


async def write(device, value, timeout: Optional[float] = None):
    """Write a single device value."""
    drf = _resolve_drf(device)
    backend = _get_global_async_backend()
    return await backend.write(drf, value, timeout=timeout)


async def write_many(settings, timeout: Optional[float] = None):
    """Write multiple device values in a single batch."""
    items = settings.items() if isinstance(settings, dict) else settings
    resolved = [(_resolve_drf(d), v) for d, v in items]
    backend = _get_global_async_backend()
    return await backend.write_many(resolved, timeout=timeout)


async def subscribe(drfs: list, callback=None, on_error=None):
    """Subscribe to devices for streaming."""
    resolved = [_resolve_drf(d) for d in drfs]
    backend = _get_global_async_backend()
    return await backend.subscribe(resolved, callback=callback, on_error=on_error)
