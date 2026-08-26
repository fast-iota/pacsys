"""pacsys.aio - async API for pacsys backends.

Module-level convenience API mirrors pacsys.read(), pacsys.get(), etc.
for use in async code. Uses lazy-initialized global async backend.

    import pacsys.aio as aio

    aio.configure(backend="dpm")
    value = await aio.read("M:OUTTMP")
    await aio.shutdown()
"""

import asyncio
import logging

from pacsys import _validate_config_host, _validate_config_port, _validate_config_positive
from pacsys.aio._backends import AsyncBackend
from pacsys.aio._device import AsyncDevice
from pacsys.aio._subscription import AsyncSubscriptionHandle
from pacsys.auth import Auth, JWTAuth, KerberosAuth

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
    from pacsys import _env_dpm_host, _env_dpm_port, _env_pool_size, _env_timeout
    from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

    return AsyncDPMHTTPBackend(
        host=host if host is not None else (_env_dpm_host if _env_dpm_host is not None else "acsys-proxy.fnal.gov"),
        port=port if port is not None else (_env_dpm_port if _env_dpm_port is not None else 6802),
        pool_size=pool_size if pool_size is not None else (_env_pool_size if _env_pool_size is not None else 4),
        timeout=timeout if timeout is not None else (_env_timeout if _env_timeout is not None else 5.0),
        auth=auth,
        role=role,
    )


# ── Module-level Global Backend ──────────────────────────────────────────


class _Unset:
    """Sentinel distinguishing omitted configuration from an explicit None."""


_UNSET = _Unset()
_VALID_ASYNC_BACKENDS = {"dpm", "grpc"}

_config_backend: str | None = None
_config_auth: Auth | None = None
_config_role: str | None = None
_config_host: str | None = None
_config_port: int | None = None
_config_pool_size: int | None = None
_config_timeout: float | None = None

_global_async_backend: AsyncBackend | None = None
_async_backend_initialized: bool = False
_owner_loop: asyncio.AbstractEventLoop | None = None
_background_tasks: set[asyncio.Task[None]] = set()


logger = logging.getLogger(__name__)


def _retain_background_task(task: asyncio.Task[None]) -> None:
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _close_old_backend(backend: AsyncBackend) -> None:
    try:
        await backend.close()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to close replaced async backend")


def configure(
    *,
    backend: str | _Unset | None = _UNSET,
    host: str | _Unset | None = _UNSET,
    port: int | _Unset | None = _UNSET,
    pool_size: int | _Unset | None = _UNSET,
    timeout: float | _Unset | None = _UNSET,
    auth: Auth | str | _Unset | None = _UNSET,
    role: str | _Unset | None = _UNSET,
) -> None:
    """Configure async backend settings.

    If a backend is already initialized, its cleanup is scheduled on the
    running event loop (best-effort) and it is replaced on the next
    operation. Pass None to clear a previously set value (falls back to
    default).

    For deterministic cleanup of the old backend (flushing connections),
    call ``await shutdown()`` before ``configure()``.

    Args:
        backend: Backend type - "dpm" or "grpc" (default: "dpm")
        host: Server hostname
        port: Server port
        pool_size: Connection pool size (DPM only, default: 4)
        timeout: Default timeout in seconds (default: from PACSYS_TIMEOUT or 5.0)
        auth: Authentication object (KerberosAuth for DPM, JWTAuth for gRPC),
              or "krb" as shortcut for KerberosAuth()
        role: Role for authenticated operations (DPM only)

    Raises:
        ValueError: If backend is not a valid type
        RuntimeError: If a backend is initialized but no event loop is
            running (its resources are loop-bound and cannot be cleaned up
            here -- ``await shutdown()`` on the owning loop first)
    """
    global _config_backend, _config_auth, _config_role
    global _config_host, _config_port, _config_pool_size, _config_timeout
    global _global_async_backend, _async_backend_initialized, _owner_loop

    if not isinstance(backend, _Unset) and backend is not None and backend not in _VALID_ASYNC_BACKENDS:
        raise ValueError(f"Invalid backend {backend!r}, must be one of {sorted(_VALID_ASYNC_BACKENDS)}")
    if isinstance(auth, str):
        if auth != "krb":
            raise ValueError("auth string must be 'krb'")
        normalized_auth = KerberosAuth()
    else:
        normalized_auth = auth

    if not isinstance(host, _Unset):
        _validate_config_host(host, "host")
    if not isinstance(port, _Unset):
        _validate_config_port(port, "port")
    if not isinstance(pool_size, _Unset):
        _validate_config_positive(pool_size, "pool_size", integer=True)
    if not isinstance(timeout, _Unset):
        _validate_config_positive(timeout, "timeout")

    configured_backend = backend if not isinstance(backend, _Unset) else _config_backend
    configured_auth = normalized_auth if not isinstance(normalized_auth, _Unset) else _config_auth
    effective_backend = configured_backend or "dpm"
    if effective_backend == "dpm" and configured_auth is not None and not isinstance(configured_auth, KerberosAuth):
        raise ValueError("DPM backend auth must be KerberosAuth or None")
    if effective_backend == "grpc" and configured_auth is not None and not isinstance(configured_auth, JWTAuth):
        raise ValueError("gRPC backend auth must be JWTAuth or None")

    if _async_backend_initialized:
        old_backend = _global_async_backend
        if old_backend is not None:
            # Resolve the loop before touching global state so a failed
            # configure() leaves shutdown() reachable.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                raise RuntimeError(
                    "Cannot reconfigure an initialized async backend without a running "
                    "event loop; await pacsys.aio.shutdown() on its owning loop first"
                ) from None
            owner = _owner_loop
            if owner is not None and owner is not loop and not owner.is_closed():
                raise RuntimeError(
                    "configure() called from a different event loop than the one owning "
                    "the initialized async backend; await pacsys.aio.shutdown() on the "
                    "owning loop first"
                )
            if owner is None or owner is loop:
                _retain_background_task(loop.create_task(_close_old_backend(old_backend)))
            else:
                # Owner loop already closed -- cleanup is impossible by any means.
                logger.warning(
                    "Abandoning async backend owned by a closed event loop; its connections "
                    "cannot be cleaned up. await pacsys.aio.shutdown() before the owning "
                    "loop exits to avoid this."
                )
        _global_async_backend = None
        _async_backend_initialized = False
        _owner_loop = None

    if not isinstance(backend, _Unset):
        _config_backend = backend
    if not isinstance(auth, _Unset):
        assert not isinstance(normalized_auth, _Unset)
        _config_auth = normalized_auth
    if not isinstance(role, _Unset):
        _config_role = role
    if not isinstance(host, _Unset):
        _config_host = host
    if not isinstance(port, _Unset):
        _config_port = port
    if not isinstance(pool_size, _Unset):
        _config_pool_size = pool_size
    if not isinstance(timeout, _Unset):
        _config_timeout = timeout


async def shutdown() -> None:
    """Close and release the global async backend.

    After shutdown(), configure() can be called again and the next
    operation will re-initialize the backend.

    Must run on the backend's owning event loop. If the owning loop has
    already closed, the backend is abandoned with a warning (its resources
    are unrecoverable); if the owning loop is still open elsewhere, raises
    RuntimeError instead of clobbering it.
    """
    global _global_async_backend, _async_backend_initialized, _owner_loop

    backend = _global_async_backend
    if backend is None:
        _async_backend_initialized = False
        return
    owner = _owner_loop
    if owner is not None and owner is not asyncio.get_running_loop():
        if not owner.is_closed():
            raise RuntimeError(
                "shutdown() called from a different event loop than the one owning "
                "the global async backend; await it on the owning loop"
            )
        logger.warning(
            "Abandoning async backend owned by a closed event loop; its connections "
            "cannot be cleaned up. await pacsys.aio.shutdown() before the owning "
            "loop exits to avoid this."
        )
    else:
        await backend.close()
    # Identity check: a concurrent configure() may have installed a new backend
    # while close() was awaited -- never erase that one.
    if _global_async_backend is backend:
        _global_async_backend = None
        _owner_loop = None
        _async_backend_initialized = False


def _get_global_async_backend() -> AsyncBackend:
    """Get or create the global async backend (lazy initialization).

    Backend creation is synchronous (no I/O at construction).
    Connection happens lazily on first operation.

    Raises RuntimeError when called from a loop other than the one that
    created the backend -- its pool/transports are loop-bound and would
    fail cryptically (or corrupt state) if reused cross-loop.
    """
    global _global_async_backend, _async_backend_initialized, _owner_loop

    if _global_async_backend is not None:
        if _owner_loop is not None and _owner_loop is not asyncio.get_running_loop():
            if _owner_loop.is_closed():
                raise RuntimeError(
                    "Global async backend belongs to an event loop that has closed. "
                    "await pacsys.aio.shutdown() before the owning loop exits; to "
                    "recover now, await pacsys.aio.shutdown() (abandons the stale "
                    "backend) and retry, or use explicit backend instances "
                    "(aio.dpm()/aio.grpc())."
                )
            raise RuntimeError(
                "Global async backend belongs to a different, still-open event loop. "
                "Use it from that loop, or use explicit backend instances "
                "(aio.dpm()/aio.grpc()) instead of the global API."
            )
        return _global_async_backend

    # Capture the loop before any global mutation: a raise here (no running
    # loop) must not leave a backend installed with _owner_loop=None, which
    # would disable the cross-loop guard above.
    owner_loop = asyncio.get_running_loop()

    from pacsys import _env_timeout

    timeout = _config_timeout if _config_timeout is not None else (_env_timeout if _env_timeout is not None else 5.0)
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

    _owner_loop = owner_loop
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


async def read(device, timeout: float | None = None):
    """Read a single device value using the global async backend."""
    drf = _resolve_drf(device)
    backend = _get_global_async_backend()
    return await backend.read(drf, timeout=timeout)


async def get(device, timeout: float | None = None):
    """Read a single device with full metadata."""
    drf = _resolve_drf(device)
    backend = _get_global_async_backend()
    return await backend.get(drf, timeout=timeout)


async def get_many(devices: list, timeout: float | None = None):
    """Read multiple devices in a single batch."""
    drfs = [_resolve_drf(d) for d in devices]
    backend = _get_global_async_backend()
    return await backend.get_many(drfs, timeout=timeout)


async def read_many(devices: list, timeout: float | None = None):
    """Read multiple device values in a single batch.

    Returns bare values. Raises ReadError if any device fails.
    """
    drfs = [_resolve_drf(d) for d in devices]
    backend = _get_global_async_backend()
    return await backend.read_many(drfs, timeout=timeout)


async def write(device, value, timeout: float | None = None):
    """Write a single device value."""
    drf = _resolve_drf(device)
    backend = _get_global_async_backend()
    return await backend.write(drf, value, timeout=timeout)


async def write_many(settings, timeout: float | None = None):
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
