"""Async gRPC backend - thin shell over _DaqCore."""

import asyncio
import logging
from typing import Any, cast

from pacsys.aio._backends import AsyncBackend
from pacsys.aio._subscription import AsyncSubscriptionHandle, _call_on_error, _callback_feeder
from pacsys.drf_utils import prepare_for_write
from pacsys.errors import AuthenticationError, DeviceError
from pacsys.types import (
    BackendCapability,
    ErrorCallback,
    Reading,
    ReadingCallback,
    Value,
    WriteResult,
    _validate_callback,
)

logger = logging.getLogger(__name__)

try:
    from pacsys.backends.grpc_backend import GRPC_AVAILABLE, _DaqCore
except ImportError:
    GRPC_AVAILABLE = False
    _DaqCore = cast("Any", None)


class AsyncGRPCBackend(AsyncBackend):
    """Async gRPC backend. Wraps _DaqCore directly, no reactor thread."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        auth=None,
        timeout: float | None = None,
    ):
        if not GRPC_AVAILABLE:
            raise ImportError("grpc package not available")
        from pacsys.backends.grpc_backend import _resolve_config

        self._host, self._port, self._auth, self._timeout = _resolve_config(host, port, auth, timeout)
        self._core: _DaqCore | None = None
        self._connected = False
        self._closed = False
        self._connect_lock = asyncio.Lock()
        self._handles: list[AsyncSubscriptionHandle] = []

    async def _ensure_connected(self):
        if self._closed:
            raise RuntimeError("Backend is closed")
        if self._connected:
            return
        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("Backend is closed")
            if self._connected:
                return
            # Publish only after connect + closed re-check so a concurrent
            # close() cannot strand a half-connected core.
            core = _DaqCore(self._host, self._port, self._auth, self._timeout)
            await core.connect()
            if self._closed:
                await core.close()
                raise RuntimeError("Backend is closed")
            self._core = core
            self._connected = True

    @property
    def capabilities(self) -> BackendCapability:
        caps = BackendCapability.READ | BackendCapability.STREAM | BackendCapability.BATCH
        if self._auth is not None:
            caps |= BackendCapability.WRITE | BackendCapability.AUTH_JWT
        return caps

    @property
    def authenticated(self) -> bool:
        return self._auth is not None

    @property
    def principal(self) -> str | None:
        return self._auth.principal if self._auth else None

    async def read(self, drf: str, timeout: float | None = None) -> Value:
        reading = await self.get(drf, timeout=timeout)
        if not reading.ok:
            raise DeviceError(
                drf=reading.drf,
                facility_code=reading.facility_code,
                error_code=reading.error_code,
                message=reading.message or "Read failed",
            )
        assert reading.value is not None
        return reading.value

    async def get(self, drf: str, timeout: float | None = None) -> Reading:
        readings = await self.get_many([drf], timeout=timeout)
        return readings[0]

    async def get_many(self, drfs: list[str], timeout: float | None = None) -> list[Reading]:
        if not drfs:
            return []
        await self._ensure_connected()
        assert self._core is not None
        effective_timeout = timeout if timeout is not None else self._timeout
        return await self._core.read_many(drfs, effective_timeout)

    async def write(self, drf: str, value: Value, timeout: float | None = None) -> WriteResult:
        results = await self.write_many([(drf, value)], timeout=timeout)
        return results[0]

    async def write_many(
        self,
        settings: list[tuple[str, Value]],
        timeout: float | None = None,
    ) -> list[WriteResult]:
        if not settings:
            return []
        if self._auth is None:
            raise AuthenticationError(
                "JWTAuth required for write operations. Provide auth=JWTAuth(token=...) or set PACSYS_JWT_TOKEN."
            )
        await self._ensure_connected()
        assert self._core is not None
        prepared = [(prepare_for_write(drf), value) for drf, value in settings]
        effective_timeout = timeout if timeout is not None else self._timeout
        return await self._core.write_many(prepared, effective_timeout)

    async def subscribe(
        self,
        drfs: list[str],
        callback: ReadingCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> AsyncSubscriptionHandle:
        if not drfs:
            raise ValueError("drfs cannot be empty")
        _validate_callback(callback, on_error)
        await self._ensure_connected()
        assert self._core is not None
        handle = AsyncSubscriptionHandle(remover=self.remove)
        handle._drfs = drfs

        def _error_adapter(exc, fatal=False):
            if fatal:
                handle._signal_error(exc)
            # Sync parity: on_error sees every error. A callback feeder forwards fatal ones itself
            # (raised out of _readings); retryable ones are logged by the core and never end the handle.
            if on_error is not None and (callback is None or not fatal):
                handle._spawn(_call_on_error(on_error, exc, handle))

        core = self._core

        async def _run_stream():
            try:
                await core.stream(drfs, handle._dispatch, handle._is_stopped, _error_adapter)
            except Exception as exc:  # noqa: BLE001
                # A failure escaping the core is a subscription error, not a
                # graceful end -- consumers must see it raised.
                handle._signal_error(exc)
            finally:
                # Stream end (server onCompleted, fatal error, cancel) must
                # stop the handle or readings() blocks forever
                handle._signal_stop()
                if handle in self._handles:
                    self._handles.remove(handle)

        handle._task = asyncio.ensure_future(_run_stream())
        if callback is not None:
            handle._callback_task = asyncio.ensure_future(_callback_feeder(handle, callback, on_error))
        self._handles.append(handle)
        return handle

    async def remove(self, handle) -> None:
        if isinstance(handle, AsyncSubscriptionHandle):
            await handle.stop()
            if handle in self._handles:
                self._handles.remove(handle)

    async def stop_streaming(self) -> None:
        for h in list(self._handles):
            await h.stop()
        self._handles.clear()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.stop_streaming()
        if self._core is not None:
            await self._core.close()
            self._core = None
        self._connected = False
