"""Async gRPC backend - thin shell over _DaqCore."""

import asyncio
import logging
from typing import Optional

from pacsys.aio._backends import AsyncBackend
from pacsys.aio._subscription import AsyncSubscriptionHandle, _callback_feeder
from pacsys.drf_utils import prepare_for_write
from pacsys.errors import AuthenticationError, DeviceError
from pacsys.types import (
    Value,
    Reading,
    WriteResult,
    BackendCapability,
    ReadingCallback,
    ErrorCallback,
)

logger = logging.getLogger(__name__)

try:
    from pacsys.backends.grpc_backend import _DaqCore, GRPC_AVAILABLE
except ImportError:
    GRPC_AVAILABLE = False
    _DaqCore = None  # type: ignore[assignment,misc]


class AsyncGRPCBackend(AsyncBackend):
    """Async gRPC backend. Wraps _DaqCore directly, no reactor thread."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        auth=None,
        timeout: float = 5.0,
    ):
        if not GRPC_AVAILABLE:
            raise ImportError("grpc package not available")
        self._host = host or "localhost"
        self._port = port or 23456
        self._auth = auth
        self._timeout = timeout
        self._core: Optional[_DaqCore] = None
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
            self._core = _DaqCore(self._host, self._port, self._auth, self._timeout)
            await self._core.connect()
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
    def principal(self) -> Optional[str]:
        return self._auth.principal if self._auth else None

    async def read(self, drf: str, timeout: Optional[float] = None) -> Value:
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

    async def get(self, drf: str, timeout: Optional[float] = None) -> Reading:
        readings = await self.get_many([drf], timeout=timeout)
        return readings[0]

    async def get_many(self, drfs: list[str], timeout: Optional[float] = None) -> list[Reading]:
        if not drfs:
            return []
        await self._ensure_connected()
        assert self._core is not None
        effective_timeout = timeout if timeout is not None else self._timeout
        return await self._core.read_many(drfs, effective_timeout)

    async def write(self, drf: str, value: Value, timeout: Optional[float] = None) -> WriteResult:
        results = await self.write_many([(drf, value)], timeout=timeout)
        return results[0]

    async def write_many(
        self,
        settings: list[tuple[str, Value]],
        timeout: Optional[float] = None,
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
        callback: Optional[ReadingCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ) -> AsyncSubscriptionHandle:
        if not drfs:
            raise ValueError("drfs cannot be empty")
        await self._ensure_connected()
        assert self._core is not None
        handle = AsyncSubscriptionHandle()

        def _error_adapter(exc, fatal=False):
            if fatal:
                handle._signal_error(exc)
            else:
                logger.warning("gRPC stream transient error (will retry): %s", exc)

        handle._task = asyncio.ensure_future(
            self._core.stream(drfs, handle._dispatch, handle._is_stopped, _error_adapter)
        )
        if callback:
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
