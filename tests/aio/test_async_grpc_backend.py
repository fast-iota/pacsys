"""Tests for AsyncGRPCBackend."""

import asyncio
from unittest import mock

import pytest

from pacsys.errors import AuthenticationError, DeviceError
from pacsys.types import Reading, ValueType, WriteResult, BackendCapability

try:
    from pacsys.backends import grpc_backend

    GRPC_AVAILABLE = grpc_backend.GRPC_AVAILABLE
except ImportError:
    GRPC_AVAILABLE = False

if not GRPC_AVAILABLE:
    pytest.skip("grpc and proto files not available", allow_module_level=True)

from pacsys.aio._grpc import AsyncGRPCBackend


def _make_reading(drf="M:OUTTMP", val=72.5, error_code=0):
    return Reading(drf=drf, value_type=ValueType.SCALAR, value=val, error_code=error_code)


def _make_error_reading(drf="M:OUTTMP"):
    return Reading(drf=drf, value_type=ValueType.SCALAR, value=None, error_code=-10, message="Bad")


def _make_write_result(drf="M:OUTTMP.SETTING@N", error_code=0):
    return WriteResult(drf=drf, error_code=error_code)


@pytest.fixture
def backend():
    """Create AsyncGRPCBackend with mocked core."""
    b = AsyncGRPCBackend()
    b._core = mock.AsyncMock()
    b._connected = True
    return b


class TestAsyncGRPCRead:
    @pytest.mark.asyncio
    async def test_read_single(self, backend):
        backend._core.read_many = mock.AsyncMock(return_value=[_make_reading()])
        val = await backend.read("M:OUTTMP")
        assert val == 72.5
        backend._core.read_many.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_many(self, backend):
        readings = [_make_reading("M:OUTTMP"), _make_reading("G:AMANDA", val=1.0)]
        backend._core.read_many = mock.AsyncMock(return_value=readings)
        result = await backend.get_many(["M:OUTTMP", "G:AMANDA"])
        assert len(result) == 2
        assert result[0].value == 72.5
        assert result[1].value == 1.0

    @pytest.mark.asyncio
    async def test_get_many_empty(self, backend):
        result = await backend.get_many([])
        assert result == []

    @pytest.mark.asyncio
    async def test_read_error_raises(self, backend):
        backend._core.read_many = mock.AsyncMock(return_value=[_make_error_reading()])
        with pytest.raises(DeviceError):
            await backend.read("M:OUTTMP")


class TestAsyncGRPCWrite:
    @pytest.mark.asyncio
    async def test_write(self, backend):
        from pacsys.auth import JWTAuth
        from tests.devices import make_jwt_token

        backend._auth = JWTAuth(token=make_jwt_token({"sub": "test@fnal.gov"}))
        backend._core.write_many = mock.AsyncMock(return_value=[_make_write_result()])
        result = await backend.write("M:OUTTMP", 72.5)
        assert result.success
        backend._core.write_many.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_no_auth_raises(self, backend):
        with pytest.raises(AuthenticationError):
            await backend.write("M:OUTTMP", 72.5)

    @pytest.mark.asyncio
    async def test_write_many_empty(self, backend):
        from pacsys.auth import JWTAuth
        from tests.devices import make_jwt_token

        backend._auth = JWTAuth(token=make_jwt_token({"sub": "test@fnal.gov"}))
        result = await backend.write_many([])
        assert result == []


class TestAsyncGRPCSubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_creates_task(self, backend):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        async def fake_stream(drfs, dispatch_fn, stop_check, error_fn):
            dispatch_fn(_make_reading())
            return

        backend._core.stream = fake_stream
        handle = await backend.subscribe(["M:OUTTMP@p,1000"])
        assert isinstance(handle, AsyncSubscriptionHandle)
        # Wait for the stream task to complete
        await asyncio.sleep(0.05)
        assert not handle._queue.empty()

    @pytest.mark.asyncio
    async def test_subscribe_with_callback(self, backend):
        collected = []

        async def fake_stream(drfs, dispatch_fn, stop_check, error_fn):
            dispatch_fn(_make_reading(val=10.0))
            dispatch_fn(_make_reading(val=20.0))
            return

        backend._core.stream = fake_stream

        async def on_reading(reading, handle):
            collected.append(reading.value)

        handle = await backend.subscribe(["M:OUTTMP@p,1000"], callback=on_reading)
        await asyncio.sleep(0.1)
        handle._signal_stop()
        if handle._callback_task:
            await handle._callback_task
        assert collected == [10.0, 20.0]


class TestAsyncGRPCMisc:
    @pytest.mark.asyncio
    async def test_context_manager_closes(self, backend):
        close_mock = mock.AsyncMock()
        backend._core.close = close_mock
        async with backend:
            pass
        close_mock.assert_awaited_once()

    def test_capabilities_read_only(self):
        b = AsyncGRPCBackend()
        assert BackendCapability.READ in b.capabilities
        assert BackendCapability.STREAM in b.capabilities
        assert BackendCapability.WRITE not in b.capabilities

    def test_capabilities_with_auth(self):
        from pacsys.auth import JWTAuth
        from tests.devices import make_jwt_token

        b = AsyncGRPCBackend(auth=JWTAuth(token=make_jwt_token({"sub": "test@fnal.gov"})))
        assert BackendCapability.WRITE in b.capabilities
        assert BackendCapability.AUTH_JWT in b.capabilities

    def test_auth_defaults(self):
        b = AsyncGRPCBackend()
        assert b.authenticated is False
        assert b.principal is None

    @pytest.mark.asyncio
    async def test_closed_backend_raises(self, backend):
        await backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            await backend.read("M:OUTTMP")

    @pytest.mark.asyncio
    async def test_subscribe_empty_drfs_raises(self, backend):
        with pytest.raises(ValueError, match="drfs cannot be empty"):
            await backend.subscribe([])

    @pytest.mark.asyncio
    async def test_subscribe_error_adapter_nonfatal(self, backend):
        """Non-fatal gRPC errors don't kill the subscription."""

        async def fake_stream(drfs, dispatch_fn, stop_check, error_fn):
            # gRPC _DaqCore.stream calls error_fn with fatal=False for transient errors
            error_fn(RuntimeError("transport error"), fatal=False)
            # Stream should NOT be stopped — subscription continues retrying
            assert not stop_check()

        backend._core.stream = fake_stream
        handle = await backend.subscribe(["M:OUTTMP@p,1000"])
        await asyncio.sleep(0.05)
        # Handle should NOT be stopped by a non-fatal error
        assert not handle.stopped

    @pytest.mark.asyncio
    async def test_subscribe_error_adapter_fatal(self, backend):
        """Fatal gRPC errors stop the subscription."""

        async def fake_stream(drfs, dispatch_fn, stop_check, error_fn):
            error_fn(RuntimeError("fatal error"), fatal=True)

        backend._core.stream = fake_stream
        handle = await backend.subscribe(["M:OUTTMP@p,1000"])
        await asyncio.sleep(0.05)
        assert handle.stopped
        assert isinstance(handle.exc, RuntimeError)
