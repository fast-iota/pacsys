"""Tests for AsyncDPMHTTPBackend."""

import asyncio
from unittest import mock

import pytest

from pacsys.auth import KerberosAuth
from pacsys.errors import AuthenticationError, DeviceError
from pacsys.types import Reading, ValueType, WriteResult, BackendCapability

from pacsys.aio._dpm_http import AsyncDPMHTTPBackend


def _make_reading(drf="M:OUTTMP", val=72.5, error_code=0):
    return Reading(drf=drf, value_type=ValueType.SCALAR, value=val, error_code=error_code)


def _make_error_reading(drf="M:OUTTMP"):
    return Reading(drf=drf, value_type=ValueType.SCALAR, value=None, error_code=-10, message="Bad")


def _make_write_result(drf="M:OUTTMP.SETTING@N", error_code=0):
    return WriteResult(drf=drf, error_code=error_code)


def _mock_core():
    """Create a mock _AsyncDpmCore."""
    core = mock.AsyncMock()
    core.read_many = mock.AsyncMock(return_value=[_make_reading()])
    core.write_many = mock.AsyncMock(return_value=[_make_write_result()])
    core.connect = mock.AsyncMock()
    core.close = mock.AsyncMock()
    return core


@pytest.fixture
def backend():
    """AsyncDPMHTTPBackend with mocked core creation."""
    b = AsyncDPMHTTPBackend(host="localhost", port=6802)

    async def fake_create():
        return _mock_core()

    b._create_core = fake_create
    return b


class TestAsyncDPMRead:
    @pytest.mark.asyncio
    async def test_read_single(self, backend):
        val = await backend.read("M:OUTTMP")
        assert val == 72.5

    @pytest.mark.asyncio
    async def test_get_many(self, backend):
        readings = [_make_reading("M:OUTTMP"), _make_reading("G:AMANDA", val=1.0)]

        async def fake_create():
            core = _mock_core()
            core.read_many = mock.AsyncMock(return_value=readings)
            return core

        backend._create_core = fake_create
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
        async def fake_create():
            core = _mock_core()
            core.read_many = mock.AsyncMock(return_value=[_make_error_reading()])
            return core

        backend._create_core = fake_create
        with pytest.raises(DeviceError):
            await backend.read("M:OUTTMP")

    @pytest.mark.asyncio
    async def test_read_borrows_and_returns(self, backend):
        """Successful read returns core to pool."""
        await backend.read("M:OUTTMP")
        assert not backend._pool.empty()

    @pytest.mark.asyncio
    async def test_read_error_discards_core(self, backend):
        """On exception, core is discarded (not returned to pool)."""

        async def fake_create():
            core = _mock_core()
            core.read_many = mock.AsyncMock(side_effect=ConnectionError("broken"))
            return core

        backend._create_core = fake_create
        with pytest.raises(ConnectionError):
            await backend.get_many(["M:OUTTMP"])
        assert backend._pool.empty()


class TestAsyncDPMWrite:
    @pytest.mark.asyncio
    async def test_write_requires_auth(self, backend):
        with pytest.raises(AuthenticationError):
            await backend.write("M:OUTTMP", 72.5)

    @pytest.mark.asyncio
    async def test_write_single(self):
        auth = mock.MagicMock(spec=KerberosAuth)
        auth.principal = "test@FNAL.GOV"
        b = AsyncDPMHTTPBackend(host="localhost", port=6802, auth=auth)

        async def fake_create():
            return _mock_core()

        b._create_core = fake_create
        result = await b.write("M:OUTTMP", 72.5)
        assert result.success

    @pytest.mark.asyncio
    async def test_write_many_empty(self):
        auth = mock.MagicMock(spec=KerberosAuth)
        auth.principal = "test@FNAL.GOV"
        b = AsyncDPMHTTPBackend(host="localhost", port=6802, auth=auth)
        result = await b.write_many([])
        assert result == []


class TestAsyncDPMSubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_creates_dedicated_core(self, backend):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        async def fake_stream(drfs, dispatch, stop, error):
            dispatch(_make_reading())
            return

        async def fake_create():
            core = _mock_core()
            core.stream = fake_stream
            return core

        backend._create_core = fake_create
        handle = await backend.subscribe(["M:OUTTMP@p,1000"])
        assert isinstance(handle, AsyncSubscriptionHandle)
        await asyncio.sleep(0.05)
        assert not handle._queue.empty()


class TestAsyncDPMMisc:
    @pytest.mark.asyncio
    async def test_close_drains_pool(self, backend):
        await backend.read("M:OUTTMP")
        assert not backend._pool.empty()
        await backend.close()
        assert backend._pool.empty()
        assert backend._closed

    @pytest.mark.asyncio
    async def test_context_manager(self, backend):
        async with backend:
            await backend.read("M:OUTTMP")
        assert backend._closed

    def test_capabilities_read_only(self):
        b = AsyncDPMHTTPBackend(host="localhost", port=6802)
        assert BackendCapability.READ in b.capabilities
        assert BackendCapability.STREAM in b.capabilities
        assert BackendCapability.WRITE not in b.capabilities

    def test_capabilities_with_auth(self):
        auth = mock.MagicMock(spec=KerberosAuth)
        b = AsyncDPMHTTPBackend(host="localhost", port=6802, auth=auth)
        assert BackendCapability.WRITE in b.capabilities
        assert BackendCapability.AUTH_KERBEROS in b.capabilities

    @pytest.mark.asyncio
    async def test_closed_backend_raises(self, backend):
        await backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            await backend.read("M:OUTTMP")

    @pytest.mark.asyncio
    async def test_closed_backend_write_raises(self):
        auth = mock.MagicMock(spec=KerberosAuth)
        b = AsyncDPMHTTPBackend(host="localhost", port=6802, auth=auth)
        await b.close()
        with pytest.raises(RuntimeError, match="closed"):
            await b.write("M:OUTTMP", 72.5)

    @pytest.mark.asyncio
    async def test_closed_backend_subscribe_raises(self, backend):
        await backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            await backend.subscribe(["M:OUTTMP@p,1000"])

    def test_pool_size_zero_raises(self):
        with pytest.raises(ValueError, match="pool_size"):
            AsyncDPMHTTPBackend(host="localhost", port=6802, pool_size=0)

    @pytest.mark.asyncio
    async def test_subscribe_empty_drfs_raises(self, backend):
        with pytest.raises(ValueError, match="drfs cannot be empty"):
            await backend.subscribe([])
