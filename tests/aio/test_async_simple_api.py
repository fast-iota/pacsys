"""Tests for pacsys.aio module-level convenience API."""

import pytest
from unittest import mock

import pacsys.aio as aio
from pacsys.errors import ReadError
from pacsys.types import Reading, ValueType, WriteResult


@pytest.fixture(autouse=True)
def reset_aio_state():
    """Reset global state before/after each test."""
    aio._global_async_backend = None
    aio._async_backend_initialized = False
    aio._config_backend = None
    aio._config_auth = None
    aio._config_role = None
    aio._config_host = None
    aio._config_port = None
    aio._config_pool_size = None
    aio._config_timeout = None
    yield
    aio._global_async_backend = None
    aio._async_backend_initialized = False
    aio._config_backend = None
    aio._config_auth = None
    aio._config_role = None
    aio._config_host = None
    aio._config_port = None
    aio._config_pool_size = None
    aio._config_timeout = None


@pytest.fixture
def fake_backend():
    """Install a mock as the global async backend."""
    backend = mock.AsyncMock()
    backend.read = mock.AsyncMock(return_value=72.5)
    backend.get = mock.AsyncMock(
        return_value=Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5, error_code=0)
    )
    backend.get_many = mock.AsyncMock(
        return_value=[Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5, error_code=0)]
    )
    backend.write = mock.AsyncMock(return_value=WriteResult(drf="M:OUTTMP.SETTING@N"))
    backend.write_many = mock.AsyncMock(return_value=[WriteResult(drf="M:OUTTMP.SETTING@N")])
    backend.close = mock.AsyncMock()

    aio._global_async_backend = backend
    aio._async_backend_initialized = True
    return backend


class TestConfigure:
    def test_configure_stores_settings(self):
        aio.configure(backend="grpc", host="myhost", port=1234, timeout=10.0)
        assert aio._config_backend == "grpc"
        assert aio._config_host == "myhost"
        assert aio._config_port == 1234
        assert aio._config_timeout == 10.0

    def test_configure_invalid_backend(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            aio.configure(backend="nosql")

    def test_configure_after_init_auto_replaces(self, fake_backend):
        aio.configure(host="other")
        assert fake_backend._closed is True
        assert aio._global_async_backend is None
        assert aio._async_backend_initialized is False
        assert aio._config_host == "other"


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_backend(self, fake_backend):
        await aio.shutdown()
        fake_backend.close.assert_awaited_once()
        assert aio._global_async_backend is None
        assert aio._async_backend_initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_allows_reconfigure(self, fake_backend):
        await aio.shutdown()
        aio.configure(host="new-host")
        assert aio._config_host == "new-host"

    @pytest.mark.asyncio
    async def test_shutdown_multiple_safe(self):
        await aio.shutdown()
        await aio.shutdown()  # No error


class TestModuleAPI:
    @pytest.mark.asyncio
    async def test_read(self, fake_backend):
        val = await aio.read("M:OUTTMP")
        assert val == 72.5
        fake_backend.read.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get(self, fake_backend):
        reading = await aio.get("M:OUTTMP")
        assert reading.value == 72.5

    @pytest.mark.asyncio
    async def test_get_many(self, fake_backend):
        readings = await aio.get_many(["M:OUTTMP"])
        assert len(readings) == 1

    @pytest.mark.asyncio
    async def test_write(self, fake_backend):
        result = await aio.write("M:OUTTMP", 72.5)
        assert result.success

    @pytest.mark.asyncio
    async def test_write_many(self, fake_backend):
        results = await aio.write_many([("M:OUTTMP", 72.5)])
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_read_with_device(self, fake_backend):
        """Module-level read() accepts AsyncDevice."""
        from pacsys.aio._device import AsyncDevice

        device = AsyncDevice("M:OUTTMP")
        val = await aio.read(device)
        assert val == 72.5

    @pytest.mark.asyncio
    async def test_read_invalid_type(self, fake_backend):
        with pytest.raises(TypeError, match="Expected str"):
            await aio.read(12345)

    @pytest.mark.asyncio
    async def test_read_many(self, fake_backend):
        fake_backend.get_many = mock.AsyncMock(
            return_value=[
                Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5, error_code=0),
                Reading(drf="G:AMANDA", value_type=ValueType.SCALAR, value=1.0, error_code=0),
            ]
        )
        values = await aio.read_many(["M:OUTTMP", "G:AMANDA"])
        assert values == [72.5, 1.0]

    @pytest.mark.asyncio
    async def test_read_many_raises_on_device_error(self, fake_backend):
        fake_backend.get_many = mock.AsyncMock(
            return_value=[
                Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5, error_code=0),
                Reading(drf="M:BADDEV", error_code=-42, message="not found"),
            ]
        )
        with pytest.raises(ReadError) as exc_info:
            await aio.read_many(["M:OUTTMP", "M:BADDEV"])
        assert len(exc_info.value.readings) == 2
        assert exc_info.value.readings[0].ok
        assert exc_info.value.readings[1].is_error

    @pytest.mark.asyncio
    async def test_read_many_transport_error_passthrough(self, fake_backend):
        partial = [Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5)]
        transport_err = ReadError(partial, "connection lost")
        fake_backend.get_many = mock.AsyncMock(side_effect=transport_err)
        with pytest.raises(ReadError) as exc_info:
            await aio.read_many(["M:OUTTMP"])
        assert exc_info.value is transport_err


class TestLazyInit:
    @pytest.mark.asyncio
    async def test_lazy_creates_dpm_by_default(self):
        with mock.patch("pacsys.aio._dpm_http.AsyncDPMHTTPBackend") as MockDPM:
            mock_instance = mock.AsyncMock()
            mock_instance.read = mock.AsyncMock(return_value=72.5)
            MockDPM.return_value = mock_instance

            val = await aio.read("M:OUTTMP")

        assert val == 72.5
        MockDPM.assert_called_once()

    @pytest.mark.asyncio
    async def test_lazy_creates_grpc_when_configured(self):
        aio.configure(backend="grpc")

        with mock.patch("pacsys.aio._grpc.AsyncGRPCBackend") as MockGRPC:
            mock_instance = mock.AsyncMock()
            mock_instance.read = mock.AsyncMock(return_value=42.0)
            MockGRPC.return_value = mock_instance

            val = await aio.read("M:OUTTMP")

        assert val == 42.0

    @pytest.mark.asyncio
    async def test_backend_reused_on_subsequent_calls(self):
        with mock.patch("pacsys.aio._dpm_http.AsyncDPMHTTPBackend") as MockDPM:
            mock_instance = mock.AsyncMock()
            mock_instance.read = mock.AsyncMock(return_value=1.0)
            MockDPM.return_value = mock_instance

            await aio.read("M:OUTTMP")
            await aio.read("G:AMANDA")

        assert MockDPM.call_count == 1
