"""
Integration tests for pacsys.aio async Simple API.

Tests the complete path: aio.read() -> global async backend -> AsyncDPMHTTPBackend

Mirrors test_simple_api.py for async code.

Run with: pytest tests/real/test_async_simple_api.py -v -s -o "addopts="
"""

import asyncio

import pytest
import pytest_asyncio

import pacsys.aio as aio
from pacsys.aio import AsyncDevice
from pacsys.errors import DeviceError
from pacsys.types import Reading

from .devices import (
    ARRAY_DEVICE,
    NONEXISTENT_DEVICE,
    PERIODIC_DEVICE,
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_ELEMENT,
    requires_dpm_http,
    requires_grpc,
    TIMEOUT_BATCH,
    TIMEOUT_READ,
    TIMEOUT_STREAM_EVENT,
    TIMEOUT_STREAM_ITER,
)


@pytest_asyncio.fixture(autouse=True)
async def _reset_aio():
    """Reset async global backend before and after each test."""
    aio._global_async_backend = None
    aio._async_backend_initialized = False
    aio._config_backend = None
    aio._config_host = None
    aio._config_port = None
    aio._config_pool_size = None
    aio._config_timeout = None
    aio._config_auth = None
    aio._config_role = None
    yield
    await aio.shutdown()


# =============================================================================
# Simple API Read Tests
# =============================================================================


@requires_dpm_http
class TestAsyncSimpleAPIRead:
    """Tests for aio.read() - simplest async API."""

    @pytest.mark.asyncio
    async def test_read_scalar_device(self):
        value = await aio.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    @pytest.mark.asyncio
    async def test_read_array_device(self):
        value = await aio.read(ARRAY_DEVICE, timeout=TIMEOUT_READ)
        assert hasattr(value, "__len__")
        assert len(value) == 11

    @pytest.mark.asyncio
    async def test_read_with_device_object(self):
        device = AsyncDevice(SCALAR_DEVICE)
        value = await aio.read(device, timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    @pytest.mark.asyncio
    async def test_read_nonexistent_raises(self):
        with pytest.raises(DeviceError) as exc_info:
            await aio.read(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert exc_info.value.error_code != 0


# =============================================================================
# Simple API Get Tests
# =============================================================================


@requires_dpm_http
class TestAsyncSimpleAPIGet:
    """Tests for aio.get() - returns Reading with metadata."""

    @pytest.mark.asyncio
    async def test_get_returns_reading(self):
        reading = await aio.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)

        assert isinstance(reading, Reading)
        assert reading.ok
        assert reading.value is not None

    @pytest.mark.asyncio
    async def test_get_with_device_object(self):
        device = AsyncDevice(SCALAR_DEVICE)
        reading = await aio.get(device, timeout=TIMEOUT_READ)
        assert reading.ok
        assert isinstance(reading.value, (int, float))

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_error_reading(self):
        reading = await aio.get(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code != 0


# =============================================================================
# Simple API Get Many Tests
# =============================================================================


@requires_dpm_http
class TestAsyncSimpleAPIGetMany:
    """Tests for aio.get_many() - batch reads."""

    @pytest.mark.asyncio
    async def test_get_many_multiple_devices(self):
        devices = [SCALAR_DEVICE, SCALAR_ELEMENT, SCALAR_DEVICE_2]
        readings = await aio.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 3

    @pytest.mark.asyncio
    async def test_get_many_mixed_device_types(self):
        devices = [SCALAR_DEVICE, AsyncDevice(SCALAR_ELEMENT), AsyncDevice(SCALAR_DEVICE_2)]
        readings = await aio.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 3
        assert all(r.ok for r in readings)

    @pytest.mark.asyncio
    async def test_get_many_partial_failure(self):
        devices = [SCALAR_DEVICE, NONEXISTENT_DEVICE]
        readings = await aio.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 2
        assert readings[0].ok
        assert not readings[1].ok


# =============================================================================
# Configuration Tests
# =============================================================================


@requires_dpm_http
class TestAsyncConfiguration:
    """Tests for aio.configure() and lifecycle."""

    @pytest.mark.asyncio
    async def test_configure_before_use(self):
        aio.configure(timeout=TIMEOUT_BATCH)
        value = await aio.read(SCALAR_DEVICE)
        assert value is not None

    @pytest.mark.asyncio
    async def test_shutdown_allows_reconfigure(self):
        await aio.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        await aio.shutdown()
        aio.configure(timeout=TIMEOUT_BATCH)
        value = await aio.read(SCALAR_DEVICE)
        assert value is not None


# =============================================================================
# Backend Factory Tests
# =============================================================================


@requires_dpm_http
class TestAsyncBackendFactories:
    """Tests for async backend factory functions."""

    @pytest.mark.asyncio
    async def test_dpm_factory(self):
        async with aio.dpm() as backend:
            value = await backend.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
            assert isinstance(value, (int, float))

    @requires_grpc
    @pytest.mark.asyncio
    async def test_grpc_factory(self):
        async with aio.grpc() as backend:
            value = await backend.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
            assert isinstance(value, (int, float))


# =============================================================================
# Global Backend Lifecycle Tests
# =============================================================================


@requires_dpm_http
class TestAsyncGlobalBackendLifecycle:
    """Tests for global async backend initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_lazy_initialization(self):
        assert aio._global_async_backend is None

        value = await aio.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert value is not None
        assert aio._global_async_backend is not None

    @pytest.mark.asyncio
    async def test_backend_reused_across_calls(self):
        await aio.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        backend1 = aio._global_async_backend

        await aio.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        backend2 = aio._global_async_backend

        assert backend1 is backend2

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self):
        await aio.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert aio._global_async_backend is not None

        await aio.shutdown()
        assert aio._global_async_backend is None

        value = await aio.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert value is not None


# =============================================================================
# Module-Level Streaming Tests
# =============================================================================


@requires_dpm_http
@pytest.mark.streaming
class TestAsyncModuleLevelStreaming:
    """Tests for aio.subscribe() module-level function."""

    @pytest.mark.asyncio
    async def test_subscribe_callback_mode(self):
        readings: list[Reading] = []
        event = asyncio.Event()

        def on_reading(reading, handle):
            readings.append(reading)
            if len(readings) >= 1:
                event.set()

        handle = await aio.subscribe([PERIODIC_DEVICE], callback=on_reading)
        try:
            await asyncio.wait_for(event.wait(), timeout=TIMEOUT_STREAM_EVENT)
        finally:
            await handle.stop()

        assert len(readings) >= 1

    @pytest.mark.asyncio
    async def test_subscribe_iterator_mode(self):
        handle = await aio.subscribe([PERIODIC_DEVICE])
        async with handle:
            count = 0
            async for reading, _ in handle.readings(timeout=TIMEOUT_STREAM_ITER):
                count += 1
                if count >= 2:
                    break

        assert count >= 2


# =============================================================================
# Concurrency Tests
# =============================================================================


@requires_dpm_http
class TestAsyncConcurrency:
    """Tests for concurrent async operations."""

    @pytest.mark.asyncio
    async def test_concurrent_reads_via_gather(self):
        results = await asyncio.gather(*[aio.read(SCALAR_DEVICE, timeout=TIMEOUT_READ) for _ in range(5)])

        assert len(results) == 5
        assert all(isinstance(v, (int, float)) for v in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
