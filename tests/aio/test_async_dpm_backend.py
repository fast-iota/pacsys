"""Tests for AsyncDPMHTTPBackend."""

import asyncio
from unittest import mock

import pytest

from pacsys.acnet.errors import ERR_RETRY
from pacsys.aio._dpm_http import AsyncDPMHTTPBackend
from pacsys.auth import KerberosAuth
from pacsys.dpm_connection import DPMConnectionError
from pacsys.errors import AuthenticationError, DeviceError, ReadError
from pacsys.types import BackendCapability, Reading, ValueType, WriteResult


def _make_reading(drf="M:OUTTMP", val=72.5, error_code=0):
    return Reading(drf=drf, value_type=ValueType.SCALAR, value=val, error_code=error_code)


def _make_error_reading(drf="M:OUTTMP"):
    return Reading(drf=drf, value=None, error_code=-10, message="Bad")


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

    @pytest.mark.asyncio
    async def test_write_connect_failure_returns_write_results(self):
        auth = mock.MagicMock(spec=KerberosAuth)
        auth.principal = "test@FNAL.GOV"
        b = AsyncDPMHTTPBackend(host="localhost", port=6802, auth=auth)
        b._create_core = mock.AsyncMock(side_effect=DPMConnectionError("Connection refused"))

        results = await b.write_many([("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)])
        assert len(results) == 2
        for r in results:
            assert not r.success
            assert r.error_code == ERR_RETRY
            assert "Connection refused" in r.message

    @pytest.mark.asyncio
    async def test_write_many_prevalidates_before_connecting(self):
        auth = KerberosAuth(_lazy=True)
        backend = AsyncDPMHTTPBackend(host="localhost", port=6802, auth=auth)
        backend._create_core = mock.AsyncMock()

        with pytest.raises(TypeError, match="only strings"):
            await backend.write_many([("M:OUTTMP", ["on", 1])])

        backend._create_core.assert_not_awaited()


class TestAsyncDPMSubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_creates_dedicated_core(self, backend):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        async def fake_stream(drfs, dispatch, stop, error):
            dispatch(_make_reading())

        async def fake_create():
            core = _mock_core()
            core.stream = fake_stream
            return core

        backend._create_core = fake_create
        handle = await backend.subscribe(["M:OUTTMP@p,1000"])
        assert isinstance(handle, AsyncSubscriptionHandle)
        await asyncio.sleep(0.05)
        assert not handle._queue.empty()

    @pytest.mark.asyncio
    async def test_stream_death_closes_core_and_removes_handle(self, backend):
        # Stream error exit must close the dedicated core, stop the handle,
        # and drop it from _handles -- without an explicit handle.stop()
        err = RuntimeError("StartList failed")
        core = _mock_core()

        async def fake_stream(drfs, dispatch, stop, error):
            error(err)

        core.stream = fake_stream

        async def fake_create():
            return core

        backend._create_core = fake_create
        handle = await backend.subscribe(["M:OUTTMP@p,1000"])
        await handle._task
        core.close.assert_awaited()
        assert handle.stopped
        assert handle not in backend._handles
        with pytest.raises(RuntimeError, match="StartList failed"):
            async for _ in handle.readings(timeout=0.1):
                pass

    @pytest.mark.asyncio
    async def test_normal_stream_end_closes_core(self, backend):
        core = _mock_core()

        async def fake_stream(drfs, dispatch, stop, error):
            dispatch(_make_reading())

        core.stream = fake_stream

        async def fake_create():
            return core

        backend._create_core = fake_create
        handle = await backend.subscribe(["M:OUTTMP@p,1000"])
        await handle._task
        core.close.assert_awaited()
        assert handle.stopped
        assert handle not in backend._handles

    @pytest.mark.asyncio
    async def test_user_stop_still_safe_after_wrapper(self, backend):
        # Explicit stop() after stream end: remover's second close is a no-op
        core = _mock_core()

        async def fake_stream(drfs, dispatch, stop, error):
            dispatch(_make_reading())

        core.stream = fake_stream

        async def fake_create():
            return core

        backend._create_core = fake_create
        handle = await backend.subscribe(["M:OUTTMP@p,1000"])
        await handle._task
        await handle.stop()
        assert handle.stopped
        assert handle not in backend._handles


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


class TestAsyncDPMCloseRaces:
    """close() racing in-flight operations must not leak connected cores."""

    @pytest.mark.asyncio
    async def test_release_after_close_discards(self):
        b = AsyncDPMHTTPBackend(host="localhost", port=6802)
        b._pool_count = 1
        await b.close()
        core = _mock_core()
        await b._release_core(core)
        core.close.assert_awaited_once()
        assert b._pool.empty()
        assert b._pool_count == 0

    @pytest.mark.asyncio
    async def test_release_queue_full_decrements_count(self):
        b = AsyncDPMHTTPBackend(host="localhost", port=6802, pool_size=1)
        await b._pool.put(_mock_core())
        b._pool_count = 2
        extra = _mock_core()
        await b._release_core(extra)
        extra.close.assert_awaited_once()
        assert b._pool_count == 1

    @pytest.mark.asyncio
    async def test_get_many_racing_close_discards_core(self):
        b = AsyncDPMHTTPBackend(host="localhost", port=6802)
        core = _mock_core()
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_read(*args, **kwargs):
            started.set()
            await release.wait()
            return [_make_reading()]

        core.read_many = slow_read
        b._create_core = mock.AsyncMock(return_value=core)

        task = asyncio.create_task(b.get_many(["M:OUTTMP"]))
        await started.wait()
        await b.close()
        release.set()
        readings = await task
        assert readings[0].ok
        core.close.assert_awaited()
        assert b._pool.empty()
        assert b._pool_count == 0

    @pytest.mark.asyncio
    async def test_create_core_close_during_connect(self):
        core = _mock_core()
        connect_started = asyncio.Event()
        connect_release = asyncio.Event()

        async def slow_connect():
            connect_started.set()
            await connect_release.wait()

        core.connect = slow_connect
        with mock.patch("pacsys.aio._dpm_http._AsyncDpmCore", return_value=core):
            b = AsyncDPMHTTPBackend(host="localhost", port=6802)
            task = asyncio.create_task(b._borrow_core())
            await connect_started.wait()
            await b.close()
            connect_release.set()
            with pytest.raises(RuntimeError, match="Backend is closed"):
                await task
        core.close.assert_awaited_once()
        assert b._pool_count == 0

    @pytest.mark.asyncio
    async def test_subscribe_close_during_connect(self):
        core = _mock_core()
        connect_started = asyncio.Event()
        connect_release = asyncio.Event()

        async def slow_connect():
            connect_started.set()
            await connect_release.wait()

        core.connect = slow_connect
        with mock.patch("pacsys.aio._dpm_http._AsyncDpmCore", return_value=core):
            b = AsyncDPMHTTPBackend(host="localhost", port=6802)
            task = asyncio.create_task(b.subscribe(["M:OUTTMP@p,1000"]))
            await connect_started.wait()
            await b.close()
            connect_release.set()
            with pytest.raises(RuntimeError, match="Backend is closed"):
                await task
        core.close.assert_awaited_once()
        assert b._handles == []

    @pytest.mark.asyncio
    async def test_borrow_wait_path_reports_closed(self):
        b = AsyncDPMHTTPBackend(host="localhost", port=6802, pool_size=1, timeout=0.2)
        b._pool_count = 1  # one core checked out, pool queue empty
        task = asyncio.create_task(b._borrow_core())
        await asyncio.sleep(0.05)  # waiter parked in pool.get()
        await b.close()
        with pytest.raises(RuntimeError, match="Backend is closed"):
            await task

    @pytest.mark.asyncio
    async def test_pool_exhaustion_raises_read_error(self):
        b = AsyncDPMHTTPBackend(host="localhost", port=6802, pool_size=1, timeout=0.2)
        b._pool_count = 1  # one core checked out, pool queue empty
        with pytest.raises(ReadError, match="pool exhausted") as exc_info:
            await b.get_many(["M:OUTTMP", "G:AMANDA"])
        readings = exc_info.value.readings
        assert len(readings) == 2
        assert all(r.error_code == ERR_RETRY for r in readings)

    @pytest.mark.asyncio
    async def test_connect_failure_raises_read_error(self):
        with mock.patch(
            "pacsys.aio._dpm_http._AsyncDpmCore.connect",
            mock.AsyncMock(side_effect=DPMConnectionError("Connection refused")),
        ):
            b = AsyncDPMHTTPBackend(host="localhost", port=6802)
            with pytest.raises(ReadError, match="Connection refused") as exc_info:
                await b.get_many(["M:OUTTMP"])
            assert exc_info.value.readings[0].error_code == ERR_RETRY

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
