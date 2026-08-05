"""Tests for AsyncSubscriptionHandle."""

import asyncio

import pytest

from pacsys.types import Reading, ValueType


@pytest.fixture
def make_reading():
    def _make(val):
        return Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=val, error_code=0)

    return _make


class TestAsyncSubscriptionHandle:
    @pytest.mark.asyncio
    async def test_dispatch_and_iterate(self, make_reading):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        handle = AsyncSubscriptionHandle()
        handle._dispatch(make_reading(1.0))
        handle._dispatch(make_reading(2.0))
        handle._signal_stop()

        results = []
        async for reading, h in handle.readings():
            results.append(reading.value)
            assert h is handle
        assert results == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_error_propagation(self, make_reading):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        handle = AsyncSubscriptionHandle()
        handle._dispatch(make_reading(1.0))
        handle._signal_error(RuntimeError("boom"))

        results = []
        with pytest.raises(RuntimeError, match="boom"):  # noqa: PT012 -- verifies buffered values first
            async for reading, h in handle.readings():
                results.append(reading.value)
        assert results == [1.0]

    @pytest.mark.asyncio
    async def test_timeout(self):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        handle = AsyncSubscriptionHandle()
        with pytest.raises(asyncio.TimeoutError):
            async for _ in handle.readings(timeout=0.05):
                pass

    @pytest.mark.asyncio
    async def test_overflow_drops(self, make_reading):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        handle = AsyncSubscriptionHandle()
        handle._maxsize = 3
        handle._queue = asyncio.Queue(maxsize=3)
        for i in range(5):
            handle._dispatch(make_reading(float(i)))
        handle._signal_stop()

        results = []
        async for reading, _ in handle.readings():
            results.append(reading.value)
        assert len(results) == 3  # 2 dropped

    @pytest.mark.asyncio
    async def test_stopped_property(self):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        handle = AsyncSubscriptionHandle()
        assert handle.stopped is False
        handle._signal_stop()
        assert handle.stopped is True

    @pytest.mark.asyncio
    async def test_callback_feeder_async(self, make_reading):
        from pacsys.aio._subscription import AsyncSubscriptionHandle, _callback_feeder

        handle = AsyncSubscriptionHandle()
        collected = []

        async def cb(reading, h):
            collected.append(reading.value)

        task = asyncio.ensure_future(_callback_feeder(handle, cb, None))
        handle._dispatch(make_reading(10.0))
        handle._dispatch(make_reading(20.0))
        handle._signal_stop()
        await task
        assert collected == [10.0, 20.0]

    @pytest.mark.asyncio
    async def test_callback_feeder_sync(self, make_reading):
        from pacsys.aio._subscription import AsyncSubscriptionHandle, _callback_feeder

        handle = AsyncSubscriptionHandle()
        collected = []

        def cb(reading, h):
            collected.append(reading.value)

        task = asyncio.ensure_future(_callback_feeder(handle, cb, None))
        handle._dispatch(make_reading(5.0))
        handle._signal_stop()
        await task
        assert collected == [5.0]
