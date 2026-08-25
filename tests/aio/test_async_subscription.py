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
    async def test_timeout_is_total_window_and_returns_normally(self, make_reading):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        handle = AsyncSubscriptionHandle()

        async def produce():
            while True:
                handle._dispatch(make_reading(1.0))
                await asyncio.sleep(0.01)

        async def consume():
            return sum([1 async for _ in handle.readings(timeout=0.05)])

        producer = asyncio.create_task(produce())
        try:
            readings = await asyncio.wait_for(consume(), timeout=0.2)
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        assert readings > 0

    @pytest.mark.asyncio
    async def test_timeout_zero_drains_buffer(self, make_reading):
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        handle = AsyncSubscriptionHandle()
        handle._dispatch(make_reading(1.0))
        handle._dispatch(make_reading(2.0))

        readings = [reading.value async for reading, _ in handle.readings(timeout=0)]
        assert readings == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_callback_mode_cannot_iterate(self):
        from pacsys.aio._subscription import AsyncSubscriptionHandle, _callback_feeder

        handle = AsyncSubscriptionHandle()
        handle._callback_task = asyncio.create_task(_callback_feeder(handle, lambda reading, h: None, None))
        try:
            with pytest.raises(RuntimeError, match="Cannot iterate subscription with callback"):
                async for _ in handle.readings():
                    pass
        finally:
            await handle.stop()

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
    async def test_concurrent_stop_waits_for_task_completion(self):
        """A concurrent stop() must not return before the stream task unwinds
        (close() drains the pool right after)."""
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        handle = AsyncSubscriptionHandle()
        unwound = False

        async def slow_stream():
            nonlocal unwound
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.2)  # slow unwind
                unwound = True
                raise

        handle._task = asyncio.ensure_future(slow_stream())
        await asyncio.sleep(0)  # let it start

        await asyncio.gather(handle.stop(), handle.stop())
        assert unwound
        assert handle._task.done()

    @pytest.mark.asyncio
    async def test_stop_via_remover_does_not_deadlock(self):
        """stop() -> _remover -> backend.remove -> handle.stop() re-enters from
        the same task and must return immediately, not wait on itself."""
        from pacsys.aio._subscription import AsyncSubscriptionHandle

        async def remover(h):
            await h.stop()  # mirrors Backend.remove

        handle = AsyncSubscriptionHandle(remover=remover)
        await asyncio.wait_for(handle.stop(), timeout=1.0)

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
