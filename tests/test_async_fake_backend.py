"""Tests for AsyncFakeBackend."""

import asyncio

import pytest

from pacsys.testing import AsyncFakeBackend
from pacsys.types import ValueType


class TestAsyncFakeBackendRead:
    def test_read_value(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        reading = asyncio.run(fb.get("M:OUTTMP"))
        assert reading.ok
        assert reading.value == pytest.approx(72.5)

    def test_read_error(self):
        fb = AsyncFakeBackend()
        fb.set_error("M:BADDEV", -42, "Device not found")
        reading = asyncio.run(fb.get("M:BADDEV"))
        assert reading.is_error
        assert reading.error_code == -42

    def test_get_many(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        fb.set_reading("G:AMANDA", 42.0)
        readings = asyncio.run(fb.get_many(["M:OUTTMP", "G:AMANDA"]))
        assert len(readings) == 2
        assert readings[0].value == pytest.approx(72.5)
        assert readings[1].value == pytest.approx(42.0)


class TestAsyncFakeBackendWrite:
    def test_write(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        result = asyncio.run(fb.write("M:OUTTMP", 80.0))
        assert result.success
        assert fb.was_written("M:OUTTMP")

    def test_write_many(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        fb.set_reading("G:AMANDA", 42.0)
        results = asyncio.run(fb.write_many([("M:OUTTMP", 80.0), ("G:AMANDA", 50.0)]))
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_partial_ranged_write_needs_backing_value(self):
        fb = AsyncFakeBackend()
        drf = "B:HS23T.SETTING{4:2}.RAW@N"
        assert not asyncio.run(fb.write(drf, b"xy")).success
        fb.set_reading("B:HS23T.SETTING.RAW", b"\x00" * 8, value_type=ValueType.RAW)
        results = asyncio.run(fb.write_many([(drf, b"xy")]))
        assert results[0].success
        assert asyncio.run(fb.read("B:HS23T.SETTING{4:2}.RAW@I")) == b"xy"


class TestAsyncFakeBackendStreaming:
    @pytest.mark.parametrize("kwargs", [{"callback": "bad"}, {"on_error": "bad"}])
    def test_subscribe_validates_callbacks_before_creation(self, kwargs):
        async def _run():
            fb = AsyncFakeBackend()
            with pytest.raises(TypeError, match="must be callable"):
                await fb.subscribe(["M:OUTTMP"], **kwargs)
            assert fb._handles == []

        asyncio.run(_run())

    def test_subscribe_rejects_empty_drfs(self):
        async def _run():
            fb = AsyncFakeBackend()
            with pytest.raises(ValueError, match="drfs cannot be empty"):
                await fb.subscribe([])
            assert fb._handles == []

        asyncio.run(_run())

    def test_subscribe_and_emit(self):
        async def _run():
            fb = AsyncFakeBackend()
            fb.set_reading("M:OUTTMP", 72.5)
            handle = await fb.subscribe(["M:OUTTMP"])

            async def _emit():
                await asyncio.sleep(0.05)
                fb.emit_reading("M:OUTTMP", 73.0)
                fb.emit_reading("M:OUTTMP", 74.0)
                await asyncio.sleep(0.05)
                await handle.stop()

            emit_task = asyncio.create_task(_emit())
            readings = []
            async for reading, _ in handle.readings(timeout=2.0):
                readings.append(reading)
            await emit_task
            assert len(readings) == 2
            assert readings[0].value == pytest.approx(73.0)
            assert readings[1].value == pytest.approx(74.0)

        asyncio.run(_run())

    @pytest.mark.parametrize("async_callback", [False, True], ids=["sync", "async"])
    def test_iterator_delivers_error_callback(self, async_callback):
        async def _run():
            async with AsyncFakeBackend() as fb:
                errors = []

                def on_error(exc, handle):
                    errors.append((exc, handle))

                async def async_on_error(exc, handle):
                    on_error(exc, handle)

                handle = await fb.subscribe(["M:OUTTMP"], on_error=async_on_error if async_callback else on_error)
                error = ConnectionError("Simulated disconnect")
                fb.emit_error(error)
                fb.emit_error(ConnectionError("Repeated disconnect"))

                with pytest.raises(ConnectionError) as caught:
                    await anext(handle.readings(timeout=0.1))
                assert caught.value is error
                await handle.stop()
                assert errors == [(error, handle)]

        asyncio.run(_run())

    def test_callback_mode(self):
        async def _run():
            fb = AsyncFakeBackend()
            readings = []
            errors = []
            handle = await fb.subscribe(
                ["M:OUTTMP"],
                callback=lambda reading, _handle: readings.append(reading.value),
                on_error=lambda exc, _handle: errors.append(exc),
            )

            with pytest.raises(RuntimeError, match="Cannot iterate subscription with callback"):
                await anext(handle.readings(timeout=0))

            fb.emit_reading("M:OUTTMP", 73.0)
            error = ConnectionError("Simulated disconnect")
            fb.emit_error(error)
            assert handle._callback_task is not None
            await asyncio.wait_for(handle._callback_task, timeout=0.1)

            assert readings == [73.0]
            assert errors == [error]
            await fb.close()

        asyncio.run(_run())

    def test_close(self):
        async def _run():
            fb = AsyncFakeBackend()
            fb.set_reading("M:OUTTMP", 72.5)
            await fb.close()
            with pytest.raises(RuntimeError):
                await fb.get("M:OUTTMP")

        asyncio.run(_run())

    def test_close_stops_all_handles(self):
        """close() must stop every handle; remove() mutates _handles mid-loop."""

        async def _run():
            fb = AsyncFakeBackend()
            fb.set_reading("M:OUTTMP", 72.5)
            handles = [await fb.subscribe(["M:OUTTMP"]) for _ in range(4)]
            await fb.close()
            assert [h.stopped for h in handles] == [True] * 4
            assert fb._handles == []
            assert fb._sync_handles == []

        asyncio.run(_run())
