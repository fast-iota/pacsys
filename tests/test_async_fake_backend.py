"""Tests for AsyncFakeBackend."""

import asyncio
import pytest
from pacsys.testing import AsyncFakeBackend


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


class TestAsyncFakeBackendStreaming:
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

            asyncio.ensure_future(_emit())
            readings = []
            async for reading, _ in handle.readings(timeout=2.0):
                readings.append(reading)
            assert len(readings) == 2
            assert readings[0].value == pytest.approx(73.0)
            assert readings[1].value == pytest.approx(74.0)

        asyncio.run(_run())

    def test_close(self):
        async def _run():
            fb = AsyncFakeBackend()
            fb.set_reading("M:OUTTMP", 72.5)
            await fb.close()
            with pytest.raises(RuntimeError):
                await fb.get("M:OUTTMP")

        asyncio.run(_run())
