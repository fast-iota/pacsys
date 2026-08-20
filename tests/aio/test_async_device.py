"""Tests for AsyncDevice."""

from unittest import mock

import pytest

from pacsys.aio._device import AsyncDevice
from pacsys.testing import AsyncFakeBackend
from pacsys.types import BasicControl


class TestAsyncDeviceRead:
    @pytest.mark.asyncio
    async def test_read(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP.READING", 72.5)

        device = AsyncDevice("M:OUTTMP", backend=fb)
        val = await device.read()
        assert val == 72.5
        assert fb.was_read("M:OUTTMP.READING")

    @pytest.mark.asyncio
    async def test_setting(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP.SETTING", 50.0)

        device = AsyncDevice("M:OUTTMP", backend=fb)
        val = await device.setting()
        assert val == 50.0
        assert fb.was_read("M:OUTTMP.SETTING")

    @pytest.mark.asyncio
    async def test_status_bool(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP.STATUS.ON", 1)

        device = AsyncDevice("M:OUTTMP", backend=fb)
        val = await device.status(field="on")
        assert val is True

    @pytest.mark.asyncio
    async def test_status_raw(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP.STATUS.RAW", 0xFF)

        device = AsyncDevice("M:OUTTMP", backend=fb)
        val = await device.status(field="raw")
        assert val == 0xFF

    @pytest.mark.asyncio
    async def test_get_with_invalid_prop_raises(self):
        device = AsyncDevice("M:OUTTMP", backend=AsyncFakeBackend())
        with pytest.raises(ValueError, match="Invalid property"):
            await device.get(prop="nonexistent")

    @pytest.mark.asyncio
    async def test_subscribe_with_invalid_prop_raises(self):
        device = AsyncDevice("M:OUTTMP", backend=AsyncFakeBackend())
        with pytest.raises(ValueError, match="Invalid property"):
            await device.subscribe(prop="nonexistent", event="p,1000")


class TestAsyncDeviceWrite:
    @pytest.mark.asyncio
    async def test_write(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP.SETTING", 72.5)

        device = AsyncDevice("M:OUTTMP", backend=fb)
        result = await device.write(72.5)
        assert result.success
        assert fb.was_written("M:OUTTMP.SETTING")

    @pytest.mark.asyncio
    async def test_write_with_verify(self):
        from pacsys.verify import Verify

        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP.SETTING", 72.5)
        fb.set_reading("M:OUTTMP.READING", 72.5)

        device = AsyncDevice("M:OUTTMP", backend=fb)
        v = Verify(initial_delay=0.0, retry_delay=0.0)
        result = await device.write(72.5, verify=v)
        assert result.verified

    @pytest.mark.asyncio
    async def test_write_rejects_basic_control(self):
        fb = AsyncFakeBackend()
        device = AsyncDevice("Z:ACLTST", backend=fb)
        with pytest.raises(TypeError, match="control\\(\\)"):
            await device.write(BasicControl.RESET)
        assert fb.writes == []

    @pytest.mark.asyncio
    async def test_control_on(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP.CONTROL", 0)

        device = AsyncDevice("M:OUTTMP", backend=fb)
        result = await device.on()
        assert result.success
        assert fb.was_written("M:OUTTMP.CONTROL")
        # Verify the command value was BasicControl.ON
        _, written_value = fb.writes[-1]
        assert written_value == BasicControl.ON


class TestAsyncDeviceFluent:
    def test_with_event(self):
        device = AsyncDevice("M:OUTTMP")
        d2 = device.with_event("p,1000")
        assert isinstance(d2, AsyncDevice)
        assert d2.is_periodic

    def test_with_range(self):
        device = AsyncDevice("B:HS23T")
        d2 = device.with_range(start=0, end=10)
        assert isinstance(d2, AsyncDevice)
        assert "[0:10]" in d2.drf

    def test_with_backend(self):
        backend1 = mock.AsyncMock()
        backend2 = mock.AsyncMock()

        device = AsyncDevice("M:OUTTMP", backend=backend1)
        d2 = device.with_backend(backend2)
        assert isinstance(d2, AsyncDevice)
        assert d2._backend is backend2

    @pytest.mark.parametrize(
        ("drf", "method", "args"),
        [
            ("M:OUTTMP.ANALOG.ALL", "with_event", ("P,1000",)),
            ("M:OUTTMP.ANALOG.ALL", "with_range", (0, 1)),
            ("M:OUTTMP.ANALOG[0:1].ALL", "without_range", ()),
            ("M:OUTTMP.ANALOG.ALL@P,1000", "without_event", ()),
            ("M:OUTTMP.ANALOG.ALL", "with_extra", ("FTP",)),
        ],
    )
    def test_fluent_modifiers_preserve_explicit_default_field(self, drf, method, args):
        modified = getattr(AsyncDevice(drf), method)(*args)
        assert modified.request.field_explicit

    @pytest.mark.asyncio
    async def test_fluent_modifier_preserves_default_field_behavior(self):
        backend = AsyncFakeBackend()
        backend.set_reading("M:OUTTMP.STATUS.ALL", 1)
        device = AsyncDevice("M:OUTTMP.ANALOG.ALL", backend=backend)
        await device.status()
        await device.with_event("P,1000").status()
        assert backend.reads == ["M:OUTTMP.STATUS.ALL@I", "M:OUTTMP.STATUS.ALL@I"]

    def test_equality(self):
        d1 = AsyncDevice("M:OUTTMP")
        d2 = AsyncDevice("M:OUTTMP")
        assert d1 == d2

    def test_hash(self):
        d1 = AsyncDevice("M:OUTTMP")
        d2 = AsyncDevice("M:OUTTMP")
        assert hash(d1) == hash(d2)
        assert len({d1, d2}) == 1
