"""Tests for AsyncDevice."""

import asyncio
from unittest import mock

import pytest

from pacsys.aio._device import AsyncArrayDevice, AsyncDevice, AsyncScalarDevice, AsyncTextDevice
from pacsys.errors import DeviceError
from pacsys.testing import AsyncFakeBackend
from pacsys.types import BasicControl, ValueType


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
    @pytest.mark.parametrize(
        "warning_drf",
        [
            "Z:ACLTST.STATUS.BIT_VALUE",
            "Z:ACLTST.STATUS.BIT_NAMES",
            "Z:ACLTST.STATUS.BIT_VALUES",
        ],
    )
    async def test_digital_status_rejects_warning_without_value(self, warning_drf):
        fb = AsyncFakeBackend()
        fb.set_reading("Z:ACLTST.STATUS.BIT_VALUE", 2)
        fb.set_reading("Z:ACLTST.STATUS.BIT_NAMES", ["On", "Ready"], value_type=ValueType.TEXT_ARRAY)
        fb.set_reading("Z:ACLTST.STATUS.BIT_VALUES", ["No", "Yes"], value_type=ValueType.TEXT_ARRAY)
        fb.set_error(warning_drf, 1, "DPM_PEND")

        with pytest.raises(DeviceError) as exc_info:
            await AsyncDevice("Z:ACLTST", backend=fb).digital_status()

        assert exc_info.value.error_code == 1

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
    async def test_verify_context_does_not_skip_sibling_task_write(self):
        from pacsys.verify import Verify

        backend = AsyncFakeBackend()
        backend.set_reading("M:VERIFY.SETTING", 123.0)
        context_entered = asyncio.Event()
        release_context = asyncio.Event()

        async def hold_context():
            with Verify(always=True, check_first=True, readback="M:VERIFY.SETTING@I"):
                context_entered.set()
                await release_context.wait()

        async def write_sibling():
            await context_entered.wait()
            try:
                return await AsyncDevice("M:TARGET", backend=backend).write(123.0)
            finally:
                release_context.set()

        _, result = await asyncio.gather(hold_context(), write_sibling())

        assert result.success
        assert not result.skipped
        assert backend.was_written("M:TARGET.SETTING")

    @pytest.mark.asyncio
    async def test_write_rejects_basic_control(self):
        fb = AsyncFakeBackend()
        device = AsyncDevice("Z:ACLTST", backend=fb)
        with pytest.raises(TypeError, match="control\\(\\)"):
            await device.write(BasicControl.RESET)
        assert fb.writes == []

    @pytest.mark.asyncio
    async def test_control_verify_extracts_basic_status_field(self):
        from pacsys.verify import Verify

        fb = AsyncFakeBackend()
        fb.set_reading(
            "Z:ACLTST.STATUS.ON",
            {"on": True, "ready": False},
            value_type=ValueType.BASIC_STATUS,
        )
        device = AsyncDevice("Z:ACLTST", backend=fb)

        result = await device.on(verify=Verify(initial_delay=0, retry_delay=0, max_attempts=1))

        assert result.verified is True
        assert result.readback is True

    @pytest.mark.asyncio
    async def test_control_text_readback_fails_without_boolean_coercion(self):
        from pacsys.verify import Verify

        fb = AsyncFakeBackend()
        fb.set_reading("Z:ACLTST.STATUS.ON", "False", value_type=ValueType.TEXT)
        device = AsyncDevice("Z:ACLTST", backend=fb)

        result = await device.on(verify=Verify(check_first=True, initial_delay=0, retry_delay=0, max_attempts=1))

        assert not result.skipped
        assert result.verified is False
        assert result.readback == "False"
        assert fb.was_written("Z:ACLTST.CONTROL")

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


class TestAsyncTypedDevices:
    @pytest.mark.asyncio
    async def test_scalar_array_text(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        fb.set_reading("B:IRMS06", [1.0, 2.0], value_type=ValueType.SCALAR_ARRAY)
        fb.set_reading("G:AMANDA.READING", "some text", value_type=ValueType.TEXT)
        assert await AsyncScalarDevice("M:OUTTMP", backend=fb).read() == 72.5
        assert (await AsyncArrayDevice("B:IRMS06", backend=fb).read()).tolist() == [1.0, 2.0]
        assert await AsyncTextDevice("G:AMANDA", backend=fb).read() == "some text"

    @pytest.mark.asyncio
    async def test_type_mismatch_raises(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", "text", value_type=ValueType.TEXT)
        with pytest.raises(TypeError, match="Expected scalar"):
            await AsyncScalarDevice("M:OUTTMP", backend=fb).read()

    def test_fluent_keeps_subclass(self):
        assert type(AsyncScalarDevice("M:OUTTMP").with_event("p,1000")) is AsyncScalarDevice


class TestAsyncAwaitNext:
    @pytest.mark.asyncio
    async def test_returns_next_reading_and_unsubscribes(self):
        fb = AsyncFakeBackend()
        dev = AsyncDevice("M:OUTTMP@p,1000", backend=fb)
        asyncio.get_running_loop().call_later(0.02, fb.emit_reading, "M:OUTTMP.READING@p,1000", 99.0)
        reading = await dev.await_next(timeout=2.0)
        assert reading.value == 99.0
        assert not fb._sync._subscriptions

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        dev = AsyncDevice("M:OUTTMP", backend=AsyncFakeBackend())
        with pytest.raises(TimeoutError):
            await dev.await_next(event="p,1000", timeout=0.05)

    @pytest.mark.asyncio
    async def test_requires_event(self):
        dev = AsyncDevice("M:OUTTMP", backend=AsyncFakeBackend())
        with pytest.raises(ValueError, match="await_next requires an event"):
            await dev.await_next()
        with pytest.raises(ValueError, match="cannot use @N"):
            await dev.await_next(event="N")


class TestAsyncControlVerify:
    """Control verification shares the sync plan: STATUS readback, check_first skip, field extraction."""

    @pytest.mark.asyncio
    async def test_control_with_verify_reads_status(self):
        from pacsys.verify import Verify

        fb = AsyncFakeBackend()
        fb.set_reading("Z:ACLTST.STATUS.ON", True)
        result = await AsyncDevice("Z:ACLTST", backend=fb).on(verify=Verify(initial_delay=0, retry_delay=0))
        assert result.verified is True
        assert fb.was_written("Z:ACLTST.CONTROL")

    @pytest.mark.asyncio
    async def test_control_check_first_skips(self):
        from pacsys.verify import Verify

        fb = AsyncFakeBackend()
        fb.set_reading("Z:ACLTST.STATUS.ON", True)
        result = await AsyncDevice("Z:ACLTST", backend=fb).on(verify=Verify(check_first=True, initial_delay=0))
        assert result.skipped is True
        assert not fb.was_written("Z:ACLTST.CONTROL")

    @pytest.mark.asyncio
    async def test_control_verify_extracts_basic_status_field(self):
        from pacsys.verify import Verify

        fb = AsyncFakeBackend()
        fb.set_reading("Z:ACLTST.STATUS.ON", {"on": True, "ready": False}, value_type=ValueType.BASIC_STATUS)
        result = await AsyncDevice("Z:ACLTST", backend=fb).on(
            verify=Verify(initial_delay=0, retry_delay=0, max_attempts=1)
        )
        assert result.verified is True and result.readback is True
