"""
Integration tests for AsyncDevice API against real DPM/HTTP backend.

Tests the AsyncDevice-centric interface: read(), setting(), status(), description(),
get(), write(), control(), and fluent modifiers (with_backend, with_event, with_range).

Mirrors test_device_api.py for async code.

Run with: pytest tests/real/test_async_device_api.py -v -s -o "addopts="
"""

import asyncio

import pytest

from pacsys.aio import AsyncDevice
from pacsys.digital_status import DigitalStatus, StatusBit
from pacsys.errors import DeviceError
from pacsys.types import Reading, ValueType
from pacsys.verify import Verify

from .devices import (
    ANALOG_ALARM_SETPOINT,
    ARRAY_DEVICE,
    NONEXISTENT_DEVICE,
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_DEVICE_3,
    SCALAR_ELEMENT,
    STATUS_DEVICE,
    requires_dpm_http,
    requires_kerberos,
    requires_write_enabled,
    TIMEOUT_READ,
)


async def _create_async_dpm_write_backend():
    from pacsys.auth import KerberosAuth
    from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

    return AsyncDPMHTTPBackend(auth=KerberosAuth(), role="testing")


# =============================================================================
# Read Tests
# =============================================================================


@requires_dpm_http
class TestAsyncDeviceRead:
    """Read operations via AsyncDevice API."""

    @pytest.mark.asyncio
    async def test_read_scalar(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE, backend=async_dpm_http_backend)
        value = await dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    @pytest.mark.asyncio
    async def test_read_scalar_second_device(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE_2, backend=async_dpm_http_backend)
        value = await dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    @pytest.mark.asyncio
    async def test_read_array(self, async_dpm_http_backend):
        dev = AsyncDevice(ARRAY_DEVICE, backend=async_dpm_http_backend)
        value = await dev.read(timeout=TIMEOUT_READ)
        assert hasattr(value, "__len__")
        assert len(value) == 11

    @pytest.mark.asyncio
    async def test_read_single_element(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_ELEMENT, backend=async_dpm_http_backend)
        value = await dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    @pytest.mark.asyncio
    async def test_read_raw(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE, backend=async_dpm_http_backend)
        value = await dev.read(field="raw", timeout=TIMEOUT_READ)
        assert isinstance(value, bytes)
        assert len(value) > 0

    @pytest.mark.asyncio
    async def test_read_description(self, async_dpm_http_backend):
        dev = AsyncDevice("M:OUTTMP", backend=async_dpm_http_backend)
        desc = await dev.description(timeout=TIMEOUT_READ)
        assert isinstance(desc, str)
        assert len(desc) > 0

    @pytest.mark.asyncio
    async def test_read_status(self, async_dpm_http_backend):
        dev = AsyncDevice(STATUS_DEVICE, backend=async_dpm_http_backend)
        value = await dev.status(timeout=TIMEOUT_READ)
        assert isinstance(value, dict)
        assert "on" in value

    @pytest.mark.asyncio
    async def test_read_status_field(self, async_dpm_http_backend):
        """status(field='on') returns a bool."""
        dev = AsyncDevice(SCALAR_DEVICE_3, backend=async_dpm_http_backend)
        on_val = await dev.status(field="on", timeout=TIMEOUT_READ)
        assert isinstance(on_val, bool)

    @pytest.mark.asyncio
    async def test_read_analog_alarm(self, async_dpm_http_backend):
        dev = AsyncDevice("N:H801", backend=async_dpm_http_backend)
        value = await dev.analog_alarm(timeout=TIMEOUT_READ)
        assert isinstance(value, dict)
        assert "minimum" in value
        assert "maximum" in value

    @pytest.mark.asyncio
    async def test_read_digital_alarm(self, async_dpm_http_backend):
        dev = AsyncDevice("N:H801", backend=async_dpm_http_backend)
        value = await dev.digital_alarm(timeout=TIMEOUT_READ)
        assert isinstance(value, dict)


# =============================================================================
# Setting Tests
# =============================================================================


@requires_dpm_http
class TestAsyncDeviceSetting:
    """Read SETTING property via AsyncDevice API."""

    @pytest.mark.asyncio
    async def test_read_setting(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE_3, backend=async_dpm_http_backend)
        value = await dev.setting(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    @pytest.mark.asyncio
    async def test_read_setting_raw(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE_3, backend=async_dpm_http_backend)
        value = await dev.setting(field="raw", timeout=TIMEOUT_READ)
        assert isinstance(value, bytes)
        assert len(value) > 0


# =============================================================================
# digital_status() Tests
# =============================================================================


@requires_dpm_http
class TestAsyncDeviceDigitalStatus:
    """digital_status() fetches BIT_VALUE/BIT_NAMES/BIT_VALUES."""

    @pytest.mark.asyncio
    async def test_returns_digital_status(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE_3, backend=async_dpm_http_backend)
        status = await dev.digital_status(timeout=TIMEOUT_READ)
        assert isinstance(status, DigitalStatus)
        assert status.device == "Z:ACLTST"
        assert isinstance(status.raw_value, int)
        assert len(status.bits) > 0
        assert all(isinstance(b, StatusBit) for b in status.bits)

    @pytest.mark.asyncio
    async def test_legacy_attributes(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE_3, backend=async_dpm_http_backend)
        status = await dev.digital_status(timeout=TIMEOUT_READ)
        assert isinstance(status.on, bool)
        assert isinstance(status.ready, bool)

    @pytest.mark.asyncio
    async def test_bit_lookup(self, async_dpm_http_backend):
        """Bits can be looked up by name."""
        dev = AsyncDevice(SCALAR_DEVICE_3, backend=async_dpm_http_backend)
        status = await dev.digital_status(timeout=TIMEOUT_READ)
        bit = status["On"]
        assert bit is not None
        assert isinstance(bit, StatusBit)
        assert bit.name.lower() == "on"


# =============================================================================
# get() Tests
# =============================================================================


@requires_dpm_http
class TestAsyncDeviceGet:
    """get() returns full Reading."""

    @pytest.mark.asyncio
    async def test_get_returns_reading(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE, backend=async_dpm_http_backend)
        reading = await dev.get(timeout=TIMEOUT_READ)
        assert isinstance(reading, Reading)
        assert reading.ok
        assert reading.value is not None
        assert reading.value_type == ValueType.SCALAR

    @pytest.mark.asyncio
    async def test_get_has_metadata(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE, backend=async_dpm_http_backend)
        reading = await dev.get(timeout=TIMEOUT_READ)
        assert reading.ok
        assert reading.name is not None
        assert reading.timestamp is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


@requires_dpm_http
class TestAsyncDeviceErrors:
    """Error handling via AsyncDevice API."""

    @pytest.mark.asyncio
    async def test_read_nonexistent_raises(self, async_dpm_http_backend):
        dev = AsyncDevice(NONEXISTENT_DEVICE, backend=async_dpm_http_backend)
        with pytest.raises(DeviceError):
            await dev.read(timeout=TIMEOUT_READ)

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_error(self, async_dpm_http_backend):
        dev = AsyncDevice(NONEXISTENT_DEVICE, backend=async_dpm_http_backend)
        reading = await dev.get(timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code != 0

    @pytest.mark.asyncio
    async def test_invalid_field_raises_valueerror(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE, backend=async_dpm_http_backend)
        with pytest.raises(ValueError, match="not allowed"):
            await dev.read(field="on")

    @pytest.mark.asyncio
    async def test_status_invalid_field_raises(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE_3, backend=async_dpm_http_backend)
        with pytest.raises(ValueError, match="not allowed"):
            await dev.status(field="scaled")


# =============================================================================
# Fluent Modifier Tests
# =============================================================================


@requires_dpm_http
class TestAsyncDeviceFluent:
    """with_backend(), with_event(), with_range() return new AsyncDevice."""

    @pytest.mark.asyncio
    async def test_with_backend(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE)
        bound = dev.with_backend(async_dpm_http_backend)
        assert bound is not dev
        value = await bound.read(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    @pytest.mark.asyncio
    async def test_with_event(self, async_dpm_http_backend):
        dev = AsyncDevice(SCALAR_DEVICE, backend=async_dpm_http_backend)
        immediate = dev.with_event("I")
        assert immediate.has_event
        value = await immediate.read(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    @pytest.mark.asyncio
    async def test_with_range(self, async_dpm_http_backend):
        dev = AsyncDevice("B:IRMS06", backend=async_dpm_http_backend)
        ranged = dev.with_range(0, 5)
        assert "[0:5]" in ranged.drf
        value = await ranged.read(timeout=TIMEOUT_READ)
        assert hasattr(value, "__len__")
        assert len(value) == 6


# =============================================================================
# Write Tests (requires Kerberos + PACSYS_TEST_WRITE=1)
# =============================================================================


@requires_dpm_http
@requires_kerberos
@pytest.mark.kerberos
class TestAsyncDeviceWrite:
    """AsyncDevice.write() operations."""

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_write_scalar(self):
        """Write a different value, verify readback, restore."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice(SCALAR_DEVICE_3, backend=backend)
            original = await dev.setting(timeout=TIMEOUT_READ)

            new_value = original + 0.1
            result = await dev.write(new_value, timeout=TIMEOUT_READ)
            assert result.success

            await asyncio.sleep(1.0)
            readback = await dev.setting(timeout=TIMEOUT_READ)
            assert abs(readback - new_value) < 0.01, f"Wrote {new_value}, read back {readback}"

            # Restore
            result2 = await dev.write(original, timeout=TIMEOUT_READ)
            assert result2.success
        finally:
            await backend.close()

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_write_with_verify(self):
        """AsyncDevice.write(verify=Verify(...)) reads back the value."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice(SCALAR_DEVICE_3, backend=backend)
            original = await dev.setting(timeout=TIMEOUT_READ)

            new_value = original + 0.1
            result = await dev.write(new_value, verify=Verify(tolerance=0.01), timeout=TIMEOUT_READ)
            assert result.success
            assert result.verified is True
            assert result.readback is not None
            assert abs(result.readback - new_value) < 0.01

            # Restore
            await dev.write(original, timeout=TIMEOUT_READ)
        finally:
            await backend.close()

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_write_verify_check_first(self):
        """Verify(check_first=True) skips write when value already matches."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice(SCALAR_DEVICE_3, backend=backend)
            current = await dev.setting(timeout=TIMEOUT_READ)

            result = await dev.write(current, verify=Verify(check_first=True, tolerance=0.01), timeout=TIMEOUT_READ)
            assert result.success
            assert result.skipped is True
            assert result.verified is True
        finally:
            await backend.close()

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_write_raw(self):
        """Write raw bytes via field='raw'."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice(SCALAR_DEVICE_3, backend=backend)
            original_raw = await dev.setting(field="raw", timeout=TIMEOUT_READ)
            assert isinstance(original_raw, bytes)

            # DEC F_float for 45.0
            raw_45 = b"\x34\x43\x00\x00"
            result = await dev.write(raw_45, field="raw", timeout=TIMEOUT_READ)
            assert result.success

            await asyncio.sleep(1.0)
            readback_raw = await dev.setting(field="raw", timeout=TIMEOUT_READ)
            assert readback_raw == raw_45
            readback_scaled = await dev.setting(timeout=TIMEOUT_READ)
            assert readback_scaled == 45.0

            # Restore
            await dev.write(original_raw, field="raw", timeout=TIMEOUT_READ)
        finally:
            await backend.close()

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_control_on_off(self):
        """dev.on() / dev.off() toggle the on status bit."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice(SCALAR_DEVICE_3, backend=backend)
            initial_on = await dev.status(field="on", timeout=TIMEOUT_READ)

            result = await dev.on(timeout=TIMEOUT_READ)
            assert result.success
            await asyncio.sleep(1.0)
            assert await dev.status(field="on", timeout=TIMEOUT_READ) is True

            result = await dev.off(timeout=TIMEOUT_READ)
            assert result.success
            await asyncio.sleep(1.0)
            assert await dev.status(field="on", timeout=TIMEOUT_READ) is False

            # Restore
            if initial_on:
                await dev.on(timeout=TIMEOUT_READ)
        finally:
            await backend.close()

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_positive_negative(self):
        """dev.positive() / dev.negative() toggle the positive status bit."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice(SCALAR_DEVICE_3, backend=backend)
            initial_positive = await dev.status(field="positive", timeout=TIMEOUT_READ)

            result = await dev.positive(timeout=TIMEOUT_READ)
            assert result.success
            await asyncio.sleep(1.0)
            assert await dev.status(field="positive", timeout=TIMEOUT_READ) is True

            result = await dev.negative(timeout=TIMEOUT_READ)
            assert result.success
            await asyncio.sleep(1.0)
            assert await dev.status(field="positive", timeout=TIMEOUT_READ) is False

            # Restore
            if initial_positive:
                await dev.positive(timeout=TIMEOUT_READ)
            else:
                await dev.negative(timeout=TIMEOUT_READ)
        finally:
            await backend.close()

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_ramp_dc(self):
        """dev.ramp() / dev.dc() toggle the ramp status bit."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice(SCALAR_DEVICE_3, backend=backend)
            initial_ramp = await dev.status(field="ramp", timeout=TIMEOUT_READ)

            result = await dev.ramp(timeout=TIMEOUT_READ)
            assert result.success
            await asyncio.sleep(1.0)
            assert await dev.status(field="ramp", timeout=TIMEOUT_READ) is True

            result = await dev.dc(timeout=TIMEOUT_READ)
            assert result.success
            await asyncio.sleep(1.0)
            assert await dev.status(field="ramp", timeout=TIMEOUT_READ) is False

            # Restore
            if initial_ramp:
                await dev.ramp(timeout=TIMEOUT_READ)
            else:
                await dev.dc(timeout=TIMEOUT_READ)
        finally:
            await backend.close()

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_reset(self):
        """dev.reset() succeeds and status has on/ready."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice(SCALAR_DEVICE_3, backend=backend)
            result = await dev.reset(timeout=TIMEOUT_READ)
            assert result.success

            await asyncio.sleep(1.0)
            status = await dev.status(timeout=TIMEOUT_READ)
            assert isinstance(status, dict)
            assert "on" in status
        finally:
            await backend.close()

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_control_with_verify(self):
        """dev.on(verify=True) verifies STATUS.ON is True after write."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice(SCALAR_DEVICE_3, backend=backend)
            initial_on = await dev.status(field="on", timeout=TIMEOUT_READ)

            result = await dev.on(verify=True, timeout=TIMEOUT_READ)
            assert result.success
            assert result.verified is True
            assert result.readback is True

            # Restore
            if not initial_on:
                await dev.off(timeout=TIMEOUT_READ)
        finally:
            await backend.close()


# =============================================================================
# Alarm Write Tests
# =============================================================================


@requires_dpm_http
@requires_kerberos
@pytest.mark.kerberos
class TestAsyncDeviceAlarmWrite:
    """Alarm write operations via AsyncDevice API."""

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_write_analog_alarm_max(self):
        """Write analog alarm MAX via backend, verify via device.analog_alarm()."""
        backend = await _create_async_dpm_write_backend()
        try:
            dev = AsyncDevice("Z:ACLTST", backend=backend)
            alarm = await dev.analog_alarm(timeout=TIMEOUT_READ)
            assert isinstance(alarm, dict)
            orig_max = alarm["maximum"]

            new_max = orig_max + 0.5
            result = await backend.write(f"{ANALOG_ALARM_SETPOINT}.MAX", new_max, timeout=TIMEOUT_READ)
            assert result.success

            await asyncio.sleep(1.0)
            after = await dev.analog_alarm(timeout=TIMEOUT_READ)
            assert after["maximum"] == new_max

            # Restore
            await backend.write(f"{ANALOG_ALARM_SETPOINT}.MAX", orig_max, timeout=TIMEOUT_READ)
        finally:
            await backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
