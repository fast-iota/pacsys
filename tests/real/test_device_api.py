"""
Integration tests for Device API against real DPM/HTTP backend.

Tests the Device-centric interface: read(), setting(), status(), description(),
analog_alarm(), digital_alarm(), digital_status(), write(), control(), and
fluent modifiers (with_backend, with_event, with_range).

Run with: pytest tests/real/test_device_api.py -v -s
"""

import time

import numpy as np
import pytest

from pacsys.device import Device, ScalarDevice, ArrayDevice, TextDevice
from pacsys.digital_status import DigitalStatus, StatusBit
from pacsys.errors import DeviceError
from pacsys.types import Reading, ValueType
from pacsys.verify import Verify

from .devices import (
    ANALOG_ALARM_SETPOINT,
    ARRAY_DEVICE,
    CONTROL_PAIRS,
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


def _create_dpm_write_backend():
    from pacsys.auth import KerberosAuth
    from pacsys.backends.dpm_http import DPMHTTPBackend

    return DPMHTTPBackend(auth=KerberosAuth(), role="testing")


# =============================================================================
# Read Tests
# =============================================================================


@requires_dpm_http
class TestDeviceRead:
    """Read operations via Device API."""

    def test_read_scalar(self, dpm_http_backend):
        dev = ScalarDevice(SCALAR_DEVICE, backend=dpm_http_backend)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, float)
        print(f"\n  {dev.name}: {value}")

    def test_read_scalar_second_device(self, dpm_http_backend):
        dev = ScalarDevice(SCALAR_DEVICE_2, backend=dpm_http_backend)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, float)

    def test_read_array(self, dpm_http_backend):
        dev = ArrayDevice(ARRAY_DEVICE, backend=dpm_http_backend)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, np.ndarray)
        assert len(value) == 11
        print(f"\n  {dev.name}: array(len={len(value)})")

    def test_read_single_element(self, dpm_http_backend):
        dev = ScalarDevice(SCALAR_ELEMENT, backend=dpm_http_backend)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, float)

    def test_read_raw(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend)
        value = dev.read(field="raw", timeout=TIMEOUT_READ)
        assert isinstance(value, bytes)
        assert len(value) > 0
        print(f"\n  {dev.name} RAW: {value.hex()} (len={len(value)})")

    def test_read_description(self, dpm_http_backend):
        dev = TextDevice("M:OUTTMP", backend=dpm_http_backend)
        desc = dev.description(timeout=TIMEOUT_READ)
        assert isinstance(desc, str)
        assert len(desc) > 0
        print(f"\n  {dev.name} DESCRIPTION: {desc!r}")

    def test_read_status(self, dpm_http_backend):
        dev = Device(STATUS_DEVICE, backend=dpm_http_backend)
        value = dev.status(timeout=TIMEOUT_READ)
        assert isinstance(value, dict)
        assert "on" in value
        print(f"\n  {dev.name} STATUS: {value}")

    def test_read_status_field(self, dpm_http_backend):
        """status(field='on') returns a bool."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend)
        on_val = dev.status(field="on", timeout=TIMEOUT_READ)
        assert isinstance(on_val, bool)
        print(f"\n  {dev.name} STATUS.ON: {on_val}")

    def test_read_analog_alarm(self, dpm_http_backend):
        dev = Device("N:H801", backend=dpm_http_backend)
        value = dev.analog_alarm(timeout=TIMEOUT_READ)
        assert isinstance(value, dict)
        assert "minimum" in value
        assert "maximum" in value
        print(f"\n  {dev.name} ANALOG alarm: {value}")

    def test_read_digital_alarm(self, dpm_http_backend):
        dev = Device("N:H801", backend=dpm_http_backend)
        value = dev.digital_alarm(timeout=TIMEOUT_READ)
        assert isinstance(value, dict)
        print(f"\n  {dev.name} DIGITAL alarm: {value}")


# =============================================================================
# Setting Tests
# =============================================================================


@requires_dpm_http
class TestDeviceSetting:
    """Read SETTING property via Device API."""

    def test_read_setting(self, dpm_http_backend):
        dev = ScalarDevice(SCALAR_DEVICE_3, backend=dpm_http_backend)
        value = dev.setting(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))
        print(f"\n  {dev.name} SETTING: {value}")

    def test_read_setting_raw(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend)
        value = dev.setting(field="raw", timeout=TIMEOUT_READ)
        assert isinstance(value, bytes)
        assert len(value) > 0
        print(f"\n  {dev.name} SETTING.RAW: {value.hex()}")


# =============================================================================
# get() Tests
# =============================================================================


@requires_dpm_http
class TestDeviceGet:
    """get() returns full Reading."""

    def test_get_returns_reading(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend)
        reading = dev.get(timeout=TIMEOUT_READ)
        assert isinstance(reading, Reading)
        assert reading.ok
        assert reading.value is not None
        assert reading.value_type == ValueType.SCALAR
        print(f"\n  {reading.drf}: {reading.value} ({reading.value_type.name})")

    def test_get_has_metadata(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend)
        reading = dev.get(timeout=TIMEOUT_READ)
        assert reading.ok
        assert reading.name is not None
        assert reading.timestamp is not None


# =============================================================================
# digital_status() Tests
# =============================================================================


@requires_dpm_http
class TestDeviceDigitalStatus:
    """digital_status() fetches BIT_VALUE/BIT_NAMES/BIT_VALUES."""

    def test_returns_digital_status(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend)
        status = dev.digital_status(timeout=TIMEOUT_READ)
        assert isinstance(status, DigitalStatus)
        assert status.device == "Z:ACLTST"
        assert isinstance(status.raw_value, int)
        assert len(status.bits) > 0
        assert all(isinstance(b, StatusBit) for b in status.bits)
        print(f"\n  {status}")

    def test_legacy_attributes(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend)
        status = dev.digital_status(timeout=TIMEOUT_READ)
        assert isinstance(status.on, bool)
        assert isinstance(status.ready, bool)
        print(f"\n  on={status.on}, ready={status.ready}, positive={status.positive}")

    def test_bit_lookup(self, dpm_http_backend):
        """Bits can be looked up by name or position."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend)
        status = dev.digital_status(timeout=TIMEOUT_READ)
        bit = status["On"]
        assert bit is not None
        assert isinstance(bit, StatusBit)
        assert bit.name.lower() == "on"


# =============================================================================
# Error Handling Tests
# =============================================================================


@requires_dpm_http
class TestDeviceErrors:
    """Error handling via Device API."""

    def test_read_nonexistent_raises(self, dpm_http_backend):
        dev = Device(NONEXISTENT_DEVICE, backend=dpm_http_backend)
        with pytest.raises(DeviceError):
            dev.read(timeout=TIMEOUT_READ)

    def test_get_nonexistent_returns_error(self, dpm_http_backend):
        dev = Device(NONEXISTENT_DEVICE, backend=dpm_http_backend)
        reading = dev.get(timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code != 0

    def test_invalid_field_raises_valueerror(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend)
        with pytest.raises(ValueError, match="not allowed"):
            dev.read(field="on")

    def test_status_invalid_field_raises(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend)
        with pytest.raises(ValueError, match="not allowed"):
            dev.status(field="scaled")


# =============================================================================
# Subclass Type Safety Tests
# =============================================================================


@requires_dpm_http
class TestDeviceSubclasses:
    """ScalarDevice, ArrayDevice, TextDevice enforce value types."""

    def test_scalar_device_returns_float(self, dpm_http_backend):
        dev = ScalarDevice(SCALAR_DEVICE, backend=dpm_http_backend)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, float)

    def test_array_device_returns_ndarray(self, dpm_http_backend):
        dev = ArrayDevice(ARRAY_DEVICE, backend=dpm_http_backend)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, np.ndarray)

    def test_text_device_returns_str(self, dpm_http_backend):
        dev = TextDevice("M:OUTTMP", backend=dpm_http_backend)
        desc = dev.description(timeout=TIMEOUT_READ)
        assert isinstance(desc, str)


# =============================================================================
# Fluent Modifier Tests
# =============================================================================


@requires_dpm_http
class TestDeviceFluent:
    """with_backend(), with_event(), with_range() return new Device."""

    def test_with_backend(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE)
        bound = dev.with_backend(dpm_http_backend)
        assert bound is not dev
        value = bound.read(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    def test_with_event(self, dpm_http_backend):
        dev = ScalarDevice(SCALAR_DEVICE, backend=dpm_http_backend)
        periodic = dev.with_event("p,1000")
        assert periodic.is_periodic
        assert isinstance(periodic, ScalarDevice)
        assert "p,1000" in periodic.drf

    def test_with_range(self, dpm_http_backend):
        dev = Device("B:IRMS06", backend=dpm_http_backend)
        ranged = dev.with_range(0, 5)
        assert "[0:5]" in ranged.drf
        value = ranged.read(timeout=TIMEOUT_READ)
        assert hasattr(value, "__len__")
        assert len(value) == 6

    def test_subclass_preserved(self, dpm_http_backend):
        """Fluent methods preserve subclass type."""
        dev = ScalarDevice(SCALAR_DEVICE, backend=dpm_http_backend)
        assert isinstance(dev.with_event("p,500"), ScalarDevice)
        assert isinstance(dev.with_backend(dpm_http_backend), ScalarDevice)


# =============================================================================
# Properties Tests
# =============================================================================


@requires_dpm_http
class TestDeviceProperties:
    """Device properties: drf, name, has_event, is_periodic."""

    def test_name(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend)
        assert dev.name == "M:OUTTMP"

    def test_drf_canonical(self, dpm_http_backend):
        dev = Device("M:OUTTMP", backend=dpm_http_backend)
        assert "READING" in dev.drf or "M:OUTTMP" in dev.drf

    def test_has_event_false_by_default(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend)
        assert not dev.has_event

    def test_has_event_true(self, dpm_http_backend):
        dev = Device("M:OUTTMP@p,1000", backend=dpm_http_backend)
        assert dev.has_event

    def test_is_periodic(self, dpm_http_backend):
        dev = Device("M:OUTTMP@p,1000", backend=dpm_http_backend)
        assert dev.is_periodic

    def test_is_not_periodic(self, dpm_http_backend):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend)
        assert not dev.is_periodic


# =============================================================================
# Write Tests (requires Kerberos + PACSYS_TEST_WRITE=1)
# =============================================================================


@requires_dpm_http
@requires_kerberos
@pytest.mark.kerberos
class TestDeviceWrite:
    """Device.write() operations."""

    @pytest.mark.write
    @requires_write_enabled
    def test_write_scalar(self):
        """Write a different value, verify readback, restore."""
        backend = _create_dpm_write_backend()
        try:
            dev = ScalarDevice(SCALAR_DEVICE_3, backend=backend)
            original = dev.setting(timeout=TIMEOUT_READ)
            print(f"\n  Original SETTING: {original}")

            new_value = original + 0.1
            result = dev.write(new_value, timeout=TIMEOUT_READ)
            assert result.success
            print(f"  Write {new_value}: success={result.success}")

            time.sleep(1.0)
            readback = dev.setting(timeout=TIMEOUT_READ)
            assert abs(readback - new_value) < 0.01, f"Wrote {new_value}, read back {readback}"
            print(f"  Readback: {readback}")

            # Restore
            result2 = dev.write(original, timeout=TIMEOUT_READ)
            assert result2.success
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    def test_write_raw(self):
        """Write raw bytes via field='raw'."""
        backend = _create_dpm_write_backend()
        try:
            dev = Device(SCALAR_DEVICE_3, backend=backend)
            original_raw = dev.setting(field="raw", timeout=TIMEOUT_READ)
            assert isinstance(original_raw, bytes)
            print(f"\n  Original raw: {original_raw.hex()}")

            # DEC F_float for 45.0
            raw_45 = b"\x34\x43\x00\x00"
            result = dev.write(raw_45, field="raw", timeout=TIMEOUT_READ)
            assert result.success

            time.sleep(1.0)
            readback_raw = dev.setting(field="raw", timeout=TIMEOUT_READ)
            assert readback_raw == raw_45
            readback_scaled = dev.setting(timeout=TIMEOUT_READ)
            assert readback_scaled == 45.0
            print(f"  After raw write: scaled={readback_scaled}")

            # Restore
            dev.write(original_raw, field="raw", timeout=TIMEOUT_READ)
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    def test_write_with_verify(self):
        """Device.write(verify=True) reads back the value."""
        backend = _create_dpm_write_backend()
        try:
            dev = ScalarDevice(SCALAR_DEVICE_3, backend=backend)
            original = dev.setting(timeout=TIMEOUT_READ)

            new_value = original + 0.1
            result = dev.write(new_value, verify=Verify(tolerance=0.01), timeout=TIMEOUT_READ)
            assert result.success
            assert result.verified is True
            assert result.readback is not None
            assert abs(result.readback - new_value) < 0.01
            print(f"\n  Verified write: readback={result.readback}, attempts={result.attempts}")

            # Restore
            dev.write(original, timeout=TIMEOUT_READ)
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    def test_write_verify_check_first(self):
        """Verify(check_first=True) skips write when value already matches."""
        backend = _create_dpm_write_backend()
        try:
            dev = ScalarDevice(SCALAR_DEVICE_3, backend=backend)
            current = dev.setting(timeout=TIMEOUT_READ)

            # Write the same value with check_first
            result = dev.write(current, verify=Verify(check_first=True, tolerance=0.01), timeout=TIMEOUT_READ)
            assert result.success
            assert result.skipped is True
            assert result.verified is True
            print(f"\n  check_first skipped: value already {current}")
        finally:
            backend.close()


# =============================================================================
# Control Tests (requires Kerberos + PACSYS_TEST_WRITE=1)
# =============================================================================


@requires_dpm_http
@requires_kerberos
@pytest.mark.kerberos
class TestDeviceControl:
    """Device.control() and shortcut methods (on/off/reset/etc)."""

    @pytest.mark.write
    @requires_write_enabled
    def test_on_off(self):
        """dev.on() / dev.off() toggle the on status bit."""
        backend = _create_dpm_write_backend()
        try:
            dev = Device(SCALAR_DEVICE_3, backend=backend)
            initial = dev.digital_status(timeout=TIMEOUT_READ)
            print(f"\n  Initial on={initial.on}")

            result = dev.on(timeout=TIMEOUT_READ)
            assert result.success
            time.sleep(1.0)
            assert dev.status(field="on", timeout=TIMEOUT_READ) is True

            result = dev.off(timeout=TIMEOUT_READ)
            assert result.success
            time.sleep(1.0)
            assert dev.status(field="on", timeout=TIMEOUT_READ) is False

            # Restore
            if initial.on:
                dev.on(timeout=TIMEOUT_READ)
            print(f"  Restored on={initial.on}")
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    def test_positive_negative(self):
        """dev.positive() / dev.negative() toggle the positive status bit."""
        backend = _create_dpm_write_backend()
        try:
            dev = Device(SCALAR_DEVICE_3, backend=backend)
            initial = dev.digital_status(timeout=TIMEOUT_READ)

            dev.positive(timeout=TIMEOUT_READ)
            time.sleep(1.0)
            assert dev.status(field="positive", timeout=TIMEOUT_READ) is True

            dev.negative(timeout=TIMEOUT_READ)
            time.sleep(1.0)
            assert dev.status(field="positive", timeout=TIMEOUT_READ) is False

            # Restore
            if initial.positive:
                dev.positive(timeout=TIMEOUT_READ)
            else:
                dev.negative(timeout=TIMEOUT_READ)
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    def test_ramp_dc(self):
        """dev.ramp() / dev.dc() toggle the ramp status bit."""
        backend = _create_dpm_write_backend()
        try:
            dev = Device(SCALAR_DEVICE_3, backend=backend)
            initial = dev.digital_status(timeout=TIMEOUT_READ)

            dev.ramp(timeout=TIMEOUT_READ)
            time.sleep(1.0)
            assert dev.status(field="ramp", timeout=TIMEOUT_READ) is True

            dev.dc(timeout=TIMEOUT_READ)
            time.sleep(1.0)
            assert dev.status(field="ramp", timeout=TIMEOUT_READ) is False

            # Restore
            if initial.ramp:
                dev.ramp(timeout=TIMEOUT_READ)
            else:
                dev.dc(timeout=TIMEOUT_READ)
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    def test_reset(self):
        """dev.reset() succeeds and status has on/ready."""
        backend = _create_dpm_write_backend()
        try:
            dev = Device(SCALAR_DEVICE_3, backend=backend)
            result = dev.reset(timeout=TIMEOUT_READ)
            assert result.success
            print(f"\n  RESET: success={result.success}")

            time.sleep(1.0)
            status = dev.status(timeout=TIMEOUT_READ)
            assert isinstance(status, dict)
            assert "on" in status
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    def test_control_with_verify(self):
        """dev.on(verify=True) verifies STATUS.ON is True after write."""
        backend = _create_dpm_write_backend()
        try:
            dev = Device(SCALAR_DEVICE_3, backend=backend)
            initial_on = dev.status(field="on", timeout=TIMEOUT_READ)

            result = dev.on(verify=True, timeout=TIMEOUT_READ)
            assert result.success
            assert result.verified is True
            assert result.readback is True
            print(f"\n  on(verify=True): verified={result.verified}, attempts={result.attempts}")

            # Restore
            if not initial_on:
                dev.off(timeout=TIMEOUT_READ)
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.parametrize(
        "cmd_true,cmd_false,field",
        CONTROL_PAIRS,
        ids=lambda x: x if isinstance(x, str) else x.name,
    )
    def test_control_pair(self, cmd_true, cmd_false, field):
        """Toggle control pair via device.control() and verify status."""
        backend = _create_dpm_write_backend()
        try:
            dev = Device(SCALAR_DEVICE_3, backend=backend)
            initial = dev.status(field=field, timeout=TIMEOUT_READ)
            print(f"\n  Initial {field}: {initial}")

            result = dev.control(cmd_true, timeout=TIMEOUT_READ)
            assert result.success
            time.sleep(1.0)
            assert dev.status(field=field, timeout=TIMEOUT_READ) is True

            result = dev.control(cmd_false, timeout=TIMEOUT_READ)
            assert result.success
            time.sleep(1.0)
            assert dev.status(field=field, timeout=TIMEOUT_READ) is False

            # Restore
            dev.control(cmd_true if initial else cmd_false, timeout=TIMEOUT_READ)
        finally:
            backend.close()


# =============================================================================
# Alarm Write Tests
# =============================================================================


@requires_dpm_http
@requires_kerberos
@pytest.mark.kerberos
class TestDeviceAlarmWrite:
    """Alarm write operations via Device API."""

    @pytest.mark.write
    @requires_write_enabled
    def test_write_analog_alarm_max(self):
        """Write analog alarm MAX via device.set_analog_alarm() isn't covered;
        test write to ANALOG field via backend write (mirrors backend test)."""
        backend = _create_dpm_write_backend()
        try:
            dev = Device("Z:ACLTST", backend=backend)
            alarm = dev.analog_alarm(timeout=TIMEOUT_READ)
            assert isinstance(alarm, dict)
            orig_max = alarm["maximum"]
            print(f"\n  Original alarm max: {orig_max}")

            new_max = orig_max + 0.5
            # Use backend.write for field-level alarm write (device API writes whole block)
            result = backend.write(f"{ANALOG_ALARM_SETPOINT}.MAX", new_max, timeout=TIMEOUT_READ)
            assert result.success

            time.sleep(1.0)
            after = dev.analog_alarm(timeout=TIMEOUT_READ)
            assert after["maximum"] == new_max

            # Restore
            backend.write(f"{ANALOG_ALARM_SETPOINT}.MAX", orig_max, timeout=TIMEOUT_READ)
        finally:
            backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
