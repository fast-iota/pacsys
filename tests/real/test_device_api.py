"""
Integration tests for Device API against real DPM/HTTP backend.

Tests the Device-centric interface: read(), setting(), status(), description(),
analog_alarm(), digital_alarm(), digital_status(), write(), control(), and
fluent modifiers (with_backend, with_event, with_range).
"""

import math
import time

import numpy as np
import pytest

from pacsys.device import ArrayDevice, Device, ScalarDevice, TextDevice
from pacsys.digital_status import DigitalStatus, StatusBit
from pacsys.errors import DeviceError
from pacsys.types import Reading, ValueType
from pacsys.verify import Verify

from .devices import (
    ANALOG_ALARM_SETPOINT,
    ARRAY_DEVICE,
    CONTROL_PAIRS,
    DPM_TEST_HOST,
    DPM_TEST_PORT,
    NONEXISTENT_DEVICE,
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_DEVICE_3,
    SCALAR_ELEMENT,
    STATUS_DEVICE,
    TIMEOUT_READ,
    requires_dpm_http,
    requires_kerberos,
    requires_write_enabled,
    wait_for_readback,
)


@pytest.fixture(scope="module")
def dpm_write_backend():
    from pacsys.auth import KerberosAuth
    from pacsys.backends.dpm_http import DPMHTTPBackend

    backend = DPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT, auth=KerberosAuth(), role="testing")
    yield backend
    backend.close()


@pytest.fixture(autouse=True)
def pause():
    """Space out load."""
    time.sleep(0.05)
    yield
    time.sleep(0.05)


# =============================================================================
# Read Tests
# =============================================================================


@requires_dpm_http
class TestDeviceRead:
    """Read operations via Device API."""

    def test_read_scalar(self, dpm_http_backend_cls):
        dev = ScalarDevice(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, float)
        assert math.isfinite(value)
        print(f"\n  {dev.name}: {value}")

    def test_read_scalar_second_device(self, dpm_http_backend_cls):
        dev = ScalarDevice(SCALAR_DEVICE_2, backend=dpm_http_backend_cls)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, float)
        assert math.isfinite(value)

    def test_read_array(self, dpm_http_backend_cls):
        dev = ArrayDevice(ARRAY_DEVICE, backend=dpm_http_backend_cls)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, np.ndarray)
        assert len(value) == 11
        print(f"\n  {dev.name}: array(len={len(value)})")

    def test_read_single_element(self, dpm_http_backend_cls):
        dev = ScalarDevice(SCALAR_ELEMENT, backend=dpm_http_backend_cls)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, float)
        assert math.isfinite(value)

    def test_read_raw(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        value = dev.read(field="raw", timeout=TIMEOUT_READ)
        assert isinstance(value, bytes)
        assert len(value) > 0
        print(f"\n  {dev.name} RAW: {value.hex()} (len={len(value)})")

    def test_read_description(self, dpm_http_backend_cls):
        dev = TextDevice("M:OUTTMP", backend=dpm_http_backend_cls)
        desc = dev.description(timeout=TIMEOUT_READ)
        assert isinstance(desc, str)
        assert len(desc) > 0
        print(f"\n  {dev.name} DESCRIPTION: {desc!r}")

    def test_read_status(self, dpm_http_backend_cls):
        dev = Device(STATUS_DEVICE, backend=dpm_http_backend_cls)
        value = dev.status(timeout=TIMEOUT_READ)
        assert isinstance(value, dict)
        assert "on" in value
        print(f"\n  {dev.name} STATUS: {value}")

    def test_read_status_field(self, dpm_http_backend_cls):
        """status(field='on') returns a bool."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend_cls)
        on_val = dev.status(field="on", timeout=TIMEOUT_READ)
        assert isinstance(on_val, bool)
        print(f"\n  {dev.name} STATUS.ON: {on_val}")

    def test_read_analog_alarm(self, dpm_http_backend_cls):
        dev = Device("N:H801", backend=dpm_http_backend_cls)
        value = dev.analog_alarm(timeout=TIMEOUT_READ)
        assert isinstance(value, dict)
        assert "minimum" in value
        assert "maximum" in value
        print(f"\n  {dev.name} ANALOG alarm: {value}")

    def test_read_digital_alarm(self, dpm_http_backend_cls):
        dev = Device("N:H801", backend=dpm_http_backend_cls)
        value = dev.digital_alarm(timeout=TIMEOUT_READ)
        assert isinstance(value, dict)
        print(f"\n  {dev.name} DIGITAL alarm: {value}")


# =============================================================================
# Setting Tests
# =============================================================================


@requires_dpm_http
class TestDeviceSetting:
    """Read SETTING property via Device API."""

    def test_read_setting(self, dpm_http_backend_cls):
        dev = ScalarDevice(SCALAR_DEVICE_3, backend=dpm_http_backend_cls)
        value = dev.setting(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(value)
        print(f"\n  {dev.name} SETTING: {value}")

    def test_read_setting_raw(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend_cls)
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

    def test_get_returns_reading(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        reading = dev.get(timeout=TIMEOUT_READ)
        assert isinstance(reading, Reading)
        assert reading.ok
        assert reading.value is not None
        assert reading.value_type == ValueType.SCALAR
        assert isinstance(reading.value, (int, float)) and not isinstance(reading.value, bool)
        assert math.isfinite(reading.value)
        print(f"\n  {reading.drf}: {reading.value} ({reading.value_type.name})")

    def test_get_has_metadata(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        reading = dev.get(timeout=TIMEOUT_READ)
        assert reading.ok
        assert reading.name is not None
        assert reading.timestamp is not None
        assert isinstance(reading.value, (int, float)) and not isinstance(reading.value, bool)
        assert math.isfinite(reading.value)


# =============================================================================
# digital_status() Tests
# =============================================================================


@requires_dpm_http
class TestDeviceDigitalStatus:
    """digital_status() fetches BIT_VALUE/BIT_NAMES/BIT_VALUES."""

    def test_returns_digital_status(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend_cls)
        status = dev.digital_status(timeout=TIMEOUT_READ)
        assert isinstance(status, DigitalStatus)
        assert status.device == "Z:ACLTST"
        assert isinstance(status.raw_value, int)
        assert len(status.bits) > 0
        assert all(isinstance(b, StatusBit) for b in status.bits)
        print(f"\n  {status}")

    def test_legacy_attributes(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend_cls)
        status = dev.digital_status(timeout=TIMEOUT_READ)
        assert isinstance(status.on, bool)
        assert isinstance(status.ready, bool)
        print(f"\n  on={status.on}, ready={status.ready}, positive={status.positive}")

    def test_bit_lookup(self, dpm_http_backend_cls):
        """Bits can be looked up by name or position."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend_cls)
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

    def test_read_nonexistent_raises(self, dpm_http_backend_cls):
        dev = Device(NONEXISTENT_DEVICE, backend=dpm_http_backend_cls)
        with pytest.raises(DeviceError):
            dev.read(timeout=TIMEOUT_READ)

    def test_get_nonexistent_returns_error(self, dpm_http_backend_cls):
        dev = Device(NONEXISTENT_DEVICE, backend=dpm_http_backend_cls)
        reading = dev.get(timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code != 0

    def test_invalid_field_raises_valueerror(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        with pytest.raises(ValueError, match="not allowed"):
            dev.read(field="on")

    def test_status_invalid_field_raises(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE_3, backend=dpm_http_backend_cls)
        with pytest.raises(ValueError, match="not allowed"):
            dev.status(field="scaled")


# =============================================================================
# Subclass Type Safety Tests
# =============================================================================


@requires_dpm_http
class TestDeviceSubclasses:
    """ScalarDevice, ArrayDevice, TextDevice enforce value types."""

    def test_scalar_device_returns_float(self, dpm_http_backend_cls):
        dev = ScalarDevice(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, float)
        assert math.isfinite(value)

    def test_array_device_returns_ndarray(self, dpm_http_backend_cls):
        dev = ArrayDevice(ARRAY_DEVICE, backend=dpm_http_backend_cls)
        value = dev.read(timeout=TIMEOUT_READ)
        assert isinstance(value, np.ndarray)

    def test_text_device_returns_str(self, dpm_http_backend_cls):
        dev = TextDevice("M:OUTTMP", backend=dpm_http_backend_cls)
        desc = dev.description(timeout=TIMEOUT_READ)
        assert isinstance(desc, str)


# =============================================================================
# Fluent Modifier Tests
# =============================================================================


@requires_dpm_http
class TestDeviceFluent:
    """with_backend(), with_event(), with_range() return new Device."""

    def test_with_backend(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE)
        bound = dev.with_backend(dpm_http_backend_cls)
        assert bound is not dev
        value = bound.read(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(value)

    def test_with_event(self, dpm_http_backend_cls):
        dev = ScalarDevice(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        periodic = dev.with_event("p,1000")
        assert periodic.is_periodic
        assert isinstance(periodic, ScalarDevice)
        assert "p,1000" in periodic.drf

    def test_with_range(self, dpm_http_backend_cls):
        dev = Device("B:IRMS06", backend=dpm_http_backend_cls)
        ranged = dev.with_range(0, 5)
        assert "[0:5]" in ranged.drf
        value = ranged.read(timeout=TIMEOUT_READ)
        assert hasattr(value, "__len__")
        assert len(value) == 6

    def test_subclass_preserved(self, dpm_http_backend_cls):
        """Fluent methods preserve subclass type."""
        dev = ScalarDevice(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        assert isinstance(dev.with_event("p,500"), ScalarDevice)
        assert isinstance(dev.with_backend(dpm_http_backend_cls), ScalarDevice)


# =============================================================================
# Properties Tests
# =============================================================================


@requires_dpm_http
class TestDeviceProperties:
    """Device properties: drf, name, has_event, is_periodic."""

    def test_name(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        assert dev.name == "M:OUTTMP"

    def test_drf_canonical(self, dpm_http_backend_cls):
        dev = Device("M:OUTTMP", backend=dpm_http_backend_cls)
        assert "READING" in dev.drf or "M:OUTTMP" in dev.drf

    def test_has_event_false_by_default(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend_cls)
        assert not dev.has_event

    def test_has_event_true(self, dpm_http_backend_cls):
        dev = Device("M:OUTTMP@p,1000", backend=dpm_http_backend_cls)
        assert dev.has_event

    def test_is_periodic(self, dpm_http_backend_cls):
        dev = Device("M:OUTTMP@p,1000", backend=dpm_http_backend_cls)
        assert dev.is_periodic

    def test_is_not_periodic(self, dpm_http_backend_cls):
        dev = Device(SCALAR_DEVICE, backend=dpm_http_backend_cls)
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
    def test_write_scalar(self, dpm_write_backend):
        """Write a different value, verify readback, restore."""
        dev = ScalarDevice(SCALAR_DEVICE_3, backend=dpm_write_backend)
        original = dev.setting(timeout=TIMEOUT_READ)
        assert isinstance(original, (int, float)) and not isinstance(original, bool)
        assert math.isfinite(original)
        print(f"\n  Original SETTING: {original}")
        try:
            new_value = original + 0.1
            result = dev.write(new_value, timeout=TIMEOUT_READ)
            assert result.success
            print(f"  Write {new_value}: success={result.success}")

            readback = wait_for_readback(
                lambda: dev.setting(timeout=TIMEOUT_READ),
                lambda value: value == pytest.approx(new_value, abs=0.01),
                description=f"{dev.name} setting={new_value}",
            )
            print(f"  Readback: {readback}")
        finally:
            result = dev.write(original, timeout=TIMEOUT_READ)
            assert result.success, f"Restore failed: {result.error_code} {result.message}"
            wait_for_readback(
                lambda: dev.setting(timeout=TIMEOUT_READ),
                lambda value: value == pytest.approx(original, abs=0.01),
                description=f"restore {dev.name} setting={original}",
            )

    @pytest.mark.write
    @requires_write_enabled
    def test_write_raw(self, dpm_write_backend):
        """Write raw bytes via field='raw'."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_write_backend)
        original_raw = dev.setting(field="raw", timeout=TIMEOUT_READ)
        assert isinstance(original_raw, bytes)
        print(f"\n  Original raw: {original_raw.hex()}")
        try:
            # DEC F_float for 45.0
            raw_45 = b"\x34\x43\x00\x00"
            result = dev.write(raw_45, field="raw", timeout=TIMEOUT_READ)
            assert result.success

            wait_for_readback(
                lambda: dev.setting(field="raw", timeout=TIMEOUT_READ),
                lambda value: value == raw_45,
                description=f"{dev.name} raw setting={raw_45.hex()}",
            )
            readback_scaled = dev.setting(timeout=TIMEOUT_READ)
            assert readback_scaled == pytest.approx(45.0)
            print(f"  After raw write: scaled={readback_scaled}")
        finally:
            result = dev.write(original_raw, field="raw", timeout=TIMEOUT_READ)
            assert result.success, f"Restore failed: {result.error_code} {result.message}"
            wait_for_readback(
                lambda: dev.setting(field="raw", timeout=TIMEOUT_READ),
                lambda value: value == original_raw,
                description=f"restore {dev.name} raw setting",
            )

    @pytest.mark.write
    @requires_write_enabled
    def test_write_with_verify(self, dpm_write_backend):
        """Device.write(verify=True) reads back the value."""
        dev = ScalarDevice(SCALAR_DEVICE_3, backend=dpm_write_backend)
        original = dev.setting(timeout=TIMEOUT_READ)
        assert isinstance(original, (int, float)) and not isinstance(original, bool)
        assert math.isfinite(original)
        try:
            new_value = original + 0.1
            result = dev.write(new_value, verify=Verify(tolerance=0.01), timeout=TIMEOUT_READ)
            assert result.success
            assert result.verified is True
            assert result.readback == pytest.approx(new_value, abs=0.01)
            print(f"\n  Verified write: readback={result.readback}, attempts={result.attempts}")
        finally:
            result = dev.write(original, timeout=TIMEOUT_READ)
            assert result.success, f"Restore failed: {result.error_code} {result.message}"
            wait_for_readback(
                lambda: dev.setting(timeout=TIMEOUT_READ),
                lambda value: value == pytest.approx(original, abs=0.01),
                description=f"restore {dev.name} setting={original}",
            )

    @pytest.mark.write
    @requires_write_enabled
    def test_write_verify_check_first(self, dpm_write_backend):
        """Verify(check_first=True) skips write when value already matches."""
        dev = ScalarDevice(SCALAR_DEVICE_3, backend=dpm_write_backend)
        current = dev.setting(timeout=TIMEOUT_READ)
        assert isinstance(current, (int, float)) and not isinstance(current, bool)
        assert math.isfinite(current)

        # Write the same value with check_first
        result = dev.write(current, verify=Verify(check_first=True, tolerance=0.01), timeout=TIMEOUT_READ)
        assert result.success
        assert result.skipped is True
        assert result.verified is True
        print(f"\n  check_first skipped: value already {current}")


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
    def test_on_off(self, dpm_write_backend):
        """dev.on() / dev.off() toggle the on status bit."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_write_backend)
        initial = dev.digital_status(timeout=TIMEOUT_READ)
        assert isinstance(initial.on, bool)
        print(f"\n  Initial on={initial.on}")
        try:
            result = dev.on(timeout=TIMEOUT_READ)
            assert result.success
            wait_for_readback(
                lambda: dev.status(field="on", timeout=TIMEOUT_READ),
                lambda value: value is True,
                description=f"{dev.name} on",
            )

            result = dev.off(timeout=TIMEOUT_READ)
            assert result.success
            wait_for_readback(
                lambda: dev.status(field="on", timeout=TIMEOUT_READ),
                lambda value: value is False,
                description=f"{dev.name} off",
            )
        finally:
            result = (dev.on if initial.on else dev.off)(timeout=TIMEOUT_READ)
            assert result.success, f"Restore failed: {result.error_code} {result.message}"
            wait_for_readback(
                lambda: dev.status(field="on", timeout=TIMEOUT_READ),
                lambda value: value is initial.on,
                description=f"restore {dev.name} on={initial.on}",
            )
        print(f"  Restored on={initial.on}")

    @pytest.mark.write
    @requires_write_enabled
    def test_positive_negative(self, dpm_write_backend):
        """dev.positive() / dev.negative() toggle the positive status bit."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_write_backend)
        initial = dev.digital_status(timeout=TIMEOUT_READ)
        assert isinstance(initial.positive, bool)
        try:
            result = dev.positive(timeout=TIMEOUT_READ)
            assert result.success
            wait_for_readback(
                lambda: dev.status(field="positive", timeout=TIMEOUT_READ),
                lambda value: value is True,
                description=f"{dev.name} positive",
            )

            result = dev.negative(timeout=TIMEOUT_READ)
            assert result.success
            wait_for_readback(
                lambda: dev.status(field="positive", timeout=TIMEOUT_READ),
                lambda value: value is False,
                description=f"{dev.name} negative",
            )
        finally:
            result = (dev.positive if initial.positive else dev.negative)(timeout=TIMEOUT_READ)
            assert result.success, f"Restore failed: {result.error_code} {result.message}"
            wait_for_readback(
                lambda: dev.status(field="positive", timeout=TIMEOUT_READ),
                lambda value: value is initial.positive,
                description=f"restore {dev.name} positive={initial.positive}",
            )

    @pytest.mark.write
    @requires_write_enabled
    def test_ramp_dc(self, dpm_write_backend):
        """dev.ramp() / dev.dc() toggle the ramp status bit."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_write_backend)
        # Basic status: without DevDB, digital_status() only sees the bits DPM names (no ramp bit)
        initial_ramp = dev.status(field="ramp", timeout=TIMEOUT_READ)
        assert isinstance(initial_ramp, bool)
        try:
            result = dev.ramp(timeout=TIMEOUT_READ)
            assert result.success
            wait_for_readback(
                lambda: dev.status(field="ramp", timeout=TIMEOUT_READ),
                lambda value: value is True,
                description=f"{dev.name} ramp",
            )

            result = dev.dc(timeout=TIMEOUT_READ)
            assert result.success
            wait_for_readback(
                lambda: dev.status(field="ramp", timeout=TIMEOUT_READ),
                lambda value: value is False,
                description=f"{dev.name} dc",
            )
        finally:
            result = (dev.ramp if initial_ramp else dev.dc)(timeout=TIMEOUT_READ)
            assert result.success, f"Restore failed: {result.error_code} {result.message}"
            wait_for_readback(
                lambda: dev.status(field="ramp", timeout=TIMEOUT_READ),
                lambda value: value is initial_ramp,
                description=f"restore {dev.name} ramp={initial_ramp}",
            )

    @pytest.mark.write
    @requires_write_enabled
    def test_reset(self, dpm_write_backend):
        """dev.reset() succeeds and status has on/ready."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_write_backend)
        result = dev.reset(timeout=TIMEOUT_READ)
        assert result.success
        print(f"\n  RESET: success={result.success}")

        status = dev.status(timeout=TIMEOUT_READ)
        assert isinstance(status, dict)
        assert "on" in status

    @pytest.mark.write
    @requires_write_enabled
    def test_control_with_verify(self, dpm_write_backend):
        """dev.on(verify=True) verifies STATUS.ON is True after write."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_write_backend)
        initial_on = dev.status(field="on", timeout=TIMEOUT_READ)
        assert isinstance(initial_on, bool)
        try:
            result = dev.on(verify=True, timeout=TIMEOUT_READ)
            assert result.success
            assert result.verified is True
            assert result.readback is True
            print(f"\n  on(verify=True): verified={result.verified}, attempts={result.attempts}")
        finally:
            result = (dev.on if initial_on else dev.off)(timeout=TIMEOUT_READ)
            assert result.success, f"Restore failed: {result.error_code} {result.message}"
            wait_for_readback(
                lambda: dev.status(field="on", timeout=TIMEOUT_READ),
                lambda value: value is initial_on,
                description=f"restore {dev.name} on={initial_on}",
            )

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.parametrize(
        ("cmd_true", "cmd_false", "field"),
        CONTROL_PAIRS,
        ids=lambda x: x if isinstance(x, str) else x.name,
    )
    def test_control_pair(self, cmd_true, cmd_false, field, dpm_write_backend):
        """Toggle control pair via device.control() and verify status."""
        dev = Device(SCALAR_DEVICE_3, backend=dpm_write_backend)
        initial = dev.status(field=field, timeout=TIMEOUT_READ)
        assert isinstance(initial, bool)
        print(f"\n  Initial {field}: {initial}")
        try:
            result = dev.control(cmd_true, timeout=TIMEOUT_READ)
            assert result.success
            wait_for_readback(
                lambda: dev.status(field=field, timeout=TIMEOUT_READ),
                lambda value: value is True,
                description=f"{dev.name} {field}=True",
            )

            result = dev.control(cmd_false, timeout=TIMEOUT_READ)
            assert result.success
            wait_for_readback(
                lambda: dev.status(field=field, timeout=TIMEOUT_READ),
                lambda value: value is False,
                description=f"{dev.name} {field}=False",
            )
        finally:
            result = dev.control(cmd_true if initial else cmd_false, timeout=TIMEOUT_READ)
            assert result.success, f"Restore failed: {result.error_code} {result.message}"
            wait_for_readback(
                lambda: dev.status(field=field, timeout=TIMEOUT_READ),
                lambda value: value is initial,
                description=f"restore {dev.name} {field}={initial}",
            )


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
    def test_write_analog_alarm_max(self, dpm_write_backend):
        """A field-level MAX write is visible through Device.analog_alarm()."""
        dev = Device("Z:ACLTST", backend=dpm_write_backend)
        alarm = dev.analog_alarm(timeout=TIMEOUT_READ)
        assert isinstance(alarm, dict)
        orig_max = alarm["maximum"]
        assert isinstance(orig_max, (int, float)) and not isinstance(orig_max, bool)
        assert math.isfinite(orig_max)
        print(f"\n  Original alarm max: {orig_max}")
        try:
            new_max = orig_max + 0.5
            # Use backend.write for field-level alarm write (device API writes whole block)
            result = dpm_write_backend.write(f"{ANALOG_ALARM_SETPOINT}.MAX", new_max, timeout=TIMEOUT_READ)
            assert result.success

            wait_for_readback(
                lambda: dev.analog_alarm(timeout=TIMEOUT_READ),
                lambda value: value["maximum"] == pytest.approx(new_max),
                description=f"{dev.name} alarm maximum={new_max}",
            )
        finally:
            result = dpm_write_backend.write(f"{ANALOG_ALARM_SETPOINT}.MAX", orig_max, timeout=TIMEOUT_READ)
            assert result.success, f"Restore failed: {result.error_code} {result.message}"
            wait_for_readback(
                lambda: dev.analog_alarm(timeout=TIMEOUT_READ),
                lambda value: value["maximum"] == pytest.approx(orig_max),
                description=f"restore {dev.name} alarm maximum={orig_max}",
            )
