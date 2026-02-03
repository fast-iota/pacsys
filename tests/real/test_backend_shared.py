"""
Shared tests that run against all read-capable backends.

These tests use the `read_backend` parametrized fixture which runs each test
against DMQ, DPM HTTP, and gRPC backends (where available).

Tests here verify cross-backend behavior consistency. Backend-specific tests
should remain in their individual test files (test_dmq_backend.py, etc.).

Run with: pytest tests/real/test_backend_shared.py -v
"""

import pytest
import time
import threading

from pacsys.drf_utils import strip_event
from pacsys.types import Reading, ValueType, BackendCapability, SubscriptionHandle
from pacsys.errors import DeviceError

from .devices import (
    CONTROL_PAIRS,
    CONTROL_RESET,
    DEVICE_TYPES,
    NONEXISTENT_DEVICE,
    NOPROP_DEVICE,
    PERIODIC_DEVICE,
    FAST_PERIODIC,
    SLOW_PERIODIC,
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_SETPOINT,
    SCALAR_SETPOINT_RAW,
    STATUS_CONTROL_DEVICE,
    ARRAY_DEVICE,
    SCALAR_ELEMENT,
    TIMEOUT_READ,
    TIMEOUT_BATCH,
    TIMEOUT_STREAM_EVENT,
    TIMEOUT_STREAM_ITER,
    requires_kerberos,
    requires_write_enabled,
)


# =============================================================================
# Connection Tests
# =============================================================================


@pytest.mark.real
class TestBackendConnection:
    """Tests for backend connection lifecycle (all backends)."""

    def test_backend_has_read_capability(self, read_backend):
        """All backends should have READ capability."""
        assert BackendCapability.READ in read_backend.capabilities

    def test_backend_not_closed_initially(self, read_backend):
        """Backend should not be closed after creation."""
        assert not read_backend._closed


# =============================================================================
# Basic Read Tests
# =============================================================================


@pytest.mark.real
class TestBackendRead:
    """Tests for basic read operations (all backends)."""

    def test_get_scalar(self, read_backend):
        """get() returns Reading with float value for scalar device."""
        reading = read_backend.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_DEVICE}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR
        assert isinstance(reading.value, float)
        assert reading.timestamp is not None

    def test_get_second_device(self, read_backend):
        """get() works for second test device."""
        reading = read_backend.get(SCALAR_DEVICE_2, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_DEVICE_2}: {reading.message}"
        assert isinstance(reading.value, float)

    def test_get_array(self, read_backend):
        """get() returns array for array device."""
        reading = read_backend.get(ARRAY_DEVICE, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {ARRAY_DEVICE}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR_ARRAY
        assert hasattr(reading.value, "__len__")
        assert len(reading.value) >= 2

    def test_get_array_element(self, read_backend):
        """get() returns scalar for array element."""
        reading = read_backend.get(SCALAR_ELEMENT, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_ELEMENT}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR
        assert isinstance(reading.value, (int, float))


# =============================================================================
# Batch Read Tests
# =============================================================================


@pytest.mark.real
class TestBackendBatchReads:
    """Tests for batch read operations (all backends)."""

    def test_get_many(self, read_backend):
        """get_many() reads multiple devices in one batch."""
        devices = [SCALAR_DEVICE, SCALAR_DEVICE_2]
        readings = read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        for i, reading in enumerate(readings):
            assert reading.ok, f"Failed: {devices[i]}: {reading.message}"
            assert isinstance(reading.value, float)

    def test_get_many_empty_list(self, read_backend):
        """get_many() with empty list returns empty list."""
        readings = read_backend.get_many([], timeout=TIMEOUT_READ)
        assert readings == []

    def test_get_many_order_preserved(self, read_backend):
        """get_many() returns readings in same order as request."""
        devices = [SCALAR_DEVICE_2, SCALAR_DEVICE, SCALAR_ELEMENT]
        readings = read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 3
        # All should succeed
        assert all(r.ok for r in readings), f"Failures: {[r.message for r in readings if not r.ok]}"

    def test_get_many_duplicate_drf(self, read_backend):
        """get_many() handles same device requested multiple times."""
        devices = [SCALAR_DEVICE, SCALAR_DEVICE_2, SCALAR_DEVICE]

        readings = read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 3
        assert readings[0].ok
        assert readings[1].ok
        assert readings[2].ok
        # Both SCALAR_DEVICE readings should have values
        assert readings[0].value is not None
        assert readings[2].value is not None


# =============================================================================
# Value Type Tests (Parametrized)
# =============================================================================


@pytest.mark.real
class TestBackendValueTypes:
    """Tests for reading all supported value types (all backends)."""

    @pytest.mark.parametrize("drf,expected_type,python_type,desc", DEVICE_TYPES)
    def test_get_value_type(self, read_backend, drf, expected_type, python_type, desc):
        """get() returns correct value_type for {desc}."""
        reading = read_backend.get(drf, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {drf}: {reading.message}"
        assert reading.value_type == expected_type, f"Expected {expected_type}, got {reading.value_type}"

        if python_type is not None:
            assert isinstance(reading.value, python_type), f"Expected {python_type}, got {type(reading.value)}"

        # Array length check
        if expected_type == ValueType.SCALAR_ARRAY:
            assert hasattr(reading.value, "__len__")
            assert len(reading.value) >= 2

    def test_get_many_all_types(self, read_backend):
        """get_many() reads all value types in one batch."""
        devices = [d[0] for d in DEVICE_TYPES]
        readings = read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == len(devices)
        for i, (drf, expected_type, _, _) in enumerate(DEVICE_TYPES):
            assert readings[i].ok, f"Failed: {drf}: {readings[i].message}"
            assert readings[i].value_type == expected_type


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.real
class TestBackendErrors:
    """Tests for error handling (all backends)."""

    def test_read_nonexistent_raises(self, read_backend):
        """read() raises DeviceError for nonexistent device."""
        with pytest.raises(DeviceError) as exc_info:
            read_backend.read(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert exc_info.value.error_code != 0

    def test_get_nonexistent_returns_error(self, read_backend):
        """get() returns error Reading for nonexistent device."""
        reading = read_backend.get(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code != 0

    def test_get_noprop_error(self, read_backend):
        """get() returns error for missing property."""
        reading = read_backend.get(NOPROP_DEVICE, timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code < 0

    def test_get_many_partial_failure(self, read_backend):
        """get_many() handles mix of success and error."""
        devices = [SCALAR_DEVICE, NONEXISTENT_DEVICE, SCALAR_DEVICE_2]
        readings = read_backend.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 3
        assert readings[0].ok
        assert not readings[1].ok
        assert readings[2].ok


# =============================================================================
# Mixed Event Types Tests
# =============================================================================


@pytest.mark.real
class TestBackendMixedEvents:
    """Tests for reading devices with different event types in one batch."""

    def test_get_many_mixed_periodic_and_clock_event(self, read_backend):
        """get_many() returns first reading per device for mixed event types.

        Tests that when combining a fast periodic (@p,100) with a slow clock
        event (@e,02 = TCLK event 02, fires every ~4-5s), we:
        1. Get the first reading for each device
        2. Don't wait for additional periodic updates while waiting for clock event
        3. Return as soon as all devices have one reading

        Timing varies depending on when the clock event fires (0-5s wait).
        """
        devices = [f"{SCALAR_DEVICE}@p,100", f"{SCALAR_DEVICE_2}@e,02"]

        readings = read_backend.get_many(devices, timeout=10.0)

        # Both readings should be present and valid
        assert len(readings) == 2
        assert readings[0].ok, f"Periodic device failed: {readings[0].message}"
        assert readings[1].ok, f"Clock event device failed: {readings[1].message}"

    def test_get_many_all_periodic_same_rate(self, read_backend):
        """get_many() with multiple periodic devices at same rate gets one reading each."""
        devices = [f"{SCALAR_DEVICE}@p,100", f"{SCALAR_DEVICE_2}@p,100"]

        readings = read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        assert all(r.ok for r in readings), f"Failures: {[r.message for r in readings if not r.ok]}"

    def test_get_many_immediate_with_periodic(self, read_backend):
        """get_many() with @I (immediate) and @p (periodic) devices."""
        devices = [f"{SCALAR_DEVICE}@I", f"{SCALAR_DEVICE_2}@p,500"]

        readings = read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        assert readings[0].ok, f"Immediate device failed: {readings[0].message}"
        assert readings[1].ok, f"Periodic device failed: {readings[1].message}"


# =============================================================================
# Streaming Tests
# =============================================================================


@pytest.mark.real
@pytest.mark.streaming
class TestBackendStreaming:
    """Streaming tests that work across all backends."""

    def _skip_if_no_stream(self, backend):
        if BackendCapability.STREAM not in backend.capabilities:
            pytest.skip("Backend does not support streaming")

    def test_subscribe_callback_mode(self, read_backend):
        """subscribe() with callback receives readings."""
        self._skip_if_no_stream(read_backend)

        readings = []
        event = threading.Event()

        def on_reading(reading: Reading, handle: SubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 1:
                event.set()

        handle = read_backend.subscribe([PERIODIC_DEVICE], callback=on_reading)
        try:
            event.wait(timeout=TIMEOUT_STREAM_EVENT)
        finally:
            handle.stop()

        assert len(readings) >= 1
        assert all(r.ok for r in readings)

    def test_subscribe_iterator_mode(self, read_backend):
        """subscribe() without callback enables iterator mode."""
        self._skip_if_no_stream(read_backend)

        with read_backend.subscribe([PERIODIC_DEVICE]) as handle:
            readings = []
            for reading, h in handle.readings(timeout=TIMEOUT_STREAM_ITER):
                readings.append(reading)
                if len(readings) >= 1:
                    break

        assert len(readings) >= 1
        assert handle.stopped

    def test_handle_stop_ends_subscription(self, read_backend):
        """handle.stop() stops receiving data."""
        self._skip_if_no_stream(read_backend)

        readings = []

        def on_reading(reading: Reading, handle: SubscriptionHandle):
            readings.append(reading)

        handle = read_backend.subscribe([FAST_PERIODIC], callback=on_reading)
        time.sleep(1.0)
        count_before = len(readings)
        handle.stop()
        time.sleep(1.5)
        count_after = len(readings)

        # Allow 1-2 in-flight readings after stop
        assert count_after <= count_before + 2

    def test_multiple_subscriptions(self, read_backend):
        """Multiple independent subscriptions work."""
        self._skip_if_no_stream(read_backend)

        h1 = read_backend.subscribe([SLOW_PERIODIC], callback=lambda r, h: None)
        h2 = read_backend.subscribe(["G:AMANDA@p,1000"], callback=lambda r, h: None)

        time.sleep(0.5)
        h1.stop()

        assert h1.stopped
        assert not h2.stopped

        h2.stop()

    def test_backend_close_stops_all(self, read_backend):
        """Backend.close() stops all subscriptions."""
        self._skip_if_no_stream(read_backend)

        handles = []
        for _ in range(2):
            handles.append(read_backend.subscribe([PERIODIC_DEVICE], callback=lambda r, h: None))
        time.sleep(0.5)

        read_backend.close()

        for h in handles:
            assert h.stopped


# =============================================================================
# Write Tests (shared across write-capable backends)
# =============================================================================


@pytest.mark.real
@pytest.mark.kerberos
@requires_kerberos
class TestBackendWrite:
    """Write tests that run against all write-capable backends (DPM HTTP, DMQ)."""

    def test_write_capabilities(self, write_backend):
        """Backend reports WRITE capability when auth is set."""
        assert BackendCapability.WRITE in write_backend.capabilities
        assert BackendCapability.AUTH_KERBEROS in write_backend.capabilities

    @pytest.mark.write
    @requires_write_enabled
    def test_write_scalar(self, write_backend):
        """Write a different value, verify readback, then restore original."""
        read_drf = strip_event(SCALAR_SETPOINT)
        original = write_backend.read(read_drf, timeout=TIMEOUT_READ)

        new_value = original + 0.1
        result = write_backend.write(SCALAR_SETPOINT, new_value, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(1.0)
        readback = write_backend.read(read_drf, timeout=TIMEOUT_READ)
        assert abs(readback - new_value) < 0.01, f"Write did not take effect: wrote {new_value}, read back {readback}"

        # Restore original value
        result2 = write_backend.write(SCALAR_SETPOINT, original, timeout=TIMEOUT_READ)
        assert result2.success

        time.sleep(1.0)
        restored = write_backend.read(read_drf, timeout=TIMEOUT_READ)
        assert abs(restored - original) < 0.01, f"Restore failed: wrote {original}, read back {restored}"

    @pytest.mark.write
    @requires_write_enabled
    def test_write_changes_raw(self, write_backend):
        """Write a scaled value and verify the .RAW readback changes accordingly."""
        original_scaled = write_backend.read(strip_event(SCALAR_SETPOINT), timeout=TIMEOUT_READ)
        original_raw = write_backend.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert isinstance(original_raw, bytes) and len(original_raw) > 0

        new_value = original_scaled + 1.0
        result = write_backend.write(SCALAR_SETPOINT, new_value, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(1.0)
        new_raw = write_backend.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert new_raw != original_raw, (
            f"Raw bytes unchanged after scaled write: {original_raw.hex()} -> {new_raw.hex()}"
        )

        # Restore
        result2 = write_backend.write(SCALAR_SETPOINT, original_scaled, timeout=TIMEOUT_READ)
        assert result2.success

        time.sleep(1.0)
        restored_raw = write_backend.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert restored_raw == original_raw, f"Restore failed: expected {original_raw.hex()}, got {restored_raw.hex()}"

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.parametrize(
        "cmd_true,cmd_false,field", CONTROL_PAIRS, ids=lambda x: x if isinstance(x, str) else x.name
    )
    def test_control_pair(self, write_backend, cmd_true, cmd_false, field):
        """Toggle control pair and verify the corresponding status bit changes."""
        reading = write_backend.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert reading.ok, f"Failed to read status: {reading.message}"
        initial = reading.value.get(field)

        # Set TRUE
        result = write_backend.write(STATUS_CONTROL_DEVICE, cmd_true, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(1.0)
        status = write_backend.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after {cmd_true.name}: {status.message}"
        assert status.value.get(field) is True, f"Expected {field}=True after {cmd_true.name}"

        # Set FALSE
        result = write_backend.write(STATUS_CONTROL_DEVICE, cmd_false, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(1.0)
        status = write_backend.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after {cmd_false.name}: {status.message}"
        assert status.value.get(field) is False, f"Expected {field}=False after {cmd_false.name}"

        # Restore
        restore = cmd_true if initial else cmd_false
        write_backend.write(STATUS_CONTROL_DEVICE, restore, timeout=TIMEOUT_READ)

    @pytest.mark.write
    @requires_write_enabled
    def test_control_reset(self, write_backend):
        """RESET command succeeds and produces a valid status."""
        result = write_backend.write(STATUS_CONTROL_DEVICE, CONTROL_RESET, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(1.0)
        status = write_backend.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after RESET: {status.message}"
        assert isinstance(status.value, dict)
        assert "on" in status.value


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
