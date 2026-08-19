"""
Shared async tests that run against all async read-capable backends.

Mirrors test_backend_shared.py for async backends (AsyncDPMHTTPBackend, AsyncGRPCBackend).
"""

import asyncio
import math

import pytest

from pacsys.aio._backends import AsyncBackend
from pacsys.aio._subscription import AsyncSubscriptionHandle
from pacsys.drf3 import parse_request
from pacsys.drf_utils import strip_event
from pacsys.errors import DeviceError
from pacsys.types import BackendCapability, BasicControl, Reading, ValueType

from .devices import (
    ACLTST_NONEXISTENT_ORDINAL,
    ACLTST_UNPAIRED_CONTROLS,
    ARRAY_DEVICE,
    CONTROL_PAIRS,
    CONTROL_RESET,
    DEVICE_TYPES,
    FAST_PERIODIC,
    NONEXISTENT_DEVICE,
    NOPROP_DEVICE,
    PERIODIC_DEVICE,
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_ELEMENT,
    SCALAR_SETPOINT,
    SCALAR_SETPOINT_RAW,
    SETTING_ON_READONLY,
    SLOW_PERIODIC,
    STATUS_CONTROL_DEVICE,
    TIMEOUT_BATCH,
    TIMEOUT_READ,
    TIMEOUT_STREAM_EVENT,
    TIMEOUT_STREAM_ITER,
    requires_kerberos,
    requires_write_enabled,
)

# =============================================================================
# Connection Tests
# =============================================================================


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncBackendConnection:
    """Tests for async backend connection lifecycle."""

    async def test_backend_has_read_capability(self, async_read_backend_cls: AsyncBackend):
        assert BackendCapability.READ in async_read_backend_cls.capabilities

    async def test_backend_not_closed_initially(self, async_read_backend_cls: AsyncBackend):
        assert not async_read_backend_cls._closed


# =============================================================================
# Basic Read Tests
# =============================================================================


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncBackendRead:
    """Tests for basic async read operations."""

    async def test_get_scalar(self, async_read_backend_cls: AsyncBackend):
        reading = await async_read_backend_cls.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_DEVICE}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR
        assert isinstance(reading.value, float)
        assert math.isfinite(reading.value)
        assert reading.timestamp is not None

    async def test_get_second_device(self, async_read_backend_cls: AsyncBackend):
        reading = await async_read_backend_cls.get(SCALAR_DEVICE_2, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_DEVICE_2}: {reading.message}"
        assert isinstance(reading.value, float)
        assert math.isfinite(reading.value)

    async def test_get_array(self, async_read_backend_cls: AsyncBackend):
        reading = await async_read_backend_cls.get(ARRAY_DEVICE, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {ARRAY_DEVICE}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR_ARRAY
        assert hasattr(reading.value, "__len__")
        assert len(reading.value) >= 2

    async def test_get_array_element(self, async_read_backend_cls: AsyncBackend):
        reading = await async_read_backend_cls.get(SCALAR_ELEMENT, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_ELEMENT}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR
        assert isinstance(reading.value, (int, float))
        assert not isinstance(reading.value, bool)
        assert math.isfinite(reading.value)


# =============================================================================
# Batch Read Tests
# =============================================================================


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncBackendBatchReads:
    """Tests for async batch read operations."""

    async def test_get_many(self, async_read_backend_cls: AsyncBackend):
        devices = [SCALAR_DEVICE, SCALAR_DEVICE_2]
        readings = await async_read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        for i, reading in enumerate(readings):
            assert reading.ok, f"Failed: {devices[i]}: {reading.message}"
            assert isinstance(reading.value, float)
            assert math.isfinite(reading.value)

    async def test_get_many_empty_list(self, async_read_backend_cls: AsyncBackend):
        readings = await async_read_backend_cls.get_many([], timeout=TIMEOUT_READ)
        assert readings == []

    async def test_get_many_order_preserved(self, async_read_backend_cls: AsyncBackend):
        devices = [SCALAR_DEVICE_2, SCALAR_DEVICE, SCALAR_ELEMENT]
        readings = await async_read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 3
        assert all(r.ok for r in readings), f"Failures: {[r.message for r in readings if not r.ok]}"

    async def test_get_many_duplicate_drf(self, async_read_backend_cls: AsyncBackend):
        devices = [SCALAR_DEVICE, SCALAR_DEVICE_2, SCALAR_DEVICE]
        readings = await async_read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 3
        assert readings[0].ok
        assert readings[1].ok
        assert readings[2].ok
        assert readings[0].value is not None
        assert readings[2].value is not None


# =============================================================================
# Value Type Tests
# =============================================================================


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncBackendValueTypes:
    """Tests for reading all supported value types."""

    @staticmethod
    def _is_grpc(backend) -> bool:
        from pacsys.aio._grpc import AsyncGRPCBackend

        return isinstance(backend, AsyncGRPCBackend)

    @pytest.mark.parametrize(("drf", "expected_type", "python_type", "desc"), DEVICE_TYPES)
    async def test_get_value_type(self, async_read_backend_cls: AsyncBackend, drf, expected_type, python_type, desc):
        """get() returns correct value_type for {desc}."""
        reading = await async_read_backend_cls.get(drf, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {drf}: {reading.message}"
        assert reading.value_type == expected_type, f"Expected {expected_type}, got {reading.value_type}"

        if python_type is not None:
            assert isinstance(reading.value, python_type), f"Expected {python_type}, got {type(reading.value)}"

        if expected_type == ValueType.SCALAR_ARRAY:
            assert hasattr(reading.value, "__len__")
            assert len(reading.value) >= 2

    async def test_get_many_all_types(self, async_read_backend_cls: AsyncBackend):
        """get_many() reads all value types in one batch."""
        if self._is_grpc(async_read_backend_cls):
            pytest.skip("Mixed-type batches of this size are not supported by the current DPM gRPC service")

        devices = [d[0] for d in DEVICE_TYPES]
        readings = await async_read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == len(devices)
        for i, (drf, expected_type, _, _) in enumerate(DEVICE_TYPES):
            assert readings[i].ok, f"Failed: {drf}: {readings[i].message}"
            assert readings[i].value_type == expected_type


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncBackendErrors:
    """Tests for async error handling."""

    async def test_read_nonexistent_raises(self, async_read_backend_cls: AsyncBackend):
        """read() raises DeviceError for nonexistent device."""
        with pytest.raises(DeviceError) as exc_info:
            await async_read_backend_cls.read(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert exc_info.value.error_code != 0

    async def test_get_nonexistent_returns_error(self, async_read_backend_cls: AsyncBackend):
        """get() returns error Reading for nonexistent device."""
        reading = await async_read_backend_cls.get(NONEXISTENT_DEVICE, timeout=10.0)
        assert not reading.ok
        assert reading.error_code != 0

    async def test_get_noprop_error(self, async_read_backend_cls: AsyncBackend):
        """get() returns error for missing property."""
        reading = await async_read_backend_cls.get(NOPROP_DEVICE, timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code < 0

    async def test_read_setting_on_readonly_raises(self, async_read_backend_cls: AsyncBackend):
        """read() raises DeviceError for SETTING on read-only device."""
        with pytest.raises(DeviceError):
            await async_read_backend_cls.read(SETTING_ON_READONLY, timeout=TIMEOUT_READ)

    async def test_get_many_partial_failure(self, async_read_backend_cls: AsyncBackend):
        """get_many() handles mix of success and error."""
        devices = [SCALAR_DEVICE, NONEXISTENT_DEVICE, SCALAR_DEVICE_2]
        readings = await async_read_backend_cls.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 3
        assert readings[0].ok
        assert not readings[1].ok
        assert readings[2].ok


# =============================================================================
# Mixed Event Types Tests
# =============================================================================


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncBackendMixedEvents:
    """Tests for reading devices with different event types in one async batch."""

    async def test_get_many_mixed_periodic_and_clock_event(self, async_read_backend_cls: AsyncBackend):
        """get_many() returns first reading per device for mixed event types."""
        devices = [f"{SCALAR_DEVICE}@p,100", f"{SCALAR_DEVICE_2}@e,0C"]

        readings = await async_read_backend_cls.get_many(devices, timeout=10.0)

        assert len(readings) == 2
        assert readings[0].ok, f"Periodic device failed: {readings[0].message}"
        assert readings[1].ok, f"Clock event device failed: {readings[1].message}"

    async def test_get_many_all_periodic_same_rate(self, async_read_backend_cls: AsyncBackend):
        """get_many() with multiple periodic devices at same rate."""
        devices = [f"{SCALAR_DEVICE}@p,100", f"{SCALAR_DEVICE_2}@p,100"]

        readings = await async_read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        assert all(r.ok for r in readings), f"Failures: {[r.message for r in readings if not r.ok]}"

    async def test_get_many_immediate_with_periodic(self, async_read_backend_cls: AsyncBackend):
        """get_many() with @I (immediate) and @p (periodic) devices."""
        devices = [f"{SCALAR_DEVICE}@I", f"{SCALAR_DEVICE_2}@p,500"]

        readings = await async_read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        assert readings[0].ok, f"Immediate device failed: {readings[0].message}"
        assert readings[1].ok, f"Periodic device failed: {readings[1].message}"

    async def test_get_many_same_device_periodic_and_clock(self, async_read_backend_cls: AsyncBackend):
        """Same device at periodic and clock event returns two distinct readings."""
        devices = [f"{SCALAR_DEVICE}@p,500", f"{SCALAR_DEVICE}@e,0C"]
        readings = await async_read_backend_cls.get_many(devices, timeout=10.0)

        assert len(readings) == 2
        assert readings[0].ok, f"Periodic failed: {readings[0].message}"
        assert readings[1].ok, f"Clock event failed: {readings[1].message}"
        assert "@p" in readings[0].drf.lower(), f"Expected periodic event in drf: {readings[0].drf}"
        assert "@e" in readings[1].drf.lower(), f"Expected clock event in drf: {readings[1].drf}"

    async def test_get_many_same_device_two_clock_events(self, async_read_backend_cls: AsyncBackend):
        """Same device at two different clock events (0C and 0F)."""
        devices = [f"{SCALAR_DEVICE}@e,0C", f"{SCALAR_DEVICE}@e,0F"]
        readings = await async_read_backend_cls.get_many(devices, timeout=10.0)

        assert len(readings) == 2
        assert readings[0].ok, f"Event 0C failed: {readings[0].message}"
        assert readings[1].ok, f"Event 0F failed: {readings[1].message}"
        assert "0C" in readings[0].drf, f"Expected event 0C in drf: {readings[0].drf}"
        assert "0F" in readings[1].drf, f"Expected event 0F in drf: {readings[1].drf}"


# =============================================================================
# Streaming Tests
# =============================================================================


def _skip_if_no_stream(backend):
    if BackendCapability.STREAM not in backend.capabilities:
        pytest.skip("Backend does not support streaming")


@pytest.mark.streaming
@pytest.mark.asyncio(loop_scope="class")
class TestAsyncBackendStreaming:
    """Async streaming tests across all backends."""

    async def test_subscribe_callback_mode(self, async_read_backend_cls: AsyncBackend):
        """subscribe() with sync callback receives readings."""
        _skip_if_no_stream(async_read_backend_cls)

        readings: list[Reading] = []
        event = asyncio.Event()

        def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 1:
                event.set()

        handle = await async_read_backend_cls.subscribe([PERIODIC_DEVICE], callback=on_reading)
        try:
            await asyncio.wait_for(event.wait(), timeout=TIMEOUT_STREAM_EVENT)
        finally:
            await handle.stop()

        assert len(readings) >= 1
        assert all(r.ok for r in readings)

    async def test_subscribe_async_callback(self, async_read_backend_cls: AsyncBackend):
        """subscribe() with async callback receives readings."""
        _skip_if_no_stream(async_read_backend_cls)

        readings: list[Reading] = []
        event = asyncio.Event()

        async def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 1:
                event.set()

        handle = await async_read_backend_cls.subscribe([PERIODIC_DEVICE], callback=on_reading)
        try:
            await asyncio.wait_for(event.wait(), timeout=TIMEOUT_STREAM_EVENT)
        finally:
            await handle.stop()

        assert len(readings) >= 1
        assert all(r.ok for r in readings)

    async def test_subscribe_iterator_mode(self, async_read_backend_cls: AsyncBackend):
        """subscribe() without callback enables async iterator mode."""
        _skip_if_no_stream(async_read_backend_cls)

        readings: list[Reading] = []
        handle = await async_read_backend_cls.subscribe([PERIODIC_DEVICE])
        async with handle:
            async for reading, h in handle.readings(timeout=TIMEOUT_STREAM_ITER):
                readings.append(reading)
                if len(readings) >= 1:
                    break

        assert len(readings) >= 1
        assert handle.stopped

    async def test_handle_stop_ends_subscription(self, async_read_backend_cls: AsyncBackend):
        """handle.stop() stops receiving data."""
        _skip_if_no_stream(async_read_backend_cls)

        readings: list[Reading] = []

        def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)

        handle = await async_read_backend_cls.subscribe([FAST_PERIODIC], callback=on_reading)
        await asyncio.sleep(0.5)
        count_before = len(readings)
        await handle.stop()
        await asyncio.sleep(0.5)
        count_after = len(readings)

        # Allow 1-2 in-flight readings after stop
        assert count_after <= count_before + 2

    async def test_multiple_subscriptions(self, async_read_backend_cls: AsyncBackend):
        """Multiple independent subscriptions work."""
        _skip_if_no_stream(async_read_backend_cls)

        h1 = await async_read_backend_cls.subscribe([SLOW_PERIODIC], callback=lambda r, h: None)
        h2 = await async_read_backend_cls.subscribe(["G:AMANDA@p,1000"], callback=lambda r, h: None)

        await asyncio.sleep(0.1)
        await h1.stop()

        assert h1.stopped
        assert not h2.stopped

        await h2.stop()

    async def test_subscribe_mixed_valid_and_invalid(self, async_read_backend_cls: AsyncBackend):
        """Subscription with valid + nonexistent device delivers both data and errors."""
        _skip_if_no_stream(async_read_backend_cls)

        readings_by_drf: dict[str, list[Reading]] = {}
        lock = asyncio.Lock()
        got_both = asyncio.Event()

        valid_drf = PERIODIC_DEVICE
        invalid_drf = f"{NONEXISTENT_DEVICE}@p,500"

        async def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            async with lock:
                readings_by_drf.setdefault(reading.drf, []).append(reading)
                if len(readings_by_drf) >= 2:
                    got_both.set()

        handle = await async_read_backend_cls.subscribe([valid_drf, invalid_drf], callback=on_reading)
        try:
            await asyncio.wait_for(got_both.wait(), timeout=10.0)
        finally:
            await handle.stop()

        async with lock:
            valid_readings = readings_by_drf.get(valid_drf, [])
            assert len(valid_readings) >= 1, f"Expected readings for {valid_drf}"
            assert valid_readings[0].ok

            invalid_readings = [r for drf, rs in readings_by_drf.items() if drf != valid_drf for r in rs]
            assert len(invalid_readings) >= 1, (
                f"Expected error reading for invalid device, got drfs: {list(readings_by_drf)}"
            )
            assert invalid_readings[0].is_error or invalid_readings[0].is_warning

    async def test_subscribe_invalid_device_reports_error(self, async_read_backend_cls: AsyncBackend):
        """Subscription to nonexistent device delivers error/warning (not silent)."""
        _skip_if_no_stream(async_read_backend_cls)

        readings: list[Reading] = []
        got_reading = asyncio.Event()

        def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)
            got_reading.set()

        handle = await async_read_backend_cls.subscribe([f"{NONEXISTENT_DEVICE}@p,500"], callback=on_reading)
        try:
            # DPM_PEND warning arrives with delay for nonexistent devices
            await asyncio.wait_for(got_reading.wait(), timeout=10.0)
        finally:
            await handle.stop()

        assert len(readings) >= 1, "Expected at least one error/warning reading for nonexistent device"
        assert readings[0].is_error or readings[0].is_warning


@pytest.mark.streaming
class TestAsyncBackendCloseStopsAll:
    """Test that Backend.close() stops all subscriptions (needs own backend)."""

    @pytest.mark.asyncio
    async def test_backend_close_stops_all(self, async_read_backend: AsyncBackend):
        """Backend.close() stops all subscriptions."""
        _skip_if_no_stream(async_read_backend)

        handles = [await async_read_backend.subscribe([PERIODIC_DEVICE], callback=lambda r, h: None) for _ in range(2)]
        await asyncio.sleep(0.1)

        await async_read_backend.close()

        for h in handles:
            assert h.stopped


# =============================================================================
# Write Tests
# =============================================================================


@pytest.mark.kerberos
@requires_kerberos
@pytest.mark.asyncio(loop_scope="class")
class TestAsyncBackendWrite:
    """Async write tests (DPM HTTP only)."""

    @pytest.mark.write
    @requires_write_enabled
    async def test_write_scalar(self, async_write_backend_cls: AsyncBackend):
        """Write a different value, verify readback, then restore original."""
        read_drf = strip_event(SCALAR_SETPOINT)
        original = await async_write_backend_cls.read(read_drf, timeout=TIMEOUT_READ)

        new_value = original + 0.1
        result = await async_write_backend_cls.write(SCALAR_SETPOINT, new_value, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(0.5)
        readback = await async_write_backend_cls.read(read_drf, timeout=TIMEOUT_READ)
        assert abs(readback - new_value) < 0.01, f"Write did not take effect: wrote {new_value}, read back {readback}"

        # Restore original value
        result2 = await async_write_backend_cls.write(SCALAR_SETPOINT, original, timeout=TIMEOUT_READ)
        assert result2.success

        await asyncio.sleep(0.5)
        restored = await async_write_backend_cls.read(read_drf, timeout=TIMEOUT_READ)
        assert abs(restored - original) < 0.01, f"Restore failed: wrote {original}, read back {restored}"

    @pytest.mark.write
    @requires_write_enabled
    async def test_write_changes_raw(self, async_write_backend_cls: AsyncBackend):
        """Write a scaled value and verify the .RAW readback changes."""
        original_scaled = await async_write_backend_cls.read(strip_event(SCALAR_SETPOINT), timeout=TIMEOUT_READ)
        original_raw = await async_write_backend_cls.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert isinstance(original_raw, bytes)
        assert len(original_raw) > 0

        new_value = original_scaled + 1.0
        result = await async_write_backend_cls.write(SCALAR_SETPOINT, new_value, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(0.5)
        new_raw = await async_write_backend_cls.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert new_raw != original_raw, (
            f"Raw bytes unchanged after scaled write: {original_raw.hex()} -> {new_raw.hex()}"
        )

        # Restore
        result2 = await async_write_backend_cls.write(SCALAR_SETPOINT, original_scaled, timeout=TIMEOUT_READ)
        assert result2.success

        await asyncio.sleep(0.5)
        restored_raw = await async_write_backend_cls.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert restored_raw == original_raw, f"Restore failed: expected {original_raw.hex()}, got {restored_raw.hex()}"

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.parametrize(
        ("cmd_true", "cmd_false", "field"), CONTROL_PAIRS, ids=lambda x: x if isinstance(x, str) else x.name
    )
    async def test_control_pair(self, async_write_backend_cls: AsyncBackend, cmd_true, cmd_false, field):
        """Toggle control pair and verify the corresponding status bit changes."""
        reading = await async_write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert reading.ok, f"Failed to read status: {reading.message}"
        initial = reading.value.get(field)

        # Set TRUE
        result = await async_write_backend_cls.write(STATUS_CONTROL_DEVICE, cmd_true, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(0.5)
        status = await async_write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after {cmd_true.name}: {status.message}"
        assert status.value.get(field) is True, f"Expected {field}=True after {cmd_true.name}"

        # Set FALSE
        result = await async_write_backend_cls.write(STATUS_CONTROL_DEVICE, cmd_false, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(0.5)
        status = await async_write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after {cmd_false.name}: {status.message}"
        assert status.value.get(field) is False, f"Expected {field}=False after {cmd_false.name}"

        # Restore
        restore = cmd_true if initial else cmd_false
        await async_write_backend_cls.write(STATUS_CONTROL_DEVICE, restore, timeout=TIMEOUT_READ)

    @pytest.mark.write
    @requires_write_enabled
    async def test_control_reset(self, async_write_backend_cls: AsyncBackend):
        """RESET command succeeds and produces a valid status."""
        result = await async_write_backend_cls.write(STATUS_CONTROL_DEVICE, CONTROL_RESET, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(0.5)
        status = await async_write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after RESET: {status.message}"
        assert isinstance(status.value, dict)
        assert "on" in status.value

    @pytest.mark.write
    @requires_write_enabled
    async def test_control_mixed(self, async_write_backend_cls: AsyncBackend):
        """Mixed R/W/stream interleave stress test."""
        readings: list[Reading] = []
        event = asyncio.Event()

        result = await async_write_backend_cls.write(STATUS_CONTROL_DEVICE, BasicControl.NEGATIVE, timeout=TIMEOUT_READ)
        assert result.success

        async def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 3:
                r = await async_write_backend_cls.write(
                    STATUS_CONTROL_DEVICE, BasicControl.POSITIVE, timeout=TIMEOUT_READ
                )
                assert r.success
                event.set()

        valid_drf = PERIODIC_DEVICE
        invalid_drf = f"{NONEXISTENT_DEVICE}@p,500"
        handle = await async_write_backend_cls.subscribe([valid_drf, invalid_drf], callback=on_reading)
        h1 = await async_write_backend_cls.subscribe([SCALAR_DEVICE + "@p,500"])

        await asyncio.sleep(0.1)

        read1 = await async_write_backend_cls.read(SCALAR_DEVICE_2)
        assert isinstance(read1, float)
        assert math.isfinite(read1)
        read2 = await async_write_backend_cls.get_many([SCALAR_DEVICE_2, SCALAR_ELEMENT, ARRAY_DEVICE])
        assert all(r.ok for r in read2)
        for _ in range(5):
            read3 = await async_write_backend_cls.get_many(
                [SCALAR_DEVICE, SCALAR_DEVICE_2, SCALAR_ELEMENT, SCALAR_SETPOINT]
            )
            assert all(r.ok for r in read3)
            w = await async_write_backend_cls.write(STATUS_CONTROL_DEVICE, BasicControl.NEGATIVE, timeout=TIMEOUT_READ)
            assert w.ok

        await asyncio.sleep(1.0)

        try:
            await asyncio.wait_for(event.wait(), timeout=TIMEOUT_STREAM_EVENT)
        finally:
            await handle.stop()

        valid_name = parse_request(valid_drf).device
        invalid_name = parse_request(invalid_drf).device
        valid_readings = [r for r in readings if r.name == valid_name]
        invalid_readings = [r for r in readings if r.name == invalid_name]
        assert valid_readings and all(r.ok for r in valid_readings)
        # The dedicated mixed-subscription test waits for delayed DPM_PEND.
        assert all(not r.ok for r in invalid_readings)

        j = 0
        async for r, _ in h1.readings(timeout=TIMEOUT_STREAM_ITER):
            j += 1
            if j > 3:
                await h1.stop()

        status = await async_write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status: {status.message}"
        assert status.value["positive"] is True


# =============================================================================
# Unpaired / Nonexistent Control Ordinals (Z:ACLTST)
# =============================================================================


@pytest.mark.kerberos
@requires_kerberos
@pytest.mark.asyncio(loop_scope="class")
class TestAsyncBackendUnpairedControls:
    """Unpaired control ordinals on Z:ACLTST (async write backend).

    Ordinals 7-11 map to device-specific TEST commands that succeed but don't
    toggle standard status bits. Ordinal 25 is beyond the device's control
    table and should be rejected.
    """

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.parametrize(
        ("ordinal", "cmd_name"),
        ACLTST_UNPAIRED_CONTROLS,
        ids=[name for _, name in ACLTST_UNPAIRED_CONTROLS],
    )
    async def test_unpaired_control_write_succeeds(self, async_write_backend_cls: AsyncBackend, ordinal, cmd_name):
        """Unpaired ordinal write is accepted without error."""
        result = await async_write_backend_cls.write(STATUS_CONTROL_DEVICE, ordinal, timeout=TIMEOUT_READ)
        assert result.success, f"Ordinal {ordinal} ({cmd_name}) write failed: {result.error_code} {result.message}"

    @pytest.mark.write
    @requires_write_enabled
    async def test_nonexistent_ordinal_rejected(self, async_write_backend_cls: AsyncBackend):
        """Ordinal beyond device's control table is rejected (DIO_SCALEFAIL)."""
        result = await async_write_backend_cls.write(
            STATUS_CONTROL_DEVICE, ACLTST_NONEXISTENT_ORDINAL, timeout=TIMEOUT_READ
        )
        assert not result.success, f"Expected error for ordinal {ACLTST_NONEXISTENT_ORDINAL}, but write succeeded"
