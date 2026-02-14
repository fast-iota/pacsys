"""
Shared async tests that run against all async read-capable backends.

Mirrors test_backend_shared.py for async backends (AsyncDPMHTTPBackend, AsyncGRPCBackend).

Run with: pytest tests/real/test_async_backend_shared.py -v -s -o "addopts="
"""

import asyncio

import pytest

from pacsys.aio._backends import AsyncBackend
from pacsys.errors import DeviceError
from pacsys.types import BackendCapability, ValueType

from pacsys.aio._subscription import AsyncSubscriptionHandle
from pacsys.types import Reading

from pacsys.drf_utils import strip_event

from pacsys.types import BasicControl

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


@pytest.mark.real
class TestAsyncBackendConnection:
    """Tests for async backend connection lifecycle."""

    @pytest.mark.asyncio
    async def test_backend_has_read_capability(self, async_read_backend: AsyncBackend):
        assert BackendCapability.READ in async_read_backend.capabilities

    @pytest.mark.asyncio
    async def test_backend_not_closed_initially(self, async_read_backend: AsyncBackend):
        assert not async_read_backend._closed


# =============================================================================
# Basic Read Tests
# =============================================================================


@pytest.mark.real
class TestAsyncBackendRead:
    """Tests for basic async read operations."""

    @pytest.mark.asyncio
    async def test_get_scalar(self, async_read_backend: AsyncBackend):
        reading = await async_read_backend.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_DEVICE}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR
        assert isinstance(reading.value, float)
        assert reading.timestamp is not None

    @pytest.mark.asyncio
    async def test_get_second_device(self, async_read_backend: AsyncBackend):
        reading = await async_read_backend.get(SCALAR_DEVICE_2, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_DEVICE_2}: {reading.message}"
        assert isinstance(reading.value, float)

    @pytest.mark.asyncio
    async def test_get_array(self, async_read_backend: AsyncBackend):
        reading = await async_read_backend.get(ARRAY_DEVICE, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {ARRAY_DEVICE}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR_ARRAY
        assert hasattr(reading.value, "__len__")
        assert len(reading.value) >= 2

    @pytest.mark.asyncio
    async def test_get_array_element(self, async_read_backend: AsyncBackend):
        reading = await async_read_backend.get(SCALAR_ELEMENT, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_ELEMENT}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR
        assert isinstance(reading.value, (int, float))


# =============================================================================
# Batch Read Tests
# =============================================================================


@pytest.mark.real
class TestAsyncBackendBatchReads:
    """Tests for async batch read operations."""

    @pytest.mark.asyncio
    async def test_get_many(self, async_read_backend: AsyncBackend):
        devices = [SCALAR_DEVICE, SCALAR_DEVICE_2]
        readings = await async_read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        for i, reading in enumerate(readings):
            assert reading.ok, f"Failed: {devices[i]}: {reading.message}"
            assert isinstance(reading.value, float)

    @pytest.mark.asyncio
    async def test_get_many_empty_list(self, async_read_backend: AsyncBackend):
        readings = await async_read_backend.get_many([], timeout=TIMEOUT_READ)
        assert readings == []

    @pytest.mark.asyncio
    async def test_get_many_order_preserved(self, async_read_backend: AsyncBackend):
        devices = [SCALAR_DEVICE_2, SCALAR_DEVICE, SCALAR_ELEMENT]
        readings = await async_read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 3
        assert all(r.ok for r in readings), f"Failures: {[r.message for r in readings if not r.ok]}"

    @pytest.mark.asyncio
    async def test_get_many_duplicate_drf(self, async_read_backend: AsyncBackend):
        devices = [SCALAR_DEVICE, SCALAR_DEVICE_2, SCALAR_DEVICE]
        readings = await async_read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 3
        assert readings[0].ok
        assert readings[1].ok
        assert readings[2].ok
        assert readings[0].value is not None
        assert readings[2].value is not None


# =============================================================================
# Value Type Tests
# =============================================================================


@pytest.mark.real
class TestAsyncBackendValueTypes:
    """Tests for reading all supported value types."""

    @staticmethod
    def _is_grpc(backend) -> bool:
        from pacsys.aio._grpc import AsyncGRPCBackend

        return isinstance(backend, AsyncGRPCBackend)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("drf,expected_type,python_type,desc", DEVICE_TYPES)
    async def test_get_value_type(self, async_read_backend: AsyncBackend, drf, expected_type, python_type, desc):
        """get() returns correct value_type for {desc}."""
        reading = await async_read_backend.get(drf, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {drf}: {reading.message}"
        assert reading.value_type == expected_type, f"Expected {expected_type}, got {reading.value_type}"

        if python_type is not None:
            assert isinstance(reading.value, python_type), f"Expected {python_type}, got {type(reading.value)}"

        if expected_type == ValueType.SCALAR_ARRAY:
            assert hasattr(reading.value, "__len__")
            assert len(reading.value) >= 2

    @pytest.mark.asyncio
    async def test_get_many_all_types(self, async_read_backend: AsyncBackend):
        """get_many() reads all value types in one batch."""
        if self._is_grpc(async_read_backend):
            pytest.skip(
                "DPM gRPC server has unsynchronized StreamObserver.onNext() - "
                "concurrent ACNET callbacks corrupt HTTP/2 framing with 12+ mixed devices"
            )

        devices = [d[0] for d in DEVICE_TYPES]
        readings = await async_read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == len(devices)
        for i, (drf, expected_type, _, _) in enumerate(DEVICE_TYPES):
            assert readings[i].ok, f"Failed: {drf}: {readings[i].message}"
            assert readings[i].value_type == expected_type


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.real
class TestAsyncBackendErrors:
    """Tests for async error handling."""

    @pytest.mark.asyncio
    async def test_read_nonexistent_raises(self, async_read_backend: AsyncBackend):
        """read() raises DeviceError for nonexistent device."""
        with pytest.raises(DeviceError) as exc_info:
            await async_read_backend.read(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert exc_info.value.error_code != 0

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_error(self, async_read_backend: AsyncBackend):
        """get() returns error Reading for nonexistent device."""
        await asyncio.sleep(0.5)  # cooldown after prior error test
        reading = await async_read_backend.get(NONEXISTENT_DEVICE, timeout=10.0)
        assert not reading.ok
        assert reading.error_code != 0

    @pytest.mark.asyncio
    async def test_get_noprop_error(self, async_read_backend: AsyncBackend):
        """get() returns error for missing property."""
        reading = await async_read_backend.get(NOPROP_DEVICE, timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code < 0

    @pytest.mark.asyncio
    async def test_read_setting_on_readonly_raises(self, async_read_backend: AsyncBackend):
        """read() raises DeviceError for SETTING on read-only device."""
        with pytest.raises(DeviceError):
            await async_read_backend.read(SETTING_ON_READONLY, timeout=TIMEOUT_READ)

    @pytest.mark.asyncio
    async def test_get_many_partial_failure(self, async_read_backend: AsyncBackend):
        """get_many() handles mix of success and error."""
        devices = [SCALAR_DEVICE, NONEXISTENT_DEVICE, SCALAR_DEVICE_2]
        readings = await async_read_backend.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 3
        assert readings[0].ok
        assert not readings[1].ok
        assert readings[2].ok


# =============================================================================
# Mixed Event Types Tests
# =============================================================================


@pytest.mark.real
class TestAsyncBackendMixedEvents:
    """Tests for reading devices with different event types in one async batch."""

    @pytest.mark.asyncio
    async def test_get_many_mixed_periodic_and_clock_event(self, async_read_backend: AsyncBackend):
        """get_many() returns first reading per device for mixed event types."""
        devices = [f"{SCALAR_DEVICE}@p,100", f"{SCALAR_DEVICE_2}@e,02"]

        readings = await async_read_backend.get_many(devices, timeout=10.0)

        assert len(readings) == 2
        assert readings[0].ok, f"Periodic device failed: {readings[0].message}"
        assert readings[1].ok, f"Clock event device failed: {readings[1].message}"

    @pytest.mark.asyncio
    async def test_get_many_all_periodic_same_rate(self, async_read_backend: AsyncBackend):
        """get_many() with multiple periodic devices at same rate."""
        devices = [f"{SCALAR_DEVICE}@p,100", f"{SCALAR_DEVICE_2}@p,100"]

        readings = await async_read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        assert all(r.ok for r in readings), f"Failures: {[r.message for r in readings if not r.ok]}"

    @pytest.mark.asyncio
    async def test_get_many_immediate_with_periodic(self, async_read_backend: AsyncBackend):
        """get_many() with @I (immediate) and @p (periodic) devices."""
        devices = [f"{SCALAR_DEVICE}@I", f"{SCALAR_DEVICE_2}@p,500"]

        readings = await async_read_backend.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        assert readings[0].ok, f"Immediate device failed: {readings[0].message}"
        assert readings[1].ok, f"Periodic device failed: {readings[1].message}"

    @pytest.mark.asyncio
    async def test_get_many_same_device_periodic_and_clock(self, async_read_backend: AsyncBackend):
        """Same device at periodic and clock event returns two distinct readings."""
        devices = [f"{SCALAR_DEVICE}@p,500", f"{SCALAR_DEVICE}@e,02"]
        readings = await async_read_backend.get_many(devices, timeout=10.0)

        assert len(readings) == 2
        assert readings[0].ok, f"Periodic failed: {readings[0].message}"
        assert readings[1].ok, f"Clock event failed: {readings[1].message}"
        assert "@p" in readings[0].drf.lower(), f"Expected periodic event in drf: {readings[0].drf}"
        assert "@e" in readings[1].drf.lower(), f"Expected clock event in drf: {readings[1].drf}"

    @pytest.mark.asyncio
    async def test_get_many_same_device_two_clock_events(self, async_read_backend: AsyncBackend):
        """Same device at two different clock events (02 and 52)."""
        devices = [f"{SCALAR_DEVICE}@e,02", f"{SCALAR_DEVICE}@e,52"]
        readings = await async_read_backend.get_many(devices, timeout=10.0)

        assert len(readings) == 2
        assert readings[0].ok, f"Event 02 failed: {readings[0].message}"
        assert readings[1].ok, f"Event 52 failed: {readings[1].message}"
        assert "02" in readings[0].drf, f"Expected event 02 in drf: {readings[0].drf}"
        assert "52" in readings[1].drf, f"Expected event 52 in drf: {readings[1].drf}"


# =============================================================================
# Streaming Tests
# =============================================================================


@pytest.mark.real
@pytest.mark.streaming
class TestAsyncBackendStreaming:
    """Async streaming tests across all backends."""

    def _skip_if_no_stream(self, backend):
        if BackendCapability.STREAM not in backend.capabilities:
            pytest.skip("Backend does not support streaming")

    @pytest.mark.asyncio
    async def test_subscribe_callback_mode(self, async_read_backend: AsyncBackend):
        """subscribe() with sync callback receives readings."""
        self._skip_if_no_stream(async_read_backend)

        readings: list[Reading] = []
        event = asyncio.Event()

        def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 1:
                event.set()

        handle = await async_read_backend.subscribe([PERIODIC_DEVICE], callback=on_reading)
        try:
            await asyncio.wait_for(event.wait(), timeout=TIMEOUT_STREAM_EVENT)
        finally:
            await handle.stop()

        assert len(readings) >= 1
        assert all(r.ok for r in readings)

    @pytest.mark.asyncio
    async def test_subscribe_async_callback(self, async_read_backend: AsyncBackend):
        """subscribe() with async callback receives readings."""
        self._skip_if_no_stream(async_read_backend)

        readings: list[Reading] = []
        event = asyncio.Event()

        async def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 1:
                event.set()

        handle = await async_read_backend.subscribe([PERIODIC_DEVICE], callback=on_reading)
        try:
            await asyncio.wait_for(event.wait(), timeout=TIMEOUT_STREAM_EVENT)
        finally:
            await handle.stop()

        assert len(readings) >= 1
        assert all(r.ok for r in readings)

    @pytest.mark.asyncio
    async def test_subscribe_iterator_mode(self, async_read_backend: AsyncBackend):
        """subscribe() without callback enables async iterator mode."""
        self._skip_if_no_stream(async_read_backend)

        readings: list[Reading] = []
        handle = await async_read_backend.subscribe([PERIODIC_DEVICE])
        async with handle:
            async for reading, h in handle.readings(timeout=TIMEOUT_STREAM_ITER):
                readings.append(reading)
                if len(readings) >= 1:
                    break

        assert len(readings) >= 1
        assert handle.stopped

    @pytest.mark.asyncio
    async def test_handle_stop_ends_subscription(self, async_read_backend: AsyncBackend):
        """handle.stop() stops receiving data."""
        self._skip_if_no_stream(async_read_backend)

        readings: list[Reading] = []

        def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)

        handle = await async_read_backend.subscribe([FAST_PERIODIC], callback=on_reading)
        await asyncio.sleep(1.0)
        count_before = len(readings)
        await handle.stop()
        await asyncio.sleep(1.5)
        count_after = len(readings)

        # Allow 1-2 in-flight readings after stop
        assert count_after <= count_before + 2

    @pytest.mark.asyncio
    async def test_multiple_subscriptions(self, async_read_backend: AsyncBackend):
        """Multiple independent subscriptions work."""
        self._skip_if_no_stream(async_read_backend)

        h1 = await async_read_backend.subscribe([SLOW_PERIODIC], callback=lambda r, h: None)
        h2 = await async_read_backend.subscribe(["G:AMANDA@p,1000"], callback=lambda r, h: None)

        await asyncio.sleep(0.5)
        await h1.stop()

        assert h1.stopped
        assert not h2.stopped

        await h2.stop()

    @pytest.mark.asyncio
    async def test_backend_close_stops_all(self, async_read_backend: AsyncBackend):
        """Backend.close() stops all subscriptions."""
        self._skip_if_no_stream(async_read_backend)

        handles = []
        for _ in range(2):
            handles.append(await async_read_backend.subscribe([PERIODIC_DEVICE], callback=lambda r, h: None))
        await asyncio.sleep(0.5)

        await async_read_backend.close()

        for h in handles:
            assert h.stopped

    @pytest.mark.asyncio
    async def test_subscribe_mixed_valid_and_invalid(self, async_read_backend: AsyncBackend):
        """Subscription with valid + nonexistent device delivers both data and errors."""
        self._skip_if_no_stream(async_read_backend)

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

        handle = await async_read_backend.subscribe([valid_drf, invalid_drf], callback=on_reading)
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

    @pytest.mark.asyncio
    async def test_subscribe_invalid_device_reports_error(self, async_read_backend: AsyncBackend):
        """Subscription to nonexistent device delivers error/warning (not silent)."""
        self._skip_if_no_stream(async_read_backend)

        readings: list[Reading] = []
        got_reading = asyncio.Event()

        def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)
            got_reading.set()

        handle = await async_read_backend.subscribe([f"{NONEXISTENT_DEVICE}@p,500"], callback=on_reading)
        try:
            # DPM_PEND warning arrives with delay for nonexistent devices
            await asyncio.wait_for(got_reading.wait(), timeout=10.0)
        finally:
            await handle.stop()

        assert len(readings) >= 1, "Expected at least one error/warning reading for nonexistent device"
        assert readings[0].is_error or readings[0].is_warning


# =============================================================================
# Write Tests
# =============================================================================


@pytest.mark.real
@pytest.mark.kerberos
@requires_kerberos
class TestAsyncBackendWrite:
    """Async write tests (DPM HTTP only)."""

    @pytest.mark.asyncio
    async def test_write_capabilities(self, async_write_backend: AsyncBackend):
        """Backend reports WRITE capability when auth is set."""
        assert BackendCapability.WRITE in async_write_backend.capabilities
        assert BackendCapability.AUTH_KERBEROS in async_write_backend.capabilities

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_write_scalar(self, async_write_backend: AsyncBackend):
        """Write a different value, verify readback, then restore original."""
        read_drf = strip_event(SCALAR_SETPOINT)
        original = await async_write_backend.read(read_drf, timeout=TIMEOUT_READ)

        new_value = original + 0.1
        result = await async_write_backend.write(SCALAR_SETPOINT, new_value, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(1.0)
        readback = await async_write_backend.read(read_drf, timeout=TIMEOUT_READ)
        assert abs(readback - new_value) < 0.01, f"Write did not take effect: wrote {new_value}, read back {readback}"

        # Restore original value
        result2 = await async_write_backend.write(SCALAR_SETPOINT, original, timeout=TIMEOUT_READ)
        assert result2.success

        await asyncio.sleep(1.0)
        restored = await async_write_backend.read(read_drf, timeout=TIMEOUT_READ)
        assert abs(restored - original) < 0.01, f"Restore failed: wrote {original}, read back {restored}"

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_write_changes_raw(self, async_write_backend: AsyncBackend):
        """Write a scaled value and verify the .RAW readback changes."""
        original_scaled = await async_write_backend.read(strip_event(SCALAR_SETPOINT), timeout=TIMEOUT_READ)
        original_raw = await async_write_backend.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert isinstance(original_raw, bytes) and len(original_raw) > 0

        new_value = original_scaled + 1.0
        result = await async_write_backend.write(SCALAR_SETPOINT, new_value, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(1.0)
        new_raw = await async_write_backend.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert new_raw != original_raw, (
            f"Raw bytes unchanged after scaled write: {original_raw.hex()} -> {new_raw.hex()}"
        )

        # Restore
        result2 = await async_write_backend.write(SCALAR_SETPOINT, original_scaled, timeout=TIMEOUT_READ)
        assert result2.success

        await asyncio.sleep(1.0)
        restored_raw = await async_write_backend.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert restored_raw == original_raw, f"Restore failed: expected {original_raw.hex()}, got {restored_raw.hex()}"

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cmd_true,cmd_false,field", CONTROL_PAIRS, ids=lambda x: x if isinstance(x, str) else x.name
    )
    async def test_control_pair(self, async_write_backend: AsyncBackend, cmd_true, cmd_false, field):
        """Toggle control pair and verify the corresponding status bit changes."""
        reading = await async_write_backend.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert reading.ok, f"Failed to read status: {reading.message}"
        initial = reading.value.get(field)

        # Set TRUE
        result = await async_write_backend.write(STATUS_CONTROL_DEVICE, cmd_true, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(1.0)
        status = await async_write_backend.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after {cmd_true.name}: {status.message}"
        assert status.value.get(field) is True, f"Expected {field}=True after {cmd_true.name}"

        # Set FALSE
        result = await async_write_backend.write(STATUS_CONTROL_DEVICE, cmd_false, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(1.0)
        status = await async_write_backend.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after {cmd_false.name}: {status.message}"
        assert status.value.get(field) is False, f"Expected {field}=False after {cmd_false.name}"

        # Restore
        restore = cmd_true if initial else cmd_false
        await async_write_backend.write(STATUS_CONTROL_DEVICE, restore, timeout=TIMEOUT_READ)

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_control_reset(self, async_write_backend: AsyncBackend):
        """RESET command succeeds and produces a valid status."""
        result = await async_write_backend.write(STATUS_CONTROL_DEVICE, CONTROL_RESET, timeout=TIMEOUT_READ)
        assert result.success

        await asyncio.sleep(1.0)
        status = await async_write_backend.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after RESET: {status.message}"
        assert isinstance(status.value, dict)
        assert "on" in status.value

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_control_mixed(self, async_write_backend: AsyncBackend):
        """Mixed R/W/stream interleave stress test."""
        readings: list[Reading] = []
        event = asyncio.Event()

        result = await async_write_backend.write(STATUS_CONTROL_DEVICE, BasicControl.NEGATIVE, timeout=TIMEOUT_READ)
        assert result.success

        async def on_reading(reading: Reading, handle: AsyncSubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 3:
                r = await async_write_backend.write(STATUS_CONTROL_DEVICE, BasicControl.POSITIVE, timeout=TIMEOUT_READ)
                assert r.success
                event.set()

        valid_drf = PERIODIC_DEVICE
        invalid_drf = f"{NONEXISTENT_DEVICE}@p,500"
        handle = await async_write_backend.subscribe([valid_drf, invalid_drf], callback=on_reading)
        h1 = await async_write_backend.subscribe([SCALAR_DEVICE + "@p,500"])

        await asyncio.sleep(0.5)

        read1 = await async_write_backend.read(SCALAR_DEVICE_2)
        assert isinstance(read1, float)
        read2 = await async_write_backend.get_many([SCALAR_DEVICE_2, SCALAR_ELEMENT, ARRAY_DEVICE])
        assert all(r.ok for r in read2)
        for _ in range(5):
            read3 = await async_write_backend.get_many(
                [SCALAR_DEVICE, SCALAR_DEVICE_2, SCALAR_ELEMENT, SCALAR_SETPOINT]
            )
            assert all(r.ok for r in read3)
            w = await async_write_backend.write(STATUS_CONTROL_DEVICE, BasicControl.NEGATIVE, timeout=TIMEOUT_READ)
            assert w.ok

        await asyncio.sleep(4.0)

        try:
            await asyncio.wait_for(event.wait(), timeout=TIMEOUT_STREAM_EVENT)
        finally:
            await handle.stop()

        assert len(readings) >= 1
        assert all(r.ok for r in readings if r.name == valid_drf)

        j = 0
        async for r, _ in h1.readings(timeout=TIMEOUT_STREAM_ITER):
            j += 1
            if j > 3:
                await h1.stop()

        status = await async_write_backend.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status: {status.message}"
        assert status.value["positive"] is True


# =============================================================================
# Unpaired / Nonexistent Control Ordinals (Z:ACLTST)
# =============================================================================


@pytest.mark.real
@pytest.mark.kerberos
@requires_kerberos
class TestAsyncBackendUnpairedControls:
    """Unpaired control ordinals on Z:ACLTST (async write backend).

    Ordinals 7-11 map to device-specific TEST commands that succeed but don't
    toggle standard status bits. Ordinal 25 is beyond the device's control
    table and should be rejected.
    """

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "ordinal,cmd_name",
        ACLTST_UNPAIRED_CONTROLS,
        ids=[name for _, name in ACLTST_UNPAIRED_CONTROLS],
    )
    async def test_unpaired_control_write_succeeds(self, async_write_backend: AsyncBackend, ordinal, cmd_name):
        """Unpaired ordinal write is accepted without error."""
        result = await async_write_backend.write(STATUS_CONTROL_DEVICE, ordinal, timeout=TIMEOUT_READ)
        assert result.success, f"Ordinal {ordinal} ({cmd_name}) write failed: {result.error_code} {result.message}"

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.asyncio
    async def test_nonexistent_ordinal_rejected(self, async_write_backend: AsyncBackend):
        """Ordinal beyond device's control table is rejected (DIO_SCALEFAIL)."""
        result = await async_write_backend.write(
            STATUS_CONTROL_DEVICE, ACLTST_NONEXISTENT_ORDINAL, timeout=TIMEOUT_READ
        )
        assert not result.success, f"Expected error for ordinal {ACLTST_NONEXISTENT_ORDINAL}, but write succeeded"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
