"""
Shared tests that run against all read-capable backends.

These tests use the `read_backend` parametrized fixture which runs each test
against DMQ, DPM HTTP, and gRPC backends (where available).

Tests here verify cross-backend behavior consistency. Backend-specific tests
should remain in their individual test files (test_dmq_backend.py, etc.).
"""

import pytest
import time
import threading

from pacsys.backends import Backend
from pacsys.drf3 import parse_request
from pacsys.drf3.property import DRF_PROPERTY
from pacsys.drf_utils import strip_event
from pacsys.types import BasicControl, Reading, ValueType, BackendCapability, SubscriptionHandle
from pacsys.errors import DeviceError

from .devices import (
    ACLTST_NONEXISTENT_ORDINAL,
    ACLTST_UNPAIRED_CONTROLS,
    CONTROL_PAIRS,
    CONTROL_RESET,
    DEVICE_TYPES,
    FTP_DEVICE,
    LOGGER_DEVICE,
    LOGGER_DEVICE_WITH_EVENT,
    NONEXISTENT_DEVICE,
    NOPROP_DEVICE,
    PERIODIC_DEVICE,
    FAST_PERIODIC,
    SLOW_PERIODIC,
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_DEVICE_3,
    SCALAR_SETPOINT,
    SCALAR_SETPOINT_RAW,
    SETTING_ON_READONLY,
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


def _skip_if_no_stream(backend):
    if BackendCapability.STREAM not in backend.capabilities:
        pytest.skip(f"Backend {backend.__class__.__name__} does not support streaming")


@pytest.fixture(autouse=True)
def pause():
    """Space out load."""
    time.sleep(0.05)
    yield
    time.sleep(0.05)


# =============================================================================
# Connection Tests
# =============================================================================


class TestBackendConnection:
    """Tests for backend connection lifecycle (all backends)."""

    def test_backend_has_read_capability(self, read_backend_cls: Backend):
        """All backends should have READ capability."""
        assert BackendCapability.READ in read_backend_cls.capabilities

    def test_backend_not_closed_initially(self, read_backend_cls: Backend):
        """Backend should not be closed after creation."""
        assert not read_backend_cls._closed


# =============================================================================
# Basic Read Tests
# =============================================================================


class TestBackendRead:
    """Tests for basic read operations (all backends)."""

    def test_get_scalar(self, read_backend_cls: Backend):
        """get() returns Reading with float value for scalar device."""
        reading = read_backend_cls.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_DEVICE}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR
        assert isinstance(reading.value, float)
        assert reading.timestamp is not None

    def test_get_second_device(self, read_backend_cls: Backend):
        """get() works for second test device."""
        reading = read_backend_cls.get(SCALAR_DEVICE_2, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_DEVICE_2}: {reading.message}"
        assert isinstance(reading.value, float)

    def test_get_array(self, read_backend_cls: Backend):
        """get() returns array for array device."""
        reading = read_backend_cls.get(ARRAY_DEVICE, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {ARRAY_DEVICE}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR_ARRAY
        assert hasattr(reading.value, "__len__")
        assert len(reading.value) >= 2

    def test_get_array_element(self, read_backend_cls: Backend):
        """get() returns scalar for array element."""
        reading = read_backend_cls.get(SCALAR_ELEMENT, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {SCALAR_ELEMENT}: {reading.message}"
        assert reading.value_type == ValueType.SCALAR
        assert isinstance(reading.value, (int, float))


# =============================================================================
# Batch Read Tests
# =============================================================================


class TestBackendBatchReads:
    """Tests for batch read operations (all backends)."""

    def test_get_many(self, read_backend_cls: Backend):
        """get_many() reads multiple devices in one batch."""
        devices = [SCALAR_DEVICE, SCALAR_DEVICE_2]
        readings = read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        for i, reading in enumerate(readings):
            assert reading.ok, f"Failed: {devices[i]}: {reading.message}"
            assert isinstance(reading.value, float)

    def test_get_many_empty_list(self, read_backend_cls: Backend):
        """get_many() with empty list returns empty list."""
        readings = read_backend_cls.get_many([], timeout=TIMEOUT_READ)
        assert readings == []

    def test_get_many_order_preserved(self, read_backend_cls: Backend):
        """get_many() returns readings in same order as request."""
        devices = [SCALAR_DEVICE_2, SCALAR_DEVICE, SCALAR_ELEMENT]
        readings = read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 3
        # All should succeed
        assert all(r.ok for r in readings), f"Failures: {[r.message for r in readings if not r.ok]}"

    def test_get_many_duplicate_drf(self, read_backend_cls: Backend):
        """get_many() handles same device requested multiple times."""
        devices = [SCALAR_DEVICE, SCALAR_DEVICE_2, SCALAR_DEVICE]

        readings = read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

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


class TestBackendValueTypes:
    """Tests for reading all supported value types (all backends)."""

    # ACL CGI doesn't understand qualifier chars (~|@$) but _acl_read_prefix
    # canonicalizes them to explicit .PROPERTY names. Status/alarm properties
    # still can't return structured dicts - ACL returns plain text for those.
    _ACL_UNSUPPORTED_PROPERTIES = {
        DRF_PROPERTY.ANALOG,
        DRF_PROPERTY.DIGITAL,
        DRF_PROPERTY.BIT_STATUS,
    }

    @staticmethod
    def _is_acl(backend) -> bool:
        from pacsys.backends.acl import ACLBackend

        return isinstance(backend, ACLBackend)

    @staticmethod
    def _is_grpc(backend) -> bool:
        from pacsys.backends.grpc_backend import GRPCBackend

        return isinstance(backend, GRPCBackend)

    @pytest.mark.parametrize("drf,expected_type,python_type,desc", DEVICE_TYPES)
    def test_get_value_type(self, read_backend_cls: Backend, drf, expected_type, python_type, desc):
        """get() returns correct value_type for {desc}."""
        if self._is_acl(read_backend_cls):
            req = parse_request(drf)
            if req.property in self._ACL_UNSUPPORTED_PROPERTIES:
                pytest.skip(f"ACL cannot return structured {req.property.name} data")

        reading = read_backend_cls.get(drf, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {drf}: {reading.message}"
        assert reading.value_type == expected_type, f"Expected {expected_type}, got {reading.value_type}"

        if python_type is not None:
            assert isinstance(reading.value, python_type), f"Expected {python_type}, got {type(reading.value)}"

        # Array length check
        if expected_type == ValueType.SCALAR_ARRAY:
            assert hasattr(reading.value, "__len__")
            assert len(reading.value) >= 2

    def test_get_many_all_types(self, read_backend_cls: Backend):
        """get_many() reads all value types in one batch."""
        if self._is_acl(read_backend_cls):
            pytest.skip("ACL does not support all structured value types")
        if self._is_grpc(read_backend_cls):
            pytest.skip("DPM gRPC server has bug with gRPC synchronization")

        devices = [d[0] for d in DEVICE_TYPES]
        readings = read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == len(devices)
        for i, (drf, expected_type, _, _) in enumerate(DEVICE_TYPES):
            assert readings[i].ok, f"Failed: {drf}: {readings[i].message}"
            assert readings[i].value_type == expected_type


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBackendErrors:
    """Tests for error handling (all backends)."""

    def test_read_nonexistent_raises(self, read_backend_cls: Backend):
        """read() raises DeviceError for nonexistent device."""
        with pytest.raises(DeviceError) as exc_info:
            read_backend_cls.read(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert exc_info.value.error_code != 0

    def test_get_nonexistent_returns_error(self, read_backend: Backend):
        """get() returns error Reading for nonexistent device."""
        time.sleep(0.1)  # DMQ needs cooldown after prior error test
        reading = read_backend.get(NONEXISTENT_DEVICE, timeout=10.0)
        assert not reading.ok
        assert reading.error_code != 0

    def test_get_noprop_error(self, read_backend_cls: Backend):
        """get() returns error for missing property."""
        reading = read_backend_cls.get(NOPROP_DEVICE, timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code < 0

    def test_read_setting_on_readonly_raises(self, read_backend_cls: Backend):
        """read() raises DeviceError for SETTING on read-only device."""
        with pytest.raises(DeviceError):
            read_backend_cls.read(SETTING_ON_READONLY, timeout=TIMEOUT_READ)

    def test_get_many_partial_failure(self, read_backend_cls: Backend):
        """get_many() handles mix of success and error."""
        devices = [SCALAR_DEVICE, NONEXISTENT_DEVICE, SCALAR_DEVICE_2]
        readings = read_backend_cls.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 3
        assert readings[0].ok
        assert not readings[1].ok
        assert readings[2].ok


# =============================================================================
# Mixed Event Types Tests
# =============================================================================


class TestBackendMixedEvents:
    """Tests for reading devices with different event types in one batch."""

    def test_get_many_mixed_periodic_and_clock_event(self, read_backend_cls: Backend):
        """get_many() returns first reading per device for mixed event types.

        Tests that when combining a periodic (@p,100) with a slow clock
        event (@e,0C = TCLK event 0C, 15Hz booster), we:
        1. Get the first reading for each device
        2. Don't wait for additional periodic updates while waiting for clock event
        3. Return as soon as all devices have one reading

        Timing varies depending on when the clock event fires (0-5s wait).
        """
        _skip_if_no_stream(read_backend_cls)

        devices = [f"{SCALAR_DEVICE}@p,100", f"{SCALAR_DEVICE_2}@e,0C"]

        readings = read_backend_cls.get_many(devices, timeout=10.0)

        # Both readings should be present and valid
        assert len(readings) == 2
        assert readings[0].ok, f"Periodic device failed: {readings[0].message}"
        assert readings[1].ok, f"Clock event device failed: {readings[1].message}"

    def test_get_many_all_periodic_same_rate(self, read_backend_cls: Backend):
        """get_many() with multiple periodic devices at same rate gets one reading each."""
        _skip_if_no_stream(read_backend_cls)

        devices = [f"{SCALAR_DEVICE}@p,100", f"{SCALAR_DEVICE_2}@p,100"]

        readings = read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        assert all(r.ok for r in readings), f"Failures: {[r.message for r in readings if not r.ok]}"

    def test_get_many_immediate_with_periodic(self, read_backend_cls: Backend):
        """get_many() with @I (immediate) and @p (periodic) devices."""
        _skip_if_no_stream(read_backend_cls)

        devices = [f"{SCALAR_DEVICE}@I", f"{SCALAR_DEVICE_2}@p,500"]

        readings = read_backend_cls.get_many(devices, timeout=TIMEOUT_BATCH)

        assert len(readings) == 2
        assert readings[0].ok, f"Immediate device failed: {readings[0].message}"
        assert readings[1].ok, f"Periodic device failed: {readings[1].message}"

    def test_get_many_same_device_periodic_and_clock(self, read_backend_cls: Backend):
        """Same device at periodic and clock event returns two distinct readings."""
        _skip_if_no_stream(read_backend_cls)

        devices = [f"{SCALAR_DEVICE}@p,500", f"{SCALAR_DEVICE}@e,0C"]
        readings = read_backend_cls.get_many(devices, timeout=10.0)

        assert len(readings) == 2
        assert readings[0].ok, f"Periodic failed: {readings[0].message}"
        assert readings[1].ok, f"Clock event failed: {readings[1].message}"
        # Verify DRF carries the correct event so callers can discriminate
        assert "@p" in readings[0].drf.lower(), f"Expected periodic event in drf: {readings[0].drf}"
        assert "@e" in readings[1].drf.lower(), f"Expected clock event in drf: {readings[1].drf}"

    def test_get_many_same_device_two_clock_events(self, read_backend_cls: Backend):
        """Same device at two different clock events (0F and 0C)."""
        _skip_if_no_stream(read_backend_cls)

        devices = [f"{SCALAR_DEVICE_3}@e,0F", f"{SCALAR_DEVICE_3}@e,0C"]
        readings = read_backend_cls.get_many(devices, timeout=10.0)

        assert len(readings) == 2
        assert readings[0].ok, f"Event 0F failed: {readings[0].message}"
        assert readings[1].ok, f"Event 0C failed: {readings[1].message}"
        # Verify each reading carries its specific clock event
        assert "0F" in readings[0].drf, f"Expected event 0F in drf: {readings[0].drf}"
        assert "0C" in readings[1].drf, f"Expected event 0C in drf: {readings[1].drf}"


# =============================================================================
# Streaming Tests
# =============================================================================


@pytest.mark.streaming
class TestBackendStreaming:
    """Streaming tests that work across all backends."""

    def test_subscribe_callback_mode(self, read_backend_cls: Backend):
        """subscribe() with callback receives readings."""
        _skip_if_no_stream(read_backend_cls)

        readings = []
        event = threading.Event()

        def on_reading(reading: Reading, handle: SubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 1:
                event.set()

        handle = read_backend_cls.subscribe([PERIODIC_DEVICE], callback=on_reading)
        try:
            event.wait(timeout=TIMEOUT_STREAM_EVENT)
        finally:
            handle.stop()

        assert len(readings) >= 1
        assert all(r.ok for r in readings)

    def test_subscribe_iterator_mode(self, read_backend_cls: Backend):
        """subscribe() without callback enables iterator mode."""
        _skip_if_no_stream(read_backend_cls)

        with read_backend_cls.subscribe([PERIODIC_DEVICE]) as handle:
            readings = []
            for reading, h in handle.readings(timeout=TIMEOUT_STREAM_ITER):
                readings.append(reading)
                if len(readings) >= 1:
                    break

        assert len(readings) >= 1
        assert handle.stopped

    def test_handle_stop_ends_subscription(self, read_backend_cls: Backend):
        """handle.stop() stops receiving data."""
        _skip_if_no_stream(read_backend_cls)

        readings = []

        def on_reading(reading: Reading, handle: SubscriptionHandle):
            readings.append(reading)

        handle = read_backend_cls.subscribe([FAST_PERIODIC], callback=on_reading)
        time.sleep(0.5)
        count_before = len(readings)
        handle.stop()
        time.sleep(0.5)
        count_after = len(readings)

        # Allow 1-2 in-flight readings after stop
        assert count_after <= count_before + 2

    def test_multiple_subscriptions(self, read_backend_cls: Backend):
        """Multiple independent subscriptions work."""
        _skip_if_no_stream(read_backend_cls)

        h1 = read_backend_cls.subscribe([SLOW_PERIODIC], callback=lambda r, h: None)
        h2 = read_backend_cls.subscribe(["G:AMANDA@p,1000"], callback=lambda r, h: None)

        time.sleep(0.1)
        h1.stop()

        assert h1.stopped
        assert not h2.stopped

        h2.stop()

    def test_backend_close_stops_all(self, read_backend: Backend):
        """Backend.close() stops all subscriptions."""
        _skip_if_no_stream(read_backend)

        handles = []
        for _ in range(2):
            handles.append(read_backend.subscribe([PERIODIC_DEVICE], callback=lambda r, h: None))
        time.sleep(0.1)

        read_backend.close()

        for h in handles:
            assert h.stopped

    def _skip_if_dmq(self, backend):
        """DMQ silently drops nonexistent devices instead of sending errors."""
        from pacsys.backends.dmq import DMQBackend

        if isinstance(backend, DMQBackend):
            pytest.skip("DMQ does not report errors for nonexistent streaming devices")

    def test_subscribe_mixed_valid_and_invalid(self, read_backend_cls: Backend):
        """Subscription with valid + nonexistent device delivers both data and errors."""
        _skip_if_no_stream(read_backend_cls)
        self._skip_if_dmq(read_backend_cls)

        readings_by_drf: dict[str, list[Reading]] = {}
        lock = threading.Lock()
        got_both = threading.Event()

        valid_drf = PERIODIC_DEVICE
        invalid_drf = f"{NONEXISTENT_DEVICE}@p,500"

        def on_reading(reading: Reading, handle: SubscriptionHandle):
            with lock:
                readings_by_drf.setdefault(reading.drf, []).append(reading)
                print(
                    f"Received reading for {reading.drf}: ok={reading.ok}, value={reading.value}, error_code={reading.error_code}, facility={reading.facility}, message={reading.message}"
                )
                if len(readings_by_drf) >= 2:
                    got_both.set()

        handle = read_backend_cls.subscribe([valid_drf, invalid_drf], callback=on_reading)
        try:
            # Nonexistent device warning (DPM_PEND) arrives with delay
            got_both.wait(timeout=10.0)
        finally:
            handle.stop()

        with lock:
            valid_readings = readings_by_drf.get(valid_drf, [])
            assert len(valid_readings) >= 1, f"Expected readings for {valid_drf}"
            assert valid_readings[0].ok

            # Nonexistent device should have delivered an error/warning reading
            # (DRF may be canonicalized by the server, so check all non-valid keys)
            invalid_readings = [r for drf, rs in readings_by_drf.items() if drf != valid_drf for r in rs]
            assert len(invalid_readings) >= 1, (
                f"Expected error reading for invalid device, got drfs: {list(readings_by_drf)}"
            )
            assert invalid_readings[0].is_error or invalid_readings[0].is_warning

    def test_subscribe_invalid_device_reports_error(self, read_backend_cls: Backend):
        """Subscription to nonexistent device delivers error/warning (not silent)."""
        _skip_if_no_stream(read_backend_cls)
        self._skip_if_dmq(read_backend_cls)

        readings: list[Reading] = []
        got_reading = threading.Event()

        def on_reading(reading: Reading, handle: SubscriptionHandle):
            readings.append(reading)
            got_reading.set()

        handle = read_backend_cls.subscribe([f"{NONEXISTENT_DEVICE}@p,500"], callback=on_reading)
        try:
            # DPM_PEND warning arrives with delay for nonexistent devices
            got_reading.wait(timeout=10.0)
        finally:
            handle.stop()

        assert len(readings) >= 1, "Expected at least one error/warning reading for nonexistent device"
        assert readings[0].is_error or readings[0].is_warning


# =============================================================================
# FTP (Fast Time Plot) Tests - DPM backends only
# =============================================================================


def _skip_if_not_dpm(backend):
    """Skip test if backend is not a DPM-based backend (only DPM understands <-FTP)."""
    from pacsys.backends.dpm_http import DPMHTTPBackend

    try:
        from pacsys.backends.grpc_backend import GRPCBackend
    except ImportError:
        GRPCBackend = None

    if not isinstance(backend, DPMHTTPBackend) and not (GRPCBackend and isinstance(backend, GRPCBackend)):
        pytest.skip(f"Backend {backend.__class__.__name__} does not support <-FTP routing")


class TestBackendFTP:
    """FTP read tests via DPM's <-FTP extra qualifier."""

    def test_get_ftp(self, read_backend_cls: Backend):
        """get() with <-FTP returns timestamped scalar array from FTPMAN at ~100 Hz."""
        _skip_if_not_dpm(read_backend_cls)
        _skip_if_no_stream(read_backend_cls)

        reading = read_backend_cls.get(FTP_DEVICE, timeout=TIMEOUT_READ)

        assert reading.ok, f"Failed to read {FTP_DEVICE}: {reading.message}"
        assert reading.value_type == ValueType.TIMED_SCALAR_ARRAY
        assert "data" in reading.value and "micros" in reading.value
        assert len(reading.value["data"]) > 0
        assert len(reading.value["data"]) == len(reading.value["micros"])
        assert reading.timestamp is not None

        # Verify sample spacing is ~10ms (100 Hz)
        micros = reading.value["micros"]
        assert len(micros) >= 2, f"FTP should return multiple samples, got {len(micros)}"
        deltas_us = [micros[i + 1] - micros[i] for i in range(len(micros) - 1)]
        avg_delta_ms = sum(deltas_us) / len(deltas_us) / 1000.0
        assert 9.0 < avg_delta_ms < 11.0, f"Expected ~10ms spacing, got {avg_delta_ms:.2f}ms"


# =============================================================================
# Logger (Historical Data) Tests - DPM backends only
# =============================================================================


class TestBackendLogger:
    """Logger read tests via DPM's <-LOGGER extra qualifier.

    Window: 2025-01-15 12:00–13:00 UTC (epoch ms 1736942400000–1736946000000).
    """

    # start/end in microseconds for timestamp validation
    _START_US = 1736942400000 * 1000
    _END_US = 1736946000000 * 1000
    _MARGIN_US = 60_000_000  # 60 s

    def _assert_logger_reading(self, reading: Reading, drf: str):
        assert reading.ok, f"Failed to read {drf}: {reading.message}"
        assert reading.value_type == ValueType.TIMED_SCALAR_ARRAY
        assert "data" in reading.value and "micros" in reading.value
        assert len(reading.value["data"]) > 0
        assert len(reading.value["data"]) == len(reading.value["micros"])
        assert reading.timestamp is not None

        micros = reading.value["micros"]
        assert micros[0] >= self._START_US - self._MARGIN_US, "First sample before window"
        assert micros[-1] <= self._END_US + self._MARGIN_US, "Last sample after window"
        for i in range(1, len(micros)):
            assert micros[i] >= micros[i - 1], f"Timestamps not monotonic at index {i}"

    def test_get_logger(self, read_backend: Backend):
        """get() with <-LOGGER (no event) returns historical timed scalar array."""
        _skip_if_not_dpm(read_backend)
        reading = read_backend.get(LOGGER_DEVICE, timeout=30.0)
        self._assert_logger_reading(reading, LOGGER_DEVICE)

        # M:OUTTMP is logged at ~1 Hz; 1 hour ≈ 3600 points (allow ±10%)
        n = len(reading.value["data"])
        assert 3200 < n < 4000, f"Expected ~3600 samples for 1h at ~1Hz, got {n}"

    def test_get_logger_with_event(self, read_backend: Backend):
        """get() with @P,1000,true<-LOGGER returns historical timed scalar array."""
        _skip_if_not_dpm(read_backend)
        reading = read_backend.get(LOGGER_DEVICE_WITH_EVENT, timeout=30.0)
        self._assert_logger_reading(reading, LOGGER_DEVICE_WITH_EVENT)

        # @P,1000 = 1 Hz delivery; 1 hour ≈ 3600 points (allow ±10%)
        n = len(reading.value["data"])
        assert 3200 < n < 4000, f"Expected ~3600 samples for 1h at 1Hz, got {n}"

        # Verify sample spacing is ~1 s (within tolerance for logger jitter)
        micros = reading.value["micros"]
        span_s = (micros[-1] - micros[0]) / 1_000_000
        avg_spacing_s = span_s / (len(micros) - 1)
        assert 0.5 < avg_spacing_s < 2.0, f"Expected ~1s spacing, got {avg_spacing_s:.1f}s"


# =============================================================================
# Write Tests (shared across write-capable backends)
# =============================================================================


@pytest.mark.kerberos
@requires_kerberos
class TestBackendWrite:
    """Write tests that run against all write-capable backends (DPM HTTP, DMQ)."""

    @pytest.mark.write
    @requires_write_enabled
    def test_write_scalar(self, write_backend_cls: Backend):
        """Write a different value, verify readback, then restore original."""
        read_drf = strip_event(SCALAR_SETPOINT)
        original = write_backend_cls.read(read_drf, timeout=TIMEOUT_READ)

        new_value = original + 0.1
        result = write_backend_cls.write(SCALAR_SETPOINT, new_value, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(0.5)
        readback = write_backend_cls.read(read_drf, timeout=TIMEOUT_READ)
        assert abs(readback - new_value) < 0.01, f"Write did not take effect: wrote {new_value}, read back {readback}"

        # Restore original value
        result2 = write_backend_cls.write(SCALAR_SETPOINT, original, timeout=TIMEOUT_READ)
        assert result2.success

        time.sleep(0.5)
        restored = write_backend_cls.read(read_drf, timeout=TIMEOUT_READ)
        assert abs(restored - original) < 0.01, f"Restore failed: wrote {original}, read back {restored}"

    @pytest.mark.write
    @requires_write_enabled
    def test_write_changes_raw(self, write_backend_cls: Backend):
        """Write a scaled value and verify the .RAW readback changes accordingly."""
        original_scaled = write_backend_cls.read(strip_event(SCALAR_SETPOINT), timeout=TIMEOUT_READ)
        original_raw = write_backend_cls.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert isinstance(original_raw, bytes) and len(original_raw) > 0

        new_value = original_scaled + 1.0
        result = write_backend_cls.write(SCALAR_SETPOINT, new_value, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(0.5)
        new_raw = write_backend_cls.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert new_raw != original_raw, (
            f"Raw bytes unchanged after scaled write: {original_raw.hex()} -> {new_raw.hex()}"
        )

        # Restore
        result2 = write_backend_cls.write(SCALAR_SETPOINT, original_scaled, timeout=TIMEOUT_READ)
        assert result2.success

        time.sleep(0.5)
        restored_raw = write_backend_cls.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
        assert restored_raw == original_raw, f"Restore failed: expected {original_raw.hex()}, got {restored_raw.hex()}"

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.parametrize(
        "cmd_true,cmd_false,field", CONTROL_PAIRS, ids=lambda x: x if isinstance(x, str) else x.name
    )
    def test_control_pair(self, write_backend_cls: Backend, cmd_true, cmd_false, field):
        """Toggle control pair and verify the corresponding status bit changes."""
        reading = write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert reading.ok, f"Failed to read status: {reading.message}"
        initial = reading.value.get(field)

        # Set TRUE
        result = write_backend_cls.write(STATUS_CONTROL_DEVICE, cmd_true, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(0.5)
        status = write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after {cmd_true.name}: {status.message}"
        assert status.value.get(field) is True, f"Expected {field}=True after {cmd_true.name}"

        # Set FALSE
        result = write_backend_cls.write(STATUS_CONTROL_DEVICE, cmd_false, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(0.5)
        status = write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after {cmd_false.name}: {status.message}"
        assert status.value.get(field) is False, f"Expected {field}=False after {cmd_false.name}"

        # Restore
        restore = cmd_true if initial else cmd_false
        write_backend_cls.write(STATUS_CONTROL_DEVICE, restore, timeout=TIMEOUT_READ)

    @pytest.mark.write
    @requires_write_enabled
    def test_control_reset(self, write_backend_cls: Backend):
        """RESET command succeeds and produces a valid status."""
        result = write_backend_cls.write(STATUS_CONTROL_DEVICE, CONTROL_RESET, timeout=TIMEOUT_READ)
        assert result.success

        time.sleep(0.5)
        status = write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after RESET: {status.message}"
        assert isinstance(status.value, dict)
        assert "on" in status.value

    @pytest.mark.write
    @requires_write_enabled
    def test_control_mixed(self, write_backend_cls: Backend):
        """MIXED R/W test."""
        readings = []
        event = threading.Event()

        result = write_backend_cls.write(STATUS_CONTROL_DEVICE, BasicControl.NEGATIVE, timeout=TIMEOUT_READ)
        assert result.success

        def on_reading(reading: Reading, handle: SubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 3:
                result = write_backend_cls.write(STATUS_CONTROL_DEVICE, BasicControl.POSITIVE, timeout=TIMEOUT_READ)
                assert result.success
                event.set()

        valid_drf = PERIODIC_DEVICE
        invalid_drf = f"{NONEXISTENT_DEVICE}@p,500"
        handle = write_backend_cls.subscribe([valid_drf, invalid_drf], callback=on_reading)
        h1 = write_backend_cls.subscribe([SCALAR_DEVICE + "@p,500"])

        time.sleep(0.1)

        read1 = write_backend_cls.read(SCALAR_DEVICE_2)
        assert isinstance(read1, float)
        read2 = write_backend_cls.get_many([SCALAR_DEVICE_2, SCALAR_ELEMENT, ARRAY_DEVICE])
        assert all(r.ok for r in read2)
        for i in range(5):
            read3 = write_backend_cls.get_many([SCALAR_DEVICE, SCALAR_DEVICE_2, SCALAR_ELEMENT, SCALAR_SETPOINT])
            assert all(r.ok for r in read3)
            write = write_backend_cls.write(STATUS_CONTROL_DEVICE, BasicControl.NEGATIVE, timeout=TIMEOUT_READ)
            assert write.ok

        time.sleep(1.0)

        try:
            event.wait(timeout=TIMEOUT_STREAM_EVENT)
        finally:
            handle.stop()

        assert len(readings) >= 1
        assert all(r.ok for r in readings if r.name == valid_drf)
        assert all(not r.ok for r in readings if r.name == invalid_drf)

        j = 0
        for r in h1.readings():
            j += 1
            if j > 3:
                h1.stop()

        status = write_backend_cls.get(STATUS_CONTROL_DEVICE, timeout=TIMEOUT_READ)
        assert status.ok, f"Failed to read status after RESET: {status.message}"
        assert status.value["positive"] is True


# =============================================================================
# Unpaired / Nonexistent Control Ordinals (Z:ACLTST)
# =============================================================================


@pytest.mark.kerberos
@requires_kerberos
class TestBackendUnpairedControls:
    """Unpaired control ordinals on Z:ACLTST (all write-capable backends).

    Ordinals 7-11 map to device-specific TEST commands that succeed but don't
    toggle standard status bits. Ordinal 25 is beyond the device's control
    table and should be rejected. See ordinal table in devices.py.
    """

    @pytest.mark.write
    @requires_write_enabled
    @pytest.mark.parametrize(
        "ordinal,cmd_name",
        ACLTST_UNPAIRED_CONTROLS,
        ids=[name for _, name in ACLTST_UNPAIRED_CONTROLS],
    )
    def test_unpaired_control_write_succeeds(self, write_backend: Backend, ordinal, cmd_name):
        """Unpaired ordinal write is accepted without error."""
        result = write_backend.write(STATUS_CONTROL_DEVICE, ordinal, timeout=TIMEOUT_READ)
        assert result.success, f"Ordinal {ordinal} ({cmd_name}) write failed: {result.error_code} {result.message}"

    @pytest.mark.write
    @requires_write_enabled
    def test_nonexistent_ordinal_rejected(self, write_backend: Backend):
        """Ordinal beyond device's control table is rejected (DIO_SCALEFAIL)."""
        result = write_backend.write(STATUS_CONTROL_DEVICE, ACLTST_NONEXISTENT_ORDINAL, timeout=TIMEOUT_READ)
        assert not result.success, f"Expected error for ordinal {ACLTST_NONEXISTENT_ORDINAL}, but write succeeded"
