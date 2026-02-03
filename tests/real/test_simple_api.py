"""
Integration tests for pacsys Simple API.

Tests the complete path: pacsys.read() -> global backend -> DPMHTTPBackend -> ConnectionPool -> TCP

Covers:
- Module-level read/get/get_many functions
- Device class integration
- Configuration and lifecycle
- Backend factory functions
- Thread safety

Run with: pytest tests/real/test_simple_api.py -v -s
"""

import pytest
import threading

import pacsys
from pacsys import Device, ScalarDevice, ArrayDevice
from pacsys import Reading, DeviceError, SubscriptionHandle

from .devices import (
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_ELEMENT,
    ARRAY_DEVICE,
    NONEXISTENT_DEVICE,
    PERIODIC_DEVICE,
    requires_dpm_http,
    TIMEOUT_READ,
    TIMEOUT_BATCH,
    TIMEOUT_STREAM_ITER,
    TIMEOUT_THREAD_JOIN,
)


# =============================================================================
# Simple API Read Tests
# =============================================================================


@requires_dpm_http
class TestSimpleAPIRead:
    """Tests for pacsys.read() - simplest API."""

    def test_read_scalar_device(self):
        """pacsys.read() returns scalar value."""
        value = pacsys.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))
        print(f"\n  pacsys.read('{SCALAR_DEVICE}') = {value}")

    def test_read_array_device(self):
        """pacsys.read() returns array for array device."""
        value = pacsys.read(ARRAY_DEVICE, timeout=TIMEOUT_READ)
        assert hasattr(value, "__len__")
        assert len(value) == 11
        print(f"\n  pacsys.read('{ARRAY_DEVICE}') = array of len {len(value)}")

    def test_read_with_device_object(self):
        """pacsys.read() accepts Device object."""
        device = Device(SCALAR_DEVICE)
        value = pacsys.read(device, timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))
        print(f"\n  pacsys.read(Device('{SCALAR_DEVICE}')) = {value}")

    def test_read_nonexistent_raises_device_error(self):
        """pacsys.read() raises DeviceError for bad device."""
        with pytest.raises(DeviceError) as exc_info:
            pacsys.read(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert exc_info.value.error_code != 0
        print(f"\n  DeviceError: {exc_info.value.message}")


# =============================================================================
# Simple API Get Tests
# =============================================================================


@requires_dpm_http
class TestSimpleAPIGet:
    """Tests for pacsys.get() - returns Reading with metadata."""

    def test_get_returns_reading(self):
        """pacsys.get() returns Reading object."""
        reading = pacsys.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)

        assert isinstance(reading, Reading)
        assert reading.ok
        assert reading.value is not None
        print(f"\n  pacsys.get('{SCALAR_DEVICE}'):")
        print(f"    value: {reading.value}")
        print(f"    units: {reading.units}")

    def test_get_with_device_object(self):
        """pacsys.get() accepts Device object."""
        device = ScalarDevice(SCALAR_DEVICE)
        reading = pacsys.get(device, timeout=TIMEOUT_READ)
        assert reading.ok
        assert isinstance(reading.value, (int, float))

    def test_get_nonexistent_returns_error_reading(self):
        """pacsys.get() returns error Reading without raising."""
        reading = pacsys.get(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert not reading.ok
        assert reading.error_code != 0
        print(f"\n  Error reading: status={reading.error_code}")


# =============================================================================
# Simple API Get Many Tests
# =============================================================================


@requires_dpm_http
class TestSimpleAPIGetMany:
    """Tests for pacsys.get_many() - batch reads."""

    def test_get_many_multiple_devices(self):
        """pacsys.get_many() reads multiple devices."""
        devices = [SCALAR_DEVICE, SCALAR_ELEMENT, SCALAR_DEVICE_2]
        readings = pacsys.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 3
        for i, r in enumerate(readings):
            print(f"\n  {devices[i]}: {r.value} (ok={r.ok})")

    def test_get_many_with_mixed_device_types(self):
        """pacsys.get_many() accepts mix of str and Device objects."""
        devices = [
            SCALAR_DEVICE,
            Device(SCALAR_ELEMENT),
            ScalarDevice(SCALAR_DEVICE_2),
        ]
        readings = pacsys.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 3
        assert all(r.ok for r in readings)

    def test_get_many_partial_failure(self):
        """pacsys.get_many() handles partial failures gracefully."""
        devices = [SCALAR_DEVICE, NONEXISTENT_DEVICE]
        readings = pacsys.get_many(devices, timeout=TIMEOUT_READ)

        assert len(readings) == 2
        assert readings[0].ok
        assert not readings[1].ok


# =============================================================================
# Device API Tests
# =============================================================================


@requires_dpm_http
class TestDeviceAPI:
    """Tests for Device class with global backend."""

    def test_device_read(self):
        """Device.read() uses global backend."""
        device = Device(SCALAR_DEVICE)
        value = device.read(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))
        print(f"\n  Device('{SCALAR_DEVICE}').read() = {value}")

    def test_device_get(self):
        """Device.get() returns Reading with metadata."""
        device = Device(SCALAR_DEVICE)
        reading = device.get(timeout=TIMEOUT_READ)

        assert isinstance(reading, Reading)
        assert reading.ok
        print(f"\n  Device.get(): value={reading.value}, units={reading.units}")

    def test_scalar_device_returns_float(self):
        """ScalarDevice.read() returns float."""
        device = ScalarDevice(SCALAR_DEVICE)
        value = device.read(timeout=TIMEOUT_READ)
        assert isinstance(value, float)
        print(f"\n  ScalarDevice.read() = {value} (type={type(value).__name__})")

    def test_array_device_returns_array(self):
        """ArrayDevice.read() returns numpy array."""
        device = ArrayDevice(ARRAY_DEVICE)
        value = device.read(timeout=TIMEOUT_READ)

        import numpy as np

        assert isinstance(value, np.ndarray)
        assert len(value) == 11
        print(f"\n  ArrayDevice.read() = array of len {len(value)}")

    def test_device_with_event_modifier(self):
        """Device.with_event() creates new device with event."""
        device = Device(SCALAR_DEVICE)
        immediate = device.with_event("I")

        assert immediate.has_event
        value = immediate.read(timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float))

    def test_device_properties(self):
        """Device exposes parsed DRF properties."""
        device = Device(ARRAY_DEVICE)

        assert device.name == "B:IRMS06"
        assert device.drf is not None
        print("\n  Device properties:")
        print(f"    name: {device.name}")
        print(f"    drf: {device.drf}")


# =============================================================================
# Configuration Tests
# =============================================================================


@requires_dpm_http
class TestConfiguration:
    """Tests for pacsys.configure() and environment."""

    def test_configure_before_use(self):
        """configure() sets parameters before backend init."""
        pacsys.shutdown()
        pacsys.configure(default_timeout=TIMEOUT_BATCH)

        value = pacsys.read(SCALAR_DEVICE)
        assert value is not None

    def test_shutdown_allows_reconfigure(self):
        """shutdown() allows reconfiguration."""
        pacsys.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)  # Ensure backend is initialized
        pacsys.shutdown()
        pacsys.configure(pool_size=2)

        value2 = pacsys.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert value2 is not None


# =============================================================================
# Backend Factory Tests
# =============================================================================


@requires_dpm_http
class TestBackendFactories:
    """Tests for backend factory functions."""

    def test_dpm_factory_creates_backend(self):
        """pacsys.dpm() creates working DPMHTTPBackend."""
        with pacsys.dpm() as backend:
            value = backend.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
            assert isinstance(value, (int, float))
            print(f"\n  pacsys.dpm().read() = {value}")

    def test_dpm_http_factory_is_alias(self):
        """pacsys.dpm_http() is alias for pacsys.dpm()."""
        with pacsys.dpm_http() as backend:
            value = backend.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
            assert isinstance(value, (int, float))

    def test_dpm_factory_with_custom_settings(self):
        """pacsys.dpm() accepts custom parameters."""
        with pacsys.dpm(pool_size=2, timeout=TIMEOUT_BATCH) as backend:
            value = backend.read(SCALAR_DEVICE)
            assert value is not None

    def test_acl_factory_creates_backend(self):
        """pacsys.acl() creates working ACLBackend."""
        with pacsys.acl() as backend:
            try:
                value = backend.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
                assert isinstance(value, (int, float))
                print(f"\n  pacsys.acl().read() = {value}")
            except DeviceError as e:
                if "403" in str(e) or "Forbidden" in str(e):
                    pytest.skip("ACL endpoint not accessible (403 Forbidden)")
                raise


# =============================================================================
# Global Backend Lifecycle Tests
# =============================================================================


@requires_dpm_http
class TestGlobalBackendLifecycle:
    """Tests for global backend initialization and cleanup."""

    def test_lazy_initialization(self):
        """Global backend initializes on first use."""
        pacsys.shutdown()
        assert pacsys._global_dpm_backend is None

        value = pacsys.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert value is not None
        assert pacsys._global_dpm_backend is not None

    def test_backend_reused_across_calls(self):
        """Same backend instance used for multiple calls."""
        pacsys.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        backend1 = pacsys._global_dpm_backend

        pacsys.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        backend2 = pacsys._global_dpm_backend

        assert backend1 is backend2

    def test_shutdown_cleans_up(self):
        """shutdown() closes backend and allows re-init."""
        pacsys.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert pacsys._global_dpm_backend is not None

        pacsys.shutdown()
        assert pacsys._global_dpm_backend is None

        value = pacsys.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert value is not None


# =============================================================================
# Module-Level Streaming Tests
# =============================================================================


@requires_dpm_http
@pytest.mark.streaming
class TestModuleLevelStreaming:
    """Tests for pacsys.subscribe() module-level function."""

    def test_subscribe_uses_global_backend(self):
        """pacsys.subscribe() uses global DPM backend."""
        import time

        readings = []

        def on_reading(reading: Reading, handle: SubscriptionHandle):
            readings.append(reading)

        handle = pacsys.subscribe([PERIODIC_DEVICE], callback=on_reading)
        try:
            time.sleep(1.5)
        finally:
            handle.stop()

        assert len(readings) >= 1
        print(f"\n  pacsys.subscribe() received {len(readings)} readings")

    def test_subscribe_iterator_mode_global(self):
        """pacsys.subscribe() works in iterator mode."""
        with pacsys.subscribe([PERIODIC_DEVICE]) as handle:
            count = 0
            for reading, _ in handle.readings(timeout=TIMEOUT_STREAM_ITER):
                count += 1
                print(f"\n  Global subscribe reading: {reading.value}")
                if count >= 2:
                    break

        assert count >= 2


# =============================================================================
# Thread Safety Tests
# =============================================================================


@requires_dpm_http
class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_reads_via_simple_api(self):
        """Concurrent pacsys.read() calls are thread-safe."""
        results = []
        errors = []

        def do_read():
            try:
                value = pacsys.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
                results.append(value)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_read) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=TIMEOUT_THREAD_JOIN)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 5
        print(f"\n  Concurrent reads: {results}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
