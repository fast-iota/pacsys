"""
Integration tests for DMQBackend (DMQ-specific behavior).

Common read/error/value-type tests are in test_backend_shared.py.
Common streaming tests are in test_backend_shared.py.
Common write tests are in test_backend_shared.py.

This file contains DMQ-specific tests:
- Connection lifecycle with Kerberos
- DMQ-specific streaming patterns (periodic, callback, multi-device)
- DMQ-specific error handling (iterator with callback, empty subscribe)
- Raw integer write (.SETTING.RAW)

Run with: pytest tests/real/test_dmq_backend.py -v -s
"""

import threading
import time

import pytest

from pacsys.auth import KerberosAuth
from pacsys.backends.dmq import DMQBackend
from pacsys.drf_utils import strip_event
from pacsys.types import BackendCapability, Reading, SubscriptionHandle

from .devices import (
    PERIODIC_DEVICE,
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_SETPOINT,
    SCALAR_SETPOINT_RAW,
    TIMEOUT_READ,
    TIMEOUT_STREAM_EVENT,
    TIMEOUT_STREAM_ITER,
    requires_dmq,
    requires_kerberos,
    requires_write_enabled,
)


def _create_dmq_backend(**kwargs) -> DMQBackend:
    """Create a DMQBackend with Kerberos auth for testing."""
    kwargs.setdefault("host", "localhost")
    kwargs.setdefault("auth", KerberosAuth())
    return DMQBackend(**kwargs)


# =============================================================================
# Connection Tests (DMQ-specific - tests Kerberos auth and close behavior)
# =============================================================================


@requires_dmq
@pytest.mark.real
class TestDMQBackendConnection:
    """Tests for DMQBackend connection and lifecycle.

    NOTE: Basic connection tests are in test_backend_shared.py.
    These tests are DMQ-specific: Kerberos auth, closed state behavior.
    """

    def test_backend_connects_and_closes(self):
        """Backend connects and closes cleanly."""
        backend = _create_dmq_backend()
        assert BackendCapability.READ in backend.capabilities
        assert BackendCapability.STREAM in backend.capabilities
        backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            backend.read(SCALAR_DEVICE)

    def test_context_manager_cleanup(self):
        """Context manager properly cleans up resources."""
        with _create_dmq_backend() as backend:
            assert backend.capabilities is not None
        with pytest.raises(RuntimeError, match="closed"):
            backend.read(SCALAR_DEVICE)

    def test_closed_backend_raises(self):
        """Operations on closed backend raise RuntimeError."""
        backend = _create_dmq_backend()
        backend.close()

        with pytest.raises(RuntimeError, match="closed"):
            backend.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)

        with pytest.raises(RuntimeError, match="closed"):
            backend.subscribe([PERIODIC_DEVICE])


# =============================================================================
# Streaming Tests (DMQ-specific patterns)
# =============================================================================


@requires_dmq
@pytest.mark.real
@pytest.mark.streaming
class TestDMQBackendStreaming:
    """DMQ-specific streaming tests.

    Common streaming tests (callback, iterator, stop, multiple subs, close)
    are in test_backend_shared.py. These test DMQ-specific patterns.
    """

    def test_subscribe_periodic(self, dmq_backend):
        """subscribe() receives periodic readings in iterator mode."""
        with dmq_backend.subscribe([PERIODIC_DEVICE]) as handle:
            readings = []
            for reading, h in handle.readings(timeout=TIMEOUT_STREAM_ITER):
                readings.append(reading)
                if len(readings) >= 2:
                    break

        assert len(readings) >= 2
        assert all(r.ok for r in readings)
        assert handle.stopped

    def test_subscribe_callback(self, dmq_backend):
        """subscribe() with callback receives periodic readings."""
        readings = []
        event = threading.Event()

        def on_reading(reading: Reading, handle: SubscriptionHandle):
            readings.append(reading)
            if len(readings) >= 2:
                event.set()

        handle = dmq_backend.subscribe([PERIODIC_DEVICE], callback=on_reading)
        try:
            event.wait(timeout=TIMEOUT_STREAM_EVENT)
        finally:
            handle.stop()

        assert len(readings) >= 2
        assert all(r.ok for r in readings)

    def test_subscribe_multiple_devices(self, dmq_backend):
        """subscribe() to multiple devices."""
        devices = [f"{SCALAR_DEVICE}@p,500", f"{SCALAR_DEVICE_2}@p,500"]

        with dmq_backend.subscribe(devices) as handle:
            readings = []
            seen_devices = set()
            for reading, h in handle.readings(timeout=TIMEOUT_STREAM_ITER):
                readings.append(reading)
                seen_devices.add(reading.drf.split("@")[0])
                if len(readings) >= 4:
                    break

        assert len(readings) >= 2

    def test_iterator_mode_error_callback(self, dmq_backend):
        """Iterator mode raises exception if callback was provided."""
        handle = dmq_backend.subscribe([PERIODIC_DEVICE], callback=lambda r, h: None)
        try:
            with pytest.raises(RuntimeError, match="callback"):
                list(handle.readings(timeout=1.0))
        finally:
            handle.stop()

    def test_subscribe_empty_raises(self, dmq_backend):
        """subscribe() with empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            dmq_backend.subscribe([])


# =============================================================================
# Write Tests (DMQ-specific)
# =============================================================================


@requires_dmq
@requires_kerberos
@pytest.mark.real
@pytest.mark.kerberos
class TestDMQBackendWrite:
    """DMQ-specific write tests.

    Common write tests (scalar, raw readback, control pair, reset) are in
    test_backend_shared.py. This tests DMQ-specific raw integer write.
    """

    @pytest.mark.write
    @requires_write_enabled
    def test_write_raw(self):
        """Write via .SETTING.RAW and verify both raw readback and common units.

        Z:ACLTST has identity transform (C1=C2=1), so the raw value written
        equals the common-units (scaled) readback.
        """
        read_drf = strip_event(SCALAR_SETPOINT)
        backend = _create_dmq_backend()
        try:
            original_scaled = backend.read(read_drf, timeout=TIMEOUT_READ)

            for val in (45, 46):
                result = backend.write(SCALAR_SETPOINT_RAW, val, timeout=TIMEOUT_READ)
                assert result.success, f"Write {val} failed: {result.error_code} {result.message}"

                time.sleep(1.0)
                backend.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
                scaled = backend.read(read_drf, timeout=TIMEOUT_READ)
                assert scaled == float(val), f"Expected scaled={float(val)}, got {scaled}"

            # Restore
            backend.write(SCALAR_SETPOINT_RAW, int(original_scaled), timeout=TIMEOUT_READ)
        finally:
            backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
