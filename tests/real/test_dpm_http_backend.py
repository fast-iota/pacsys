"""
Integration tests for DPMHTTPBackend (DPM HTTP-specific behavior).

Common read/error/value-type tests are in test_backend_shared.py.
Common streaming tests are in test_backend_shared.py.
Common write tests are in test_backend_shared.py.

This file contains DPM HTTP-specific tests:
- Connection pool behavior
- Raw bytes write (DEC F_float)
- Alarm writes
- Digital status reflects control

Run with: pytest tests/real/test_dpm_http_backend.py -v -s
"""

import time
import threading

import pytest

from pacsys.backends.dpm_http import DPMHTTPBackend
from pacsys.types import BasicControl

from .devices import (
    ANALOG_ALARM_SETPOINT,
    SCALAR_SETPOINT_RAW,
    STATUS_CONTROL_DEVICE,
    requires_dpm_http,
    requires_kerberos,
    requires_write_enabled,
    TIMEOUT_READ,
    TIMEOUT_THREAD_JOIN,
)


def _create_dpm_write_backend(**kwargs) -> DPMHTTPBackend:
    """Create a DPMHTTPBackend with Kerberos auth and role for write testing."""
    from pacsys.auth import KerberosAuth

    kwargs.setdefault("auth", KerberosAuth())
    kwargs.setdefault("role", "testing")
    return DPMHTTPBackend(**kwargs)


# =============================================================================
# Connection Pool Tests
# =============================================================================


@requires_dpm_http
class TestDPMHTTPBackendPool:
    """Tests for connection pool behavior."""

    def test_sequential_reads_reuse_connections(self):
        """Multiple sequential reads work correctly."""
        with DPMHTTPBackend(pool_size=2) as backend:
            for i in range(5):
                value = backend.read("M:OUTTMP", timeout=TIMEOUT_READ)
                assert isinstance(value, (int, float))

    def test_concurrent_reads(self):
        """Concurrent reads use pool correctly."""
        results = []
        errors = []

        def do_read(backend, device):
            try:
                results.append(backend.read(device, timeout=TIMEOUT_READ))
            except Exception as e:
                errors.append(e)

        with DPMHTTPBackend(pool_size=4) as backend:
            threads = [threading.Thread(target=do_read, args=(backend, "M:OUTTMP")) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=TIMEOUT_THREAD_JOIN)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 4


# =============================================================================
# Write Tests (DPM HTTP-specific)
# =============================================================================


@requires_dpm_http
@requires_kerberos
@pytest.mark.kerberos
class TestDPMHTTPBackendWrite:
    """DPM HTTP-specific write tests.

    Common write tests (scalar, raw readback, control pair, reset) are in
    test_backend_shared.py. These tests cover DPM HTTP-only behavior.
    """

    @pytest.mark.write
    @requires_write_enabled
    def test_write_raw(self):
        """Write DEC F_float raw bytes and verify engineering-unit readback.

        Z:ACLTST stores settings as DEC F_float (VAX single-precision).
        We write the exact raw representation and confirm the server
        converts back to the correct engineering value.
        """
        from pacsys.drf_utils import strip_event
        from .devices import SCALAR_SETPOINT

        raw_cases = [
            (b"\x34\x43\x00\x00", 45.0),
            (b"\x60\x43\x00\x00", 56.0),
        ]
        read_drf = strip_event(SCALAR_SETPOINT)
        backend = _create_dpm_write_backend()
        try:
            original_raw = backend.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)

            for raw_bytes, expected_scaled in raw_cases:
                result = backend.write(SCALAR_SETPOINT_RAW, raw_bytes, timeout=TIMEOUT_READ)
                assert result.success, f"Write {raw_bytes.hex()} failed: {result.error_code} {result.message}"

                time.sleep(1.0)
                readback_raw = backend.read(SCALAR_SETPOINT_RAW, timeout=TIMEOUT_READ)
                readback_scaled = backend.read(read_drf, timeout=TIMEOUT_READ)
                assert readback_raw == raw_bytes, f"Raw mismatch: wrote {raw_bytes.hex()}, read {readback_raw.hex()}"
                assert readback_scaled == expected_scaled, f"Expected scaled={expected_scaled}, got {readback_scaled}"

            # Restore
            backend.write(SCALAR_SETPOINT_RAW, original_raw, timeout=TIMEOUT_READ)
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    def test_write_analog_alarm_limit(self):
        """Write an analog alarm limit field and verify readback."""
        alarm_drf = ANALOG_ALARM_SETPOINT
        backend = _create_dpm_write_backend()
        try:
            reading = backend.get(alarm_drf, timeout=TIMEOUT_READ)
            assert reading.ok, f"Failed to read alarm: {reading.message}"
            orig_max = reading.value["maximum"]

            new_max = orig_max + 0.5
            result = backend.write(f"{alarm_drf}.MAX", new_max, timeout=TIMEOUT_READ)
            assert result.success

            time.sleep(1.0)
            after = backend.get(alarm_drf, timeout=TIMEOUT_READ)
            assert after.ok
            assert after.value["maximum"] == new_max, f"Expected maximum={new_max}, got {after.value['maximum']}"
            assert after.value["minimum"] == reading.value["minimum"]

            # Restore
            result2 = backend.write(f"{alarm_drf}.MAX", orig_max, timeout=TIMEOUT_READ)
            assert result2.success
        finally:
            backend.close()

    @pytest.mark.write
    @requires_write_enabled
    def test_write_analog_alarm_bypass(self):
        """Toggle analog alarm bypass (abort_inhibit) on and off."""
        alarm_drf = ANALOG_ALARM_SETPOINT
        backend = _create_dpm_write_backend()
        try:
            reading = backend.get(alarm_drf, timeout=TIMEOUT_READ)
            assert reading.ok, f"Failed to read alarm: {reading.message}"
            orig_inhibit = reading.value["abort_inhibit"]

            # Enable bypass
            result = backend.write(f"{alarm_drf}.ABORT_INHIBIT", 1, timeout=TIMEOUT_READ)
            assert result.success

            time.sleep(1.0)
            after_on = backend.get(alarm_drf, timeout=TIMEOUT_READ)
            assert after_on.ok
            assert after_on.value["abort_inhibit"] is True

            # Disable bypass
            result2 = backend.write(f"{alarm_drf}.ABORT_INHIBIT", 0, timeout=TIMEOUT_READ)
            assert result2.success

            time.sleep(1.0)
            after_off = backend.get(alarm_drf, timeout=TIMEOUT_READ)
            assert after_off.ok
            assert after_off.value["abort_inhibit"] is False

            # Restore
            restore = 1 if orig_inhibit else 0
            backend.write(f"{alarm_drf}.ABORT_INHIBIT", restore, timeout=TIMEOUT_READ)
        finally:
            backend.close()


# =============================================================================
# Digital Status + Control (DPM HTTP-specific write-based test)
# =============================================================================


@requires_dpm_http
class TestDeviceDigitalStatus:
    """DPM HTTP-specific: digital_status() reflects BasicControl writes."""

    @pytest.mark.write
    @requires_write_enabled
    @requires_kerberos
    def test_digital_status_reflects_control(self):
        """digital_status() reflects changes made via BasicControl writes."""
        from pacsys.device import Device

        backend = _create_dpm_write_backend()
        try:
            dev = Device("Z:ACLTST", backend=backend)
            initial = dev.digital_status(timeout=TIMEOUT_READ)

            # Turn ON and verify via digital_status
            backend.write(STATUS_CONTROL_DEVICE, BasicControl.ON, timeout=TIMEOUT_READ)
            time.sleep(1.0)
            after_on = dev.digital_status(timeout=TIMEOUT_READ)
            assert after_on.on is True, f"Expected on=True, got {after_on.on}"

            # Turn OFF and verify
            backend.write(STATUS_CONTROL_DEVICE, BasicControl.OFF, timeout=TIMEOUT_READ)
            time.sleep(1.0)
            after_off = dev.digital_status(timeout=TIMEOUT_READ)
            assert after_off.on is False, f"Expected on=False, got {after_off.on}"

            # Restore
            restore = BasicControl.ON if initial.on else BasicControl.OFF
            backend.write(STATUS_CONTROL_DEVICE, restore, timeout=TIMEOUT_READ)
        finally:
            backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
