"""
Low-level tests for DPM/HTTP connection (direct TCP to acsys-proxy).

Tests the raw TCP + PC binary protocol layer without the Backend abstraction.

Run with: pytest tests/real/low_level/test_dpm_http.py -v -s
"""

import pytest
import time

from tests.real.devices import (
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    ARRAY_DEVICE,
    SCALAR_ELEMENT,
    DESCRIPTION_DEVICE,
    STATUS_DEVICE,
    ANALOG_ALARM_DEVICE,
    DIGITAL_ALARM_DEVICE,
    SETTING_ON_READONLY,
    NOPROP_DEVICE,
    requires_dpm_http,
    TIMEOUT_READ,
)

from pacsys.acnet import DPMError


# =============================================================================
# Connection Tests
# =============================================================================


@requires_dpm_http
class TestDPMHTTPBasic:
    """Basic connection and read tests."""

    def test_connection_establishes(self, dpm_http_connection):
        """DPMConnection establishes and gets list_id."""
        assert dpm_http_connection.list_id is not None
        print(f"\n  Connected, list_id={dpm_http_connection.list_id}")

    def test_single_read(self, dpm_http_connection):
        """read() returns reading with data."""
        reading = dpm_http_connection.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)

        assert reading.data is not None
        assert reading.timestamp is not None
        print(f"\n  {SCALAR_DEVICE}:")
        print(f"    data: {reading.data}")
        print(f"    timestamp: {reading.timestamp}")
        print(f"    status: {reading.status}")

        if reading.device_info:
            print(f"    name: {reading.device_info.name}")
            print(f"    units: {reading.device_info.units}")

    def test_multiple_devices(self, dpm_http_connection):
        """read() works for multiple devices sequentially."""
        devices = [SCALAR_DEVICE, SCALAR_DEVICE_2]

        for device in devices:
            try:
                reading = dpm_http_connection.read(device, timeout=TIMEOUT_READ)
                print(f"\n  {device} = {reading.data}")
            except DPMError as e:
                print(f"\n  {device} = ERROR: {e}")
            except TimeoutError as e:
                print(f"\n  {device} = TIMEOUT: {e}")


# =============================================================================
# Array Tests
# =============================================================================


@requires_dpm_http
class TestDPMHTTPArrays:
    """Tests for array device reads."""

    def test_array_range(self, dpm_http_connection):
        """read() returns array for range request."""
        reading = dpm_http_connection.read(ARRAY_DEVICE, timeout=TIMEOUT_READ)

        assert reading.data is not None
        assert hasattr(reading.data, "__len__")
        assert len(reading.data) == 11  # [0:10] inclusive
        print(f"\n  {ARRAY_DEVICE}:")
        print(f"    data: {reading.data}")
        print(f"    length: {len(reading.data)}")

    def test_scalar_element(self, dpm_http_connection):
        """read() returns scalar for single array element."""
        device = f"{SCALAR_ELEMENT}@I"
        reading = dpm_http_connection.read(device, timeout=TIMEOUT_READ)

        assert reading.data is not None
        assert isinstance(reading.data, (int, float))
        print(f"\n  {device} = {reading.data}")

    def test_array_with_immediate_event(self, dpm_http_connection):
        """read() with @I returns exactly one reply."""
        device = "B:IRMS06[0:10]@I"
        reading = dpm_http_connection.read(device, timeout=TIMEOUT_READ)

        assert reading.data is not None
        assert hasattr(reading.data, "__len__")
        assert len(reading.data) == 11
        print(f"\n  {device}:")
        print(f"    length: {len(reading.data)}")


# =============================================================================
# Property Tests
# =============================================================================


@requires_dpm_http
class TestDPMHTTPProperties:
    """Tests for different device properties."""

    def test_description_property(self, dpm_http_connection):
        """read() returns text for DESCRIPTION property."""
        reading = dpm_http_connection.read(DESCRIPTION_DEVICE, timeout=TIMEOUT_READ)

        assert reading.data is not None
        assert isinstance(reading.data, str)
        print(f"\n  {DESCRIPTION_DEVICE} = '{reading.data}'")

    def test_status_property(self, dpm_http_connection):
        """read() returns dict for STATUS property."""
        reading = dpm_http_connection.read(STATUS_DEVICE, timeout=TIMEOUT_READ)

        assert reading.data is not None
        assert isinstance(reading.data, dict)
        print(f"\n  {STATUS_DEVICE}:")
        for key, val in reading.data.items():
            print(f"    {key}: {val}")

    def test_analog_alarm_property(self, dpm_http_connection):
        """read() returns dict for ANALOG_ALARM property."""
        reading = dpm_http_connection.read(ANALOG_ALARM_DEVICE, timeout=TIMEOUT_READ)

        assert reading.data is not None
        assert isinstance(reading.data, dict)
        print(f"\n  {ANALOG_ALARM_DEVICE}:")
        for key, val in reading.data.items():
            print(f"    {key}: {val}")

    def test_digital_alarm_property(self, dpm_http_connection):
        """read() returns dict for DIGITAL_ALARM property."""
        reading = dpm_http_connection.read(DIGITAL_ALARM_DEVICE, timeout=TIMEOUT_READ)

        assert reading.data is not None
        assert isinstance(reading.data, dict)
        print(f"\n  {DIGITAL_ALARM_DEVICE}:")
        for key, val in reading.data.items():
            print(f"    {key}: {val}")


# =============================================================================
# Error Tests
# =============================================================================


@requires_dpm_http
class TestDPMHTTPErrors:
    """Tests for error handling."""

    def test_setting_on_readonly_returns_error(self, dpm_http_connection):
        """read() raises DPMError for SETTING on read-only device."""
        device = f"{SETTING_ON_READONLY}@I"
        with pytest.raises(DPMError) as exc_info:
            dpm_http_connection.read(device, timeout=TIMEOUT_READ)
        assert exc_info.value.status < 0
        print(f"\n  {device}: status={exc_info.value.status}")

    def test_digital_alarm_noprop(self, dpm_http_connection):
        """read() raises DPMError for device without digital alarm."""
        with pytest.raises(DPMError) as exc_info:
            dpm_http_connection.read(NOPROP_DEVICE, timeout=TIMEOUT_READ)
        assert exc_info.value.status < 0
        print(f"\n  {NOPROP_DEVICE}: status={exc_info.value.status}")


# =============================================================================
# Streaming Tests
# =============================================================================


@requires_dpm_http
@pytest.mark.streaming
class TestDPMHTTPStreaming:
    """Tests for periodic/streaming reads."""

    def test_periodic_read_with_callback(self, dpm_http_connection):
        """add_request() with callback receives periodic readings."""
        readings = []

        def handle_data(reading):
            readings.append(reading)
            print(f"\n  Reading {len(readings)}: {reading.data}")

        ref_id = dpm_http_connection.add_request("M:OUTTMP@p,1000", callback=handle_data)
        print(f"\n  Added request, ref_id={ref_id}")

        dpm_http_connection.start()
        print("  Started acquisition...")

        time.sleep(3.0)

        dpm_http_connection.stop()
        print(f"  Stopped. Received {len(readings)} readings.")

        assert len(readings) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
