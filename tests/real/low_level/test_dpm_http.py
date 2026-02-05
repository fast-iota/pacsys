"""
Low-level tests for DPM/HTTP connection.

Tests the raw TCP + SDD DPM.proto binary protocol layer without the Backend abstraction.
Data-type, error, array, and streaming tests live in test_backend_shared.py.

Run with: pytest tests/real/low_level/test_dpm_http.py -v -s
"""

from tests.real.devices import (
    SCALAR_DEVICE,
    requires_dpm_http,
    TIMEOUT_READ,
)


@requires_dpm_http
class TestDPMHTTPConnection:
    """Raw DPMConnection protocol tests."""

    def test_connection_establishes(self, dpm_http_connection):
        """DPMConnection establishes and gets list_id."""
        assert dpm_http_connection.list_id is not None

    def test_single_read(self, dpm_http_connection):
        """read() returns reading with data."""
        reading = dpm_http_connection.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)

        assert reading.data is not None
        assert reading.timestamp is not None


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
