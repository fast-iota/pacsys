"""
Low-level tests for DPM/ACNET connection (DPM via ACNET protocol routing).

Tests the ACNET protocol path: DPMAcnet -> acnetd -> DPM

Run with: pytest tests/real/low_level/test_dpm_acnet.py -v -s
"""

import pytest
import time

from tests.real.devices import (
    ARRAY_DEVICE,
    requires_dpm_acnet,
    TIMEOUT_BATCH,
)


# =============================================================================
# DPMAcnet Tests
# =============================================================================


@requires_dpm_acnet
class TestDPMAcnetConnection:
    """Tests for DPM/ACNET protocol path."""

    def test_connection_establishes(self, dpm_acnet):
        """DPMAcnet establishes and gets list_id."""
        assert dpm_acnet.list_id is not None
        print(f"\n  Connected, list_id={dpm_acnet.list_id}")

    def test_array_read(self, dpm_acnet):
        """read() returns array via ACNET path."""
        # Brief pause to allow any previous connections to settle
        time.sleep(0.5)

        reading = dpm_acnet.read(ARRAY_DEVICE, timeout=TIMEOUT_BATCH)

        assert reading.data is not None
        assert hasattr(reading.data, "__len__")
        print(f"\n  {ARRAY_DEVICE}:")
        print(f"    data: {reading.data}")
        print(f"    length: {len(reading.data)}")

        if reading.meta:
            print(f"    name: {reading.meta.get('name')}")
            print(f"    units: {reading.meta.get('units')}")

        # Expect 11 elements for [0:10] inclusive
        assert len(reading.data) >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
