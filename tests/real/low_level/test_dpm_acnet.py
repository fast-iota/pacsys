"""
Low-level tests for DPM/ACNET connection (DPM via ACNET protocol routing).

Tests the ACNET protocol path: DPMAcnet -> acnetd -> DPM

Run with: pytest tests/real/low_level/test_dpm_acnet.py -v -s
"""

import pytest
import time

from pacsys.acnet import DPMAcnet
from tests.real.devices import (
    requires_dpm_acnet,
    TIMEOUT_BATCH,
)


@pytest.fixture(scope="class")
def dpm_acnet():
    """Class-scoped DPMAcnet connection."""
    conn = DPMAcnet()
    conn.connect()
    yield conn
    conn.close()


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

    def test_repeated_array_read(self):
        """read() B:IRMS06[0:10] with 0.5s pause, fresh connection each time."""
        for i in range(10):
            dpm = DPMAcnet()
            dpm.connect()
            try:
                reading = dpm.read("B:IRMS06[0:10]@I", timeout=TIMEOUT_BATCH)
                assert reading.data is not None
                assert hasattr(reading.data, "__len__")
                assert len(reading.data) == 11
                if i % 10 == 0:
                    print(f"\n  [{i}] B:IRMS06[0:10] len={len(reading.data)}")
            finally:
                dpm.close()
            time.sleep(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
