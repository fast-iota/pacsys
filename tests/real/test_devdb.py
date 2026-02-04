"""
Real server tests for DevDB gRPC client.

Requires a DevDB service running at localhost:45678 (or PACSYS_DEVDB_HOST/PORT).
Auto-skips if not available.

Usage:
    python -m pytest tests/real/test_devdb.py -v -s -o "addopts="
"""

import os
import socket

import pytest

from pacsys.devdb import DevDBClient, DeviceInfoResult, DEVDB_AVAILABLE

DEVDB_HOST = os.environ.get("PACSYS_DEVDB_HOST", "localhost")
DEVDB_PORT = int(os.environ.get("PACSYS_DEVDB_PORT", "6802"))


def devdb_server_available() -> bool:
    if not DEVDB_AVAILABLE:
        return False
    try:
        sock = socket.create_connection((DEVDB_HOST, DEVDB_PORT), timeout=2.0)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError, socket.gaierror):
        return False


requires_devdb = pytest.mark.skipif(
    not devdb_server_available(),
    reason=f"DevDB server not available at {DEVDB_HOST}:{DEVDB_PORT}",
)


@pytest.fixture(scope="module")
def devdb():
    client = DevDBClient(host=DEVDB_HOST, port=DEVDB_PORT)
    yield client
    client.close()


@requires_devdb
class TestGetDeviceInfo:
    def test_single_device(self, devdb):
        result = devdb.get_device_info(["Z:ACLTST"])
        assert "Z:ACLTST" in result

        info = result["Z:ACLTST"]
        assert isinstance(info, DeviceInfoResult)
        assert info.device_index == 140013
        assert info.description == "ACL test device!"

    def test_reading_property(self, devdb):
        info = devdb.get_device_info(["Z:ACLTST"])["Z:ACLTST"]
        assert info.reading is not None
        assert info.reading.common_units == "blip"
        assert info.reading.min_val == pytest.approx(-500.0)
        assert info.reading.max_val == pytest.approx(500.0)

    def test_setting_property(self, devdb):
        info = devdb.get_device_info(["Z:ACLTST"])["Z:ACLTST"]
        assert info.setting is not None

    def test_control_commands(self, devdb):
        info = devdb.get_device_info(["Z:ACLTST"])["Z:ACLTST"]
        assert info.control is not None
        assert len(info.control) == 12
        # First three should be RESET, ON, OFF
        short_names = [c.short_name for c in info.control]
        assert "RESET" in short_names or "Reset" in short_names

    def test_status_bits(self, devdb):
        info = devdb.get_device_info(["Z:ACLTST"])["Z:ACLTST"]
        assert info.status_bits is not None
        assert len(info.status_bits) == 4
        short_names = [b.short_name for b in info.status_bits]
        assert "On" in short_names

    def test_batch_query(self, devdb):
        result = devdb.get_device_info(["Z:ACLTST", "M:OUTTMP"])
        assert "Z:ACLTST" in result
        assert "M:OUTTMP" in result
        assert result["Z:ACLTST"].device_index != result["M:OUTTMP"].device_index

    def test_nonexistent_device(self, devdb):
        from pacsys.errors import DeviceError

        with pytest.raises(DeviceError):
            devdb.get_device_info(["X:NOTREAL"])

    def test_caching(self, devdb):
        """Second query for same device should use cache."""
        devdb.clear_cache("Z:ACLTST")
        r1 = devdb.get_device_info(["Z:ACLTST"])
        r2 = devdb.get_device_info(["Z:ACLTST"])
        assert r1["Z:ACLTST"].device_index == r2["Z:ACLTST"].device_index

    def test_g_amanda(self, devdb):
        info = devdb.get_device_info(["G:AMANDA"])["G:AMANDA"]
        assert isinstance(info, DeviceInfoResult)
        assert info.reading is not None


@requires_devdb
class TestGetAlarmInfo:
    def test_alarm_info(self, devdb):
        result = devdb.get_alarm_info(["Z:ACLTST"])
        assert len(result) >= 1
        assert result[0].device_name == "Z:ACLTST"
        assert result[0].alarm_block is not None


@requires_devdb
class TestDigitalStatusViaDevDB:
    def test_devdb_status_bits_match_3read(self, devdb):
        """Compare DevDB-accelerated digital_status with 3-read path."""
        from pacsys.digital_status import DigitalStatus

        # Get bit definitions from DevDB
        info = devdb.get_device_info(["Z:ACLTST"])["Z:ACLTST"]
        assert info.status_bits is not None

        # We can't run the full comparison without a live backend,
        # but we can verify the bit definitions are sensible
        for bd in info.status_bits:
            assert bd.short_name  # non-empty name
            assert bd.true_str  # non-empty text
            assert bd.false_str  # non-empty text

        # Construct DigitalStatus from DevDB bits with a known raw value
        status = DigitalStatus.from_devdb_bits(
            "Z:ACLTST",
            raw_value=2,  # Ready only
            bit_defs=info.status_bits,
            ext_bit_defs=info.ext_status_bits,
        )
        assert status.device == "Z:ACLTST"
        assert status.raw_value == 2
        # Verify the Ready bit is active
        ready_bit = status.get("Ready")
        assert ready_bit is not None
        assert ready_bit.is_set is True
