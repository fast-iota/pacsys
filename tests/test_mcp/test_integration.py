"""Integration tests: full tool flow with FakeBackend and policies."""

import pytest
from pacsys.testing import FakeBackend
from pacsys.supervised._policies import DeviceAccessPolicy, ValueRangePolicy, SlewRatePolicy, SlewLimit
from pacsys.mcp._tools import tool_read_device, tool_write_device


@pytest.fixture
def backend():
    fb = FakeBackend()
    fb.set_reading("M:OUTTMP", 72.5, units="deg F")
    fb.set_reading("Z:ACLTST.SETTING", 10.0)
    fb.set_reading("G:AMANDA", 1.23)
    return fb


@pytest.fixture
def write_policies():
    return [
        DeviceAccessPolicy(patterns=["Z:ACLTST"], mode="allow", action="set"),
        ValueRangePolicy(limits={"Z:ACLTST": (0.0, 100.0)}),
        SlewRatePolicy(limits={"Z:ACLTST": SlewLimit(max_step=10.0)}),
    ]


def test_read_then_write_then_read(backend, write_policies):
    """Full cycle: read current value, write new value, read back."""
    r1 = tool_read_device(backend, "Z:ACLTST.SETTING")
    assert r1["ok"] is True
    assert r1["value"] == 10.0

    w = tool_write_device(backend, "Z:ACLTST", 15.0, policies=write_policies)
    assert w["ok"] is True

    r2 = tool_read_device(backend, "Z:ACLTST.SETTING")
    assert r2["ok"] is True
    assert r2["value"] == 15.0


def test_write_denied_no_allowlist(backend):
    """Without write_devices config, writes are denied."""
    result = tool_write_device(backend, "Z:ACLTST", 42.0, policies=[])
    assert result["ok"] is False


def test_write_denied_wrong_device(backend, write_policies):
    """Device not in allowlist is denied."""
    result = tool_write_device(backend, "G:AMANDA", 42.0, policies=write_policies)
    assert result["ok"] is False


def test_write_denied_out_of_range(backend, write_policies):
    """Value outside configured range is denied."""
    result = tool_write_device(backend, "Z:ACLTST", 200.0, policies=write_policies)
    assert result["ok"] is False
    assert "outside range" in result["error"].lower()


def test_write_denied_slew_rate(backend, write_policies):
    """Second write exceeding step limit is denied."""
    w1 = tool_write_device(backend, "Z:ACLTST", 15.0, policies=write_policies)
    assert w1["ok"] is True

    w2 = tool_write_device(backend, "Z:ACLTST", 50.0, policies=write_policies)
    assert w2["ok"] is False
    assert "step" in w2["error"].lower() or "slew" in w2["error"].lower()
