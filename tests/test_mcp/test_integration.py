"""Integration tests: full tool flow with FakeBackend and policies."""

import pytest

from pacsys.mcp._tools import tool_read_device, tool_write_device
from pacsys.supervised._policies import DeviceAccessPolicy, SlewLimit, SlewRatePolicy, ValueRangePolicy
from pacsys.testing import FakeBackend


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
    r1 = tool_read_device(backend, "Z:ACLTST.SETTING", write_policies)
    assert r1["ok"] is True
    assert r1["value"] == 10.0

    w = tool_write_device(backend, "Z:ACLTST", 15.0, policies=write_policies)
    assert w["ok"] is True

    r2 = tool_read_device(backend, "Z:ACLTST.SETTING", write_policies)
    assert r2["ok"] is True
    assert r2["value"] == 15.0


def test_read_denied_by_deny_policy(backend):
    """Deny-mode device policy must gate MCP reads, matching the supervised server."""
    policies = [DeviceAccessPolicy(patterns=["Z:*"], mode="deny", action="read")]
    result = tool_read_device(backend, "Z:ACLTST.SETTING", policies)
    assert result["ok"] is False
    assert "denied" in result["error"].lower()

    ok = tool_read_device(backend, "M:OUTTMP", policies)
    assert ok["ok"] is True
    assert ok["value"] == 72.5


def test_read_rate_limited(backend):
    """Rate-limit policy applies to MCP reads."""
    from pacsys.supervised._policies import RateLimitPolicy

    policies = [RateLimitPolicy(max_requests=2, window_seconds=60.0)]
    assert tool_read_device(backend, "M:OUTTMP", policies)["ok"] is True
    assert tool_read_device(backend, "M:OUTTMP", policies)["ok"] is True
    denied = tool_read_device(backend, "M:OUTTMP", policies)
    assert denied["ok"] is False
    assert "rate limit" in denied["error"].lower()


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
