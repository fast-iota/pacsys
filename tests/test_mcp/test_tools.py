import pytest
from pacsys.testing import FakeBackend
from pacsys.supervised._policies import (
    DeviceAccessPolicy,
    ValueRangePolicy,
)
from pacsys.mcp._tools import tool_read_device, tool_write_device, tool_device_info


@pytest.fixture
def backend():
    fb = FakeBackend()
    fb.set_reading("M:OUTTMP", 72.5, units="deg F")
    fb.set_reading("Z:ACLTST.SETTING", 10.0)
    return fb


# ── read_device ──────────────────────────────────────────────


def test_read_device_success(backend):
    result = tool_read_device(backend, "M:OUTTMP")
    assert result["ok"] is True
    assert result["value"] == 72.5
    assert result["name"] == "M:OUTTMP"


def test_read_device_error(backend):
    backend.set_error("M:BADDEV", -42, "DIO_NO_SUCH")
    result = tool_read_device(backend, "M:BADDEV")
    assert result["ok"] is False
    assert "DIO_NO_SUCH" in result["error"]


def test_read_device_backend_exception(backend):
    """Backend raises unexpected exception — tool catches it."""
    backend.close()
    result = tool_read_device(backend, "M:DOESNOTEXIST")
    assert result["ok"] is False
    assert "error" in result


# ── write_device ─────────────────────────────────────────────


def test_write_device_no_policies(backend):
    """No policies = no policy approves writes = denied."""
    result = tool_write_device(backend, "Z:ACLTST", 42.0, policies=[])
    assert result["ok"] is False
    assert "denied" in result["error"].lower() or "no policy" in result["error"].lower()


def test_write_device_allowed(backend):
    policies = [DeviceAccessPolicy(patterns=["Z:ACLTST"], mode="allow", action="set")]
    result = tool_write_device(backend, "Z:ACLTST", 42.0, policies=policies)
    assert result["ok"] is True
    assert backend.was_written("Z:ACLTST.SETTING")


def test_write_device_denied_by_range(backend):
    policies = [
        DeviceAccessPolicy(patterns=["Z:ACLTST"], mode="allow", action="set"),
        ValueRangePolicy(limits={"Z:ACLTST": (0.0, 50.0)}),
    ]
    result = tool_write_device(backend, "Z:ACLTST", 999.0, policies=policies)
    assert result["ok"] is False
    assert "outside range" in result["error"].lower()
    assert not backend.was_written("Z:ACLTST.SETTING")


def test_write_device_unknown_device(backend):
    policies = [DeviceAccessPolicy(patterns=["Z:ACLTST"], mode="allow", action="set")]
    result = tool_write_device(backend, "Z:UNKNOWN", 42.0, policies=policies)
    assert result["ok"] is False


# ── device_info (basic, no DevDB mock needed for error path) ─


def test_device_info_no_devdb():
    result = tool_device_info(None, "M:OUTTMP")
    assert result["ok"] is False
    assert "unavailable" in result["error"].lower()
