import json
from types import SimpleNamespace
from unittest import mock

import pytest

from pacsys.mcp._tools import tool_device_info, tool_read_device, tool_write_device
from pacsys.supervised._audit import AuditLog
from pacsys.supervised._policies import (
    DeviceAccessPolicy,
    ValueRangePolicy,
)
from pacsys.testing import FakeBackend


@pytest.fixture
def backend():
    fb = FakeBackend()
    fb.set_reading("M:OUTTMP", 72.5, units="deg F")
    fb.set_reading("Z:ACLTST.SETTING", 10.0)
    return fb


# ── read_device ──────────────────────────────────────────────


def test_read_device_success(backend):
    result = tool_read_device(backend, "M:OUTTMP", [])
    assert result["ok"] is True
    assert result["value"] == 72.5
    assert result["name"] == "M:OUTTMP"


def test_read_device_error(backend):
    backend.set_error("M:BADDEV", -42, "DIO_NO_SUCH")
    result = tool_read_device(backend, "M:BADDEV", [])
    assert result["ok"] is False
    assert "DIO_NO_SUCH" in result["error"]


def test_read_device_backend_exception(backend):
    """Backend raises unexpected exception — tool catches it."""
    with mock.patch.object(backend, "get", side_effect=RuntimeError("backend failed")):
        result = tool_read_device(backend, "M:DOESNOTEXIST", [])
    assert result["ok"] is False
    assert result["error"] == "backend failed"


def test_read_device_malformed_drf_returns_error_dict(backend):
    result = tool_read_device(backend, "M:OUTTMP[[", [DeviceAccessPolicy(patterns=["Z:*"], mode="deny", action="read")])
    assert result["ok"] is False
    assert result["drf"] == "M:OUTTMP[["
    assert result["value"] is None
    assert result["error"]


# ── write_device ─────────────────────────────────────────────


def test_write_device_malformed_drf_returns_error_dict(backend, tmp_path):
    policies = [DeviceAccessPolicy(patterns=["M:*"], mode="allow", action="set")]
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    result = tool_write_device(backend, "M:OUTTMP[[", 1.0, policies, audit)
    audit.close()
    assert result["ok"] is False
    assert result["drf"] == "M:OUTTMP[["
    assert result["error"]
    assert backend.writes == []
    entries = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [(e["allowed"], e["drfs"]) for e in entries] == [(False, ["M:OUTTMP[["])]  # denial is audited too


def test_write_device_no_policies(backend):
    """No policies = no policy approves writes = denied."""
    result = tool_write_device(backend, "Z:ACLTST", 42.0, policies=[])
    assert result["ok"] is False
    assert "no policy" in result["error"].lower()


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


def test_write_audit_records_allowed_and_denied_values(backend, tmp_path):
    path = tmp_path / "mcp-audit.jsonl"
    audit = AuditLog(str(path))
    policies = [
        DeviceAccessPolicy(patterns=["Z:ACLTST"], mode="allow", action="set"),
        ValueRangePolicy(limits={"Z:ACLTST": (0.0, 50.0)}),
    ]

    assert tool_write_device(backend, "Z:ACLTST", 42.0, policies, audit)["ok"] is True
    assert tool_write_device(backend, "Z:ACLTST", 999.0, policies, audit)["ok"] is False
    audit.close()

    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(entry["allowed"], entry["values"][0]["value"]) for entry in entries] == [
        (True, 42.0),
        (False, 999.0),
    ]


def test_write_blocked_when_audit_fails(backend):
    """A configured audit log that cannot record the decision blocks the write."""
    audit = mock.Mock(spec=AuditLog)
    audit.log_request.side_effect = OSError("disk full")
    policies = [DeviceAccessPolicy(patterns=["Z:ACLTST"], mode="allow", action="set")]

    result = tool_write_device(backend, "Z:ACLTST", 42.0, policies, audit)
    assert result["ok"] is False
    assert "Audit" in result["error"]
    assert backend.writes == []


def test_write_device_unknown_device(backend):
    policies = [DeviceAccessPolicy(patterns=["Z:ACLTST"], mode="allow", action="set")]
    result = tool_write_device(backend, "Z:UNKNOWN", 42.0, policies=policies)
    assert result["ok"] is False


# ── device_info (basic, no DevDB mock needed for error path) ─


def test_device_info_no_devdb():
    result = tool_device_info(None, "M:OUTTMP")
    assert result["ok"] is False
    assert "unavailable" in result["error"].lower()


def test_device_info_success():
    reading = SimpleNamespace(primary_units="deg F", common_units="C", min_val=-50.0, max_val=200.0)
    setting = SimpleNamespace(primary_units="A", common_units=None, min_val=0.0, max_val=10.0)
    control = (SimpleNamespace(value=1, short_name="ON", long_name="Turn on"),)
    info = SimpleNamespace(
        device_index=123, description="Test device", reading=reading, setting=setting, control=control
    )
    devdb = mock.MagicMock()
    devdb.get_device_info.return_value = {"M:OUTTMP": info}

    assert tool_device_info(devdb, "M:OUTTMP") == {
        "ok": True,
        "name": "M:OUTTMP",
        "description": "Test device",
        "device_index": 123,
        "reading": {"units": "deg F", "common_units": "C", "min": -50.0, "max": 200.0},
        "setting": {"units": "A", "common_units": None, "min": 0.0, "max": 10.0},
        "control_commands": [{"value": 1, "short_name": "ON", "long_name": "Turn on"}],
    }
