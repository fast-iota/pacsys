import sys

import pytest

from pacsys.mcp._config import MCPConfig, build_policies, load_config
from pacsys.supervised._policies import DeviceAccessPolicy, SlewRatePolicy, ValueRangePolicy


def test_default_config():
    cfg = load_config(None)
    assert cfg.transport == "stdio"
    assert cfg.role is None
    assert cfg.write_devices == []
    assert cfg.audit_log is None


@pytest.mark.parametrize(
    ("data", "error_type", "match"),
    [
        (
            {"policies": {"write_devices": ["Z:ACLTST"], "value_range": {"Z:ACLTST": [0, 1]}}},
            ValueError,
            "value_range",
        ),
        ({"policies": {"write_devices": "Z:*"}}, TypeError, "must be an array"),
    ],
)
def test_unsafe_policy_config_fails_closed(data, error_type, match):
    with pytest.raises(error_type, match=match):
        MCPConfig.from_dict(data)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib requires 3.11+")
def test_load_from_toml_string():
    import tomllib

    raw = """
[server]
transport = "sse"
port = 9090
role = "testing"
audit_log = "audit.jsonl"

[policies]
write_devices = ["Z:ACLTST", "Z:CUBE_Z"]
allow_raw = ["B:HS*"]

[policies.value_ranges]
"Z:ACLTST" = [0.0, 100.0]

[policies.slew_rates]
"Z:ACLTST" = { max_step = 5.0 }
"""
    data = tomllib.loads(raw)
    cfg = MCPConfig.from_dict(data)
    assert cfg.transport == "sse"
    assert cfg.port == 9090
    assert cfg.role == "testing"
    assert cfg.write_devices == ["Z:ACLTST", "Z:CUBE_Z"]
    assert cfg.allow_raw == ["B:HS*"]
    assert cfg.value_ranges == {"Z:ACLTST": (0.0, 100.0)}
    assert cfg.slew_rates == {"Z:ACLTST": {"max_step": 5.0}}
    assert cfg.audit_log == "audit.jsonl"


def test_build_policies_empty():
    cfg = load_config(None)
    policies = build_policies(cfg)
    assert policies == []


def test_build_policies_write_devices():
    cfg = MCPConfig(write_devices=["Z:ACLTST"])
    policies = build_policies(cfg)
    assert len(policies) == 1
    assert isinstance(policies[0], DeviceAccessPolicy)
    assert policies[0].allows_writes is True


def test_build_policies_full():
    cfg = MCPConfig(
        write_devices=["Z:ACLTST"],
        value_ranges={"Z:ACLTST": (0.0, 100.0)},
        slew_rates={"Z:ACLTST": {"max_step": 5.0}},
    )
    policies = build_policies(cfg)
    assert len(policies) == 3
    assert isinstance(policies[0], DeviceAccessPolicy)
    assert isinstance(policies[1], ValueRangePolicy)
    assert isinstance(policies[2], SlewRatePolicy)


def test_transport_options_are_validated_after_overrides():
    with pytest.raises(ValueError, match="only valid"):
        MCPConfig(port=9090).finalized()

    assert MCPConfig(transport="sse").finalized().port == 8000
