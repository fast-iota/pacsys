import tomllib
from pacsys.mcp._config import load_config, build_policies, MCPConfig
from pacsys.supervised._policies import DeviceAccessPolicy, ValueRangePolicy, SlewRatePolicy


def test_default_config():
    cfg = load_config(None)
    assert cfg.transport == "stdio"
    assert cfg.role is None
    assert cfg.write_devices == []
    assert cfg.audit_log is None


def test_load_from_toml_string():
    raw = """
[server]
transport = "sse"
port = 9090
role = "testing"
audit_log = "audit.jsonl"

[policies]
write_devices = ["Z:ACLTST", "Z:CUBE_Z"]

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
