"""TOML config parsing and policy chain construction for MCP server."""

import logging
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]
from dataclasses import dataclass, field

from pacsys.supervised._policies import (
    DeviceAccessPolicy,
    Policy,
    SlewLimit,
    SlewRatePolicy,
    ValueRangePolicy,
)

logger = logging.getLogger("pacsys.mcp")


@dataclass
class MCPConfig:
    """Parsed MCP server configuration."""

    transport: str = "stdio"
    port: int = 8000
    role: str | None = None
    audit_log: str | None = None
    write_devices: list[str] = field(default_factory=list)
    value_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    slew_rates: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "MCPConfig":
        server = data.get("server", {})
        policies = data.get("policies", {})

        value_ranges = {}
        for dev, bounds in policies.get("value_ranges", {}).items():
            value_ranges[dev] = (float(bounds[0]), float(bounds[1]))

        return cls(
            transport=server.get("transport", "stdio"),
            port=server.get("port", 8000),
            role=server.get("role"),
            audit_log=server.get("audit_log"),
            write_devices=policies.get("write_devices", []),
            value_ranges=value_ranges,
            slew_rates=policies.get("slew_rates", {}),
        )


def load_config(path: str | None) -> MCPConfig:
    """Load config from TOML file, or return defaults if path is None."""
    if path is None:
        return MCPConfig()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return MCPConfig.from_dict(data)


def build_policies(cfg: MCPConfig) -> list[Policy]:
    """Construct policy chain from config."""
    policies: list[Policy] = []

    if cfg.write_devices:
        policies.append(DeviceAccessPolicy(patterns=cfg.write_devices, mode="allow", action="set"))

    if cfg.value_ranges:
        policies.append(ValueRangePolicy(limits=cfg.value_ranges))

    if cfg.slew_rates:
        limits = {}
        for dev, params in cfg.slew_rates.items():
            limits[dev] = SlewLimit(**params)
        policies.append(SlewRatePolicy(limits=limits))

    return policies
