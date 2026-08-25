"""TOML config parsing and policy chain construction for MCP server."""

import logging
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

from pacsys.supervised._policies import (
    DeviceAccessPolicy,
    Policy,
    SlewLimit,
    SlewRatePolicy,
    ValueRangePolicy,
)

logger = logging.getLogger("pacsys.mcp")

_TOP_LEVEL_KEYS = frozenset({"server", "policies"})
_SERVER_KEYS = frozenset({"transport", "port", "role", "audit_log"})
_POLICY_KEYS = frozenset({"write_devices", "value_ranges", "slew_rates", "allow_raw"})
_SLEW_KEYS = frozenset({"max_step", "max_rate"})


def _reject_unknown_keys(data: dict, allowed: frozenset[str], section: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in {section}: {', '.join(sorted(unknown))}")


def _require_table(value: object, section: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{section} must be a TOML table")
    return value


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class MCPConfig:
    """Parsed MCP server configuration."""

    transport: str = "stdio"
    port: int | None = None
    role: str | None = None
    audit_log: str | None = None
    write_devices: list[str] = field(default_factory=list)
    value_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    slew_rates: dict[str, dict] = field(default_factory=dict)
    allow_raw: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.transport not in ("stdio", "sse"):
            raise ValueError(f"transport must be 'stdio' or 'sse', got {self.transport!r}")
        if self.port is not None and (
            isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535
        ):
            raise ValueError(f"port must be an integer from 1 to 65535, got {self.port!r}")
        for field_name, value in (("role", self.role), ("audit_log", self.audit_log)):
            if value is not None:
                _nonempty_string(value, field_name)

        if not isinstance(self.write_devices, (list, tuple)):
            hint = '; use an array such as ["Z:*"]' if isinstance(self.write_devices, str) else ""
            raise TypeError(f"policies.write_devices must be an array of device patterns{hint}")
        write_devices = [_nonempty_string(pattern, "policies.write_devices entry") for pattern in self.write_devices]

        if not isinstance(self.allow_raw, (list, tuple)):
            hint = '; use an array such as ["B:HS*"]' if isinstance(self.allow_raw, str) else ""
            raise TypeError(f"policies.allow_raw must be an array of device patterns{hint}")
        allow_raw = [_nonempty_string(pattern, "policies.allow_raw entry") for pattern in self.allow_raw]

        if not isinstance(self.value_ranges, dict):
            raise TypeError("policies.value_ranges must be a TOML table")
        value_ranges: dict[str, tuple[float, float]] = {}
        for device, bounds in self.value_ranges.items():
            device = _nonempty_string(device, "policies.value_ranges key")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"policies.value_ranges.{device} must contain exactly two numbers")
            try:
                lower, upper = float(bounds[0]), float(bounds[1])
            except (TypeError, ValueError) as e:
                raise ValueError(f"policies.value_ranges.{device} must contain exactly two numbers") from e
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError(f"policies.value_ranges.{device} must contain finite bounds in ascending order")
            value_ranges[device] = (lower, upper)

        if not isinstance(self.slew_rates, dict):
            raise TypeError("policies.slew_rates must be a TOML table")
        slew_rates: dict[str, dict[str, float]] = {}
        for device, raw_params in self.slew_rates.items():
            device = _nonempty_string(device, "policies.slew_rates key")
            params = _require_table(raw_params, f"policies.slew_rates.{device}")
            _reject_unknown_keys(params, _SLEW_KEYS, f"[policies.slew_rates.{device}]")
            if not params:
                raise ValueError(f"policies.slew_rates.{device} requires max_step or max_rate")
            normalized = {}
            for name, raw_value in params.items():
                if isinstance(raw_value, bool):
                    raise TypeError(f"policies.slew_rates.{device}.{name} must be a positive finite number")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"policies.slew_rates.{device}.{name} must be a positive finite number") from e
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"policies.slew_rates.{device}.{name} must be a positive finite number")
                normalized[name] = value
            slew_rates[device] = normalized

        object.__setattr__(self, "write_devices", write_devices)
        object.__setattr__(self, "value_ranges", value_ranges)
        object.__setattr__(self, "slew_rates", slew_rates)
        object.__setattr__(self, "allow_raw", allow_raw)

    def finalized(self) -> "MCPConfig":
        """Validate transport-specific options after CLI overrides."""
        if self.transport == "stdio":
            if self.port is not None:
                raise ValueError("port is only valid with transport='sse'")
            return self
        return self if self.port is not None else replace(self, port=8000)

    @classmethod
    def from_dict(cls, data: dict) -> "MCPConfig":
        if not isinstance(data, dict):
            raise TypeError("MCP configuration must be a TOML table")
        _reject_unknown_keys(data, _TOP_LEVEL_KEYS, "top level")
        server = _require_table(data.get("server", {}), "[server]")
        policies = _require_table(data.get("policies", {}), "[policies]")
        _reject_unknown_keys(server, _SERVER_KEYS, "[server]")
        _reject_unknown_keys(policies, _POLICY_KEYS, "[policies]")

        return cls(
            transport=server.get("transport", "stdio"),
            port=server.get("port"),
            role=server.get("role"),
            audit_log=server.get("audit_log"),
            write_devices=policies.get("write_devices", []),
            value_ranges=policies.get("value_ranges", {}),
            slew_rates=policies.get("slew_rates", {}),
            allow_raw=policies.get("allow_raw", []),
        )


def load_config(path: str | None) -> MCPConfig:
    """Load config from TOML file, or return defaults if path is None."""
    if path is None:
        return MCPConfig()
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return MCPConfig.from_dict(data)


def build_policies(cfg: MCPConfig) -> list[Policy]:
    """Construct policy chain from config."""
    policies: list[Policy] = []

    if cfg.write_devices:
        policies.append(DeviceAccessPolicy(patterns=cfg.write_devices, mode="allow", action="set"))

    if cfg.value_ranges:
        policies.append(ValueRangePolicy(limits=cfg.value_ranges, allow_raw=cfg.allow_raw))

    if cfg.slew_rates:
        limits = {}
        for dev, params in cfg.slew_rates.items():
            limits[dev] = SlewLimit(**params)
        policies.append(SlewRatePolicy(limits=limits, allow_raw=cfg.allow_raw))

    return policies
