"""
Digital status (basic status) helper for ACNET devices.

DigitalStatus is a frozen value object representing a device's status bit field.
It is constructed from data already fetched by a backend or Device API — it does
no I/O itself.

Two construction paths:
  - from_bit_arrays(): from BIT_VALUE + BIT_NAMES + BIT_VALUES (full picture)
  - from_status_dict(): from a BasicStatus reading dict (legacy 5-attribute or gRPC map)

Example usage:
    # Via Device API (future)
    status = dev.digital_status()

    # Manual construction from backend responses
    readings = backend.get_many([
        "Z:ACLTST.STATUS.BIT_VALUE@I",
        "Z:ACLTST.STATUS.BIT_NAMES@I",
        "Z:ACLTST.STATUS.BIT_VALUES@I",
    ])
    status = DigitalStatus.from_bit_arrays(
        device="Z:ACLTST",
        raw_value=int(readings[0].value),
        bit_names=readings[1].value,
        bit_values=readings[2].value,
    )

    print(status)
    # Z:ACLTST status=0x0002
    #   On:       No
    #   Ready:    Yes
    #   Polarity: Minus

    status["Ready"]       # StatusBit(position=1, name='Ready', ...)
    status[0]             # StatusBit at bit position 0
    status.on             # True/False/None

    for bit in status:
        print(f"{bit.name}: {bit.value} (bit {bit.position}={'1' if bit.is_set else '0'})")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, overload

__all__ = ["StatusBit", "DigitalStatus"]

# Legacy attribute names used by PC/SDD and DMQ backends (lowercase keys)
_LEGACY_KEYS = ("on", "ready", "remote", "positive", "ramp")

# Map from legacy key to conventional display names for matching gRPC labels
_LEGACY_DISPLAY_NAMES: dict[str, tuple[str, ...]] = {
    "on": ("on", "on/off", "pwr", "power"),
    "ready": ("ready", "ready/tripped", "sts", "status"),
    "remote": ("remote", "remote/local", "ctrl", "control"),
    "positive": ("positive", "polarity", "pol"),
    "ramp": ("ramp", "ramp/dc", "ref", "reference"),
}


@dataclass(frozen=True)
class StatusBit:
    """A single bit in a digital status word.

    Attributes:
        position: Bit index (0-31).
        name: Label/description from the database.
        value: Current text representation (e.g. "Yes", "On", "Minus").
        is_set: True if the raw bit is 1.
    """

    position: int
    name: str
    value: str
    is_set: bool

    def __bool__(self) -> bool:
        return self.is_set


@dataclass(frozen=True)
class DigitalStatus:
    """Immutable representation of a device's digital status.

    Contains the raw integer status word and per-bit decoded information.
    Supports lookup by name or position, iteration, and formatted display.
    """

    device: str
    raw_value: int
    bits: tuple[StatusBit, ...]
    on: Optional[bool] = None
    ready: Optional[bool] = None
    remote: Optional[bool] = None
    positive: Optional[bool] = None
    ramp: Optional[bool] = None

    @classmethod
    def from_bit_arrays(
        cls,
        device: str,
        raw_value: int,
        bit_names: list[str],
        bit_values: list[str],
    ) -> DigitalStatus:
        """Construct from parallel arrays (BIT_NAMES + BIT_VALUES + BIT_VALUE).

        These correspond to the DPM sub-properties:
          - device.STATUS.BIT_VALUE  → raw_value (int)
          - device.STATUS.BIT_NAMES  → bit_names (list[str], indexed by bit position)
          - device.STATUS.BIT_VALUES → bit_values (list[str], indexed by bit position)

        Empty strings in bit_names indicate undefined bit positions (skipped).
        """
        n = max(len(bit_names), len(bit_values))
        bits = []
        for i in range(n):
            name = bit_names[i] if i < len(bit_names) else ""
            val = bit_values[i] if i < len(bit_values) else ""
            if not name and not val:
                continue
            is_set = bool(raw_value & (1 << i))
            bits.append(StatusBit(position=i, name=name, value=val, is_set=is_set))

        legacy = _infer_legacy_from_bits(bits)
        return cls(
            device=device,
            raw_value=raw_value,
            bits=tuple(bits),
            **legacy,
        )

    @classmethod
    def from_status_dict(
        cls,
        device: str,
        status_dict: dict,
        raw_value: Optional[int] = None,
    ) -> DigitalStatus:
        """Construct from a BasicStatus reading dict.

        Handles both backend formats:
          - PC/DMQ: {"on": True, "ready": False, ...} (bool values, lowercase keys)
          - gRPC:   {"On": "No", "Ready": "Yes", ...} (string values, display-name keys)

        If raw_value is not provided, it is reconstructed from the dict where possible.
        """
        # Detect format: bool values = PC/DMQ, string values = gRPC
        if status_dict and isinstance(next(iter(status_dict.values())), bool):
            return cls._from_legacy_dict(device, status_dict, raw_value)
        else:
            return cls._from_grpc_dict(device, status_dict, raw_value)

    @classmethod
    def _from_legacy_dict(
        cls,
        device: str,
        d: dict,
        raw_value: Optional[int],
    ) -> DigitalStatus:
        """Construct from PC/DMQ format: {"on": True, ...}."""
        legacy = {k: d.get(k) for k in _LEGACY_KEYS}

        # Build bits from what we know (limited — no bit positions or text)
        bits = []
        for i, key in enumerate(_LEGACY_KEYS):
            val = d.get(key)
            if val is None:
                continue
            bits.append(
                StatusBit(
                    position=i,
                    name=key.capitalize(),
                    value=str(val),
                    is_set=bool(val),
                )
            )

        return cls(
            device=device,
            raw_value=raw_value if raw_value is not None else _reconstruct_raw(bits),
            bits=tuple(bits),
            **legacy,
        )

    @classmethod
    def _from_grpc_dict(
        cls,
        device: str,
        d: dict,
        raw_value: Optional[int],
    ) -> DigitalStatus:
        """Construct from gRPC format: {"On": "No", "Ready": "Yes", ...}."""
        bits = []
        for i, (name, value) in enumerate(d.items()):
            # gRPC map doesn't carry bit positions; use insertion order as proxy
            bits.append(
                StatusBit(
                    position=i,
                    name=name,
                    value=value,
                    is_set=_text_is_true(value),
                )
            )

        legacy = _infer_legacy_from_bits(bits)
        return cls(
            device=device,
            raw_value=raw_value if raw_value is not None else _reconstruct_raw(bits),
            bits=tuple(bits),
            **legacy,
        )

    @classmethod
    def from_reading(cls, reading) -> DigitalStatus:
        """Construct from a Reading with value_type == BASIC_STATUS.

        Raises:
            ValueError: If the reading is not a basic status type or has no value.
        """
        from pacsys.types import ValueType

        if reading.value_type != ValueType.BASIC_STATUS:
            raise ValueError(f"Expected BASIC_STATUS reading, got {reading.value_type}")
        if reading.value is None:
            raise ValueError("Reading has no value")

        device = reading.name if hasattr(reading, "name") else reading.drf
        return cls.from_status_dict(device, reading.value)

    # --- Lookup ---

    @overload
    def __getitem__(self, key: str) -> StatusBit: ...
    @overload
    def __getitem__(self, key: int) -> StatusBit: ...

    def __getitem__(self, key):
        """Look up a bit by name (str) or position (int).

        Raises:
            KeyError: If no bit matches the name.
            IndexError: If no bit at the given position.
        """
        if isinstance(key, str):
            key_lower = key.lower()
            for bit in self.bits:
                if bit.name.lower() == key_lower:
                    return bit
            raise KeyError(f"No status bit named {key!r}")
        if isinstance(key, int):
            for bit in self.bits:
                if bit.position == key:
                    return bit
            raise IndexError(f"No status bit at position {key}")
        raise TypeError(f"Key must be str or int, got {type(key).__name__}")

    def get(self, key: str | int, default=None) -> StatusBit | None:
        """Look up a bit, returning default if not found."""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def __contains__(self, key: str | int) -> bool:
        return self.get(key) is not None

    # --- Iteration ---

    def __iter__(self) -> Iterator[StatusBit]:
        return iter(self.bits)

    def __len__(self) -> int:
        return len(self.bits)

    # --- Display ---

    def __str__(self) -> str:
        width = max((len(b.name) for b in self.bits), default=0)
        hex_digits = max(2, (self.raw_value.bit_length() + 3) // 4)
        lines = [f"{self.device} status=0x{self.raw_value:0{hex_digits}X}"]
        for bit in self.bits:
            lines.append(f"  {bit.name + ':':<{width + 1}} {bit.value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, str]:
        """Return {name: value} dict for all defined bits."""
        return {bit.name: bit.value for bit in self.bits}


def _text_is_true(text: str) -> bool:
    """Heuristic: does this text value represent a 'true'/'active' state?

    gRPC returns text values like "Yes", "On", "Ready", "Remote", "Plus", "Ramp".
    We consider a value 'true' unless it matches known 'false' patterns.
    """
    false_patterns = {
        "no",
        "off",
        "false",
        "tripped",
        "trip",
        "local",
        "negative",
        "minus",
        "dc",
        "0",
        "",
    }
    return text.strip().lower() not in false_patterns


def _infer_legacy_from_bits(bits: list[StatusBit]) -> dict:
    """Try to match bits to legacy attribute slots by name."""
    result: dict[str, Optional[bool]] = {k: None for k in _LEGACY_KEYS}
    for bit in bits:
        name_lower = bit.name.strip().lower()
        for legacy_key, patterns in _LEGACY_DISPLAY_NAMES.items():
            if name_lower in patterns:
                result[legacy_key] = bit.is_set
                break
    return result


def _reconstruct_raw(bits: list[StatusBit]) -> int:
    """Best-effort raw value reconstruction from known bits."""
    raw = 0
    for bit in bits:
        if bit.is_set:
            raw |= 1 << bit.position
    return raw
