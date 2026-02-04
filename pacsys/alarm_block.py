"""
Alarm block manipulation for ACNET analog and digital alarms.

Alarm blocks are 20-byte structures that control alarm behavior.
Values are returned in engineering units (DPM applies device transforms).

Example usage:
    # Read an analog alarm
    alarm = AnalogAlarm.read("Z:ACLTST")
    print(f"Limits: {alarm.minimum} to {alarm.maximum}")

    # Modify with context manager (recommended)
    with AnalogAlarm.modify("Z:ACLTST") as alarm:
        alarm.minimum = 0.0
        alarm.maximum = 100.0
        alarm.bypass = False

    # Or manual read/modify/write
    alarm = AnalogAlarm.read("Z:ACLTST")
    alarm.minimum = 50.0
    alarm.write("Z:ACLTST")
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pacsys.backends import Backend


def _get_backend(backend: Optional["Backend"]) -> "Backend":
    """Get backend, using global default if none specified."""
    if backend is not None:
        return backend
    from pacsys import _get_global_backend

    return _get_global_backend()


__all__ = [
    "AlarmFlags",
    "DataLength",
    "LimitType",
    "DataType",
    "FTD",
    "AlarmBlock",
    "AnalogAlarm",
    "DigitalAlarm",
]


class AlarmFlags(IntFlag):
    """Alarm block status/flag bits."""

    # Common flags (both analog and digital)
    ENABLE = 1 << 0  # Alarm enable: 0=bypassed, 1=active (Java: isEnabled())
    BYPASS = ENABLE  # Deprecated alias -- use ENABLE
    BAD = 1 << 1  # GB: 0=good, 1=bad/alarm
    ABORT = 1 << 2  # AB: 0=no abort, 1=abort on alarm
    ABORT_INHIBIT = 1 << 3  # AI: 0=enabled, 1=inhibited
    Q0 = 1 << 5  # Data length bit 0
    Q1 = 1 << 6  # Data length bit 1
    DIGITAL = 1 << 7  # AD: 0=analog, 1=digital

    # Analog-only flags
    K0 = 1 << 8  # Limit type bit 0
    K1 = 1 << 9  # Limit type bit 1
    LOW = 1 << 11  # LO: Reading low
    HIGH = 1 << 12  # HI: Reading high

    # Aeolus flags
    EVENT = 1 << 13  # EV: 0=exception, 1=event
    LOG_EVENT = 1 << 14  # LE: Log event enabled
    DISPLAY_EVENT = 1 << 15  # DE: Display event enabled


class DataLength(IntEnum):
    """Data length (Q field) - size of alarmed value."""

    BYTES_1 = 0
    BYTES_2 = 1
    BYTES_4 = 2


class LimitType(IntEnum):
    """Analog alarm limit type (K field)."""

    NOM_TOL = 0  # Nominal/Tolerance
    MIN_MAX = 2  # Minimum/Maximum


class DataType(IntEnum):
    """Data type stored in FE system data byte 2 (analog only)."""

    UNKNOWN = 0
    SIGNED_INT = 1
    UNSIGNED_INT = 2
    FLOAT = 3


@dataclass
class FTD:
    """
    Frequency Time Descriptor - controls alarm sampling.

    Two modes:
    - Periodic: sample at fixed rate (60Hz tick-based)
    - Event: sample on TCLK clock event with optional delay
    """

    is_periodic: bool
    period_ticks: int = 0  # 60Hz ticks if periodic
    clock_event: int = 0  # TCLK event if event-based
    delay_10ms: int = 0  # Delay after event (max 127 = 1.27s)

    @classmethod
    def from_word(cls, value: int) -> FTD:
        """Parse 16-bit FTD value."""
        if value & 0x8000:  # Top bit set = event-based
            return cls(
                is_periodic=False,
                clock_event=value & 0xFF,
                delay_10ms=(value >> 8) & 0x7F,
            )
        else:
            return cls(is_periodic=True, period_ticks=value & 0x7FFF)

    def to_word(self) -> int:
        """Convert to 16-bit FTD value."""
        if self.is_periodic:
            return self.period_ticks & 0x7FFF
        return 0x8000 | ((self.delay_10ms & 0x7F) << 8) | (self.clock_event & 0xFF)

    @property
    def rate_hz(self) -> float:
        """Sampling rate in Hz (periodic mode only)."""
        if not self.is_periodic or self.period_ticks == 0:
            return 0.0
        return 60.0 / self.period_ticks

    @classmethod
    def periodic_hz(cls, hz: float) -> FTD:
        """Create periodic FTD at given Hz rate."""
        ticks = int(60.0 / hz) if hz > 0 else 0
        return cls(is_periodic=True, period_ticks=ticks)

    @classmethod
    def periodic_ticks(cls, ticks: int) -> FTD:
        """Create periodic FTD with 60Hz tick count."""
        return cls(is_periodic=True, period_ticks=ticks)

    @classmethod
    def on_event(cls, event: int, delay_ms: int = 0) -> FTD:
        """Create event-triggered FTD."""
        return cls(is_periodic=False, clock_event=event, delay_10ms=min(delay_ms // 10, 127))

    @classmethod
    def default(cls) -> FTD:
        """FTD of 0 = use device's default from database."""
        return cls(is_periodic=True, period_ticks=0)

    def __repr__(self) -> str:
        if self.is_periodic:
            if self.period_ticks == 0:
                return "FTD(default)"
            return f"FTD(periodic={self.rate_hz:.2f}Hz)"
        delay = f"+{self.delay_10ms * 10}ms" if self.delay_10ms else ""
        return f"FTD(event=${self.clock_event:02X}{delay})"


@dataclass
class AlarmBlock:
    """
    Base alarm block (20 bytes).

    This is the raw representation. Use AnalogAlarm or DigitalAlarm
    for type-specific field access.
    """

    flags: int = 0  # 16-bit flags
    value1_raw: bytes = field(default_factory=lambda: b"\x00\x00\x00\x00")
    value2_raw: bytes = field(default_factory=lambda: b"\x00\x00\x00\x00")
    tries_needed: int = 1
    tries_now: int = 0
    ftd: FTD = field(default_factory=FTD.default)
    fe_data: bytes = field(default_factory=lambda: b"\x00" * 6)

    # Structured data from DPM (engineering units) - populated by modify() context
    _structured: Optional[dict] = field(default=None, repr=False, compare=False)
    _initial_structured: Optional[dict] = field(default=None, repr=False, compare=False)

    _STRUCT = struct.Struct("<H4s4sBBH6s")  # 20 bytes, little-endian

    @classmethod
    def from_bytes(cls, data: bytes) -> AlarmBlock:
        """Parse 20-byte alarm block.

        Note: ACNET uses word-wise byte swapping, so bytes 10-11 (tries fields)
        are swapped in network order: [tries_now, tries_needed] instead of
        [tries_needed, tries_now].
        """
        if len(data) < 20:
            raise ValueError(f"Alarm block requires 20 bytes, got {len(data)}")
        flags, v1, v2, tries_now, tries_n, ftd_word, fe = cls._STRUCT.unpack(data[:20])
        return cls(
            flags=flags,
            value1_raw=v1,
            value2_raw=v2,
            tries_needed=tries_n,
            tries_now=tries_now,
            ftd=FTD.from_word(ftd_word),
            fe_data=fe,
        )

    def to_bytes(self) -> bytes:
        """Serialize to 20 bytes (network order with ACNET byte swapping)."""
        return self._STRUCT.pack(
            self.flags,
            self.value1_raw,
            self.value2_raw,
            self.tries_now,  # Swapped due to ACNET word-wise byte swap
            self.tries_needed,
            self.ftd.to_word(),
            self.fe_data,
        )

    # --- Flag accessors ---

    def _get_flag(self, flag: AlarmFlags) -> bool:
        return bool(self.flags & flag)

    def _set_flag(self, flag: AlarmFlags, value: bool) -> None:
        if value:
            self.flags |= flag
        else:
            self.flags &= ~flag

    @property
    def is_active(self) -> bool:
        """True if alarm is active (not bypassed)."""
        return self._get_flag(AlarmFlags.ENABLE)

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self._set_flag(AlarmFlags.ENABLE, value)
        if self._structured is not None:
            self._structured["alarm_enable"] = value

    @property
    def bypass(self) -> bool:
        """True if alarm is bypassed (inverse of is_active)."""
        return not self.is_active

    @bypass.setter
    def bypass(self, value: bool) -> None:
        self.is_active = not value

    @property
    def is_bad(self) -> bool:
        """True if currently in alarm state."""
        return self._get_flag(AlarmFlags.BAD)

    @property
    def abort_enabled(self) -> bool:
        """True if alarm can trigger abort (AB=1 and AI=0)."""
        return self._get_flag(AlarmFlags.ABORT) and not self._get_flag(AlarmFlags.ABORT_INHIBIT)

    @property
    def abort(self) -> bool:
        """Abort flag (AB bit)."""
        return self._get_flag(AlarmFlags.ABORT)

    @abort.setter
    def abort(self, value: bool) -> None:
        self._set_flag(AlarmFlags.ABORT, value)
        if self._structured is not None:
            self._structured["abort"] = value

    @property
    def abort_inhibit(self) -> bool:
        """Abort inhibit flag (AI bit)."""
        return self._get_flag(AlarmFlags.ABORT_INHIBIT)

    @abort_inhibit.setter
    def abort_inhibit(self, value: bool) -> None:
        self._set_flag(AlarmFlags.ABORT_INHIBIT, value)
        if self._structured is not None:
            self._structured["abort_inhibit"] = value

    @property
    def is_digital(self) -> bool:
        """True if this is a digital alarm block."""
        return self._get_flag(AlarmFlags.DIGITAL)

    @property
    def data_length(self) -> DataLength:
        """Data length (Q field)."""
        q = (self.flags >> 5) & 0x03
        return DataLength(min(q, 2))

    @data_length.setter
    def data_length(self, value: DataLength) -> None:
        self.flags = (self.flags & ~0x60) | ((value & 0x03) << 5)

    @property
    def data_bytes(self) -> int:
        """Number of bytes for alarmed value (1, 2, or 4)."""
        return [1, 2, 4][self.data_length]


class AnalogAlarm(AlarmBlock):
    """
    Analog alarm block with typed value access.

    Use minimum/maximum properties for engineering-unit limits. The raw alarm
    block also has a NOM_TOL limit_type (nominal/tolerance), but no backend
    protocol supports writing it in engineering units -- only minimum/maximum
    are available in structured alarm messages (DPM PC binary, gRPC protobuf,
    DMQ SDD). To set a nominal/tolerance alarm, convert manually:
        minimum = nominal - tolerance
        maximum = nominal + tolerance
    """

    @property
    def limit_type(self) -> LimitType:
        """Limit type (K field): NOM_TOL or MIN_MAX."""
        k = (self.flags >> 8) & 0x03
        return LimitType.MIN_MAX if k == 2 else LimitType.NOM_TOL

    @limit_type.setter
    def limit_type(self, value: LimitType) -> None:
        self.flags = (self.flags & ~0x300) | ((value & 0x03) << 8)

    @property
    def is_high(self) -> bool:
        """True if reading is above high limit."""
        return self._get_flag(AlarmFlags.HIGH)

    @property
    def is_low(self) -> bool:
        """True if reading is below low limit."""
        return self._get_flag(AlarmFlags.LOW)

    @property
    def data_type(self) -> DataType:
        """Data type from FE system data byte 2."""
        if len(self.fe_data) > 2:
            return DataType(self.fe_data[2] & 0x03)
        return DataType.UNKNOWN

    @data_type.setter
    def data_type(self, value: DataType) -> None:
        if len(self.fe_data) >= 6:
            fe = bytearray(self.fe_data)
            fe[2] = (fe[2] & ~0x03) | (value & 0x03)
            self.fe_data = bytes(fe)

    # --- Value accessors based on data type ---

    def _unpack_value(self, raw: bytes) -> int | float:
        """Unpack 4-byte value based on data_type and data_length."""
        n = self.data_bytes
        dt = self.data_type
        if dt == DataType.FLOAT:
            return struct.unpack("<f", raw)[0]
        elif dt == DataType.UNSIGNED_INT:
            return int.from_bytes(raw[:n], "little", signed=False)
        else:  # SIGNED_INT or UNKNOWN
            return int.from_bytes(raw[:n], "little", signed=True)

    def _pack_value(self, value: int | float) -> bytes:
        """Pack value to 4 bytes based on data_type and data_length."""
        dt = self.data_type
        if dt == DataType.FLOAT:
            return struct.pack("<f", float(value))
        else:
            n = self.data_bytes
            signed = dt != DataType.UNSIGNED_INT
            v = int(value)
            packed = v.to_bytes(n, "little", signed=signed)
            return packed.ljust(4, b"\x00" if v >= 0 else b"\xff")

    @property
    def value1(self) -> int | float:
        """Value 1 (minimum or nominal)."""
        return self._unpack_value(self.value1_raw)

    @value1.setter
    def value1(self, value: int | float) -> None:
        self.value1_raw = self._pack_value(value)

    @property
    def value2(self) -> int | float:
        """Value 2 (maximum or tolerance)."""
        return self._unpack_value(self.value2_raw)

    @value2.setter
    def value2(self, value: int | float) -> None:
        self.value2_raw = self._pack_value(value)

    # Raw value aliases (internal - use minimum/maximum for engineering units)
    @property
    def _min_value_raw(self) -> int | float:
        """Minimum in raw/primary units (internal)."""
        return self.value1

    @_min_value_raw.setter
    def _min_value_raw(self, value: int | float) -> None:
        self.value1 = value

    @property
    def _max_value_raw(self) -> int | float:
        """Maximum in raw/primary units (internal)."""
        return self.value2

    @_max_value_raw.setter
    def _max_value_raw(self, value: int | float) -> None:
        self.value2 = value

    # --- Engineering unit accessors (from structured DPM response) ---

    @property
    def minimum(self) -> float | None:
        """Minimum in engineering units.

        Returns None if structured data not available (e.g., from from_bytes()).
        Populated automatically by read() and modify().
        """
        if self._structured is not None:
            return self._structured.get("minimum")
        return None

    @minimum.setter
    def minimum(self, value: float) -> None:
        """Set minimum in engineering units."""
        if self._structured is None:
            raise ValueError("No structured data - use read() or modify() first")
        self._structured["minimum"] = value

    @property
    def maximum(self) -> float | None:
        """Maximum in engineering units.

        Returns None if structured data not available (e.g., from from_bytes()).
        Populated automatically by read() and modify().
        """
        if self._structured is not None:
            return self._structured.get("maximum")
        return None

    @maximum.setter
    def maximum(self, value: float) -> None:
        """Set maximum in engineering units."""
        if self._structured is None:
            raise ValueError("No structured data - use read() or modify() first")
        self._structured["maximum"] = value

    @classmethod
    def from_bytes(cls, data: bytes) -> AnalogAlarm:
        """Parse 20-byte analog alarm block."""
        base = AlarmBlock.from_bytes(data)
        return cls(
            flags=base.flags,
            value1_raw=base.value1_raw,
            value2_raw=base.value2_raw,
            tries_needed=base.tries_needed,
            tries_now=base.tries_now,
            ftd=base.ftd,
            fe_data=base.fe_data,
        )

    @classmethod
    def read(cls, device: str, backend: Optional["Backend"] = None, segment: int = 0) -> AnalogAlarm:
        """Read analog alarm block from device.

        Fetches both raw bytes and structured data to provide engineering unit
        access via minimum/maximum properties.

        Args:
            device: Device name or DRF string
            backend: Optional backend. If None, uses global default.
            segment: Alarm segment index (default 0)

        Raises:
            DeviceError: If device has no analog alarm (DBM_NOPROP) or other error
            TypeError: If response is not bytes
        """
        from pacsys.drf_utils import get_device_name
        from pacsys.errors import DeviceError

        be = _get_backend(backend)
        name = get_device_name(device)
        offset = segment * 20

        # Fetch both raw and structured in parallel
        raw_drf = f"{name}.ANALOG{{{offset}:20}}.RAW@I"
        struct_drf = f"{name}.ANALOG@I"
        readings = be.get_many([raw_drf, struct_drf])

        # Parse raw block
        raw_reading = readings[0]
        if raw_reading.is_error:
            raise DeviceError(raw_drf, raw_reading.facility_code, raw_reading.error_code, raw_reading.message)
        if not isinstance(raw_reading.value, bytes):
            raise TypeError(f"Expected bytes, got {type(raw_reading.value).__name__}")

        alarm = cls.from_bytes(raw_reading.value)

        # Attach structured data for engineering unit access
        struct_reading = readings[1]
        if struct_reading.ok and isinstance(struct_reading.value, dict):
            alarm._structured = struct_reading.value
            alarm._initial_structured = dict(struct_reading.value)

        return alarm

    def write(self, device: str, backend: Optional["Backend"] = None, segment: int = 0) -> None:
        """Write analog alarm block to device.

        Args:
            device: Device name or DRF string
            backend: Optional backend. If None, uses global default.
            segment: Alarm segment index (default 0)
        """
        from pacsys.drf_utils import get_device_name

        name = get_device_name(device)
        offset = segment * 20
        drf = f"{name}.ANALOG{{{offset}:20}}.RAW@I"
        result = _get_backend(backend).write(drf, self.to_bytes())
        if not result.success:
            raise RuntimeError(f"Failed to write alarm: {result.message}")

    @classmethod
    def modify(cls, device: str, backend: Optional["Backend"] = None, segment: int = 0):
        """Context manager for read-modify-write.

        Args:
            device: Device name or DRF string
            backend: Optional backend. If None, uses global default.
            segment: Alarm segment index (default 0)
        """
        return _AlarmModifyContext(cls, device, backend, segment)

    def __repr__(self) -> str:
        status = "active" if self.is_active else "bypassed"
        state = "ALARM" if self.is_bad else "ok"
        min_v = self.minimum if self.minimum is not None else self._min_value_raw
        max_v = self.maximum if self.maximum is not None else self._max_value_raw
        limits = f"min={min_v}, max={max_v}"
        if self.limit_type == LimitType.NOM_TOL:
            limits += " (NOM_TOL raw)"
        return f"AnalogAlarm({status}, {state}, {limits}, {self.ftd})"


class DigitalAlarm(AlarmBlock):
    """
    Digital alarm block with typed value access.

    Value 1 = nominal (expected bit pattern)
    Value 2 = mask (which bits to check)
    """

    @property
    def nominal(self) -> int:
        """Expected bit pattern."""
        n = self.data_bytes
        return int.from_bytes(self.value1_raw[:n], "little", signed=False)

    @nominal.setter
    def nominal(self, value: int) -> None:
        n = self.data_bytes
        self.value1_raw = value.to_bytes(n, "little", signed=False).ljust(4, b"\x00")

    @property
    def mask(self) -> int:
        """Bit mask for comparison."""
        n = self.data_bytes
        return int.from_bytes(self.value2_raw[:n], "little", signed=False)

    @mask.setter
    def mask(self, value: int) -> None:
        n = self.data_bytes
        self.value2_raw = value.to_bytes(n, "little", signed=False).ljust(4, b"\x00")

    @classmethod
    def from_bytes(cls, data: bytes) -> DigitalAlarm:
        """Parse 20-byte digital alarm block."""
        base = AlarmBlock.from_bytes(data)
        return cls(
            flags=base.flags,
            value1_raw=base.value1_raw,
            value2_raw=base.value2_raw,
            tries_needed=base.tries_needed,
            tries_now=base.tries_now,
            ftd=base.ftd,
            fe_data=base.fe_data,
        )

    @classmethod
    def read(cls, device: str, backend: Optional["Backend"] = None, segment: int = 0) -> DigitalAlarm:
        """Read digital alarm block from device.

        Fetches both raw bytes and structured data for consistent behavior
        with analog alarms.

        Args:
            device: Device name or DRF string
            backend: Optional backend. If None, uses global default.
            segment: Alarm segment index (default 0)

        Raises:
            DeviceError: If device has no digital alarm (DBM_NOPROP) or other error
            TypeError: If response is not bytes
        """
        from pacsys.drf_utils import get_device_name
        from pacsys.errors import DeviceError

        be = _get_backend(backend)
        name = get_device_name(device)
        offset = segment * 20

        # Fetch both raw and structured in parallel
        raw_drf = f"{name}.DIGITAL{{{offset}:20}}.RAW@I"
        struct_drf = f"{name}.DIGITAL@I"
        readings = be.get_many([raw_drf, struct_drf])

        # Parse raw block
        raw_reading = readings[0]
        if raw_reading.is_error:
            raise DeviceError(raw_drf, raw_reading.facility_code, raw_reading.error_code, raw_reading.message)
        if not isinstance(raw_reading.value, bytes):
            raise TypeError(f"Expected bytes, got {type(raw_reading.value).__name__}")

        alarm = cls.from_bytes(raw_reading.value)

        # Attach structured data
        struct_reading = readings[1]
        if struct_reading.ok and isinstance(struct_reading.value, dict):
            alarm._structured = struct_reading.value
            alarm._initial_structured = dict(struct_reading.value)

        return alarm

    def write(self, device: str, backend: Optional["Backend"] = None, segment: int = 0) -> None:
        """Write digital alarm block to device.

        Args:
            device: Device name or DRF string
            backend: Optional backend. If None, uses global default.
            segment: Alarm segment index (default 0)
        """
        from pacsys.drf_utils import get_device_name

        name = get_device_name(device)
        offset = segment * 20
        drf = f"{name}.DIGITAL{{{offset}:20}}.RAW@I"
        result = _get_backend(backend).write(drf, self.to_bytes())
        if not result.success:
            raise RuntimeError(f"Failed to write alarm: {result.message}")

    @classmethod
    def modify(cls, device: str, backend: Optional["Backend"] = None, segment: int = 0):
        """Context manager for read-modify-write.

        Args:
            device: Device name or DRF string
            backend: Optional backend. If None, uses global default.
            segment: Alarm segment index (default 0)
        """
        return _AlarmModifyContext(cls, device, backend, segment)

    def __repr__(self) -> str:
        status = "active" if self.is_active else "bypassed"
        state = "ALARM" if self.is_bad else "ok"
        return f"DigitalAlarm({status}, {state}, nom=0x{self.nominal:X}, mask=0x{self.mask:X}, {self.ftd})"


class _AlarmModifyContext:
    """Context manager for read-modify-write pattern.

    Reads both raw bytes and structured data on entry. On exit, determines
    whether to write raw, structured, or both based on what fields changed.

    Structured write is preferred when possible (min/max in engineering units).
    Raw write is required for fields not in structured response (ftd, fe_data, etc.).
    """

    # Fields available in structured response (can be written via structured write)
    _STRUCTURED_FIELDS = frozenset(
        {
            "is_active",
            "bypass",
            "abort",
            "abort_inhibit",
            "tries_needed",
            # min/max handled specially via engineering units
        }
    )

    # Fields only in raw block (require raw write)
    _RAW_ONLY_FIELDS = frozenset(
        {
            "ftd",
            "fe_data",
            "data_length",
            "limit_type",
            "data_type",
        }
    )

    def __init__(self, cls, device: str, backend: Optional["Backend"], segment: int):
        self._cls = cls
        self._device = device
        self._backend = backend
        self._segment = segment
        self._block: AnalogAlarm | DigitalAlarm | None = None
        self._initial_raw: bytes | None = None
        self._structured: dict | None = None

    def __enter__(self):
        from pacsys.drf_utils import get_device_name

        backend = _get_backend(self._backend)
        name = get_device_name(self._device)
        offset = self._segment * 20

        # Determine property name based on alarm type
        prop = "ANALOG" if self._cls is AnalogAlarm else "DIGITAL"

        # Read both raw and structured in parallel (same connection)
        raw_drf = f"{name}.{prop}{{{offset}:20}}.RAW@I"
        struct_drf = f"{name}.{prop}@I"

        readings = backend.get_many([raw_drf, struct_drf])

        # Parse raw block
        raw_reading = readings[0]
        if raw_reading.is_error:
            from pacsys.errors import DeviceError

            raise DeviceError(
                raw_drf,
                raw_reading.facility_code,
                raw_reading.error_code,
                raw_reading.message,
            )
        if not isinstance(raw_reading.value, bytes):
            raise TypeError(f"Expected bytes, got {type(raw_reading.value).__name__}")

        self._initial_raw = raw_reading.value
        self._block = self._cls.from_bytes(raw_reading.value)

        # Parse structured response
        struct_reading = readings[1]
        if struct_reading.ok and isinstance(struct_reading.value, dict):
            self._structured = struct_reading.value
            # Attach engineering unit values to block
            self._block._structured = self._structured
            self._block._initial_structured = dict(self._structured)

        return self._block

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        if exc_type is not None or self._block is None or self._initial_raw is None:
            return False

        from pacsys.drf_utils import get_device_name

        backend = _get_backend(self._backend)
        name = get_device_name(self._device)
        offset = self._segment * 20
        prop = "ANALOG" if self._cls is AnalogAlarm else "DIGITAL"

        # Determine what changed
        current_raw = self._block.to_bytes()
        raw_changed = current_raw != self._initial_raw

        eng_changed = False
        struct_changed = False
        s = self._block._structured
        init = self._block._initial_structured
        if s is not None and init is not None:
            # Check if engineering unit values changed
            if self._cls is AnalogAlarm:
                eng_changed = s.get("minimum") != init.get("minimum") or s.get("maximum") != init.get("maximum")
            else:  # DigitalAlarm
                eng_changed = s.get("nominal") != init.get("nominal") or s.get("mask") != init.get("mask")
            # Check if any structured flag changed
            for key in ("alarm_enable", "abort", "abort_inhibit", "tries_needed"):
                if s.get(key) != init.get(key):
                    struct_changed = True
                    break

        # Check if raw-only fields changed (ftd, fe_data, etc.)
        raw_only_changed = False
        if raw_changed:
            init_block = self._cls.from_bytes(self._initial_raw)
            if self._block.ftd.to_word() != init_block.ftd.to_word():
                raw_only_changed = True
            if self._block.fe_data != init_block.fe_data:
                raw_only_changed = True
            if self._block.data_length != init_block.data_length:
                raw_only_changed = True
            if isinstance(self._block, AnalogAlarm) and isinstance(init_block, AnalogAlarm):
                if self._block.limit_type != init_block.limit_type:
                    raw_only_changed = True

        # Decide write strategy
        if raw_only_changed and (eng_changed or struct_changed):
            # Both raw-only fields (ftd, fe_data) and engineering values changed.
            # Write structured first (eng values), then raw (for ftd/fe_data).
            self._write_structured(backend, name, prop)
            # Re-read to get updated raw bytes with new eng values, then
            # patch raw-only fields on top and write.
            raw_drf = f"{name}.{prop}{{{offset}:20}}.RAW@I"
            fresh = backend.read(raw_drf)
            if isinstance(fresh, bytes):
                patched = self._cls.from_bytes(fresh)
            else:
                patched = self._cls.from_bytes(current_raw)
            # Apply raw-only field changes from user's block onto the fresh read
            patched.ftd = self._block.ftd
            patched.fe_data = self._block.fe_data
            patched.data_length = self._block.data_length
            if isinstance(self._block, AnalogAlarm) and isinstance(patched, AnalogAlarm):
                patched.limit_type = self._block.limit_type
            result = backend.write(raw_drf, patched.to_bytes())
            if not result.success:
                raise RuntimeError(f"Failed to write alarm (raw): {result.message}")
        elif raw_only_changed:
            # Only raw-only fields changed (ftd, fe_data, etc.)
            raw_drf = f"{name}.{prop}{{{offset}:20}}.RAW@I"
            result = backend.write(raw_drf, current_raw)
            if not result.success:
                raise RuntimeError(f"Failed to write alarm (raw): {result.message}")
        elif eng_changed or struct_changed:
            # Use structured write for engineering unit values
            self._write_structured(backend, name, prop)
        elif raw_changed:
            # Flags changed but not via structured - use structured write anyway
            self._write_structured(backend, name, prop)

        return False

    def _write_structured(self, backend, name: str, prop: str):
        """Write alarm using structured (engineering unit) values."""
        if self._block is None:
            return

        s = self._block._structured
        if s is None:
            # Fall back to raw write if no structured data
            offset = self._segment * 20
            raw_drf = f"{name}.{prop}{{{offset}:20}}.RAW@I"
            result = backend.write(raw_drf, self._block.to_bytes())
            if not result.success:
                raise RuntimeError(f"Failed to write alarm (raw fallback): {result.message}")
            return

        drf = f"{name}.{prop}@I"

        # Build write value from structured data
        if self._cls is AnalogAlarm:
            # Write as array: [minimum, maximum] with appropriate flags via separate writes
            # For now, write the full structured dict
            write_val = {
                "minimum": s["minimum"],
                "maximum": s["maximum"],
                "alarm_enable": s["alarm_enable"],
                "abort": s["abort"],
                "abort_inhibit": s["abort_inhibit"],
                "tries_needed": s["tries_needed"],
            }
        else:  # DigitalAlarm
            write_val = {
                "nominal": s["nominal"],
                "mask": s["mask"],
                "alarm_enable": s["alarm_enable"],
                "abort": s["abort"],
                "abort_inhibit": s["abort_inhibit"],
                "tries_needed": s["tries_needed"],
            }

        result = backend.write(drf, write_val)
        if not result.success:
            raise RuntimeError(f"Failed to write alarm (structured): {result.message}")
