"""
Ramp table manipulation for ACNET corrector magnets.

Ramp tables are 64-point arrays stored as raw bytes in the SETTING property.
Each point is 4 bytes little-endian: int16 value followed by int16 time.
A slot has 64 points (256 bytes total).

Wire format per point (matches Java RampDevice):
    byte[0:1] = value (int16 LE)  -- Java: table[j][1]
    byte[2:3] = time  (int16 LE)  -- Java: table[j][0]

Java convention uses table[j][0]=time, table[j][1]=value, but the wire
format is value-first. See RampDevice.getFtTable() and setFtTable().

Value scaling follows the standard ACNET two-stage transform chain:
    Forward:  engineering = common_transform(primary_transform(raw))
    Inverse:  raw = inverse_primary_transform(inverse_common_transform(eng))
Java equivalent: pdudcu (raw→eng) and pdcuud (eng→raw) via ReadSetScaling.

Time scaling: raw times on the wire are clock ticks. The card's update rate
(update_rate_hz) determines the tick period. Times are converted to/from
microseconds:
    Forward:  time_us = raw_ticks * (1e6 / update_rate_hz)
    Inverse:  raw_ticks = round(time_us * update_rate_hz / 1e6)
Java equivalent: getFtScaledTable() / setFtScaledTable() via
getScaledUpdateFrequency(). Card types and their rates:
    453 (CAMAC):  720 Hz fixed
    465/466 (CAMAC): 1 / 5 / 10 KHz (configurable)
    473 (CAMAC):  100 KHz fixed (Booster correctors)

Example usage:
    ramp = BoosterRamp.read("B:HS23T", slot=0)
    print(ramp.values)  # engineering units (Amps)
    print(ramp.times)   # microseconds (float64)

    with BoosterRamp.modify("B:HS23T", slot=0) as ramp:
        ramp.values[10] = 2.5  # set point 10 to 2.5 Amps

    # Custom machine type -- override transform functions
    class MyRamp(Ramp):
        update_rate_hz = 5000  # 5 KHz card

        @classmethod
        def primary_transform(cls, raw):
            return raw / 1638.4
        @classmethod
        def common_transform(cls, primary):
            return primary * 2.0
        @classmethod
        def inverse_common_transform(cls, common):
            return common / 2.0
        @classmethod
        def inverse_primary_transform(cls, primary):
            return primary * 1638.4
"""

from __future__ import annotations

import struct

import numpy as np
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from pacsys.backends import Backend
    from pacsys.types import WriteResult


def _get_backend(backend: Optional["Backend"]) -> "Backend":
    """Get backend, using global default if none specified."""
    if backend is not None:
        return backend
    from pacsys import _get_global_backend

    return _get_global_backend()


__all__ = [
    "Ramp",
    "CorrectorRamp",
    "BoosterRamp",
]


class Ramp:
    """Ramp table (64 points of time/value pairs).

    Subclasses MUST override the four transform classmethods to define
    the raw <-> primary <-> common (engineering) unit conversions.

    Forward:  engineering = common_transform(primary_transform(raw))
    Inverse:  raw = inverse_primary_transform(inverse_common_transform(engineering))
    """

    POINTS_PER_SLOT: ClassVar[int] = 64
    BYTES_PER_POINT: ClassVar[int] = 4  # int16 value + int16 time

    # Card update rate in Hz. Determines tick period for time conversion.
    # Card types: 453=720Hz, 465/466=1/5/10KHz, 473=100KHz.
    update_rate_hz: ClassVar[int] = 10_000

    # Optional validation bounds (override in subclass).
    # max_time is in microseconds.
    max_value: ClassVar[float | None] = None
    max_time: ClassVar[float | None] = None

    # --- Transform functions (override in subclass) ---

    @classmethod
    def primary_transform(cls, raw: np.ndarray) -> np.ndarray:
        """Convert raw int16 values to primary (database) units."""
        raise NotImplementedError("Subclass must implement primary_transform")

    @classmethod
    def common_transform(cls, primary: np.ndarray) -> np.ndarray:
        """Convert primary units to common (engineering) units."""
        raise NotImplementedError("Subclass must implement common_transform")

    @classmethod
    def inverse_common_transform(cls, common: np.ndarray) -> np.ndarray:
        """Convert common (engineering) units back to primary units."""
        raise NotImplementedError("Subclass must implement inverse_common_transform")

    @classmethod
    def inverse_primary_transform(cls, primary: np.ndarray) -> np.ndarray:
        """Convert primary units back to raw int16 values."""
        raise NotImplementedError("Subclass must implement inverse_primary_transform")

    def __init__(self, values: np.ndarray, times: np.ndarray):
        """Args:
        values: Engineering-unit amplitudes (64 points, float64).
        times: Delta times in microseconds (64 points, float64).
        """
        if len(values) != self.POINTS_PER_SLOT:
            raise ValueError(f"Expected {self.POINTS_PER_SLOT} values, got {len(values)}")
        if len(times) != self.POINTS_PER_SLOT:
            raise ValueError(f"Expected {self.POINTS_PER_SLOT} times, got {len(times)}")
        self.values = np.asarray(values, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)

    @classmethod
    def _slot_bytes(cls) -> int:
        return cls.POINTS_PER_SLOT * cls.BYTES_PER_POINT

    @classmethod
    def _make_drf(cls, device: str, slot: int) -> str:
        """Build DRF for a ramp slot using byte range (like alarm_block.py)."""
        offset = slot * cls._slot_bytes()
        length = cls._slot_bytes()
        return f"{device}.SETTING{{{offset}:{length}}}.RAW@I"

    @classmethod
    def _tick_us(cls) -> float:
        """Microseconds per clock tick."""
        return 1_000_000.0 / cls.update_rate_hz

    def _validate(self) -> None:
        """Validate ramp values and times are within bounds."""
        if self.max_value is not None and np.any(np.abs(self.values) > self.max_value):
            raise ValueError(
                f"Ramp values exceed max {self.max_value}: max(|values|) = {np.max(np.abs(self.values)):.4f}"
            )
        if self.max_time is not None and np.any(np.abs(self.times) > self.max_time):
            raise ValueError(
                f"Ramp times exceed max {self.max_time} us: max(|times|) = {np.max(np.abs(self.times)):.1f} us"
            )

    @classmethod
    def from_bytes(cls, data: bytes) -> "Ramp":
        """Parse raw ramp table bytes.

        Wire format per 4-byte point (little-endian, same as Java RampDevice):
            byte[0:1] = value (int16 LE)  -- F(t) amplitude
            byte[2:3] = time  (int16 LE)  -- delta time
        """
        expected = cls._slot_bytes()
        if len(data) != expected:
            raise ValueError(f"Ramp table requires {expected} bytes, got {len(data)}")

        fmt = f"<{cls.POINTS_PER_SLOT * 2}h"
        unpacked = struct.unpack(fmt, data)

        # Even indices = values, odd indices = times (value-first wire order)
        raw_values = np.array(unpacked[0::2], dtype=np.int16)
        raw_times = np.array(unpacked[1::2], dtype=np.int16)

        primary = cls.primary_transform(raw_values.astype(np.float64))
        eng_values = cls.common_transform(primary)
        times_us = raw_times.astype(np.float64) * cls._tick_us()
        return cls(values=eng_values, times=times_us)

    def to_bytes(self) -> bytes:
        """Serialize ramp table to raw bytes (value-first wire order)."""
        self._validate()
        primary = self.inverse_common_transform(self.values)
        raw_values_f = np.round(self.inverse_primary_transform(primary))
        raw_times_f = np.round(self.times / self._tick_us())

        # Check for non-finite values (NaN/Inf bypass comparison checks)
        if not np.all(np.isfinite(raw_values_f)):
            raise ValueError("Raw values contain NaN or Inf after inverse transform")
        if not np.all(np.isfinite(raw_times_f)):
            raise ValueError("Raw times contain NaN or Inf after time conversion")

        # Check int16 bounds before cast (numpy silently wraps on overflow)
        i16 = np.iinfo(np.int16)
        if np.any(raw_values_f < i16.min) or np.any(raw_values_f > i16.max):
            raise ValueError(
                f"Raw values overflow int16 [{i16.min}, {i16.max}]: "
                f"range [{raw_values_f.min():.0f}, {raw_values_f.max():.0f}]"
            )
        if np.any(raw_times_f < i16.min) or np.any(raw_times_f > i16.max):
            raise ValueError(
                f"Raw times overflow int16 [{i16.min}, {i16.max}]: "
                f"range [{raw_times_f.min():.0f}, {raw_times_f.max():.0f}]"
            )

        raw_values = raw_values_f.astype(np.int16)
        raw_times = raw_times_f.astype(np.int16)

        fmt = f"<{self.POINTS_PER_SLOT * 2}h"
        pairs = []
        for v, t in zip(raw_values, raw_times):
            pairs.append(int(v))  # value first (matches Java setFtTable)
            pairs.append(int(t))  # time second
        return struct.pack(fmt, *pairs)

    @classmethod
    def read(
        cls,
        device: str,
        slot: int = 0,
        backend: Optional["Backend"] = None,
    ) -> "Ramp":
        """Read a ramp table from a corrector magnet.

        Args:
            device: Device name or DRF string
            slot: Ramp slot index (default 0)
            backend: Optional backend. If None, uses global default.

        Raises:
            DeviceError: If read fails
            TypeError: If response is not bytes
        """
        from pacsys.drf_utils import get_device_name
        from pacsys.errors import DeviceError

        be = _get_backend(backend)
        name = get_device_name(device)
        drf = cls._make_drf(name, slot)

        reading = be.get(drf)
        if reading.is_error:
            raise DeviceError(drf, reading.facility_code, reading.error_code, reading.message)
        if not isinstance(reading.value, bytes):
            raise TypeError(f"Expected bytes, got {type(reading.value).__name__}")

        return cls.from_bytes(reading.value)

    def write(
        self,
        device: str,
        slot: int = 0,
        backend: Optional["Backend"] = None,
    ) -> WriteResult:
        """Write ramp table to a corrector magnet.

        Args:
            device: Device name or DRF string
            slot: Ramp slot index (default 0)
            backend: Optional backend. If None, uses global default.

        Returns:
            WriteResult from the backend

        Raises:
            RuntimeError: If write fails
        """
        from pacsys.drf_utils import get_device_name

        be = _get_backend(backend)
        name = get_device_name(device)
        drf = self._make_drf(name, slot)

        result = be.write(drf, self.to_bytes())
        if not result.success:
            raise RuntimeError(f"Failed to write ramp table: {result.message}")
        return result

    @classmethod
    def modify(
        cls,
        device: str,
        slot: int = 0,
        backend: Optional["Backend"] = None,
    ):
        """Context manager for read-modify-write.

        Reads on entry, writes on exit only if bytes changed.
        """
        return _RampModifyContext(cls, device, slot, backend)

    def __repr__(self) -> str:
        n_active = int(np.count_nonzero(self.values))
        return f"{type(self).__name__}({n_active}/{self.POINTS_PER_SLOT} active points)"

    def __str__(self) -> str:
        lines = [f"{type(self).__name__} ({self.POINTS_PER_SLOT} points):"]
        for i, (v, t) in enumerate(zip(self.values, self.times)):
            if v != 0.0 or t != 0.0:
                lines.append(f"  [{i:2d}] t={t:8.1f}us  value={v:.4f}")
        if len(lines) == 1:
            lines.append("  (all zeros)")
        return "\n".join(lines)


class BoosterRamp(Ramp):
    """Booster corrector ramp table (473 CAMAC card).

    Primary transform: raw / 3276.8
    Common transform:  primary * 4.0
    Combined: engineering = raw / 819.2 (Amps)
    Update rate: 100 KHz fixed (10 us/tick). One Booster cycle = 66.67 ms (15 Hz).
    Java ref: RampDevice473.UPDATE_FREQUENCY = 100000.
    """

    update_rate_hz: ClassVar[int] = 100_000  # 473 CAMAC card: 100 KHz fixed, 10 us/tick
    max_value: ClassVar[float | None] = 1000.0
    max_time: ClassVar[float | None] = 66_660.0  # 6666 ticks * 10 us ≈ one Booster cycle

    @classmethod
    def primary_transform(cls, raw: np.ndarray) -> np.ndarray:
        return raw / 3276.8

    @classmethod
    def common_transform(cls, primary: np.ndarray) -> np.ndarray:
        return primary * 4.0

    @classmethod
    def inverse_common_transform(cls, common: np.ndarray) -> np.ndarray:
        return common / 4.0

    @classmethod
    def inverse_primary_transform(cls, primary: np.ndarray) -> np.ndarray:
        return primary * 3276.8


class _RampModifyContext:
    """Context manager for ramp read-modify-write."""

    def __init__(self, cls, device: str, slot: int, backend: Optional["Backend"]):
        self._cls = cls
        self._device = device
        self._slot = slot
        self._backend = backend
        self._ramp: Ramp | None = None
        self._initial_bytes: bytes | None = None

    def __enter__(self) -> Ramp:
        from pacsys.drf_utils import get_device_name
        from pacsys.errors import DeviceError

        be = _get_backend(self._backend)
        name = get_device_name(self._device)
        drf = self._cls._make_drf(name, self._slot)

        reading = be.get(drf)
        if reading.is_error:
            raise DeviceError(drf, reading.facility_code, reading.error_code, reading.message)
        if not isinstance(reading.value, bytes):
            raise TypeError(f"Expected bytes, got {type(reading.value).__name__}")

        self._initial_bytes = reading.value
        self._ramp = self._cls.from_bytes(reading.value)
        return self._ramp

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        if exc_type is not None or self._ramp is None or self._initial_bytes is None:
            return False

        current_bytes = self._ramp.to_bytes()
        if current_bytes != self._initial_bytes:
            from pacsys.drf_utils import get_device_name

            be = _get_backend(self._backend)
            name = get_device_name(self._device)
            drf = self._cls._make_drf(name, self._slot)

            result = be.write(drf, current_bytes)
            if not result.success:
                raise RuntimeError(f"Failed to write ramp table: {result.message}")

        return False


# Deprecated alias
CorrectorRamp = Ramp
