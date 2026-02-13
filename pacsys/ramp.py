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
    ramp = BoosterHVRamp.read("B:HS23T", slot=0)
    print(ramp.values)  # engineering units (Amps)
    print(ramp.times)   # microseconds (float64)

    with BoosterHVRamp.modify("B:HS23T", slot=0) as ramp:
        ramp.values[10] = 2.5  # set point 10 to 2.5 Amps

    # Custom machine type using Scaler (recommended for standard ACNET transforms):
    from pacsys import Scaler
    class MyRamp(Ramp):
        update_rate_hz = 5000
        scaler = Scaler(p_index=2, c_index=2, constants=(2.0, 1.0, 0.0), input_len=2)

    # Custom machine type with manual transforms (for non-standard scaling):
    class MyCustomRamp(Ramp):
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

from pacsys.scaling import Scaler

if TYPE_CHECKING:
    from pacsys.backends import Backend
    from pacsys.types import WriteResult


def _get_backend(backend: Optional["Backend"]) -> "Backend":
    """Get backend, using global default if none specified."""
    if backend is not None:
        return backend
    from pacsys import _get_global_backend

    return _get_global_backend()


def _validate_device_name(drf: str) -> None:
    """Raise ValueError if drf contains property, range, event, field, or extra.

    Ramp functions build their own DRFs from the device name, so callers
    must pass bare device names (e.g. "B:HS23T"), not full DRFs.
    Qualifier-based names like "B_HS23T" are accepted (implicit property).
    """
    from pacsys.drf3 import parse_request

    req = parse_request(drf)
    if req.property_explicit:
        raise ValueError(
            f"Expected bare device name, got explicit property in {drf!r}. Pass only the device name (e.g. 'B:HS23T')."
        )
    if req.range is not None:
        raise ValueError(
            f"Expected bare device name, got range in {drf!r}. Pass only the device name (e.g. 'B:HS23T')."
        )
    if req.event is not None and req.event.mode != "U":
        raise ValueError(
            f"Expected bare device name, got event in {drf!r}. Pass only the device name (e.g. 'B:HS23T')."
        )
    if req.extra is not None:
        raise ValueError(
            f"Expected bare device name, got extra in {drf!r}. Pass only the device name (e.g. 'B:HS23T')."
        )


__all__ = [
    "Ramp",
    "RecyclerQRamp",
    "RecyclerQRampGroup",
    "RecyclerSRamp",
    "RecyclerSRampGroup",
    "RecyclerSCRamp",
    "RecyclerSCRampGroup",
    "RecyclerHVSQRamp",
    "RecyclerHVSQRampGroup",
    "BoosterHVRamp",
    "BoosterHVRampGroup",
    "BoosterQRamp",
    "BoosterQRampGroup",
    "RampGroup",
    "read_ramps",
    "write_ramps",
]


class Ramp:
    """Ramp table (64 points of time/value pairs).

    Value scaling is handled in one of two ways:

    1. Set ``scaler`` to a ``Scaler`` instance (recommended for standard
       ACNET transforms — use parameters from the device database).

    2. Override the four transform classmethods for custom/non-standard
       transforms.

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

    # Scaler for standard ACNET transforms (alternative to overriding classmethods).
    scaler: ClassVar[Scaler | None] = None

    # --- Transform functions (override in subclass if scaler is not set) ---

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

    def __init__(
        self,
        values: np.ndarray,
        times: np.ndarray,
        device: str | None = None,
        slot: int | None = None,
    ):
        """Args:
        values: Engineering-unit amplitudes (64 points, float64).
        times: Delta times in microseconds (64 points, float64).
        device: Canonical device name (set by read(), None from from_bytes()).
        slot: Ramp slot index (set by read(), None from from_bytes()).
        """
        if len(values) != self.POINTS_PER_SLOT:
            raise ValueError(f"Expected {self.POINTS_PER_SLOT} values, got {len(values)}")
        if len(times) != self.POINTS_PER_SLOT:
            raise ValueError(f"Expected {self.POINTS_PER_SLOT} times, got {len(times)}")
        self.values = np.asarray(values, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.device = device
        self.slot = slot

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

        if cls.scaler is not None:
            eng_values = cls.scaler.scale(raw_values.astype(np.int64))
        else:
            primary = cls.primary_transform(raw_values.astype(np.float64))
            eng_values = cls.common_transform(primary)
        times_us = raw_times.astype(np.float64) * cls._tick_us()
        return cls(values=eng_values, times=times_us)

    def to_bytes(self) -> bytes:
        """Serialize ramp table to raw bytes (value-first wire order)."""
        from pacsys.scaling import ScalingError

        self._validate()
        scaler = type(self).scaler
        if scaler is not None:
            if not np.all(np.isfinite(self.values)):
                raise ValueError("Raw values contain NaN or Inf")
            try:
                raw_values_f = scaler.unscale(self.values).astype(np.float64)
            except ScalingError as e:
                raise ValueError(f"Raw values overflow int16: {e}") from None
        else:
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
            device: Bare device name DRF (e.g. "B:HS23T")
            slot: Ramp slot index (default 0)
            backend: Optional backend. If None, uses global default.

        Raises:
            ValueError: If device is not a bare device name
            DeviceError: If read fails
            TypeError: If response is not bytes
        """
        from pacsys.drf_utils import get_device_name
        from pacsys.errors import DeviceError

        _validate_device_name(device)
        be = _get_backend(backend)
        name = get_device_name(device)
        drf = cls._make_drf(name, slot)

        reading = be.get(drf)
        if reading.is_error:
            raise DeviceError(drf, reading.facility_code, reading.error_code, reading.message)
        if not isinstance(reading.value, bytes):
            raise TypeError(f"Expected bytes, got {type(reading.value).__name__}")

        ramp = cls.from_bytes(reading.value)
        ramp.device = name
        ramp.slot = slot
        return ramp

    def write(
        self,
        device: str | None = None,
        slot: int | None = None,
        backend: Optional["Backend"] = None,
    ) -> "WriteResult":
        """Write ramp table to a corrector magnet.

        Args:
            device: Bare device name DRF (e.g. "B:HS23T"). If None, uses stored device from read().
            slot: Ramp slot index. If None, uses stored slot from read().
            backend: Optional backend. If None, uses global default.

        Returns:
            WriteResult from the backend

        Raises:
            ValueError: If no device or slot available, or device is not a bare device name
            RuntimeError: If write fails
        """
        from pacsys.drf_utils import get_device_name

        if device is not None:
            _validate_device_name(device)
        device = device or self.device
        if device is None:
            raise ValueError("No device specified and none stored from read()")
        slot = slot if slot is not None else self.slot
        if slot is None:
            raise ValueError("No slot specified and none stored from read()")

        be = _get_backend(backend)
        name = get_device_name(device)
        drf = self._make_drf(name, slot)

        result = be.write(drf, self.to_bytes())
        if not result.success:
            raise RuntimeError(f"Failed to write ramp table: {result.message}")
        return result

    @classmethod
    def read_many(
        cls,
        devices: list[str],
        slot: int = 0,
        backend: Optional["Backend"] = None,
    ) -> list["Ramp"]:
        """Batched read of ramp tables from multiple devices.

        Args:
            devices: List of bare device names (e.g. ["B:HS23T", "B:HS24T"])
            slot: Ramp slot index (default 0)
            backend: Optional backend. If None, uses global default.

        Returns:
            List of Ramp instances with .device and .slot set
        """
        return read_ramps(cls, devices, slot=slot, backend=backend)

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
        _validate_device_name(device)
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


class RecyclerQRamp(Ramp):
    """Recycler quad ramp table (453 CAMAC card).

    Scaling: p_index=2 (raw / 3276.8), c_index=6 with C1=2.0, C2=1.0
    Combined: engineering = 2.0 * raw / 3276.8
    Update rate: 720 Hz fixed (1389 us/tick).
    """

    update_rate_hz: ClassVar[int] = 720  # 453 CAMAC card: 720 Hz fixed
    scaler: ClassVar[Scaler | None] = Scaler(p_index=2, c_index=6, constants=(2.0, 1.0), input_len=2)


class RecyclerSRamp(Ramp):
    """Recycler sextupole ramp table (453 CAMAC card).

    Scaling: p_index=2 (raw / 3276.8), c_index=6 with C1=12.0, C2=10.0
    Combined: engineering = 12.0 * raw / (3276.8 * 10.0)
    Update rate: 720 Hz fixed (1389 us/tick).
    """

    update_rate_hz: ClassVar[int] = 720  # 453 CAMAC card: 720 Hz fixed
    scaler: ClassVar[Scaler | None] = Scaler(p_index=2, c_index=6, constants=(12.0, 10.0), input_len=2)


class RecyclerSCRamp(Ramp):
    """Recycler sextupole corrector ramp table (C475 CAMAC card).

    Scaling: p_index=2 (raw / 3276.8), c_index=6 with C1=1.2000000477, C2=1.0
    Combined: engineering = 1.2000000477 * raw / 3276.8
    Update rate: 100 KHz fixed (10 us/tick).
    """

    update_rate_hz: ClassVar[int] = 100_000  # C475 CAMAC card: 100 KHz fixed
    scaler: ClassVar[Scaler | None] = Scaler(p_index=2, c_index=6, constants=(1.2000000477, 1.0), input_len=2)


class RecyclerHVSQRamp(Ramp):
    """Recycler H/V and skew quad corrector ramp table (453 CAMAC card).

    Scaling: p_index=2 (raw / 3276.8), c_index=6 with C1=12.0, C2=10.0
    Combined: engineering = 12.0 * raw / (3276.8 * 10.0)
    Update rate: 720 Hz fixed (1389 us/tick).
    """

    update_rate_hz: ClassVar[int] = 720  # 453 CAMAC card: 720 Hz fixed
    scaler: ClassVar[Scaler | None] = Scaler(p_index=2, c_index=6, constants=(12.0, 10.0), input_len=2)


class BoosterHVRamp(Ramp):
    """Booster corrector ramp table (473 CAMAC card).

    Scaling: p_index=2 (raw / 3276.8), c_index=6 with C1=4.0, C2=1.0
    Combined: engineering = 4.0 * raw / 3276.8
    Update rate: 100 KHz fixed (10 us/tick). One Booster cycle = 66.67 ms (15 Hz).
    Java ref: RampDevice473.UPDATE_FREQUENCY = 100000.
    """

    update_rate_hz: ClassVar[int] = 100_000  # 473 CAMAC card: 100 KHz fixed, 10 us/tick
    max_value: ClassVar[float | None] = 1000.0
    max_time: ClassVar[float | None] = 66_660.0  # 6666 ticks * 10 us ≈ one Booster cycle
    scaler: ClassVar[Scaler | None] = Scaler(p_index=2, c_index=6, constants=(4.0, 1.0), input_len=2)


class BoosterQRamp(Ramp):
    """Booster quad ramp table (C473 CAMAC card).

    Scaling: p_index=2 (raw / 3276.8), c_index=6 with C1=6.5, C2=1.0
    Combined: engineering = 6.5 * raw / 3276.8
    Update rate: 100 KHz fixed (10 us/tick).
    """

    update_rate_hz: ClassVar[int] = 100_000  # C473 CAMAC card: 100 KHz fixed
    max_time: ClassVar[float | None] = 66_660.0  # 6666 ticks * 10 us ≈ one Booster cycle
    scaler: ClassVar[Scaler | None] = Scaler(p_index=2, c_index=6, constants=(6.5, 1.0), input_len=2)


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
        self._ramp.device = name
        self._ramp.slot = self._slot
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


def read_ramps(
    cls: type[Ramp],
    devices: list[str],
    slot: int = 0,
    backend: Optional["Backend"] = None,
) -> list[Ramp]:
    """Batched read of ramp tables from multiple devices.

    Args:
        cls: Ramp subclass to use for parsing
        devices: List of bare device names (e.g. ["B:HS23T", "B:HS24T"])
        slot: Ramp slot index (default 0)
        backend: Optional backend. If None, uses global default.

    Returns:
        List of Ramp instances with .device and .slot set

    Raises:
        ValueError: If devices list is empty or any device is not a bare device name
        DeviceError: If any reading is an error
        TypeError: If any reading is not bytes
    """
    if not devices:
        raise ValueError("devices list must not be empty")
    for d in devices:
        _validate_device_name(d)

    from pacsys.drf_utils import get_device_name
    from pacsys.errors import DeviceError

    be = _get_backend(backend)
    names = [get_device_name(d) for d in devices]
    drfs = [cls._make_drf(name, slot) for name in names]

    readings = be.get_many(drfs)
    ramps: list[Ramp] = []
    for reading, name in zip(readings, names):
        if reading.is_error:
            raise DeviceError(reading.drf, reading.facility_code, reading.error_code, reading.message)
        if not isinstance(reading.value, bytes):
            raise TypeError(f"Expected bytes for {name}, got {type(reading.value).__name__}")
        ramp = cls.from_bytes(reading.value)
        ramp.device = name
        ramp.slot = slot
        ramps.append(ramp)
    return ramps


def write_ramps(
    ramps: "Ramp | list[Ramp] | RampGroup | list[Ramp | RampGroup]",
    *,
    slot: int | None = None,
    backend: Optional["Backend"] = None,
) -> list["WriteResult"]:
    """Batched write of ramp tables.

    Accepts a single Ramp, a list of Ramps, a RampGroup, or a mixed list.
    All are flattened into a single write_many call.

    Args:
        ramps: Ramp(s) or RampGroup(s) to write
        slot: Optional slot override (applies to all ramps)
        backend: Optional backend. If None, uses global default.

    Returns:
        List of WriteResult in same order as flattened ramps
    """
    from pacsys.drf_utils import get_device_name

    # Normalize to flat list[Ramp]
    flat: list[Ramp] = []
    items: list[Ramp | RampGroup] = [ramps] if isinstance(ramps, (Ramp, RampGroup)) else list(ramps)
    for item in items:
        if isinstance(item, RampGroup):
            flat.extend(item._to_ramps(slot))
        else:
            flat.append(item)

    be = _get_backend(backend)
    settings: list[tuple[str, bytes]] = []
    for ramp in flat:
        dev = ramp.device
        if dev is None:
            raise ValueError("Ramp has no device set — read() first or set .device")
        s = slot if slot is not None else ramp.slot
        if s is None:
            raise ValueError(f"No slot for device {dev} — read() first or set .slot")
        name = get_device_name(dev)
        drf = type(ramp)._make_drf(name, s)
        settings.append((drf, ramp.to_bytes()))
    return be.write_many(settings)  # type: ignore[arg-type]  # bytes is a valid Value


class RampGroup:
    """Group of ramp tables stored as 2D arrays (64 points x N devices).

    Subclasses must set the `base` class variable to a Ramp subclass.
    Use __getitem__ to get a view-backed Ramp for a single device.
    """

    base: ClassVar[type[Ramp]]

    def __init__(
        self,
        devices: list[str],
        values: np.ndarray,
        times: np.ndarray,
        slot: int = 0,
    ):
        n = len(devices)
        if len(set(devices)) != n:
            raise ValueError("Duplicate device names in RampGroup")
        pts = type(self).base.POINTS_PER_SLOT
        if values.shape != (pts, n):
            raise ValueError(f"Expected values shape ({pts}, {n}), got {values.shape}")
        if times.shape != (pts, n):
            raise ValueError(f"Expected times shape ({pts}, {n}), got {times.shape}")
        self.devices = list(devices)
        self.values = np.asarray(values, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.slot = slot
        self._device_map: dict[str, int] = {d: i for i, d in enumerate(devices)}

    def __getitem__(self, device: str) -> Ramp:
        idx = self._device_map[device]
        ramp = object.__new__(self.base)
        ramp.values = self.values[:, idx]
        ramp.times = self.times[:, idx]
        ramp.device = device
        ramp.slot = self.slot
        return ramp

    def __len__(self) -> int:
        return len(self.devices)

    def __iter__(self):
        return iter(self.devices)

    def __contains__(self, device: str) -> bool:
        return device in self._device_map

    def _to_ramps(self, slot: int | None = None, devices: list[str] | None = None) -> list[Ramp]:
        """Demux 2D arrays into individual Ramp objects."""
        targets = devices if devices is not None else self.devices
        target_slot = slot if slot is not None else self.slot
        ramps: list[Ramp] = []
        for i, dev in enumerate(targets):
            ramp = self.base(values=self.values[:, i], times=self.times[:, i])
            ramp.device = dev
            ramp.slot = target_slot
            ramps.append(ramp)
        return ramps

    @classmethod
    def read(
        cls,
        devices: list[str],
        slot: int = 0,
        backend: Optional["Backend"] = None,
    ) -> "RampGroup":
        """Batched read into a RampGroup.

        Args:
            devices: List of device names
            slot: Ramp slot index (default 0)
            backend: Optional backend. If None, uses global default.
        """
        ramps = read_ramps(cls.base, devices, slot=slot, backend=backend)
        values = np.column_stack([r.values for r in ramps])
        times = np.column_stack([r.times for r in ramps])
        return cls(
            devices=[r.device for r in ramps],  # type: ignore[misc]  # read_ramps sets .device
            values=values,
            times=times,
            slot=slot,
        )

    def write(
        self,
        *,
        devices: list[str] | None = None,
        slot: int | None = None,
        backend: Optional["Backend"] = None,
    ) -> list["WriteResult"]:
        """Write group to devices.

        Args:
            devices: Override target device names (must match column count)
            slot: Override slot index
            backend: Optional backend
        """
        if devices is not None:
            for d in devices:
                _validate_device_name(d)
        targets = devices if devices is not None else self.devices
        if len(targets) != self.values.shape[1]:
            raise ValueError(f"Expected {self.values.shape[1]} devices, got {len(targets)}")
        ramps = self._to_ramps(slot, targets)
        return write_ramps(ramps, backend=backend)

    @classmethod
    def modify(
        cls,
        devices: list[str],
        slot: int = 0,
        backend: Optional["Backend"] = None,
    ):
        """Context manager for batched read-modify-write.

        Reads on entry, writes changed devices on exit.
        """
        return _RampGroupModifyContext(cls, devices, slot, backend)


class _RampGroupModifyContext:
    """Context manager for RampGroup read-modify-write."""

    def __init__(
        self,
        cls: type[RampGroup],
        devices: list[str],
        slot: int,
        backend: Optional["Backend"],
    ):
        self._cls = cls
        self._devices = devices
        self._slot = slot
        self._backend = backend
        self._group: RampGroup | None = None
        self._initial_bytes: dict[str, bytes] = {}

    def __enter__(self) -> RampGroup:
        self._group = self._cls.read(self._devices, self._slot, self._backend)
        # Store initial bytes per device for change detection
        for i, dev in enumerate(self._group.devices):
            ramp = self._cls.base(
                values=self._group.values[:, i].copy(),
                times=self._group.times[:, i].copy(),
            )
            self._initial_bytes[dev] = ramp.to_bytes()
        return self._group

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        if exc_type is not None or self._group is None:
            return False

        from pacsys.drf_utils import get_device_name

        be = _get_backend(self._backend)
        settings: list[tuple[str, bytes]] = []
        for i, dev in enumerate(self._group.devices):
            ramp = self._cls.base(
                values=self._group.values[:, i],
                times=self._group.times[:, i],
            )
            current_bytes = ramp.to_bytes()
            if current_bytes != self._initial_bytes[dev]:
                name = get_device_name(dev)
                drf = self._cls.base._make_drf(name, self._slot)
                settings.append((drf, current_bytes))

        if settings:
            results = be.write_many(settings)  # type: ignore[arg-type]
            failures = [(drf, r.message) for (drf, _), r in zip(settings, results) if not r.success]
            if failures:
                raise RuntimeError(
                    f"Partial write failure: {len(failures)}/{len(settings)} failed: "
                    + ", ".join(f"{drf}: {msg}" for drf, msg in failures)
                )

        return False


class RecyclerQRampGroup(RampGroup):
    """RampGroup for Recycler quads using RecyclerQRamp transforms."""

    base = RecyclerQRamp


class RecyclerSRampGroup(RampGroup):
    """RampGroup for Recycler sextupoles using RecyclerSRamp transforms."""

    base = RecyclerSRamp


class RecyclerSCRampGroup(RampGroup):
    """RampGroup for Recycler sextupole correctors using RecyclerSCRamp transforms."""

    base = RecyclerSCRamp


class RecyclerHVSQRampGroup(RampGroup):
    """RampGroup for Recycler H/V and skew quad correctors using RecyclerHVSQRamp transforms."""

    base = RecyclerHVSQRamp


class BoosterHVRampGroup(RampGroup):
    """RampGroup for Booster correctors using BoosterHVRamp transforms."""

    base = BoosterHVRamp


class BoosterQRampGroup(RampGroup):
    """RampGroup for Booster quads using BoosterQRamp transforms."""

    base = BoosterQRamp
