"""
Device API - immutable Device objects with DRF3 validation at construction.

Subclasses: ScalarDevice, ArrayDevice, TextDevice.
Fluent API: with_event(), with_range(), with_backend().
Property-specific reads: read(), setting(), status(), analog_alarm(), digital_alarm(), description().
Write methods: write(), control(), on(), off(), reset(), etc.
"""

from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

import numpy as np

from pacsys.drf3 import DataRequest, parse_request, parse_event
from pacsys.drf3.event import PeriodicEvent
from pacsys.drf3.field import (
    DRF_FIELD,
    parse_field,
    DEFAULT_FIELD_FOR_PROPERTY,
    ALLOWED_FIELD_FOR_PROPERTY,
)
from pacsys.drf3.property import DRF_PROPERTY
from pacsys.drf3.range import ARRAY_RANGE
from pacsys.types import Value, Reading, WriteResult, BasicControl

if TYPE_CHECKING:
    from pacsys.backends import Backend
    from pacsys.digital_status import DigitalStatus
    from pacsys.verify import Verify

# Control command -> (status field, expected value when command succeeds)
_CONTROL_STATUS_MAP: dict[BasicControl, tuple[str, bool]] = {
    BasicControl.ON: ("on", True),
    BasicControl.OFF: ("on", False),
    BasicControl.RESET: ("ready", True),
    BasicControl.TRIP: ("ready", False),
    BasicControl.POSITIVE: ("positive", True),
    BasicControl.NEGATIVE: ("positive", False),
    BasicControl.RAMP: ("ramp", True),
    BasicControl.DC: ("ramp", False),
    BasicControl.REMOTE: ("remote", True),
    BasicControl.LOCAL: ("remote", False),
}


class Device:
    """Device wrapper with DRF3 validation at construction.

    Devices are immutable - modification methods return NEW Device instances.
    """

    __slots__ = ("_request", "_backend", "_drf_string")

    def __init__(self, drf: str, backend: Optional[Backend] = None):
        """Create a device from DRF string.

        Args:
            drf: Device request string (e.g., "M:OUTTMP", "B:HS23T[0:10]@p,1000")
            backend: Optional backend instance. If None, uses global default.

        Raises:
            ValueError: If DRF syntax is invalid (at construction, not read time)
        """
        self._request = parse_request(drf)  # Validates DRF syntax
        self._backend = backend
        self._drf_string = drf

    # ─── Properties ───────────────────────────────────────────────────────

    @property
    def drf(self) -> str:
        """Canonical DRF string (normalized form)."""
        return self._request.to_canonical()

    @property
    def name(self) -> str:
        """Device name without property/range/event modifiers."""
        return self._request.device

    @property
    def request(self) -> DataRequest:
        """Parsed DRF3 request with all components."""
        return self._request

    @property
    def has_event(self) -> bool:
        """True if DRF specifies an explicit event (not default)."""
        return self._request.event is not None and self._request.event.mode != "U"

    @property
    def is_periodic(self) -> bool:
        """True if DRF specifies periodic acquisition."""
        return isinstance(self._request.event, PeriodicEvent)

    # ─── DRF Building Helpers ─────────────────────────────────────────────

    def _build_drf(
        self,
        prop: DRF_PROPERTY,
        field: DRF_FIELD | None,
        event: str,
    ) -> str:
        """Build DRF from device name + explicit property/field/event.

        Avoids to_canonical() field-fallthrough issues when switching properties.
        """
        out = self.name
        out += f".{prop.name}"
        if self._request.range is not None:
            out += str(self._request.range)
        if field is not None:
            default = DEFAULT_FIELD_FOR_PROPERTY.get(prop)
            if field != default:
                out += f".{field.name}"
        out += f"@{event}"
        if self._request.extra is not None:
            out += f"<-{self._request.extra.name}"
        return out

    def _resolve_field(self, field: str | None, prop: DRF_PROPERTY) -> DRF_FIELD | None:
        """Resolve a user-supplied field string to a DRF_FIELD.

        Args:
            field: Field name (e.g., "raw", "on") or None for default.
            prop: The property this field applies to.

        Returns:
            DRF_FIELD instance, or None if property has no default field and none given.

        Raises:
            ValueError: If field is not allowed for the property.
        """
        if field is None:
            return DEFAULT_FIELD_FOR_PROPERTY.get(prop)
        f = parse_field(field.upper())
        allowed = ALLOWED_FIELD_FOR_PROPERTY.get(prop, [])
        if f not in allowed:
            raise ValueError(f"Field '{field}' not allowed for {prop.name}")
        return f

    # ─── Read Methods ─────────────────────────────────────────────────────

    def read(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read READING property. Raises DeviceError on failure.

        Args:
            field: Optional field override ('raw', 'primary', 'scaled').
            timeout: Operation timeout in seconds.
        """
        drf = self._build_drf(
            DRF_PROPERTY.READING,
            self._resolve_field(field, DRF_PROPERTY.READING),
            "I",
        )
        return self._get_backend().read(drf, timeout)

    def setting(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read SETTING property."""
        drf = self._build_drf(
            DRF_PROPERTY.SETTING,
            self._resolve_field(field, DRF_PROPERTY.SETTING),
            "I",
        )
        return self._get_backend().read(drf, timeout)

    _BOOL_STATUS_FIELDS = frozenset({"ON", "READY", "REMOTE", "POSITIVE", "RAMP"})

    def status(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read STATUS property.

        Args:
            field: Optional field ('on', 'ready', 'remote', 'positive', 'ramp',
                   'all', 'text', 'raw', etc.).

        Returns:
            For boolean fields (on/ready/remote/positive/ramp): bool.
            For other fields or no field: raw backend value (dict, bytes, etc.).
        """
        resolved = self._resolve_field(field, DRF_PROPERTY.STATUS)
        drf = self._build_drf(DRF_PROPERTY.STATUS, resolved, "I")
        value = self._get_backend().read(drf, timeout)
        if resolved is not None and resolved.name in self._BOOL_STATUS_FIELDS:
            return bool(value)
        return value

    def analog_alarm(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read ANALOG alarm property."""
        drf = self._build_drf(
            DRF_PROPERTY.ANALOG,
            self._resolve_field(field, DRF_PROPERTY.ANALOG),
            "I",
        )
        return self._get_backend().read(drf, timeout)

    def digital_alarm(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read DIGITAL alarm property."""
        drf = self._build_drf(
            DRF_PROPERTY.DIGITAL,
            self._resolve_field(field, DRF_PROPERTY.DIGITAL),
            "I",
        )
        return self._get_backend().read(drf, timeout)

    def description(self, *, field: str | None = None, timeout: float | None = None) -> str:
        """Read DESCRIPTION property."""
        drf = self._build_drf(
            DRF_PROPERTY.DESCRIPTION,
            self._resolve_field(field, DRF_PROPERTY.DESCRIPTION),
            "I",
        )
        value = self._get_backend().read(drf, timeout)
        assert isinstance(value, str), f"Expected str from DESCRIPTION, got {type(value).__name__}"
        return value

    def get(
        self,
        *,
        prop: str | None = None,
        field: str | None = None,
        timeout: float | None = None,
    ) -> Reading:
        """Read device with full metadata (timestamp, cycle, meta).

        Args:
            prop: Property name ('reading', 'setting', 'status', 'analog',
                  'digital', 'description'). If None, uses device's base DRF.
            field: Optional field override (e.g., 'raw', 'on', 'scaled').
                   Only valid when prop is specified.
            timeout: Operation timeout in seconds.
        """
        if prop is None:
            if field is not None:
                raise ValueError("field requires prop to be specified")
            return self._get_backend().get(self.drf, timeout)
        p = DRF_PROPERTY[prop.upper()]
        resolved = self._resolve_field(field, p)
        drf = self._build_drf(p, resolved, "I")
        return self._get_backend().get(drf, timeout)

    def info(self, timeout: float | None = None):
        """Fetch device metadata from DevDB (cached).

        Provides scaling, limits, control commands, status bit definitions.

        Args:
            timeout: gRPC timeout in seconds (default: client's configured timeout).

        Raises:
            RuntimeError: If DevDB is not configured or unreachable.
            DeviceError: If the device is not found in DevDB.
        """
        from pacsys.devdb import DeviceInfoResult  # noqa: F811

        devdb = self._get_devdb()
        if devdb is None:
            raise RuntimeError(
                "DevDB not available. Configure with pacsys.configure(devdb_host=...) "
                "or set PACSYS_DEVDB_HOST environment variable."
            )
        results: dict[str, DeviceInfoResult] = devdb.get_device_info([self.name], timeout=timeout)
        return results[self.name]

    def digital_status(self, timeout: float | None = None) -> DigitalStatus:
        """Fetch full digital status (BIT_VALUE + BIT_NAMES + BIT_VALUES).

        Returns a DigitalStatus with per-bit labels and values, richer than
        the 5-bool legacy dict from get() on a STATUS property.

        When DevDB is available and has cached status_bits for this device,
        only 1 read (BIT_VALUE) is needed instead of 3. Falls back to the
        3-read path if DevDB is unavailable.

        Raises:
            DeviceError: If the device has no status property.
        """
        from pacsys.digital_status import DigitalStatus
        from pacsys.errors import DeviceError

        backend = self._get_backend()
        name = self.name
        extra = f"<-{self._request.extra.name}" if self._request.extra else ""

        # Try DevDB-accelerated path (1 read instead of 3)
        devdb = self._get_devdb()
        if devdb is not None:
            try:
                info = devdb.get_device_info([name])[name]
            except Exception:
                info = None  # DevDB unavailable, fall through to 3-read path
            if info is not None and info.status_bits is not None:
                reading = backend.get(f"{name}.STATUS.BIT_VALUE@I{extra}", timeout)
                if reading.is_error:
                    raise DeviceError(reading.drf, reading.facility_code, reading.error_code, reading.message)
                raw_value = reading.value
                if not isinstance(raw_value, (int, float)):
                    raise TypeError(f"Expected numeric BIT_VALUE, got {type(raw_value).__name__}")
                return DigitalStatus.from_devdb_bits(
                    device=name,
                    raw_value=int(raw_value),
                    bit_defs=info.status_bits,
                    ext_bit_defs=info.ext_status_bits,
                )

        # Standard 3-read path
        readings = backend.get_many(
            [
                f"{name}.STATUS.BIT_VALUE@I{extra}",
                f"{name}.STATUS.BIT_NAMES@I{extra}",
                f"{name}.STATUS.BIT_VALUES@I{extra}",
            ],
            timeout=timeout,
        )
        for r in readings:
            if r.is_error:
                raise DeviceError(r.drf, r.facility_code, r.error_code, r.message)

        raw_value = readings[0].value
        bit_names = readings[1].value
        bit_values = readings[2].value
        assert isinstance(raw_value, (int, float)), f"Expected numeric BIT_VALUE, got {type(raw_value).__name__}"
        assert isinstance(bit_names, list), f"Expected list for BIT_NAMES, got {type(bit_names).__name__}"
        assert isinstance(bit_values, list), f"Expected list for BIT_VALUES, got {type(bit_values).__name__}"
        return DigitalStatus.from_bit_arrays(
            device=name,
            raw_value=int(raw_value),
            bit_names=bit_names,  # type: ignore[arg-type]  # list narrowing limitation
            bit_values=bit_values,  # type: ignore[arg-type]  # list narrowing limitation
        )

    # ─── Write Methods ────────────────────────────────────────────────────

    def write(
        self,
        value: Value,
        *,
        field: str | None = None,
        verify: bool | Verify | None = None,
        timeout: float | None = None,
    ) -> WriteResult:
        """Write to SETTING property.

        Args:
            value: Value to write.
            field: Optional field override ('raw', 'primary', 'scaled').
            verify: Verification config (True/False/Verify/None).
            timeout: Operation timeout in seconds.
        """
        from pacsys.verify import resolve_verify, values_match

        v = resolve_verify(verify)
        resolved_field = self._resolve_field(field, DRF_PROPERTY.SETTING)
        write_drf = self._build_drf(DRF_PROPERTY.SETTING, resolved_field, "N")

        backend = self._get_backend()

        # check_first: skip write if value already matches
        if v is not None and v.check_first:
            read_drf = self._build_drf(DRF_PROPERTY.SETTING, resolved_field, "I")
            if v.readback:
                read_drf = v.readback
            current = backend.read(read_drf, timeout)
            if values_match(current, value, v.tolerance):
                return WriteResult(
                    drf=write_drf,
                    verified=True,
                    readback=current,
                    skipped=True,
                    attempts=0,
                )

        result = backend.write(write_drf, value, timeout=timeout)
        if not result.success:
            return result

        # Verification readback loop
        if v is not None:
            read_drf = self._build_drf(DRF_PROPERTY.SETTING, resolved_field, "I")
            if v.readback:
                read_drf = v.readback
            return self._verify_readback(result, read_drf, value, v, timeout)

        return result

    def control(
        self,
        command: BasicControl,
        *,
        verify: bool | Verify | None = None,
        timeout: float | None = None,
    ) -> WriteResult:
        """Write CONTROL command.

        Args:
            command: BasicControl enum value (ON, OFF, RESET, etc.).
            verify: Verification reads STATUS to confirm command effect.
            timeout: Operation timeout in seconds.
        """
        from pacsys.verify import resolve_verify, values_match

        v = resolve_verify(verify)
        write_drf = self._build_drf(DRF_PROPERTY.CONTROL, None, "N")
        backend = self._get_backend()

        mapping = _CONTROL_STATUS_MAP.get(command)
        status_field_name: str | None = mapping[0] if mapping else None
        expected: bool | None = mapping[1] if mapping else None

        if v is not None and status_field_name is None:
            raise ValueError(f"Cannot verify control command {command!r}: no STATUS field mapping defined")

        # check_first: skip write if status already matches
        if v is not None and v.check_first and expected is not None:
            status_field = self._resolve_field(status_field_name, DRF_PROPERTY.STATUS)
            read_drf = v.readback or self._build_drf(DRF_PROPERTY.STATUS, status_field, "I")
            current = bool(backend.read(read_drf, timeout))
            if values_match(current, expected, v.tolerance):
                return WriteResult(
                    drf=write_drf,
                    verified=True,
                    readback=current,
                    skipped=True,
                    attempts=0,
                )

        result = backend.write(write_drf, command, timeout=timeout)
        if not result.success:
            return result

        # Verification: read STATUS field to confirm
        if v is not None and expected is not None:
            status_field = self._resolve_field(status_field_name, DRF_PROPERTY.STATUS)
            read_drf = v.readback or self._build_drf(DRF_PROPERTY.STATUS, status_field, "I")
            vr = self._verify_readback(result, read_drf, expected, v, timeout)
            if vr.readback is not None:
                vr = WriteResult(
                    drf=vr.drf,
                    facility_code=vr.facility_code,
                    error_code=vr.error_code,
                    message=vr.message,
                    verified=vr.verified,
                    readback=bool(vr.readback),
                    skipped=vr.skipped,
                    attempts=vr.attempts,
                )
            return vr

        return result

    # ─── Control Shortcuts ────────────────────────────────────────────────

    def on(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.ON, verify=verify, timeout=timeout)

    def off(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.OFF, verify=verify, timeout=timeout)

    def reset(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.RESET, verify=verify, timeout=timeout)

    def positive(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.POSITIVE, verify=verify, timeout=timeout)

    def negative(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.NEGATIVE, verify=verify, timeout=timeout)

    def ramp(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.RAMP, verify=verify, timeout=timeout)

    def dc(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.DC, verify=verify, timeout=timeout)

    def local(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.LOCAL, verify=verify, timeout=timeout)

    def remote(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.REMOTE, verify=verify, timeout=timeout)

    def trip(self, *, verify: bool | Verify | None = None, timeout: float | None = None) -> WriteResult:
        return self.control(BasicControl.TRIP, verify=verify, timeout=timeout)

    # ─── Alarm Setters ────────────────────────────────────────────────────

    def set_analog_alarm(self, settings: dict, *, timeout: float | None = None) -> WriteResult:
        """Write ANALOG alarm property."""
        write_drf = self._build_drf(DRF_PROPERTY.ANALOG, None, "N")
        return self._get_backend().write(write_drf, settings, timeout=timeout)

    def set_digital_alarm(self, settings: dict, *, timeout: float | None = None) -> WriteResult:
        """Write DIGITAL alarm property."""
        write_drf = self._build_drf(DRF_PROPERTY.DIGITAL, None, "N")
        return self._get_backend().write(write_drf, settings, timeout=timeout)

    # ─── Verify Internals ─────────────────────────────────────────────────

    def _verify_readback(
        self,
        write_result: WriteResult,
        read_drf: str,
        expected: Value,
        v: Verify,
        timeout: float | None,
    ) -> WriteResult:
        """Run the readback verification loop after a write."""
        from pacsys.errors import DeviceError
        from pacsys.verify import values_match

        backend = self._get_backend()
        time.sleep(v.initial_delay)

        last_readback: Value | None = None
        last_error: DeviceError | None = None
        for attempt in range(1, v.max_attempts + 1):
            try:
                last_readback = backend.read(read_drf, timeout)
                last_error = None
            except DeviceError as e:
                last_error = e
                if attempt < v.max_attempts:
                    time.sleep(v.retry_delay)
                continue
            if values_match(last_readback, expected, v.tolerance):
                return WriteResult(
                    drf=write_result.drf,
                    facility_code=write_result.facility_code,
                    error_code=write_result.error_code,
                    message=write_result.message,
                    verified=True,
                    readback=last_readback,
                    attempts=attempt,
                )
            if attempt < v.max_attempts:
                time.sleep(v.retry_delay)

        msg = write_result.message
        if last_error is not None:
            msg = f"Readback failed: {last_error}"
        return WriteResult(
            drf=write_result.drf,
            facility_code=write_result.facility_code,
            error_code=write_result.error_code,
            message=msg,
            verified=False,
            readback=last_readback,
            attempts=v.max_attempts,
        )

    # ─── Fluent Modifications ─────────────────────────────────────────────

    def with_backend(self, backend: Backend) -> Device:
        """Return new Device bound to a specific backend."""
        return self.__class__(self.drf, backend)

    def with_event(self, event: str) -> Device:
        """Return new Device with different event."""
        new_event = parse_event(event)
        new_drf = self._request.to_canonical(event=new_event)
        return self.__class__(new_drf, self._backend)

    def with_range(self, start: int, end: int | None = None) -> Device:
        """Return new Device with array range."""
        new_range = ARRAY_RANGE(mode="std", low=start, high=end)
        new_drf = self._request.to_canonical(range=new_range)
        return self.__class__(new_drf, self._backend)

    # ─── Internal ─────────────────────────────────────────────────────────

    def _get_backend(self) -> Backend:
        """Get backend, using global default if none specified."""
        if self._backend is not None:
            return self._backend
        from pacsys import _get_global_backend

        return _get_global_backend()

    def _get_devdb(self):
        """Get global DevDB client, or None if not configured."""
        from pacsys import _get_global_devdb

        return _get_global_devdb()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.drf!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Device):
            return NotImplemented
        return self.drf == other.drf

    def __hash__(self) -> int:
        return hash(self.drf)


class ScalarDevice(Device):
    """Device that returns scalar values (float)."""

    def read(self, *, field: str | None = None, timeout: float | None = None) -> float:
        """Read scalar value. Raises TypeError if not scalar."""
        value = Device.read(self, field=field, timeout=timeout)
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected scalar, got {type(value).__name__}")
        return float(value)


class ArrayDevice(Device):
    """Device that returns array values."""

    def read(self, *, field: str | None = None, timeout: float | None = None) -> np.ndarray:
        """Read array value. Raises TypeError if not array."""
        value = Device.read(self, field=field, timeout=timeout)
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.array(value)
        raise TypeError(f"Expected array, got {type(value).__name__}")


class TextDevice(Device):
    """Device that returns text/string values."""

    def read(self, *, field: str | None = None, timeout: float | None = None) -> str:
        """Read text value. Raises TypeError if not string."""
        value = Device.read(self, field=field, timeout=timeout)
        if not isinstance(value, str):
            raise TypeError(f"Expected string, got {type(value).__name__}")
        return value


__all__ = ["Device", "ScalarDevice", "ArrayDevice", "TextDevice"]
