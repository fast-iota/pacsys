"""
Device API - immutable Device objects with DRF3 validation at construction.

Subclasses: ScalarDevice, ArrayDevice, TextDevice.
Fluent API: with_event(), with_range(), with_backend().
Property-specific reads: read(), setting(), status(), analog_alarm(), digital_alarm(), description().
Write methods: write(), control(), on(), off(), reset(), etc.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from typing_extensions import Self

from pacsys._device_base import _as_array, _as_scalar, _as_text, _DeviceBase, _require_str_list, _WritePlan
from pacsys.drf3 import parse_request
from pacsys.drf3.property import DRF_PROPERTY
from pacsys.types import (
    BasicControl,
    ErrorCallback,
    Reading,
    ReadingCallback,
    SubscriptionHandle,
    Value,
    WriteResult,
    _validate_callback,
)

if TYPE_CHECKING:
    import numpy as np

    from pacsys.backends import Backend
    from pacsys.devdb import DeviceInfoResult
    from pacsys.digital_status import DigitalStatus
    from pacsys.verify import Verify


class Device(_DeviceBase):
    """Device wrapper with DRF3 validation at construction.

    Devices are immutable - modification methods return NEW Device instances.
    """

    __slots__ = ("_backend",)
    _backend: Backend | None

    def __init__(self, drf: str, backend: Backend | None = None):
        """Create a device from DRF string.

        Args:
            drf: Device request string (e.g., "M:OUTTMP", "B:HS23T[0:10]@p,1000")
            backend: Optional backend instance. If None, uses global default.

        Raises:
            ValueError: If DRF syntax is invalid (at construction, not read time)
        """
        super().__init__(parse_request(drf))
        object.__setattr__(self, "_backend", backend)

    # ─── Read Methods ─────────────────────────────────────────────────────

    def read(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read READING property. Raises DeviceError on failure."""
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

    def status(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read STATUS property."""
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
        if not isinstance(value, str):
            raise TypeError(f"Expected str from DESCRIPTION, got {type(value).__name__}")
        return value

    def get(
        self,
        *,
        prop: str | None = None,
        field: str | None = None,
        timeout: float | None = None,
    ) -> Reading:
        """Read device with full metadata (timestamp, cycle, meta)."""
        if prop is None:
            if field is not None:
                raise ValueError("field requires prop to be specified")
            return self._get_backend().get(self.drf, timeout)
        p = self._parse_prop(prop)
        resolved = self._resolve_field(field, p)
        drf = self._build_drf(p, resolved, "I")
        return self._get_backend().get(drf, timeout)

    def info(self, timeout: float | None = None):
        """Fetch device metadata from DevDB (cached)."""

        devdb = self._get_devdb()
        if devdb is None:
            raise RuntimeError(
                "DevDB not available. Configure with pacsys.configure(devdb_host=...) "
                "or set PACSYS_DEVDB_HOST environment variable."
            )
        results: dict[str, DeviceInfoResult] = devdb.get_device_info([self.name], timeout=timeout)
        if self.name not in results:
            from pacsys.acnet.errors import ERR_NOPROP, FACILITY_DBM
            from pacsys.errors import DeviceError

            raise DeviceError(self.drf, FACILITY_DBM, ERR_NOPROP, f"Device '{self.name}' not found in DevDB")
        return results[self.name]

    def digital_status(self, timeout: float | None = None) -> DigitalStatus:
        """Fetch full digital status (BIT_VALUE + BIT_NAMES + BIT_VALUES)."""
        from pacsys.digital_status import DigitalStatus
        from pacsys.errors import DeviceError

        backend = self._get_backend()
        name = self.name
        extra = f"<-{self._request.extra_raw}" if self._request.extra else ""

        # Try DevDB-accelerated path (1 read instead of 3)
        devdb = self._get_devdb()
        if devdb is not None:
            try:
                info = devdb.get_device_info([name], timeout=timeout)[name]
            except Exception:  # noqa: BLE001
                info = None
            if info is not None and info.status_bits is not None:
                reading = backend.get(f"{name}.STATUS.BIT_VALUE@I{extra}", timeout)
                if not reading.ok:
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
            if not r.ok:
                raise DeviceError(r.drf, r.facility_code, r.error_code, r.message)

        raw_value = readings[0].value
        bit_names = readings[1].value
        bit_values = readings[2].value
        if not isinstance(raw_value, (int, float)):
            raise TypeError(f"Expected numeric BIT_VALUE, got {type(raw_value).__name__}")
        return DigitalStatus.from_bit_arrays(
            device=name,
            raw_value=int(raw_value),
            bit_names=_require_str_list(bit_names, "BIT_NAMES"),
            bit_values=_require_str_list(bit_values, "BIT_VALUES"),
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
        """Write to SETTING property."""
        return self._execute(self._plan_write(value, field, verify), timeout)

    def control(
        self,
        command: BasicControl,
        *,
        verify: bool | Verify | None = None,
        timeout: float | None = None,
    ) -> WriteResult:
        """Write CONTROL command."""
        return self._execute(self._plan_control(command, verify), timeout)

    def _execute(self, plan: _WritePlan, timeout: float | None) -> WriteResult:
        backend = self._get_backend()
        if plan.check_first:
            assert plan.read_drf is not None
            current = plan.normalize(backend.read(plan.read_drf, timeout))
            if plan.matches(current):
                return plan.skipped(current)
        result = backend.write(plan.write_drf, plan.value, timeout=timeout)
        if not result.success or plan.read_drf is None:
            return result
        return self._verify_readback(result, plan, timeout)

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

    # ─── Streaming Methods ────────────────────────────────────────────────

    def subscribe(
        self,
        callback: ReadingCallback | None = None,
        on_error: ErrorCallback | None = None,
        *,
        prop: str | None = None,
        field: str | None = None,
        event: str | None = None,
    ) -> SubscriptionHandle:
        """Subscribe to device for streaming data.

        Args:
            prop: Property to subscribe to (default: READING)
            field: Sub-field (requires prop)
            event: Event string (e.g. "p,1000"). Uses device's event if None.
            callback: Optional callback for push-mode
            on_error: Optional error handler

        Returns:
            SubscriptionHandle (usable as context manager)

        Raises:
            ValueError: If no event available, or field given without prop
        """
        _validate_callback(callback, on_error, event_hint=True)
        drf = self._stream_drf(prop, field, event, "subscribe")
        return self._get_backend().subscribe([drf], callback, on_error)

    def await_next(
        self,
        *,
        prop: str | None = None,
        field: str | None = None,
        event: str | None = None,
        timeout: float = 5.0,
    ) -> Reading:
        """Block until the next reading arrives and return it.

        Subscribes, waits for one reading, unsubscribes. Requires an event
        (either on the device or via event= kwarg).
        """
        from pacsys.exp._watch import watch

        drf = self._stream_drf(prop, field, event, "await_next")
        return watch(drf, lambda r: True, timeout=timeout, backend=self._get_backend())

    # ─── Verify Internals ─────────────────────────────────────────────────

    def _verify_readback(self, result: WriteResult, plan: _WritePlan, timeout: float | None) -> WriteResult:
        """Run the readback verification loop after a write."""
        from pacsys.errors import DeviceError

        assert plan.verify is not None and plan.read_drf is not None
        v, backend = plan.verify, self._get_backend()
        time.sleep(v.initial_delay)
        last_readback: Value | None = None
        last_error: DeviceError | None = None
        for attempt in range(1, v.max_attempts + 1):
            try:
                last_readback = plan.normalize(backend.read(plan.read_drf, timeout))
                last_error = None
            except DeviceError as e:
                last_error = e
            else:
                if plan.matches(last_readback):
                    return plan.verified(result, last_readback, attempt)
            if attempt < v.max_attempts:
                time.sleep(v.retry_delay)
        return plan.failed(result, last_readback, last_error)

    # ─── Fluent Modifications ─────────────────────────────────────────────

    def with_backend(self, backend: Backend) -> Self:
        """Return new Device bound to a specific backend."""
        return self._new(self.drf, backend)

    def _from_drf(self, drf: str) -> Self:
        return self._new(drf, self._backend)

    def _new(self, drf: str, backend: Backend | None) -> Self:
        device = self.__class__(drf, backend)
        device._request.field_explicit = self._request.field_explicit
        return device

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


class ScalarDevice(Device):
    """Device that returns scalar values (float)."""

    def read(self, *, field: str | None = None, timeout: float | None = None) -> float:
        """Read scalar value. Raises TypeError if not scalar."""
        return _as_scalar(Device.read(self, field=field, timeout=timeout))


class ArrayDevice(Device):
    """Device that returns array values."""

    def read(self, *, field: str | None = None, timeout: float | None = None) -> np.ndarray:
        """Read array value. Raises TypeError if not array."""
        return _as_array(Device.read(self, field=field, timeout=timeout))


class TextDevice(Device):
    """Device that returns text/string values."""

    def read(self, *, field: str | None = None, timeout: float | None = None) -> str:
        """Read text value. Raises TypeError if not string."""
        return _as_text(Device.read(self, field=field, timeout=timeout))


__all__ = ["ArrayDevice", "Device", "ScalarDevice", "TextDevice"]
