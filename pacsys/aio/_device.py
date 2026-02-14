"""Async Device API - mirrors Device with async I/O methods."""

from __future__ import annotations

import asyncio
from typing import Optional, TYPE_CHECKING

from pacsys._device_base import _DeviceBase, CONTROL_STATUS_MAP
from pacsys.drf3 import parse_request
from pacsys.drf3.property import DRF_PROPERTY
from pacsys.types import Value, Reading, WriteResult, BasicControl

if TYPE_CHECKING:
    from pacsys.aio._backends import AsyncBackend
    from pacsys.verify import Verify


class AsyncDevice(_DeviceBase):
    """Async device wrapper. All I/O methods are async."""

    __slots__ = ("_backend",)

    def __init__(self, drf: str, backend: Optional[AsyncBackend] = None):
        super().__init__(parse_request(drf))
        self._backend = backend

    # ─── Read Methods ─────────────────────────────────────────────────────

    async def read(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read READING property."""
        drf = self._build_drf(DRF_PROPERTY.READING, self._resolve_field(field, DRF_PROPERTY.READING), "I")
        return await self._get_backend().read(drf, timeout)

    async def setting(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read SETTING property."""
        drf = self._build_drf(DRF_PROPERTY.SETTING, self._resolve_field(field, DRF_PROPERTY.SETTING), "I")
        return await self._get_backend().read(drf, timeout)

    async def status(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read STATUS property."""
        resolved = self._resolve_field(field, DRF_PROPERTY.STATUS)
        drf = self._build_drf(DRF_PROPERTY.STATUS, resolved, "I")
        value = await self._get_backend().read(drf, timeout)
        if resolved is not None and resolved.name in self._BOOL_STATUS_FIELDS:
            return bool(value)
        return value

    async def digital_status(self, timeout: float | None = None):
        """Fetch full digital status (BIT_VALUE + BIT_NAMES + BIT_VALUES)."""
        from pacsys.digital_status import DigitalStatus
        from pacsys.errors import DeviceError

        backend = self._get_backend()
        name = self.name
        extra = f"<-{self._request.extra.name}" if self._request.extra else ""

        readings = await backend.get_many(
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
        if not isinstance(raw_value, (int, float)):
            raise TypeError(f"Expected numeric BIT_VALUE, got {type(raw_value).__name__}")
        if not isinstance(bit_names, list):
            raise TypeError(f"Expected list for BIT_NAMES, got {type(bit_names).__name__}")
        if not isinstance(bit_values, list):
            raise TypeError(f"Expected list for BIT_VALUES, got {type(bit_values).__name__}")
        return DigitalStatus.from_bit_arrays(
            device=name,
            raw_value=int(raw_value),
            bit_names=bit_names,  # type: ignore[arg-type]
            bit_values=bit_values,  # type: ignore[arg-type]
        )

    async def analog_alarm(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read ANALOG alarm property."""
        drf = self._build_drf(DRF_PROPERTY.ANALOG, self._resolve_field(field, DRF_PROPERTY.ANALOG), "I")
        return await self._get_backend().read(drf, timeout)

    async def digital_alarm(self, *, field: str | None = None, timeout: float | None = None) -> Value:
        """Read DIGITAL alarm property."""
        drf = self._build_drf(DRF_PROPERTY.DIGITAL, self._resolve_field(field, DRF_PROPERTY.DIGITAL), "I")
        return await self._get_backend().read(drf, timeout)

    async def description(self, *, field: str | None = None, timeout: float | None = None) -> str:
        """Read DESCRIPTION property."""
        drf = self._build_drf(DRF_PROPERTY.DESCRIPTION, self._resolve_field(field, DRF_PROPERTY.DESCRIPTION), "I")
        value = await self._get_backend().read(drf, timeout)
        if not isinstance(value, str):
            raise TypeError(f"Expected str from DESCRIPTION, got {type(value).__name__}")
        return value

    async def get(
        self,
        *,
        prop: str | None = None,
        field: str | None = None,
        timeout: float | None = None,
    ) -> Reading:
        """Read device with full metadata."""
        if prop is None:
            if field is not None:
                raise ValueError("field requires prop to be specified")
            return await self._get_backend().get(self.drf, timeout)
        p = DRF_PROPERTY[prop.upper()]
        resolved = self._resolve_field(field, p)
        drf = self._build_drf(p, resolved, "I")
        return await self._get_backend().get(drf, timeout)

    # ─── Write Methods ────────────────────────────────────────────────────

    async def write(
        self,
        value: Value,
        *,
        field: str | None = None,
        verify: bool | Verify | None = None,
        timeout: float | None = None,
    ) -> WriteResult:
        """Write to SETTING property."""
        from pacsys.verify import resolve_verify, values_match

        v = resolve_verify(verify)
        resolved_field = self._resolve_field(field, DRF_PROPERTY.SETTING)
        write_drf = self._build_drf(DRF_PROPERTY.SETTING, resolved_field, "N")
        backend = self._get_backend()

        if v is not None and v.check_first:
            read_drf = self._build_drf(DRF_PROPERTY.SETTING, resolved_field, "I")
            if v.readback:
                read_drf = v.readback
            current = await backend.read(read_drf, timeout)
            if values_match(current, value, v.tolerance):
                return WriteResult(drf=write_drf, verified=True, readback=current, skipped=True, attempts=0)

        result = await backend.write(write_drf, value, timeout=timeout)
        if not result.success:
            return result

        if v is not None:
            read_drf = self._build_drf(DRF_PROPERTY.SETTING, resolved_field, "I")
            if v.readback:
                read_drf = v.readback
            return await self._verify_readback(result, read_drf, value, v, timeout)

        return result

    async def control(
        self,
        command: BasicControl,
        *,
        verify: bool | Verify | None = None,
        timeout: float | None = None,
    ) -> WriteResult:
        """Write CONTROL command."""
        from pacsys.verify import resolve_verify, values_match

        v = resolve_verify(verify)
        write_drf = self._build_drf(DRF_PROPERTY.CONTROL, None, "N")
        backend = self._get_backend()

        mapping = CONTROL_STATUS_MAP.get(command)
        status_field_name: str | None = mapping[0] if mapping else None
        expected: bool | None = mapping[1] if mapping else None

        if v is not None and status_field_name is None:
            raise ValueError(f"Cannot verify control command {command!r}: no STATUS field mapping")

        if v is not None and v.check_first and expected is not None:
            status_field = self._resolve_field(status_field_name, DRF_PROPERTY.STATUS)
            read_drf = v.readback or self._build_drf(DRF_PROPERTY.STATUS, status_field, "I")
            current = bool(await backend.read(read_drf, timeout))
            if values_match(current, expected, v.tolerance):
                return WriteResult(drf=write_drf, verified=True, readback=current, skipped=True, attempts=0)

        result = await backend.write(write_drf, command, timeout=timeout)
        if not result.success:
            return result

        if v is not None and expected is not None:
            status_field = self._resolve_field(status_field_name, DRF_PROPERTY.STATUS)
            read_drf = v.readback or self._build_drf(DRF_PROPERTY.STATUS, status_field, "I")
            vr = await self._verify_readback(result, read_drf, expected, v, timeout)
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

    async def on(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.ON, verify=verify, timeout=timeout)

    async def off(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.OFF, verify=verify, timeout=timeout)

    async def reset(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.RESET, verify=verify, timeout=timeout)

    async def positive(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.POSITIVE, verify=verify, timeout=timeout)

    async def negative(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.NEGATIVE, verify=verify, timeout=timeout)

    async def ramp(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.RAMP, verify=verify, timeout=timeout)

    async def dc(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.DC, verify=verify, timeout=timeout)

    async def local(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.LOCAL, verify=verify, timeout=timeout)

    async def remote(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.REMOTE, verify=verify, timeout=timeout)

    async def trip(self, *, verify=None, timeout=None) -> WriteResult:
        return await self.control(BasicControl.TRIP, verify=verify, timeout=timeout)

    # ─── Alarm Setters ────────────────────────────────────────────────────

    async def set_analog_alarm(self, settings: dict, *, timeout=None) -> WriteResult:
        write_drf = self._build_drf(DRF_PROPERTY.ANALOG, None, "N")
        return await self._get_backend().write(write_drf, settings, timeout=timeout)

    async def set_digital_alarm(self, settings: dict, *, timeout=None) -> WriteResult:
        write_drf = self._build_drf(DRF_PROPERTY.DIGITAL, None, "N")
        return await self._get_backend().write(write_drf, settings, timeout=timeout)

    # ─── Verify Internals ─────────────────────────────────────────────────

    async def _verify_readback(
        self,
        write_result: WriteResult,
        read_drf: str,
        expected: Value,
        v: Verify,
        timeout: float | None,
    ) -> WriteResult:
        from pacsys.errors import DeviceError
        from pacsys.verify import values_match

        backend = self._get_backend()
        await asyncio.sleep(v.initial_delay)

        last_readback: Value | None = None
        last_error: DeviceError | None = None
        for attempt in range(1, v.max_attempts + 1):
            try:
                last_readback = await backend.read(read_drf, timeout)
                last_error = None
            except DeviceError as e:
                last_error = e
                if attempt < v.max_attempts:
                    await asyncio.sleep(v.retry_delay)
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
                await asyncio.sleep(v.retry_delay)

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

    def with_backend(self, backend: AsyncBackend) -> AsyncDevice:
        """Return new AsyncDevice bound to a specific backend."""
        return self.__class__(self.drf, backend)

    def _from_drf(self, drf: str) -> AsyncDevice:
        return self.__class__(drf, self._backend)

    # ─── Internal ─────────────────────────────────────────────────────────

    def _get_backend(self) -> AsyncBackend:
        if self._backend is not None:
            return self._backend
        from pacsys.aio import _get_global_async_backend

        return _get_global_async_backend()
