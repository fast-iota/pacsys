"""Async Device API - mirrors Device with async I/O methods."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from typing_extensions import Self

from pacsys._device_base import _as_array, _as_scalar, _as_text, _DeviceBase, _require_str_list, _WritePlan
from pacsys.drf3 import parse_request
from pacsys.drf3.property import DRF_PROPERTY
from pacsys.types import BasicControl, ErrorCallback, Reading, ReadingCallback, Value, WriteResult, _validate_callback

if TYPE_CHECKING:
    import numpy as np

    from pacsys.aio._backends import AsyncBackend
    from pacsys.aio._subscription import AsyncSubscriptionHandle
    from pacsys.verify import Verify


class AsyncDevice(_DeviceBase):
    """Async device wrapper. All I/O methods are async."""

    __slots__ = ("_backend",)
    _backend: AsyncBackend | None

    def __init__(self, drf: str, backend: AsyncBackend | None = None):
        super().__init__(parse_request(drf))
        object.__setattr__(self, "_backend", backend)

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
        extra = f"<-{self._request.extra_raw}" if self._request.extra else ""

        readings = await backend.get_many(
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
        p = self._parse_prop(prop)
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
        return await self._execute(self._plan_write(value, field, verify), timeout)

    async def control(
        self,
        command: BasicControl,
        *,
        verify: bool | Verify | None = None,
        timeout: float | None = None,
    ) -> WriteResult:
        """Write CONTROL command."""
        return await self._execute(self._plan_control(command, verify), timeout)

    async def _execute(self, plan: _WritePlan, timeout: float | None) -> WriteResult:
        backend = self._get_backend()
        if plan.check_first:
            assert plan.read_drf is not None
            current = plan.normalize(await backend.read(plan.read_drf, timeout))
            if plan.matches(current):
                return plan.skipped(current)
        result = await backend.write(plan.write_drf, plan.value, timeout=timeout)
        if not result.success or plan.read_drf is None:
            return result
        return await self._verify_readback(result, plan, timeout)

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

    # ─── Streaming Methods ────────────────────────────────────────────────

    async def subscribe(
        self,
        callback: ReadingCallback | None = None,
        on_error: ErrorCallback | None = None,
        *,
        prop: str | None = None,
        field: str | None = None,
        event: str | None = None,
    ) -> AsyncSubscriptionHandle:
        """Subscribe to device for streaming data.

        Args:
            prop: Property to subscribe to (default: READING)
            field: Sub-field (requires prop)
            event: Event string (e.g. "p,1000"). Uses device's event if None.
            callback: Optional callback for push-mode
            on_error: Optional error handler

        Returns:
            AsyncSubscriptionHandle (usable as async context manager)

        Raises:
            ValueError: If no event available, or field given without prop
        """
        _validate_callback(callback, on_error, event_hint=True)
        drf = self._stream_drf(prop, field, event, "subscribe")
        return await self._get_backend().subscribe([drf], callback, on_error)

    async def await_next(
        self,
        *,
        prop: str | None = None,
        field: str | None = None,
        event: str | None = None,
        timeout: float = 5.0,
    ) -> Reading:
        """Wait for the next reading and return it (subscribe, take one, unsubscribe)."""
        drf = self._stream_drf(prop, field, event, "await_next")
        handle = await self._get_backend().subscribe([drf])
        try:
            async for reading, _ in handle.readings(timeout=timeout):
                return reading
        finally:
            await handle.stop()
        raise TimeoutError(f"No reading within {timeout}s for {drf}")

    # ─── Verify Internals ─────────────────────────────────────────────────

    async def _verify_readback(self, result: WriteResult, plan: _WritePlan, timeout: float | None) -> WriteResult:
        from pacsys.errors import DeviceError

        assert plan.verify is not None and plan.read_drf is not None
        v, backend = plan.verify, self._get_backend()
        await asyncio.sleep(v.initial_delay)
        last_readback: Value | None = None
        last_error: DeviceError | None = None
        for attempt in range(1, v.max_attempts + 1):
            try:
                last_readback = plan.normalize(await backend.read(plan.read_drf, timeout))
                last_error = None
            except DeviceError as e:
                last_error = e
            else:
                if plan.matches(last_readback):
                    return plan.verified(result, last_readback, attempt)
            if attempt < v.max_attempts:
                await asyncio.sleep(v.retry_delay)
        return plan.failed(result, last_readback, last_error)

    # ─── Fluent Modifications ─────────────────────────────────────────────

    def with_backend(self, backend: AsyncBackend) -> Self:
        """Return new AsyncDevice bound to a specific backend."""
        return self._new(self.drf, backend)

    def _from_drf(self, drf: str) -> Self:
        return self._new(drf, self._backend)

    def _new(self, drf: str, backend: AsyncBackend | None) -> Self:
        device = self.__class__(drf, backend)
        device._request.field_explicit = self._request.field_explicit
        return device

    # ─── Internal ─────────────────────────────────────────────────────────

    def _get_backend(self) -> AsyncBackend:
        if self._backend is not None:
            return self._backend
        from pacsys.aio import _get_global_async_backend

        return _get_global_async_backend()


class AsyncScalarDevice(AsyncDevice):
    """AsyncDevice that returns scalar values (float)."""

    async def read(self, *, field: str | None = None, timeout: float | None = None) -> float:
        return _as_scalar(await AsyncDevice.read(self, field=field, timeout=timeout))


class AsyncArrayDevice(AsyncDevice):
    """AsyncDevice that returns array values."""

    async def read(self, *, field: str | None = None, timeout: float | None = None) -> np.ndarray:
        return _as_array(await AsyncDevice.read(self, field=field, timeout=timeout))


class AsyncTextDevice(AsyncDevice):
    """AsyncDevice that returns text/string values."""

    async def read(self, *, field: str | None = None, timeout: float | None = None) -> str:
        return _as_text(await AsyncDevice.read(self, field=field, timeout=timeout))
