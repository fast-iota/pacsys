"""Shared base for Device and AsyncDevice - pure DRF logic, no I/O."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from typing_extensions import Self

from pacsys.drf3 import DataRequest, parse_event, parse_extra
from pacsys.drf3.event import DefaultEvent, NeverEvent, PeriodicEvent
from pacsys.drf3.field import (
    ALLOWED_FIELD_FOR_PROPERTY,
    DEFAULT_FIELD_FOR_PROPERTY,
    DRF_FIELD,
    parse_field,
)
from pacsys.drf3.property import DRF_PROPERTY, parse_property
from pacsys.drf3.range import ARRAY_RANGE
from pacsys.types import BasicControl, Value, WriteResult

if TYPE_CHECKING:
    import numpy as np

    from pacsys.verify import Verify

CONTROL_STATUS_MAP: dict[BasicControl, tuple[str, bool]] = {
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


def _require_str_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"Expected list for {field}, got {type(value).__name__}")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"Expected list[str] for {field}")
    return cast("list[str]", value)


def _as_scalar(value: Value) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected scalar, got {type(value).__name__}")
    return float(value)


def _as_array(value: Value) -> np.ndarray:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.array(value)
    raise TypeError(f"Expected array, got {type(value).__name__}")


def _as_text(value: Value) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected string, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class _WritePlan:
    """Backend-free plan for one write: what to send and, if verifying, how to read it back.

    Device and AsyncDevice both drive this; only the I/O calls and the sleep flavor differ.
    """

    write_drf: str
    value: Value  # payload sent to the backend
    expected: Value  # what the readback must match
    verify: Verify | None
    read_drf: str | None  # None when no readback happens
    readback_field: str | None = None  # STATUS field name for control readback normalization

    @property
    def check_first(self) -> bool:
        return self.verify is not None and self.verify.check_first

    def normalize(self, readback: Value) -> Value:
        if self.readback_field is None:
            return readback
        from pacsys.verify import _normalize_control_readback

        return _normalize_control_readback(readback, self.readback_field)

    def matches(self, readback: Value) -> bool:
        from pacsys.verify import values_match

        assert self.verify is not None
        return values_match(readback, self.expected, self.verify.tolerance)

    def skipped(self, current: Value) -> WriteResult:
        return WriteResult(drf=self.write_drf, verified=True, readback=current, skipped=True, attempts=0)

    @staticmethod
    def verified(result: WriteResult, readback: Value, attempt: int) -> WriteResult:
        return WriteResult(
            drf=result.drf,
            facility_code=result.facility_code,
            error_code=result.error_code,
            message=result.message,
            verified=True,
            readback=readback,
            attempts=attempt,
        )

    def failed(self, result: WriteResult, readback: Value | None, last_error: Exception | None) -> WriteResult:
        assert self.verify is not None
        return WriteResult(
            drf=result.drf,
            facility_code=result.facility_code,
            error_code=result.error_code,
            message=f"Readback failed: {last_error}" if last_error is not None else result.message,
            verified=False,
            readback=readback,
            attempts=self.verify.max_attempts,
        )


class _DeviceBase:
    """DRF building, field resolution, fluent modification. No I/O."""

    __slots__ = ("_request",)
    _request: DataRequest

    _BOOL_STATUS_FIELDS = frozenset({"ON", "READY", "REMOTE", "POSITIVE", "RAMP"})

    def __init__(self, request: DataRequest):
        object.__setattr__(self, "_request", request)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{self.__class__.__name__} instances are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{self.__class__.__name__} instances are immutable")

    def __setstate__(
        self,
        state: dict[str, object] | tuple[dict[str, object] | None, dict[str, object]],
    ) -> None:
        states = state if isinstance(state, tuple) else (state,)
        for values in states:
            for name, value in (values or {}).items():
                object.__setattr__(self, name, value)

    @property
    def drf(self) -> str:
        return self._request.to_canonical()

    @property
    def name(self) -> str:
        return self._request.device

    @property
    def request(self) -> DataRequest:
        """Snapshot of the parsed DRF; DataRequest is mutable, so never hand out the live one."""
        return copy.deepcopy(self._request)

    @property
    def has_event(self) -> bool:
        return self._request.event is not None and self._request.event.mode != "U"

    @property
    def is_periodic(self) -> bool:
        return isinstance(self._request.event, PeriodicEvent)

    def _build_drf(self, prop: DRF_PROPERTY, field: DRF_FIELD | None, event: str) -> str:
        if not self._request.is_acnet:
            # EPICS: read/write/subscribe target the PV itself; ACNET-only properties
            # and fields have no analogue and must fail loudly, never be synthesized.
            if prop not in (DRF_PROPERTY.READING, DRF_PROPERTY.SETTING):
                raise ValueError(f"{prop.name} is ACNET-specific, not supported for non-ACNET device {self.name}")
            if field is not None and field != DEFAULT_FIELD_FOR_PROPERTY.get(prop):
                raise ValueError(f"ACNET field {field.name} not supported for non-ACNET device {self.name}")
            return self._request.to_canonical(event=parse_event(event))
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
            out += f"<-{self._request.extra_raw}"
        return out

    def _parse_prop(self, prop: str) -> DRF_PROPERTY:
        """Resolve a property argument using the DRF parser's aliases."""
        try:
            return parse_property(prop)
        except (AttributeError, ValueError):
            raise ValueError(f"Invalid property {prop!r} for {self.name}") from None

    def _resolve_field(self, field: str | None, prop: DRF_PROPERTY) -> DRF_FIELD | None:
        if field is None:
            # A field typed in the constructor DRF (e.g. Device("X.SETTING.RAW")) carries
            # over when valid for the target property; cross-property mismatches drop to
            # the target's default (same policy as prepare_for_write/to_canonical).
            if self._request.field_explicit and self._request.field in ALLOWED_FIELD_FOR_PROPERTY.get(prop, []):
                return self._request.field
            return DEFAULT_FIELD_FOR_PROPERTY.get(prop)
        f = parse_field(field.upper())
        allowed = ALLOWED_FIELD_FOR_PROPERTY.get(prop, [])
        if f not in allowed:
            raise ValueError(f"Field '{field}' not allowed for {prop.name}")
        return f

    def _plan_write(self, value: Value, field: str | None, verify: bool | Verify | None) -> _WritePlan:
        from pacsys.verify import resolve_verify

        if isinstance(value, BasicControl):
            raise TypeError(
                f"BasicControl.{value.name} targets the CONTROL property - use control() instead of write()"
            )
        v = resolve_verify(verify)
        f = self._resolve_field(field, DRF_PROPERTY.SETTING)
        read_drf = (v.readback or self._build_drf(DRF_PROPERTY.SETTING, f, "I")) if v is not None else None
        return _WritePlan(self._build_drf(DRF_PROPERTY.SETTING, f, "N"), value, value, v, read_drf)

    def _plan_control(self, command: BasicControl, verify: bool | Verify | None) -> _WritePlan:
        from pacsys.verify import resolve_verify

        v = resolve_verify(verify)
        write_drf = self._build_drf(DRF_PROPERTY.CONTROL, None, "N")
        if v is None:
            return _WritePlan(write_drf, command, command, None, None)
        mapping = CONTROL_STATUS_MAP.get(command)
        if mapping is None:
            raise ValueError(f"Cannot verify control command {command!r}: no STATUS field mapping defined")
        field_name, expected = mapping
        status_field = self._resolve_field(field_name, DRF_PROPERTY.STATUS)
        read_drf = v.readback or self._build_drf(DRF_PROPERTY.STATUS, status_field, "I")
        return _WritePlan(write_drf, command, expected, v, read_drf, readback_field=field_name)

    def _stream_drf(self, prop: str | None, field: str | None, event: str | None, what: str) -> str:
        """DRF for subscribe()/await_next(): device event unless overridden; @N is rejected."""
        if prop is None:
            if field is not None:
                raise ValueError("field requires prop to be specified")
            p = DRF_PROPERTY.READING
        else:
            p = self._parse_prop(prop)
        resolved = self._resolve_field(field, p)
        if event is not None:
            ev = parse_event(event)
        elif self.has_event:
            ev = self._request.event
        else:
            raise ValueError(f"{what} requires an event — use event= or dev.with_event()")
        if isinstance(ev, NeverEvent):
            raise ValueError(f"{what} cannot use @N (never) event")  # noqa: TRY004 - documented ValueError
        return self._build_drf(p, resolved, ev.raw_string)

    def with_event(self, event: str) -> Self:
        new_event = parse_event(event)
        new_drf = self._request.to_canonical(event=new_event)
        return self._from_drf(new_drf)

    def with_range(self, start: int | None = None, end: int | None = None, *, at: int | None = None) -> Self:
        if at is not None:
            if start is not None or end is not None:
                raise ValueError("'at' cannot be combined with 'start'/'end'")
            new_range = ARRAY_RANGE(mode="single", low=at)
        elif start is not None or end is not None:
            new_range = ARRAY_RANGE(mode="std", low=start, high=end)
        else:
            new_range = ARRAY_RANGE(mode="full")
        new_drf = self._request.to_canonical(range=new_range)
        return self._from_drf(new_drf)

    def without_range(self) -> Self:
        """Return new device with array range removed."""
        new_drf = self._request.to_canonical(range=None)
        return self._from_drf(new_drf)

    def without_event(self) -> Self:
        """Return new device with event removed (back to default/unspecified)."""
        new_drf = self._request.to_canonical(event=DefaultEvent())
        return self._from_drf(new_drf)

    def with_extra(self, extra: str | None) -> Self:
        """Return new device with extra modifier set or cleared.

        Args:
            extra: Extra modifier string (e.g. "FTP", "LOGGER:123:456"),
                   or None to remove the extra.
        """
        if extra is not None:
            parse_extra(extra)  # validate
        canonical = self._request.to_canonical(extra=None)
        if extra is not None:
            canonical += f"<-{extra}"
        return self._from_drf(canonical)

    def _from_drf(self, drf: str) -> Self:
        """Create new instance of same type from DRF. Override in subclasses."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.drf!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _DeviceBase):
            return NotImplemented
        return self.drf == other.drf

    def __hash__(self) -> int:
        return hash(self.drf)
