"""Shared base for Device and AsyncDevice - pure DRF logic, no I/O."""

from __future__ import annotations

import inspect
from typing import Any

from pacsys.drf3 import DataRequest, parse_event
from pacsys.drf3.field import (
    DRF_FIELD,
    parse_field,
    DEFAULT_FIELD_FOR_PROPERTY,
    ALLOWED_FIELD_FOR_PROPERTY,
)
from pacsys.drf3.property import DRF_PROPERTY
from pacsys.drf3.range import ARRAY_RANGE
from pacsys.drf3.event import PeriodicEvent
from pacsys.types import BasicControl

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


def _min_positional_params(fn: Any) -> int | None:
    """Count minimum positional args *fn* accepts, or None if uninspectable."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    count = 0
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return None  # *args/**kwargs — can't tell
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            count += 1
    return count


def _validate_callback(callback: object, on_error: object) -> None:
    """Validate callback/on_error: must be callable with correct arity."""
    if callback is not None:
        if not callable(callback):
            raise TypeError(
                f"callback must be callable, got {type(callback).__name__} "
                f"— did you mean subscribe(event={callback!r})?"
            )
        n = _min_positional_params(callback)
        if n is not None and n < 2:
            raise TypeError(f"callback must accept 2 arguments (reading, handle), but {callback!r} accepts {n}")
    if on_error is not None:
        if not callable(on_error):
            raise TypeError(f"on_error must be callable, got {type(on_error).__name__}")
        n = _min_positional_params(on_error)
        if n is not None and n < 2:
            raise TypeError(f"on_error must accept 2 arguments (exception, handle), but {on_error!r} accepts {n}")


class _DeviceBase:
    """DRF building, field resolution, fluent modification. No I/O."""

    __slots__ = ("_request",)
    _request: DataRequest

    _BOOL_STATUS_FIELDS = frozenset({"ON", "READY", "REMOTE", "POSITIVE", "RAMP"})

    def __init__(self, request: DataRequest):
        object.__setattr__(self, "_request", request)

    @property
    def drf(self) -> str:
        return self._request.to_canonical()

    @property
    def name(self) -> str:
        return self._request.device

    @property
    def request(self) -> DataRequest:
        return self._request

    @property
    def has_event(self) -> bool:
        return self._request.event is not None and self._request.event.mode != "U"

    @property
    def is_periodic(self) -> bool:
        return isinstance(self._request.event, PeriodicEvent)

    def _build_drf(self, prop: DRF_PROPERTY, field: DRF_FIELD | None, event: str) -> str:
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
        if field is None:
            return DEFAULT_FIELD_FOR_PROPERTY.get(prop)
        f = parse_field(field.upper())
        allowed = ALLOWED_FIELD_FOR_PROPERTY.get(prop, [])
        if f not in allowed:
            raise ValueError(f"Field '{field}' not allowed for {prop.name}")
        return f

    def with_event(self, event: str) -> _DeviceBase:
        new_event = parse_event(event)
        new_drf = self._request.to_canonical(event=new_event)
        return self._from_drf(new_drf)

    def with_range(self, start: int | None = None, end: int | None = None, *, at: int | None = None) -> _DeviceBase:
        if at is not None:
            if start is not None or end is not None:
                raise ValueError("'at' cannot be combined with 'start'/'end'")
            new_range = ARRAY_RANGE(mode="single", low=at)
        elif start is not None:
            new_range = ARRAY_RANGE(mode="std", low=start, high=end)
        else:
            new_range = ARRAY_RANGE(mode="full")
        new_drf = self._request.to_canonical(range=new_range)
        return self._from_drf(new_drf)

    def _from_drf(self, drf: str) -> _DeviceBase:
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
