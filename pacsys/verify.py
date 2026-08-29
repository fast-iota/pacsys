"""
Verify - write-and-verify configuration for Device write operations.

Verify instances configure how a write is verified by reading back the value
after writing. They can be used directly, as context managers (to set defaults
for a block of code), or via the task-local context stack.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pacsys.types import Value


@dataclass(frozen=True)
class Verify:
    """Configuration for write-and-verify operations.

    Can be used directly as a parameter to Device.write()/control(), or as
    a context manager to set defaults for a block of code.

    Attributes:
        check_first: Read current value before writing; skip if already matches.
        tolerance: Comparison tolerance for numeric readback.
        initial_delay: Seconds to wait after write before first readback.
        retry_delay: Seconds between readback attempts.
        max_attempts: Max readback attempts before declaring failure.
        readback: Optional DRF override for readback (default: same property).
        always: When used as context default, auto-verify calls with verify=None.
    """

    check_first: bool = False
    tolerance: float = 0.0
    initial_delay: float = 0.3
    retry_delay: float = 0.5
    max_attempts: int = 3
    readback: str | None = None
    always: bool = False

    def __post_init__(self) -> None:
        for name in ("tolerance", "initial_delay", "retry_delay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, Integral):
            raise TypeError("max_attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        for name in ("check_first", "always"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.readback is not None:
            if not isinstance(self.readback, str):
                raise TypeError("readback must be a DRF string or None")
            from pacsys.drf3 import parse_request

            parse_request(self.readback)  # ValueError on invalid DRF, before any write

    def __enter__(self) -> Verify:
        _push_verify(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _pop_verify()
        return False


# Immutable values keep inherited asyncio task contexts independent.
_verify_stack: ContextVar[tuple[Verify, ...]] = ContextVar("pacsys_verify_stack", default=())


def _push_verify(v: Verify) -> None:
    _verify_stack.set((*_verify_stack.get(), v))


def _pop_verify() -> None:
    stack = _verify_stack.get()
    if stack:
        _verify_stack.set(stack[:-1])


def get_active_verify() -> Verify | None:
    """Return the current Verify from the task-local stack, or None."""
    stack = _verify_stack.get()
    return stack[-1] if stack else None


def resolve_verify(verify: bool | Verify | None) -> Verify | None:
    """Resolve a verify parameter to a Verify instance or None.

    Args:
        verify: User-supplied verify parameter:
            - False  -> None (never verify)
            - True   -> active context or Verify() defaults
            - Verify -> use that instance
            - None   -> if context.always: use context; else: None
    """
    if verify is False:
        return None
    if verify is True:
        active = get_active_verify()
        return active if active is not None else Verify()
    if isinstance(verify, Verify):
        return verify
    active = get_active_verify()
    if active is not None and active.always:
        return active
    return None


def values_match(a: Value, b: Value, tolerance: float = 0.0) -> bool:
    """Compare two values within tolerance."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            if a_arr.shape != b_arr.shape:
                return False
            a_bool = a_arr.dtype.kind == "b"
            b_bool = b_arr.dtype.kind == "b"
            if a_bool or b_bool:
                return a_bool and b_bool and bool(np.array_equal(a_arr, b_arr))
            if a_arr.dtype.kind not in "iufc" or b_arr.dtype.kind not in "iufc":
                return bool(np.array_equal(a_arr, b_arr))
            return bool(np.allclose(a_arr, b_arr, atol=tolerance, rtol=0.0, equal_nan=False))
        except (TypeError, ValueError):
            return False

    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != len(b):
            return False
        return all(values_match(x, y, tolerance) for x, y in zip(a, b, strict=True))

    a_scalar: object = a.item() if isinstance(a, np.generic) else a
    b_scalar: object = b.item() if isinstance(b, np.generic) else b

    a_bool = isinstance(a_scalar, bool)
    b_bool = isinstance(b_scalar, bool)
    if a_bool or b_bool:
        return a_bool and b_bool and a_scalar == b_scalar
    if isinstance(a_scalar, Real) and isinstance(b_scalar, Real):
        return bool(np.isclose(a_scalar, b_scalar, atol=tolerance, rtol=0.0, equal_nan=False))
    try:
        result = a_scalar == b_scalar
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _normalize_control_readback(value: Value, field: str) -> Value:
    """Extract a mapped field from a backend basic-status response.

    DPM delivers ``.STATUS.<field>`` as a 0.0/1.0 double (DPMProtocolReplierPC.sendReply(boolean));
    only those two values are coerced to bool so text or other numbers still fail to match.
    """
    if isinstance(value, dict) and field in value:
        value = value[field]
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, Real) and not isinstance(scalar, bool) and scalar in (0, 1):
        return bool(scalar)
    return value


__all__ = ["Verify", "get_active_verify", "resolve_verify", "values_match"]
