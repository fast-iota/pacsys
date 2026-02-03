"""
Verify - write-and-verify configuration for Device write operations.

Verify instances configure how a write is verified by reading back the value
after writing. They can be used directly, as context managers (to set defaults
for a block of code), or via the thread-local context stack.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np

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
    readback: Optional[str] = None
    always: bool = False

    @classmethod
    def defaults(cls, **kwargs) -> Verify:
        """Create Verify with custom defaults."""
        return cls(**kwargs)

    def __enter__(self) -> Verify:
        _push_verify(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _pop_verify()
        return False


# Thread-local context stack
_local = threading.local()


def _push_verify(v: Verify) -> None:
    stack = getattr(_local, "stack", None)
    if stack is None:
        _local.stack = stack = []
    stack.append(v)


def _pop_verify() -> None:
    stack = getattr(_local, "stack", None)
    if stack:
        stack.pop()


@contextmanager
def verify_context(verify: Verify):
    """Context manager that pushes a Verify onto the thread-local stack."""
    _push_verify(verify)
    try:
        yield verify
    finally:
        _pop_verify()


def get_active_verify() -> Optional[Verify]:
    """Return the current Verify from the thread-local stack, or None."""
    stack = getattr(_local, "stack", None)
    return stack[-1] if stack else None


def resolve_verify(verify: bool | Verify | None) -> Optional[Verify]:
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
    # verify is None
    active = get_active_verify()
    if active is not None and active.always:
        return active
    return None


def values_match(a: Value, b: Value, tolerance: float = 0.0) -> bool:
    """Compare two values within tolerance."""
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tolerance
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        return bool(np.allclose(a, b, atol=tolerance))
    return a == b


__all__ = ["Verify", "verify_context", "get_active_verify", "resolve_verify", "values_match"]
