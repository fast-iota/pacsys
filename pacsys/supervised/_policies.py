"""Pluggable policy system for supervised proxy server."""

import fnmatch
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from pacsys.drf_utils import get_device_name


@dataclass(frozen=True)
class RequestContext:
    """Context for a single RPC request, passed to policy checks."""

    drfs: list[str]
    rpc_method: str  # "Read" or "Set"
    peer: str
    metadata: dict[str, str]
    values: list[tuple[str, object]]  # [(DRF, value), ...] — empty for reads
    raw_request: object  # raw protobuf message


@dataclass(frozen=True)
class PolicyDecision:
    """Result of a policy check.

    On deny: ``reason`` is required, ``ctx`` is ignored.
    On allow without modification: ``ctx`` is None.
    On allow with modification: ``ctx`` is a new RequestContext.
    """

    allowed: bool
    reason: Optional[str] = None  # required when denied
    ctx: Optional[RequestContext] = None  # modified context, None = no change

    def __post_init__(self):
        if not self.allowed and not self.reason:
            raise ValueError("PolicyDecision must include a reason when denied")


_ALLOW = PolicyDecision(allowed=True)


class Policy(ABC):
    """Abstract base for policy checks. Implement check() to allow or deny requests."""

    @abstractmethod
    def check(self, ctx: RequestContext) -> PolicyDecision: ...


class ReadOnlyPolicy(Policy):
    """Denies Set RPCs, allows everything else."""

    def check(self, ctx: RequestContext) -> PolicyDecision:
        if ctx.rpc_method == "Set":
            return PolicyDecision(allowed=False, reason="Write operations disabled")
        return _ALLOW


class DeviceAccessPolicy(Policy):
    """Allow or deny access based on device name patterns.

    Args:
        patterns: List of patterns (e.g. ["M:*", "G:AMANDA"])
        mode: "allow" = only matching devices allowed, "deny" = matching devices blocked
        syntax: "glob" (fnmatch, default) or "regex" (full-match, case-insensitive)
    """

    def __init__(self, patterns: list[str], mode: str = "allow", syntax: str = "glob"):
        if not patterns:
            raise ValueError("patterns must not be empty")
        if mode not in ("allow", "deny"):
            raise ValueError(f"mode must be 'allow' or 'deny', got {mode!r}")
        if syntax not in ("glob", "regex"):
            raise ValueError(f"syntax must be 'glob' or 'regex', got {syntax!r}")
        self._patterns = patterns
        self._mode = mode
        self._syntax = syntax
        if syntax == "regex":
            self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def _matches(self, device_name: str) -> bool:
        if self._syntax == "regex":
            return any(r.fullmatch(device_name) for r in self._compiled)
        return any(fnmatch.fnmatchcase(device_name.upper(), p.upper()) for p in self._patterns)

    def check(self, ctx: RequestContext) -> PolicyDecision:
        for drf in ctx.drfs:
            name = get_device_name(drf)
            matched = self._matches(name)
            if self._mode == "allow" and not matched:
                return PolicyDecision(allowed=False, reason=f"Device {name} not in allow list")
            if self._mode == "deny" and matched:
                return PolicyDecision(allowed=False, reason=f"Device {name} is denied")
        return _ALLOW


class RateLimitPolicy(Policy):
    """Sliding window rate limit per peer.

    Args:
        max_requests: Maximum requests per window
        window_seconds: Window size in seconds (default: 60)
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        if max_requests <= 0:
            raise ValueError(f"max_requests must be positive, got {max_requests}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self._max_requests = max_requests
        self._window = window_seconds
        self._lock = threading.Lock()
        self._timestamps: dict[str, list[float]] = {}

    def check(self, ctx: RequestContext) -> PolicyDecision:
        now = time.monotonic()
        cutoff = now - self._window

        with self._lock:
            times = self._timestamps.get(ctx.peer, [])
            # Prune expired entries
            times = [t for t in times if t > cutoff]

            if len(times) >= self._max_requests:
                self._timestamps[ctx.peer] = times
                return PolicyDecision(
                    allowed=False,
                    reason=f"Rate limit exceeded ({self._max_requests} per {self._window}s)",
                )

            times.append(now)
            self._timestamps[ctx.peer] = times

        return _ALLOW


class ValueRangePolicy(Policy):
    """Deny writes where numeric values fall outside allowed ranges.

    Args:
        limits: Mapping of device name glob pattern to (min, max) bounds.
    """

    def __init__(self, limits: dict[str, tuple[float, float]]):
        if not limits:
            raise ValueError("limits must not be empty")
        self._limits = limits

    def _bound_for(self, device_name: str) -> Optional[tuple[float, float]]:
        upper = device_name.upper()
        for pattern, bound in self._limits.items():
            if fnmatch.fnmatchcase(upper, pattern.upper()):
                return bound
        return None

    def check(self, ctx: RequestContext) -> PolicyDecision:
        if ctx.rpc_method != "Set":
            return _ALLOW
        for drf, value in ctx.values:
            if not isinstance(value, (int, float)):
                continue
            bound = self._bound_for(get_device_name(drf))
            if bound is None:
                continue
            lo, hi = bound
            if not (lo <= value <= hi):
                name = get_device_name(drf)
                return PolicyDecision(
                    allowed=False,
                    reason=f"Value {value} for {name} outside range [{lo}, {hi}]",
                )
        return _ALLOW


@dataclass(frozen=True)
class SlewLimit:
    """Constraints for a single device pattern in :class:`SlewRatePolicy`.

    At least one of ``max_step`` or ``max_rate`` must be set.

    Attributes:
        max_step: Maximum absolute change per write (units).
        max_rate: Maximum rate of change (units/second).
    """

    max_step: Optional[float] = None
    max_rate: Optional[float] = None

    def __post_init__(self):
        if self.max_step is None and self.max_rate is None:
            raise ValueError("SlewLimit requires at least one of max_step or max_rate")


class SlewRatePolicy(Policy):
    """Deny writes that change too fast or by too much.

    Args:
        limits: Mapping of device name glob pattern to :class:`SlewLimit`.

    First write to any device is always allowed (no history).
    History is updated on allow (accepts that failed backend writes will
    leave stale history).
    """

    def __init__(self, limits: dict[str, SlewLimit]):
        if not limits:
            raise ValueError("limits must not be empty")
        self._limits = limits
        self._lock = threading.Lock()
        self._history: dict[str, tuple[float, float]] = {}  # device -> (value, timestamp)

    def _limit_for(self, device_name: str) -> Optional[SlewLimit]:
        upper = device_name.upper()
        for pattern, limit in self._limits.items():
            if fnmatch.fnmatchcase(upper, pattern.upper()):
                return limit
        return None

    def check(self, ctx: RequestContext) -> PolicyDecision:
        if ctx.rpc_method != "Set":
            return _ALLOW

        now = time.monotonic()

        with self._lock:
            # First pass: validate all values
            for drf, value in ctx.values:
                if not isinstance(value, (int, float)):
                    continue
                name = get_device_name(drf)
                limit = self._limit_for(name)
                if limit is None:
                    continue
                prev = self._history.get(name)
                if prev is None:
                    continue  # first write always allowed
                prev_value, prev_time = prev
                delta = abs(value - prev_value)

                if limit.max_step is not None and delta > limit.max_step:
                    return PolicyDecision(
                        allowed=False,
                        reason=f"Step {delta:.4g} for {name} exceeds limit {limit.max_step}",
                    )

                if limit.max_rate is not None:
                    dt = max(now - prev_time, 1e-9)
                    rate = delta / dt
                    if rate > limit.max_rate:
                        return PolicyDecision(
                            allowed=False,
                            reason=f"Slew rate {rate:.1f}/s for {name} exceeds limit {limit.max_rate}/s",
                        )

            # Second pass: update history (only if all passed)
            for drf, value in ctx.values:
                if not isinstance(value, (int, float)):
                    continue
                name = get_device_name(drf)
                if self._limit_for(name) is not None:
                    self._history[name] = (float(value), now)

        return _ALLOW


def evaluate_policies(policies: list[Policy], ctx: RequestContext) -> PolicyDecision:
    """Evaluate a chain of policies. First denial short-circuits.

    Each policy sees the (potentially modified) context from the previous
    policy. The final decision always carries ``ctx`` set to the final context.
    """
    current = ctx
    for policy in policies:
        decision = policy.check(current)
        if not decision.allowed:
            return decision
        if decision.ctx is not None:
            current = decision.ctx
    return PolicyDecision(allowed=True, ctx=current)
