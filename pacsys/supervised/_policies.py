"""Pluggable policy system for supervised proxy server."""

import enum
import fnmatch
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pacsys.drf3 import parse_request
from pacsys.drf3.field import DRF_FIELD
from pacsys.drf3.property import DRF_PROPERTY
from pacsys.drf_utils import prepare_for_write
from pacsys.types import Value

# Payloads in these fields are device counts/volts, not comparable to limits in engineering units.
_UNSCALED_FIELDS = frozenset(
    {
        DRF_FIELD.RAW,
        DRF_FIELD.PRIMARY,
        DRF_FIELD.VOLTS,
        DRF_FIELD.RAW_MIN,
        DRF_FIELD.RAW_MAX,
        DRF_FIELD.RAW_NOM,
        DRF_FIELD.RAW_TOL,
    }
)


def _numeric_elements(value: object) -> list[float] | None:
    """Flatten *value* to its numeric elements; None if any element is non-numeric.

    Policies must fail closed: a value they cannot interpret numerically
    (str, dict, mixed list, ...) returns None and must be denied for
    limited devices, not skipped. Enums (e.g. BasicControl.ON) are commands,
    not setpoints — their ordinals must never pass a numeric range check.
    Exception: raw bytes block writes (alarm blocks, ramp tables) are
    structured payloads, not setpoints — callers exempt them explicitly.
    """
    if isinstance(value, enum.Enum):
        return None
    if isinstance(value, (bool, int, float)):  # covers np.float64 (subclasses float)
        return [float(value)]
    if isinstance(value, np.generic):
        return [float(value)] if isinstance(value, (np.bool_, np.integer, np.floating)) else None
    if isinstance(value, np.ndarray):
        if value.dtype == bool or np.issubdtype(value.dtype, np.number):
            return [float(x) for x in np.asarray(value).ravel()]
        return None
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            sub = _numeric_elements(item)
            if sub is None:
                return None
            out.extend(sub)
        return out
    return None


def _raw_patterns(patterns: list[str] | None) -> tuple[str, ...]:
    if patterns is None:
        return ()
    if not isinstance(patterns, (list, tuple)):
        raise TypeError("allow_raw must be a list of device patterns")
    if any(not isinstance(pattern, str) or not pattern.strip() for pattern in patterns):
        raise ValueError("allow_raw patterns must be non-empty strings")
    return tuple(patterns)


def _glob_matches(device_name: str, pattern: str, is_acnet: bool) -> bool:
    if is_acnet:
        return fnmatch.fnmatchcase(device_name.upper(), pattern.upper())
    return fnmatch.fnmatchcase(device_name, pattern)


def _raw_allowed(device_name: str, patterns: tuple[str, ...], is_acnet: bool) -> bool:
    return any(_glob_matches(device_name, pattern, is_acnet) for pattern in patterns)


@dataclass(frozen=True)
class RequestContext:
    """Context for a single RPC request, passed to policy checks."""

    drfs: list[str]  # Fixed target slots; policies must not modify or reorder
    rpc_method: str  # "Read" or "Set"
    peer: str
    metadata: dict[str, str]
    values: list[tuple[str, Value]]  # DRFs fixed and aligned with drfs; payloads may change
    raw_request: object  # raw protobuf message
    allowed: frozenset[int] = frozenset()  # slot indices approved for this operation


@dataclass(frozen=True)
class PolicyDecision:
    """Result of a policy check.

    On deny: ``reason`` is required, ``ctx`` is ignored.
    On allow without modification: ``ctx`` is None.
    On allow with modification: ``ctx`` is a new RequestContext.
    """

    allowed: bool
    reason: str | None = None
    ctx: RequestContext | None = None

    def __post_init__(self):
        if not self.allowed and not self.reason:
            raise ValueError("PolicyDecision must include a reason when denied")


_ALLOW = PolicyDecision(allowed=True)


class Policy(ABC):
    """Abstract base for policy checks. Implement check() to allow or deny requests."""

    @property
    def allows_writes(self) -> bool:
        """Whether this policy explicitly gates write access."""
        return False

    @abstractmethod
    def check(self, ctx: RequestContext) -> PolicyDecision: ...


class ReadOnlyPolicy(Policy):
    """Denies Set RPCs, allows everything else."""

    def check(self, ctx: RequestContext) -> PolicyDecision:
        if ctx.rpc_method == "Set":
            return PolicyDecision(allowed=False, reason="Write operations disabled")
        return _ALLOW


# TODO: patterns match device names only — operators cannot express property scope
# (e.g. "deny raw/byte-range writes to B:*" or "allow ANALOG_ALARM on B:*"). Adding
# property/field matching requires normalizing the wire DRF first (prepare_for_write),
# ideally once at the gateway boundary (_server.py) so all policies see canonical DRFs.
class DeviceAccessPolicy(Policy):
    """Allow or deny access based on device name patterns.

    Reads are allowed by default and can only be restricted with mode="deny";
    writes are denied by default and require mode="allow" approval. Hence
    mode="allow" with action="read" would be a silent no-op and is rejected.

    Args:
        patterns: List of patterns (e.g. ["M:*", "G:AMANDA"])
        mode: "allow" = approve matching devices for writes, "deny" = block matching devices
        action: "all" (both RPCs), "read" (Read only), "set" (Set only)
        syntax: "glob" (fnmatch, default) or "regex" (full-match). Matching is
            case-insensitive for ACNET devices and case-sensitive for EPICS PVs.
    """

    def __init__(
        self,
        patterns: list[str],
        mode: str = "allow",
        action: str = "all",
        syntax: str = "glob",
    ):
        if not patterns:
            raise ValueError("patterns must not be empty")
        if mode not in ("allow", "deny"):
            raise ValueError(f"mode must be 'allow' or 'deny', got {mode!r}")
        if action not in ("all", "read", "set"):
            raise ValueError(f"action must be 'all', 'read', or 'set', got {action!r}")
        if syntax not in ("glob", "regex"):
            raise ValueError(f"syntax must be 'glob' or 'regex', got {syntax!r}")
        if mode == "allow" and action == "read":
            raise ValueError(
                "mode='allow' with action='read' has no effect: reads are allowed by default. "
                "Use mode='deny' to block specific devices, or mode='deny' with syntax='regex' "
                "and a negated pattern (e.g. r'(?!M:).*') for a read allowlist."
            )
        self._patterns = patterns
        self._mode = mode
        self._action = action
        self._syntax = syntax
        if syntax == "regex":
            self._compiled_acnet = [re.compile(p, re.IGNORECASE) for p in patterns]
            self._compiled_epics = [re.compile(p) for p in patterns]

    @property
    def allows_writes(self) -> bool:
        return self._mode == "allow" and self._action in ("set", "all")

    def _matches(self, device_name: str, is_acnet: bool) -> bool:
        if self._syntax == "regex":
            patterns = self._compiled_acnet if is_acnet else self._compiled_epics
            return any(pattern.fullmatch(device_name) for pattern in patterns)
        return any(_glob_matches(device_name, pattern, is_acnet) for pattern in self._patterns)

    def _applies(self, rpc_method: str) -> bool:
        if self._action == "all":
            return True
        return self._action == rpc_method.lower()

    def check(self, ctx: RequestContext) -> PolicyDecision:
        if not self._applies(ctx.rpc_method):
            return _ALLOW

        if self._mode == "deny":
            for drf in ctx.drfs:
                req = parse_request(drf)
                name = req.device
                if self._matches(name, req.is_acnet):
                    return PolicyDecision(allowed=False, reason=f"Device {name} is denied")
            return _ALLOW

        # mode="allow": approve matching slots, pass through non-matching
        approved = set(ctx.allowed)
        for i, drf in enumerate(ctx.drfs):
            req = parse_request(drf)
            if self._matches(req.device, req.is_acnet):
                approved.add(i)
        new_allowed = frozenset(approved)
        if new_allowed == ctx.allowed:
            return _ALLOW
        from dataclasses import replace

        return PolicyDecision(allowed=True, ctx=replace(ctx, allowed=new_allowed))


def _write_target(drf: str) -> tuple[str, DRF_PROPERTY, DRF_FIELD | None, bool]:
    """Return the device, property, field, and ACNET classification of a write."""
    req = parse_request(prepare_for_write(drf))
    dev = req.device.upper() if req.is_acnet else req.device
    return dev, req.property, req.field, req.is_acnet


def _unscaled_denial(device_name: str, field: DRF_FIELD) -> PolicyDecision:
    """Counts/volts payloads require an allow_raw exemption from engineering-unit limits."""
    label = DRF_FIELD.PRIMARY if field is DRF_FIELD.VOLTS else field
    return PolicyDecision(
        allowed=False,
        reason=f"{label.name} field write to range-limited device {device_name} is not comparable to its engineering-unit limit",
    )


def _peer_key(peer: str) -> str:
    """Bucket key: gRPC peer without its ephemeral port, so a reconnect cannot reset the limit."""
    return peer.rsplit(":", 1)[0] if peer.startswith(("ipv4:", "ipv6:")) else peer


class RateLimitPolicy(Policy):
    """Sliding window rate limit per client address (port ignored).

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

        key = _peer_key(ctx.peer)
        with self._lock:
            # Prune buckets with no activity inside the window (they would be empty anyway)
            if len(self._timestamps) > 100:
                stale = [peer for peer, ts_list in self._timestamps.items() if not ts_list or ts_list[-1] <= cutoff]
                for peer in stale:
                    del self._timestamps[peer]

            times = self._timestamps.get(key, [])
            times = [t for t in times if t > cutoff]

            if len(times) >= self._max_requests:
                self._timestamps[key] = times
                return PolicyDecision(
                    allowed=False,
                    reason=f"Rate limit exceeded ({self._max_requests} per {self._window}s)",
                )

            times.append(now)
            self._timestamps[key] = times

        return _ALLOW


class ValueRangePolicy(Policy):
    """Deny writes where numeric values fall outside allowed ranges.

    Args:
        limits: Mapping of device name glob pattern to (min, max) bounds. All
            matching bounds are intersected.
        allow_raw: Device patterns explicitly allowed to bypass numeric limits
            for structured raw writes such as ramp or alarm blocks, and for
            RAW/PRIMARY/VOLTS field writes (otherwise denied as not comparable).

    CONTROL writes (``M&X``, ``.CONTROL``, ``.STATUS.ON``) are command ordinals, not
    values, and pass through; restrict them with ``DeviceAccessPolicy``.
    """

    def __init__(self, limits: dict[str, tuple[float, float]], *, allow_raw: list[str] | None = None):
        if not limits:
            raise ValueError("limits must not be empty")
        self._limits = limits
        self._allow_raw = _raw_patterns(allow_raw)

    def _bound_for(self, device_name: str, is_acnet: bool) -> tuple[float, float] | None:
        bounds = [bound for pattern, bound in self._limits.items() if _glob_matches(device_name, pattern, is_acnet)]
        if not bounds:
            return None
        return max(bound[0] for bound in bounds), min(bound[1] for bound in bounds)

    def check(self, ctx: RequestContext) -> PolicyDecision:
        if ctx.rpc_method != "Set":
            return _ALLOW
        for drf, value in ctx.values:
            name, property_, field, is_acnet = _write_target(drf)
            bound = self._bound_for(name, is_acnet)
            if bound is None or property_ is DRF_PROPERTY.CONTROL:
                continue  # CONTROL payloads are command ordinals, not values (gate them by device access)
            lo, hi = bound
            if lo > hi:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Contradictory value ranges for {name}: empty intersection [{lo}, {hi}]",
                )
            if isinstance(value, (bytes, bytearray)) or field in _UNSCALED_FIELDS:
                if _raw_allowed(name, self._allow_raw, is_acnet):
                    continue  # exempt raw write: nothing comparable to check
                if field in _UNSCALED_FIELDS:
                    return _unscaled_denial(name, field)
                return PolicyDecision(
                    allowed=False,
                    reason=f"Raw byte write to range-limited device {name} requires an allow_raw exemption",
                )
            elements = _numeric_elements(value)
            if elements is None:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Non-numeric value {value!r} for range-limited device {name}",
                )
            for v in elements:
                # NaN/inf fail the comparison and are denied (fail closed)
                if not (lo <= v <= hi):
                    return PolicyDecision(
                        allowed=False,
                        reason=f"Value {v} for {name} outside range [{lo}, {hi}]",
                    )
        return _ALLOW


def _validate_policy_context(policy: Policy, ctx: RequestContext, original_drfs: list[str], rpc_method: str) -> None:
    """Enforce stable target slots while allowing value transformations."""
    policy_name = type(policy).__name__
    if ctx.drfs != original_drfs:
        raise ValueError(f"{policy_name} modified ctx.drfs; target transformations are not supported")

    valid_slots = frozenset(range(len(original_drfs)))
    if not ctx.allowed <= valid_slots:
        raise ValueError(f"{policy_name} produced out-of-range allowed slots")

    if rpc_method == "Read":
        if ctx.values:
            raise ValueError(f"{policy_name} added values to a Read request")
        return

    if len(ctx.values) != len(original_drfs):
        raise ValueError(
            f"{policy_name} produced mismatched drfs/values: {len(original_drfs)} drfs vs {len(ctx.values)} values"
        )
    if [drf for drf, _ in ctx.values] != original_drfs:
        raise ValueError(f"{policy_name} modified or reordered value DRFs; target transformations are not supported")


def evaluate_policies(policies: list[Policy], ctx: RequestContext) -> PolicyDecision:
    """Evaluate a chain of policies. First denial short-circuits.

    Each policy sees the (potentially modified) context from the previous
    policy. The final decision always carries ``ctx`` set to the final context.
    """
    original_drfs = list(ctx.drfs)
    rpc_method = ctx.rpc_method
    current = ctx
    for policy in policies:
        decision = policy.check(current)
        if not decision.allowed:
            return decision
        current = decision.ctx if decision.ctx is not None else current
        _validate_policy_context(policy, current, original_drfs, rpc_method)
    return PolicyDecision(allowed=True, ctx=current)
