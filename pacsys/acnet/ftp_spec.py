"""
Parser for DataEventFactory-style FTP/snapshot event strings.

ACL and the legacy Java DAQ library use event strings like
``f,type=ftp,rate=60,dur=1;trig=e,2,1000;null`` to specify FTP/snapshot
requests with full control over rate, duration, trigger, and rearm.

This module turns those strings into frozen dataclass request objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

# ---------------------------------------------------------------------------
# Sample modes (snapshot only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodicSample:
    """Default periodic sampling."""


@dataclass(frozen=True)
class ClockSample:
    """Sample on clock events (up to 4 hex event bytes)."""

    events: tuple[int, ...]


@dataclass(frozen=True)
class ExternalSample:
    """Sample on external trigger with modifier 0-3."""

    modifier: int


SampleMode = Union[PeriodicSample, ClockSample, ExternalSample]

# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockTrigger:
    """Trigger on clock events (up to 8 hex events) with optional delay."""

    events: tuple[int, ...]
    delay_ms: int = 0


@dataclass(frozen=True)
class DeviceTrigger:
    """Trigger on a device reading matching mask/value."""

    device: str
    mask: int = 0
    value: int = 0
    delay_ms: int = 0


@dataclass(frozen=True)
class ExternalTrigger:
    """Trigger on external source with modifier."""

    modifier: int = 0


@dataclass(frozen=True)
class StateTrigger:
    """Trigger on device state comparison."""

    device: str
    value: int
    delay_ms: int = 0
    flag: str = "="


TriggerSpec = Union[ClockTrigger, DeviceTrigger, ExternalTrigger, StateTrigger]

# ---------------------------------------------------------------------------
# ReArm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReArmSpec:
    """Re-arm configuration for FTP/snapshot."""

    enabled: bool
    delay_event: str | None = None
    max_per_hour: int = -1


# ---------------------------------------------------------------------------
# Top-level specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FTPSpec:
    """Parsed FTP (continuous fast time plot) request."""

    rate_hz: float
    duration_s: float
    trigger: TriggerSpec | None = None
    rearm: ReArmSpec | None = None


@dataclass(frozen=True)
class SnapshotSpec:
    """Parsed snapshot request."""

    rate_hz: int
    duration_s: float
    num_points: int
    preference: str
    sample: SampleMode
    trigger: TriggerSpec | None = None
    rearm: ReArmSpec | None = None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_VALID_PREFERENCES = {"rate", "dur", "both", "none"}


def _parse_trigger(s: str) -> TriggerSpec:
    """Parse a trigger section like ``trig=e,02,FE,1000``."""
    # Split on first '=' to get "trig" and the rest, then split rest on ','
    # (avoids breaking flags like ">=" that contain '=')
    if not s.startswith("trig=") or len(s) < 6:
        raise ValueError(f"Invalid trigger section: {s}")

    parts = s[5:].split(",")  # after "trig="
    kind = parts[0].lower()
    args = parts[1:]

    if kind == "e":
        # Clock trigger: hex events, last decimal token is delay_ms
        if not args:
            return ClockTrigger(events=(), delay_ms=0)
        delay_ms = int(args[-1])
        events = tuple(int(p, 16) for p in args[:-1])
        if len(events) > 8:
            raise ValueError(f"Clock trigger supports max 8 events, got {len(events)}")
        return ClockTrigger(events=events, delay_ms=delay_ms)

    if kind == "d":
        # Device trigger: trig=d,<device>,mask=<hex>,val=<hex>,dly=<ms>
        # Java canonical format uses key=value for mask/val/dly (hex values)
        if not args:
            raise ValueError("Device trigger requires at least a device name")
        device = args[0]
        mask = 0
        value = 0
        delay_ms = 0
        for token in args[1:]:
            if token.startswith("mask="):
                mask = int(token[5:], 16)
            elif token.startswith("val="):
                value = int(token[4:], 16)
            elif token.startswith("dly="):
                delay_ms = int(token[4:])
            else:
                raise ValueError(f"Unknown device trigger field: {token!r}")
        return DeviceTrigger(device=device, mask=mask, value=value, delay_ms=delay_ms)

    if kind == "x":
        # External trigger: trig=x,mod=<0-3> (Java canonical format)
        modifier = 0
        if args:
            token = args[0]
            if token.startswith("mod="):
                modifier = int(token[4:])
            else:
                modifier = int(token)
        if modifier < 0 or modifier > 3:
            raise ValueError(f"External trigger modifier must be 0-3, got {modifier}")
        return ExternalTrigger(modifier=modifier)

    if kind == "s":
        # State trigger: device,value,delay,flag
        device = args[0] if len(args) > 0 else ""
        value = int(args[1]) if len(args) > 1 else 0
        delay_ms = int(args[2]) if len(args) > 2 else 0
        flag = args[3] if len(args) > 3 else "="
        return StateTrigger(device=device, value=value, delay_ms=delay_ms, flag=flag)

    raise ValueError(f"Unknown trigger type: {kind}")


def _parse_rearm(s: str) -> ReArmSpec:
    """Parse a rearm section like ``rearm=true,dly=p,60000,false,nmhr=30``."""
    tokens = re.split("[,=]", s)
    # tokens[0] == "rearm", tokens[1] == "true"/"false"
    if len(tokens) < 2 or tokens[0] != "rearm":
        raise ValueError(f"Invalid rearm section: {s}")

    enabled_s = tokens[1].lower()
    if enabled_s not in ("true", "false"):
        raise ValueError(f"Invalid rearm enabled value: {tokens[1]!r}")
    enabled = enabled_s == "true"

    # Scan for delay event (between dly= and nmhr) and max_per_hour
    delay_event: str | None = None
    max_per_hour = -1

    # Find dly= and nmhr= positions in the raw string
    dly_idx = s.find("dly=")
    nmhr_idx = s.find("nmhr=")

    if dly_idx >= 0:
        # Extract delay event string: from after "dly=" to just before ",nmhr"
        if nmhr_idx >= 0:
            delay_part = s[dly_idx + 4 : nmhr_idx].rstrip(",")
        else:
            delay_part = s[dly_idx + 4 :]
        # Java emits literal "null" for absent delay events
        if delay_part and delay_part != "null":
            delay_event = delay_part

    if nmhr_idx >= 0:
        nmhr_val = s[nmhr_idx + 5 :].split(",")[0]
        if nmhr_val:
            max_per_hour = int(nmhr_val)

    return ReArmSpec(enabled=enabled, delay_event=delay_event, max_per_hour=max_per_hour)


def _parse_sample(tokens: list[str], idx: int) -> tuple[SampleMode, int]:
    """Parse sample mode from scope tokens starting at idx (after smpl key).

    Returns (sample, new_idx) where new_idx is past the consumed tokens.
    """
    kind = tokens[idx].lower()
    idx += 1

    if kind == "p":
        return PeriodicSample(), idx

    if kind == "e":
        # Clock sample: collect hex event bytes (max 4)
        events = []
        while idx < len(tokens):
            try:
                events.append(int(tokens[idx], 16))
                idx += 1
            except ValueError:
                break
        if len(events) > 4:
            raise ValueError(f"Clock sample supports max 4 events, got {len(events)}")
        return ClockSample(events=tuple(events)), idx

    if kind == "x":
        # External sample: mod=N (key-value) or bare positional integer
        modifier = 0
        if idx < len(tokens) and tokens[idx] == "mod":
            idx += 1  # skip "mod" key
            if idx < len(tokens):
                modifier = int(tokens[idx])
                idx += 1
        elif idx < len(tokens):
            try:
                modifier = int(tokens[idx])
                idx += 1
            except ValueError:
                pass  # not a modifier token, leave for caller
        if modifier < 0 or modifier > 3:
            raise ValueError(f"External sample modifier must be 0-3, got {modifier}")
        return ExternalSample(modifier=modifier), idx

    raise ValueError(f"Unknown sample type: {kind}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_ftp_event(s: str) -> FTPSpec | SnapshotSpec:
    """Parse a DataEventFactory-style FTP/snapshot event string.

    Examples::

        parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,2,1000;null")
        parse_ftp_event("f,type=snp,rate=2048,dur=1.0,npts=2048,pref=rate,smpl=p;trig=e,2,0;null")

    Raises ``ValueError`` on invalid input.
    """
    if not s or len(s) < 12:
        raise ValueError(f"Event string too short: {s!r}")

    # Determine type from prefix
    prefix = s[:11].lower()
    if prefix == "f,type=ftp,":
        return _parse_ftp(s[11:])
    elif prefix == "f,type=snp,":
        return _parse_snapshot(s[11:])
    else:
        raise ValueError(f"Unknown event type prefix: {s[:11]!r}")


def _parse_ftp(remainder: str) -> FTPSpec:
    """Parse the FTP portion after stripping ``f,type=ftp,``."""
    sections = remainder.split(";")

    scope = sections[0]
    trigger_s = sections[1] if len(sections) > 1 else None
    rearm_s = sections[2] if len(sections) > 2 else None

    # Parse scope: rate=60.0,dur=1.0
    tokens = re.split("[,=]", scope)
    found_rate = False
    found_dur = False
    rate_hz = 0.0
    duration_s = 0.0
    i = 0
    while i < len(tokens):
        key = tokens[i].lower()
        if key == "rate" and i + 1 < len(tokens):
            rate_hz = float(tokens[i + 1])
            found_rate = True
            i += 2
        elif key == "dur" and i + 1 < len(tokens):
            duration_s = float(tokens[i + 1])
            found_dur = True
            i += 2
        else:
            i += 1

    if not found_rate:
        raise ValueError("FTP scope missing required 'rate' field")
    if not found_dur:
        raise ValueError("FTP scope missing required 'dur' field")

    trigger = _parse_trigger(trigger_s) if trigger_s and trigger_s.lower() != "null" else None
    rearm = _parse_rearm(rearm_s) if rearm_s and rearm_s.lower() != "null" else None

    return FTPSpec(rate_hz=rate_hz, duration_s=duration_s, trigger=trigger, rearm=rearm)


def _parse_snapshot(remainder: str) -> SnapshotSpec:
    """Parse the snapshot portion after stripping ``f,type=snp,``."""
    sections = remainder.split(";")

    scope = sections[0]
    trigger_s = sections[1] if len(sections) > 1 else None
    rearm_s = sections[2] if len(sections) > 2 else None

    # Parse scope tokens
    tokens = re.split("[,=]", scope)
    found: set[str] = set()
    rate_hz = 0
    duration_s = 0.0
    num_points = 0
    preference = "rate"
    sample: SampleMode = PeriodicSample()

    i = 0
    while i < len(tokens):
        key = tokens[i].lower()
        if key == "rate" and i + 1 < len(tokens):
            rate_hz = int(float(tokens[i + 1]))
            found.add("rate")
            i += 2
        elif key == "dur" and i + 1 < len(tokens):
            duration_s = float(tokens[i + 1])
            found.add("dur")
            i += 2
        elif key == "npts" and i + 1 < len(tokens):
            num_points = int(tokens[i + 1])
            found.add("npts")
            i += 2
        elif key == "pref" and i + 1 < len(tokens):
            pref = tokens[i + 1].lower()
            if pref not in _VALID_PREFERENCES:
                raise ValueError(f"Invalid preference: {pref!r} (expected one of {_VALID_PREFERENCES})")
            preference = pref
            found.add("pref")
            i += 2
        elif key == "smpl" and i + 1 < len(tokens):
            sample, i = _parse_sample(tokens, i + 1)
            found.add("smpl")
        else:
            i += 1

    missing = {"rate", "dur", "npts", "pref", "smpl"} - found
    if missing:
        raise ValueError(f"Snapshot scope missing required fields: {', '.join(sorted(missing))}")

    trigger = _parse_trigger(trigger_s) if trigger_s and trigger_s.lower() != "null" else None
    rearm = _parse_rearm(rearm_s) if rearm_s and rearm_s.lower() != "null" else None

    return SnapshotSpec(
        rate_hz=rate_hz,
        duration_s=duration_s,
        num_points=num_points,
        preference=preference,
        sample=sample,
        trigger=trigger,
        rearm=rearm,
    )
