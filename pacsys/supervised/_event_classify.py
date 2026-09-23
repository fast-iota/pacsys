"""Classify DRF events as one-shot or streaming for request routing."""

from pacsys.drf3 import parse_request
from pacsys.drf3.event import (
    ImmediateEvent,
    NeverEvent,
)
from pacsys.drf3.extra import HISTORICAL_EXTRAS


def is_oneshot_event(drf: str) -> bool:
    """True for @I, @N, and bounded historical (logger) requests.

    Everything else is streaming: no event and @U both resolve to the
    device's default event which is typically @p,1000 (periodic).
    @P, @Q, @E, @S are all explicitly repetitive. Logger sources carry
    no event (DPM rejects @I with them) but return a finite result.
    """
    req = parse_request(drf)
    return req.extra in HISTORICAL_EXTRAS or isinstance(req.event, (ImmediateEvent, NeverEvent))


def all_oneshot(drfs: list[str]) -> bool:
    """True if ALL drfs are one-shot. Mixed list -> streaming path."""
    return all(is_oneshot_event(drf) for drf in drfs)
