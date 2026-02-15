"""Classify DRF events as one-shot or streaming for request routing."""

from pacsys.drf3 import parse_request
from pacsys.drf3.event import (
    ImmediateEvent,
    NeverEvent,
)


def is_oneshot_event(drf: str) -> bool:
    """True only for @I or @N.

    Everything else is streaming: no event and @U both resolve to the
    device's default event which is typically @p,1000 (periodic).
    @P, @Q, @E, @S are all explicitly repetitive.
    """
    req = parse_request(drf)
    event = req.event
    if isinstance(event, (ImmediateEvent, NeverEvent)):
        return True
    return False


def all_oneshot(drfs: list[str]) -> bool:
    """True if ALL drfs are one-shot. Mixed list -> streaming path."""
    return all(is_oneshot_event(drf) for drf in drfs)
