"""read_fresh: wait for one new reading per channel via temporary subscription."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from pacsys.types import DeviceSpec, Reading
from pacsys.drf_utils import has_event, replace_event
from pacsys.exp._resolve import resolve_drf, resolve_backend

if TYPE_CHECKING:
    from pacsys.backends import Backend


def read_fresh(
    devices: list[DeviceSpec],
    *,
    default_event: str | None = None,
    timeout: float = 5.0,
    backend: Backend | None = None,
) -> list[Reading]:
    """Wait for one fresh reading per channel via temporary subscription.

    Args:
        devices: List of DRF strings or Device objects.
        default_event: Event to apply to DRFs that lack one (e.g. "p,1000").
        timeout: Max seconds to wait for all channels.
        backend: Optional backend (uses global default if None).

    Returns:
        list[Reading] in same order as input.

    Raises:
        TimeoutError: If any channel doesn't deliver within timeout.
        ValueError: If devices is empty.
    """
    if not devices:
        raise ValueError("devices cannot be empty")

    drfs = []
    for d in devices:
        drf = resolve_drf(d)
        if default_event is not None and not has_event(drf):
            drf = replace_event(drf, default_event)
        drfs.append(drf)

    be = resolve_backend(backend)

    # Deduplicate DRFs for subscription
    unique_drfs = list(dict.fromkeys(drfs))

    results: dict[str, Reading] = {}
    error_box: list[Exception] = []
    lock = threading.Lock()
    done = threading.Event()

    def on_reading(reading, handle):
        with lock:
            if reading.drf not in results:
                results[reading.drf] = reading
                if len(results) >= len(unique_drfs):
                    done.set()

    def on_error(exc, handle):
        error_box.append(exc)
        done.set()

    handle = None
    try:
        handle = be.subscribe(unique_drfs, callback=on_reading, on_error=on_error)
        if not done.wait(timeout=timeout):
            with lock:
                missing = [d for d in unique_drfs if d not in results]
            raise TimeoutError(f"Timed out waiting for: {missing}")
        # Success takes priority over late-arriving errors
        with lock:
            if len(results) >= len(unique_drfs):
                return [results[drf] for drf in drfs]
        if error_box:
            raise error_box[0]
    finally:
        if handle is not None:
            handle.stop()

    return [results[drf] for drf in drfs]
