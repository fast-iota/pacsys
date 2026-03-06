"""watch: block until a condition is met on a channel."""

from __future__ import annotations

import threading
from typing import Callable, TYPE_CHECKING

from pacsys.types import DeviceSpec, Reading
from pacsys.exp._resolve import resolve_drf, resolve_backend

if TYPE_CHECKING:
    from pacsys.backends import Backend


def watch(
    device: DeviceSpec,
    condition: Callable[[Reading], bool],
    *,
    timeout: float = 30.0,
    backend: Backend | None = None,
) -> Reading:
    """Block until condition(reading) returns True.

    Args:
        device: DRF string or Device object (must have an event for streaming).
        condition: Predicate receiving a Reading, returns True to stop.
        timeout: Max seconds to wait.
        backend: Optional backend.

    Returns:
        The Reading that satisfied the condition.

    Raises:
        TimeoutError: If condition not met within timeout.
    """
    drf = resolve_drf(device)
    be = resolve_backend(backend)

    result_box: list[Reading] = []
    error_box: list[Exception] = []
    done = threading.Event()

    def on_reading(reading, handle):
        try:
            if condition(reading):
                result_box.append(reading)
                done.set()
        except Exception as exc:
            error_box.append(exc)
            done.set()

    def on_error(exc, handle):
        error_box.append(exc)
        done.set()

    handle = None
    try:
        handle = be.subscribe([drf], callback=on_reading, on_error=on_error)
        if not done.wait(timeout=timeout):
            raise TimeoutError(f"Condition not met within {timeout}s for {drf}")
        if result_box:
            return result_box[0]
        if error_box:
            raise error_box[0]
    finally:
        if handle is not None:
            handle.stop()

    return result_box[0]
