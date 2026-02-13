"""Shared buffered subscription handle using Condition + deque."""

import logging
import threading
import time
from collections import deque
from typing import Iterator, Optional

from pacsys.types import Reading, SubscriptionHandle

logger = logging.getLogger(__name__)

_DEFAULT_BUFFER_MAXSIZE = 10_000


class BufferedSubscriptionHandle(SubscriptionHandle):
    """Base subscription handle with Condition + deque for zero-polling iteration.

    Subclasses must set ``_ref_ids`` and provide ``stop()`` (calling ``_signal_stop()``).
    Producer threads call ``_dispatch()``, ``_signal_stop()``, ``_signal_error()``.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._buf: deque[Reading] = deque()
        self._maxsize = _DEFAULT_BUFFER_MAXSIZE
        self._stopped = False
        self._exc: Optional[Exception] = None
        self._ref_ids: list[int] = []
        self._drop_count = 0
        self._last_drop_log = 0.0

    # -- Properties -----------------------------------------------------------

    @property
    def ref_ids(self) -> list[int]:
        return list(self._ref_ids)

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def exc(self) -> Optional[Exception]:
        return self._exc

    # -- Producer API (called from reactor / IO thread) -----------------------

    def _dispatch(self, reading: Reading) -> None:
        """Enqueue a reading, waking any blocked reader."""
        if self._stopped:
            return
        with self._cond:
            if self._stopped:
                return
            if len(self._buf) >= self._maxsize:
                self._drop_count += 1
                now = time.monotonic()
                if now - self._last_drop_log >= 5.0:
                    logger.warning(
                        "Subscription buffer full (%d), dropped %d readings",
                        self._maxsize,
                        self._drop_count,
                    )
                    self._drop_count = 0
                    self._last_drop_log = now
                return
            self._buf.append(reading)
            self._cond.notify()

    def _signal_stop(self) -> None:
        """Idempotent stop - wake all waiters."""
        if self._stopped:
            return
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def _signal_error(self, exc: Exception) -> None:
        """Idempotent error - first error wins, then stop."""
        with self._cond:
            if self._exc is None:
                self._exc = exc
            if not self._stopped:
                self._stopped = True
            self._cond.notify_all()

    # -- Consumer API (called from user thread) -------------------------------

    def readings(
        self,
        timeout: Optional[float] = None,
    ) -> Iterator[tuple[Reading, SubscriptionHandle]]:
        if getattr(self, "_is_callback_mode", False):
            raise RuntimeError("Cannot iterate subscription with callback; readings are pushed to callback")
        start = time.monotonic()

        while True:
            reading = None
            with self._cond:
                # Try to pop a buffered reading
                if self._buf:
                    reading = self._buf.popleft()
                else:
                    # No data - check terminal conditions
                    if self._exc is not None:
                        raise self._exc
                    if self._stopped:
                        return

                    # Compute wait time
                    if timeout == 0:
                        return
                    elif timeout is not None:
                        remaining = timeout - (time.monotonic() - start)
                        if remaining <= 0:
                            return
                        wait = remaining
                    else:
                        wait = None

                    self._cond.wait(timeout=wait)

                    # Re-check after wakeup
                    if self._buf:
                        reading = self._buf.popleft()
                    elif self._exc is not None:
                        raise self._exc
                    elif self._stopped:
                        return
                    elif timeout == 0:
                        return
                    elif timeout is not None and time.monotonic() - start >= timeout:
                        return
                    else:
                        continue  # spurious wakeup, loop again

            # Yield outside the lock
            if reading is not None:
                yield (reading, self)
