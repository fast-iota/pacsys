"""Callback dispatch for streaming backends.

CallbackDispatcher decouples reactor threads from user callbacks.
WORKER mode (default) delivers on a dedicated daemon thread.
DIRECT mode delivers inline with a slow-callback warning.
"""

import logging
import queue
import threading
import time

from pacsys.types import DispatchMode, ErrorCallback, Reading, ReadingCallback, SubscriptionHandle

logger = logging.getLogger(__name__)

_SLOW_THRESHOLD = 0.050  # 50ms
_WARN_INTERVAL = 10.0  # rate-limit slow-callback warnings
_QUEUE_MAX_SIZE = 10_000  # bounded queue to prevent OOM from slow callbacks


class CallbackDispatcher:
    """Dispatch streaming callbacks in WORKER or DIRECT mode.

    WORKER: lazy-starts a single daemon thread with a bounded Queue.
    DIRECT: calls inline, warns if callback exceeds 50ms.
    """

    def __init__(self, mode: DispatchMode = DispatchMode.WORKER):
        self._mode = mode
        # WORKER state (lazy-initialized)
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue | None = None
        self._stop = threading.Event()
        self._started = False
        self._lock = threading.Lock()
        # DIRECT rate-limiting
        self._last_warn_time = 0.0
        # Drop warning rate-limiting (WORKER mode)
        self._last_drop_warn_time = 0.0

    @property
    def mode(self) -> DispatchMode:
        return self._mode

    def dispatch_reading(
        self,
        callback: ReadingCallback,
        reading: Reading,
        handle: SubscriptionHandle,
    ) -> None:
        """Dispatch a reading callback."""
        if self._stop.is_set():
            return
        if self._mode is DispatchMode.WORKER:
            self._ensure_worker()
            assert self._queue is not None
            self._enqueue((callback, reading, handle, False))
        else:
            self._call_direct(callback, reading, handle)

    def dispatch_error(
        self,
        on_error: ErrorCallback,
        exc: Exception,
        handle: SubscriptionHandle,
    ) -> None:
        """Dispatch an error callback."""
        if self._stop.is_set():
            return
        if self._mode is DispatchMode.WORKER:
            self._ensure_worker()
            assert self._queue is not None
            self._enqueue((on_error, exc, handle, True))
        else:
            self._call_direct_error(on_error, exc, handle)

    def close(self) -> None:
        """Stop worker thread if running."""
        self._stop.set()
        if self._queue is not None:
            # Sentinel must bypass the maxsize bound
            # Since stop event is set, one of puts will always succeed
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                # Queue is full - drain one item to make room for sentinel
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass
        # Snapshot guards concurrent close(); never join the worker from itself
        # (close() from a callback runs on the worker thread -- _stop is set, so
        # the loop exits right after the callback returns; thread is a daemon).
        t = self._thread
        if t is not None and t is not threading.current_thread():
            # If thread doesn't shutdown, ignore - it is a daemon
            t.join(timeout=2.0)
            # Keep _thread on the self-join path so a later cross-thread
            # close() still joins the worker.
            self._thread = None

    # ── internal ──

    def _enqueue(self, item: tuple) -> None:
        """Put item on queue, dropping with a warning if full."""
        assert self._queue is not None
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            now = time.monotonic()
            if now - self._last_drop_warn_time >= _WARN_INTERVAL:
                logger.warning(
                    "Dispatch queue full (%d items), dropping callback - callback is too slow for the data rate",
                    _QUEUE_MAX_SIZE,
                )
                self._last_drop_warn_time = now

    def _ensure_worker(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._queue = queue.Queue(maxsize=_QUEUE_MAX_SIZE)
            t = threading.Thread(target=self._worker_loop, daemon=True, name="pacsys-dispatch")
            t.start()
            self._thread = t
            self._started = True

    def _worker_loop(self) -> None:
        assert self._queue is not None
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:  # sentinel
                break
            fn, arg, handle, is_error = item
            try:
                fn(arg, handle)
            except Exception:
                kind = "on_error" if is_error else "reading"
                logger.exception("Error in %s callback", kind)

    def _call_direct(
        self,
        callback: ReadingCallback,
        reading: Reading,
        handle: SubscriptionHandle,
    ) -> None:
        t0 = time.monotonic()
        try:
            callback(reading, handle)
        except Exception:
            logger.exception("Error in reading callback")
            return
        elapsed = time.monotonic() - t0
        if elapsed > _SLOW_THRESHOLD:
            self._maybe_warn(elapsed)

    def _call_direct_error(
        self,
        on_error: ErrorCallback,
        exc: Exception,
        handle: SubscriptionHandle,
    ) -> None:
        t0 = time.monotonic()
        try:
            on_error(exc, handle)
        except Exception:
            logger.exception("Error in on_error callback")
            return
        elapsed = time.monotonic() - t0
        if elapsed > _SLOW_THRESHOLD:
            self._maybe_warn(elapsed)

    def _maybe_warn(self, elapsed: float) -> None:
        now = time.monotonic()
        if now - self._last_warn_time >= _WARN_INTERVAL:
            logger.warning(
                "Slow callback in DIRECT mode: %.0fms (threshold: %.0fms)", elapsed * 1000, _SLOW_THRESHOLD * 1000
            )
            self._last_warn_time = now
