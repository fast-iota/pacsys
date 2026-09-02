"""Async subscription handle using asyncio.Queue."""

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from pacsys.backends import summarize_drfs
from pacsys.types import Reading

logger = logging.getLogger(__name__)

_DEFAULT_BUFFER_MAXSIZE = 10_000


class AsyncSubscriptionHandle:
    """Async counterpart of BufferedSubscriptionHandle.

    Uses asyncio.Queue for zero-polling async iteration.
    Producer calls _dispatch() (sync, non-blocking).
    Consumer uses async for reading, handle in handle.readings().
    """

    def __init__(self, remover=None) -> None:
        self._maxsize = _DEFAULT_BUFFER_MAXSIZE
        self._queue: asyncio.Queue[Reading | None] = asyncio.Queue(maxsize=self._maxsize)
        self._stopped = False
        self._stopping = False
        self._stop_complete = asyncio.Event()
        self._stop_task: asyncio.Task | None = None
        self._exc: Exception | None = None
        self._task: asyncio.Task | None = None
        self._callback_task: asyncio.Task | None = None
        self._aux_tasks: set[asyncio.Task] = set()  # owned notifications (on_error), reaped by stop()
        self._drop_count = 0  # cumulative, never reset
        self._drops_since_log = 0
        self._last_drop_log = 0.0
        self._core: Any = None
        self._drfs: list[str] = []
        self._remover = remover

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def exc(self) -> Exception | None:
        return self._exc

    @property
    def dropped(self) -> int:
        return self._drop_count

    # -- Producer API (called from core's dispatch_fn) -------------------------

    def _dispatch(self, reading: Reading) -> None:
        if self._stopped:
            return
        try:
            self._queue.put_nowait(reading)
        except asyncio.QueueFull:
            self._drop_count += 1
            self._drops_since_log += 1
            now = time.monotonic()
            if now - self._last_drop_log >= 5.0:
                logger.warning(
                    "Async subscription buffer full (%d), dropped %d readings (devices: %s)",
                    self._maxsize,
                    self._drops_since_log,
                    summarize_drfs(self._drfs),
                )
                self._last_drop_log = now
                self._drops_since_log = 0

    def _signal_stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass  # consumer will see _stopped flag

    def _signal_error(self, exc: Exception) -> None:
        if self._exc is None:
            self._exc = exc
        self._signal_stop()

    def _is_stopped(self) -> bool:
        return self._stopped

    def _spawn(self, coro) -> None:
        """Run an auxiliary coroutine owned by this handle (kept referenced; awaited by stop())."""
        task = asyncio.ensure_future(coro)
        self._aux_tasks.add(task)
        task.add_done_callback(self._aux_tasks.discard)

    # -- Consumer API ----------------------------------------------------------

    async def readings(self, timeout: float | None = None) -> AsyncIterator[tuple[Reading, "AsyncSubscriptionHandle"]]:
        """Yield buffered readings within one total wall-clock window."""
        if self._callback_task is not None:
            raise RuntimeError("Cannot iterate subscription with callback; readings are pushed to callback")

        async for item in self._readings(timeout):
            yield item

    async def _readings(self, timeout: float | None = None) -> AsyncIterator[tuple[Reading, "AsyncSubscriptionHandle"]]:
        if timeout == 0:
            for _ in range(self._queue.qsize()):
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    if self._exc is not None:
                        raise self._exc
                    return
                yield (item, self)
            if self._exc is not None:
                raise self._exc
            return

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                if self._exc is not None:
                    raise self._exc
                return
            if self._stopped and self._queue.empty():
                if self._exc is not None:
                    raise self._exc
                return
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    if self._exc is not None:
                        raise self._exc from None
                    return
            if item is None:
                if self._exc is not None:
                    raise self._exc
                return
            yield (item, self)

    async def stop(self) -> None:
        cur = asyncio.current_task()
        if self._stopping:
            # Reentrant call from the stopping task itself (stop -> _remover ->
            # backend.remove -> handle.stop) must not wait on its own
            # completion; genuinely concurrent callers wait until tasks are
            # fully unwound (close() drains the pool right after).
            if self._stop_task is cur:
                return
            await self._stop_complete.wait()
            return
        self._stopping = True
        self._stop_task = cur
        try:
            self._signal_stop()
            if self._remover is not None:
                await self._remover(self)
            # Never cancel/await the current task (callback calling stop());
            # it ends naturally via the stop sentinel.
            owned = [
                (t, what)
                for t, what in ((self._task, "task"), (self._callback_task, "callback task"))
                if t is not None and t is not cur and not t.done()
            ]
            # Cancel both before awaiting either: a stop() cancelled mid-reap still leaves nothing running
            for t, _ in owned:
                t.cancel()
            for t, what in owned:
                await _reap(t, what)
            # on_error notifications are always delivered: awaited, not cancelled
            for t in list(self._aux_tasks):
                if t is not cur:
                    await _reap(t, "on_error task")
        finally:
            # A cancelled stop() releases waiters without a completion
            # guarantee -- acceptable while no caller wraps stop in a timeout.
            self._stop_complete.set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.stop()
        return False


async def _reap(task: asyncio.Task, what: str) -> None:
    """Wait for a cancelled child task to unwind, logging any failure.

    asyncio.wait never re-raises the child's CancelledError, so only an external
    cancellation of the awaiting task itself propagates from here.
    """
    await asyncio.wait({task})
    if not task.cancelled() and task.exception() is not None:
        logger.error("Subscription %s failed during shutdown", what, exc_info=task.exception())


async def _call_on_error(on_error, exc: Exception, handle: AsyncSubscriptionHandle) -> None:
    """Invoke a sync or async on_error callback; its own failures are logged, never raised."""
    try:
        if inspect.iscoroutinefunction(on_error):
            await on_error(exc, handle)
        else:
            on_error(exc, handle)
    except Exception as err_exc:  # noqa: BLE001
        logger.error("Error in on_error callback: %s", err_exc)


async def _callback_feeder(handle: AsyncSubscriptionHandle, callback, on_error) -> None:
    """Feed readings from handle to callback (async or sync)."""
    is_async_cb = inspect.iscoroutinefunction(callback)
    stream_error_delivered = False

    async def deliver_stream_error(exc: Exception) -> None:
        nonlocal stream_error_delivered
        if stream_error_delivered:
            return
        stream_error_delivered = True
        if on_error:
            await _call_on_error(on_error, exc, handle)
        else:
            logger.error("Unhandled error in stream: %s", exc)

    try:
        async for reading, h in handle._readings():
            if handle._stopping:  # set only by stop(): a stream that ends on its own still delivers its tail
                continue  # queued before stop(); keep draining so a pending error is still raised below
            try:
                if is_async_cb:
                    await callback(reading, h)
                else:
                    callback(reading, h)
            except Exception as exc:  # noqa: BLE001
                if on_error:
                    await _call_on_error(on_error, exc, h)
                else:
                    logger.error("Unhandled error in subscription callback: %s", exc)
    except asyncio.CancelledError:
        if handle._exc is not None:
            await deliver_stream_error(handle._exc)
    except Exception as exc:  # noqa: BLE001
        await deliver_stream_error(exc)
