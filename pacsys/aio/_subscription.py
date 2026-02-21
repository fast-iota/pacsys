"""Async subscription handle using asyncio.Queue."""

import asyncio
import inspect
import logging
import time
from typing import Any, AsyncIterator, Optional

from pacsys.types import Reading

logger = logging.getLogger(__name__)

_DEFAULT_BUFFER_MAXSIZE = 10_000


class AsyncSubscriptionHandle:
    """Async counterpart of BufferedSubscriptionHandle.

    Uses asyncio.Queue for zero-polling async iteration.
    Producer calls _dispatch() (sync, non-blocking).
    Consumer uses async for reading, handle in handle.readings().
    """

    def __init__(self) -> None:
        self._maxsize = _DEFAULT_BUFFER_MAXSIZE
        self._queue: asyncio.Queue[Reading | None] = asyncio.Queue(maxsize=self._maxsize)
        self._stopped = False
        self._exc: Optional[Exception] = None
        self._task: Optional[asyncio.Task] = None
        self._callback_task: Optional[asyncio.Task] = None
        self._drop_count = 0
        self._last_drop_log = 0.0
        self._core: Any = None

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def exc(self) -> Optional[Exception]:
        return self._exc

    # -- Producer API (called from core's dispatch_fn) -------------------------

    def _dispatch(self, reading: Reading) -> None:
        if self._stopped:
            return
        try:
            self._queue.put_nowait(reading)
        except asyncio.QueueFull:
            self._drop_count += 1
            now = time.monotonic()
            if now - self._last_drop_log >= 5.0:
                logger.warning(
                    "Async subscription buffer full (%d), dropped %d readings",
                    self._maxsize,
                    self._drop_count,
                )
                self._drop_count = 0
                self._last_drop_log = now

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

    # -- Consumer API ----------------------------------------------------------

    async def readings(
        self, timeout: Optional[float] = None
    ) -> AsyncIterator[tuple[Reading, "AsyncSubscriptionHandle"]]:
        while True:
            # Stop sentinel may not have been enqueued if queue was full
            if self._stopped and self._queue.empty():
                if self._exc is not None:
                    raise self._exc
                return
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if self._stopped:
                    if self._exc is not None:
                        raise self._exc
                    return
                raise
            if item is None:
                if self._exc is not None:
                    raise self._exc
                return
            yield (item, self)

    async def stop(self) -> None:
        self._signal_stop()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._callback_task is not None and not self._callback_task.done():
            self._callback_task.cancel()
            try:
                await self._callback_task
            except (asyncio.CancelledError, Exception):
                pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.stop()
        return False


async def _callback_feeder(handle: AsyncSubscriptionHandle, callback, on_error) -> None:
    """Feed readings from handle to callback (async or sync)."""
    is_async_cb = inspect.iscoroutinefunction(callback)
    is_async_err = inspect.iscoroutinefunction(on_error) if on_error else False

    try:
        async for reading, h in handle.readings():
            try:
                if is_async_cb:
                    await callback(reading, h)
                else:
                    callback(reading, h)
            except Exception as exc:
                try:
                    if on_error:
                        if is_async_err:
                            await on_error(exc, h)
                        else:
                            on_error(exc, h)
                    else:
                        logger.error("Unhandled error in subscription callback: %s", exc)
                except Exception as err_exc:
                    logger.error("Error in on_error callback: %s", err_exc)
    except asyncio.CancelledError:
        pass
