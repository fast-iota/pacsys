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
        self._drop_count = 0
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
                drf_summary = summarize_drfs(self._drfs)
                logger.warning(
                    "Async subscription buffer full (%d), dropped %d readings (devices: %s)",
                    self._maxsize,
                    self._drop_count,
                    drf_summary,
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

    async def readings(self, timeout: float | None = None) -> AsyncIterator[tuple[Reading, "AsyncSubscriptionHandle"]]:
        """Yield buffered readings within one total wall-clock window."""
        if self._callback_task is not None:
            raise RuntimeError("Cannot iterate subscription with callback; readings are pushed to callback")

        async for item in self._readings(timeout):
            yield item

    async def _readings(self, timeout: float | None = None) -> AsyncIterator[tuple[Reading, "AsyncSubscriptionHandle"]]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._queue.empty():
                if self._stopped:
                    if self._exc is not None:
                        raise self._exc
                    return
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    if self._exc is not None:
                        raise self._exc from None
                    return
            else:
                item = self._queue.get_nowait()
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
            if self._task is not None and self._task is not cur and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001
                    logger.exception("Subscription task failed during shutdown")
            if self._callback_task is not None and self._callback_task is not cur and not self._callback_task.done():
                self._callback_task.cancel()
                try:
                    await self._callback_task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001
                    logger.exception("Subscription callback task failed during shutdown")
        finally:
            # A cancelled stop() releases waiters without a completion
            # guarantee -- acceptable while no caller wraps stop in a timeout.
            self._stop_complete.set()

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
        async for reading, h in handle._readings():
            try:
                if is_async_cb:
                    await callback(reading, h)
                else:
                    callback(reading, h)
            except Exception as exc:  # noqa: BLE001
                try:
                    if on_error:
                        if is_async_err:
                            await on_error(exc, h)
                        else:
                            on_error(exc, h)
                    else:
                        logger.error("Unhandled error in subscription callback: %s", exc)
                except Exception as err_exc:  # noqa: BLE001
                    logger.error("Error in on_error callback: %s", err_exc)
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001
        if on_error:
            try:
                if is_async_err:
                    await on_error(exc, handle)
                else:
                    on_error(exc, handle)
            except Exception as err_exc:  # noqa: BLE001
                logger.error("Error in on_error callback: %s", err_exc)
        else:
            logger.error("Unhandled error in stream: %s", exc)
