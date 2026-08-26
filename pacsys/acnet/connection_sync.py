"""
Synchronous ACNET connections - thin wrappers around async connection classes.

Manages a dedicated asyncio reactor thread and delegates all operations
to the async core via run_coroutine_threadsafe.

_SyncAcnetConnectionBase holds all reactor/delegation boilerplate.
Subclasses just provide a factory for the specific async connection type.
"""

import asyncio
import logging
import threading

from .async_connection import (
    ACSYS_PROXY_HOST,
    AsyncAcnetConnectionBase,
    AsyncAcnetConnectionTCP,
    AsyncAcnetConnectionUDP,
    AsyncRequestContext,
    CancelHandler,
    MessageHandler,
    NodeStats,
    ReplyHandler,
    RequestHandler,
    _encode_connection_name,
)
from .constants import ACNET_CLIENT_PORT, ACNET_TCP_PORT, DEFAULT_TIMEOUT
from .packet import AcnetRequest, RequestId

logger = logging.getLogger(__name__)

__all__ = ["AcnetConnectionTCP", "AcnetConnectionUDP", "AcnetRequestContext", "NodeStats", "ACSYS_PROXY_HOST"]


class AcnetRequestContext:
    """Sync wrapper around AsyncRequestContext."""

    def __init__(self, async_ctx: AsyncRequestContext, loop: asyncio.AbstractEventLoop):
        self._async_ctx = async_ctx
        self._loop = loop

    def cancel(self):
        """Cancel this request (fire-and-forget, safe from any thread)."""
        if not self._async_ctx._cancelled:
            asyncio.run_coroutine_threadsafe(self._async_ctx.cancel(), self._loop)

    @property
    def request_id(self) -> RequestId:
        return self._async_ctx.request_id

    @property
    def cancelled(self) -> bool:
        return self._async_ctx.cancelled

    @property
    def multiple_reply(self) -> bool:
        return self._async_ctx.multiple_reply

    @property
    def timeout(self) -> int:
        return self._async_ctx.timeout

    @property
    def task(self) -> str:
        return self._async_ctx.task

    @property
    def node(self) -> int:
        return self._async_ctx.node

    @property
    def reply_handler(self) -> ReplyHandler:
        return self._async_ctx.reply_handler


class _SyncAcnetConnectionBase:
    """
    Base class for synchronous ACNET connections.

    Thin wrapper that delegates to an AsyncAcnetConnectionBase subclass
    running on a dedicated reactor thread. All public methods block until
    the async operation completes.
    """

    # Expose class-level dicts for tracing compatibility
    _CMD_NAMES = AsyncAcnetConnectionBase._CMD_NAMES
    _MSG_TYPE_NAMES = AsyncAcnetConnectionBase._MSG_TYPE_NAMES

    def __init__(
        self,
        host: str = ACSYS_PROXY_HOST,
        port: int = ACNET_TCP_PORT,
        name: str = "",
        *,
        vnode: str = "",
        trace: bool = False,
    ):
        self._host = host
        self._port = port
        self._requested_name = name
        self._vnode_name = vnode
        self._trace = trace
        _encode_connection_name(name, "Task name", allow_empty=True)
        _encode_connection_name(vnode, "Vnode name", allow_empty=True)

        self._async: AsyncAcnetConnectionBase | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reactor_thread: threading.Thread | None = None

    def _create_async(self) -> AsyncAcnetConnectionBase:
        """Factory method - subclasses return the appropriate async connection type."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Properties - proxy to async core
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._async.name if self._async else ""

    @property
    def handle(self) -> str:
        return self._async.handle if self._async else ""

    @property
    def raw_handle(self) -> int:
        return self._async.raw_handle if self._async else 0

    @property
    def connected(self) -> bool:
        return self._async.connected if self._async else False

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    # ------------------------------------------------------------------
    # Reactor management
    # ------------------------------------------------------------------

    def _start_reactor(self):
        """Start the asyncio reactor thread."""
        ready = threading.Event()
        loop_holder: list[asyncio.AbstractEventLoop] = []

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_holder.append(loop)
            ready.set()
            loop.run_forever()
            # Cleanup after stop
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        self._reactor_thread = threading.Thread(
            target=_run,
            name="ACNET-Reactor",
            daemon=True,
        )
        self._reactor_thread.start()
        ready.wait(timeout=5.0)

        if not loop_holder:
            raise RuntimeError("Failed to start ACNET reactor")

        self._loop = loop_holder[0]

    @property
    def _core(self) -> AsyncAcnetConnectionBase:
        """Return the async core, asserting it is initialized."""
        assert self._async is not None, "not connected"
        return self._async

    def _run_sync(self, coro, timeout=10.0):
        """Schedule a coroutine on the reactor and block for its result."""
        assert self._loop is not None, "reactor not started"
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Connect to the configured ACNET daemon transport.

        A failed connect stops the reactor and clears state; the object can
        be re-connected (fresh reactor and core).
        """
        try:
            self._start_reactor()
            self._async = self._create_async()
            self._run_sync(self._core.connect())
        except BaseException:
            self.close()
            raise

    def close(self):
        """Close the connection and clean up resources."""
        if self._async and self._loop:
            try:
                self._run_sync(self._core.close(), timeout=5.0)
            except Exception:  # noqa: BLE001
                logger.debug("Async ACNET close failed", exc_info=True)

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._reactor_thread and self._reactor_thread.is_alive():
            self._reactor_thread.join(timeout=2.0)

        self._async = None
        self._loop = None
        self._reactor_thread = None

    # ------------------------------------------------------------------
    # Public commands - delegate to async core
    # ------------------------------------------------------------------

    def send_request(
        self,
        node: int,
        task: str,
        data: bytes,
        reply_handler: ReplyHandler,
        multiple_reply: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> "AcnetRequestContext":
        assert self._loop is not None, "reactor not started"
        async_ctx = self._run_sync(self._core.send_request(node, task, data, reply_handler, multiple_reply, timeout))
        return AcnetRequestContext(async_ctx, self._loop)

    def request_single(
        self,
        node: int,
        task: str,
        data: bytes,
        reply_handler: ReplyHandler,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> "AcnetRequestContext":
        return self.send_request(node, task, data, reply_handler, multiple_reply=False, timeout=timeout)

    def request_multiple(
        self,
        node: int,
        task: str,
        data: bytes,
        reply_handler: ReplyHandler,
        timeout: int = 0,
    ) -> "AcnetRequestContext":
        return self.send_request(node, task, data, reply_handler, multiple_reply=True, timeout=timeout)

    def get_node(self, name: str) -> int:
        return self._run_sync(self._core.get_node(name))

    def get_name(self, node: int) -> str:
        return self._run_sync(self._core.get_name(node))

    def get_local_node(self) -> int:
        return self._run_sync(self._core.get_local_node())

    def get_default_node(self) -> int:
        return self._run_sync(self._core.get_default_node())

    def rename_task(self, new_name: str):
        self._run_sync(self._core.rename_task(new_name))

    def send_message(self, node: int, task: str, data: bytes):
        self._run_sync(self._core.send_message(node, task, data))

    def ignore_request(self, request: AcnetRequest):
        self._run_sync(self._core.ignore_request(request))

    def get_node_stats(self) -> NodeStats:
        return self._run_sync(self._core.get_node_stats())

    def get_task_pid(self, task: str) -> int:
        return self._run_sync(self._core.get_task_pid(task))

    def disconnect_single(self):
        self._run_sync(self._core.disconnect_single())

    def handle_messages(self, handler: MessageHandler):
        self._run_sync(self._core.handle_messages(handler))

    def handle_requests(self, handler: RequestHandler):
        self._run_sync(self._core.handle_requests(handler))

    def handle_cancels(self, handler: CancelHandler):
        self._core.handle_cancels(handler)

    def send_reply(self, request: AcnetRequest, data: bytes, status: int, last: bool = True):
        self._run_sync(self._core.send_reply(request, data, status, last))

    def _send_keepalive(self):
        """Send keepalive - exposed for tests."""
        self._run_sync(self._core._send_keepalive())

    # ------------------------------------------------------------------
    # Internal - exposed for test mocking compatibility
    # ------------------------------------------------------------------

    def _xact(self, content: bytes) -> bytes:
        """Send command and wait for ACK - sync wrapper for test mocking."""
        return self._run_sync(self._core._xact(content))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class AcnetConnectionTCP(_SyncAcnetConnectionBase):
    """Synchronous ACNET connection over TCP."""

    def _create_async(self) -> AsyncAcnetConnectionTCP:
        return AsyncAcnetConnectionTCP(
            host=self._host,
            port=self._port,
            name=self._requested_name,
            vnode=self._vnode_name,
            trace=self._trace,
        )


class AcnetConnectionUDP(_SyncAcnetConnectionBase):
    """Synchronous connection to the official local acnetd UDP interface."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = ACNET_CLIENT_PORT,
        name: str = "",
        *,
        vnode: str = "",
        trace: bool = False,
    ):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("acnetd's official UDP client interface is local-only; use host='127.0.0.1'")
        super().__init__(host, port, name, vnode=vnode, trace=trace)

    def _create_async(self) -> AsyncAcnetConnectionUDP:
        return AsyncAcnetConnectionUDP(
            host=self._host,
            port=self._port,
            name=self._requested_name,
            vnode=self._vnode_name,
            trace=self._trace,
        )
