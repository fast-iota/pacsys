"""
Async ACNET connection - pure asyncio implementation with pluggable transport.

AsyncAcnetConnectionBase holds all protocol logic (commands, dispatch, tracking).
Subclasses provide transport-specific framing:
- AsyncAcnetConnectionTCP: TCP stream with 4-byte length prefix + handshake
- AsyncAcnetConnectionUDP: local acnetd command/data datagrams

The sync wrappers (AcnetConnectionTCP / AcnetConnectionUDP) schedule calls via
run_coroutine_threadsafe.

Example:
    async with AsyncAcnetConnectionTCP("acsys-proxy.fnal.gov") as conn:
        await conn.send_request(node, "DPM", data, reply_handler)
"""

import asyncio
import logging
import os
import socket
import struct
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from .constants import (
    ACNET_CLIENT_PORT,
    ACNET_FLG_RPY,
    ACNET_TCP_PORT,
    CMD_BLOCK_REQUESTS,
    CMD_CANCEL,
    CMD_CONNECT,
    CMD_DEFAULT_NODE,
    CMD_DISCONNECT,
    CMD_DISCONNECT_SINGLE,
    CMD_IGNORE_REQUEST,
    CMD_KEEPALIVE,
    CMD_LOCAL_NODE,
    CMD_NAME_LOOKUP,
    CMD_NODE_LOOKUP,
    CMD_NODE_STATS,
    CMD_RECEIVE_REQUESTS,
    CMD_RENAME_TASK,
    CMD_REQUEST_ACK,
    CMD_SEND,
    CMD_SEND_REPLY,
    CMD_SEND_REQUEST_TIMEOUT,
    CMD_TASK_PID,
    DEFAULT_TIMEOUT,
    INFINITE_TIMEOUT,
    RECV_BUFFER_SIZE,
    REPLY_ENDMULT,
    REPLY_NORMAL,
    SEND_BUFFER_SIZE,
)
from .errors import (
    ACNET_DISCONNECTED,
    ACNET_NOT_CONNECTED,
    ACNET_REQREJ,
    AcnetError,
    AcnetRequestRejectedError,
    AcnetUnavailableError,
)
from .packet import (
    AcnetCancel,
    AcnetMessage,
    AcnetPacket,
    AcnetReply,
    AcnetRequest,
    ReplyId,
    RequestId,
)
from .rad50 import decode_stripped as _rad50_decode
from .rad50 import encode as _rad50_encode

logger = logging.getLogger(__name__)

# Default proxy host
ACSYS_PROXY_HOST = "acsys-proxy.fnal.gov"

# TCP bridge message types. Raw UDP has no message-type envelope.
TCP_CLIENT_PING = 0
ACNETD_COMMAND = 1
ACNETD_ACK = 2
ACNETD_DATA = 3

# Handshake string (TCP only)
TCP_HANDSHAKE = b"RAW\r\n\r\n"

# Keepalive interval (seconds)
KEEPALIVE_INTERVAL = 30

# Max buffered replies per request ID before treating as orphaned.
# Legitimate buffering is 1-2 replies (ACK+reply in same TCP batch).
_MAX_BUFFERED_REPLIES = 16


@dataclass(frozen=True)
class NodeStats:
    """ACNET node statistics from acnetd."""

    usm_received: int
    requests_received: int
    replies_received: int
    usm_sent: int
    requests_sent: int
    replies_sent: int
    request_queue_limit: int


# Type aliases for handlers
ReplyHandler = Callable
RequestHandler = Callable
MessageHandler = Callable
CancelHandler = Callable


class AsyncRequestContext:
    """Context for tracking an outgoing request (async version)."""

    def __init__(
        self,
        connection: "AsyncAcnetConnectionBase",
        task: str,
        node: int,
        request_id: RequestId,
        multiple_reply: bool,
        timeout: int,
        reply_handler: ReplyHandler,
    ):
        self.connection = connection
        self.task = task
        self.node = node
        self.request_id = request_id
        self.multiple_reply = multiple_reply
        self.timeout = timeout
        self.reply_handler = reply_handler
        self._cancelled = False

    async def cancel(self):
        """Cancel this request."""
        if not self._cancelled:
            self._cancelled = True
            await self.connection._send_cancel(self)

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class AsyncAcnetConnectionBase:
    """
    Base class for async ACNET connections.

    Contains all protocol logic (commands, dispatch, tracking).
    Subclasses implement transport-specific framing via abstract methods:
    - _open_transport()
    - _close_transport()
    - _send_frame(content)
    - _start_read_loop()
    """

    # True when the transport guarantees acnetd's SEND_REQUEST ack is observed
    # before any reply for the new request (TCP stream ordering). UDP has no
    # such guarantee: acnetd sends acks and data on separate sockets.
    _ORDERED_TRANSPORT: ClassVar[bool] = True

    # Command code → name for tracing
    _CMD_NAMES: ClassVar[dict[int, str]] = {
        0: "KEEPALIVE",
        1: "CONNECT",
        2: "RENAME",
        3: "DISCONNECT",
        4: "SEND",
        5: "SEND_REQ",
        6: "RECV_REQ",
        7: "SEND_REPLY",
        8: "CANCEL",
        9: "REQ_ACK",
        11: "NAME_LOOKUP",
        12: "NODE_LOOKUP",
        13: "LOCAL_NODE",
        14: "TASK_PID",
        15: "NODE_STATS",
        17: "DISCONNECT1",
        18: "SEND_REQ_TMO",
        19: "IGNORE_REQ",
        20: "BLOCK_REQ",
        22: "DEFAULT_NODE",
    }
    _MSG_TYPE_NAMES: ClassVar[dict[int, str]] = {0: "PING", 1: "CMD", 2: "ACK", 3: "DATA"}

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

        # Handle assigned by daemon
        self._raw_handle = _rad50_encode(name) if name else 0
        self._raw_vnode = _rad50_encode(vnode) if vnode else 0
        self._handle_name = ""

        # Command serialization - one command at a time
        self._cmd_lock = asyncio.Lock()
        # ACK delivery - only one pending at a time (under _cmd_lock)
        self._pending_ack: asyncio.Future | None = None

        # State
        self._connected = False
        self._receiving = False
        self._disposed = False

        # Outgoing request tracking
        self._reply_handlers: dict[RequestId, AsyncRequestContext] = {}
        # Buffered replies for requests not yet registered, keyed by (possibly
        # reused) request ID. Stores (reply, recv_seq) for causality: on TCP,
        # acnetd writes the SEND_REQUEST ack before any reply for the new
        # request, so a buffered reply is fresh iff its recv_seq is after the
        # ack's. On UDP no such ordering exists (_ORDERED_TRANSPORT=False) and
        # all buffered replies are delivered.
        self._reply_buffer: dict[RequestId, deque[tuple]] = defaultdict(lambda: deque(maxlen=_MAX_BUFFERED_REPLIES))
        # Receive sequence: counts reply frames as processed; ACK handler
        # records its position so send_request can discard pre-ack strays.
        self._recv_seq = 0
        self._ack_recv_seq = 0
        # Registration windows: buffering is only meaningful while a
        # send_request is between sending and draining. Outside any window an
        # unknown-ID reply is provably stale.
        self._pending_sends = 0
        # Incoming request tracking
        self._requests_in: dict[ReplyId, AcnetRequest] = {}

        # Handlers
        self._message_handler: MessageHandler | None = None
        self._request_handler: RequestHandler | None = None
        self._cancel_handler: CancelHandler | None = None

        # Tasks
        self._read_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def name(self) -> str:
        return self._handle_name

    @property
    def handle(self) -> str:
        return self._handle_name

    @property
    def raw_handle(self) -> int:
        return self._raw_handle

    @property
    def connected(self) -> bool:
        return self._connected and not self._disposed

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    # ------------------------------------------------------------------
    # Abstract transport methods - subclasses must implement
    # ------------------------------------------------------------------

    async def _open_transport(self):
        """Open the underlying transport (TCP stream / UDP socket)."""
        raise NotImplementedError

    async def _close_transport(self):
        """Close the underlying transport."""
        raise NotImplementedError

    async def _send_frame(self, content: bytes):
        """Send one raw acnetd command using the transport's framing."""
        raise NotImplementedError

    def _start_read_loop(self):
        """Start the transport-specific read loop / callback registration."""
        raise NotImplementedError

    def _connect_pid(self) -> int:
        """PID placed in CONNECT; the TCP bridge replaces zero."""
        return 0

    def _connect_data_port(self) -> int:
        """Data port placed in CONNECT; the TCP bridge replaces zero."""
        return 0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self):
        """Connect to the remote ACNET daemon.

        A failed connect closes the connection (transport, read task) and
        disposes it -- create a new instance to retry.
        """
        if self._disposed:
            raise AcnetError(ACNET_NOT_CONNECTED, "Connection disposed")

        try:
            await self._open_transport()
            self._start_read_loop()
            await self._do_connect()
        except BaseException:
            await self.close()
            raise

        logger.info("Connected to ACNET via %s:%s as %s", self._host, self._port, self._handle_name)

        self._start_keepalive_loop()

    async def close(self):
        """Close the connection and clean up."""
        # Send DISCONNECT before marking disposed
        if self._connected:
            try:
                await self._do_disconnect()
            except Exception:  # noqa: BLE001
                logger.debug("Disconnect failed while closing %s", self._handle_name, exc_info=True)

        self._disposed = True

        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        try:
            await self._close_transport()
        finally:
            if self._pending_ack and not self._pending_ack.done():
                self._pending_ack.set_exception(AcnetUnavailableError())

            # Also needed when the connection already dropped (_connected is
            # False, so _do_disconnect never ran): handlers would otherwise be
            # left waiting forever. Idempotent - the dict is snapshot+cleared.
            self._fail_reply_handlers()
            self._reply_buffer.clear()

        logger.info("Closed async ACNET connection %s", self._handle_name)

    async def _do_connect(self):
        """Send CONNECT command and process response."""
        content = struct.pack(
            ">H3IH",
            CMD_CONNECT,
            self._raw_handle,
            self._raw_vnode,
            self._connect_pid(),
            self._connect_data_port(),
        )
        ack = await self._xact(content)

        if len(ack) < 9:
            raise AcnetUnavailableError

        _ack_code, status, _task_id, handle = struct.unpack(">HhBI", ack[:9])

        if status < 0:
            raise AcnetError(status, f"CONNECT failed with status {status}")

        self._raw_handle = handle
        self._handle_name = _rad50_decode(handle)
        self._connected = True

        logger.debug("Connected with handle %s (%#x)", self._handle_name, handle)

    async def _do_disconnect(self):
        """Send DISCONNECT command."""
        content = struct.pack(">H2I", CMD_DISCONNECT, self._raw_handle, self._raw_vnode)
        try:
            await self._xact(content, timeout=0.5)
        except Exception:  # noqa: BLE001
            logger.debug("Disconnect command failed for %s", self._handle_name, exc_info=True)

        self._fail_reply_handlers()
        self._reply_buffer.clear()

        self._connected = False

    # ------------------------------------------------------------------
    # Connection lost handler (called by read loop / UDP protocol)
    # ------------------------------------------------------------------

    def _on_connection_lost(self):
        self._connected = False

        if self._pending_ack and not self._pending_ack.done():
            self._pending_ack.set_exception(AcnetUnavailableError())

        # Fail all active request contexts so consumers don't hang
        self._fail_reply_handlers()
        self._reply_buffer.clear()

        logger.debug("Connection lost for %s", self._handle_name)

    def _fail_reply_handlers(self) -> None:
        """Cancel all request contexts and deliver a synthetic final
        DISCONNECTED reply to each handler.

        Without this, mult-request consumers (FTP snapshots/streams,
        DPM-over-ACNET) wait forever on queues that will never be fed again.
        """
        contexts = list(self._reply_handlers.values())
        self._reply_handlers.clear()
        for ctx in contexts:
            ctx._cancelled = True
            reply = AcnetReply(ACNET_FLG_RPY, ACNET_DISCONNECTED, 0, 0, 0, 0, ctx.request_id.id, 0, b"")
            try:
                ctx.reply_handler(reply)
            except Exception as e:  # noqa: BLE001
                logger.warning("Reply handler exception (disconnect): %s", e)

    # ------------------------------------------------------------------
    # Command transaction
    # ------------------------------------------------------------------

    async def _xact(self, content: bytes, timeout: float = 5.0) -> bytes:
        """Send a command and wait for ACK. Serialized via _cmd_lock.

        ``content`` starts with the raw acnetd command code. The TCP transport
        adds the bridge message type and length; UDP sends it unchanged.
        """
        if self._disposed:
            raise AcnetError(ACNET_NOT_CONNECTED, "Connection disposed")

        async with self._cmd_lock:
            if self._disposed:
                raise AcnetError(ACNET_NOT_CONNECTED, "Connection disposed")

            loop = asyncio.get_running_loop()
            self._pending_ack = loop.create_future()

            if self._trace:
                cmd = struct.unpack(">H", content[:2])[0]
                cmd_name = self._CMD_NAMES.get(cmd, f"?{cmd}")
                logger.info("TRACE> %s(%s) len=%s %s", cmd_name, cmd, len(content), content[:20].hex())

            try:
                await self._send_frame(content)
            except (OSError, AcnetUnavailableError) as e:
                self._pending_ack = None
                self._connected = False
                self._on_connection_lost()
                await self._close_transport()
                logger.error("Failed to send command: %s", e)
                raise AcnetUnavailableError from e

            try:
                ack_data = await asyncio.wait_for(self._pending_ack, timeout=timeout)
            except asyncio.TimeoutError as e:
                self._pending_ack = None
                # A late ACK could arrive and be consumed by the next
                # command, permanently desynchronizing the stream.
                # Kill the transport so the connection cannot be reused.
                logger.error("Timeout waiting for ack - closing transport to prevent desync")
                self._connected = False
                self._on_connection_lost()
                await self._close_transport()
                raise AcnetUnavailableError from e
            except asyncio.CancelledError:
                self._pending_ack = None
                logger.error("Task cancelled during ack wait - closing transport to prevent desync")
                self._connected = False
                self._on_connection_lost()
                await asyncio.shield(self._close_transport())
                raise

            if self._trace:
                logger.info("TRACE< ACK len=%s %s", len(ack_data), ack_data.hex())

            return ack_data

    # ------------------------------------------------------------------
    # Keepalive loop
    # ------------------------------------------------------------------

    def _start_keepalive_loop(self):
        self._keepalive_task = asyncio.ensure_future(self._keepalive_loop())

    async def _keepalive_loop(self):
        try:
            while not self._disposed:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                try:
                    await self._send_keepalive()
                except AcnetError as e:
                    if not self._disposed:
                        logger.warning("Keepalive failed: %s", e)
        except asyncio.CancelledError:
            pass

    async def _send_keepalive(self):
        content = struct.pack(">H2I", CMD_KEEPALIVE, self._raw_handle, self._raw_vnode)
        await self._xact(content)

    # ------------------------------------------------------------------
    # Frame dispatch (sync - runs in read loop / UDP callback, no awaits)
    # ------------------------------------------------------------------

    def _dispatch_frame(self, msg_type: int, data: bytes):
        if self._trace and msg_type != TCP_CLIENT_PING:
            type_name = self._MSG_TYPE_NAMES.get(msg_type, f"?{msg_type}")
            logger.info("TRACE  RECV %s len=%s %s", type_name, len(data), data[:20].hex())

        if msg_type == TCP_CLIENT_PING:
            pass
        elif msg_type == ACNETD_ACK:
            if self._pending_ack and not self._pending_ack.done():
                self._ack_recv_seq = self._recv_seq
                self._pending_ack.set_result(data)
            else:
                logger.debug("Dropping unexpected ACK (no pending command)")
        elif msg_type == ACNETD_DATA:
            self._dispatch_data(data)
        else:
            logger.warning("Unknown message type: %s", msg_type)

    def _dispatch_data(self, data: bytes) -> None:
        """Parse and dispatch every ACNET packet in a data frame/datagram."""
        offset = 0
        while offset < len(data):
            remaining = len(data) - offset
            if remaining < 18:
                logger.warning("Trailing short ACNET packet: %d bytes", remaining)
                return
            length = struct.unpack_from("<H", data, offset + 16)[0]
            if length < 18 or length > remaining:
                logger.warning("Invalid ACNET packet length %d in %d-byte payload", length, remaining)
                return
            try:
                packet = AcnetPacket.parse(data[offset : offset + length])
            except (ValueError, struct.error) as e:
                logger.warning("Error parsing ACNET packet: %s", e)
                return
            self._handle_packet(packet)
            offset += (length + 1) & ~1

    def _handle_packet(self, packet: AcnetPacket):
        try:
            if isinstance(packet, AcnetReply):
                self._handle_reply(packet)
            elif isinstance(packet, AcnetRequest):
                self._handle_request(packet)
            elif isinstance(packet, AcnetMessage):
                self._handle_message(packet)
            elif isinstance(packet, AcnetCancel):
                self._handle_cancel(packet)
        except Exception:
            logger.exception("Error handling packet")

    def _handle_reply(self, reply: AcnetReply):
        """Handle an incoming reply.

        If the handler is not registered yet, buffer the reply: acnetd reuses
        request IDs, so a finished ID may already belong to a new request.
        send_request() drains the buffer and, on ordered transports, discards
        entries received before the request ACK. UDP has no cross-socket
        ordering, so it delivers all buffered replies. The event loop owns all
        of this state, so no locks are needed.
        """
        self._recv_seq += 1
        context = self._reply_handlers.get(reply.request_id)

        if context:
            try:
                context.reply_handler(reply)
            except Exception as e:  # noqa: BLE001
                logger.warning("Reply handler exception: %s", e)

            if reply.last:
                self._reply_handlers.pop(reply.request_id, None)
                context._cancelled = True
        elif self._pending_sends > 0:
            # deque maxlen evicts oldest: newest replies are the fresh ones
            self._reply_buffer[reply.request_id].append((reply, self._recv_seq))
        else:
            # No registration in flight - reply cannot be fresh
            logger.debug("Dropping stale reply for unknown request %s", reply.request_id)

    def _handle_request(self, request: AcnetRequest):
        task = asyncio.create_task(self._request_ack(request.reply_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        self._requests_in[request.reply_id] = request

        if self._request_handler:
            try:
                self._request_handler(request)
            except Exception as e:  # noqa: BLE001
                logger.warning("Request handler exception: %s", e)
        else:
            logger.debug("No request handler, ignoring incoming request")

    def _handle_message(self, message: AcnetMessage):
        if self._message_handler:
            try:
                self._message_handler(message)
            except Exception as e:  # noqa: BLE001
                logger.warning("Message handler exception: %s", e)

    def _handle_cancel(self, cancel: AcnetCancel):
        if self._cancel_handler:
            try:
                self._cancel_handler(cancel)
            except Exception as e:  # noqa: BLE001
                logger.warning("Cancel handler exception: %s", e)

    # ------------------------------------------------------------------
    # Public commands
    # ------------------------------------------------------------------

    async def send_request(
        self,
        node: int,
        task: str,
        data: bytes,
        reply_handler: ReplyHandler,
        multiple_reply: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> AsyncRequestContext:
        """Send a request and register a reply handler."""
        task_rad50 = _rad50_encode(task)
        mult_flag = 1 if multiple_reply else 0
        tmo = timeout if timeout > 0 else INFINITE_TIMEOUT

        content = (
            struct.pack(
                ">H3I2HI",
                CMD_SEND_REQUEST_TIMEOUT,
                self._raw_handle,
                self._raw_vnode,
                task_rad50,
                node,
                mult_flag,
                tmo,
            )
            + data
        )

        # Open a registration window: replies for our (possibly reused) req_id
        # may arrive in the same TCP batch as the ack, before the handler is
        # registered below - _handle_reply buffers them while a window is open.
        self._pending_sends += 1
        try:
            ack = await self._xact(content)

            if len(ack) < 4:
                raise AcnetUnavailableError

            if len(ack) < 6:
                _ack_code, status = struct.unpack(">Hh", ack[:4])
                if status == ACNET_REQREJ:
                    raise AcnetRequestRejectedError(task)
                if status < 0:
                    raise AcnetError(status, f"SEND_REQUEST to '{task}' failed")
                raise AcnetUnavailableError

            _ack_code, status, req_id = struct.unpack(">HhH", ack[:6])

            if status < 0:
                raise AcnetError(status, f"SEND_REQUEST to '{task}' failed")

            # Snapshot our ack's position in the receive stream (still valid:
            # _cmd_lock serializes commands, so no other ack can overwrite it
            # before we resume).
            ack_seq = self._ack_recv_seq

            context = AsyncRequestContext(
                connection=self,
                task=task,
                node=node,
                request_id=RequestId(req_id),
                multiple_reply=multiple_reply,
                timeout=timeout,
                reply_handler=reply_handler,
            )

            self._reply_handlers[context.request_id] = context

            # Drain buffered replies (ACK+reply arrived in same batch).
            # On TCP acnetd writes our ack before forwarding any reply for the
            # new request, so pre-ack entries are stale from the ID's prior
            # life. On UDP acks and data use separate sockets — no ordering to
            # classify by, so deliver everything (bounded stale risk beats
            # dropping reordered fresh replies).
            buffered = self._reply_buffer.pop(context.request_id, ())
            for reply, recv_seq in buffered:
                if self._ORDERED_TRANSPORT and recv_seq <= ack_seq:
                    logger.debug("Discarding pre-ack stale reply for %s", context.request_id)
                    continue
                try:
                    context.reply_handler(reply)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Reply handler exception (buffered): %s", e)
                if reply.last:
                    self._reply_handlers.pop(context.request_id, None)
                    context._cancelled = True
                    break

            return context
        finally:
            self._pending_sends -= 1
            if self._pending_sends == 0:
                # No registration in flight - remaining buffered strays are stale
                self._reply_buffer.clear()

    async def request_single(
        self,
        node: int,
        task: str,
        data: bytes,
        reply_handler: ReplyHandler,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> AsyncRequestContext:
        return await self.send_request(node, task, data, reply_handler, multiple_reply=False, timeout=timeout)

    async def request_multiple(
        self,
        node: int,
        task: str,
        data: bytes,
        reply_handler: ReplyHandler,
        timeout: int = 0,
    ) -> AsyncRequestContext:
        return await self.send_request(node, task, data, reply_handler, multiple_reply=True, timeout=timeout)

    async def get_node(self, name: str) -> int:
        """Look up a node address by name."""
        name_rad50 = _rad50_encode(name)
        content = struct.pack(">H3I", CMD_NAME_LOOKUP, self._raw_handle, self._raw_vnode, name_rad50)
        ack = await self._xact(content)

        if len(ack) < 6:
            raise AcnetUnavailableError

        _ack_code, status, high, low = struct.unpack(">HhBB", ack[:6])

        if status < 0:
            raise AcnetError(status, f"GET_NODE failed for {name}")

        return high * 256 + low

    async def get_name(self, node: int) -> str:
        """Look up a node name by address."""
        content = struct.pack(">H2IH", CMD_NODE_LOOKUP, self._raw_handle, self._raw_vnode, node)
        ack = await self._xact(content)

        if len(ack) < 8:
            raise AcnetUnavailableError

        _ack_code, status, name_rad50 = struct.unpack(">HhI", ack[:8])

        if status < 0:
            raise AcnetError(status, f"GET_NAME failed for node {node}")

        return _rad50_decode(name_rad50)

    async def get_local_node(self) -> int:
        """Get the local node address."""
        content = struct.pack(">H2I", CMD_LOCAL_NODE, self._raw_handle, self._raw_vnode)
        ack = await self._xact(content)

        if len(ack) < 6:
            raise AcnetUnavailableError

        _ack_code, status, high, low = struct.unpack(">HhBB", ack[:6])

        if status < 0:
            raise AcnetError(status, "GET_LOCAL_NODE failed")

        return high * 256 + low

    async def get_default_node(self) -> int:
        """Get the default routing node."""
        content = struct.pack(">H2I", CMD_DEFAULT_NODE, self._raw_handle, self._raw_vnode)
        ack = await self._xact(content)

        if len(ack) < 6:
            raise AcnetUnavailableError

        _ack_code, status, high, low = struct.unpack(">HhBB", ack[:6])

        if status < 0:
            raise AcnetError(status, "GET_DEFAULT_NODE failed")

        return high * 256 + low

    async def rename_task(self, new_name: str):
        """Rename this connection's task handle."""
        if not new_name or len(new_name) > 6:
            raise ValueError("Task name must be 1-6 characters")

        name_rad50 = _rad50_encode(new_name)
        content = struct.pack(">H3I", CMD_RENAME_TASK, self._raw_handle, self._raw_vnode, name_rad50)
        ack = await self._xact(content)

        if len(ack) < 4:
            raise AcnetUnavailableError

        _ack_code, status = struct.unpack(">Hh", ack[:4])

        if status < 0:
            raise AcnetError(status, f"RENAME_TASK failed for '{new_name}'")

        self._raw_handle = name_rad50
        self._handle_name = _rad50_decode(name_rad50)
        logger.info("Renamed task to %s", self._handle_name)

    async def send_message(self, node: int, task: str, data: bytes):
        """Send an unsolicited message (no reply expected)."""
        task_rad50 = _rad50_encode(task)
        content = (
            struct.pack(
                ">H3IH",
                CMD_SEND,
                self._raw_handle,
                self._raw_vnode,
                task_rad50,
                node,
            )
            + data
        )
        ack = await self._xact(content)

        if len(ack) < 4:
            raise AcnetUnavailableError

        _ack_code, status = struct.unpack(">Hh", ack[:4])

        if status < 0:
            raise AcnetError(status, "SEND_MESSAGE failed")

    async def ignore_request(self, request: AcnetRequest):
        """Ignore an incoming request without sending a reply."""
        if request.cancelled:
            return

        self._requests_in.pop(request.reply_id, None)
        request.cancel()

        reply_id = request.reply_id.value & 0xFFFF
        content = struct.pack(">H2IH", CMD_IGNORE_REQUEST, self._raw_handle, self._raw_vnode, reply_id)
        ack = await self._xact(content)

        if len(ack) < 4:
            raise AcnetUnavailableError

        _ack_code, status = struct.unpack(">Hh", ack[:4])
        if status < 0:
            raise AcnetError(status, "IGNORE_REQUEST failed")

    async def get_node_stats(self) -> NodeStats:
        """Get ACNET node statistics."""
        content = struct.pack(">H2I", CMD_NODE_STATS, self._raw_handle, self._raw_vnode)
        ack = await self._xact(content)

        if len(ack) < 32:
            raise AcnetUnavailableError

        _ack_code, status = struct.unpack(">Hh", ack[:4])

        if status < 0:
            raise AcnetError(status, "GET_NODE_STATS failed")

        counters = struct.unpack(">7I", ack[4:32])
        return NodeStats(*counters)

    async def get_task_pid(self, task: str) -> int:
        """Get the OS process ID for an ACNET task."""
        task_rad50 = _rad50_encode(task)
        content = struct.pack(">H3I", CMD_TASK_PID, self._raw_handle, self._raw_vnode, task_rad50)
        ack = await self._xact(content)

        if len(ack) < 8:
            raise AcnetUnavailableError

        _ack_code, status, pid = struct.unpack(">HhI", ack[:8])

        if status < 0:
            raise AcnetError(status, f"GET_TASK_PID failed for '{task}'")

        return pid

    async def disconnect_single(self):
        """Disconnect this single task instance."""
        content = struct.pack(">H2I", CMD_DISCONNECT_SINGLE, self._raw_handle, self._raw_vnode)
        ack = await self._xact(content)

        if len(ack) < 4:
            raise AcnetUnavailableError

        _ack_code, status = struct.unpack(">Hh", ack[:4])
        if status < 0:
            raise AcnetError(status, "DISCONNECT_SINGLE failed")

        self._fail_reply_handlers()
        self._reply_buffer.clear()

        for request in self._requests_in.values():
            request.cancel()
        self._requests_in.clear()

        self._receiving = False
        self._connected = False

        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

    async def handle_messages(self, handler: MessageHandler):
        """Register a handler for unsolicited messages."""
        self._message_handler = handler
        await self._start_receiving()

    async def handle_requests(self, handler: RequestHandler):
        """Register a handler for incoming requests."""
        self._request_handler = handler
        await self._start_receiving()

    def handle_cancels(self, handler: CancelHandler):
        """Register a handler for cancel notifications."""
        self._cancel_handler = handler

    async def send_reply(self, request: AcnetRequest, data: bytes, status: int, last: bool = True):
        """Send a reply to an incoming request."""
        if request.cancelled:
            raise AcnetError(ACNET_NOT_CONNECTED, "Request was cancelled")

        flags = REPLY_ENDMULT if last else REPLY_NORMAL

        if not request.multiple_reply or last:
            self._requests_in.pop(request.reply_id, None)
            request.cancel()

        reply_id = request.reply_id.value & 0xFFFF
        content = (
            struct.pack(
                ">H2IHHh",
                CMD_SEND_REPLY,
                self._raw_handle,
                self._raw_vnode,
                reply_id,
                flags,
                status,
            )
            + data
        )

        await self._xact(content)

    async def _send_cancel(self, context: AsyncRequestContext):
        """Send a cancel for an outgoing request."""
        self._reply_handlers.pop(context.request_id, None)
        self._reply_buffer.pop(context.request_id, None)

        content = struct.pack(
            ">H2IH",
            CMD_CANCEL,
            self._raw_handle,
            self._raw_vnode,
            context.request_id.id,
        )

        try:
            await self._xact(content)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to cancel ACNET request %s", context.request_id, exc_info=True)

    async def _request_ack(self, reply_id: ReplyId) -> None:
        """Acknowledge receipt of an incoming request."""
        content = struct.pack(
            ">H2IH",
            CMD_REQUEST_ACK,
            self._raw_handle,
            self._raw_vnode,
            reply_id.value & 0xFFFF,
        )

        try:
            await self._xact(content)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to send request ack for %s", reply_id, exc_info=True)

    async def _start_receiving(self):
        """Start receiving incoming packets."""
        if not self._receiving:
            content = struct.pack(">H2I", CMD_RECEIVE_REQUESTS, self._raw_handle, self._raw_vnode)
            ack = await self._xact(content)

            if len(ack) < 4:
                raise AcnetUnavailableError

            _ack_code, status = struct.unpack(">Hh", ack[:4])
            if status < 0:
                raise AcnetError(status, "RECEIVE_REQUESTS failed")

            self._receiving = True

    async def _stop_receiving(self):
        """Stop receiving incoming packets."""
        if self._receiving:
            content = struct.pack(">H2I", CMD_BLOCK_REQUESTS, self._raw_handle, self._raw_vnode)
            ack = await self._xact(content)

            if len(ack) < 4:
                raise AcnetUnavailableError

            _ack_code, status = struct.unpack(">Hh", ack[:4])
            if status < 0:
                raise AcnetError(status, "BLOCK_REQUESTS failed")

            self._receiving = False

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    def __repr__(self):
        state = "connected" if self.connected else "disconnected"
        cls = type(self).__name__
        return f"{cls}({self._host}:{self._port}, {self._handle_name}, {state})"


# ======================================================================
# TCP transport
# ======================================================================


class AsyncAcnetConnectionTCP(AsyncAcnetConnectionBase):
    """Async ACNET connection over TCP with 4-byte length-prefix framing."""

    def __init__(
        self,
        host: str = ACSYS_PROXY_HOST,
        port: int = ACNET_TCP_PORT,
        name: str = "",
        *,
        vnode: str = "",
        trace: bool = False,
    ):
        super().__init__(host, port, name, vnode=vnode, trace=trace)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def _open_transport(self):
        """Open TCP connection and send handshake."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=5.0,
            )

            self._writer.write(TCP_HANDSHAKE)
            await self._writer.drain()

            sock = self._writer.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER_SIZE)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUFFER_SIZE)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            logger.debug("Opened async TCP channel to %s:%s", self._host, self._port)

        except OSError as e:
            logger.error("Failed to open TCP channel: %s", e)
            raise AcnetUnavailableError from e

    async def _close_transport(self):
        """Close the TCP writer."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except OSError:
                pass
            self._writer = None
            self._reader = None

    async def _send_frame(self, content: bytes):
        """Wrap a raw command in the TCP bridge envelope and length prefix."""
        if self._writer is None:
            raise AcnetUnavailableError
        frame = struct.pack(">H", ACNETD_COMMAND) + content
        self._writer.write(struct.pack(">I", len(frame)) + frame)
        await self._writer.drain()

    def _start_read_loop(self):
        """Start the TCP deframing read loop."""
        self._read_task = asyncio.ensure_future(self._tcp_read_loop())

    async def _tcp_read_loop(self):
        """Read and dispatch TCP frames (4-byte length prefix deframing)."""
        assert self._reader is not None
        buffer = bytearray()

        try:
            while not self._disposed:
                chunk = await self._reader.read(8192)
                if not chunk:
                    logger.warning("Connection closed by remote")
                    break
                buffer.extend(chunk)

                while len(buffer) >= 4:
                    pkt_len = struct.unpack(">I", buffer[:4])[0]

                    if pkt_len < 2 or pkt_len > 65535:
                        logger.error("Invalid packet length: %s, stream desynchronized", pkt_len)
                        self._disposed = True
                        break

                    if len(buffer) < 4 + pkt_len:
                        break

                    pkt_data = bytes(buffer[4 : 4 + pkt_len])
                    del buffer[: 4 + pkt_len]

                    msg_type = struct.unpack(">H", pkt_data[:2])[0]
                    msg_data = pkt_data[2:]

                    self._dispatch_frame(msg_type, msg_data)

        except asyncio.CancelledError:
            raise
        except OSError as e:
            if not self._disposed:
                logger.warning("Socket error in read loop: %s", e)
        except Exception:
            logger.exception("Error in read loop")

        self._on_connection_lost()


# ======================================================================
# Local UDP transport
# ======================================================================


class _AcnetUDPCommandProtocol(asyncio.DatagramProtocol):
    """Deliver raw acnetd ACK datagrams from the command socket."""

    def __init__(self, connection: "AsyncAcnetConnectionUDP"):
        self._conn = connection

    def datagram_received(self, data: bytes, addr):
        self._conn._dispatch_frame(ACNETD_ACK, data)

    def error_received(self, exc):
        logger.error("ACNET UDP command socket error: %s", exc)
        self._conn._transport_failed()

    def connection_lost(self, exc):
        if not self._conn._disposed:
            self._conn._transport_failed()


class _AcnetUDPDataProtocol(asyncio.DatagramProtocol):
    """Deliver raw ACNET packets from the data socket."""

    def __init__(self, connection: "AsyncAcnetConnectionUDP"):
        self._conn = connection

    def datagram_received(self, data: bytes, addr):
        self._conn._dispatch_data(data)

    def error_received(self, exc):
        logger.error("ACNET UDP data socket error: %s", exc)
        self._conn._transport_failed()

    def connection_lost(self, exc):
        if not self._conn._disposed:
            self._conn._transport_failed()


class AsyncAcnetConnectionUDP(AsyncAcnetConnectionBase):
    """Async connection to the official local acnetd UDP client interface."""

    # acnetd sends acks and data on separate sockets — no ack/reply ordering.
    _ORDERED_TRANSPORT: ClassVar[bool] = False

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
        self._cmd_transport: asyncio.DatagramTransport | None = None
        self._data_transport: asyncio.DatagramTransport | None = None
        self._cmd_protocol: _AcnetUDPCommandProtocol | None = None
        self._data_protocol: _AcnetUDPDataProtocol | None = None

    async def _open_transport(self):
        """Open separate local command and data sockets."""
        loop = asyncio.get_running_loop()
        try:
            data_transport, data_protocol = await asyncio.wait_for(
                loop.create_datagram_endpoint(
                    lambda: _AcnetUDPDataProtocol(self),
                    remote_addr=(self._host, self._port),
                    family=socket.AF_INET,
                ),
                timeout=5.0,
            )
            self._data_transport = data_transport
            self._data_protocol = data_protocol
            data_socket = data_transport.get_extra_info("socket")
            if data_socket is not None:
                data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUFFER_SIZE)

            cmd_transport, cmd_protocol = await asyncio.wait_for(
                loop.create_datagram_endpoint(
                    lambda: _AcnetUDPCommandProtocol(self),
                    remote_addr=(self._host, self._port),
                    family=socket.AF_INET,
                ),
                timeout=5.0,
            )
            self._cmd_transport = cmd_transport
            self._cmd_protocol = cmd_protocol
            cmd_socket = cmd_transport.get_extra_info("socket")
            if cmd_socket is not None:
                cmd_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER_SIZE)
            logger.debug("Opened local async UDP channels to %s:%s", self._host, self._port)
        except (OSError, TimeoutError) as e:
            await self._close_transport()
            logger.error("Failed to open local ACNET UDP channels: %s", e)
            raise AcnetUnavailableError from e

    async def _close_transport(self):
        """Close both local UDP sockets."""
        self._drop_transports()

    def _drop_transports(self) -> None:
        cmd_transport, self._cmd_transport = self._cmd_transport, None
        data_transport, self._data_transport = self._data_transport, None
        self._cmd_protocol = None
        self._data_protocol = None
        if cmd_transport is not None:
            cmd_transport.close()
        if data_transport is not None:
            data_transport.close()

    def _transport_failed(self) -> None:
        """Poison both UDP channels after either channel fails."""
        if self._cmd_transport is None and self._data_transport is None:
            return
        self._drop_transports()
        self._on_connection_lost()

    def _connect_pid(self) -> int:
        return os.getpid()

    def _connect_data_port(self) -> int:
        if self._data_transport is None:
            raise AcnetUnavailableError
        sockname = self._data_transport.get_extra_info("sockname")
        if not isinstance(sockname, tuple) or len(sockname) < 2 or not isinstance(sockname[1], int):
            raise AcnetUnavailableError
        return sockname[1]

    async def _send_frame(self, content: bytes):
        """Send one raw command datagram to local acnetd."""
        if self._cmd_transport is None:
            raise AcnetUnavailableError
        self._cmd_transport.sendto(content)

    def _start_read_loop(self):
        """Datagram protocol callbacks drive both receive paths."""
        self._read_task = None
