"""
ACNET UDP connection implementation.

This module provides a thread-based ACNET communication layer over UDP.
It connects to the local ACNET daemon and handles:
- Sending requests and receiving replies
- Handling incoming requests
- Unsolicited messages
- Connection monitoring and auto-reconnect
"""

import logging
import os
import select
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from . import rad50
from .constants import (
    ACNET_TCP_PORT,
    CMD_BLOCK_REQUESTS,
    CMD_CANCEL,
    CMD_CONNECT,
    CMD_CONNECT_EXT,
    CMD_DISCONNECT,
    CMD_IGNORE_REQUEST,
    CMD_KEEPALIVE,
    CMD_LOCAL_NODE,
    CMD_NAME_LOOKUP,
    CMD_NODE_LOOKUP,
    CMD_RECEIVE_REQUESTS,
    CMD_REQUEST_ACK,
    CMD_SEND,
    CMD_SEND_REPLY,
    CMD_SEND_REQUEST_TIMEOUT,
    DEFAULT_TIMEOUT,
    HEARTBEAT_TIMEOUT,
    RECONNECT_DELAY,
    RECV_BUFFER_SIZE,
    REPLY_ENDMULT,
    REPLY_NORMAL,
    SEND_BUFFER_SIZE,
)
from .errors import ACNET_NOT_CONNECTED, ACNET_SUCCESS, AcnetError, AcnetUnavailableError
from .packet import AcnetCancel, AcnetMessage, AcnetPacket, AcnetReply, AcnetRequest, ReplyId, RequestId

logger = logging.getLogger(__name__)


# Type aliases for handlers
ReplyHandler = Callable[[AcnetReply], None]
RequestHandler = Callable[[AcnetRequest], None]
MessageHandler = Callable[[AcnetMessage], None]
CancelHandler = Callable[[AcnetCancel], None]


@dataclass
class AcnetRequestContext:
    """Context for tracking an outgoing request."""

    connection: "AcnetConnection"
    task: str
    node: int
    request_id: RequestId
    multiple_reply: bool
    timeout: int
    reply_handler: ReplyHandler
    _cancelled: bool = field(default=False, init=False)

    def cancel(self):
        """Cancel this request."""
        if not self._cancelled:
            self._cancelled = True
            self.connection._send_cancel(self)

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class AcnetConnection:
    """
    Thread-based ACNET connection over UDP.

    This class manages communication with the local ACNET daemon,
    handling outgoing requests and incoming packets in separate threads.

    Example usage:
        conn = AcnetConnection("MYTASK")
        conn.connect()

        # Send a request
        def handle_reply(reply):
            print(f"Got reply: {reply.status}")

        ctx = conn.send_request(node, "TARGET", data, handle_reply)

        # Clean up
        conn.close()
    """

    def __init__(self, name: str = "", vnode: str = ""):
        """
        Initialize an ACNET connection.

        Args:
            name: Task name (up to 6 characters, auto-assigned if empty)
            vnode: Virtual node name for this connection
        """
        self._name = rad50.encode(name)
        self._vnode_name = vnode
        self._vnode = rad50.encode(vnode) if vnode else 0
        self._task_id = -1
        self._pid = os.getpid()

        # Sockets
        self._cmd_socket: Optional[socket.socket] = None
        self._data_socket: Optional[socket.socket] = None
        self._cmd_lock = threading.Lock()

        # State
        self._connected = False
        self._receiving = False
        self._disposed = False

        # Request tracking
        self._requests_out: dict[RequestId, AcnetRequestContext] = {}
        self._requests_out_lock = threading.Lock()
        self._requests_in: dict[ReplyId, AcnetRequest] = {}
        self._requests_in_lock = threading.Lock()

        # Handlers
        self._message_handler: Optional[MessageHandler] = None
        self._request_handler: Optional[RequestHandler] = None
        self._cancel_handler: Optional[CancelHandler] = None

        # Threads
        self._data_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def name(self) -> str:
        """Get the connected task name."""
        return rad50.decode_stripped(self._name)

    @property
    def task_id(self) -> int:
        """Get the assigned task ID."""
        return self._task_id

    @property
    def connected(self) -> bool:
        """Check if connected to daemon."""
        return self._task_id != -1 and not self._disposed

    def connect(self):
        """
        Connect to the ACNET daemon.

        Raises:
            AcnetUnavailableError: If daemon is not available
            AcnetError: If connection fails
        """
        if self._disposed:
            raise AcnetError(ACNET_NOT_CONNECTED, "Connection disposed")

        self._open_sockets()

        try:
            self._connect_8bit()
        except AcnetError:
            logger.warning("8-bit connect failed, trying 16-bit")
            self._connect_16bit()

        logger.info(f"Connected to ACNET as {self.name} (task_id={self._task_id})")

        # Start threads
        self._start_data_thread()
        self._start_monitor_thread()

    def close(self):
        """Close the connection and clean up resources."""
        self._disposed = True
        self._stop_event.set()

        # Disconnect from daemon
        if self._connected:
            try:
                self._disconnect()
            except Exception:
                pass

        # Close sockets
        if self._cmd_socket:
            try:
                self._cmd_socket.close()
            except Exception:
                pass
            self._cmd_socket = None

        if self._data_socket:
            try:
                self._data_socket.close()
            except Exception:
                pass
            self._data_socket = None

        # Wait for threads
        if self._data_thread and self._data_thread.is_alive():
            self._data_thread.join(timeout=2.0)
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        logger.info(f"Closed ACNET connection {self.name}")

    def send(self, node: int, task: str, data: bytes):
        """
        Send an unsolicited message.

        Args:
            node: Destination node value
            task: Destination task name
            data: Message payload
        """
        cmd_data = struct.pack("<I", rad50.encode(task)) + struct.pack("<H", node)
        self._send_command(CMD_SEND, 0, cmd_data, data)

    def send_request(
        self,
        node: int,
        task: str,
        data: bytes,
        reply_handler: ReplyHandler,
        multiple_reply: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> AcnetRequestContext:
        """
        Send a request and register a reply handler.

        Args:
            node: Destination node value
            task: Destination task name
            data: Request payload
            reply_handler: Callback for handling replies
            multiple_reply: Whether to expect multiple replies
            timeout: Request timeout in milliseconds (0 = infinite)

        Returns:
            Request context that can be used to cancel the request
        """
        task_rad50 = rad50.encode(task)
        mult_flag = 1 if multiple_reply else 0
        tmo = timeout if timeout > 0 else 0x7FFFFFFF

        cmd_data = struct.pack("<IHhI", task_rad50, node, mult_flag, tmo)
        ack = self._send_command(CMD_SEND_REQUEST_TIMEOUT, 2, cmd_data, data)

        req_id = struct.unpack_from("<H", ack, 4)[0]
        context = AcnetRequestContext(
            connection=self,
            task=task,
            node=node,
            request_id=RequestId(req_id),
            multiple_reply=multiple_reply,
            timeout=timeout,
            reply_handler=reply_handler,
        )

        with self._requests_out_lock:
            self._requests_out[context.request_id] = context

        return context

    def request_single(
        self,
        node: int,
        task: str,
        data: bytes,
        reply_handler: ReplyHandler,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> AcnetRequestContext:
        """Send a single-reply request."""
        return self.send_request(node, task, data, reply_handler, multiple_reply=False, timeout=timeout)

    def request_multiple(
        self,
        node: int,
        task: str,
        data: bytes,
        reply_handler: ReplyHandler,
        timeout: int = 0,
    ) -> AcnetRequestContext:
        """Send a multiple-reply request."""
        return self.send_request(node, task, data, reply_handler, multiple_reply=True, timeout=timeout)

    def handle_messages(self, handler: MessageHandler):
        """Register a handler for unsolicited messages."""
        self._message_handler = handler
        self._start_receiving()

    def handle_requests(self, handler: RequestHandler):
        """Register a handler for incoming requests."""
        self._request_handler = handler
        self._start_receiving()

    def handle_cancels(self, handler: CancelHandler):
        """Register a handler for cancel notifications."""
        self._cancel_handler = handler

    def get_node(self, name: str) -> int:
        """Look up a node value by name (cmdNameLookup)."""
        cmd_data = struct.pack("<I", rad50.encode(name))
        ack = self._send_command(CMD_NAME_LOOKUP, 4, cmd_data)
        return struct.unpack_from("<H", ack, 4)[0]

    def get_name(self, node: int) -> str:
        """Look up a node name by value (cmdNodeLookup)."""
        cmd_data = struct.pack("<H", node)
        ack = self._send_command(CMD_NODE_LOOKUP, 5, cmd_data)
        return rad50.decode_stripped(struct.unpack_from("<I", ack, 4)[0])

    def get_local_node(self) -> int:
        """Get the local node value (cmdLocalNode)."""
        ack = self._send_command(CMD_LOCAL_NODE, 4, b"")
        return struct.unpack_from("<H", ack, 4)[0]

    def send_reply(self, request: AcnetRequest, data: bytes, status: int, last: bool = True):
        """
        Send a reply to an incoming request.

        Args:
            request: The request to reply to
            data: Reply payload
            status: Status code
            last: True if this is the last reply
        """
        if request.cancelled:
            raise AcnetError(ACNET_NOT_CONNECTED, "Request was cancelled")

        flags = REPLY_ENDMULT if last else REPLY_NORMAL

        if not request.multiple_reply or last:
            with self._requests_in_lock:
                self._requests_in.pop(request.reply_id, None)
                request.cancel()

        reply_id = request.reply_id.value
        cmd_data = struct.pack("<HHh", reply_id & 0xFFFF, flags, status)
        self._send_command(CMD_SEND_REPLY, 3, cmd_data, data)

    def _open_sockets(self):
        """Open command and data UDP sockets."""
        try:
            # Command socket
            self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._cmd_socket.connect(("127.0.0.1", ACNET_TCP_PORT))
            self._cmd_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER_SIZE)
            self._cmd_socket.setblocking(False)

            # Data socket (separate for receiving)
            self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._data_socket.connect(("127.0.0.1", ACNET_TCP_PORT))
            self._data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUFFER_SIZE)
            self._data_socket.setblocking(False)

        except OSError as e:
            logger.error(f"Failed to open ACNET sockets: {e}")
            raise AcnetUnavailableError()

    def _local_port(self) -> int:
        """Get the local port of the data socket."""
        if self._data_socket:
            return self._data_socket.getsockname()[1]
        return 0

    def _connect_8bit(self):
        """Connect to daemon with 8-bit task ID (cmdConnect)."""
        cmd_data = struct.pack("<IH", self._pid, self._local_port())
        ack = self._send_command(CMD_CONNECT, 1, cmd_data)

        self._task_id = ack[4] & 0xFF
        self._name = struct.unpack_from("<I", ack, 5)[0]
        self._connected = True

    def _connect_16bit(self):
        """Connect to daemon with 16-bit task ID (cmdConnectExt)."""
        cmd_data = struct.pack("<IH", self._pid, self._local_port())
        ack = self._send_command(CMD_CONNECT_EXT, 16, cmd_data)

        self._task_id = struct.unpack_from("<H", ack, 4)[0]
        self._name = struct.unpack_from("<I", ack, 6)[0]
        self._connected = True

    def _disconnect(self):
        """Disconnect from the daemon."""
        try:
            self._send_command(CMD_DISCONNECT, 0, b"")
        except Exception:
            pass

        # Cancel all outstanding requests
        with self._requests_out_lock:
            for ctx in self._requests_out.values():
                ctx._cancelled = True
            self._requests_out.clear()

        self._task_id = -1
        self._connected = False

    def _ping(self):
        """Send a ping to the daemon (cmdKeepAlive)."""
        self._send_command(CMD_KEEPALIVE, 0, b"")

    def _start_receiving(self):
        """Start receiving incoming packets (cmdReceiveRequests)."""
        if not self._receiving:
            self._receiving = True
            self._send_command(CMD_RECEIVE_REQUESTS, 0, b"")

    def _stop_receiving(self):
        """Stop receiving incoming packets (cmdBlockRequests)."""
        if self._receiving:
            self._receiving = False
            self._send_command(CMD_BLOCK_REQUESTS, 0, b"")

    def _request_ack(self, reply_id: ReplyId):
        """Acknowledge receipt of a request."""
        cmd_data = struct.pack("<H", reply_id.value & 0xFFFF)
        self._send_command(CMD_REQUEST_ACK, 0, cmd_data)

    def _send_cancel(self, context: AcnetRequestContext):
        """Send a cancel for an outgoing request (cmdCancel)."""
        with self._requests_out_lock:
            self._requests_out.pop(context.request_id, None)

        cmd_data = struct.pack("<H", context.request_id.id)
        self._send_command(CMD_CANCEL, 0, cmd_data)

    def _ignore_request(self, request: AcnetRequest):
        """Ignore an incoming request without sending a reply."""
        if request.cancelled:
            return

        with self._requests_in_lock:
            self._requests_in.pop(request.reply_id, None)
            request.cancel()

        cmd_data = struct.pack("<H", request.reply_id.value & 0xFFFF)
        self._send_command(CMD_IGNORE_REQUEST, 0, cmd_data)

    def _send_command(self, cmd: int, expected_ack: int, cmd_data: bytes, payload: bytes = b"") -> bytes:
        """
        Send a command to the daemon and wait for acknowledgement.

        Args:
            cmd: Command code
            expected_ack: Expected acknowledgement code
            cmd_data: Command-specific data
            payload: Additional payload data

        Returns:
            Acknowledgement packet data

        Raises:
            AcnetUnavailableError: If daemon doesn't respond
            AcnetError: If daemon returns an error
        """
        if self._disposed:
            raise AcnetError(ACNET_NOT_CONNECTED, "Connection disposed")

        with self._cmd_lock:
            # Build header: cmd(2) + name(4) + vnode(4)
            header = struct.pack("<HII", cmd, self._name, self._vnode)

            # Send command
            packet = header + cmd_data + payload
            try:
                self._cmd_socket.send(packet)
            except OSError as e:
                logger.error(f"Failed to send command {cmd}: {e}")
                raise AcnetUnavailableError()

            # Wait for acknowledgement (up to 5 seconds)
            ready, _, _ = select.select([self._cmd_socket], [], [], 5.0)
            if not ready:
                logger.error(f"Timeout waiting for ack on command {cmd}")
                raise AcnetUnavailableError()

            try:
                ack_data = self._cmd_socket.recv(256)
            except OSError as e:
                logger.error(f"Failed to receive ack: {e}")
                raise AcnetUnavailableError()

            if len(ack_data) < 4:
                raise AcnetUnavailableError()

            # Parse acknowledgement
            r_ack, status = struct.unpack_from("<Hh", ack_data, 0)

            if status != ACNET_SUCCESS:
                raise AcnetError(status, f"Command {cmd} failed")

            if expected_ack != 0 and r_ack != expected_ack:
                raise RuntimeError(f"Command/ack mismatch: expected {expected_ack}, got {r_ack}")

            return ack_data

    def _start_data_thread(self):
        """Start the data receiving thread."""
        self._data_thread = threading.Thread(target=self._data_thread_run, name=f"ACNET-data-{self.name}", daemon=True)
        self._data_thread.start()

    def _data_thread_run(self):
        """Data thread main loop - receives and dispatches packets."""
        logger.debug(f"Data thread started for {self.name}")

        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([self._data_socket], [], [], 0.5)
                if not ready:
                    continue

                data = self._data_socket.recv(65536)
                if not data:
                    continue

                packet = AcnetPacket.parse(data)
                self._handle_packet(packet)

            except OSError:
                if not self._stop_event.is_set():
                    logger.warning(f"Socket error in data thread for {self.name}")
                break
            except Exception as e:
                logger.exception(f"Error in data thread: {e}")

        logger.debug(f"Data thread stopped for {self.name}")

    def _handle_packet(self, packet: AcnetPacket):
        """Dispatch a received packet to the appropriate handler."""
        try:
            if packet.is_reply():
                self._handle_reply(packet)
            elif packet.is_request():
                self._handle_request(packet)
            elif packet.is_message():
                self._handle_message(packet)
            elif packet.is_cancel():
                self._handle_cancel(packet)
        except Exception as e:
            logger.exception(f"Error handling packet: {e}")

    def _handle_reply(self, reply: AcnetReply):
        """Handle an incoming reply."""
        with self._requests_out_lock:
            context = self._requests_out.get(reply.request_id)

        if context:
            try:
                context.reply_handler(reply)
            except Exception as e:
                logger.warning(f"Reply handler exception: {e}")

            if reply.last:
                with self._requests_out_lock:
                    self._requests_out.pop(reply.request_id, None)
                context._cancelled = True
        else:
            # No handler - send cancel
            try:
                cmd_data = struct.pack("<H", reply.request_id.id)
                self._send_command(CMD_CANCEL, 0, cmd_data)
            except Exception:
                pass

    def _handle_request(self, request: AcnetRequest):
        """Handle an incoming request."""
        try:
            self._request_ack(request.reply_id)
        except Exception as e:
            logger.warning(f"Failed to ack request: {e}")
            return

        with self._requests_in_lock:
            self._requests_in[request.reply_id] = request

        if self._request_handler:
            try:
                self._request_handler(request)
            except Exception as e:
                logger.warning(f"Request handler exception: {e}")
                try:
                    self.send_reply(request, b"", REPLY_ENDMULT, last=True)
                except Exception:
                    pass
        else:
            # No handler - send end-mult status
            try:
                self.send_reply(request, b"", REPLY_ENDMULT, last=True)
            except Exception:
                pass

    def _handle_message(self, message: AcnetMessage):
        """Handle an unsolicited message."""
        if self._message_handler:
            try:
                self._message_handler(message)
            except Exception as e:
                logger.warning(f"Message handler exception: {e}")

    def _handle_cancel(self, cancel: AcnetCancel):
        """Handle a cancel notification."""
        if self._cancel_handler:
            try:
                self._cancel_handler(cancel)
            except Exception as e:
                logger.warning(f"Cancel handler exception: {e}")

    def _start_monitor_thread(self):
        """Start the connection monitor thread."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_thread_run, name=f"ACNET-monitor-{self.name}", daemon=True
        )
        self._monitor_thread.start()

    def _monitor_thread_run(self):
        """Monitor thread main loop - pings daemon and handles reconnection."""
        logger.debug(f"Monitor thread started for {self.name}")

        while not self._stop_event.is_set():
            time.sleep(HEARTBEAT_TIMEOUT / 1000.0)

            if self._stop_event.is_set():
                break

            try:
                if self._connected:
                    self._ping()
                else:
                    logger.info(f"Attempting reconnect for {self.name}")
                    self._connect_8bit()
                    if self._receiving:
                        self._start_receiving()
            except AcnetError:
                logger.warning(f"Lost connection for {self.name}")
                self._disconnect()
                time.sleep(RECONNECT_DELAY / 1000.0)
            except Exception as e:
                logger.exception(f"Monitor thread error: {e}")

        logger.debug(f"Monitor thread stopped for {self.name}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
