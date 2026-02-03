"""
ACNET TCP connection implementation.

This module provides a thread-based ACNET communication layer over TCP.
TCP connections are used for remote access to ACNET daemon (acnetd) via
acsys-proxy, and will be the primary channel for DPM communication.

Protocol (matching acsys-python reference):
1. Connect to remote host (e.g., acsys-proxy.fnal.gov:6802)
2. Send handshake: "RAW\r\n\r\n"
3. All messages use 4-byte big-endian length prefix
4. Message types: PING(0), COMMAND(1), ACK(2), DATA(3)
5. Commands use big-endian encoding with handle-based addressing
"""

import logging
import queue
import select
import socket
import struct
import threading
import time
from typing import Optional

from .constants import (
    ACNET_TCP_PORT,
    DEFAULT_TIMEOUT,
    HEARTBEAT_TIMEOUT,
    RECV_BUFFER_SIZE,
    REPLY_ENDMULT,
    REPLY_NORMAL,
    SEND_BUFFER_SIZE,
)
from .errors import ACNET_NOT_CONNECTED, AcnetError, AcnetUnavailableError
from .packet import (
    AcnetCancel,
    AcnetMessage,
    AcnetPacket,
    AcnetReply,
    AcnetRequest,
    ReplyId,
    RequestId,
)

logger = logging.getLogger(__name__)

# Default proxy host
ACSYS_PROXY_HOST = "acsys-proxy.fnal.gov"

# TCP message types
TCP_CLIENT_PING = 0
ACNETD_COMMAND = 1
ACNETD_ACK = 2
ACNETD_DATA = 3

# Handshake string
TCP_HANDSHAKE = b"RAW\r\n\r\n"

# acnetd command codes (official names from acnetd CommandList enum)
CMD_KEEPALIVE = 0               # cmdKeepAlive
CMD_CONNECT = 1                 # cmdConnect (8-bit task ID)
CMD_DISCONNECT = 3              # cmdDisconnect
CMD_SEND = 4                    # cmdSend (unsolicited message)
CMD_RECEIVE_REQUESTS = 6        # cmdReceiveRequests
CMD_SEND_REPLY = 7              # cmdSendReply
CMD_CANCEL = 8                  # cmdCancel
CMD_REQUEST_ACK = 9             # cmdRequestAck
CMD_NAME_LOOKUP = 11            # cmdNameLookup (name -> trunk/node)
CMD_NODE_LOOKUP = 12            # cmdNodeLookup (trunk/node -> name)
CMD_LOCAL_NODE = 13             # cmdLocalNode
CMD_CONNECT_EXT = 16            # cmdConnectExt (16-bit task ID)
CMD_SEND_REQUEST_TIMEOUT = 18   # cmdSendRequestWithTimeout

# Type aliases for handlers
ReplyHandler = type(lambda reply: None)
RequestHandler = type(lambda request: None)
MessageHandler = type(lambda message: None)
CancelHandler = type(lambda cancel: None)


class AcnetRequestContext:
    """Context for tracking an outgoing request."""

    def __init__(
        self,
        connection: "AcnetConnectionTCP",
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

    def cancel(self):
        """Cancel this request."""
        if not self._cancelled:
            self._cancelled = True
            self.connection._send_cancel(self)

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class AcnetConnectionTCP:
    """
    Thread-based ACNET connection over TCP.

    This class manages TCP communication with a remote ACNET daemon (acnetd),
    typically accessed via acsys-proxy. It handles:
    - Length-prefixed framing with big-endian encoding
    - Outgoing requests and incoming replies
    - Incoming requests from remote tasks
    - Connection monitoring and auto-reconnect

    Protocol format matches the acsys-python reference implementation.

    Example usage:
        conn = AcnetConnectionTCP("acsys-proxy.fnal.gov")
        conn.connect()

        def handle_reply(reply):
            print(f"Got reply: {reply.status}")

        ctx = conn.send_request(node, "DPM", data, handle_reply)

        conn.close()
    """

    # RAD50 character set for encoding/decoding
    _rad50_chars = b" ABCDEFGHIJKLMNOPQRSTUVWXYZ$.%0123456789"

    def __init__(self, host: str = ACSYS_PROXY_HOST, port: int = ACNET_TCP_PORT, name: str = ""):
        """
        Initialize a TCP ACNET connection.

        Args:
            host: Remote host to connect to (default: acsys-proxy.fnal.gov)
            port: Remote port (default: 6802)
            name: Task name (up to 6 characters, auto-assigned if empty)
        """
        self._host = host
        self._port = port
        self._requested_name = name

        # Handle assigned by daemon (used in all commands)
        self._raw_handle = 0
        self._handle_name = ""

        # Socket
        self._socket: Optional[socket.socket] = None
        self._socket_lock = threading.Lock()
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

        # Ack queue for synchronous command responses
        self._ack_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)

        # Handlers
        self._message_handler: Optional[MessageHandler] = None
        self._request_handler: Optional[RequestHandler] = None
        self._cancel_handler: Optional[CancelHandler] = None

        # Threads
        self._read_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def name(self) -> str:
        """Get the connected task name (handle)."""
        return self._handle_name

    @property
    def handle(self) -> str:
        """Alias for name - the assigned handle."""
        return self._handle_name

    @property
    def raw_handle(self) -> int:
        """Get the raw handle value."""
        return self._raw_handle

    @property
    def connected(self) -> bool:
        """Check if connected to daemon."""
        return self._connected and not self._disposed

    @property
    def host(self) -> str:
        """Get the remote host."""
        return self._host

    @property
    def port(self) -> int:
        """Get the remote port."""
        return self._port

    @staticmethod
    def _rtoa(r50: int) -> str:
        """Convert RAD50 value to string."""
        chars = AcnetConnectionTCP._rad50_chars
        result = bytearray(6)

        first_bit = r50 & 0xFFFF
        second_bit = (r50 >> 16) & 0xFFFF

        for i in range(3):
            result[2 - i] = chars[int(first_bit % 40)]
            first_bit //= 40
            result[5 - i] = chars[int(second_bit % 40)]
            second_bit //= 40

        return result.decode("ascii").strip()

    @staticmethod
    def _ator(s: str) -> int:
        """Convert string to RAD50 value."""

        def char_to_index(c):
            if "A" <= c <= "Z":
                return ord(c) - ord("A") + 1
            if "a" <= c <= "z":
                return ord(c) - ord("a") + 1
            if "0" <= c <= "9":
                return ord(c) - ord("0") + 30
            if c == "$":
                return 27
            if c == ".":
                return 28
            if c == "%":
                return 29
            return 0

        first_bit = 0
        second_bit = 0
        s_len = len(s)

        for i in range(6):
            c = s[i] if i < s_len else " "
            if i < 3:
                first_bit = first_bit * 40 + char_to_index(c)
            else:
                second_bit = second_bit * 40 + char_to_index(c)

        return (second_bit << 16) | first_bit

    def connect(self):
        """
        Connect to the remote ACNET daemon.

        Raises:
            AcnetUnavailableError: If connection fails
            AcnetError: If registration fails
        """
        if self._disposed:
            raise AcnetError(ACNET_NOT_CONNECTED, "Connection disposed")

        self._open_channel()

        # Start read thread BEFORE sending commands (needed to receive acks)
        self._start_read_thread()

        # Send CONNECT command
        self._do_connect()

        logger.info(f"Connected to ACNET via TCP {self._host}:{self._port} as {self._handle_name}")

        # Start monitor thread
        self._start_monitor_thread()

    def _do_connect(self):
        """Send CONNECT command and process response."""
        # CONNECT command format (from reference):
        # [length: 4 BE][type: 2 BE][cmd: 2 BE][handle: 4 BE][0: 4 BE][0: 4 BE][0: 2 BE]
        # Total content: 18 bytes
        buf = struct.pack(">I2H3IH", 18, ACNETD_COMMAND, CMD_CONNECT, self._raw_handle, 0, 0, 0)

        ack = self._xact(buf)

        # ACK format for CMD_CONNECT (code 1) - type already stripped:
        # [ack_code: 2 BE][status: 2 BE signed][task_id: 1][handle: 4 BE]
        if len(ack) < 9:
            raise AcnetUnavailableError()

        ack_code, status, task_id, handle = struct.unpack(">HhBI", ack[:9])

        if status < 0:
            raise AcnetError(status, f"CONNECT failed with status {status}")

        self._raw_handle = handle
        self._handle_name = self._rtoa(handle)
        self._connected = True

        logger.debug(f"Connected with handle {self._handle_name} ({handle:#x})")

    def close(self):
        """Close the connection and clean up resources."""
        self._disposed = True
        self._stop_event.set()

        # Disconnect from daemon
        if self._connected:
            try:
                self._do_disconnect()
            except Exception:
                pass

        # Close socket
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        # Wait for threads
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        logger.info(f"Closed TCP ACNET connection {self._handle_name}")

    def _do_disconnect(self):
        """Send DISCONNECT command."""
        # DISCONNECT command: [length][type][cmd][handle][0]
        buf = struct.pack(">I2H2I", 12, ACNETD_COMMAND, CMD_DISCONNECT, self._raw_handle, 0)
        try:
            self._xact(buf)
        except Exception:
            pass

        # Cancel all outstanding requests
        with self._requests_out_lock:
            for ctx in self._requests_out.values():
                ctx._cancelled = True
            self._requests_out.clear()

        self._connected = False

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
        task_rad50 = self._ator(task)
        mult_flag = 1 if multiple_reply else 0
        tmo = timeout if timeout > 0 else 1000

        # SEND_REQUEST command format (from reference):
        # [length: 4 BE][type: 2 BE][cmd: 2 BE][handle: 4 BE][0: 4 BE]
        # [task: 4 BE][node: 2 BE][mult: 2 BE][timeout: 4 BE][payload]
        content_len = 24 + len(data)
        buf = (
            struct.pack(
                ">I2H3I2HI",
                content_len,
                ACNETD_COMMAND,
                CMD_SEND_REQUEST_TIMEOUT,
                self._raw_handle,
                0,
                task_rad50,
                node,
                mult_flag,
                tmo,
            )
            + data
        )

        ack = self._xact(buf)

        # ACK format for SEND_REQUEST (code 2) - type already stripped:
        # [ack_code: 2 BE][status: 2 BE signed][req_id: 2 BE]
        if len(ack) < 6:
            raise AcnetUnavailableError()

        ack_code, status, req_id = struct.unpack(">HhH", ack[:6])

        if status < 0:
            raise AcnetError(status, "SEND_REQUEST failed")

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

    def get_node(self, name: str) -> int:
        """Look up a node address by name."""
        # NAME_LOOKUP command (11) - cmdNameLookup: name -> trunk/node
        # [length: 4 BE][type: 2 BE][cmd: 2 BE][handle: 4 BE][0: 4 BE][name: 4 BE]
        name_rad50 = self._ator(name)
        buf = struct.pack(">I2H3I", 16, ACNETD_COMMAND, CMD_NAME_LOOKUP, self._raw_handle, 0, name_rad50)

        ack = self._xact(buf)

        # ACK format (code 4) - type already stripped:
        # [ack_code: 2 BE][status: 2 BE signed][high: 1][low: 1]
        if len(ack) < 6:
            raise AcnetUnavailableError()

        ack_code, status, high, low = struct.unpack(">HhBB", ack[:6])

        if status < 0:
            raise AcnetError(status, f"GET_NODE failed for {name}")

        return high * 256 + low

    def get_name(self, node: int) -> str:
        """Look up a node name by address."""
        # NODE_LOOKUP command (12) - cmdNodeLookup: trunk/node -> name
        # [length: 4 BE][type: 2 BE][cmd: 2 BE][handle: 4 BE][0: 4 BE][addr: 2 BE]
        buf = struct.pack(">I2H2IH", 14, ACNETD_COMMAND, CMD_NODE_LOOKUP, self._raw_handle, 0, node)

        ack = self._xact(buf)

        # ACK format (code 5) - type already stripped:
        # [ack_code: 2 BE][status: 2 BE signed][name: 4 BE]
        if len(ack) < 8:
            raise AcnetUnavailableError()

        ack_code, status, name_rad50 = struct.unpack(">HhI", ack[:8])

        if status < 0:
            raise AcnetError(status, f"GET_NAME failed for node {node}")

        return self._rtoa(name_rad50)

    def get_local_node(self) -> int:
        """Get the local node address."""
        # LOCAL_NODE command (13) - cmdLocalNode
        # [length: 4 BE][type: 2 BE][cmd: 2 BE][handle: 4 BE][0: 4 BE]
        buf = struct.pack(">I2H2I", 12, ACNETD_COMMAND, CMD_LOCAL_NODE, self._raw_handle, 0)

        ack = self._xact(buf)

        # ACK format - type already stripped:
        # [ack_code: 2 BE][status: 2 BE signed][high: 1][low: 1]
        if len(ack) < 6:
            raise AcnetUnavailableError()

        ack_code, status, high, low = struct.unpack(">HhBB", ack[:6])

        if status < 0:
            raise AcnetError(status, "GET_LOCAL_NODE failed")

        return high * 256 + low

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

        # SEND_REPLY command (7)
        reply_id = request.reply_id.value & 0xFFFF
        content_len = 14 + len(data)
        buf = (
            struct.pack(
                ">I2H2IHHh",
                content_len,
                ACNETD_COMMAND,
                CMD_SEND_REPLY,
                self._raw_handle,
                0,
                reply_id,
                flags,
                status,
            )
            + data
        )

        self._xact(buf)

    def _open_channel(self):
        """Open TCP socket and send handshake."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self._host, self._port))

            # Send handshake
            self._socket.sendall(TCP_HANDSHAKE)

            # Set socket options
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER_SIZE)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUFFER_SIZE)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            logger.debug(f"Opened TCP channel to {self._host}:{self._port}")

        except OSError as e:
            logger.error(f"Failed to open TCP channel: {e}")
            raise AcnetUnavailableError()

    def _xact(self, buf: bytes) -> bytes:
        """
        Send a command and wait for acknowledgement.

        Args:
            buf: Complete command buffer (with length prefix)

        Returns:
            ACK packet data (without length prefix)
        """
        if self._disposed:
            raise AcnetError(ACNET_NOT_CONNECTED, "Connection disposed")

        with self._cmd_lock:
            try:
                with self._socket_lock:
                    self._socket.sendall(buf)
            except OSError as e:
                logger.error(f"Failed to send command: {e}")
                raise AcnetUnavailableError()

            # Wait for acknowledgement
            try:
                ack_data = self._ack_queue.get(timeout=5.0)
            except queue.Empty:
                logger.error("Timeout waiting for ack")
                raise AcnetUnavailableError()

            return ack_data

    def _send_cancel(self, context: AcnetRequestContext):
        """Send a cancel for an outgoing request."""
        with self._requests_out_lock:
            self._requests_out.pop(context.request_id, None)

        # CANCEL command (8) - cmdCancel:
        # [length: 4 BE][type: 2 BE][cmd: 2 BE][handle: 4 BE][0: 4 BE][req_id: 2 BE]
        buf = struct.pack(
            ">I2H2IH",
            14,
            ACNETD_COMMAND,
            CMD_CANCEL,
            self._raw_handle,
            0,
            context.request_id.id,
        )

        try:
            self._xact(buf)
        except Exception:
            pass

    def _request_ack(self, reply_id: ReplyId):
        """Acknowledge receipt of a request."""
        # REQUEST_ACK command (9)
        buf = struct.pack(
            ">I2H2IH",
            14,
            ACNETD_COMMAND,
            CMD_REQUEST_ACK,
            self._raw_handle,
            0,
            reply_id.value & 0xFFFF,
        )

        try:
            self._xact(buf)
        except Exception as e:
            logger.warning(f"Failed to send request ack: {e}")

    def _start_receiving(self):
        """Start receiving incoming packets."""
        if not self._receiving:
            self._receiving = True
            # RECEIVE_REQUESTS command (6) - cmdReceiveRequests
            buf = struct.pack(">I2H2I", 12, ACNETD_COMMAND, CMD_RECEIVE_REQUESTS, self._raw_handle, 0)
            try:
                self._xact(buf)
            except Exception as e:
                logger.warning(f"Failed to start receiving: {e}")

    def _start_read_thread(self):
        """Start the read thread."""
        self._read_thread = threading.Thread(
            target=self._read_thread_run,
            name=f"ACNET-TCP-read-{self._handle_name}",
            daemon=True,
        )
        self._read_thread.start()

    def _read_thread_run(self):
        """Read thread main loop - receives and dispatches packets."""
        logger.debug(f"TCP read thread started for {self._handle_name}")

        # Set socket to non-blocking for select
        self._socket.setblocking(False)

        buffer = bytearray()

        while not self._stop_event.is_set():
            try:
                # Check if socket is still valid
                sock = self._socket
                if sock is None:
                    break

                # Wait for data with timeout
                ready, _, _ = select.select([sock], [], [], 0.5)
                if not ready:
                    continue

                # Read available data
                try:
                    chunk = sock.recv(8192)
                    if not chunk:
                        logger.warning("Connection closed")
                        break
                    buffer.extend(chunk)
                except BlockingIOError:
                    continue

                # Process complete packets
                while len(buffer) >= 4:
                    pkt_len = struct.unpack(">I", buffer[:4])[0]

                    if pkt_len < 2 or pkt_len > 65535:
                        logger.warning(f"Invalid packet length: {pkt_len}")
                        buffer = bytearray()
                        break

                    if len(buffer) < 4 + pkt_len:
                        break  # Need more data

                    # Extract packet
                    pkt_data = bytes(buffer[4 : 4 + pkt_len])
                    buffer = buffer[4 + pkt_len :]

                    # Parse message type
                    msg_type = struct.unpack(">H", pkt_data[:2])[0]
                    msg_data = pkt_data[2:]

                    self._handle_tcp_message(msg_type, msg_data)

            except OSError as e:
                if not self._stop_event.is_set():
                    logger.warning(f"Socket error in read thread: {e}")
                break
            except Exception as e:
                logger.exception(f"Error in read thread: {e}")

        logger.debug(f"TCP read thread stopped for {self._handle_name}")

    def _handle_tcp_message(self, msg_type: int, data: bytes):
        """Handle a received TCP message based on type."""
        if msg_type == TCP_CLIENT_PING:
            pass  # Ignore pings
        elif msg_type == ACNETD_ACK:
            # Put ACK in queue for waiting command
            try:
                self._ack_queue.put_nowait(data)
            except queue.Full:
                logger.warning("Ack queue full, dropping ack")
        elif msg_type == ACNETD_DATA:
            # ACNET packet data
            if len(data) >= 18:
                try:
                    packet = AcnetPacket.parse(data)
                    self._handle_packet(packet)
                except Exception as e:
                    logger.warning(f"Error parsing ACNET packet: {e}")
        else:
            logger.warning(f"Unknown TCP message type: {msg_type}")

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
                buf = struct.pack(
                    ">I2H2IH",
                    14,
                    ACNETD_COMMAND,
                    CMD_CANCEL,
                    self._raw_handle,
                    0,
                    reply.request_id.id,
                )
                self._xact(buf)
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
            target=self._monitor_thread_run,
            name=f"ACNET-TCP-monitor-{self._handle_name}",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_thread_run(self):
        """Monitor thread main loop - handles reconnection."""
        logger.debug(f"TCP monitor thread started for {self._handle_name}")

        while not self._stop_event.is_set():
            time.sleep(HEARTBEAT_TIMEOUT / 1000.0)

            if self._stop_event.is_set():
                break

            # Monitor connection health - could add ping here if needed

        logger.debug(f"TCP monitor thread stopped for {self._handle_name}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
