"""
Low-level DPM (Data Pool Manager) connection layer.

This module provides the foundational TCP connection to DPM with:
- Safe HTTP-style handshake with error detection
- Length-prefixed message I/O
- PC binary protocol marshaling/unmarshaling

This is a building block for DPMBackend and other higher-level APIs.
For full-featured DPM client with callbacks and threading, see acnet.connection_dpm.

Protocol Flow:
1. Connect to acsys-proxy.fnal.gov:6802 via TCP
2. Send HTTP-style handshake: GET /dpm HTTP/1.1\r\nContent-Type: application/pc\r\n\r\n
3. Safe response detection: read first 4 bytes
   - If "HTTP" -> server returned HTTP error, read rest and raise ConnectionError
   - Otherwise -> interpret as big-endian uint32 length prefix
4. Read OpenList reply and extract list_id
5. Send/receive length-prefixed PC messages
"""

import logging
import socket
import struct
from typing import Optional, Union

from pacsys.dpm_protocol import (
    OpenList_reply,
    ProtocolError,
    unmarshal_reply,
)

logger = logging.getLogger(__name__)

# Default DPM proxy settings
DEFAULT_HOST = "acsys-proxy.fnal.gov"
DEFAULT_PORT = 6802
DEFAULT_TIMEOUT = 5.0

# HTTP-style handshake for DPM connection
DPM_HANDSHAKE = b"GET /dpm HTTP/1.1\r\nContent-Type: application/pc\r\n\r\n"

# Maximum message size to accept (1MB)
MAX_MESSAGE_SIZE = 1024 * 1024


class DPMConnectionError(Exception):
    """Exception raised for DPM connection errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class DPMConnection:
    """
    Low-level TCP connection to DPM server.

    Handles connection establishment, handshake, and message I/O.
    Use as context manager for automatic resource cleanup.

    Example:
        with DPMConnection(host="acsys-proxy.fnal.gov", port=6802) as conn:
            print(f"List ID: {conn.list_id}")

            # Send a message
            from pacsys.dpm_protocol import AddToList_request
            add_req = AddToList_request()
            add_req.list_id = conn.list_id
            add_req.ref_id = 1
            add_req.drf_request = "M:OUTTMP@I"
            conn.send_message(add_req)

            # Receive a reply
            reply = conn.recv_message()

    Attributes:
        list_id: The DPM list ID obtained from OpenList reply (read-only)
        connected: True if connection is active (read-only)
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize DPM connection parameters.

        Args:
            host: DPM proxy hostname
            port: DPM proxy port
            timeout: Socket timeout in seconds for all operations

        Note:
            Connection is NOT established until connect() is called or
            the context manager is entered.
        """
        if not host:
            raise ValueError("host cannot be empty")
        if port <= 0 or port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {port}")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        self._host = host
        self._port = port
        self._timeout = timeout

        self._socket: Optional[socket.socket] = None
        self._list_id: Optional[int] = None
        self._recv_buffer = bytearray()
        self._connected = False

    @property
    def list_id(self) -> Optional[int]:
        """The DPM list ID from OpenList reply, or None if not connected."""
        return self._list_id

    @property
    def connected(self) -> bool:
        """True if the connection is active."""
        return self._connected and self._socket is not None

    def connect(self) -> None:
        """
        Establish connection to DPM server.

        Performs TCP connect, HTTP-style handshake, and extracts list_id
        from the OpenList response.

        Raises:
            DPMConnectionError: If connection fails or server returns HTTP error
            ValueError: If response is not a valid OpenList reply
        """
        if self._connected:
            raise DPMConnectionError("Already connected")

        try:
            # Create and connect socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self._timeout)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            logger.debug(f"Connecting to {self._host}:{self._port}")
            self._socket.connect((self._host, self._port))

            # Send HTTP-style handshake
            self._socket.sendall(DPM_HANDSHAKE)
            logger.debug("Sent DPM handshake")

            # Safe response detection: read first 4 bytes
            first_bytes = self._recv_exact(4)

            # Check if server returned HTTP error
            if first_bytes == b"HTTP":
                self._handle_http_error(first_bytes)

            # Otherwise, interpret as length prefix and read OpenList reply
            length = struct.unpack(">I", first_bytes)[0]
            if length == 0 or length > MAX_MESSAGE_SIZE:
                raise DPMConnectionError(f"Invalid message length: {length}")

            message_data = self._recv_exact(length)
            reply = self._unmarshal_reply(message_data)

            if not isinstance(reply, OpenList_reply):
                raise DPMConnectionError(f"Expected OpenList reply, got {type(reply).__name__}")

            self._list_id = reply.list_id
            self._connected = True

            logger.info(f"Connected to DPM at {self._host}:{self._port}, list_id={self._list_id}")

        except socket.error as e:
            self._cleanup_socket()
            raise DPMConnectionError(f"Failed to connect to {self._host}:{self._port}: {e}")
        except Exception:
            self._cleanup_socket()
            raise

    def _handle_http_error(self, first_bytes: bytes) -> None:
        """Handle HTTP error response from server."""
        assert self._socket is not None
        # Read rest of HTTP response line and body
        response_data = first_bytes
        try:
            # Read until we get enough data to see the HTTP status line
            # HTTP responses are text-based, read until double CRLF
            while b"\r\n\r\n" not in response_data and len(response_data) < 4096:
                chunk = self._socket.recv(1024)
                if not chunk:
                    break
                response_data += chunk
        except socket.timeout:
            pass
        except socket.error:
            pass

        # Extract HTTP status from response
        try:
            response_str = response_data.decode("utf-8", errors="replace")
            # Parse first line: "HTTP/1.1 404 Not Found"
            lines = response_str.split("\r\n")
            status_line = lines[0] if lines else "HTTP error (unknown status)"
        except Exception:
            status_line = "HTTP error (could not parse response)"

        raise DPMConnectionError(f"DPM server at {self._host}:{self._port} returned HTTP error: {status_line}")

    def close(self) -> None:
        """
        Close the connection and release resources.

        Safe to call multiple times. After close(), the connection
        cannot be reused.
        """
        self._connected = False
        self._cleanup_socket()
        self._list_id = None
        self._recv_buffer.clear()
        logger.debug("DPM connection closed")

    def _cleanup_socket(self) -> None:
        """Clean up socket resources."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def send_message(self, msg: Union[object, bytes]) -> None:
        """
        Send a PC-encoded message with length prefix.

        Args:
            msg: Either a protocol message object (with marshal() method)
                 or raw bytes to send

        Raises:
            DPMConnectionError: If not connected or send fails
        """
        if not self._connected or self._socket is None:
            raise DPMConnectionError("Not connected")

        # Marshal if it's a protocol message object
        if hasattr(msg, "marshal"):
            data = bytes(msg.marshal())  # type: ignore[call-arg]
        elif isinstance(msg, (bytes, bytearray)):
            data = bytes(msg)
        else:
            raise TypeError(f"msg must be a protocol message or bytes, got {type(msg).__name__}")

        # Send with length prefix
        length_prefix = struct.pack(">I", len(data))
        try:
            self._socket.sendall(length_prefix + data)
            logger.debug(f"Sent message: {len(data)} bytes")
        except socket.error as e:
            self._connected = False
            raise DPMConnectionError(f"Send failed: {e}")

    def recv_message(self, timeout: Optional[float] = None) -> object:
        """
        Receive and unmarshal a single reply message.

        Args:
            timeout: Optional timeout override for this operation.
                     If None, uses the connection's default timeout.

        Returns:
            Unmarshaled protocol reply object

        Raises:
            DPMConnectionError: If not connected or receive fails
            TimeoutError: If receive times out
        """
        if not self._connected or self._socket is None:
            raise DPMConnectionError("Not connected")

        # Save original timeout and set new one if specified
        original_timeout = self._socket.gettimeout()
        if timeout is not None:
            self._socket.settimeout(timeout)

        try:
            # Read length prefix (4 bytes)
            len_bytes = self._recv_exact(4)
            length = struct.unpack(">I", len_bytes)[0]

            if length == 0 or length > MAX_MESSAGE_SIZE:
                raise DPMConnectionError(f"Invalid message length: {length}")

            # Read message body
            data = self._recv_exact(length)

            # Unmarshal
            reply = self._unmarshal_reply(data)
            logger.debug(f"Received message: {type(reply).__name__}")
            return reply

        except socket.timeout:
            raise TimeoutError("Receive timeout")
        except socket.error as e:
            self._connected = False
            raise DPMConnectionError(f"Receive failed: {e}")
        finally:
            # Restore original timeout
            if timeout is not None and self._socket is not None:
                self._socket.settimeout(original_timeout)

    def _recv_exact(self, n: int) -> bytes:
        """
        Receive exactly n bytes from socket.

        Handles partial reads by looping until all bytes are received.
        Uses internal buffer to handle cases where more data is received
        than requested.

        Args:
            n: Number of bytes to receive

        Returns:
            Exactly n bytes

        Raises:
            DPMConnectionError: If connection is closed before all bytes received
            socket.timeout: If receive times out
            socket.error: If socket error occurs
        """
        assert self._socket is not None
        while len(self._recv_buffer) < n:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise DPMConnectionError("Connection closed by server")
            self._recv_buffer.extend(chunk)

        # Extract requested bytes and keep remainder in buffer
        data = bytes(self._recv_buffer[:n])
        self._recv_buffer = self._recv_buffer[n:]
        return data

    def _unmarshal_reply(self, data: bytes) -> object:
        """
        Unmarshal bytes into a protocol reply object.

        Args:
            data: Raw message bytes

        Returns:
            Unmarshaled protocol reply object

        Raises:
            DPMConnectionError: If unmarshaling fails
        """
        try:
            return unmarshal_reply(iter(data))
        except ProtocolError as e:
            raise DPMConnectionError(f"Protocol error: {e}")
        except StopIteration:
            raise DPMConnectionError("Unexpected end of message data")

    def __enter__(self) -> "DPMConnection":
        """Enter context manager - establishes connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager - closes connection."""
        self.close()
        return False

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"DPMConnection({self._host}:{self._port}, {status})"
