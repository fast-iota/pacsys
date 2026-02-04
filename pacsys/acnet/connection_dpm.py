"""
DPM (Data Pool Manager) connection implementation.

This module provides TCP-based access to Fermilab's Data Pool Manager
through acsys-proxy using the PC binary protocol.

Protocol Flow:
1. Connect to acsys-proxy.fnal.gov:6802 via TCP
2. Send HTTP-style handshake: GET /dpm HTTP/1.1\r\nContent-Type: application/pc\r\n\r\n
3. Receive OpenList reply with list_id (length-prefixed PC message)
4. Send AddToList requests for each device
5. Send StartList request to begin acquisition
6. Receive streaming data replies

Message Format:
- All messages after handshake use length-prefixed framing:
  [4 bytes length (big-endian)][PC-encoded message]
"""

import logging
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from pacsys.dpm_protocol import (
    AddToList_reply,
    AddToList_request,
    AnalogAlarm_reply,
    BasicStatus_reply,
    ClearList_request,
    DeviceInfo_reply,
    DigitalAlarm_reply,
    ListStatus_reply,
    OpenList_reply,
    ProtocolError,
    Raw_reply,
    Scalar_reply,
    ScalarArray_reply,
    StartList_reply,
    StartList_request,
    Status_reply,
    StopList_request,
    Text_reply,
    TextArray_reply,
    TimedScalarArray_reply,
    unmarshal_reply,
)

logger = logging.getLogger(__name__)

# Default DPM proxy
DPM_PROXY_HOST = "acsys-proxy.fnal.gov"
DPM_PROXY_PORT = 6802

# Connection timeouts
CONNECT_TIMEOUT = 10.0
RECV_TIMEOUT = 10.0

# HTTP-style handshake for DPM
DPM_HANDSHAKE = b"GET /dpm HTTP/1.1\r\nContent-Type: application/pc\r\n\r\n"


@dataclass
class DPMReading:
    """Data reading from DPM."""

    ref_id: int
    timestamp: int = 0
    cycle: int = 0
    status: int = 0
    data: object = None
    device_info: Optional["DPMDeviceInfo"] = None
    micros: object = None  # Per-sample timestamps from TimedScalarArray (int64 list)


@dataclass
class DPMDeviceInfo:
    """Device metadata from DPM."""

    ref_id: int
    di: int = 0
    name: str = ""
    description: str = ""
    units: str = ""
    format_hint: int = 0


class DPMError(Exception):
    """DPM-specific error."""

    def __init__(self, status: int, message: str = ""):
        self.status = status
        self.message = message
        super().__init__(f"DPM error {status}: {message}" if message else f"DPM error {status}")


class DPMConnection:
    """
    TCP-based connection to DPM server via acsys-proxy.

    Uses the PC binary protocol over HTTP-style handshake.

    Example usage:
        with DPMConnection() as dpm:
            # Read a single device immediately
            reading = dpm.read("M:OUTTMP")
            print(f"Value: {reading.data}")

            # Or with periodic updates
            dpm.add_request("M:OUTTMP@p,1000", callback=handle_data)
            dpm.start()
            time.sleep(5)
            dpm.stop()
    """

    def __init__(self, host: str = DPM_PROXY_HOST, port: int = DPM_PROXY_PORT):
        """
        Initialize DPM connection.

        Args:
            host: DPM proxy hostname (default: acsys-proxy.fnal.gov)
            port: DPM proxy port (default: 6802)
        """
        self._host = host
        self._port = port

        # Socket
        self._socket: Optional[socket.socket] = None
        self._socket_lock = threading.Lock()

        # State
        self._connected = False
        self._disposed = False
        self._running = False
        self._list_id: Optional[int] = None

        # Request tracking
        self._ref_counter = 0
        self._ref_counter_lock = threading.Lock()
        self._requests: dict[int, dict] = {}  # ref_id -> {drf, callback, ...}
        self._device_info: dict[int, DPMDeviceInfo] = {}  # ref_id -> device info

        # Read thread
        self._read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Receive buffer
        self._recv_buffer = bytearray()

    @property
    def connected(self) -> bool:
        """Check if connected to DPM."""
        return self._connected and not self._disposed

    @property
    def list_id(self) -> Optional[int]:
        """Get the current list ID."""
        return self._list_id

    def connect(self):
        """
        Connect to the DPM proxy.

        Raises:
            DPMError: If connection fails
        """
        if self._disposed:
            raise DPMError(-1, "Connection disposed")

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(CONNECT_TIMEOUT)
            self._socket.connect((self._host, self._port))
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Send HTTP-style handshake
            self._socket.sendall(DPM_HANDSHAKE)

            # Receive OpenList reply
            reply = self._recv_reply()
            if not isinstance(reply, OpenList_reply):
                raise DPMError(-1, f"Expected OpenList reply, got {type(reply).__name__}")

            self._list_id = reply.list_id
            self._connected = True

            logger.info(f"Connected to DPM at {self._host}:{self._port}, list_id={self._list_id}")

        except socket.error as e:
            logger.error(f"Failed to connect to DPM: {e}")
            raise DPMError(-1, f"Connection failed: {e}")

    def close(self):
        """Close the connection and clean up resources."""
        self._disposed = True
        self._stop_event.set()

        # Stop acquisition if running
        if self._running and self._socket:
            try:
                self._send_stop_list()
            except Exception:
                pass

        # Close socket
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        # Wait for read thread
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)

        self._connected = False
        logger.info("Closed DPM connection")

    def _next_ref_id(self) -> int:
        """Get next reference ID for a request."""
        with self._ref_counter_lock:
            self._ref_counter += 1
            return self._ref_counter

    def _send_message(self, data: bytes):
        """Send a length-prefixed message."""
        assert self._socket is not None, "not connected"
        length = struct.pack(">I", len(data))
        with self._socket_lock:
            self._socket.sendall(length + data)

    def _recv_exact(self, n: int, timeout: float | None = None) -> bytes:
        """Receive exactly n bytes from socket."""
        assert self._socket is not None, "not connected"
        if timeout:
            self._socket.settimeout(timeout)
        try:
            while len(self._recv_buffer) < n:
                chunk = self._socket.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection closed")
                self._recv_buffer.extend(chunk)
        finally:
            if timeout:
                self._socket.settimeout(RECV_TIMEOUT)

        data = bytes(self._recv_buffer[:n])
        del self._recv_buffer[:n]
        return data

    def _recv_reply(self, timeout: float | None = None) -> object:
        """Receive and unmarshal a single reply."""
        # Read length prefix
        len_bytes = self._recv_exact(4, timeout)
        length = struct.unpack(">I", len_bytes)[0]

        if length == 0 or length > 1024 * 1024:
            raise DPMError(-1, f"Invalid message length: {length}")

        # Read message
        data = self._recv_exact(length, timeout)

        # Unmarshal
        try:
            return unmarshal_reply(iter(data))
        except ProtocolError as e:
            raise DPMError(-1, f"Protocol error: {e}")

    def _send_add_to_list(self, ref_id: int, drf_request: str):
        """Send AddToList request."""
        msg = AddToList_request()
        msg.list_id = self._list_id
        msg.ref_id = ref_id
        msg.drf_request = drf_request

        data = bytes(msg.marshal())
        self._send_message(data)

    def _send_start_list(self, model: str | None = None):
        """Send StartList request."""
        msg = StartList_request()
        msg.list_id = self._list_id
        if model:
            msg.model = model

        data = bytes(msg.marshal())
        self._send_message(data)

    def _send_stop_list(self):
        """Send StopList request."""
        msg = StopList_request()
        msg.list_id = self._list_id

        data = bytes(msg.marshal())
        self._send_message(data)

    def _send_clear_list(self):
        """Send ClearList request."""
        msg = ClearList_request()
        msg.list_id = self._list_id

        data = bytes(msg.marshal())
        self._send_message(data)

    def add_request(self, drf_request: str, callback: Optional[Callable[[DPMReading], None]] = None) -> int:
        """
        Add a device request to the list.

        Args:
            drf_request: DRF3 format device request string (e.g., "M:OUTTMP@p,1000")
            callback: Optional callback for data updates

        Returns:
            Reference ID for this request
        """
        ref_id = self._next_ref_id()

        self._requests[ref_id] = {
            "drf": drf_request,
            "callback": callback,
        }

        self._send_add_to_list(ref_id, drf_request)
        logger.debug(f"Added request {ref_id}: {drf_request}")

        return ref_id

    def start(self, model: str | None = None):
        """
        Start data acquisition.

        Args:
            model: Optional model parameter for StartList
        """
        if self._running:
            return

        # Set running flag BEFORE starting thread
        self._running = True
        self._stop_event.clear()

        # Start read thread
        self._read_thread = threading.Thread(target=self._read_thread_run, name="DPM-read", daemon=True)
        self._read_thread.start()

        # Send start request
        self._send_start_list(model)
        logger.debug("Started DPM acquisition")

    def stop(self):
        """Stop data acquisition."""
        if not self._running:
            return

        self._running = False

        try:
            self._send_stop_list()
        except Exception as e:
            logger.warning(f"Error stopping list: {e}")

        self._stop_event.set()

        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)

        logger.debug("Stopped DPM acquisition")

    def clear(self):
        """Clear all requests from the list."""
        if self._running:
            self.stop()

        self._send_clear_list()
        self._requests.clear()
        self._device_info.clear()
        logger.debug("Cleared DPM list")

    def read(self, drf_request: str, timeout: float = 5.0) -> DPMReading:
        """
        Read a device value synchronously (one-shot).

        Args:
            drf_request: DRF3 format device request (e.g., "M:OUTTMP")
            timeout: Read timeout in seconds

        Returns:
            DPMReading with the device value

        Raises:
            DPMError: If read fails
            TimeoutError: If read times out
        """
        # Add @I for immediate if no event specified
        if "@" not in drf_request:
            drf_request = f"{drf_request}@I"

        result_event = threading.Event()
        result_holder = {"reading": None}

        def handle_result(reading: DPMReading):
            result_holder["reading"] = reading
            result_event.set()

        # Add request with callback
        self.add_request(drf_request, callback=handle_result)

        # Start if not already running
        was_running = self._running
        if not was_running:
            self.start()

        try:
            # Wait for result
            if not result_event.wait(timeout=timeout):
                raise TimeoutError(f"Timeout reading {drf_request}")

            reading = result_holder["reading"]
            if reading is None:
                raise DPMError(-1, f"No reading received for {drf_request}")
            if reading.status != 0:
                raise DPMError(reading.status, f"Error reading {drf_request}")

            return reading

        finally:
            if not was_running:
                self.stop()

    def _read_thread_run(self):
        """Read thread main loop."""
        logger.debug("DPM read thread started")
        assert self._socket is not None, "not connected"

        self._socket.settimeout(RECV_TIMEOUT)

        while not self._stop_event.is_set() and self._running:
            try:
                reply = self._recv_reply(timeout=RECV_TIMEOUT)
                self._handle_reply(reply)

            except socket.timeout:
                continue
            except ConnectionError as e:
                if not self._stop_event.is_set():
                    logger.warning(f"DPM connection error: {e}")
                break
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.warning(f"DPM read error: {e}")
                break

        logger.debug("DPM read thread stopped")

    def _handle_reply(self, reply):
        """Handle a received reply."""
        if isinstance(reply, ListStatus_reply):
            # Heartbeat - connection is alive
            logger.debug(f"ListStatus: list_id={reply.list_id}, status={reply.status}")
            return

        if isinstance(reply, StartList_reply):
            logger.debug(f"StartList: list_id={reply.list_id}, status={reply.status}")
            if reply.status != 0:
                logger.warning(f"StartList failed with status {reply.status}")
            return

        if isinstance(reply, AddToList_reply):
            logger.debug(f"AddToList: ref_id={reply.ref_id}, status={reply.status}")
            if reply.status != 0:
                req = self._requests.get(reply.ref_id)
                if req and req.get("callback"):
                    reading = DPMReading(ref_id=reply.ref_id, status=reply.status)
                    req["callback"](reading)
            return

        # Data replies
        if isinstance(
            reply,
            (
                Scalar_reply,
                ScalarArray_reply,
                Raw_reply,
                Text_reply,
                TextArray_reply,
                Status_reply,
                TimedScalarArray_reply,
            ),
        ):
            micros = None
            if isinstance(reply, TimedScalarArray_reply) and hasattr(reply, "micros") and reply.micros:
                micros = reply.micros
            reading = DPMReading(
                ref_id=reply.ref_id,
                timestamp=reply.timestamp,
                cycle=reply.cycle,
                status=reply.status,
                data=reply.data if hasattr(reply, "data") else None,
                micros=micros,
            )

            # Attach device info if available
            if reply.ref_id in self._device_info:
                reading.device_info = self._device_info[reply.ref_id]

            # Call callback
            req = self._requests.get(reply.ref_id)
            if req and req.get("callback"):
                req["callback"](reading)

            logger.debug(f"Data reply: ref_id={reply.ref_id}, data={reading.data}")
            return

        if isinstance(reply, DeviceInfo_reply):
            info = DPMDeviceInfo(
                ref_id=reply.ref_id,
                di=reply.di,
                name=reply.name,
                description=reply.description,
                units=getattr(reply, "units", ""),
                format_hint=getattr(reply, "format_hint", 0),
            )

            self._device_info[reply.ref_id] = info
            logger.debug(f"DeviceInfo: {info.name} ({info.description})")
            return

        if isinstance(reply, AnalogAlarm_reply):
            data = {
                "minimum": reply.minimum,
                "maximum": reply.maximum,
                "alarm_enable": reply.alarm_enable,
                "alarm_status": reply.alarm_status,
                "abort": reply.abort,
                "abort_inhibit": reply.abort_inhibit,
                "tries_needed": reply.tries_needed,
                "tries_now": reply.tries_now,
            }
            reading = DPMReading(
                ref_id=reply.ref_id,
                timestamp=reply.timestamp,
                cycle=reply.cycle,
                data=data,
            )
            # Attach device info if available
            if reply.ref_id in self._device_info:
                reading.device_info = self._device_info[reply.ref_id]

            req = self._requests.get(reply.ref_id)
            if req and req.get("callback"):
                req["callback"](reading)

            logger.debug(f"AnalogAlarm reply: ref_id={reply.ref_id}")
            return

        if isinstance(reply, DigitalAlarm_reply):
            data = {
                "nominal": reply.nominal,
                "mask": reply.mask,
                "alarm_enable": reply.alarm_enable,
                "alarm_status": reply.alarm_status,
                "abort": reply.abort,
                "abort_inhibit": reply.abort_inhibit,
                "tries_needed": reply.tries_needed,
                "tries_now": reply.tries_now,
            }
            reading = DPMReading(
                ref_id=reply.ref_id,
                timestamp=reply.timestamp,
                cycle=reply.cycle,
                data=data,
            )
            # Attach device info if available
            if reply.ref_id in self._device_info:
                reading.device_info = self._device_info[reply.ref_id]

            req = self._requests.get(reply.ref_id)
            if req and req.get("callback"):
                req["callback"](reading)

            logger.debug(f"DigitalAlarm reply: ref_id={reply.ref_id}")
            return

        if isinstance(reply, BasicStatus_reply):
            # BasicStatus has optional fields
            data = {}
            if hasattr(reply, "on"):
                data["on"] = reply.on
            if hasattr(reply, "ready"):
                data["ready"] = reply.ready
            if hasattr(reply, "remote"):
                data["remote"] = reply.remote
            if hasattr(reply, "positive"):
                data["positive"] = reply.positive
            if hasattr(reply, "ramp"):
                data["ramp"] = reply.ramp

            reading = DPMReading(
                ref_id=reply.ref_id,
                timestamp=reply.timestamp,
                cycle=reply.cycle,
                data=data,
            )
            # Attach device info if available
            if reply.ref_id in self._device_info:
                reading.device_info = self._device_info[reply.ref_id]

            req = self._requests.get(reply.ref_id)
            if req and req.get("callback"):
                req["callback"](reading)

            logger.debug(f"BasicStatus reply: ref_id={reply.ref_id}")
            return

        logger.debug(f"Unhandled reply type: {type(reply).__name__}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
