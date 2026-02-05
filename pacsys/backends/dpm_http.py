"""
DPM HTTP Backend - primary backend for ACNET device access.

Uses TCP/PC protocol via acsys-proxy. Connection pool for reads,
independent TCP connections per subscribe() for streaming.
See SPECIFICATION.md for protocol details.
"""

import asyncio
import logging
import queue
import struct
import threading
import time
from typing import Iterator, Optional

import numpy as np

from pacsys.acnet.errors import (
    ERR_OK,
    ERR_RETRY,
    ERR_TIMEOUT,
    FACILITY_ACNET,
    parse_error,
    status_message,
)
from pacsys.auth import Auth, KerberosAuth
from pacsys.backends import Backend, timestamp_from_millis
import socket

from pacsys.dpm_connection import DPM_HANDSHAKE, MAX_MESSAGE_SIZE, DPMConnection, DPMConnectionError
from pacsys.errors import AuthenticationError, DeviceError
from pacsys.pool import ConnectionPool
from pacsys.types import (
    BackendCapability,
    DeviceMeta,
    ErrorCallback,
    Reading,
    ReadingCallback,
    SubscriptionHandle,
    Value,
    ValueType,
    WriteResult,
)

from pacsys.dpm_protocol import (
    AddToList_reply,
    AddToList_request,
    AnalogAlarm_reply,
    ApplySettings_reply,
    ApplySettings_request,
    Authenticate_reply,
    Authenticate_request,
    BasicStatus_reply,
    ClearList_request,
    DeviceInfo_reply,
    DigitalAlarm_reply,
    EnableSettings_request,
    ListStatus_reply,
    OpenList_reply,
    ProtocolError,
    Raw_reply,
    RawSetting_struct,
    Scalar_reply,
    ScalarArray_reply,
    ScaledSetting_struct,
    StartList_reply,
    StartList_request,
    Status_reply,
    StopList_request,
    Text_reply,
    TextArray_reply,
    TextSetting_struct,
    TimedScalarArray_reply,
    unmarshal_reply,
)
from pacsys.drf_utils import ensure_immediate_event, prepare_for_write

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_HOST = "acsys-proxy.fnal.gov"
DEFAULT_PORT = 6802
DEFAULT_POOL_SIZE = 4
DEFAULT_TIMEOUT = 5.0
_DEFAULT_QUEUE_MAXSIZE = 10000

# Kerberos service principal for DPM


def _reply_to_value_and_type(reply) -> tuple[Optional[Value], Optional[ValueType]]:
    """Extract value and type from a DPM data reply."""
    if isinstance(reply, Scalar_reply):
        return reply.data, ValueType.SCALAR
    elif isinstance(reply, ScalarArray_reply):
        return np.array(reply.data), ValueType.SCALAR_ARRAY
    elif isinstance(reply, TimedScalarArray_reply):
        data = np.array(reply.data)
        if hasattr(reply, "micros") and reply.micros:
            micros = np.array(reply.micros, dtype=np.int64)
            return {"data": data, "micros": micros}, ValueType.TIMED_SCALAR_ARRAY
        return data, ValueType.SCALAR_ARRAY
    elif isinstance(reply, Raw_reply):
        return bytes(reply.data), ValueType.RAW
    elif isinstance(reply, Text_reply):
        return reply.data, ValueType.TEXT
    elif isinstance(reply, TextArray_reply):
        return list(reply.data), ValueType.TEXT_ARRAY
    elif isinstance(reply, AnalogAlarm_reply):
        return {
            "minimum": reply.minimum,
            "maximum": reply.maximum,
            "alarm_enable": reply.alarm_enable,
            "alarm_status": reply.alarm_status,
            "abort": reply.abort,
            "abort_inhibit": reply.abort_inhibit,
            "tries_needed": reply.tries_needed,
            "tries_now": reply.tries_now,
        }, ValueType.ANALOG_ALARM
    elif isinstance(reply, DigitalAlarm_reply):
        return {
            "nominal": reply.nominal,
            "mask": reply.mask,
            "alarm_enable": reply.alarm_enable,
            "alarm_status": reply.alarm_status,
            "abort": reply.abort,
            "abort_inhibit": reply.abort_inhibit,
            "tries_needed": reply.tries_needed,
            "tries_now": reply.tries_now,
        }, ValueType.DIGITAL_ALARM
    elif isinstance(reply, BasicStatus_reply):
        status_dict = {}
        if hasattr(reply, "on"):
            status_dict["on"] = reply.on
        if hasattr(reply, "ready"):
            status_dict["ready"] = reply.ready
        if hasattr(reply, "remote"):
            status_dict["remote"] = reply.remote
        if hasattr(reply, "positive"):
            status_dict["positive"] = reply.positive
        if hasattr(reply, "ramp"):
            status_dict["ramp"] = reply.ramp
        return status_dict, ValueType.BASIC_STATUS
    elif isinstance(reply, Status_reply):
        return None, ValueType.SCALAR

    logger.error(f"Unknown reply type: {type(reply).__name__}, cannot extract value")
    return None, None


def _device_info_to_meta(info: DeviceInfo_reply) -> DeviceMeta:
    """Convert DeviceInfo_reply to DeviceMeta."""
    return DeviceMeta(
        device_index=info.di,
        name=info.name,
        description=info.description,
        units=getattr(info, "units", None),
        format_hint=getattr(info, "format_hint", None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal Streaming Classes
# ─────────────────────────────────────────────────────────────────────────────


class _AsyncDPMConnection:
    """Async TCP connection to DPM server for streaming subscriptions.

    Uses asyncio StreamReader/StreamWriter for non-blocking I/O.
    Handles partial packets natively via readexactly().
    """

    # DPM server sends ListStatus_reply heartbeats every ~2s.
    # If no data arrives within this window, the connection is presumed dead.
    _RECV_TIMEOUT = 10.0

    def __init__(self, host: str, port: int, timeout: float = DEFAULT_TIMEOUT):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._list_id: Optional[int] = None

    @property
    def list_id(self) -> Optional[int]:
        return self._list_id

    async def connect(self) -> None:
        """Connect to DPM, send handshake, read OpenList_reply."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port, limit=MAX_MESSAGE_SIZE),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            raise DPMConnectionError(f"Connection to {self._host}:{self._port} timed out")

        # Set TCP_NODELAY and SO_KEEPALIVE on the underlying socket
        sock = self._writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        self._writer.write(DPM_HANDSHAKE)
        await self._writer.drain()

        # Read OpenList reply (same detection as sync: first 4 bytes)
        first_bytes = await self._reader.readexactly(4)
        if first_bytes == b"HTTP":
            # Read rest of HTTP status line for useful error message
            try:
                rest = await asyncio.wait_for(self._reader.readline(), timeout=2.0)
                status_line = "HTTP" + rest.decode("utf-8", errors="replace").rstrip()
            except Exception:
                status_line = "HTTP error (could not read status)"
            raise DPMConnectionError(f"DPM server at {self._host}:{self._port} returned HTTP error: {status_line}")

        length = struct.unpack(">I", first_bytes)[0]
        if length == 0 or length > MAX_MESSAGE_SIZE:
            raise DPMConnectionError(f"Invalid message length: {length}")

        data = await self._reader.readexactly(length)
        try:
            reply = unmarshal_reply(iter(data))
        except (ProtocolError, StopIteration) as e:
            raise DPMConnectionError(f"Protocol error during handshake: {e}")

        if not isinstance(reply, OpenList_reply):
            raise DPMConnectionError(f"Expected OpenList reply, got {type(reply).__name__}")
        self._list_id = reply.list_id

    async def send_message(self, msg) -> None:
        """Send a length-prefixed SDD message."""
        assert self._writer is not None
        if hasattr(msg, "marshal"):
            data = bytes(msg.marshal())
        else:
            data = bytes(msg)
        self._writer.write(struct.pack(">I", len(data)) + data)
        await self._writer.drain()

    async def send_messages_batch(self, msgs: list) -> None:
        """Send multiple length-prefixed messages in a single TCP write."""
        assert self._writer is not None
        buf = bytearray()
        for msg in msgs:
            data = bytes(msg.marshal()) if hasattr(msg, "marshal") else bytes(msg)
            buf.extend(struct.pack(">I", len(data)))
            buf.extend(data)
        self._writer.write(buf)
        await self._writer.drain()

    async def recv_message(self):
        """Receive and unmarshal one reply. Handles partial packets natively.

        Uses a read timeout to detect silent connection drops. The DPM server
        sends ListStatus_reply heartbeats every ~2s, so if nothing arrives
        within _RECV_TIMEOUT seconds, the connection is presumed dead.
        """
        assert self._reader is not None
        try:
            len_bytes = await asyncio.wait_for(self._reader.readexactly(4), timeout=self._RECV_TIMEOUT)
        except asyncio.TimeoutError:
            raise DPMConnectionError(
                f"No data received for {self._RECV_TIMEOUT}s (missed heartbeats), connection presumed dead"
            )
        length = struct.unpack(">I", len_bytes)[0]
        if length == 0 or length > MAX_MESSAGE_SIZE:
            raise DPMConnectionError(f"Invalid message length: {length}")
        data = await self._reader.readexactly(length)
        try:
            return unmarshal_reply(iter(data))
        except (ProtocolError, StopIteration) as e:
            raise DPMConnectionError(f"Protocol error: {e}")

    async def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None


class _WriteConnection:
    """An authenticated DPM connection for write operations.

    Authentication persists at the DPM list level. Once authenticated and
    EnableSettings is called, the connection can be reused for multiple
    write operations by clearing and restarting the list.

    Reuse flow: StopList -> ClearList -> AddToList -> StartList -> ApplySettings
    """

    def __init__(self, conn: DPMConnection, principal: str, role: str):
        self.conn = conn
        self.principal = principal
        self.role = role
        self.authenticated = False
        self.last_used = time.monotonic()

    def is_stale(self, max_idle: float = 60.0) -> bool:
        """Check if connection has been idle too long."""
        return time.monotonic() - self.last_used > max_idle

    def close(self) -> None:
        """Close the underlying connection."""
        try:
            self.conn.close()
        except Exception:
            pass


class _DPMHTTPSubscriptionHandle(SubscriptionHandle):
    """Subscription handle for DPMHTTPBackend.

    Each handle corresponds to one async task with its own TCP connection.
    """

    def __init__(
        self,
        backend: "DPMHTTPBackend",
        drfs: list[str],
        callback: Optional[ReadingCallback],
        on_error: Optional[ErrorCallback] = None,
    ):
        self._backend = backend
        self._drfs = drfs
        self._callback = callback
        self._is_callback_mode = callback is not None
        self._on_error = on_error
        self._queue: queue.Queue[Reading] = queue.Queue(maxsize=_DEFAULT_QUEUE_MAXSIZE)
        self._stopped = False
        self._exc: Optional[Exception] = None
        self._ref_ids: list[int] = list(range(1, len(drfs) + 1))
        self._task: Optional[asyncio.Task] = None

    @property
    def ref_ids(self) -> list[int]:
        """Reference IDs for devices in this subscription."""
        return list(self._ref_ids)

    @property
    def stopped(self) -> bool:
        """True if this subscription has been stopped."""
        return self._stopped

    @property
    def exc(self) -> Optional[Exception]:
        """Exception if an error occurred, else None."""
        return self._exc

    def readings(
        self,
        timeout: Optional[float] = None,
    ) -> Iterator[tuple[Reading, SubscriptionHandle]]:
        """Yield (reading, handle) pairs for this subscription.

        Args:
            timeout: Seconds to wait for readings.
                    None = block forever (until stop() called)
                    0 = non-blocking (drain buffered readings only)

        Yields:
            (reading, handle) pairs

        Raises:
            RuntimeError: If subscription has a callback
            Exception: If connection error occurred
        """
        if self._is_callback_mode:
            raise RuntimeError("Cannot iterate subscription with callback")

        start_time = time.monotonic()

        while not self._stopped:
            if self._exc is not None:
                raise self._exc

            try:
                reading = self._queue.get(timeout=0.1)
                yield (reading, self)
            except queue.Empty:
                if self._exc is not None:
                    raise self._exc

                if self._stopped:
                    break

                if timeout == 0:
                    break
                elif timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        break
                continue

    def stop(self) -> None:
        """Stop this subscription and cancel its async task."""
        if not self._stopped:
            self._backend.remove(self)


class DPMHTTPBackend(Backend):
    """
    DPM HTTP Backend for ACNET device access.

    Uses TCP/HTTP protocol to communicate with DPM via acsys-proxy.
    Supports multiple independent streaming subscriptions, each with its
    own async TCP connection on a shared asyncio reactor thread.

    Design note: RemoveFromList is not implemented. Each subscription gets
    its own TCP connection with an independent DPM list, so partial device
    removal has no use case -- call remove(handle) to tear down an entire
    subscription instead.

    Capabilities:
        - READ: Always enabled
        - WRITE: Enabled when auth is KerberosAuth and role is set
        - STREAM: Always enabled (multiple independent subscriptions)
        - AUTH_KERBEROS: Enabled when auth is KerberosAuth
        - BATCH: Always enabled (get_many)
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        pool_size: int = DEFAULT_POOL_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        auth: Optional[Auth] = None,
        role: Optional[str] = None,
    ):
        """
        Initialize DPM HTTP backend.

        Args:
            host: DPM proxy hostname (default: acsys-proxy.fnal.gov)
            port: DPM proxy port (default: 6802)
            pool_size: Connection pool size for reads (default: 4)
            timeout: Default operation timeout in seconds (default: 5.0)
            auth: Authentication object (KerberosAuth for writes)
            role: Role for authenticated operations (e.g., "testing")
        """
        if not host:
            raise ValueError("host cannot be empty")
        if port <= 0 or port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {port}")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if auth is not None and not isinstance(auth, KerberosAuth):
            raise ValueError(f"auth must be KerberosAuth or None, got {type(auth).__name__}")

        self._host = host
        self._port = port
        self._pool_size = pool_size
        self._timeout = timeout
        self._auth: Optional[KerberosAuth] = auth
        self._role = role
        self._pool: Optional[ConnectionPool] = None
        self._pool_lock = threading.Lock()
        self._closed = False

        # Streaming state -- asyncio reactor (matches gRPC backend pattern)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reactor_thread: Optional[threading.Thread] = None
        self._reactor_lock = threading.Lock()
        self._handles: list[_DPMHTTPSubscriptionHandle] = []
        self._handles_lock = threading.Lock()

        # Write connection pool - authenticated connections for writes
        # Authentication persists at list level; reuse via StopList + ClearList
        self._write_connections: list[_WriteConnection] = []
        self._write_pool_size = 2  # Max authenticated write connections
        self._write_lock = threading.Lock()
        self._write_idle_timeout = 60.0  # Close connections idle > 60s

        # Validate auth eagerly (fail fast)
        if self._auth is not None:
            _ = self._auth.principal  # This validates credentials

        logger.debug(
            f"DPMHTTPBackend initialized: host={host}, port={port}, "
            f"pool_size={pool_size}, timeout={timeout}, auth={type(auth).__name__ if auth else None}, role={role}"
        )

    @property
    def capabilities(self) -> BackendCapability:
        """Backend capabilities based on configuration."""
        caps = BackendCapability.READ | BackendCapability.BATCH | BackendCapability.STREAM

        if isinstance(self._auth, KerberosAuth):
            caps |= BackendCapability.AUTH_KERBEROS
            if self._role is not None:
                caps |= BackendCapability.WRITE

        return caps

    @property
    def authenticated(self) -> bool:
        """True if backend is configured for authenticated operations."""
        return self._auth is not None

    @property
    def principal(self) -> Optional[str]:
        """Principal name if authenticated, else None."""
        if self._auth is not None:
            return self._auth.principal
        return None

    @property
    def host(self) -> str:
        """DPM proxy hostname."""
        return self._host

    @property
    def port(self) -> int:
        """DPM proxy port."""
        return self._port

    @property
    def pool_size(self) -> int:
        """Connection pool size."""
        return self._pool_size

    @property
    def timeout(self) -> float:
        """Default operation timeout."""
        return self._timeout

    def _get_pool(self) -> ConnectionPool:
        """Get or create the connection pool (lazy initialization with double-checked locking)."""
        if self._closed:
            raise RuntimeError("Backend is closed")

        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    self._pool = ConnectionPool(
                        host=self._host,
                        port=self._port,
                        pool_size=self._pool_size,
                        timeout=self._timeout,
                    )
        pool = self._pool
        assert pool is not None
        return pool

    # ─────────────────────────────────────────────────────────────────────────
    # Read Methods
    # ─────────────────────────────────────────────────────────────────────────

    def read(self, drf: str, timeout: Optional[float] = None) -> Value:
        """Read a single device value."""
        reading = self.get(drf, timeout=timeout)

        if not reading.ok:
            raise DeviceError(
                drf=reading.drf,
                facility_code=reading.facility_code,
                error_code=reading.error_code,
                message=reading.message,
            )

        assert reading.value is not None
        return reading.value

    def get(self, drf: str, timeout: Optional[float] = None) -> Reading:
        """Read a single device with full metadata."""
        readings = self.get_many([drf], timeout=timeout)
        return readings[0]

    def get_many(self, drfs: list[str], timeout: Optional[float] = None) -> list[Reading]:
        """Read multiple devices in a single batch."""
        if not drfs:
            return []

        effective_timeout = timeout if timeout is not None else self._timeout
        deadline = time.monotonic() + effective_timeout

        prepared_drfs = [ensure_immediate_event(drf) for drf in drfs]

        device_infos: dict[int, DeviceInfo_reply] = {}
        data_replies: dict[int, object] = {}
        add_errors: dict[int, AddToList_reply] = {}  # ref_id -> failed AddToList
        received_count = 0
        expected_count = len(drfs)

        pool = self._get_pool()
        conn_broken = False

        with pool.connection() as conn:
            list_id = conn.list_id

            # Pipeline: batch all AddToList + StartList into a single TCP send
            setup_msgs = []
            for i, drf in enumerate(prepared_drfs):
                add_req = AddToList_request()
                add_req.list_id = list_id
                add_req.ref_id = i + 1
                add_req.drf_request = drf
                setup_msgs.append(add_req)

            start_req = StartList_request()
            start_req.list_id = list_id
            setup_msgs.append(start_req)
            conn.send_messages_batch(setup_msgs)

            try:
                while received_count < expected_count:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break

                    try:
                        reply = conn.recv_message(timeout=min(remaining, 2.0))
                    except TimeoutError:
                        if time.monotonic() >= deadline:
                            break
                        continue

                    if isinstance(reply, AddToList_reply):
                        if reply.status != 0:
                            add_errors[reply.ref_id] = reply
                            received_count += 1
                    elif isinstance(reply, DeviceInfo_reply):
                        device_infos[reply.ref_id] = reply
                    elif isinstance(reply, StartList_reply):
                        if reply.status != 0:
                            logger.warning(f"StartList returned status {reply.status}")
                            break  # No data will arrive
                    elif isinstance(reply, ListStatus_reply):
                        pass
                    elif isinstance(reply, Status_reply):
                        if reply.ref_id not in data_replies:
                            data_replies[reply.ref_id] = reply
                            received_count += 1
                    elif hasattr(reply, "ref_id"):
                        if reply.ref_id not in data_replies:
                            data_replies[reply.ref_id] = reply
                            received_count += 1
            except (BrokenPipeError, ConnectionResetError, OSError):
                conn_broken = True
                raise
            finally:
                if not conn_broken:
                    try:
                        stop_req = StopList_request()
                        stop_req.list_id = list_id
                        clear_req = ClearList_request()
                        clear_req.list_id = list_id
                        conn.send_messages_batch([stop_req, clear_req])
                    except Exception:
                        pass  # Best-effort cleanup on broken connection

        readings: list[Reading] = []

        for i, original_drf in enumerate(drfs):
            ref_id = i + 1
            info = device_infos.get(ref_id)
            reply = data_replies.get(ref_id)
            add_err = add_errors.get(ref_id)

            meta = _device_info_to_meta(info) if info else None

            if add_err is not None:
                # AddToList failed for this device (bad DRF, etc.)
                facility, error = parse_error(add_err.status)
                readings.append(
                    Reading(
                        drf=original_drf,
                        value_type=ValueType.SCALAR,
                        tag=ref_id,
                        facility_code=facility,
                        error_code=error,
                        value=None,
                        message=status_message(facility, error) or f"AddToList failed (status={add_err.status})",
                        timestamp=None,
                        cycle=0,
                        meta=meta,
                    )
                )
            elif reply is None:
                readings.append(
                    Reading(
                        drf=original_drf,
                        value_type=ValueType.SCALAR,
                        tag=ref_id,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_TIMEOUT,
                        value=None,
                        message="Request timeout",
                        timestamp=None,
                        cycle=0,
                        meta=meta,
                    )
                )
            else:
                readings.append(self._reply_to_reading(reply, original_drf, meta))

        return readings

    # ─────────────────────────────────────────────────────────────────────────
    # Write Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _authenticate_connection(self, conn) -> tuple[bytes, bytes]:
        """Authenticate a connection via Kerberos GSSAPI.

        Two-phase protocol:
        1. Send empty token to request service name from DPM server
        2. Server replies with its Kerberos service name (e.g. "dpm@<host>")
        3. Create GSSAPI context targeting that service, send initial token
        4. Server accepts, optional mutual-auth token exchange
        """
        import gssapi

        # Phase 1: request service name with empty token
        auth_req = Authenticate_request()
        auth_req.list_id = conn.list_id
        auth_req.token = b""
        conn.send_message(auth_req)

        reply = conn.recv_message(timeout=self._timeout)
        if not isinstance(reply, Authenticate_reply):
            raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        raw_service_name = reply.serviceName
        if not raw_service_name:
            raise AuthenticationError("Server did not provide a service name")

        # Server sends Java GSS-API format: "daeset/bd@host" (with possible \ escaping)
        # Translate @ → /, strip \, append explicit realm
        gss_name = raw_service_name.translate({ord("@"): "/", ord("\\"): None}) + "@FNAL.GOV"
        logger.debug(f"DPM service name: {gss_name}")

        # Phase 2: create GSSAPI context with server's actual service name
        service_name = gssapi.Name(gss_name, gssapi.NameType.kerberos_principal)

        creds = gssapi.creds.Credentials(usage="initiate")
        ctx = gssapi.SecurityContext(
            name=service_name,
            usage="initiate",
            creds=creds,
            flags=[  # type: ignore[arg-type]  # no stubs for gssapi
                gssapi.RequirementFlag.replay_detection,
                gssapi.RequirementFlag.integrity,
                gssapi.RequirementFlag.out_of_sequence_detection,
            ],
            mech=gssapi.MechType.kerberos,
        )

        token = ctx.step()

        auth_req = Authenticate_request()
        auth_req.list_id = conn.list_id
        auth_req.token = bytes(token) if token else b""
        conn.send_message(auth_req)

        reply = conn.recv_message(timeout=self._timeout)
        if not isinstance(reply, Authenticate_reply):
            raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        if hasattr(reply, "token") and reply.token and not ctx.complete:
            token = ctx.step(reply.token)
            if token:
                auth_req = Authenticate_request()
                auth_req.list_id = conn.list_id
                auth_req.token = bytes(token)
                conn.send_message(auth_req)

                reply = conn.recv_message(timeout=self._timeout)
                if not isinstance(reply, Authenticate_reply):
                    raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        if not ctx.complete:
            raise AuthenticationError("Kerberos authentication incomplete")

        # MIC signs an arbitrary message (server just verifies the signature)
        message = b"1234"
        mic = ctx.get_signature(message)

        logger.debug(f"Kerberos authentication complete for {self._auth.principal if self._auth else 'unknown'}")
        return bytes(mic), message

    def _enable_settings(self, conn, mic: bytes, message: bytes) -> None:
        """Enable settings on a connection after authentication."""
        enable_req = EnableSettings_request()
        enable_req.list_id = conn.list_id
        enable_req.MIC = mic
        enable_req.message = message
        conn.send_message(enable_req)

        # Server replies with Status_reply (status=0 on success, DPM_PRIV on failure).
        # Skip any ListStatus_reply heartbeats that may arrive first.
        while True:
            reply = conn.recv_message(timeout=self._timeout)
            if isinstance(reply, ListStatus_reply):
                continue
            if isinstance(reply, Status_reply):
                if reply.status != 0:
                    facility, error = parse_error(reply.status)
                    raise AuthenticationError(
                        f"EnableSettings failed: facility={facility}, error={error} (DPM_PRIV = privilege denied)"
                    )
                break
            raise AuthenticationError(f"Unexpected reply during EnableSettings: {type(reply).__name__}")
        logger.debug("EnableSettings accepted")

    # ─────────────────────────────────────────────────────────────────────────
    # Write Connection Pool Management
    # ─────────────────────────────────────────────────────────────────────────

    def _get_write_connection(self) -> _WriteConnection:
        """Get or create an authenticated write connection.

        Returns an existing idle connection from the pool, or creates
        and authenticates a new one if needed.

        Returns:
            _WriteConnection with authentication completed

        Raises:
            AuthenticationError: If authentication fails
            DPMConnectionError: If connection fails
        """
        with self._write_lock:
            # Close and remove stale connections
            fresh = []
            for wc in self._write_connections:
                if wc.is_stale(self._write_idle_timeout):
                    wc.close()
                else:
                    fresh.append(wc)
            self._write_connections = fresh

            # Try to get an existing live connection
            while self._write_connections:
                wc = self._write_connections.pop(0)
                if not wc.conn.connected:
                    logger.debug(f"Discarding dead write connection (list_id={wc.conn.list_id})")
                    wc.close()
                    continue
                wc.last_used = time.monotonic()
                logger.debug(f"Reusing authenticated write connection (list_id={wc.conn.list_id})")
                return wc

        # Create new connection outside the lock
        assert self._auth is not None, "Auth required for write connections"
        conn = DPMConnection(host=self._host, port=self._port, timeout=self._timeout)
        conn.connect()
        wc = _WriteConnection(conn, self._auth.principal, self._role)  # type: ignore[arg-type]  # narrowed by assert above

        # Authenticate the new connection
        try:
            mic, message = self._authenticate_connection(conn)
            self._enable_settings(conn, mic, message)
            wc.authenticated = True
            logger.debug(f"Created new authenticated write connection (list_id={conn.list_id})")
        except Exception:
            wc.close()
            raise

        return wc

    def _release_write_connection(self, wc: _WriteConnection) -> None:
        """Return a write connection to the pool for reuse.

        The connection should be in a clean state (list stopped).
        """
        wc.last_used = time.monotonic()

        with self._write_lock:
            if len(self._write_connections) < self._write_pool_size:
                self._write_connections.append(wc)
                logger.debug(f"Returned write connection to pool (list_id={wc.conn.list_id})")
            else:
                # Pool full, close this one
                wc.close()
                logger.debug(f"Write pool full, closed connection (list_id={wc.conn.list_id})")

    def _discard_write_connection(self, wc: _WriteConnection) -> None:
        """Discard a broken write connection without returning to pool."""
        wc.close()
        logger.debug(f"Discarded broken write connection (list_id={wc.conn.list_id})")

    def _close_write_connections(self) -> None:
        """Close all write connections."""
        with self._write_lock:
            for wc in self._write_connections:
                wc.close()
            self._write_connections.clear()
            logger.debug("Closed all write connections")

    def _value_to_setting(
        self,
        ref_id: int,
        value: Value,
    ) -> tuple[Optional[RawSetting_struct], Optional[ScaledSetting_struct], Optional[TextSetting_struct]]:
        """Convert a value to the appropriate setting struct."""
        raw_setting = None
        scaled_setting = None
        text_setting = None

        if isinstance(value, bytes):
            raw_setting = RawSetting_struct()
            raw_setting.ref_id = ref_id
            raw_setting.data = value
        elif isinstance(value, str):
            text_setting = TextSetting_struct()
            text_setting.ref_id = ref_id
            text_setting.data = [value]
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) > 0 and isinstance(value[0], str):  # type: ignore[arg-type]  # numpy indexing
                text_setting = TextSetting_struct()
                text_setting.ref_id = ref_id
                text_setting.data = list(value)
            else:
                scaled_setting = ScaledSetting_struct()
                scaled_setting.ref_id = ref_id
                scaled_setting.data = [float(v) for v in value]
        else:
            scaled_setting = ScaledSetting_struct()
            scaled_setting.ref_id = ref_id
            scaled_setting.data = [float(value)]  # type: ignore[arg-type]  # numeric Value

        return raw_setting, scaled_setting, text_setting

    # Writable alarm dict keys → DRF field names, keyed by DRF property.
    # "abort" and "alarm_status" are read-only status bits, not settable.
    _ANALOG_ALARM_FIELDS: dict[str, str] = {
        "minimum": "MIN",
        "maximum": "MAX",
        "alarm_enable": "ALARM_ENABLE",
        "abort_inhibit": "ABORT_INHIBIT",
        "tries_needed": "TRIES_NEEDED",
    }
    _DIGITAL_ALARM_FIELDS: dict[str, str] = {
        "nominal": "NOM",
        "mask": "MASK",
        "alarm_enable": "ALARM_ENABLE",
        "abort_inhibit": "ABORT_INHIBIT",
        "tries_needed": "TRIES_NEEDED",
    }

    def _expand_alarm_dict(self, drf: str, alarm_dict: dict) -> list[tuple[str, Value]]:
        """Expand an alarm dict into per-field (drf.FIELD, value) pairs.

        DPM/HTTP ApplySettings only supports scalar/raw/text -- not structured
        alarm messages. The DRF property (ANALOG/DIGITAL) determines which
        field map to use; keys that don't belong raise ValueError.
        """
        from pacsys.drf3 import parse_request
        from pacsys.drf3.property import DRF_PROPERTY
        from pacsys.drf_utils import get_device_name

        prop = parse_request(drf).property
        if prop == DRF_PROPERTY.ANALOG:
            field_map = self._ANALOG_ALARM_FIELDS
            prop_name = "ANALOG"
        elif prop == DRF_PROPERTY.DIGITAL:
            field_map = self._DIGITAL_ALARM_FIELDS
            prop_name = "DIGITAL"
        elif prop in (DRF_PROPERTY.STATUS, DRF_PROPERTY.CONTROL):
            raise ValueError(
                f"Cannot write a dict to {prop.name} property. "
                f'Use BasicControl enum values instead: backend.write("{drf}", BasicControl.ON)'
            )
        else:
            raise ValueError(f"Cannot write dict to {prop.name} property (DRF: {drf})")

        # Validate keys
        writable = set(field_map) | {"abort", "alarm_status", "tries_now"}  # read-only keys are silently skipped
        bad_keys = set(alarm_dict) - writable
        if bad_keys:
            raise ValueError(f"Unknown {prop_name} alarm keys: {bad_keys}")

        base = get_device_name(drf)
        pairs: list[tuple[str, Value]] = []
        for key, field_name in field_map.items():
            if key not in alarm_dict:
                continue
            val = alarm_dict[key]
            if isinstance(val, bool):
                val = 1 if val else 0
            pairs.append((f"{base}.{prop_name}.{field_name}", val))
        return pairs

    def write(
        self,
        drf: str,
        value: Value,
        timeout: Optional[float] = None,
    ) -> WriteResult:
        """Write a single device value."""
        # DPM/HTTP has no structured alarm setting type -- expand dict to
        # sequential per-field writes.  They must be sequential because alarm
        # fields share the same 20-byte block on the server; a single
        # ApplySettings with multiple fields on the same device causes the
        # later field to overwrite the earlier one.
        if isinstance(value, dict):
            pairs = self._expand_alarm_dict(drf, value)
            if not pairs:
                return WriteResult(drf=drf, error_code=ERR_OK)
            for field_drf, field_val in pairs:
                results = self.write_many([(field_drf, field_val)], timeout=timeout)
                if not results[0].success:
                    r = results[0]
                    return WriteResult(
                        drf=drf, facility_code=r.facility_code, error_code=r.error_code, message=r.message
                    )
            return WriteResult(drf=drf, error_code=ERR_OK)

        results = self.write_many([(drf, value)], timeout=timeout)
        return results[0]

    def _execute_write(
        self,
        conn: DPMConnection,
        list_id: int,
        prepared_settings: list[tuple[str, Value]],
        deadline: float,
    ) -> Optional[ApplySettings_reply]:
        """Execute the write protocol on an authenticated connection.

        Returns the ApplySettings_reply or None on timeout.
        Raises connection errors for retry handling by caller.
        """
        # Stop and clear previous requests from reused connection
        stop_req = StopList_request()
        stop_req.list_id = list_id
        conn.send_message(stop_req)

        clear_req = ClearList_request()
        clear_req.list_id = list_id
        conn.send_message(clear_req)

        # Set ROLE list property
        role_req = AddToList_request()
        role_req.list_id = list_id
        role_req.ref_id = 0
        role_req.drf_request = f"#ROLE:{self._role}"
        conn.send_message(role_req)

        # Add devices to list
        for i, (drf, _) in enumerate(prepared_settings):
            add_req = AddToList_request()
            add_req.list_id = list_id
            add_req.ref_id = i + 1
            add_req.drf_request = drf
            conn.send_message(add_req)

        # Start list
        start_req = StartList_request()
        start_req.list_id = list_id
        conn.send_message(start_req)

        # Wait for device info
        device_infos: dict[int, DeviceInfo_reply | Status_reply] = {}
        received_infos = 0
        expected_count = len(prepared_settings)

        while received_infos < expected_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                reply = conn.recv_message(timeout=min(remaining, 2.0))
            except TimeoutError:
                if time.monotonic() >= deadline:
                    break
                continue

            if isinstance(reply, (AddToList_reply, ListStatus_reply)):
                pass
            elif isinstance(reply, DeviceInfo_reply):
                device_infos[reply.ref_id] = reply
                received_infos += 1
            elif isinstance(reply, StartList_reply):
                if reply.status != 0:
                    logger.warning(f"StartList returned status {reply.status}")
            elif isinstance(reply, Status_reply):
                device_infos[reply.ref_id] = reply
                received_infos += 1

        # Build and send ApplySettings
        apply_req = ApplySettings_request()
        apply_req.user_name = self._auth.principal if self._auth else ""
        apply_req.list_id = list_id

        raw_settings = []
        scaled_settings = []
        text_settings = []

        for i, (_, value) in enumerate(prepared_settings):
            ref_id = i + 1
            raw, scaled, text = self._value_to_setting(ref_id, value)
            if raw:
                raw_settings.append(raw)
            if scaled:
                scaled_settings.append(scaled)
            if text:
                text_settings.append(text)

        if raw_settings:
            apply_req.raw_array = raw_settings
        if scaled_settings:
            apply_req.scaled_array = scaled_settings
        if text_settings:
            apply_req.text_array = text_settings

        conn.send_message(apply_req)

        # Wait for ApplySettings reply
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                reply = conn.recv_message(timeout=min(remaining, 2.0))
            except TimeoutError:
                if time.monotonic() >= deadline:
                    break
                continue

            if isinstance(reply, ApplySettings_reply):
                return reply
            elif isinstance(reply, ListStatus_reply):
                pass

        return None

    def write_many(
        self,
        settings: list[tuple[str, Value]],
        timeout: Optional[float] = None,
    ) -> list[WriteResult]:
        """Write multiple device values.

        Uses a pool of authenticated connections for efficient repeated writes.
        Authentication is cached at the DPM list level and reused across calls.
        """
        if not settings:
            return []

        if not isinstance(self._auth, KerberosAuth):
            raise AuthenticationError("Backend not configured for authenticated operations. Pass auth=KerberosAuth().")

        if self._role is None:
            raise AuthenticationError("Role required for writes. Pass role='Operator' (or appropriate role).")

        effective_timeout = timeout if timeout is not None else self._timeout

        # Prepare settings (add .SETTING and @I if needed)
        prepared_settings = [(prepare_for_write(drf), value) for drf, value in settings]

        # Try up to twice: first attempt may hit a stale pooled connection
        last_error = None
        for attempt in range(2):
            deadline = time.monotonic() + effective_timeout

            try:
                wc = self._get_write_connection()
            except Exception as e:
                error_msg = f"Failed to get write connection: {e}"
                return [
                    WriteResult(drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=error_msg)
                    for drf, _ in settings
                ]

            conn = wc.conn
            list_id = conn.list_id
            apply_reply = None

            try:
                assert list_id is not None, "list_id must be set after connect"
                result = self._execute_write(conn, list_id, prepared_settings, deadline)
                apply_reply = result

                # Stop list (but keep connection and auth for reuse)
                stop_req = StopList_request()
                stop_req.list_id = list_id
                conn.send_message(stop_req)

                self._release_write_connection(wc)
                last_error = None
                break  # Success

            except (BrokenPipeError, ConnectionResetError, OSError, DPMConnectionError) as e:
                logger.warning(f"Write connection error (attempt {attempt + 1}): {e}")
                self._discard_write_connection(wc)
                last_error = e
                if attempt == 0:
                    continue  # Retry with fresh connection
            except Exception as e:
                logger.warning(f"Unexpected write error: {e}")
                self._discard_write_connection(wc)
                raise

        if last_error is not None:
            return [
                WriteResult(
                    drf=drf,
                    facility_code=FACILITY_ACNET,
                    error_code=ERR_RETRY,
                    message=f"Connection error: {last_error}",
                )
                for drf, _ in settings
            ]

        # Parse results
        if apply_reply is None:
            return [
                WriteResult(drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_TIMEOUT, message="Request timeout")
                for drf, _ in settings
            ]

        status_map: dict[int, int] = {}
        for status_struct in apply_reply.status:
            status_map[status_struct.ref_id] = status_struct.status

        results: list[WriteResult] = []
        for i, (drf, _) in enumerate(settings):
            ref_id = i + 1
            status = status_map.get(ref_id, -1)

            if status == 0:
                results.append(WriteResult(drf=drf, error_code=ERR_OK))
            else:
                facility, error = parse_error(status)
                results.append(
                    WriteResult(
                        drf=drf,
                        facility_code=facility,
                        error_code=error,
                        message=f"Write error (facility={facility}, error={error})",
                    )
                )

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Streaming Methods -- asyncio reactor
    # ─────────────────────────────────────────────────────────────────────────

    def _start_reactor(self) -> None:
        """Start the reactor thread and event loop. Must hold _reactor_lock."""
        ready = threading.Event()
        loop_holder: list[asyncio.AbstractEventLoop] = []

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_holder.append(loop)
            ready.set()
            loop.run_forever()
            # Cleanup pending tasks on shutdown
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        self._reactor_thread = threading.Thread(target=_run, name="DPMHTTPBackend-Reactor", daemon=True)
        self._reactor_thread.start()
        ready.wait(timeout=5.0)
        if not loop_holder:
            raise RuntimeError("DPMHTTPBackend: failed to start reactor event loop")
        self._loop = loop_holder[0]

    def _ensure_reactor(self) -> None:
        """Lazily start the reactor thread (double-check locking)."""
        if self._loop is not None:
            return
        with self._reactor_lock:
            if self._loop is not None:
                return
            if self._closed:
                raise RuntimeError("Backend is closed")
            self._start_reactor()

    async def _stream_subscription(self, handle: _DPMHTTPSubscriptionHandle) -> None:
        """Async coroutine that manages a single streaming subscription.

        Creates its own TCP connection, sends AddToList/StartList, then
        loops receiving replies until cancelled or connection error.
        """
        conn = _AsyncDPMConnection(self._host, self._port, self._timeout)
        drfs = handle._drfs
        callback = handle._callback
        # Per-device metadata, keyed by ref_id
        metas: dict[int, DeviceMeta] = {}
        # DRF by ref_id for reading construction
        drf_map: dict[int, str] = {}
        # Rate-limit queue overflow warnings
        drop_count = 0
        last_drop_log = 0.0

        try:
            await conn.connect()
            list_id = conn.list_id

            # Batch all AddToList + StartList into a single TCP write
            setup_msgs = []
            for i, drf in enumerate(drfs):
                ref_id = i + 1
                drf_map[ref_id] = drf
                add_req = AddToList_request()
                add_req.list_id = list_id
                add_req.ref_id = ref_id
                add_req.drf_request = drf
                setup_msgs.append(add_req)

            start_req = StartList_request()
            start_req.list_id = list_id
            setup_msgs.append(start_req)
            await conn.send_messages_batch(setup_msgs)

            # Receive loop
            while not handle._stopped:
                reply = await conn.recv_message()

                if isinstance(reply, AddToList_reply):
                    if reply.status != 0:
                        # Report failed device addition as error reading
                        drf = drf_map.get(reply.ref_id)
                        if drf is not None:
                            facility, error = parse_error(reply.status)
                            reading = Reading(
                                drf=drf,
                                value_type=ValueType.SCALAR,
                                tag=reply.ref_id,
                                facility_code=facility,
                                error_code=error,
                                value=None,
                                message=status_message(facility, error) or f"AddToList failed (status={reply.status})",
                                timestamp=None,
                                cycle=0,
                                meta=None,
                            )
                            if callback is not None:
                                try:
                                    callback(reading, handle)
                                except Exception as e:
                                    logger.error(f"Error in callback: {e}")
                            else:
                                try:
                                    handle._queue.put_nowait(reading)
                                except queue.Full:
                                    pass
                    continue

                if isinstance(reply, StartList_reply):
                    if reply.status != 0:
                        logger.warning(f"StartList returned status {reply.status}")
                    continue

                if isinstance(reply, ListStatus_reply):
                    continue

                if isinstance(reply, DeviceInfo_reply):
                    metas[reply.ref_id] = _device_info_to_meta(reply)
                    continue

                if hasattr(reply, "ref_id"):
                    ref_id = reply.ref_id
                    drf = drf_map.get(ref_id)
                    if drf is None:
                        logger.warning(f"Data for unknown ref_id={ref_id}")
                        continue
                    meta = metas.get(ref_id)
                    reading = self._reply_to_reading(reply, drf, meta)

                    if callback is not None:
                        try:
                            callback(reading, handle)
                        except Exception as e:
                            logger.error(f"Error in callback: {e}")
                    else:
                        try:
                            handle._queue.put_nowait(reading)
                        except queue.Full:
                            drop_count += 1
                            now = time.monotonic()
                            if now - last_drop_log >= 5.0:
                                logger.warning(
                                    f"DPM subscription queue full ({_DEFAULT_QUEUE_MAXSIZE}), "
                                    f"dropped {drop_count} readings"
                                )
                                drop_count = 0
                                last_drop_log = now

        except asyncio.CancelledError:
            pass  # Normal shutdown via task.cancel()
        except (asyncio.IncompleteReadError, DPMConnectionError, OSError) as e:
            if not handle._stopped:
                handle._exc = e
                handle._stopped = True
                if handle._on_error is not None:
                    try:
                        handle._on_error(e, handle)
                    except Exception as cb_err:
                        logger.error(f"Error in on_error callback: {cb_err}")
        except Exception as e:
            if not handle._stopped:
                handle._exc = e
                handle._stopped = True
                logger.error(f"Unexpected streaming error: {e}")
                if handle._on_error is not None:
                    try:
                        handle._on_error(e, handle)
                    except Exception as cb_err:
                        logger.error(f"Error in on_error callback: {cb_err}")
        finally:
            await conn.close()
            # Remove handle so dead subscriptions don't accumulate
            with self._handles_lock:
                if handle in self._handles:
                    self._handles.remove(handle)

    def _reply_to_reading(self, reply, drf: str, meta: Optional[DeviceMeta]) -> Reading:
        """Convert a DPM reply to a Reading object."""
        if isinstance(reply, Status_reply):
            facility, error = parse_error(reply.status)
            return Reading(
                drf=drf,
                value_type=ValueType.SCALAR,
                tag=reply.ref_id,
                facility_code=facility,
                error_code=error,
                value=None,
                message=status_message(facility, error),
                timestamp=timestamp_from_millis(reply.timestamp) if reply.timestamp else None,
                cycle=reply.cycle,
                meta=meta,
            )

        value, value_type = _reply_to_value_and_type(reply)

        # Unknown reply type -- return error reading
        if value_type is None:
            return Reading(
                drf=drf,
                value_type=ValueType.SCALAR,
                tag=reply.ref_id if hasattr(reply, "ref_id") else 0,
                facility_code=FACILITY_ACNET,
                error_code=ERR_RETRY,
                value=None,
                message=f"Unknown reply type: {type(reply).__name__}",
                timestamp=None,
                cycle=0,
                meta=meta,
            )

        # Alarm/status replies have no status field -- receiving them means success (0)
        status = reply.status if hasattr(reply, "status") else 0
        timestamp = reply.timestamp
        cycle = reply.cycle

        facility, error = parse_error(status)

        return Reading(
            drf=drf,
            value_type=value_type,
            tag=reply.ref_id,
            facility_code=facility,
            error_code=error,
            value=value,
            message=status_message(facility, error),
            timestamp=timestamp_from_millis(timestamp) if timestamp else None,
            cycle=cycle,
            meta=meta,
        )

    def subscribe(
        self,
        drfs: list[str],
        callback: Optional[ReadingCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ) -> SubscriptionHandle:
        """Subscribe to devices for streaming data.

        Each subscribe() call creates an async task with its own TCP connection
        and DPM list. Subscriptions are independent -- stopping one does not
        affect others.

        Args:
            drfs: List of device request strings (e.g., "M:OUTTMP@p,1000")
            callback: Optional function called for each reading.
                     If None, use handle.readings() to iterate.
            on_error: Optional function called on connection errors.

        Returns:
            SubscriptionHandle for managing this subscription
        """
        if not drfs:
            raise ValueError("drfs cannot be empty")

        if self._closed:
            raise RuntimeError("Backend is closed")

        self._ensure_reactor()
        assert self._loop is not None

        handle = _DPMHTTPSubscriptionHandle(
            backend=self,
            drfs=drfs,
            callback=callback,
            on_error=on_error,
        )

        # Create the streaming task on the reactor loop
        async def _create_task():
            return asyncio.ensure_future(self._stream_subscription(handle))

        handle._task = asyncio.run_coroutine_threadsafe(_create_task(), self._loop).result(timeout=5.0)

        with self._handles_lock:
            self._handles.append(handle)

        mode_str = "callback" if handle._is_callback_mode else "iterator"
        logger.info(f"Created {mode_str} subscription for {len(drfs)} devices")
        return handle

    def remove(self, handle: SubscriptionHandle) -> None:
        """Remove a subscription. Cancels the associated async task."""
        if not isinstance(handle, _DPMHTTPSubscriptionHandle):
            raise TypeError(f"Expected _DPMHTTPSubscriptionHandle, got {type(handle).__name__}")

        handle._stopped = True

        if handle._task is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(handle._task.cancel)

        with self._handles_lock:
            if handle in self._handles:
                self._handles.remove(handle)

        logger.info(f"Removed DPM subscription for {len(handle._drfs)} devices")

    def stop_streaming(self) -> None:
        """Stop all streaming subscriptions."""
        with self._handles_lock:
            handles = list(self._handles)
            self._handles.clear()

        for handle in handles:
            handle._stopped = True
            if handle._task is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(handle._task.cancel)

        logger.info("All DPM streaming stopped")

    def close(self) -> None:
        """Close the backend and release all resources."""
        if self._closed:
            return

        self._closed = True

        # Stop streaming first
        self.stop_streaming()

        # Stop the event loop and join reactor thread
        loop = self._loop
        thread = self._reactor_thread
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._loop = None
        self._reactor_thread = None

        # Close write connections
        self._close_write_connections()

        # Close connection pool
        if self._pool is not None:
            self._pool.close()
            self._pool = None

        logger.info("DPMHTTPBackend closed")

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        auth_info = f", auth={self._auth.auth_type}" if self._auth else ""
        with self._handles_lock:
            n_subs = len(self._handles)
        return f"DPMHTTPBackend({self._host}:{self._port}, pool_size={self._pool_size}{auth_info}, subs={n_subs}, {status})"


__all__ = ["DPMHTTPBackend"]
