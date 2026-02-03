"""
DPM HTTP Backend - primary backend for ACNET device access.

Uses TCP/PC protocol via acsys-proxy. Connection pool for reads,
independent TCP connections per subscribe() for streaming.
See SPECIFICATION.md for protocol details.
"""

import logging
import queue
import selectors
import socket
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
from pacsys.dpm_connection import DPMConnection, DPMConnectionError
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


def _ensure_immediate_event(drf: str) -> str:
    """Ensure DRF has immediate event (@I) if no event specified."""
    return ensure_immediate_event(drf)


def _reply_to_value_and_type(reply) -> tuple[Optional[Value], ValueType]:
    """Extract value and type from a DPM data reply."""
    if isinstance(reply, Scalar_reply):
        return reply.data, ValueType.SCALAR
    elif isinstance(reply, ScalarArray_reply):
        return np.array(reply.data), ValueType.SCALAR_ARRAY
    elif isinstance(reply, TimedScalarArray_reply):
        return np.array(reply.data), ValueType.SCALAR_ARRAY
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

    logger.warning(f"Unknown reply type: {type(reply).__name__}")
    return None, ValueType.SCALAR


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


class _StreamSubscription:
    """Internal state for a single device within a subscription."""

    def __init__(
        self,
        ref_id: int,
        drf: str,
        callback: Optional[ReadingCallback],
        handle: "_DPMHTTPSubscriptionHandle",
    ):
        self.ref_id = ref_id
        self.drf = drf
        self.callback = callback
        self.handle = handle
        self.meta: Optional[DeviceMeta] = None


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


class _StreamConnection:
    """State for a single streaming TCP connection (one per subscribe() call)."""

    def __init__(
        self,
        sub_id: int,
        conn: DPMConnection,
        handle: "_DPMHTTPSubscriptionHandle",
    ):
        self.sub_id = sub_id
        self.conn = conn
        self.handle = handle
        self.list_started = False
        self.subscriptions: dict[int, _StreamSubscription] = {}  # ref_id -> subscription
        self.next_ref_id = 1


class _DPMHTTPSubscriptionHandle(SubscriptionHandle):
    """Subscription handle for DPMHTTPBackend.

    Each handle corresponds to one TCP connection with its own DPM list.
    """

    def __init__(
        self,
        backend: "DPMHTTPBackend",
        sub_id: int,
        is_callback_mode: bool,
        on_error: Optional[ErrorCallback] = None,
    ):
        self._backend = backend
        self._sub_id = sub_id
        self._is_callback_mode = is_callback_mode
        self._on_error = on_error
        self._queue: queue.Queue[Reading] = queue.Queue(maxsize=_DEFAULT_QUEUE_MAXSIZE)
        self._stopped = False
        self._exc: Optional[Exception] = None
        self._ref_ids: list[int] = []

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
        """Stop this subscription and close its connection."""
        if not self._stopped:
            self._backend.remove(self)
            self._stopped = True


class DPMHTTPBackend(Backend):
    """
    DPM HTTP Backend for ACNET device access.

    Uses TCP/HTTP protocol to communicate with DPM via acsys-proxy.
    Supports multiple independent streaming subscriptions, each with its
    own TCP connection, multiplexed on a single receiver thread.

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

        # Streaming state - multiple connections, single thread
        self._selector: Optional[selectors.DefaultSelector] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_lock = threading.Lock()
        self._stream_stop_event = threading.Event()
        self._stream_connections: dict[int, _StreamConnection] = {}  # sub_id -> connection state
        self._next_sub_id = 1
        # Self-pipe trick: selector ops are queued and executed by the receiver
        # thread so selector.select() is never held under _stream_lock.
        self._selector_ops: queue.SimpleQueue = queue.SimpleQueue()
        self._wakeup_r: Optional[socket.socket] = None
        self._wakeup_w: Optional[socket.socket] = None

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

        prepared_drfs = [_ensure_immediate_event(drf) for drf in drfs]

        device_infos: dict[int, DeviceInfo_reply] = {}
        data_replies: dict[int, object] = {}
        received_count = 0
        expected_count = len(drfs)

        pool = self._get_pool()

        with pool.connection() as conn:
            list_id = conn.list_id

            for i, drf in enumerate(prepared_drfs):
                add_req = AddToList_request()
                add_req.list_id = list_id
                add_req.ref_id = i + 1
                add_req.drf_request = drf
                conn.send_message(add_req)

            start_req = StartList_request()
            start_req.list_id = list_id
            conn.send_message(start_req)

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

                if isinstance(reply, DeviceInfo_reply):
                    device_infos[reply.ref_id] = reply
                elif isinstance(reply, StartList_reply):
                    if reply.status != 0:
                        logger.warning(f"StartList returned status {reply.status}")
                elif isinstance(reply, ListStatus_reply):
                    pass
                elif isinstance(reply, Status_reply):
                    # Only count first reply per ref_id (ignore subsequent periodic updates)
                    if reply.ref_id not in data_replies:
                        data_replies[reply.ref_id] = reply
                        received_count += 1
                elif hasattr(reply, "ref_id"):
                    # Only count first reply per ref_id (ignore subsequent periodic updates)
                    if reply.ref_id not in data_replies:
                        data_replies[reply.ref_id] = reply
                        received_count += 1

            clear_req = ClearList_request()
            clear_req.list_id = list_id
            conn.send_message(clear_req)

        readings: list[Reading] = []

        for i, original_drf in enumerate(drfs):
            ref_id = i + 1
            info = device_infos.get(ref_id)
            reply = data_replies.get(ref_id)

            meta = _device_info_to_meta(info) if info else None

            if reply is None:
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
        2. Server replies with its Kerberos service name (e.g. "dpm@dce03.fnal.gov")
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

        # Server replies with Status_reply (status=0 on success, DPM_PRIV on failure)
        reply = conn.recv_message(timeout=self._timeout)
        if isinstance(reply, Status_reply):
            status = reply.status
            if status != 0:
                facility, error = parse_error(status)
                raise AuthenticationError(
                    f"EnableSettings failed: facility={facility}, error={error} (DPM_PRIV = privilege denied)"
                )
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

            # Try to get an existing connection
            if self._write_connections:
                wc = self._write_connections.pop(0)
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
        elif isinstance(value, (list, np.ndarray)):
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

        DPM/HTTP ApplySettings only supports scalar/raw/text — not structured
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
        # DPM/HTTP has no structured alarm setting type — expand dict to
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
        deadline = time.monotonic() + effective_timeout

        # Prepare settings (add .SETTING and @I if needed)
        prepared_settings = [(prepare_for_write(drf), value) for drf, value in settings]

        # Get authenticated write connection (from pool or create new)
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
            # Clear previous requests from reused connection
            clear_req = ClearList_request()
            clear_req.list_id = list_id
            conn.send_message(clear_req)

            # Set ROLE list property (DPM server checks list.property("ROLE") for write auth)
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
            expected_count = len(settings)

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

                if isinstance(reply, DeviceInfo_reply):
                    device_infos[reply.ref_id] = reply
                    received_infos += 1
                elif isinstance(reply, StartList_reply):
                    if reply.status != 0:
                        logger.warning(f"StartList returned status {reply.status}")
                elif isinstance(reply, ListStatus_reply):
                    pass
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
                    apply_reply = reply
                    break
                elif isinstance(reply, ListStatus_reply):
                    pass

            # Stop list (but keep connection and auth for reuse)
            stop_req = StopList_request()
            stop_req.list_id = list_id
            conn.send_message(stop_req)

            # Return connection to pool for reuse
            self._release_write_connection(wc)

        except (BrokenPipeError, ConnectionResetError, OSError, DPMConnectionError) as e:
            # Connection broken, discard it
            logger.warning(f"Write connection error: {e}")
            self._discard_write_connection(wc)
            return [
                WriteResult(
                    drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=f"Connection error: {e}"
                )
                for drf, _ in settings
            ]
        except Exception as e:
            # Unexpected error, discard connection to be safe
            logger.warning(f"Unexpected write error: {e}")
            self._discard_write_connection(wc)
            raise

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
    # Streaming Methods - Multi-connection, single-thread
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_receiver_thread(self) -> None:
        """Ensure the receiver thread and selector are running."""
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return

        with self._stream_lock:
            if self._stream_thread is not None and self._stream_thread.is_alive():
                return

            if self._closed:
                raise RuntimeError("Backend is closed")

            # Create selector + wakeup pipe if needed
            if self._selector is None:
                self._selector = selectors.DefaultSelector()
                r, w = socket.socketpair()
                r.setblocking(False)
                w.setblocking(False)
                self._wakeup_r = r
                self._wakeup_w = w
                self._selector.register(r, selectors.EVENT_READ, data=None)

            self._stream_stop_event.clear()

            self._stream_thread = threading.Thread(
                target=self._receiver_loop,
                name="DPMHTTPBackend-Receiver",
                daemon=True,
            )
            self._stream_thread.start()

            logger.debug("Started receiver thread")

    def _wakeup(self) -> None:
        """Wake the receiver thread from selector.select()."""
        if self._wakeup_w is not None:
            try:
                self._wakeup_w.send(b"\x00")
            except OSError:
                pass

    def _process_selector_ops(self) -> None:
        """Process queued selector register/unregister ops (receiver thread only)."""
        while True:
            try:
                action, sock, sub_id = self._selector_ops.get_nowait()
            except queue.Empty:
                break
            try:
                assert self._selector is not None
                if action == "register":
                    self._selector.register(sock, selectors.EVENT_READ, data=sub_id)
                elif action == "unregister":
                    self._selector.unregister(sock)
            except (KeyError, ValueError, OSError):
                pass

    def _receiver_loop(self) -> None:
        """Single thread that receives from all streaming connections."""
        logger.debug("Receiver loop started")

        while not self._stream_stop_event.is_set():
            try:
                # All selector mutations happen here on the receiver thread
                self._process_selector_ops()

                if self._selector is None:
                    break

                # select() WITHOUT holding _stream_lock — no contention
                ready = self._selector.select(timeout=1.0)

                for key, events in ready:
                    if self._stream_stop_event.is_set():
                        break

                    # Drain wakeup pipe (data=None sentinel)
                    if key.data is None:
                        try:
                            assert self._wakeup_r is not None
                            self._wakeup_r.recv(1024)
                        except (BlockingIOError, OSError):
                            pass
                        continue

                    sub_id = key.data
                    self._handle_socket_ready(sub_id)

            except Exception as e:
                if self._stream_stop_event.is_set():
                    break
                logger.error(f"Error in receiver loop: {e}")

        logger.debug("Receiver loop exiting")

    def _handle_socket_ready(self, sub_id: int) -> None:
        """Handle a socket that has data ready to read."""
        with self._stream_lock:
            stream_conn = self._stream_connections.get(sub_id)
            if stream_conn is None:
                return

            conn = stream_conn.conn

        try:
            # Receive message (non-blocking since selector said it's ready)
            reply = conn.recv_message(timeout=1.0)
            self._dispatch_reply(sub_id, reply)

        except DPMConnectionError as e:
            logger.error(f"Connection error for sub_id={sub_id}: {e}")
            self._handle_connection_error(sub_id, e)

        except TimeoutError:
            # Shouldn't happen since selector said ready, but handle gracefully
            pass

        except Exception as e:
            logger.error(f"Unexpected error for sub_id={sub_id}: {e}")
            self._handle_connection_error(sub_id, e)

    def _dispatch_reply(self, sub_id: int, reply) -> None:
        """Dispatch a received reply to appropriate handler."""
        with self._stream_lock:
            stream_conn = self._stream_connections.get(sub_id)
            if stream_conn is None:
                return

        # Handle StartList reply
        if isinstance(reply, StartList_reply):
            if reply.status != 0:
                logger.warning(f"StartList returned status {reply.status} for sub_id={sub_id}")
            return

        # Handle heartbeat
        if isinstance(reply, ListStatus_reply):
            logger.debug(f"Heartbeat: sub_id={sub_id}, status={reply.status}")
            return

        # Handle DeviceInfo
        if isinstance(reply, DeviceInfo_reply):
            ref_id = reply.ref_id
            with self._stream_lock:
                sub = stream_conn.subscriptions.get(ref_id)
                if sub is not None:
                    sub.meta = _device_info_to_meta(reply)
            return

        # Handle data replies
        if hasattr(reply, "ref_id"):
            ref_id = reply.ref_id
            with self._stream_lock:
                sub = stream_conn.subscriptions.get(ref_id)
                if sub is None:
                    logger.warning(f"Data for unknown ref_id={ref_id} in sub_id={sub_id}")
                    return
                drf = sub.drf
                meta = sub.meta
                callback = sub.callback
                handle = sub.handle

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
                    logger.warning(
                        f"DPM subscription queue full ({_DEFAULT_QUEUE_MAXSIZE}), dropping reading for {reading.drf}"
                    )

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
        status = getattr(reply, "status", 0)
        timestamp = getattr(reply, "timestamp", 0)
        cycle = getattr(reply, "cycle", 0)

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

    def _handle_connection_error(self, sub_id: int, exc: Exception) -> None:
        """Handle a connection error for a specific subscription."""
        with self._stream_lock:
            stream_conn = self._stream_connections.get(sub_id)
            if stream_conn is None:
                return

            handle = stream_conn.handle

        # Store exception on handle
        handle._exc = exc
        handle._stopped = True

        # Call on_error callback if provided
        if handle._on_error is not None:
            try:
                handle._on_error(exc, handle)
            except Exception as cb_err:
                logger.error(f"Error in on_error callback: {cb_err}")

        # Clean up this connection
        self._cleanup_subscription(sub_id)

    def _cleanup_subscription(self, sub_id: int) -> None:
        """Clean up a subscription's connection and state."""
        with self._stream_lock:
            stream_conn = self._stream_connections.pop(sub_id, None)
            if stream_conn is None:
                return

            conn = stream_conn.conn

        # Queue selector unregister (processed by receiver thread)
        self._selector_ops.put(("unregister", conn._socket, None))
        self._wakeup()

        # Close connection (outside lock)
        try:
            conn.close()
        except Exception as e:
            logger.debug(f"Error closing connection for sub_id={sub_id}: {e}")

    def subscribe(
        self,
        drfs: list[str],
        callback: Optional[ReadingCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ) -> SubscriptionHandle:
        """Subscribe to devices for streaming data.

        Each subscribe() call creates a new TCP connection with its own
        DPM list. Subscriptions are independent - stopping one does not
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

        # Create new connection for this subscription
        conn = DPMConnection(
            host=self._host,
            port=self._port,
            timeout=self._timeout,
        )
        conn.connect()

        is_callback_mode = callback is not None

        with self._stream_lock:
            sub_id = self._next_sub_id
            self._next_sub_id += 1

            # Create handle
            handle = _DPMHTTPSubscriptionHandle(
                backend=self,
                sub_id=sub_id,
                is_callback_mode=is_callback_mode,
                on_error=on_error,
            )

            # Create stream connection state
            stream_conn = _StreamConnection(
                sub_id=sub_id,
                conn=conn,
                handle=handle,
            )

            list_id = conn.list_id
            ref_ids = []

            # Add each device to this connection's list
            for drf in drfs:
                ref_id = stream_conn.next_ref_id
                stream_conn.next_ref_id += 1

                add_req = AddToList_request()
                add_req.list_id = list_id
                add_req.ref_id = ref_id
                add_req.drf_request = drf
                conn.send_message(add_req)

                sub = _StreamSubscription(
                    ref_id=ref_id,
                    drf=drf,
                    callback=callback,
                    handle=handle,
                )
                stream_conn.subscriptions[ref_id] = sub
                ref_ids.append(ref_id)

                logger.debug(f"Added device: sub_id={sub_id}, ref_id={ref_id}, drf={drf}")

            # Start the list
            start_req = StartList_request()
            start_req.list_id = list_id
            conn.send_message(start_req)
            stream_conn.list_started = True

            # Store connection state
            self._stream_connections[sub_id] = stream_conn
            handle._ref_ids = ref_ids

        # Queue selector registration (processed by receiver thread)
        self._selector_ops.put(("register", conn._socket, sub_id))
        self._wakeup()

        # Ensure receiver thread is running
        self._ensure_receiver_thread()

        mode_str = "callback" if is_callback_mode else "iterator"
        logger.info(f"Created {mode_str} subscription sub_id={sub_id} for {len(drfs)} devices")

        return handle

    def remove(self, handle: SubscriptionHandle) -> None:
        """Remove a subscription and close its connection."""
        if not isinstance(handle, _DPMHTTPSubscriptionHandle):
            raise TypeError(f"Expected _DPMHTTPSubscriptionHandle, got {type(handle).__name__}")

        sub_id = handle._sub_id

        with self._stream_lock:
            stream_conn = self._stream_connections.get(sub_id)
            if stream_conn is None:
                return  # Already removed

            conn = stream_conn.conn
            list_id = conn.list_id

            # Send StopList and ClearList
            if stream_conn.list_started:
                try:
                    stop_req = StopList_request()
                    stop_req.list_id = list_id
                    conn.send_message(stop_req)

                    clear_req = ClearList_request()
                    clear_req.list_id = list_id
                    conn.send_message(clear_req)
                except Exception as e:
                    logger.debug(f"Error stopping list for sub_id={sub_id}: {e}")

        # Clean up (also closes connection)
        self._cleanup_subscription(sub_id)

        handle._stopped = True
        logger.info(f"Removed subscription sub_id={sub_id}")

    def stop_streaming(self) -> None:
        """Stop all streaming subscriptions and close all streaming connections."""
        logger.debug("Stopping all streaming")

        # Signal receiver thread to stop and wake it
        self._stream_stop_event.set()
        self._wakeup()

        # Get all sub_ids
        with self._stream_lock:
            sub_ids = list(self._stream_connections.keys())

        # Stop each subscription
        for sub_id in sub_ids:
            with self._stream_lock:
                stream_conn = self._stream_connections.get(sub_id)
                if stream_conn:
                    stream_conn.handle._stopped = True
            self._cleanup_subscription(sub_id)

        # Wait for receiver thread
        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

        # Drain any remaining queued ops
        while not self._selector_ops.empty():
            try:
                self._selector_ops.get_nowait()
            except queue.Empty:
                break

        # Close wakeup sockets
        for s in (self._wakeup_r, self._wakeup_w):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._wakeup_r = None
        self._wakeup_w = None

        # Close selector
        if self._selector is not None:
            try:
                self._selector.close()
            except Exception:
                pass
            self._selector = None

        logger.info("All streaming stopped")

    def close(self) -> None:
        """Close the backend and release all resources."""
        if self._closed:
            return

        self._closed = True

        # Stop streaming first
        self.stop_streaming()

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
        n_subs = len(self._stream_connections)
        return f"DPMHTTPBackend({self._host}:{self._port}, pool_size={self._pool_size}{auth_info}, subs={n_subs}, {status})"


__all__ = ["DPMHTTPBackend"]
