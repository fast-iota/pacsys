"""
DPM HTTP Backend - primary backend for ACNET device access.

Uses TCP/PC protocol via acsys-proxy. Connection pool for reads,
independent TCP connections per subscribe() for streaming.
See specs/protocols.md and specs/backends.md for protocol details.

TODO: unify with _dpm_core.py (parallel sync/async implementations of the same
protocol: read/write paths ~83% similar, GSSAPI handshake duplicated, and the
conn_broken never-re-pool invariant enforced in both). Plan: delegate this sync
backend to _AsyncDpmCore on the shared reactor thread. Big refactor - needs its
own branch and a full live-server test pass (royal test flush).
"""

import logging
import socket
import struct
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar, SupportsFloat, SupportsIndex, cast

if TYPE_CHECKING:
    import asyncio

from pacsys.acnet.errors import (
    ERR_OK,
    ERR_RETRY,
    ERR_TIMEOUT,
    FACILITY_ACNET,
    parse_error,
    status_message,
)
from pacsys.auth import Auth, KerberosAuth, _require_gssapi
from pacsys.backends import ALARM_READONLY_KEYS, Backend, summarize_drfs, timestamp_from_millis
from pacsys.backends._dispatch import CallbackDispatcher
from pacsys.backends._subscription import BufferedSubscriptionHandle
from pacsys.dpm_connection import DPM_HANDSHAKE, MAX_MESSAGE_SIZE, DPMConnection, DPMConnectionError
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
from pacsys.drf_utils import (
    ensure_immediate_event,
    is_chunked_historical_drf,
    is_historical_drf,
    is_immediate_only,
    prepare_for_write,
)
from pacsys.errors import AuthenticationError, DeviceError, ReadError
from pacsys.pool import ConnectionPool, PoolClosedError, PoolExhaustedError
from pacsys.types import (
    BackendCapability,
    DeviceMeta,
    DispatchMode,
    ErrorCallback,
    Reading,
    ReadingCallback,
    SubscriptionHandle,
    Value,
    ValueType,
    WriteResult,
    _loaded_numpy_types,
)

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_HOST = "acsys-proxy.fnal.gov"
DEFAULT_PORT = 6802
DEFAULT_POOL_SIZE = 4
DEFAULT_TIMEOUT = 5.0
_MAX_WRITE_CONNECTIONS = 4  # max concurrent write connections (pooled + in-flight)

_SettingPayload = tuple[RawSetting_struct | None, ScaledSetting_struct | None, TextSetting_struct | None]


def _coerce_setting_float(value: object) -> float:
    if isinstance(value, (SupportsFloat, SupportsIndex)):
        return float(value)
    raise TypeError(f"DPM scaled setting value is not numeric: {type(value).__name__}")


def _value_to_setting(
    ref_id: int,
    value: Value,
) -> tuple[RawSetting_struct | None, ScaledSetting_struct | None, TextSetting_struct | None]:
    """Convert a public value into exactly one DPM setting payload."""
    if isinstance(value, bytes):
        setting = RawSetting_struct()
        setting.ref_id = ref_id
        setting.data = value
        return setting, None, None

    if isinstance(value, str):
        setting = TextSetting_struct()
        setting.ref_id = ref_id
        setting.data = [value]
        return None, None, setting

    if isinstance(value, dict):
        raise TypeError("write_many() does not support alarm dicts; use write() instead")

    if isinstance(value, _loaded_numpy_types("ndarray")):
        array = cast("Any", value)
        if array.ndim != 1:
            raise TypeError("DPM array settings must be one-dimensional")
        items = cast("list[object]", array.tolist())
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = None

    if items is not None and items:
        text_items = [isinstance(item, str) for item in items]
        if all(text_items):
            setting = TextSetting_struct()
            setting.ref_id = ref_id
            setting.data = cast("list[str]", items)
            return None, None, setting
        if any(text_items):
            raise TypeError("DPM text array settings must contain only strings")

    try:
        data = [_coerce_setting_float(item) for item in items] if items is not None else [_coerce_setting_float(value)]
    except (TypeError, ValueError) as e:
        raise TypeError("DPM scaled settings must contain numeric values") from e

    setting = ScaledSetting_struct()
    setting.ref_id = ref_id
    setting.data = data
    return None, setting, None


def _aggregate_logger_chunks(chunks: list, drf: str, meta) -> Reading:
    """Merge multiple logger reply chunks into a single TIMED_SCALAR_ARRAY Reading."""
    import numpy as np

    all_data: list[np.ndarray] = []
    all_micros: list[np.ndarray] = []
    first_ts = None

    for chunk in chunks:
        # Propagate first error chunk as the result
        if hasattr(chunk, "status") and chunk.status != 0:
            facility, error = parse_error(chunk.status)
            return Reading(
                drf=drf,
                facility_code=facility,
                error_code=error,
                value=None,
                message=status_message(facility, error),
                timestamp=timestamp_from_millis(chunk.timestamp) if chunk.timestamp else None,
                meta=meta,
            )
        if isinstance(chunk, TimedScalarArray_reply):
            all_data.append(np.array(chunk.data))
            if hasattr(chunk, "micros") and chunk.micros:
                all_micros.append(np.array(chunk.micros, dtype=np.int64))
            if first_ts is None and chunk.timestamp:
                first_ts = timestamp_from_millis(chunk.timestamp)
        elif isinstance(chunk, ScalarArray_reply):
            all_data.append(np.array(chunk.data))
            if first_ts is None and chunk.timestamp:
                first_ts = timestamp_from_millis(chunk.timestamp)

    data = np.concatenate(all_data) if all_data else np.array([], dtype=float)
    if all_micros:
        micros = np.concatenate(all_micros)
        value = {"data": data, "micros": micros}
        vtype = ValueType.TIMED_SCALAR_ARRAY
    else:
        value = data
        vtype = ValueType.SCALAR_ARRAY

    return Reading(
        drf=drf,
        value_type=vtype,
        value=value,
        timestamp=first_ts,
        meta=meta,
    )


def _reply_to_value_and_type(reply) -> tuple[Value | None, ValueType | None]:
    """Extract value and type from a DPM data reply."""
    if isinstance(reply, Scalar_reply):
        return reply.data, ValueType.SCALAR
    if isinstance(reply, (ScalarArray_reply, TimedScalarArray_reply)):
        import numpy as np

        if isinstance(reply, ScalarArray_reply):
            return np.array(reply.data), ValueType.SCALAR_ARRAY
        data = np.array(reply.data)
        if hasattr(reply, "micros") and reply.micros:
            micros = np.array(reply.micros, dtype=np.int64)
            return {"data": data, "micros": micros}, ValueType.TIMED_SCALAR_ARRAY
        return data, ValueType.SCALAR_ARRAY
    if isinstance(reply, Raw_reply):
        return bytes(reply.data), ValueType.RAW
    if isinstance(reply, Text_reply):
        return reply.data, ValueType.TEXT
    if isinstance(reply, TextArray_reply):
        return list(reply.data), ValueType.TEXT_ARRAY
    if isinstance(reply, AnalogAlarm_reply):
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
    if isinstance(reply, DigitalAlarm_reply):
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
    if isinstance(reply, BasicStatus_reply):
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
    if isinstance(reply, Status_reply):
        return None, ValueType.SCALAR

    logger.error("Unknown reply type: %s, cannot extract value", type(reply).__name__)
    return None, None


def _reply_to_reading(reply, drf: str, meta: DeviceMeta | None) -> Reading:
    """Convert a DPM reply to a Reading object."""
    if isinstance(reply, Status_reply):
        facility, error = parse_error(reply.status)
        return Reading(
            drf=drf,
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
        facility_code=facility,
        error_code=error,
        value=value,
        message=status_message(facility, error),
        timestamp=timestamp_from_millis(timestamp) if timestamp else None,
        cycle=cycle,
        meta=meta,
    )


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
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._list_id: int | None = None

    @property
    def list_id(self) -> int:
        if self._list_id is None:
            raise DPMConnectionError("DPM connection has no list ID; connect() must complete first")
        return self._list_id

    async def connect(self) -> None:
        """Connect to DPM, send handshake, read OpenList_reply."""
        import asyncio

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port, limit=MAX_MESSAGE_SIZE),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as e:
            raise DPMConnectionError(f"Connection to {self._host}:{self._port} timed out") from e

        try:
            # Set TCP_NODELAY and SO_KEEPALIVE on the underlying socket
            sock = self._writer.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            self._writer.write(DPM_HANDSHAKE)
            await self._writer.drain()

            # Read OpenList reply (same detection as sync: first 4 bytes)
            try:
                first_bytes = await asyncio.wait_for(self._reader.readexactly(4), timeout=self._timeout)
            except asyncio.TimeoutError as e:
                raise DPMConnectionError("Handshake timed out reading initial reply") from e
            if first_bytes == b"HTTP":
                # Read rest of HTTP status line for useful error message
                try:
                    rest = await asyncio.wait_for(self._reader.readline(), timeout=2.0)
                    status_line = "HTTP" + rest.decode("utf-8", errors="replace").rstrip()
                except Exception:  # noqa: BLE001
                    status_line = "HTTP error (could not read status)"
                raise DPMConnectionError(f"DPM server at {self._host}:{self._port} returned HTTP error: {status_line}")

            length = struct.unpack(">I", first_bytes)[0]
            if length == 0 or length > MAX_MESSAGE_SIZE:
                raise DPMConnectionError(f"Invalid message length: {length}")

            try:
                data = await asyncio.wait_for(self._reader.readexactly(length), timeout=self._timeout)
            except asyncio.TimeoutError as e:
                raise DPMConnectionError("Handshake timed out reading message body") from e
            try:
                reply = unmarshal_reply(iter(data))
            except (ProtocolError, StopIteration) as e:
                raise DPMConnectionError(f"Protocol error during handshake: {e}") from e

            if not isinstance(reply, OpenList_reply):
                raise DPMConnectionError(f"Expected OpenList reply, got {type(reply).__name__}")
            self._list_id = reply.list_id
        except BaseException:
            await self.close()
            raise

    async def send_message(self, msg) -> None:
        """Send a length-prefixed SDD message."""
        if self._writer is None:
            raise DPMConnectionError("Not connected")
        if hasattr(msg, "marshal"):
            data = bytes(msg.marshal())
        else:
            data = bytes(msg)
        self._writer.write(struct.pack(">I", len(data)) + data)
        await self._writer.drain()

    async def send_messages_batch(self, msgs: list) -> None:
        """Send multiple length-prefixed messages in a single TCP write."""
        if self._writer is None:
            raise DPMConnectionError("Not connected")
        buf = bytearray()
        for msg in msgs:
            data = bytes(msg.marshal()) if hasattr(msg, "marshal") else bytes(msg)
            buf.extend(struct.pack(">I", len(data)))
            buf.extend(data)
        self._writer.write(buf)
        await self._writer.drain()

    async def recv_message(self, timeout: float | None = None):
        """Receive and unmarshal one reply. Handles partial packets natively.

        Uses a read timeout to detect silent connection drops. The DPM server
        sends ListStatus_reply heartbeats every ~2s, so if nothing arrives
        within _RECV_TIMEOUT seconds, the connection is presumed dead.
        """
        import asyncio

        if self._reader is None:
            raise DPMConnectionError("Not connected")
        effective_timeout = timeout if timeout is not None else self._RECV_TIMEOUT
        try:
            len_bytes = await asyncio.wait_for(self._reader.readexactly(4), timeout=effective_timeout)
        except asyncio.TimeoutError as e:
            if timeout is not None:
                raise asyncio.TimeoutError("Receive timeout") from e
            raise DPMConnectionError(
                f"No data received for {self._RECV_TIMEOUT}s (missed heartbeats), connection presumed dead"
            ) from e
        length = struct.unpack(">I", len_bytes)[0]
        if length == 0 or length > MAX_MESSAGE_SIZE:
            raise DPMConnectionError(f"Invalid message length: {length}")
        try:
            data = await asyncio.wait_for(self._reader.readexactly(length), timeout=self._RECV_TIMEOUT)
        except asyncio.TimeoutError as e:
            raise DPMConnectionError(f"Timed out reading {length}-byte message body") from e
        try:
            return unmarshal_reply(iter(data))
        except (ProtocolError, StopIteration) as e:
            raise DPMConnectionError(f"Protocol error: {e}") from e

    async def close(self) -> None:
        writer = self._writer
        self._writer = None
        self._reader = None
        self._list_id = None
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to close async DPM connection", exc_info=True)


class _WriteConnection:
    """An authenticated DPM connection for write operations.

    Authentication persists at the DPM list level. Once authenticated and
    EnableSettings is called, the connection can be reused for multiple
    write operations by clearing and restarting the list.

    Reuse flow: StopList -> ClearList -> AddToList -> StartList -> ApplySettings
    """

    def __init__(self, conn: DPMConnection, principal: str, role: str | None):
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
        except Exception:  # noqa: BLE001
            logger.debug("Failed to close DPM write connection", exc_info=True)


class _DPMHTTPSubscriptionHandle(BufferedSubscriptionHandle):
    """Subscription handle for DPMHTTPBackend.

    Each handle corresponds to one async task with its own TCP connection.
    """

    def __init__(
        self,
        backend: "DPMHTTPBackend",
        drfs: list[str],
        callback: ReadingCallback | None,
        on_error: ErrorCallback | None = None,
    ):
        super().__init__()
        self._backend = backend
        self._drfs = drfs
        self._callback = callback
        self._is_callback_mode = callback is not None
        self._on_error = on_error
        self._ref_ids = list(range(1, len(drfs) + 1))
        self._task: asyncio.Task | None = None

    def _dispatch(self, reading: Reading) -> None:
        """Called from the reactor thread to deliver a reading."""
        if self._stopped:
            return
        if self._callback is not None:
            self._backend._dispatcher.dispatch_reading(self._callback, reading, self)
        else:
            super()._dispatch(reading)

    def _dispatch_error(self, exc: Exception) -> None:
        """Called from the reactor thread on stream error (always fatal for DPM)."""
        self._signal_error(exc)
        if self._on_error is not None:
            self._backend._dispatcher.dispatch_error(self._on_error, exc, self)

    def stop(self) -> None:
        """Stop this subscription and cancel its async task."""
        if not self._stopped:
            self._backend.remove(self)


class _DpmStreamCore:
    """Pure-async DPM streaming protocol logic.

    Manages AddToList/StartList setup and the recv loop for a single
    streaming subscription. Takes functional callbacks for dispatch,
    stop checking, and error handling - knows nothing about threads,
    handles, or user callbacks.
    """

    def __init__(self, conn: _AsyncDPMConnection):
        self._conn = conn

    async def stream(
        self,
        drfs: list[str],
        dispatch_fn,
        stop_check,
        error_fn,
    ) -> None:
        import asyncio

        metas: dict[int, DeviceMeta] = {}
        drf_map: dict[int, str] = {}

        try:
            list_id = self._conn.list_id

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
            await self._conn.send_messages_batch(setup_msgs)

            # Receive loop
            while not stop_check():
                reply = await self._conn.recv_message()

                if isinstance(reply, AddToList_reply):
                    if reply.status != 0:
                        drf = drf_map.get(reply.ref_id)
                        if drf is not None:
                            facility, error = parse_error(reply.status)
                            reading = Reading(
                                drf=drf,
                                facility_code=facility,
                                error_code=error,
                                value=None,
                                message=status_message(facility, error) or f"AddToList failed (status={reply.status})",
                                timestamp=None,
                                cycle=0,
                                meta=None,
                            )
                            dispatch_fn(reading)
                    continue

                if isinstance(reply, StartList_reply):
                    if reply.status != 0:
                        drf_summary = summarize_drfs(drfs)
                        logger.warning("StartList returned status %d (devices: %s)", reply.status, drf_summary)
                        error_fn(
                            DPMConnectionError(f"StartList failed (status={reply.status}, devices: {drf_summary})")
                        )
                        return
                    continue

                if isinstance(reply, ListStatus_reply):
                    continue

                if isinstance(reply, DeviceInfo_reply):
                    metas[reply.ref_id] = _device_info_to_meta(reply)
                    continue

                if isinstance(reply, Status_reply) and reply.ref_id == 0:
                    if reply.status != 0:
                        facility, error = parse_error(reply.status)
                        message = status_message(facility, error) or f"status={reply.status}"
                        error_fn(DPMConnectionError(f"DPM job start failed: {message}"))
                        return
                    continue

                if hasattr(reply, "ref_id"):
                    ref_id = reply.ref_id
                    drf = drf_map.get(ref_id)
                    if drf is None:
                        logger.warning("Data for unknown ref_id=%s", ref_id)
                        continue
                    meta = metas.get(ref_id)
                    reading = _reply_to_reading(reply, drf, meta)
                    dispatch_fn(reading)

        except asyncio.CancelledError:
            pass  # Normal shutdown via task.cancel()
        except (asyncio.IncompleteReadError, DPMConnectionError, OSError) as e:
            if not stop_check():
                drf_summary = summarize_drfs(drfs)
                wrapped = DPMConnectionError(f"{e} (devices: {drf_summary})")
                wrapped.__cause__ = e
                error_fn(wrapped)
        except Exception as e:  # noqa: BLE001
            if not stop_check():
                drf_summary = summarize_drfs(drfs)
                logger.error("Unexpected streaming error: %s (devices: %s)", e, drf_summary)
                error_fn(e)


class DPMHTTPBackend(Backend):
    """
    DPM HTTP Backend for ACNET device access.

    Uses the TCP/PC protocol to communicate with DPM via acsys-proxy.
    Supports multiple independent streaming subscriptions, each with its
    own async TCP connection on a shared asyncio reactor thread.

    Design note: RemoveFromList is not implemented. Each subscription gets
    its own TCP connection with an independent DPM list, so partial device
    removal has no use case -- call remove(handle) to tear down an entire
    subscription instead.

    Capabilities:
        - READ: Always enabled
        - WRITE: Enabled when auth is KerberosAuth (role optional — console class writes don't need it)
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
        auth: Auth | None = None,
        role: str | None = None,
        dispatch_mode: DispatchMode = DispatchMode.WORKER,
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
        self._auth: KerberosAuth | None = auth
        self._role = role
        self._pool: ConnectionPool | None = None
        self._pool_lock = threading.Lock()
        self._closed = False

        # Callback dispatcher
        self._dispatch_mode = dispatch_mode
        self._dispatcher = CallbackDispatcher(dispatch_mode)

        # Streaming state -- asyncio reactor (matches gRPC backend pattern)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reactor_thread: threading.Thread | None = None
        self._reactor_lock = threading.Lock()
        self._handles: list[_DPMHTTPSubscriptionHandle] = []
        self._handles_lock = threading.Lock()

        # Write connection pool - authenticated connections for writes
        # Authentication persists at list level; reuse via StopList + ClearList
        self._write_connections: list[_WriteConnection] = []
        self._write_pool_size = 2  # Max authenticated write connections
        self._write_lock = threading.Lock()
        self._write_idle_timeout = 60.0  # Close connections idle > 60s
        self._write_in_flight = 0  # Connections currently checked out for writes

        # Validate auth eagerly — but skip for lazy auth (validated on first write)
        if self._auth is not None and not getattr(self._auth, "_lazy", False):
            _ = self._auth.principal  # This validates credentials

        logger.debug(
            "DPMHTTPBackend initialized: host=%s, port=%s, pool_size=%s, timeout=%s, auth=%s, role=%s",
            host,
            port,
            pool_size,
            timeout,
            type(auth).__name__ if auth else None,
            role,
        )

    @property
    def capabilities(self) -> BackendCapability:
        """Backend capabilities based on configuration."""
        caps = BackendCapability.READ | BackendCapability.BATCH | BackendCapability.STREAM

        if isinstance(self._auth, KerberosAuth):
            caps |= BackendCapability.AUTH_KERBEROS | BackendCapability.WRITE

        return caps

    @property
    def authenticated(self) -> bool:
        """True if backend is configured for authenticated operations."""
        return self._auth is not None

    @property
    def principal(self) -> str | None:
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
                if self._closed:
                    raise RuntimeError("Backend is closed")
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

    def read(self, drf: str, timeout: float | None = None) -> Value:
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

    def get(self, drf: str, timeout: float | None = None) -> Reading:
        """Read a single device with full metadata."""
        readings = self.get_many([drf], timeout=timeout)
        return readings[0]

    def get_many(self, drfs: list[str], timeout: float | None = None) -> list[Reading]:
        """Read multiple devices in a single batch."""
        if not drfs:
            return []

        effective_timeout = timeout if timeout is not None else self._timeout
        deadline = time.monotonic() + effective_timeout

        prepared_drfs = [ensure_immediate_event(drf) for drf in drfs]

        # Chunked logger DRFs arrive in 487-point chunks with a final empty chunk.
        # Pre-detect so we accumulate chunks instead of stopping at the first.
        chunked_logger_refs: set[int] = set()
        for i, drf in enumerate(prepared_drfs):
            if is_chunked_historical_drf(drf):
                chunked_logger_refs.add(i + 1)

        device_infos: dict[int, DeviceInfo_reply] = {}
        data_replies: dict[int, object] = {}
        logger_chunks: dict[int, list] = {}  # ref_id -> accumulated chunks
        logger_complete: set[int] = set()  # ref_ids that received the empty terminator
        add_errors: dict[int, AddToList_reply] = {}  # ref_id -> failed AddToList
        received_count = 0
        expected_count = len(drfs)
        job_error: int | None = None  # ref-0 Status_reply = job start failure

        # Repeating events (@p/@e/...) keep producing replies after the first —
        # a connection that carried one must be closed, not re-pooled, or stale
        # replies get attributed to the next borrower's refs.
        reuse_safe = all(is_historical_drf(d) or is_immediate_only(d) for d in prepared_drfs)

        pool = self._get_pool()
        conn_broken = False
        transport_error: BaseException | None = None

        try:
            with pool.connection(wait_timeout=effective_timeout) as conn:
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
                            if (
                                reply.status != 0
                                and 1 <= reply.ref_id <= expected_count
                                and reply.ref_id not in add_errors
                            ):
                                add_errors[reply.ref_id] = reply
                                received_count += 1
                        elif isinstance(reply, DeviceInfo_reply):
                            if 1 <= reply.ref_id <= expected_count:
                                device_infos[reply.ref_id] = reply
                        elif isinstance(reply, StartList_reply):
                            if reply.status != 0:
                                drf_summary = summarize_drfs(drfs)
                                logger.warning("StartList returned status %d (devices: %s)", reply.status, drf_summary)
                                break  # No data will arrive
                        elif isinstance(reply, ListStatus_reply):
                            pass
                        elif isinstance(reply, Status_reply):
                            ref_id = reply.ref_id
                            if ref_id == 0:
                                # On the TCP transport StartList_reply.status is hardwired OK;
                                # a ref-0 Status_reply is the real job-start-failure signal.
                                if reply.status != 0 and job_error is None:
                                    job_error = reply.status
                            elif ref_id in chunked_logger_refs:
                                # Error for a logger DRF — record as an error chunk
                                if ref_id not in logger_complete:
                                    logger_chunks.setdefault(ref_id, []).append(reply)
                                    logger_complete.add(ref_id)
                                    received_count += 1
                            elif 1 <= ref_id <= expected_count and ref_id not in data_replies:
                                data_replies[ref_id] = reply
                                received_count += 1
                        elif hasattr(reply, "ref_id"):
                            ref_id = reply.ref_id
                            if not (1 <= ref_id <= expected_count):
                                pass  # stale/unknown ref — never count toward expected_count
                            elif ref_id in chunked_logger_refs:
                                # Logger: accumulate chunks; empty chunk = done
                                is_empty = (
                                    isinstance(reply, (TimedScalarArray_reply, ScalarArray_reply))
                                    and len(reply.data) == 0
                                )
                                if is_empty:
                                    if ref_id not in logger_complete:
                                        if hasattr(reply, "status") and reply.status != 0:
                                            # Accumulate error terminators so aggregation surfaces the error.
                                            logger_chunks.setdefault(ref_id, []).append(reply)
                                        logger_complete.add(ref_id)
                                        received_count += 1
                                else:
                                    logger_chunks.setdefault(ref_id, []).append(reply)
                            elif ref_id not in data_replies:
                                data_replies[ref_id] = reply
                                received_count += 1
                except (BrokenPipeError, ConnectionResetError, OSError, DPMConnectionError) as e:
                    conn_broken = True
                    transport_error = e
                finally:
                    if not conn_broken:
                        if job_error is not None or received_count < expected_count:
                            conn.close()
                        else:
                            try:
                                stop_req = StopList_request()
                                stop_req.list_id = list_id
                                clear_req = ClearList_request()
                                clear_req.list_id = list_id
                                conn.send_messages_batch([stop_req, clear_req])
                            except Exception as e:  # noqa: BLE001
                                # Failed StopList send means unknown connection state —
                                # close so it is not re-pooled dirty. Data is already
                                # complete; don't destroy the readings over cleanup.
                                logger.warning("StopList cleanup failed: %s", e, exc_info=True)
                                conn.close()
                            else:
                                if not reuse_safe:
                                    conn.close()
                    else:
                        # The decode boundary is intact, but the list is still active.
                        conn.close()
        except (PoolClosedError, PoolExhaustedError, DPMConnectionError, OSError) as e:
            transport_error = e

        readings: list[Reading] = []
        has_timeout = False

        for i, original_drf in enumerate(drfs):
            ref_id = i + 1
            info = device_infos.get(ref_id)
            reply = data_replies.get(ref_id)
            chunks = logger_chunks.get(ref_id)
            add_err = add_errors.get(ref_id)

            meta = _device_info_to_meta(info) if info else None

            if add_err is not None:
                facility, error = parse_error(add_err.status)
                readings.append(
                    Reading(
                        drf=original_drf,
                        facility_code=facility,
                        error_code=error,
                        value=None,
                        message=status_message(facility, error) or f"AddToList failed (status={add_err.status})",
                        timestamp=None,
                        cycle=0,
                        meta=meta,
                    )
                )
                continue

            if ref_id in chunked_logger_refs:
                if ref_id in logger_complete and chunks:
                    # Complete logger response with data
                    readings.append(_aggregate_logger_chunks(chunks, original_drf, meta))
                elif ref_id in logger_complete:
                    # Empty window (terminator received, no data chunks) — valid empty result
                    import numpy as np

                    readings.append(
                        Reading(
                            drf=original_drf,
                            value_type=ValueType.TIMED_SCALAR_ARRAY,
                            value={"data": np.array([], dtype=float), "micros": np.array([], dtype=np.int64)},
                            timestamp=None,
                            meta=meta,
                        )
                    )
                else:
                    # No terminator — partial data or timeout
                    has_timeout = True
                    if job_error is not None:
                        fc, ec = parse_error(job_error)
                        msg = status_message(fc, ec) or f"DPM job start failed (status={job_error})"
                    else:
                        fc = FACILITY_ACNET
                        ec = ERR_RETRY if transport_error is not None else ERR_TIMEOUT
                        msg = (
                            f"Connection error: {transport_error}"
                            if transport_error is not None
                            else "Logger response incomplete"
                        )
                    readings.append(
                        Reading(
                            drf=original_drf,
                            facility_code=fc,
                            error_code=ec,
                            value=None,
                            message=msg,
                            timestamp=None,
                            meta=meta,
                        )
                    )
                continue

            if reply is None:
                has_timeout = True
                if job_error is not None:
                    fc, ec = parse_error(job_error)
                    msg = status_message(fc, ec) or f"DPM job start failed (status={job_error})"
                else:
                    fc = FACILITY_ACNET
                    ec = ERR_RETRY if transport_error is not None else ERR_TIMEOUT
                    msg = f"Connection error: {transport_error}" if transport_error is not None else "Request timeout"
                readings.append(
                    Reading(
                        drf=original_drf,
                        facility_code=fc,
                        error_code=ec,
                        value=None,
                        message=msg,
                        timestamp=None,
                        cycle=0,
                        meta=meta,
                    )
                )
            else:
                readings.append(_reply_to_reading(reply, original_drf, meta))

        if transport_error is not None or has_timeout:
            if job_error is not None:
                fc, ec = parse_error(job_error)
                raise ReadError(readings, f"DPM job start failed: {status_message(fc, ec) or job_error}")
            raise ReadError(readings, str(transport_error or "Request timeout")) from transport_error

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
        gssapi = _require_gssapi()
        from gssapi import exceptions as gssapi_exceptions

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
        logger.debug("DPM service name: %s", gss_name)

        # Phase 2: create GSSAPI context with server's actual service name
        try:
            service_name = gssapi.Name(gss_name, gssapi.NameType.kerberos_principal)

            assert self._auth is not None
            creds = self._auth._get_credentials()
            ctx = gssapi.SecurityContext(
                name=service_name,
                usage="initiate",
                creds=creds,
                flags=(
                    gssapi.RequirementFlag.replay_detection
                    | gssapi.RequirementFlag.integrity
                    | gssapi.RequirementFlag.out_of_sequence_detection
                ),
                mech=gssapi.MechType.kerberos,
            )

            token = ctx.step()
        except gssapi_exceptions.GSSError as e:
            raise AuthenticationError(f"Kerberos authentication failed for {gss_name}: {e}") from e

        auth_req = Authenticate_request()
        auth_req.list_id = conn.list_id
        auth_req.token = bytes(token) if token else b""
        conn.send_message(auth_req)

        reply = conn.recv_message(timeout=self._timeout)
        if not isinstance(reply, Authenticate_reply):
            raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        if hasattr(reply, "token") and reply.token and not ctx.complete:
            try:
                token = ctx.step(reply.token)
            except gssapi_exceptions.GSSError as e:
                raise AuthenticationError(f"Kerberos authentication failed for {gss_name}: {e}") from e
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
        try:
            mic = ctx.get_signature(message)
        except gssapi_exceptions.GSSError as e:
            raise AuthenticationError(f"Kerberos authentication failed for {gss_name}: {e}") from e

        logger.debug("Kerberos authentication complete for %s", self._auth.principal if self._auth else "unknown")
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
            RuntimeError: If the backend was closed concurrently
        """
        assert self._auth is not None, "Auth required for write connections"
        current_principal = self._auth.principal
        current_role = self._role

        with self._write_lock:
            if self._closed:
                raise RuntimeError("Backend is closed")

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
                wc = self._write_connections.pop()
                if not wc.conn.connected:
                    logger.debug("Discarding dead write connection (list_id=%s)", wc.conn.list_id)
                    wc.close()
                    continue
                if wc.principal != current_principal or wc.role != current_role:
                    logger.debug(
                        "Discarding write connection with stale auth context (list_id=%s, principal=%s, role=%s)",
                        wc.conn.list_id,
                        wc.principal,
                        wc.role,
                    )
                    wc.close()
                    continue
                wc.last_used = time.monotonic()
                self._write_in_flight += 1
                logger.debug("Reusing authenticated write connection (list_id=%s)", wc.conn.list_id)
                return wc

            # Pool exhausted - check concurrent limit before creating new
            if self._write_in_flight >= _MAX_WRITE_CONNECTIONS:
                raise PoolExhaustedError(f"Too many concurrent write connections ({_MAX_WRITE_CONNECTIONS})")
            self._write_in_flight += 1

        # Create new connection outside the lock
        conn = DPMConnection(host=self._host, port=self._port, timeout=self._timeout)
        try:
            conn.connect()
            wc = _WriteConnection(conn, current_principal, current_role)
            mic, message = self._authenticate_connection(conn)
            self._enable_settings(conn, mic, message)
            wc.authenticated = True
            logger.debug("Created new authenticated write connection (list_id=%s)", conn.list_id)
        except Exception:
            with self._write_lock:
                self._write_in_flight -= 1
            conn.close()
            raise

        return wc

    def _release_write_connection(self, wc: _WriteConnection) -> None:
        """Return a write connection to the pool for reuse.

        The connection should be in a clean state (list stopped).
        """
        wc.last_used = time.monotonic()

        with self._write_lock:
            self._write_in_flight -= 1
            if self._closed:
                wc.close()
                return
            if len(self._write_connections) < self._write_pool_size:
                self._write_connections.append(wc)
                logger.debug("Returned write connection to pool (list_id=%s)", wc.conn.list_id)
            else:
                # Pool full, close this one
                wc.close()
                logger.debug("Write pool full, closed connection (list_id=%s)", wc.conn.list_id)

    def _discard_write_connection(self, wc: _WriteConnection) -> None:
        """Discard a broken write connection without returning to pool."""
        with self._write_lock:
            self._write_in_flight -= 1
        wc.close()
        logger.debug("Discarded broken write connection (list_id=%s)", wc.conn.list_id)

    def _close_write_connections(self) -> None:
        """Close all write connections."""
        with self._write_lock:
            for wc in self._write_connections:
                wc.close()
            self._write_connections.clear()
            logger.debug("Closed all write connections")

    # Writable alarm dict keys → DRF field names, keyed by DRF property.
    # "abort" and "alarm_status" are read-only status bits, not settable.
    _ANALOG_ALARM_FIELDS: ClassVar[dict[str, str]] = {
        "minimum": "MIN",
        "maximum": "MAX",
        "alarm_enable": "ALARM_ENABLE",
        "abort_inhibit": "ABORT_INHIBIT",
        "tries_needed": "TRIES_NEEDED",
    }
    _DIGITAL_ALARM_FIELDS: ClassVar[dict[str, str]] = {
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

        keys = set(alarm_dict)
        readonly = keys & ALARM_READONLY_KEYS
        if readonly:
            raise ValueError(f"Read-only alarm dict keys cannot be written: {readonly}")
        bad_keys = keys - set(field_map)
        if bad_keys:
            raise ValueError(f"Unknown {prop_name} alarm keys: {bad_keys}")
        if not keys:
            raise ValueError("Alarm dict must include at least one writable key")

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
        timeout: float | None = None,
    ) -> WriteResult:
        """Write a single device value."""
        # DPM/HTTP has no structured alarm setting type -- expand dict to
        # sequential per-field writes.  They must be sequential because alarm
        # fields share the same 20-byte block on the server; a single
        # ApplySettings with multiple fields on the same device causes the
        # later field to overwrite the earlier one.
        if isinstance(value, dict):
            pairs = self._expand_alarm_dict(drf, value)
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
        setting_payloads: list[_SettingPayload],
        deadline: float,
    ) -> tuple[ApplySettings_reply | None, dict[int, int]]:
        """Execute the write protocol on an authenticated connection.

        Returns (ApplySettings_reply, add_errors) or (None, add_errors) on timeout.
        Raises connection errors for retry handling by caller.
        """
        add_errors: dict[int, int] = {}
        # Batch all setup messages into a single TCP write
        setup_msgs: list = []

        # Stop and clear previous requests from reused connection
        stop_req = StopList_request()
        stop_req.list_id = list_id
        setup_msgs.append(stop_req)

        clear_req = ClearList_request()
        clear_req.list_id = list_id
        setup_msgs.append(clear_req)

        # Set ROLE list property (optional — console class writes don't need it)
        if self._role is not None:
            role_req = AddToList_request()
            role_req.list_id = list_id
            role_req.ref_id = 0
            role_req.drf_request = f"#ROLE:{self._role}"
            setup_msgs.append(role_req)

        # Add devices to list
        for i, (drf, _) in enumerate(prepared_settings):
            add_req = AddToList_request()
            add_req.list_id = list_id
            add_req.ref_id = i + 1
            add_req.drf_request = drf
            setup_msgs.append(add_req)

        # Start list
        start_req = StartList_request()
        start_req.list_id = list_id
        setup_msgs.append(start_req)

        conn.send_messages_batch(setup_msgs)

        # Wait for device info / add replies before sending settings
        received_infos = 0
        expected_count = len(prepared_settings)
        received_start_list_reply = False
        seen_refs: set[int] = set()  # count each ref at most once

        while received_infos < expected_count or not received_start_list_reply:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                reply = conn.recv_message(timeout=min(remaining, 2.0))
            except TimeoutError:
                if time.monotonic() >= deadline:
                    break
                continue

            if isinstance(reply, ListStatus_reply):
                pass
            elif isinstance(reply, AddToList_reply):
                if reply.status != 0 and 1 <= reply.ref_id <= expected_count and reply.ref_id not in seen_refs:
                    add_errors[reply.ref_id] = reply.status
                    seen_refs.add(reply.ref_id)
                    received_infos += 1
            elif isinstance(reply, DeviceInfo_reply):
                if 1 <= reply.ref_id <= expected_count and reply.ref_id not in seen_refs:
                    seen_refs.add(reply.ref_id)
                    received_infos += 1
            elif isinstance(reply, StartList_reply):
                received_start_list_reply = True
                if reply.status != 0:
                    write_drfs = [drf for drf, _ in prepared_settings]
                    drf_summary = summarize_drfs(write_drfs)
                    logger.warning("StartList returned status %d (devices: %s)", reply.status, drf_summary)
                    return None, add_errors
            elif isinstance(reply, Status_reply):
                ref_id = reply.ref_id
                if ref_id == 0:
                    # Job-start failure: StartList_reply.status is hardwired OK on the
                    # TCP transport — this is the real signal. Surface via add_errors[0].
                    if reply.status != 0:
                        add_errors[0] = reply.status
                        return None, add_errors
                elif 1 <= ref_id <= expected_count and ref_id not in seen_refs:
                    if reply.status != 0:
                        add_errors[ref_id] = reply.status
                    seen_refs.add(ref_id)
                    received_infos += 1

        if received_infos < expected_count or not received_start_list_reply:
            write_drfs = [drf for drf, _ in prepared_settings]
            drf_summary = summarize_drfs(write_drfs)
            logger.warning(
                "Write setup timed out: received %d/%d device infos, StartList_reply=%s (devices: %s)",
                received_infos,
                expected_count,
                received_start_list_reply,
                drf_summary,
            )
            return None, add_errors

        # Build and send ApplySettings
        apply_req = ApplySettings_request()
        apply_req.user_name = self._auth.principal if self._auth else ""
        apply_req.list_id = list_id

        raw_settings = []
        scaled_settings = []
        text_settings = []

        for raw, scaled, text in setting_payloads:
            if raw:
                raw_settings.append(raw)
            if scaled:
                scaled_settings.append(scaled)
            if text:
                text_settings.append(text)

        if raw_settings:
            setattr(apply_req, "raw_array", raw_settings)
        if scaled_settings:
            setattr(apply_req, "scaled_array", scaled_settings)
        if text_settings:
            setattr(apply_req, "text_array", text_settings)

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
                return reply, add_errors
            if isinstance(reply, ListStatus_reply):
                pass

        return None, add_errors

    def write_many(
        self,
        settings: list[tuple[str, Value]],
        timeout: float | None = None,
    ) -> list[WriteResult]:
        """Write multiple device values.

        Uses a pool of authenticated connections for efficient repeated writes.
        Authentication is cached at the DPM list level and reused across calls.
        """
        if not settings:
            return []

        if self._closed:
            raise RuntimeError("Backend is closed")

        if not isinstance(self._auth, KerberosAuth):
            raise AuthenticationError("Backend not configured for authenticated operations. Pass auth=KerberosAuth().")

        effective_timeout = timeout if timeout is not None else self._timeout

        # Prepare settings (add .SETTING and @I if needed)
        prepared_settings = [(prepare_for_write(drf), value) for drf, value in settings]
        setting_payloads = [_value_to_setting(i, value) for i, (_, value) in enumerate(settings, 1)]

        # Try up to twice: first attempt may hit a stale pooled connection
        add_errors: dict[int, int] = {}
        last_error = None
        for attempt in range(2):
            deadline = time.monotonic() + effective_timeout

            try:
                wc = self._get_write_connection()
            except (AuthenticationError, ImportError, RuntimeError):
                # Auth failures and closed-backend are caller bugs - fail fast
                raise
            except (DPMConnectionError, OSError, PoolExhaustedError) as e:
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
                apply_reply, add_errors = self._execute_write(
                    conn, list_id, prepared_settings, setting_payloads, deadline
                )

                if apply_reply is None:
                    # Timeout: server's late reply may still be in the TCP stream,
                    # so discard the connection to avoid corrupting the next write.
                    self._discard_write_connection(wc)
                    break

                # Stop list (but keep connection and auth for reuse)
                stop_req = StopList_request()
                stop_req.list_id = list_id
                conn.send_message(stop_req)

                self._release_write_connection(wc)
                last_error = None
                break  # Success

            except (BrokenPipeError, ConnectionResetError, OSError, DPMConnectionError) as e:
                logger.warning("Write connection error (attempt %s): %s", attempt + 1, e)
                self._discard_write_connection(wc)
                last_error = e
                if attempt == 0:
                    continue  # Retry with fresh connection
            except Exception as e:  # noqa: BLE001
                logger.warning("Unexpected write error: %s", e)
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
            job_err = add_errors.get(0)  # ref-0 status = job start failure
            results: list[WriteResult] = []
            for i, (drf, _) in enumerate(settings):
                ref_id = i + 1
                if ref_id in add_errors:
                    facility, error = parse_error(add_errors[ref_id])
                    results.append(
                        WriteResult(
                            drf=drf,
                            facility_code=facility,
                            error_code=error,
                            message=status_message(facility, error)
                            or f"AddToList failed (status={add_errors[ref_id]})",
                        )
                    )
                elif job_err is not None:
                    facility, error = parse_error(job_err)
                    results.append(
                        WriteResult(
                            drf=drf,
                            facility_code=facility,
                            error_code=error,
                            message=status_message(facility, error) or f"DPM job start failed (status={job_err})",
                        )
                    )
                else:
                    results.append(
                        WriteResult(
                            drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_TIMEOUT, message="Request timeout"
                        )
                    )
            return results

        status_map: dict[int, int] = {}
        for status_struct in apply_reply.status:
            status_map[status_struct.ref_id] = status_struct.status

        global_err = status_map.get(0)

        results: list[WriteResult] = []
        for i, (drf, _) in enumerate(settings):
            ref_id = i + 1

            if ref_id in add_errors:
                facility, error = parse_error(add_errors[ref_id])
                results.append(
                    WriteResult(
                        drf=drf,
                        facility_code=facility,
                        error_code=error,
                        message=status_message(facility, error) or f"AddToList failed (status={add_errors[ref_id]})",
                    )
                )
                continue

            if ref_id in status_map:
                status = status_map[ref_id]
                if status == 0:
                    results.append(WriteResult(drf=drf, error_code=ERR_OK))
                else:
                    facility, error = parse_error(status)
                    results.append(
                        WriteResult(
                            drf=drf,
                            facility_code=facility,
                            error_code=error,
                            message=status_message(facility, error)
                            or f"Write error (facility={facility}, error={error})",
                        )
                    )
            elif global_err is not None and global_err != 0:
                facility, error = parse_error(global_err)
                results.append(
                    WriteResult(
                        drf=drf,
                        facility_code=facility,
                        error_code=error,
                        message=status_message(facility, error) or f"Global error {global_err}",
                    )
                )
            else:
                results.append(
                    WriteResult(
                        drf=drf,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_TIMEOUT,
                        message="No reply from server",
                    )
                )

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Streaming Methods -- asyncio reactor
    # ─────────────────────────────────────────────────────────────────────────

    def _start_reactor(self) -> None:
        """Start the reactor thread and event loop. Must hold _reactor_lock."""
        import asyncio

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

        Creates its own TCP connection, delegates protocol logic to
        _DpmStreamCore, owns connection lifecycle and handle cleanup.
        """
        conn = _AsyncDPMConnection(self._host, self._port, self._timeout)
        try:
            # Catch connect failures only: core.stream reports its own errors
            # via error_fn, so a broader catch could double-dispatch.
            try:
                await conn.connect()
            except Exception as e:  # noqa: BLE001
                if not handle._stopped:
                    drfs = handle._drfs
                    summary = summarize_drfs(drfs)
                    logger.error("Subscription connection failed for %s: %s", summary, e)
                    handle._dispatch_error(e)
                return
            core = _DpmStreamCore(conn)
            await core.stream(
                drfs=handle._drfs,
                dispatch_fn=handle._dispatch,
                stop_check=lambda: handle._stopped,
                error_fn=handle._dispatch_error,
            )
        finally:
            await conn.close()
            handle._signal_stop()
            with self._handles_lock:
                if handle in self._handles:
                    self._handles.remove(handle)

    def subscribe(
        self,
        drfs: list[str],
        callback: ReadingCallback | None = None,
        on_error: ErrorCallback | None = None,
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

        import asyncio

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

        with self._handles_lock:
            # Re-check under the lock: close() sets _closed before
            # stop_streaming() drains _handles, so an append here either
            # happens-before the drain or raises (grpc has the same guard).
            if self._closed:
                raise RuntimeError("Backend is closed")
            self._handles.append(handle)
        future = None
        try:
            future = asyncio.run_coroutine_threadsafe(_create_task(), self._loop)
            handle._task = future.result(timeout=5.0)
        except Exception:
            handle._signal_stop()
            if future is not None:
                future.cancel()
            with self._handles_lock:
                if handle in self._handles:
                    self._handles.remove(handle)
            raise

        mode_str = "callback" if handle._is_callback_mode else "iterator"
        logger.info("Created %s subscription for %s devices", mode_str, len(drfs))
        return handle

    def remove(self, handle: SubscriptionHandle) -> None:
        """Remove a subscription. Cancels the associated async task."""
        if not isinstance(handle, _DPMHTTPSubscriptionHandle):
            raise TypeError(f"Expected _DPMHTTPSubscriptionHandle, got {type(handle).__name__}")

        handle._signal_stop()

        if handle._task is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(handle._task.cancel)

        with self._handles_lock:
            if handle in self._handles:
                self._handles.remove(handle)

        logger.info("Removed DPM subscription for %s devices", len(handle._drfs))

    def stop_streaming(self) -> None:
        """Stop all streaming subscriptions."""
        with self._handles_lock:
            handles = list(self._handles)
            self._handles.clear()

        for handle in handles:
            handle._signal_stop()
            if handle._task is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(handle._task.cancel)

        logger.info("All DPM streaming stopped")

    def close(self) -> None:
        """Close the backend and release all resources."""
        with self._reactor_lock:
            if self._closed:
                return
            self._closed = True

        # Stop streaming first
        self.stop_streaming()
        self._dispatcher.close()

        # Stop the event loop and join reactor thread
        loop = self._loop
        thread = self._reactor_thread
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("Reactor thread did not stop within 2s")
            else:
                self._reactor_thread = None
        else:
            self._reactor_thread = None
        self._loop = None

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
        return (
            f"DPMHTTPBackend({self._host}:{self._port}, pool_size={self._pool_size}{auth_info}, "
            f"subs={n_subs}, {status})"
        )


__all__ = ["DPMHTTPBackend"]
