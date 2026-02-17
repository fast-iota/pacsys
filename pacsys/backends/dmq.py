"""
DMQ Backend - RabbitMQ/AMQP backend for ACNET device access.

Uses RabbitMQ message broker to communicate with ACNET via the DMQ impl1
server (reference_code/gov/fnal/controls/service/dmq/impl/).
"""

import logging
import os
import socket
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pika
import pika.spec
from pika.adapters.select_connection import SelectConnection
from pika.channel import Channel

from pacsys.acnet.errors import ERR_OK, ERR_RETRY, ERR_TIMEOUT, FACILITY_ACNET, FACILITY_DMQ
from pacsys.auth import Auth, KerberosAuth
from pacsys.backends import Backend, timestamp_from_millis, validate_alarm_dict
from pacsys.backends._dispatch import CallbackDispatcher
from pacsys.backends._subscription import BufferedSubscriptionHandle
from pacsys.errors import AuthenticationError, DeviceError, ReadError
from pacsys.backends.dmq_protocol import (
    ReadingRequest_request,
    SettingRequest_request,
    unmarshal_reply,
    DoubleSample_reply,
    DoubleArraySample_reply,
    StringSample_reply,
    StringArraySample_reply,
    ErrorSample_reply,
    BasicStatusSample_reply,
    BasicControlSample_reply,
    AnalogAlarmSample_reply,
    AnalogAlarm_struct,
    DigitalAlarmSample_reply,
    DigitalAlarm_struct,
    BinarySample_reply,
    IntegerSample_reply,
    IntegerArraySample_reply,
    ShortSample_reply,
    ShortArraySample_reply,
    LongSample_reply,
    LongArraySample_reply,
    BooleanSample_reply,
    BooleanArraySample_reply,
    BasicControl_Reset,
    BasicControl_On,
    BasicControl_Off,
    BasicControl_Positive,
    BasicControl_Negative,
    BasicControl_Ramp,
    BasicControl_DC,
)
from pacsys.types import (
    BackendCapability,
    BasicControl,
    DispatchMode,
    ErrorCallback,
    Reading,
    ReadingCallback,
    SubscriptionHandle,
    Value,
    ValueType,
    WriteResult,
)
from pacsys.drf_utils import ensure_immediate_event, get_device_name, prepare_for_write

logger = logging.getLogger(__name__)

# Enable verbose protocol tracing for write channel messages
TRACE = False

DEFAULT_HOST = os.environ.get("PACSYS_DMQ_HOST", "appsrv2.fnal.gov")
DEFAULT_PORT = int(os.environ.get("PACSYS_DMQ_PORT", "5672"))
DEFAULT_VHOST = "/"
DEFAULT_TIMEOUT = 10.0
INIT_EXCHANGE = "amq.topic"

# PENDING status in DMQ - errorNumber=1 means init in progress
DMQ_PENDING_ERROR = 1

# GSS-API service principal for DMQ authenticated operations
DMQ_SERVICE_PRINCIPAL = "daeset/bd/dmq.fnal.gov@FNAL.GOV"

# Write connection heartbeat interval (seconds)
WRITE_HEARTBEAT_INTERVAL = 5.0

# Default write session idle TTL (seconds) - close session if unused for this long
DEFAULT_WRITE_SESSION_TTL = 600.0

# Maximum concurrent write sessions (protects against channel exhaustion)
MAX_WRITE_SESSIONS = 1024

# Map backend-agnostic BasicControl enum → SDD protocol constants.
# Only commands 0-6 have SDD enum values; LOCAL/REMOTE/TRIP (7-9) are
# sent as DoubleSample since the DMQ proto enum lacks them.
_BASIC_CONTROL_TO_SDD = {
    BasicControl.RESET: BasicControl_Reset,
    BasicControl.ON: BasicControl_On,
    BasicControl.OFF: BasicControl_Off,
    BasicControl.POSITIVE: BasicControl_Positive,
    BasicControl.NEGATIVE: BasicControl_Negative,
    BasicControl.RAMP: BasicControl_Ramp,
    BasicControl.DC: BasicControl_DC,
}


def _dict_to_alarm_sample(d: dict, ref_id: int, timestamp_ms: int):
    """Convert alarm dict to AnalogAlarmSample_reply or DigitalAlarmSample_reply.

    Requires at least one type-specific key (minimum/maximum for analog,
    nominal/mask for digital) to disambiguate alarm type.
    """
    alarm_type = validate_alarm_dict(d)
    if alarm_type == "analog":
        alarm = AnalogAlarm_struct()
        if "minimum" in d:
            alarm.minimum = float(d["minimum"])
        if "maximum" in d:
            alarm.maximum = float(d["maximum"])
        if "alarm_enable" in d:
            alarm.alarm_enable = bool(d["alarm_enable"])
        if "abort_inhibit" in d:
            alarm.abort_inhibit = bool(d["abort_inhibit"])
        if "tries_needed" in d:
            alarm.tries_needed = int(d["tries_needed"])
        sample = AnalogAlarmSample_reply()
        sample.value = alarm
        sample.time = timestamp_ms
        sample.ref_id = ref_id  # type: ignore[attr-defined]
        return sample
    else:
        alarm = DigitalAlarm_struct()
        if "nominal" in d:
            alarm.nominal = int(d["nominal"])
        if "mask" in d:
            alarm.mask = int(d["mask"])
        if "alarm_enable" in d:
            alarm.alarm_enable = bool(d["alarm_enable"])
        if "abort_inhibit" in d:
            alarm.abort_inhibit = bool(d["abort_inhibit"])
        if "tries_needed" in d:
            alarm.tries_needed = int(d["tries_needed"])
        sample = DigitalAlarmSample_reply()
        sample.value = alarm
        sample.time = timestamp_ms
        sample.ref_id = ref_id  # type: ignore[attr-defined]
        return sample


@dataclass
class _ReadJob:
    """State for an async one-shot read on the IO thread."""

    drfs: list[str]
    prepared_drfs: list[str]
    drf_to_idx: dict[str, int]
    # Reverse index: prepared_drf → all indices (handles duplicates in O(1))
    drf_to_all_indices: dict[str, list[int]] = field(default_factory=dict)
    readings: dict[int, Reading] = field(default_factory=dict)
    done_event: threading.Event = field(default_factory=threading.Event)
    error: Optional[Exception] = None  # GSS or setup error
    channel: Optional[Channel] = None
    exchange_name: str = ""
    queue_name: str = ""
    consumer_tag: Optional[str] = None


@dataclass
class _WriteSession:
    """Active write session for a device.

    Created on-demand when a write targets a new device.
    Reused for subsequent writes to the same device.
    """

    device: str
    init_drf: str  # exact DRF used in INIT dataRequest -- must match SETTING routing key
    channel: Channel
    exchange_name: str
    queue_name: str
    gss_context: Any  # gssapi.SecurityContext
    last_used: float  # time.monotonic()
    consumer_tag: Optional[str] = None
    heartbeat_handle: Optional[object] = None
    cleanup_handle: Optional[object] = None  # idle TTL timer
    # Pending writes: correlation_id -> (index, drf, results_list, completion_tracker)
    pending: dict[str, tuple[int, str, list, "_WriteCompletionTracker"]] = field(default_factory=dict)
    # Writes queued until server confirms INIT via PENDING response (S.# binding ready)
    init_confirmed: bool = False
    queued_sends: list[tuple[list[tuple[int, str, Value]], list, "_WriteCompletionTracker"]] = field(
        default_factory=list
    )
    init_timer: Optional[object] = None  # safety timer if PENDING never arrives


@dataclass
class _WriteCompletionTracker:
    """Tracks completion of a write_many request across multiple devices."""

    total_devices: int
    completed_devices: int = 0
    done_event: threading.Event = field(default_factory=threading.Event)
    exception: Optional[Exception] = None

    def device_complete(self) -> None:
        """Called when all writes for one device are done."""
        self.completed_devices += 1
        if self.completed_devices >= self.total_devices:
            self.done_event.set()

    def abort(self, exc: Exception) -> None:
        """Abort with exception -- signals done immediately."""
        self.exception = exc
        self.done_event.set()


# Fermilab public AS3152 prefixes -- if the client's IP falls in one of these,
# report it verbatim; otherwise use the proxy address to avoid leaking
# private/home IPs to the server.
_FNAL_PREFIXES = (
    (socket.inet_aton("131.225.0.0"), 16),
    (socket.inet_aton("192.190.216.0"), 22),
    (socket.inet_aton("198.49.208.0"), 24),
)
_FNAL_PROXY = socket.inet_aton("131.225.142.68")


def _in_fnal_range(ip_bytes: bytes) -> bool:
    """Check if IP (4 bytes) falls within any Fermilab prefix."""
    ip_int = int.from_bytes(ip_bytes, "big")
    for net_bytes, prefix_len in _FNAL_PREFIXES:
        net_int = int.from_bytes(net_bytes, "big")
        mask = (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF
        if (ip_int & mask) == (net_int & mask):
            return True
    return False


def _get_host_address() -> bytes:
    """Get host-address bytes for AMQP headers.

    Returns the real local IP if it's on the Fermilab network (AS3152),
    otherwise returns a fixed proxy address to avoid leaking private IPs.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = socket.inet_aton(s.getsockname()[0])
    except OSError as e:
        logger.warning("Cannot determine local IP, falling back to FNAL proxy: %s", e)
        local_ip = _FNAL_PROXY

    return local_ip if _in_fnal_range(local_ip) else _FNAL_PROXY


def _extract_basic_status(reply):
    """Extract BasicStatus value as dict.

    BasicStatus_struct fields are set by the unmarshaler only if present
    in the wire data, so getattr is necessary here.
    """
    if not hasattr(reply, "value"):
        return {}
    v = reply.value
    d = {
        "on": getattr(v, "on", None),
        "ready": getattr(v, "ready", None),
        "remote": getattr(v, "remote", None),
        "positive": getattr(v, "positive", None),
        "ramp": getattr(v, "ramp", None),
    }
    return {k: val for k, val in d.items() if val is not None}


def _extract_analog_alarm(reply):
    """Extract AnalogAlarm value as dict."""
    if not hasattr(reply, "value"):
        return {}
    v = reply.value
    return {
        "minimum": v.minimum,
        "maximum": v.maximum,
        "alarm_enable": v.alarm_enable,
        "alarm_status": v.alarm_status,
        "abort": v.abort,
        "abort_inhibit": v.abort_inhibit,
        "tries_needed": v.tries_needed,
        "tries_now": v.tries_now,
    }


def _extract_digital_alarm(reply):
    """Extract DigitalAlarm value as dict."""
    if not hasattr(reply, "value"):
        return {}
    v = reply.value
    return {
        "nominal": v.nominal,
        "mask": v.mask,
        "alarm_enable": v.alarm_enable,
        "alarm_status": v.alarm_status,
        "abort": v.abort,
        "abort_inhibit": v.abort_inhibit,
        "tries_needed": v.tries_needed,
        "tries_now": v.tries_now,
    }


# Dispatch table: reply_type -> (value_type, value_extractor)
_REPLY_VALUE_MAP: dict[type, tuple[ValueType, Any]] = {
    DoubleSample_reply: (ValueType.SCALAR, lambda r: r.value),
    DoubleArraySample_reply: (ValueType.SCALAR_ARRAY, lambda r: np.array(r.value)),
    StringSample_reply: (ValueType.TEXT, lambda r: r.value),
    StringArraySample_reply: (ValueType.TEXT_ARRAY, lambda r: list(r.value)),
    BinarySample_reply: (ValueType.RAW, lambda r: bytes(r.value)),
    BasicControlSample_reply: (ValueType.SCALAR, lambda r: r.value if hasattr(r, "value") else None),
    BasicStatusSample_reply: (ValueType.BASIC_STATUS, _extract_basic_status),
    AnalogAlarmSample_reply: (ValueType.ANALOG_ALARM, _extract_analog_alarm),
    DigitalAlarmSample_reply: (ValueType.DIGITAL_ALARM, _extract_digital_alarm),
    ShortSample_reply: (ValueType.SCALAR, lambda r: int(r.value)),
    ShortArraySample_reply: (ValueType.SCALAR_ARRAY, lambda r: np.array(r.value, dtype=np.int16)),
    IntegerSample_reply: (ValueType.SCALAR, lambda r: int(r.value)),
    IntegerArraySample_reply: (ValueType.SCALAR_ARRAY, lambda r: np.array(r.value, dtype=np.int32)),
    LongSample_reply: (ValueType.SCALAR, lambda r: int(r.value)),
    LongArraySample_reply: (ValueType.SCALAR_ARRAY, lambda r: np.array(r.value, dtype=np.int64)),
    BooleanSample_reply: (ValueType.SCALAR, lambda r: bool(r.value)),
    BooleanArraySample_reply: (ValueType.SCALAR_ARRAY, lambda r: np.array(r.value, dtype=bool)),
}


def _reply_to_reading(reply, drf: str) -> Reading:
    """Convert a DMQ reply to a Reading object."""
    if isinstance(reply, ErrorSample_reply):
        return Reading(
            drf=drf,
            value_type=ValueType.SCALAR,
            facility_code=reply.facilityCode,
            error_code=reply.errorNumber,
            value=None,
            message=getattr(reply, "message", None),
            timestamp=timestamp_from_millis(reply.time) if reply.time else None,
            cycle=getattr(reply, "cycle_time", 0),
        )

    entry = _REPLY_VALUE_MAP.get(type(reply))
    if entry:
        vtype, extract = entry
        return Reading(
            drf=drf,
            value_type=vtype,
            error_code=ERR_OK,
            value=extract(reply),
            timestamp=timestamp_from_millis(reply.time) if reply.time else None,
            cycle=getattr(reply, "cycle_time", 0),
        )

    logger.warning(f"Unknown reply type: {type(reply).__name__}")
    return Reading(
        drf=drf,
        value_type=ValueType.SCALAR,
        facility_code=FACILITY_ACNET,
        error_code=ERR_RETRY,
        value=None,
        message=f"Unknown reply type: {type(reply).__name__}",
        timestamp=None,
        cycle=0,
    )


def _resolve_reply(
    routing_key: str, body: bytes, drfs: list[str], drf_to_idx: dict[str, int]
) -> tuple[Any, int, int | None] | None:
    """Parse reply and resolve device index from routing key or ref_id.

    Shared by read and subscription message handlers.
    Returns (reply, idx, ref_id) or None if message should be skipped.
    """
    if routing_key == "Q":
        return None
    try:
        reply = unmarshal_reply(iter(body))
    except Exception as e:
        logger.warning(f"Failed to unmarshal reply: {e}")
        # Try to resolve index via routing key so callers get an immediate
        # error instead of waiting for a timeout.
        if routing_key.startswith("R."):
            idx = drf_to_idx.get(routing_key[2:])
            if idx is not None:
                err = ErrorSample_reply()
                err.facilityCode = FACILITY_ACNET
                err.errorNumber = ERR_RETRY
                return err, idx, None
        return None
    if (
        isinstance(reply, ErrorSample_reply)
        and reply.errorNumber == DMQ_PENDING_ERROR
        and reply.facilityCode == FACILITY_DMQ
    ):
        return None
    ref_id = getattr(reply, "ref_id", None)
    idx: int | None = None
    if ref_id is not None and 1 <= ref_id <= len(drfs):
        idx = ref_id - 1
    elif routing_key.startswith("R."):
        idx = drf_to_idx.get(routing_key[2:])
    if idx is None:
        return None
    return reply, idx, ref_id


class _DMQSubscriptionHandle(BufferedSubscriptionHandle):
    """Subscription handle for DMQBackend."""

    def __init__(
        self,
        backend: "DMQBackend",
        sub_id: str,
        drfs: list[str],
        is_callback_mode: bool,
        on_error: Optional[ErrorCallback] = None,
    ):
        super().__init__()
        self._backend = backend
        self._sub_id = sub_id
        self._drfs = drfs
        self._is_callback_mode = is_callback_mode
        self._on_error = on_error
        self._ref_ids = list(range(1, len(drfs) + 1))

    def stop(self) -> None:
        if not self._stopped:
            self._backend.remove(self)
            self._signal_stop()


@dataclass
class _SelectSubscription:
    """State for a single subscription on the shared SelectConnection."""

    sub_id: str
    drfs: list[str]
    drf_to_idx: dict[str, int]  # device_name -> index for O(1) routing key lookup
    drf_to_all_indices: dict[str, list[int]]  # drf -> all indices (handles duplicates)
    handle: "_DMQSubscriptionHandle"
    callback: Optional[ReadingCallback]
    exchange_name: str
    queue_name: str = ""
    channel: Optional[Channel] = None
    consumer_tag: Optional[str] = None
    heartbeat_handle: Optional[object] = None  # ioloop timer handle
    setup_complete: threading.Event = field(default_factory=threading.Event)
    setup_error: Optional[Exception] = None


class DMQBackend(Backend):
    """
    DMQ Backend for ACNET device access via RabbitMQ.

    Uses AMQP to communicate with ACNET via the DMQ server.
    Supports read, write, and streaming operations.

    NOTE: DMQ requires Kerberos authentication for ALL operations,
    including reads. This differs from DPM/HTTP which allows anonymous reads.

    Capabilities:
        - READ: Requires KerberosAuth
        - STREAM: Requires KerberosAuth
        - WRITE: Requires KerberosAuth
        - BATCH: Always enabled (get_many)
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        vhost: str = DEFAULT_VHOST,
        timeout: float = DEFAULT_TIMEOUT,
        auth: Optional[Auth] = None,
        write_session_ttl: float = DEFAULT_WRITE_SESSION_TTL,
        dispatch_mode: DispatchMode = DispatchMode.WORKER,
    ):
        """
        Initialize DMQ backend.

        Args:
            host: RabbitMQ broker hostname (default: appsrv2.fnal.gov)
            port: RabbitMQ broker port (default: 5672)
            vhost: RabbitMQ virtual host (default: /)
            timeout: Default operation timeout in seconds (default: 10.0)
            auth: KerberosAuth required for all DMQ operations
            write_session_ttl: Idle timeout for write sessions in seconds (default: 600)

        Raises:
            AuthenticationError: If auth is not provided or not KerberosAuth
        """
        if not host:
            raise ValueError("host cannot be empty")
        if port <= 0 or port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {port}")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if auth is None:
            raise AuthenticationError("DMQ requires KerberosAuth for all operations (run kinit first)")
        if not isinstance(auth, KerberosAuth):
            raise AuthenticationError(f"DMQ requires KerberosAuth, got {type(auth).__name__}")

        self._host = host
        self._port = port
        self._vhost = vhost
        self._timeout = timeout
        self._auth: Optional[KerberosAuth] = auth
        self._write_session_ttl = write_session_ttl
        self._closed = False

        # Callback dispatcher
        self._dispatch_mode = dispatch_mode
        self._dispatcher = CallbackDispatcher(dispatch_mode)

        # Streaming state (SelectConnection model)
        self._stream_lock = threading.Lock()
        self._subscriptions: dict[str, _SelectSubscription] = {}
        self._select_connection: Optional[SelectConnection] = None
        self._io_thread: Optional[threading.Thread] = None
        self._connection_ready = threading.Event()
        self._connection_error: Optional[Exception] = None

        # Write state (unified with SelectConnection)
        self._write_sessions: dict[str, _WriteSession] = {}  # init_drf -> active session
        # Writes queued while channel setup is in progress (prevents duplicate sessions)
        self._pending_session_setups: dict[
            str, list[tuple[list[tuple[int, str, Value]], list[WriteResult | None], _WriteCompletionTracker]]
        ] = {}

        # Cache local IP (used in INIT, SETTING messages)
        self._local_ip = _get_host_address()

        # Validate auth eagerly — DMQ requires auth for all operations
        if self._auth is not None:
            _ = self._auth.principal  # This validates credentials

        logger.debug(
            f"DMQBackend initialized: host={host}, port={port}, vhost={vhost}, "
            f"timeout={timeout}, auth={type(auth).__name__ if auth else None}"
        )

    @property
    def capabilities(self) -> BackendCapability:
        caps = BackendCapability.READ | BackendCapability.STREAM | BackendCapability.BATCH

        if isinstance(self._auth, KerberosAuth):
            caps |= BackendCapability.AUTH_KERBEROS | BackendCapability.WRITE

        return caps

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def authenticated(self) -> bool:
        return self._auth is not None

    @property
    def principal(self) -> Optional[str]:
        return self._auth.principal if self._auth is not None else None

    def _check_not_io_thread(self) -> None:
        """Raise if called from the IO thread to prevent deadlock."""
        if self._io_thread is not None and threading.current_thread() is self._io_thread:
            raise RuntimeError(
                "Cannot call blocking backend methods from the IO thread "
                "(e.g. from a DIRECT mode streaming callback). "
                "Use DispatchMode.WORKER or offload to another thread."
            )

    def _sign_message(
        self,
        ctx,
        binary: bytes,
        message_id: str,
        correlation_id: Optional[str] = None,
        reply_to: str = "",
        app_id: str = "",
    ) -> bytes:
        """Compute MIC signature over message body and AMQP properties.

        Builds the signing body matching Java's GSSUtil.createBody format:
        binary + '\\0' + messageId + '\\0' + correlationId + '\\0' +
        replyTo + '\\0' + appId + '\\0' + hostAddress + '\\0'

        Then calls gssapi get_signature (getMIC) on the result.
        """
        parts = [
            binary,
            b"\x00",
            message_id.encode("utf-8") if message_id else b"",
            b"\x00",
            correlation_id.encode("utf-8") if correlation_id else b"",
            b"\x00",
            reply_to.encode("utf-8") if reply_to else b"",
            b"\x00",
            app_id.encode("utf-8") if app_id else b"",
            b"\x00",
            self._local_ip,
            b"\x00",
        ]
        return bytes(ctx.get_signature(b"".join(parts)))

    def _send_init(
        self,
        channel: Channel,
        exchange_name: str,
        drfs: list[str],
        ctx: Any,
        token: bytes,
    ) -> str:
        """Send INIT request to start a job (IO thread).

        ctx and token must be pre-computed before calling this method.

        Returns:
            message_id for correlation
        """
        # Create ReadingRequest
        req = ReadingRequest_request()
        req.dataRequest = list(drfs)
        request_bytes = bytes(req.marshal())

        message_id = str(uuid.uuid4())
        app_id = "pacsys"

        mic = self._sign_message(ctx, request_bytes, message_id, reply_to=exchange_name, app_id=app_id)

        # Publish INIT to amq.topic with GSS token and signature
        channel.basic_publish(
            exchange=INIT_EXCHANGE,
            routing_key="I",
            body=request_bytes,
            properties=pika.BasicProperties(
                message_id=message_id,
                reply_to=exchange_name,
                app_id=app_id,
                headers={
                    "host-address": self._local_ip,
                    "gss-token": bytes(token) if token else b"",
                    "signature": mic,
                },
            ),
        )

        logger.debug(f"Sent INIT for {len(drfs)} devices, message_id={message_id}")
        return message_id

    def _send_drop(
        self,
        channel: Channel,
        exchange_name: str,
    ) -> None:
        """Send DROP message to cleanup job."""
        try:
            channel.basic_publish(
                exchange=exchange_name,
                routing_key="D",
                body=b"",
            )
            logger.debug(f"Sent DROP to exchange={exchange_name}")
        except Exception as e:
            logger.debug(f"Error sending DROP: {e}")

    def _do_oneshot_read(self, drfs: list[str], timeout: float) -> list[Reading]:
        """Perform one-shot read via the shared SelectConnection."""
        if not drfs:
            return []

        prepared_drfs = [ensure_immediate_event(drf) for drf in drfs]
        # Build reverse index: prepared_drf → all indices (for dedup-aware O(1) lookup)
        drf_to_all: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(prepared_drfs):
            drf_to_all[d].append(i)
        job = _ReadJob(
            drfs=drfs,
            prepared_drfs=prepared_drfs,
            drf_to_idx={drf: i for i, drf in enumerate(prepared_drfs)},
            drf_to_all_indices=dict(drf_to_all),
        )

        self._ensure_io_thread()
        conn = self._select_connection
        if conn is None:
            cause = ConnectionError(f"No connection to RabbitMQ at {self._host}:{self._port}")
            readings = [
                Reading(
                    drf=drf,
                    value_type=ValueType.SCALAR,
                    facility_code=FACILITY_ACNET,
                    error_code=ERR_RETRY,
                    value=None,
                    message=str(cause),
                    timestamp=None,
                    cycle=0,
                )
                for drf in drfs
            ]
            raise ReadError(readings, str(cause)) from cause

        conn.ioloop.add_callback_threadsafe(lambda: self._start_read_async(job))

        timed_out = not job.done_event.wait(timeout)
        if timed_out:
            # Timeout -- schedule cleanup on IO thread
            if conn.is_open:
                conn.ioloop.add_callback_threadsafe(lambda: self._complete_read(job))
                job.done_event.wait(timeout=5.0)

        # Build result list
        if job.error is not None:
            backfill_code = ERR_RETRY
            backfill_msg = str(job.error)
        else:
            backfill_code = ERR_TIMEOUT
            backfill_msg = "Request timeout"

        result: list[Reading] = []
        has_backfill = False
        for i, drf in enumerate(drfs):
            if i in job.readings:
                result.append(job.readings[i])
            else:
                has_backfill = True
                result.append(
                    Reading(
                        drf=drf,
                        value_type=ValueType.SCALAR,
                        facility_code=FACILITY_ACNET,
                        error_code=backfill_code,
                        value=None,
                        message=backfill_msg,
                        timestamp=None,
                        cycle=0,
                    )
                )

        # Connection-level errors or total timeout → raise
        if job.error is not None:
            raise ReadError(result, backfill_msg) from job.error
        if has_backfill and not job.readings:
            raise ReadError(result, backfill_msg)

        return result

    def _start_read_async(self, job: _ReadJob) -> None:
        """Start async read on IO thread."""
        if self._select_connection is None or not self._select_connection.is_open:
            job.error = ConnectionError(f"No connection to RabbitMQ at {self._host}:{self._port}")
            job.done_event.set()
            return

        def on_ready(channel, exchange_name, queue_name):
            if job.done_event.is_set():
                # Timeout already fired -- clean up the channel we just opened
                try:
                    if channel.is_open:
                        channel.close()
                except Exception:
                    pass
                return
            job.channel = channel
            job.exchange_name = exchange_name
            job.queue_name = queue_name

            try:
                ctx = self._create_gss_context()
                token = ctx.step()
            except Exception as exc:
                logger.error(f"GSS context creation failed for read: {exc}")
                job.error = exc
                self._complete_read(job)
                return

            if job.done_event.is_set():
                # Timeout fired while GSS was executing - don't start a ghost job
                try:
                    if channel.is_open:
                        channel.close()
                except Exception:
                    pass
                return

            try:
                self._send_init(channel, exchange_name, job.prepared_drfs, ctx, token)
                job.consumer_tag = channel.basic_consume(
                    queue=queue_name,
                    on_message_callback=lambda ch, method, props, body: self._on_read_message(job, ch, method, body),
                    auto_ack=False,
                )
            except Exception as exc:
                logger.error(f"Read setup failed after INIT: {exc}")
                job.error = exc
                self._complete_read(job)
                try:
                    if channel.is_open:
                        channel.close()
                except Exception:
                    pass

        self._setup_channel_async(on_ready=on_ready)

    def _on_read_message(self, job: _ReadJob, channel: Channel, method, body: bytes) -> None:
        """Handle a message for an async read (IO thread)."""
        result = _resolve_reply(method.routing_key, body, job.prepared_drfs, job.drf_to_idx)
        channel.basic_ack(method.delivery_tag)
        if result is None:
            return
        reply, idx, ref_id = result
        # Fill this index and any unfilled duplicates with the same prepared DRF.
        # The server deduplicates requests, so only one reply arrives per unique device.
        # Uses pre-built reverse index for O(K) lookup (K = duplicates) instead of O(N).
        drf = job.prepared_drfs[idx]
        filled = False
        for i in job.drf_to_all_indices.get(drf, ()):
            if i not in job.readings:
                job.readings[i] = _reply_to_reading(reply, job.drfs[i])
                filled = True
        if filled and len(job.readings) >= len(job.drfs):
            self._complete_read(job)

    def _complete_read(self, job: _ReadJob) -> None:
        """Finish async read: send DROP, cancel consumer, close channel (IO thread)."""
        if job.done_event.is_set():
            return
        try:
            if job.channel and job.exchange_name:
                self._send_drop(job.channel, job.exchange_name)
            if job.channel and job.consumer_tag:
                job.channel.basic_cancel(job.consumer_tag)
            if job.channel and job.channel.is_open:
                job.channel.close()
        except Exception as e:
            logger.debug(f"Error cleaning up read channel: {e}")
        finally:
            job.done_event.set()

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

        assert reading.value is not None  # ok implies value is set
        return reading.value

    def get(self, drf: str, timeout: Optional[float] = None) -> Reading:
        """Read a single device with full metadata."""
        readings = self.get_many([drf], timeout=timeout)
        return readings[0]

    def get_many(self, drfs: list[str], timeout: Optional[float] = None) -> list[Reading]:
        """Read multiple devices in a single batch."""
        self._check_not_io_thread()
        if self._closed:
            raise RuntimeError("Backend is closed")

        if not drfs:
            return []

        effective_timeout = timeout if timeout is not None else self._timeout
        return self._do_oneshot_read(drfs, effective_timeout)

    def _create_gss_context(self):
        """Create a GSS-API security context for DMQ authentication.

        The DMQ service principal is "daeset/bd/dmq.fnal.gov@FNAL.GOV" in
        standard Kerberos format: service/instance@REALM

        Flags requested match Java client:
        - integrity: for MIC signatures
        - replay_detection: prevent replay attacks
        - out_of_sequence_detection: detect out-of-order messages

        Returns:
            gssapi.SecurityContext initialized for the DMQ service

        Raises:
            ImportError: If gssapi library is not installed
            AuthenticationError: If context creation fails
        """
        import gssapi

        try:
            name = gssapi.Name(DMQ_SERVICE_PRINCIPAL, gssapi.NameType.kerberos_principal)
            assert self._auth is not None
            creds = self._auth._get_credentials()
            ctx = gssapi.SecurityContext(
                name=name,
                usage="initiate",
                creds=creds,
                flags=(
                    gssapi.RequirementFlag.integrity
                    | gssapi.RequirementFlag.replay_detection
                    | gssapi.RequirementFlag.out_of_sequence_detection
                ),
            )
            return ctx
        except Exception as e:
            raise AuthenticationError(f"Failed to create GSS context: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Shared AMQP Channel Setup (runs on IO thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_channel_async(self, on_ready, exchange_name=None, on_channel_open=None) -> None:
        """Open channel with queue, exchange, and bindings (IO thread).

        Executes: open_channel → queue_declare → exchange_declare → bind(R.#) → bind(Q) → on_ready.
        Shared by read, write, and subscription setup.

        Args:
            on_ready: Callback(channel, exchange_name, queue_name) when setup completes.
            exchange_name: Exchange name to use (default: generate UUID).
            on_channel_open: Optional callback(channel) invoked right after channel opens.
        """
        ex = exchange_name or str(uuid.uuid4())

        def _on_open(channel: Channel):
            if on_channel_open:
                on_channel_open(channel)
            channel.queue_declare(
                queue="",
                exclusive=True,
                auto_delete=True,
                callback=lambda frame: _on_queue(channel, frame.method.queue),
            )

        def _on_queue(ch, q):
            ch.exchange_declare(
                exchange=ex,
                exchange_type="topic",
                auto_delete=True,
                callback=lambda _: _on_exchange(ch, q),
            )

        def _on_exchange(ch, q):
            ch.queue_bind(
                queue=q,
                exchange=ex,
                routing_key="R.#",
                callback=lambda _: _on_r_bind(ch, q),
            )

        def _on_r_bind(ch, q):
            ch.queue_bind(
                queue=q,
                exchange=ex,
                routing_key="Q",
                callback=lambda _: on_ready(ch, ex, q),
            )

        assert self._select_connection is not None, "No connection"
        self._select_connection.channel(on_open_callback=_on_open)

    # ─────────────────────────────────────────────────────────────────────────
    # Write Session Heartbeats (runs on IO thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_write_session_heartbeat(self, session: _WriteSession) -> None:
        """Schedule heartbeat for write session (IO thread)."""
        if self._select_connection is None or not self._select_connection.is_open:
            return
        if session.init_drf not in self._write_sessions:
            return

        def send_heartbeat():
            if session.init_drf not in self._write_sessions:
                return
            if session.channel is None or not session.channel.is_open:
                return
            try:
                session.channel.basic_publish(exchange=session.exchange_name, routing_key="H", body=b"")
                logger.debug(f"Sent write session heartbeat for {session.device}")
            except Exception as e:
                logger.warning(f"Write session heartbeat failed for {session.device}: {e}")
            self._schedule_write_session_heartbeat(session)

        session.heartbeat_handle = self._select_connection.ioloop.call_later(5.0, send_heartbeat)

    # ─────────────────────────────────────────────────────────────────────────
    # Write Session Lifecycle (runs on IO thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_write_session_cleanup(self, session: _WriteSession) -> None:
        """Schedule idle cleanup for write session (IO thread)."""
        if self._select_connection is None or not self._select_connection.is_open:
            return

        # Cancel existing cleanup timer
        if session.cleanup_handle is not None:
            try:
                self._select_connection.ioloop.remove_timeout(session.cleanup_handle)
            except Exception:
                pass
            session.cleanup_handle = None

        def do_cleanup():
            if session.init_drf not in self._write_sessions:
                return
            # Don't cleanup if there are pending writes
            if session.pending:
                self._schedule_write_session_cleanup(session)
                return
            self._close_write_session(session.init_drf, reason=f"idle for {self._write_session_ttl}s")

        session.cleanup_handle = self._select_connection.ioloop.call_later(self._write_session_ttl, do_cleanup)

    def _close_write_session(self, init_drf: str, reason: str = "") -> None:
        """Close a write session and release resources (IO thread)."""
        session = self._write_sessions.pop(init_drf, None)
        if session is None:
            return

        logger.debug(f"Closing write session for {session.device} ({init_drf}): {reason}")

        # Cancel timers
        conn = self._select_connection
        for handle in (session.heartbeat_handle, session.cleanup_handle, session.init_timer):
            if handle is not None and conn is not None:
                try:
                    conn.ioloop.remove_timeout(handle)
                except Exception:
                    pass

        # Fail any queued and pending writes - signal each tracker exactly once
        trackers: dict[int, _WriteCompletionTracker] = {}
        for q_settings, q_results, q_tracker in session.queued_sends:
            for i, drf, _ in q_settings:
                if q_results[i] is None:
                    q_results[i] = WriteResult(
                        drf=drf,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_RETRY,
                        message=f"Session closed: {reason}",
                    )
            trackers.setdefault(id(q_tracker), q_tracker)
        session.queued_sends = []
        for corr_id, (i, drf, results_list, pending_tracker) in list(session.pending.items()):
            if results_list[i] is None:
                results_list[i] = WriteResult(
                    drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=f"Session closed: {reason}"
                )
            trackers.setdefault(id(pending_tracker), pending_tracker)
        session.pending.clear()
        for tracker in trackers.values():
            tracker.device_complete()

        # Send DROP and close channel
        if session.channel is not None and session.channel.is_open:
            try:
                session.channel.basic_publish(exchange=session.exchange_name, routing_key="D", body=b"")
            except Exception:
                pass
            if session.consumer_tag:
                try:
                    session.channel.basic_cancel(session.consumer_tag)
                except Exception:
                    pass
            try:
                session.channel.close()
            except Exception:
                pass

    def _evict_lru_write_session(self) -> bool:
        """Evict the least-recently-used idle write session (IO thread).

        Returns True if a session was evicted. Returns False if all sessions
        have pending writes (refuses to destroy in-flight work).
        """
        if not self._write_sessions:
            return False
        lru_key = None
        lru_time = float("inf")
        for key, session in self._write_sessions.items():
            if not session.pending and not session.queued_sends and session.last_used < lru_time:
                lru_time = session.last_used
                lru_key = key
        if lru_key is None:
            return False
        self._close_write_session(lru_key, reason="evicted (session limit)")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Async Write Execution (runs on IO thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_write_many_async(
        self,
        settings: list[tuple[str, Value]],
        results: list[WriteResult | None],
        tracker: _WriteCompletionTracker,
    ) -> None:
        """Execute write_many on IO thread."""
        try:
            # Group by init_drf (property-aware, not just device name)
            by_init_drf: dict[str, list[tuple[int, str, Value]]] = defaultdict(list)
            for i, (drf, value) in enumerate(settings):
                init_drf = prepare_for_write(drf)
                by_init_drf[init_drf].append((i, drf, value))

            tracker.total_devices = len(by_init_drf)

            if tracker.total_devices == 0:
                tracker.done_event.set()
                return

            # Process each init_drf (parallel on IO thread - all non-blocking)
            for init_drf, drf_settings in by_init_drf.items():
                self._write_to_device_async(init_drf, drf_settings, results, tracker)
        except Exception as e:
            logger.error(f"Failed to prepare write batch: {e}")
            tracker.abort(e)

    def _write_to_device_async(
        self,
        init_drf: str,
        drf_settings: list[tuple[int, str, Value]],
        results: list[WriteResult | None],
        tracker: _WriteCompletionTracker,
    ) -> None:
        """Write to single device/property (IO thread). Keyed by init_drf."""
        # Check for existing session
        if init_drf in self._write_sessions:
            session = self._write_sessions[init_drf]
            if session.channel is not None and session.channel.is_open:
                # Reset idle TTL since session is being reused
                self._schedule_write_session_cleanup(session)
                if session.init_confirmed:
                    self._send_settings_async(session, drf_settings, results, tracker)
                else:
                    # S.# not yet bound - queue until PENDING arrives
                    session.queued_sends.append((drf_settings, results, tracker))
                return
            else:
                # Session dead, clean up properly
                self._close_write_session(init_drf, reason="channel dead")

        # Queue if channel setup already in progress for this device
        if init_drf in self._pending_session_setups:
            self._pending_session_setups[init_drf].append((drf_settings, results, tracker))
            return

        # Enforce session limit before creating new one
        if len(self._write_sessions) >= MAX_WRITE_SESSIONS:
            if not self._evict_lru_write_session():
                for i, drf, _ in drf_settings:
                    results[i] = WriteResult(
                        drf=drf,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_RETRY,
                        message="Too many active write sessions",
                    )
                tracker.device_complete()
                return

        # Create channel on-demand
        if self._select_connection is None or not self._select_connection.is_open:
            for i, drf, _ in drf_settings:
                results[i] = WriteResult(
                    drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message="Connection not open"
                )
            tracker.device_complete()
            return

        # Register pending setup before async call to prevent duplicate sessions
        self._pending_session_setups[init_drf] = [(drf_settings, results, tracker)]

        def on_ready(channel, exchange_name, queue_name):
            self._create_write_session(channel, exchange_name, queue_name, init_drf)

        self._setup_channel_async(on_ready=on_ready)

    def _create_write_session(
        self,
        channel: Channel,
        exchange_name: str,
        queue_name: str,
        init_drf: str,
    ) -> None:
        """Create write session: GSS auth + INIT + session setup (IO thread).

        Drains all writes queued in ``_pending_session_setups[init_drf]``.
        GSS context created inline - accepts first-call KDC latency (~50-500ms);
        cached Kerberos ticket thereafter (<1ms).
        """
        queued_writes = self._pending_session_setups.pop(init_drf, [])

        # Aborted before channel was ready (timeout drained the queue)
        if not queued_writes:
            try:
                if channel.is_open:
                    channel.close()
            except Exception:
                pass
            return

        device = get_device_name(init_drf)
        try:
            ctx = self._create_gss_context()
            token = ctx.step()
        except (AuthenticationError, ImportError) as exc:
            # Auth/import errors propagate to caller via tracker.abort()
            logger.error(f"GSS auth failed for {device}: {exc}")
            try:
                if channel.is_open:
                    channel.close()
            except Exception:
                pass
            for _, _, q_tracker in queued_writes:
                q_tracker.abort(exc)
            return
        except Exception as exc:
            logger.error(f"GSS context creation failed for {device}: {exc}")
            try:
                if channel.is_open:
                    channel.close()
            except Exception:
                pass
            msg = f"GSS context creation failed: {exc}"
            for q_settings, q_results, q_tracker in queued_writes:
                for i, drf, _ in q_settings:
                    q_results[i] = WriteResult(
                        drf=drf,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_RETRY,
                        message=msg,
                    )
                q_tracker.device_complete()
            return

        # Build and send INIT -- init_drf is the exact string registered
        # on the server as the setter key; SETTING routing key must match it
        req = SettingRequest_request()
        req.dataRequest = [init_drf]
        body = bytes(req.marshal())

        message_id = str(uuid.uuid4())

        mic = self._sign_message(ctx, body, message_id, reply_to=exchange_name, app_id="pacsys")

        channel.basic_publish(
            exchange=INIT_EXCHANGE,
            routing_key="I",
            body=body,
            properties=pika.BasicProperties(
                message_id=message_id,
                reply_to=exchange_name,
                app_id="pacsys",
                headers={
                    "gss-token": bytes(token) if token else b"",
                    "signature": mic,
                    "host-address": self._local_ip,
                },
            ),
        )

        # Create session and populate queued writes BEFORE starting consumer.
        # PENDING may arrive immediately after basic_consume (triggering
        # _flush_queued_writes), so queued_sends must be populated first.
        session = _WriteSession(
            device=device,
            init_drf=init_drf,
            channel=channel,
            exchange_name=exchange_name,
            queue_name=queue_name,
            gss_context=ctx,
            last_used=time.monotonic(),
        )

        # Queue writes until server confirms INIT via PENDING response.
        # The server binds S.# on our exchange during INIT processing;
        # PENDING arrives after the binding is in place (see ServerJobProxy.init()).
        session.queued_sends = [(s, r, t) for s, r, t in queued_writes]

        # Register session before consumer starts (channel close callback needs it)
        channel.add_on_close_callback(lambda ch, reason: self._on_write_session_channel_closed(init_drf, reason))
        self._write_sessions[init_drf] = session

        # Start consuming - PENDING may arrive immediately after this
        session.consumer_tag = channel.basic_consume(
            queue=queue_name,
            on_message_callback=lambda ch, m, p, b: self._on_write_message(session, ch, m, p, b),
            auto_ack=False,
        )

        # Fail fast: if PENDING doesn't arrive within 5s, the server failed to
        # process INIT.  impl1 sends PENDING for settings only after full job
        # creation (InitTask.run), so allow up to CLIENT_INIT_RATE (5000ms).
        if self._select_connection is not None:
            session.init_timer = self._select_connection.ioloop.call_later(
                5.0, lambda: self._fail_unconfirmed_session(session)
            )

        # Schedule heartbeat and idle cleanup
        self._schedule_write_session_heartbeat(session)
        self._schedule_write_session_cleanup(session)

        logger.debug(f"Write session created for {device} ({init_drf}), exchange={exchange_name[:8]}")

    def _flush_queued_writes(self, session: _WriteSession) -> None:
        """Send queued writes after server INIT confirmation (IO thread).

        Called when PENDING response arrives, proving S.# is bound.
        """
        if session.init_timer is not None and self._select_connection is not None:
            try:
                self._select_connection.ioloop.remove_timeout(session.init_timer)
            except Exception:
                pass
            session.init_timer = None
        queued = session.queued_sends
        session.queued_sends = []
        session.init_confirmed = True
        for q_settings, q_results, q_tracker in queued:
            self._send_settings_async(session, q_settings, q_results, q_tracker)

    def _fail_unconfirmed_session(self, session: _WriteSession) -> None:
        """Fail-fast: server didn't confirm INIT within deadline (IO thread)."""
        if session.init_confirmed:
            return
        session.init_timer = None
        logger.error(
            f"Write session for {session.device} ({session.init_drf}): "
            "server did not confirm INIT (no PENDING received)"
        )
        self._close_write_session(session.init_drf, reason="no INIT confirmation from server")

    def _on_write_session_channel_closed(self, init_drf: str, reason: Exception) -> None:
        """Write session channel closed (IO thread)."""
        session = self._write_sessions.pop(init_drf, None)
        if session is not None:
            # Cancel timers
            conn = self._select_connection
            for handle in (session.heartbeat_handle, session.cleanup_handle, session.init_timer):
                if handle is not None and conn is not None:
                    try:
                        conn.ioloop.remove_timeout(handle)
                    except Exception:
                        pass
            # Fail any pending and queued writes - signal each tracker exactly once
            trackers: dict[int, _WriteCompletionTracker] = {}
            for q_settings, q_results, q_tracker in session.queued_sends:
                for i, drf, _ in q_settings:
                    if q_results[i] is None:
                        q_results[i] = WriteResult(
                            drf=drf,
                            facility_code=FACILITY_ACNET,
                            error_code=ERR_RETRY,
                            message=f"Channel closed: {reason}",
                        )
                trackers.setdefault(id(q_tracker), q_tracker)
            session.queued_sends = []
            for corr_id, (i, drf, results_list, pending_tracker) in list(session.pending.items()):
                if results_list[i] is None:
                    results_list[i] = WriteResult(
                        drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=f"Channel closed: {reason}"
                    )
                trackers.setdefault(id(pending_tracker), pending_tracker)
            session.pending.clear()
            for tracker in trackers.values():
                tracker.device_complete()
        logger.debug(f"Write session closed for {init_drf}: {reason}")

    def _send_settings_async(
        self,
        session: _WriteSession,
        device_settings: list[tuple[int, str, Value]],
        results: list[WriteResult | None],
        tracker: _WriteCompletionTracker,
    ) -> None:
        """Send SETTING messages for device (IO thread)."""
        if session.channel is None or not session.channel.is_open:
            for i, drf, _ in device_settings:
                results[i] = WriteResult(
                    drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message="Channel not open"
                )
            tracker.device_complete()
            return

        pending_for_device = 0

        for idx, (i, drf, value) in enumerate(device_settings):
            try:
                sample = self._value_to_sample(value, ref_id=1)
                body = bytes(sample.marshal())

                # Use message_id as the pending key. impl1 sends empty
                # correlationId on write responses; FIFO fallback handles it.
                message_id = str(uuid.uuid4())

                mic = self._sign_message(session.gss_context, body, message_id)

                session.channel.basic_publish(
                    exchange=session.exchange_name,
                    routing_key=f"S.{session.init_drf}",
                    body=body,
                    properties=pika.BasicProperties(
                        message_id=message_id,
                        headers={"signature": mic, "host-address": self._local_ip},
                    ),
                )
                # Register pending AFTER successful publish to avoid orphaned entries
                session.pending[message_id] = (i, drf, results, tracker)
                pending_for_device += 1
                logger.debug(f"Sent SETTING for {session.device} (rk=S.{session.init_drf}), msg_id={message_id[:8]}")

            except Exception as e:
                logger.error(f"Failed to send setting for {drf}: {e}")
                results[i] = WriteResult(drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=str(e))
                # Fail remaining unsent settings and close poisoned session
                for i2, drf2, _ in device_settings[idx + 1 :]:
                    results[i2] = WriteResult(
                        drf=drf2,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_RETRY,
                        message=f"Session closed: {e}",
                    )
                # _close_write_session handles already-published pending writes
                self._close_write_session(session.init_drf, reason=f"sign/publish error: {e}")
                if pending_for_device == 0:
                    tracker.device_complete()
                return

        session.last_used = time.monotonic()

        # If no settings were sent, complete immediately
        if pending_for_device == 0:
            tracker.device_complete()

    def _on_write_message(
        self,
        session: _WriteSession,
        channel: Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.BasicProperties,
        body: bytes,
    ) -> None:
        """Handle write response (IO thread)."""
        channel.basic_ack(method.delivery_tag)

        # Skip heartbeats (server sends Q routing key periodically)
        rk: str = method.routing_key
        if rk == "Q":
            return

        # Extract correlation_id early (before unmarshal) so we can fail
        # the specific request if unmarshal throws
        corr_id = properties.correlation_id

        # Parse reply
        try:
            reply = unmarshal_reply(iter(body))
        except Exception as e:
            logger.warning(f"Failed to unmarshal write response: {e}")
            # Fail the specific pending request instead of silently dropping
            if corr_id and corr_id in session.pending:
                i, drf, results, tracker = session.pending.pop(corr_id)
                results[i] = WriteResult(
                    drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=f"Unmarshal error: {e}"
                )
                logger.warning(f"Write result[{i}] set to error=-1 for {drf}")
                device_still_has_pending = any(t is tracker for _, _, _, t in session.pending.values())
                if not device_still_has_pending:
                    tracker.device_complete()
            return

        if TRACE:
            rk = method.routing_key
            rtype = type(reply).__name__
            in_pending = corr_id in session.pending if corr_id else False
            if isinstance(reply, ErrorSample_reply):
                logger.debug(
                    f"Write msg: rk={rk} type={rtype} facility={reply.facilityCode} "
                    f"error={reply.errorNumber} corr={corr_id and corr_id[:8]} pending={in_pending}"
                )
            else:
                logger.debug(
                    f"Write msg: rk={rk} type={rtype} value={getattr(reply, 'value', '?')} "
                    f"corr={corr_id and corr_id[:8]} pending={in_pending}"
                )

        # PENDING confirms S.# binding is in place - flush queued writes
        if (
            isinstance(reply, ErrorSample_reply)
            and reply.errorNumber == DMQ_PENDING_ERROR
            and reply.facilityCode == FACILITY_DMQ
        ):
            if not session.init_confirmed:
                self._flush_queued_writes(session)
            return

        # Match by correlation_id. impl1 sends correlationId="" on write
        # responses (ServerSettingJob.dataChanged). Fall back to FIFO (oldest
        # pending) when correlationId is missing or unknown.
        # FIFO is safe: single channel + single device = ordered responses.
        if corr_id and corr_id in session.pending:
            i, drf, results, tracker = session.pending.pop(corr_id)
        elif session.pending:
            oldest_key = next(iter(session.pending))
            i, drf, results, tracker = session.pending.pop(oldest_key)
            if TRACE:
                logger.debug(f"Write response matched via FIFO fallback (corr_id={corr_id!r})")
        else:
            return

        # Build WriteResult
        if isinstance(reply, ErrorSample_reply):
            results[i] = WriteResult(
                drf=drf,
                facility_code=reply.facilityCode,
                error_code=reply.errorNumber,
                message=getattr(reply, "message", None),
            )
        else:
            # Non-error response = success
            results[i] = WriteResult(drf=drf, error_code=ERR_OK)

        # Check if all settings for this device are done
        device_still_has_pending = any(t is tracker for _, _, _, t in session.pending.values())
        if not device_still_has_pending:
            tracker.device_complete()

    def _abort_pending_writes(self, init_drfs: set[str]) -> None:
        """Abort pending writes for given init_drfs (IO thread).

        Also drains writes queued in ``_pending_session_setups`` so that
        in-flight channel setups don't produce ghost writes after timeout.
        Sessions with timed-out writes are closed since they're in a bad state.
        """
        for init_drf in init_drfs:
            # Drain writes queued during channel setup (prevents ghost writes)
            queued = self._pending_session_setups.pop(init_drf, [])
            for q_settings, q_results, q_tracker in queued:
                for i, drf, _ in q_settings:
                    if q_results[i] is None:
                        q_results[i] = WriteResult(
                            drf=drf,
                            facility_code=FACILITY_ACNET,
                            error_code=ERR_TIMEOUT,
                            message="Request timeout",
                        )
                q_tracker.device_complete()

            session = self._write_sessions.get(init_drf)
            if session is None:
                continue
            if not session.pending and not session.queued_sends:
                continue
            # Mark all queued and pending writes as timed out - signal each tracker once
            trackers: dict[int, _WriteCompletionTracker] = {}
            for q_settings, q_results, q_tracker in session.queued_sends:
                for i, drf, _ in q_settings:
                    if q_results[i] is None:
                        q_results[i] = WriteResult(
                            drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_TIMEOUT, message="Request timeout"
                        )
                trackers.setdefault(id(q_tracker), q_tracker)
            session.queued_sends = []
            for corr_id, (i, drf, results, tracker) in list(session.pending.items()):
                if results[i] is None:
                    results[i] = WriteResult(
                        drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_TIMEOUT, message="Request timeout"
                    )
                trackers.setdefault(id(tracker), tracker)
            session.pending.clear()
            for tracker in trackers.values():
                tracker.device_complete()
            # Close the session - it's in a bad state (e.g. stuck after PENDING)
            self._close_write_session(init_drf, reason="timed out")

    def _value_to_sample(self, value: Value, ref_id: int = 1):
        """Convert a Python value to an appropriate DMQ Sample reply object.

        Args:
            value: The value to convert (use BasicControl_* constants for control writes)
            ref_id: Reference ID for the sample

        Returns:
            Sample reply object suitable for sending as a SETTING/CONTROL message
        """
        timestamp_ms = int(time.time() * 1000)

        # BasicControl enum → BasicControlSample for commands 0-6,
        # DoubleSample for LOCAL/REMOTE/TRIP (7-9) since the DMQ proto
        # enum only defines 7 values. The server accepts the ordinal as
        # a double and looks it up in the per-device control attribute table.
        if isinstance(value, BasicControl):
            sdd_const = _BASIC_CONTROL_TO_SDD.get(value)
            if sdd_const is not None:
                sample = BasicControlSample_reply()
                sample.value = sdd_const
                sample.time = timestamp_ms
                sample.ref_id = ref_id  # type: ignore[attr-defined]
                return sample
            # LOCAL, REMOTE, TRIP -- send as double ordinal
            sample = DoubleSample_reply()
            sample.value = float(value)
            sample.time = timestamp_ms
            sample.ref_id = ref_id  # type: ignore[attr-defined]
            return sample

        if isinstance(value, dict):
            return _dict_to_alarm_sample(value, ref_id, timestamp_ms)

        if isinstance(value, int) and not isinstance(value, bool):
            sample = IntegerSample_reply()
            sample.value = value
            sample.time = timestamp_ms
            sample.ref_id = ref_id  # type: ignore[attr-defined]
            return sample
        elif isinstance(value, float):
            sample = DoubleSample_reply()
            sample.value = value
            sample.time = timestamp_ms
            sample.ref_id = ref_id  # type: ignore[attr-defined]
            return sample
        elif isinstance(value, (list, np.ndarray)):
            sample = DoubleArraySample_reply()
            sample.value = [float(v) for v in value]
            sample.time = timestamp_ms
            sample.ref_id = ref_id  # type: ignore[attr-defined]
            return sample
        elif isinstance(value, str):
            sample = StringSample_reply()
            sample.value = value
            sample.time = timestamp_ms
            sample.ref_id = ref_id  # type: ignore[attr-defined]
            return sample
        elif isinstance(value, bytes):
            raise ValueError(
                "DMQ backend does not support writing bytes values. "
                "The DMQ server rejects BinarySample messages. "
                "For .RAW writes, pass an integer that the server will "
                "inverse-transform into the desired raw value (the server "
                "applies primary and common inverse transforms to the integer)."
            )
        else:
            # Unsupported type - raise error
            raise ValueError(f"Unsupported value type for DMQ write: {type(value).__name__}")

    def write(
        self,
        drf: str,
        value: Value,
        timeout: Optional[float] = None,
    ) -> WriteResult:
        """Write a single device value.

        Args:
            drf: Device Request Format string
            value: Value to write
            timeout: Operation timeout in seconds

        Returns:
            WriteResult indicating success or failure

        Raises:
            AuthenticationError: If no KerberosAuth configured
        """
        results = self.write_many([(drf, value)], timeout=timeout)
        return results[0]

    def write_many(
        self,
        settings: list[tuple[str, Value]],
        timeout: Optional[float] = None,
    ) -> list[WriteResult]:
        """Write multiple device values.

        Uses a single shared SelectConnection with on-demand channels.
        Write sessions are cached per device and reused for subsequent writes.

        Args:
            settings: List of (drf, value) tuples to write
            timeout: Operation timeout in seconds

        Returns:
            List of WriteResult for each setting

        Raises:
            AuthenticationError: If no KerberosAuth configured
        """
        if not settings:
            return []

        self._check_not_io_thread()
        if self._closed:
            raise RuntimeError("Backend is closed")

        if not isinstance(self._auth, KerberosAuth):
            raise AuthenticationError("KerberosAuth required for writes. Pass auth=KerberosAuth().")

        # Ensure IO thread is running (shared with streaming)
        self._ensure_io_thread()

        # Prepare results container (shared with IO thread)
        results: list[WriteResult | None] = [None] * len(settings)
        tracker = _WriteCompletionTracker(total_devices=0)

        # Schedule async execution on IO thread
        if self._select_connection is None:
            raise RuntimeError("Connection not available")

        self._select_connection.ioloop.add_callback_threadsafe(
            lambda: self._execute_write_many_async(settings, results, tracker)
        )

        # Block until done or timeout
        effective_timeout = timeout if timeout is not None else self._timeout
        if not tracker.done_event.wait(effective_timeout):
            # Timeout - schedule abort on IO thread and wait for it to finish
            # before touching results.  This avoids a race where the main thread
            # overwrites a successful result with ERR_TIMEOUT.
            init_drfs_involved = {prepare_for_write(drf) for drf, _ in settings}
            abort_done = threading.Event()

            def do_abort():
                self._abort_pending_writes(init_drfs_involved)
                abort_done.set()

            self._select_connection.ioloop.add_callback_threadsafe(do_abort)
            if not abort_done.wait(timeout=2.0):
                logger.warning(
                    "Write abort for %d device(s) did not complete within 2s; IO thread may be unresponsive",
                    len(init_drfs_involved),
                )

            # Fill timeout errors for results the IO thread didn't cover
            # (e.g. devices that never got a session).  Safe now because
            # _abort_pending_writes has finished and cleared pending state.
            for i, (drf, _) in enumerate(settings):
                if results[i] is None:
                    results[i] = WriteResult(
                        drf=drf,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_TIMEOUT,
                        message="Request timeout",
                    )

        # Propagate auth/GSS errors from IO thread
        if tracker.exception is not None:
            raise tracker.exception

        # Ensure all results are filled (shouldn't happen, but be safe)
        final_results: list[WriteResult] = []
        for i, (drf, _) in enumerate(settings):
            r = results[i]
            if r is not None:
                final_results.append(r)
            else:
                final_results.append(
                    WriteResult(drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message="Unknown error")
                )

        return final_results

    # ─────────────────────────────────────────────────────────────────────────
    # SelectConnection streaming infrastructure
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_io_thread(self) -> None:
        """Start the IO thread and SelectConnection if not already running."""
        with self._stream_lock:
            if self._io_thread is not None and self._io_thread.is_alive():
                # Thread exists - still verify connection is ready before returning
                if self._connection_ready.is_set() and self._connection_error is None:
                    return
                # Connection not ready yet (startup in progress) - fall through to wait
            else:
                if self._closed:
                    raise RuntimeError("Backend is closed")

                self._connection_ready.clear()
                self._connection_error = None

                self._io_thread = threading.Thread(
                    target=self._io_loop_thread,
                    name="DMQBackend-IOLoop",
                    daemon=True,
                )
                self._io_thread.start()

        # Wait for connection to be ready
        if not self._connection_ready.wait(timeout=self._timeout):
            raise ConnectionError(
                f"Failed to connect to RabbitMQ at {self._host}:{self._port}: timed out after {self._timeout}s"
            )
        if self._connection_error is not None:
            err = self._connection_error
            detail = str(err) or type(err).__name__
            raise ConnectionError(f"Failed to connect to RabbitMQ at {self._host}:{self._port}: {detail}") from err

    def _io_loop_thread(self) -> None:
        """Background thread running the SelectConnection event loop."""
        params = pika.ConnectionParameters(
            host=self._host,
            port=self._port,
            virtual_host=self._vhost,
            heartbeat=15,
            blocked_connection_timeout=self._timeout,
        )

        try:
            self._select_connection = SelectConnection(
                parameters=params,
                on_open_callback=self._on_connection_open,
                on_open_error_callback=self._on_connection_open_error,
                on_close_callback=self._on_connection_closed,
            )
            logger.debug("Starting SelectConnection ioloop")
            self._select_connection.ioloop.start()
        except Exception as e:
            logger.error(f"IO loop thread error: {e}")
            self._connection_error = e
            self._connection_ready.set()
        finally:
            # Reset connection so _ensure_io_thread can restart if needed
            self._select_connection = None
            logger.debug("IO loop thread exiting")

    def _on_connection_open(self, connection: SelectConnection) -> None:
        """Called when SelectConnection is established."""
        logger.info(f"SelectConnection opened to {self._host}:{self._port}")
        self._connection_ready.set()

    def _on_connection_open_error(self, connection: SelectConnection, error: Exception) -> None:
        """Called when SelectConnection fails to open."""
        logger.error(f"SelectConnection open error: {error}")
        self._connection_error = error
        self._connection_ready.set()

    def _on_connection_closed(self, connection: SelectConnection, reason: Exception) -> None:
        """Called when SelectConnection is closed."""
        logger.info(f"SelectConnection closed: {reason}")
        self._connection_ready.clear()

        # Fail all pending writes - signal each tracker exactly once
        trackers: dict[int, _WriteCompletionTracker] = {}

        # Fail writes queued during channel setup
        for init_drf, queued in self._pending_session_setups.items():
            for q_settings, q_results, q_tracker in queued:
                for i, drf, _ in q_settings:
                    if q_results[i] is None:
                        q_results[i] = WriteResult(
                            drf=drf,
                            facility_code=FACILITY_ACNET,
                            error_code=ERR_RETRY,
                            message=f"Connection closed: {reason}",
                        )
                trackers.setdefault(id(q_tracker), q_tracker)
        self._pending_session_setups.clear()

        for session in self._write_sessions.values():
            # Cancel init timer
            if session.init_timer is not None:
                try:
                    connection.ioloop.remove_timeout(session.init_timer)
                except Exception:
                    pass
            # Fail queued sends
            for q_settings, q_results, q_tracker in session.queued_sends:
                for i, drf, _ in q_settings:
                    if q_results[i] is None:
                        q_results[i] = WriteResult(
                            drf=drf,
                            facility_code=FACILITY_ACNET,
                            error_code=ERR_RETRY,
                            message=f"Connection closed: {reason}",
                        )
                trackers.setdefault(id(q_tracker), q_tracker)
            session.queued_sends = []
            # Fail pending writes
            for corr_id, (i, drf, results, tracker) in list(session.pending.items()):
                if results[i] is None:
                    results[i] = WriteResult(
                        drf=drf,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_RETRY,
                        message=f"Connection closed: {reason}",
                    )
                trackers.setdefault(id(tracker), tracker)
            session.pending.clear()
        for tracker in trackers.values():
            tracker.device_complete()

        # Clear write state
        self._write_sessions.clear()

        # Notify all active subscriptions of the connection loss
        # Copy under lock, dispatch outside - callbacks may call remove()/stop_streaming()
        with self._stream_lock:
            subs = list(self._subscriptions.values())
            self._subscriptions.clear()
        for sub in subs:
            # Unblock any thread waiting on subscription setup
            sub.setup_error = reason
            sub.setup_complete.set()
            sub.handle._signal_error(reason)
            if sub.handle._on_error is not None:
                self._dispatcher.dispatch_error(sub.handle._on_error, reason, sub.handle)
        # Stop the ioloop (will exit the thread)
        try:
            connection.ioloop.stop()
        except Exception:
            pass

    def _start_subscription_async(
        self, sub: _SelectSubscription, init_body: bytes, init_headers: dict, init_message_id: str
    ) -> None:
        """Schedule subscription setup on the IO loop (called from main thread)."""
        if self._select_connection is None:
            sub.setup_error = RuntimeError("No connection")
            sub.setup_complete.set()
            return

        self._select_connection.ioloop.add_callback_threadsafe(
            lambda: self._open_channel_for_subscription(sub, init_body, init_headers, init_message_id)
        )

    def _open_channel_for_subscription(
        self, sub: _SelectSubscription, init_body: bytes, init_headers: dict, init_message_id: str
    ) -> None:
        """Open a channel for a subscription (runs in IO thread)."""
        if self._select_connection is None or not self._select_connection.is_open:
            sub.setup_error = RuntimeError("Connection not open")
            sub.setup_complete.set()
            return

        def on_ch_open(channel):
            if sub.handle._stopped:
                try:
                    channel.close()
                except Exception:
                    pass
                return
            sub.channel = channel
            channel.add_on_close_callback(lambda ch, reason: self._on_channel_closed(ch, reason, sub))

        def on_ready(channel, exchange_name, queue_name):
            if sub.handle._stopped:
                try:
                    channel.close()
                except Exception:
                    pass
                return
            sub.queue_name = queue_name

            # Send INIT to amq.topic
            channel.basic_publish(
                exchange=INIT_EXCHANGE,
                routing_key="I",
                body=init_body,
                properties=pika.BasicProperties(
                    message_id=init_message_id,
                    reply_to=sub.exchange_name,
                    app_id="pacsys",
                    headers=init_headers,
                ),
            )
            logger.debug(f"Sent INIT for sub {sub.sub_id[:8]}, {len(sub.drfs)} devices")

            # Start consuming
            sub.consumer_tag = channel.basic_consume(
                queue=sub.queue_name,
                on_message_callback=lambda ch, method, props, body: self._on_message(sub, ch, method, props, body),
                auto_ack=False,
            )

            self._schedule_heartbeat(sub)
            sub.setup_complete.set()
            logger.info(f"Subscription {sub.sub_id[:8]} setup complete")

        self._setup_channel_async(
            on_ready=on_ready,
            exchange_name=sub.exchange_name,
            on_channel_open=on_ch_open,
        )

    def _schedule_heartbeat(self, sub: _SelectSubscription) -> None:
        """Schedule next heartbeat for subscription (runs in IO thread)."""
        if self._select_connection is None or not self._select_connection.is_open:
            return
        if sub.handle._stopped:
            return

        def send_heartbeat():
            if sub.handle._stopped or sub.channel is None or not sub.channel.is_open:
                return
            try:
                sub.channel.basic_publish(exchange=sub.exchange_name, routing_key="H", body=b"")
                logger.debug(f"Sent heartbeat for sub {sub.sub_id[:8]}")
            except Exception as e:
                logger.warning(f"Failed to send heartbeat: {e}")
            # Schedule next heartbeat
            self._schedule_heartbeat(sub)

        sub.heartbeat_handle = self._select_connection.ioloop.call_later(5.0, send_heartbeat)

    def _on_channel_closed(self, channel: Channel, reason: Exception, sub: _SelectSubscription) -> None:
        """Channel closed callback (runs in IO thread)."""
        logger.debug(f"Channel closed for sub {sub.sub_id[:8]}: {reason}")
        with self._stream_lock:
            was_active = self._subscriptions.pop(sub.sub_id, None) is not None
        if not was_active:
            return  # user-initiated close via remove() - already cleaned up
        sub.handle._signal_error(reason)
        if sub.handle._on_error is not None:
            self._dispatcher.dispatch_error(sub.handle._on_error, reason, sub.handle)

    def _on_message(
        self,
        sub: _SelectSubscription,
        channel: Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.BasicProperties,
        body: bytes,
    ) -> None:
        """Handle incoming message for subscription (runs in IO thread)."""
        result = _resolve_reply(method.routing_key, body, sub.drfs, sub.drf_to_idx)
        channel.basic_ack(method.delivery_tag)
        if result is None:
            return
        reply, idx, ref_id = result
        drf = sub.drfs[idx]
        handle = sub.handle

        # Deliver to all indices sharing this DRF (handles duplicate subscriptions)
        for i in sub.drf_to_all_indices.get(drf, (idx,)):
            reading = _reply_to_reading(reply, sub.drfs[i])
            if sub.callback is not None:
                self._dispatcher.dispatch_reading(sub.callback, reading, handle)
            else:
                handle._dispatch(reading)

    def _cancel_subscription_async(self, sub: _SelectSubscription) -> None:
        """Schedule subscription cancellation on the IO loop."""
        if self._select_connection is None or not self._select_connection.is_open:
            return

        def do_cancel():
            # Cancel heartbeat timer
            if sub.heartbeat_handle is not None:
                try:
                    self._select_connection.ioloop.remove_timeout(sub.heartbeat_handle)  # type: ignore
                except Exception:
                    pass
                sub.heartbeat_handle = None

            # Send DROP
            if sub.channel is not None and sub.channel.is_open:
                try:
                    sub.channel.basic_publish(exchange=sub.exchange_name, routing_key="D", body=b"")
                    logger.debug(f"Sent DROP for sub {sub.sub_id[:8]}")
                except Exception:
                    pass

                # Cancel consumer
                if sub.consumer_tag is not None:
                    try:
                        sub.channel.basic_cancel(sub.consumer_tag)
                    except Exception:
                        pass

                # Close channel
                try:
                    sub.channel.close()
                except Exception:
                    pass

            sub.handle._signal_stop()

        self._select_connection.ioloop.add_callback_threadsafe(do_cancel)

    def subscribe(
        self,
        drfs: list[str],
        callback: Optional[ReadingCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ) -> SubscriptionHandle:
        """Subscribe to devices for streaming data.

        Uses SelectConnection with a shared IO thread for all subscriptions.
        Each subscription gets its own channel on the shared connection.
        """
        self._check_not_io_thread()
        if self._closed:
            raise RuntimeError("Backend is closed")

        if not drfs:
            raise ValueError("drfs cannot be empty")

        # Ensure IO thread is running
        self._ensure_io_thread()

        sub_id = str(uuid.uuid4())
        is_callback_mode = callback is not None
        exchange_name = str(uuid.uuid4())

        # Create handle
        handle = _DMQSubscriptionHandle(
            backend=self,
            sub_id=sub_id,
            drfs=drfs,
            is_callback_mode=is_callback_mode,
            on_error=on_error,
        )

        # Create subscription state (with reverse index for duplicate DRFs)
        drf_to_all: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(drfs):
            drf_to_all[d].append(i)
        sub = _SelectSubscription(
            sub_id=sub_id,
            drfs=drfs,
            drf_to_idx={d: i for i, d in enumerate(drfs)},
            drf_to_all_indices=dict(drf_to_all),
            handle=handle,
            callback=callback,
            exchange_name=exchange_name,
        )

        # Prepare INIT message in main thread (GSS context creation can block)
        ctx = self._create_gss_context()
        token = ctx.step()

        req = ReadingRequest_request()
        req.dataRequest = list(drfs)
        init_body = bytes(req.marshal())

        message_id = str(uuid.uuid4())
        mic = self._sign_message(ctx, init_body, message_id, reply_to=exchange_name, app_id="pacsys")

        init_headers = {
            "host-address": self._local_ip,
            "gss-token": bytes(token) if token else b"",
            "signature": mic,
        }

        with self._stream_lock:
            self._subscriptions[sub_id] = sub

        # Schedule async setup on IO loop
        self._start_subscription_async(sub, init_body, init_headers, message_id)

        # Wait for setup to complete
        if not sub.setup_complete.wait(timeout=self._timeout):
            with self._stream_lock:
                self._subscriptions.pop(sub_id, None)
            # Cancel any partially-setup resources on IO loop
            self._cancel_subscription_async(sub)
            raise RuntimeError("Timeout waiting for subscription setup")

        if sub.setup_error is not None:
            with self._stream_lock:
                self._subscriptions.pop(sub_id, None)
            # Cancel any partially-setup resources on IO loop
            self._cancel_subscription_async(sub)
            raise sub.setup_error

        mode_str = "callback" if is_callback_mode else "iterator"
        logger.info(f"Created {mode_str} subscription sub_id={sub_id[:8]} for {len(drfs)} devices")

        return handle

    def remove(self, handle: SubscriptionHandle) -> None:
        """Remove a subscription.

        Schedules cancellation on the IO loop (channel close, DROP message).
        """
        if not isinstance(handle, _DMQSubscriptionHandle):
            raise TypeError(f"Expected _DMQSubscriptionHandle, got {type(handle).__name__}")

        sub_id = handle._sub_id

        with self._stream_lock:
            sub = self._subscriptions.pop(sub_id, None)

        if sub is None:
            return

        self._cancel_subscription_async(sub)
        handle._signal_stop()
        logger.info(f"Removed subscription sub_id={sub_id[:8]}")

    def stop_streaming(self) -> None:
        """Stop all streaming subscriptions.

        Cancels all subscriptions and closes the SelectConnection.
        """
        logger.debug("Stopping all streaming")

        with self._stream_lock:
            subs = list(self._subscriptions.values())
            self._subscriptions.clear()

        for sub in subs:
            sub.handle._signal_stop()
            self._cancel_subscription_async(sub)

        # Close the SelectConnection and stop the IO thread
        if self._select_connection is not None and self._select_connection.is_open:

            def close_connection():
                if self._select_connection is not None:
                    self._select_connection.close()

            try:
                self._select_connection.ioloop.add_callback_threadsafe(close_connection)
            except Exception:
                pass

        if self._io_thread is not None:
            self._io_thread.join(timeout=3.0)
            if not self._io_thread.is_alive():
                self._io_thread = None

        self._select_connection = None
        logger.info("All streaming stopped")

    def close(self) -> None:
        """Close the backend and release all resources."""
        if self._closed:
            return

        self._closed = True

        # Stop streaming and close SelectConnection
        # Write state (_write_sessions) is cleared on the IO thread during
        # connection close to avoid cross-thread mutation.
        self.stop_streaming()
        self._dispatcher.close()

        # After IO thread has joined, safe to clear any remnants
        self._write_sessions.clear()
        self._pending_session_setups.clear()

        logger.info("DMQBackend closed")

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        n_subs = len(self._subscriptions)
        return f"DMQBackend({self._host}:{self._port}, subs={n_subs}, {status})"


__all__ = ["DMQBackend"]
