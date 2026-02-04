"""
DMQ Backend - RabbitMQ/AMQP backend for ACNET device access.

Uses RabbitMQ message broker to communicate with ACNET via the DMQ server.
See SPECIFICATION.md and diodmq_impl.md for protocol details.
"""

import logging
import os
import queue
import socket
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np
import pika
import pika.spec
from pika.adapters.select_connection import SelectConnection
from pika.channel import Channel

from pacsys.acnet.errors import ERR_OK, ERR_RETRY, ERR_TIMEOUT, FACILITY_ACNET
from pacsys.auth import Auth, KerberosAuth
from pacsys.backends import Backend, timestamp_from_millis, validate_alarm_dict
from pacsys.errors import AuthenticationError, DeviceError
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
_DEFAULT_QUEUE_MAXSIZE = 10000
DEFAULT_VHOST = "/"
DEFAULT_TIMEOUT = 5.0
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
MAX_WRITE_SESSIONS = 256

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
    readings: dict[int, Reading] = field(default_factory=dict)
    done_event: threading.Event = field(default_factory=threading.Event)
    channel: Optional[Channel] = None
    exchange_name: str = ""
    queue_name: str = ""
    consumer_tag: Optional[str] = None


@dataclass
class _StandbyChannel:
    """Pre-warmed channel ready for write assignment.

    Standby channels have exchange/queue declared and bound to R.#/Q routing keys,
    but no DMQ job (no INIT sent). GSS context is generated at assignment time
    (fast, <1ms) to avoid Kerberos token expiry.
    """

    channel: Channel
    exchange_name: str
    queue_name: str


@dataclass
class _WriteSession:
    """Active write session for a device.

    Created when a standby channel is promoted for a specific device.
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
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = socket.inet_aton(s.getsockname()[0])
        s.close()
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


def _reply_to_reading(reply, drf: str, ref_id: Optional[int] = None) -> Reading:
    """Convert a DMQ reply to a Reading object."""
    if isinstance(reply, ErrorSample_reply):
        return Reading(
            drf=drf,
            value_type=ValueType.SCALAR,
            tag=getattr(reply, "ref_id", ref_id),
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
            tag=getattr(reply, "ref_id", ref_id),
            error_code=ERR_OK,
            value=extract(reply),
            timestamp=timestamp_from_millis(reply.time) if reply.time else None,
            cycle=getattr(reply, "cycle_time", 0),
        )

    logger.warning(f"Unknown reply type: {type(reply).__name__}")
    return Reading(
        drf=drf,
        value_type=ValueType.SCALAR,
        tag=ref_id,
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
        return None
    if isinstance(reply, ErrorSample_reply) and reply.errorNumber == DMQ_PENDING_ERROR:
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


class _DMQSubscriptionHandle(SubscriptionHandle):
    """Subscription handle for DMQBackend."""

    def __init__(
        self,
        backend: "DMQBackend",
        sub_id: str,
        drfs: list[str],
        is_callback_mode: bool,
        on_error: Optional[ErrorCallback] = None,
    ):
        self._backend = backend
        self._sub_id = sub_id
        self._drfs = drfs
        self._is_callback_mode = is_callback_mode
        self._on_error = on_error
        self._queue: queue.Queue[Reading] = queue.Queue(maxsize=_DEFAULT_QUEUE_MAXSIZE)
        self._stopped = False
        self._exc: Optional[Exception] = None
        self._ref_ids: list[int] = list(range(1, len(drfs) + 1))

    @property
    def ref_ids(self) -> list[int]:
        return list(self._ref_ids)

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def exc(self) -> Optional[Exception]:
        return self._exc

    def readings(
        self,
        timeout: Optional[float] = None,
    ) -> Iterator[tuple[Reading, SubscriptionHandle]]:
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
        if not self._stopped:
            self._backend.remove(self)
            self._stopped = True


@dataclass
class _SelectSubscription:
    """State for a single subscription on the shared SelectConnection."""

    sub_id: str
    drfs: list[str]
    drf_to_idx: dict[str, int]  # device_name -> index for O(1) routing key lookup
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
    ):
        """
        Initialize DMQ backend.

        Args:
            host: RabbitMQ broker hostname (default: appsrv2.fnal.gov)
            port: RabbitMQ broker port (default: 5672)
            vhost: RabbitMQ virtual host (default: /)
            timeout: Default operation timeout in seconds (default: 5.0)
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

        # Streaming state (SelectConnection model)
        self._stream_lock = threading.Lock()
        self._subscriptions: dict[str, _SelectSubscription] = {}
        self._select_connection: Optional[SelectConnection] = None
        self._io_thread: Optional[threading.Thread] = None
        self._connection_ready = threading.Event()
        self._connection_error: Optional[Exception] = None

        # Write state (unified with SelectConnection)
        self._write_sessions: dict[str, _WriteSession] = {}  # init_drf -> active session
        self._standby_channels: list[_StandbyChannel] = []  # pre-warmed channels
        self._standby_target = 5  # target pool size
        self._standby_pending = 0  # channels being created

        # Cache local IP (used in INIT, SETTING messages)
        self._local_ip = _get_host_address()

        # Validate auth eagerly (fail fast)
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
    ) -> str:
        """Send INIT request to start a job.

        Creates a GSS context, includes the token and MIC signature in the INIT message
        for server authentication.

        Returns:
            message_id for correlation
        """
        # Create GSS context and get token for authentication
        ctx = self._create_gss_context()
        token = ctx.step()

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
        job = _ReadJob(
            drfs=drfs,
            prepared_drfs=prepared_drfs,
            drf_to_idx={drf: i for i, drf in enumerate(prepared_drfs)},
        )

        self._ensure_io_thread()
        conn = self._select_connection
        if conn is None:
            raise ConnectionError(f"No connection to RabbitMQ at {self._host}:{self._port}")

        conn.ioloop.add_callback_threadsafe(lambda: self._start_read_async(job))

        if not job.done_event.wait(timeout):
            # Timeout -- schedule cleanup on IO thread
            if conn.is_open:
                conn.ioloop.add_callback_threadsafe(lambda: self._complete_read(job))
                job.done_event.wait(timeout=5.0)

        # Build result list
        result: list[Reading] = []
        for i, drf in enumerate(drfs):
            if i in job.readings:
                result.append(job.readings[i])
            else:
                result.append(
                    Reading(
                        drf=drf,
                        value_type=ValueType.SCALAR,
                        tag=i + 1,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_TIMEOUT,
                        value=None,
                        message="Request timeout",
                        timestamp=None,
                        cycle=0,
                    )
                )

        return result

    def _start_read_async(self, job: _ReadJob) -> None:
        """Start async read on IO thread."""
        if self._select_connection is None or not self._select_connection.is_open:
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
            self._send_init(channel, exchange_name, job.prepared_drfs)
            job.consumer_tag = channel.basic_consume(
                queue=queue_name,
                on_message_callback=lambda ch, method, props, body: self._on_read_message(job, ch, method, body),
                auto_ack=False,
            )

        self._setup_channel_async(on_ready=on_ready)

    def _on_read_message(self, job: _ReadJob, channel: Channel, method, body: bytes) -> None:
        """Handle a message for an async read (IO thread)."""
        channel.basic_ack(method.delivery_tag)
        result = _resolve_reply(method.routing_key, body, job.prepared_drfs, job.drf_to_idx)
        if result is None:
            return
        reply, idx, ref_id = result
        # Fill this index and any unfilled duplicates with the same prepared DRF.
        # The server deduplicates requests, so only one reply arrives per unique device.
        drf = job.prepared_drfs[idx]
        filled = False
        for i, d in enumerate(job.prepared_drfs):
            if d == drf and i not in job.readings:
                job.readings[i] = _reply_to_reading(reply, job.drfs[i], ref_id)
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
            ctx = gssapi.SecurityContext(
                name=name,
                usage="initiate",
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
        Shared by standby pool, on-demand write, and subscription setup.

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
    # Standby Channel Pool Management (runs on IO thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _replenish_standby_pool(self) -> None:
        """Ensure standby pool has target number of channels (IO thread)."""
        needed = self._standby_target - len(self._standby_channels) - self._standby_pending
        for _ in range(needed):
            self._standby_pending += 1
            self._open_standby_channel()

    def _open_standby_channel(self) -> None:
        """Open a new standby channel (IO thread)."""
        if self._select_connection is None or not self._select_connection.is_open:
            self._standby_pending = max(0, self._standby_pending - 1)
            return

        exchange_name = str(uuid.uuid4())

        def on_ch_open(channel):
            channel.add_on_close_callback(lambda ch, reason: self._on_standby_channel_closed(ch, reason, exchange_name))

        def on_ready(channel, ex, queue_name):
            self._standby_pending = max(0, self._standby_pending - 1)
            standby = _StandbyChannel(channel=channel, exchange_name=ex, queue_name=queue_name)
            self._standby_channels.append(standby)
            logger.debug(f"Standby channel ready: exchange={ex[:8]}, pool_size={len(self._standby_channels)}")

        self._setup_channel_async(on_ready=on_ready, exchange_name=exchange_name, on_channel_open=on_ch_open)

    def _on_standby_channel_closed(self, channel: Channel, reason: Exception, exchange_name: str) -> None:
        """Standby channel closed unexpectedly (IO thread)."""
        # Remove from pool if present
        self._standby_channels = [s for s in self._standby_channels if s.channel is not channel]
        logger.debug(f"Standby channel closed: {reason}")

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

        # Cancel heartbeat timer
        if session.heartbeat_handle is not None and self._select_connection is not None:
            try:
                self._select_connection.ioloop.remove_timeout(session.heartbeat_handle)
            except Exception:
                pass

        # Cancel cleanup timer
        if session.cleanup_handle is not None and self._select_connection is not None:
            try:
                self._select_connection.ioloop.remove_timeout(session.cleanup_handle)
            except Exception:
                pass

        # Fail any pending writes
        for corr_id, (i, drf, results_list, pending_tracker) in list(session.pending.items()):
            if results_list[i] is None:
                results_list[i] = WriteResult(
                    drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=f"Session closed: {reason}"
                )
            pending_tracker.device_complete()
        session.pending.clear()

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

    def _evict_lru_write_session(self) -> None:
        """Evict the least-recently-used write session to make room (IO thread)."""
        if not self._write_sessions:
            return
        # Find session with oldest last_used that has no pending writes
        lru_key = None
        lru_time = float("inf")
        for key, session in self._write_sessions.items():
            if not session.pending and session.last_used < lru_time:
                lru_time = session.last_used
                lru_key = key
        # If all sessions have pending writes, evict the oldest anyway
        if lru_key is None:
            lru_key = min(self._write_sessions, key=lambda k: self._write_sessions[k].last_used)
        self._close_write_session(lru_key, reason="evicted (session limit)")

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
                self._send_settings_async(session, drf_settings, results, tracker)
                return
            else:
                # Session dead, clean up properly
                self._close_write_session(init_drf, reason="channel dead")

        # Enforce session limit before creating new one
        if len(self._write_sessions) >= MAX_WRITE_SESSIONS:
            self._evict_lru_write_session()

        # Need new session - pop from standby pool
        if self._standby_channels:
            standby = self._standby_channels.pop()
            self._promote_to_write_session(standby, init_drf, drf_settings, results, tracker)
            # Replenish pool
            self._replenish_standby_pool()
            return

        # No standby available - create channel on-demand
        self._create_channel_for_write(init_drf, drf_settings, results, tracker)

    def _create_channel_for_write(
        self,
        init_drf: str,
        drf_settings: list[tuple[int, str, Value]],
        results: list[WriteResult | None],
        tracker: _WriteCompletionTracker,
    ) -> None:
        """Create channel on-demand for write (IO thread)."""
        if self._select_connection is None or not self._select_connection.is_open:
            for i, drf, _ in drf_settings:
                results[i] = WriteResult(
                    drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message="Connection not open"
                )
            tracker.device_complete()
            return

        def on_ready(channel, exchange_name, queue_name):
            standby = _StandbyChannel(channel=channel, exchange_name=exchange_name, queue_name=queue_name)
            self._promote_to_write_session(standby, init_drf, drf_settings, results, tracker)

        self._setup_channel_async(on_ready=on_ready)

    def _promote_to_write_session(
        self,
        standby: _StandbyChannel,
        init_drf: str,
        drf_settings: list[tuple[int, str, Value]],
        results: list[WriteResult | None],
        tracker: _WriteCompletionTracker,
    ) -> None:
        """Convert standby channel to active write session (IO thread)."""
        device = get_device_name(init_drf)
        try:
            # Generate GSS context (fast, <1ms)
            ctx = self._create_gss_context()
            token = ctx.step()
        except Exception as e:
            logger.error(f"GSS context creation failed for {device}: {e}")
            tracker.abort(AuthenticationError(f"GSS context creation failed: {e}"))
            return

        # Channel already has R.# and Q bindings from standby setup.
        # Server replies use R.# routing keys, no S.# binding needed.
        self._on_write_bind_complete(standby, init_drf, ctx, token, drf_settings, results, tracker)

    def _on_write_bind_complete(
        self,
        standby: _StandbyChannel,
        init_drf: str,
        ctx: Any,
        token: bytes,
        drf_settings: list[tuple[int, str, Value]],
        results: list[WriteResult | None],
        tracker: _WriteCompletionTracker,
    ) -> None:
        """Binding complete, send INIT and create session (IO thread)."""
        device = get_device_name(init_drf)
        # Build and send INIT -- init_drf is the exact string registered
        # on the server as the setter key; SETTING routing key must match it
        req = SettingRequest_request()
        req.dataRequest = [init_drf]
        body = bytes(req.marshal())

        message_id = str(uuid.uuid4())

        mic = self._sign_message(ctx, body, message_id, reply_to=standby.exchange_name, app_id="pacsys")

        standby.channel.basic_publish(
            exchange=INIT_EXCHANGE,
            routing_key="I",
            body=body,
            properties=pika.BasicProperties(
                message_id=message_id,
                reply_to=standby.exchange_name,
                app_id="pacsys",
                headers={
                    "gss-token": bytes(token) if token else b"",
                    "signature": mic,
                    "host-address": self._local_ip,
                },
            ),
        )

        # Create session
        session = _WriteSession(
            device=device,
            init_drf=init_drf,
            channel=standby.channel,
            exchange_name=standby.exchange_name,
            queue_name=standby.queue_name,
            gss_context=ctx,
            last_used=time.monotonic(),
        )

        # Add on-close callback
        standby.channel.add_on_close_callback(
            lambda ch, reason: self._on_write_session_channel_closed(init_drf, reason)
        )

        # Start consuming (for responses)
        session.consumer_tag = standby.channel.basic_consume(
            queue=standby.queue_name,
            on_message_callback=lambda ch, m, p, b: self._on_write_message(session, ch, m, p, b),
            auto_ack=False,
        )

        self._write_sessions[init_drf] = session

        # Schedule heartbeat and idle cleanup
        self._schedule_write_session_heartbeat(session)
        self._schedule_write_session_cleanup(session)

        # Brief delay for INIT processing, then send settings
        if self._select_connection is not None:
            self._select_connection.ioloop.call_later(
                0.05,  # 50ms for INIT to process
                lambda: self._send_settings_async(session, drf_settings, results, tracker),
            )

        logger.debug(f"Write session created for {device} ({init_drf}), exchange={standby.exchange_name[:8]}")

    def _on_write_session_channel_closed(self, init_drf: str, reason: Exception) -> None:
        """Write session channel closed (IO thread)."""
        session = self._write_sessions.pop(init_drf, None)
        if session is not None:
            # Cancel timers
            conn = self._select_connection
            for handle in (session.heartbeat_handle, session.cleanup_handle):
                if handle is not None and conn is not None:
                    try:
                        conn.ioloop.remove_timeout(handle)
                    except Exception:
                        pass
            # Fail any pending writes
            for corr_id, (i, drf, results_list, pending_tracker) in list(session.pending.items()):
                if results_list[i] is None:
                    results_list[i] = WriteResult(
                        drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=f"Channel closed: {reason}"
                    )
                pending_tracker.device_complete()
            session.pending.clear()
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

        for i, drf, value in device_settings:
            try:
                sample = self._value_to_sample(value, ref_id=1)
                body = bytes(sample.marshal())

                # Use message_id as the pending key. The Java server echoes
                # message_id back as the response's correlationId (impl2), or
                # sends empty string (impl). Using message_id avoids mismatch.
                message_id = str(uuid.uuid4())

                mic = self._sign_message(session.gss_context, body, message_id)

                session.pending[message_id] = (i, drf, results, tracker)
                pending_for_device += 1

                session.channel.basic_publish(
                    exchange=session.exchange_name,
                    routing_key=f"S.{session.init_drf}",
                    body=body,
                    properties=pika.BasicProperties(
                        message_id=message_id,
                        headers={"signature": mic, "host-address": self._local_ip},
                    ),
                )
                logger.debug(f"Sent SETTING for {session.device} (rk=S.{session.init_drf}), msg_id={message_id[:8]}")

            except Exception as e:
                logger.error(f"Failed to send setting for {drf}: {e}")
                results[i] = WriteResult(drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=str(e))

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
        rk: str = method.routing_key  # type: ignore[assignment]
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

        # Skip PENDING
        if isinstance(reply, ErrorSample_reply) and reply.errorNumber == DMQ_PENDING_ERROR:
            return

        # Match by correlation_id. Server echoes message_id as the response's
        # correlationId; older impl sends empty string. Fall back to FIFO
        # (oldest pending) when correlationId is missing or unknown.
        if corr_id and corr_id in session.pending:
            i, drf, results, tracker = session.pending.pop(corr_id)
        elif session.pending:
            # FIFO fallback: match oldest pending write
            oldest_key = next(iter(session.pending))
            i, drf, results, tracker = session.pending.pop(oldest_key)
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

        Sessions with timed-out writes are closed since they're in a bad state
        (e.g. DMQ sent PENDING but never followed up with a final status).
        """
        for init_drf in init_drfs:
            session = self._write_sessions.get(init_drf)
            if session is None:
                continue
            if not session.pending:
                continue
            # Mark all pending writes as timed out
            for corr_id, (i, drf, results, tracker) in list(session.pending.items()):
                if results[i] is None:
                    results[i] = WriteResult(
                        drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_TIMEOUT, message="Request timeout"
                    )
            session.pending.clear()
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

        Uses a single shared SelectConnection with pre-warmed standby channels.
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
            # Timeout - abort and fill in errors
            init_drfs_involved = {prepare_for_write(drf) for drf, _ in settings}
            self._select_connection.ioloop.add_callback_threadsafe(
                lambda: self._abort_pending_writes(init_drfs_involved)
            )
            # Fill timeout errors for missing results
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
                return
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
        # Start pre-warming the standby channel pool
        self._replenish_standby_pool()

    def _on_connection_open_error(self, connection: SelectConnection, error: Exception) -> None:
        """Called when SelectConnection fails to open."""
        logger.error(f"SelectConnection open error: {error}")
        self._connection_error = error
        self._connection_ready.set()

    def _on_connection_closed(self, connection: SelectConnection, reason: Exception) -> None:
        """Called when SelectConnection is closed."""
        logger.info(f"SelectConnection closed: {reason}")

        # Fail all pending writes
        for session in self._write_sessions.values():
            for corr_id, (i, drf, results, tracker) in list(session.pending.items()):
                if results[i] is None:
                    results[i] = WriteResult(
                        drf=drf,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_RETRY,
                        message=f"Connection closed: {reason}",
                    )
                tracker.device_complete()
            session.pending.clear()

        # Clear write state
        self._write_sessions.clear()
        self._standby_channels.clear()
        self._standby_pending = 0

        # Notify all active subscriptions of the connection loss
        with self._stream_lock:
            for sub in self._subscriptions.values():
                sub.handle._exc = reason
                sub.handle._stopped = True
                if sub.handle._on_error is not None:
                    try:
                        sub.handle._on_error(reason, sub.handle)
                    except Exception:
                        pass
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
            sub.channel = channel
            channel.add_on_close_callback(lambda ch, reason: self._on_channel_closed(ch, reason, sub))

        def on_ready(channel, exchange_name, queue_name):
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
        sub.handle._stopped = True
        if sub.handle._on_error is not None and sub.handle._exc is None:
            sub.handle._exc = reason
            try:
                sub.handle._on_error(reason, sub.handle)
            except Exception:
                pass

    def _on_message(
        self,
        sub: _SelectSubscription,
        channel: Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.BasicProperties,
        body: bytes,
    ) -> None:
        """Handle incoming message for subscription (runs in IO thread)."""
        channel.basic_ack(method.delivery_tag)
        result = _resolve_reply(method.routing_key, body, sub.drfs, sub.drf_to_idx)  # type: ignore[arg-type]
        if result is None:
            return
        reply, idx, ref_id = result
        reading = _reply_to_reading(reply, sub.drfs[idx], ref_id)
        handle = sub.handle

        if sub.callback is not None:
            # Run callback (note: blocking callbacks will stall IO loop)
            try:
                sub.callback(reading, handle)
            except Exception as e:
                logger.error(f"Error in subscription callback: {e}")
        else:
            try:
                handle._queue.put_nowait(reading)
            except queue.Full:
                logger.warning(
                    f"DMQ subscription queue full ({_DEFAULT_QUEUE_MAXSIZE}), dropping reading for {reading.drf}"
                )

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

            sub.handle._stopped = True

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

        # Create subscription state
        sub = _SelectSubscription(
            sub_id=sub_id,
            drfs=drfs,
            drf_to_idx={d: i for i, d in enumerate(drfs)},
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
        handle._stopped = True
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
            sub.handle._stopped = True
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
            self._io_thread = None

        self._select_connection = None
        logger.info("All streaming stopped")

    def close(self) -> None:
        """Close the backend and release all resources."""
        if self._closed:
            return

        self._closed = True

        # Stop streaming and close SelectConnection
        # Write state (_write_sessions, _standby_channels) is cleared on the
        # IO thread during connection close to avoid cross-thread mutation.
        self.stop_streaming()

        # After IO thread has joined, safe to clear any remnants
        self._write_sessions.clear()
        self._standby_channels.clear()

        logger.info("DMQBackend closed")

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        n_subs = len(self._subscriptions)
        return f"DMQBackend({self._host}:{self._port}, subs={n_subs}, {status})"


__all__ = ["DMQBackend"]
