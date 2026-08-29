"""
gRPC Backend - device access via DAQ gRPC service with JWT authentication.

Architecture: single reactor thread running grpc.aio on a dedicated asyncio loop.
Each subscribe() creates an asyncio.Task (not a thread). Bounded FIFO queues
prevent OOM. Auto-reconnection with exponential backoff for UNAVAILABLE errors.

Requires grpcio package. See specs/backends.md for protocol details.
"""

import asyncio
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, TypeGuard, cast

import numpy as np

from pacsys.acnet.errors import ERR_RETRY, ERR_TIMEOUT, FACILITY_ACNET, normalize_error_code
from pacsys.auth import Auth, JWTAuth
from pacsys.backends import Backend, summarize_drfs, validate_alarm_dict
from pacsys.backends._dispatch import CallbackDispatcher
from pacsys.backends._subscription import BufferedSubscriptionHandle
from pacsys.drf_utils import is_chunked_historical_drf, prepare_for_write
from pacsys.errors import AuthenticationError, DeviceError, ReadError
from pacsys.types import (
    BackendCapability,
    DispatchMode,
    ErrorCallback,
    Reading,
    ReadingCallback,
    SubscriptionHandle,
    Value,
    ValueType,
    WriteResult,
    _normalize_numpy_scalar,
    _validate_callback,
)

logger = logging.getLogger(__name__)

# Check if grpc and proto files are available
_import_error = ""
try:
    import grpc
    from google.protobuf import timestamp_pb2
    from grpc import aio as grpc_aio

    # Import generated proto files (common protos first -- DAQ depends on them)
    from pacsys._proto.controls.common.v1 import device_pb2, status_pb2
    from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2, DAQ_pb2_grpc

    GRPC_AVAILABLE = True
except (ImportError, TypeError) as e:
    GRPC_AVAILABLE = False
    grpc = cast("Any", None)
    grpc_aio = cast("Any", None)
    timestamp_pb2 = cast("Any", None)
    DAQ_pb2 = cast("Any", None)
    DAQ_pb2_grpc = cast("Any", None)
    device_pb2 = cast("Any", None)
    status_pb2 = cast("Any", None)
    _import_error = str(e)

DEFAULT_HOST = os.environ.get("PACSYS_GRPC_HOST", "dce08.fnal.gov")
DEFAULT_PORT = int(os.environ.get("PACSYS_GRPC_PORT", "50051"))
DEFAULT_TIMEOUT = 5.0

# Reconnection constants
_RECONNECT_INITIAL_DELAY = 1.0
_RECONNECT_MAX_DELAY = 30.0
_RECONNECT_BACKOFF_FACTOR = 2.0

# gRPC status codes the DPM server actually produces:
#   UNAVAILABLE  - server down / connection refused (gRPC transport)
#   CANCELLED    - client cancel or server shutdown mid-stream
#   DEADLINE_EXCEEDED - client-side timeout (handled separately in _grpc_error_code)
#   UNKNOWN      - unhandled Java exception in set() path only
# The DPM server never returns UNAUTHENTICATED, PERMISSION_DENIED, or
# INVALID_ARGUMENT - auth failures are ACNET errors inside ReadingReply.status.
_RETRYABLE_STATUS_CODES = (
    frozenset({grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED}) if GRPC_AVAILABLE else frozenset()
)


def _grpc_error_code(e: "grpc.aio.AioRpcError") -> int:
    """Map gRPC status to ACNET error code. DEADLINE_EXCEEDED → ERR_TIMEOUT, else ERR_RETRY."""
    if GRPC_AVAILABLE and e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        return ERR_TIMEOUT
    return ERR_RETRY


def _grpc_facility_code(e: "grpc.aio.AioRpcError") -> int:
    """Map gRPC status to ACNET facility code. DEADLINE_EXCEEDED → FACILITY_ACNET, else 0."""
    if GRPC_AVAILABLE and e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        return FACILITY_ACNET
    return 0


def _resolve_config(
    host: str | None, port: int | None, auth: Auth | None, timeout: float | None
) -> "tuple[str, int, JWTAuth | None, float]":
    """Apply defaults and validate constructor arguments (shared by the sync and async backends)."""
    host = host if host is not None else DEFAULT_HOST
    port = port if port is not None else DEFAULT_PORT
    timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
    if auth is not None and not isinstance(auth, JWTAuth):
        raise ValueError(f"auth must be JWTAuth or None, got {type(auth).__name__}")
    if not host:
        raise ValueError("host cannot be empty")
    if port <= 0:
        raise ValueError(f"port must be positive, got {port}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"timeout must be positive and finite, got {timeout}")
    return host, port, auth if auth is not None else JWTAuth.from_env(), timeout


def _value_to_proto_value(value: Value, *, for_write: bool = False) -> "device_pb2.Value":
    """Convert Python value to proto Value message."""
    proto_value = device_pb2.Value()
    value = _normalize_numpy_scalar(value)  # NumPy scalars (int64, bool_, ...) behave like their Python peers
    if isinstance(value, (list, tuple)):
        value = [_normalize_numpy_scalar(v) for v in value]

    if isinstance(value, float):
        proto_value.scalar = value
    elif isinstance(value, int):
        proto_value.scalar = float(value)
    elif isinstance(value, str):
        proto_value.text = value
    elif isinstance(value, bytes):
        proto_value.raw = value
    elif isinstance(value, dict):
        if for_write:
            raise NotImplementedError(
                "Alarm writes are not supported via gRPC -- the DPM server does not handle "
                "alarm settings over this protocol. Use the DPM/HTTP or DMQ backend instead."
            )
        if _is_basic_status_dict(value):
            for k, v in value.items():
                proto_value.basicStatus.value[k] = str(v)
        else:
            _dict_to_proto_alarm(value, proto_value)
    elif isinstance(value, (list, tuple)):
        if all(isinstance(v, (int, float)) for v in value):
            proto_value.scalarArr.value.extend([float(v) for v in value])
        elif all(isinstance(v, str) for v in value):
            proto_value.textArr.value.extend(v for v in value if isinstance(v, str))
        else:
            raise ValueError(f"Cannot convert mixed list to proto value: {value}")
    elif isinstance(value, np.ndarray):
        proto_value.scalarArr.value.extend(value.tolist())
    else:
        raise TypeError(f"Cannot convert value of type {type(value)} to proto value")

    return proto_value


_BASIC_STATUS_KEYS = frozenset({"on", "ready", "remote", "positive", "ramp"})


def _is_basic_status_dict(d: dict) -> TypeGuard[dict[str, bool]]:
    """True if dict looks like a basic status reading (bool values, status keys)."""
    return bool(d) and set(d.keys()) <= _BASIC_STATUS_KEYS and all(isinstance(v, bool) for v in d.values())


def _dict_to_proto_alarm(d: dict, proto_value: "device_pb2.Value") -> None:
    """Populate proto Value with an alarm dict (analog or digital).

    Requires at least one type-specific key (minimum/maximum for analog,
    nominal/mask for digital) to disambiguate alarm type.
    """
    alarm_type = validate_alarm_dict(d)
    if alarm_type == "analog":
        a = proto_value.anaAlarm
        if "minimum" in d:
            a.minimum = float(d["minimum"])
        if "maximum" in d:
            a.maximum = float(d["maximum"])
        if "alarm_enable" in d:
            a.alarmEnable = bool(d["alarm_enable"])
        if "abort_inhibit" in d:
            a.abortInhibit = bool(d["abort_inhibit"])
        if "tries_needed" in d:
            a.triesNeeded = int(d["tries_needed"])
    else:
        a = proto_value.digAlarm
        if "nominal" in d:
            a.nominal = int(d["nominal"])
        if "mask" in d:
            a.mask = int(d["mask"])
        if "alarm_enable" in d:
            a.alarmEnable = bool(d["alarm_enable"])
        if "abort_inhibit" in d:
            a.abortInhibit = bool(d["abort_inhibit"])
        if "tries_needed" in d:
            a.triesNeeded = int(d["tries_needed"])


def _proto_value_to_python(proto_value: "device_pb2.Value") -> tuple[Value, ValueType]:
    """Convert proto Value message to Python value."""
    value_type = proto_value.WhichOneof("value")

    if value_type == "scalar":
        return proto_value.scalar, ValueType.SCALAR
    if value_type == "scalarArr":
        values = list(proto_value.scalarArr.value)
        return np.array(values, dtype=float), ValueType.SCALAR_ARRAY
    if value_type == "raw":
        return proto_value.raw, ValueType.RAW
    if value_type == "text":
        return proto_value.text, ValueType.TEXT
    if value_type == "textArr":
        return list(proto_value.textArr.value), ValueType.TEXT_ARRAY
    if value_type == "anaAlarm":
        alarm = proto_value.anaAlarm
        return {
            "minimum": alarm.minimum,
            "maximum": alarm.maximum,
            "alarm_enable": alarm.alarmEnable,
            "alarm_status": alarm.alarmStatus,
            "abort": alarm.abort,
            "abort_inhibit": alarm.abortInhibit,
            "tries_needed": alarm.triesNeeded,
            "tries_now": alarm.triesNow,
        }, ValueType.ANALOG_ALARM
    if value_type == "digAlarm":
        alarm = proto_value.digAlarm
        return {
            "nominal": alarm.nominal,
            "mask": alarm.mask,
            "alarm_enable": alarm.alarmEnable,
            "alarm_status": alarm.alarmStatus,
            "abort": alarm.abort,
            "abort_inhibit": alarm.abortInhibit,
            "tries_needed": alarm.triesNeeded,
            "tries_now": alarm.triesNow,
        }, ValueType.DIGITAL_ALARM
    if value_type == "basicStatus":
        return {k: v == "True" for k, v in proto_value.basicStatus.value.items()}, ValueType.BASIC_STATUS
    if value_type is None:
        raise ValueError("Proto Value has no value set (empty oneof)")
    raise ValueError(f"Unknown proto value type: {value_type!r}")


def _proto_timestamp_to_datetime(ts: "timestamp_pb2.Timestamp") -> datetime | None:
    """Convert proto Timestamp to Python datetime.

    NOTE: Python datetime only supports microsecond precision, so nanoseconds
    from the proto are truncated.  The return type may change in the future to
    preserve full nanosecond fidelity.
    """
    if ts is None:
        return None
    try:
        seconds = ts.seconds
        nanos = ts.nanos
        if seconds == 0 and nanos == 0:
            return None
        return datetime.fromtimestamp(seconds + nanos / 1e9, tz=timezone.utc)
    except (ValueError, OSError, AttributeError):
        return None


def _proto_status_to_codes(status: "status_pb2.Status") -> tuple[int, int, str | None]:
    """Extract status codes from proto Status message."""
    if status is None:
        return 0, 0, None
    return (
        status.facility_code,
        normalize_error_code(status.status_code),
        status.message if status.message else None,
    )


def _reply_to_readings(reply, drfs: list[str]) -> list[Reading]:
    """Convert a ReadingReply to a list of Readings.

    The server may pack multiple (value, timestamp) pairs into a single
    Readings message for high-frequency buffered data (see
    DPMListGRPC.sendReply(WhatDaq, double[], long[], long)).
    """
    index = reply.index
    if index < 0 or index >= len(drfs):
        logger.warning("Received reply for unknown index %s", index)
        return []

    drf = drfs[index]
    now = datetime.now(timezone.utc)
    value_field = reply.WhichOneof("value")

    if value_field == "status":
        # DPM routes nonzero device statuses through receiveStatus(), so this
        # top-level status reply is mutually exclusive with a data payload.
        facility, error, message = _proto_status_to_codes(reply.status)
        # Status-only reply with code 0 means "success but no data" which is
        # an error for a data acquisition call (downstream expects a value).
        return [
            Reading(
                drf=drf,
                facility_code=facility,
                error_code=error if error != 0 else ERR_RETRY,
                message=message or "gRPC error",
                timestamp=now,
            )
        ]
    if value_field == "readings":
        reading_list = reply.readings.reading
        results = []
        for rd in reading_list:
            ts = _proto_timestamp_to_datetime(rd.timestamp)
            facility, error, message = _proto_status_to_codes(rd.status)
            if error != 0:
                results.append(
                    Reading(
                        drf=drf,
                        facility_code=facility,
                        error_code=error,
                        message=message,
                        timestamp=ts or now,
                    )
                )
                continue

            try:
                value, value_type = _proto_value_to_python(rd.data)
            except ValueError as e:
                # If there's no data and it wasn't an error, it's malformed
                results.append(
                    Reading(
                        drf=drf,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_RETRY,
                        message=str(e),
                        timestamp=ts or now,
                    )
                )
                continue

            results.append(
                Reading(
                    drf=drf,
                    value_type=value_type,
                    value=value,
                    facility_code=facility,
                    error_code=error,
                    message=message,
                    timestamp=ts or now,
                )
            )
        return results
    return []


def _aggregate_proto_readings(reading_list, drf: str, now: datetime) -> Reading:
    """Aggregate proto readings directly into a TIMED_SCALAR_ARRAY.

    Skips intermediate Reading objects - extracts values and timestamps
    straight from the proto messages to avoid N datetime round-trips
    and N throwaway allocations.

    If any reading carries a nonzero error status, the first error is
    propagated immediately instead of silently mixing errors into data.
    """
    # Check for error status in samples before aggregating
    for rd in reading_list:
        if rd.status is not None:
            facility, error, message = _proto_status_to_codes(rd.status)
            if error != 0:
                ts = _proto_timestamp_to_datetime(rd.timestamp) or now
                return Reading(
                    drf=drf,
                    facility_code=facility,
                    error_code=error,
                    message=message,
                    timestamp=ts,
                )

    try:
        data = np.array([_proto_value_to_python(rd.data)[0] for rd in reading_list], dtype=float)
    except (TypeError, ValueError) as e:
        return Reading(drf=drf, facility_code=FACILITY_ACNET, error_code=ERR_RETRY, message=str(e), timestamp=now)
    micros = np.array(
        [
            rd.timestamp.seconds * 1_000_000 + rd.timestamp.nanos // 1_000
            if rd.timestamp.seconds or rd.timestamp.nanos
            else 0
            for rd in reading_list
        ],
        dtype=np.int64,
    )
    ts = _proto_timestamp_to_datetime(reading_list[0].timestamp) or now
    return Reading(
        drf=drf,
        value_type=ValueType.TIMED_SCALAR_ARRAY,
        value={"data": data, "micros": micros},
        timestamp=ts,
    )


def _merge_logger_readings(chunks: list[Reading], drf: str) -> Reading:
    """Merge multiple logger chunk Readings into a single TIMED_SCALAR_ARRAY."""
    # Propagate error readings immediately instead of silently dropping them
    for r in chunks:
        if r.error_code and r.error_code != 0:
            return r

    all_data: list[np.ndarray] = []
    all_micros: list[np.ndarray] = []
    first_ts = chunks[0].timestamp if chunks else None

    for r in chunks:
        if isinstance(r.value, dict):
            all_data.append(r.value["data"])
            all_micros.append(r.value["micros"])
        elif isinstance(r.value, np.ndarray):
            all_data.append(r.value)

    data = np.concatenate(all_data) if all_data else np.array([], dtype=float)
    if all_micros:
        micros = np.concatenate(all_micros)
        return Reading(
            drf=drf, value_type=ValueType.TIMED_SCALAR_ARRAY, value={"data": data, "micros": micros}, timestamp=first_ts
        )
    return Reading(drf=drf, value_type=ValueType.SCALAR_ARRAY, value=data, timestamp=first_ts)


# ─────────────────────────────────────────────────────────────────────────────
# Async Core -- all gRPC I/O lives here
# ─────────────────────────────────────────────────────────────────────────────


class _DaqCore:
    """Pure-async gRPC logic. Owns the aio channel and stub."""

    def __init__(self, host: str, port: int, auth: JWTAuth | None, timeout: float):
        self._host = host
        self._port = port
        self._auth = auth
        self._timeout = timeout
        self._channel: grpc_aio.Channel | None = None
        self._stub: DAQ_pb2_grpc.DAQStub | None = None

    async def connect(self):
        target = f"{self._host}:{self._port}"
        options = [
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
        ]
        self._channel = grpc_aio.insecure_channel(target, options=options)
        self._stub = DAQ_pb2_grpc.DAQStub(self._channel)
        logger.debug("Created async gRPC channel to %s", target)

    async def close(self):
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
            logger.debug("Async gRPC channel closed")

    def _metadata(self) -> list[tuple[str, str]] | None:
        if self._auth is not None:
            return [("authorization", f"Bearer {self._auth.token}")]
        return None

    async def read_many(self, drfs: list[str], timeout: float) -> list[Reading]:
        assert self._stub is not None, "Not connected"
        request = DAQ_pb2.ReadingList()
        for drf in drfs:
            request.drf.append(drf)

        logger.debug("gRPC async Read request: %s devices", len(drfs))

        # Chunked logger DRFs arrive in 487-point chunks with a final empty chunk.
        chunked_logger_indices = {i for i, drf in enumerate(drfs) if is_chunked_historical_drf(drf)}
        logger_chunks: dict[int, list[Reading]] = {}  # index -> accumulated chunks
        logger_complete: set[int] = set()  # indices that received the terminator

        results = cast("list[Reading | None]", [None] * len(drfs))
        received_count = 0
        expected_count = len(drfs)
        now = datetime.now(timezone.utc)
        transport_error: BaseException | None = None

        call = None
        try:
            call = self._stub.Read(
                request,
                timeout=timeout,
                metadata=self._metadata(),
            )

            async for reply in call:
                if reply is None:
                    logger.warning(
                        "gRPC stream yielded None (received=%s/%s), missing=[%s]",
                        received_count,
                        expected_count,
                        ", ".join(drfs[i] for i in range(len(drfs)) if results[i] is None),
                    )
                    continue

                index = reply.index
                if index < 0 or index >= len(drfs):
                    continue

                value_field = reply.WhichOneof("value")

                if index in chunked_logger_indices:
                    # Logger: accumulate chunks; empty readings = done
                    if (
                        value_field == "status"
                        and index not in logger_complete
                        and (codes := _proto_status_to_codes(reply.status))[1] != 0
                    ):
                        # Real error (bad device, auth failure, ...) — not an
                        # end-of-stream terminator (mirrors _dpm_core Status_reply handling)
                        facility, error, message = codes
                        results[index] = Reading(
                            drf=drfs[index],
                            facility_code=facility,
                            error_code=error,
                            message=message or "gRPC error",
                            timestamp=now,
                        )
                        logger_complete.add(index)
                        received_count += 1
                    elif value_field == "readings" and len(reply.readings.reading) > 0:
                        chunk = _aggregate_proto_readings(reply.readings.reading, drfs[index], now)
                        logger_chunks.setdefault(index, []).append(chunk)
                    elif index not in logger_complete:
                        # Empty or non-readings reply = completion signal (guard against double-count)
                        logger_complete.add(index)
                        chunks = logger_chunks.get(index, [])
                        if chunks:
                            results[index] = _merge_logger_readings(chunks, drfs[index])
                        else:
                            # Empty window — valid empty result
                            results[index] = Reading(
                                drf=drfs[index],
                                value_type=ValueType.TIMED_SCALAR_ARRAY,
                                value={"data": np.array([], dtype=float), "micros": np.array([], dtype=np.int64)},
                                timestamp=now,
                            )
                        received_count += 1
                elif results[index] is None:
                    if value_field == "readings" and len(reply.readings.reading) > 1:
                        results[index] = _aggregate_proto_readings(reply.readings.reading, drfs[index], now)
                        received_count += 1
                    else:
                        readings = _reply_to_readings(reply, drfs)
                        if readings:
                            results[index] = readings[0]
                            received_count += 1

                if received_count >= expected_count:
                    call.cancel()
                    break

        except grpc.aio.AioRpcError as e:
            transport_error = e
            target = f"{self._host}:{self._port}"
            error_message = f"gRPC error ({target}): {e.code().name}: {e.details()}"
            missing = [drfs[i] for i in range(len(drfs)) if results[i] is None]
            logger.error(
                "%s (received %s/%s, missing: %s)",
                error_message,
                received_count,
                expected_count,
                missing,
            )
            ec = _grpc_error_code(e)
            fc = _grpc_facility_code(e)
            for i in range(len(drfs)):
                if results[i] is None:
                    results[i] = Reading(
                        drf=drfs[i],
                        facility_code=fc,
                        error_code=ec,
                        message=error_message,
                        timestamp=now,
                    )

        except Exception as e:  # noqa: BLE001
            transport_error = e
            error_message = f"gRPC error ({self._host}:{self._port}): {type(e).__name__}: {e}"
            logger.error("%s (received %s/%s)", error_message, received_count, expected_count)
            if call is not None:
                call.cancel()
            for i in range(len(drfs)):
                if results[i] is None:
                    results[i] = Reading(drf=drfs[i], error_code=ERR_RETRY, message=error_message, timestamp=now)

        # Backfill missing (stream ended without responses for some devices)
        has_missing = False
        for i in range(len(drfs)):
            if results[i] is None:
                has_missing = True
                results[i] = Reading(
                    drf=drfs[i],
                    error_code=ERR_RETRY,
                    message="No response received",
                    timestamp=now,
                )

        complete_results: list[Reading] = []
        for result in results:
            if result is None:
                raise RuntimeError("gRPC read result backfill left an empty slot")
            complete_results.append(result)

        if transport_error is not None or has_missing:
            raise ReadError(complete_results, str(transport_error or "Incomplete response")) from transport_error

        return complete_results

    async def write_many(self, settings: list[tuple[str, Value]], timeout: float) -> list[WriteResult]:
        assert self._stub is not None, "Not connected"
        # Phase 1: Validate
        valid_items: list[tuple[int, str, DAQ_pb2.Setting]] = []
        validation_errors: dict[int, str] = {}

        for i, (drf, value) in enumerate(settings):
            try:
                setting = DAQ_pb2.Setting()
                setting.device = drf
                setting.value.CopyFrom(_value_to_proto_value(value, for_write=True))
                valid_items.append((i, drf, setting))
            except (TypeError, ValueError, NotImplementedError) as e:
                logger.error("Failed to convert value for %s: %s", drf, e)
                validation_errors[i] = str(e)

        # Phase 2: RPC
        rpc_results: dict[int, WriteResult] = {}

        if valid_items:
            request = DAQ_pb2.SettingList()
            for _, _, proto_setting in valid_items:
                request.setting.append(proto_setting)

            logger.debug("gRPC async Set request: %s devices (of %s total)", len(valid_items), len(settings))

            try:
                response = await self._stub.Set(
                    request,
                    timeout=timeout,
                    metadata=self._metadata(),
                )

                status_list = response.status

                for rpc_idx, (orig_idx, drf, _) in enumerate(valid_items):
                    if rpc_idx < len(status_list):
                        facility, error, message = _proto_status_to_codes(status_list[rpc_idx])
                        rpc_results[orig_idx] = WriteResult(
                            drf=drf,
                            facility_code=facility,
                            error_code=error,
                            message=message,
                        )
                    else:
                        rpc_results[orig_idx] = WriteResult(
                            drf=drf,
                            error_code=ERR_RETRY,
                            message="No status received from server",
                        )

            except grpc.aio.AioRpcError as e:
                target = f"{self._host}:{self._port}"
                error_message = f"gRPC error ({target}): {e.code().name}: {e.details()}"
                logger.error(error_message)
                ec = _grpc_error_code(e)
                fc = _grpc_facility_code(e)
                for orig_idx, drf, _ in valid_items:
                    rpc_results[orig_idx] = WriteResult(drf=drf, facility_code=fc, error_code=ec, message=error_message)

            except Exception as e:  # noqa: BLE001
                error_message = f"gRPC error ({self._host}:{self._port}): {type(e).__name__}: {e}"
                logger.error(error_message)
                for orig_idx, drf, _ in valid_items:
                    rpc_results[orig_idx] = WriteResult(drf=drf, error_code=ERR_RETRY, message=error_message)

        # Phase 3: Merge in original order
        results = []
        for i, (drf, _) in enumerate(settings):
            if i in validation_errors:
                results.append(WriteResult(drf=drf, error_code=ERR_RETRY, message=validation_errors[i]))
            elif i in rpc_results:
                results.append(rpc_results[i])
            else:
                results.append(WriteResult(drf=drf, error_code=ERR_RETRY, message="Internal error: no result"))

        return results

    async def stream(
        self,
        drfs: list[str],
        dispatch_fn,
        stop_check,
        error_fn,
    ):
        """Long-running stream with reconnection on errors.

        Normal stream completion (server onCompleted) is treated as a graceful
        end -- the subscription stops without reconnecting. This is the correct
        behavior for @I (immediate) events and also safe for periodic events
        since the server never calls onCompleted on periodic streams. Network
        errors and UNAVAILABLE trigger exponential backoff reconnection.
        """
        assert self._stub is not None, "Not connected"
        backoff = _RECONNECT_INITIAL_DELAY

        while not stop_check():
            try:
                request = DAQ_pb2.ReadingList()
                for drf in drfs:
                    request.drf.append(drf)

                call = self._stub.Read(
                    request,
                    timeout=None,
                    metadata=self._metadata(),
                )

                stream_start = time.monotonic()

                async for reply in call:
                    if stop_check():
                        call.cancel()
                        return

                    for reading in _reply_to_readings(reply, drfs):
                        dispatch_fn(reading)

                    # Only reset backoff after sustained healthy streaming to
                    # prevent flapping connections from retrying at min interval.
                    if time.monotonic() - stream_start >= _RECONNECT_MAX_DELAY:
                        backoff = _RECONNECT_INITIAL_DELAY

                # Stream ended normally (server called onCompleted). This
                # happens for @I events and on graceful server shutdown.
                # Do NOT reconnect -- treat as subscription end.
                if not stop_check():
                    logger.info("gRPC stream completed normally, subscription ending")
                return

            except asyncio.CancelledError:
                return

            except grpc.aio.AioRpcError as e:
                if stop_check():
                    return
                target = f"{self._host}:{self._port}"
                code = e.code()
                ec = _grpc_error_code(e)
                fc = _grpc_facility_code(e)
                drf_summary = summarize_drfs(drfs)
                exc = DeviceError(
                    drf=drfs[0] if drfs else "?",
                    facility_code=fc,
                    error_code=ec,
                    message=f"gRPC stream error ({target}): {code.name}: {e.details()} (devices: {drf_summary})",
                )
                if code in _RETRYABLE_STATUS_CODES:
                    error_fn(exc, fatal=False)
                    logger.warning(
                        "gRPC stream %s (%s), retrying in %.1fs: %s",
                        code.name,
                        target,
                        backoff,
                        e.details(),
                    )
                else:
                    error_fn(exc, fatal=True)
                    logger.error("gRPC stream %s (%s), non-retryable: %s", code.name, target, e.details())
                    return

            except Exception as e:  # noqa: BLE001
                if stop_check():
                    return
                target = f"{self._host}:{self._port}"
                drf_summary = summarize_drfs(drfs)
                exc = DeviceError(
                    drf=drfs[0] if drfs else "?",
                    facility_code=FACILITY_ACNET,
                    error_code=ERR_RETRY,
                    message=f"gRPC stream error ({target}): {type(e).__name__}: {e} (devices: {drf_summary})",
                )
                error_fn(exc, fatal=True)
                logger.error("gRPC stream error (%s): %s", target, e)
                return

            # Backoff before retry
            if stop_check():
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * _RECONNECT_BACKOFF_FACTOR, _RECONNECT_MAX_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# Subscription Handle
# ─────────────────────────────────────────────────────────────────────────────


class _GRPCSubscriptionHandle(BufferedSubscriptionHandle):
    """Concrete SubscriptionHandle for the async GRPCBackend."""

    def __init__(
        self,
        backend: "GRPCBackend",
        drfs: list[str],
        callback: ReadingCallback | None,
        on_error: ErrorCallback | None,
    ):
        super().__init__()
        self._backend = backend
        self._drfs = drfs
        self._callback = callback
        self._on_error = on_error
        self._is_callback_mode = callback is not None
        self._ref_ids = list(range(len(drfs)))
        self._task: asyncio.Task | None = None

    def _dispatch(self, reading: Reading) -> None:
        """Called from the reactor thread to deliver a reading."""
        if self._stopped:
            return
        if self._callback is not None:
            self._backend._dispatcher.dispatch_reading(self._callback, reading, self)
        else:
            super()._dispatch(reading)

    def _dispatch_error(self, exc: Exception, *, fatal: bool) -> None:
        """Called from the reactor thread on stream error.

        Only fatal errors stop the iterator - retryable errors are handled by
        the stream's reconnection loop and should not terminate consumption.
        """
        if fatal:
            self._signal_error(exc)
        if self._on_error is not None:
            self._backend._dispatcher.dispatch_error(self._on_error, exc, self)

    def stop(self) -> None:
        if not self._stopped:
            self._backend.remove(self)


# ─────────────────────────────────────────────────────────────────────────────
# GRPCBackend -- sync facade over the async reactor
# ─────────────────────────────────────────────────────────────────────────────


class GRPCBackend(Backend):
    """
    gRPC backend for device access with JWT authentication.

    Uses a single reactor thread with grpc.aio for all I/O. Each subscribe()
    creates an asyncio.Task on the shared event loop (not a new thread).
    Bounded FIFO queues prevent OOM. Auto-reconnection with exponential
    backoff for UNAVAILABLE errors.

    Capabilities:
        - READ: Always enabled
        - STREAM: Always enabled (gRPC streaming)
        - BATCH: Always enabled (multiple devices in one call)
        - WRITE: Only when JWTAuth is provided
        - AUTH_JWT: Only when JWTAuth is provided

    Example (read-only):
        with GRPCBackend() as backend:
            temp = backend.read("M:OUTTMP")

    Example (with JWT):
        auth = JWTAuth(token="eyJ...")
        with GRPCBackend(auth=auth) as backend:
            print(f"Authenticated as: {backend.principal}")
            result = backend.write("M:OUTTMP", 72.5)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        auth: Auth | None = None,
        timeout: float | None = None,
        dispatch_mode: DispatchMode = DispatchMode.WORKER,
    ):
        if not GRPC_AVAILABLE:
            raise ImportError(
                f"grpc package and proto files required for GRPCBackend. "
                f"Install with: pip install grpcio grpcio-tools. Error: {_import_error}"
            )

        self._host, self._port, self._auth, self._timeout = _resolve_config(host, port, auth, timeout)

        # Reactor state -- all lazy
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reactor_thread: threading.Thread | None = None
        self._core: _DaqCore | None = None
        self._closed = False
        self._reactor_lock = threading.Lock()

        # Callback dispatcher
        self._dispatch_mode = dispatch_mode
        self._dispatcher = CallbackDispatcher(dispatch_mode)

        # Tracked subscriptions
        self._handles: list[_GRPCSubscriptionHandle] = []
        self._handles_lock = threading.Lock()

        logger.debug(
            "GRPCBackend initialized: host=%s, port=%s, authenticated=%s",
            self._host,
            self._port,
            self.authenticated,
        )

    # ── Reactor lifecycle ─────────────────────────────────────────────────

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

        self._reactor_thread = threading.Thread(target=_run, name="GRPCBackend-Reactor", daemon=True)
        self._reactor_thread.start()
        ready.wait(timeout=5.0)
        if not loop_holder:
            raise RuntimeError("GRPCBackend: failed to start reactor event loop")
        self._loop = loop_holder[0]

    def _ensure_reactor(self) -> None:
        """Lazily start reactor and connect core."""
        self._check_not_reactor_thread()
        if self._core is not None:
            return
        with self._reactor_lock:
            if self._core is not None:
                return
            if self._closed:
                raise RuntimeError("Backend is closed")
            self._start_reactor()
            assert self._loop is not None, "Reactor loop not initialized"
            core = _DaqCore(self._host, self._port, self._auth, self._timeout)
            try:
                fut = asyncio.run_coroutine_threadsafe(core.connect(), self._loop)
                fut.result(timeout=self._timeout)
            except Exception:
                # Connect failed - tear down reactor so next call retries cleanly
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._reactor_thread is not None:
                    self._reactor_thread.join(timeout=2.0)
                    if not self._reactor_thread.is_alive():
                        self._reactor_thread = None
                    else:
                        logger.warning("Reactor thread did not stop within 2s")
                self._loop = None
                raise
            self._core = core

    def _check_not_reactor_thread(self) -> None:
        """Raise if called from the reactor thread to prevent deadlock."""
        if threading.current_thread() is getattr(self, "_reactor_thread", None):
            raise RuntimeError(
                "Cannot call blocking backend methods from the reactor thread "
                "(e.g. from a DIRECT mode streaming callback). "
                "Use DispatchMode.WORKER or offload to another thread."
            )

    def _run_sync(self, coro, timeout: float | None = None):
        """Bridge sync → async: submit coroutine to reactor and wait."""
        self._ensure_reactor()
        assert self._loop is not None, "Reactor loop not initialized"
        effective_timeout = timeout if timeout is not None else self._timeout
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=effective_timeout + 1.0)
        except Exception:
            fut.cancel()
            raise

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def capabilities(self) -> BackendCapability:
        caps = BackendCapability.READ | BackendCapability.STREAM | BackendCapability.BATCH
        if self._auth is not None:
            caps |= BackendCapability.WRITE | BackendCapability.AUTH_JWT
        return caps

    @property
    def authenticated(self) -> bool:
        return self._auth is not None

    @property
    def principal(self) -> str | None:
        if self._auth is not None:
            return self._auth.principal
        return None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def timeout(self) -> float:
        return self._timeout

    # ── Read methods ──────────────────────────────────────────────────────

    def read(self, drf: str, timeout: float | None = None) -> Value:
        reading = self.get(drf, timeout=timeout)
        if not reading.ok:
            raise DeviceError(
                drf=reading.drf,
                facility_code=reading.facility_code,
                error_code=reading.error_code,
                message=reading.message
                or f"Read failed (facility={reading.facility_code}, error={reading.error_code})",
            )
        assert reading.value is not None
        return reading.value

    def get(self, drf: str, timeout: float | None = None) -> Reading:
        readings = self.get_many([drf], timeout=timeout)
        return readings[0]

    def get_many(self, drfs: list[str], timeout: float | None = None) -> list[Reading]:
        if self._closed:
            raise RuntimeError("Backend is closed")
        if not drfs:
            return []
        effective_timeout = timeout if timeout is not None else self._timeout
        self._ensure_reactor()
        assert self._core is not None
        return self._run_sync(self._core.read_many(drfs, effective_timeout), timeout=effective_timeout)

    # ── Write methods ─────────────────────────────────────────────────────

    def write(
        self,
        drf: str,
        value: Value,
        timeout: float | None = None,
    ) -> WriteResult:
        results = self.write_many([(drf, value)], timeout=timeout)
        return results[0]

    def write_many(
        self,
        settings: list[tuple[str, Value]],
        timeout: float | None = None,
    ) -> list[WriteResult]:
        if self._closed:
            raise RuntimeError("Backend is closed")
        if not settings:
            return []
        if self._auth is None:
            raise AuthenticationError(
                "JWTAuth required for write operations. "
                "Provide auth=JWTAuth(token=...) or set PACSYS_JWT_TOKEN environment variable."
            )
        prepared_settings = [(prepare_for_write(drf), value) for drf, value in settings]
        effective_timeout = timeout if timeout is not None else self._timeout
        self._ensure_reactor()
        assert self._core is not None
        return self._run_sync(self._core.write_many(prepared_settings, effective_timeout), timeout=effective_timeout)

    # ── Streaming ─────────────────────────────────────────────────────────

    def subscribe(
        self,
        drfs: list[str],
        callback: ReadingCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> SubscriptionHandle:
        """Subscribe to devices for streaming data.

        Creates an asyncio.Task on the shared reactor loop. The handle uses
        a bounded FIFO queue (default 10000); on overflow, newest readings
        are dropped with a warning.

        Note: data may be dropped at two independent levels under high load:
        1. Server-side: DPM drops readings when gRPC flow control signals the
           client is slow (no notification to client).
        2. Client-side: queue overflow drops newest readings with a log warning.
        Monitor logs for 'queue full' warnings. Use callbacks for lowest-latency
        consumption, or increase consumer throughput.
        """
        if not GRPC_AVAILABLE:
            raise ImportError("grpc package required for streaming")
        if not drfs:
            raise ValueError("drfs cannot be empty")
        if self._closed:
            raise RuntimeError("Backend is closed")
        _validate_callback(callback, on_error)

        self._ensure_reactor()
        assert self._loop is not None, "Reactor loop not initialized"
        assert self._core is not None
        core = self._core

        handle = _GRPCSubscriptionHandle(
            backend=self,
            drfs=drfs,
            callback=callback,
            on_error=on_error,
        )

        # Create the streaming task on the reactor loop
        async def _create_task():
            async def _run_stream():
                try:
                    await core.stream(
                        drfs,
                        handle._dispatch,
                        lambda: handle._stopped,
                        handle._dispatch_error,
                    )
                finally:
                    handle._signal_stop()
                    with self._handles_lock:
                        if handle in self._handles:
                            self._handles.remove(handle)

            return asyncio.ensure_future(_run_stream())

        with self._handles_lock:
            if self._closed:
                raise RuntimeError("Backend is closed")
            self._handles.append(handle)

        fut = asyncio.run_coroutine_threadsafe(_create_task(), self._loop)
        try:
            handle._task = fut.result(timeout=5.0)
        except Exception:
            fut.cancel()
            handle._signal_stop()
            with self._handles_lock:
                if handle in self._handles:
                    self._handles.remove(handle)
            raise

        mode_str = "callback" if handle._is_callback_mode else "iterator"
        logger.info("Created %s subscription for %s devices via gRPC", mode_str, len(drfs))
        return handle

    def remove(self, handle: SubscriptionHandle) -> None:
        """Remove a subscription. Cancels the associated async task."""
        if not isinstance(handle, _GRPCSubscriptionHandle):
            raise TypeError(f"Expected _GRPCSubscriptionHandle, got {type(handle).__name__}")

        handle._stop_requested = True
        handle._signal_stop()

        if handle._task is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(handle._task.cancel)

        with self._handles_lock:
            if handle in self._handles:
                self._handles.remove(handle)

        logger.info("Removed gRPC subscription for %s devices", len(handle._drfs))

    def stop_streaming(self) -> None:
        """Stop all streaming subscriptions."""
        with self._handles_lock:
            handles = list(self._handles)
            self._handles.clear()

        for handle in handles:
            handle._stop_requested = True
            handle._signal_stop()
            if handle._task is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(handle._task.cancel)

        logger.info("gRPC streaming stopped")

    def close(self) -> None:
        """Close the backend: stop streams, close core, stop reactor."""
        with self._reactor_lock:
            if self._closed:
                return
            self._closed = True

        self.stop_streaming()
        self._dispatcher.close()

        # Close the async core
        if self._core is not None and self._loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._core.close(), self._loop)
                fut.result(timeout=2.0)
            except Exception as e:  # noqa: BLE001
                logger.debug("Error closing gRPC core: %s", e)
            self._core = None

        # Stop the event loop and join reactor thread
        # Note: set _loop to None AFTER join so reactor cleanup can finish
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

        logger.debug("GRPCBackend closed")

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        auth = f", authenticated as {self.principal}" if self._auth else ""
        return f"GRPCBackend({self._host}:{self._port}, {status}{auth})"


__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "DEFAULT_TIMEOUT", "GRPC_AVAILABLE", "GRPCBackend"]
