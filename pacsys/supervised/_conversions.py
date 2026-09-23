"""Convert between Backend types (Reading, WriteResult) and proto messages.

Reuses _value_to_proto_value from grpc_backend.py for the server->proto direction.
"""

import logging

import numpy as np

from pacsys._proto.controls.common.v1 import status_pb2
from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2
from pacsys.acnet.errors import ERR_RETRY, FACILITY_ACNET
from pacsys.backends.grpc_backend import _value_to_proto_value
from pacsys.types import Reading, ValueType, WriteResult

logger = logging.getLogger("pacsys.supervised")

_LOGGER_CHUNK_POINTS = 487  # DPM logger chunk size for scalar devices; array devices send one record per reply


def reading_to_proto_replies(
    reading: Reading, index: int, *, complete_history: bool = False
) -> "list[DAQ_pb2.ReadingReply]":
    """Convert a Reading to the ReadingReply messages the DPM gRPC server would send.

    Readings without usable data (errors AND warnings-without-data) use the
    status oneof. Timed arrays become one proto Reading per sample (per record
    for 2-D array-device data). ``complete_history`` marks a whole
    LOGGER/LOGGERDURATION result: it is sent in DPM-sized chunks followed by
    the empty terminator the client waits for. A reading that cannot be
    encoded becomes an error status reply for its index.
    """
    try:
        return _reading_to_replies(reading, index, complete_history)
    except (TypeError, ValueError, KeyError, OverflowError) as e:
        logger.error(
            "Cannot encode reading for %s (index=%d, value_type=%s): %s", reading.drf, index, reading.value_type, e
        )
        return [_status_reply(index, FACILITY_ACNET, ERR_RETRY, f"Cannot encode reading: {e}")]


def _reading_to_replies(reading: Reading, index: int, complete_history: bool) -> "list[DAQ_pb2.ReadingReply]":
    if not reading.ok:
        if reading.error_code == 0:  # no data; status 0 would read as success (or a logger terminator) to clients
            return [_status_reply(index, FACILITY_ACNET, ERR_RETRY, reading.message or "No data")]
        return [_status_reply(index, reading.facility_code, reading.error_code, reading.message)]

    value = reading.value
    if complete_history or (reading.value_type == ValueType.TIMED_SCALAR_ARRAY and isinstance(value, dict)):
        data, micros = _timed_samples(value)
        if not complete_history:
            return [_samples_reply(reading, index, data, micros)]
        step = 1 if data.ndim == 2 else _LOGGER_CHUNK_POINTS
        chunks = [
            _samples_reply(reading, index, data[i : i + step], micros[i : i + step]) for i in range(0, len(data), step)
        ]
        return [*chunks, _samples_reply(reading, index, data[:0], micros[:0])]

    reply = DAQ_pb2.ReadingReply(index=index)
    rd = reply.readings.reading.add()
    if reading.timestamp is not None:
        rd.timestamp.FromDatetime(reading.timestamp)
    if reading.value_type == ValueType.BASIC_STATUS and isinstance(value, dict):
        # DPM/gRPC status maps are device-specific display strings (e.g. {"On": "Yes"}), not alarm dicts
        rd.data.basicStatus.SetInParent()
        rd.data.basicStatus.value.update({str(k): str(v) for k, v in value.items()})
    elif value is not None:
        rd.data.CopyFrom(_value_to_proto_value(value))
    _set_status(rd.status, reading.facility_code, reading.error_code, reading.message)
    return [reply]


def _timed_samples(value) -> tuple[np.ndarray, np.ndarray]:
    """Split a TIMED_SCALAR_ARRAY value into (data, micros); data is 1-D or 2-D (records x elements)."""
    if isinstance(value, dict):
        data = np.asarray(value["data"], dtype=float)
        micros = np.asarray(value["micros"], dtype=np.int64)
    elif value is not None and np.size(value) == 0:  # empty logger window without micros
        return np.empty(0), np.empty(0, dtype=np.int64)
    else:
        raise ValueError(f"expected timed data dict, got {type(value).__name__}")
    if data.ndim not in (1, 2) or micros.ndim != 1 or len(data) != len(micros):
        raise ValueError(f"timed data shape {data.shape} does not match micros shape {micros.shape}")
    return data, micros


def _samples_reply(reading: Reading, index: int, data: np.ndarray, micros: np.ndarray) -> "DAQ_pb2.ReadingReply":
    """One reply with a proto Reading per sample; no samples yields the empty (terminator) reply."""
    reply = DAQ_pb2.ReadingReply(index=index)
    reply.readings.SetInParent()
    for sample, us in zip(data, micros):
        rd = reply.readings.reading.add()
        seconds, rem = divmod(int(us), 1_000_000)
        rd.timestamp.seconds, rd.timestamp.nanos = seconds, rem * 1_000
        if data.ndim == 2:
            rd.data.scalarArr.SetInParent()
            rd.data.scalarArr.value.extend(sample.tolist())
        else:
            rd.data.scalar = float(sample)
    if reply.readings.reading:
        _set_status(reply.readings.reading[0].status, reading.facility_code, reading.error_code, reading.message)
    return reply


def _status_reply(index: int, facility: int, error: int, message: str | None) -> "DAQ_pb2.ReadingReply":
    reply = DAQ_pb2.ReadingReply(index=index)
    reply.status.SetInParent()
    _set_status(reply.status, facility, error, message)
    return reply


def _set_status(status: "status_pb2.Status", facility: int, error: int, message: str | None) -> None:
    status.facility_code = facility
    status.status_code = error
    if message:
        status.message = message


def write_result_to_proto_status(result: WriteResult) -> "status_pb2.Status":
    """Convert a WriteResult to a Status proto message."""
    status = status_pb2.Status()
    _set_status(status, result.facility_code, result.error_code, result.message)
    return status
