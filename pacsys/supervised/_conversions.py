"""Convert between Backend types (Reading, WriteResult) and proto messages.

Reuses _value_to_proto_value from grpc_backend.py for the server->proto direction.
"""

from google.protobuf import timestamp_pb2

from pacsys._proto.controls.common.v1 import status_pb2
from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2
from pacsys.backends.grpc_backend import _value_to_proto_value
from pacsys.types import Reading, WriteResult


def reading_to_proto_reply(reading: Reading, index: int) -> "DAQ_pb2.ReadingReply":  # type: ignore[unresolved-attribute]
    """Convert a Reading to a ReadingReply proto message.

    Readings without usable data (errors AND warnings-without-data) use the
    status oneof - matching real DPM gRPC server behavior. Only readings with
    actual values use the readings oneof.
    """
    reply = DAQ_pb2.ReadingReply()  # type: ignore[unresolved-attribute]
    reply.index = index

    if not reading.ok:
        reply.status.facility_code = reading.facility_code
        reply.status.status_code = reading.error_code
        if reading.message:
            reply.status.message = reading.message
        return reply

    # Success path: pack into Readings
    rd = DAQ_pb2.Reading()  # type: ignore[unresolved-attribute]
    if reading.timestamp is not None:
        ts = timestamp_pb2.Timestamp()
        ts.FromDatetime(reading.timestamp)
        rd.timestamp.CopyFrom(ts)
    if reading.value is not None:
        rd.data.CopyFrom(_value_to_proto_value(reading.value))
    rd.status.facility_code = reading.facility_code
    rd.status.status_code = reading.error_code
    if reading.message:
        rd.status.message = reading.message

    reply.readings.reading.append(rd)
    return reply


def write_result_to_proto_status(result: WriteResult) -> "status_pb2.Status":  # type: ignore[unresolved-attribute]
    """Convert a WriteResult to a Status proto message."""
    status = status_pb2.Status()  # type: ignore[unresolved-attribute]
    status.facility_code = result.facility_code
    status.status_code = result.error_code
    if result.message:
        status.message = result.message
    return status
