"""RETDAT raw acquisition using explicit 16-bit frequency-time descriptors.

See ref_docs/MIP RETDAT Protocol Notes.html and ref_docs/dataserv.txt. The FTD
is transmitted unchanged; frontend-specific legacy flags are never reinterpreted.
"""

import struct
from collections.abc import Sequence
from dataclasses import dataclass

from ._read import (
    Connection,
    ReadDevice,
    ReadStream,
    ReadValue,
    _devices,
    _first,
    _message_limits,
    _message_size,
    _parse_values,
    _reply_size,
    _timeout,
    _uint,
)
from .constants import MAX_ACNET_MESSAGE_SIZE

__all__ = ["RetdatClient", "RetdatReply", "ReadDevice", "ReadValue", "ReadStream", "build_request", "parse_reply"]


@dataclass(frozen=True)
class RetdatReply:
    values: tuple[ReadValue, ...]
    acnet_status: int | None = None
    received_at_ns: int | None = None


def build_request(
    devices: Sequence[ReadDevice],
    ftd: int = 0,
    *,
    max_request_size: int = MAX_ACNET_MESSAGE_SIZE,
    max_reply_size: int = MAX_ACNET_MESSAGE_SIZE,
) -> bytes:
    """Encode a request with unsigned 16-bit lengths, offsets, and FTD.

    Standard FTDs: 0 = immediate, periodic values specify 60 Hz ticks,
    0x8000 | event_number selects a clock event. The ACNET multiple-reply flag
    is separate from the FTD and is chosen by read() versus stream().
    """
    _message_limits(max_request_size, max_reply_size)
    _uint(ftd, 0xFFFF, "ftd")
    devices = _devices(devices, 0xFFFF)
    size = _reply_size(devices)
    _message_size(size, max_reply_size, "reply")
    _message_size(6 + 16 * len(devices), max_request_size, "request")
    return struct.pack("<HHH", size, len(devices), ftd) + b"".join(
        struct.pack("<I8sHH", d.dipi, d.ssdn, d.length, d.offset) for d in devices
    )


def parse_reply(
    data: bytes,
    devices: Sequence[ReadDevice],
    *,
    acnet_status: int | None = None,
    received_at_ns: int | None = None,
) -> RetdatReply:
    """Decode per-device statuses and padded wire data; RETDAT has no timestamps."""
    devices = _devices(devices, 0xFFFF)
    return RetdatReply(_parse_values(data, devices, 0), acnet_status, received_at_ns)


class RetdatClient:
    """Raw RETDAT requests over an already-connected ACNET TCP/UDP connection.

    max_request_size/max_reply_size set destination payload limits in bytes,
    including RETDAT headers and padding, excluding the ACNET header.
    """

    def __init__(
        self,
        connection: Connection,
        *,
        max_request_size: int = MAX_ACNET_MESSAGE_SIZE,
        max_reply_size: int = MAX_ACNET_MESSAGE_SIZE,
    ):
        _message_limits(max_request_size, max_reply_size)
        if connection is None:
            raise TypeError("connection is required")
        self._connection = connection
        self._max_request_size = max_request_size
        self._max_reply_size = max_reply_size

    def stream(
        self,
        node: int,
        devices: Sequence[ReadDevice],
        *,
        ftd: int,
        buffer_size: int = 256,
    ) -> ReadStream[RetdatReply]:
        """Start a multiple-reply request with the exact supplied FTD."""
        return self._request(node, devices, ftd, True, buffer_size)

    def _request(
        self,
        node: int,
        devices: Sequence[ReadDevice],
        ftd: int,
        repetitive: bool,
        buffer_size: int,
    ) -> ReadStream[RetdatReply]:
        devices = _devices(devices, 0xFFFF)
        payload = build_request(
            devices,
            ftd,
            max_request_size=self._max_request_size,
            max_reply_size=self._max_reply_size,
        )
        return ReadStream(
            self._connection,
            node,
            "RETDAT",
            payload,
            lambda reply, received: parse_reply(
                reply.data, devices, acnet_status=reply.status, received_at_ns=received
            ),
            repetitive=repetitive,
            buffer_size=buffer_size,
        )

    def read(
        self,
        node: int,
        devices: Sequence[ReadDevice],
        *,
        ftd: int = 0,
        timeout: float = 1.0,
    ) -> RetdatReply:
        """Request one batch, then cancel. FTD may also select a periodic/clock event.

        timeout bounds waiting for data after ACNET request setup. The connection
        retains its own command/ACK timeout. Per-device statuses are returned.
        """
        _timeout(timeout)
        return _first(self._request(node, devices, ftd, False, 1), timeout)
