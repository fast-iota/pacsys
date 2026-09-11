"""GETS32 raw acquisition: wire codecs, explicit events, and synchronous requests.

See ref_docs/gets32.pdf and the DPM DaqSendTransaction32 reference. Data remains
in frontend wire representation; there is no device lookup or scaling here.
"""

import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..drf3.event import ClockEvent, ImmediateEvent, NeverEvent, PeriodicEvent, StateEvent, parse_event
from ..drf_utils import get_device_name
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

__all__ = [
    "Gets32Client",
    "Gets32Event",
    "Gets32Header",
    "Gets32Reply",
    "ReadDevice",
    "ReadValue",
    "ReadStream",
    "build_request",
    "parse_reply",
]


@dataclass(frozen=True)
class Gets32Event:
    """Explicit wire event, including legacy/frontend-specific acquisition modes.

    ``from_string`` canonicalizes standard DRF events. Direct construction keeps
    the wire text verbatim and requires its classic FTD and repetition flag.
    An incomplete/unrepresentable classic FTD is -1, per the GETS32 specification.
    Frontends determine which event types they implement.
    """

    text: str
    classic_ftd: int
    repetitive: bool

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip() or any(not 32 <= ord(c) <= 126 for c in self.text):
            raise ValueError("event text must be nonempty printable ASCII")
        _uint(len(self.text) + (len(self.text) & 1), 0xFFFF, "padded event length")
        if isinstance(self.classic_ftd, bool) or not isinstance(self.classic_ftd, int):
            raise TypeError("classic_ftd must be an integer")
        if self.classic_ftd != -1:
            _uint(self.classic_ftd, 0xFFFF, "classic_ftd")
        if not isinstance(self.repetitive, bool):
            raise TypeError("repetitive must be a bool")

    @classmethod
    def from_string(cls, text: str) -> "Gets32Event":
        """Canonicalize I, P/Q, E, S, or N using the existing DRF event parser.

        P/Q periods and clock/state delays are emitted as decimal milliseconds.
        State devices may be names or indices; names are resolved by the frontend.
        Q means return on change, not a one-shot request. U needs database metadata
        and is deliberately not resolved by this low-level client.
        """
        event = parse_event(text)
        if isinstance(event, ImmediateEvent):
            return cls("I", 0, False)
        if isinstance(event, NeverEvent):
            return cls("N", 2, False)
        if isinstance(event, PeriodicEvent):
            _uint(event.freq, 0x7FFFFFFF, "period in milliseconds")
            if event.freq == 0:
                raise ValueError("period must be positive")
            ticks = max(4, 60 * event.freq // 1000)
            ftd = ticks if event.cont and ticks <= 0x7FFF else -1
            return cls(f"{event.mode},{event.freq},{str(event.imm).lower()}", ftd, True)
        if isinstance(event, ClockEvent):
            _uint(event.evt, 0xFF, "clock event")
            _uint(event.delay, 0x7FFFFFFF, "clock delay in milliseconds")
            clock = "E" if event.clock_type == "either" else event.clock_type.upper()
            ftd = 0x8000 | event.evt if event.delay == 0 and clock != "S" else -1
            return cls(f"E,{event.evt:X},{clock},{event.delay}", ftd, True)
        if isinstance(event, StateEvent):
            _uint(event.value, 0xFFFF, "state value")
            _uint(event.delay, 0x7FFFFFFF, "state delay in milliseconds")
            if event.device.isdecimal():
                device = str(int(event.device))
                _uint(int(device), 0xFFFFFF, "state device index")
            else:
                device = get_device_name(event.device)
                if device != event.device.upper():
                    raise ValueError("state target must be a device name or index, without property/range/event")
            return cls(f"S,{device},{event.value},{event.delay},{event.expression}", -1, True)
        raise ValueError("Use an explicit I, P/Q, E, S, or N event, or construct Gets32Event for a wire event")


@dataclass(frozen=True)
class Gets32Header:
    """Native header fields, without interpreting or synthesizing timestamps.

    Time fields normally contain epoch milliseconds. cycle_timestamp may instead
    contain a UCD cycle sequence number with a zero high word (gets32.pdf).
    """

    global_status: int
    type_code: int
    major_version: int
    minor_version: int
    order_flag: int
    sequence: int
    cycle_timestamp: int
    collection_timestamp: int
    reply_timestamp: int


@dataclass(frozen=True)
class Gets32Reply:
    header: Gets32Header
    values: tuple[ReadValue, ...]
    acnet_status: int | None = None
    received_at_ns: int | None = None


def build_request(
    node: int,
    devices: Sequence[ReadDevice],
    event: str | Gets32Event = "I",
    *,
    priority: int = 1,
    max_request_size: int = MAX_ACNET_MESSAGE_SIZE,
    max_reply_size: int = MAX_ACNET_MESSAGE_SIZE,
) -> bytes:
    """Encode GETS32 with byte limits including protocol headers, excluding ACNET's."""
    _message_limits(max_request_size, max_reply_size)
    _uint(node, 0xFFFF, "node")
    _uint(priority, 2, "priority")
    devices = _devices(devices, 0xFFFFFFFF)
    event = Gets32Event.from_string(event) if isinstance(event, str) else event
    if not isinstance(event, Gets32Event):
        raise TypeError("event must be a string or Gets32Event")
    encoded = event.text.encode("ascii")
    encoded += b" " * (len(encoded) & 1)
    reply_size = 34 + _reply_size(devices)
    _message_size(reply_size, max_reply_size, "reply")
    _message_size(18 + len(encoded) + 20 * len(devices), max_request_size, "request")
    prefix = bytes((1, 1, 0, node >> 8, node & 0xFF, 0, priority, int(event.repetitive)))
    header = struct.pack("<IHHH", reply_size, len(devices), event.classic_ftd & 0xFFFF, len(encoded))
    return (
        prefix + header + encoded + b"".join(struct.pack("<I8sII", d.dipi, d.ssdn, d.length, d.offset) for d in devices)
    )


def parse_reply(
    data: bytes,
    devices: Sequence[ReadDevice],
    *,
    acnet_status: int | None = None,
    received_at_ns: int | None = None,
) -> Gets32Reply:
    """Decode the complete reply, preserving header and all padded data slots."""
    devices = _devices(devices, 0xFFFFFFFF)
    if len(data) < 34:
        raise ValueError(f"GETS32 reply too short: {len(data)} bytes, expected at least 34")
    header = Gets32Header(*struct.unpack_from("<hBBBBIQQQ", data))
    if (header.type_code, header.major_version, header.minor_version, header.order_flag) != (1, 1, 0, 0):
        raise ValueError(f"Unsupported GETS32 reply header: {header}")
    return Gets32Reply(header, _parse_values(data, devices, 34), acnet_status, received_at_ns)


class Gets32Client:
    """Raw GETS32 requests over an already-connected ACNET TCP/UDP connection.

    max_request_size/max_reply_size set destination payload limits in bytes,
    including GETS32 headers and padding, excluding the ACNET header.
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
        event: str | Gets32Event,
        priority: int = 1,
        buffer_size: int = 256,
    ) -> ReadStream[Gets32Reply]:
        """Start an acquisition. I is finite; P/Q, E and S are repetitive.

        N and quiet state/clock requests can remain silent; use an appropriate
        readings timeout. Pending heartbeats are consumed without yielding data.
        """
        devices = _devices(devices, 0xFFFFFFFF)
        event = Gets32Event.from_string(event) if isinstance(event, str) else event
        payload = build_request(
            node,
            devices,
            event,
            priority=priority,
            max_request_size=self._max_request_size,
            max_reply_size=self._max_reply_size,
        )
        return ReadStream(
            self._connection,
            node,
            "GETS32",
            payload,
            lambda reply, received: parse_reply(
                reply.data, devices, acnet_status=reply.status, received_at_ns=received
            ),
            repetitive=event.repetitive,
            buffer_size=buffer_size,
        )

    def read(
        self,
        node: int,
        devices: Sequence[ReadDevice],
        *,
        event: str | Gets32Event = "I",
        priority: int = 1,
        timeout: float = 1.0,
    ) -> Gets32Reply:
        """Request one batch, optionally on an event, and cancel on every exit.

        timeout bounds waiting for data after ACNET request setup. The connection
        retains its own command/ACK timeout. Device and global statuses are returned.
        """
        _timeout(timeout)
        event = Gets32Event.from_string(event) if isinstance(event, str) else event
        if not isinstance(event, Gets32Event):
            raise TypeError("event must be a string or Gets32Event")
        return _first(self.stream(node, devices, event=replace(event, repetitive=False), priority=priority), timeout)
