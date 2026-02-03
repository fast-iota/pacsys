"""
ACNET packet parsing and creation.

ACNET packets have an 18-byte header followed by variable-length data:

Offset  Size  Byte Order     Field             Description
------  ----  ----------     -----             -----------
0       2     Little-endian  flags             Message type and control flags
2       2     Little-endian  status            Status code (or reply ID)
4       2     Big-endian     server            Server node (trunk:node)
6       2     Big-endian     client            Client node (trunk:node)
8       4     Little-endian  serverTask        Server task (Rad50 encoded)
12      2     Little-endian  clientTaskId      Client task identifier
14      2     Little-endian  id                Message/request ID
16      2     Little-endian  length            Total packet length
18+     var   Little-endian  data              Payload (variable)
"""

import struct
from dataclasses import dataclass
from typing import Optional

from . import rad50
from .constants import (
    ACNET_FLG_CAN,
    ACNET_FLG_MLT,
    ACNET_FLG_REQ,
    ACNET_FLG_RPY,
    ACNET_FLG_TYPE,
    ACNET_FLG_USM,
    ACNET_HEADER_SIZE,
)


@dataclass
class RequestId:
    """Identifier for an outgoing request."""

    id: int

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, RequestId):
            return self.id == other.id
        return False


@dataclass
class ReplyId:
    """Identifier for an incoming request (used when sending replies)."""

    value: int

    @classmethod
    def from_client_and_id(cls, client: int, msg_id: int) -> "ReplyId":
        """Create reply ID from client node and message ID."""
        return cls((client << 16) | msg_id)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, ReplyId):
            return self.value == other.value
        return False


class AcnetPacket:
    """Base class for ACNET packets."""

    def __init__(
        self,
        flags: int,
        status: int,
        server: int,
        client: int,
        server_task: int,
        client_task_id: int,
        msg_id: int,
        length: int,
        data: bytes,
    ):
        self.flags = flags
        self.status = status
        self.server = server
        self.client = client
        self.server_task = server_task
        self.client_task_id = client_task_id
        self.id = msg_id
        self.length = length
        self.data = data

    @property
    def server_task_name(self) -> str:
        """Get server task name as string."""
        return rad50.decode_stripped(self.server_task)

    @property
    def server_trunk(self) -> int:
        """Get server trunk number."""
        return (self.server >> 8) & 0xFF

    @property
    def server_node(self) -> int:
        """Get server node number."""
        return self.server & 0xFF

    @property
    def client_trunk(self) -> int:
        """Get client trunk number."""
        return (self.client >> 8) & 0xFF

    @property
    def client_node(self) -> int:
        """Get client node number."""
        return self.client & 0xFF

    def is_request(self) -> bool:
        return False

    def is_reply(self) -> bool:
        return False

    def is_message(self) -> bool:
        return False

    def is_cancel(self) -> bool:
        return False

    @staticmethod
    def parse(data: bytes) -> "AcnetPacket":
        """
        Parse raw bytes into an AcnetPacket subclass.

        Args:
            data: Raw packet bytes (at least 18 bytes for header)

        Returns:
            Appropriate AcnetPacket subclass based on flags
        """
        if len(data) < ACNET_HEADER_SIZE:
            raise ValueError(f"Packet too short: {len(data)} < {ACNET_HEADER_SIZE}")

        # Parse header
        flags = struct.unpack_from("<H", data, 0)[0]
        status = struct.unpack_from("<h", data, 2)[0]  # Signed
        server = struct.unpack_from(">H", data, 4)[0]  # Big-endian
        client = struct.unpack_from(">H", data, 6)[0]  # Big-endian
        server_task = struct.unpack_from("<I", data, 8)[0]
        client_task_id = struct.unpack_from("<H", data, 12)[0]
        msg_id = struct.unpack_from("<H", data, 14)[0]
        length = struct.unpack_from("<H", data, 16)[0]

        # Extract payload
        payload = data[ACNET_HEADER_SIZE:]

        # Determine packet type from flags
        msg_type = flags & (ACNET_FLG_TYPE | ACNET_FLG_CAN)

        if msg_type == ACNET_FLG_RPY:
            return AcnetReply(flags, status, server, client, server_task, client_task_id, msg_id, length, payload)
        elif msg_type == ACNET_FLG_USM:
            return AcnetMessage(flags, status, server, client, server_task, client_task_id, msg_id, length, payload)
        elif msg_type == ACNET_FLG_REQ:
            return AcnetRequest(flags, status, server, client, server_task, client_task_id, msg_id, length, payload)
        elif msg_type == ACNET_FLG_CAN:
            return AcnetCancel(flags, status, server, client, server_task, client_task_id, msg_id, length, payload)
        else:
            return AcnetPacket(flags, status, server, client, server_task, client_task_id, msg_id, length, payload)


class AcnetReply(AcnetPacket):
    """An ACNET reply packet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_id = RequestId(self.id)
        # MLT flag=0 means this is the last (or only) reply
        self._last = (self.flags & ACNET_FLG_MLT) == 0

    @property
    def request_id(self) -> RequestId:
        """Get the request ID this reply is for."""
        return self._request_id

    @property
    def last(self) -> bool:
        """Check if this is the last reply in a multiple-reply sequence."""
        return self._last

    def is_reply(self) -> bool:
        return True

    def success(self) -> bool:
        """Check if the reply indicates success."""
        return self.status == 0

    def __repr__(self):
        return f"AcnetReply(status=0x{self.status & 0xFFFF:04x}, last={self.last})"


class AcnetRequest(AcnetPacket):
    """An ACNET request packet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reply ID comes from status field (if non-zero) or client+id
        if self.status == 0:
            self._reply_id = ReplyId.from_client_and_id(self.client, self.id)
        else:
            self._reply_id = ReplyId(self.status)
        self._multiple_reply = (self.flags & ACNET_FLG_MLT) > 0
        self._cancelled = False
        self.user_object: Optional[object] = None

    @property
    def reply_id(self) -> ReplyId:
        """Get the reply ID to use when responding."""
        return self._reply_id

    @property
    def multiple_reply(self) -> bool:
        """Check if this request expects multiple replies."""
        return self._multiple_reply

    @property
    def cancelled(self) -> bool:
        """Check if this request has been cancelled."""
        return self._cancelled

    def cancel(self):
        """Mark this request as cancelled."""
        self._cancelled = True

    def is_request(self) -> bool:
        return True

    def is_multicast(self) -> bool:
        """Check if this is a multicast request."""
        return (self.server & 0xFFFF) == 0xFF

    def __repr__(self):
        return f"AcnetRequest(reply_id={self._reply_id.value:#x}, mult={self._multiple_reply})"


class AcnetMessage(AcnetPacket):
    """An ACNET unsolicited message packet."""

    def is_message(self) -> bool:
        return True

    def __repr__(self):
        return f"AcnetMessage(from={self.server:#x})"


class AcnetCancel(AcnetPacket):
    """An ACNET cancel packet."""

    def is_cancel(self) -> bool:
        return True

    def __repr__(self):
        return f"AcnetCancel(id={self.id})"


def node_value(trunk: int, node: int) -> int:
    """Create a 16-bit node value from trunk and node numbers."""
    return ((trunk & 0xFF) << 8) | (node & 0xFF)


def node_parts(value: int) -> tuple[int, int]:
    """Split a node value into (trunk, node) tuple."""
    return (value >> 8) & 0xFF, value & 0xFF
