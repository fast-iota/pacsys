"""Tests for the pacsys.acnet module (no network, pure unit tests)."""

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pacsys.acnet import (
    ACNET_HEADER_SIZE,
    ACNET_PEND,
    AcnetConnectionTCP,
    AcnetError,
    AcnetPacket,
    AcnetReply,
    AcnetRequest,
    NodeStats,
    ReplyId,
    decode,
    decode_stripped,
    encode,
    node_parts,
    node_value,
)
from pacsys.acnet.async_connection import AsyncAcnetConnectionTCP
from pacsys.acnet.constants import (
    ACNET_FLG_MLT,
    ACNET_FLG_REQ,
    ACNET_FLG_RPY,
    ACNET_FLG_USM,
    CMD_DEFAULT_NODE,
    CMD_IGNORE_REQUEST,
    CMD_KEEPALIVE,
    CMD_NODE_STATS,
    CMD_RENAME_TASK,
    CMD_SEND,
    CMD_TASK_PID,
)
from pacsys.acnet.errors import (
    make_error,
    normalize_error_code,
    parse_error,
)


class TestRad50:
    """Tests for RAD50 encoding/decoding."""

    def test_encode_simple(self):
        """Test encoding simple names."""
        # "DPM" should encode to a specific value
        result = encode("DPM")
        assert isinstance(result, int)
        assert result == encode("DPM   ")  # Should pad with spaces

    def test_decode_roundtrip(self):
        """Test that encode/decode roundtrips correctly."""
        test_names = ["DPM", "ACNET", "TEST", "A", "123456", "A.B$C%"]
        for name in test_names:
            padded = name.ljust(6)[:6]
            encoded = encode(name)
            decoded = decode(encoded)
            assert decoded == padded.upper(), f"Failed for {name}"

    def test_decode_stripped(self):
        """Test decode_stripped removes trailing spaces."""
        encoded = encode("ABC")
        assert decode(encoded) == "ABC   "
        assert decode_stripped(encoded) == "ABC"

    def test_encode_case_insensitive(self):
        """Test that encoding is case insensitive."""
        assert encode("dpm") == encode("DPM")
        assert encode("AcNeT") == encode("ACNET")

    def test_special_characters(self):
        """Test encoding of special characters ($, ., %)."""
        test_names = ["A$B", "X.Y", "P%Q", "$.%"]
        for name in test_names:
            encoded = encode(name)
            decoded = decode_stripped(encoded)
            assert decoded == name.upper()

    def test_numeric_characters(self):
        """Test encoding of numeric characters."""
        assert decode_stripped(encode("123")) == "123"
        assert decode_stripped(encode("A1B2C3")) == "A1B2C3"

    def test_truncation(self):
        """Test that names longer than 6 chars are truncated."""
        long_name = "TOOLONGNAME"
        encoded = encode(long_name)
        decoded = decode(encoded)
        assert decoded == "TOOLON"

    def test_known_values(self):
        """Test against known encoded values."""
        # Space encodes to 0
        assert encode("      ") == 0
        # "A" in first position: index 1 * 40^2 = 1600
        assert encode("A     ") == 1600
        # "A" in second position: index 1 * 40^1 = 40
        assert encode(" A    ") == 40
        # "A" in third position: index 1 * 40^0 = 1
        assert encode("  A   ") == 1


class TestNodeAddressing:
    """Tests for node addressing utilities."""

    def test_node_value(self):
        """Test creating node values from trunk/node."""
        # Trunk 9, node 1
        value = node_value(9, 1)
        assert value == 0x0901

        # Trunk 0, node 255
        value = node_value(0, 255)
        assert value == 0x00FF

    def test_node_parts(self):
        """Test splitting node values."""
        trunk, node = node_parts(0x0901)
        assert trunk == 9
        assert node == 1

        trunk, node = node_parts(0xFF00)
        assert trunk == 255
        assert node == 0

    def test_roundtrip(self):
        """Test node_value/node_parts roundtrip."""
        for trunk in [0, 9, 128, 255]:
            for node in [0, 1, 128, 255]:
                value = node_value(trunk, node)
                t, n = node_parts(value)
                assert t == trunk and n == node


class TestErrorCodes:
    """Tests for error code utilities."""

    def test_make_error(self):
        """Test creating error codes."""
        # ACNET_PEND = 1 + (1 * 256) = 257 = 0x0101
        assert make_error(1, 1) == 0x0101
        # ACNET_ENDMULT = 1 + (2 * 256) = 513 = 0x0201
        assert make_error(1, 2) == 0x0201

    def test_parse_error(self):
        """Test parsing error codes."""
        facility, error_num = parse_error(ACNET_PEND)
        assert facility == 1
        assert error_num == 1

    def test_negative_errors(self):
        """Test negative error numbers."""
        # ACNET_NO_NODE = 1 + (-30 * 256)
        code = make_error(1, -30)
        facility, error_num = parse_error(code)
        assert facility == 1
        assert error_num == -30


class TestNormalizeErrorCode:
    """Tests for unsigned -> signed error code normalization."""

    @pytest.mark.parametrize(
        "input_code,expected",
        [
            (0, 0),
            (1, 1),
            (42, 42),
            (127, 127),
            (128, -128),
            (200, -56),
            (227, -29),
            (255, -1),
            (-1, -1),
            (-29, -29),
        ],
    )
    def test_normalize_error_code(self, input_code, expected):
        assert normalize_error_code(input_code) == expected


class TestRequestReplyIds:
    """Tests for RequestId and ReplyId."""

    def test_reply_id_from_client_and_id(self):
        """Test ReplyId creation from client and message ID."""
        reply_id = ReplyId.from_client_and_id(0x0901, 0x1234)
        assert reply_id.value == 0x09011234


class TestPacketParsing:
    """Tests for ACNET packet parsing."""

    def _make_packet(self, flags: int, status: int = 0, server: int = 0, client: int = 0, data: bytes = b"") -> bytes:
        """Helper to construct a raw ACNET packet."""
        server_task = 0
        client_task_id = 0
        msg_id = 1
        length = ACNET_HEADER_SIZE + len(data)

        # Build header
        header = struct.pack("<H", flags)  # flags - little endian
        header += struct.pack("<h", status)  # status - little endian signed
        header += struct.pack(">H", server)  # server - big endian
        header += struct.pack(">H", client)  # client - big endian
        header += struct.pack("<I", server_task)  # server task - little endian
        header += struct.pack("<H", client_task_id)  # client task id
        header += struct.pack("<H", msg_id)  # message id
        header += struct.pack("<H", length)  # length

        return header + data

    def test_parse_reply(self):
        """Test parsing a reply packet."""
        raw = self._make_packet(ACNET_FLG_RPY, status=0)
        packet = AcnetPacket.parse(raw)

        assert isinstance(packet, AcnetReply)
        assert packet.is_reply()
        assert not packet.is_request()
        assert packet.status == 0
        assert packet.last  # No MLT flag = last reply

    def test_parse_reply_multiple(self):
        """Test parsing a multi-reply packet."""
        raw = self._make_packet(ACNET_FLG_RPY | ACNET_FLG_MLT, status=ACNET_PEND)
        packet = AcnetPacket.parse(raw)

        assert isinstance(packet, AcnetReply)
        assert not packet.last  # MLT flag set = more replies coming
        assert packet.status == ACNET_PEND

    def test_parse_request(self):
        """Test parsing a request packet."""
        raw = self._make_packet(ACNET_FLG_REQ, server=0x0901, client=0x0902)
        packet = AcnetPacket.parse(raw)

        assert isinstance(packet, AcnetRequest)
        assert packet.is_request()
        assert packet.server == 0x0901
        assert packet.client == 0x0902

    def test_parse_request_multiple_reply(self):
        """Test parsing a request expecting multiple replies."""
        raw = self._make_packet(ACNET_FLG_REQ | ACNET_FLG_MLT)
        packet = AcnetPacket.parse(raw)

        assert isinstance(packet, AcnetRequest)
        assert packet.multiple_reply

    def test_parse_message(self):
        """Test parsing an unsolicited message."""
        raw = self._make_packet(ACNET_FLG_USM)
        packet = AcnetPacket.parse(raw)

        assert packet.is_message()
        assert not packet.is_reply()
        assert not packet.is_request()

    def test_parse_with_data(self):
        """Test parsing a packet with payload data."""
        payload = b"Hello ACNET"
        raw = self._make_packet(ACNET_FLG_RPY, data=payload)
        packet = AcnetPacket.parse(raw)

        assert packet.data == payload

    def test_packet_too_short(self):
        """Test that short packets raise an error."""
        with pytest.raises(ValueError, match="too short"):
            AcnetPacket.parse(b"short")

    def test_server_task_name(self):
        """Test getting server task name from packet."""
        raw = self._make_packet(ACNET_FLG_RPY)
        packet = AcnetPacket.parse(raw)
        # Server task is 0, which decodes to spaces
        assert packet.server_task_name == ""

    def test_node_properties(self):
        """Test trunk/node extraction from packet."""
        raw = self._make_packet(ACNET_FLG_RPY, server=0x0901, client=0x0A02)
        packet = AcnetPacket.parse(raw)

        assert packet.server_trunk == 9
        assert packet.server_node == 1
        assert packet.client_trunk == 10
        assert packet.client_node == 2


class TestAcnetReply:
    """Tests for AcnetReply class."""

    def test_success(self):
        """Test success() method."""
        # Create a success reply
        raw = struct.pack("<HhHHIHHH", ACNET_FLG_RPY, 0, 0, 0, 0, 0, 1, 18)
        reply = AcnetPacket.parse(raw)
        assert reply.success()

        # Create a failure reply
        raw = struct.pack("<HhHHIHHH", ACNET_FLG_RPY, -1, 0, 0, 0, 0, 1, 18)
        reply = AcnetPacket.parse(raw)
        assert not reply.success()

    def test_request_id(self):
        """Test getting request ID from reply."""
        # Message ID is at offset 14
        raw = struct.pack("<HhHHIHHH", ACNET_FLG_RPY, 0, 0, 0, 0, 0, 0x1234, 18)
        reply = AcnetPacket.parse(raw)
        assert reply.request_id.id == 0x1234


class TestAcnetRequest:
    """Tests for AcnetRequest class."""

    def test_reply_id_from_status(self):
        """Test reply ID extraction when status is non-zero."""
        # When status field is non-zero, it contains the reply ID
        raw = struct.pack("<HhHHIHHH", ACNET_FLG_REQ, 0x5678, 0, 0, 0, 0, 1, 18)
        request = AcnetPacket.parse(raw)
        assert request.reply_id.value == 0x5678

    def test_reply_id_from_client(self):
        """Test reply ID extraction when status is zero."""
        # When status is 0, reply ID comes from client and message ID
        # Client at big-endian offset 6, msg_id at little-endian offset 14
        raw = struct.pack("<H", ACNET_FLG_REQ)  # flags
        raw += struct.pack("<h", 0)  # status = 0
        raw += struct.pack(">H", 0)  # server
        raw += struct.pack(">H", 0x0901)  # client
        raw += struct.pack("<I", 0)  # server task
        raw += struct.pack("<H", 0)  # client task id
        raw += struct.pack("<H", 0x1234)  # msg id
        raw += struct.pack("<H", 18)  # length

        request = AcnetPacket.parse(raw)
        # Reply ID should be (client << 16) | msg_id
        expected = (0x0901 << 16) | 0x1234
        assert request.reply_id.value == expected

    def test_is_multicast(self):
        """Test multicast detection."""
        # Server node 0xFF indicates multicast
        raw = struct.pack("<H", ACNET_FLG_REQ)  # flags
        raw += struct.pack("<h", 0)  # status
        raw += struct.pack(">H", 0x00FF)  # server = multicast
        raw += struct.pack(">H", 0)  # client
        raw += struct.pack("<I", 0)  # server task
        raw += struct.pack("<H", 0)  # client task id
        raw += struct.pack("<H", 1)  # msg id
        raw += struct.pack("<H", 18)  # length

        request = AcnetPacket.parse(raw)
        assert request.is_multicast()


# =============================================================================
# AcnetConnectionTCP command protocol tests (mocked _xact)
# =============================================================================


def _make_conn():
    """Create a TCP connection with fake handle, bypassing real connect.

    Sets up a real reactor thread and async core so that _run_sync works,
    but mocks the underlying stream writer to avoid real network I/O.
    """
    conn = AcnetConnectionTCP("localhost", port=9999)
    conn._start_reactor()
    conn._async = AsyncAcnetConnectionTCP("localhost", port=9999)
    conn._async._raw_handle = encode("TEST")
    conn._async._handle_name = "TEST"
    conn._async._connected = True
    conn._async._writer = MagicMock()
    conn._async._writer.write = MagicMock()
    conn._async._writer.drain = AsyncMock()
    return conn


@pytest.fixture
def conn():
    c = _make_conn()
    try:
        yield c
    finally:
        c.close()


class TestTCPGetDefaultNode:
    """Tests for get_default_node (cmdDefaultNode)."""

    def test_returns_node_address(self, conn):
        # ack: [ack_code=4][status=0][trunk=12][node=6]
        ack = struct.pack(">HhBB", 4, 0, 12, 6)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            result = conn.get_default_node()
        assert result == 12 * 256 + 6
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[2:4])[0]
        assert cmd == CMD_DEFAULT_NODE

    def test_error_raises(self, conn):
        ack = struct.pack(">HhBB", 4, -1, 0, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)):
            with pytest.raises(AcnetError):
                conn.get_default_node()


class TestTCPRenameTask:
    """Tests for rename_task (cmdRenameTask)."""

    def test_renames_and_updates_handle(self, conn):
        ack = struct.pack(">Hh", 0, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            conn.rename_task("NEWNAM")
        assert conn.name == "NEWNAM"
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[2:4])[0]
        assert cmd == CMD_RENAME_TASK
        # RAD50 name after handle(4) + virtual_node(4) = offset 16
        name_rad50 = struct.unpack(">I", buf[12:16])[0]
        assert decode_stripped(name_rad50) == "NEWNAM"

    def test_empty_name_raises(self, conn):
        with pytest.raises(ValueError):
            conn.rename_task("")

    def test_long_name_raises(self, conn):
        with pytest.raises(ValueError):
            conn.rename_task("TOOLONGNAME")

    def test_error_raises(self, conn):
        ack = struct.pack(">Hh", 0, -1)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)):
            with pytest.raises(AcnetError):
                conn.rename_task("FOO")


class TestTCPSendMessage:
    """Tests for send_message (cmdSend)."""

    def test_sends_with_payload(self, conn):
        ack = struct.pack(">Hh", 0, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            conn.send_message(node=0x0A06, task="DPM", data=b"\x01\x02")
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[2:4])[0]
        assert cmd == CMD_SEND
        # Payload at the end
        assert buf[-2:] == b"\x01\x02"

    def test_error_raises(self, conn):
        ack = struct.pack(">Hh", 0, -3)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)):
            with pytest.raises(AcnetError):
                conn.send_message(0x0A06, "DPM", b"")


class TestTCPIgnoreRequest:
    """Tests for ignore_request (cmdIgnoreRequest)."""

    def test_sends_ignore_and_cleans_up(self, conn):
        # Build a fake AcnetRequest
        raw = struct.pack("<HhHHIHHH", ACNET_FLG_REQ, 0, 0, 0, 0, 0, 42, 18)
        request = AcnetPacket.parse(raw)
        # Register it as an incoming request (on the async core)
        conn._async._requests_in[request.reply_id] = request

        ack = struct.pack(">Hh", 0, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            conn.ignore_request(request)

        assert request.reply_id not in conn._async._requests_in
        assert request.cancelled
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[2:4])[0]
        assert cmd == CMD_IGNORE_REQUEST


class TestTCPGetNodeStats:
    """Tests for get_node_stats (cmdNodeStats)."""

    def test_returns_stats_dataclass(self, conn):
        counters = (10, 20, 30, 40, 50, 60, 100)
        ack = struct.pack(">Hh7I", 7, 0, *counters)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            stats = conn.get_node_stats()
        assert isinstance(stats, NodeStats)
        assert stats.usm_received == 10
        assert stats.requests_received == 20
        assert stats.replies_received == 30
        assert stats.usm_sent == 40
        assert stats.requests_sent == 50
        assert stats.replies_sent == 60
        assert stats.request_queue_limit == 100
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[2:4])[0]
        assert cmd == CMD_NODE_STATS

    def test_error_raises(self, conn):
        ack = struct.pack(">Hh7I", 7, -1, 0, 0, 0, 0, 0, 0, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)):
            with pytest.raises(AcnetError):
                conn.get_node_stats()


class TestTCPGetTaskPid:
    """Tests for get_task_pid (cmdTaskPid)."""

    def test_returns_pid(self, conn):
        ack = struct.pack(">HhI", 6, 0, 12345)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            pid = conn.get_task_pid("DPM")
        assert pid == 12345
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[2:4])[0]
        assert cmd == CMD_TASK_PID

    def test_error_raises(self, conn):
        ack = struct.pack(">HhI", 6, -1, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)):
            with pytest.raises(AcnetError):
                conn.get_task_pid("NOPE")


class TestTCPKeepalive:
    """Tests for keepalive command."""

    def test_keepalive_sends_correct_command(self, conn):
        ack = struct.pack(">Hh", 0, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            conn._send_keepalive()
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[2:4])[0]
        assert cmd == CMD_KEEPALIVE
