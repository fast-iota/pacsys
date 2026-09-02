"""Tests for the pacsys.acnet module (no network, pure unit tests)."""

import asyncio
import struct
import threading
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pacsys.acnet import (
    ACNET_HEADER_SIZE,
    ACNET_PEND,
    AcnetConnectionTCP,
    AcnetConnectionUDP,
    AcnetError,
    AcnetPacket,
    AcnetReply,
    AcnetRequest,
    AcnetUnavailableError,
    DPMAcnet,
    DPMError,
    DPMReading,
    NodeStats,
    ReplyId,
    RequestId,
    decode,
    decode_stripped,
    encode,
    node_parts,
    node_value,
)
from pacsys.acnet.async_connection import AsyncAcnetConnectionTCP, AsyncAcnetConnectionUDP
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
    DAE_LJ_NO_DATA,
    DPM_NOT_YET,
    make_error,
    normalize_error_code,
    parse_error,
    status_message,
)
from pacsys.dpm_protocol import ListStatus_reply, Status_reply


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
                assert t == trunk
                assert n == node


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
        ("input_code", "expected"),
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

    def test_dpm_not_yet_message(self):
        assert DPM_NOT_YET == make_error(17, -48)
        assert "not implemented" in status_message(17, -48)

    def test_logger_no_data_message(self):
        assert DAE_LJ_NO_DATA == make_error(66, -64)
        assert "no logger data" in status_message(66, -64)


class TestRequestReplyIds:
    """Tests for RequestId and ReplyId."""

    def test_reply_id_from_client_and_id(self):
        """Test ReplyId creation from client and message ID."""
        reply_id = ReplyId.from_client_and_id(0x0901, 0x1234)
        assert reply_id.value == 0x09011234

    def test_ids_are_immutable_dictionary_keys(self):
        request_id = RequestId(1)
        reply_id = ReplyId(2)
        ids = {request_id: "request", reply_id: "reply"}

        with pytest.raises(FrozenInstanceError):
            request_id.id = 3
        with pytest.raises(FrozenInstanceError):
            reply_id.value = 4

        assert ids[RequestId(1)] == "request"
        assert ids[ReplyId(2)] == "reply"


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

    @pytest.mark.parametrize("declared_length", [17, 19])
    def test_invalid_declared_packet_length_raises(self, declared_length):
        raw = bytearray(self._make_packet(ACNET_FLG_RPY))
        struct.pack_into("<H", raw, 16, declared_length)

        with pytest.raises(ValueError, match="Bad packet length"):
            AcnetPacket.parse(bytes(raw))

    def test_payload_excludes_bytes_beyond_declared_length(self):
        raw = self._make_packet(ACNET_FLG_RPY, data=b"payload") + b"padding"
        packet = AcnetPacket.parse(raw)
        assert packet.data == b"payload"

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
        c._async._connected = False  # skip disconnect handshake (0.5s timeout)
        c.close()


def test_sync_method_from_reactor_thread_raises_immediately():
    conn = AcnetConnectionTCP("localhost", port=9999)
    conn._async = AsyncAcnetConnectionTCP("localhost", port=9999)
    conn._reactor_thread = threading.current_thread()

    with pytest.raises(RuntimeError, match="cannot be called from the ACNET reactor thread"):
        conn.get_local_node()


def test_sync_connect_rejects_second_connection():
    conn = AcnetConnectionTCP("localhost", port=9999)
    conn._async = MagicMock()

    with patch.object(conn, "_start_reactor") as start_reactor:
        with pytest.raises(RuntimeError, match="already connected"):
            conn.connect()

    start_reactor.assert_not_called()


@pytest.mark.asyncio
async def test_async_connect_rejects_second_connection():
    conn = AsyncAcnetConnectionTCP("localhost", port=9999)
    conn._connected = True

    with pytest.raises(RuntimeError, match="already connected"):
        await conn.connect()


class TestTCPGetDefaultNode:
    """Tests for get_default_node (cmdDefaultNode)."""

    def test_returns_node_address(self, conn):
        # ack: [ack_code=4][status=0][trunk=12][node=6]
        ack = struct.pack(">HhBB", 4, 0, 12, 6)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            result = conn.get_default_node()
        assert result == 12 * 256 + 6
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[:2])[0]
        assert cmd == CMD_DEFAULT_NODE

    def test_error_raises(self, conn):
        ack = struct.pack(">HhBB", 4, -1, 0, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)):
            with pytest.raises(AcnetError):
                conn.get_default_node()


def test_sync_udp_wraps_official_local_async_transport():
    conn = AcnetConnectionUDP(name="UDP001")
    core = conn._create_async()

    assert isinstance(core, AsyncAcnetConnectionUDP)
    assert core.host == "127.0.0.1"
    assert core.raw_handle == encode("UDP001")


class TestTCPRenameTask:
    """Tests for rename_task (cmdRenameTask)."""

    def test_renames_and_updates_handle(self, conn):
        ack = struct.pack(">Hh", 0, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            conn.rename_task("NEWNAM")
        assert conn.name == "NEWNAM"
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[:2])[0]
        assert cmd == CMD_RENAME_TASK
        name_rad50 = struct.unpack(">I", buf[10:14])[0]
        assert decode_stripped(name_rad50) == "NEWNAM"

    def test_empty_name_raises(self, conn):
        with pytest.raises(ValueError, match="Task name must be 1-6 characters"):
            conn.rename_task("")

    def test_long_name_raises(self, conn):
        with pytest.raises(ValueError, match="Task name must be 1-6 characters"):
            conn.rename_task("TOOLONGNAME")

    def test_error_raises(self, conn):
        ack = struct.pack(">Hh", 0, -1)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)):
            with pytest.raises(AcnetError):
                conn.rename_task("FOO")


def test_get_node_rejects_long_name_before_io(conn):
    with patch.object(conn._async, "_xact", new=AsyncMock()) as xact:
        with pytest.raises(ValueError, match="Node name must be 1-6 characters"):
            conn.get_node("TOOLONGNAME")

    xact.assert_not_awaited()


class TestTCPSendMessage:
    """Tests for send_message (cmdSend)."""

    def test_sends_with_payload(self, conn):
        ack = struct.pack(">Hh", 0, 0)
        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)) as mock:
            conn.send_message(node=0x0A06, task="DPM", data=b"\x01\x02")
        buf = mock.call_args[0][0]
        cmd = struct.unpack(">H", buf[:2])[0]
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
        cmd = struct.unpack(">H", buf[:2])[0]
        assert cmd == CMD_IGNORE_REQUEST

    def test_error_propagates_after_local_cleanup(self, conn):
        raw = struct.pack("<HhHHIHHH", ACNET_FLG_REQ, 0, 0, 0, 0, 0, 42, 18)
        request = AcnetPacket.parse(raw)
        conn._async._requests_in[request.reply_id] = request

        with patch.object(conn._async, "_xact", new=AsyncMock(side_effect=AcnetUnavailableError)):
            with pytest.raises(AcnetUnavailableError):
                conn.ignore_request(request)

        assert request.cancelled
        assert request.reply_id not in conn._async._requests_in

    def test_negative_ack_raises(self, conn):
        raw = struct.pack("<HhHHIHHH", ACNET_FLG_REQ, 0, 0, 0, 0, 0, 42, 18)
        request = AcnetPacket.parse(raw)
        ack = struct.pack(">Hh", 0, -1)

        with patch.object(conn._async, "_xact", new=AsyncMock(return_value=ack)):
            with pytest.raises(AcnetError, match="IGNORE_REQUEST failed"):
                conn.ignore_request(request)

    def test_cancelled_request_is_idempotent(self, conn):
        raw = struct.pack("<HhHHIHHH", ACNET_FLG_REQ, 0, 0, 0, 0, 0, 42, 18)
        request = AcnetPacket.parse(raw)
        request.cancel()

        with patch.object(conn._async, "_xact", new=AsyncMock()) as mock:
            conn.ignore_request(request)

        mock.assert_not_awaited()


class TestDPMAcnetNodeResolution:
    @pytest.mark.parametrize(("configured", "expected"), [(None, "DPM06"), ("DPM01", "DPM01")])
    def test_resolves_configured_or_default_node(self, configured, expected):
        dpm = DPMAcnet(dpm_node=configured)
        dpm._con = MagicMock()
        dpm._con.get_node.return_value = 123

        dpm._find_dpm()

        dpm._con.get_node.assert_called_once_with(expected)
        assert dpm._dpm_node == 123


class TestDPMAcnetListState:
    @pytest.fixture
    def dpm(self):
        dpm = DPMAcnet()
        dpm._list_id = 123
        dpm._active = True
        dpm._dev_list = {1: "M:OUTTMP@I"}
        dpm._meta = {1: {"name": "M:OUTTMP"}}
        return dpm

    def test_stop_commits_after_success(self, dpm):
        reply = Status_reply()
        reply.status = 0
        dpm._send_request = MagicMock(return_value=reply)

        dpm.stop()

        assert not dpm._active

    @pytest.mark.parametrize("reply", [object(), pytest.param(None, id="no-reply")])
    def test_stop_rejects_unexpected_reply(self, dpm, reply):
        dpm._send_request = MagicMock(return_value=reply)

        with pytest.raises(DPMError, match="Expected Status_reply"):
            dpm.stop()

        assert not dpm._active

    def test_stop_failure_clears_active_state(self, dpm):
        reply = Status_reply()
        reply.status = -1
        dpm._send_request = MagicMock(return_value=reply)

        with pytest.raises(DPMError, match="StopList failed"):
            dpm.stop()

        assert not dpm._active

    def test_stop_transport_failure_clears_active_state(self, dpm):
        dpm._send_request = MagicMock(side_effect=DPMError(-1, "timeout"))

        with pytest.raises(DPMError, match="timeout"):
            dpm.stop()

        assert not dpm._active

    def test_read_preserves_primary_error_when_stop_fails(self, dpm, caplog):
        dpm._active = False
        dpm._dev_list = {}
        dpm.add_entry = MagicMock()
        dpm.start = MagicMock(side_effect=lambda: setattr(dpm, "_active", True))
        dpm.readings = MagicMock(return_value=iter(()))
        dpm.stop = MagicMock(side_effect=DPMError(-1, "stop timeout"))
        dpm.close = MagicMock()

        with pytest.raises(TimeoutError, match=r"Timeout reading M:OUTTMP\.READING@I"):
            dpm.read("M:OUTTMP", timeout=0)

        dpm.close.assert_called_once_with()
        assert "Failed to stop DPM acquisition after reading M:OUTTMP.READING@I" in caplog.text

    def test_read_uses_parser_for_immediate_event(self, dpm):
        expected_drf = "M:UTEST.ANALOG@I"
        tag = hash(expected_drf) & 0x7FFFFFFF

        dpm._active = True
        dpm._dev_list = {}
        dpm.add_entry = MagicMock()
        dpm.readings = MagicMock(return_value=iter([DPMReading(ref_id=tag, data=1.0)]))

        reading = dpm.read("M@UTEST")

        assert reading.data == 1.0
        dpm.add_entry.assert_called_once_with(tag, expected_drf)

    def test_read_propagates_stop_failure_after_success(self, dpm, caplog):
        drf = "M:OUTTMP@I"
        tag = hash(drf) & 0x7FFFFFFF
        dpm._active = False
        dpm._dev_list = {}
        dpm.add_entry = MagicMock()
        dpm.start = MagicMock(side_effect=lambda: setattr(dpm, "_active", True))
        dpm.readings = MagicMock(return_value=iter([DPMReading(ref_id=tag, data=1.0)]))
        dpm.stop = MagicMock(side_effect=DPMError(-1, "stop timeout"))
        dpm.close = MagicMock()

        with pytest.raises(DPMError, match="stop timeout"):
            dpm.read(drf)

        dpm.close.assert_called_once_with()
        assert "Failed to stop DPM acquisition after reading M:OUTTMP@I" in caplog.text

    def test_clear_list_commits_after_success(self, dpm):
        reply = ListStatus_reply()
        reply.status = 0
        dpm._send_request = MagicMock(return_value=reply)

        dpm.clear_list()

        assert not dpm._dev_list
        assert not dpm._meta

    def test_clear_list_rejects_unexpected_reply(self, dpm):
        dpm._send_request = MagicMock(return_value=object())

        with pytest.raises(DPMError, match="Expected ListStatus_reply"):
            dpm.clear_list()

        assert dpm._dev_list == {1: "M:OUTTMP@I"}
        assert dpm._meta == {1: {"name": "M:OUTTMP"}}

    def test_clear_list_failure_preserves_maps(self, dpm):
        reply = ListStatus_reply()
        reply.status = -1
        dpm._send_request = MagicMock(return_value=reply)

        with pytest.raises(DPMError, match="ClearList failed"):
            dpm.clear_list()

        assert dpm._dev_list == {1: "M:OUTTMP@I"}
        assert dpm._meta == {1: {"name": "M:OUTTMP"}}

    def test_clear_list_transport_failure_preserves_maps(self, dpm):
        dpm._send_request = MagicMock(side_effect=DPMError(-1, "timeout"))

        with pytest.raises(DPMError, match="timeout"):
            dpm.clear_list()

        assert dpm._dev_list == {1: "M:OUTTMP@I"}
        assert dpm._meta == {1: {"name": "M:OUTTMP"}}


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
        cmd = struct.unpack(">H", buf[:2])[0]
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
        cmd = struct.unpack(">H", buf[:2])[0]
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
        cmd = struct.unpack(">H", buf[:2])[0]
        assert cmd == CMD_KEEPALIVE

    @pytest.mark.asyncio
    async def test_loop_stops_after_connection_loss(self):
        conn = AsyncAcnetConnectionTCP("localhost", 6802)
        conn._connected = False
        conn._send_keepalive = AsyncMock()

        await conn._keepalive_loop()

        conn._send_keepalive.assert_not_awaited()


class TestXactCancelledError:
    """_xact must close transport on CancelledError, same as TimeoutError."""

    @pytest.mark.asyncio
    async def test_xact_cleans_up_on_cancelled_error(self):
        """CancelledError during ACK wait must close transport to prevent desync."""
        conn = AsyncAcnetConnectionTCP("localhost", 6802)
        conn._connected = True
        conn._disposed = False
        conn._cmd_lock = asyncio.Lock()
        conn._pending_ack = None
        conn._trace = False

        # Mock transport methods
        conn._send_frame = AsyncMock()
        conn._close_transport = AsyncMock()

        async def cancel_during_wait():
            """Simulate external cancellation while awaiting ACK."""
            await conn._xact(b"\x00\x00\x00\x01", timeout=5.0)

        task = asyncio.create_task(cancel_during_wait())
        await asyncio.sleep(0.01)  # let task start and block on _pending_ack
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Transport must be closed (same as timeout path)
        conn._close_transport.assert_awaited_once()
        assert conn._connected is False
        assert conn._pending_ack is None


class TestDPMAcnetOpenList:
    def test_open_list_rejects_non_openlist_first_reply(self):
        """A wrong-type first reply must fail the handshake, not leave list_id=None."""
        import queue
        from types import SimpleNamespace

        d = DPMAcnet.__new__(DPMAcnet)
        d._reply_queue = queue.Queue()
        d._meta = {}
        d._con = MagicMock()
        d._dpm_node = 0x0901

        status = ListStatus_reply()
        status.status = 0
        status.list_id = 5
        fake = SimpleNamespace(last=False, status=0, data=bytes(status.marshal()))

        def request_multiple(**kwargs):
            kwargs["reply_handler"](fake)
            return MagicMock()

        d._con.request_multiple.side_effect = request_multiple

        with pytest.raises(DPMError, match="expected OpenList_reply, got ListStatus_reply"):
            d._open_list()


class TestDPMAcnetSendRequest:
    def test_negative_reply_status_is_preserved(self):
        dpm = DPMAcnet()
        dpm._dpm_node = 1

        def request_single(**kwargs):
            kwargs["reply_handler"](SimpleNamespace(last=True, status=-42, data=b""))
            return MagicMock()

        dpm._con = MagicMock(request_single=request_single)
        msg = MagicMock()
        msg.marshal.return_value = b""

        with pytest.raises(DPMError) as exc_info:
            dpm._send_request(msg)

        assert exc_info.value.status == -42


class TestDPMAcnetStreamTermination:
    def test_readings_stops_on_termination_sentinel(self):
        """A None sentinel (connection lost) terminates the readings() generator."""
        import queue

        d = DPMAcnet.__new__(DPMAcnet)
        d._reply_queue = queue.Queue()
        d._terminal_status = None
        d._reply_queue.put("r1")
        d._reply_queue.put(None)
        d._reply_queue.put("r2")
        assert list(d.readings(timeout=0.5)) == ["r1"]

    @staticmethod
    def _terminal_reply(status):
        from types import SimpleNamespace

        return SimpleNamespace(last=True, status=status, data=b"")

    def _open(self, dpm):
        """Drive _open_list with a fake connection; returns the captured reply handler."""
        from pacsys.dpm_protocol import OpenList_reply

        captured = {}

        def request_multiple(node, task, data, reply_handler, timeout):
            captured["handler"] = reply_handler
            ol = OpenList_reply()
            ol.list_id = 7
            reply_handler(SimpleNamespace(last=False, status=0, data=bytes(ol.marshal())))
            return MagicMock()

        dpm._con = MagicMock(request_multiple=request_multiple)
        dpm._dpm_node = 1
        dpm._open_list()
        assert dpm._list_id == 7
        return captured["handler"]

    def test_terminal_reply_wakes_consumer_when_queue_full(self):
        """The sentinel must land even when a slow consumer filled the queue (oldest reading dropped)."""
        import queue

        from pacsys.acnet.errors import ACNET_DISCONNECTED

        dpm = DPMAcnet()
        dpm._reply_queue = queue.Queue(maxsize=2)
        handler = self._open(dpm)
        dpm._reply_queue.put(DPMReading(ref_id=1))
        dpm._reply_queue.put(DPMReading(ref_id=2))

        got = []
        t = threading.Thread(target=lambda: got.extend(dpm.readings(timeout=None)), daemon=True)
        t.start()
        time.sleep(0.05)  # consumer drains and blocks in queue.get()
        handler(self._terminal_reply(ACNET_DISCONNECTED))
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert [r.ref_id for r in got] == [1, 2]
        assert dpm._terminal_status == ACNET_DISCONNECTED
        # Later readings after termination are dropped; a second iteration returns at once
        dpm._handle_dpm_reply(Status_reply())
        assert list(dpm.readings(timeout=None)) == []

    def test_terminal_reply_when_full_and_idle_drops_oldest(self):
        import queue

        from pacsys.acnet.errors import ACNET_DISCONNECTED

        dpm = DPMAcnet()
        dpm._reply_queue = queue.Queue(maxsize=1)
        handler = self._open(dpm)
        dpm._reply_queue.put(DPMReading(ref_id=1))
        handler(self._terminal_reply(ACNET_DISCONNECTED))
        assert list(dpm.readings(timeout=0.5)) == []

    def test_nonnegative_terminal_reply_ends_stream_cleanly(self):
        dpm = DPMAcnet()
        handler = self._open(dpm)

        handler(self._terminal_reply(0))

        assert list(dpm.readings(timeout=0.5)) == []
        assert dpm._terminal_status == 0

    def test_read_reports_termination_not_timeout(self):
        from pacsys.acnet.errors import ACNET_DISCONNECTED

        dpm = DPMAcnet()
        handler = self._open(dpm)
        dpm._active = True
        dpm._send_request = MagicMock(return_value=Status_reply())
        handler(self._terminal_reply(ACNET_DISCONNECTED))

        with pytest.raises(DPMError) as exc_info:
            dpm.read("M:OUTTMP", timeout=0.5)
        assert exc_info.value.status == ACNET_DISCONNECTED

    def test_close_wakes_blocked_iterator_and_connect_resets(self):
        from pacsys.acnet.errors import ACNET_CANCELLED

        dpm = DPMAcnet()
        dpm._reply_queue.put(DPMReading(ref_id=1))
        got = []
        t = threading.Thread(target=lambda: got.extend(dpm.readings(timeout=None)), daemon=True)
        t.start()
        time.sleep(0.05)
        dpm.close()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert dpm._terminal_status == ACNET_CANCELLED
        dpm._dev_list[1] = "M:OUTTMP"
        dpm._reply_queue.put(DPMReading(ref_id=1))  # stale item left behind

        with (
            patch("pacsys.acnet.dpm_acnet.AcnetConnectionTCP"),
            patch.object(dpm, "_find_dpm"),
            patch.object(dpm, "_open_list"),
        ):
            dpm.connect()
            assert dpm._terminal_status is None
            assert dpm._reply_queue.empty()
            assert dpm._dev_list == {}
            with pytest.raises(RuntimeError, match="already connected"):
                dpm.connect()
