"""Tests for the pacsys.acnet module."""

import struct

import pytest

from pacsys.acnet import (
    ACNET_HEADER_SIZE,
    ACNET_NO_NODE,
    ACNET_OK,
    ACNET_PEND,
    ACNET_SUCCESS,
    ACSYS_PROXY_HOST,
    AcnetConnection,
    AcnetConnectionTCP,
    AcnetError,
    AcnetNodeError,
    AcnetPacket,
    AcnetReply,
    AcnetRequest,
    ReplyId,
    RequestId,
    decode,
    decode_stripped,
    encode,
    node_parts,
    node_value,
)
from pacsys.acnet.constants import (
    ACNET_FLG_MLT,
    ACNET_FLG_REQ,
    ACNET_FLG_RPY,
    ACNET_FLG_USM,
)
from pacsys.acnet.errors import (
    ERR_OK,
    ERR_RETRY,
    ERR_TIMEOUT,
    FACILITY_ACNET,
    make_error,
    normalize_error_code,
    parse_error,
    status_message,
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
        """Test against known encoded values from Java implementation."""
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

    def test_success_codes(self):
        """Test success codes."""
        assert ACNET_OK == 0
        assert ACNET_SUCCESS == 0

    def test_decomposed_constants(self):
        """Decomposed constants match error numbers in composite constants."""
        assert FACILITY_ACNET == 1
        assert ERR_OK == 0
        assert ERR_RETRY == -1
        assert ERR_TIMEOUT == -6
        # Verify consistency with composite constants
        _, error_num = parse_error(make_error(1, -1))
        assert error_num == ERR_RETRY
        _, error_num = parse_error(make_error(1, -6))
        assert error_num == ERR_TIMEOUT


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


class TestStatusMessage:
    """Tests for status_message helper."""

    def test_success_returns_none(self):
        assert status_message(0, 0) is None
        assert status_message(1, 0) is None

    def test_error_message(self):
        msg = status_message(1, -6)
        assert msg == "Device error (facility=1, error=-6)"

    def test_warning_message(self):
        msg = status_message(17, 2)
        assert msg == "Warning (facility=17, error=2)"


class TestAcnetError:
    """Tests for AcnetError exception."""

    def test_basic_error(self):
        """Test creating basic error."""
        err = AcnetError(ACNET_PEND, "test message")
        assert err.status == ACNET_PEND
        assert "test message" in str(err)

    def test_node_error(self):
        """Test AcnetNodeError."""
        err = AcnetNodeError("MISSING")
        assert err.status == ACNET_NO_NODE
        assert "MISSING" in str(err)


class TestRequestReplyIds:
    """Tests for RequestId and ReplyId."""

    def test_request_id_equality(self):
        """Test RequestId equality."""
        id1 = RequestId(123)
        id2 = RequestId(123)
        id3 = RequestId(456)

        assert id1 == id2
        assert id1 != id3

    def test_request_id_hash(self):
        """Test RequestId is hashable."""
        ids = {RequestId(1), RequestId(2), RequestId(1)}
        assert len(ids) == 2

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

    def test_cancel_state(self):
        """Test request cancellation state."""
        raw = struct.pack("<HhHHIHHH", ACNET_FLG_REQ, 0, 0, 0, 0, 0, 1, 18)
        request = AcnetPacket.parse(raw)

        assert not request.cancelled
        request.cancel()
        assert request.cancelled

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


class TestConnectionClasses:
    """Tests for connection class instantiation."""

    def test_udp_connection_instantiation(self):
        """Test that AcnetConnection can be instantiated."""
        conn = AcnetConnection("TEST")
        assert conn.name == "TEST"
        assert not conn.connected

    def test_tcp_connection_instantiation(self):
        """Test that AcnetConnectionTCP can be instantiated."""
        conn = AcnetConnectionTCP("localhost", name="TEST")
        # name returns empty string before connecting (handle is assigned by daemon)
        assert conn.name == ""
        assert conn.host == "localhost"
        assert conn.port == 6802
        assert not conn.connected

    def test_tcp_connection_default_host(self):
        """Test default host for TCP connection."""
        conn = AcnetConnectionTCP()
        assert conn.host == ACSYS_PROXY_HOST

    def test_proxy_host_constant(self):
        """Test the proxy host constant."""
        assert ACSYS_PROXY_HOST == "acsys-proxy.fnal.gov"
