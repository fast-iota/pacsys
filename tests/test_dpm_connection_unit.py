"""
Unit tests for DPMConnection class.

These tests use mocking for socket operations - no real network calls.
For integration tests that actually connect to DPM, see test_dpm_connection.py.
"""

import socket
import struct
import pytest
from unittest import mock

from pacsys.dpm_connection import (
    DPMConnection,
    DPMConnectionError,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    DPM_HANDSHAKE,
    MAX_MESSAGE_SIZE,
)
from pacsys.dpm_protocol import (
    AddToList_request,
    OpenList_reply,
    Scalar_reply,
)


class TestDPMConnectionInit:
    """Tests for DPMConnection initialization."""

    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        conn = DPMConnection()
        assert conn._host == DEFAULT_HOST
        assert conn._port == DEFAULT_PORT
        assert conn._timeout == DEFAULT_TIMEOUT
        assert conn._socket is None
        assert conn._list_id is None
        assert not conn.connected

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        conn = DPMConnection(host="test.example.com", port=1234, timeout=5.0)
        assert conn._host == "test.example.com"
        assert conn._port == 1234
        assert conn._timeout == 5.0

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"host": ""}, "host cannot be empty"),
            ({"port": 0}, "port must be between"),
            ({"port": -1}, "port must be between"),
            ({"port": 65536}, "port must be between"),
            ({"timeout": 0}, "timeout must be positive"),
            ({"timeout": -1.0}, "timeout must be positive"),
        ],
    )
    def test_invalid_init_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DPMConnection(**kwargs)


class TestMessageFraming:
    """Tests for message framing (length prefix encoding/decoding)."""

    def test_length_prefix_encoding(self):
        """Test that length prefix is encoded as big-endian uint32."""
        # Create a mock OpenList reply for the handshake
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 12345
        reply_data = bytes(open_list_reply.marshal())
        reply_length = struct.pack(">I", len(reply_data))

        # Mock socket that returns the framed reply
        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = reply_length + reply_data

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            # Verify handshake was sent
            mock_socket.sendall.assert_called_with(DPM_HANDSHAKE)

            # Verify list_id was extracted
            assert conn.list_id == 12345

    def test_send_message_adds_length_prefix(self):
        """Test that send_message adds 4-byte big-endian length prefix."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 1
        reply_data = bytes(open_list_reply.marshal())
        reply_length = struct.pack(">I", len(reply_data))

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = reply_length + reply_data

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            # Clear the call history from connect()
            mock_socket.sendall.reset_mock()

            # Create and send a message
            add_req = AddToList_request()
            add_req.list_id = 1
            add_req.ref_id = 1
            add_req.drf_request = "M:OUTTMP@I"
            conn.send_message(add_req)

            # Verify sendall was called
            assert mock_socket.sendall.called
            sent_data = mock_socket.sendall.call_args[0][0]

            # Verify length prefix
            sent_length = struct.unpack(">I", sent_data[:4])[0]
            sent_body = sent_data[4:]
            assert sent_length == len(sent_body)

    def test_recv_message_reads_length_prefix(self):
        """Test that recv_message correctly reads length prefix."""
        # Create mock replies
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 1
        open_list_data = bytes(open_list_reply.marshal())
        open_list_frame = struct.pack(">I", len(open_list_data)) + open_list_data

        scalar_reply = Scalar_reply()
        scalar_reply.ref_id = 1
        scalar_reply.timestamp = 12345
        scalar_reply.cycle = 0
        scalar_reply.status = 0
        scalar_reply.data = 42.0
        scalar_data = bytes(scalar_reply.marshal())
        scalar_frame = struct.pack(">I", len(scalar_data)) + scalar_data

        mock_socket = mock.Mock(spec=socket.socket)
        # First recv for connect(), subsequent for recv_message()
        mock_socket.recv.side_effect = [open_list_frame, scalar_frame]

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            # Receive scalar reply
            reply = conn.recv_message()
            assert isinstance(reply, Scalar_reply)
            assert reply.ref_id == 1
            assert reply.data == 42.0


class TestHTTPErrorDetection:
    """Tests for HTTP error detection during handshake."""

    def test_http_error_404(self):
        """Test detection of HTTP 404 error response."""
        http_response = b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nNot Found"

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = http_response

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            with pytest.raises(DPMConnectionError) as exc_info:
                conn.connect()

            assert "HTTP error" in str(exc_info.value)
            assert "404" in str(exc_info.value)

    def test_http_error_503(self):
        """Test detection of HTTP 503 error response."""
        http_response = b"HTTP/1.1 503 Service Unavailable\r\n\r\nService down"

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = http_response

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            with pytest.raises(DPMConnectionError) as exc_info:
                conn.connect()

            assert "HTTP error" in str(exc_info.value)
            assert "503" in str(exc_info.value)

    def test_http_error_partial_response(self):
        """Test handling when HTTP response arrives in parts or with timeout."""
        # First recv returns "HTTP" (first 4 bytes), subsequent reads get rest
        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.side_effect = [
            b"HTTP",  # First 4 bytes read by _recv_exact(4)
            b"/1.1 500 Internal Server Error\r\n\r\n",  # Read by _handle_http_error
            socket.timeout("recv timeout"),  # Stop reading more
        ]

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            with pytest.raises(DPMConnectionError) as exc_info:
                conn.connect()

            assert "HTTP error" in str(exc_info.value)
            assert "500" in str(exc_info.value)


class TestOpenListReplyParsing:
    """Tests for parsing OpenList reply."""

    def test_valid_openlist_reply(self):
        """Test successful parsing of valid OpenList reply."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 99999
        reply_data = bytes(open_list_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = reply_frame

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            assert conn.list_id == 99999
            assert conn.connected

    def test_unexpected_reply_type(self):
        """Test error when server sends unexpected reply type."""
        # Send a Scalar_reply instead of OpenList_reply
        scalar_reply = Scalar_reply()
        scalar_reply.ref_id = 1
        scalar_reply.timestamp = 0
        scalar_reply.cycle = 0
        scalar_reply.status = 0
        scalar_reply.data = 0.0
        reply_data = bytes(scalar_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = reply_frame

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            with pytest.raises(DPMConnectionError) as exc_info:
                conn.connect()

            assert "Expected OpenList reply" in str(exc_info.value)


class TestConnectionLifecycle:
    """Tests for connection lifecycle management."""

    def test_context_manager_connects_and_closes(self):
        """Test that context manager handles connect and close."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 1
        reply_data = bytes(open_list_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = reply_frame

        with mock.patch("socket.socket", return_value=mock_socket):
            with DPMConnection() as conn:
                assert conn.connected
                assert conn.list_id == 1

            # After context exit, should be closed
            assert not conn.connected
            mock_socket.close.assert_called()

    def test_double_connect_raises_error(self):
        """Test that connecting twice raises error."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 1
        reply_data = bytes(open_list_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = reply_frame

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            with pytest.raises(DPMConnectionError, match="Already connected"):
                conn.connect()

    def test_close_multiple_times_safe(self):
        """Test that close() can be called multiple times safely."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 1
        reply_data = bytes(open_list_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = reply_frame

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            conn.close()
            conn.close()  # Should not raise
            conn.close()  # Should not raise

            assert not conn.connected

    def test_send_when_not_connected_raises(self):
        """Test that send_message raises when not connected."""
        conn = DPMConnection()

        with pytest.raises(DPMConnectionError, match="Not connected"):
            conn.send_message(b"test")

    def test_recv_when_not_connected_raises(self):
        """Test that recv_message raises when not connected."""
        conn = DPMConnection()

        with pytest.raises(DPMConnectionError, match="Not connected"):
            conn.recv_message()


class TestTimeoutHandling:
    """Tests for timeout handling."""

    def test_socket_timeout_on_connect(self):
        """Test socket timeout during connection."""
        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.connect.side_effect = socket.timeout("Connection timed out")

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection(timeout=1.0)
            with pytest.raises(DPMConnectionError, match="Failed to connect"):
                conn.connect()

    def test_recv_message_timeout(self):
        """Test recv_message timeout."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 1
        reply_data = bytes(open_list_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.side_effect = [
            reply_frame,  # For connect()
            socket.timeout("recv timeout"),  # For recv_message()
        ]

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            with pytest.raises(TimeoutError, match="Receive timeout"):
                conn.recv_message()


class TestPartialReads:
    """Tests for handling partial reads."""

    def test_partial_read_length_prefix(self):
        """Test handling when length prefix arrives in parts."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 123
        reply_data = bytes(open_list_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        # Simulate partial reads: length comes in 2-byte chunks
        mock_socket.recv.side_effect = [
            reply_frame[:2],
            reply_frame[2:4],
            reply_frame[4:],
        ]

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            assert conn.list_id == 123

    def test_partial_read_message_body(self):
        """Test handling when message body arrives in parts."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 456
        reply_data = bytes(open_list_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        # Split frame into many small chunks
        chunks = [reply_frame[i : i + 5] for i in range(0, len(reply_frame), 5)]
        mock_socket.recv.side_effect = chunks

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            assert conn.list_id == 456


class TestErrorConditions:
    """Tests for various error conditions."""

    def test_invalid_message_length_zero(self):
        """Test error on zero-length message."""
        # Frame with zero length
        frame = struct.pack(">I", 0)

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = frame

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            with pytest.raises(DPMConnectionError, match="Invalid message length"):
                conn.connect()

    def test_invalid_message_length_too_large(self):
        """Test error on message length exceeding maximum."""
        # Frame with very large length (> MAX_MESSAGE_SIZE)
        frame = struct.pack(">I", MAX_MESSAGE_SIZE + 1)

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = frame

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            with pytest.raises(DPMConnectionError, match="Invalid message length"):
                conn.connect()

    def test_connection_closed_by_server(self):
        """Test error when server closes connection."""
        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = b""  # Empty read = connection closed

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            with pytest.raises(DPMConnectionError, match="Connection closed"):
                conn.connect()

    def test_send_message_with_bytes(self):
        """Test sending raw bytes instead of message object."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 1
        reply_data = bytes(open_list_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = reply_frame

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()
            mock_socket.sendall.reset_mock()

            # Send raw bytes
            test_data = b"test message data"
            conn.send_message(test_data)

            sent_data = mock_socket.sendall.call_args[0][0]
            sent_length = struct.unpack(">I", sent_data[:4])[0]
            sent_body = sent_data[4:]
            assert sent_length == len(test_data)
            assert sent_body == test_data

    def test_send_message_invalid_type(self):
        """Test error when sending invalid message type."""
        open_list_reply = OpenList_reply()
        open_list_reply.list_id = 1
        reply_data = bytes(open_list_reply.marshal())
        reply_frame = struct.pack(">I", len(reply_data)) + reply_data

        mock_socket = mock.Mock(spec=socket.socket)
        mock_socket.recv.return_value = reply_frame

        with mock.patch("socket.socket", return_value=mock_socket):
            conn = DPMConnection()
            conn.connect()

            with pytest.raises(TypeError, match="must be a protocol message or bytes"):
                conn.send_message(12345)  # Invalid type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
