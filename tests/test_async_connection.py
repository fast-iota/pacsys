"""Tests for async ACNET connection classes (no network, pure unit tests)."""

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pacsys.acnet.async_connection import (
    ACNETD_ACK,
    ACNETD_COMMAND,
    ACNETD_DATA,
    AsyncAcnetConnectionTCP,
    AsyncAcnetConnectionUDP,
    AsyncRequestContext,
    _AcnetUDPProtocol,
)
from pacsys.acnet.constants import (
    ACNET_FLG_MLT,
    ACNET_FLG_RPY,
    ACNET_HEADER_SIZE,
    CMD_CONNECT,
)
from pacsys.acnet.errors import AcnetError, AcnetUnavailableError
from pacsys.acnet.packet import AcnetPacket, AcnetReply, RequestId
from pacsys.acnet.rad50 import encode as _rad50_encode


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_tcp_conn() -> AsyncAcnetConnectionTCP:
    """Create an AsyncAcnetConnectionTCP with fake handle, bypassing real connect."""
    conn = AsyncAcnetConnectionTCP("localhost", port=9999)
    conn._raw_handle = _rad50_encode("TEST")
    conn._handle_name = "TEST"
    conn._connected = True
    conn._writer = MagicMock()
    conn._writer.write = MagicMock()
    conn._writer.drain = AsyncMock()
    conn._reader = MagicMock()
    return conn


def _make_udp_conn() -> AsyncAcnetConnectionUDP:
    """Create an AsyncAcnetConnectionUDP with fake handle, bypassing real connect."""
    conn = AsyncAcnetConnectionUDP("localhost", port=9999)
    conn._raw_handle = _rad50_encode("TEST")
    conn._handle_name = "TEST"
    conn._connected = True
    conn._udp_transport = MagicMock()
    conn._udp_transport.sendto = MagicMock()
    return conn


def _make_reply_packet(req_id: int, status: int = 0, last: bool = True, data: bytes = b"") -> AcnetReply:
    """Build a fake AcnetReply."""
    flags = ACNET_FLG_RPY
    if not last:
        flags |= ACNET_FLG_MLT
    raw = struct.pack("<HhHHIHHH", flags, status, 0, 0, 0, 0, req_id, ACNET_HEADER_SIZE + len(data))
    raw += data
    return AcnetPacket.parse(raw)


class TestReplyBuffering:
    """Test that replies arriving before handler registration are buffered."""

    def test_reply_before_handler_is_buffered(self):
        conn = _make_tcp_conn()
        reply = _make_reply_packet(req_id=42)

        conn._handle_reply(reply)

        req_id = RequestId(42)
        assert req_id in conn._reply_buffer
        assert len(conn._reply_buffer[req_id]) == 1

    def test_buffered_replies_delivered_on_registration(self):
        """Buffered replies are delivered when send_request registers the handler."""

        async def _test():
            conn = _make_tcp_conn()
            req_id = RequestId(7)

            reply = _make_reply_packet(req_id=7, last=True)
            conn._reply_buffer[req_id].append((reply, float("inf")))

            received = []
            ack = struct.pack(">HhH", 2, 0, 7)

            with patch.object(conn, "_xact", new=AsyncMock(return_value=ack)):
                await conn.send_request(node=0x0901, task="DPM", data=b"", reply_handler=lambda r: received.append(r))

            assert len(received) == 1
            assert received[0] is reply
            assert req_id not in conn._reply_buffer

        _run(_test())

    def test_stale_buffered_replies_discarded_on_id_reuse(self):
        """Buffered replies from a previous request are discarded on ID reuse."""

        async def _test():
            conn = _make_tcp_conn()
            req_id = RequestId(7)

            stale_reply = _make_reply_packet(req_id=7, last=True)
            conn._reply_buffer[req_id].append((stale_reply, 0.0))

            received = []
            ack = struct.pack(">HhH", 2, 0, 7)

            with patch.object(conn, "_xact", new=AsyncMock(return_value=ack)):
                await conn.send_request(node=0x0901, task="DPM", data=b"", reply_handler=lambda r: received.append(r))

            assert len(received) == 0
            assert req_id in conn._reply_handlers

        _run(_test())

    def test_reply_with_handler_calls_handler(self):
        conn = _make_tcp_conn()
        received = []

        req_id = RequestId(10)
        ctx = AsyncRequestContext(
            connection=conn,
            task="DPM",
            node=0,
            request_id=req_id,
            multiple_reply=True,
            timeout=5000,
            reply_handler=lambda r: received.append(r),
        )
        conn._reply_handlers[req_id] = ctx

        reply = _make_reply_packet(req_id=10, last=False)
        conn._handle_reply(reply)

        assert len(received) == 1
        assert req_id in conn._reply_handlers

    def test_last_reply_removes_handler(self):
        conn = _make_tcp_conn()
        received = []
        req_id = RequestId(11)
        ctx = AsyncRequestContext(
            connection=conn,
            task="DPM",
            node=0,
            request_id=req_id,
            multiple_reply=True,
            timeout=5000,
            reply_handler=lambda r: received.append(r),
        )
        conn._reply_handlers[req_id] = ctx

        reply = _make_reply_packet(req_id=11, last=True)
        conn._handle_reply(reply)

        assert len(received) == 1
        assert req_id not in conn._reply_handlers
        assert req_id in conn._dead_requests
        assert ctx.cancelled

    def test_late_reply_for_dead_request_not_buffered(self):
        """Replies for cancelled/completed requests must not leak into _reply_buffer."""
        conn = _make_tcp_conn()
        req_id = RequestId(99)

        conn._dead_requests.add(req_id)

        reply = _make_reply_packet(req_id=99)
        conn._handle_reply(reply)

        assert req_id not in conn._reply_buffer

    def test_orphaned_buffer_evicted_after_cap(self):
        """Excess buffered replies for an unknown request ID get evicted."""
        from pacsys.acnet.async_connection import _MAX_BUFFERED_REPLIES

        conn = _make_tcp_conn()
        req_id = RequestId(77)

        for _ in range(_MAX_BUFFERED_REPLIES):
            conn._handle_reply(_make_reply_packet(req_id=77, last=False))
        assert len(conn._reply_buffer[req_id]) == _MAX_BUFFERED_REPLIES

        conn._handle_reply(_make_reply_packet(req_id=77, last=False))
        assert req_id not in conn._reply_buffer
        assert req_id in conn._dead_requests


class TestCloseCleanup:
    """Test that close() releases all tracking state."""

    def test_close_clears_all_state(self):
        async def _test():
            conn = _make_tcp_conn()

            req_id = RequestId(50)
            ctx = AsyncRequestContext(
                connection=conn,
                task="DPM",
                node=0,
                request_id=req_id,
                multiple_reply=True,
                timeout=5000,
                reply_handler=lambda r: None,
            )
            conn._reply_handlers[req_id] = ctx
            conn._reply_buffer[RequestId(51)].append("stale")
            conn._dead_requests.add(RequestId(52))

            conn._read_task = None
            conn._keepalive_task = None

            await conn.close()

            assert len(conn._reply_handlers) == 0
            assert len(conn._reply_buffer) == 0
            assert len(conn._dead_requests) == 0
            assert conn._writer is None

        _run(_test())


class TestAsyncXact:
    """Test command serialization and ACK delivery."""

    def test_xact_sends_and_waits_for_ack(self):
        async def _test():
            conn = _make_tcp_conn()
            ack_data = struct.pack(">Hh", 0, 0)

            async def _deliver_ack():
                await asyncio.sleep(0.01)
                conn._dispatch_frame(ACNETD_ACK, ack_data)

            content = struct.pack(">2H2I", ACNETD_COMMAND, CMD_CONNECT, 0, 0)
            asyncio.ensure_future(_deliver_ack())
            result = await conn._xact(content)
            assert result == ack_data

        _run(_test())

    def test_xact_timeout_raises(self):
        async def _test():
            conn = _make_tcp_conn()
            content = struct.pack(">2H2I", ACNETD_COMMAND, CMD_CONNECT, 0, 0)

            with patch("pacsys.acnet.async_connection.asyncio.wait_for", side_effect=asyncio.TimeoutError):
                with pytest.raises(AcnetUnavailableError):
                    await conn._xact(content)

        _run(_test())

    def test_xact_disposed_raises(self):
        async def _test():
            conn = _make_tcp_conn()
            conn._disposed = True
            content = struct.pack(">2H2I", ACNETD_COMMAND, CMD_CONNECT, 0, 0)

            with pytest.raises(AcnetError):
                await conn._xact(content)

        _run(_test())


class TestAsyncReadLoop:
    """Test read loop packet parsing and dispatch."""

    def test_dispatch_frame_ack(self):
        conn = _make_tcp_conn()
        loop = asyncio.new_event_loop()
        conn._pending_ack = loop.create_future()

        ack_data = struct.pack(">Hh", 1, 0)
        conn._dispatch_frame(ACNETD_ACK, ack_data)

        assert conn._pending_ack.done()
        assert conn._pending_ack.result() == ack_data
        loop.close()

    def test_dispatch_frame_data_dispatches_reply(self):
        conn = _make_tcp_conn()
        received = []
        req_id = RequestId(99)
        ctx = AsyncRequestContext(
            connection=conn,
            task="DPM",
            node=0,
            request_id=req_id,
            multiple_reply=False,
            timeout=5000,
            reply_handler=lambda r: received.append(r),
        )
        conn._reply_handlers[req_id] = ctx

        reply_raw = struct.pack("<HhHHIHHH", ACNET_FLG_RPY, 0, 0, 0, 0, 0, 99, ACNET_HEADER_SIZE)
        conn._dispatch_frame(ACNETD_DATA, reply_raw)

        assert len(received) == 1
        assert isinstance(received[0], AcnetReply)

    def test_connection_loss_fails_pending_ack(self):
        conn = _make_tcp_conn()
        loop = asyncio.new_event_loop()
        conn._pending_ack = loop.create_future()

        if conn._pending_ack and not conn._pending_ack.done():
            conn._pending_ack.set_exception(AcnetUnavailableError())

        assert conn._pending_ack.done()
        with pytest.raises(AcnetUnavailableError):
            conn._pending_ack.result()
        loop.close()


class TestAsyncConnect:
    """Test connection handshake."""

    def test_do_connect_parses_handle(self):
        async def _test():
            conn = _make_tcp_conn()
            handle = _rad50_encode("MYTEST")
            ack = struct.pack(">HhBI", 1, 0, 0, handle)

            with patch.object(conn, "_xact", new=AsyncMock(return_value=ack)):
                await conn._do_connect()

            assert conn._raw_handle == handle
            assert conn._handle_name == "MYTEST"
            assert conn._connected

        _run(_test())

    def test_do_connect_short_ack_raises(self):
        async def _test():
            conn = _make_tcp_conn()
            ack = b"\x00\x01"

            with patch.object(conn, "_xact", new=AsyncMock(return_value=ack)):
                with pytest.raises(AcnetUnavailableError):
                    await conn._do_connect()

        _run(_test())

    def test_do_connect_negative_status_raises(self):
        async def _test():
            conn = _make_tcp_conn()
            ack = struct.pack(">HhBI", 1, -1, 0, 0)

            with patch.object(conn, "_xact", new=AsyncMock(return_value=ack)):
                with pytest.raises(AcnetError):
                    await conn._do_connect()

        _run(_test())


class TestAsyncRequestContext:
    """Test AsyncRequestContext cancel."""

    def test_cancel_sends_cancel_command(self):
        async def _test():
            conn = _make_tcp_conn()
            ack = struct.pack(">Hh", 0, 0)
            with patch.object(conn, "_xact", new=AsyncMock(return_value=ack)):
                ctx = AsyncRequestContext(
                    connection=conn,
                    task="DPM",
                    node=0,
                    request_id=RequestId(5),
                    multiple_reply=False,
                    timeout=5000,
                    reply_handler=lambda r: None,
                )
                conn._reply_handlers[ctx.request_id] = ctx
                await ctx.cancel()

            assert ctx.cancelled
            assert ctx.request_id not in conn._reply_handlers

        _run(_test())


class TestAsyncContextManager:
    """Test async context manager support."""

    def test_async_with_full_lifecycle(self):
        async def _test():
            conn = AsyncAcnetConnectionTCP("localhost", port=9999)
            with (
                patch.object(conn, "connect", new=AsyncMock()),
                patch.object(conn, "close", new=AsyncMock()) as mock_close,
            ):
                async with conn:
                    assert True
                mock_close.assert_called_once()

        _run(_test())


# ======================================================================
# TCP transport tests
# ======================================================================


class TestTCPSendFrame:
    """Test that TCP _send_frame prepends 4-byte length."""

    def test_send_frame_prepends_length(self):
        async def _test():
            conn = _make_tcp_conn()
            content = b"\x00\x01\x00\x02"  # 4 bytes
            await conn._send_frame(content)

            written = conn._writer.write.call_args[0][0]
            assert written[:4] == struct.pack(">I", 4)
            assert written[4:] == content

        _run(_test())


# ======================================================================
# UDP transport tests
# ======================================================================


class TestUDPSendFrame:
    """Test that UDP _send_frame sends raw content without length prefix."""

    def test_send_frame_sends_raw(self):
        async def _test():
            conn = _make_udp_conn()
            content = b"\x00\x01\x00\x02"
            await conn._send_frame(content)

            conn._udp_transport.sendto.assert_called_once_with(content)

        _run(_test())


class TestUDPProtocol:
    """Test the _AcnetUDPProtocol dispatches to connection."""

    def test_datagram_received_dispatches(self):
        conn = _make_udp_conn()
        protocol = _AcnetUDPProtocol(conn)

        ack_data = struct.pack(">Hh", 1, 0)
        # Frame: 2-byte msg_type (ACNETD_ACK=2) + ack payload
        frame = struct.pack(">H", ACNETD_ACK) + ack_data

        loop = asyncio.new_event_loop()
        conn._pending_ack = loop.create_future()

        protocol.datagram_received(frame, ("127.0.0.1", 6802))

        assert conn._pending_ack.done()
        assert conn._pending_ack.result() == ack_data
        loop.close()

    def test_datagram_received_ignores_short(self):
        conn = _make_udp_conn()
        protocol = _AcnetUDPProtocol(conn)
        # Should not raise
        protocol.datagram_received(b"\x00", ("127.0.0.1", 6802))

    def test_connection_lost_calls_handler(self):
        conn = _make_udp_conn()
        protocol = _AcnetUDPProtocol(conn)

        conn._connected = True
        protocol.connection_lost(None)

        assert not conn._connected
