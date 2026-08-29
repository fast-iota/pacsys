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
    _AcnetUDPCommandProtocol,
    _AcnetUDPDataProtocol,
)
from pacsys.acnet.constants import (
    ACNET_FLG_MLT,
    ACNET_FLG_REQ,
    ACNET_FLG_RPY,
    ACNET_HEADER_SIZE,
    CMD_BLOCK_REQUESTS,
    CMD_CONNECT,
    CMD_DISCONNECT_SINGLE,
    CMD_RECEIVE_REQUESTS,
)
from pacsys.acnet.errors import ACNET_DISCONNECTED, AcnetError, AcnetUnavailableError
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
    conn._writer.wait_closed = AsyncMock()
    conn._reader = MagicMock()
    return conn


def _make_udp_conn() -> AsyncAcnetConnectionUDP:
    """Create an AsyncAcnetConnectionUDP with fake handle, bypassing real connect."""
    conn = AsyncAcnetConnectionUDP("localhost", port=9999, name="TEST")
    conn._raw_handle = _rad50_encode("TEST")
    conn._handle_name = "TEST"
    conn._connected = True
    conn._cmd_transport = MagicMock()
    conn._cmd_transport.sendto = MagicMock()
    conn._data_transport = MagicMock()
    conn._data_transport.get_extra_info.return_value = ("127.0.0.1", 43210)
    return conn


@pytest.mark.parametrize("name", ["TOOLONG", "BAD_NAME"])
def test_connection_names_must_be_lossless_rad50(name):
    from pacsys.acnet.connection_sync import AcnetConnectionTCP

    with pytest.raises(ValueError):
        AsyncAcnetConnectionTCP("localhost", name=name)
    with pytest.raises(ValueError):
        AcnetConnectionTCP("localhost", name=name)


def _make_reply_packet(req_id: int, status: int = 0, last: bool = True, data: bytes = b"") -> AcnetReply:
    """Build a fake AcnetReply."""
    flags = ACNET_FLG_RPY
    if not last:
        flags |= ACNET_FLG_MLT
    raw = struct.pack("<HhHHIHHH", flags, status, 0, 0, 0, 0, req_id, ACNET_HEADER_SIZE + len(data))
    raw += data
    return AcnetPacket.parse(raw)


class TestReceivingCommands:
    def test_start_commits_after_success_and_is_idempotent(self):
        async def run_test():
            conn = _make_tcp_conn()
            conn._xact = AsyncMock(return_value=struct.pack(">Hh", 0, 0))

            await conn._start_receiving()
            await conn._start_receiving()

            assert conn._receiving
            conn._xact.assert_awaited_once()
            command = struct.unpack_from(">H", conn._xact.call_args.args[0])[0]
            assert command == CMD_RECEIVE_REQUESTS

        _run(run_test())


class TestDisconnectSingle:
    def test_success_clears_task_state(self):
        async def run_test():
            conn = _make_tcp_conn()
            conn._receiving = True
            outgoing = MagicMock()
            outgoing._cancelled = False
            conn._reply_handlers[RequestId(1)] = outgoing
            conn._reply_buffer[RequestId(2)].append((MagicMock(), 0))
            raw = struct.pack("<HhHHIHHH", ACNET_FLG_REQ, 0, 0, 0, 0, 0, 42, 18)
            incoming = AcnetPacket.parse(raw)
            conn._requests_in[incoming.reply_id] = incoming
            keepalive = asyncio.create_task(asyncio.sleep(60))
            conn._keepalive_task = keepalive
            conn._xact = AsyncMock(return_value=struct.pack(">Hh", 0, 0))

            await conn.disconnect_single()

            assert not conn.connected
            assert not conn._receiving
            assert outgoing._cancelled
            assert not conn._reply_handlers
            assert not conn._reply_buffer
            assert incoming.cancelled
            assert not conn._requests_in
            assert keepalive.cancelled()
            command = struct.unpack_from(">H", conn._xact.call_args.args[0])[0]
            assert command == CMD_DISCONNECT_SINGLE

        _run(run_test())

    @pytest.mark.parametrize("ack", [b"", struct.pack(">Hh", 0, -1)])
    def test_ack_failure_preserves_state(self, ack):
        async def run_test():
            conn = _make_tcp_conn()
            conn._receiving = True
            outgoing = MagicMock()
            outgoing._cancelled = False
            conn._reply_handlers[RequestId(1)] = outgoing
            conn._xact = AsyncMock(return_value=ack)

            with pytest.raises(AcnetError):
                await conn.disconnect_single()

            assert conn.connected
            assert conn._receiving
            assert conn._reply_handlers[RequestId(1)] is outgoing
            assert not outgoing._cancelled

        _run(run_test())

    def test_transport_failure_preserves_state(self):
        async def run_test():
            conn = _make_tcp_conn()
            conn._receiving = True
            conn._xact = AsyncMock(side_effect=AcnetUnavailableError)

            with pytest.raises(AcnetUnavailableError):
                await conn.disconnect_single()

            assert conn.connected
            assert conn._receiving

        _run(run_test())


class TestReceivingFailures:
    @pytest.mark.parametrize("ack", [b"", struct.pack(">Hh", 0, -1)])
    def test_start_failure_preserves_retry_state(self, ack):
        async def run_test():
            conn = _make_tcp_conn()
            conn._xact = AsyncMock(return_value=ack)

            with pytest.raises(AcnetError):
                await conn._start_receiving()

            assert not conn._receiving

        _run(run_test())

    def test_stop_commits_after_success(self):
        async def run_test():
            conn = _make_tcp_conn()
            conn._receiving = True
            conn._xact = AsyncMock(return_value=struct.pack(">Hh", 0, 0))

            await conn._stop_receiving()

            assert not conn._receiving
            command = struct.unpack_from(">H", conn._xact.call_args.args[0])[0]
            assert command == CMD_BLOCK_REQUESTS

        _run(run_test())

    @pytest.mark.parametrize("ack", [b"", struct.pack(">Hh", 0, -1)])
    def test_stop_failure_preserves_retry_state(self, ack):
        async def run_test():
            conn = _make_tcp_conn()
            conn._receiving = True
            conn._xact = AsyncMock(return_value=ack)

            with pytest.raises(AcnetError):
                await conn._stop_receiving()

            assert conn._receiving

        _run(run_test())


class TestReplyBuffering:
    """Test that replies arriving before handler registration are buffered."""

    def test_reply_before_handler_is_buffered(self):
        """During a registration window, unknown-ID replies are buffered."""
        conn = _make_tcp_conn()
        conn._pending_sends = 1
        reply = _make_reply_packet(req_id=42)

        conn._handle_reply(reply)

        req_id = RequestId(42)
        assert req_id in conn._reply_buffer
        assert len(conn._reply_buffer[req_id]) == 1

    def test_reply_without_pending_send_dropped(self):
        """Outside any registration window an unknown-ID reply is provably stale."""
        conn = _make_tcp_conn()
        assert conn._pending_sends == 0

        conn._handle_reply(_make_reply_packet(req_id=42))

        assert RequestId(42) not in conn._reply_buffer

    def test_buffered_replies_delivered_on_registration(self):
        """Buffered replies are delivered when send_request registers the handler."""

        async def _test():
            conn = _make_tcp_conn()
            req_id = RequestId(7)

            reply = _make_reply_packet(req_id=7, last=True)
            conn._reply_buffer[req_id].append((reply, 1))  # post-ack seq (ack_seq stays 0)

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
            conn._reply_buffer[req_id].append((stale_reply, 0))  # pre-ack seq

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
        assert ctx.cancelled

    def test_late_reply_after_completion_not_buffered(self):
        """Replies for completed requests must not leak into _reply_buffer when idle."""
        conn = _make_tcp_conn()
        req_id = RequestId(99)

        reply = _make_reply_packet(req_id=99)
        conn._handle_reply(reply)

        assert req_id not in conn._reply_buffer

    def test_orphaned_buffer_fifo_eviction_keeps_newest(self):
        """Overflow evicts oldest entries; the newest (possibly fresh) survive."""
        from pacsys.acnet.async_connection import _MAX_BUFFERED_REPLIES

        conn = _make_tcp_conn()
        conn._pending_sends = 1
        req_id = RequestId(77)

        replies = [_make_reply_packet(req_id=77, last=False) for _ in range(_MAX_BUFFERED_REPLIES + 1)]
        for r in replies:
            conn._handle_reply(r)

        buf = conn._reply_buffer[req_id]
        assert len(buf) == _MAX_BUFFERED_REPLIES
        kept = [r for r, _ in buf]
        assert replies[0] not in kept  # oldest evicted
        assert kept[-1] is replies[-1]  # newest kept


class TestRequestIdReuseRace:
    """acnetd recycles request IDs; a fresh first reply batched with the
    SEND_REQUEST ack must be delivered, while stale replies from the ID's
    previous life must be discarded (ack-sequence causality)."""

    def _complete_request(self, conn, req_id_int):
        """Run a request to completion so its ID is retired."""
        received = []
        ctx = AsyncRequestContext(
            connection=conn,
            task="DPM",
            node=0,
            request_id=RequestId(req_id_int),
            multiple_reply=False,
            timeout=5000,
            reply_handler=lambda r: received.append(r),
        )
        conn._reply_handlers[RequestId(req_id_int)] = ctx
        conn._handle_reply(_make_reply_packet(req_id=req_id_int, last=True))
        assert RequestId(req_id_int) not in conn._reply_handlers
        return received

    def test_reused_id_reply_batched_with_ack_delivered(self):
        """The core race: ID retired, reused, first reply arrives before
        send_request resumes from the ack."""

        async def _test():
            conn = _make_tcp_conn()
            self._complete_request(conn, 7)

            fresh = _make_reply_packet(req_id=7, last=True)
            ack = struct.pack(">HhH", 2, 0, 7)

            async def fake_xact(content, timeout=5.0):
                # Simulate the read loop processing ACK then the batched reply
                # before send_request resumes.
                conn._ack_recv_seq = conn._recv_seq
                conn._handle_reply(fresh)
                return ack

            received = []
            with patch.object(conn, "_xact", new=fake_xact):
                await conn.send_request(node=0x0901, task="DPM", data=b"", reply_handler=lambda r: received.append(r))

            assert received == [fresh]

        _run(_test())

    def test_stale_reply_during_send_discarded_fresh_kept(self):
        """A stale reply landing before our ack is discarded even though it
        arrives during the registration window; the post-ack reply survives."""

        async def _test():
            conn = _make_tcp_conn()
            self._complete_request(conn, 7)

            stale = _make_reply_packet(req_id=7, last=True)
            fresh = _make_reply_packet(req_id=7, last=True)
            ack = struct.pack(">HhH", 2, 0, 7)

            async def fake_xact(content, timeout=5.0):
                conn._handle_reply(stale)  # e.g. arrived while waiting on _cmd_lock
                conn._ack_recv_seq = conn._recv_seq
                conn._handle_reply(fresh)
                return ack

            received = []
            with patch.object(conn, "_xact", new=fake_xact):
                await conn.send_request(node=0x0901, task="DPM", data=b"", reply_handler=lambda r: received.append(r))

            assert received == [fresh]  # stale never misattributed

        _run(_test())

    def test_udp_reply_overtaking_ack_delivered(self):
        """UDP: acnetd sends acks and data on separate sockets, so a fresh
        reply datagram can overtake its ack — it must not be dropped as stale."""

        async def _test():
            conn = _make_udp_conn()
            self._complete_request(conn, 7)

            fresh = _make_reply_packet(req_id=7, last=True)
            ack = struct.pack(">HhH", 2, 0, 7)

            async def fake_xact(content, timeout=5.0):
                conn._handle_reply(fresh)  # datagram overtakes the ack
                conn._ack_recv_seq = conn._recv_seq
                return ack

            received = []
            with patch.object(conn, "_xact", new=fake_xact):
                await conn.send_request(node=0x0901, task="DPM", data=b"", reply_handler=lambda r: received.append(r))

            assert received == [fresh]

        _run(_test())

    def test_stray_buffers_cleared_when_windows_close(self):
        async def _test():
            conn = _make_tcp_conn()
            stray = _make_reply_packet(req_id=99, last=False)
            ack = struct.pack(">HhH", 2, 0, 7)

            async def fake_xact(content, timeout=5.0):
                conn._ack_recv_seq = conn._recv_seq
                conn._handle_reply(stray)  # unrelated ID, buffered during window
                return ack

            with patch.object(conn, "_xact", new=fake_xact):
                await conn.send_request(node=0x0901, task="DPM", data=b"", reply_handler=lambda r: None)

            assert conn._pending_sends == 0
            assert len(conn._reply_buffer) == 0

        _run(_test())

    def test_window_closes_on_send_failure(self):
        async def _test():
            conn = _make_tcp_conn()
            with patch.object(conn, "_xact", new=AsyncMock(side_effect=AcnetUnavailableError)):
                with pytest.raises(AcnetUnavailableError):
                    await conn.send_request(node=0x0901, task="DPM", data=b"", reply_handler=lambda r: None)
            assert conn._pending_sends == 0

        _run(_test())


class TestConnectFailureCleanup:
    """Failed connect() must not leak the transport or read task."""

    def _make_conn_with_fakes(self):
        conn = AsyncAcnetConnectionTCP("localhost", port=9999)
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        async def fake_open():
            conn._reader = MagicMock()
            conn._writer = writer

        def fake_start_read_loop():
            conn._read_task = asyncio.ensure_future(asyncio.sleep(3600))

        conn._open_transport = fake_open
        conn._start_read_loop = fake_start_read_loop
        return conn, writer

    def test_do_connect_failure_closes_transport_and_read_task(self):
        async def _test():
            conn, writer = self._make_conn_with_fakes()

            async def fail_connect():
                raise AcnetError(-1, "CONNECT rejected")

            conn._do_connect = fail_connect
            with pytest.raises(AcnetError, match="CONNECT rejected"):
                await conn.connect()

            read_task = conn._read_task
            assert read_task is not None and read_task.done()
            writer.close.assert_called_once()
            assert conn._writer is None
            assert conn._disposed
            # Caller-side double close is safe
            await conn.close()
            writer.close.assert_called_once()
            # Disposed object refuses reconnect
            with pytest.raises(AcnetError, match="disposed"):
                await conn.connect()

        _run(_test())

    def test_do_connect_cancellation_cleans_up(self):
        async def _test():
            conn, writer = self._make_conn_with_fakes()

            async def cancelled_connect():
                raise asyncio.CancelledError

            conn._do_connect = cancelled_connect
            with pytest.raises(asyncio.CancelledError):
                await conn.connect()
            writer.close.assert_called_once()
            assert conn._writer is None

        _run(_test())


class TestSyncConnectFailureCleanup:
    """Sync wrapper: failed connect() must stop the reactor and stay retryable."""

    def test_core_connect_failure_stops_reactor_then_retry_works(self):
        from pacsys.acnet.connection_sync import AcnetConnectionTCP

        conn = AcnetConnectionTCP("localhost", port=9999)
        fail_core = MagicMock()
        fail_core.connect = AsyncMock(side_effect=AcnetUnavailableError)
        fail_core.close = AsyncMock()
        ok_core = MagicMock()
        ok_core.connect = AsyncMock()
        ok_core.close = AsyncMock()
        cores = iter([fail_core, ok_core])
        conn._create_async = lambda: next(cores)

        with pytest.raises(AcnetUnavailableError):
            conn.connect()
        fail_core.close.assert_awaited()
        assert conn._async is None
        assert conn._loop is None
        assert conn._reactor_thread is None

        conn.connect()
        assert conn._async is ok_core
        assert conn._reactor_thread is not None and conn._reactor_thread.is_alive()
        conn.close()
        assert conn._reactor_thread is None


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

            conn._read_task = None
            conn._keepalive_task = None

            await conn.close()

            assert len(conn._reply_handlers) == 0
            assert len(conn._reply_buffer) == 0
            assert conn._writer is None

        _run(_test())

    @staticmethod
    def _arm(conn, received):
        ctx = AsyncRequestContext(
            connection=conn,
            task="DPM",
            node=0,
            request_id=RequestId(50),
            multiple_reply=True,
            timeout=5000,
            reply_handler=received.append,
        )
        conn._reply_handlers[RequestId(50)] = ctx
        return ctx

    def test_close_fails_handlers_when_already_disconnected(self):
        """Connection already dropped (_xact ack-timeout nulled the writer), so
        _do_disconnect never runs - handlers must still get ACNET_DISCONNECTED
        or a multi-reply consumer waits forever."""

        async def _test():
            conn = _make_tcp_conn()
            conn._connected = False
            conn._writer = None
            conn._read_task = None
            conn._keepalive_task = None

            received = []
            ctx = self._arm(conn, received)

            await conn.close()

            assert len(conn._reply_handlers) == 0
            assert [r.status for r in received] == [ACNET_DISCONNECTED]
            assert ctx.cancelled

        _run(_test())

    def test_close_delivers_synthetic_reply_exactly_once(self):
        """Connected path calls _fail_reply_handlers via _do_disconnect; the
        unconditional call in close() must not double-deliver."""

        async def _test():
            conn = _make_tcp_conn()
            conn._read_task = None
            conn._keepalive_task = None
            conn._xact = AsyncMock(side_effect=Exception("boom"))

            received = []
            self._arm(conn, received)

            await conn.close()

            assert len(received) == 1
            assert len(conn._reply_handlers) == 0

        _run(_test())

    def test_close_fails_handlers_even_if_transport_close_raises(self):
        async def _test():
            conn = _make_tcp_conn()
            conn._connected = False
            conn._read_task = None
            conn._keepalive_task = None
            conn._writer.wait_closed = AsyncMock(side_effect=RuntimeError("transport blew up"))

            received = []
            self._arm(conn, received)

            with pytest.raises(RuntimeError):
                await conn.close()

            assert len(conn._reply_handlers) == 0
            assert [r.status for r in received] == [ACNET_DISCONNECTED]

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

            content = struct.pack(">H2I", CMD_CONNECT, 0, 0)
            ack_task = asyncio.create_task(_deliver_ack())
            result = await conn._xact(content)
            await ack_task
            assert result == ack_data

        _run(_test())

    def test_xact_timeout_raises(self):
        async def _test():
            conn = _make_tcp_conn()
            content = struct.pack(">H2I", CMD_CONNECT, 0, 0)

            with patch("pacsys.acnet.async_connection.asyncio.wait_for", side_effect=asyncio.TimeoutError):
                with pytest.raises(AcnetUnavailableError):
                    await conn._xact(content)

        _run(_test())

    def test_xact_disposed_raises(self):
        async def _test():
            conn = _make_tcp_conn()
            conn._disposed = True
            content = struct.pack(">H2I", CMD_CONNECT, 0, 0)

            with pytest.raises(AcnetError):
                await conn._xact(content)

        _run(_test())

    def test_xact_cancelled_during_send_drops_transport(self):
        """Cancel while the frame is being written must not leave _pending_ack for the next command."""

        async def _test():
            conn = _make_tcp_conn()
            sending = asyncio.Event()

            async def slow_drain():
                sending.set()
                await asyncio.sleep(10)

            conn._writer.drain = slow_drain
            content = struct.pack(">H2I", CMD_CONNECT, 0, 0)
            task = asyncio.create_task(conn._xact(content))
            await sending.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert conn._pending_ack is None
            assert conn._writer is None
            assert not conn._connected
            conn._dispatch_frame(ACNETD_ACK, b"OLD-ACK")  # late ACK for the cancelled command
            with pytest.raises(AcnetUnavailableError):  # transport gone: cannot bind it to a new command
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

        conn._on_connection_lost()

        assert not conn._connected
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
    """Test that TCP _send_frame adds the bridge envelope and length."""

    def test_send_frame_prepends_length(self):
        async def _test():
            conn = _make_tcp_conn()
            content = b"\x00\x01\x00\x02"  # 4 bytes
            await conn._send_frame(content)

            written = conn._writer.write.call_args[0][0]
            assert written[:4] == struct.pack(">I", 6)
            assert written[4:6] == struct.pack(">H", ACNETD_COMMAND)
            assert written[6:] == content

        _run(_test())

    def test_send_frame_closed_transport_raises_acnet_error(self):
        async def _test():
            conn = _make_tcp_conn()
            conn._writer = None  # close-race: _close_transport ran, connection object alive
            with pytest.raises(AcnetUnavailableError):
                await conn._send_frame(b"\x00\x01")

        _run(_test())


class TestRequestAckErrorHandling:
    """Fire-and-forget ack must never leak an unretrieved task exception."""

    def test_request_ack_swallows_transport_failure(self, caplog):
        async def _test():
            conn = _make_tcp_conn()
            conn._writer = None
            from pacsys.acnet.packet import ReplyId

            await conn._request_ack(ReplyId(1))  # must not raise

        with caplog.at_level("WARNING"):
            _run(_test())
        assert any("Failed to send request ack" in r.message for r in caplog.records)


# ======================================================================
# UDP transport tests
# ======================================================================


class TestUDPSendFrame:
    """Test that official UDP sends raw command structures."""

    def test_send_frame_sends_raw(self):
        async def _test():
            conn = _make_udp_conn()
            content = b"\x00\x01\x00\x02"
            await conn._send_frame(content)

            conn._cmd_transport.sendto.assert_called_once_with(content)

        _run(_test())

    def test_send_frame_closed_transport_raises_acnet_error(self):
        async def _test():
            conn = _make_udp_conn()
            conn._cmd_transport = None
            with pytest.raises(AcnetUnavailableError):
                await conn._send_frame(b"\x00\x01")

        _run(_test())


class TestUDPProtocol:
    """Test raw ACK and ACNET data datagram dispatch."""

    def test_command_datagram_delivers_raw_ack(self):
        conn = _make_udp_conn()
        protocol = _AcnetUDPCommandProtocol(conn)

        ack_data = struct.pack(">Hh", 1, 0)

        loop = asyncio.new_event_loop()
        conn._pending_ack = loop.create_future()

        protocol.datagram_received(ack_data, ("127.0.0.1", 6802))

        assert conn._pending_ack.done()
        assert conn._pending_ack.result() == ack_data
        loop.close()

    def test_data_datagram_dispatches_multiple_packets(self):
        conn = _make_udp_conn()
        protocol = _AcnetUDPDataProtocol(conn)
        received = []
        for req_id in (7, 8):
            conn._reply_handlers[RequestId(req_id)] = AsyncRequestContext(
                connection=conn,
                task="DPM",
                node=0,
                request_id=RequestId(req_id),
                multiple_reply=False,
                timeout=5000,
                reply_handler=received.append,
            )

        first = struct.pack("<HhHHIHHH", ACNET_FLG_RPY, 0, 0, 0, 0, 0, 7, ACNET_HEADER_SIZE + 1)
        first += b"x\x00"  # odd declared packet length plus alignment padding
        second = struct.pack("<HhHHIHHH", ACNET_FLG_RPY, 0, 0, 0, 0, 0, 8, ACNET_HEADER_SIZE)
        raw = first + second
        protocol.datagram_received(raw, ("127.0.0.1", 6802))

        assert [reply.request_id for reply in received] == [RequestId(7), RequestId(8)]
        assert received[0].data == b"x"

    def test_connection_lost_calls_handler(self):
        conn = _make_udp_conn()
        protocol = _AcnetUDPCommandProtocol(conn)

        conn._connected = True
        protocol.connection_lost(None)

        assert not conn._connected
        assert conn._cmd_transport is None
        assert conn._data_transport is None


class TestUDPConnect:
    def test_connect_uses_raw_daemon_layout_with_real_data_port(self):
        async def _test():
            conn = _make_udp_conn()
            handle = _rad50_encode("UDP001")
            conn._xact = AsyncMock(return_value=struct.pack(">HhBI", 1, 0, 7, handle))

            with patch("pacsys.acnet.async_connection.os.getpid", return_value=1234):
                await conn._do_connect()

            raw = conn._xact.call_args.args[0]
            assert raw == struct.pack(
                ">HIIIH",
                CMD_CONNECT,
                _rad50_encode("TEST"),
                0,
                1234,
                43210,
            )
            assert conn.name == "UDP001"

        _run(_test())

    def test_remote_host_is_rejected(self):
        with pytest.raises(ValueError, match="local-only"):
            AsyncAcnetConnectionUDP("acsys-proxy.fnal.gov")


class TestConnectionLossNotifiesHandlers:
    """Connection loss delivers a final DISCONNECTED reply to multi-reply consumers."""

    def test_connection_lost_delivers_final_disconnected_reply(self):
        conn = _make_tcp_conn()
        received = []
        ctx = AsyncRequestContext(
            connection=conn,
            task="FTPMAN",
            node=1,
            request_id=RequestId(7),
            multiple_reply=True,
            timeout=0,
            reply_handler=received.append,
        )
        conn._reply_handlers[RequestId(7)] = ctx

        conn._on_connection_lost()

        assert not conn._reply_handlers
        assert ctx.cancelled
        assert len(received) == 1
        reply = received[0]
        assert reply.last
        assert reply.status == ACNET_DISCONNECTED
        assert reply.request_id == RequestId(7)

    def test_handler_exception_does_not_block_others(self):
        conn = _make_tcp_conn()
        received = []

        def bad_handler(reply):
            raise RuntimeError("boom")

        for req_id, handler in [(1, bad_handler), (2, received.append)]:
            conn._reply_handlers[RequestId(req_id)] = AsyncRequestContext(
                connection=conn,
                task="T",
                node=1,
                request_id=RequestId(req_id),
                multiple_reply=True,
                timeout=0,
                reply_handler=handler,
            )

        conn._on_connection_lost()

        assert len(received) == 1
