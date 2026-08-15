"""
Unit tests for DMQBackend.

Tests cover:
- Backend initialization and capabilities
- Single device read/get
- Multiple device get_many
- Subscribe operations (iterator and callback modes)
- Error handling
- Uses mock pika classes for unit tests (no real RabbitMQ needed)

NOTE: DMQ requires KerberosAuth for all operations. Tests use MockGSSAPIModule
to avoid needing real Kerberos credentials.
"""

import threading
import time
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from unittest import mock

import numpy as np
import pytest
from pika.adapters.select_connection import SelectConnection
from pika.exceptions import ChannelWrongStateError

from pacsys.acnet.errors import ERR_TIMEOUT, FACILITY_DMQ
from pacsys.backends.dmq import (
    DMQBackend,
    _reply_to_reading,
    _resolve_reply,
    _WriteCompletionTracker,
    _WriteSession,
)
from pacsys.backends.dmq_protocol import (
    AnalogAlarmSample_reply,
    BasicControlSample_reply,
    BasicStatusSample_reply,
    BinarySample_reply,
    DigitalAlarmSample_reply,
    DoubleArraySample_reply,
    DoubleSample_reply,
    ErrorSample_reply,
    IntegerSample_reply,
    StringArraySample_reply,
    StringSample_reply,
)
from pacsys.drf_utils import prepare_for_write
from pacsys.errors import AuthenticationError, DeviceError, ReadError
from pacsys.types import Reading, ValueType
from tests.devices import (
    ARRAY_DEVICE,
    ARRAY_VALUES,
    ERROR_NOT_FOUND,
    TEMP_DEVICE,
    TEMP_DEVICE_2,
    TEMP_VALUE,
    TIMESTAMP_MILLIS,
    MockGSSAPIModule,
)

_MOCK_GSSAPI = MockGSSAPIModule()


def _create_mock_auth():
    """Create a mock KerberosAuth for testing without real Kerberos.

    Returns a KerberosAuth instance that uses MockGSSAPIModule.
    """
    with mock.patch.dict("sys.modules", {"gssapi": _MOCK_GSSAPI}):
        from pacsys.auth import KerberosAuth

        return KerberosAuth()


@contextmanager
def _mock_gssapi():
    """Keep gssapi mocked for the duration of a block.

    Needed because KerberosAuth._get_credentials() re-imports gssapi on every
    call (e.g. via auth.principal), so the mock must be active whenever a
    DMQBackend is constructed or uses the auth object.
    """
    with mock.patch.dict("sys.modules", {"gssapi": _MOCK_GSSAPI}):
        yield


def _mock_gss_context():
    """Return a mock GSS context for testing.

    The context returns a dummy token on step().
    """
    ctx = mock.MagicMock()
    ctx.step.return_value = b"mock_gss_token"
    ctx.get_mic.return_value = b"mock_mic_signature"
    return ctx


# =============================================================================
# Mock pika classes for SelectConnection (async/callback-based)
# =============================================================================


class MockIOLoop:
    """Mock IOLoop for SelectConnection testing."""

    def __init__(self, connection: "MockSelectConnection"):
        self._connection = connection
        self._running = False
        self._callbacks = []
        self._timers = {}
        self._timer_id = 0
        self._stop_event = threading.Event()

    def start(self):
        """Start the event loop (blocks in real code, here we process callbacks)."""
        self._running = True
        # Process initial callback queue
        self._process_callbacks()
        # Wait for stop signal (simulating blocking)
        while self._running and not self._stop_event.is_set():
            self._process_callbacks()
            time.sleep(0.01)

    def stop(self):
        """Stop the event loop."""
        self._running = False
        self._stop_event.set()

    def add_callback_threadsafe(self, callback):
        """Add a callback to be executed on the IO loop thread."""
        self._callbacks.append(callback)

    def call_later(self, delay: float, callback):
        """Schedule a callback after delay seconds."""
        self._timer_id += 1
        timer_id = self._timer_id
        self._timers[timer_id] = callback

        # Actually execute the callback after the delay
        def execute_later():
            time.sleep(delay)
            if timer_id in self._timers:
                cb = self._timers.pop(timer_id)
                self._callbacks.append(cb)

        threading.Thread(target=execute_later, daemon=True).start()
        return timer_id

    def remove_timeout(self, timer_id):
        """Cancel a scheduled timer."""
        self._timers.pop(timer_id, None)

    def _process_callbacks(self):
        """Process pending callbacks."""
        while self._callbacks:
            cb = self._callbacks.pop(0)
            cb()


class MockSelectChannel:
    """Mock Channel for SelectConnection with callback-based operations."""

    def __init__(self, connection: "MockSelectConnection", replies: list[bytes], routing_keys: list[str]):
        self._connection = connection
        self._replies = replies
        self._routing_keys = routing_keys
        self._reply_idx = 0
        self._queue_name = f"test-queue-{id(self)}"
        self._exchange_name = None
        self._on_message_callback = None
        self._consumer_tag = None
        self._published_messages = []
        self._is_open = True
        self._close_callbacks = []

    @property
    def is_open(self):
        return self._is_open

    def add_on_close_callback(self, callback):
        self._close_callbacks.append(callback)

    def queue_declare(self, queue="", exclusive=False, auto_delete=False, callback=None):
        frame = mock.MagicMock()
        frame.method.queue = self._queue_name
        if callback:
            callback(frame)

    def exchange_declare(self, exchange="", exchange_type="", auto_delete=False, callback=None):
        self._exchange_name = exchange
        if callback:
            callback(None)

    def queue_bind(self, queue="", exchange="", routing_key="", callback=None):
        if callback:
            callback(None)

    def basic_publish(self, exchange="", routing_key="", body=b"", properties=None):
        self._published_messages.append(
            {
                "exchange": exchange,
                "routing_key": routing_key,
                "body": body,
                "properties": properties,
            }
        )

    def basic_consume(self, queue, on_message_callback=None, auto_ack=False):
        self._on_message_callback = on_message_callback
        self._consumer_tag = f"ctag-{id(self)}"
        # Start delivering messages in background
        self._start_message_delivery()
        return self._consumer_tag

    def basic_cancel(self, consumer_tag):
        self._on_message_callback = None

    def basic_ack(self, delivery_tag):
        pass

    def close(self):
        self._is_open = False
        for cb in self._close_callbacks:
            cb(self, Exception("Channel closed"))

    def _start_message_delivery(self):
        """Deliver messages to the callback in a background thread."""

        def deliver():
            while self._on_message_callback and self._reply_idx < len(self._replies):
                if not self._is_open:
                    break
                reply_bytes = self._replies[self._reply_idx]
                if self._reply_idx < len(self._routing_keys):
                    routing_key = self._routing_keys[self._reply_idx]
                else:
                    routing_key = f"R.{TEMP_DEVICE}"
                self._reply_idx += 1

                method = mock.MagicMock()
                method.routing_key = routing_key
                method.delivery_tag = self._reply_idx

                callback = self._on_message_callback
                if callback:
                    self._connection.ioloop.add_callback_threadsafe(partial(callback, self, method, None, reply_bytes))
                time.sleep(0.01)

        thread = threading.Thread(target=deliver, daemon=True)
        thread.start()


class MockSelectConnection:
    """Mock SelectConnection for testing."""

    def __init__(self, replies: list[bytes] | None = None, routing_keys: list[str] | None = None):
        self._replies = replies or []
        self._routing_keys = routing_keys or []
        self._is_open = False
        self._on_open_callback = None
        self._on_close_callback = None
        self.ioloop = MockIOLoop(self)

    @property
    def is_open(self):
        return self._is_open

    def channel(self, on_open_callback=None):
        """Open a new channel."""
        ch = MockSelectChannel(self, self._replies, self._routing_keys)
        if on_open_callback:
            # Schedule callback to simulate async behavior
            self.ioloop.add_callback_threadsafe(lambda: on_open_callback(ch))
        return ch

    def close(self):
        """Close the connection."""
        self._is_open = False
        if self._on_close_callback:
            self._on_close_callback(self, Exception("Connection closed"))
        self.ioloop.stop()

    def _trigger_open(self):
        """Trigger the on_open callback (called after connection setup)."""
        self._is_open = True
        if self._on_open_callback:
            self._on_open_callback(self)


def create_mock_select_connection_factory(replies: list[bytes] | None = None, routing_keys: list[str] | None = None):
    """Create a factory function that returns a MockSelectConnection.

    This simulates pika.SelectConnection's constructor behavior.
    """
    mock_conn = MockSelectConnection(replies, routing_keys)

    def factory(
        cls=None,
        parameters=None,
        on_open_callback=None,
        on_open_error_callback=None,
        on_close_callback=None,
    ):
        mock_conn._on_open_callback = on_open_callback
        mock_conn._on_close_callback = on_close_callback

        # Trigger open callback after a brief delay (simulate async connect)
        def do_open():
            mock_conn._trigger_open()

        threading.Timer(0.01, do_open).start()
        return mock_conn

    return factory, mock_conn


def make_double_reply(
    value: float,
    time_ms: int = TIMESTAMP_MILLIS,
    ref_id: int | None = None,
) -> bytes:
    """Create marshaled DoubleSample_reply bytes."""
    reply = DoubleSample_reply()
    reply.value = value
    reply.time = time_ms
    if ref_id is not None:
        reply.ref_id = ref_id
    return bytes(reply.marshal())


def make_double_array_reply(
    values: list[float],
    time_ms: int = TIMESTAMP_MILLIS,
    ref_id: int | None = None,
) -> bytes:
    """Create marshaled DoubleArraySample_reply bytes."""
    reply = DoubleArraySample_reply()
    reply.value = values
    reply.time = time_ms
    if ref_id is not None:
        reply.ref_id = ref_id
    return bytes(reply.marshal())


def make_error_reply(
    facility_code: int = 0,
    error_number: int = -1,
    time_ms: int = TIMESTAMP_MILLIS,
    message: str | None = None,
    ref_id: int | None = None,
) -> bytes:
    """Create marshaled ErrorSample_reply bytes."""
    reply = ErrorSample_reply()
    reply.facilityCode = facility_code
    reply.errorNumber = error_number
    reply.time = time_ms
    if message is not None:
        reply.message = message
    if ref_id is not None:
        reply.ref_id = ref_id
    return bytes(reply.marshal())


def make_string_reply(
    value: str,
    time_ms: int = TIMESTAMP_MILLIS,
    ref_id: int | None = None,
) -> bytes:
    """Create marshaled StringSample_reply bytes."""
    reply = StringSample_reply()
    reply.value = value
    reply.time = time_ms
    if ref_id is not None:
        reply.ref_id = ref_id
    return bytes(reply.marshal())


# =============================================================================
# Backend context managers (eliminate repeated mock.patch boilerplate)
# =============================================================================


@contextmanager
def _mock_dmq_backend(replies=None, routing_keys=None, **kwargs):
    """Create a DMQBackend with mocked SelectConnection.

    Yields the backend; closes it on exit.
    """
    factory, mock_conn = create_mock_select_connection_factory(replies or [], routing_keys)
    with (
        _mock_gssapi(),
        mock.patch.object(SelectConnection, "__new__", side_effect=factory),
        mock.patch.object(DMQBackend, "_create_gss_context", return_value=_mock_gss_context()),
    ):
        backend = DMQBackend(host="localhost", auth=_create_mock_auth(), **kwargs)
        try:
            yield backend
        finally:
            backend.close()


@contextmanager
def _mock_dmq_write_backend(write_response_factory=None, **kwargs):
    """Create a DMQBackend with mocked SelectConnection supporting writes.

    Yields the backend; closes it on exit.
    """
    factory, mock_conn = create_write_select_connection_factory(write_response_factory)
    with (
        _mock_gssapi(),
        mock.patch("pika.BlockingConnection"),
        mock.patch.object(SelectConnection, "__new__", side_effect=factory),
        mock.patch.object(DMQBackend, "_create_gss_context", return_value=_mock_gss_context()),
    ):
        backend = DMQBackend(host="localhost", auth=_create_mock_auth(), **kwargs)
        try:
            yield backend
        finally:
            backend.close()


# =============================================================================
# Test Backend Initialization
# =============================================================================


class TestMockPikaBehavior:
    def test_threadsafe_callback_is_queued(self):
        connection = MockSelectConnection()
        callback = mock.MagicMock()
        connection.ioloop._running = True

        connection.ioloop.add_callback_threadsafe(callback)

        callback.assert_not_called()
        connection.ioloop._process_callbacks()
        callback.assert_called_once_with()

    def test_io_callback_error_propagates(self):
        connection = MockSelectConnection()
        connection.ioloop.add_callback_threadsafe(mock.MagicMock(side_effect=RuntimeError("callback failed")))

        with pytest.raises(RuntimeError, match="callback failed"):
            connection.ioloop._process_callbacks()

    def test_channel_close_callback_error_propagates(self):
        connection = MockSelectConnection()
        channel = MockSelectChannel(connection, [], [])
        channel.add_on_close_callback(mock.MagicMock(side_effect=RuntimeError("close callback failed")))

        with pytest.raises(RuntimeError, match="close callback failed"):
            channel.close()

    def test_connection_close_callback_error_propagates(self):
        connection = MockSelectConnection()
        connection._on_close_callback = mock.MagicMock(side_effect=RuntimeError("close callback failed"))

        with pytest.raises(RuntimeError, match="close callback failed"):
            connection.close()


class TestDMQCleanup:
    @staticmethod
    def _write_session(**overrides):
        session = mock.MagicMock()
        session.device = TEMP_DEVICE
        session.init_drf = f"{TEMP_DEVICE}.SETTING@N"
        session.channel = mock.MagicMock()
        session.channel.is_open = True
        session.exchange_name = "write-exchange"
        session.consumer_tag = None
        session.heartbeat_handle = None
        session.cleanup_handle = None
        session.init_timer = None
        session.queued_sends = []
        session.pending = {}
        session.init_confirmed = False
        for name, value in overrides.items():
            setattr(session, name, value)
        return session

    def test_cancel_failure_still_closes_read_channel(self):
        backend = DMQBackend.__new__(DMQBackend)
        backend._send_drop = mock.MagicMock()
        channel = mock.MagicMock()
        channel.is_open = True
        channel.basic_cancel.side_effect = ChannelWrongStateError("channel closed")
        job = mock.MagicMock(
            channel=channel,
            exchange_name="reply-exchange",
            consumer_tag="consumer",
            done_event=threading.Event(),
        )

        backend._complete_read(job)

        channel.close.assert_called_once_with()
        assert job.done_event.is_set()

    def test_unexpected_close_error_propagates_after_signalling_done(self):
        backend = DMQBackend.__new__(DMQBackend)
        backend._send_drop = mock.MagicMock()
        channel = mock.MagicMock()
        channel.is_open = True
        channel.close.side_effect = RuntimeError("close failed")
        job = mock.MagicMock(
            channel=channel,
            exchange_name="reply-exchange",
            consumer_tag="consumer",
            done_event=threading.Event(),
        )

        with pytest.raises(RuntimeError, match="close failed"):
            backend._complete_read(job)

        assert job.done_event.is_set()

    def test_cleanup_timer_failure_preserves_handle(self):
        backend = DMQBackend.__new__(DMQBackend)
        backend._select_connection = mock.MagicMock()
        backend._select_connection.is_open = True
        backend._select_connection.ioloop.remove_timeout.side_effect = RuntimeError("invalid timer")
        old_handle = object()
        session = mock.MagicMock(cleanup_handle=old_handle)

        with pytest.raises(RuntimeError, match="invalid timer"):
            backend._schedule_write_session_cleanup(session)

        assert session.cleanup_handle is old_handle
        backend._select_connection.ioloop.call_later.assert_not_called()

    def test_close_session_timer_failure_logs_and_finishes_cleanup(self, caplog):
        backend = DMQBackend.__new__(DMQBackend)
        backend._select_connection = mock.MagicMock()
        backend._select_connection.ioloop.remove_timeout.side_effect = RuntimeError("invalid timer")
        session = self._write_session(heartbeat_handle=object())
        backend._write_sessions = {session.init_drf: session}

        backend._close_write_session(session.init_drf, reason="test")

        session.channel.close.assert_called_once_with()
        assert session.init_drf not in backend._write_sessions
        assert "Failed to cancel heartbeat timer" in caplog.text

    def test_flush_timer_failure_logs_and_dispatches_queued_writes(self, caplog):
        backend = DMQBackend.__new__(DMQBackend)
        backend._select_connection = mock.MagicMock()
        backend._select_connection.ioloop.remove_timeout.side_effect = RuntimeError("invalid timer")
        backend._send_settings_async = mock.MagicMock()
        queued = ([(0, TEMP_DEVICE, 1.0)], [None], mock.MagicMock())
        session = self._write_session(init_timer=object(), queued_sends=[queued])

        backend._flush_queued_writes(session)

        assert session.init_timer is None
        assert session.init_confirmed
        assert not session.queued_sends
        backend._send_settings_async.assert_called_once_with(session, *queued)
        assert "Failed to cancel INIT timer" in caplog.text

    def test_channel_close_timer_failure_logs_and_completes_tracker(self, caplog):
        backend = DMQBackend.__new__(DMQBackend)
        backend._select_connection = mock.MagicMock()
        backend._select_connection.ioloop.remove_timeout.side_effect = RuntimeError("invalid timer")
        tracker = mock.MagicMock()
        results = [None]
        queued = ([(0, TEMP_DEVICE, 1.0)], results, tracker)
        session = self._write_session(cleanup_handle=object(), queued_sends=[queued])
        backend._write_sessions = {session.init_drf: session}

        backend._on_write_session_channel_closed(session.init_drf, RuntimeError("channel closed"))

        assert results[0] is not None
        tracker.device_complete.assert_called_once_with()
        assert "Failed to cancel cleanup timer" in caplog.text

    def test_connection_close_timer_failure_logs_and_completes_tracker(self, caplog):
        backend = DMQBackend.__new__(DMQBackend)
        backend._connection_ready = threading.Event()
        backend._connection_ready.set()
        backend._pending_session_setups = {}
        tracker = mock.MagicMock()
        results = [None]
        queued = ([(0, TEMP_DEVICE, 1.0)], results, tracker)
        session = self._write_session(init_timer=object(), queued_sends=[queued])
        backend._write_sessions = {session.init_drf: session}
        backend._stream_lock = threading.Lock()
        backend._subscriptions = {}
        backend._dispatcher = mock.MagicMock()
        connection = mock.MagicMock()
        connection.ioloop.remove_timeout.side_effect = RuntimeError("invalid timer")

        backend._on_connection_closed(connection, RuntimeError("connection closed"))

        assert results[0] is not None
        tracker.device_complete.assert_called_once_with()
        assert not backend._write_sessions
        connection.ioloop.stop.assert_called_once_with()
        assert "Failed to cancel INIT timer after connection loss" in caplog.text

    def test_subscription_timer_failure_logs_and_finishes_cleanup(self, caplog):
        backend = DMQBackend.__new__(DMQBackend)
        connection = mock.MagicMock()
        connection.is_open = True
        connection.ioloop.remove_timeout.side_effect = RuntimeError("invalid timer")
        connection.ioloop.add_callback_threadsafe.side_effect = lambda callback: callback()
        backend._select_connection = connection
        sub = mock.MagicMock()
        sub.sub_id = "subscription-id"
        sub.heartbeat_handle = object()
        sub.channel.is_open = True
        sub.exchange_name = "subscription-exchange"
        sub.consumer_tag = "consumer"

        backend._cancel_subscription_async(sub)

        assert sub.heartbeat_handle is None
        sub.channel.basic_publish.assert_called_once_with(exchange="subscription-exchange", routing_key="D", body=b"")
        sub.channel.basic_cancel.assert_called_once_with("consumer")
        sub.channel.close.assert_called_once_with()
        sub.handle._signal_stop.assert_called_once_with()
        assert "Failed to cancel heartbeat timer for subscription" in caplog.text


class TestDMQBackendInit:
    """Tests for DMQBackend initialization."""

    def test_backend_requires_auth(self):
        """Test that auth is required for DMQ backend."""
        with pytest.raises(AuthenticationError, match="DMQ requires KerberosAuth"):
            DMQBackend()

    def test_backend_init_invalid_host(self):
        """Test that empty host raises ValueError."""
        auth = _create_mock_auth()
        with pytest.raises(ValueError, match="host cannot be empty"):
            DMQBackend(host="", auth=auth)

    def test_backend_init_invalid_port(self):
        """Test that invalid port raises ValueError."""
        auth = _create_mock_auth()
        with pytest.raises(ValueError, match="port must be between"):
            DMQBackend(port=0, auth=auth)
        with pytest.raises(ValueError, match="port must be between"):
            DMQBackend(port=70000, auth=auth)

    def test_backend_init_invalid_timeout(self):
        """Test that invalid timeout raises ValueError."""
        auth = _create_mock_auth()
        with pytest.raises(ValueError, match="timeout must be positive"):
            DMQBackend(timeout=0, auth=auth)
        with pytest.raises(ValueError, match="timeout must be positive"):
            DMQBackend(timeout=-1, auth=auth)


class TestDMQConnectionOpenError:
    """A failed initial connect must not permanently brick the backend (dmq H1)."""

    def test_open_error_stops_io_thread_and_allows_retry(self):
        conns = []

        def factory(
            cls=None,
            parameters=None,
            on_open_callback=None,
            on_open_error_callback=None,
            on_close_callback=None,
        ):
            conn = MockSelectConnection()
            conn._on_open_callback = on_open_callback
            conn._on_close_callback = on_close_callback
            conns.append(conn)
            if len(conns) == 1:
                # First attempt: connection open fails
                threading.Timer(
                    0.01, lambda: on_open_error_callback(conn, ConnectionRefusedError("connection refused"))
                ).start()
            else:
                threading.Timer(0.01, conn._trigger_open).start()
            return conn

        with (
            _mock_gssapi(),
            mock.patch.object(SelectConnection, "__new__", side_effect=factory),
        ):
            backend = DMQBackend(host="localhost", auth=_create_mock_auth(), timeout=2.0)
            try:
                with pytest.raises(ConnectionError, match="connection refused"):
                    backend._ensure_io_thread()

                # IO thread must exit after the failed open (ioloop stopped)
                thread = backend._io_thread
                assert thread is not None
                thread.join(timeout=2.0)
                assert not thread.is_alive()

                # Next attempt starts a fresh connection and recovers
                backend._ensure_io_thread()
                assert backend._connection_error is None
                assert len(conns) == 2
            finally:
                start = time.monotonic()
                backend.close()
                # close() must not burn the full 3s join timeout on a stuck thread
                assert time.monotonic() - start < 2.0


# =============================================================================
# Job-level INIT errors (plain "R" routing key, no ref_id) — dmq H2
# =============================================================================

SECURITY_VIOLATION = -99


class _InitFailChannel(MockSelectChannel):
    """Responds to INIT with a job-level ErrorSample on plain 'R' (no ref_id),
    correlated to the INIT message_id — the ServerJobManager.sendStatus shape."""

    def basic_publish(self, exchange="", routing_key="", body=b"", properties=None):
        super().basic_publish(exchange, routing_key, body, properties)
        if routing_key == "I":
            self._init_message_id = getattr(properties, "message_id", None)

    def basic_consume(self, queue, on_message_callback=None, auto_ack=False):
        self._on_message_callback = on_message_callback
        self._consumer_tag = f"ctag-{id(self)}"
        mid = getattr(self, "_init_message_id", None)
        if mid and not getattr(self, "_delivered", False):
            self._delivered = True
            method = mock.MagicMock()
            method.routing_key = "R"
            method.delivery_tag = 1
            props = mock.MagicMock()
            props.correlation_id = mid
            body = make_error_reply(
                facility_code=FACILITY_DMQ, error_number=SECURITY_VIOLATION, message="Security violation: test"
            )
            self._connection.ioloop.add_callback_threadsafe(partial(on_message_callback, self, method, props, body))
        return self._consumer_tag


class _InitFailConnection(MockSelectConnection):
    def channel(self, on_open_callback=None):
        ch = _InitFailChannel(self, [], [])
        if on_open_callback:
            self.ioloop.add_callback_threadsafe(lambda: on_open_callback(ch))
        return ch


@contextmanager
def _init_fail_backend(**kwargs):
    """DMQBackend whose server rejects every INIT with a security violation."""
    mock_conn = _InitFailConnection()

    def factory(
        cls=None,
        parameters=None,
        on_open_callback=None,
        on_open_error_callback=None,
        on_close_callback=None,
    ):
        mock_conn._on_open_callback = on_open_callback
        mock_conn._on_close_callback = on_close_callback
        threading.Timer(0.01, mock_conn._trigger_open).start()
        return mock_conn

    with (
        _mock_gssapi(),
        mock.patch.object(SelectConnection, "__new__", side_effect=factory),
        mock.patch.object(DMQBackend, "_create_gss_context", return_value=_mock_gss_context()),
    ):
        backend = DMQBackend(host="localhost", auth=_create_mock_auth(), **kwargs)
        try:
            yield backend
        finally:
            backend.close()


class TestDMQJobLevelErrors:
    """Server INIT failures (auth/server errors) must surface, not time out."""

    def test_resolve_reply_job_error_correlation(self):
        body = make_error_reply(facility_code=FACILITY_DMQ, error_number=SECURITY_VIOLATION, message="denied")
        props = mock.MagicMock()
        props.correlation_id = "init-123"
        # Matching correlation id -> job-level error (idx None)
        result = _resolve_reply("R", body, ["M:OUTTMP"], {"M:OUTTMP": 0}, props, "init-123")
        assert result is not None
        reply, idx, _ref_id = result
        assert idx is None
        assert reply.errorNumber == SECURITY_VIOLATION
        # Mismatched correlation id -> dropped
        assert _resolve_reply("R", body, ["M:OUTTMP"], {"M:OUTTMP": 0}, props, "other-id") is None
        # No properties (empty correlation id) -> accepted
        result = _resolve_reply("R", body, ["M:OUTTMP"], {"M:OUTTMP": 0}, None, "init-123")
        assert result is not None and result[1] is None
        # Device-keyed error still resolves per-device
        result = _resolve_reply("R.M:OUTTMP", body, ["M:OUTTMP"], {"M:OUTTMP": 0}, props, "init-123")
        assert result is not None and result[1] == 0

    def test_read_init_failure_fails_fast(self):
        error = make_error_reply(
            facility_code=FACILITY_DMQ, error_number=SECURITY_VIOLATION, message="Security violation: test"
        )
        with _mock_dmq_backend(replies=[error], routing_keys=["R"]) as backend:
            start = time.monotonic()
            with pytest.raises(ReadError, match="job start failed") as exc_info:
                backend.get(TEMP_DEVICE, timeout=5.0)
            assert time.monotonic() - start < 2.0  # not a timeout
            reading = exc_info.value.readings[0]
            assert not reading.ok
            assert reading.error_code == SECURITY_VIOLATION
            assert reading.facility_code == FACILITY_DMQ
            assert "Security violation" in reading.message

    def test_get_many_init_failure_fails_all_devices(self):
        with _init_fail_backend() as backend:
            with pytest.raises(ReadError) as exc_info:
                backend.get_many([TEMP_DEVICE, TEMP_DEVICE_2], timeout=5.0)
            readings = exc_info.value.readings
            assert len(readings) == 2
            for reading in readings:
                assert not reading.ok
                assert reading.error_code == SECURITY_VIOLATION

    def test_subscribe_init_failure_signals_error(self):
        errors = []
        err_event = threading.Event()

        def on_err(exc, handle):
            errors.append(exc)
            err_event.set()

        with _init_fail_backend() as backend:
            try:
                handle = backend.subscribe([TEMP_DEVICE], callback=lambda r, h: None, on_error=on_err)
            except DeviceError as e:
                # Error arrived before subscribe() returned
                assert e.error_code == SECURITY_VIOLATION
                return
            assert err_event.wait(2.0), "on_error was never called for rejected INIT"
            assert isinstance(errors[0], DeviceError)
            assert errors[0].error_code == SECURITY_VIOLATION
            assert handle.stopped
            assert handle.exc is not None

    def test_write_init_failure_returns_server_error(self):
        with _init_fail_backend() as backend:
            start = time.monotonic()
            result = backend.write(TEMP_DEVICE, TEMP_VALUE, timeout=8.0)
            elapsed = time.monotonic() - start
            assert not result.success
            assert result.error_code == SECURITY_VIOLATION
            assert "Security violation" in (result.message or "")
            assert elapsed < 3.0  # not the 5s INIT timer, not the 8s timeout


# =============================================================================
# write_many vs connection loss (dmq H3)
# =============================================================================


class TestDMQWriteConnectionLoss:
    """Multi-device write_many must complete per (tracker, device) on connection loss."""

    def test_connection_close_completes_tracker_per_device(self):
        """Direct accounting: one device_complete per (tracker, init_drf)."""

        def fake_session(init_drf):
            return _WriteSession(
                device=init_drf,
                init_drf=init_drf,
                channel=None,
                exchange_name="x",
                queue_name="q",
                gss_context=None,
                last_used=0.0,
            )

        with _mock_dmq_backend() as backend:
            backend._ensure_io_thread()
            conn = backend._select_connection
            assert conn is not None

            # Tracker A spans two sessions (a 2-device write_many)
            tracker_a = _WriteCompletionTracker(total_devices=2)
            results_a: list = [None, None]
            # Tracker B has queued AND pending work in one session (1 device)
            tracker_b = _WriteCompletionTracker(total_devices=1)
            results_b: list = [None, None]
            # Tracker C waits in a pending session setup (1 device)
            tracker_c = _WriteCompletionTracker(total_devices=1)
            results_c: list = [None]

            s1 = fake_session("D:EV1.SETTING@N")
            s1.queued_sends = [([(0, "D:EV1", 1.0)], results_a, tracker_a)]
            s2 = fake_session("D:EV2.SETTING@N")
            s2.pending = {"corr1": (1, "D:EV2", results_a, tracker_a), "corr2": (1, "D:EV2", results_b, tracker_b)}
            s2.queued_sends = [([(0, "D:EV2", 3.0)], results_b, tracker_b)]
            backend._write_sessions = {s1.init_drf: s1, s2.init_drf: s2}
            backend._pending_session_setups = {"D:EV3.SETTING@N": [([(0, "D:EV3", 4.0)], results_c, tracker_c)]}

            backend._on_connection_closed(conn, Exception("link lost"))

            assert tracker_a.completed_devices == 2 and tracker_a.done_event.is_set()
            assert tracker_b.completed_devices == 1 and tracker_b.done_event.is_set()
            assert tracker_c.completed_devices == 1 and tracker_c.done_event.is_set()
            for results in (results_a, results_b, results_c):
                for r in results:
                    assert r is not None and "Connection closed" in r.message

    def test_write_many_two_devices_connection_loss_returns_promptly(self):
        """2-device write_many + connection drop: prompt error results, no hang/crash."""
        holder = {}
        with _mock_dmq_backend() as backend:  # no PENDING ever arrives -> writes stay queued

            def do_writes():
                holder["results"] = backend.write_many([(TEMP_DEVICE, 1.0), (TEMP_DEVICE_2, 2.0)], timeout=8.0)

            t = threading.Thread(target=do_writes)
            t.start()
            time.sleep(0.3)  # let both sessions get created
            conn = backend._select_connection
            assert conn is not None
            conn.ioloop.add_callback_threadsafe(conn.close)
            t.join(timeout=3.0)
            assert not t.is_alive(), "write_many hung after connection loss"
            results = holder["results"]
            assert len(results) == 2
            for r in results:
                assert not r.success
                assert "Connection closed" in (r.message or "")

    def test_write_many_timeout_with_dead_connection_no_crash(self):
        """IO loop dies without close callback: timeout backfill, no AttributeError."""
        holder = {}
        with _mock_dmq_backend() as backend:

            def do_write():
                holder["results"] = backend.write_many([(TEMP_DEVICE, 1.0)], timeout=1.5)

            t = threading.Thread(target=do_write)
            t.start()
            time.sleep(0.3)
            conn = backend._select_connection
            assert conn is not None
            conn.ioloop.stop()  # thread exits, _select_connection -> None, results unfilled
            t.join(timeout=6.0)
            assert not t.is_alive()
            assert "results" in holder, "write_many raised instead of returning results"
            result = holder["results"][0]
            assert not result.success
            assert result.error_code == ERR_TIMEOUT


# =============================================================================
# Test Read Operations
# =============================================================================


class TestDMQBackendRead:
    """Tests for DMQBackend read operations (uses SelectConnection via IO thread)."""

    def test_read_scalar(self):
        """Test reading a scalar value."""
        with _mock_dmq_backend([make_double_reply(TEMP_VALUE, ref_id=1)]) as backend:
            assert backend.read(TEMP_DEVICE) == TEMP_VALUE

    def test_get_scalar(self):
        """Test get() returns Reading with correct fields."""
        with _mock_dmq_backend([make_double_reply(TEMP_VALUE, ref_id=1)]) as backend:
            reading = backend.get(TEMP_DEVICE)
            assert isinstance(reading, Reading)
            assert reading.value == TEMP_VALUE
            assert reading.value_type == ValueType.SCALAR
            assert reading.is_success
            assert reading.ok
            assert reading.error_code == 0

    def test_get_scalar_array(self):
        """Test reading an array value."""
        with _mock_dmq_backend([make_double_array_reply(ARRAY_VALUES, ref_id=1)]) as backend:
            reading = backend.get(ARRAY_DEVICE)
            assert reading.value_type == ValueType.SCALAR_ARRAY
            assert isinstance(reading.value, np.ndarray)
            np.testing.assert_array_equal(reading.value, np.array(ARRAY_VALUES))

    def test_get_many(self):
        """Test batch read with get_many()."""
        replies = [make_double_reply(TEMP_VALUE, ref_id=1), make_double_reply(1.234, ref_id=2)]
        with _mock_dmq_backend(replies) as backend:
            readings = backend.get_many([TEMP_DEVICE, TEMP_DEVICE_2])
            assert len(readings) == 2
            assert readings[0].value == TEMP_VALUE
            assert readings[1].value == 1.234

    def test_get_many_same_device_different_properties(self):
        """Regression: routing key matching when ref_id is missing."""
        replies = [make_double_reply(1.0), make_double_reply(2.0)]
        routing_keys = [f"R.{TEMP_DEVICE}.READING@I", f"R.{TEMP_DEVICE}.SETTING@I"]
        with _mock_dmq_backend(replies, routing_keys) as backend:
            readings = backend.get_many([TEMP_DEVICE, f"{TEMP_DEVICE}.SETTING"])
            assert len(readings) == 2
            assert readings[0].value == 1.0
            assert readings[1].value == 2.0

    def test_get_many_out_of_order_routing_keys(self):
        """Test correct matching when responses arrive out of order."""
        replies = [make_double_reply(99.0), make_double_reply(42.0)]
        routing_keys = [f"R.{TEMP_DEVICE}.SETTING@I", f"R.{TEMP_DEVICE}.READING@I"]
        with _mock_dmq_backend(replies, routing_keys) as backend:
            readings = backend.get_many([TEMP_DEVICE, f"{TEMP_DEVICE}.SETTING"])
            assert len(readings) == 2
            assert readings[0].value == 42.0  # READING
            assert readings[1].value == 99.0  # SETTING

    def test_read_error(self):
        """Test that read() raises DeviceError on error reply."""
        reply = make_error_reply(error_number=ERROR_NOT_FOUND, message="Device not found", ref_id=1)
        with _mock_dmq_backend([reply]) as backend:
            with pytest.raises(DeviceError) as exc_info:
                backend.read(TEMP_DEVICE)
            assert exc_info.value.error_code == ERROR_NOT_FOUND

    def test_get_error(self):
        """Test that get() returns error Reading instead of raising."""
        reply = make_error_reply(error_number=ERROR_NOT_FOUND, message="Device not found", ref_id=1)
        with _mock_dmq_backend([reply]) as backend:
            reading = backend.get(TEMP_DEVICE)
            assert reading.is_error
            assert not reading.ok
            assert reading.error_code == ERROR_NOT_FOUND

    def test_read_timeout(self):
        """Test that timeout raises ReadError."""
        with _mock_dmq_backend([], timeout=0.5) as backend:
            with pytest.raises(ReadError) as exc_info:
                backend.get(TEMP_DEVICE, timeout=0.5)
            readings = exc_info.value.readings
            assert len(readings) == 1
            assert readings[0].is_error
            assert readings[0].error_code == -6  # ACNET_REQTMO
            assert "timeout" in readings[0].message.lower()

    def test_read_gss_failure_reports_auth_error(self):
        """Test that GSS auth failure during read raises ReadError with ERR_RETRY.

        Regression: previously, GSS errors were masked as ERR_TIMEOUT because
        on_gss_error just called _complete_read without storing the error.
        """
        factory, mock_conn = create_mock_select_connection_factory([])
        with (
            _mock_gssapi(),
            mock.patch.object(SelectConnection, "__new__", side_effect=factory),
            mock.patch.object(
                DMQBackend,
                "_create_gss_context",
                side_effect=AuthenticationError("Kerberos ticket expired"),
            ),
        ):
            backend = DMQBackend(host="localhost", auth=_create_mock_auth())
            try:
                with pytest.raises(ReadError) as exc_info:
                    backend.get(TEMP_DEVICE, timeout=2.0)
                reading = exc_info.value.readings[0]
                assert reading.is_error
                assert reading.error_code == -1, f"Expected ERR_RETRY (-1) for auth failure, got {reading.error_code}"
                assert "Kerberos ticket expired" in reading.message
                assert isinstance(exc_info.value.__cause__, AuthenticationError)
            finally:
                backend.close()

    def test_read_closed_backend(self):
        """Test that read on closed backend raises RuntimeError."""
        with _mock_gssapi():
            backend = DMQBackend(host="localhost", auth=_create_mock_auth())
            backend.close()
            with pytest.raises(RuntimeError, match="closed"):
                backend.read(TEMP_DEVICE)


# =============================================================================
# Test Subscribe Operations
# =============================================================================


class TestDMQBackendSubscribe:
    """Tests for DMQBackend subscribe operations."""

    def test_subscribe_close_race_raises(self):
        """_closed flipping between the pre-check and the locked insert must raise."""
        with _mock_dmq_backend([]) as backend:
            orig = backend._ensure_io_thread

            def racy():
                orig()
                backend._closed = True  # concurrent close() lands here

            with (
                mock.patch.object(backend, "_ensure_io_thread", side_effect=racy),
                pytest.raises(RuntimeError, match="Backend is closed"),
            ):
                backend.subscribe([TEMP_DEVICE])
            assert backend._subscriptions == {}
            backend._closed = False  # let the context manager close() run fully

    def test_subscribe_iterator_mode(self):
        """Test subscribe with iterator mode yields readings."""
        replies = [make_double_reply(TEMP_VALUE + i, ref_id=1) for i in range(3)]
        with _mock_dmq_backend(replies) as backend:
            handle = backend.subscribe([TEMP_DEVICE])
            readings_received = []
            start = time.monotonic()
            for reading, h in handle.readings(timeout=1.0):
                readings_received.append(reading)
                if len(readings_received) >= 3 or time.monotonic() - start > 2.0:
                    break
            handle.stop()
            assert len(readings_received) >= 1
            assert readings_received[0].value == TEMP_VALUE

    def test_subscribe_callback_mode(self):
        """Test subscribe with callback mode calls callback."""
        replies = [make_double_reply(TEMP_VALUE + i, ref_id=1) for i in range(2)]
        callback_results = []
        with _mock_dmq_backend(replies) as backend:
            handle = backend.subscribe([TEMP_DEVICE], callback=lambda r, h: callback_results.append(r))
            deadline = time.monotonic() + 2.0
            while len(callback_results) < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            handle.stop()
            assert len(callback_results) >= 1
            assert callback_results[0].value == TEMP_VALUE

    def test_subscribe_stop(self):
        """Test that handle.stop() stops subscription."""
        replies = [make_double_reply(TEMP_VALUE, ref_id=1) for _ in range(10)]
        with _mock_dmq_backend(replies) as backend:
            handle = backend.subscribe([TEMP_DEVICE])
            assert not handle.stopped
            handle.stop()
            assert handle.stopped

    def test_subscribe_empty_drfs_raises(self):
        """Test that subscribe with empty drfs raises ValueError."""
        with _mock_gssapi():
            backend = DMQBackend(host="localhost", auth=_create_mock_auth())
            try:
                with pytest.raises(ValueError, match="drfs cannot be empty"):
                    backend.subscribe([])
            finally:
                backend.close()

    def test_subscribe_callback_cannot_iterate(self):
        """Test that callback-mode subscription cannot use readings()."""
        with _mock_dmq_backend([make_double_reply(TEMP_VALUE, ref_id=1)]) as backend:
            handle = backend.subscribe([TEMP_DEVICE], callback=lambda r, h: None)
            with pytest.raises(RuntimeError, match="callback"):
                list(handle.readings(timeout=0.1))
            handle.stop()

    def test_subscribe_context_manager(self):
        """Test that subscription handle works as context manager."""
        with _mock_dmq_backend([make_double_reply(TEMP_VALUE, ref_id=1)]) as backend:
            handle = backend.subscribe([TEMP_DEVICE])
            with handle:
                assert not handle.stopped
            assert handle.stopped


# =============================================================================
# Test Write Operations
# =============================================================================


class MockSelectChannelWithWriteSupport(MockSelectChannel):
    """Extended MockSelectChannel with write response support."""

    def __init__(
        self,
        connection: "MockSelectConnection",
        replies: list[bytes],
        routing_keys: list[str],
        write_response_factory=None,
    ):
        super().__init__(connection, replies, routing_keys)
        self._write_response_factory = write_response_factory
        self._pending_writes: list[str] = []
        self._write_thread_started = False

    def basic_publish(self, exchange="", routing_key="", body=b"", properties=None):
        """Capture INIT and SETTING messages."""
        super().basic_publish(exchange, routing_key, body, properties)
        # INIT triggers a PENDING response (confirms S.# binding)
        if routing_key == "I":
            self._init_received = True
        # Queue write response if this is a SETTING message (use message_id
        # as the response correlation_id, matching Java server behavior)
        if routing_key.startswith("S.") and properties and getattr(properties, "message_id", None):
            self._pending_writes.append(properties.message_id)

    def basic_consume(self, queue, on_message_callback=None, auto_ack=False):
        """Start consuming and also deliver write responses."""
        self._on_message_callback = on_message_callback
        self._consumer_tag = f"ctag-{id(self)}"

        # Start the regular message delivery for streaming
        if self._replies:
            self._start_message_delivery()

        # Start write response delivery
        if self._write_response_factory and not self._write_thread_started:
            self._write_thread_started = True

            def deliver_write_responses():
                pending_sent = False
                for _ in range(300):  # ~6 seconds max
                    if not self._is_open:
                        break
                    # Send PENDING once after INIT (confirms S.# binding)
                    if not pending_sent and getattr(self, "_init_received", False) and self._on_message_callback:
                        pending_sent = True
                        pending_reply = ErrorSample_reply()
                        pending_reply.facilityCode = FACILITY_DMQ
                        pending_reply.errorNumber = 1  # DMQ_PENDING_ERROR
                        pending_reply.time = 0
                        method = mock.MagicMock()
                        method.routing_key = "R.pending"
                        method.delivery_tag = 0
                        props = mock.MagicMock()
                        props.correlation_id = None
                        callback = self._on_message_callback
                        reply_bytes = bytes(pending_reply.marshal())
                        self._connection.ioloop.add_callback_threadsafe(
                            partial(callback, self, method, props, reply_bytes)
                        )
                    while self._pending_writes and self._on_message_callback:
                        corr_id = self._pending_writes.pop(0)
                        response_bytes = self._write_response_factory(corr_id)

                        method = mock.MagicMock()
                        method.routing_key = f"R.{TEMP_DEVICE}"
                        method.delivery_tag = 1

                        props = mock.MagicMock()
                        props.correlation_id = corr_id

                        callback = self._on_message_callback
                        self._connection.ioloop.add_callback_threadsafe(
                            partial(callback, self, method, props, response_bytes)
                        )
                    time.sleep(0.02)

            threading.Thread(target=deliver_write_responses, daemon=True).start()

        return self._consumer_tag


def _default_write_response_factory(corr_id: str) -> bytes:
    """Default write response - success."""
    reply = DoubleSample_reply()
    reply.value = TEMP_VALUE
    reply.time = TIMESTAMP_MILLIS
    reply.ref_id = 1
    return bytes(reply.marshal())


class MockSelectConnectionWithWriteSupport(MockSelectConnection):
    """Mock SelectConnection that supports write operations."""

    def __init__(
        self, replies: list[bytes] | None = None, routing_keys: list[str] | None = None, write_response_factory=None
    ):
        super().__init__(replies, routing_keys)
        self._write_response_factory = write_response_factory or _default_write_response_factory

    def channel(self, on_open_callback=None):
        """Open a channel with write support."""
        ch = MockSelectChannelWithWriteSupport(self, self._replies, self._routing_keys, self._write_response_factory)
        if on_open_callback:
            self.ioloop.add_callback_threadsafe(lambda: on_open_callback(ch))
        return ch


def create_write_select_connection_factory(write_response_factory=None):
    """Create a factory for MockSelectConnectionWithWriteSupport."""
    mock_conn = MockSelectConnectionWithWriteSupport(
        replies=[], routing_keys=[], write_response_factory=write_response_factory
    )

    def factory(
        cls=None,
        parameters=None,
        on_open_callback=None,
        on_open_error_callback=None,
        on_close_callback=None,
    ):
        mock_conn._on_open_callback = on_open_callback
        mock_conn._on_close_callback = on_close_callback

        def do_open():
            mock_conn._trigger_open()

        threading.Timer(0.01, do_open).start()
        return mock_conn

    return factory, mock_conn


class TestDMQBackendWrite:
    """Tests for DMQBackend write operations with Kerberos auth."""

    def test_write_returns_write_result(self):
        """Test that write() returns WriteResult on success."""
        with _mock_dmq_write_backend() as backend:
            result = backend.write(TEMP_DEVICE, TEMP_VALUE, timeout=5.0)
            assert result.success
            assert result.error_code == 0

    def test_write_many_returns_results(self):
        """Test that write_many() returns list of WriteResult."""
        with _mock_dmq_write_backend() as backend:
            results = backend.write_many([(TEMP_DEVICE, TEMP_VALUE), (TEMP_DEVICE_2, 1.234)], timeout=5.0)
            assert len(results) == 2
            assert results[0].success
            assert results[1].success

    def test_write_handles_error_response(self):
        """Test that write() handles error response correctly."""

        def error_response_factory(corr_id):
            reply = ErrorSample_reply()
            reply.facilityCode = 1
            reply.errorNumber = ERROR_NOT_FOUND
            reply.time = TIMESTAMP_MILLIS
            reply.message = "Device not found"
            reply.ref_id = 1
            return bytes(reply.marshal())

        with _mock_dmq_write_backend(error_response_factory) as backend:
            result = backend.write(TEMP_DEVICE, TEMP_VALUE, timeout=5.0)
            assert not result.success
            assert result.error_code == ERROR_NOT_FOUND

    def test_write_session_reuse(self):
        """Test that write sessions are reused for same device."""
        with _mock_dmq_write_backend() as backend:
            assert backend.write(TEMP_DEVICE, TEMP_VALUE, timeout=5.0).success
            assert backend.write(TEMP_DEVICE, TEMP_VALUE + 1, timeout=5.0).success
            # Sessions keyed by init_drf (e.g. "M:OUTTMP.SETTING@N")
            assert prepare_for_write(TEMP_DEVICE) in backend._write_sessions

    def test_write_session_per_device(self):
        """Test that each device gets its own write session."""
        with _mock_dmq_write_backend() as backend:
            assert backend.write(TEMP_DEVICE, TEMP_VALUE, timeout=5.0).success
            assert backend.write(TEMP_DEVICE_2, 1.234, timeout=5.0).success
            assert len(backend._write_sessions) == 2
            assert prepare_for_write(TEMP_DEVICE) in backend._write_sessions
            assert prepare_for_write(TEMP_DEVICE_2) in backend._write_sessions

    def test_write_session_cleanup_on_close(self):
        """Test that write sessions are cleaned up on backend close."""
        with _mock_dmq_write_backend() as backend:
            assert backend.write(TEMP_DEVICE, TEMP_VALUE, timeout=5.0).success
            assert len(backend._write_sessions) == 1
        # After context exit (close), sessions cleared
        assert len(backend._write_sessions) == 0

    def test_write_bytes_returns_error(self):
        """DMQ server rejects BinarySample; writing bytes must fail."""
        factory, mock_conn = create_write_select_connection_factory()
        with (
            _mock_gssapi(),
            mock.patch("pika.BlockingConnection"),
            mock.patch.object(SelectConnection, "__new__", side_effect=factory),
        ):
            backend = DMQBackend(host="localhost", auth=_create_mock_auth())
            try:
                result = backend.write(TEMP_DEVICE, b"\x00\x01\x02\x03", timeout=5.0)
                assert not result.success
                assert "does not support writing bytes" in result.message
            finally:
                backend.close()

    def test_write_auth_failure_returns_error_result(self):
        """Test that GSS context failure during async write returns error WriteResult."""
        factory, mock_conn = create_write_select_connection_factory()

        def failing_gss():
            raise RuntimeError("Kerberos ticket expired")

        with (
            _mock_gssapi(),
            mock.patch("pika.BlockingConnection"),
            mock.patch.object(SelectConnection, "__new__", side_effect=factory),
            mock.patch.object(DMQBackend, "_create_gss_context", side_effect=failing_gss),
        ):
            backend = DMQBackend(host="localhost", auth=_create_mock_auth())
            try:
                result = backend.write(TEMP_DEVICE, TEMP_VALUE, timeout=5.0)
                assert not result.success
                assert "GSS context creation failed" in (result.message or "")
            finally:
                backend.close()


# =============================================================================
# Test Alarm Dict to Sample Conversion
# =============================================================================


class TestDictToAlarmSample:
    """Tests for _dict_to_alarm_sample helper."""

    def test_analog_alarm_dict(self):
        from pacsys.backends.dmq import _dict_to_alarm_sample

        sample = _dict_to_alarm_sample(
            {"minimum": 1.5, "maximum": 99.0, "alarm_enable": True, "tries_needed": 3},
            ref_id=7,
            timestamp_ms=1000,
        )
        assert isinstance(sample, AnalogAlarmSample_reply)
        assert sample.value.minimum == 1.5
        assert sample.value.maximum == 99.0
        assert sample.value.alarm_enable is True
        assert sample.value.tries_needed == 3
        assert sample.ref_id == 7

    def test_analog_alarm_partial(self):
        from pacsys.backends.dmq import _dict_to_alarm_sample

        sample = _dict_to_alarm_sample({"minimum": 10.0}, ref_id=1, timestamp_ms=0)
        assert isinstance(sample, AnalogAlarmSample_reply)
        assert sample.value.minimum == 10.0
        assert sample.value.maximum == 0.0  # default

    def test_digital_alarm_dict(self):
        from pacsys.backends.dmq import _dict_to_alarm_sample

        sample = _dict_to_alarm_sample(
            {"nominal": 0xFF, "mask": 0x0F, "alarm_enable": False},
            ref_id=1,
            timestamp_ms=0,
        )
        assert isinstance(sample, DigitalAlarmSample_reply)
        assert sample.value.nominal == 0xFF
        assert sample.value.mask == 0x0F
        assert sample.value.alarm_enable is False

    def test_readonly_keys_skipped(self):
        from pacsys.backends.dmq import _dict_to_alarm_sample

        sample = _dict_to_alarm_sample(
            {"minimum": 1.0, "maximum": 2.0, "alarm_status": True, "abort": False, "tries_now": 5},
            ref_id=1,
            timestamp_ms=0,
        )
        assert isinstance(sample, AnalogAlarmSample_reply)
        assert sample.value.minimum == 1.0

    def test_unknown_keys_raises(self):
        from pacsys.backends.dmq import _dict_to_alarm_sample

        with pytest.raises(ValueError, match="Unknown alarm dict keys"):
            _dict_to_alarm_sample({"minimum": 1.0, "bogus": 42}, ref_id=1, timestamp_ms=0)

    def test_mixed_keys_raises(self):
        from pacsys.backends.dmq import _dict_to_alarm_sample

        with pytest.raises(ValueError, match=r"Cannot mix analog.*and digital"):
            _dict_to_alarm_sample({"minimum": 1.0, "nominal": 5}, ref_id=1, timestamp_ms=0)

    def test_shared_only_keys_raises(self):
        from pacsys.backends.dmq import _dict_to_alarm_sample

        with pytest.raises(ValueError, match="type-specific key"):
            _dict_to_alarm_sample({"alarm_enable": True}, ref_id=1, timestamp_ms=0)

    def test_empty_dict_raises(self):
        from pacsys.backends.dmq import _dict_to_alarm_sample

        with pytest.raises(ValueError, match="type-specific key"):
            _dict_to_alarm_sample({}, ref_id=1, timestamp_ms=0)


# =============================================================================
# Test Value-to-Sample Conversion
# =============================================================================


class TestValueToSample:
    """Tests for _value_to_sample helper (BasicControl → wire format)."""

    def test_basic_control_on_uses_sdd_enum(self):
        """Commands 0-6 use BasicControlSample with SDD enum constants."""
        from pacsys.backends.dmq import _BASIC_CONTROL_TO_SDD, DMQBackend
        from pacsys.types import BasicControl

        backend = object.__new__(DMQBackend)
        sample = backend._value_to_sample(BasicControl.ON)
        assert isinstance(sample, BasicControlSample_reply)
        assert sample.value == _BASIC_CONTROL_TO_SDD[BasicControl.ON]

    def test_basic_control_local_uses_double(self):
        """LOCAL/REMOTE/TRIP (7-9) use DoubleSample since DMQ proto lacks them."""
        from pacsys.backends.dmq import DMQBackend
        from pacsys.types import BasicControl

        backend = object.__new__(DMQBackend)
        for cmd in (BasicControl.LOCAL, BasicControl.REMOTE, BasicControl.TRIP):
            sample = backend._value_to_sample(cmd)
            assert isinstance(sample, DoubleSample_reply), f"{cmd.name} should use DoubleSample"
            assert sample.value == float(cmd), f"{cmd.name} ordinal mismatch"

    def test_bool_true_converts_to_integer_sample(self):
        """bool True should convert to IntegerSample with value 1."""
        from pacsys.backends.dmq import DMQBackend

        backend = object.__new__(DMQBackend)
        sample = backend._value_to_sample(True)
        assert isinstance(sample, IntegerSample_reply)
        assert sample.value == 1

    def test_bool_false_converts_to_integer_sample(self):
        """bool False should convert to IntegerSample with value 0."""
        from pacsys.backends.dmq import DMQBackend

        backend = object.__new__(DMQBackend)
        sample = backend._value_to_sample(False)
        assert isinstance(sample, IntegerSample_reply)
        assert sample.value == 0

    def test_string_list_converts_to_string_array_sample(self):
        """List of strings should use StringArraySample_reply, not DoubleArraySample."""
        from pacsys.backends.dmq import DMQBackend

        backend = object.__new__(DMQBackend)
        sample = backend._value_to_sample(["hello", "world"])
        assert isinstance(sample, StringArraySample_reply)
        assert list(sample.value) == ["hello", "world"]


# =============================================================================
# Test Reply Conversion
# =============================================================================


class TestReplyToReading:
    """Tests for _reply_to_reading helper function."""

    def test_double_sample_to_reading(self):
        """Test converting DoubleSample_reply to Reading."""
        reply = DoubleSample_reply()
        reply.value = TEMP_VALUE
        reply.time = TIMESTAMP_MILLIS

        reading = _reply_to_reading(reply, TEMP_DEVICE)
        assert reading.value == TEMP_VALUE
        assert reading.value_type == ValueType.SCALAR
        assert reading.error_code == 0
        assert isinstance(reading.timestamp, datetime)

    def test_double_array_sample_to_reading(self):
        """Test converting DoubleArraySample_reply to Reading."""
        reply = DoubleArraySample_reply()
        reply.value = ARRAY_VALUES
        reply.time = TIMESTAMP_MILLIS

        reading = _reply_to_reading(reply, ARRAY_DEVICE)
        assert reading.value_type == ValueType.SCALAR_ARRAY
        np.testing.assert_array_equal(reading.value, np.array(ARRAY_VALUES))

    def test_error_sample_to_reading(self):
        """Test converting ErrorSample_reply to Reading."""
        reply = ErrorSample_reply()
        reply.facilityCode = 1
        reply.errorNumber = ERROR_NOT_FOUND
        reply.time = TIMESTAMP_MILLIS
        reply.message = "Device not found"

        reading = _reply_to_reading(reply, TEMP_DEVICE)
        assert reading.value is None
        assert reading.error_code == ERROR_NOT_FOUND
        assert reading.facility_code == 1
        assert reading.is_error
        assert reading.message == "Device not found"

    def test_string_sample_to_reading(self):
        """Test converting StringSample_reply to Reading."""
        reply = StringSample_reply()
        reply.value = "test string"
        reply.time = TIMESTAMP_MILLIS

        reading = _reply_to_reading(reply, TEMP_DEVICE)
        assert reading.value == "test string"
        assert reading.value_type == ValueType.TEXT

    def test_basic_status_to_reading(self):
        """Test converting BasicStatusSample_reply to Reading."""
        reply = BasicStatusSample_reply()
        reply.time = TIMESTAMP_MILLIS
        # BasicStatus has a nested value with boolean fields
        reply.value = mock.MagicMock(on=True, ready=True, remote=False, positive=None, ramp=None)

        reading = _reply_to_reading(reply, TEMP_DEVICE)
        assert reading.value_type == ValueType.BASIC_STATUS
        assert reading.error_code == 0
        assert isinstance(reading.value, dict)
        assert reading.value["on"] is True
        assert reading.value["ready"] is True
        assert reading.value["remote"] is False
        # None values should be filtered out
        assert "positive" not in reading.value
        assert "ramp" not in reading.value

    def test_analog_alarm_to_reading(self):
        """Test converting AnalogAlarmSample_reply to Reading."""
        reply = AnalogAlarmSample_reply()
        reply.time = TIMESTAMP_MILLIS
        reply.value = mock.MagicMock(
            minimum=0.0,
            maximum=100.0,
            alarm_enable=True,
            alarm_status=False,
            abort=False,
            abort_inhibit=False,
            tries_needed=3,
            tries_now=0,
        )

        reading = _reply_to_reading(reply, TEMP_DEVICE)
        assert reading.value_type == ValueType.ANALOG_ALARM
        assert reading.error_code == 0
        assert reading.value["minimum"] == 0.0
        assert reading.value["maximum"] == 100.0
        assert reading.value["alarm_enable"] is True
        assert reading.value["tries_needed"] == 3

    def test_digital_alarm_to_reading(self):
        """Test converting DigitalAlarmSample_reply to Reading."""
        reply = DigitalAlarmSample_reply()
        reply.time = TIMESTAMP_MILLIS
        reply.value = mock.MagicMock(
            nominal=0,
            mask=0xFF,
            alarm_enable=True,
            alarm_status=True,
            abort=False,
            abort_inhibit=False,
            tries_needed=1,
            tries_now=1,
        )

        reading = _reply_to_reading(reply, TEMP_DEVICE)
        assert reading.value_type == ValueType.DIGITAL_ALARM
        assert reading.error_code == 0
        assert reading.value["nominal"] == 0
        assert reading.value["mask"] == 0xFF
        assert reading.value["alarm_status"] is True

    def test_basic_control_to_reading(self):
        """Test converting BasicControlSample_reply to Reading."""
        reply = BasicControlSample_reply()
        reply.time = TIMESTAMP_MILLIS
        reply.value = 42.0

        reading = _reply_to_reading(reply, TEMP_DEVICE)
        assert reading.value_type == ValueType.SCALAR
        assert reading.value == 42.0
        assert reading.error_code == 0

    def test_binary_sample_to_reading(self):
        """Test converting BinarySample_reply to Reading."""
        reply = BinarySample_reply()
        reply.time = TIMESTAMP_MILLIS
        reply.value = b"\x01\x02\x03"

        reading = _reply_to_reading(reply, TEMP_DEVICE)
        assert reading.value_type == ValueType.RAW
        assert reading.value == b"\x01\x02\x03"
        assert reading.error_code == 0


# =============================================================================
# Test Backend Lifecycle
# =============================================================================


class TestDMQBackendLifecycle:
    """Tests for DMQBackend lifecycle operations."""

    def test_close_idempotent(self):
        """Test that close() can be called multiple times."""
        with _mock_gssapi():
            backend = DMQBackend(host="localhost", auth=_create_mock_auth())
            backend.close()
            backend.close()  # Should not raise

    def test_stop_streaming(self):
        """Test stop_streaming() stops all subscriptions."""
        replies = [make_double_reply(TEMP_VALUE, ref_id=1) for _ in range(10)]
        with _mock_dmq_backend(replies) as backend:
            handle1 = backend.subscribe([TEMP_DEVICE])
            handle2 = backend.subscribe([TEMP_DEVICE_2])
            backend.stop_streaming()
            assert handle1.stopped
            assert handle2.stopped


# =============================================================================
# Test Partial Timeout
# =============================================================================


class TestDMQPartialTimeout:
    def test_partial_timeout_raises_read_error(self):
        """If any device times out in get_many, ReadError must be raised."""
        from unittest.mock import MagicMock

        backend = DMQBackend.__new__(DMQBackend)
        backend._host = "localhost"
        backend._port = 5672
        backend._timeout = 1.0
        backend._closed = False
        backend._auth = MagicMock()
        backend._auth.principal = "test@FNAL.GOV"
        backend._io_thread = None
        backend._channel = None

        # Mock connection with ioloop that calls callbacks synchronously
        mock_conn = MagicMock()
        mock_conn.is_open = True
        mock_conn.ioloop.add_callback_threadsafe = lambda cb: cb()
        backend._select_connection = mock_conn

        def fake_start_read(job):
            r = Reading(
                drf="M:OUTTMP",
                value_type=ValueType.SCALAR,
                value=72.5,
                timestamp=None,
                cycle=0,
            )
            job.readings[0] = r
            job.done_event.set()

        backend._start_read_async = fake_start_read
        backend._ensure_io_thread = lambda: None

        with pytest.raises(ReadError) as exc_info:
            backend.get_many(["M:OUTTMP", "G:AMANDA"], timeout=1.0)

        assert len(exc_info.value.readings) == 2
