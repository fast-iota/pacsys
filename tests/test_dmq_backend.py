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
from unittest import mock

import numpy as np
import pytest
from pika.adapters.select_connection import SelectConnection

from pacsys.backends.dmq import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    DMQBackend,
    _reply_to_reading,
)
from pacsys.drf_utils import prepare_for_write
from pacsys.errors import AuthenticationError, DeviceError
from pacsys.resources import (
    AnalogAlarmSample_reply,
    BasicControlSample_reply,
    BasicStatusSample_reply,
    BinarySample_reply,
    DigitalAlarmSample_reply,
    DoubleSample_reply,
    DoubleArraySample_reply,
    ErrorSample_reply,
    StringSample_reply,
)
from pacsys.types import BackendCapability, Reading, ValueType
from tests.devices import (
    TEMP_DEVICE,
    TEMP_DEVICE_2,
    TEMP_VALUE,
    ARRAY_DEVICE,
    ARRAY_VALUES,
    TIMESTAMP_MILLIS,
    ERROR_NOT_FOUND,
    MockGSSAPIModule,
)


def _create_mock_auth():
    """Create a mock KerberosAuth for testing without real Kerberos.

    Returns a KerberosAuth instance that uses MockGSSAPIModule.
    """
    with mock.patch.dict("sys.modules", {"gssapi": MockGSSAPIModule()}):
        from pacsys.auth import KerberosAuth

        return KerberosAuth()


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
        if self._running:
            self._process_callbacks()

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
            try:
                cb()
            except Exception:
                pass  # Swallow errors like real pika


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
            try:
                cb(self, Exception("Channel closed"))
            except Exception:
                pass

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

                if self._on_message_callback:
                    try:
                        self._on_message_callback(self, method, None, reply_bytes)
                    except Exception:
                        pass
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
            try:
                self._on_close_callback(self, Exception("Connection closed"))
            except Exception:
                pass
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


class TestDMQBackendInit:
    """Tests for DMQBackend initialization."""

    def test_backend_requires_auth(self):
        """Test that auth is required for DMQ backend."""
        with pytest.raises(AuthenticationError, match="DMQ requires KerberosAuth"):
            DMQBackend()

    def test_backend_init_defaults(self):
        """Test that default parameters are set correctly."""
        with mock.patch("pika.BlockingConnection"):
            auth = _create_mock_auth()
            backend = DMQBackend(auth=auth)
            try:
                assert backend.host == DEFAULT_HOST
                assert backend.port == DEFAULT_PORT
                assert backend.timeout == DEFAULT_TIMEOUT
            finally:
                backend.close()

    def test_backend_init_custom_params(self):
        """Test initialization with custom parameters."""
        with mock.patch("pika.BlockingConnection"):
            auth = _create_mock_auth()
            backend = DMQBackend(
                host="custom.example.com",
                port=5673,
                timeout=5.0,
                auth=auth,
            )
            try:
                assert backend.host == "custom.example.com"
                assert backend.port == 5673
                assert backend.timeout == 5.0
            finally:
                backend.close()

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

    def test_backend_capabilities(self):
        """Test that backend has correct capabilities (including WRITE since auth required)."""
        with mock.patch("pika.BlockingConnection"):
            auth = _create_mock_auth()
            backend = DMQBackend(auth=auth)
            try:
                caps = backend.capabilities
                assert BackendCapability.READ in caps
                assert BackendCapability.WRITE in caps  # Auth always required now
                assert BackendCapability.AUTH_KERBEROS in caps
                assert BackendCapability.STREAM in caps
                assert BackendCapability.BATCH in caps
            finally:
                backend.close()


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

    def test_get_many_empty_list(self):
        """Test get_many with empty list returns empty list."""
        backend = DMQBackend(host="localhost", auth=_create_mock_auth())
        try:
            assert backend.get_many([]) == []
        finally:
            backend.close()

    def test_get_many_same_device_different_properties(self):
        """Regression: routing key matching when ref_id is missing."""
        replies = [make_double_reply(1.0), make_double_reply(2.0)]
        routing_keys = [f"R.{TEMP_DEVICE}@I", f"R.{TEMP_DEVICE}.SETTING@I"]
        with _mock_dmq_backend(replies, routing_keys) as backend:
            readings = backend.get_many([TEMP_DEVICE, f"{TEMP_DEVICE}.SETTING"])
            assert len(readings) == 2
            assert readings[0].value == 1.0
            assert readings[1].value == 2.0

    def test_get_many_out_of_order_routing_keys(self):
        """Test correct matching when responses arrive out of order."""
        replies = [make_double_reply(99.0), make_double_reply(42.0)]
        routing_keys = [f"R.{TEMP_DEVICE}.SETTING@I", f"R.{TEMP_DEVICE}@I"]
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
        """Test that timeout returns error reading."""
        with _mock_dmq_backend([], timeout=0.5) as backend:
            reading = backend.get(TEMP_DEVICE, timeout=0.5)
            assert reading.is_error
            assert reading.error_code == -6  # ACNET_REQTMO
            assert "timeout" in reading.message.lower()

    def test_read_closed_backend(self):
        """Test that read on closed backend raises RuntimeError."""
        backend = DMQBackend(host="localhost", auth=_create_mock_auth())
        backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            backend.read(TEMP_DEVICE)


# =============================================================================
# Test Subscribe Operations
# =============================================================================


class TestDMQBackendSubscribe:
    """Tests for DMQBackend subscribe operations."""

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

    def test_subscribe_ref_ids(self):
        """Test that handle has correct ref_ids."""
        with _mock_dmq_backend([make_double_reply(TEMP_VALUE, ref_id=1)]) as backend:
            handle = backend.subscribe([TEMP_DEVICE, TEMP_DEVICE_2])
            assert handle.ref_ids == [1, 2]
            handle.stop()


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
        """Capture SETTING messages."""
        super().basic_publish(exchange, routing_key, body, properties)
        # Queue write response if this is a SETTING message
        if (
            routing_key.startswith("S.")
            and properties
            and hasattr(properties, "correlation_id")
            and properties.correlation_id
        ):
            self._pending_writes.append(properties.correlation_id)

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
                for _ in range(300):  # ~6 seconds max
                    if not self._is_open:
                        break
                    while self._pending_writes and self._on_message_callback:
                        corr_id = self._pending_writes.pop(0)
                        response_bytes = self._write_response_factory(corr_id)

                        method = mock.MagicMock()
                        method.routing_key = f"R.{TEMP_DEVICE}"
                        method.delivery_tag = 1

                        props = mock.MagicMock()
                        props.correlation_id = corr_id

                        try:
                            self._on_message_callback(self, method, props, response_bytes)
                        except Exception:
                            pass
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

    def test_write_many_empty_list(self):
        """Test that write_many() with empty list returns empty list."""
        backend = DMQBackend(host="localhost", auth=_create_mock_auth())
        try:
            assert backend.write_many([]) == []
        finally:
            backend.close()

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

    def test_write_auth_failure_raises(self):
        """Test that GSS context failure during async write raises AuthenticationError."""
        factory, mock_conn = create_write_select_connection_factory()

        def failing_gss():
            raise RuntimeError("Kerberos ticket expired")

        with (
            mock.patch("pika.BlockingConnection"),
            mock.patch.object(SelectConnection, "__new__", side_effect=factory),
            mock.patch.object(DMQBackend, "_create_gss_context", side_effect=failing_gss),
        ):
            backend = DMQBackend(host="localhost", auth=_create_mock_auth())
            try:
                with pytest.raises(AuthenticationError, match="GSS context creation failed"):
                    backend.write(TEMP_DEVICE, TEMP_VALUE, timeout=5.0)
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

        with pytest.raises(ValueError, match="Cannot mix analog.*and digital"):
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

    def test_context_manager(self):
        """Test backend as context manager."""
        with mock.patch.object(DMQBackend, "_create_gss_context", return_value=_mock_gss_context()):
            with DMQBackend(host="localhost", auth=_create_mock_auth()) as backend:
                assert backend is not None

    def test_close_idempotent(self):
        """Test that close() can be called multiple times."""
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
