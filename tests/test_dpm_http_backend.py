"""
Unit tests for DPMHTTPBackend.

Tests cover:
- Backend initialization and capabilities
- Single device read/get
- Multiple device get_many
- Error handling (device not found, timeout)
- Reply type mapping (scalar, array, text, raw, alarms, status)
- Timestamp conversion
- Context manager usage
- Streaming API surface
- Factory function
"""

from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock

import pytest

from pacsys.backends import timestamp_from_millis
from pacsys.backends.dpm_http import DPMHTTPBackend
from pacsys.types import Reading, ValueType
from pacsys.errors import DeviceError, AuthenticationError
from pacsys.acnet.errors import make_error
from pacsys.dpm_protocol import ListStatus_reply, Raw_reply, StartList_reply

# Shared test helpers
from tests.devices import (
    MockSocketWithReplies,
    make_add_to_list_reply,
    make_device_info,
    make_start_list,
    make_scalar_reply,
    make_scalar_array_reply,
    make_text_reply,
    make_status_reply,
    make_read_sequence,
    TEMP_DEVICE,
    TEMP_VALUE,
    TIMESTAMP_MILLIS,
)


# =============================================================================
# Backend Abstract Base Class Tests
# =============================================================================


class TestDPMHTTPBackendInit:
    """Tests for DPMHTTPBackend input validation."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"host": ""}, "host cannot be empty"),
            ({"port": 0}, "port must be between"),
            ({"port": -1}, "port must be between"),
            ({"port": 65536}, "port must be between"),
            ({"pool_size": 0}, "pool_size must be positive"),
            ({"pool_size": -1}, "pool_size must be positive"),
            ({"timeout": 0}, "timeout must be positive"),
            ({"timeout": -1.0}, "timeout must be positive"),
        ],
    )
    def test_invalid_init_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DPMHTTPBackend(**kwargs)


# =============================================================================
# Timestamp Helpers
# =============================================================================


def test_timestamp_from_millis():
    """Timestamp conversion from milliseconds."""
    millis = 1704067200_000  # Jan 1, 2024 00:00:00 UTC
    dt = timestamp_from_millis(millis)
    assert dt.timestamp() == millis / 1_000


def test_timestamp_from_millis_zero():
    """Timestamp conversion with zero."""
    dt = timestamp_from_millis(0)
    assert dt == datetime.fromtimestamp(0)


# =============================================================================
# Single Device Read Tests
# =============================================================================


class TestSingleDeviceRead:
    """Tests for single device read/get operations."""

    def test_read_scalar_success(self):
        """Successful scalar read."""
        replies = [
            make_device_info(units="degF"),
            make_start_list(),
            make_scalar_reply(),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                value = backend.read(TEMP_DEVICE, timeout=5.0)
                assert value == TEMP_VALUE
            finally:
                backend.close()

    def test_get_returns_reading(self):
        """get() returns a Reading object."""
        replies = [
            make_device_info(units="degF"),
            make_start_list(),
            make_scalar_reply(),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                reading = backend.get(TEMP_DEVICE, timeout=5.0)
                assert isinstance(reading, Reading)
                assert reading.value == TEMP_VALUE
                assert reading.value_type == ValueType.SCALAR
                assert reading.is_success
                assert reading.ok
                assert reading.meta is not None
                assert reading.meta.name == TEMP_DEVICE
                assert reading.meta.units == "degF"
            finally:
                backend.close()

    def test_read_error_raises_device_error(self):
        """read() raises DeviceError on failure."""
        replies = [
            make_start_list(),
            make_status_reply(status=make_error(1, -42)),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                with pytest.raises(DeviceError) as exc_info:
                    backend.read("M:BADDEV", timeout=5.0)
                assert exc_info.value.error_code == -42
            finally:
                backend.close()

    def test_get_error_returns_reading_with_error(self):
        """get() returns Reading with is_error=True on failure."""
        replies = [
            make_start_list(),
            make_status_reply(status=make_error(1, -42)),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                reading = backend.get("M:BADDEV", timeout=5.0)
                assert reading.is_error
                assert not reading.ok
                assert reading.error_code == -42
            finally:
                backend.close()


# =============================================================================
# Multiple Device Read Tests
# =============================================================================


class TestMultipleDeviceRead:
    """Tests for multiple device get_many operations."""

    def test_get_many_multiple_devices(self):
        """Reading multiple devices."""
        replies = [
            make_device_info(name="M:OUTTMP", ref_id=1),
            make_device_info(name="G:AMANDA", ref_id=2, di=12346),
            make_start_list(),
            make_scalar_reply(value=72.5, ref_id=1),
            make_scalar_reply(value=1.234, ref_id=2),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "G:AMANDA"], timeout=5.0)
                assert len(readings) == 2
                assert readings[0].value == 72.5
                assert readings[1].value == 1.234
            finally:
                backend.close()

    def test_get_many_partial_failure(self):
        """Partial failures are returned as error readings."""
        replies = [
            make_device_info(ref_id=1),
            make_start_list(),
            make_scalar_reply(ref_id=1),
            make_status_reply(status=make_error(1, -42), ref_id=2),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many([TEMP_DEVICE, "M:BADDEV"], timeout=5.0)
                assert len(readings) == 2
                assert readings[0].ok
                assert readings[0].value == TEMP_VALUE
                assert readings[1].is_error
                assert readings[1].error_code == -42
            finally:
                backend.close()


# =============================================================================
# Batch Edge Cases Tests
# =============================================================================


class TestBatchEdgeCases:
    """Tests for batch operation edge cases."""

    def test_get_many_duplicate_drfs(self):
        """get_many() handles same device requested multiple times."""
        replies = [
            make_device_info(name="M:OUTTMP", ref_id=1),
            make_device_info(name="G:AMANDA", ref_id=2, di=12346),
            make_device_info(name="M:OUTTMP", ref_id=3),
            make_start_list(),
            make_scalar_reply(value=72.5, ref_id=1),
            make_scalar_reply(value=1.234, ref_id=2),
            make_scalar_reply(value=72.5, ref_id=3),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "G:AMANDA", "M:OUTTMP"], timeout=5.0)
                assert len(readings) == 3
                assert all(r.ok for r in readings)
                assert readings[0].value is not None
                assert readings[2].value is not None
            finally:
                backend.close()

    def test_get_many_order_preserved(self):
        """get_many() returns readings in same order as request."""
        # Replies come back in different order (B, C, A)
        replies = [
            make_device_info(name="C:DEV", ref_id=1, di=12347),
            make_device_info(name="A:DEV", ref_id=2, di=12345),
            make_device_info(name="B:DEV", ref_id=3, di=12346),
            make_start_list(),
            make_scalar_reply(value=2.0, ref_id=3),  # B arrives first
            make_scalar_reply(value=3.0, ref_id=1),  # then C
            make_scalar_reply(value=1.0, ref_id=2),  # then A
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many(["C:DEV", "A:DEV", "B:DEV"], timeout=5.0)
                # Results should be in request order, not reply order
                assert len(readings) == 3
                assert readings[0].value == 3.0  # C:DEV
                assert readings[1].value == 1.0  # A:DEV
                assert readings[2].value == 2.0  # B:DEV
            finally:
                backend.close()

    def test_get_many_single_device(self):
        """get_many() with single device works correctly."""
        replies = make_read_sequence()
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many([TEMP_DEVICE], timeout=5.0)
                assert len(readings) == 1
                assert readings[0].ok
                assert readings[0].value == TEMP_VALUE
            finally:
                backend.close()

    def test_get_many_all_errors(self):
        """get_many() handles all devices returning errors."""
        replies = [
            make_start_list(),
            make_status_reply(status=make_error(1, -42), ref_id=1),
            make_status_reply(status=make_error(1, -43), ref_id=2),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many(["M:BAD1", "M:BAD2"], timeout=5.0)
                assert len(readings) == 2
                assert readings[0].is_error
                assert readings[0].error_code == -42
                assert readings[1].is_error
                assert readings[1].error_code == -43
            finally:
                backend.close()

    def test_get_many_mixed_types(self):
        """get_many() handles mixed value types in one batch."""
        replies = [
            make_device_info(name="M:OUTTMP", ref_id=1, description="Scalar device"),
            make_device_info(name="B:HS23T", ref_id=2, di=12346, description="Array device"),
            make_device_info(name="M:DESC", ref_id=3, di=12347, description="Text device"),
            make_start_list(),
            make_scalar_reply(value=72.5, ref_id=1),
            make_scalar_array_reply(values=[1.0, 2.0, 3.0], ref_id=2),
            make_text_reply(text="Description text", ref_id=3),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many(
                    ["M:OUTTMP", "B:HS23T[0:2]", "M:DESC.DESCRIPTION"],
                    timeout=5.0,
                )
                assert len(readings) == 3
                assert readings[0].value_type == ValueType.SCALAR
                assert readings[0].value == 72.5
                assert readings[1].value_type == ValueType.SCALAR_ARRAY
                assert list(readings[1].value) == [1.0, 2.0, 3.0]
                assert readings[2].value_type == ValueType.TEXT
                assert readings[2].value == "Description text"
            finally:
                backend.close()

    def test_get_many_large_batch(self):
        """get_many() handles larger batches."""
        num_devices = 10
        replies = []
        for i in range(num_devices):
            replies.append(make_device_info(name=f"D:DEV{i:02d}", ref_id=i + 1, di=12345 + i))
        replies.append(make_start_list())
        for i in range(num_devices):
            replies.append(make_scalar_reply(value=float(i * 10), ref_id=i + 1))

        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                drfs = [f"D:DEV{i:02d}" for i in range(num_devices)]
                readings = backend.get_many(drfs, timeout=5.0)
                assert len(readings) == num_devices
                for i, reading in enumerate(readings):
                    assert reading.ok
                    assert reading.value == float(i * 10)
            finally:
                backend.close()


# =============================================================================
# Reply Types Tests
# =============================================================================


class TestReplyTypes:
    """Tests for different reply types."""

    def test_scalar_array_reply(self):
        """ScalarArray_reply handling."""
        replies = [
            make_start_list(),
            make_scalar_array_reply(values=[1.0, 2.0, 3.0, 4.0, 5.0]),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                reading = backend.get("B:HS23T[0:4]", timeout=5.0)
                assert reading.value_type == ValueType.SCALAR_ARRAY
                assert list(reading.value) == [1.0, 2.0, 3.0, 4.0, 5.0]
            finally:
                backend.close()

    def test_text_reply(self):
        """Text_reply handling."""
        replies = [make_start_list(), make_text_reply(text="Hello, ACNET!")]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                reading = backend.get("M:OUTTMP.DESCRIPTION", timeout=5.0)
                assert reading.value_type == ValueType.TEXT
                assert reading.value == "Hello, ACNET!"
            finally:
                backend.close()

    def test_raw_reply(self):
        """Raw_reply handling."""
        start_reply = StartList_reply()
        start_reply.list_id = 1
        start_reply.status = 0

        raw_reply = Raw_reply()
        raw_reply.ref_id = 1
        raw_reply.timestamp = TIMESTAMP_MILLIS
        raw_reply.cycle = 0
        raw_reply.status = 0
        raw_reply.data = b"\x01\x02\x03\x04"

        mock_socket = MockSocketWithReplies(list_id=1, replies=[start_reply, raw_reply])

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                reading = backend.get("M:OUTTMP.RAW", timeout=5.0)
                assert reading.value_type == ValueType.RAW
                assert reading.value == b"\x01\x02\x03\x04"
            finally:
                backend.close()


# =============================================================================
# Streaming API Surface Tests
# =============================================================================


class TestDPMHTTPBackendStreaming:
    """Tests for DPMHTTPBackend streaming methods."""

    def test_subscribe_requires_drfs(self):
        """subscribe() requires non-empty drfs."""
        backend = DPMHTTPBackend()
        try:
            with pytest.raises(ValueError, match="drfs cannot be empty"):
                backend.subscribe([])
        finally:
            backend.close()

    def test_subscribe_on_closed_backend_raises(self):
        """subscribe() on closed backend raises."""
        backend = DPMHTTPBackend()
        backend.close()

        with pytest.raises(RuntimeError, match="Backend is closed"):
            backend.subscribe(["M:OUTTMP@p,1000"])

    def test_callback_mode_cannot_iterate(self):
        """Callback-mode handle cannot be iterated."""
        from pacsys.backends.dpm_http import _DPMHTTPSubscriptionHandle

        backend = DPMHTTPBackend()
        try:
            handle = _DPMHTTPSubscriptionHandle(
                backend=backend,
                drfs=["M:OUTTMP@p,1000"],
                callback=lambda r, h: None,
            )

            with pytest.raises(RuntimeError, match="Cannot iterate subscription with callback"):
                list(handle.readings(timeout=0))
        finally:
            backend.close()

    def test_subscription_handle_context_manager(self):
        """Handle works as context manager."""
        from pacsys.backends.dpm_http import _DPMHTTPSubscriptionHandle

        backend = DPMHTTPBackend()
        try:
            handle = _DPMHTTPSubscriptionHandle(
                backend=backend,
                drfs=["M:OUTTMP@p,1000"],
                callback=None,
            )

            with handle as h:
                assert h is handle
                assert not handle.stopped

            assert handle.stopped
        finally:
            backend.close()

    def test_remove_with_wrong_handle_type(self):
        """remove() rejects wrong handle types."""
        backend = DPMHTTPBackend()
        try:
            with pytest.raises(TypeError, match="Expected _DPMHTTPSubscriptionHandle"):
                backend.remove("not a handle")
        finally:
            backend.close()

    def test_close_calls_stop_streaming(self):
        """close() calls stop_streaming()."""
        backend = DPMHTTPBackend()
        backend.stop_streaming = MagicMock()
        backend.close()

        backend.stop_streaming.assert_called_once()


# =============================================================================
# Multi-Connection Architecture Tests
# =============================================================================


class TestDPMHTTPBackendMultiConnection:
    """Tests for multi-connection streaming architecture."""

    def test_stop_streaming_clears_all_handles(self):
        """stop_streaming() clears all handles."""
        backend = DPMHTTPBackend()
        try:
            backend._handles.append(MagicMock())
            backend._handles.append(MagicMock())

            backend.stop_streaming()

            assert len(backend._handles) == 0
        finally:
            backend.close()


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager_closes(self):
        """Context manager closes backend."""
        with DPMHTTPBackend() as backend:
            assert not backend._closed
        assert backend._closed

    def test_context_manager_on_exception(self):
        """Backend is closed even on exception."""
        try:
            with DPMHTTPBackend() as backend:
                raise ValueError("test error")
        except ValueError:
            pass
        assert backend._closed

    def test_close_multiple_times_safe(self):
        """close() can be called multiple times safely."""
        backend = DPMHTTPBackend()
        backend.close()
        backend.close()
        backend.close()
        assert backend._closed

    def test_close_after_operations(self):
        """close after operations."""
        replies = [make_start_list(), make_scalar_reply()]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            _ = backend.get(TEMP_DEVICE, timeout=5.0)
            backend.close()
            assert backend._closed
            assert backend._pool is None


# =============================================================================
# Write Not Supported Tests
# =============================================================================


class TestWriteNotSupported:
    """Tests for write operations without authentication."""

    def test_write_raises_authentication_error(self):
        """write() without auth raises AuthenticationError."""
        backend = DPMHTTPBackend()
        try:
            with pytest.raises(AuthenticationError, match="not configured for authenticated"):
                backend.write("M:OUTTMP", 72.5)
        finally:
            backend.close()

    def test_write_many_raises_authentication_error(self):
        """write_many() without auth raises AuthenticationError."""
        backend = DPMHTTPBackend()
        try:
            with pytest.raises(AuthenticationError, match="not configured for authenticated"):
                backend.write_many([("M:OUTTMP", 72.5)])
        finally:
            backend.close()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling edge cases."""

    def test_operation_after_close_raises(self):
        """Operations after close raise RuntimeError."""
        backend = DPMHTTPBackend()
        backend.close()

        with pytest.raises(RuntimeError, match="Backend is closed"):
            backend.get("M:OUTTMP")

    def test_heartbeat_ignored(self):
        """ListStatus heartbeat is ignored."""
        heartbeat = ListStatus_reply()
        heartbeat.list_id = 1
        heartbeat.status = 0

        replies = [make_start_list(), heartbeat, make_scalar_reply()]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                reading = backend.get(TEMP_DEVICE, timeout=5.0)
                assert reading.value == TEMP_VALUE
            finally:
                backend.close()

    def test_warning_status(self):
        """Positive status is treated as warning."""
        replies = [
            make_start_list(),
            make_scalar_reply(status=make_error(1, 1)),  # Warning
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                reading = backend.get(TEMP_DEVICE, timeout=5.0)
                assert reading.is_warning
                assert reading.ok  # Warning with data is still usable
                assert reading.value == TEMP_VALUE
            finally:
                backend.close()


# =============================================================================
# AddToList Reply Handling Tests
# =============================================================================


class TestAddToListReplyHandling:
    """Tests that AddToList_reply is correctly filtered in get_many."""

    def test_add_to_list_replies_not_counted_as_data(self):
        """AddToList_reply should be ignored, not treated as data replies."""
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_add_to_list_reply(ref_id=2, status=0),
            make_device_info(name="M:OUTTMP", ref_id=1),
            make_device_info(name="G:AMANDA", ref_id=2, di=12346),
            make_start_list(),
            make_scalar_reply(value=72.5, ref_id=1),
            make_scalar_reply(value=1.234, ref_id=2),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "G:AMANDA"], timeout=5.0)
                assert len(readings) == 2
                assert readings[0].value == 72.5
                assert readings[1].value == 1.234
            finally:
                backend.close()

    def test_add_to_list_error_returns_error_reading(self):
        """AddToList_reply with non-zero status produces error reading."""
        error_status = make_error(1, -42)
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_add_to_list_reply(ref_id=2, status=error_status),
            make_device_info(name="M:OUTTMP", ref_id=1),
            make_start_list(),
            make_scalar_reply(value=72.5, ref_id=1),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "M:BADDEV"], timeout=5.0)
                assert len(readings) == 2
                assert readings[0].ok
                assert readings[0].value == 72.5
                assert readings[1].is_error
                assert readings[1].error_code == -42
            finally:
                backend.close()


# =============================================================================
# StartList Failure Abort Tests
# =============================================================================


class TestStartListFailureAbort:
    """Tests that failed StartList aborts get_many instead of waiting."""

    def test_start_list_failure_returns_timeout_readings(self):
        """Failed StartList should not block -- return timeout readings promptly."""
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_device_info(name="M:OUTTMP", ref_id=1),
            make_start_list(status=make_error(1, -1)),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                import time

                start = time.monotonic()
                readings = backend.get_many(["M:OUTTMP"], timeout=5.0)
                elapsed = time.monotonic() - start
                assert len(readings) == 1
                assert readings[0].is_error
                # Should return quickly, not wait for the full 5s timeout
                assert elapsed < 2.0
            finally:
                backend.close()


# =============================================================================
# Connection Pool Contamination Tests
# =============================================================================


class TestPoolTimeoutDiscard:
    """Tests that pool discards connections on TimeoutError."""

    def test_pool_discards_on_timeout(self):
        """ConnectionPool should discard connection on TimeoutError."""
        from pacsys.pool import ConnectionPool

        mock_conn = MagicMock()
        mock_conn.connected = True

        pool = ConnectionPool(host="localhost", port=6802)
        # Manually inject mock connection to avoid real network
        pool._available = [mock_conn]
        pool._in_use = set()

        try:
            with pool.connection() as conn:
                assert conn is mock_conn
                raise TimeoutError("Receive timeout")
        except TimeoutError:
            pass

        # Connection should have been discarded, not returned to available
        assert mock_conn not in pool._available
        assert mock_conn not in pool._in_use
        mock_conn.close.assert_called_once()

    def test_pool_keeps_connection_on_normal_exit(self):
        """ConnectionPool should return connection on normal exit."""
        from pacsys.pool import ConnectionPool

        mock_conn = MagicMock()
        mock_conn.connected = True

        pool = ConnectionPool(host="localhost", port=6802)
        pool._available = [mock_conn]
        pool._in_use = set()

        with pool.connection() as conn:
            assert conn is mock_conn

        # Connection should be returned to available pool
        assert mock_conn in pool._available
        mock_conn.close.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
