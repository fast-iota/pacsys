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
- Factory function
"""

import time
from unittest import mock
from unittest.mock import MagicMock

import pytest

from pacsys.acnet.errors import DAE_LJ_NO_DATA, ERR_TIMEOUT, make_error
from pacsys.backends.dpm_http import DPMHTTPBackend, _value_to_setting
from pacsys.dpm_connection import DPMConnectionError
from pacsys.dpm_protocol import ListStatus_reply, Raw_reply, StartList_reply
from pacsys.errors import AuthenticationError, DeviceError, ReadError
from pacsys.pool import PoolExhaustedError
from pacsys.types import Reading, ValueType

# Shared test helpers
from tests.devices import (
    TEMP_DEVICE,
    TEMP_VALUE,
    TIMESTAMP_MILLIS,
    MockSocketWithReplies,
    make_add_to_list_reply,
    make_apply_settings_reply,
    make_device_info,
    make_read_sequence,
    make_scalar_array_reply,
    make_scalar_reply,
    make_start_list,
    make_status_reply,
    make_text_reply,
)
from tests.test_dpm_http_auth import create_mock_kerberos_auth

# =============================================================================
# Backend Abstract Base Class Tests
# =============================================================================


class TestDPMHTTPBackendInit:
    """Tests for DPMHTTPBackend input validation."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"host": ""}, "host cannot be empty"),
            ({"port": 0}, "port must be between"),
            ({"port": -1}, "port must be between"),
            ({"port": 65536}, "port must be between"),
            ({"pool_size": 0}, "pool_size must be positive"),
            ({"pool_size": -1}, "pool_size must be positive"),
            ({"timeout": 0}, "timeout must be positive"),
            ({"timeout": -1.0}, "timeout must be positive"),
            ({"timeout": None}, "timeout must be positive"),
            ({"timeout": float("inf")}, "timeout must be positive"),
            ({"timeout": float("nan")}, "timeout must be positive"),
        ],
    )
    def test_invalid_init_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DPMHTTPBackend(**kwargs)


class TestValueToSetting:
    def test_rejects_multidimensional_array(self):
        import numpy as np

        with pytest.raises(TypeError, match="one-dimensional"):
            _value_to_setting(1, np.ones((2, 2)))

    @pytest.mark.parametrize("value", [["on", 1], [1, "on"]])
    def test_rejects_mixed_text_array(self, value):
        with pytest.raises(TypeError, match="only strings"):
            _value_to_setting(1, value)

    def test_write_prevalidates_before_getting_connection(self):
        backend = DPMHTTPBackend(auth=create_mock_kerberos_auth())
        try:
            with mock.patch.object(backend, "_get_write_connection") as get_connection:
                with pytest.raises(TypeError, match="only strings"):
                    backend.write_many([("M:OUTTMP", ["on", 1])])
            get_connection.assert_not_called()
        finally:
            backend.close()

    def test_write_rejects_nonpositive_call_timeout_before_connecting(self):
        backend = DPMHTTPBackend(auth=create_mock_kerberos_auth())
        try:
            with mock.patch.object(backend, "_get_write_connection") as get_connection:
                with pytest.raises(ValueError, match="timeout must be positive"):
                    backend.write_many([("M:OUTTMP", 1.0)], timeout=0)
            get_connection.assert_not_called()
        finally:
            backend.close()


class TestReactorThreadGuard:
    @pytest.mark.parametrize("method", ["get", "write", "subscribe"])
    def test_blocking_call_from_reactor_thread_raises(self, method):
        """Pool I/O on the reactor thread stalls every subscription - fail immediately instead."""
        import asyncio

        backend = DPMHTTPBackend(auth=create_mock_kerberos_auth())
        backend._ensure_reactor()
        calls = {
            "get": lambda: backend.get("M:OUTTMP", timeout=0.01),
            "write": lambda: backend.write("Z:ACLTST", 1.0, timeout=0.01),
            "subscribe": lambda: backend.subscribe(["M:OUTTMP@p,1000"]),
        }

        async def on_reactor():
            return calls[method]()

        try:
            fut = asyncio.run_coroutine_threadsafe(on_reactor(), backend._loop)
            with pytest.raises(RuntimeError, match="reactor thread"):
                fut.result(timeout=2.0)
        finally:
            backend.close()


class TestAlarmDictExpansion:
    @pytest.mark.parametrize("key", ["alarm_status", "abort", "tries_now"])
    def test_rejects_readonly_keys(self, key):
        backend = DPMHTTPBackend.__new__(DPMHTTPBackend)

        with pytest.raises(ValueError, match="Read-only alarm dict keys"):
            backend.write("Z:TEST.ANALOG", {key: 1})

    def test_rejects_empty_dict(self):
        backend = DPMHTTPBackend.__new__(DPMHTTPBackend)

        with pytest.raises(ValueError, match="at least one writable key"):
            backend.write("Z:TEST.ANALOG", {})

    def test_accepts_shared_only_dict_when_property_identifies_type(self):
        backend = DPMHTTPBackend.__new__(DPMHTTPBackend)

        assert backend._expand_alarm_dict("Z:TEST.ANALOG", {"alarm_enable": True}) == [
            ("Z:TEST.ANALOG.ALARM_ENABLE", 1)
        ]


# =============================================================================
# Single Device Read Tests
# =============================================================================


class TestSingleDeviceRead:
    """Tests for single device read/get operations."""

    def test_read_scalar_success(self):
        """Successful scalar read."""
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
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
            make_add_to_list_reply(ref_id=1, status=0),
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

    def test_get_many_partial_failure(self):
        """Partial failures are returned as error readings."""
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_add_to_list_reply(ref_id=2, status=0),
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
# Historical Data Source Tests
# =============================================================================


LOGGERSINGLE_DRF = "M:OUTTMP<-LOGGERSINGLE:ArkIv:1736942400:60"


class TestHistoricalReads:
    def test_loggersingle_mixed_batch_completes_and_reuses_connection(self):
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_add_to_list_reply(ref_id=2, status=0),
            make_device_info(ref_id=1),
            make_device_info(name="G:AMANDA", ref_id=2),
            make_start_list(),
            make_scalar_reply(value=71.25, ref_id=1),
            make_scalar_reply(value=1.234, ref_id=2),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many([LOGGERSINGLE_DRF, "G:AMANDA"], timeout=1.0)
                assert backend._get_pool().available_count == 1
            finally:
                backend.close()

        assert readings[0].ok
        assert readings[0].value == 71.25
        assert readings[0].value_type == ValueType.SCALAR
        assert readings[1].value == 1.234

    def test_loggersingle_error_is_preserved(self):
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_device_info(ref_id=1),
            make_start_list(),
            make_status_reply(status=DAE_LJ_NO_DATA, ref_id=1),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            with DPMHTTPBackend() as backend:
                reading = backend.get(LOGGERSINGLE_DRF, timeout=1.0)

        assert reading.facility_code == 66
        assert reading.error_code == -64

    @pytest.mark.parametrize(
        "drf",
        [
            "M:OUTTMP<-LOGGER:1736942400000:1736946000000",
            "M:OUTTMP<-LOGGERDURATION:60000",
        ],
    )
    def test_chunked_logger_still_waits_for_terminator(self, drf):
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_device_info(ref_id=1),
            make_start_list(),
            make_scalar_array_reply(values=[1.0, 2.0], ref_id=1),
            make_scalar_array_reply(values=[3.0], ref_id=1),
            make_scalar_array_reply(values=[], ref_id=1),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            with DPMHTTPBackend() as backend:
                reading = backend.get(drf, timeout=1.0)

        assert reading.ok
        assert reading.value_type == ValueType.SCALAR_ARRAY
        assert reading.value.tolist() == [1.0, 2.0, 3.0]


# =============================================================================
# Batch Edge Cases Tests
# =============================================================================


class TestBatchEdgeCases:
    """Tests for batch operation edge cases."""

    def test_get_many_duplicate_drfs(self):
        """get_many() handles same device requested multiple times."""
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_add_to_list_reply(ref_id=2, status=0),
            make_add_to_list_reply(ref_id=3, status=0),
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
            make_add_to_list_reply(ref_id=1, status=0),
            make_add_to_list_reply(ref_id=2, status=0),
            make_add_to_list_reply(ref_id=3, status=0),
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
            make_add_to_list_reply(ref_id=1, status=0),
            make_add_to_list_reply(ref_id=2, status=0),
            make_add_to_list_reply(ref_id=3, status=0),
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
        replies = [make_add_to_list_reply(ref_id=i + 1, status=0) for i in range(num_devices)]
        replies.extend(make_device_info(name=f"D:DEV{i:02d}", ref_id=i + 1, di=12345 + i) for i in range(num_devices))
        replies.append(make_start_list())
        replies.extend(make_scalar_reply(value=float(i * 10), ref_id=i + 1) for i in range(num_devices))

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
            make_add_to_list_reply(ref_id=1, status=0),
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
        replies = [make_add_to_list_reply(ref_id=1, status=0), make_start_list(), make_text_reply(text="Hello, ACNET!")]
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

        add_reply = make_add_to_list_reply(ref_id=1, status=0)
        mock_socket = MockSocketWithReplies(list_id=1, replies=[add_reply, start_reply, raw_reply])

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                reading = backend.get("M:OUTTMP.RAW", timeout=5.0)
                assert reading.value_type == ValueType.RAW
                assert reading.value == b"\x01\x02\x03\x04"
            finally:
                backend.close()


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


class TestWriteConnectionAuthContext:
    """Tests for write connection pooling across auth context changes."""

    def test_reuse_discards_pooled_connection_if_principal_changed(self):
        backend = DPMHTTPBackend()
        acquired = None
        try:
            backend._auth = create_mock_kerberos_auth("new-user@FNAL.GOV")

            stale_wc = MagicMock()
            stale_wc.is_stale.return_value = False
            stale_wc.principal = "old-user@FNAL.GOV"
            stale_wc.role = None
            stale_wc.conn.connected = True
            stale_wc.conn.list_id = 100
            backend._write_connections = [stale_wc]

            new_conn = MagicMock()
            new_conn.connected = True
            new_conn.list_id = 200

            with mock.patch("pacsys.backends.dpm_http.DPMConnection", return_value=new_conn):
                with mock.patch.object(backend, "_authenticate_connection", return_value=(b"mic", b"1234")):
                    with mock.patch.object(backend, "_enable_settings"):
                        acquired = backend._get_write_connection()

            stale_wc.close.assert_called_once()
            assert acquired.conn is new_conn
            assert acquired.principal == "new-user@FNAL.GOV"
            assert acquired.role is None
        finally:
            if acquired is not None:
                backend._release_write_connection(acquired)
            backend.close()

    def test_capacity_failure_returns_retry_results(self):
        backend = DPMHTTPBackend(auth=create_mock_kerberos_auth())
        try:
            with mock.patch.object(
                backend, "_get_write_connection", side_effect=PoolExhaustedError("too many connections")
            ):
                results = backend.write_many([(TEMP_DEVICE, 1.0)])

            assert len(results) == 1
            assert not results[0].success
            assert "too many connections" in results[0].message
        finally:
            backend.close()

    def test_write_pool_uses_typed_capacity_error(self):
        backend = DPMHTTPBackend(auth=create_mock_kerberos_auth())
        backend._write_in_flight = 4
        try:
            with pytest.raises(PoolExhaustedError, match="Too many concurrent write connections"):
                backend._get_write_connection()
        finally:
            backend.close()

    @pytest.mark.parametrize("error", [TypeError("programming bug"), RuntimeError("programming bug")])
    def test_unexpected_acquisition_error_propagates(self, error):
        backend = DPMHTTPBackend(auth=create_mock_kerberos_auth())
        try:
            with mock.patch.object(backend, "_get_write_connection", side_effect=error):
                with pytest.raises(type(error), match="programming bug"):
                    backend.write_many([(TEMP_DEVICE, 1.0)])
        finally:
            backend.close()

    def test_connect_deadline_expiry_returns_err_timeout(self):
        """Deadline expiry inside DPMConnection.connect() is ERR_TIMEOUT, not a retryable connection error."""
        backend = DPMHTTPBackend(auth=create_mock_kerberos_auth())
        conn = MagicMock()
        conn.connect.side_effect = TimeoutError("Timed out during DPM connection")
        try:
            with mock.patch("pacsys.backends.dpm_http.DPMConnection", return_value=conn):
                results = backend.write_many([(TEMP_DEVICE, 1.0)], timeout=0.5)
        finally:
            backend.close()
        assert results[0].error_code == ERR_TIMEOUT
        conn.close.assert_called_once()
        assert backend._write_in_flight == 0

    def test_expired_setup_sends_nothing_and_returns_timeout(self):
        backend = DPMHTTPBackend(auth=create_mock_kerberos_auth())
        wc = MagicMock()
        wc.conn.list_id = 1
        backend._write_in_flight = 1

        def delayed_checkout(_deadline):
            time.sleep(0.02)
            return wc

        try:
            with mock.patch.object(backend, "_get_write_connection", side_effect=delayed_checkout):
                results = backend.write_many([(TEMP_DEVICE, 1.0)], timeout=0.01)
        finally:
            backend.close()

        assert results[0].error_code == ERR_TIMEOUT
        wc.conn.send_messages_batch.assert_not_called()
        wc.close.assert_called_once()

    def test_retry_reuses_original_deadline(self):
        backend = DPMHTTPBackend(auth=create_mock_kerberos_auth())
        first = MagicMock()
        first.conn.list_id = 1
        second = MagicMock()
        second.conn.list_id = 2
        apply_reply = make_apply_settings_reply([(1, 0)])
        backend._write_in_flight = 2

        try:
            with (
                mock.patch.object(backend, "_get_write_connection", side_effect=[first, second]) as get_connection,
                mock.patch.object(
                    backend,
                    "_execute_write",
                    side_effect=[DPMConnectionError("stale"), (apply_reply, {})],
                ) as execute,
            ):
                results = backend.write_many([(TEMP_DEVICE, 1.0)], timeout=1.0)
        finally:
            backend.close()

        assert results[0].success
        assert get_connection.call_args_list[0].args[0] == get_connection.call_args_list[1].args[0]
        assert execute.call_args_list[0].args[-1] == execute.call_args_list[1].args[-1]


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

    def test_pool_exhaustion_is_normalized(self):
        backend = DPMHTTPBackend()
        pool = MagicMock()
        pool.connection.return_value.__enter__.side_effect = PoolExhaustedError("pool exhausted")
        backend._pool = pool
        try:
            with pytest.raises(ReadError) as exc_info:
                backend.get(TEMP_DEVICE)
            assert isinstance(exc_info.value.__cause__, PoolExhaustedError)
        finally:
            backend.close()

    def test_unexpected_pool_error_propagates(self):
        backend = DPMHTTPBackend()
        pool = MagicMock()
        pool.connection.return_value.__enter__.side_effect = TypeError("programming bug")
        backend._pool = pool
        try:
            with pytest.raises(TypeError, match="programming bug"):
                backend.get(TEMP_DEVICE)
        finally:
            backend.close()

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

    def test_decode_error_connection_not_repooled(self):
        """A boundary-safe decode error keeps connected=True, but the un-stopped
        connection must be closed, not released back into the pool."""

        class _UndecodableReply:
            def marshal(self):
                return b"\xff\xfe\xfd"  # well-framed, unmarshal fails

        replies = [make_start_list(), _UndecodableReply()]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                with pytest.raises(ReadError) as exc_info:
                    backend.get_many([TEMP_DEVICE], timeout=2.0)
                assert not exc_info.value.readings[0].ok
                assert backend._get_pool().available_count == 0
                assert mock_socket._closed
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
# Delayed Error Response Tests
# =============================================================================


class TestDelayedErrorResponse:
    """Tests for DPM error responses that arrive after a mini-timeout.

    DPM returns DPM_PEND for nonexistent devices with a ~3s delay.
    The recv loop uses 2s mini-timeouts. A transient timeout must NOT kill
    the connection — it should retry and receive the delayed response.
    """

    def test_read_nonexistent_raises_device_error_after_delay(self):
        """read() raises DeviceError even when the error reply is delayed.

        Simulates: AddToList ok, StartList ok, then a 2s gap (socket.timeout)
        before the DPM_PEND Status_reply arrives.
        """
        error_status = make_error(17, 1)  # DPM_PEND: facility=17, error=1
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_start_list(),
            TimeoutError("delayed response"),  # 2s mini-timeout fires here
            make_status_reply(status=error_status, ref_id=1),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                with pytest.raises(DeviceError) as exc_info:
                    backend.read("Z:NOTFND", timeout=10.0)
                assert exc_info.value.error_code == 1
                assert exc_info.value.facility_code == 17
            finally:
                backend.close()

    def test_get_nonexistent_returns_error_reading_after_delay(self):
        """get() returns error Reading when DPM_PEND arrives after delay."""
        error_status = make_error(17, 1)
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_start_list(),
            TimeoutError("delayed response"),
            make_status_reply(status=error_status, ref_id=1),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                reading = backend.get("Z:NOTFND", timeout=10.0)
                assert not reading.ok
                assert reading.error_code == 1
                assert reading.facility_code == 17
            finally:
                backend.close()

    def test_get_many_partial_failure_with_delay(self):
        """get_many() returns mix of success and delayed error readings."""
        error_status = make_error(17, 1)
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_add_to_list_reply(ref_id=2, status=0),
            make_add_to_list_reply(ref_id=3, status=0),
            make_device_info(name="M:OUTTMP", ref_id=1),
            make_device_info(name="G:AMANDA", ref_id=3, di=12346),
            make_start_list(),
            make_scalar_reply(value=72.5, ref_id=1),
            TimeoutError("delayed DPM_PEND"),
            make_status_reply(status=error_status, ref_id=2),
            make_scalar_reply(value=1.234, ref_id=3),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "Z:NOTFND", "G:AMANDA"], timeout=10.0)
                assert len(readings) == 3
                assert readings[0].ok
                assert readings[0].value == 72.5
                assert not readings[1].ok
                assert readings[1].error_code == 1
                assert readings[2].ok
                assert readings[2].value == 1.234
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
        """Failed StartList should not block -- raise ReadError promptly."""
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
                with pytest.raises(ReadError) as exc_info:
                    backend.get_many(["M:OUTTMP"], timeout=5.0)
                elapsed = time.monotonic() - start
                readings = exc_info.value.readings
                assert len(readings) == 1
                assert readings[0].is_error
                # Should raise quickly, not wait for the full 5s timeout
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


# =============================================================================
# Get Many Timeout Connection Cleanup Tests
# =============================================================================


class TestGetManyTimeoutConnectionCleanup:
    """Connection must be discarded (not reused) when get_many times out."""

    def test_timeout_closes_connection(self):
        """When recv loop gets fewer replies than expected, connection must be closed."""
        import threading
        from contextlib import contextmanager

        backend = DPMHTTPBackend.__new__(DPMHTTPBackend)
        backend._timeout = 0.5
        backend._pool_lock = threading.Lock()
        backend._pool_size = 2
        backend._closed = False

        mock_conn = MagicMock()
        mock_conn.list_id = 1
        mock_conn.connected = True

        # Simulate: 2 devices requested, only 1 replies, then timeout
        call_count = [0]

        def mock_recv(timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return make_add_to_list_reply(ref_id=1, status=0)
            if call_count[0] == 2:
                return make_add_to_list_reply(ref_id=2, status=0)
            if call_count[0] == 3:
                return make_start_list(status=0)
            if call_count[0] == 4:
                return make_device_info(ref_id=1)
            if call_count[0] == 5:
                return make_scalar_reply(ref_id=1)
            # Device 2 never replies — always timeout
            raise TimeoutError

        mock_conn.recv_message = mock_recv
        mock_conn.send_messages_batch = MagicMock()
        mock_conn.send_message = MagicMock()
        mock_conn.close = MagicMock()

        @contextmanager
        def mock_connection(wait_timeout=None):
            yield mock_conn

        mock_pool = MagicMock()
        mock_pool.connection = mock_connection
        backend._get_pool = MagicMock(return_value=mock_pool)

        # Request 2 devices but only 1 will reply — should raise ReadError
        with pytest.raises(ReadError):
            backend.get_many(["Z:ACLTST", "Z:MISSING"], timeout=0.5)

        # The connection MUST be closed so the pool discards it
        mock_conn.close.assert_called()

    def test_recv_gets_full_remaining_budget(self):
        """No 2 s recv slice: a reply body straddling a slice boundary used to kill the connection."""
        import threading
        from contextlib import contextmanager

        backend = DPMHTTPBackend.__new__(DPMHTTPBackend)
        backend._timeout = 10.0
        backend._pool_lock = threading.Lock()
        backend._pool_size = 2
        backend._closed = False

        mock_conn = MagicMock()
        mock_conn.list_id = 1
        mock_conn.connected = True
        replies = iter(
            [
                make_add_to_list_reply(ref_id=1, status=0),
                make_start_list(status=0),
                make_device_info(ref_id=1),
                make_scalar_reply(ref_id=1),
            ]
        )
        timeouts = []

        def mock_recv(timeout=None):
            timeouts.append(timeout)
            return next(replies)

        mock_conn.recv_message = mock_recv
        mock_conn.send_messages_batch = MagicMock()
        mock_conn.send_message = MagicMock()

        @contextmanager
        def mock_connection(wait_timeout=None):
            yield mock_conn

        mock_pool = MagicMock()
        mock_pool.connection = mock_connection
        backend._get_pool = MagicMock(return_value=mock_pool)

        backend.get_many(["Z:ACLTST"], timeout=10.0)
        assert timeouts and all(t > 2.0 for t in timeouts), timeouts

    def test_stop_send_failure_returns_readings(self):
        """A failed StopList/ClearList send after a complete read keeps the readings."""
        import threading
        from contextlib import contextmanager

        backend = DPMHTTPBackend.__new__(DPMHTTPBackend)
        backend._timeout = 0.5
        backend._pool_lock = threading.Lock()
        backend._pool_size = 2
        backend._closed = False

        mock_conn = MagicMock()
        mock_conn.list_id = 1
        mock_conn.connected = True

        call_count = [0]

        def mock_recv(timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return make_add_to_list_reply(ref_id=1, status=0)
            if call_count[0] == 2:
                return make_start_list(status=0)
            if call_count[0] == 3:
                return make_device_info(ref_id=1)
            if call_count[0] == 4:
                return make_scalar_reply(ref_id=1)
            raise TimeoutError

        send_calls = [0]

        def mock_send_batch(msgs):
            send_calls[0] += 1
            if send_calls[0] == 2:  # first batch = setup, second = stop/clear
                raise RuntimeError("unexpected")

        mock_conn.recv_message = mock_recv
        mock_conn.send_messages_batch = mock_send_batch
        mock_conn.send_message = MagicMock()
        mock_conn.close = MagicMock()

        @contextmanager
        def mock_connection(wait_timeout=None):
            yield mock_conn

        mock_pool = MagicMock()
        mock_pool.connection = mock_connection
        backend._get_pool = MagicMock(return_value=mock_pool)

        readings = backend.get_many(["Z:ACLTST"], timeout=0.5)
        assert readings[0].value == TEMP_VALUE
        mock_conn.close.assert_called()


# =============================================================================
# Pooled-connection hygiene
# =============================================================================


class TestPooledConnectionHygiene:
    """Stale/spurious replies must never be counted or re-pooled into the next call."""

    def test_ref0_status_is_job_error(self):
        """Ref-0 Status_reply = job start failure: surfaced, connection closed."""
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_start_list(),
            make_status_reply(status=make_error(1, -42), ref_id=0),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)
        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                with pytest.raises(ReadError, match="job start failed"):
                    backend.get_many([TEMP_DEVICE], timeout=1.0)
                assert mock_socket._closed
            finally:
                backend.close()

    def test_ref0_status_preserves_later_healthy_reply(self):
        error_status = make_error(1, -42)
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_add_to_list_reply(ref_id=2, status=0),
            make_device_info(name=TEMP_DEVICE, ref_id=1),
            make_device_info(name="G:AMANDA", ref_id=2),
            make_start_list(),
            make_status_reply(status=error_status, ref_id=0),
            make_scalar_reply(value=1.234, ref_id=2),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)
        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                with pytest.raises(ReadError) as exc_info:
                    backend.get_many([TEMP_DEVICE, "G:AMANDA"], timeout=0.01)
                assert exc_info.value.readings[0].error_code == -42
                assert exc_info.value.readings[1].value == 1.234
            finally:
                backend.close()

    def test_ref0_status_is_used_for_incomplete_logger(self):
        error_status = make_error(1, -42)
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_start_list(),
            make_status_reply(status=error_status, ref_id=0),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)
        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                with pytest.raises(ReadError) as exc_info:
                    backend.get_many(["M:OUTTMP<-LOGGER:1736942400000:1736946000000"], timeout=0.01)
                assert exc_info.value.readings[0].error_code == -42
                assert "incomplete" not in (exc_info.value.readings[0].message or "").lower()
            finally:
                backend.close()

    def test_stale_out_of_range_reply_ignored(self):
        """A reply with a ref outside 1..N is stale/unknown — never counted."""
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_device_info(),
            make_start_list(),
            make_scalar_reply(value=99.0, ref_id=7),  # stale ref from a previous borrow
            make_scalar_reply(value=TEMP_VALUE, ref_id=1),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)
        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many([TEMP_DEVICE], timeout=2.0)
                assert readings[0].ok
                assert readings[0].value == TEMP_VALUE
                assert not mock_socket._closed  # @I read — safe to re-pool
            finally:
                backend.close()

    def test_repeating_event_closes_connection(self):
        """A connection that carried @p must be closed, not re-pooled."""
        replies = [
            make_add_to_list_reply(ref_id=1, status=0),
            make_device_info(),
            make_start_list(),
            make_scalar_reply(value=1.0, ref_id=1),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)
        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            try:
                readings = backend.get_many(["M:OUTTMP@p,1000"], timeout=2.0)
                assert readings[0].value == 1.0
                assert mock_socket._closed
            finally:
                backend.close()

    def test_no_cross_call_misattribution(self):
        """End-to-end: an extra periodic reply must not become the next call's value."""
        sock1 = MockSocketWithReplies(
            list_id=1,
            replies=[
                make_add_to_list_reply(ref_id=1, status=0),
                make_device_info(),
                make_start_list(),
                make_scalar_reply(value=1.0, ref_id=1),
                make_scalar_reply(value=2.0, ref_id=1),  # extra periodic sample left in stream
            ],
        )
        sock2 = MockSocketWithReplies(
            list_id=1,
            replies=[
                make_add_to_list_reply(ref_id=1, status=0),
                make_device_info(name="G:AMANDA"),
                make_start_list(),
                make_scalar_reply(value=55.5, ref_id=1),
            ],
        )
        with mock.patch("socket.socket", side_effect=[sock1, sock2]):
            backend = DPMHTTPBackend()
            try:
                r1 = backend.get_many(["M:OUTTMP@p,1000"], timeout=2.0)
                assert r1[0].value == 1.0
                r2 = backend.get_many(["G:AMANDA"], timeout=2.0)
                assert r2[0].ok
                assert r2[0].value == 55.5  # NOT 2.0 — the stale sample from call 1
            finally:
                backend.close()


# =============================================================================
# Subscription connect failure
# =============================================================================


class TestDPMSubscribeCloseRace:
    def test_subscribe_close_race_raises(self):
        """_closed flipping between the pre-check and the locked append must raise."""
        backend = DPMHTTPBackend()
        try:
            orig = backend._ensure_reactor

            def racy():
                orig()
                backend._closed = True  # concurrent close() lands here

            with (
                mock.patch.object(backend, "_ensure_reactor", side_effect=racy),
                pytest.raises(RuntimeError, match="Backend is closed"),
            ):
                backend.subscribe(["M:OUTTMP@p,1000"], callback=lambda r, h: None)
            assert backend._handles == []
        finally:
            backend._closed = False  # let close() run fully
            backend.close()


class TestDPMSubscribeConnectFailure:
    """A dead server must surface through the handle, not silent empty readings."""

    def test_subscribe_connect_failure_dispatches_error(self):
        import threading
        import time

        from pacsys.backends.dpm_http import _AsyncDPMConnection
        from pacsys.dpm_connection import DPMConnectionError

        errors = []
        err_event = threading.Event()

        def on_err(exc, handle):
            errors.append(exc)
            err_event.set()

        backend = DPMHTTPBackend()
        try:
            with mock.patch.object(
                _AsyncDPMConnection, "connect", side_effect=DPMConnectionError("connection refused")
            ):
                handle = backend.subscribe(["M:OUTTMP@p,1000"], callback=lambda r, h: None, on_error=on_err)
                assert err_event.wait(2.0), "on_error never fired for failed connect"
                assert isinstance(errors[0], DPMConnectionError)
                assert handle.exc is errors[0]
                assert handle.stopped
                deadline = time.monotonic() + 2.0
                while handle in backend._handles and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert handle not in backend._handles
        finally:
            backend.close()
