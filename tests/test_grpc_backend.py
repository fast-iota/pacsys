"""
Unit tests for GRPCBackend.

Tests cover:
- Backend initialization and capabilities
- Single device read/get
- Multiple device get_many
- Write operations (requires token)
- JWT token parsing for principal
- Environment variable token (PACSYS_JWT_TOKEN)
- Error handling
- Bounded queue overflow
- Reactor lifecycle
- _DaqCore.stream reconnection and backoff
- Uses stub mocking for unit tests (requires real proto files)
"""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import CancelledError
from unittest import mock

import numpy as np
import pytest

from pacsys import DigitalStatus
from pacsys.acnet.errors import ERR_RETRY, ERR_TIMEOUT, FACILITY_ACNET
from pacsys.auth import JWTAuth
from pacsys.errors import AuthenticationError, DeviceError, ReadError
from pacsys.types import Reading, ValueType, WriteResult
from tests.devices import make_jwt_token

# sample_jwt fixture is provided by conftest.py

# Check if grpc and proto files are available
try:
    from pacsys._proto.controls.common.v1 import device_pb2, status_pb2
    from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2
    from pacsys.backends import grpc_backend

    GRPC_AVAILABLE = grpc_backend.GRPC_AVAILABLE
except ImportError:
    GRPC_AVAILABLE = False
    grpc_backend = None
    DAQ_pb2 = None
    device_pb2 = None
    status_pb2 = None

if not GRPC_AVAILABLE:
    pytest.skip("grpc and proto files not available", allow_module_level=True)

import grpc

# ─────────────────────────────────────────────────────────────────────────────
# Async mock helpers
# ─────────────────────────────────────────────────────────────────────────────


class AsyncMockIterator:
    """Wraps a list of proto replies as an async iterator with cancel()/done()."""

    def __init__(self, replies):
        self._replies = list(replies)
        self._index = 0
        self._cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._cancelled or self._index >= len(self._replies):
            raise StopAsyncIteration
        reply = self._replies[self._index]
        self._index += 1
        return reply

    def cancel(self):
        self._cancelled = True

    def done(self):
        return self._cancelled or self._index >= len(self._replies)


class AsyncMockRpcError(grpc.aio.AioRpcError):
    """Lightweight AioRpcError mock that can be raised in async code."""

    def __init__(self, code, details=""):
        self._code_val = code
        self._details_val = details

    def code(self):
        return self._code_val

    def details(self):
        return self._details_val

    def __str__(self):
        return f"AioRpcError({self._code_val.name}: {self._details_val})"


class AsyncErrorIterator:
    """Async iterator that immediately raises an AioRpcError."""

    def __init__(self, code, details=""):
        self._error = AsyncMockRpcError(code, details)

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._error

    def cancel(self):
        pass


class AsyncReplyThenError:
    """Yields replies, then raises error. Tracks cancel()."""

    def __init__(self, replies, error=None):
        self._replies = list(replies)
        self._idx = 0
        self._error = error
        self.cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.cancelled:
            raise StopAsyncIteration
        if self._idx < len(self._replies):
            r = self._replies[self._idx]
            self._idx += 1
            return r
        if self._error:
            raise self._error
        raise StopAsyncIteration

    def cancel(self):
        self.cancelled = True


# ─────────────────────────────────────────────────────────────────────────────
# Reply factories
# ─────────────────────────────────────────────────────────────────────────────


def make_reading_reply(
    index: int,
    scalar_value: float | None = None,
    text_value: str | None = None,
    error_code: int | None = None,
    error_message: str | None = None,
) -> DAQ_pb2.ReadingReply:
    """Create a ReadingReply protobuf message for testing."""
    reply = DAQ_pb2.ReadingReply()
    reply.index = index

    if error_code is not None:
        reply.status.status_code = error_code
        reply.status.message = error_message or "Error"
    else:
        reading = DAQ_pb2.Reading()
        reading.timestamp.seconds = 1234567890
        reading.timestamp.nanos = 0

        if scalar_value is not None:
            reading.data.scalar = scalar_value
        elif text_value is not None:
            reading.data.text = text_value

        reply.readings.reading.append(reading)

    return reply


def make_setting_reply(status_codes: list[int]) -> DAQ_pb2.SettingReply:
    """Create a SettingReply protobuf message for testing."""
    reply = DAQ_pb2.SettingReply()
    for code in status_codes:
        status = status_pb2.Status()
        status.status_code = code
        status.message = "OK" if code == 0 else "Error"
        reply.status.append(status)
    return reply


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_stub():
    """Create a mock gRPC stub for testing."""
    return mock.MagicMock()


def _make_backend_with_stub(stub, auth=None):
    """Create a GRPCBackend with mocked async core stub."""
    backend = grpc_backend.GRPCBackend(auth=auth)
    backend._start_reactor()
    core = grpc_backend._DaqCore(backend._host, backend._port, backend._auth, backend._timeout)
    core._stub = stub
    backend._core = core
    return backend


def _close_backend_fast_for_tests(backend):
    """Fast shutdown for mocked backends to avoid per-test join timeout tax."""
    if backend._closed:
        return
    backend._closed = True
    backend.stop_streaming()
    backend._dispatcher.close()
    backend._core = None

    loop = backend._loop
    thread = backend._reactor_thread
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=0.1)
    backend._loop = None
    backend._reactor_thread = None


@pytest.fixture
def backend_with_mock_stub(mock_stub):
    """GRPCBackend (no auth) with mocked stub -- for read tests."""
    backend = _make_backend_with_stub(mock_stub)
    yield backend, mock_stub
    _close_backend_fast_for_tests(backend)


@pytest.fixture
def auth_backend_with_mock_stub(mock_stub, sample_jwt):
    """GRPCBackend (with JWT auth) and mocked stub -- for write tests."""
    auth = JWTAuth(token=sample_jwt)
    backend = _make_backend_with_stub(mock_stub, auth=auth)
    yield backend, mock_stub
    _close_backend_fast_for_tests(backend)


# ─────────────────────────────────────────────────────────────────────────────
# Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGRPCBackendInit:
    """Tests for GRPCBackend initialization."""

    def test_token_from_environment(self, sample_jwt):
        with mock.patch.dict(os.environ, {"PACSYS_JWT_TOKEN": sample_jwt}):
            backend = grpc_backend.GRPCBackend()
            try:
                assert backend.authenticated
                assert backend.principal == "testuser@fnal.gov"
            finally:
                backend.close()

    def test_explicit_auth_overrides_environment(self, sample_jwt):
        env_token = make_jwt_token({"sub": "env_user@fnal.gov"})
        explicit_token = make_jwt_token({"sub": "explicit_user@fnal.gov"})

        with mock.patch.dict(os.environ, {"PACSYS_JWT_TOKEN": env_token}):
            auth = JWTAuth(token=explicit_token)
            backend = grpc_backend.GRPCBackend(auth=auth)
            try:
                assert backend.principal == "explicit_user@fnal.gov"
            finally:
                backend.close()


# ─────────────────────────────────────────────────────────────────────────────
# Capabilities Tests
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# JWT Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestJWTDecoding:
    """Tests for JWT token decoding via JWTAuth."""

    def test_decode_valid_jwt(self):
        token = make_jwt_token({"sub": "user@example.com", "name": "Test User"})
        auth = JWTAuth(token=token)
        payload = auth._decode_payload()
        assert payload["sub"] == "user@example.com"
        assert payload["name"] == "Test User"

    def test_decode_invalid_jwt_format(self):
        auth = JWTAuth(token="not.a.valid.jwt.token")
        with pytest.raises(ValueError, match="Invalid JWT format"):
            auth._decode_payload()

    def test_decode_jwt_missing_parts(self):
        auth = JWTAuth(token="only.two")
        with pytest.raises(ValueError, match="Invalid JWT format"):
            auth._decode_payload()

    def test_extract_principal_valid(self):
        token = make_jwt_token({"sub": "user@example.com"})
        auth = JWTAuth(token=token)
        assert auth.principal == "user@example.com"

    def test_extract_principal_no_sub(self):
        token = make_jwt_token({"name": "No Subject"})
        auth = JWTAuth(token=token)
        with pytest.raises(ValueError, match="no 'sub' claim"):
            _ = auth.principal

    def test_extract_principal_invalid_token(self):
        auth = JWTAuth(token="invalid")
        with pytest.raises(ValueError, match="Invalid JWT format"):
            _ = auth.principal


# ─────────────────────────────────────────────────────────────────────────────
# Single Device Read Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSingleDeviceRead:
    """Tests for single device read/get operations."""

    def test_read_scalar_success(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator([make_reading_reply(0, scalar_value=72.5)])

        value = backend.read("M:OUTTMP")
        assert value == 72.5

    def test_read_text_success(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator([make_reading_reply(0, text_value="Outdoor Temperature")])

        value = backend.read("M:OUTTMP.DESCRIPTION")
        assert value == "Outdoor Temperature"

    def test_get_returns_reading(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator([make_reading_reply(0, scalar_value=72.5)])

        reading = backend.get("M:OUTTMP")
        assert isinstance(reading, Reading)
        assert reading.value == 72.5
        assert reading.value_type == ValueType.SCALAR
        assert reading.is_success
        assert reading.ok

    def test_get_preserves_status_display_text(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        original = {"On": "False", "Ready": "Yes"}
        reply = DAQ_pb2.ReadingReply(index=0)
        reply.readings.reading.add().data.basicStatus.value.update(original)
        mock_stub.Read.return_value = AsyncMockIterator([reply])

        reading = backend.get("Z:ACLTST.STATUS")

        assert reading.ok
        assert reading.value == original
        status = DigitalStatus.from_reading(reading)
        assert len(status) == 2
        assert status["On"].value == "False"
        assert status["On"].is_set is False
        assert status["Ready"].value == "Yes"
        assert status["Ready"].is_set is True
        assert status.on is False
        assert status.ready is True

    def test_read_error_raises_device_error(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [make_reading_reply(0, error_code=-42, error_message="Device not found")]
        )

        with pytest.raises(DeviceError) as exc_info:
            backend.read("M:BADDEV")
        assert "Device not found" in exc_info.value.message

    @pytest.mark.parametrize("code", [-42, 0])
    def test_get_error_returns_reading_with_error(self, backend_with_mock_stub, code):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [make_reading_reply(0, error_code=code, error_message="Device not found")]
        )

        reading = backend.get("M:BADDEV")
        assert (reading.facility_code, reading.error_code) == ((FACILITY_ACNET, ERR_RETRY) if code == 0 else (0, code))
        assert reading.is_error
        assert not reading.ok
        assert "Device not found" in reading.message

    def test_get_positive_status_returns_warning_without_value(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [make_reading_reply(0, error_code=1, error_message="Request pending")]
        )

        reading = backend.get("M:OUTTMP")

        assert reading.is_warning
        assert reading.value is None
        assert not reading.ok
        assert reading.message == "Request pending"

    def test_read_positive_status_raises_device_error(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [make_reading_reply(0, error_code=1, error_message="Request pending")]
        )

        with pytest.raises(DeviceError) as exc_info:
            backend.read("M:OUTTMP")

        assert exc_info.value.error_code == 1
        assert exc_info.value.message == "Request pending"


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Device Read Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWarningData:
    @staticmethod
    def _reply(*samples):
        reply = DAQ_pb2.ReadingReply(index=0)
        for i, (value, status) in enumerate(samples):
            reading = reply.readings.reading.add()
            reading.timestamp.seconds = 1234567890 + i
            if value is not None:
                reading.data.scalar = value
            reading.status.facility_code = 66
            reading.status.status_code = status
            reading.status.message = f"status {status}"
        return reply

    @pytest.mark.parametrize("streaming", [False, True])
    def test_supervised_warning_keeps_value(self, backend_with_mock_stub, streaming):
        from pacsys.supervised._conversions import reading_to_proto_replies

        backend, stub = backend_with_mock_stub
        original = Reading(
            drf="M:OUTTMP", value=12.5, value_type=ValueType.SCALAR, facility_code=66, error_code=1, message="warning"
        )
        stub.Read.return_value = AsyncMockIterator(reading_to_proto_replies(original, 0))
        if streaming:
            with backend.subscribe([original.drf]) as handle:
                readings = [r for r, _h in handle.readings(timeout=0.2)]
        else:
            readings = [backend.get(original.drf)]
        assert len(readings) == 1
        reading = readings[0]
        assert reading.ok and reading.value == original.value and reading.value_type == original.value_type
        assert (reading.facility_code, reading.error_code, reading.message) == (66, 1, "warning")

    @pytest.mark.parametrize("logger", [False, True])
    def test_aggregate_keeps_all_values_and_first_warning(self, backend_with_mock_stub, logger):
        backend, stub = backend_with_mock_stub
        if logger:
            replies = [self._reply((1.0, 0), (2.0, 1)), self._reply((3.0, 2)), self._reply()]
        else:
            replies = [self._reply((1.0, 0), (2.0, 1), (3.0, 2))]
        stub.Read.return_value = AsyncMockIterator(replies)
        reading = backend.get("M:OUTTMP<-LOGGERDURATION:60000" if logger else "M:OUTTMP")
        assert reading.ok
        assert reading.value_type == ValueType.TIMED_SCALAR_ARRAY
        assert reading.value["data"].tolist() == [1.0, 2.0, 3.0]
        assert (reading.facility_code, reading.error_code, reading.message) == (66, 1, "status 1")

    @pytest.mark.parametrize("layout", ["batch", "logger"])
    @pytest.mark.parametrize(("value", "status"), [(9.0, -42), (None, 2)])
    def test_unusable_sample_keeps_status(self, backend_with_mock_stub, layout, value, status):
        backend, stub = backend_with_mock_stub
        if layout == "batch":
            replies = [self._reply((1.0, 1), (value, status))]
        else:
            replies = [self._reply((1.0, 1)), self._reply((value, status)), self._reply()]
        stub.Read.return_value = AsyncMockIterator(replies)
        reading = backend.get("M:OUTTMP<-LOGGERDURATION:60000" if layout == "logger" else "M:OUTTMP")
        assert not reading.ok and reading.value is None
        assert (reading.facility_code, reading.error_code, reading.message) == (66, status, f"status {status}")


def test_subscribe_survives_caller_mutating_drfs(backend_with_mock_stub):
    backend, stub = backend_with_mock_stub
    gate = threading.Event()
    requests = []

    class GatedIterator(AsyncMockIterator):
        async def __anext__(self):
            while not gate.is_set():
                await asyncio.sleep(0.005)
            return await super().__anext__()

    def read(request, **kwargs):
        requests.append(list(request.drf))
        return GatedIterator([make_reading_reply(0, scalar_value=1.5)])

    stub.Read.side_effect = read
    drfs = ["M:OUTTMP@p,1000"]
    with backend.subscribe(drfs) as handle:
        drfs.clear()
        gate.set()
        readings = [r for r, _h in handle.readings(timeout=0.5)]
    assert requests == [["M:OUTTMP@p,1000"]]
    assert [(r.drf, r.value) for r in readings] == [("M:OUTTMP@p,1000", 1.5)]


class TestMultipleDeviceRead:
    """Tests for multiple device get_many operations."""

    def test_get_many_normalizes_default_events_on_wire(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        drfs = [
            "M:OUTTMP",
            "M_OUTTMP@U",
            "M:OUTTMP@p,1000",
            "M:OUTTMP@E,0F",
            "pv:name.VAL",
            "M:OUTTMP<-LOGGER:1700000000000:1700000060000",
            "M:OUTTMP<-LOGGERDURATION:60000",
            "M:OUTTMP<-LOGGERSINGLE:ArkIv:1736942400:60",
        ]
        replies = []
        for index in range(len(drfs)):
            replies.append(make_reading_reply(index, scalar_value=float(index)))
            if index in (5, 6):
                terminator = DAQ_pb2.ReadingReply(index=index)
                terminator.readings.SetInParent()
                replies.append(terminator)
        mock_stub.Read.return_value = AsyncMockIterator(replies)

        readings = backend.get_many(drfs)

        mock_stub.Read.assert_called_once()
        assert list(mock_stub.Read.call_args.args[0].drf) == [
            "M:OUTTMP.READING@I",
            "M:OUTTMP.SETTING@I",
            *drfs[2:4],
            "pv:name.VAL@I",
            *drfs[5:],
        ]
        assert [reading.drf for reading in readings] == drfs
        assert all(reading.ok for reading in readings)
        assert readings[5].value["data"].tolist() == [5.0]
        assert readings[6].value["data"].tolist() == [6.0]
        assert readings[7].value == 7.0

    def test_get_many_malformed_batch_raises_before_rpc(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub

        with pytest.raises(ValueError):
            backend.get_many(["M:OUTTMP", "M:OUTTMP{bad}"])

        mock_stub.Read.assert_not_called()

    def test_get_many_multiple_devices(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [
                make_reading_reply(0, scalar_value=72.5),
                make_reading_reply(1, scalar_value=1.234),
            ]
        )

        readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
        assert len(readings) == 2
        assert readings[0].value == 72.5
        assert readings[1].value == 1.234

    def test_get_many_partial_failure(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [
                make_reading_reply(0, scalar_value=72.5),
                make_reading_reply(1, error_code=-1, error_message="Bad device"),
            ]
        )

        readings = backend.get_many(["M:OUTTMP", "M:BADDEV"])
        assert len(readings) == 2
        assert readings[0].ok
        assert readings[0].value == 72.5
        assert readings[1].is_error
        assert "Bad device" in readings[1].message


# ─────────────────────────────────────────────────────────────────────────────
# Write Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteOperations:
    """Tests for write operations."""

    def test_write_requires_token(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        with pytest.raises(AuthenticationError, match="JWTAuth required"):
            backend.write("M:OUTTMP", 72.5)

    def test_write_success(self, auth_backend_with_mock_stub):
        backend, mock_stub = auth_backend_with_mock_stub
        mock_stub.Set = mock.AsyncMock(return_value=make_setting_reply([0]))

        result = backend.write("M:OUTTMP", 72.5)
        assert isinstance(result, WriteResult)
        assert result.success

    def test_write_many_requires_token(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        with pytest.raises(AuthenticationError, match="JWTAuth required"):
            backend.write_many([("M:OUTTMP", 72.5)])

    def test_write_many_success(self, auth_backend_with_mock_stub):
        backend, mock_stub = auth_backend_with_mock_stub
        mock_stub.Set = mock.AsyncMock(return_value=make_setting_reply([0, 0]))

        results = backend.write_many([("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)])
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_write_many_numpy_arrays(self, auth_backend_with_mock_stub):
        backend, mock_stub = auth_backend_with_mock_stub
        mock_stub.Set = mock.AsyncMock(return_value=make_setting_reply([0, -42]))
        settings = [
            ("M:BAD", np.array([1, "text"], dtype=object)),
            ("M:TEXT", np.array(["hello", "世界"])),
            ("M:MATRIX", np.array([[1, 2]])),
            ("M:NUMBER", np.array([1.0, 2.5])),
            ("M:ZERO", np.array(3)),
        ]

        results = backend.write_many(settings)

        mock_stub.Set.assert_awaited_once()
        request = mock_stub.Set.call_args.args[0]
        assert [s.device for s in request.setting] == ["M:TEXT.SETTING@N", "M:NUMBER.SETTING@N"]
        assert request.setting[0].value.WhichOneof("value") == "textArr"
        assert list(request.setting[0].value.textArr.value) == ["hello", "世界"]
        assert request.setting[1].value.WhichOneof("value") == "scalarArr"
        assert list(request.setting[1].value.scalarArr.value) == [1.0, 2.5]
        assert [r.drf for r in results] == [f"{drf}.SETTING@N" for drf, _ in settings]
        assert results[1].success
        assert results[3].error_code == -42
        for index in (0, 2, 4):
            assert not results[index].success
            assert (results[index].facility_code, results[index].error_code) == (FACILITY_ACNET, ERR_RETRY)
            assert results[index].message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_backend", [False, True], ids=["sync", "async"])
    async def test_write_many_reserved_directives_preserve_status_order(self, mock_stub, sample_jwt, async_backend):
        import pacsys.aio

        auth = JWTAuth(token=sample_jwt)
        if async_backend:
            backend = pacsys.aio.grpc(auth=auth)
            backend._core = grpc_backend._DaqCore(backend._host, backend._port, auth, backend._timeout)
            backend._core._stub = mock_stub
            backend._connected = True
        else:
            backend = _make_backend_with_stub(mock_stub, auth=auth)

        server_statuses = {
            "M:OUTTMP.SETTING@N": 0,
            "#:123.SETTING@N": -42,
            "M:OUTTMP.LONG_NAME@N": -43,
            "EPICS:PV:SET@N": 0,
        }

        async def set_reply(request, **kwargs):
            # DPM consumes these list directives without allocating setting/status slots.
            return make_setting_reply(
                [
                    server_statuses[setting.device]
                    for setting in request.setting
                    if setting.device not in {"#AB:CD@N", "#ROLE:x@N", "#plain@N"}
                ]
            )

        mock_stub.Set = mock.AsyncMock(side_effect=set_reply)
        settings = [
            ("#AB:CD", 1.0),
            ("M:OUTTMP", 72.5),
            ("M:BAD", object()),
            ("#ROLE:x", 2.0),
            ("#:123", 3.0),
            ("M#OUTTMP", "name"),
            ("#plain", 4.0),
            ("EPICS:PV:SET", 5.0),
        ]
        try:
            results = await backend.write_many(settings) if async_backend else backend.write_many(settings)
            assert results[1].success
            assert results[4].error_code == -42
            assert results[5].error_code == -43
            assert results[7].success
            assert [r.drf for r in results] == [
                "#AB:CD@N",
                "M:OUTTMP.SETTING@N",
                "M:BAD.SETTING@N",
                "#ROLE:x@N",
                "#:123.SETTING@N",
                "M:OUTTMP.LONG_NAME@N",
                "#plain@N",
                "EPICS:PV:SET@N",
            ]
            for index in (0, 2, 3, 6):
                assert not results[index].success
                assert (results[index].facility_code, results[index].error_code) == (FACILITY_ACNET, ERR_RETRY)
            for index in (0, 3, 6):
                assert "reserved for DPM list directives" in results[index].message
            assert "Cannot convert value of type" in results[2].message
            mock_stub.Set.assert_awaited_once()
            assert [s.device for s in mock_stub.Set.call_args.args[0].setting] == list(server_statuses)
        finally:
            if async_backend:
                await backend.close()
            else:
                _close_backend_fast_for_tests(backend)

    def test_write_prepares_drf(self, auth_backend_with_mock_stub):
        """write() applies prepare_for_write to convert shorthand DRFs."""
        backend, mock_stub = auth_backend_with_mock_stub
        mock_stub.Set = mock.AsyncMock(return_value=make_setting_reply([0]))

        backend.write("M:OUTTMP", 72.5)

        call_args = mock_stub.Set.call_args
        request = call_args[0][0]
        assert request.setting[0].device == "M:OUTTMP.SETTING@N"

    def test_write_many_prepares_drfs(self, auth_backend_with_mock_stub):
        """write_many() applies prepare_for_write to all DRFs."""
        backend, mock_stub = auth_backend_with_mock_stub
        mock_stub.Set = mock.AsyncMock(return_value=make_setting_reply([0, 0]))

        backend.write_many([("M:OUTTMP", 72.5), ("M_OUTTMP.STATUS", 1)])

        call_args = mock_stub.Set.call_args
        request = call_args[0][0]
        assert request.setting[0].device == "M:OUTTMP.SETTING@N"
        assert request.setting[1].device == "M:OUTTMP.CONTROL@N"

    def test_write_many_missing_server_response(self, auth_backend_with_mock_stub):
        """Missing server statuses produce write errors."""
        backend, mock_stub = auth_backend_with_mock_stub
        mock_stub.Set = mock.AsyncMock(return_value=make_setting_reply([0]))

        results = backend.write_many([("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)])
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success
        assert (results[1].facility_code, results[1].error_code) == (FACILITY_ACNET, ERR_RETRY)
        assert "No status received" in results[1].message

    def test_write_unsupported_type_returns_error(self, auth_backend_with_mock_stub):
        backend, mock_stub = auth_backend_with_mock_stub

        result = backend.write("M:OUTTMP", object())

        assert not result.success
        assert (result.facility_code, result.error_code) == (FACILITY_ACNET, ERR_RETRY)
        assert "Cannot convert value of type" in result.message
        mock_stub.Set.assert_not_called()

    @pytest.mark.parametrize(
        "error, code",
        [
            (RuntimeError("programming bug"), ERR_RETRY),
            (AsyncMockRpcError(grpc.StatusCode.UNAVAILABLE, "programming bug"), ERR_RETRY),
            (AsyncMockRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "programming bug"), ERR_TIMEOUT),
        ],
    )
    def test_unexpected_write_error_becomes_write_results(self, auth_backend_with_mock_stub, error, code):
        backend, mock_stub = auth_backend_with_mock_stub
        mock_stub.Set = mock.AsyncMock(side_effect=error)

        results = backend.write_many([("M:OUTTMP", 72.5)])
        assert len(results) == 1
        assert not results[0].success
        assert (results[0].facility_code, results[0].error_code) == (FACILITY_ACNET, code)
        assert "programming bug" in results[0].message


# ─────────────────────────────────────────────────────────────────────────────
# gRPC Error Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGRPCErrors:
    """Tests for gRPC error handling."""

    def test_grpc_error_on_read(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "Connection refused")

        with pytest.raises(ReadError) as exc_info:
            backend.read("M:OUTTMP")
        assert (exc_info.value.readings[0].facility_code, exc_info.value.readings[0].error_code) == (
            FACILITY_ACNET,
            ERR_RETRY,
        )
        assert "UNAVAILABLE" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, grpc.aio.AioRpcError)

    def test_grpc_error_on_get_many(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncErrorIterator(grpc.StatusCode.DEADLINE_EXCEEDED, "Timeout")

        with pytest.raises(ReadError) as exc_info:
            backend.get_many(["M:OUTTMP", "G:AMANDA"])
        readings = exc_info.value.readings
        assert len(readings) == 2
        assert all(r.is_error for r in readings)
        assert all((r.facility_code, r.error_code) == (FACILITY_ACNET, ERR_TIMEOUT) for r in readings)
        assert all("DEADLINE_EXCEEDED" in r.message for r in readings)
        assert isinstance(exc_info.value.__cause__, grpc.aio.AioRpcError)

    def test_unexpected_read_error_becomes_read_error(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncReplyThenError([], RuntimeError("programming bug"))

        with pytest.raises(ReadError) as exc_info:
            backend.get("M:OUTTMP")
        readings = exc_info.value.readings
        assert len(readings) == 1
        assert readings[0].is_error
        assert (readings[0].facility_code, readings[0].error_code) == (FACILITY_ACNET, ERR_RETRY)
        assert "programming bug" in readings[0].message
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_multisample_text_reading_degrades_per_device(self, backend_with_mock_stub):
        # Text device packed 2+ samples per message: aggregation cannot coerce to
        # float; that device gets ERR_RETRY while the rest of the batch succeeds.
        backend, mock_stub = backend_with_mock_stub
        reply0 = DAQ_pb2.ReadingReply()
        reply0.index = 0
        for txt in ("a", "b"):
            rd = DAQ_pb2.Reading()
            rd.timestamp.seconds = 1234567890
            rd.data.text = txt
            reply0.readings.reading.append(rd)
        reply1 = make_reading_reply(1, scalar_value=72.5)
        mock_stub.Read.return_value = AsyncReplyThenError([reply0, reply1])

        readings = backend.get_many(["Z:ACLTST", "M:OUTTMP"])
        assert readings[0].is_error
        assert readings[1].value == 72.5


# ─────────────────────────────────────────────────────────────────────────────
# Context Manager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager_closes(self):
        with grpc_backend.GRPCBackend() as backend:
            assert not backend._closed
        assert backend._closed

    def test_context_manager_on_exception(self):
        try:
            with grpc_backend.GRPCBackend() as backend:
                raise ValueError("test error")
        except ValueError:
            pass
        assert backend._closed

    def test_close_multiple_times_safe(self):
        backend = grpc_backend.GRPCBackend()
        backend.close()
        backend.close()
        backend.close()
        assert backend._closed


# ─────────────────────────────────────────────────────────────────────────────
# Operations After Close
# ─────────────────────────────────────────────────────────────────────────────


class TestOperationAfterClose:
    """Tests for operations after close."""

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("read", ("M:OUTTMP",)),
            ("get", ("M:OUTTMP",)),
            ("get_many", (["M:OUTTMP"],)),
            ("write_many", ([("M:OUTTMP", 72.5)],)),
        ],
    )
    def test_operation_after_close_raises(self, sample_jwt, method, args):
        backend = grpc_backend.GRPCBackend(auth=JWTAuth(token=sample_jwt))
        backend.close()
        with pytest.raises(RuntimeError, match="Backend is closed"):
            getattr(backend, method)(*args)


# ─────────────────────────────────────────────────────────────────────────────
# Value Conversion Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestValueConversion:
    """Tests for value type conversion."""

    @pytest.mark.parametrize(
        ("input_val", "field", "expected"),
        [
            (72.5, "scalar", 72.5),
            (42, "scalar", 42.0),
            ("hello", "text", "hello"),
            (b"\x00\x01\x02", "raw", b"\x00\x01\x02"),
        ],
    )
    def test_value_to_proto_simple(self, input_val, field, expected):
        proto = grpc_backend._value_to_proto_value(input_val)
        assert getattr(proto, field) == expected

    def test_list_to_scalar_array_conversion(self):
        proto = grpc_backend._value_to_proto_value([1.0, 2.0, 3.0])
        assert list(proto.scalarArr.value) == [1.0, 2.0, 3.0]

    def test_text_array_conversion(self):
        proto = grpc_backend._value_to_proto_value(["a", "b", "c"])
        assert list(proto.textArr.value) == ["a", "b", "c"]

    def test_numpy_scalars_match_python_peers(self):
        """NumPy scalars (alone or in lists) are accepted like DPM does."""
        assert grpc_backend._value_to_proto_value(np.int64(3)).scalar == 3.0
        assert grpc_backend._value_to_proto_value(np.bool_(True)).scalar == 1.0
        assert grpc_backend._value_to_proto_value(np.longdouble(1.5)).scalar == 1.5
        proto = grpc_backend._value_to_proto_value([np.int64(1), np.float64(2.5)])
        assert list(proto.scalarArr.value) == [1.0, 2.5]
        with pytest.raises(TypeError):  # never silently encoded as a number
            grpc_backend._value_to_proto_value(np.datetime64("2020-01-01T00:00:00.000000001"))

    @pytest.mark.parametrize(
        "value",
        [
            np.array([1.5], dtype=np.longdouble),
            np.array([np.longdouble(1.5)], dtype=object),
            np.array([np.str_("hello")], dtype=object),
            np.array([], dtype=str),
        ],
    )
    def test_numpy_arrays_match_lists(self, value):
        assert grpc_backend._value_to_proto_value(value) == grpc_backend._value_to_proto_value(value.tolist())

    @pytest.mark.parametrize(
        ("field", "set_val", "expected_val", "expected_type"),
        [
            ("scalar", 72.5, 72.5, ValueType.SCALAR),
            ("text", "hello", "hello", ValueType.TEXT),
            ("raw", b"\x00\x01", b"\x00\x01", ValueType.RAW),
        ],
    )
    def test_proto_to_python(self, field, set_val, expected_val, expected_type):
        proto = device_pb2.Value()
        setattr(proto, field, set_val)
        value, vtype = grpc_backend._proto_value_to_python(proto)
        assert value == expected_val
        assert vtype == expected_type

    def test_analog_alarm_returns_snake_case_keys(self):
        proto = device_pb2.Value()
        alarm = proto.anaAlarm
        alarm.minimum = 1.0
        alarm.maximum = 100.0
        alarm.alarmEnable = True
        alarm.alarmStatus = False
        alarm.abort = True
        alarm.abortInhibit = False
        alarm.triesNeeded = 3
        alarm.triesNow = 1
        value, vtype = grpc_backend._proto_value_to_python(proto)
        assert vtype == ValueType.ANALOG_ALARM
        assert value == {
            "minimum": 1.0,
            "maximum": 100.0,
            "alarm_enable": True,
            "alarm_status": False,
            "abort": True,
            "abort_inhibit": False,
            "tries_needed": 3,
            "tries_now": 1,
        }

    def test_digital_alarm_returns_snake_case_keys(self):
        proto = device_pb2.Value()
        alarm = proto.digAlarm
        alarm.nominal = 5
        alarm.mask = 0xFF
        alarm.alarmEnable = False
        alarm.alarmStatus = True
        alarm.abort = False
        alarm.abortInhibit = True
        alarm.triesNeeded = 2
        alarm.triesNow = 0
        value, vtype = grpc_backend._proto_value_to_python(proto)
        assert vtype == ValueType.DIGITAL_ALARM
        assert value == {
            "nominal": 5,
            "mask": 0xFF,
            "alarm_enable": False,
            "alarm_status": True,
            "abort": False,
            "abort_inhibit": True,
            "tries_needed": 2,
            "tries_now": 0,
        }

    def test_analog_alarm_dict_to_proto(self):
        d = {"minimum": 1.5, "maximum": 99.0, "alarm_enable": True, "abort_inhibit": False, "tries_needed": 3}
        proto = grpc_backend._value_to_proto_value(d)
        assert proto.WhichOneof("value") == "anaAlarm"
        assert proto.anaAlarm.minimum == 1.5
        assert proto.anaAlarm.maximum == 99.0
        assert proto.anaAlarm.alarmEnable is True
        assert proto.anaAlarm.abortInhibit is False
        assert proto.anaAlarm.triesNeeded == 3

    def test_analog_alarm_dict_partial_keys(self):
        proto = grpc_backend._value_to_proto_value({"minimum": 10.0})
        assert proto.WhichOneof("value") == "anaAlarm"
        assert proto.anaAlarm.minimum == 10.0
        assert proto.anaAlarm.maximum == 0.0  # proto default

    def test_digital_alarm_dict_to_proto(self):
        d = {"nominal": 0xFF, "mask": 0x0F, "alarm_enable": False, "tries_needed": 2}
        proto = grpc_backend._value_to_proto_value(d)
        assert proto.WhichOneof("value") == "digAlarm"
        assert proto.digAlarm.nominal == 0xFF
        assert proto.digAlarm.mask == 0x0F
        assert proto.digAlarm.alarmEnable is False
        assert proto.digAlarm.triesNeeded == 2

    def test_alarm_dict_readonly_keys_round_trip(self):
        """A full backend alarm reading (with status keys) survives proxy forwarding."""
        original = {
            "nominal": 9,
            "mask": 61,
            "alarm_enable": False,
            "alarm_status": True,
            "abort": True,
            "abort_inhibit": False,
            "tries_needed": 1,
            "tries_now": 3,
        }
        result, vtype = grpc_backend._proto_value_to_python(grpc_backend._value_to_proto_value(original))
        assert vtype == ValueType.DIGITAL_ALARM
        assert result == original

    def test_alarm_dict_unknown_keys_raises(self):
        with pytest.raises(ValueError, match="Unknown alarm dict keys"):
            grpc_backend._value_to_proto_value({"minimum": 1.0, "bogus": 42})

    def test_alarm_dict_mixed_keys_raises(self):
        with pytest.raises(ValueError, match=r"Cannot mix analog.*and digital"):
            grpc_backend._value_to_proto_value({"minimum": 1.0, "nominal": 5})

    def test_alarm_dict_shared_only_raises(self):
        with pytest.raises(ValueError, match="type-specific key"):
            grpc_backend._value_to_proto_value({"alarm_enable": True})

    def test_alarm_dict_empty_raises(self):
        with pytest.raises(ValueError, match="type-specific key"):
            grpc_backend._value_to_proto_value({})

    def test_alarm_dict_round_trip(self):
        """Dict → proto → dict preserves writable fields."""
        original = {"minimum": -5.0, "maximum": 105.0, "alarm_enable": True, "abort_inhibit": True, "tries_needed": 4}
        proto = grpc_backend._value_to_proto_value(original)
        result, vtype = grpc_backend._proto_value_to_python(proto)
        assert vtype == ValueType.ANALOG_ALARM
        for key, expected in original.items():
            assert result[key] == expected

    def test_basic_status_round_trip_returns_bools(self):
        """BasicStatus dict → proto → dict preserves bool types, not strings."""
        original = {"on": True, "ready": False, "remote": True, "positive": False, "ramp": True}
        proto = grpc_backend._value_to_proto_value(original)
        result, vtype = grpc_backend._proto_value_to_python(proto)
        assert vtype == ValueType.BASIC_STATUS
        assert result == original
        assert all(isinstance(v, bool) for v in result.values())

    def test_basic_status_preserves_server_bit_text(self):
        """Real DPM/gRPC sends per-bit display text, which must not collapse to False."""
        proto = device_pb2.Value()
        proto.basicStatus.value.update({"On": "Yes", "Shutter Status": "Closed", "Heartbeat": " [-]"})
        result, vtype = grpc_backend._proto_value_to_python(proto)
        assert vtype == ValueType.BASIC_STATUS
        assert result == {"On": "Yes", "Shutter Status": "Closed", "Heartbeat": " [-]"}


# ─────────────────────────────────────────────────────────────────────────────
# Status Code Normalization
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusCodeNormalization:
    """Tests for uint8 -> int8 status code normalization."""

    @pytest.mark.parametrize(
        ("input_code", "expected"),
        [
            (0, 0),
            (1, 1),
            (42, 42),
            (127, 127),
            (227, -29),
            (255, -1),
            (128, -128),
            (200, -56),
            (-1, -1),
            (-29, -29),
        ],
    )
    def test_normalize_error_code(self, input_code, expected):
        from pacsys.acnet.errors import normalize_error_code

        assert normalize_error_code(input_code) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Backend Inheritance
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Bounded Queue Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBoundedQueue:
    """Tests for bounded buffer in subscription handle."""

    def test_queue_overflow_drops_and_warns(self, caplog):
        """When the buffer is full, new readings are dropped with a warning."""
        backend = grpc_backend.GRPCBackend()
        try:
            handle = grpc_backend._GRPCSubscriptionHandle(
                backend=backend,
                drfs=["M:OUTTMP@p,1000"],
                callback=None,
                on_error=None,
            )
            handle._maxsize = 2

            r1 = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=1.0)
            r2 = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=2.0)
            r3 = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=3.0)

            handle._dispatch(r1)
            handle._dispatch(r2)

            with caplog.at_level(logging.WARNING, logger="pacsys.backends._subscription"):
                handle._dispatch(r3)

            assert len(handle._buf) == 2
            # Oldest readings survive (FIFO)
            assert handle._buf[0].value == 1.0
            assert handle._buf[1].value == 2.0
            assert any("buffer full" in rec.message.lower() for rec in caplog.records)
        finally:
            backend.close()


# ─────────────────────────────────────────────────────────────────────────────
# Reactor Lifecycle Tests
# ─────────────────────────────────────────────────────────────────────────────


async def _call_get(backend):
    return backend.get("M:OUTTMP", timeout=0.01)


class TestReactorLifecycle:
    """Tests for lazy reactor startup and clean shutdown."""

    def test_no_reactor_on_init(self):
        """Reactor thread is NOT started on construction."""
        backend = grpc_backend.GRPCBackend()
        try:
            assert backend._reactor_thread is None
            assert backend._loop is None
            assert backend._core is None
        finally:
            backend.close()

    def test_reactor_starts_on_first_operation(self, mock_stub):
        """Reactor thread starts when first I/O is needed."""
        backend = grpc_backend.GRPCBackend()
        # Manually start reactor (would normally happen via _ensure_reactor)
        backend._start_reactor()
        try:
            assert backend._reactor_thread is not None
            assert backend._reactor_thread.is_alive()
            assert backend._loop is not None
        finally:
            backend.close()

    def test_reactor_cleans_up_on_close(self):
        """Reactor thread and loop are cleaned up on close."""
        backend = grpc_backend.GRPCBackend()
        backend._start_reactor()
        assert backend._reactor_thread.is_alive()

        backend.close()
        assert backend._closed
        assert backend._reactor_thread is None
        assert backend._loop is None

    def test_blocking_call_from_reactor_thread_raises(self):
        """A sync facade on the reactor thread would deadlock until timeout - fail immediately instead."""
        backend = grpc_backend.GRPCBackend()
        backend._start_reactor()
        try:
            fut = asyncio.run_coroutine_threadsafe(_call_get(backend), backend._loop)
            with pytest.raises(RuntimeError, match="reactor thread"):
                fut.result(timeout=2.0)
        finally:
            backend.close()

    def test_close_from_direct_callback_awaits_core_without_blocking(self):
        backend = grpc_backend.GRPCBackend(dispatch_mode=grpc_backend.DispatchMode.DIRECT)
        backend._start_reactor()
        callback_returned = threading.Event()
        core_closed = threading.Event()
        elapsed = []

        class Core:
            async def close(self):
                core_closed.set()

        backend._core = Core()
        loop = backend._loop
        thread = backend._reactor_thread
        assert loop is not None and thread is not None

        def callback(_reading, _handle):
            start = time.monotonic()
            backend.close()
            elapsed.append(time.monotonic() - start)
            callback_returned.set()

        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=1.0)
        loop.call_soon_threadsafe(backend._dispatcher.dispatch_reading, callback, reading, mock.MagicMock())

        assert callback_returned.wait(0.5)
        assert elapsed[0] < 0.5
        assert core_closed.wait(1.0)
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert backend._core is None
        assert backend._loop is None
        assert backend._reactor_thread is None

    def test_properties_dont_start_reactor(self):
        """Accessing properties does not start the reactor thread."""
        backend = grpc_backend.GRPCBackend()
        try:
            _ = backend.host
            _ = backend.port
            _ = backend.timeout
            _ = backend.capabilities
            _ = backend.authenticated
            _ = backend.principal
            assert backend._reactor_thread is None
        finally:
            backend.close()


# ─────────────────────────────────────────────────────────────────────────────
# _DaqCore.stream Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDaqCoreStream:
    """Tests for _DaqCore.stream reconnection and backoff logic."""

    @staticmethod
    def _core(stub):
        core = grpc_backend._DaqCore("localhost", 23456, None, 5.0)
        core._stub = stub
        return core

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    # -- Normal completion: no reconnect -----------------------------------

    def test_normal_completion_no_reconnect(self):
        """Stream that ends normally exits without retry."""
        stub = mock.MagicMock()
        stub.Read.return_value = AsyncMockIterator([make_reading_reply(0, scalar_value=42.0)])

        dispatched, errors = [], []
        self._run(
            self._core(stub).stream(
                drfs=["M:OUTTMP@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e, fatal: errors.append(e),
            )
        )

        assert stub.Read.call_count == 1
        assert len(dispatched) == 1
        assert dispatched[0].value == 42.0
        assert not errors

    # -- CancelledError: clean exit ----------------------------------------

    def test_cancelled_error_no_retry_no_callback(self):
        """CancelledError exits without error callback or retry."""
        stub = mock.MagicMock()

        class _Cancelled:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError

            def cancel(self):
                pass

        stub.Read.return_value = _Cancelled()
        errors = []

        self._run(
            self._core(stub).stream(
                drfs=["M:OUTTMP@p,1000"],
                dispatch_fn=lambda r: None,
                stop_check=lambda: False,
                error_fn=lambda e, fatal: errors.append((e, fatal)),
            )
        )

        assert stub.Read.call_count == 1
        assert not errors

    # -- Backoff exponential growth + ceiling ------------------------------

    def test_backoff_sequence_and_ceiling(self):
        """Backoff doubles per retry, capped at 30s."""
        stub = mock.MagicMock()
        n = [0]

        def make_call(*a, **kw):
            n[0] += 1
            if n[0] <= 7:
                return AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "down")
            return AsyncMockIterator([])  # normal end

        stub.Read.side_effect = make_call
        sleeps = []

        async def fake_sleep(t):
            sleeps.append(t)

        with mock.patch("asyncio.sleep", side_effect=fake_sleep):
            self._run(
                self._core(stub).stream(
                    drfs=["M:OUTTMP@p,1000"],
                    dispatch_fn=lambda r: None,
                    stop_check=lambda: False,
                    error_fn=lambda e, fatal: None,
                )
            )

        assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]

    # -- Backoff resets after sustained healthy streaming -------------------

    def test_backoff_resets_after_sustained_streaming(self):
        """After 30s of healthy data, backoff resets to initial on next error."""
        stub = mock.MagicMock()
        reply = make_reading_reply(0, scalar_value=1.0)
        n = [0]

        def make_call(*a, **kw):
            n[0] += 1
            if n[0] == 1:
                return AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "down")
            if n[0] == 2:
                return AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "down")
            if n[0] == 3:
                # Healthy stream for "31s" (mocked) then error
                return AsyncReplyThenError(
                    [reply] * 3,
                    AsyncMockRpcError(grpc.StatusCode.UNAVAILABLE, "down"),
                )
            if n[0] == 4:
                return AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "down")
            return AsyncMockIterator([])

        stub.Read.side_effect = make_call
        sleeps = []

        async def fake_sleep(t):
            sleeps.append(t)

        # Proxy time module - only intercept monotonic(), leave asyncio alone
        mono_values = iter(
            [
                0,  # attempt 1: stream_start (error before any reply check)
                10,  # attempt 2: stream_start (error before any reply check)
                100,  # attempt 3: stream_start
                110,  # attempt 3, reply 1: 110-100=10 < 30
                120,  # attempt 3, reply 2: 120-100=20 < 30
                131,  # attempt 3, reply 3: 131-100=31 >= 30 → RESET
                200,  # attempt 4: stream_start (error before any reply check)
                300,  # attempt 5: stream_start
            ]
        )

        class _TimeProxy:
            """Intercept monotonic() without breaking asyncio's time usage."""

            def monotonic(self):
                return next(mono_values)

            def __getattr__(self, name):
                return getattr(time, name)

        with (
            mock.patch("asyncio.sleep", side_effect=fake_sleep),
            mock.patch.object(grpc_backend, "time", _TimeProxy()),
        ):
            self._run(
                self._core(stub).stream(
                    drfs=["M:OUTTMP@p,1000"],
                    dispatch_fn=lambda r: None,
                    stop_check=lambda: False,
                    error_fn=lambda e, fatal: None,
                )
            )

        # sleeps: 1.0 (err1), 2.0 (err2), 1.0 (reset! err3), 2.0 (err4)
        assert sleeps == [1.0, 2.0, 1.0, 2.0]

    # -- stop_check during iteration cancels call --------------------------

    def test_stop_during_iteration_cancels_call(self):
        """stop_check=True mid-stream cancels the gRPC call."""
        stub = mock.MagicMock()
        replies = [make_reading_reply(0, scalar_value=float(i)) for i in range(5)]
        call = AsyncReplyThenError(replies)
        stub.Read.return_value = call

        count = [0]

        def stop_after_2():
            return count[0] >= 2

        def dispatch(r):
            count[0] += 1

        self._run(
            self._core(stub).stream(
                drfs=["M:OUTTMP@p,1000"],
                dispatch_fn=dispatch,
                stop_check=stop_after_2,
                error_fn=lambda e, fatal: None,
            )
        )

        assert count[0] == 2
        assert call.cancelled

    # -- Retryable vs fatal errors ------------------------------------------

    def test_retryable_grpc_then_generic_error_is_fatal(self):
        """Retryable RPC errors reconnect, but unexpected errors terminate."""
        stub = mock.MagicMock()
        n = [0]

        def make_call(*a, **kw):
            n[0] += 1
            if n[0] == 1:
                return AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "srv down")
            if n[0] == 2:
                return AsyncReplyThenError([], RuntimeError("boom"))
            raise AssertionError("stream retried after fatal error")

        stub.Read.side_effect = make_call
        errors = []

        async def fake_sleep(t):
            pass

        with mock.patch("asyncio.sleep", side_effect=fake_sleep):
            self._run(
                self._core(stub).stream(
                    drfs=["M:OUTTMP@p,1000"],
                    dispatch_fn=lambda r: None,
                    stop_check=lambda: False,
                    error_fn=lambda e, fatal: errors.append((e, fatal)),
                )
            )

        assert len(errors) == 2
        assert [fatal for _, fatal in errors] == [False, True]
        assert all(isinstance(e, DeviceError) for e, _ in errors)
        assert all((e.facility_code, e.error_code) == (FACILITY_ACNET, ERR_RETRY) for e, _ in errors)
        assert "UNAVAILABLE" in errors[0][0].message
        assert "boom" in errors[1][0].message
        assert stub.Read.call_count == 2

    # -- Retryable vs non-retryable log levels -----------------------------

    def test_retryable_status_logs_warning(self, caplog):
        """UNAVAILABLE logs WARNING; other codes log ERROR."""
        stub = mock.MagicMock()
        n = [0]

        def make_call(*a, **kw):
            n[0] += 1
            if n[0] == 1:
                return AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "down")
            if n[0] == 2:
                return AsyncErrorIterator(grpc.StatusCode.UNKNOWN, "oops")
            return AsyncMockIterator([])

        stub.Read.side_effect = make_call

        async def fake_sleep(t):
            pass

        with (
            mock.patch("asyncio.sleep", side_effect=fake_sleep),
            caplog.at_level(logging.WARNING, logger="pacsys.backends.grpc_backend"),
        ):
            self._run(
                self._core(stub).stream(
                    drfs=["M:OUTTMP@p,1000"],
                    dispatch_fn=lambda r: None,
                    stop_check=lambda: False,
                    error_fn=lambda e, fatal: None,
                )
            )

        warn_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        err_msgs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("UNAVAILABLE" in r.message for r in warn_msgs)
        assert any("UNKNOWN" in r.message for r in err_msgs)

    # -- stop_check before backoff sleep exits immediately -----------------

    def test_stop_before_backoff_skips_sleep(self):
        """stop_check True after error_fn but before sleep → zero sleeps."""
        stub = mock.MagicMock()
        stub.Read.side_effect = lambda *a, **kw: AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "down")
        stop = [False]
        mock_sleep = mock.AsyncMock()

        with mock.patch("asyncio.sleep", mock_sleep):
            self._run(
                self._core(stub).stream(
                    drfs=["M:OUTTMP@p,1000"],
                    dispatch_fn=lambda r: None,
                    stop_check=lambda: stop[0],
                    error_fn=lambda e, fatal: stop.__setitem__(0, True),
                )
            )

        # The pre-sleep guard avoids backoff after stop.
        mock_sleep.assert_not_called()
        assert stub.Read.call_count == 1

    # -- stop_check True at entry → immediate return -----------------------

    def test_stop_at_entry_does_nothing(self):
        """stop_check=True from start → no Read call, no sleep."""
        stub = mock.MagicMock()
        mock_sleep = mock.AsyncMock()

        with mock.patch("asyncio.sleep", mock_sleep):
            self._run(
                self._core(stub).stream(
                    drfs=["M:OUTTMP@p,1000"],
                    dispatch_fn=lambda r: None,
                    stop_check=lambda: True,
                    error_fn=lambda e, fatal: None,
                )
            )

        stub.Read.assert_not_called()
        mock_sleep.assert_not_called()

    # -- Short stream does NOT reset backoff (negative control) ------------

    def test_short_stream_does_not_reset_backoff(self):
        """Stream healthy for <30s does NOT reset backoff."""
        stub = mock.MagicMock()
        reply = make_reading_reply(0, scalar_value=1.0)
        n = [0]

        def make_call(*a, **kw):
            n[0] += 1
            if n[0] == 1:
                return AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "down")
            if n[0] == 2:
                # Short healthy stream (10s < 30s) then error
                return AsyncReplyThenError(
                    [reply] * 2,
                    AsyncMockRpcError(grpc.StatusCode.UNAVAILABLE, "down"),
                )
            if n[0] == 3:
                return AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "down")
            return AsyncMockIterator([])

        stub.Read.side_effect = make_call
        sleeps = []

        async def fake_sleep(t):
            sleeps.append(t)

        # All monotonic deltas stay < 30s
        mono_values = iter(
            [
                0,  # attempt 1: stream_start (error immediately)
                100,  # attempt 2: stream_start
                105,  # attempt 2, reply 1: 105-100=5 < 30
                110,  # attempt 2, reply 2: 110-100=10 < 30  → NO reset
                200,  # attempt 3: stream_start (error immediately)
                300,  # attempt 4: stream_start
            ]
        )

        class _TimeProxy:
            def monotonic(self):
                return next(mono_values)

            def __getattr__(self, name):
                return getattr(time, name)

        with (
            mock.patch("asyncio.sleep", side_effect=fake_sleep),
            mock.patch.object(grpc_backend, "time", _TimeProxy()),
        ):
            self._run(
                self._core(stub).stream(
                    drfs=["M:OUTTMP@p,1000"],
                    dispatch_fn=lambda r: None,
                    stop_check=lambda: False,
                    error_fn=lambda e, fatal: None,
                )
            )

        # sleeps: 1.0 (err1), 2.0 (err2 - NOT reset), 4.0 (err3 - keeps growing)
        assert sleeps == [1.0, 2.0, 4.0]


# ─────────────────────────────────────────────────────────────────────────────
# Logger DRF error handling
# ─────────────────────────────────────────────────────────────────────────────


class TestLoggerReadErrors:
    """Logger status errors remain error readings."""

    def test_loggersingle_scalar_completes_without_terminator(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [
                make_reading_reply(0, scalar_value=71.25),
                make_reading_reply(1, scalar_value=72.5),
            ]
        )

        readings = backend.get_many(["M:OUTTMP<-LOGGERSINGLE:ArkIv:1736942400:60", "M:OUTTMP"])

        assert readings[0].ok
        assert readings[0].value == 71.25
        assert readings[0].value_type == ValueType.SCALAR
        assert readings[1].value == 72.5

    def test_loggersingle_error_is_preserved(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        reply = make_reading_reply(0, error_code=-64, error_message="DAE_LJ_NO_DATA")
        reply.status.facility_code = 66
        mock_stub.Read.return_value = AsyncMockIterator([reply])

        readings = backend.get_many(["M:OUTTMP<-LOGGERSINGLE:ArkIv:1736942400:60"])

        assert readings[0].facility_code == 66
        assert readings[0].error_code == -64
        assert "DAE_LJ_NO_DATA" in (readings[0].message or "")

    def test_logger_error_status_surfaced(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [make_reading_reply(0, error_code=-34, error_message="DBM_NOREC")]
        )
        readings = backend.get_many(["M:OUTTMP<-LOGGERDURATION:60000"])
        assert len(readings) == 1
        assert not readings[0].ok
        assert readings[0].error_code != 0
        assert "DBM_NOREC" in (readings[0].message or "")

    def test_logger_status_zero_still_terminates(self, backend_with_mock_stub):
        """A status-code-0 reply remains the end-of-stream terminator."""
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [
                make_reading_reply(0, scalar_value=1.5),  # data chunk
                make_reading_reply(0, error_code=0),  # terminator
            ]
        )
        readings = backend.get_many(["M:OUTTMP<-LOGGERDURATION:60000"])
        assert len(readings) == 1
        assert readings[0].ok
        assert len(readings[0].value["data"]) == 1

    @pytest.mark.parametrize("last_len", [10, 5])
    def test_array_logger_short_final_record(self, backend_with_mock_stub, last_len):
        """DataLoggerFetchJob may send a shorter final array record; only that device fails."""
        from pacsys.acnet.errors import ERR_RETRY, FACILITY_ACNET

        backend, mock_stub = backend_with_mock_stub
        replies = []
        for n in (10, 10, last_len):
            reply = DAQ_pb2.ReadingReply(index=0)
            rd = reply.readings.reading.add()
            rd.data.scalarArr.value.extend(range(n))
            rd.timestamp.seconds = 1234567890
            replies.append(reply)
        terminator = DAQ_pb2.ReadingReply(index=0)
        terminator.readings.SetInParent()
        mock_stub.Read.return_value = AsyncMockIterator([*replies, terminator, make_reading_reply(1, scalar_value=1.5)])

        readings = backend.get_many(["B:ARR[0:10]<-LOGGER:1700000000000:1700000060000", "M:OUTTMP"])

        if last_len == 10:
            assert readings[0].value_type == ValueType.TIMED_SCALAR_ARRAY
            assert readings[0].value["data"].shape == (3, 10)
            assert readings[0].value["micros"].shape == (3,)
        else:
            assert (readings[0].facility_code, readings[0].error_code) == (FACILITY_ACNET, ERR_RETRY)
        assert readings[1].ok and readings[1].value == 1.5


@pytest.mark.parametrize("operation", ["read", "write", "subscribe"])
@pytest.mark.parametrize("close_first", [True, False])
def test_close_races_operation_submission(sample_jwt, operation, close_first):
    backend = grpc_backend.GRPCBackend(auth=JWTAuth(token=sample_jwt))
    backend._start_reactor()
    entered = threading.Event()
    release = threading.Event()
    created = threading.Event()
    core = mock.Mock()
    result = [object()]

    async def run(*args):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.001)
        return result

    def create(*args):
        created.set()
        return run(*args)

    async def close():
        if not close_first:
            release.set()

    core.read_many.side_effect = create
    core.write_many.side_effect = create
    core.stream.side_effect = create
    core.close = close
    backend._core = core
    ensure = backend._ensure_reactor

    def gated_ensure():
        ensure()
        if close_first:
            entered.set()
            assert release.wait(2.0)

    outcome = []

    def call():
        try:
            if operation == "read":
                outcome.append(backend.get_many(["M:OUTTMP"]))
            elif operation == "write":
                outcome.append(backend.write_many([("M:OUTTMP", 1.0)]))
            else:
                outcome.append(backend.subscribe(["M:OUTTMP"]))
        except Exception as exc:  # noqa: BLE001 -- report worker failures to the test thread
            outcome.append(exc)

    with (
        mock.patch.object(backend, "_ensure_reactor", side_effect=gated_ensure),
        mock.patch.object(
            grpc_backend, "_GRPCSubscriptionHandle", wraps=grpc_backend._GRPCSubscriptionHandle
        ) as handle_factory,
    ):
        worker = threading.Thread(target=call)
        worker.start()
        try:
            assert entered.wait(1.0)
            backend.close()
        finally:
            release.set()
            worker.join(2.0)
            backend.close()
    assert not worker.is_alive()
    assert len(outcome) == 1
    if close_first:
        assert not created.is_set()
        handle_factory.assert_not_called()
        assert not backend._handles
        assert isinstance(outcome[0], RuntimeError)
        assert str(outcome[0]) == "Backend is closed"
    else:
        assert created.is_set()
        if operation == "subscribe" and not isinstance(outcome[0], CancelledError):
            assert outcome[0].stopped
            assert outcome[0]._task.done()
            assert not backend._handles
        else:
            assert outcome[0] == result or isinstance(outcome[0], CancelledError)
