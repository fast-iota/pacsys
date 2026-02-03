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
- Uses stub mocking for unit tests (requires real proto files)
"""

import logging
import os
import queue
from unittest import mock

import pytest

from pacsys.auth import JWTAuth
from pacsys.backends import Backend
from pacsys.errors import AuthenticationError, DeviceError
from pacsys.types import BackendCapability, Reading, ValueType, WriteResult
from tests.devices import make_jwt_token

# sample_jwt fixture is provided by conftest.py

# Check if grpc and proto files are available
try:
    from pacsys.backends import grpc_backend
    from proto.controls.common.v1 import device_pb2, status_pb2
    from proto.controls.service.DAQ.v1 import DAQ_pb2

    GRPC_AVAILABLE = grpc_backend.GRPC_AVAILABLE
except ImportError:
    GRPC_AVAILABLE = False
    grpc_backend = None
    DAQ_pb2 = None
    device_pb2 = None
    status_pb2 = None

if not GRPC_AVAILABLE:
    pytest.skip("grpc and proto files not available", allow_module_level=True)

import grpc  # noqa: E402


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


# ─────────────────────────────────────────────────────────────────────────────
# Reply factories
# ─────────────────────────────────────────────────────────────────────────────


def make_reading_reply(
    index: int,
    scalar_value: float = None,
    text_value: str = None,
    error_code: int = None,
    error_message: str = None,
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


@pytest.fixture
def backend_with_mock_stub(mock_stub):
    """GRPCBackend (no auth) with mocked stub — for read tests."""
    backend = _make_backend_with_stub(mock_stub)
    yield backend, mock_stub
    backend.close()


@pytest.fixture
def auth_backend_with_mock_stub(mock_stub, sample_jwt):
    """GRPCBackend (with JWT auth) and mocked stub — for write tests."""
    auth = JWTAuth(token=sample_jwt)
    backend = _make_backend_with_stub(mock_stub, auth=auth)
    yield backend, mock_stub
    backend.close()


# ─────────────────────────────────────────────────────────────────────────────
# Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGRPCBackendInit:
    """Tests for GRPCBackend initialization."""

    def test_default_parameters(self):
        backend = grpc_backend.GRPCBackend()
        try:
            assert backend.host == grpc_backend.DEFAULT_HOST
            assert backend.port == grpc_backend.DEFAULT_PORT
            assert backend.timeout == grpc_backend.DEFAULT_TIMEOUT
            assert not backend.authenticated
        finally:
            backend.close()

    def test_custom_parameters(self, sample_jwt):
        auth = JWTAuth(token=sample_jwt)
        backend = grpc_backend.GRPCBackend(
            host="custom.example.com",
            port=50052,
            auth=auth,
            timeout=5.0,
        )
        try:
            assert backend.host == "custom.example.com"
            assert backend.port == 50052
            assert backend.timeout == 5.0
            assert backend.authenticated
        finally:
            backend.close()

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

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"host": ""}, "host cannot be empty"),
            ({"port": 0}, "port must be positive"),
            ({"port": -1}, "port must be positive"),
            ({"timeout": 0}, "timeout must be positive"),
            ({"timeout": -1.0}, "timeout must be positive"),
        ],
    )
    def test_invalid_init_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            grpc_backend.GRPCBackend(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Capabilities Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCapabilities:
    """Tests for backend capabilities."""

    @pytest.mark.parametrize(
        "cap,present",
        [
            (BackendCapability.READ, True),
            (BackendCapability.STREAM, True),
            (BackendCapability.BATCH, True),
            (BackendCapability.WRITE, False),
            (BackendCapability.AUTH_JWT, False),
            (BackendCapability.AUTH_KERBEROS, False),
        ],
    )
    def test_capabilities_no_auth(self, cap, present):
        backend = grpc_backend.GRPCBackend()
        try:
            assert (cap in backend.capabilities) == present
        finally:
            backend.close()

    @pytest.mark.parametrize(
        "cap,present",
        [
            (BackendCapability.READ, True),
            (BackendCapability.WRITE, True),
            (BackendCapability.AUTH_JWT, True),
        ],
    )
    def test_capabilities_with_auth(self, sample_jwt, cap, present):
        auth = JWTAuth(token=sample_jwt)
        backend = grpc_backend.GRPCBackend(auth=auth)
        try:
            assert (cap in backend.capabilities) == present
        finally:
            backend.close()

    def test_not_authenticated_without_token(self):
        backend = grpc_backend.GRPCBackend()
        try:
            assert not backend.authenticated
            assert backend.principal is None
        finally:
            backend.close()

    def test_authenticated_with_auth(self, sample_jwt):
        auth = JWTAuth(token=sample_jwt)
        backend = grpc_backend.GRPCBackend(auth=auth)
        try:
            assert backend.authenticated
            assert backend.principal == "testuser@fnal.gov"
        finally:
            backend.close()


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

    def test_read_error_raises_device_error(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [make_reading_reply(0, error_code=-42, error_message="Device not found")]
        )

        with pytest.raises(DeviceError) as exc_info:
            backend.read("M:BADDEV")
        assert "Device not found" in exc_info.value.message

    def test_get_error_returns_reading_with_error(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncMockIterator(
            [make_reading_reply(0, error_code=-42, error_message="Device not found")]
        )

        reading = backend.get("M:BADDEV")
        assert reading.is_error
        assert not reading.ok
        assert "Device not found" in reading.message


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Device Read Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleDeviceRead:
    """Tests for multiple device get_many operations."""

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

    def test_get_many_empty_list(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        readings = backend.get_many([])
        assert readings == []


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
        """BUG FIX: missing server responses should be errors, not success."""
        backend, mock_stub = auth_backend_with_mock_stub
        mock_stub.Set = mock.AsyncMock(return_value=make_setting_reply([0]))

        results = backend.write_many([("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)])
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success
        assert "No status received" in results[1].message


# ─────────────────────────────────────────────────────────────────────────────
# gRPC Error Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGRPCErrors:
    """Tests for gRPC error handling."""

    def test_grpc_error_on_read(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncErrorIterator(grpc.StatusCode.UNAVAILABLE, "Connection refused")

        with pytest.raises(DeviceError) as exc_info:
            backend.read("M:OUTTMP")
        assert "UNAVAILABLE" in exc_info.value.message

    def test_grpc_error_on_get_many(self, backend_with_mock_stub):
        backend, mock_stub = backend_with_mock_stub
        mock_stub.Read.return_value = AsyncErrorIterator(grpc.StatusCode.DEADLINE_EXCEEDED, "Timeout")

        readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
        assert len(readings) == 2
        assert all(r.is_error for r in readings)
        assert all("DEADLINE_EXCEEDED" in r.message for r in readings)


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
        "method,args",
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
        "input_val,field,expected",
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

    @pytest.mark.parametrize(
        "field,set_val,expected_val,expected_type",
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

    def test_alarm_dict_skips_readonly_keys(self):
        d = {"minimum": 1.0, "maximum": 2.0, "alarm_status": True, "abort": False, "tries_now": 5}
        proto = grpc_backend._value_to_proto_value(d)
        assert proto.WhichOneof("value") == "anaAlarm"
        assert proto.anaAlarm.minimum == 1.0

    def test_alarm_dict_unknown_keys_raises(self):
        with pytest.raises(ValueError, match="Unknown alarm dict keys"):
            grpc_backend._value_to_proto_value({"minimum": 1.0, "bogus": 42})

    def test_alarm_dict_mixed_keys_raises(self):
        with pytest.raises(ValueError, match="Cannot mix analog.*and digital"):
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
        for key in original:
            assert result[key] == original[key]


# ─────────────────────────────────────────────────────────────────────────────
# Status Code Normalization
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusCodeNormalization:
    """Tests for uint8 -> int8 status code normalization."""

    @pytest.mark.parametrize(
        "input_code,expected",
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


class TestBackendInheritance:
    """Tests for Backend abstract base class inheritance."""

    def test_grpc_backend_is_subclass(self):
        assert issubclass(grpc_backend.GRPCBackend, Backend)


# ─────────────────────────────────────────────────────────────────────────────
# Bounded Queue Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBoundedQueue:
    """Tests for bounded FIFO queue in subscription handle."""

    def test_queue_is_bounded(self):
        """Verify the queue has the expected maxsize."""
        backend = grpc_backend.GRPCBackend()
        try:
            handle = grpc_backend._GRPCSubscriptionHandle(
                backend=backend,
                drfs=["M:OUTTMP@p,1000"],
                callback=None,
                on_error=None,
            )
            assert handle._queue.maxsize == grpc_backend._DEFAULT_QUEUE_MAXSIZE
        finally:
            backend.close()

    def test_queue_overflow_drops_and_warns(self, caplog):
        """When the queue is full, new readings are dropped with a warning."""
        backend = grpc_backend.GRPCBackend()
        try:
            handle = grpc_backend._GRPCSubscriptionHandle(
                backend=backend,
                drfs=["M:OUTTMP@p,1000"],
                callback=None,
                on_error=None,
            )
            # Use a tiny queue for testing
            handle._queue = queue.Queue(maxsize=2)

            r1 = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=1.0)
            r2 = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=2.0)
            r3 = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=3.0)

            handle._dispatch(r1)
            handle._dispatch(r2)

            with caplog.at_level(logging.WARNING, logger="pacsys.backends.grpc_backend"):
                handle._dispatch(r3)

            assert handle._queue.qsize() == 2
            # Oldest readings survive (FIFO)
            assert handle._queue.get_nowait().value == 1.0
            assert handle._queue.get_nowait().value == 2.0
            assert any("queue full" in rec.message for rec in caplog.records)
        finally:
            backend.close()


# ─────────────────────────────────────────────────────────────────────────────
# Reactor Lifecycle Tests
# ─────────────────────────────────────────────────────────────────────────────


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
