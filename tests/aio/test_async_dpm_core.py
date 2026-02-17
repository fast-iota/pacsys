"""Tests for _AsyncDpmCore."""

import asyncio
from unittest import mock

import numpy as np
import pytest

from pacsys.errors import AuthenticationError, ReadError
from pacsys.types import ValueType

from pacsys.dpm_protocol import (
    AddToList_reply,
    Authenticate_reply,
    Authenticate_request,
    DeviceInfo_reply,
    ListStatus_reply,
    Scalar_reply,
    SettingStatus_struct,
    StartList_reply,
    Status_reply,
    TimedScalarArray_reply,
    ApplySettings_reply,
)
from pacsys.backends._dpm_core import _AsyncDpmCore
from tests.devices import MockGSSAPIModule, make_auth_reply


def _scalar_reply(ref_id, value):
    r = Scalar_reply()
    r.ref_id = ref_id
    r.data = value
    r.timestamp = 1000
    r.cycle = 0
    return r


def _device_info(ref_id):
    r = DeviceInfo_reply()
    r.ref_id = ref_id
    r.units = "degF"
    r.format_hint = "%f"
    return r


def _add_ok(ref_id):
    r = AddToList_reply()
    r.ref_id = ref_id
    r.status = 0
    return r


def _add_error(ref_id, status=0xBB06):
    r = AddToList_reply()
    r.ref_id = ref_id
    r.status = status
    return r


def _start_ok():
    r = StartList_reply()
    r.status = 0
    return r


def _status_ok(ref_id=0):
    r = Status_reply()
    r.ref_id = ref_id
    r.status = 0
    return r


def _list_status():
    return ListStatus_reply()


def _apply_settings_reply(pairs):
    """Build ApplySettings_reply with SettingStatus_struct list."""
    reply = ApplySettings_reply()
    reply.status = []
    for ref_id, status in pairs:
        s = SettingStatus_struct()
        s.ref_id = ref_id
        s.status = status
        reply.status.append(s)
    return reply


class FakeAsyncConn:
    """Fake _AsyncDPMConnection that replays canned messages."""

    def __init__(self, replies):
        self._replies = list(replies)
        self._idx = 0
        self.list_id = 42
        self.sent = []
        self._closed = False

    async def connect(self):
        pass

    async def send_message(self, msg):
        self.sent.append(msg)

    async def send_messages_batch(self, msgs):
        self.sent.extend(msgs)

    async def recv_message(self):
        if self._idx >= len(self._replies):
            # Block forever (simulate no more data)
            await asyncio.sleep(100)
        reply = self._replies[self._idx]
        self._idx += 1
        return reply

    async def close(self):
        self._closed = True


@pytest.fixture
def make_core():
    def _make(replies, auth=None, role=None):
        core = _AsyncDpmCore("localhost", 6802, timeout=2.0, auth=auth, role=role)
        conn = FakeAsyncConn(replies)
        core._conn = conn
        return core, conn

    return _make


class TestReadMany:
    @pytest.mark.asyncio
    async def test_read_single(self, make_core):
        replies = [_add_ok(1), _device_info(1), _start_ok(), _scalar_reply(1, 72.5)]
        core, conn = make_core(replies)
        readings = await core.read_many(["M:OUTTMP"], timeout=2.0)
        assert len(readings) == 1
        assert readings[0].value == 72.5
        assert readings[0].ok

    @pytest.mark.asyncio
    async def test_read_many_multiple(self, make_core):
        replies = [
            _add_ok(1),
            _add_ok(2),
            _device_info(1),
            _device_info(2),
            _start_ok(),
            _scalar_reply(1, 10.0),
            _scalar_reply(2, 20.0),
        ]
        core, conn = make_core(replies)
        readings = await core.read_many(["M:OUTTMP", "G:AMANDA"], timeout=2.0)
        assert len(readings) == 2
        assert readings[0].value == 10.0
        assert readings[1].value == 20.0

    @pytest.mark.asyncio
    async def test_read_with_add_error(self, make_core):
        # 0xBB06 -> (facility=6, error=-69) which is a real ACNET error
        replies = [_add_error(1, status=0xBB06), _start_ok()]
        core, conn = make_core(replies)
        # AddToList error is a server-side error, returned as error Reading (not ReadError)
        readings = await core.read_many(["M:OUTTMP"], timeout=2.0)
        assert len(readings) == 1
        assert readings[0].is_error

    @pytest.mark.asyncio
    async def test_read_timeout(self, make_core):
        # No data replies - will timeout
        replies = [_add_ok(1), _start_ok(), _list_status()]
        core, conn = make_core(replies)
        with pytest.raises(ReadError):
            await core.read_many(["M:OUTTMP"], timeout=0.1)

    @pytest.mark.asyncio
    async def test_read_sends_stop_clear(self, make_core):
        replies = [_add_ok(1), _device_info(1), _start_ok(), _scalar_reply(1, 1.0)]
        core, conn = make_core(replies)
        await core.read_many(["M:OUTTMP"], timeout=2.0)
        # Last two messages should be StopList + ClearList

        sent_types = [type(m).__name__ for m in conn.sent]
        assert "StopList_request" in sent_types
        assert "ClearList_request" in sent_types

    @pytest.mark.asyncio
    async def test_read_ignores_list_status(self, make_core):
        replies = [_add_ok(1), _list_status(), _device_info(1), _start_ok(), _list_status(), _scalar_reply(1, 5.0)]
        core, conn = make_core(replies)
        readings = await core.read_many(["M:OUTTMP"], timeout=2.0)
        assert readings[0].value == 5.0


class TestWriteMany:
    @pytest.mark.asyncio
    async def test_write_single(self, make_core):
        replies = [
            _add_ok(1),
            _device_info(1),
            _start_ok(),
            _apply_settings_reply([(1, 0)]),
        ]
        core, conn = make_core(replies)
        core._settings_enabled = True
        core._auth = mock.MagicMock()
        core._auth.principal = "test@fnal.gov"

        results = await core.write_many([("M:OUTTMP.SETTING@N", 72.5)], timeout=2.0)
        assert len(results) == 1
        assert results[0].success

    @pytest.mark.asyncio
    async def test_write_auto_authenticates(self, make_core):
        """Write auto-authenticates if settings not enabled."""
        auth = mock.MagicMock()
        auth.principal = "test@fnal.gov"

        core, conn = make_core([], auth=auth)
        # Patch authenticate and enable_settings
        core.authenticate = mock.AsyncMock()
        core.enable_settings = mock.AsyncMock()
        core._settings_enabled = False  # will be set by mocked enable_settings

        # After auth, mock enable_settings sets _settings_enabled
        async def fake_enable():
            core._settings_enabled = True

        core.enable_settings = fake_enable

        conn._replies = [_add_ok(1), _device_info(1), _start_ok(), _apply_settings_reply([(1, 0)])]

        results = await core.write_many([("M:OUTTMP.SETTING@N", 72.5)], timeout=2.0)
        core.authenticate.assert_awaited_once()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_write_with_role(self, make_core):
        replies = [_add_ok(1), _device_info(1), _start_ok(), _apply_settings_reply([(1, 0)])]
        core, conn = make_core(replies, role="testing")
        core._settings_enabled = True
        core._auth = mock.MagicMock()
        core._auth.principal = "test@fnal.gov"

        await core.write_many([("M:OUTTMP.SETTING@N", 72.5)], timeout=2.0)
        # Check that a #ROLE: AddToList was sent
        from pacsys.dpm_protocol import AddToList_request

        role_msgs = [m for m in conn.sent if isinstance(m, AddToList_request) and m.ref_id == 0]
        assert len(role_msgs) == 1
        assert "ROLE:testing" in role_msgs[0].drf_request


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_dispatches_readings(self, make_core):
        replies = [_add_ok(1), _device_info(1), _start_ok(), _scalar_reply(1, 42.0)]
        core, conn = make_core(replies)
        dispatched = []
        stopped = False

        def dispatch(reading):
            nonlocal stopped
            dispatched.append(reading)
            stopped = True

        await core.stream(["M:OUTTMP@p,1000"], dispatch, lambda: stopped, lambda e: None)
        assert len(dispatched) == 1
        assert dispatched[0].value == 42.0

    @pytest.mark.asyncio
    async def test_stream_handles_add_error(self, make_core):
        replies = [_add_error(1, status=0xBB06), _start_ok()]
        core, conn = make_core(replies)
        dispatched = []
        stopped = False

        def dispatch(reading):
            nonlocal stopped
            dispatched.append(reading)
            stopped = True

        await core.stream(["M:OUTTMP@p,1000"], dispatch, lambda: stopped, lambda e: None)
        assert len(dispatched) == 1
        assert dispatched[0].is_error


class TestEnableSettings:
    @pytest.mark.asyncio
    async def test_enable_settings_success(self, make_core):
        replies = [_list_status(), _status_ok()]
        core, conn = make_core(replies)
        core._mic = b"fake_mic"
        core._mic_message = b"1234"
        await core.enable_settings()
        assert core._settings_enabled is True

    @pytest.mark.asyncio
    async def test_enable_settings_failure(self, make_core):
        fail = Status_reply()
        fail.ref_id = 0
        fail.status = 0x10002  # some error status
        replies = [fail]
        core, conn = make_core(replies)
        core._mic = b"fake_mic"
        core._mic_message = b"1234"
        with pytest.raises(AuthenticationError, match="EnableSettings failed"):
            await core.enable_settings()

    @pytest.mark.asyncio
    async def test_enable_settings_requires_mic(self, make_core):
        core, conn = make_core([])
        with pytest.raises(AuthenticationError, match="Must authenticate"):
            await core.enable_settings()


def _timed_scalar_array(ref_id, data, micros, timestamp=1000):
    r = TimedScalarArray_reply()
    r.ref_id = ref_id
    r.data = list(data)
    r.micros = list(micros)
    r.timestamp = timestamp
    r.cycle = 0
    r.status = 0
    return r


def _empty_timed_scalar_array(ref_id):
    r = TimedScalarArray_reply()
    r.ref_id = ref_id
    r.data = []
    r.micros = []
    r.timestamp = 0
    r.cycle = 0
    r.status = 0
    return r


LOGGER_DRF = "M:OUTTMP<-LOGGER:1736942400000:1736946000000"


class TestLoggerRead:
    @pytest.mark.asyncio
    async def test_logger_accumulates_chunks(self, make_core):
        """Two data chunks + empty terminator -> merged TIMED_SCALAR_ARRAY."""
        chunk1 = _timed_scalar_array(1, [1.0, 2.0, 3.0], [100, 200, 300], timestamp=1000)
        chunk2 = _timed_scalar_array(1, [4.0, 5.0], [400, 500], timestamp=2000)
        terminator = _empty_timed_scalar_array(1)
        replies = [_add_ok(1), _device_info(1), _start_ok(), chunk1, chunk2, terminator]
        core, conn = make_core(replies)

        readings = await core.read_many([LOGGER_DRF], timeout=2.0)

        assert len(readings) == 1
        r = readings[0]
        assert r.ok
        assert r.value_type == ValueType.TIMED_SCALAR_ARRAY
        assert isinstance(r.value, dict)
        np.testing.assert_array_equal(r.value["data"], [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(r.value["micros"], [100, 200, 300, 400, 500])

    @pytest.mark.asyncio
    async def test_logger_empty_window(self, make_core):
        """Only empty terminator -> empty arrays (valid empty time window)."""
        terminator = _empty_timed_scalar_array(1)
        replies = [_add_ok(1), _device_info(1), _start_ok(), terminator]
        core, conn = make_core(replies)

        readings = await core.read_many([LOGGER_DRF], timeout=2.0)

        assert len(readings) == 1
        r = readings[0]
        assert r.ok
        assert r.value_type == ValueType.TIMED_SCALAR_ARRAY
        assert len(r.value["data"]) == 0
        assert len(r.value["micros"]) == 0

    @pytest.mark.asyncio
    async def test_logger_mixed_with_normal_read(self, make_core):
        """Logger DRF + normal DRF in same batch both return correctly."""
        chunk1 = _timed_scalar_array(1, [10.0, 20.0], [100, 200])
        terminator = _empty_timed_scalar_array(1)
        normal_reply = _scalar_reply(2, 72.5)
        replies = [
            _add_ok(1),
            _add_ok(2),
            _device_info(1),
            _device_info(2),
            _start_ok(),
            chunk1,
            terminator,
            normal_reply,
        ]
        core, conn = make_core(replies)

        readings = await core.read_many([LOGGER_DRF, "M:OUTTMP"], timeout=2.0)

        assert len(readings) == 2
        # Logger reading
        assert readings[0].value_type == ValueType.TIMED_SCALAR_ARRAY
        np.testing.assert_array_equal(readings[0].value["data"], [10.0, 20.0])
        # Normal reading
        assert readings[1].value == 72.5

    @pytest.mark.asyncio
    async def test_logger_status_reply_error(self, make_core):
        """Status_reply error for a logger DRF surfaces as a proper ACNET error."""
        error_status = Status_reply()
        error_status.ref_id = 1
        error_status.status = 0xBB06
        replies = [_add_ok(1), _device_info(1), _start_ok(), error_status]
        core, conn = make_core(replies)

        readings = await core.read_many([LOGGER_DRF], timeout=2.0)

        assert len(readings) == 1
        assert readings[0].is_error
        assert readings[0].error_code != 0

    @pytest.mark.asyncio
    async def test_logger_error_terminator(self, make_core):
        """Empty terminator with nonzero status surfaces as error, not empty success."""
        error_term = TimedScalarArray_reply()
        error_term.ref_id = 1
        error_term.data = []
        error_term.micros = []
        error_term.timestamp = 0
        error_term.cycle = 0
        error_term.status = 0xBB06
        replies = [_add_ok(1), _device_info(1), _start_ok(), error_term]
        core, conn = make_core(replies)

        readings = await core.read_many([LOGGER_DRF], timeout=2.0)

        assert len(readings) == 1
        assert readings[0].is_error


def _make_auth_reply_with_token():
    """Auth reply with token for mutual auth phase."""
    reply = Authenticate_reply()
    reply.serviceName = ""
    reply.token = b"server_token"
    return reply


@pytest.fixture
def mock_kerberos_auth():
    """Context manager that patches KerberosAuth._get_credentials."""
    from pacsys.auth import KerberosAuth

    def _make(mock_gssapi):
        auth = KerberosAuth(_lazy=True)
        patcher = mock.patch.object(KerberosAuth, "_get_credentials", return_value=mock_gssapi.Credentials())
        patcher.start()
        return auth, patcher

    patchers = []

    def factory(mock_gssapi):
        auth, patcher = _make(mock_gssapi)
        patchers.append(patcher)
        return auth

    yield factory
    for p in patchers:
        p.stop()


class TestAuthenticate:
    """Tests for _AsyncDpmCore.authenticate() Kerberos handshake."""

    @pytest.mark.asyncio
    async def test_happy_path(self, make_core, mock_kerberos_auth):
        """Two-phase handshake completes, MIC is stored."""
        mock_gssapi = MockGSSAPIModule()
        mock_gssapi.SecurityContext = MockGSSAPIContextForAuth
        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = mock_kerberos_auth(mock_gssapi)
            replies = [make_auth_reply("dpm\\@host"), make_auth_reply()]
            core, conn = make_core(replies, auth=auth)
            await core.authenticate()
            assert core._mic == b"mock_mic_signature"
            assert core._mic_message == b"1234"
            auth_reqs = [m for m in conn.sent if isinstance(m, Authenticate_request)]
            assert len(auth_reqs) == 2
            assert auth_reqs[0].token == b""

    @pytest.mark.asyncio
    async def test_no_auth_raises(self, make_core):
        """authenticate() without KerberosAuth raises AuthenticationError."""
        core, conn = make_core([])
        with pytest.raises(AuthenticationError, match="KerberosAuth required"):
            await core.authenticate()

    @pytest.mark.asyncio
    async def test_empty_service_name_raises(self, make_core, mock_kerberos_auth):
        """Empty service name from server raises AuthenticationError."""
        mock_gssapi = MockGSSAPIModule()
        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = mock_kerberos_auth(mock_gssapi)
            reply = Authenticate_reply()
            reply.serviceName = ""
            core, conn = make_core([reply], auth=auth)
            with pytest.raises(AuthenticationError, match="service name"):
                await core.authenticate()

    @pytest.mark.asyncio
    async def test_wrong_reply_type_phase1(self, make_core, mock_kerberos_auth):
        """Non-Authenticate_reply in phase 1 raises AuthenticationError."""
        mock_gssapi = MockGSSAPIModule()
        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = mock_kerberos_auth(mock_gssapi)
            core, conn = make_core([_status_ok()], auth=auth)
            with pytest.raises(AuthenticationError, match="Expected Authenticate_reply"):
                await core.authenticate()

    @pytest.mark.asyncio
    async def test_wrong_reply_type_phase2(self, make_core, mock_kerberos_auth):
        """Non-Authenticate_reply in phase 2 raises AuthenticationError."""
        mock_gssapi = MockGSSAPIModule()
        mock_gssapi.SecurityContext = MockGSSAPIContextForAuth
        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = mock_kerberos_auth(mock_gssapi)
            core, conn = make_core([make_auth_reply("dpm"), _status_ok()], auth=auth)
            with pytest.raises(AuthenticationError, match="Expected Authenticate_reply"):
                await core.authenticate()

    @pytest.mark.asyncio
    async def test_context_incomplete_raises(self, make_core, mock_kerberos_auth):
        """Context never completes -> AuthenticationError."""
        mock_gssapi = MockGSSAPIModule()
        mock_gssapi.SecurityContext = MockGSSAPIContextNeverComplete
        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = mock_kerberos_auth(mock_gssapi)
            replies = [make_auth_reply("dpm"), make_auth_reply()]
            core, conn = make_core(replies, auth=auth)
            with pytest.raises(AuthenticationError, match="incomplete"):
                await core.authenticate()

    @pytest.mark.asyncio
    async def test_service_name_transformation(self, make_core, mock_kerberos_auth):
        """Verifies Java-escaped service name is correctly transformed."""
        mock_gssapi = MockGSSAPIModule()
        captured_names = []
        original_name = mock_gssapi.Name

        def capture_name(name, name_type=None):
            captured_names.append(name)
            return original_name(name, name_type)

        mock_gssapi.Name = staticmethod(capture_name)
        mock_gssapi.SecurityContext = MockGSSAPIContextForAuth
        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = mock_kerberos_auth(mock_gssapi)
            replies = [make_auth_reply("daeset\\@somehost"), make_auth_reply()]
            core, conn = make_core(replies, auth=auth)
            await core.authenticate()
            assert "daeset/somehost@FNAL.GOV" in captured_names

    @pytest.mark.asyncio
    async def test_timeout_on_recv(self, make_core, mock_kerberos_auth):
        """Timeout waiting for server reply raises TimeoutError."""
        mock_gssapi = MockGSSAPIModule()
        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = mock_kerberos_auth(mock_gssapi)
            core, conn = make_core([], auth=auth)
            core._timeout = 0.1
            with pytest.raises(asyncio.TimeoutError):
                await core.authenticate()


class MockGSSAPIContextForAuth:
    """Mock GSSAPI context that completes after first step."""

    def __init__(self, name=None, usage=None, flags=None, creds=None, mech=None):
        self.complete = False
        self._step_count = 0

    def step(self, token=None):
        self._step_count += 1
        self.complete = True
        return b"mock_kerberos_token"

    def get_signature(self, message):
        return b"mock_mic_signature"


class MockGSSAPIContextNeverComplete:
    """Mock GSSAPI context that never completes."""

    def __init__(self, name=None, usage=None, flags=None, creds=None, mech=None):
        self.complete = False

    def step(self, token=None):
        return b"mock_kerberos_token"

    def get_signature(self, message):
        return b"mock_mic_signature"
