"""Tests for _AsyncDpmCore."""

import asyncio
from unittest import mock

import pytest

from pacsys.errors import AuthenticationError, ReadError

from pacsys.dpm_protocol import (
    AddToList_reply,
    DeviceInfo_reply,
    ListStatus_reply,
    Scalar_reply,
    SettingStatus_struct,
    StartList_reply,
    Status_reply,
    ApplySettings_reply,
)
from pacsys.backends._dpm_core import _AsyncDpmCore


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
