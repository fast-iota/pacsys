"""Tests for supervised gRPC proxy server using FakeBackend + in-process channel."""

import threading
import time

import grpc
import pytest

from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2, DAQ_pb2_grpc
from pacsys.supervised import DeviceAccessPolicy, ReadOnlyPolicy, SupervisedServer, ValueRangePolicy
from pacsys.supervised._conversions import reading_to_proto_reply, write_result_to_proto_status
from pacsys.supervised._event_classify import all_oneshot, is_oneshot_event
from pacsys.supervised._policies import Policy, PolicyDecision, RequestContext
from pacsys.testing import AsyncFakeBackend, FakeBackend
from pacsys.types import Reading, ValueType, WriteResult

_ALLOW_ALL_WRITES = DeviceAccessPolicy(patterns=["*"], action="set", mode="allow")


# ── Event Classification ──────────────────────────────────────────────────


class TestEventClassify:
    def test_no_event_is_streaming(self):
        assert not is_oneshot_event("M:OUTTMP")

    def test_immediate_is_oneshot(self):
        assert is_oneshot_event("M:OUTTMP@I")

    def test_default_is_streaming(self):
        assert not is_oneshot_event("M:OUTTMP@U")

    def test_never_is_oneshot(self):
        assert is_oneshot_event("M:OUTTMP@N")

    def test_change_monitored_periodic_is_streaming(self):
        assert not is_oneshot_event("M:OUTTMP@Q,1000")

    def test_continuous_periodic_is_streaming(self):
        assert not is_oneshot_event("M:OUTTMP@p,1000")

    def test_clock_is_streaming(self):
        assert not is_oneshot_event("M:OUTTMP@E,0F")

    def test_state_is_streaming(self):
        assert not is_oneshot_event("M:OUTTMP@S,G:AMANDA,0,12,>")

    def test_all_oneshot_true(self):
        assert all_oneshot(["M:OUTTMP@I", "G:AMANDA@N"])

    def test_all_oneshot_false_mixed(self):
        assert not all_oneshot(["M:OUTTMP@I", "G:AMANDA@p,1000"])

    def test_all_oneshot_false_bare(self):
        assert not all_oneshot(["M:OUTTMP@I", "G:AMANDA"])

    def test_all_oneshot_empty(self):
        assert all_oneshot([])


# ── Server Lifecycle ──────────────────────────────────────────────────────


def _seed_backend(backend) -> None:
    backend.set_reading("M:OUTTMP", 72.5)
    backend.set_reading("G:AMANDA", 42.0)


@pytest.fixture(scope="module")
def fake_backend():
    fb = FakeBackend()
    _seed_backend(fb)
    return fb


@pytest.fixture(scope="module")
def policy_backend():
    fb = FakeBackend()
    _seed_backend(fb)
    return fb


@pytest.fixture(autouse=True)
def reset_fake_backend(request):
    if "fake_backend" in request.fixturenames or "server" in request.fixturenames:
        fb = request.getfixturevalue("fake_backend")
        fb.reset()
        _seed_backend(fb)


@pytest.fixture(autouse=True)
def reset_policy_backend(request):
    if "server_with_policy" in request.fixturenames:
        policy_backend = request.getfixturevalue("policy_backend")
        policy_backend.reset()
        _seed_backend(policy_backend)


@pytest.fixture(scope="module")
def server(fake_backend):
    """Start a SupervisedServer on an OS-assigned port, yield it, then stop."""
    srv = SupervisedServer(fake_backend, port=0, policies=[_ALLOW_ALL_WRITES])
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def server_with_policy(policy_backend):
    """Server with ReadOnlyPolicy."""
    srv = SupervisedServer(policy_backend, port=0, policies=[ReadOnlyPolicy()])
    srv.start()
    yield srv
    srv.stop()


def _make_channel(server):
    """Create a gRPC channel to the server."""
    return grpc.insecure_channel(f"localhost:{server.port}")


def _wait_subscribed(backend, timeout=2.0):
    """Poll until a new subscription appears on the backend."""
    subs = backend._subscriptions if isinstance(backend, FakeBackend) else backend._sync._subscriptions
    initial = len(subs)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(subs) > initial:
            return
        time.sleep(0.01)
    raise TimeoutError("No new subscription appeared within timeout")


# ── Server Lifecycle Tests ────────────────────────────────────────────────


class TestServerLifecycle:
    def test_start_stop(self, fake_backend):
        srv = SupervisedServer(fake_backend, port=0)
        srv.start()
        assert srv.port > 0
        srv.stop()

    def test_context_manager(self, fake_backend):
        with SupervisedServer(fake_backend, port=0) as srv:
            assert srv.port > 0

    def test_invalid_backend_type(self):
        with pytest.raises(TypeError, match="Backend"):
            SupervisedServer("not a backend")

    def test_invalid_port(self, fake_backend):
        with pytest.raises(ValueError, match="port"):
            SupervisedServer(fake_backend, port=-1)


# ── One-shot Read Tests ───────────────────────────────────────────────────


class TestOneshotRead:
    def test_single_device(self, server):
        with _make_channel(server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1
            reply = replies[0]
            assert reply.index == 0
            assert reply.WhichOneof("value") == "readings"
            reading = reply.readings.reading[0]
            assert reading.data.scalar == pytest.approx(72.5)

    def test_multiple_devices(self, server):
        with _make_channel(server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")
            request.drf.append("G:AMANDA@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 2

            values = {}
            for r in replies:
                rd = r.readings.reading[0]
                values[r.index] = rd.data.scalar

            assert values[0] == pytest.approx(72.5)
            assert values[1] == pytest.approx(42.0)

    def test_error_reading(self, fake_backend, server):
        fake_backend.set_error("M:BADDEV", -42, "Device not found")

        with _make_channel(server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:BADDEV@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1
            reply = replies[0]
            assert reply.WhichOneof("value") == "status"
            assert reply.status.status_code == -42

    def test_empty_drfs_returns_error(self, server):
        with _make_channel(server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()

            with pytest.raises(grpc.RpcError) as exc_info:
                list(stub.Read(request, timeout=5.0))
            assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# ── Streaming Read Tests ──────────────────────────────────────────────────


class TestStreamingRead:
    def test_streaming_read(self, fake_backend):
        with SupervisedServer(fake_backend, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@p,1000")

                def emit():
                    _wait_subscribed(fake_backend)
                    for i in range(3):
                        fake_backend.emit_reading("M:OUTTMP@p,1000", 70.0 + i)
                        time.sleep(0.05)

                emitter = threading.Thread(target=emit, daemon=True)
                emitter.start()

                replies = []
                for reply in stub.Read(request, timeout=3.0):
                    replies.append(reply)
                    if len(replies) >= 3:
                        break

                emitter.join(timeout=2.0)
                assert len(replies) == 3
                values = [r.readings.reading[0].data.scalar for r in replies]
                assert values == [pytest.approx(70.0), pytest.approx(71.0), pytest.approx(72.0)]

    def test_duplicate_drfs_fan_out(self, fake_backend):
        """Subscribe to [X, Y, X] — one X emit yields exactly 2 replies at indices 0 and 2."""
        with SupervisedServer(fake_backend, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@p,1000")
                request.drf.append("G:AMANDA@p,1000")
                request.drf.append("M:OUTTMP@p,1000")

                def emit():
                    _wait_subscribed(fake_backend)
                    fake_backend.emit_reading("M:OUTTMP@p,1000", 99.0)
                    time.sleep(0.5)  # wait so we can detect over-duplication
                    fake_backend.emit_reading("G:AMANDA@p,1000", 55.0)

                emitter = threading.Thread(target=emit, daemon=True)
                emitter.start()

                replies = []
                for reply in stub.Read(request, timeout=3.0):
                    replies.append(reply)
                    if len(replies) >= 3:
                        break

                emitter.join(timeout=2.0)
                # First emit (M:OUTTMP) → exactly 2 replies at index 0 and 2
                # Second emit (G:AMANDA) → 1 reply at index 1
                assert len(replies) == 3
                m_replies = [r for r in replies if r.readings.reading[0].data.scalar == pytest.approx(99.0)]
                assert len(m_replies) == 2
                assert sorted(r.index for r in m_replies) == [0, 2]
                g_replies = [r for r in replies if r.readings.reading[0].data.scalar == pytest.approx(55.0)]
                assert len(g_replies) == 1
                assert g_replies[0].index == 1


# ── Set Tests ─────────────────────────────────────────────────────────────


class TestSet:
    def test_single_write(self, fake_backend, server):
        with _make_channel(server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 80.0
            request.setting.append(setting)

            reply = stub.Set(request, timeout=5.0)
            assert len(reply.status) == 1
            assert reply.status[0].status_code == 0

            # Verify backend received the write
            assert fake_backend.was_written("M:OUTTMP")

    def test_empty_settings_returns_error(self, server):
        with _make_channel(server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()

            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Set(request, timeout=5.0)
            assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# ── Policy Enforcement Tests ──────────────────────────────────────────────


class TestPolicyEnforcement:
    def test_read_allowed_with_readonly_policy(self, server_with_policy):
        with _make_channel(server_with_policy) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1

    def test_set_blocked_with_readonly_policy(self, server_with_policy):
        with _make_channel(server_with_policy) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 80.0
            request.setting.append(setting)

            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Set(request, timeout=5.0)
            assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED


# ── Default-Deny Writes Tests ─────────────────────────────────────────────


class TestDefaultDenyWrites:
    """Sets are denied by default; reads are allowed by default."""

    def test_read_allowed_no_policies(self):
        fb = FakeBackend()
        _seed_backend(fb)
        with SupervisedServer(fb, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@I")
                replies = list(stub.Read(request, timeout=5.0))
                assert len(replies) == 1

    def test_set_denied_no_policies(self):
        fb = FakeBackend()
        _seed_backend(fb)
        with SupervisedServer(fb, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                setting = DAQ_pb2.Setting()
                setting.device = "M:OUTTMP"
                setting.value.scalar = 80.0
                request.setting.append(setting)
                with pytest.raises(grpc.RpcError) as exc_info:
                    stub.Set(request, timeout=5.0)
                assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED

    def test_set_denied_only_rate_limit_policy(self):
        """RateLimitPolicy doesn't gate writes — sets still denied."""
        fb = FakeBackend()
        _seed_backend(fb)
        from pacsys.supervised import RateLimitPolicy

        with SupervisedServer(fb, port=0, policies=[RateLimitPolicy(max_requests=100)]) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                setting = DAQ_pb2.Setting()
                setting.device = "M:OUTTMP"
                setting.value.scalar = 80.0
                request.setting.append(setting)
                with pytest.raises(grpc.RpcError) as exc_info:
                    stub.Set(request, timeout=5.0)
                assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED

    def test_set_allowed_with_access_policy(self):
        fb = FakeBackend()
        _seed_backend(fb)
        with SupervisedServer(
            fb,
            port=0,
            policies=[
                DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow"),
            ],
        ) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                setting = DAQ_pb2.Setting()
                setting.device = "M:OUTTMP"
                setting.value.scalar = 80.0
                request.setting.append(setting)
                reply = stub.Set(request, timeout=5.0)
                assert len(reply.status) == 1
                assert reply.status[0].status_code == 0

    def test_set_partial_approval_denied(self):
        """Request with one approved and one unapproved device is denied."""
        fb = FakeBackend()
        _seed_backend(fb)
        with SupervisedServer(
            fb,
            port=0,
            policies=[
                DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow"),
            ],
        ) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                s1 = DAQ_pb2.Setting()
                s1.device = "M:OUTTMP"
                s1.value.scalar = 80.0
                request.setting.append(s1)
                s2 = DAQ_pb2.Setting()
                s2.device = "G:AMANDA"
                s2.value.scalar = 50.0
                request.setting.append(s2)
                with pytest.raises(grpc.RpcError) as exc_info:
                    stub.Set(request, timeout=5.0)
                assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED

    def test_set_composable_access_policies(self):
        """Two access policies covering different device groups compose correctly."""
        fb = FakeBackend()
        _seed_backend(fb)
        with SupervisedServer(
            fb,
            port=0,
            policies=[
                DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow"),
                DeviceAccessPolicy(patterns=["G:*"], action="set", mode="allow"),
            ],
        ) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                s1 = DAQ_pb2.Setting()
                s1.device = "M:OUTTMP"
                s1.value.scalar = 80.0
                request.setting.append(s1)
                s2 = DAQ_pb2.Setting()
                s2.device = "G:AMANDA"
                s2.value.scalar = 50.0
                request.setting.append(s2)
                reply = stub.Set(request, timeout=5.0)
                assert len(reply.status) == 2


# ── ValueRangePolicy Integration Tests ───────────────────────────────────


@pytest.fixture(scope="module")
def range_backend():
    fb = FakeBackend()
    _seed_backend(fb)
    return fb


@pytest.fixture(autouse=True)
def reset_range_backend(request):
    if "server_with_range_policy" in request.fixturenames:
        fb = request.getfixturevalue("range_backend")
        fb.reset()
        _seed_backend(fb)


@pytest.fixture(scope="module")
def server_with_range_policy(range_backend):
    """Server with ValueRangePolicy limiting M:* to [0, 100]."""
    srv = SupervisedServer(
        range_backend, port=0, policies=[_ALLOW_ALL_WRITES, ValueRangePolicy(limits={"M:*": (0.0, 100.0)})]
    )
    srv.start()
    yield srv
    srv.stop()


class TestValueRangePolicyIntegration:
    def test_in_range_write_succeeds(self, server_with_range_policy):
        with _make_channel(server_with_range_policy) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 50.0
            request.setting.append(setting)

            reply = stub.Set(request, timeout=5.0)
            assert len(reply.status) == 1
            assert reply.status[0].status_code == 0

    def test_out_of_range_write_denied(self, server_with_range_policy):
        with _make_channel(server_with_range_policy) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 200.0
            request.setting.append(setting)

            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Set(request, timeout=5.0)
            assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED


# ── Async Backend Tests ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def async_backend():
    fb = AsyncFakeBackend()
    _seed_backend(fb)
    return fb


@pytest.fixture(autouse=True)
def reset_async_backend(request):
    if "async_server" in request.fixturenames:
        fb = request.getfixturevalue("async_backend")
        fb.reset()
        _seed_backend(fb)


@pytest.fixture(scope="module")
def async_server(async_backend):
    """Start a SupervisedServer with AsyncFakeBackend on an OS-assigned port."""
    srv = SupervisedServer(async_backend, port=0, policies=[_ALLOW_ALL_WRITES])
    srv.start()
    yield srv
    srv.stop()


class TestAsyncOneshotRead:
    def test_single_device(self, async_server):
        with _make_channel(async_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1
            reply = replies[0]
            assert reply.WhichOneof("value") == "readings"
            reading = reply.readings.reading[0]
            assert reading.data.scalar == pytest.approx(72.5)

    def test_multiple_devices(self, async_server):
        with _make_channel(async_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")
            request.drf.append("G:AMANDA@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 2
            values = {}
            for r in replies:
                rd = r.readings.reading[0]
                values[r.index] = rd.data.scalar
            assert values[0] == pytest.approx(72.5)
            assert values[1] == pytest.approx(42.0)

    def test_error_reading(self, async_backend, async_server):
        async_backend.set_error("M:BADDEV", -42, "Device not found")
        with _make_channel(async_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:BADDEV@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1
            reply = replies[0]
            assert reply.WhichOneof("value") == "status"
            assert reply.status.status_code == -42


class TestAsyncSet:
    def test_single_write(self, async_backend, async_server):
        with _make_channel(async_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 80.0
            request.setting.append(setting)

            reply = stub.Set(request, timeout=5.0)
            assert len(reply.status) == 1
            assert reply.status[0].status_code == 0
            assert async_backend.was_written("M:OUTTMP")


class TestAsyncStreamingRead:
    def test_streaming_read(self, async_backend):
        with SupervisedServer(async_backend, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@p,1000")

                def emit():
                    _wait_subscribed(async_backend)
                    for i in range(3):
                        async_backend.emit_reading("M:OUTTMP@p,1000", 70.0 + i)
                        time.sleep(0.05)

                emitter = threading.Thread(target=emit, daemon=True)
                emitter.start()

                replies = []
                for reply in stub.Read(request, timeout=3.0):
                    replies.append(reply)
                    if len(replies) >= 3:
                        break

                emitter.join(timeout=2.0)
                assert len(replies) == 3
                values = [r.readings.reading[0].data.scalar for r in replies]
                assert values == [pytest.approx(70.0), pytest.approx(71.0), pytest.approx(72.0)]

    def test_duplicate_drfs_fan_out(self, async_backend):
        """Async path: [X, Y, X] — one X emit yields exactly 2 replies at indices 0 and 2."""
        with SupervisedServer(async_backend, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@p,1000")
                request.drf.append("G:AMANDA@p,1000")
                request.drf.append("M:OUTTMP@p,1000")

                def emit():
                    _wait_subscribed(async_backend)
                    async_backend.emit_reading("M:OUTTMP@p,1000", 99.0)
                    time.sleep(0.5)
                    async_backend.emit_reading("G:AMANDA@p,1000", 55.0)

                emitter = threading.Thread(target=emit, daemon=True)
                emitter.start()

                replies = []
                for reply in stub.Read(request, timeout=3.0):
                    replies.append(reply)
                    if len(replies) >= 3:
                        break

                emitter.join(timeout=2.0)
                assert len(replies) == 3
                m_replies = [r for r in replies if r.readings.reading[0].data.scalar == pytest.approx(99.0)]
                assert len(m_replies) == 2
                assert sorted(r.index for r in m_replies) == [0, 2]
                g_replies = [r for r in replies if r.readings.reading[0].data.scalar == pytest.approx(55.0)]
                assert len(g_replies) == 1
                assert g_replies[0].index == 1


# ── Async Policy Enforcement Tests ───────────────────────────────────────


@pytest.fixture(scope="module")
def async_policy_backend():
    fb = AsyncFakeBackend()
    _seed_backend(fb)
    return fb


@pytest.fixture(autouse=True)
def reset_async_policy_backend(request):
    if "async_server_with_policy" in request.fixturenames:
        fb = request.getfixturevalue("async_policy_backend")
        fb.reset()
        _seed_backend(fb)


@pytest.fixture(scope="module")
def async_server_with_policy(async_policy_backend):
    srv = SupervisedServer(async_policy_backend, port=0, policies=[ReadOnlyPolicy()])
    srv.start()
    yield srv
    srv.stop()


class TestAsyncPolicyEnforcement:
    def test_read_allowed_with_readonly_policy(self, async_server_with_policy):
        with _make_channel(async_server_with_policy) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")
            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1

    def test_set_blocked_with_readonly_policy(self, async_server_with_policy):
        with _make_channel(async_server_with_policy) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 80.0
            request.setting.append(setting)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Set(request, timeout=5.0)
            assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED


# ── Token Authentication ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def token_backend():
    fb = FakeBackend()
    _seed_backend(fb)
    return fb


@pytest.fixture(scope="module")
def token_server(token_backend):
    srv = SupervisedServer(token_backend, port=0, token="test-secret", policies=[_ALLOW_ALL_WRITES])
    srv.start()
    yield srv
    srv.stop()


class TestTokenAuthentication:
    def test_read_allowed_without_token(self, token_server):
        """Reads are open — token only guards writes."""
        with _make_channel(token_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")
            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1

    def test_set_with_valid_token(self, token_server):
        with _make_channel(token_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 80.0
            request.setting.append(setting)
            md = [("authorization", "Bearer test-secret")]
            reply = stub.Set(request, timeout=5.0, metadata=md)
            assert len(reply.status) == 1
            assert reply.status[0].status_code == 0

    def test_set_rejected_without_token(self, token_server):
        with _make_channel(token_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 80.0
            request.setting.append(setting)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Set(request, timeout=5.0)
            assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED

    def test_set_rejected_with_wrong_token(self, token_server):
        with _make_channel(token_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 80.0
            request.setting.append(setting)
            md = [("authorization", "Bearer wrong-token")]
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Set(request, timeout=5.0, metadata=md)
            assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED

    def test_no_token_config_allows_all(self):
        """Server without token= accepts any request including writes."""
        fb = FakeBackend()
        _seed_backend(fb)
        with SupervisedServer(fb, port=0, policies=[_ALLOW_ALL_WRITES]) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                setting = DAQ_pb2.Setting()
                setting.device = "M:OUTTMP"
                setting.value.scalar = 80.0
                request.setting.append(setting)
                reply = stub.Set(request, timeout=5.0)
                assert len(reply.status) == 1
                assert reply.status[0].status_code == 0


# ── Backend Exception Mapping Tests ──────────────────────────────────────


class _ErrorBackend(FakeBackend):
    """FakeBackend that raises a configured exception on read/write."""

    def __init__(self, exc: Exception):
        super().__init__()
        self._exc = exc

    def get_many(self, drfs, timeout=None):
        raise self._exc

    def write_many(self, settings, timeout=None):
        raise self._exc


class TestBackendExceptionMapping:
    def test_read_not_implemented(self):
        fb = _ErrorBackend(NotImplementedError("subscribe not supported"))
        _seed_backend(fb)
        with SupervisedServer(fb, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@I")
                with pytest.raises(grpc.RpcError) as exc_info:
                    list(stub.Read(request, timeout=5.0))
                assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED

    def test_read_authentication_error(self):
        from pacsys.errors import AuthenticationError

        fb = _ErrorBackend(AuthenticationError("ticket expired"))
        _seed_backend(fb)
        with SupervisedServer(fb, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@I")
                with pytest.raises(grpc.RpcError) as exc_info:
                    list(stub.Read(request, timeout=5.0))
                assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED

    def test_read_generic_exception(self):
        fb = _ErrorBackend(RuntimeError("connection lost"))
        _seed_backend(fb)
        with SupervisedServer(fb, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@I")
                with pytest.raises(grpc.RpcError) as exc_info:
                    list(stub.Read(request, timeout=5.0))
                assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    def test_set_not_implemented(self):
        fb = _ErrorBackend(NotImplementedError("writes not supported"))
        _seed_backend(fb)
        with SupervisedServer(fb, port=0, policies=[_ALLOW_ALL_WRITES]) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                setting = DAQ_pb2.Setting()
                setting.device = "M:OUTTMP"
                setting.value.scalar = 80.0
                request.setting.append(setting)
                with pytest.raises(grpc.RpcError) as exc_info:
                    stub.Set(request, timeout=5.0)
                assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED

    def test_set_authentication_error(self):
        from pacsys.errors import AuthenticationError

        fb = _ErrorBackend(AuthenticationError("no credentials"))
        _seed_backend(fb)
        with SupervisedServer(fb, port=0, policies=[_ALLOW_ALL_WRITES]) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                setting = DAQ_pb2.Setting()
                setting.device = "M:OUTTMP"
                setting.value.scalar = 80.0
                request.setting.append(setting)
                with pytest.raises(grpc.RpcError) as exc_info:
                    stub.Set(request, timeout=5.0)
                assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED

    def test_set_generic_exception(self):
        fb = _ErrorBackend(RuntimeError("backend crashed"))
        _seed_backend(fb)
        with SupervisedServer(fb, port=0, policies=[_ALLOW_ALL_WRITES]) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                setting = DAQ_pb2.Setting()
                setting.device = "M:OUTTMP"
                setting.value.scalar = 80.0
                request.setting.append(setting)
                with pytest.raises(grpc.RpcError) as exc_info:
                    stub.Set(request, timeout=5.0)
                assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    def test_set_malformed_value(self):
        """Setting with unset value oneof returns INVALID_ARGUMENT."""
        fb = FakeBackend()
        _seed_backend(fb)
        with SupervisedServer(fb, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                setting = DAQ_pb2.Setting()
                setting.device = "M:OUTTMP"
                # Leave setting.value unset (no scalar/raw/text)
                request.setting.append(setting)
                with pytest.raises(grpc.RpcError) as exc_info:
                    stub.Set(request, timeout=5.0)
                assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# ── Conversion Edge Case Tests ───────────────────────────────────────────


class TestReadingToProtoReply:
    def test_error_reading(self):
        reading = Reading(drf="M:BAD", value_type=ValueType.SCALAR, facility_code=1, error_code=-42, message="broken")
        reply = reading_to_proto_reply(reading, 3)
        assert reply.index == 3
        assert reply.WhichOneof("value") == "status"
        assert reply.status.facility_code == 1
        assert reply.status.status_code == -42
        assert reply.status.message == "broken"

    def test_error_reading_no_message(self):
        reading = Reading(drf="M:BAD", value_type=ValueType.SCALAR, error_code=-1)
        reply = reading_to_proto_reply(reading, 0)
        assert reply.WhichOneof("value") == "status"
        assert reply.status.message == ""

    def test_success_with_timestamp(self):
        from datetime import datetime, timezone

        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        reading = Reading(drf="M:OK", value_type=ValueType.SCALAR, value=42.0, timestamp=ts)
        reply = reading_to_proto_reply(reading, 0)
        assert reply.WhichOneof("value") == "readings"
        rd = reply.readings.reading[0]
        assert rd.data.scalar == pytest.approx(42.0)
        assert rd.timestamp.seconds > 0

    def test_success_without_timestamp(self):
        reading = Reading(drf="M:OK", value_type=ValueType.SCALAR, value=10.0, timestamp=None)
        reply = reading_to_proto_reply(reading, 0)
        rd = reply.readings.reading[0]
        assert rd.data.scalar == pytest.approx(10.0)
        assert rd.timestamp.seconds == 0  # unset proto timestamp

    def test_success_with_none_value(self):
        reading = Reading(drf="M:OK", value_type=ValueType.SCALAR, value=None, error_code=0)
        reply = reading_to_proto_reply(reading, 0)
        # ok=False because value is None, so status oneof is used
        assert reply.WhichOneof("value") == "status"

    def test_success_message_propagated(self):
        reading = Reading(drf="M:OK", value_type=ValueType.SCALAR, value=1.0, message="info")
        reply = reading_to_proto_reply(reading, 0)
        rd = reply.readings.reading[0]
        assert rd.status.message == "info"

    def test_success_no_message(self):
        reading = Reading(drf="M:OK", value_type=ValueType.SCALAR, value=1.0)
        reply = reading_to_proto_reply(reading, 0)
        rd = reply.readings.reading[0]
        assert rd.status.message == ""

    def test_facility_code_propagated(self):
        reading = Reading(drf="M:OK", value_type=ValueType.SCALAR, value=1.0, facility_code=16)
        reply = reading_to_proto_reply(reading, 0)
        rd = reply.readings.reading[0]
        assert rd.status.facility_code == 16


class TestWriteResultToProtoStatus:
    def test_success(self):
        result = WriteResult(drf="M:OK", error_code=0)
        status = write_result_to_proto_status(result)
        assert status.status_code == 0
        assert status.facility_code == 0
        assert status.message == ""

    def test_error_with_message(self):
        result = WriteResult(drf="M:BAD", facility_code=1, error_code=-99, message="write failed")
        status = write_result_to_proto_status(result)
        assert status.status_code == -99
        assert status.facility_code == 1
        assert status.message == "write failed"

    def test_error_without_message(self):
        result = WriteResult(drf="M:BAD", error_code=-1)
        status = write_result_to_proto_status(result)
        assert status.status_code == -1
        assert status.message == ""


# ── Policy Reorder Index Mapping Tests ───────────────────────────────────


class _SwapPolicy(Policy):
    """Policy that reverses the DRF order to test index mapping."""

    def check(self, ctx: RequestContext) -> PolicyDecision:
        from dataclasses import replace

        new_ctx = replace(
            ctx,
            drfs=list(reversed(ctx.drfs)),
            values=list(reversed(ctx.values)),
        )
        return PolicyDecision(allowed=True, ctx=new_ctx)


class TestPolicyReorderIndexMapping:
    def test_read_reorder_preserves_original_indices(self):
        """Policy reverses [M:OUTTMP, G:AMANDA] but client gets correct index mapping."""
        fb = FakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        fb.set_reading("G:AMANDA", 42.0)
        with SupervisedServer(fb, port=0, policies=[_SwapPolicy()]) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@I")
                request.drf.append("G:AMANDA@I")

                replies = list(stub.Read(request, timeout=5.0))
                assert len(replies) == 2
                by_index = {r.index: r.readings.reading[0].data.scalar for r in replies}
                # Index 0 = M:OUTTMP (72.5), Index 1 = G:AMANDA (42.0)
                assert by_index[0] == pytest.approx(72.5)
                assert by_index[1] == pytest.approx(42.0)

    def test_set_reorder_preserves_original_indices(self):
        """Policy reverses settings order but status indices match original request."""
        fb = FakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        fb.set_reading("G:AMANDA", 42.0)
        fb.set_write_result("M:OUTTMP", success=True)
        fb.set_write_result("G:AMANDA", success=False, error_code=-1, message="fail")
        with SupervisedServer(fb, port=0, policies=[_ALLOW_ALL_WRITES, _SwapPolicy()]) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.SettingList()
                s1 = DAQ_pb2.Setting()
                s1.device = "M:OUTTMP"
                s1.value.scalar = 80.0
                request.setting.append(s1)
                s2 = DAQ_pb2.Setting()
                s2.device = "G:AMANDA"
                s2.value.scalar = 50.0
                request.setting.append(s2)

                reply = stub.Set(request, timeout=5.0)
                assert len(reply.status) == 2
                # Index 0 = M:OUTTMP (success), Index 1 = G:AMANDA (fail)
                assert reply.status[0].status_code == 0
                assert reply.status[1].status_code == -1
