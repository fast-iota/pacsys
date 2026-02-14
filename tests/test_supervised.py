"""Tests for supervised gRPC proxy server using FakeBackend + in-process channel."""

import threading
import time

import grpc
import pytest
from grpc import aio as grpc_aio

from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2, DAQ_pb2_grpc
from pacsys.supervised import ReadOnlyPolicy, SupervisedServer
from pacsys.supervised._event_classify import all_oneshot, is_oneshot_event
from pacsys.testing import FakeBackend


# ── Event Classification ──────────────────────────────────────────────────


class TestEventClassify:
    def test_no_event_is_oneshot(self):
        assert is_oneshot_event("M:OUTTMP")

    def test_immediate_is_oneshot(self):
        assert is_oneshot_event("M:OUTTMP@I")

    def test_default_is_oneshot(self):
        assert is_oneshot_event("M:OUTTMP@U")

    def test_never_is_oneshot(self):
        assert is_oneshot_event("M:OUTTMP@N")

    def test_oneshot_periodic_is_oneshot(self):
        assert is_oneshot_event("M:OUTTMP@Q,1000")

    def test_continuous_periodic_is_streaming(self):
        assert not is_oneshot_event("M:OUTTMP@p,1000")

    def test_clock_is_streaming(self):
        assert not is_oneshot_event("M:OUTTMP@E,0F")

    def test_state_is_streaming(self):
        assert not is_oneshot_event("M:OUTTMP@S,G:AMANDA,0,12,>")

    def test_all_oneshot_true(self):
        assert all_oneshot(["M:OUTTMP@I", "G:AMANDA"])

    def test_all_oneshot_false_mixed(self):
        assert not all_oneshot(["M:OUTTMP@I", "G:AMANDA@p,1000"])

    def test_all_oneshot_empty(self):
        assert all_oneshot([])


# ── Server Lifecycle ──────────────────────────────────────────────────────


def _seed_backend(backend: FakeBackend) -> None:
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
    srv = SupervisedServer(fake_backend, port=0)
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


def _make_async_channel(server):
    """Create an async gRPC channel to the server."""
    return grpc_aio.insecure_channel(f"localhost:{server.port}")


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

    def test_repr(self, server):
        r = repr(server)
        assert "SupervisedServer" in r
        assert "running" in r


# ── One-shot Read Tests ───────────────────────────────────────────────────


class TestOneshotRead:
    def test_single_device(self, server):
        with _make_channel(server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP")

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
            request.drf.append("M:OUTTMP")
            request.drf.append("G:AMANDA")

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
            request.drf.append("M:BADDEV")

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

                # Emit readings in a separate thread after a short delay
                def emit():
                    time.sleep(0.3)
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

                assert len(replies) == 3
                values = [r.readings.reading[0].data.scalar for r in replies]
                assert values == [pytest.approx(70.0), pytest.approx(71.0), pytest.approx(72.0)]


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
            request.drf.append("M:OUTTMP")

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


# ── Alarms Tests ──────────────────────────────────────────────────────────


class TestAlarms:
    def test_alarms_unimplemented(self, server):
        with _make_channel(server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.DeviceList()
            request.device.append("M:OUTTMP")

            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Alarms(request, timeout=5.0)
            assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
