"""
Real integration tests for supervised gRPC proxy server.

Architecture: GRPCBackend(client) -> SupervisedServer(gRPC proxy) -> DPMHTTPBackend -> live DPM server

Run:
    PACSYS_TEST_REAL=1 python -m pytest tests/real/test_supervised.py -v -s

With writes:
    PACSYS_TEST_REAL=1 PACSYS_TEST_WRITE=1 python -m pytest tests/real/test_supervised.py -v -s
"""

import time

import grpc
import pytest

from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2, DAQ_pb2_grpc
from pacsys.backends.dpm_http import DPMHTTPBackend
from pacsys.backends.grpc_backend import GRPCBackend
from pacsys.errors import ReadError
from pacsys.supervised import (
    DeviceAccessPolicy,
    RateLimitPolicy,
    ReadOnlyPolicy,
    SupervisedServer,
)

from pacsys.drf_utils import strip_event

from .devices import (
    ANALOG_ALARM_DEVICE,
    ARRAY_DEVICE,
    CONTROL_PAIRS,
    CONTROL_RESET,
    DESCRIPTION_DEVICE,
    DIGITAL_ALARM_DEVICE,
    NONEXISTENT_DEVICE,
    PERIODIC_DEVICE,
    RAW_DEVICE,
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_ELEMENT,
    SCALAR_SETPOINT,
    SCALAR_SETPOINT_RAW,
    STATUS_CONTROL_DEVICE,
    STATUS_DEVICE,
    TIMEOUT_BATCH,
    TIMEOUT_READ,
    TIMEOUT_STREAM_EVENT,
    dpm_server_available,
    grpc_server_available,
    kerberos_available,
    requires_dpm_http,
    requires_grpc,
    requires_kerberos,
    requires_write_enabled,
)


def assert_permission_denied(exc_info: pytest.ExceptionInfo[ReadError]) -> None:
    """Assert that a ReadError was caused by PERMISSION_DENIED gRPC status."""
    cause = exc_info.value.__cause__
    assert cause is not None, "ReadError has no chained cause"
    assert hasattr(cause, "code"), f"Chained cause is not a gRPC error: {type(cause)}"
    assert cause.code() == grpc.StatusCode.PERMISSION_DENIED, f"Expected PERMISSION_DENIED, got {cause.code()}"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="class")
def real_backend():
    """DPMHTTPBackend for proxying through supervised server."""
    if not dpm_server_available():
        pytest.skip("DPM server not available")
    backend = DPMHTTPBackend()
    yield backend
    backend.close()


@pytest.fixture
def real_write_backend():
    """DPMHTTPBackend with Kerberos auth for write tests."""
    if not dpm_server_available():
        pytest.skip("DPM server not available")
    if not kerberos_available():
        pytest.skip("Kerberos credentials not available")
    from pacsys.auth import KerberosAuth

    backend = DPMHTTPBackend(auth=KerberosAuth(), role="testing")
    yield backend
    backend.close()


@pytest.fixture(scope="class")
def supervised_server(real_backend):
    """SupervisedServer wrapping the real DPM HTTP backend."""
    srv = SupervisedServer(real_backend, port=0)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture(scope="class")
def proxy_client(supervised_server):
    """GRPCBackend client connected to the supervised proxy."""
    client = GRPCBackend(host="localhost", port=supervised_server.port, timeout=TIMEOUT_READ)
    yield client
    client.close()


@pytest.fixture
def readonly_server(real_backend):
    """SupervisedServer with ReadOnlyPolicy."""
    srv = SupervisedServer(real_backend, port=0, policies=[ReadOnlyPolicy()])
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def device_limited_server(real_backend):
    """SupervisedServer denying G:* devices (allows M:* by default for reads)."""
    policy = DeviceAccessPolicy(patterns=["G:*"], mode="deny")
    srv = SupervisedServer(real_backend, port=0, policies=[policy])
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def device_deny_server(real_backend):
    """SupervisedServer denying Z:* devices."""
    policy = DeviceAccessPolicy(patterns=["Z:*"], mode="deny")
    srv = SupervisedServer(real_backend, port=0, policies=[policy])
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def rate_limited_server(real_backend):
    """SupervisedServer with a tight rate limit (3 requests per 60s window)."""
    policy = RateLimitPolicy(max_requests=3, window_seconds=60.0)
    srv = SupervisedServer(real_backend, port=0, policies=[policy])
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture(scope="class")
def direct_grpc():
    """Direct GRPCBackend → real DPM gRPC server."""
    if not grpc_server_available():
        pytest.skip("gRPC server not available")
    backend = GRPCBackend()  # localhost:23456
    yield backend
    backend.close()


# =============================================================================
# Comparison Helper
# =============================================================================


def assert_readings_equivalent(direct, proxied, *, value_tol=0.5):
    """Assert two readings from different paths are structurally equivalent.

    Allows for live-data drift (float tolerance) and timestamp differences.
    """
    assert direct.ok == proxied.ok, (
        f"ok mismatch: direct={direct.ok} ({direct.message}), proxied={proxied.ok} ({proxied.message})"
    )
    assert direct.value_type == proxied.value_type, (
        f"value_type mismatch: direct={direct.value_type}, proxied={proxied.value_type}"
    )

    # Error sign must agree (both >=0 or both <0)
    d_sign = direct.error_code >= 0
    p_sign = proxied.error_code >= 0
    assert d_sign == p_sign, f"error_code sign mismatch: direct={direct.error_code}, proxied={proxied.error_code}"

    # Facility code must match when both are errors
    if direct.error_code < 0 and proxied.error_code < 0:
        assert direct.facility_code == proxied.facility_code, (
            f"facility_code mismatch: direct={direct.facility_code}, proxied={proxied.facility_code}"
        )

    # Timestamp: both present or both absent
    assert (direct.timestamp is not None) == (proxied.timestamp is not None), (
        f"timestamp presence mismatch: direct={direct.timestamp}, proxied={proxied.timestamp}"
    )

    if not direct.ok:
        return  # no value to compare for errors

    _assert_values_equivalent(direct.value, proxied.value, value_tol)


def _assert_values_equivalent(direct_val, proxied_val, tol):
    """Recursively compare values with tolerance for live-data drift."""
    if isinstance(direct_val, float) and isinstance(proxied_val, float):
        assert direct_val == pytest.approx(proxied_val, abs=tol), (
            f"float mismatch: direct={direct_val}, proxied={proxied_val}"
        )
    elif isinstance(direct_val, str) and isinstance(proxied_val, str):
        assert direct_val == proxied_val, f"string mismatch: direct={direct_val!r}, proxied={proxied_val!r}"
    elif isinstance(direct_val, bytes) and isinstance(proxied_val, bytes):
        # Both paths go through the same gRPC proto so raw bytes must match
        assert len(direct_val) == len(proxied_val), (
            f"raw bytes length mismatch: direct={len(direct_val)}, proxied={len(proxied_val)}"
        )
    elif isinstance(direct_val, dict) and isinstance(proxied_val, dict):
        # Both paths return dicts but status dicts may differ structurally:
        # gRPC server returns digital status bit labels, HTTP returns basic status booleans
        assert direct_val, "direct dict is empty"
        assert proxied_val, "proxied dict is empty"
    elif hasattr(direct_val, "__len__") and hasattr(proxied_val, "__len__"):
        # Array length must match; element values drift for live beam data
        assert len(direct_val) == len(proxied_val), (
            f"array length mismatch: direct={len(direct_val)}, proxied={len(proxied_val)}"
        )
    else:
        assert type(direct_val) is type(proxied_val), (
            f"type mismatch: direct={type(direct_val).__name__}, proxied={type(proxied_val).__name__}"
        )


# =============================================================================
# Read Tests
# =============================================================================


@requires_dpm_http
class TestSupervisedProxyRead:
    """Reads through supervised gRPC proxy return valid results."""

    def test_read_scalar(self, proxy_client):
        """Read M:OUTTMP through proxy, verify float value."""
        reading = proxy_client.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert reading.ok, f"Read failed: {reading.message}"
        assert isinstance(reading.value, float)

    def test_read_batch(self, proxy_client):
        """Batch read through proxy returns results for all devices."""
        readings = proxy_client.get_many([SCALAR_DEVICE, SCALAR_DEVICE_2], timeout=TIMEOUT_BATCH)
        assert len(readings) == 2
        for r in readings:
            assert r.ok, f"Read failed: {r.message}"
            assert isinstance(r.value, float)

    def test_read_array(self, proxy_client):
        """Read array device through proxy."""
        reading = proxy_client.get(ARRAY_DEVICE, timeout=TIMEOUT_READ)
        assert reading.ok, f"Read failed: {reading.message}"
        assert hasattr(reading.value, "__len__")
        assert len(reading.value) > 0

    def test_read_nonexistent_error(self, proxy_client):
        """Nonexistent device returns error Reading (via status oneof)."""
        reading = proxy_client.get(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ * 2.0)
        assert not reading.ok
        assert reading.error_code != 0
        assert reading.drf == NONEXISTENT_DEVICE

    def test_read_partial_failure(self, proxy_client):
        """Batch with valid + invalid device returns partial results."""
        readings = proxy_client.get_many([SCALAR_DEVICE, NONEXISTENT_DEVICE], timeout=TIMEOUT_BATCH * 2.0)
        assert len(readings) == 2
        by_drf = {r.drf: r for r in readings}
        assert by_drf[SCALAR_DEVICE].ok, f"Valid device failed: {by_drf[SCALAR_DEVICE].message}"
        assert not by_drf[NONEXISTENT_DEVICE].ok


# =============================================================================
# Streaming Tests (exercise the server's subscribe/queue bridge path)
# =============================================================================


@requires_dpm_http
@pytest.mark.streaming
class TestSupervisedProxyStreaming:
    """Streaming reads through supervised proxy exercise the async queue bridge."""

    def test_get_periodic_snapshot(self, proxy_client):
        """get() on a periodic DRF uses the streaming path but returns one result.

        This exercises the server's subscribe → queue → cancel cleanup path,
        which is different from the oneshot get_many path.
        """
        reading = proxy_client.get(PERIODIC_DEVICE, timeout=TIMEOUT_READ)
        assert reading.ok, f"Periodic snapshot failed: {reading.message}"
        assert isinstance(reading.value, float)

    def test_subscribe_periodic(self, supervised_server):
        """subscribe() through proxy receives multiple streaming updates."""
        client = GRPCBackend(host="localhost", port=supervised_server.port, timeout=TIMEOUT_READ)
        try:
            handle = client.subscribe([PERIODIC_DEVICE])
            try:
                count = 0
                for reading, _h in handle.readings(timeout=TIMEOUT_STREAM_EVENT * 3):
                    assert reading.ok, f"Streaming read failed: {reading.message}"
                    assert isinstance(reading.value, float)
                    count += 1
                    if count >= 3:
                        break
                assert count >= 3, f"Expected >=3 streaming readings, got {count}"
            finally:
                handle.stop()
        finally:
            client.close()


# =============================================================================
# Write Tests - raw gRPC stub helpers
# =============================================================================
#
# GRPCBackend.write() requires client-side JWTAuth, but the supervised proxy
# delegates auth to the underlying backend. All write tests use raw stubs.


def _ensure_oneshot(drf: str) -> str:
    """Append @I if no event so supervised proxy treats as one-shot."""
    if "@" not in drf:
        return drf + "@I"
    return drf


def _stub_read_scalar(stub, drf, *, timeout=TIMEOUT_READ):
    """Read a scalar value via raw gRPC stub. Returns float."""
    req = DAQ_pb2.ReadingList()
    req.drf.append(_ensure_oneshot(drf))
    replies = list(stub.Read(req, timeout=timeout))
    assert len(replies) >= 1, f"No reply for {drf}"
    rd = replies[0].readings.reading[0]
    return rd.data.scalar


def _stub_read_raw(stub, drf, *, timeout=TIMEOUT_READ):
    """Read raw bytes via raw gRPC stub. Returns bytes."""
    req = DAQ_pb2.ReadingList()
    req.drf.append(_ensure_oneshot(drf))
    replies = list(stub.Read(req, timeout=timeout))
    assert len(replies) >= 1, f"No reply for {drf}"
    rd = replies[0].readings.reading[0]
    return rd.data.raw


def _stub_read_status(stub, drf, *, timeout=TIMEOUT_READ):
    """Read basic status via raw gRPC stub. Returns dict[str, str]."""
    req = DAQ_pb2.ReadingList()
    req.drf.append(_ensure_oneshot(drf))
    replies = list(stub.Read(req, timeout=timeout))
    assert len(replies) >= 1, f"No reply for {drf}"
    rd = replies[0].readings.reading[0]
    return dict(rd.data.basicStatus.value)


def _stub_write(stub, drf, value, *, timeout=TIMEOUT_READ):
    """Write a scalar value via raw gRPC stub. Returns status_code."""
    req = DAQ_pb2.SettingList()
    setting = DAQ_pb2.Setting()
    setting.device = drf
    setting.value.scalar = float(value)
    req.setting.append(setting)
    reply = stub.Set(req, timeout=timeout)
    assert len(reply.status) == 1, f"Expected 1 status, got {len(reply.status)}"
    return reply.status[0].status_code


_ALLOW_ALL_WRITES = DeviceAccessPolicy(patterns=["*"], action="set", mode="allow")


@pytest.fixture
def write_proxy(real_write_backend):
    """SupervisedServer wrapping authenticated backend + raw gRPC stub.

    Yields (stub, server) - stub is a DAQStub connected to the proxy.
    """
    srv = SupervisedServer(real_write_backend, port=0, policies=[_ALLOW_ALL_WRITES])
    srv.start()
    ch = grpc.insecure_channel(f"localhost:{srv.port}")
    stub = DAQ_pb2_grpc.DAQStub(ch)
    yield stub
    ch.close()
    srv.stop()


# =============================================================================
# Write Test Class
# =============================================================================


@requires_dpm_http
@requires_write_enabled
@requires_kerberos
@pytest.mark.write
@pytest.mark.kerberos
class TestSupervisedProxyWrite:
    """Writes through supervised proxy to real devices (mirrors TestBackendWrite)."""

    def test_write_scalar(self, write_proxy):
        """Write float, verify readback, restore original."""
        stub = write_proxy
        read_drf = strip_event(SCALAR_SETPOINT)
        original = _stub_read_scalar(stub, read_drf)

        try:
            new_val = original + 0.1
            status = _stub_write(stub, SCALAR_SETPOINT, new_val)
            assert status == 0, f"Write failed: status_code={status}"

            time.sleep(1.0)
            readback = _stub_read_scalar(stub, read_drf)
            assert readback == pytest.approx(new_val, abs=0.01)
        finally:
            _stub_write(stub, SCALAR_SETPOINT, original)
            time.sleep(1.0)

        restored = _stub_read_scalar(stub, read_drf)
        assert restored == pytest.approx(original, abs=0.01)

    def test_write_changes_raw(self, write_proxy):
        """Write scaled value, verify raw bytes change, restore."""
        stub = write_proxy
        read_drf = strip_event(SCALAR_SETPOINT)
        original_scaled = _stub_read_scalar(stub, read_drf)
        original_raw = _stub_read_raw(stub, SCALAR_SETPOINT_RAW)
        assert isinstance(original_raw, bytes) and len(original_raw) > 0

        try:
            new_val = original_scaled + 1.0
            status = _stub_write(stub, SCALAR_SETPOINT, new_val)
            assert status == 0

            time.sleep(1.0)
            new_raw = _stub_read_raw(stub, SCALAR_SETPOINT_RAW)
            assert new_raw != original_raw, "Raw bytes unchanged after write"
        finally:
            _stub_write(stub, SCALAR_SETPOINT, original_scaled)
            time.sleep(1.0)

        restored_raw = _stub_read_raw(stub, SCALAR_SETPOINT_RAW)
        assert restored_raw == original_raw

    @pytest.mark.parametrize(
        "cmd_true,cmd_false,field",
        CONTROL_PAIRS,
        ids=[f"{f}" for _, _, f in CONTROL_PAIRS],
    )
    def test_control_pair(self, write_proxy, cmd_true, cmd_false, field):
        """Toggle control pair (ON/OFF, POSITIVE/NEGATIVE, RAMP/DC), verify status."""
        stub = write_proxy
        initial_status = _stub_read_status(stub, STATUS_CONTROL_DEVICE)
        initial = initial_status.get(field)

        try:
            # Set TRUE
            status = _stub_write(stub, STATUS_CONTROL_DEVICE, cmd_true)
            assert status == 0, f"Control {cmd_true} failed"
            time.sleep(1.0)
            st = _stub_read_status(stub, STATUS_CONTROL_DEVICE)
            assert st.get(field) == "True", f"Expected {field}=True after {cmd_true}, got {st.get(field)}"

            # Set FALSE
            status = _stub_write(stub, STATUS_CONTROL_DEVICE, cmd_false)
            assert status == 0, f"Control {cmd_false} failed"
            time.sleep(1.0)
            st = _stub_read_status(stub, STATUS_CONTROL_DEVICE)
            assert st.get(field) == "False", f"Expected {field}=False after {cmd_false}, got {st.get(field)}"
        finally:
            restore = cmd_true if initial == "True" else cmd_false
            _stub_write(stub, STATUS_CONTROL_DEVICE, restore)

    def test_control_reset(self, write_proxy):
        """RESET command succeeds and status is readable afterwards."""
        stub = write_proxy
        status = _stub_write(stub, STATUS_CONTROL_DEVICE, CONTROL_RESET)
        assert status == 0, f"RESET failed: status_code={status}"

        time.sleep(1.0)
        st = _stub_read_status(stub, STATUS_CONTROL_DEVICE)
        assert "on" in st, f"Status missing 'on' key after RESET: {st}"


# =============================================================================
# Read-Only Policy Tests
# =============================================================================


@requires_dpm_http
class TestSupervisedReadOnlyPolicy:
    """ReadOnlyPolicy allows reads but blocks writes."""

    def test_readonly_allows_read(self, readonly_server):
        """Reads succeed through readonly proxy."""
        client = GRPCBackend(host="localhost", port=readonly_server.port, timeout=TIMEOUT_READ)
        try:
            reading = client.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)
            assert reading.ok, f"Read failed: {reading.message}"
        finally:
            client.close()

    def test_readonly_blocks_write(self, readonly_server):
        """Write attempt returns PERMISSION_DENIED gRPC error."""
        # Use raw gRPC stub - GRPCBackend.write() requires client-side JWTAuth
        with grpc.insecure_channel(f"localhost:{readonly_server.port}") as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = SCALAR_SETPOINT
            setting.value.scalar = 42.0
            request.setting.append(setting)

            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Set(request, timeout=TIMEOUT_READ)
            assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED


# =============================================================================
# Device Access Policy Tests
# =============================================================================


@requires_dpm_http
class TestSupervisedDeviceAccessPolicy:
    """DeviceAccessPolicy restricts which devices are accessible."""

    def test_device_allow_list(self, device_limited_server):
        """M:* devices pass through; G:AMANDA blocked by deny policy."""
        client = GRPCBackend(host="localhost", port=device_limited_server.port, timeout=TIMEOUT_READ)
        try:
            # M:OUTTMP should work
            reading = client.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)
            assert reading.ok, f"Allowed device read failed: {reading.message}"

            # G:AMANDA should be blocked - server returns PERMISSION_DENIED
            with pytest.raises(ReadError) as exc_info:
                client.get(SCALAR_DEVICE_2, timeout=TIMEOUT_READ)
            assert_permission_denied(exc_info)
        finally:
            client.close()

    def test_device_deny_list(self, device_deny_server):
        """Z:* devices denied; M:OUTTMP allowed."""
        client = GRPCBackend(host="localhost", port=device_deny_server.port, timeout=TIMEOUT_READ)
        try:
            # M:OUTTMP should work
            reading = client.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)
            assert reading.ok, f"Allowed device read failed: {reading.message}"

            # Z:NOTFND should be blocked by policy (not just nonexistent)
            with pytest.raises(ReadError) as exc_info:
                client.get(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
            assert_permission_denied(exc_info)
        finally:
            client.close()


# =============================================================================
# Rate Limit Policy Tests
# =============================================================================


@requires_dpm_http
class TestSupervisedRateLimitPolicy:
    """RateLimitPolicy enforces request rate limits."""

    def test_rate_limit_allows_within_limit(self, rate_limited_server):
        """Several reads within the limit succeed."""
        client = GRPCBackend(host="localhost", port=rate_limited_server.port, timeout=TIMEOUT_READ)
        try:
            for _ in range(3):
                reading = client.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)
                assert reading.ok, f"Read within limit failed: {reading.message}"
        finally:
            client.close()

    def test_rate_limit_blocks_over_limit(self, rate_limited_server):
        """Exceeding rate limit returns PERMISSION_DENIED."""
        client = GRPCBackend(host="localhost", port=rate_limited_server.port, timeout=TIMEOUT_READ)
        try:
            # Use up the quota
            for _ in range(3):
                client.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)

            # Next request should be blocked
            with pytest.raises(ReadError) as exc_info:
                client.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)
            assert_permission_denied(exc_info)
        finally:
            client.close()


# =============================================================================
# Proxy vs Direct gRPC Comparison Tests
# =============================================================================

# DRFs to compare: (drf, description)
_COMPARISON_DRFS = [
    (SCALAR_DEVICE, "scalar"),
    (SCALAR_DEVICE_2, "scalar 2"),
    (ARRAY_DEVICE, "array"),
    (SCALAR_ELEMENT, "array element"),
    (RAW_DEVICE, "raw"),
    (DESCRIPTION_DEVICE, "text/description"),
    (STATUS_DEVICE, "basic status"),
    (ANALOG_ALARM_DEVICE, "analog alarm"),
    (DIGITAL_ALARM_DEVICE, "digital alarm"),
]


@requires_dpm_http
@requires_grpc
class TestSupervisedVsDirectGRPC:
    """Compare proxy path vs direct gRPC for identical device reads."""

    @pytest.mark.parametrize(
        "drf,desc",
        [(drf, desc) for drf, desc in _COMPARISON_DRFS],
        ids=[desc for _, desc in _COMPARISON_DRFS],
    )
    def test_read_value_match(self, direct_grpc, proxy_client, drf, desc):
        """Same DRF through both paths produces structurally equivalent readings."""
        direct_reading = direct_grpc.get(drf, timeout=TIMEOUT_READ)
        proxied_reading = proxy_client.get(drf, timeout=TIMEOUT_READ)
        assert_readings_equivalent(direct_reading, proxied_reading)

    def test_batch_read_match(self, direct_grpc, proxy_client):
        """Batch read through both paths produces pairwise-equivalent results."""
        drfs = [SCALAR_DEVICE, SCALAR_DEVICE_2]
        direct_readings = direct_grpc.get_many(drfs, timeout=TIMEOUT_BATCH)
        proxied_readings = proxy_client.get_many(drfs, timeout=TIMEOUT_BATCH)
        assert len(direct_readings) == len(proxied_readings)
        direct_by_drf = {r.drf: r for r in direct_readings}
        proxied_by_drf = {r.drf: r for r in proxied_readings}
        for drf in drfs:
            assert drf in direct_by_drf, f"Direct missing {drf}"
            assert drf in proxied_by_drf, f"Proxied missing {drf}"
            assert_readings_equivalent(direct_by_drf[drf], proxied_by_drf[drf])

    def test_error_match(self, direct_grpc, proxy_client):
        """Nonexistent device produces equivalent error through both paths."""
        direct_reading = direct_grpc.get(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        proxied_reading = proxy_client.get(NONEXISTENT_DEVICE, timeout=TIMEOUT_READ)
        assert not direct_reading.ok
        assert not proxied_reading.ok
        assert_readings_equivalent(direct_reading, proxied_reading)
