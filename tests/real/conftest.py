"""
Configuration for real/integration tests.

These tests require network access to actual servers:
- DPM: localhost:33232
- gRPC: localhost:23456

Gated by environment variables:
    PACSYS_TEST_REAL=1   - enable real tests (required)
    PACSYS_TEST_WRITE=1  - enable write tests (optional, requires Kerberos)

Run:
    PACSYS_TEST_REAL=1 pytest tests/real/ -v -s
"""

import functools
import os

import pytest
import pytest_asyncio

from pacsys.drf_utils import get_device_name

# Import server availability checks and markers from shared module
from .devices import (
    ACL_TEST_URL,
    ACNET_TCP_TEST_HOST,
    ACNET_TCP_TEST_PORT,
    ALLOWED_WRITE_DEVICES,
    DPM_TEST_HOST,
    DPM_TEST_PORT,
    acnet_tcp_server_available,
    dpm_server_available,
    grpc_server_available,
    acl_server_available,
    dmq_server_available,
    kerberos_available,
    requires_acnet_tcp,
    requires_dpm_http,
    requires_dpm_acnet,
    requires_grpc,
    requires_acl,
    requires_dmq,
    requires_ssh,
    SSH_JUMP_HOST,
    SSH_DEST_HOST,
)

# Re-export for backward compatibility
__all__ = [
    "acnet_tcp_server_available",
    "dpm_server_available",
    "grpc_server_available",
    "ACL_TEST_URL",
    "acl_server_available",
    "dmq_server_available",
    "requires_acnet_tcp",
    "requires_dpm_http",
    "requires_dpm_acnet",
    "requires_grpc",
    "requires_acl",
    "requires_dmq",
    "requires_ssh",
    "SSH_JUMP_HOST",
    "SSH_DEST_HOST",
]


def pytest_collection_modifyitems(config, items):
    """Skip all real tests unless PACSYS_TEST_REAL=1 is set.

    This catches direct file invocations (e.g. pytest tests/real/some_file.py)
    that bypass pytest_ignore_collect in the parent conftest.
    """
    if os.environ.get("PACSYS_TEST_REAL"):
        return
    skip = pytest.mark.skip(reason="Set PACSYS_TEST_REAL=1 to run real tests")
    for item in items:
        item.add_marker(skip)


# =============================================================================
# Backend Fixtures
# =============================================================================


@pytest.fixture
def dpm_http_backend():
    """Create a DPMHTTPBackend for testing (per-test)."""
    from pacsys.backends.dpm_http import DPMHTTPBackend

    backend = DPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT)
    yield backend
    backend.close()


@pytest.fixture(scope="class")
def dpm_http_backend_cls():
    """Class-scoped DPMHTTPBackend shared across tests in a class."""
    from pacsys.backends.dpm_http import DPMHTTPBackend

    backend = DPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT)
    yield backend
    backend.close()


@pytest.fixture
def grpc_backend():
    """Create a GRPCBackend for testing."""
    from pacsys.backends.grpc_backend import GRPCBackend

    backend = GRPCBackend()
    yield backend
    backend.close()


@pytest.fixture
def acl_backend():
    """Create an ACLBackend for testing (uses local proxy)."""
    from pacsys.backends.acl import ACLBackend

    backend = ACLBackend(base_url=ACL_TEST_URL, verify_ssl=False)
    yield backend
    backend.close()


@pytest.fixture
def dmq_backend():
    """Create a DMQBackend for testing."""
    from pacsys.auth import KerberosAuth
    from pacsys.backends.dmq import DMQBackend

    auth = KerberosAuth()
    backend = DMQBackend(host="localhost", port=5672, auth=auth)
    yield backend
    backend.close()


@pytest_asyncio.fixture
async def async_dpm_http_backend():
    """Create an AsyncDPMHTTPBackend for testing."""
    from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

    backend = AsyncDPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT)
    yield backend
    await backend.close()


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def async_dpm_http_backend_cls():
    """Class-scoped AsyncDPMHTTPBackend shared across tests in a class."""
    from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

    backend = AsyncDPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT)
    yield backend
    await backend.close()


@pytest_asyncio.fixture
async def async_grpc_backend():
    """Create an AsyncGRPCBackend for testing."""
    from pacsys.aio._grpc import AsyncGRPCBackend

    backend = AsyncGRPCBackend()
    yield backend
    await backend.close()


@pytest_asyncio.fixture(params=["dpm_http", "grpc"])
async def async_read_backend(request):
    """Parametrized fixture that yields each async read-capable backend."""
    backend_type = request.param

    if backend_type == "dpm_http":
        if not dpm_server_available():
            pytest.skip("DPM server not available")
        from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

        backend = AsyncDPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT)
    elif backend_type == "grpc":
        if not grpc_server_available():
            pytest.skip("gRPC server not available")
        from pacsys.aio._grpc import AsyncGRPCBackend

        backend = AsyncGRPCBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    yield backend
    await backend.close()


@pytest_asyncio.fixture(params=["dpm_http", "grpc"], scope="class", loop_scope="class")
async def async_read_backend_cls(request):
    """Parametrized fixture that yields each async read-capable backend."""
    backend_type = request.param

    if backend_type == "dpm_http":
        if not dpm_server_available():
            pytest.skip("DPM server not available")
        from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

        backend = AsyncDPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT)
    elif backend_type == "grpc":
        if not grpc_server_available():
            pytest.skip("gRPC server not available")
        from pacsys.aio._grpc import AsyncGRPCBackend

        backend = AsyncGRPCBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    yield backend
    await backend.close()


@pytest_asyncio.fixture(params=["dpm_http"], scope="class", loop_scope="class")
async def async_write_backend_cls(request):
    """Class-scoped parametrized async write backend."""
    if not kerberos_available():
        pytest.skip("Kerberos credentials not available")
    if request.param == "dpm_http":
        if not dpm_server_available():
            pytest.skip("DPM server not available")
        from pacsys.auth import KerberosAuth
        from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

        backend = AsyncDPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT, auth=KerberosAuth(), role="testing")
    else:
        raise ValueError(f"Unknown backend type: {request.param}")

    yield backend
    await backend.close()


@pytest_asyncio.fixture(params=["dpm_http"])
async def async_write_backend(request):
    """Parametrized fixture that yields each async write-capable backend."""
    if not kerberos_available():
        pytest.skip("Kerberos credentials not available")
    if request.param == "dpm_http":
        if not dpm_server_available():
            pytest.skip("DPM server not available")
        from pacsys.auth import KerberosAuth
        from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

        backend = AsyncDPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT, auth=KerberosAuth(), role="testing")
    else:
        raise ValueError(f"Unknown backend type: {request.param}")

    yield backend
    await backend.close()


@pytest.fixture(params=["dmq", "dpm_http", "grpc", "acl"])
def read_backend(request):
    """Parametrized fixture that yields each read-capable backend.

    Tests using this fixture will run against all available backends.
    Skips backends whose servers are not available.
    """
    backend_type = request.param

    if backend_type == "dmq":
        if not dmq_server_available():
            pytest.skip("DMQ server not available")
        from pacsys.auth import KerberosAuth
        from pacsys.backends.dmq import DMQBackend

        backend = DMQBackend(host="localhost", port=5672, auth=KerberosAuth())
    elif backend_type == "dpm_http":
        if not dpm_server_available():
            pytest.skip("DPM server not available")
        from pacsys.backends.dpm_http import DPMHTTPBackend

        backend = DPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT)
    elif backend_type == "grpc":
        if not grpc_server_available():
            pytest.skip("gRPC server not available")
        from pacsys.backends.grpc_backend import GRPCBackend

        backend = GRPCBackend()
    elif backend_type == "acl":
        if not acl_server_available():
            pytest.skip("ACL server not available at localhost:10443")
        from pacsys.backends.acl import ACLBackend

        backend = ACLBackend(base_url=ACL_TEST_URL, verify_ssl=False)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    yield backend
    backend.close()


@pytest.fixture(params=["dmq", "dpm_http", "grpc", "acl"], scope="class")
def read_backend_cls(request):
    """Class-level version of read_backend fixture."""
    backend_type = request.param

    if backend_type == "dmq":
        if not dmq_server_available():
            pytest.skip("DMQ server not available")
        from pacsys.auth import KerberosAuth
        from pacsys.backends.dmq import DMQBackend

        backend = DMQBackend(host="localhost", port=5672, auth=KerberosAuth())
    elif backend_type == "dpm_http":
        if not dpm_server_available():
            pytest.skip("DPM server not available")
        from pacsys.backends.dpm_http import DPMHTTPBackend

        backend = DPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT)
    elif backend_type == "grpc":
        if not grpc_server_available():
            pytest.skip("gRPC server not available")
        from pacsys.backends.grpc_backend import GRPCBackend

        backend = GRPCBackend()
    elif backend_type == "acl":
        if not acl_server_available():
            pytest.skip("ACL server not available at localhost:10443")
        from pacsys.backends.acl import ACLBackend

        backend = ACLBackend(base_url=ACL_TEST_URL, verify_ssl=False)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    yield backend
    backend.close()


@pytest.fixture(params=["dpm_http", "dmq"])
def write_backend(request):
    """Parametrized fixture that yields each write-capable backend.

    Tests using this fixture will run against DPM HTTP and DMQ backends.
    Skips if server not available or Kerberos credentials missing.
    """
    backend_type = request.param

    if not kerberos_available():
        pytest.skip("Kerberos credentials not available")

    if backend_type == "dpm_http":
        if not dpm_server_available():
            pytest.skip("DPM server not available")
        from pacsys.auth import KerberosAuth
        from pacsys.backends.dpm_http import DPMHTTPBackend

        backend = DPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT, auth=KerberosAuth(), role="testing")
    elif backend_type == "dmq":
        if not dmq_server_available():
            pytest.skip("DMQ server not available")
        from pacsys.auth import KerberosAuth
        from pacsys.backends.dmq import DMQBackend

        backend = DMQBackend(host="localhost", auth=KerberosAuth())
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    yield backend
    backend.close()


@pytest.fixture(params=["dpm_http", "dmq"], scope="class")
def write_backend_cls(request):
    """Class-level version of write_backend fixture."""
    backend_type = request.param

    if not kerberos_available():
        pytest.skip("Kerberos credentials not available")

    if backend_type == "dpm_http":
        if not dpm_server_available():
            pytest.skip("DPM server not available")
        from pacsys.auth import KerberosAuth
        from pacsys.backends.dpm_http import DPMHTTPBackend

        backend = DPMHTTPBackend(host=DPM_TEST_HOST, port=DPM_TEST_PORT, auth=KerberosAuth(), role="testing")
    elif backend_type == "dmq":
        if not dmq_server_available():
            pytest.skip("DMQ server not available")
        from pacsys.auth import KerberosAuth
        from pacsys.backends.dmq import DMQBackend

        backend = DMQBackend(host="localhost", auth=KerberosAuth())
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    yield backend
    backend.close()


# =============================================================================
# Low-Level Connection Fixtures
# =============================================================================


@pytest.fixture
def acnet_tcp_connection():
    """Create an AcnetConnectionTCP via the localhost tunnel."""
    from pacsys.acnet import AcnetConnectionTCP

    conn = AcnetConnectionTCP(ACNET_TCP_TEST_HOST, ACNET_TCP_TEST_PORT)
    conn.connect()
    yield conn
    conn.close()


@pytest.fixture
def dpm_acnet():
    """Create a DPMAcnet for low-level testing via localhost tunnel."""
    from pacsys.acnet import DPMAcnet

    conn = DPMAcnet(host=ACNET_TCP_TEST_HOST, port=ACNET_TCP_TEST_PORT)
    conn.connect()
    yield conn
    conn.close()


# =============================================================================
# Global Backend Reset
# =============================================================================


@pytest.fixture(autouse=True)
def reset_global_backend():
    """Reset global backend before and after each test.

    Configures the global backend to use the test tunnel so that
    pacsys.read() / pacsys.get() etc. connect to localhost, not
    acsys-proxy.fnal.gov.
    """
    import pacsys

    pacsys.shutdown()
    pacsys.configure(dpm_host=DPM_TEST_HOST, dpm_port=DPM_TEST_PORT)
    yield
    pacsys.shutdown()


@pytest.fixture(autouse=True)
def reset_async_global_backend():
    """Reset async global backend and config before and after each test.

    Configures the async global backend to use the test tunnel.
    """
    import pacsys.aio as aio

    _reset_aio_state(aio)
    aio._config_host = DPM_TEST_HOST
    aio._config_port = DPM_TEST_PORT
    yield
    _reset_aio_state(aio)


def _reset_aio_state(aio):
    """Reset all async module-level state (backend + config)."""
    if aio._global_async_backend is not None:
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(aio._global_async_backend.close())
            else:
                loop.run_until_complete(aio._global_async_backend.close())
        except Exception:
            pass  # best-effort cleanup from sync context
    aio._global_async_backend = None
    aio._async_backend_initialized = False
    aio._config_backend = None
    aio._config_auth = None
    aio._config_role = None
    aio._config_host = None
    aio._config_port = None
    aio._config_pool_size = None
    aio._config_timeout = None


# =============================================================================
# Write Safety Guard
# =============================================================================


def _assert_allowed_write(drf: str) -> None:
    """Raise if drf targets a device not in ALLOWED_WRITE_DEVICES."""
    device = get_device_name(drf).upper()
    if device not in ALLOWED_WRITE_DEVICES:
        raise RuntimeError(
            f"BLOCKED: write to '{device}' (from '{drf}') is not in "
            f"ALLOWED_WRITE_DEVICES. Add it to tests/real/devices.py if intentional."
        )


def _guard_write(original):
    """Wrap a backend .write() to enforce the allowlist."""

    @functools.wraps(original)
    def wrapper(self, drf, *args, **kwargs):
        _assert_allowed_write(drf)
        return original(self, drf, *args, **kwargs)

    return wrapper


def _guard_write_many(original):
    """Wrap a backend .write_many() to enforce the allowlist."""

    @functools.wraps(original)
    def wrapper(self, settings, *args, **kwargs):
        for drf, _value in settings:
            _assert_allowed_write(drf)
        return original(self, settings, *args, **kwargs)

    return wrapper


@pytest.fixture(autouse=True)
def _enforce_write_allowlist(monkeypatch):
    """Patch all backend classes to reject writes to unlisted devices.

    Patches at the class level so it covers backends created via fixtures
    AND manually-created backends (e.g., _create_dmq_backend()).
    """
    from pacsys.backends.dpm_http import DPMHTTPBackend
    from pacsys.backends.dmq import DMQBackend

    for cls in (DPMHTTPBackend, DMQBackend):
        if hasattr(cls, "write"):
            monkeypatch.setattr(cls, "write", _guard_write(cls.write))
        if hasattr(cls, "write_many"):
            monkeypatch.setattr(cls, "write_many", _guard_write_many(cls.write_many))

    try:
        from pacsys.backends.grpc_backend import GRPCBackend

        if hasattr(GRPCBackend, "write"):
            monkeypatch.setattr(GRPCBackend, "write", _guard_write(GRPCBackend.write))
        if hasattr(GRPCBackend, "write_many"):
            monkeypatch.setattr(GRPCBackend, "write_many", _guard_write_many(GRPCBackend.write_many))
    except ImportError:
        pass  # grpcio not installed

    # Async backends
    from pacsys.aio._dpm_http import AsyncDPMHTTPBackend

    monkeypatch.setattr(AsyncDPMHTTPBackend, "write", _guard_write(AsyncDPMHTTPBackend.write))
    monkeypatch.setattr(AsyncDPMHTTPBackend, "write_many", _guard_write_many(AsyncDPMHTTPBackend.write_many))

    try:
        from pacsys.aio._grpc import AsyncGRPCBackend

        monkeypatch.setattr(AsyncGRPCBackend, "write", _guard_write(AsyncGRPCBackend.write))
        monkeypatch.setattr(AsyncGRPCBackend, "write_many", _guard_write_many(AsyncGRPCBackend.write_many))
    except ImportError:
        pass
