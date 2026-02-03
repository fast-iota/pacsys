"""
Configuration for real/integration tests.

These tests require network access to actual servers:
- DPM: acsys-proxy.fnal.gov:6802
- gRPC: localhost:23456 (tunnel to dce08.fnal.gov:50051)

Run these tests explicitly:
    pytest tests/real/ -v -s

Or run specific test file:
    pytest tests/real/test_dpm_http_backend.py -v -s
    pytest tests/real/test_grpc_backend.py -v -s
"""

import functools

import pytest

from pacsys.drf_utils import get_device_name

# Import server availability checks and markers from shared module
from .devices import (
    ALLOWED_WRITE_DEVICES,
    dpm_server_available,
    grpc_server_available,
    acl_server_available,
    dmq_server_available,
    kerberos_available,
    requires_dpm_http,
    requires_dpm_acnet,
    requires_grpc,
    requires_acl,
    requires_dmq,
)

# Re-export for backward compatibility
__all__ = [
    "dpm_server_available",
    "grpc_server_available",
    "acl_server_available",
    "dmq_server_available",
    "requires_dpm_http",
    "requires_dpm_acnet",
    "requires_grpc",
    "requires_acl",
    "requires_dmq",
]


# =============================================================================
# Pytest Markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "real: marks tests that require real server connections",
    )
    config.addinivalue_line(
        "markers",
        "dpm: marks tests that require DPM server (acsys-proxy)",
    )
    config.addinivalue_line(
        "markers",
        "grpc: marks tests that require gRPC server",
    )
    config.addinivalue_line(
        "markers",
        "acl: marks tests that require ACL CGI endpoint",
    )
    config.addinivalue_line(
        "markers",
        "dmq: marks tests that require DMQ/RabbitMQ broker",
    )
    config.addinivalue_line(
        "markers",
        "streaming: marks streaming tests (may be slow)",
    )
    config.addinivalue_line(
        "markers",
        "kerberos: marks tests that require valid Kerberos ticket",
    )
    config.addinivalue_line(
        "markers",
        "write: marks tests that write to real devices (enable with PACSYS_TEST_WRITE=1)",
    )


def pytest_collection_modifyitems(config, items):
    """Add 'real' marker to all tests in this directory and apply skip conditions."""
    for item in items:
        # Add 'real' marker to all tests in tests/real/
        if "tests/real" in str(item.fspath) or "tests\\real" in str(item.fspath):
            item.add_marker(pytest.mark.real)


# =============================================================================
# Backend Fixtures
# =============================================================================


@pytest.fixture
def dpm_http_backend():
    """Create a DPMHTTPBackend for testing."""
    from pacsys.backends.dpm_http import DPMHTTPBackend

    backend = DPMHTTPBackend()
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
    """Create an ACLBackend for testing."""
    from pacsys.backends.acl import ACLBackend

    backend = ACLBackend()
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


@pytest.fixture(params=["dmq", "dpm_http", "grpc"])
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

        backend = DPMHTTPBackend()
    elif backend_type == "grpc":
        if not grpc_server_available():
            pytest.skip("gRPC server not available")
        from pacsys.backends.grpc_backend import GRPCBackend

        backend = GRPCBackend()
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

        backend = DPMHTTPBackend(auth=KerberosAuth(), role="testing")
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
def dpm_http_connection():
    """Create a DPMConnection for low-level testing."""
    from pacsys.acnet import DPMConnection

    conn = DPMConnection()
    conn.connect()
    yield conn
    conn.close()


@pytest.fixture
def dpm_acnet():
    """Create a DPMAcnet for low-level testing."""
    from pacsys.acnet import DPMAcnet

    conn = DPMAcnet()
    conn.connect()
    yield conn
    conn.close()


# =============================================================================
# Global Backend Reset
# =============================================================================


@pytest.fixture(autouse=True)
def reset_global_backend():
    """Reset global backend before and after each test."""
    import pacsys

    pacsys.shutdown()
    yield
    pacsys.shutdown()


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
