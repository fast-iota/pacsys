"""
Tests for the Simple API (Phase 6).

Tests cover:
- read() with DRF string and Device object
- get() returning Reading
- get_many() with mixed DRF strings and Device objects
- configure() before and after initialization
- shutdown() and re-initialization
- Environment variable configuration
- dpm() factory function
"""

import os
import pytest
from unittest import mock

import pacsys
from pacsys import (
    Device,
    ScalarDevice,
    DeviceError,
    Reading,
)
from pacsys.backends.dpm_http import DPMHTTPBackend
from pacsys.testing import FakeBackend


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before and after each test."""
    pacsys.shutdown()
    pacsys._config_backend = None
    pacsys._config_auth = None
    pacsys._config_role = None
    pacsys._config_dpm_host = None
    pacsys._config_dpm_port = None
    pacsys._config_pool_size = None
    pacsys._config_timeout = None
    yield
    pacsys.shutdown()
    pacsys._config_backend = None
    pacsys._config_auth = None
    pacsys._config_role = None
    pacsys._config_dpm_host = None
    pacsys._config_dpm_port = None
    pacsys._config_pool_size = None
    pacsys._config_timeout = None


@pytest.fixture
def fake():
    """FakeBackend patched as the global pacsys backend."""
    backend = FakeBackend()
    pacsys._global_backend = backend
    pacsys._backend_initialized = True
    yield backend
    pacsys._global_backend = None
    pacsys._backend_initialized = False


@pytest.fixture
def mock_backend():
    """MagicMock - only for backend factory/init tests."""
    backend = mock.MagicMock(spec=DPMHTTPBackend)
    return backend


# ─────────────────────────────────────────────────────────────────────────────
# read() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRead:
    """Tests for pacsys.read()."""

    def test_read_with_drf_string(self, fake):
        """read() accepts DRF string and returns value."""
        fake.set_reading("M:OUTTMP", 72.5)
        result = pacsys.read("M:OUTTMP")
        assert result == 72.5
        assert fake.was_read("M:OUTTMP")

    def test_read_with_device_object(self, fake):
        """read() accepts Device object."""
        fake.set_reading("M:OUTTMP.READING", 72.5)
        device = Device("M:OUTTMP")
        result = pacsys.read(device)
        assert result == 72.5

    def test_read_with_timeout(self, fake):
        """read() passes timeout to backend (FakeBackend ignores it)."""
        fake.set_reading("M:OUTTMP", 72.5)
        result = pacsys.read("M:OUTTMP", timeout=5.0)
        assert result == 72.5

    def test_read_raises_device_error_on_failure(self, fake):
        """read() raises DeviceError when device has error configured."""
        fake.set_error("M:BADDEV", -42, "Device not found")
        with pytest.raises(DeviceError) as exc_info:
            pacsys.read("M:BADDEV")
        assert exc_info.value.drf == "M:BADDEV"
        assert exc_info.value.error_code == -42

    def test_read_with_invalid_device_type(self, fake):
        """read() raises TypeError for invalid device type."""
        with pytest.raises(TypeError) as exc_info:
            pacsys.read(12345)  # Not a str or Device
        assert "Expected str or Device" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# get() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGet:
    """Tests for pacsys.get()."""

    def test_get_returns_reading(self, fake):
        """get() returns Reading object."""
        fake.set_reading("M:OUTTMP", 72.5, units="degF", description="Outdoor Temperature")
        result = pacsys.get("M:OUTTMP")
        assert isinstance(result, Reading)
        assert result.value == 72.5
        assert result.is_success
        assert result.ok

    def test_get_with_device_object(self, fake):
        """get() accepts Device object."""
        fake.set_reading("M:OUTTMP.READING", 72.5)
        device = ScalarDevice("M:OUTTMP")
        result = pacsys.get(device)
        assert result.value == 72.5

    def test_get_error_reading(self, fake):
        """get() returns error Reading without raising."""
        fake.set_error("M:BADDEV", -42, "Device not found")
        result = pacsys.get("M:BADDEV")
        assert result.is_error
        assert result.error_code == -42
        assert not result.ok


# ─────────────────────────────────────────────────────────────────────────────
# get_many() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetMany:
    """Tests for pacsys.get_many()."""

    def test_get_many_with_drf_strings(self, fake):
        """get_many() accepts list of DRF strings."""
        fake.set_reading("M:OUTTMP", 72.5)
        fake.set_error("M:BADDEV", -42, "Device not found")
        results = pacsys.get_many(["M:OUTTMP", "M:BADDEV"])
        assert len(results) == 2
        assert results[0].ok
        assert results[1].is_error

    def test_get_many_with_mixed_types(self, fake):
        """get_many() accepts mixed DRF strings and Device objects."""
        fake.set_reading("M:OUTTMP", 72.5)
        fake.set_reading("G:AMANDA.READING", 1.234)
        fake.set_reading("B:VIMIN.READING", 0.5)
        device = Device("G:AMANDA")
        scalar = ScalarDevice("B:VIMIN")
        results = pacsys.get_many(["M:OUTTMP", device, scalar])
        assert len(results) == 3

    def test_get_many_with_timeout(self, fake):
        """get_many() works with timeout parameter."""
        fake.set_reading("M:OUTTMP", 72.5)
        results = pacsys.get_many(["M:OUTTMP"], timeout=5.0)
        assert len(results) == 1
        assert results[0].value == 72.5


# ─────────────────────────────────────────────────────────────────────────────
# configure() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigure:
    """Tests for pacsys.configure()."""

    def test_configure_before_initialization(self):
        """configure() succeeds before backend initialization."""
        pacsys.configure(
            dpm_host="custom-proxy.fnal.gov",
            dpm_port=7000,
            pool_size=8,
            default_timeout=20.0,
        )

        # Check that configuration was stored
        assert pacsys._config_dpm_host == "custom-proxy.fnal.gov"
        assert pacsys._config_dpm_port == 7000
        assert pacsys._config_pool_size == 8
        assert pacsys._config_timeout == 20.0

    def test_configure_after_initialization_raises(self, mock_backend):
        """configure() raises RuntimeError after backend initialization."""
        # Simulate backend initialization
        with mock.patch.object(pacsys, "_backend_initialized", True):
            with pytest.raises(RuntimeError) as exc_info:
                pacsys.configure(dpm_host="custom.fnal.gov")

        assert "configure() must be called before" in str(exc_info.value)

    def test_configure_partial_settings(self):
        """configure() only sets specified parameters."""
        pacsys.configure(dpm_host="custom.fnal.gov")

        assert pacsys._config_dpm_host == "custom.fnal.gov"
        assert pacsys._config_dpm_port is None
        assert pacsys._config_pool_size is None
        assert pacsys._config_timeout is None

    def test_configure_invalid_backend_raises(self):
        """configure() raises ValueError for invalid backend name."""
        with pytest.raises(ValueError, match="Invalid backend"):
            pacsys.configure(backend="nosql")


# ─────────────────────────────────────────────────────────────────────────────
# shutdown() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestShutdown:
    """Tests for pacsys.shutdown()."""

    def test_shutdown_closes_backend(self, mock_backend):
        """shutdown() closes the global backend."""
        pacsys._global_backend = mock_backend
        pacsys._backend_initialized = True

        pacsys.shutdown()

        mock_backend.close.assert_called_once()
        assert pacsys._global_backend is None
        assert pacsys._backend_initialized is False

    def test_shutdown_multiple_times_safe(self):
        """shutdown() is safe to call multiple times."""
        pacsys.shutdown()
        pacsys.shutdown()
        pacsys.shutdown()  # Should not raise

    def test_shutdown_preserves_configuration(self):
        """shutdown() preserves configuration so re-init uses same settings."""
        pacsys.configure(dpm_host="custom.fnal.gov", pool_size=8)
        pacsys.shutdown()

        assert pacsys._config_dpm_host == "custom.fnal.gov"
        assert pacsys._config_pool_size == 8

    def test_shutdown_allows_reconfigure(self, mock_backend):
        """After shutdown(), configure() can be called again."""
        # Initialize backend
        pacsys._global_backend = mock_backend
        pacsys._backend_initialized = True

        # Shutdown and reconfigure
        pacsys.shutdown()
        pacsys.configure(dpm_host="new-proxy.fnal.gov")  # Should not raise

        assert pacsys._config_dpm_host == "new-proxy.fnal.gov"


# ─────────────────────────────────────────────────────────────────────────────
# Environment Variable Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""

    def test_env_dpm_host(self):
        """PACSYS_DPM_HOST environment variable is read."""
        with mock.patch.dict(os.environ, {"PACSYS_DPM_HOST": "env-proxy.fnal.gov"}):
            # Re-import to pick up new env value
            import importlib

            importlib.reload(pacsys)

            assert pacsys._env_dpm_host == "env-proxy.fnal.gov"

        # Restore original module state
        importlib.reload(pacsys)

    def test_env_dpm_port(self):
        """PACSYS_DPM_PORT environment variable is read as int."""
        with mock.patch.dict(os.environ, {"PACSYS_DPM_PORT": "7000"}):
            import importlib

            importlib.reload(pacsys)

            assert pacsys._env_dpm_port == 7000

        importlib.reload(pacsys)

    def test_env_pool_size(self):
        """PACSYS_POOL_SIZE environment variable is read as int."""
        with mock.patch.dict(os.environ, {"PACSYS_POOL_SIZE": "16"}):
            import importlib

            importlib.reload(pacsys)

            assert pacsys._env_pool_size == 16

        importlib.reload(pacsys)

    def test_env_timeout(self):
        """PACSYS_TIMEOUT environment variable is read as float."""
        with mock.patch.dict(os.environ, {"PACSYS_TIMEOUT": "30.5"}):
            import importlib

            importlib.reload(pacsys)

            assert pacsys._env_timeout == 30.5

        importlib.reload(pacsys)

    def test_env_invalid_port_raises(self):
        """Invalid PACSYS_DPM_PORT raises ValueError on import."""
        with mock.patch.dict(os.environ, {"PACSYS_DPM_PORT": "not-a-number"}):
            import importlib

            with pytest.raises(ValueError) as exc_info:
                importlib.reload(pacsys)

            assert "must be an integer" in str(exc_info.value)

        # Restore (need to remove the bad env var first)
        with mock.patch.dict(os.environ, clear=True):
            importlib.reload(pacsys)


# ─────────────────────────────────────────────────────────────────────────────
# Backend Factory Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestACLFactory:
    """Tests for pacsys.acl() factory function."""

    def test_acl_context_manager(self):
        """acl() returns a context manager."""
        with pacsys.acl() as backend:
            assert not backend._closed
        assert backend._closed


# ─────────────────────────────────────────────────────────────────────────────
# Session Tests
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Global Backend Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGlobalBackendInitialization:
    """Tests for global backend lazy initialization."""

    def test_backend_not_initialized_at_import(self):
        """Backend is not initialized at import."""
        assert pacsys._global_backend is None
        assert pacsys._backend_initialized is False

    def test_backend_initialized_on_first_use(self):
        """Backend is initialized on first use."""
        with mock.patch("pacsys.backends.dpm_http.DPMHTTPBackend") as MockBackend:
            mock_instance = mock.MagicMock(spec=DPMHTTPBackend)
            MockBackend.return_value = mock_instance

            backend = pacsys._get_global_backend()

        assert backend is mock_instance
        assert pacsys._backend_initialized

    def test_backend_uses_configured_settings(self):
        """Backend uses settings from configure()."""
        pacsys.configure(
            dpm_host="custom.fnal.gov",
            dpm_port=7000,
            pool_size=8,
            default_timeout=30.0,
        )

        with mock.patch("pacsys.backends.dpm_http.DPMHTTPBackend") as MockBackend:
            mock_instance = mock.MagicMock(spec=DPMHTTPBackend)
            MockBackend.return_value = mock_instance

            pacsys._get_global_backend()

        MockBackend.assert_called_once_with(
            host="custom.fnal.gov",
            port=7000,
            pool_size=8,
            timeout=30.0,
        )

    def test_backend_reused_on_subsequent_calls(self):
        """Backend is reused on subsequent calls."""
        with mock.patch("pacsys.backends.dpm_http.DPMHTTPBackend") as MockBackend:
            mock_instance = mock.MagicMock(spec=DPMHTTPBackend)
            MockBackend.return_value = mock_instance

            backend1 = pacsys._get_global_backend()
            backend2 = pacsys._get_global_backend()

        assert backend1 is backend2
        assert MockBackend.call_count == 1

    def test_global_backend_grpc(self):
        """configure(backend='grpc') creates GRPCBackend."""
        pacsys.configure(backend="grpc")

        with mock.patch("pacsys.backends.grpc_backend.GRPCBackend") as MockGRPC:
            mock_instance = mock.MagicMock()
            MockGRPC.return_value = mock_instance

            backend = pacsys._get_global_backend()

        assert backend is mock_instance
        MockGRPC.assert_called_once_with(timeout=5.0)

    def test_global_backend_dmq(self):
        """configure(backend='dmq') creates DMQBackend."""
        auth = mock.MagicMock(spec=pacsys.Auth)
        pacsys.configure(backend="dmq", auth=auth)

        with mock.patch("pacsys.backends.dmq.DMQBackend") as MockDMQ:
            mock_instance = mock.MagicMock()
            MockDMQ.return_value = mock_instance

            backend = pacsys._get_global_backend()

        assert backend is mock_instance
        MockDMQ.assert_called_once_with(timeout=5.0, auth=auth)

    def test_global_backend_acl(self):
        """configure(backend='acl') creates ACLBackend."""
        pacsys.configure(backend="acl")

        with mock.patch("pacsys.backends.acl.ACLBackend") as MockACL:
            mock_instance = mock.MagicMock()
            MockACL.return_value = mock_instance

            backend = pacsys._get_global_backend()

        assert backend is mock_instance
        MockACL.assert_called_once_with(timeout=5.0)

    def test_global_backend_dpm_with_auth(self):
        """configure(auth=..., role=...) passes auth/role to DPM backend."""
        auth = mock.MagicMock(spec=pacsys.Auth)
        pacsys.configure(auth=auth, role="testing")

        with mock.patch("pacsys.backends.dpm_http.DPMHTTPBackend") as MockDPM:
            mock_instance = mock.MagicMock(spec=DPMHTTPBackend)
            MockDPM.return_value = mock_instance

            pacsys._get_global_backend()

        MockDPM.assert_called_once_with(
            host="acsys-proxy.fnal.gov",
            port=6802,
            pool_size=4,
            timeout=5.0,
            auth=auth,
            role="testing",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration with Device API Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeviceIntegration:
    """Tests for integration between Simple API and Device API."""

    def test_device_uses_global_backend(self, fake):
        """Device uses global backend when none specified."""
        fake.set_reading("M:OUTTMP.READING", 72.5)
        device = Device("M:OUTTMP")
        value = device.read()
        assert value == 72.5

    def test_scalar_device_uses_global_backend(self, fake):
        """ScalarDevice uses global backend when none specified."""
        fake.set_reading("M:OUTTMP.READING", 72.5)
        device = ScalarDevice("M:OUTTMP")
        value = device.read()
        assert value == 72.5
        assert isinstance(value, float)


# ─────────────────────────────────────────────────────────────────────────────
# Thread Safety Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestThreadSafety:
    """Tests for thread-safe behavior."""

    def test_concurrent_initialization(self):
        """Concurrent calls to _get_global_backend() are thread-safe."""
        import threading

        results = []
        errors = []

        def get_backend():
            try:
                backend = pacsys._get_global_backend()
                results.append(backend)
            except Exception as e:
                errors.append(e)

        with mock.patch("pacsys.backends.dpm_http.DPMHTTPBackend") as MockBackend:
            mock_instance = mock.MagicMock(spec=DPMHTTPBackend)
            MockBackend.return_value = mock_instance

            threads = [threading.Thread(target=get_backend) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All threads should get the same backend instance
        assert all(r is results[0] for r in results)
        # Backend should only be created once
        assert MockBackend.call_count == 1
