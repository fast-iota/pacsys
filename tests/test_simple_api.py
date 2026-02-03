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
from datetime import datetime

import pacsys
from pacsys import (
    Device,
    ScalarDevice,
    DeviceError,
    Reading,
    ValueType,
    DeviceMeta,
)
from pacsys.backends.dpm_http import DPMHTTPBackend


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before and after each test."""
    pacsys.shutdown()
    pacsys._config_dpm_host = None
    pacsys._config_dpm_port = None
    pacsys._config_pool_size = None
    pacsys._config_timeout = None
    yield
    pacsys.shutdown()
    pacsys._config_dpm_host = None
    pacsys._config_dpm_port = None
    pacsys._config_pool_size = None
    pacsys._config_timeout = None


@pytest.fixture
def mock_backend():
    """Create a mock DPMHTTPBackend."""
    backend = mock.MagicMock(spec=DPMHTTPBackend)
    return backend


@pytest.fixture
def sample_reading():
    """Create a sample Reading for tests."""
    return Reading(
        drf="M:OUTTMP",
        value_type=ValueType.SCALAR,
        tag=1,
        facility_code=0,
        error_code=0,
        value=72.5,
        message=None,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        cycle=1234,
        meta=DeviceMeta(
            device_index=12345,
            name="M:OUTTMP",
            description="Outdoor Temperature",
            units="degF",
        ),
    )


@pytest.fixture
def error_reading():
    """Create an error Reading for tests."""
    return Reading(
        drf="M:BADDEV",
        value_type=ValueType.SCALAR,
        tag=1,
        facility_code=0,
        error_code=-42,
        value=None,
        message="Device not found",
        timestamp=None,
        cycle=0,
        meta=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# read() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRead:
    """Tests for pacsys.read()."""

    def test_read_with_drf_string(self, mock_backend, sample_reading):
        """read() accepts DRF string."""
        mock_backend.read.return_value = 72.5

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                result = pacsys.read("M:OUTTMP")

        assert result == 72.5
        mock_backend.read.assert_called_once_with("M:OUTTMP", timeout=None)

    def test_read_with_device_object(self, mock_backend):
        """read() accepts Device object."""
        mock_backend.read.return_value = 72.5
        device = Device("M:OUTTMP")

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                result = pacsys.read(device)

        assert result == 72.5
        # Device normalizes DRF to canonical form
        mock_backend.read.assert_called_once()

    def test_read_with_timeout(self, mock_backend):
        """read() passes timeout to backend."""
        mock_backend.read.return_value = 72.5

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                pacsys.read("M:OUTTMP", timeout=5.0)

        mock_backend.read.assert_called_once_with("M:OUTTMP", timeout=5.0)

    def test_read_raises_device_error_on_failure(self, mock_backend):
        """read() raises DeviceError when backend fails."""
        mock_backend.read.side_effect = DeviceError(
            drf="M:BADDEV",
            facility_code=0,
            error_code=-42,
            message="Device not found",
        )

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                with pytest.raises(DeviceError) as exc_info:
                    pacsys.read("M:BADDEV")

        assert exc_info.value.drf == "M:BADDEV"
        assert exc_info.value.error_code == -42

    def test_read_with_invalid_device_type(self, mock_backend):
        """read() raises TypeError for invalid device type."""
        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                with pytest.raises(TypeError) as exc_info:
                    pacsys.read(12345)  # Not a str or Device

        assert "Expected str or Device" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# get() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGet:
    """Tests for pacsys.get()."""

    def test_get_returns_reading(self, mock_backend, sample_reading):
        """get() returns Reading object."""
        mock_backend.get.return_value = sample_reading

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                result = pacsys.get("M:OUTTMP")

        assert isinstance(result, Reading)
        assert result.value == 72.5
        assert result.is_success
        assert result.ok

    def test_get_with_device_object(self, mock_backend, sample_reading):
        """get() accepts Device object."""
        mock_backend.get.return_value = sample_reading
        device = ScalarDevice("M:OUTTMP")

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                result = pacsys.get(device)

        assert result.value == 72.5

    def test_get_error_reading(self, mock_backend, error_reading):
        """get() returns error Reading without raising."""
        mock_backend.get.return_value = error_reading

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                result = pacsys.get("M:BADDEV")

        assert result.is_error
        assert result.error_code == -42
        assert not result.ok


# ─────────────────────────────────────────────────────────────────────────────
# get_many() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetMany:
    """Tests for pacsys.get_many()."""

    def test_get_many_with_drf_strings(self, mock_backend, sample_reading, error_reading):
        """get_many() accepts list of DRF strings."""
        mock_backend.get_many.return_value = [sample_reading, error_reading]

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                results = pacsys.get_many(["M:OUTTMP", "M:BADDEV"])

        assert len(results) == 2
        assert results[0].ok
        assert results[1].is_error

    def test_get_many_with_mixed_types(self, mock_backend, sample_reading):
        """get_many() accepts mixed DRF strings and Device objects."""
        mock_backend.get_many.return_value = [sample_reading, sample_reading, sample_reading]
        device = Device("G:AMANDA")
        scalar = ScalarDevice("B:VIMIN")

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                results = pacsys.get_many(["M:OUTTMP", device, scalar])

        assert len(results) == 3
        mock_backend.get_many.assert_called_once()

    def test_get_many_empty_list(self, mock_backend):
        """get_many() returns empty list for empty input."""
        mock_backend.get_many.return_value = []

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                results = pacsys.get_many([])

        assert results == []

    def test_get_many_with_timeout(self, mock_backend, sample_reading):
        """get_many() passes timeout to backend."""
        mock_backend.get_many.return_value = [sample_reading]

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                pacsys.get_many(["M:OUTTMP"], timeout=5.0)

        mock_backend.get_many.assert_called_once_with(["M:OUTTMP"], timeout=5.0)


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


# ─────────────────────────────────────────────────────────────────────────────
# shutdown() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestShutdown:
    """Tests for pacsys.shutdown()."""

    def test_shutdown_closes_backend(self, mock_backend):
        """shutdown() closes the global backend."""
        pacsys._global_dpm_backend = mock_backend
        pacsys._backend_initialized = True

        pacsys.shutdown()

        mock_backend.close.assert_called_once()
        assert pacsys._global_dpm_backend is None
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
        pacsys._global_dpm_backend = mock_backend
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


class TestDPMFactory:
    """Tests for pacsys.dpm() factory function."""

    def test_dpm_creates_backend_with_defaults(self):
        """dpm() creates backend with default settings."""
        with mock.patch("pacsys.backends.dpm_http.DPMHTTPBackend") as MockBackend:
            MockBackend.return_value = mock.MagicMock(spec=DPMHTTPBackend)
            pacsys.dpm()

        MockBackend.assert_called_once_with(
            host="acsys-proxy.fnal.gov",
            port=6802,
            pool_size=4,
            timeout=5.0,
            auth=None,
            role=None,
        )

    def test_dpm_creates_backend_with_custom_settings(self):
        """dpm() creates backend with custom settings."""
        with mock.patch("pacsys.backends.dpm_http.DPMHTTPBackend") as MockBackend:
            MockBackend.return_value = mock.MagicMock(spec=DPMHTTPBackend)
            pacsys.dpm(
                host="custom.fnal.gov",
                port=7000,
                pool_size=8,
                timeout=30.0,
            )

        MockBackend.assert_called_once_with(
            host="custom.fnal.gov",
            port=7000,
            pool_size=8,
            timeout=30.0,
            auth=None,
            role=None,
        )

    def test_dpm_with_auth_passes_to_backend(self):
        """dpm() with auth passes it to backend."""
        with mock.patch("pacsys.backends.dpm_http.DPMHTTPBackend") as MockBackend:
            MockBackend.return_value = mock.MagicMock(spec=DPMHTTPBackend)
            pacsys.dpm(auth="kerberos")

        MockBackend.assert_called_once_with(
            host="acsys-proxy.fnal.gov",
            port=6802,
            pool_size=4,
            timeout=5.0,
            auth="kerberos",
            role=None,
        )

    def test_dpm_with_role_passes_to_backend(self):
        """dpm() with role passes it to backend."""
        with mock.patch("pacsys.backends.dpm_http.DPMHTTPBackend") as MockBackend:
            MockBackend.return_value = mock.MagicMock(spec=DPMHTTPBackend)
            pacsys.dpm(role="Operator")

        MockBackend.assert_called_once_with(
            host="acsys-proxy.fnal.gov",
            port=6802,
            pool_size=4,
            timeout=5.0,
            auth=None,
            role="Operator",
        )


class TestGRPCFactory:
    """Tests for pacsys.grpc() factory function."""

    def test_grpc_returns_backend_or_raises_import_error(self):
        """grpc() returns a GRPCBackend or raises ImportError if grpc is not installed."""
        try:
            # If grpc is installed, this returns a backend
            backend = pacsys.grpc()
            backend.close()
        except ImportError as e:
            # If grpc is not installed, ImportError is expected
            assert "grpc package required" in str(e)


class TestACLFactory:
    """Tests for pacsys.acl() factory function."""

    def test_acl_returns_backend(self):
        """acl() returns an ACLBackend instance."""
        from pacsys.backends.acl import ACLBackend

        backend = pacsys.acl()
        try:
            assert isinstance(backend, ACLBackend)
            assert backend.base_url == "https://www-ad.fnal.gov/cgi-bin/acl.pl"
            assert backend.timeout == 5.0
        finally:
            backend.close()

    def test_acl_custom_parameters(self):
        """acl() accepts custom parameters."""
        from pacsys.backends.acl import ACLBackend

        backend = pacsys.acl(base_url="https://custom.example.com/acl", timeout=5.0)
        try:
            assert isinstance(backend, ACLBackend)
            assert backend.base_url == "https://custom.example.com/acl"
            assert backend.timeout == 5.0
        finally:
            backend.close()

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
        assert pacsys._global_dpm_backend is None
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


# ─────────────────────────────────────────────────────────────────────────────
# Integration with Device API Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeviceIntegration:
    """Tests for integration between Simple API and Device API."""

    def test_device_uses_global_backend(self, mock_backend, sample_reading):
        """Device uses global backend when none specified."""
        mock_backend.read.return_value = 72.5
        mock_backend.get.return_value = sample_reading

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
                device = Device("M:OUTTMP")
                value = device.read()

        assert value == 72.5

    def test_scalar_device_uses_global_backend(self, mock_backend):
        """ScalarDevice uses global backend when none specified."""
        mock_backend.read.return_value = 72.5

        with mock.patch.object(pacsys, "_global_dpm_backend", mock_backend):
            with mock.patch.object(pacsys, "_backend_initialized", True):
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


# ─────────────────────────────────────────────────────────────────────────────
# Exports Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExports:
    """Tests for module exports."""

    def test_all_exports_accessible(self):
        """All __all__ exports are accessible."""
        for name in pacsys.__all__:
            if name == "acnet":
                # Lazy import - skip
                continue
            assert hasattr(pacsys, name), f"Missing export: {name}"

    def test_types_exported(self):
        """Core types are exported."""
        assert hasattr(pacsys, "Value")
        assert hasattr(pacsys, "DeviceSpec")
        assert hasattr(pacsys, "ValueType")
        assert hasattr(pacsys, "BackendCapability")
        assert hasattr(pacsys, "DeviceMeta")
        assert hasattr(pacsys, "Reading")
        assert hasattr(pacsys, "WriteResult")

    def test_errors_exported(self):
        """Error classes are exported."""
        assert hasattr(pacsys, "DeviceError")
        assert hasattr(pacsys, "AuthenticationError")

    def test_device_classes_exported(self):
        """Device classes are exported."""
        assert hasattr(pacsys, "Device")
        assert hasattr(pacsys, "ScalarDevice")
        assert hasattr(pacsys, "ArrayDevice")
        assert hasattr(pacsys, "TextDevice")

    def test_simple_api_functions_exported(self):
        """Simple API functions are exported."""
        assert hasattr(pacsys, "read")
        assert hasattr(pacsys, "get")
        assert hasattr(pacsys, "get_many")

    def test_configuration_functions_exported(self):
        """Configuration functions are exported."""
        assert hasattr(pacsys, "configure")
        assert hasattr(pacsys, "shutdown")

    def test_backend_factories_exported(self):
        """Backend factory functions are exported."""
        assert hasattr(pacsys, "dpm")
        assert hasattr(pacsys, "grpc")
        assert hasattr(pacsys, "acl")
