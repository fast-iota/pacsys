"""
Unit tests for ACLBackend.

Tests cover:
- Backend initialization and capabilities
- Single device read/get
- Multiple device read
- Error handling (HTTP errors, device errors)
- Timeout handling
- URL encoding of device names
- Uses mocking for HTTP requests (no real network calls)
"""

import urllib.error
import urllib.request
from unittest import mock

import pytest

from pacsys.backends import Backend
from pacsys.backends.acl import (
    ACLBackend,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    _parse_acl_value,
    _is_error_response,
)
from pacsys.types import BackendCapability, Reading, ValueType
from pacsys.errors import DeviceError
from tests.conftest import MockACLResponse


class TestBackendAbstract:
    """Tests for Backend abstract base class."""

    def test_backend_is_abstract(self):
        """Test that Backend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Backend()

    def test_acl_backend_is_subclass(self):
        """Test that ACLBackend is a Backend subclass."""
        assert issubclass(ACLBackend, Backend)


class TestACLBackendInit:
    """Tests for ACLBackend initialization."""

    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        backend = ACLBackend()
        assert backend.base_url == DEFAULT_BASE_URL
        assert backend.timeout == DEFAULT_TIMEOUT

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        backend = ACLBackend(
            base_url="https://custom.example.com/acl",
            timeout=5.0,
        )
        assert backend.base_url == "https://custom.example.com/acl"
        assert backend.timeout == 5.0

    def test_none_parameters_use_defaults(self):
        """Test that None parameters use defaults."""
        backend = ACLBackend(base_url=None, timeout=None)
        assert backend.base_url == DEFAULT_BASE_URL
        assert backend.timeout == DEFAULT_TIMEOUT

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"base_url": ""}, "base_url cannot be empty"),
            ({"timeout": 0}, "timeout must be positive"),
            ({"timeout": -1.0}, "timeout must be positive"),
        ],
    )
    def test_invalid_init_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ACLBackend(**kwargs)


class TestCapabilities:
    """Tests for backend capabilities."""

    def test_capabilities_include_read(self):
        """Test that capabilities include READ."""
        backend = ACLBackend()
        assert BackendCapability.READ in backend.capabilities

    def test_capabilities_include_batch(self):
        """Test that capabilities include BATCH."""
        backend = ACLBackend()
        assert BackendCapability.BATCH in backend.capabilities

    def test_capabilities_exclude_write(self):
        """Test that capabilities do not include WRITE."""
        backend = ACLBackend()
        assert BackendCapability.WRITE not in backend.capabilities

    def test_capabilities_exclude_stream(self):
        """Test that capabilities do not include STREAM."""
        backend = ACLBackend()
        assert BackendCapability.STREAM not in backend.capabilities

    def test_capabilities_exclude_auth(self):
        """Test that capabilities do not include AUTH_KERBEROS or AUTH_JWT."""
        backend = ACLBackend()
        assert BackendCapability.AUTH_KERBEROS not in backend.capabilities
        assert BackendCapability.AUTH_JWT not in backend.capabilities

    def test_not_authenticated(self):
        """Test that backend is not authenticated."""
        backend = ACLBackend()
        assert not backend.authenticated

    def test_principal_is_none(self):
        """Test that principal is None."""
        backend = ACLBackend()
        assert backend.principal is None


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.parametrize(
        "raw,expected_value,expected_type",
        [
            ("72.5", 72.5, ValueType.SCALAR),
            ("42", 42.0, ValueType.SCALAR),
            ("-3.14", -3.14, ValueType.SCALAR),
            ("1.23e-4", pytest.approx(1.23e-4), ValueType.SCALAR),
            ("1.0 2.0 3.0", [1.0, 2.0, 3.0], ValueType.SCALAR_ARRAY),
            ("Hello World", "Hello World", ValueType.TEXT),
            ("  72.5  \n", 72.5, ValueType.SCALAR),
        ],
    )
    def test_parse_acl_value(self, raw, expected_value, expected_type):
        value, vtype = _parse_acl_value(raw)
        assert value == expected_value
        assert vtype == expected_type

    def test_is_error_response_exclamation(self):
        """Test error detection with exclamation mark."""
        is_error, msg = _is_error_response("! Device not found")
        assert is_error is True
        assert msg == "Device not found"

    def test_is_error_response_error_word(self):
        """Test error detection with 'error' in message."""
        is_error, msg = _is_error_response("Error: Invalid device name")
        assert is_error is True
        assert "Error" in msg

    def test_is_error_response_normal(self):
        """Test normal response is not an error."""
        is_error, msg = _is_error_response("72.5")
        assert is_error is False
        assert msg is None


class TestSingleDeviceRead:
    """Tests for single device read/get operations."""

    def test_read_scalar_success(self):
        """Test successful scalar read."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5")

            backend = ACLBackend()
            try:
                value = backend.read("M:OUTTMP")
                assert value == 72.5

                # Verify URL was built correctly
                call_args = mock_urlopen.call_args
                assert "M%3AOUTTMP" in call_args[0][0]  # URL-encoded ":"
                assert "read/" in call_args[0][0]
            finally:
                backend.close()

    def test_read_text_success(self):
        """Test successful text read."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("Outdoor Temperature")

            backend = ACLBackend()
            try:
                value = backend.read("M:OUTTMP.DESCRIPTION")
                assert value == "Outdoor Temperature"
            finally:
                backend.close()

    def test_get_returns_reading(self):
        """Test that get() returns a Reading object."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5")

            backend = ACLBackend()
            try:
                reading = backend.get("M:OUTTMP")
                assert isinstance(reading, Reading)
                assert reading.value == 72.5
                assert reading.value_type == ValueType.SCALAR
                assert reading.is_success
                assert reading.ok
            finally:
                backend.close()

    def test_read_error_raises_device_error(self):
        """Test that read() raises DeviceError on ACL error."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("! Device not found")

            backend = ACLBackend()
            try:
                with pytest.raises(DeviceError) as exc_info:
                    backend.read("M:BADDEV")
                assert "Device not found" in exc_info.value.message
            finally:
                backend.close()

    def test_get_error_returns_reading_with_error(self):
        """Test that get() returns Reading with is_error=True on failure."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("! Device not found")

            backend = ACLBackend()
            try:
                reading = backend.get("M:BADDEV")
                assert reading.is_error
                assert not reading.ok
                assert "Device not found" in reading.message
            finally:
                backend.close()


class TestMultipleDeviceRead:
    """Tests for multiple device get_many operations."""

    def test_get_many_multiple_devices(self):
        """Test reading multiple devices."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            # ACL returns one value per line
            mock_urlopen.return_value = MockACLResponse("72.5\n1.234")

            backend = ACLBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
                assert len(readings) == 2
                assert readings[0].value == 72.5
                assert readings[1].value == 1.234

                # Verify URL contains both devices with +
                call_args = mock_urlopen.call_args
                assert "+" in call_args[0][0]
            finally:
                backend.close()

    def test_get_many_partial_failure(self):
        """Test that partial failures are returned as error readings."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5\n! Bad device")

            backend = ACLBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "M:BADDEV"])
                assert len(readings) == 2
                assert readings[0].ok
                assert readings[0].value == 72.5
                assert readings[1].is_error
                assert "Bad device" in readings[1].message
            finally:
                backend.close()

    def test_get_many_empty_list(self):
        """Test that empty list returns empty list."""
        backend = ACLBackend()
        try:
            readings = backend.get_many([])
            assert readings == []
        finally:
            backend.close()


class TestHTTPErrors:
    """Tests for HTTP error handling."""

    def test_http_error_404(self):
        """Test handling of HTTP 404 error."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="http://example.com",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=None,
            )

            backend = ACLBackend()
            try:
                with pytest.raises(DeviceError) as exc_info:
                    backend.read("M:OUTTMP")
                assert "HTTP 404" in exc_info.value.message
            finally:
                backend.close()

    def test_http_error_500(self):
        """Test handling of HTTP 500 error."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="http://example.com",
                code=500,
                msg="Internal Server Error",
                hdrs={},
                fp=None,
            )

            backend = ACLBackend()
            try:
                with pytest.raises(DeviceError) as exc_info:
                    backend.read("M:OUTTMP")
                assert "HTTP 500" in exc_info.value.message
            finally:
                backend.close()

    def test_url_error(self):
        """Test handling of URL error (e.g., connection refused)."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            backend = ACLBackend()
            try:
                with pytest.raises(DeviceError) as exc_info:
                    backend.read("M:OUTTMP")
                assert "ACL request failed" in exc_info.value.message
            finally:
                backend.close()

    def test_timeout_error(self):
        """Test handling of timeout error."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Connection timed out")

            backend = ACLBackend()
            try:
                with pytest.raises(DeviceError) as exc_info:
                    backend.read("M:OUTTMP")
                assert "timed out" in exc_info.value.message.lower()
            finally:
                backend.close()

    def test_get_many_http_error_returns_error_readings(self):
        """Test that get_many() returns error readings on HTTP error."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="http://example.com",
                code=503,
                msg="Service Unavailable",
                hdrs={},
                fp=None,
            )

            backend = ACLBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
                assert len(readings) == 2
                assert all(r.is_error for r in readings)
                assert all("HTTP 503" in r.message for r in readings)
            finally:
                backend.close()


class TestURLEncoding:
    """Tests for URL encoding of device names."""

    def test_device_with_colon_encoded(self):
        """Test that colon in device name is URL encoded."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5")

            backend = ACLBackend()
            try:
                backend.read("M:OUTTMP")
                call_args = mock_urlopen.call_args
                url = call_args[0][0]
                # Colon should be encoded as %3A
                assert "M%3AOUTTMP" in url
            finally:
                backend.close()

    def test_device_with_special_chars_encoded(self):
        """Test that special characters are URL encoded."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5")

            backend = ACLBackend()
            try:
                backend.read("M:OUTTMP[0:10]@p,1000")
                call_args = mock_urlopen.call_args
                url = call_args[0][0]
                # Brackets, @ and comma should be encoded
                assert "[" not in url or "%5B" in url
                assert "]" not in url or "%5D" in url
                assert "@" not in url or "%40" in url
            finally:
                backend.close()


class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager_closes(self):
        """Test that context manager closes backend."""
        with ACLBackend() as backend:
            assert not backend._closed
        assert backend._closed

    def test_context_manager_on_exception(self):
        """Test that backend is closed even on exception."""
        try:
            with ACLBackend() as backend:
                raise ValueError("test error")
        except ValueError:
            pass
        assert backend._closed

    def test_close_multiple_times_safe(self):
        """Test that close() can be called multiple times safely."""
        backend = ACLBackend()
        backend.close()
        backend.close()  # Should not raise
        backend.close()  # Should not raise
        assert backend._closed


class TestWriteNotSupported:
    """Tests for write operations."""

    def test_write_raises_not_implemented(self):
        """Test that write() raises NotImplementedError."""
        backend = ACLBackend()
        try:
            with pytest.raises(NotImplementedError):
                backend.write("M:OUTTMP", 72.5)
        finally:
            backend.close()

    def test_write_many_raises_not_implemented(self):
        """Test that write_many() raises NotImplementedError."""
        backend = ACLBackend()
        try:
            with pytest.raises(NotImplementedError):
                backend.write_many([("M:OUTTMP", 72.5)])
        finally:
            backend.close()


class TestOperationAfterClose:
    """Tests for operations after close."""

    def test_read_after_close_raises(self):
        """Test that read() after close raises RuntimeError."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5")

            backend = ACLBackend()
            backend.close()

            with pytest.raises(RuntimeError, match="Backend is closed"):
                backend.read("M:OUTTMP")

    def test_get_after_close_raises(self):
        """Test that get() after close raises RuntimeError."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5")

            backend = ACLBackend()
            backend.close()

            with pytest.raises(RuntimeError, match="Backend is closed"):
                backend.get("M:OUTTMP")

    def test_get_many_after_close_raises(self):
        """Test that get_many() after close raises RuntimeError."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5")

            backend = ACLBackend()
            backend.close()

            with pytest.raises(RuntimeError, match="Backend is closed"):
                backend.get_many(["M:OUTTMP"])


class TestTimeout:
    """Tests for timeout handling."""

    def test_custom_timeout_passed_to_urlopen(self):
        """Test that custom timeout is passed to urlopen."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5")

            backend = ACLBackend(timeout=3.0)
            try:
                backend.read("M:OUTTMP")
                call_args = mock_urlopen.call_args
                assert call_args[1]["timeout"] == 3.0
            finally:
                backend.close()

    def test_per_call_timeout_overrides_default(self):
        """Test that per-call timeout overrides default."""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockACLResponse("72.5")

            backend = ACLBackend(timeout=10.0)
            try:
                backend.read("M:OUTTMP", timeout=2.0)
                call_args = mock_urlopen.call_args
                assert call_args[1]["timeout"] == 2.0
            finally:
                backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
