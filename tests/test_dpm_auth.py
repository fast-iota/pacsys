"""
Unit tests for DPM Backend authentication and write support.

Tests cover:
- Backend initialization with auth parameters
- Capabilities based on auth configuration
- Write single value
- Write multiple values
- Error handling (no auth, failed auth)
- Kerberos authentication flow (mocked)
"""

import pytest
from unittest import mock

from pacsys.backends.dpm_http import DPMHTTPBackend
from pacsys.auth import KerberosAuth
from pacsys.types import WriteResult
from pacsys.errors import AuthenticationError
from pacsys.acnet.errors import make_error
from tests.devices import (
    MockGSSAPIModule,
    MockSocketWithReplies,
    make_device_info,
    make_start_list,
    make_auth_reply,
    make_apply_settings_reply,
    make_enable_settings_reply,
    make_write_sequence,
    TEMP_DEVICE,
    TEMP_VALUE,
)


def create_mock_kerberos_auth(principal="user@FNAL.GOV"):
    """Create a mock KerberosAuth object."""
    mock_auth = mock.MagicMock(spec=KerberosAuth)
    mock_auth.auth_type = "kerberos"
    mock_auth.principal = principal
    return mock_auth


class TestAuthenticationInit:
    """Tests for backend initialization with authentication."""

    def test_invalid_auth_method(self):
        """Test that invalid auth type raises ValueError."""
        with pytest.raises(ValueError, match="auth must be KerberosAuth or None"):
            DPMHTTPBackend(auth="invalid")

    def test_auth_kerberos_requires_gssapi(self):
        """Test that KerberosAuth raises ImportError without gssapi."""
        # gssapi is not installed, so this should raise ImportError
        # If gssapi IS installed, skip this test
        try:
            import gssapi  # noqa: F401

            pytest.skip("gssapi is installed - cannot test missing gssapi")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="gssapi library required"):
            KerberosAuth()

    def test_auth_kerberos_initializes_principal(self):
        """Test that KerberosAuth extracts principal name."""
        mock_gssapi = MockGSSAPIModule()

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = KerberosAuth()
            backend = DPMHTTPBackend(auth=auth)
            assert backend.authenticated
            assert backend.principal == "user@FNAL.GOV"
            backend.close()

    def test_auth_kerberos_no_credentials(self):
        """Test that missing Kerberos credentials raises AuthenticationError."""
        # This test requires mocking gssapi. Use the mock.
        mock_gssapi = MockGSSAPIModule()

        # Override the Credentials method to raise an error
        class MockGSSError(Exception):
            pass

        mock_gssapi.exceptions.GSSError = MockGSSError

        def bad_credentials(usage=None):
            raise MockGSSError("No credentials found")

        mock_gssapi.Credentials = staticmethod(bad_credentials)

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with pytest.raises(AuthenticationError, match="No valid Kerberos credentials"):
                KerberosAuth()


class TestWriteWithoutAuth:
    """Tests for write operations without authentication."""

    def test_write_raises_authentication_error(self):
        """Test that write without auth raises AuthenticationError."""
        backend = DPMHTTPBackend()
        try:
            with pytest.raises(AuthenticationError, match="not configured for authenticated"):
                backend.write("M:OUTTMP", 72.5)
        finally:
            backend.close()

    def test_write_many_raises_authentication_error(self):
        """Test that write_many without auth raises AuthenticationError."""
        backend = DPMHTTPBackend()
        try:
            with pytest.raises(AuthenticationError, match="not configured for authenticated"):
                backend.write_many([("M:OUTTMP", 72.5)])
        finally:
            backend.close()


class TestWriteWithAuthNoRole:
    """Tests for write operations with auth but no role."""

    def test_write_raises_role_error(self):
        """Test that write with auth but no role raises AuthenticationError."""
        mock_gssapi = MockGSSAPIModule()

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = KerberosAuth()
            backend = DPMHTTPBackend(auth=auth)
            try:
                with pytest.raises(AuthenticationError, match="Role required"):
                    backend.write("M:OUTTMP", 72.5)
            finally:
                backend.close()


class TestWriteSuccess:
    """Tests for successful write operations."""

    def test_write_single_scalar(self):
        """Test writing a single scalar value."""
        mock_gssapi = MockGSSAPIModule()
        replies = make_write_sequence()
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")
                try:
                    result = backend.write(TEMP_DEVICE, TEMP_VALUE)
                    assert isinstance(result, WriteResult)
                    assert result.success
                    assert result.drf == TEMP_DEVICE
                finally:
                    backend.close()

    def test_write_many_scalars(self):
        """Test writing multiple scalar values."""
        mock_gssapi = MockGSSAPIModule()
        replies = [
            make_auth_reply(),
            make_auth_reply(),
            make_enable_settings_reply(),
            make_device_info(name="M:OUTTMP", ref_id=1),
            make_device_info(name="G:AMANDA", ref_id=2, di=12346),
            make_start_list(),
            make_apply_settings_reply([(1, 0), (2, 0)]),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")
                try:
                    results = backend.write_many([("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)])
                    assert len(results) == 2
                    assert all(r.success for r in results)
                finally:
                    backend.close()


class TestWriteFailure:
    """Tests for write operation failures."""

    def test_write_device_error(self):
        """Test write failure returns error status."""
        mock_gssapi = MockGSSAPIModule()
        replies = make_write_sequence(success=False)
        # Override the apply settings reply with specific error code
        replies[-1] = make_apply_settings_reply([(1, make_error(1, -42))])
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")
                try:
                    result = backend.write(TEMP_DEVICE, TEMP_VALUE)
                    assert not result.success
                    assert result.error_code == -42
                finally:
                    backend.close()

    def test_write_partial_failure(self):
        """Test partial failure in write_many."""
        mock_gssapi = MockGSSAPIModule()
        replies = [
            make_auth_reply(),
            make_auth_reply(),
            make_enable_settings_reply(),
            make_device_info(name="M:OUTTMP", ref_id=1),
            make_device_info(name="M:BADDEV", ref_id=2, di=12346),
            make_start_list(),
            make_apply_settings_reply([(1, 0), (2, make_error(1, -100))]),
        ]
        mock_socket = MockSocketWithReplies(list_id=1, replies=replies)

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")
                try:
                    results = backend.write_many([("M:OUTTMP", 72.5), ("M:BADDEV", 1.0)])
                    assert len(results) == 2
                    assert results[0].success
                    assert not results[1].success
                    assert results[1].error_code == -100
                finally:
                    backend.close()


class TestWriteValueTypes:
    """Tests for different value types in write operations."""

    def _create_mock_for_write(self):
        """Create common mock setup for write tests."""
        return MockSocketWithReplies(list_id=1, replies=make_write_sequence())

    def test_write_string_value(self):
        """Test writing a string value."""
        mock_gssapi = MockGSSAPIModule()
        mock_socket = self._create_mock_for_write()

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")
                try:
                    result = backend.write(TEMP_DEVICE, "test_value")
                    assert result.success
                finally:
                    backend.close()

    def test_write_bytes_value(self):
        """Test writing raw bytes value."""
        mock_gssapi = MockGSSAPIModule()
        mock_socket = self._create_mock_for_write()

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")
                try:
                    result = backend.write(TEMP_DEVICE, b"\x01\x02\x03")
                    assert result.success
                finally:
                    backend.close()

    def test_write_array_value(self):
        """Test writing an array value."""
        import numpy as np

        mock_gssapi = MockGSSAPIModule()
        mock_socket = self._create_mock_for_write()

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")
                try:
                    result = backend.write(TEMP_DEVICE, np.array([1.0, 2.0, 3.0]))
                    assert result.success
                finally:
                    backend.close()

    def test_write_list_value(self):
        """Test writing a list value."""
        mock_gssapi = MockGSSAPIModule()
        mock_socket = self._create_mock_for_write()

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")
                try:
                    result = backend.write(TEMP_DEVICE, [1.0, 2.0, 3.0])
                    assert result.success
                finally:
                    backend.close()


class TestWriteVerify:
    """Tests for write verification (now handled at Device layer, not backend)."""

    def test_backend_write_has_no_verify_param(self):
        """Backend.write() no longer accepts verify/tolerance -- verification is Device-layer only."""
        mock_gssapi = MockGSSAPIModule()
        mock_socket = MockSocketWithReplies(list_id=1, replies=make_write_sequence())

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")
                try:
                    with pytest.raises(TypeError):
                        backend.write(TEMP_DEVICE, TEMP_VALUE, verify=True)
                finally:
                    backend.close()


class TestWriteConnectionPool:
    """Tests for write connection caching behavior."""

    def test_write_connection_closed_on_backend_close(self):
        """Test that write connections are closed when backend closes."""
        mock_gssapi = MockGSSAPIModule()
        mock_socket = MockSocketWithReplies(list_id=1, replies=make_write_sequence())

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            with mock.patch("socket.socket", return_value=mock_socket):
                auth = KerberosAuth()
                backend = DPMHTTPBackend(auth=auth, role="Operator")

                # Do a write which should create a write connection
                result = backend.write(TEMP_DEVICE, TEMP_VALUE)
                assert result.success

                # Check connection was returned to pool
                assert len(backend._write_connections) == 1

                # Close backend
                backend.close()

                # Pool should be empty
                assert len(backend._write_connections) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
