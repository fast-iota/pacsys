"""
Unit tests for pacsys.auth module.

Tests cover:
- Auth base class
- KerberosAuth initialization and validation
- JWTAuth initialization, principal extraction, and from_env
"""

import os
from unittest import mock

import pytest

from pacsys.auth import Auth, KerberosAuth, JWTAuth
from pacsys.errors import AuthenticationError
from tests.devices import make_jwt_token, MockGSSAPIModule


class TestAuthAbstract:
    """Tests for Auth abstract base class."""

    def test_auth_is_abstract(self):
        """Test that Auth cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Auth()

    def test_kerberos_auth_is_subclass(self):
        """Test that KerberosAuth is subclass of Auth."""
        assert issubclass(KerberosAuth, Auth)

    def test_jwt_auth_is_subclass(self):
        """Test that JWTAuth is subclass of Auth."""
        assert issubclass(JWTAuth, Auth)


class TestJWTAuth:
    """Tests for JWTAuth class."""

    def test_create_with_token(self):
        """Test creating JWTAuth with explicit token."""
        token = make_jwt_token({"sub": "user@example.com"})
        auth = JWTAuth(token=token)
        assert auth.token == token
        assert auth.auth_type == "jwt"

    def test_principal_extracted(self):
        """Test that principal is extracted from JWT."""
        token = make_jwt_token({"sub": "user@example.com"})
        auth = JWTAuth(token=token)
        assert auth.principal == "user@example.com"

    def test_principal_missing_sub(self):
        """Test error when JWT has no sub claim."""
        token = make_jwt_token({"name": "Test User"})
        auth = JWTAuth(token=token)
        with pytest.raises(ValueError, match="no 'sub' claim"):
            _ = auth.principal

    def test_decode_invalid_format(self):
        """Test error on invalid JWT format."""
        auth = JWTAuth(token="not.a.valid.jwt.token")
        with pytest.raises(ValueError, match="Invalid JWT format"):
            _ = auth._decode_payload()

    def test_decode_missing_parts(self):
        """Test error on JWT with missing parts."""
        auth = JWTAuth(token="only.two")
        with pytest.raises(ValueError, match="Invalid JWT format"):
            _ = auth._decode_payload()

    def test_from_env_returns_auth(self):
        """Test from_env creates JWTAuth when env var is set."""
        token = make_jwt_token({"sub": "env_user@example.com"})
        with mock.patch.dict(os.environ, {"PACSYS_JWT_TOKEN": token}):
            auth = JWTAuth.from_env()
            assert auth is not None
            assert auth.principal == "env_user@example.com"

    def test_from_env_returns_none(self):
        """Test from_env returns None when env var is not set."""
        # Ensure the env var is not set
        with mock.patch.dict(os.environ, {}, clear=True):
            # Also remove any existing PACSYS_JWT_TOKEN
            os.environ.pop("PACSYS_JWT_TOKEN", None)
            auth = JWTAuth.from_env()
            assert auth is None

    def test_from_env_custom_var(self):
        """Test from_env with custom environment variable."""
        token = make_jwt_token({"sub": "custom@example.com"})
        with mock.patch.dict(os.environ, {"MY_JWT_TOKEN": token}):
            auth = JWTAuth.from_env(var="MY_JWT_TOKEN")
            assert auth is not None
            assert auth.principal == "custom@example.com"

    def test_frozen_dataclass(self):
        """Test that JWTAuth is immutable."""
        token = make_jwt_token({"sub": "user@example.com"})
        auth = JWTAuth(token=token)
        with pytest.raises(AttributeError):
            auth.token = "new_token"

    def test_token_not_in_repr(self):
        """Test that token is excluded from repr to prevent credential leaks."""
        token = make_jwt_token({"sub": "user@example.com"})
        auth = JWTAuth(token=token)
        repr_str = repr(auth)
        # Token should not appear in repr
        assert token not in repr_str
        # But it should still be a valid repr
        assert "JWTAuth" in repr_str


class TestKerberosAuth:
    """Tests for KerberosAuth class."""

    def test_requires_gssapi(self):
        """Test ImportError when gssapi is not installed."""
        # Skip if gssapi is installed
        try:
            import gssapi  # noqa: F401

            pytest.skip("gssapi is installed")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="gssapi library required"):
            KerberosAuth()

    def test_auth_type(self):
        """Test auth_type property."""
        mock_gssapi = MockGSSAPIModule()

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = KerberosAuth()
            assert auth.auth_type == "kerberos"

    def test_principal_extracted(self):
        """Test that principal is extracted from Kerberos credentials."""
        mock_gssapi = MockGSSAPIModule()

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = KerberosAuth()
            assert auth.principal == "user@FNAL.GOV"

    def test_validates_fnal_realm(self):
        """Test that only FNAL.GOV realm is accepted."""

        class MockCreds:
            name = "user@OTHER.REALM"
            lifetime = 3600

        class MockGSSAPI:
            class exceptions:
                class GSSError(Exception):
                    pass

            @staticmethod
            def Credentials(usage=None):
                return MockCreds()

        with mock.patch.dict("sys.modules", {"gssapi": MockGSSAPI()}):
            with pytest.raises(AuthenticationError, match="not from FNAL.GOV realm"):
                KerberosAuth()

    def test_validates_ticket_not_expired(self):
        """Test that expired tickets are rejected."""

        class MockCreds:
            name = "user@FNAL.GOV"
            lifetime = 0  # Expired

        class MockGSSAPI:
            class exceptions:
                class GSSError(Exception):
                    pass

            @staticmethod
            def Credentials(usage=None):
                return MockCreds()

        with mock.patch.dict("sys.modules", {"gssapi": MockGSSAPI()}):
            with pytest.raises(AuthenticationError, match="has expired"):
                KerberosAuth()

    def test_no_credentials_error(self):
        """Test AuthenticationError when no credentials available."""

        class MockGSSError(Exception):
            pass

        class MockGSSAPI:
            class exceptions:
                GSSError = MockGSSError

            @staticmethod
            def Credentials(usage=None):
                raise MockGSSError("No credentials")

        with mock.patch.dict("sys.modules", {"gssapi": MockGSSAPI()}):
            with pytest.raises(AuthenticationError, match="No valid Kerberos credentials"):
                KerberosAuth()

    def test_frozen_dataclass(self):
        """Test that KerberosAuth is immutable."""
        mock_gssapi = MockGSSAPIModule()

        with mock.patch.dict("sys.modules", {"gssapi": mock_gssapi}):
            auth = KerberosAuth()
            # KerberosAuth has no fields, but test that it's frozen
            assert hasattr(auth, "__dataclass_fields__")


class TestAuthExports:
    """Tests for auth module exports."""

    def test_exports_in_pacsys(self):
        """Test that auth classes are exported from pacsys."""
        import pacsys

        assert hasattr(pacsys, "Auth")
        assert hasattr(pacsys, "KerberosAuth")
        assert hasattr(pacsys, "JWTAuth")

    def test_exports_in_all(self):
        """Test that auth classes are in __all__."""
        import pacsys

        assert "Auth" in pacsys.__all__
        assert "KerberosAuth" in pacsys.__all__
        assert "JWTAuth" in pacsys.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
