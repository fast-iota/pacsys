"""
Authentication - KerberosAuth (DPM) and JWTAuth (gRPC).

Auth objects validate at construction (fail fast) and are reusable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import base64
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class Auth(ABC):
    """Base class for authentication credentials.

    Authentication objects are reusable across different backends and services.
    They represent "who you are" (identity), not "what you can do" (authorization).
    """

    @property
    @abstractmethod
    def auth_type(self) -> str:
        """Type identifier (e.g., 'kerberos', 'jwt')."""
        ...

    @property
    @abstractmethod
    def principal(self) -> str:
        """Identity name (e.g., user@FNAL.GOV or JWT subject)."""
        ...


@dataclass(frozen=True)
class KerberosAuth(Auth):
    """Kerberos authentication using system credential cache.

    Requires:
        - Valid Kerberos ticket (run `kinit` first)
        - gssapi library (`pip install gssapi`)

    The credentials are obtained from the system credential cache,
    so the same KerberosAuth instance can be used across multiple
    backends and services.

    Example:
        auth = KerberosAuth()
        print(f"Authenticated as: {auth.principal}")

        # Use with DPM backend
        backend = DPMHTTPBackend(auth=auth, role="testing")

        # Same auth could be used with other services
        # ssh_client = SSHClient(auth=auth)  # future

    Note:
        Credentials are validated at construction time (fail fast).
    """

    def __post_init__(self):
        """Validate credentials at construction time (fail fast)."""
        # This will raise ImportError or AuthenticationError if invalid
        _ = self._get_credentials()

    @property
    def auth_type(self) -> str:
        return "kerberos"

    @property
    def principal(self) -> str:
        """Get principal name from credential cache."""
        creds = self._get_credentials()
        return str(creds.name)

    def _get_credentials(self):
        """Get and validate Kerberos credentials from cache.

        Returns:
            gssapi.Credentials object

        Raises:
            ImportError: If gssapi library is not installed
            AuthenticationError: If no valid credentials or wrong realm
        """
        try:
            import gssapi
        except ImportError:
            raise ImportError("gssapi library required for Kerberos authentication. Install with: pip install gssapi")

        from pacsys.errors import AuthenticationError

        try:
            creds = gssapi.Credentials(usage="initiate")
        except gssapi.exceptions.GSSError as e:
            raise AuthenticationError(f"No valid Kerberos credentials. Run 'kinit' first. Error: {e}")

        principal_parts = str(creds.name).split("@")
        if len(principal_parts) != 2:
            raise AuthenticationError(f"Invalid Kerberos principal format: {creds.name}")

        if principal_parts[1] != "FNAL.GOV":
            raise AuthenticationError(f"Kerberos principal {principal_parts[0]} is not from FNAL.GOV realm")

        if creds.lifetime <= 0:
            raise AuthenticationError(f"Kerberos ticket for {principal_parts[0]} has expired")

        return creds


@dataclass(frozen=True)
class JWTAuth(Auth):
    """JWT token authentication.

    Can be used with gRPC backend and other JWT-accepting services.
    Token is stored as-is; validation is done server-side.

    Example:
        # Explicit token
        auth = JWTAuth(token="eyJ...")

        # From environment variable
        auth = JWTAuth.from_env()  # reads PACSYS_JWT_TOKEN

        # Use with gRPC backend
        backend = GRPCBackend(auth=auth)

    Note:
        Token is excluded from __repr__ to prevent credential leaks in logs.
    """

    token: str = field(repr=False)

    @property
    def auth_type(self) -> str:
        return "jwt"

    @property
    def principal(self) -> str:
        """Extract subject claim from JWT token.

        Note: This decodes without verification. Server validates the token.

        Returns:
            Subject ('sub') claim from token

        Raises:
            ValueError: If token format is invalid
        """
        payload = self._decode_payload()
        sub = payload.get("sub")
        if sub is None:
            raise ValueError("JWT token has no 'sub' claim")
        return sub

    def _decode_payload(self) -> dict:
        """Decode JWT payload without verification."""
        parts = self.token.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid JWT format: expected 3 parts, got {len(parts)}")

        payload_b64 = parts[1]
        # Add padding if needed (base64url doesn't always have padding)
        padding = len(payload_b64) % 4
        if padding > 0:
            payload_b64 += "=" * (4 - padding)

        try:
            decoded = base64.urlsafe_b64decode(payload_b64)
            return json.loads(decoded)
        except Exception as e:
            raise ValueError(f"Failed to decode JWT payload: {e}")

    @classmethod
    def from_env(cls, var: str = "PACSYS_JWT_TOKEN") -> Optional["JWTAuth"]:
        """Create JWTAuth from environment variable.

        Args:
            var: Environment variable name (default: PACSYS_JWT_TOKEN)

        Returns:
            JWTAuth if environment variable is set, None otherwise
        """
        token = os.environ.get(var)
        if token:
            return cls(token=token)
        return None


__all__ = ["Auth", "KerberosAuth", "JWTAuth"]
