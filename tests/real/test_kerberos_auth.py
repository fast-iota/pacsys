"""
Integration tests for Kerberos authentication.

Tests KerberosAuth against real system Kerberos credentials.

Requires:
- gssapi library installed
- Valid Kerberos ticket (run `kinit` first)
- Ticket from FNAL.GOV realm
"""

import subprocess

import pytest


def has_gssapi():
    """Check if gssapi library is installed."""
    try:
        import gssapi  # noqa: F401

        return True
    except ImportError:
        return False


def has_valid_ticket():
    """Check if user has a valid Kerberos ticket via klist."""
    try:
        result = subprocess.run(
            ["klist", "-s"],  # -s = silent, exit code only
            capture_output=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_principal_from_klist():
    """Get principal name from klist output."""
    try:
        result = subprocess.run(
            ["klist"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if "Default principal:" in line:
                return line.split(":", 1)[1].strip()
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# Skip conditions
requires_gssapi = pytest.mark.skipif(not has_gssapi(), reason="gssapi library not installed")
requires_ticket = pytest.mark.skipif(not has_valid_ticket(), reason="No valid Kerberos ticket (run kinit first)")


@requires_gssapi
@requires_ticket
@pytest.mark.kerberos
class TestRealKerberosAuth:
    """Tests using real system Kerberos credentials."""

    def test_kerberos_auth_retrieves_real_credentials(self):
        """Test that KerberosAuth can retrieve real credentials from cache."""
        from pacsys.auth import KerberosAuth
        from pacsys.errors import AuthenticationError

        principal_from_klist = get_principal_from_klist()

        # Check if principal is from FNAL.GOV realm
        if principal_from_klist and "@FNAL.GOV" not in principal_from_klist:
            pytest.skip(f"Ticket is not from FNAL.GOV realm: {principal_from_klist}")

        try:
            auth = KerberosAuth()
            assert auth.auth_type == "kerberos"
            assert auth.principal is not None
            assert "@FNAL.GOV" in auth.principal

            if principal_from_klist:
                assert auth.principal == principal_from_klist

            print(f"Successfully authenticated as: {auth.principal}")

        except AuthenticationError as e:
            if "expired" in str(e).lower():
                pytest.skip(f"Credential validation failed: {e}")
            raise


@requires_gssapi
@pytest.mark.kerberos
class TestKerberosWithoutTicket:
    """Tests for Kerberos behavior when no valid ticket exists."""

    def test_no_ticket_raises_auth_error(self, tmp_path, monkeypatch):
        """Test that missing ticket raises AuthenticationError."""
        # Point KRB5CCNAME at an empty file so gssapi finds no credentials,
        # regardless of whether the real user has a valid ticket.
        empty_ccache = tmp_path / "krb5cc_empty"
        empty_ccache.touch()
        monkeypatch.setenv("KRB5CCNAME", f"FILE:{empty_ccache}")

        from pacsys.auth import KerberosAuth
        from pacsys.errors import AuthenticationError

        with pytest.raises(AuthenticationError, match="No valid Kerberos credentials"):
            KerberosAuth()


if __name__ == "__main__":
    print(f"gssapi installed: {has_gssapi()}")
    print(f"Valid ticket: {has_valid_ticket()}")
    print(f"Principal: {get_principal_from_klist()}")
    print()

    pytest.main([__file__, "-v"])
