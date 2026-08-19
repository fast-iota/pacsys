"""
Integration tests for ACL backend against the real CGI endpoint.

The ACL CGI endpoint (www-bd.fnal.gov) is only accessible from the Fermilab
network. These tests will auto-skip when the endpoint is unreachable.

NOTE: acl.pl is blocked externally - proxy access will be needed for
CI/remote testing.
"""

import math

import pytest

from pacsys.backends.acl import ACLBackend
from pacsys.errors import DeviceError
from pacsys.types import Reading

from .conftest import requires_acl
from .devices import (
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    SCALAR_DEVICE_3,
    TIMEOUT_BATCH,
    TIMEOUT_READ,
)

# =============================================================================
# URL format verification (offline - no server needed)
# =============================================================================


class TestACLURLFormat:
    """Verify URL construction matches ACL CGI protocol."""

    def test_single_device_url(self):
        """read command uses space (+) before device, no semicolons."""
        backend = ACLBackend()
        url = backend._build_url(["M:OUTTMP"])
        assert "?acl=read+M:OUTTMP" in url
        assert "\\;" not in url
        backend.close()

    def test_batch_url_uses_semicolons(self):
        """Multiple devices use \\; separated read commands."""
        backend = ACLBackend()
        url = backend._build_url(["M:OUTTMP", "G:AMANDA"])
        assert "read+M:OUTTMP\\;read+G:AMANDA" in url
        backend.close()

    def test_default_base_url(self):
        """Default URL points to www-bd.fnal.gov."""
        backend = ACLBackend()
        assert "www-bd.fnal.gov" in backend.base_url
        assert "acl.pl" in backend.base_url
        backend.close()


# =============================================================================
# Live read tests (require ACL endpoint access)
# =============================================================================


@requires_acl
class TestACLLiveReads:
    """Live tests against the ACL CGI endpoint."""

    def test_read_scalar(self, acl_backend):
        """Read a known scalar device."""
        value = acl_backend.read(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(value)

    def test_get_scalar(self, acl_backend):
        """Get returns a Reading with metadata."""
        reading = acl_backend.get(SCALAR_DEVICE, timeout=TIMEOUT_READ)
        assert isinstance(reading, Reading)
        assert reading.ok
        assert isinstance(reading.value, (int, float)) and not isinstance(reading.value, bool)
        assert math.isfinite(reading.value)

    def test_get_many(self, acl_backend):
        """Batch read multiple devices."""
        drfs = [SCALAR_DEVICE, SCALAR_DEVICE_2]
        readings = acl_backend.get_many(drfs, timeout=TIMEOUT_BATCH)
        assert len(readings) == 2
        ok_count = sum(1 for r in readings if r.ok)
        assert ok_count >= 1, f"Expected at least 1 success, got {ok_count}"
        assert all(
            isinstance(r.value, (int, float)) and not isinstance(r.value, bool) and math.isfinite(r.value)
            for r in readings
            if r.ok
        )

    def test_get_many_mixed_valid_invalid(self, acl_backend):
        """Batch with valid and invalid devices falls back and isolates errors."""
        drfs = [SCALAR_DEVICE, "Z:DOESNOTEXIST99", SCALAR_DEVICE_2]
        readings = acl_backend.get_many(drfs, timeout=TIMEOUT_BATCH)
        assert len(readings) == 3
        # First and third should succeed, middle should error
        assert readings[0].ok
        assert isinstance(readings[0].value, (int, float)) and not isinstance(readings[0].value, bool)
        assert math.isfinite(readings[0].value)
        assert readings[1].is_error
        assert readings[2].ok
        assert isinstance(readings[2].value, (int, float)) and not isinstance(readings[2].value, bool)
        assert math.isfinite(readings[2].value)

    def test_read_nonexistent_device(self, acl_backend):
        """Reading a nonexistent device should error."""
        with pytest.raises(DeviceError):
            acl_backend.read("Z:DOESNOTEXIST99", timeout=TIMEOUT_READ)

    def test_read_alarm_field(self, acl_backend):
        """Read .MAX alarm field - returns numeric value."""
        value = acl_backend.read(f"{SCALAR_DEVICE_3}.MAX", timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(value)

    def test_read_z_acltst(self, acl_backend):
        """Read the OAC test device Z:ACLTST."""
        value = acl_backend.read(SCALAR_DEVICE_3, timeout=TIMEOUT_READ)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(value)
