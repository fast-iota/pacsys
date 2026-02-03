"""
ACL Backend implementation.

Provides read-only access to ACNET devices via the ACL CGI endpoint.
This is the simplest backend - just HTTP GET requests.

Usage:
    from pacsys.backends.acl import ACLBackend

    with ACLBackend() as backend:
        temp = backend.read("M:OUTTMP")
        reading = backend.get("M:OUTTMP")
        readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
"""

import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Optional

from pacsys.acnet.errors import ERR_OK, ERR_RETRY, ERR_TIMEOUT
from pacsys.backends import Backend
from pacsys.errors import DeviceError
from pacsys.types import (
    BackendCapability,
    Reading,
    Value,
    ValueType,
)

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_BASE_URL = "https://www-ad.fnal.gov/cgi-bin/acl.pl"
DEFAULT_TIMEOUT = 5.0


def _parse_acl_value(text: str) -> tuple[Value, ValueType]:
    """
    Parse ACL response text into a value and type.

    ACL returns plain text which we try to interpret:
    - Try parsing as float first
    - If that fails, return as string

    Args:
        text: Raw response text from ACL

    Returns:
        Tuple of (value, ValueType)
    """
    text = text.strip()

    # Try parsing as float
    try:
        value = float(text)
        return value, ValueType.SCALAR
    except ValueError:
        pass

    # Check for array-like response (whitespace or newline separated numbers)
    parts = text.split()
    if len(parts) > 1:
        try:
            values = [float(p) for p in parts]
            return values, ValueType.SCALAR_ARRAY
        except ValueError:
            pass

    # Return as text
    return text, ValueType.TEXT


def _is_error_response(text: str) -> tuple[bool, Optional[str]]:
    """
    Check if ACL response indicates an error.

    Args:
        text: Raw response text from ACL

    Returns:
        Tuple of (is_error, error_message)
    """
    text = text.strip()

    # ACL errors typically start with "!" or contain "error"
    if text.startswith("!"):
        return True, text[1:].strip()
    if "error" in text.lower():
        return True, text

    return False, None


class ACLBackend(Backend):
    """
    ACL backend for HTTP-based device reads (read-only).

    Uses the ACL CGI endpoint for simple device access.
    No streaming, no authentication, no writes.

    Capabilities:
        - READ: Always enabled
        - WRITE: No (read-only backend)
        - STREAM: No (HTTP is request/response only)
        - AUTH: No (public endpoint)
        - BATCH: Yes (get_many supported)

    Example:
        with ACLBackend() as backend:
            temp = backend.read("M:OUTTMP")
            reading = backend.get("M:OUTTMP")
            readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize ACL backend.

        Args:
            base_url: ACL CGI URL (default: https://www-ad.fnal.gov/cgi-bin/acl.pl)
            timeout: HTTP request timeout in seconds (default: 5.0)

        Raises:
            ValueError: If parameters are invalid
        """
        effective_url = base_url if base_url is not None else DEFAULT_BASE_URL
        effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        if not effective_url:
            raise ValueError("base_url cannot be empty")
        if effective_timeout <= 0:
            raise ValueError(f"timeout must be positive, got {effective_timeout}")

        self._base_url = effective_url
        self._timeout = effective_timeout
        self._closed = False

        logger.debug(f"ACLBackend initialized: base_url={effective_url}, timeout={effective_timeout}")

    @property
    def capabilities(self) -> BackendCapability:
        """ACL only supports READ and BATCH."""
        return BackendCapability.READ | BackendCapability.BATCH

    @property
    def base_url(self) -> str:
        """ACL CGI base URL."""
        return self._base_url

    @property
    def timeout(self) -> float:
        """Default request timeout."""
        return self._timeout

    def _build_url(self, drfs: list[str]) -> str:
        """
        Build ACL URL for one or more devices.

        Args:
            drfs: List of device request strings

        Returns:
            Complete URL with query parameters
        """
        # ACL format: acl=read/device1+device2+device3
        devices = "+".join(urllib.parse.quote(drf, safe="") for drf in drfs)
        return f"{self._base_url}?acl=read/{devices}"

    def _fetch(self, url: str, timeout: float) -> str:
        """
        Fetch URL content.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Response text

        Raises:
            DeviceError: If HTTP request fails
        """
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise DeviceError(
                drf="",
                facility_code=0,
                error_code=ERR_RETRY,
                message=f"ACL request failed ({url}): HTTP {e.code} {e.reason}",
            )
        except urllib.error.URLError as e:
            raise DeviceError(
                drf="",
                facility_code=0,
                error_code=ERR_RETRY,
                message=f"ACL request failed ({self._base_url}): {e.reason}",
            )
        except TimeoutError:
            raise DeviceError(
                drf="",
                facility_code=0,
                error_code=ERR_TIMEOUT,
                message=f"ACL request timed out after {timeout}s ({self._base_url})",
            )

    def read(self, drf: str, timeout: Optional[float] = None) -> Value:
        """
        Read a single device value via HTTP.

        Args:
            drf: Device request string
            timeout: Request timeout in seconds (None=default)

        Returns:
            The device value

        Raises:
            RuntimeError: If backend is closed
            DeviceError: If the read fails
        """
        reading = self.get(drf, timeout=timeout)

        # Raise if not ok: handles both negative errors AND positive status without data
        if not reading.ok:
            raise DeviceError(
                drf=reading.drf,
                facility_code=reading.facility_code,
                error_code=reading.error_code,
                message=reading.message or f"Read failed with status {reading.error_code}",
            )

        assert reading.value is not None
        return reading.value

    def get(self, drf: str, timeout: Optional[float] = None) -> Reading:
        """
        Read a single device with metadata via HTTP.

        Args:
            drf: Device request string
            timeout: Request timeout in seconds (None=default)

        Returns:
            Reading object
        """
        readings = self.get_many([drf], timeout=timeout)
        return readings[0]

    def get_many(self, drfs: list[str], timeout: Optional[float] = None) -> list[Reading]:
        """
        Read multiple devices via HTTP.

        Args:
            drfs: List of device request strings
            timeout: Total timeout for request (None=default)

        Returns:
            List of Reading objects in same order as input
        """
        if self._closed:
            raise RuntimeError("Backend is closed")

        if not drfs:
            return []

        effective_timeout = timeout if timeout is not None else self._timeout

        # Build URL and fetch
        url = self._build_url(drfs)
        logger.debug(f"ACL request: {url}")

        try:
            response_text = self._fetch(url, effective_timeout)
        except DeviceError as e:
            # Return error readings for all devices
            return [
                Reading(
                    drf=drf,
                    value_type=ValueType.SCALAR,
                    facility_code=e.facility_code,
                    error_code=e.error_code,
                    message=e.message,
                    timestamp=datetime.now(),
                )
                for drf in drfs
            ]

        logger.debug(f"ACL response: {response_text[:200]}...")

        # Parse response - ACL returns one value per line for multiple devices
        lines = response_text.strip().split("\n")

        readings: list[Reading] = []
        now = datetime.now()

        for i, drf in enumerate(drfs):
            if i < len(lines):
                line = lines[i]

                # Check for error
                is_error, error_msg = _is_error_response(line)
                if is_error:
                    readings.append(
                        Reading(
                            drf=drf,
                            value_type=ValueType.SCALAR,
                            facility_code=0,
                            error_code=ERR_RETRY,
                            message=error_msg,
                            timestamp=now,
                        )
                    )
                else:
                    # Parse value
                    value, value_type = _parse_acl_value(line)
                    readings.append(
                        Reading(
                            drf=drf,
                            value_type=value_type,
                            value=value,
                            error_code=ERR_OK,
                            timestamp=now,
                        )
                    )
            else:
                # No response for this device
                readings.append(
                    Reading(
                        drf=drf,
                        value_type=ValueType.SCALAR,
                        error_code=ERR_RETRY,
                        message="No response from ACL",
                        timestamp=now,
                    )
                )

        return readings

    def close(self) -> None:
        """Close the backend. No resources to clean up for HTTP client."""
        self._closed = True
        logger.debug("ACLBackend closed")

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"ACLBackend({self._base_url}, timeout={self._timeout}, {status})"


__all__ = ["ACLBackend"]
