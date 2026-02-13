"""
High-level exceptions for pacsys API.

These exceptions are raised by the user-facing API (read, get, write, etc.)
rather than the low-level ACNET protocol layer.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pacsys.types import Reading


class ReadError(Exception):
    """Transport failure during a batch read.

    Raised when get_many() encounters a client-side transport error
    (timeout, connection drop, auth failure) - NOT for server-side ACNET
    errors like DIO_NOATT which are returned as error Readings.

    Attributes:
        readings: Complete results list (same length as input drfs).
            Includes both successful readings and error-backfilled entries.
        message: Human-readable description of the transport failure.

    The underlying network exception is chained via ``__cause__``
    (standard ``raise ReadError(...) from cause`` pattern).
    """

    def __init__(self, readings: "list[Reading]", message: str):
        self.readings = readings
        super().__init__(message)

    def __repr__(self) -> str:
        ok = sum(1 for r in self.readings if r.ok)
        total = len(self.readings)
        return f"ReadError({ok}/{total} ok, {self.args[0]!r})"


class DeviceError(Exception):
    """Raised when a device read fails.

    Attributes:
        drf: Device request string that failed
        facility_code: ACNET facility identifier (1=ACNET, 16=DBM, 17=DPM)
        error_code: Error code (negative indicates error, positive is warning)
        message: Human-readable error description
    """

    def __init__(
        self,
        drf: str,
        facility_code: int,
        error_code: int,
        message: Optional[str] = None,
    ):
        self.drf = drf
        self.facility_code = facility_code
        self.error_code = error_code
        self.message = message

        if message:
            super().__init__(f"{drf}: {message} (facility={facility_code}, error={error_code})")
        else:
            super().__init__(f"{drf}: error (facility={facility_code}, error={error_code})")

    def __repr__(self) -> str:
        return (
            f"DeviceError(drf={self.drf!r}, facility_code={self.facility_code}, "
            f"error_code={self.error_code}, message={self.message!r})"
        )


class AuthenticationError(Exception):
    """Raised when authentication fails or is required.

    This exception is raised when:
    - A write is attempted without authentication
    - Kerberos ticket is expired or missing
    - JWT token is invalid or expired
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __repr__(self) -> str:
        return f"AuthenticationError({self.message!r})"


class ACLError(Exception):
    """Raised when an ACL command fails.

    This exception is raised when:
    - A one-shot ACL command exits with non-zero status
    - An ACL session receives an error response
    - The ACL prompt times out
    - The ACL process exits unexpectedly
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
