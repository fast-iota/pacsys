"""
ACNET error codes.

Error codes follow the pattern: FACILITY + (error_number * 256)
where error_number is signed (-128 to +127) and facility is 1-255.
"""

from enum import IntEnum


class AcnetFacility(IntEnum):
    """ACNET facility codes."""

    ACNET = 1  # Core ACNET errors
    DIO = 14  # Device I/O
    FTP = 15  # Fast Time Plot
    DBM = 16  # Database Manager
    DPM = 17  # Data Pool Manager


def make_error(facility: int, error_number: int) -> int:
    """Create an error code from facility and error number."""
    return facility + (error_number * 256)


def parse_error(code: int) -> tuple[int, int]:
    """Parse error code into facility and error number."""
    facility = code & 0xFF
    error_number = (code >> 8) & 0xFF
    if error_number > 127:
        error_number -= 256
    return facility, error_number


# ACNET facility errors
ACNET_OK = 0
ACNET_SUCCESS = 0
ACNET_DEPRECATED = make_error(1, 4)  # Used a deprecated feature
ACNET_REPLY_TIMEOUT = make_error(1, 3)  # Reply timeout (not fatal)
ACNET_ENDMULT = make_error(1, 2)  # End multiple replies
ACNET_PEND = make_error(1, 1)  # Operation pending

# Negative error codes (failures)
ACNET_RETRY = make_error(1, -1)  # Retryable I/O error
ACNET_NOLCLMEM = make_error(1, -2)  # No local memory
ACNET_NOREMMEM = make_error(1, -3)  # No remote memory
ACNET_RPLYPACK = make_error(1, -4)  # Reply packet assembly error
ACNET_REQPACK = make_error(1, -5)  # Request packet assembly error
ACNET_REQTMO = make_error(1, -6)  # Request timeout (6.5 min)
ACNET_QUEFULL = make_error(1, -7)  # Destination queue full
ACNET_BUSY = make_error(1, -8)  # Destination task busy
ACNET_NOT_CONNECTED = make_error(1, -21)  # Not connected to network
ACNET_ARG = make_error(1, -22)  # Missing argument(s)
ACNET_IVM = make_error(1, -23)  # Invalid message length/buffer
ACNET_NO_SUCH = make_error(1, -24)  # No such request or reply
ACNET_REQREJ = make_error(1, -25)  # Request rejected
ACNET_CANCELLED = make_error(1, -26)  # Request cancelled
ACNET_NAME_IN_USE = make_error(1, -27)  # Task name already in use
ACNET_NCR = make_error(1, -28)  # Not connected as RUM task
ACNET_NO_NODE = make_error(1, -30)  # No such logical node
ACNET_TRUNC_REQUEST = make_error(1, -31)  # Truncated request
ACNET_TRUNC_REPLY = make_error(1, -32)  # Truncated reply
ACNET_NO_TASK = make_error(1, -33)  # No such destination task
ACNET_DISCONNECTED = make_error(1, -34)  # Replier disconnected
ACNET_LEVEL2 = make_error(1, -35)  # Level II function error
ACNET_HARD_IO = make_error(1, -36)  # Hard I/O error
ACNET_NODE_DOWN = make_error(1, -42)  # Node offline
ACNET_UTIME = make_error(1, -49)  # User timeout
ACNET_INVARG = make_error(1, -50)  # Invalid argument

# DBM facility errors (Database Manager)
DBM_NOPROP = make_error(16, -13)  # Property not found

# DPM facility errors
DPM_PEND = make_error(17, 1)  # Request pending
DPM_STALE = make_error(17, 2)  # Stale data warning
DPM_BAD_REQUEST = make_error(17, -24)  # Malformed request
DPM_NO_SUCH_DEVICE = make_error(17, -26)  # Device not found
DPM_NO_SUCH_PROP = make_error(17, -27)  # Property not found
DPM_BAD_RANGE = make_error(17, -28)  # Invalid range
DPM_NO_SCALE = make_error(17, -31)  # Scaling not available
DPM_BAD_EVENT = make_error(17, -33)  # Invalid event
DPM_INTERNAL_ERROR = make_error(17, -45)  # Internal error

# Decomposed error numbers (signed int8) for use with Reading/WriteResult fields.
# These match the error_number component of the composite constants above.
FACILITY_ACNET = int(AcnetFacility.ACNET)  # 1
FACILITY_DBM = int(AcnetFacility.DBM)  # 16
ERR_OK = 0
ERR_RETRY = -1  # Generic retryable error (error number of ACNET_RETRY)
ERR_TIMEOUT = -6  # Request timeout (error number of ACNET_REQTMO)
ERR_NOPROP = -13  # Property not found (error number of DBM_NOPROP)


def normalize_error_code(code: int) -> int:
    """Normalize unsigned error code to signed int8 convention.

    Backends receive error codes as unsigned values (uint8/uint32).
    ACNET convention: negative=error, 0=ok, positive=warning.
    Values > 127 are negative when interpreted as signed int8.
    """
    if code > 127:
        return code - 256
    return code


def status_message(facility: int, error: int) -> str | None:
    """Build human-readable status message from decomposed error codes.

    Returns None for success (error == 0).
    """
    if error < 0:
        return f"Device error (facility={facility}, error={error})"
    elif error > 0:
        return f"Warning (facility={facility}, error={error})"
    return None


class AcnetError(Exception):
    """Exception for ACNET protocol errors."""

    def __init__(self, status: int, message: str | None = None):
        self.status = status
        facility, error_num = parse_error(status)
        self.facility = facility
        self.error_number = error_num
        if message:
            super().__init__(f"ACNET error {status:#06x}: {message}")
        else:
            super().__init__(f"ACNET error {status:#06x}")

    def __repr__(self):
        return f"AcnetError(status={self.status:#06x})"


class AcnetUnavailableError(AcnetError):
    """Exception when ACNET daemon is unavailable."""

    def __init__(self):
        super().__init__(ACNET_NOT_CONNECTED, "ACNET daemon unavailable")


class AcnetTimeoutError(AcnetError):
    """Exception when a request times out."""

    def __init__(self, timeout_ms: int | None = None):
        msg = f"timeout after {timeout_ms}ms" if timeout_ms else "timeout"
        super().__init__(ACNET_REQTMO, msg)


class AcnetNodeError(AcnetError):
    """Exception when a node is not found."""

    def __init__(self, node: str | int):
        super().__init__(ACNET_NO_NODE, f"node not found: {node}")


class AcnetTaskError(AcnetError):
    """Exception when a task is not found."""

    def __init__(self, task: str):
        super().__init__(ACNET_NO_TASK, f"task not found: {task}")
