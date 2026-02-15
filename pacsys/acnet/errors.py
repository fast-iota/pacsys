"""
ACNET error codes.

Error codes follow the pattern: FACILITY + (error_number * 256)
where error_number is signed (-128 to +127) and facility is 1-255.
"""

from enum import IntEnum


class AcnetFacility(IntEnum):
    """ACNET facility codes."""

    ACNET = 1  # Core ACNET errors
    FSHARE = 2  # File sharing
    APM = 3  # Application program manager
    LJ = 4  # Local journaling
    CBS = 5  # Console basic services
    CMU = 6  # Console management utilities
    CLIB = 7  # C library
    LOCK = 8  # Lock manager
    APPDS = 9  # Application data services
    DIO = 14  # Device I/O
    FTP = 15  # Fast Time Plot
    DBM = 16  # Database Manager
    DPM = 17  # Data Pool Manager
    DMQ = 72  # Data Multiplexer Queue (RabbitMQ-based)


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
ACNET_SYS = make_error(1, -43)  # System service error
ACNET_NXE = make_error(1, -44)  # Untranslatable error
ACNET_BUG = make_error(1, -45)  # Network internal error
ACNET_NE1 = make_error(1, -46)  # Network error #1
ACNET_NE2 = make_error(1, -47)  # Network error #2
ACNET_NE3 = make_error(1, -48)  # Network error #3
ACNET_UTIME = make_error(1, -49)  # User timeout
ACNET_INVARG = make_error(1, -50)  # Invalid argument
ACNET_MEMFAIL = make_error(1, -51)  # Dynamic memory allocation failed
ACNET_NO_HANDLE = make_error(1, -52)  # Requested ACNET handle does not exist

# DIO facility errors (Device I/O, facility 14)
# Positive status codes (warnings)
DIO_NOT_SCALABLE = make_error(14, 12)  # Value not scalable as floating point
DIO_NOT_UNIQUE = make_error(14, 11)  # Request did not have unique result
DIO_TOO_HIGH = make_error(14, 10)  # Value too high (clipped to max)
DIO_TOO_LOW = make_error(14, 9)  # Value too low (clipped to min)
DIO_INCOMPLETE_DATA = make_error(14, 8)  # Some but not all data returned
DIO_INCONSISTENT_TIMES = make_error(14, 7)  # Data has inconsistent timestamps
DIO_NO_CHANGE = make_error(14, 6)  # Data has not changed
DIO_MORE_VALUES = make_error(14, 5)  # Additional values available
DIO_NOW_EMPTY = make_error(14, 4)  # Object is now empty
DIO_PEND = make_error(14, 3)  # No new data available
DIO_ALREADY_SET = make_error(14, 2)  # Value already at requested setting
DIO_TOO_MANY_ARGS = make_error(14, 1)  # Too many arguments
# Negative error codes
DIO_MEMFAIL = make_error(14, -1)  # Memory allocation failed
DIO_CANFAIL = make_error(14, -2)  # Cancel of list/device failed
DIO_NOLIST = make_error(14, -3)  # Nonexistent list
DIO_STALE = make_error(14, -4)  # Data same as previous call
DIO_INVATT = make_error(14, -5)  # Device attribute does not exist
DIO_NOATT = make_error(14, -6)  # Attribute doesn't exist for device
DIO_NOSCALE = make_error(14, -7)  # No scaling information
DIO_BADARG = make_error(14, -8)  # Invalid argument
DIO_BADSCALE = make_error(14, -9)  # Invalid PDB (scaling) data
DIO_NOFAMILY = make_error(14, -10)  # Not a family device
DIO_NOINFO = make_error(14, -11)  # No database-information node
DIO_INVDBDAT = make_error(14, -12)  # DBM error for device
DIO_INVLEN = make_error(14, -13)  # Invalid raw data length
DIO_SETDIS = make_error(14, -14)  # Setting inhibited
DIO_SMOFF = make_error(14, -15)  # Stepping motor is off
DIO_SMITER = make_error(14, -16)  # Stepping motor iteration limit
DIO_NO_SUCH = make_error(14, -17)  # Object does not exist
DIO_UNAVAIL = make_error(14, -18)  # Service unavailable
DIO_INVDEV = make_error(14, -19)  # Invalid device
DIO_SCALEFAIL = make_error(14, -20)  # Scaling failed
DIO_NOTYET = make_error(14, -21)  # Feature not yet implemented
DIO_MISMATCH = make_error(14, -22)  # Data does not match request
DIO_TOO_MANY = make_error(14, -28)  # Too many objects
DIO_GENERIC_ERROR = make_error(14, -29)  # Generic error
DIO_BUGCHK = make_error(14, -32)  # Internal program bug
DIO_CORRUPT = make_error(14, -33)  # Data corrupted; math exception in scaling
DIO_INSUFFICIENT_ARGS = make_error(14, -34)  # Not enough arguments
DIO_INVDATA = make_error(14, -35)  # Invalid data in context
DIO_INVOFF = make_error(14, -36)  # Invalid raw data offset
DIO_DUPREQ = make_error(14, -37)  # Duplicate request
DIO_TRUNCATED = make_error(14, -38)  # Result truncated
DIO_SYNTAX = make_error(14, -39)  # Syntax error
DIO_TOO_BIG = make_error(14, -40)  # Request too large
DIO_NOT_ENABLED = make_error(14, -41)  # Feature not enabled
DIO_INVPROP = make_error(14, -42)  # Invalid property in context
DIO_TIMEOUT = make_error(14, -43)  # Request timed out
DIO_MATH_EXCEPTION = make_error(14, -46)  # Math exception during calculation
DIO_NOT_PROCESSED = make_error(14, -47)  # Entry was not processed
DIO_RETIRED = make_error(14, -48)  # Service has been retired
DIO_INVALID_RATE = make_error(14, -49)  # Invalid data rate
DIO_RANGE = make_error(14, -58)  # Value out of range
DIO_NOPRIV = make_error(14, -59)  # No privilege for action
DIO_READONLY = make_error(14, -60)  # Read-only access
DIO_EMPTY = make_error(14, -65)  # Empty request
DIO_NOT_SUPPORTED = make_error(14, -69)  # Operation not supported
DIO_AMBIGUOUS = make_error(14, -70)  # Ambiguous request
DIO_NOT_A_NUMBER = make_error(14, -71)  # Data value is NaN
DIO_RECURSION_LIMIT = make_error(14, -72)  # Recursion limit reached
DIO_INVALID_DEVICE_TYPE = make_error(14, -73)  # Invalid device type
DIO_TOO_SMALL = make_error(14, -74)  # Request too small
DIO_INVALID_DB_REQUEST = make_error(14, -76)  # Invalid database request (SQL error)
DIO_UNTRANSLATABLE_ERROR = make_error(14, -77)  # Untranslatable low-level error
DIO_NOREGULAR = make_error(14, -78)  # Not a regular device (family device)
DIO_SETONLY = make_error(14, -79)  # Unreadable property in readable context
DIO_OUT_OF_BOUNDS = make_error(14, -80)  # Array index out of bounds
DIO_CONTROLLED_SET = make_error(14, -81)  # Attempt to set controlled property
DIO_NO_DATA = make_error(14, -82)  # No data for this request

# DBM facility errors (Database Manager)
DBM_NOPROP = make_error(16, -13)  # Property not found

# FTP facility errors (Fast Time Plot, facility 15)
FACILITY_FTP = int(AcnetFacility.FTP)  # 15

# Positive status codes (informational, not errors)
FTP_COLLECTING = make_error(15, 4)  # Snapshot data collection in progress
FTP_WAIT_DELAY = make_error(15, 3)  # Snapshot armed, waiting for time delay
FTP_WAIT_EVENT = make_error(15, 2)  # Snapshot armed, waiting for arm event
FTP_PEND = make_error(15, 1)  # Snapshot pending (setup accepted)

# Negative error codes (actual failures)
FTP_INVTYP = make_error(15, -1)  # Invalid request typecode (software bug)
FTP_INVSSDN = make_error(15, -2)  # Invalid SSDN from database
FTP_FE_OUTOFMEM = make_error(15, -5)  # Front-end out of memory
FTP_NOCHAN = make_error(15, -6)  # No more MADC plot channels available
FTP_NO_DECODER = make_error(15, -7)  # No more MADC clock decoders
FTP_FE_PLOTLIM = make_error(15, -8)  # Front-end plot limit exceeded
FTP_INVNUMDEV = make_error(15, -9)  # Invalid number of devices in request
FTP_ENDOFDATA = make_error(15, -10)  # End of data
FTP_FE_PLOTLEN = make_error(15, -11)  # Front-end buffer length computation error
FTP_INVREQLEN = make_error(15, -12)  # Invalid request length (software bug)
FTP_NO_DATA = make_error(15, -13)  # No data from MADC (transient or hardware)
FTP_INVREQ = make_error(15, -14)  # Snapshot retrieval doesn't match active setup
FTP_BADEV = make_error(15, -15)  # Wrong set of clock events
FTP_BUMPED = make_error(15, -16)  # Bumped by higher priority plot
FTP_REROUTE = make_error(15, -17)  # Internal front-end reroute error
FTP_UNSFREQ = make_error(15, -19)  # Unsupported frequency (FRIG: only 1 KHz)
FTP_BIGDLY = make_error(15, -20)  # Delay too long (FRIG: max 16.384s)
FTP_UNSDEV = make_error(15, -21)  # Unsupported device type (FRIG: ADC only)
FTP_SOFTWARE = make_error(15, -22)  # Internal front-end software error
FTP_NOTRDY = make_error(15, -23)  # Snapshot data not yet ready (FRIG)
FTP_ARCNET = make_error(15, -24)  # ARCNET communication error (FRIG)
FTP_BADARM = make_error(15, -25)  # Bad arm value, can't decode arm word
FTP_INVFREQ_FOR_HARDWARE = make_error(15, -26)  # Frequency unsupported by hardware
FTP_BAD_PLOT_MODE = make_error(15, -27)  # Bad plot mode in arm/trigger word
FTP_NO_SUCH_DEVICE = make_error(15, -28)  # Device not found for retrieval
FTP_DEVICE_IN_USE = make_error(15, -29)  # Device already has active retrieval
FTP_FREQ_TOO_HIGH = make_error(15, -30)  # Frequency exceeds front-end capability
FTP_NO_SETUP = make_error(15, -31)  # No matching setup for retrieval/restart
FTP_UNSUPPORTED_PROP = make_error(15, -32)  # Unsupported property
FTP_INVALID_CHANNEL = make_error(15, -33)  # Channel in SSDN doesn't exist
FTP_NO_FIFO = make_error(15, -34)  # Missing FIFO board (FRIG)
FTP_BAD_DATA_LENGTH = make_error(15, -35)  # Data length not 2 or 4 (class bug)
FTP_BUFFER_OVERFLOW = make_error(15, -36)  # Front-end buffer overflow
FTP_NO_EVENT_SUPPORT = make_error(15, -37)  # Event-triggered sampling unsupported
FTP_TRIGGER_ERROR = make_error(15, -38)  # Internal trigger definition error
FTP_INV_CLASS_DEF = make_error(15, -39)  # Invalid class definition (software bug)
FTP_NO_RANDOM_ACCESS = make_error(15, -40)  # Random access not yet supported
FTP_INVALID_OFFSET = make_error(15, -41)  # Non-zero data offset unsupported
FTP_NO_SNAPSHOT = make_error(15, -42)  # Device doesn't support snapshot plots
FTP_EVENT_UNAVAILABLE = make_error(15, -43)  # Clock event not decoded by front-end
FTP_NO_FTPMAN_INIT = make_error(15, -44)  # FTPMAN not initialized (need class query first)
FTP_BADTIMES = make_error(15, -100)  # UCD module timestamp disagreement
FTP_BADRESETS = make_error(15, -101)  # Device timestamp didn't reset properly
FTP_BADARG = make_error(15, -102)  # Invalid argument in ACNET request
FTP_BADRPY = make_error(15, -103)  # Invalid reply from front-end

# FTP status code descriptions (composite code -> message)
_FTP_STATUS_MESSAGES = {
    FTP_COLLECTING: "collecting data",
    FTP_WAIT_DELAY: "waiting for arm delay",
    FTP_WAIT_EVENT: "waiting for arm event",
    FTP_PEND: "snapshot pending",
    FTP_INVTYP: "invalid request typecode",
    FTP_INVSSDN: "invalid SSDN",
    FTP_FE_OUTOFMEM: "front-end out of memory",
    FTP_NOCHAN: "no available MADC plot channels",
    FTP_NO_DECODER: "no available clock decoders",
    FTP_FE_PLOTLIM: "front-end plot limit exceeded",
    FTP_INVNUMDEV: "invalid number of devices",
    FTP_ENDOFDATA: "end of data",
    FTP_FE_PLOTLEN: "buffer length computation error",
    FTP_INVREQLEN: "invalid request length",
    FTP_NO_DATA: "no data from MADC",
    FTP_INVREQ: "retrieval doesn't match active setup",
    FTP_BADEV: "wrong set of clock events",
    FTP_BUMPED: "bumped by higher priority plot",
    FTP_REROUTE: "internal reroute error",
    FTP_UNSFREQ: "unsupported frequency",
    FTP_BIGDLY: "delay too long",
    FTP_UNSDEV: "unsupported device type",
    FTP_SOFTWARE: "internal software error",
    FTP_NOTRDY: "data not ready",
    FTP_ARCNET: "ARCNET communication error",
    FTP_BADARM: "bad arm value",
    FTP_INVFREQ_FOR_HARDWARE: "frequency unsupported by hardware",
    FTP_BAD_PLOT_MODE: "bad plot mode",
    FTP_NO_SUCH_DEVICE: "device not found for retrieval",
    FTP_DEVICE_IN_USE: "device already has active retrieval",
    FTP_FREQ_TOO_HIGH: "frequency exceeds front-end capability",
    FTP_NO_SETUP: "no matching setup for retrieval/restart",
    FTP_UNSUPPORTED_PROP: "unsupported property",
    FTP_INVALID_CHANNEL: "channel doesn't exist on device",
    FTP_NO_FIFO: "missing FIFO board",
    FTP_BAD_DATA_LENGTH: "invalid data length (expected 2 or 4)",
    FTP_BUFFER_OVERFLOW: "front-end buffer overflow",
    FTP_NO_EVENT_SUPPORT: "event-triggered sampling unsupported",
    FTP_TRIGGER_ERROR: "trigger definition error",
    FTP_INV_CLASS_DEF: "invalid class definition",
    FTP_NO_RANDOM_ACCESS: "random access not supported",
    FTP_INVALID_OFFSET: "non-zero data offset unsupported",
    FTP_NO_SNAPSHOT: "device doesn't support snapshots",
    FTP_EVENT_UNAVAILABLE: "clock event not available on front-end",
    FTP_NO_FTPMAN_INIT: "FTPMAN not initialized (send class query first)",
    FTP_BADTIMES: "UCD module timestamp error",
    FTP_BADRESETS: "device timestamp reset error",
    FTP_BADARG: "invalid argument",
    FTP_BADRPY: "invalid reply from front-end",
}


def ftp_status_message(composite_status: int) -> str:
    """Return human-readable message for an FTP composite status code.

    Works for both positive (informational) and negative (error) codes.
    """
    msg = _FTP_STATUS_MESSAGES.get(composite_status)
    if msg:
        return msg
    facility, error_num = parse_error(composite_status)
    if facility != FACILITY_FTP:
        return f"non-FTP status (facility={facility}, error={error_num})"
    return f"unknown FTP status (error={error_num})"


# DPM facility errors (Data Pool Manager, facility 17)
# Positive status codes (warnings)
DPM_LARGE_LIST = make_error(17, 3)  # Suspiciously large device list
DPM_STALE = make_error(17, 2)  # Stale data warning
DPM_PEND = make_error(17, 1)  # Request pending
# Negative error codes
DPM_RESTART_TOO_FAST = make_error(17, -1)  # Restarting acquisition too fast
DPM_INVFCN = make_error(17, -2)  # Invalid function code
DPM_INVRINX = make_error(17, -3)  # Invalid IRINX
DPM_IVRXNM = make_error(17, -4)  # IRINX belongs to another task
DPM_NOTPROC = make_error(17, -5)  # IRINX not processed
DPM_MARKDELET = make_error(17, -6)  # Entry marked for deletion
DPM_MXLBIG = make_error(17, -7)  # DPGET maxlen too big
DPM_OFFLEN = make_error(17, -8)  # Error in offset or length
DPM_OUTOMEM = make_error(17, -9)  # Out of memory
DPM_TMOSET = make_error(17, -10)  # Setting reply timeout from front end
DPM_LENSML = make_error(17, -11)  # Requested length too small
DPM_DPMSER = make_error(17, -12)  # System error (DBM or SMLDPM down)
DPM_ILLFTD = make_error(17, -13)  # Illegal frequency (faster than 15 Hz)
DPM_SETBIG = make_error(17, -14)  # Setting longer than one packet
DPM_NOSET = make_error(17, -15)  # Setting not allowed from this console
DPM_SETLOCK = make_error(17, -16)  # Console is locked for settings
DPM_PRIV = make_error(17, -17)  # Insufficient privileges for setting
DPM_DBPRIV = make_error(17, -18)  # Console class not allowed to set device
DPM_REDIRECT = make_error(17, -19)  # Redirect not in 'Redirect Ok' state
DPM_WHACKEDSETS = make_error(17, -20)  # Settings disabled (safety interlock)
DPM_NODEFAULT = make_error(17, -21)  # No default value for device
DPM_DUPLICATES = make_error(17, -22)  # Duplicate requests (resource leak)
DPM_NO_TCLK = make_error(17, -23)  # Front end not receiving TCLK events
DPM_BAD_REQUEST = make_error(17, -24)  # Malformed request
DPM_LOOKUP_FAILED = make_error(17, -25)  # Device lookup failed
DPM_NO_SUCH_DEVICE = make_error(17, -26)  # Device not found
DPM_NO_SUCH_PROP = make_error(17, -27)  # Property not found
DPM_BAD_RANGE = make_error(17, -28)  # Invalid array range
DPM_OUT_OF_BOUNDS = make_error(17, -29)  # Array range out of bounds
DPM_BAD_FRAMING = make_error(17, -30)  # Range not multiple of atomic size
DPM_NO_SCALE = make_error(17, -31)  # Scaling not available
DPM_NO_SUCH_FIELD = make_error(17, -32)  # Invalid DRF field
DPM_BAD_EVENT = make_error(17, -33)  # Invalid event format
DPM_BAD_DEF_EVENT = make_error(17, -34)  # No valid default event
DPM_BAD_LENGTH = make_error(17, -35)  # Bad data length for device
DPM_SCALING_FAILED = make_error(17, -36)  # Scaling failed
DPM_NO_SUCH_LIST = make_error(17, -37)  # List ID not found
DPM_SERVICE_NOT_FOUND = make_error(17, -38)  # DPM service not found
DPM_CALLBACK_NOT_FOUND = make_error(17, -39)  # Callback not found
DPM_DOC_DEVICE = make_error(17, -40)  # Documentation-only device
DPM_GETS32_DISABLED = make_error(17, -41)  # 32-bit gets disabled
DPM_INVALID_DATASOURCE = make_error(17, -42)  # Invalid datasource
DPM_BAD_DATASOURCE_FORMAT = make_error(17, -43)  # Bad datasource format
DPM_REPLY_OVERFLOW = make_error(17, -44)  # Reply too large
DPM_INTERNAL_ERROR = make_error(17, -45)  # Internal error

# DMQ facility errors (Data Multiplexer Queue, facility 72)
FACILITY_DMQ = int(AcnetFacility.DMQ)  # 72
DMQ_PENDING = make_error(72, 1)  # Data acquisition pending initialization
DMQ_OK = make_error(72, 0)  # Normal condition
DMQ_INVALID_DATA_TYPE = make_error(72, -93)  # Invalid data type in setting job
DMQ_SETTING_DISABLED = make_error(72, -94)  # Settings disabled
DMQ_SYSTEM_ERROR = make_error(72, -95)  # Unchecked exception on server
DMQ_CHANNEL_NOT_READY = make_error(72, -96)  # Communication channel not ready
DMQ_LOGIN_REQUIRED = make_error(72, -97)  # Login required
DMQ_INVALID_REQUEST = make_error(72, -98)  # Invalid data request
DMQ_SECURITY_VIOLATION = make_error(72, -99)  # Security violation (invalid credentials)

# Decomposed error numbers (signed int8) for use with Reading/WriteResult fields.
# These match the error_number component of the composite constants above.
FACILITY_ACNET = int(AcnetFacility.ACNET)  # 1
FACILITY_DIO = int(AcnetFacility.DIO)  # 14
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


# Reverse lookup tables for named error messages.
# Built lazily on first use from the module-level constants.
_STATUS_NAMES: dict[int, str] | None = None

# Human-readable descriptions for common error codes (composite code -> description).
_STATUS_DESCRIPTIONS: dict[int, str] = {
    # ACNET
    ACNET_RETRY: "retryable I/O error",
    ACNET_NOLCLMEM: "no local memory",
    ACNET_REQTMO: "request timeout",
    ACNET_QUEFULL: "destination queue full",
    ACNET_BUSY: "destination task busy",
    ACNET_NOT_CONNECTED: "not connected to network",
    ACNET_INVARG: "invalid argument",
    ACNET_NO_SUCH: "no such request or reply",
    ACNET_REQREJ: "request rejected",
    ACNET_CANCELLED: "request cancelled",
    ACNET_NO_NODE: "no such logical node",
    ACNET_NO_TASK: "no such destination task",
    ACNET_DISCONNECTED: "replier disconnected",
    ACNET_NODE_DOWN: "node offline",
    ACNET_UTIME: "user timeout",
    # DIO
    DIO_NOATT: "attribute doesn't exist for device",
    DIO_NOSCALE: "no scaling information",
    DIO_BADARG: "invalid argument",
    DIO_NO_SUCH: "object does not exist",
    DIO_UNAVAIL: "service unavailable",
    DIO_INVDEV: "invalid device",
    DIO_SCALEFAIL: "scaling failed",
    DIO_SETDIS: "setting inhibited",
    DIO_NOPRIV: "no privilege for action",
    DIO_READONLY: "read-only access",
    DIO_TIMEOUT: "request timed out",
    DIO_RANGE: "value out of range",
    DIO_NOT_SUPPORTED: "operation not supported",
    DIO_OUT_OF_BOUNDS: "array index out of bounds",
    DIO_CONTROLLED_SET: "attempt to set controlled property",
    DIO_NO_DATA: "no data for this request",
    # DBM
    DBM_NOPROP: "property not found",
    # DPM
    DPM_LARGE_LIST: "suspiciously large device list (possible resource leak)",
    DPM_STALE: "data returned more than one clock cycle late",
    DPM_PEND: "request pending, no data yet",
    DPM_RESTART_TOO_FAST: "restarting data acquisition too fast",
    DPM_OUTOMEM: "DPM out of memory",
    DPM_TMOSET: "setting reply timeout from front end",
    DPM_DPMSER: "DPM system error (DBM or SMLDPM down)",
    DPM_NOSET: "setting not allowed from this console",
    DPM_SETLOCK: "console is locked for settings",
    DPM_PRIV: "insufficient privileges for setting",
    DPM_DBPRIV: "console class not allowed to set this device",
    DPM_REDIRECT: "field not in 'Redirect Ok' state",
    DPM_WHACKEDSETS: "settings disabled (safety interlock)",
    DPM_NO_TCLK: "front end not receiving TCLK events",
    DPM_BAD_REQUEST: "malformed request",
    DPM_LOOKUP_FAILED: "device lookup service not found",
    DPM_NO_SUCH_DEVICE: "device not found",
    DPM_NO_SUCH_PROP: "property not found for device",
    DPM_BAD_RANGE: "invalid array range",
    DPM_OUT_OF_BOUNDS: "array range out of bounds",
    DPM_NO_SCALE: "scaling not available for device",
    DPM_NO_SUCH_FIELD: "invalid DRF field specifier",
    DPM_BAD_EVENT: "invalid event format",
    DPM_BAD_DEF_EVENT: "device has no valid default event",
    DPM_BAD_LENGTH: "bad data length for device",
    DPM_SCALING_FAILED: "scaling failed",
    DPM_NO_SUCH_LIST: "list ID not found",
    DPM_SERVICE_NOT_FOUND: "DPM service not found",
    DPM_DOC_DEVICE: "documentation-only device (no live data)",
    DPM_INVALID_DATASOURCE: "invalid datasource",
    DPM_REPLY_OVERFLOW: "reply too large, split into smaller requests",
    DPM_INTERNAL_ERROR: "internal DPM error",
    # DMQ
    DMQ_INVALID_DATA_TYPE: "invalid data type in setting",
    DMQ_SETTING_DISABLED: "settings disabled",
    DMQ_SYSTEM_ERROR: "unchecked exception on server",
    DMQ_CHANNEL_NOT_READY: "communication channel not ready",
    DMQ_LOGIN_REQUIRED: "login required",
    DMQ_INVALID_REQUEST: "invalid data request",
    DMQ_SECURITY_VIOLATION: "security violation (invalid credentials)",
}


def _build_status_names() -> dict[int, str]:
    """Build composite-code -> name map from module constants."""
    names: dict[int, str] = {}
    for name, val in globals().items():
        if isinstance(val, int) and name.isupper() and "_" in name:
            prefix = name.split("_")[0]
            if prefix in ("ACNET", "DIO", "DPM", "DBM", "DMQ"):
                names[val] = name
    return names


def status_message(facility: int, error: int) -> str | None:
    """Build human-readable status message from decomposed error codes.

    Returns None for success (error == 0).
    Format: "DPM_PRIV: insufficient privileges for setting (facility=17, error=-17)"
    """
    if error == 0:
        return None

    global _STATUS_NAMES
    if _STATUS_NAMES is None:
        _STATUS_NAMES = _build_status_names()

    composite = make_error(facility, error)
    name = _STATUS_NAMES.get(composite)
    desc = _STATUS_DESCRIPTIONS.get(composite)
    if name:
        if desc:
            return f"{name}: {desc} (facility={facility}, error={error})"
        kind = "warning" if error > 0 else "error"
        return f"{name}: {kind} (facility={facility}, error={error})"
    if error < 0:
        return f"Device error (facility={facility}, error={error})"
    return f"Warning (facility={facility}, error={error})"


class AcnetError(Exception):
    """Exception for ACNET protocol errors."""

    def __init__(self, status: int, message: str | None = None):
        self.status = status
        facility, error_num = parse_error(status)
        self.facility = facility
        self.error_number = error_num
        tag = f"ACNET error [{facility} {error_num}]"
        if message:
            super().__init__(f"{tag}: {message}")
        else:
            super().__init__(tag)

    def __repr__(self):
        return f"AcnetError(facility={self.facility}, error={self.error_number})"


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


class AcnetRequestRejectedError(AcnetError):
    """Exception when acnetd rejects a request to a restricted task.

    acnetd can be configured with -r to reject TCP client requests
    to specific task handles (e.g., FTPMAN). This error provides a
    clear message identifying the rejected task.
    """

    def __init__(self, task: str):
        self.task = task
        super().__init__(
            ACNET_REQREJ,
            f"request to task '{task}' rejected by acnetd (task is on the TCP reject list, see acnetd -r flag)",
        )
