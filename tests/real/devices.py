"""
Shared test device configurations and utilities for real/integration tests.

This module is the SINGLE SOURCE OF TRUTH for:
- Server availability checks
- Skip markers (requires_dpm_http, requires_dpm_acnet, requires_grpc, requires_acl)
- Test device constants
- Helper functions

All real tests should import from this module.
"""

import os
import socket

import pytest

from pacsys.acnet.errors import ERR_TIMEOUT
from pacsys.types import BasicControl, Reading, ValueType


# =============================================================================
# Server Availability Checks
# =============================================================================


def dpm_server_available() -> bool:
    """Check if DPM server (acsys-proxy) is reachable."""
    try:
        sock = socket.create_connection(("acsys-proxy.fnal.gov", 6802), timeout=2.0)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError, socket.gaierror):
        return False


def grpc_server_available() -> bool:
    """Check if gRPC server is reachable at localhost:23456."""
    try:
        sock = socket.create_connection(("localhost", 23456), timeout=2.0)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


ACL_TEST_URL = "https://localhost:10443/cgi-bin/acl.pl"


def acl_server_available() -> bool:
    """Check if ACL CGI endpoint is reachable (test proxy at localhost:10443)."""
    try:
        import ssl
        import urllib.request

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(
            f"{ACL_TEST_URL}?acl=read+M:OUTTMP",
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=3.0, context=ctx):
            return True
    except Exception:
        return False


def dmq_server_available() -> bool:
    """Check if RabbitMQ broker is reachable at localhost:5672."""
    try:
        sock = socket.create_connection(("localhost", 5672), timeout=2.0)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError, socket.gaierror):
        return False


def acnet_tcp_server_available() -> bool:
    """Check if acnetd is reachable via TCP (localhost:6802 or PACSYS_DPM_HOST)."""
    host = os.environ.get("PACSYS_DPM_HOST", "localhost")
    port = int(os.environ.get("PACSYS_DPM_PORT", "34567"))
    try:
        sock = socket.create_connection((host, port), timeout=2.0)
        sock.sendall(b"RAW\r\n\r\n")
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError, socket.gaierror):
        return False


# =============================================================================
# Skip Markers
# =============================================================================

requires_dpm_http = pytest.mark.skipif(
    not dpm_server_available(),
    reason="DPM/HTTP server not available at acsys-proxy.fnal.gov:6802",
)

requires_dpm_acnet = pytest.mark.skipif(
    not dpm_server_available(),
    reason="DPM/ACNET server not available at acsys-proxy.fnal.gov:6802",
)

requires_grpc = pytest.mark.skipif(
    not grpc_server_available(),
    reason="gRPC server not available at localhost:23456 (set up tunnel first)",
)

requires_acl = pytest.mark.skipif(
    not acl_server_available(),
    reason="ACL CGI endpoint not available",
)

requires_dmq = pytest.mark.skipif(
    not dmq_server_available(),
    reason="DMQ/RabbitMQ broker not available at localhost:5672",
)

requires_acnet_tcp = pytest.mark.skipif(
    not acnet_tcp_server_available(),
    reason="acnetd not reachable via TCP (set PACSYS_DPM_HOST/PORT or start acnetd)",
)


# =============================================================================
# Timeout Constants
# =============================================================================

# Standard timeouts for different operation types
# Server latency is expected <100ms for valid devices, but error responses
# (e.g., DPM_PEND for nonexistent devices) can take ~3s
TIMEOUT_READ = 10.0  # Single device read (includes error response delay)
TIMEOUT_BATCH = 10.0  # Batch read (may include slow error responses)
TIMEOUT_THREAD_JOIN = 6.0  # Thread join for concurrent operations
TIMEOUT_STREAM_EVENT = 2.0  # Wait for streaming event
TIMEOUT_STREAM_ITER = 3.0  # Streaming iterator timeout

# Expected fast response time for VALID devices (not error cases)
# First request may take longer (channel warmup), network adds variability
EXPECTED_FAST_RESPONSE = 1.0  # Valid device reads should complete within this
EXPECTED_ERROR_RESPONSE = 4.0  # Error responses (DPM_PEND) may take longer


# =============================================================================
# Test Devices - Scalar
# =============================================================================

SCALAR_DEVICE = "M:OUTTMP"
SCALAR_DEVICE_2 = "G:AMANDA"
SCALAR_DEVICE_3 = "Z:ACLTST"
SCALAR_ELEMENT = "B:IRMS06[0]"

# =============================================================================
# Test Devices - Array
# =============================================================================

ARRAY_DEVICE = "B:IRMS06[0:10]"

# =============================================================================
# Test Devices - Properties
# =============================================================================

DESCRIPTION_DEVICE = "M~OUTTMP"
RAW_DEVICE = "M:OUTTMP.RAW"
STATUS_DEVICE = "N|LGXS"
ANALOG_ALARM_DEVICE = "N@H801"
DIGITAL_ALARM_DEVICE = "N$H801"

# =============================================================================
# Allowed Write Devices
# =============================================================================
# Safety: ONLY these devices may be written to in tests.
# conftest.py enforces this -- any write to an unlisted device raises an error.
ALLOWED_WRITE_DEVICES = frozenset(
    [
        "G:AMANDA",
        "Z:ACLTS1",
        "Z:ACLTS2",
        "Z:ACLTS3",
        "Z:ACLTS4",
        "Z:ACLTS5",
        "Z:ACLTS6",
        "Z:ACLTS7",
        "Z:ACLTS8",
        "Z:ACLTST",
        "Z:CUBE",
        "Z:CUBE_X",
        "Z:CUBE_Y",
        "Z:CUBE_Z",
        "Z:REMSE2",
        "Z:RICH",
        "Z:SNDBOXSTJOHN",
        "Z:STRINGTEST",
        "Z:STRINGTEST2",
        "Z:STRTST",
    ]
)

# =============================================================================
# Test Devices - Settings
# =============================================================================

SCALAR_SETPOINT = "Z_ACLTST"
SCALAR_SETPOINT_RAW = "Z:ACLTST.SETTING.RAW"
ARRAY_SETPOINT = "Z_ACLTS3[0:3]"
ARRAY_SETPOINT_RAW = "Z:ACLTS3.SETTING[0:3].RAW"
ANALOG_ALARM_SETPOINT = "Z@ACLTST"

# =============================================================================
# Test Devices - Control (on/off)
# =============================================================================

# Z:ACLTST is an OAC (Open Access Client) device -- software-emulated.
# Its DB control attribute table has 12 commands, but ordinals 7-9 map to
# TEST/TEST2/TEST3 -- NOT to LOCAL/REMOTE/TRIP:
#
#   ordinal 0 (RESET)    → value 3   (RESET)
#   ordinal 1 (ON)       → value 1   (ON)
#   ordinal 2 (OFF)      → value 2   (OFF)
#   ordinal 3 (POSITIVE) → value 4   (POSITIVE)
#   ordinal 4 (NEGATIVE) → value 16  (NEGATIVE)
#   ordinal 5 (RAMP)     → value 5   (RAMP)
#   ordinal 6 (DC)       → value 6   (DC)
#   ordinal 7 (LOCAL)    → value 8   (TEST)      ← no remote status bit
#   ordinal 8 (REMOTE)   → value 256 (TEST2)     ← no remote status bit
#   ordinal 9 (TRIP)     → value 257 (TEST3)     ← does NOT toggle ready
#
# Status bits (DB): on (0x1), ready (0x2), positive (0x8), ramp (0x200).
# No "remote" status bit exists on this device.
#
# STATUS qualifier (Z|ACLTST) reads the device status bits.
# CONTROL qualifier (Z&ACLTST) writes control commands.
STATUS_CONTROL_DEVICE = "Z|ACLTST"
CONTROL_ON = BasicControl.ON
CONTROL_OFF = BasicControl.OFF
CONTROL_POSITIVE = BasicControl.POSITIVE
CONTROL_NEGATIVE = BasicControl.NEGATIVE
CONTROL_RAMP = BasicControl.RAMP
CONTROL_DC = BasicControl.DC
CONTROL_RESET = BasicControl.RESET
CONTROL_LOCAL = BasicControl.LOCAL
CONTROL_REMOTE = BasicControl.REMOTE
CONTROL_TRIP = BasicControl.TRIP

# Unpaired control ordinals on Z:ACLTST -- ordinals that map to device-specific
# TEST commands rather than standard BasicControl actions. These write successfully
# but don't toggle any standard status bit. See ordinal table above.
# (ordinal, device_command_name)
ACLTST_UNPAIRED_CONTROLS = [
    (BasicControl.LOCAL, "TEST"),  # ordinal 7 → value 8
    (BasicControl.REMOTE, "TEST2"),  # ordinal 8 → value 256
    (BasicControl.TRIP, "TEST3"),  # ordinal 9 → value 257
    (10, "TEST4"),  # ordinal 10 → value 258 (beyond BasicControl enum)
    (11, "TEST5"),  # ordinal 11 → value 259 (beyond BasicControl enum)
]

# Ordinal with no entry in Z:ACLTST's 12-command control table.
ACLTST_NONEXISTENT_ORDINAL = 25

# (command_true, command_false, status_field) -- each pair toggles a status bit.
# Only pairs whose ordinals map to real control actions on Z:ACLTST.
# REMOTE/LOCAL and RESET/TRIP are excluded -- see ordinal table above.
# Unpaired ordinals are tested separately in test_dpm_http_backend.py.
CONTROL_PAIRS = [
    (BasicControl.ON, BasicControl.OFF, "on"),
    (BasicControl.POSITIVE, BasicControl.NEGATIVE, "positive"),
    (BasicControl.RAMP, BasicControl.DC, "ramp"),
]

# =============================================================================
# Test Devices - Error Cases
# =============================================================================

NONEXISTENT_DEVICE = "Z:NOTFND"  # DPM_PEND: facility=17, error=1 (warning, no data, arrives with delay)
NOPROP_DEVICE = "N$HBLK2V"  # DBM_NOPROP: facility=16, error=-13
SETTING_ON_READONLY = "B_IRMS06[0]"  # SETTING property on read-only device

# =============================================================================
# Test Devices - Streaming
# =============================================================================

PERIODIC_DEVICE = "M:OUTTMP@p,500"
FAST_PERIODIC = "M:OUTTMP@p,300"
SLOW_PERIODIC = "M:OUTTMP@p,1000"

# FTP (Fast Time Plot) - routed via DPM's <-FTP extra qualifier
FTP_DEVICE = "M:OUTTMP@p,100H<-FTP"

# Logger - historical data via DPM's <-LOGGER extra qualifier
# 1 hour window: 2025-01-15 12:00–13:00 UTC (epoch ms)
LOGGER_DEVICE = "M:OUTTMP<-LOGGER:1736942400000:1736946000000"
LOGGER_DEVICE_WITH_EVENT = "M:OUTTMP@P,1000,true<-LOGGER:1736942400000:1736946000000"

# =============================================================================
# Alarm Value Transformations
# =============================================================================
# Raw alarm blocks store values in native encoding. DPM applies database
# transformations to convert to engineering units. These are the known
# transformations for our test devices:
#
# G:AMANDA (SCALAR_DEVICE_2):
#   Primary: X = IEEE native floating point
#   Common:  X' = X
#   Result:  raw float bytes ARE the engineering value
#
# Z:ACLTST:
#   Primary: X = input (input DEC float; output native float)
#   Common:  X' = C1*X/C2 with C1 = C2 = 1
#   Result:  raw DEC F_floating bytes converted to IEEE float
#
# N:H801 (ANALOG_ALARM_DEVICE):
#   Primary: FLOAT(input)/3276.8
#   Common:  (C1*X/C2) + C3 with C1=C2=0 and C3=0
#   Result:  raw_uint16 / 3276.8 = engineering value
#
# M:OUTTMP (SCALAR_DEVICE):
#   (transformation unknown - skip value comparison)


def alarm_raw_to_engineering(device: str, raw_value: int | float) -> float | None:
    """Convert raw alarm block value to engineering units for known devices.

    Args:
        device: Device name (e.g., "G:AMANDA", "N:H801")
        raw_value: Raw value from alarm block (already decoded based on data_type)

    Returns:
        Engineering value, or None if transformation unknown
    """
    import struct

    device = device.upper()

    if device == "G:AMANDA":
        # IEEE float - raw bytes are the value (need to reinterpret if passed as int)
        if isinstance(raw_value, float):
            return raw_value
        # raw_value is int representation of float bytes
        return struct.unpack("<f", struct.pack("<I", raw_value & 0xFFFFFFFF))[0]

    if device == "Z:ACLTST":
        # DEC F_floating to IEEE float conversion
        # DEC format: sign(1) | exponent(8) excess-128 | mantissa(23) with hidden 0.1
        if isinstance(raw_value, float):
            return raw_value
        word0 = raw_value & 0xFFFF
        word1 = (raw_value >> 16) & 0xFFFF
        if word0 == 0:
            return 0.0
        sign = (word0 >> 15) & 1
        exp = (word0 >> 7) & 0xFF
        frac_hi = word0 & 0x7F
        frac_lo = word1
        mantissa = (frac_hi << 16) | frac_lo
        # DEC: value = (0.5 + mantissa/2^24) * 2^(exp-128)
        value = (0.5 + mantissa / (1 << 24)) * (2 ** (exp - 128))
        return -value if sign else value

    if device == "N:H801":
        # Signed 16-bit int / 3276.8 (transform uses signed despite fe_data saying unsigned)
        if raw_value >= 32768:
            raw_value = raw_value - 65536
        return float(raw_value) / 3276.8

    # Unknown device - can't convert
    return None


# =============================================================================
# Device Type Test Matrix
# =============================================================================

# Device configurations: (drf, expected_type, python_type, description)
# Used by parametrized tests in backend test files
DEVICE_TYPES = [
    ("M:OUTTMP", ValueType.SCALAR, float, "scalar reading"),
    ("G:AMANDA", ValueType.SCALAR, float, "scalar reading 2"),
    ("B:IRMS06[0:10]", ValueType.SCALAR_ARRAY, None, "array reading"),
    ("B:IRMS06[0]", ValueType.SCALAR, float, "array element"),
    ("M:OUTTMP.RAW", ValueType.RAW, bytes, "raw property"),
    ("M~OUTTMP", ValueType.TEXT, str, "description property"),
    ("N|LGXS", ValueType.BASIC_STATUS, dict, "status property"),
    ("N@H801", ValueType.ANALOG_ALARM, dict, "analog alarm"),
    ("N$H801", ValueType.DIGITAL_ALARM, dict, "digital alarm"),
    ("Z_ACLTST@I", ValueType.SCALAR, float, "scalar setpoint"),
    ("Z_ACLTS3[0:3]@I", ValueType.SCALAR_ARRAY, None, "array setpoint"),
    ("Z:ACLTS3.SETTING[0:3].RAW", ValueType.RAW, bytes, "array setpoint raw"),
]


# =============================================================================
# Kerberos / Write Markers
# =============================================================================


def kerberos_available() -> bool:
    """Check if valid Kerberos credentials are available."""
    try:
        from pacsys.auth import KerberosAuth

        auth = KerberosAuth()
        _ = auth.principal
        return True
    except Exception:
        return False


requires_kerberos = pytest.mark.skipif(
    not kerberos_available(),
    reason="Valid Kerberos credentials required (run kinit first)",
)


# =============================================================================
# SSH Server Config & Availability
# =============================================================================

SSH_JUMP_HOST = os.environ.get("PACSYS_TEST_SSH_JUMP", "")
SSH_DEST_HOST = os.environ.get("PACSYS_TEST_SSH_DEST", "")


def ssh_jump_available() -> bool:
    """Check if SSH jump host is configured and reachable."""
    if not SSH_JUMP_HOST:
        return False
    try:
        sock = socket.create_connection((SSH_JUMP_HOST, 22), timeout=3.0)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError, socket.gaierror):
        return False


requires_ssh = pytest.mark.skipif(
    not SSH_JUMP_HOST or not SSH_DEST_HOST,
    reason="SSH tests require PACSYS_TEST_SSH_JUMP and PACSYS_TEST_SSH_DEST env vars (see tests/real/.env.ssh)",
)

ACL_JUMP_HOST = os.environ.get("PACSYS_TEST_ACL_JUMP", "")
ACL_DEST_HOST = os.environ.get("PACSYS_TEST_ACL_DEST", "")

requires_acl_ssh = pytest.mark.skipif(
    not ACL_JUMP_HOST or not ACL_DEST_HOST,
    reason="ACL-over-SSH tests require PACSYS_TEST_ACL_JUMP and PACSYS_TEST_ACL_DEST env vars (see tests/real/.env.ssh)",
)

requires_write_enabled = pytest.mark.skipif(
    not os.environ.get("PACSYS_TEST_WRITE"),
    reason="Set PACSYS_TEST_WRITE=1 to enable write tests",
)


# =============================================================================
# Timing Helpers
# =============================================================================


def assert_fast_response(elapsed: float, operation: str, threshold: float = EXPECTED_FAST_RESPONSE):
    """Assert operation completed quickly (not hitting timeout)."""
    assert elapsed < threshold, (
        f"{operation} took {elapsed:.2f}s, expected <{threshold}s. Server may be slow or operation may be timing out."
    )


def assert_not_timeout_error(reading: Reading, drf: str):
    """Assert reading is not a timeout error (checks ACNET error code, not message string)."""
    if not reading.ok:
        assert reading.error_code != ERR_TIMEOUT, f"Operation on {drf} timed out: {reading.message}"


# =============================================================================
# Helper Functions
# =============================================================================


def format_value(value):
    """Format value for display in test output."""
    if isinstance(value, bytes):
        return f"{value!r} (len={len(value)})"
    if hasattr(value, "__len__") and not isinstance(value, (str, dict)):
        return f"array(len={len(value)})"
    if isinstance(value, dict):
        return f"dict({len(value)} keys)"
    return str(value)
