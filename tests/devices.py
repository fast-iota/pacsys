"""
Shared test device constants and mock helpers for unit tests.

This module provides a SINGLE SOURCE OF TRUTH for:
- Test device names and DRFs
- Common test values (temperatures, arrays, etc.)
- Mock socket helpers for DPM protocol testing
- Mock GSSAPI classes for Kerberos testing
- JWT token helpers

For integration tests with real servers, see tests/real/devices.py instead.
"""

import base64
import json
import socket
import struct

from pacsys.dpm_protocol import (
    ApplySettings_reply,
    Authenticate_reply,
    DeviceInfo_reply,
    OpenList_reply,
    Scalar_reply,
    ScalarArray_reply,
    SettingStatus_struct,
    StartList_reply,
    Status_reply,
    Text_reply,
)
from pacsys.types import ValueType


# =============================================================================
# Test Device Names
# =============================================================================

# Scalar devices
TEMP_DEVICE = "M:OUTTMP"  # Temperature sensor (canonical test device)
TEMP_DEVICE_2 = "G:AMANDA"  # Another scalar device

# Array devices
ARRAY_DEVICE = "B:HS23T"
ARRAY_DEVICE_RANGE = "B:HS23T[0:3]"

# Error/invalid devices
BAD_DEVICE = "M:BADDEV"
NONEXISTENT_DEVICE = "Z:NOTFND"

# Generic test devices (for value type tests)
GENERIC_SCALAR = "D:SCALAR"
GENERIC_ARRAY = "D:ARR"
GENERIC_RAW = "D:RAW"
GENERIC_TEXT = "D:TEXT"
GENERIC_TEXT_ARRAY = "D:TARR"


# =============================================================================
# Test Values
# =============================================================================

# Temperature values (Fahrenheit)
TEMP_VALUE = 72.5
TEMP_UNITS = "degF"
TEMP_DESCRIPTION = "Outdoor temperature"

# Secondary device value
AMANDA_VALUE = 1.234

# Array values
ARRAY_VALUES = [1.0, 2.0, 3.0]
ARRAY_5_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]

# Raw bytes
RAW_BYTES = b"\x00\x01\x02\x03\xff\xfe\xfd"

# Text values
TEXT_VALUE = "Temperature sensor"
TEXT_ARRAY_VALUES = ["Line 1", "Line 2", "Line 3"]

# Alarm data structures
ANALOG_ALARM_DATA = {
    "minimum": -10.0,
    "maximum": 100.0,
    "nominal": 72.5,
    "tolerance": 5.0,
    "status": "OK",
    "alarm_enable": True,
    "alarm_flags": 0,
}

DIGITAL_ALARM_DATA = {
    "nom_mask": 0xFF,
    "alarm_mask": 0x0F,
    "status": 0x00,
    "alarm_enable": True,
    "alarm_text": ["OFF", "ON", "FAULT", "UNKNOWN"],
}

BASIC_STATUS_DATA = {
    "on": True,
    "ready": True,
    "remote": True,
    "positive": False,
    "ramp": False,
}

# Timestamp (2024-01-01 00:00:00 UTC as milliseconds)
TIMESTAMP_MILLIS = 1704067200_000


# =============================================================================
# Error Codes
# =============================================================================

# Common error codes for testing
ERROR_NOT_FOUND = -42
ERROR_TIMEOUT = -1
ERROR_NO_READING = -1


# =============================================================================
# Mock Helpers for DPM Protocol Testing
# =============================================================================


def create_openlist_frame(list_id: int = 1) -> bytes:
    """Create a valid OpenList reply frame for mocking.

    Args:
        list_id: The list ID to include in the reply

    Returns:
        Length-prefixed frame bytes
    """
    reply = OpenList_reply()
    reply.list_id = list_id
    reply_data = bytes(reply.marshal())
    return struct.pack(">I", len(reply_data)) + reply_data


def create_reply_frame(reply) -> bytes:
    """Create a length-prefixed frame from a reply object.

    Args:
        reply: Any DPM protocol reply object with marshal() method

    Returns:
        Length-prefixed frame bytes
    """
    reply_data = bytes(reply.marshal())
    return struct.pack(">I", len(reply_data)) + reply_data


class MockSocketWithReplies:
    """Mock socket that returns predetermined DPM protocol replies.

    This class simulates a TCP socket connected to a DPM server. It
    automatically sends an OpenList reply on "connection", then returns
    each configured reply in order.

    Example:
        scalar = Scalar_reply()
        scalar.ref_id = 1
        scalar.data = 72.5

        mock_socket = MockSocketWithReplies(list_id=1, replies=[scalar])

        with mock.patch("socket.socket", return_value=mock_socket):
            backend = DPMHTTPBackend()
            reading = backend.get("M:OUTTMP")
    """

    def __init__(self, list_id: int = 1, replies: list = None):
        """Initialize mock socket.

        Args:
            list_id: List ID for OpenList reply
            replies: List of DPM reply objects to return
        """
        self.list_id = list_id
        self.replies = replies or []
        self._reply_index = 0
        self._recv_buffer = bytearray()
        self._connected = False
        self._closed = False
        self._timeout = None

        # Build initial OpenList frame
        self._recv_buffer.extend(create_openlist_frame(list_id))

    def connect(self, addr):
        """Simulate connection."""
        self._connected = True

    def close(self):
        """Simulate close."""
        self._closed = True
        self._connected = False

    def settimeout(self, timeout):
        """Set socket timeout."""
        self._timeout = timeout

    def gettimeout(self):
        """Get socket timeout."""
        return self._timeout

    def setsockopt(self, level, optname, value):
        """Set socket option (no-op)."""
        pass

    def sendall(self, data):
        """Simulate send (no-op, but raises if closed)."""
        if self._closed:
            raise BrokenPipeError("Socket closed")

    def recv(self, bufsize):
        """Receive next chunk of data.

        Returns data from internal buffer. When buffer is empty,
        adds the next reply from the replies list.
        """
        if self._closed:
            raise BrokenPipeError("Socket closed")

        # If buffer is empty, add next reply
        if not self._recv_buffer and self._reply_index < len(self.replies):
            reply = self.replies[self._reply_index]
            self._reply_index += 1
            self._recv_buffer.extend(create_reply_frame(reply))

        if not self._recv_buffer:
            raise socket.timeout("No more replies")

        # Return up to bufsize bytes
        chunk = bytes(self._recv_buffer[:bufsize])
        self._recv_buffer = self._recv_buffer[bufsize:]
        return chunk


# =============================================================================
# Mock GSSAPI Classes for Kerberos Testing
# =============================================================================


class MockGSSAPICredentials:
    """Mock GSSAPI credentials for testing Kerberos auth."""

    def __init__(self, name="user@FNAL.GOV", lifetime=3600):
        self.name = name
        self.lifetime = lifetime


class MockGSSAPIContext:
    """Mock GSSAPI security context for testing Kerberos handshake."""

    def __init__(self, name=None, usage=None, flags=None, creds=None, mech=None):
        self.complete = False
        self._step_count = 0

    def step(self, token=None):
        """Mock step - returns token on first call, completes."""
        self._step_count += 1
        if self._step_count >= 1:
            self.complete = True
        return b"mock_kerberos_token"

    def get_signature(self, message):
        """Mock get_signature - returns fake MIC."""
        return b"mock_mic_signature"


class _MockCredsModule:
    """Mock gssapi.creds module."""

    Credentials = staticmethod(lambda usage=None: MockGSSAPICredentials())


class MockGSSAPIModule:
    """Mock gssapi module for testing without real Kerberos.

    Usage:
        with mock.patch.dict("sys.modules", {"gssapi": MockGSSAPIModule()}):
            auth = KerberosAuth()  # Uses mock instead of real gssapi
    """

    class NameType:
        hostbased_service = "hostbased_service"
        kerberos_principal = "kerberos_principal"

    class RequirementFlag:
        mutual_authentication = 1
        replay_detection = 2
        sequence_detection = 4
        integrity = 8
        out_of_sequence_detection = 16

        def __or__(self, other):
            return self

        def __ror__(self, other):
            return self

    class MechType:
        kerberos = "kerberos"

    class exceptions:
        class GSSError(Exception):
            pass

    creds = _MockCredsModule()
    Name = staticmethod(lambda name, name_type=None: name)
    Credentials = staticmethod(lambda usage=None: MockGSSAPICredentials())
    SecurityContext = MockGSSAPIContext


# =============================================================================
# JWT Token Helpers
# =============================================================================


def make_jwt_token(payload: dict) -> str:
    """Create a fake JWT token with given payload for testing.

    Args:
        payload: Dict to encode as JWT payload (e.g., {"sub": "user@fnal.gov"})

    Returns:
        JWT token string (header.payload.signature format)
    """
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(b"fake_signature").decode().rstrip("=")
    return f"{header}.{payload_b64}.{signature}"


# =============================================================================
# DPM Reply Factories
# =============================================================================


def make_device_info(
    name: str = TEMP_DEVICE,
    ref_id: int = 1,
    di: int = 12345,
    description: str = "Test Device",
    units: str | None = None,
) -> DeviceInfo_reply:
    """Create a DeviceInfo_reply with sensible defaults."""
    reply = DeviceInfo_reply()
    reply.ref_id = ref_id
    reply.di = di
    reply.name = name
    reply.description = description
    if units is not None:
        reply.units = units
    return reply


def make_start_list(list_id: int = 1, status: int = 0) -> StartList_reply:
    """Create a StartList_reply with sensible defaults."""
    reply = StartList_reply()
    reply.list_id = list_id
    reply.status = status
    return reply


def make_scalar_reply(
    value: float = TEMP_VALUE,
    ref_id: int = 1,
    timestamp: int = TIMESTAMP_MILLIS,
    status: int = 0,
) -> Scalar_reply:
    """Create a Scalar_reply with sensible defaults."""
    reply = Scalar_reply()
    reply.ref_id = ref_id
    reply.timestamp = timestamp
    reply.cycle = 0
    reply.status = status
    reply.data = value
    return reply


def make_scalar_array_reply(
    values: list[float] | None = None,
    ref_id: int = 1,
    timestamp: int = TIMESTAMP_MILLIS,
    status: int = 0,
) -> ScalarArray_reply:
    """Create a ScalarArray_reply with sensible defaults."""
    reply = ScalarArray_reply()
    reply.ref_id = ref_id
    reply.timestamp = timestamp
    reply.cycle = 0
    reply.status = status
    reply.data = list(values) if values is not None else list(ARRAY_VALUES)
    return reply


def make_text_reply(
    text: str = TEXT_VALUE,
    ref_id: int = 1,
    timestamp: int = TIMESTAMP_MILLIS,
    status: int = 0,
) -> Text_reply:
    """Create a Text_reply with sensible defaults."""
    reply = Text_reply()
    reply.ref_id = ref_id
    reply.timestamp = timestamp
    reply.cycle = 0
    reply.status = status
    reply.data = text
    return reply


def make_status_reply(
    status: int,
    ref_id: int = 1,
    timestamp: int = TIMESTAMP_MILLIS,
) -> Status_reply:
    """Create a Status_reply (error response) with given status code."""
    reply = Status_reply()
    reply.ref_id = ref_id
    reply.timestamp = timestamp
    reply.cycle = 0
    reply.status = status
    return reply


def make_auth_reply(service_name: str = "dpm") -> Authenticate_reply:
    """Create an Authenticate_reply for DPM auth handshake."""
    reply = Authenticate_reply()
    reply.serviceName = service_name
    # Token is required for marshaling (even if empty)
    del reply.token  # Remove the None attribute so marshal() skips it
    return reply


def make_apply_settings_reply(
    statuses: list[tuple[int, int]] | None = None,
) -> ApplySettings_reply:
    """Create ApplySettings_reply with given (ref_id, status) pairs.

    Args:
        statuses: List of (ref_id, status) tuples. Defaults to [(1, 0)] (success).
    """
    reply = ApplySettings_reply()
    pairs = statuses if statuses is not None else [(1, 0)]
    reply.status = []
    for ref_id, status in pairs:
        s = SettingStatus_struct()
        s.ref_id = ref_id
        s.status = status
        reply.status.append(s)
    return reply


def make_enable_settings_reply(status: int = 0) -> Status_reply:
    """Create EnableSettings response (Status_reply with status=0 for success)."""
    reply = Status_reply()
    reply.ref_id = 0
    reply.timestamp = 0
    reply.cycle = 0
    reply.status = status
    return reply


def make_write_sequence(
    device: str = TEMP_DEVICE,
    ref_id: int = 1,
    success: bool = True,
) -> list:
    """Create the standard reply sequence for a successful DPM write.

    Two-phase auth protocol:
    1. Authenticate_reply (service name, phase 1)
    2. Authenticate_reply (phase 2, context established)
    3. Status_reply (EnableSettings response, status=0)
    4. DeviceInfo_reply
    5. StartList_reply
    6. ApplySettings_reply
    """
    status = 0 if success else -1
    return [
        make_auth_reply(),
        make_auth_reply(),
        make_enable_settings_reply(),
        make_device_info(name=device, ref_id=ref_id),
        make_start_list(),
        make_apply_settings_reply([(ref_id, status)]),
    ]


def make_read_sequence(
    device: str = TEMP_DEVICE,
    value: float = TEMP_VALUE,
    ref_id: int = 1,
    with_device_info: bool = True,
) -> list:
    """Create the standard reply sequence for a successful DPM read.

    Returns: [DeviceInfo_reply (optional), StartList_reply, Scalar_reply]
    """
    replies = []
    if with_device_info:
        replies.append(make_device_info(name=device, ref_id=ref_id))
    replies.append(make_start_list())
    replies.append(make_scalar_reply(value=value, ref_id=ref_id))
    return replies


# =============================================================================
# Value Type Test Matrix
# =============================================================================

# (name, value, value_type, description)
# For parametrized testing of all value types
VALUE_TYPE_MATRIX = [
    ("scalar", TEMP_VALUE, ValueType.SCALAR, "float scalar"),
    ("array", ARRAY_VALUES, ValueType.SCALAR_ARRAY, "float array"),
    ("raw", RAW_BYTES, ValueType.RAW, "raw bytes"),
    ("text", TEXT_VALUE, ValueType.TEXT, "string"),
    ("text_array", TEXT_ARRAY_VALUES, ValueType.TEXT_ARRAY, "string list"),
    ("analog_alarm", ANALOG_ALARM_DATA, ValueType.ANALOG_ALARM, "analog alarm dict"),
    ("digital_alarm", DIGITAL_ALARM_DATA, ValueType.DIGITAL_ALARM, "digital alarm dict"),
    ("basic_status", BASIC_STATUS_DATA, ValueType.BASIC_STATUS, "status dict"),
]
