"""
FTPMAN (Fast Time Plot) protocol for ACNET.

Provides high-frequency data acquisition from front-end controllers:
- Class code query: determine FTP/snapshot capabilities (typecode 1)
- Continuous plot: high-frequency streaming data (typecode 6)
- Snapshot plot: triggered data capture and retrieval (typecodes 7, 8, 5)

All FTP packets use little-endian byte order (ACNET data representation).
"""

import itertools
import logging
import queue
import struct
import threading
from dataclasses import dataclass

from . import rad50
from enum import IntEnum

from .connection_sync import AcnetConnectionTCP, AcnetRequestContext
from .errors import (
    FACILITY_FTP,
    FTP_COLLECTING,
    FTP_PEND,
    FTP_WAIT_DELAY,
    FTP_WAIT_EVENT,
    AcnetError,
    AcnetTimeoutError,
    ftp_status_message,
    parse_error,
)

logger = logging.getLogger(__name__)

# Protocol constants
FTPMAN_TASK = "FTPMAN"
MAX_ACNET_MSG_SIZE = 8320  # Max ACNET message size in bytes (CLIB limit)
OVERSIZE_BUFFER_FACTOR = 1.5

# Typecodes
TYPECODE_CLASS_INFO = 1
TYPECODE_CONTINUOUS = 6
TYPECODE_SNAPSHOT_SETUP = 7
TYPECODE_SNAPSHOT_RETRIEVE = 8
TYPECODE_SNAPSHOT_CONTROL = 5

# Reply types
REPLY_TYPE_SETUP = 1
REPLY_TYPE_DATA = 2

# Default return period (15 Hz ticks between data replies)
DEFAULT_RETURN_PERIOD = 3

# Task name counters -- each pool gets a unique RAD50 name (matches Java FTPPool/SnapShotPool)
_ftp_counter = itertools.count(1)
_snap_counter = itertools.count(1)


def _next_ftp_task_name() -> int:
    """Generate next unique RAD50 task name for continuous FTP (FTP001, FTP002, ...)."""
    n = next(_ftp_counter) % 999 + 1  # wrap to 1-999, keeps name within 6-char RAD50 limit
    return rad50.encode(f"FTP{n:03d}")


def _next_snap_task_name() -> int:
    """Generate next unique RAD50 task name for snapshot (SNP001, SNP002, ...)."""
    n = next(_snap_counter) % 999 + 1  # wrap to 1-999, keeps name within 6-char RAD50 limit
    return rad50.encode(f"SNP{n:03d}")


# =============================================================================
# Data Model
# =============================================================================


@dataclass(frozen=True)
class FTPClassCode:
    """Result of a class code query for a device."""

    ftp: int  # FTP class code (11-23, or 0=unsupported)
    snap: int  # Snapshot class code (11-28, or 0=unsupported)
    error: int  # 0 on success


@dataclass(frozen=True)
class FTPClassInfo:
    """Metadata about an FTP continuous plot class code."""

    code: int
    description: str
    max_rate: int  # Hz
    from_pool: bool  # True if data comes from data pool (not direct FE)


@dataclass(frozen=True)
class SnapClassInfo:
    """Metadata about a snapshot class code."""

    code: int
    description: str
    max_rate: int  # Hz
    max_points: int
    has_timestamps: bool
    supports_triggers: bool
    skip_first_point: bool
    retrieval_max: int  # Max points per retrieval request


@dataclass(frozen=True)
class FTPDevice:
    """Device descriptor for FTP requests."""

    di: int  # Device Index (0-16777215)
    pi: int  # Property Index (0-255)
    ssdn: bytes  # 8-byte SSDN
    offset: int = 0  # Byte offset into device data
    data_length: int = 2  # Data word size: 2 or 4 bytes

    def __post_init__(self):
        if len(self.ssdn) != 8:
            raise ValueError(f"SSDN must be 8 bytes, got {len(self.ssdn)}")
        if self.data_length not in (2, 4):
            raise ValueError(f"data_length must be 2 or 4, got {self.data_length}")

    @property
    def dipi(self) -> int:
        """Combined Property-Device Index Pair for protocol."""
        return (self.pi << 24) | self.di


@dataclass(frozen=True)
class FTPDataPoint:
    """Single data point from FTP stream.

    NOTE: ``timestamp_us`` is relative to the last TCLK 0x02 event, NOT an
    absolute wall-clock time.  The TCLK supercycle is ~5 seconds, so these
    timestamps wrap around roughly every 50 000 ticks at 100 µs resolution.
    For streams lasting longer than one supercycle the caller must track
    0x02 events externally and reconstruct absolute times (as the Java
    ``FTPPool`` does via ``baseTime02``).
    """

    timestamp_us: int  # Microseconds (100us resolution, relative to TCLK 0x02)
    raw_value: int  # Raw integer value (signed 16 or 32 bit)


class SnapshotState(IntEnum):
    """Snapshot device readiness state.

    Tracks the FTP status progression from setup to data-ready:
    PENDING → WAIT_EVENT → WAIT_DELAY → COLLECTING → READY.
    Not all devices go through every state; some skip straight to READY.
    """

    PENDING = 1  # Setup accepted, waiting for FE to begin
    WAIT_EVENT = 2  # Armed, waiting for clock/external event
    WAIT_DELAY = 3  # Event received, waiting for arm delay
    COLLECTING = 4  # Collecting data points
    READY = 5  # Data ready for retrieval
    ERROR = 6  # Error occurred


# Composite FTP status → SnapshotState mapping
_FTP_STATUS_TO_STATE: dict[int, SnapshotState] = {
    FTP_PEND: SnapshotState.PENDING,
    FTP_WAIT_EVENT: SnapshotState.WAIT_EVENT,
    FTP_WAIT_DELAY: SnapshotState.WAIT_DELAY,
    FTP_COLLECTING: SnapshotState.COLLECTING,
}


def _ftp_status_to_state(composite_status: int, is_first_reply: bool = False) -> SnapshotState:
    """Map FTP composite status code to SnapshotState.

    On the first reply (setup ack), status == 0 means "accepted" not "ready"
    (CAMAC front-ends send 0 for some devices on the first reply).
    """
    if composite_status < 0:
        return SnapshotState.ERROR
    if composite_status == 0:
        return SnapshotState.PENDING if is_first_reply else SnapshotState.READY
    return _FTP_STATUS_TO_STATE.get(composite_status, SnapshotState.ERROR)


def _parse_status_update_states(data: bytes, num_devices: int) -> list[int]:
    """Extract per-device composite status codes from a snapshot status update.

    Status updates use the same wire format as setup replies:
    [2B: overall_error][22B: new-protocol header]
    [18B per device: error(2) + ref_point(4) + arm_sec(4) + arm_nsec(4) + reserved(4)]

    Returns list of per-device composite status codes (signed 16-bit).
    Returns empty list if the reply is too short to parse.
    """
    if len(data) < 24:
        return []

    overall = struct.unpack_from("<h", data, 0)[0]
    if overall < 0:
        return [overall] * num_devices

    statuses = []
    offset = 24  # skip 2B overall + 22B new-protocol header
    for _ in range(num_devices):
        if offset + 2 > len(data):
            break
        status = struct.unpack_from("<h", data, offset)[0]
        statuses.append(status)
        offset += 18  # each per-device block is 18 bytes

    return statuses


@dataclass(frozen=True)
class SnapshotSetupReply:
    """Parameters the front-end chose for a snapshot setup.

    Contains the actual values the front-end chose (may differ from request).
    """

    arm_trigger_word: int  # Arm/trigger selection word
    sample_rate_hz: int  # Actual sample rate chosen by FE
    arm_delay: int  # Actual arm delay (usec or samples)
    arm_events: bytes  # 8-byte arm clock events
    num_points: int  # Actual number of points
    per_device_errors: list[int]  # Signed status per device (negative=error)
    per_device_ref_points: list[int]  # Reference point (pre-trigger only)
    per_device_arm_time: list[tuple[int, int]]  # (seconds, nanoseconds)


# =============================================================================
# Class Code Registry
# =============================================================================

FTP_CLASS_INFO: dict[int, FTPClassInfo] = {
    1: FTPClassInfo(1, "C190 MADC channel (old)", 720, False),
    2: FTPClassInfo(2, "Internet Rack Monitor (old)", 1000, False),
    3: FTPClassInfo(3, "MRRF MAC MADC channel (old)", 100, False),
    4: FTPClassInfo(4, "Booster MAC MADC channel (old)", 15, True),
    5: FTPClassInfo(5, "15 Hz Linac/DA (old)", 15, False),
    6: FTPClassInfo(6, "1 Hz from data pool FRIG (old)", 1, True),
    7: FTPClassInfo(7, "15 Hz from data pool (old)", 15, True),
    8: FTPClassInfo(8, "60 Hz internal (old)", 60, False),
    9: FTPClassInfo(9, "68K MECAR (old)", 1440, False),
    10: FTPClassInfo(10, "Tev Collimators (old)", 240, False),
    11: FTPClassInfo(11, "C190 MADC channel", 720, False),
    12: FTPClassInfo(12, "Internet Rack Monitor", 1000, False),
    13: FTPClassInfo(13, "MRRF MAC MADC channel", 100, False),
    14: FTPClassInfo(14, "Booster MAC MADC channel", 15, True),
    15: FTPClassInfo(15, "15 Hz (Linac, D/A's etc.)", 15, False),
    16: FTPClassInfo(16, "C290 MADC channel", 1440, False),
    17: FTPClassInfo(17, "15 Hz from data pool", 15, True),
    18: FTPClassInfo(18, "60 Hz internal", 60, False),
    19: FTPClassInfo(19, "68K (MECAR)", 1440, False),
    20: FTPClassInfo(20, "Tev Collimators", 240, False),
    21: FTPClassInfo(21, "IRM 1KHz Digitizer", 1000, False),
    22: FTPClassInfo(22, "DAE 1 Hz", 1, True),
    23: FTPClassInfo(23, "DAE 15 Hz", 15, True),
}

SNAP_CLASS_INFO: dict[int, SnapClassInfo] = {
    1: SnapClassInfo(1, "C190 MADC channel (old)", 66666, 2048, True, False, True, 512),
    2: SnapClassInfo(2, "1440 Hz internal (old)", 1440, 2048, True, False, True, 512),
    4: SnapClassInfo(4, "15 Hz internal (old)", 15, 2048, True, False, True, 512),
    5: SnapClassInfo(5, "60 Hz internal (old)", 60, 2048, True, False, True, 512),
    6: SnapClassInfo(6, "Quick Digitizer Linac (old)", 10000000, 4096, False, False, True, 512),
    7: SnapClassInfo(7, "720 Hz internal (old)", 720, 2048, True, False, True, 512),
    8: SnapClassInfo(8, "New FRIG Trigger device", 0, 0, False, False, False, 0),
    9: SnapClassInfo(9, "New FRIG circ buffer (old)", 1000, 16384, True, True, True, 512),
    11: SnapClassInfo(11, "C190 MADC channel", 66666, 2048, True, False, True, 512),
    12: SnapClassInfo(12, "1440 Hz internal", 1440, 2048, True, False, True, 512),
    13: SnapClassInfo(13, "C290 MADC channel", 90000, 2048, True, False, True, 512),
    14: SnapClassInfo(14, "15 Hz internal", 15, 2048, True, False, True, 512),
    15: SnapClassInfo(15, "60 Hz internal", 60, 2048, True, False, True, 512),
    16: SnapClassInfo(16, "Quick Digitizer (Linac)", 10000000, 4096, False, False, True, 512),
    17: SnapClassInfo(17, "720 Hz internal", 720, 2048, True, False, True, 512),
    18: SnapClassInfo(18, "New FRIG circ buffer", 1000, 16384, True, True, True, 512),
    19: SnapClassInfo(19, "Swift Digitizer", 800000, 4096, False, False, True, 512),
    20: SnapClassInfo(20, "IRM 20 MHz Quick Digitizer", 20000000, 4096, False, False, True, 512),
    21: SnapClassInfo(21, "IRM 1KHz Digitizer", 1000, 4096, False, False, True, 512),
    22: SnapClassInfo(22, "DAE 1 Hz", 1, 4096, True, True, True, 4096),
    23: SnapClassInfo(23, "DAE 15 Hz", 15, 4096, True, True, True, 4096),
    24: SnapClassInfo(24, "IRM 12.5KHz Digitizer", 12500, 4096, False, False, True, 512),
    25: SnapClassInfo(25, "IRM 10KHz Digitizer", 10000, 4096, False, False, True, 512),
    26: SnapClassInfo(26, "IRM 10MHz Digitizer", 10000000, 4096, False, False, True, 512),
    28: SnapClassInfo(28, "New Booster BLM", 12500, 4096, False, False, True, 512),
}


def get_ftp_class_info(code: int) -> FTPClassInfo | None:
    """Look up FTP class info by code. Returns None if unknown."""
    return FTP_CLASS_INFO.get(code)


def get_snap_class_info(code: int) -> SnapClassInfo | None:
    """Look up snapshot class info by code. Returns None if unknown."""
    return SNAP_CLASS_INFO.get(code)


# =============================================================================
# Packet Builders
# =============================================================================


def build_class_info_request(devices: list[FTPDevice]) -> bytes:
    """Build a class code query packet (typecode 1)."""
    buf = struct.pack("<HH", TYPECODE_CLASS_INFO, len(devices))
    for dev in devices:
        buf += struct.pack("<I", dev.dipi)
        buf += dev.ssdn
    return buf


def _calculate_msg_size(num_devices: int, num_data_words: int, rate: int, return_period: int) -> int:
    """Calculate the FTP reply buffer size in 16-bit words.

    Matches Java FTPPool formula exactly::

        msgSize = (numDataBytes + numDevices*2) * rate  // data + timestamps
        msgSize += 6 * numDevices                       // per-device header
        msgSize += msgSize >> 1                         // 50% oversize
        msgSize /= returnPeriod * 2                     // to 16-bit words

    ``num_data_words`` is the total 16-bit words per sample (timestamp + value
    words).  Multiplying by 2 converts back to bytes for the Java-equivalent
    calculation.  Capped at ``MAX_ACNET_MSG_SIZE // 2``.
    """
    # bytes per second: (data_bytes + timestamp_bytes) * rate
    msg_size = num_data_words * 2 * rate
    msg_size += 6 * num_devices  # per-device header overhead (error+index+npts)
    msg_size += msg_size >> 1  # 50% oversize
    msg_size //= return_period * 2  # to 16-bit words
    return min(msg_size, MAX_ACNET_MSG_SIZE // 2)


def build_continuous_setup(
    devices: list[FTPDevice],
    rate_hz: float,
    return_period: int = DEFAULT_RETURN_PERIOD,
    priority: int = 0,
    reference_event: int = 0,
    task_name: int = 0,
) -> bytes:
    """Build a continuous plot setup packet (typecode 6, new protocol).

    Args:
        devices: Devices to stream.
        rate_hz: Sample rate in Hz.
        return_period: 15Hz ticks between replies (1-7).
        priority: Request priority (0=normal).
        reference_event: Clock event number (0-255) for timestamp alignment,
            or group event code | 0x8000.  0 means no reference.
        task_name: RAD50-encoded task name (0 = anonymous).
    """
    # Total data words per sample period across all devices
    # (timestamp word + value words: 2 for int16, 3 for int32)
    num_data_words = sum(2 if d.data_length == 2 else 3 for d in devices)

    effective_rate = int(rate_hz)
    sample_period_10us = max(1, int(100000 / effective_rate))
    msg_size = _calculate_msg_size(len(devices), num_data_words, effective_rate, return_period)

    buf = struct.pack("<H", TYPECODE_CONTINUOUS)
    buf += struct.pack("<I", task_name)
    buf += struct.pack("<H", len(devices))
    buf += struct.pack("<H", return_period)
    buf += struct.pack("<H", msg_size)
    buf += struct.pack("<H", reference_event)
    buf += struct.pack("<H", 0)  # start time
    buf += struct.pack("<H", 0)  # stop time
    buf += struct.pack("<H", priority)
    buf += struct.pack("<H", 0)  # current time (15Hz)
    buf += b"\x00" * 10  # reserved (5 x short)

    for dev in devices:
        buf += struct.pack("<I", dev.dipi)
        buf += struct.pack("<I", dev.offset)
        buf += dev.ssdn
        buf += struct.pack("<H", sample_period_10us)
        buf += b"\x00" * 4  # reserved

    return buf


def _build_arm_trigger_word(
    arm_source: int = 2,
    arm_modifier: int = 0,
    plot_mode: int = 2,
    trigger_source: int = 0,
    trigger_modifier: int = 0,
) -> int:
    """Build the arm and trigger selection word (matches Java SnapShotPool).

    Bit layout (15 is MSB):
      15..12: zeros
      11..10: TM (trigger modifier, only if TS=3)
      9..8:   TS (trigger source: 0=periodic, 2=clock, 3=external)
      7:      new protocol flag (always 1)
      6..5:   PM (plot mode: 2=post-trigger, 3=pre-trigger)
      3..2:   AM (arm modifier, only if AS=3)
      1..0:   AS (arm source: 0=device, 1=immediate, 2=clock, 3=external)

    NOTE: Java SnapShotPool never sends ARM_IMMEDIATELY (1) on the wire --
    even for immediate arming it uses ARM_CLOCK_EVENTS (2) with all-0xFF
    events and arm_delay=0.  Comment from Java: "ecbpm doesn't like;
    other don't care".  We follow the same convention.
    """
    return (
        (trigger_modifier & 0x3) << 10
        | (trigger_source & 0x3) << 8
        | (1 << 7)  # new protocol flag
        | (plot_mode & 0x3) << 5
        | (arm_modifier & 0x3) << 2
        | (arm_source & 0x3)
    )


def build_snapshot_setup(
    devices: list[FTPDevice],
    rate_hz: int,
    num_points: int = 2048,
    arm_source: int = 2,
    arm_modifier: int = 0,
    plot_mode: int = 2,
    arm_delay: int = 0,
    arm_events: bytes = b"\xff" * 8,
    trigger_source: int = 0,
    trigger_modifier: int = 0,
    sample_events: bytes = b"\xff" * 4,
    priority: int = 0,
    arm_device: FTPDevice | None = None,
    arm_mask: int = 0,
    arm_value: int = 0,
    task_name: int = 0,
) -> bytes:
    """Build a snapshot setup packet (typecode 7, per FTPMAN Protocol spec).

    Header is 68 bytes, each device adds 20 bytes.

    The default ``arm_source=2`` (ARM_CLOCK_EVENTS) with all-0xFF arm_events
    gives immediate arming -- matching Java SnapShotPool which never sends
    ARM_IMMEDIATELY (1) on the wire ("ecbpm doesn't like; other don't care").
    For clock-event arming, pass arm_events with literal event numbers, e.g.
    ``b"\\x02" + b"\\xff" * 7`` for TCLK event 0x02.
    """
    arm_trigger_word = _build_arm_trigger_word(arm_source, arm_modifier, plot_mode, trigger_source, trigger_modifier)

    # Header (68 bytes)
    buf = struct.pack("<H", TYPECODE_SNAPSHOT_SETUP)
    buf += struct.pack("<I", task_name)
    buf += struct.pack("<H", len(devices))
    buf += struct.pack("<H", arm_trigger_word)
    buf += struct.pack("<H", priority)
    buf += struct.pack("<I", rate_hz)
    buf += struct.pack("<I", arm_delay)
    buf += arm_events[:8].ljust(8, b"\xff")
    buf += sample_events[:4].ljust(4, b"\xff")
    buf += struct.pack("<I", num_points)

    # Arm device fields (24 bytes)
    if arm_device is not None:
        buf += struct.pack("<I", arm_device.dipi)
        buf += struct.pack("<I", arm_device.offset)
        buf += arm_device.ssdn
        buf += struct.pack("<I", arm_mask)
        buf += struct.pack("<I", arm_value)
    else:
        buf += b"\x00" * 24  # dipi(4)+offset(4)+ssdn(8)+mask(4)+value(4)

    buf += b"\x00" * 8  # reserved

    # Per-device blocks (20 bytes each)
    for dev in devices:
        buf += struct.pack("<I", dev.dipi)
        buf += struct.pack("<I", dev.offset)
        buf += dev.ssdn
        buf += b"\x00" * 4  # reserved

    return buf


def build_retrieve_request(
    item_number: int,
    num_points: int = 512,
    point_number: int = -1,
    task_name: int = 0,
) -> bytes:
    """Build a snapshot retrieve packet (typecode 8, per FTPMAN Protocol spec).

    Args:
        item_number: 1-based device index from the setup request.
        num_points: Number of points to retrieve.
        point_number: Starting point (0-based), or -1 for sequential access.
        task_name: RAD50-encoded task name (must match setup request).
    """
    buf = struct.pack("<H", TYPECODE_SNAPSHOT_RETRIEVE)
    buf += struct.pack("<I", task_name)
    buf += struct.pack("<H", item_number)
    buf += struct.pack("<H", num_points)
    # Point number is 4B unsigned; -1 becomes 0xFFFFFFFF
    buf += struct.pack("<I", point_number & 0xFFFFFFFF)
    return buf


SNAPSHOT_CONTROL_RESTART = 1
SNAPSHOT_CONTROL_RESET = 2


def build_snapshot_control(subtype: int, task_name: int = 0) -> bytes:
    """Build a snapshot control packet (typecode 5, per FTPMAN Protocol spec).

    Args:
        subtype: ``SNAPSHOT_CONTROL_RESTART`` (1) = re-arm and re-trigger,
            ``SNAPSHOT_CONTROL_RESET`` (2) = reset retrieval pointers.
        task_name: RAD50-encoded task name (must match setup request).
    """
    buf = struct.pack("<H", TYPECODE_SNAPSHOT_CONTROL)
    buf += struct.pack("<I", task_name)
    buf += struct.pack("<H", subtype)
    return buf


# =============================================================================
# Reply Parsers
# =============================================================================


def parse_class_info_reply(data: bytes, num_devices: int) -> list[FTPClassCode]:
    """Parse class code query reply.

    Reply format per device: [2B: error][2B: ftp_class][2B: snap_class]
    with a 2B overall status prefix.
    """
    results = []
    offset = 0

    # Overall device status
    if len(data) < 2:
        raise ValueError(f"Class info reply too short: {len(data)} bytes")
    overall_status = struct.unpack_from("<h", data, offset)[0]
    offset += 2

    if overall_status < 0:
        raise AcnetError(overall_status, "Class info request failed")

    for _ in range(num_devices):
        if offset + 6 > len(data):
            raise ValueError(f"Truncated class info reply at offset {offset}")
        error, ftp, snap = struct.unpack_from("<hHH", data, offset)
        offset += 6
        results.append(FTPClassCode(ftp=ftp, snap=snap, error=error))

    return results


def parse_continuous_first_reply(data: bytes, num_devices: int) -> list[int]:
    """Parse the first reply (setup ack) from continuous plot.

    Returns per-device status codes.
    """
    if len(data) < 2:
        raise ValueError(f"Setup reply too short: {len(data)} bytes")

    error = struct.unpack_from("<h", data, 0)[0]
    if error < 0:
        raise AcnetError(error, "Continuous plot setup failed")

    if len(data) < 4:
        raise ValueError(f"Setup reply too short: {len(data)} bytes")

    reply_type = struct.unpack_from("<H", data, 2)[0]
    if reply_type != REPLY_TYPE_SETUP:
        raise ValueError(f"Expected reply type {REPLY_TYPE_SETUP}, got {reply_type}")

    statuses = []
    offset = 4
    for _ in range(num_devices):
        if offset + 2 > len(data):
            raise ValueError(f"Truncated setup reply at offset {offset}")
        s = struct.unpack_from("<h", data, offset)[0]
        statuses.append(s)
        offset += 2

    return statuses


def parse_continuous_data_reply(
    data: bytes,
    devices: list[FTPDevice],
) -> dict[int, list[FTPDataPoint]]:
    """Parse a data reply (reply_type=2) from continuous plot.

    Returns {device_di: [FTPDataPoint, ...]} for devices that had data.
    The `index` field from the FE is an absolute byte offset into the reply.

    Timestamps are relative to the last TCLK 0x02 event and wrap every
    ~5 seconds.  See :class:`FTPDataPoint` for details.
    """
    if len(data) < 8:
        return {}

    error = struct.unpack_from("<h", data, 0)[0]
    reply_type = struct.unpack_from("<H", data, 2)[0]

    if error < 0 or reply_type != REPLY_TYPE_DATA:
        return {}

    # Skip 4 reserved bytes at offset 4
    results: dict[int, list[FTPDataPoint]] = {}
    hdr_offset = 8  # past error(2) + type(2) + reserved(4)

    for dev in devices:
        if hdr_offset + 6 > len(data):
            break

        dev_error = struct.unpack_from("<h", data, hdr_offset)[0]
        hdr_offset += 2

        if dev_error != 0:
            # Skip index and num_points for errored device
            hdr_offset += 4
            continue

        index, num_points = struct.unpack_from("<HH", data, hdr_offset)
        hdr_offset += 4

        if num_points == 0:
            continue

        # index is an absolute byte offset into the reply data buffer
        point_size = 2 + dev.data_length  # timestamp(2) + value(2 or 4)
        points = []

        cursor = index
        for _ in range(num_points):
            if cursor + point_size > len(data):
                break

            ts = struct.unpack_from("<H", data, cursor)[0]
            cursor += 2

            if dev.data_length == 4:
                val = struct.unpack_from("<i", data, cursor)[0]
                cursor += 4
            else:
                val = struct.unpack_from("<h", data, cursor)[0]
                cursor += 2

            points.append(FTPDataPoint(timestamp_us=ts * 100, raw_value=val))

        if points:
            results[dev.di] = points

    return results


def parse_snapshot_setup_reply(data: bytes, num_devices: int) -> SnapshotSetupReply:
    """Parse the snapshot setup reply (per FTPMAN Protocol spec).

    Reply format:
      [2B: error (signed)][2B: arm_trigger_word][4B: rate_hz]
      [4B: arm_delay][8B: arm_events][4B: num_points]
      Per device (18B):
        [2B: error (signed)][4B: ref_point][4B: arm_sec][4B: arm_nsec][4B: reserved]
    """
    if len(data) < 2:
        raise ValueError(f"Snapshot setup reply too short: {len(data)} bytes")

    # First 2 bytes are always the FTP-level error code.
    # When the front-end rejects the request, this may be the ONLY data.
    error = struct.unpack_from("<h", data, 0)[0]
    if error < 0:
        raise AcnetError(error, "Snapshot setup failed")

    if len(data) < 24:
        raise ValueError(f"Snapshot setup reply too short: {len(data)} bytes")

    arm_trigger_word = struct.unpack_from("<H", data, 2)[0]
    sample_rate_hz = struct.unpack_from("<I", data, 4)[0]
    arm_delay = struct.unpack_from("<I", data, 8)[0]
    arm_events = data[12:20]
    num_points = struct.unpack_from("<I", data, 20)[0]

    per_device_errors = []
    per_device_ref_points = []
    per_device_arm_time = []
    offset = 24

    for _ in range(num_devices):
        if offset + 18 > len(data):
            break
        dev_error = struct.unpack_from("<h", data, offset)[0]
        ref_point = struct.unpack_from("<I", data, offset + 2)[0]
        arm_sec = struct.unpack_from("<I", data, offset + 6)[0]
        arm_nsec = struct.unpack_from("<I", data, offset + 10)[0]
        # 4B reserved at offset+14
        offset += 18

        per_device_errors.append(dev_error)
        per_device_ref_points.append(ref_point)
        per_device_arm_time.append((arm_sec, arm_nsec))

    return SnapshotSetupReply(
        arm_trigger_word=arm_trigger_word,
        sample_rate_hz=sample_rate_hz,
        arm_delay=arm_delay,
        arm_events=arm_events,
        num_points=num_points,
        per_device_errors=per_device_errors,
        per_device_ref_points=per_device_ref_points,
        per_device_arm_time=per_device_arm_time,
    )


def parse_snapshot_data_reply(
    data: bytes,
    device: FTPDevice,
    has_timestamps: bool = True,
    skip_first_point: bool = False,
) -> list[FTPDataPoint]:
    """Parse a snapshot retrieve reply (typecode 8 response).

    Args:
        data: Raw reply bytes.
        device: Device descriptor (for data_length).
        has_timestamps: True if the snapshot class includes timestamps
            (most classes do; Quick Digitizer and Swift do not).
        skip_first_point: If True, discard the first data point (it contains
            arm time / metadata, not real data).  Nearly all snapshot classes
            require this -- see ``SnapClassInfo.skip_first_point``.
    """
    if len(data) < 4:
        return []

    error = struct.unpack_from("<h", data, 0)[0]
    if error < 0:
        raise AcnetError(error, "Snapshot retrieve failed")

    num_points = struct.unpack_from("<H", data, 2)[0]
    if num_points == 0:
        return []

    point_size = (2 + device.data_length) if has_timestamps else device.data_length
    offset = 4
    points = []

    for i in range(num_points):
        if offset + point_size > len(data):
            break

        if has_timestamps:
            ts = struct.unpack_from("<H", data, offset)[0]
            offset += 2
        else:
            ts = 0

        if device.data_length == 4:
            val = struct.unpack_from("<i", data, offset)[0]
            offset += 4
        else:
            val = struct.unpack_from("<h", data, offset)[0]
            offset += 2

        if skip_first_point and i == 0:
            continue  # first point is metadata, not real data

        points.append(FTPDataPoint(timestamp_us=ts * 100, raw_value=val))

    return points


# =============================================================================
# FTPStream - Continuous Plot
# =============================================================================


class FTPStream:
    """Context manager for continuous FTP data streaming.

    Usage:
        client = FTPClient(connection)
        with client.start_continuous(node, devices) as stream:
            for batch in stream.readings(timeout=1.0):
                for di, points in batch.items():
                    print(f"Device {di}: {len(points)} points")
    """

    def __init__(
        self,
        ctx: AcnetRequestContext,
        devices: list[FTPDevice],
        reply_queue: queue.Queue,
        setup_statuses: list[int],
    ):
        self._ctx = ctx
        self._devices = list(devices)
        self._reply_queue = reply_queue
        self._setup_statuses = setup_statuses
        self._stopped = False

    @property
    def setup_statuses(self) -> list[int]:
        """Per-device status codes from the setup reply."""
        return self._setup_statuses

    @property
    def stopped(self) -> bool:
        return self._stopped

    def readings(self, timeout: float = 1.0):
        """Yield dicts of {device_di: [FTPDataPoint, ...]} per reply.

        Blocks up to `timeout` seconds waiting for each reply.
        Stops iteration when the stream is stopped or connection ends.
        """
        while not self._stopped:
            try:
                reply = self._reply_queue.get(timeout=timeout)
            except queue.Empty:
                continue

            if reply is None:
                # Sentinel: stream ended
                self._stopped = True
                return

            status, data, is_last = reply

            if status < 0:
                logger.warning(f"FTP data reply error: status={status}")
                if is_last:
                    self._stopped = True
                    return
                continue

            batch = parse_continuous_data_reply(data, self._devices)
            if batch:
                yield batch

            if is_last:
                self._stopped = True
                return

    def stop(self):
        """Cancel the FTP stream."""
        if not self._stopped:
            self._stopped = True
            try:
                self._ctx.cancel()
            except Exception as e:
                logger.debug(f"Error cancelling FTP stream: {e}")

    def __enter__(self) -> "FTPStream":
        return self

    def __exit__(self, *args):
        self.stop()


# =============================================================================
# SnapshotHandle
# =============================================================================


class SnapshotHandle:
    """Handle for a snapshot plot setup with state tracking.

    A background monitor thread reads status updates from the front-end and
    tracks each device through the snapshot lifecycle:
    ``PENDING → WAIT_EVENT → WAIT_DELAY → COLLECTING → READY``.

    Use :meth:`wait` to block until data is ready, or poll :attr:`is_ready`.
    Always use as a context manager (or call :meth:`cancel`) to ensure the
    monitor thread is cleaned up.

    Usage::

        with client.start_snapshot(node, devices, rate_hz=1440,
                                   snap_class_code=13) as snap:
            snap.wait(timeout=30)
            points = snap.retrieve(device_index=0)
    """

    def __init__(
        self,
        connection: AcnetConnectionTCP,
        node: int,
        ctx: AcnetRequestContext,
        devices: list[FTPDevice],
        setup_reply: SnapshotSetupReply,
        reply_queue: queue.Queue,
        task_name: int = 0,
        snap_class_code: int | None = None,
    ):
        self._connection = connection
        self._node = node
        self._ctx = ctx
        self._devices = list(devices)
        self._setup_reply = setup_reply
        self._reply_queue = reply_queue
        self._task_name = task_name
        self._snap_class_info = get_snap_class_info(snap_class_code) if snap_class_code else None
        self._cancelled = False
        self._lock = threading.Lock()

        # State tracking: keyed by device di
        self._ready_event = threading.Event()
        self._device_states: dict[int, SnapshotState] = {}
        self._device_errors: dict[int, int] = {}

        # Initialize states from setup reply (first reply uses CAMAC quirk)
        for i, dev in enumerate(self._devices):
            if i < len(setup_reply.per_device_errors):
                status = setup_reply.per_device_errors[i]
                state = _ftp_status_to_state(status, is_first_reply=True)
            else:
                state = SnapshotState.PENDING
            self._device_states[dev.di] = state
            if state == SnapshotState.ERROR:
                self._device_errors[dev.di] = setup_reply.per_device_errors[i]

        if self._all_terminal():
            self._ready_event.set()

        # Start monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"ftp-snap-monitor-{task_name}",
            daemon=True,
        )
        self._monitor_thread.start()

    def _all_terminal(self) -> bool:
        """True when every device is READY or ERROR."""
        return all(s in (SnapshotState.READY, SnapshotState.ERROR) for s in self._device_states.values())

    def _monitor_loop(self):
        """Background thread: read status updates and track device states."""
        while not self._cancelled:
            try:
                item = self._reply_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            status, data, is_last = item

            if status < 0:
                with self._lock:
                    for dev in self._devices:
                        if self._device_states[dev.di] != SnapshotState.READY:
                            self._device_states[dev.di] = SnapshotState.ERROR
                            self._device_errors[dev.di] = status
                    self._ready_event.set()
                if is_last:
                    break
                continue

            per_device = _parse_status_update_states(data, len(self._devices))

            with self._lock:
                for i, dev in enumerate(self._devices):
                    if i >= len(per_device):
                        break
                    # Don't downgrade a terminal state
                    if self._device_states[dev.di] in (SnapshotState.READY, SnapshotState.ERROR):
                        continue
                    new_state = _ftp_status_to_state(per_device[i], is_first_reply=False)
                    self._device_states[dev.di] = new_state
                    if new_state == SnapshotState.ERROR:
                        self._device_errors[dev.di] = per_device[i]

                if self._all_terminal():
                    self._ready_event.set()

            if is_last:
                break

    # ----- State API -----

    @property
    def state(self) -> SnapshotState:
        """Aggregate snapshot state.

        Returns ``READY`` when all devices are ready, ``ERROR`` if any device
        has an error, otherwise the least-progressed device state.
        """
        with self._lock:
            states = set(self._device_states.values())
        if SnapshotState.ERROR in states:
            return SnapshotState.ERROR
        if all(s == SnapshotState.READY for s in states):
            return SnapshotState.READY
        non_ready = {s for s in states if s != SnapshotState.READY}
        return min(non_ready) if non_ready else SnapshotState.PENDING

    @property
    def device_states(self) -> dict[int, SnapshotState]:
        """Per-device state dict keyed by device index (di)."""
        with self._lock:
            return dict(self._device_states)

    @property
    def is_ready(self) -> bool:
        """True when all devices have data ready for retrieval."""
        return self._ready_event.is_set() and self.state == SnapshotState.READY

    def wait(self, timeout: float | None = None) -> bool:
        """Block until all devices reach a terminal state (READY or ERROR).

        Returns True if all devices are ready.  Returns False on timeout.
        Raises :class:`AcnetError` if any device ended with an error.
        """
        if not self._ready_event.wait(timeout=timeout):
            return False
        with self._lock:
            for dev in self._devices:
                if self._device_states[dev.di] == SnapshotState.ERROR:
                    err = self._device_errors.get(dev.di, 0)
                    raise AcnetError(err, f"Snapshot device {dev.di} failed")
        return True

    # ----- Existing API -----

    @property
    def setup_reply(self) -> SnapshotSetupReply:
        return self._setup_reply

    @property
    def snap_class_info(self) -> SnapClassInfo | None:
        """Snapshot class info, if snap_class_code was provided."""
        return self._snap_class_info

    def retrieve(
        self,
        device_index: int = 0,
        num_points: int | None = None,
        point_number: int = -1,
        has_timestamps: bool | None = None,
        skip_first_point: bool | None = None,
        timeout: float = 5.0,
    ) -> list[FTPDataPoint]:
        """Retrieve snapshot data for a device.

        Args:
            device_index: 0-based index into the setup device list.
            num_points: Max points per request.  Defaults to the class-specific
                ``retrieval_max`` (usually 512, 4096 for DAE) or 512.
            point_number: Starting point (0-based), or -1 for sequential.
            has_timestamps: Override timestamp parsing.  When None, auto-detect
                from class info (True for most classes).
            skip_first_point: Override first-point skipping.  When None,
                auto-detect from class info (True for nearly all classes).
            timeout: Timeout in seconds.
        """
        # Validate and resolve params under lock (brief)
        with self._lock:
            if self._cancelled:
                raise RuntimeError("Snapshot has been cancelled")
            ci = self._snap_class_info

        if num_points is None:
            num_points = ci.retrieval_max if ci else 512
        if has_timestamps is None:
            has_timestamps = ci.has_timestamps if ci else True
        if skip_first_point is None:
            skip_first_point = ci.skip_first_point if ci else False
        if ci and num_points > ci.retrieval_max:
            raise ValueError(
                f"num_points ({num_points}) exceeds class retrieval_max ({ci.retrieval_max}) for snap class {ci.code}"
            )

        dev = self._devices[device_index]
        item_number = device_index + 1  # 1-based per FTPMAN spec
        payload = build_retrieve_request(item_number, num_points, point_number, task_name=self._task_name)

        result_q: queue.Queue = queue.Queue()

        def handler(reply):
            result_q.put((reply.status, reply.data, reply.last))

        self._connection.request_single(
            node=self._node,
            task=FTPMAN_TASK,
            data=payload,
            reply_handler=handler,
            timeout=int(timeout * 1000),
        )

        try:
            status, data, _ = result_q.get(timeout=timeout)
        except queue.Empty:
            raise AcnetTimeoutError(int(timeout * 1000))

        if status < 0:
            raise AcnetError(status, "Snapshot retrieve failed")

        return parse_snapshot_data_reply(data, dev, has_timestamps=has_timestamps, skip_first_point=skip_first_point)

    def restart(self, timeout: float = 5.0):
        """Re-arm and re-trigger the snapshot.

        Resets all device states to PENDING and clears the ready event,
        so :meth:`wait` can be used again for the new capture cycle.
        """
        with self._lock:
            if self._cancelled:
                raise RuntimeError("Snapshot has been cancelled")
            # Reset states before sending command
            self._ready_event.clear()
            for dev in self._devices:
                self._device_states[dev.di] = SnapshotState.PENDING
                self._device_errors.pop(dev.di, None)

        payload = build_snapshot_control(subtype=SNAPSHOT_CONTROL_RESTART, task_name=self._task_name)
        result_q: queue.Queue = queue.Queue()

        def handler(reply):
            result_q.put((reply.status, reply.data))

        self._connection.request_single(
            node=self._node,
            task=FTPMAN_TASK,
            data=payload,
            reply_handler=handler,
            timeout=int(timeout * 1000),
        )

        try:
            status, _ = result_q.get(timeout=timeout)
        except queue.Empty:
            raise AcnetTimeoutError(int(timeout * 1000))

        if status < 0:
            raise AcnetError(status, "Snapshot restart failed")

    def reset_pointers(self, timeout: float = 5.0):
        """Reset retrieval pointers."""
        with self._lock:
            if self._cancelled:
                raise RuntimeError("Snapshot has been cancelled")

        payload = build_snapshot_control(subtype=SNAPSHOT_CONTROL_RESET, task_name=self._task_name)
        result_q: queue.Queue = queue.Queue()

        def handler(reply):
            result_q.put((reply.status, reply.data))

        self._connection.request_single(
            node=self._node,
            task=FTPMAN_TASK,
            data=payload,
            reply_handler=handler,
            timeout=int(timeout * 1000),
        )

        try:
            status, _ = result_q.get(timeout=timeout)
        except queue.Empty:
            raise AcnetTimeoutError(int(timeout * 1000))

        if status < 0:
            raise AcnetError(status, "Snapshot reset pointers failed")

    def cancel(self):
        """Cancel the snapshot and stop the monitor thread."""
        if not self._cancelled:
            self._cancelled = True
            # Wake the monitor thread so it can exit
            self._reply_queue.put(None)
            try:
                self._ctx.cancel()
            except Exception as e:
                logger.debug(f"Error cancelling snapshot: {e}")
            # Wait for monitor thread to finish
            self._monitor_thread.join(timeout=5.0)
            if self._monitor_thread.is_alive():
                logger.warning("Snapshot monitor thread did not terminate within 5s")

    def __enter__(self) -> "SnapshotHandle":
        return self

    def __exit__(self, *args):
        self.cancel()


# =============================================================================
# FTPClient
# =============================================================================


class FTPClient:
    """High-level FTPMAN protocol client.

    Wraps an existing AcnetConnectionTCP to provide FTP operations.

    Example:
        with AcnetConnectionTCP() as conn:
            conn.connect()
            ftp = FTPClient(conn)

            # Query class codes
            codes = ftp.get_class_codes(node, device)
            print(f"FTP class: {codes.ftp}, Snap class: {codes.snap}")

            # Continuous stream
            with ftp.start_continuous(node, [device]) as stream:
                for batch in stream.readings(timeout=1.0):
                    for di, points in batch.items():
                        print(f"{len(points)} points")
    """

    def __init__(self, connection: AcnetConnectionTCP):
        self._connection = connection

    def get_class_codes(
        self,
        node: int,
        device: FTPDevice,
        timeout: float = 5.0,
    ) -> FTPClassCode:
        """Query FTP/snapshot class codes for a device.

        Args:
            node: Target front-end node address.
            device: Device to query.
            timeout: Timeout in seconds.

        Returns:
            FTPClassCode with ftp/snap class codes.
        """
        payload = build_class_info_request([device])

        result_q: queue.Queue = queue.Queue()

        def handler(reply):
            result_q.put((reply.status, reply.data, reply.last))

        self._connection.request_single(
            node=node,
            task=FTPMAN_TASK,
            data=payload,
            reply_handler=handler,
            timeout=int(timeout * 1000),
        )

        try:
            status, data, _ = result_q.get(timeout=timeout)
        except queue.Empty:
            raise AcnetTimeoutError(int(timeout * 1000))

        if status < 0:
            raise AcnetError(status, "Class code query failed")

        results = parse_class_info_reply(data, 1)
        return results[0]

    def get_class_codes_many(
        self,
        node: int,
        devices: list[FTPDevice],
        timeout: float = 5.0,
    ) -> list[FTPClassCode]:
        """Query FTP/snapshot class codes for multiple devices on one node."""
        payload = build_class_info_request(devices)

        result_q: queue.Queue = queue.Queue()

        def handler(reply):
            result_q.put((reply.status, reply.data, reply.last))

        self._connection.request_single(
            node=node,
            task=FTPMAN_TASK,
            data=payload,
            reply_handler=handler,
            timeout=int(timeout * 1000),
        )

        try:
            status, data, _ = result_q.get(timeout=timeout)
        except queue.Empty:
            raise AcnetTimeoutError(int(timeout * 1000))

        if status < 0:
            raise AcnetError(status, "Class code query failed")

        return parse_class_info_reply(data, len(devices))

    def start_continuous(
        self,
        node: int,
        devices: list[FTPDevice],
        rate_hz: float,
        return_period: int = DEFAULT_RETURN_PERIOD,
        priority: int = 0,
        reference_event: int = 0,
        timeout: float = 10.0,
    ) -> FTPStream:
        """Start continuous FTP streaming.

        Args:
            node: Target front-end node address.
            devices: Devices to stream.
            rate_hz: Sample rate in Hz.
            return_period: 15Hz ticks between data replies (1-7).
            priority: Request priority (0=normal).
            reference_event: Clock event number (0-255) for timestamp alignment,
                or group event code | 0x8000.  0 means no reference.
            timeout: Timeout for setup reply in seconds.

        Returns:
            FTPStream context manager that yields data batches.
        """
        task_name = _next_ftp_task_name()
        payload = build_continuous_setup(
            devices=devices,
            rate_hz=rate_hz,
            return_period=return_period,
            priority=priority,
            reference_event=reference_event,
            task_name=task_name,
        )

        reply_queue: queue.Queue = queue.Queue()
        setup_event = threading.Event()
        setup_result: list = []  # [status, data] or [exception]

        def handler(reply):
            if not setup_event.is_set():
                # First reply is setup ack
                setup_result.append((reply.status, reply.data, reply.last))
                setup_event.set()
            else:
                # Subsequent replies are data
                if reply.last and reply.status < 0:
                    reply_queue.put(None)  # sentinel
                else:
                    reply_queue.put((reply.status, reply.data, reply.last))

        ctx = self._connection.request_multiple(
            node=node,
            task=FTPMAN_TASK,
            data=payload,
            reply_handler=handler,
            timeout=int(timeout * 1000),
        )

        # Wait for setup reply
        if not setup_event.wait(timeout=timeout):
            ctx.cancel()
            raise AcnetTimeoutError(int(timeout * 1000))

        status, data, is_last = setup_result[0]

        if status < 0:
            ctx.cancel()
            raise AcnetError(status, "Continuous plot setup failed")

        if is_last:
            raise AcnetError(0, "Unexpected end of replies after setup")

        setup_statuses = parse_continuous_first_reply(data, len(devices))

        # Check per-device statuses (negative=error, positive=informational)
        for i, s in enumerate(setup_statuses):
            if s < 0:
                fac, err = parse_error(s)
                msg = ftp_status_message(s) if fac == FACILITY_FTP else f"facility={fac}, error={err}"
                logger.error(f"Device {i} (di={devices[i].di}) setup error: {msg} [{fac} {err}]")
            elif s > 0:
                fac, err = parse_error(s)
                msg = ftp_status_message(s) if fac == FACILITY_FTP else f"facility={fac}, error={err}"
                logger.info(f"Device {i} (di={devices[i].di}) setup status: {msg} [{fac} {err}]")

        return FTPStream(
            ctx=ctx,
            devices=devices,
            reply_queue=reply_queue,
            setup_statuses=setup_statuses,
        )

    def start_snapshot(
        self,
        node: int,
        devices: list[FTPDevice],
        rate_hz: int,
        num_points: int = 2048,
        arm_source: int = 2,
        arm_modifier: int = 0,
        plot_mode: int = 2,
        arm_delay: int = 0,
        arm_events: bytes = b"\xff" * 8,
        trigger_source: int = 0,
        trigger_modifier: int = 0,
        sample_events: bytes = b"\xff" * 4,
        priority: int = 0,
        arm_device: FTPDevice | None = None,
        arm_mask: int = 0,
        arm_value: int = 0,
        snap_class_code: int | None = None,
        timeout: float = 10.0,
    ) -> SnapshotHandle:
        """Setup a snapshot plot.

        The default ``arm_source=2`` (ARM_CLOCK_EVENTS) with all-0xFF
        arm_events gives immediate arming.  Java SnapShotPool never sends
        ARM_IMMEDIATELY (1) on the wire because some FEs (ecbpm) reject it.
        For clock-event arming, keep ``arm_source=2`` and pass the event
        numbers in ``arm_events``.

        Args:
            node: Target front-end node address.
            devices: Devices to capture.
            rate_hz: Sample rate in Hz.
            num_points: Number of points to capture (default 2048).
            arm_source: 0=device, 2=clock (default, also used for immediate),
                3=external.  Avoid 1 (ARM_IMMEDIATELY) -- some FEs reject it.
            arm_modifier: Modifier for external arm source (0-3, only if arm_source=3).
            plot_mode: 2=post-trigger, 3=pre-trigger.
            arm_delay: Microseconds (post-trigger) or samples (pre-trigger) after arm.
            arm_events: 8-byte array of literal TCLK clock event numbers.
                Each byte is an event number (0x00-0xFD); 0xFF means unused.
                Example: ``b"\\x02" + b"\\xff" * 7`` for event 0x02 only.
                All-0xFF (default) with arm_source=2 gives immediate arming.
            trigger_source: 0=periodic, 2=clock, 3=external.
            trigger_modifier: Modifier for external trigger (0-3, only if trigger_source=3).
            sample_events: 4-byte sample trigger clock events (0xFF=unused).
            priority: Request priority (0=user, 1=other CR, 2=main CR, 3=SDA).
            arm_device: Optional device for device-triggered arm (arm_source=0).
            arm_mask: Arm device bit mask.
            arm_value: Arm device match value.
            snap_class_code: Snapshot class code (from get_class_codes).
                When provided, enables auto skip_first_point and retrieval_max
                enforcement in :meth:`SnapshotHandle.retrieve`.
            timeout: Timeout for setup reply in seconds.

        Returns:
            SnapshotHandle context manager.
        """
        task_name = _next_snap_task_name()
        payload = build_snapshot_setup(
            devices=devices,
            rate_hz=rate_hz,
            num_points=num_points,
            arm_source=arm_source,
            arm_modifier=arm_modifier,
            plot_mode=plot_mode,
            arm_delay=arm_delay,
            arm_events=arm_events,
            trigger_source=trigger_source,
            trigger_modifier=trigger_modifier,
            sample_events=sample_events,
            priority=priority,
            arm_device=arm_device,
            arm_mask=arm_mask,
            arm_value=arm_value,
            task_name=task_name,
        )

        reply_queue: queue.Queue = queue.Queue()
        setup_event = threading.Event()
        setup_result: list = []

        def handler(reply):
            if not setup_event.is_set():
                setup_result.append((reply.status, reply.data, reply.last))
                setup_event.set()
            else:
                reply_queue.put((reply.status, reply.data, reply.last))

        ctx = self._connection.request_multiple(
            node=node,
            task=FTPMAN_TASK,
            data=payload,
            reply_handler=handler,
            timeout=int(timeout * 1000),
        )

        if not setup_event.wait(timeout=timeout):
            ctx.cancel()
            raise AcnetTimeoutError(int(timeout * 1000))

        status, data, _ = setup_result[0]

        if status < 0:
            ctx.cancel()
            raise AcnetError(status, "Snapshot setup failed")

        setup_reply = parse_snapshot_setup_reply(data, len(devices))

        # Log per-device statuses from setup reply
        for i, s in enumerate(setup_reply.per_device_errors):
            if s < 0:
                fac, err = parse_error(s)
                msg = ftp_status_message(s) if fac == FACILITY_FTP else f"facility={fac}, error={err}"
                logger.error(f"Snapshot device {i} (di={devices[i].di}) error: {msg} [{fac} {err}]")
            elif s > 0:
                fac, err = parse_error(s)
                msg = ftp_status_message(s) if fac == FACILITY_FTP else f"facility={fac}, error={err}"
                logger.info(f"Snapshot device {i} (di={devices[i].di}) status: {msg} [{fac} {err}]")

        return SnapshotHandle(
            connection=self._connection,
            node=node,
            ctx=ctx,
            devices=devices,
            setup_reply=setup_reply,
            reply_queue=reply_queue,
            task_name=task_name,
            snap_class_code=snap_class_code,
        )
