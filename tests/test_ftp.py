"""
Unit tests for pacsys.acnet.ftp — FTPMAN protocol.

Tests packet construction, reply parsing, buffer size calculations,
and class code registry. Uses no network — all ACNET interactions are mocked.
"""

import struct
from unittest.mock import MagicMock

import pytest

from pacsys.acnet.errors import AcnetError
from pacsys.acnet.errors import FTP_COLLECTING, FTP_PEND, FTP_WAIT_DELAY, FTP_WAIT_EVENT
from pacsys.acnet.ftp import (
    FTP_CLASS_INFO,
    MAX_ACNET_MSG_SIZE,
    REPLY_TYPE_DATA,
    REPLY_TYPE_SETUP,
    SNAP_CLASS_INFO,
    SNAPSHOT_CONTROL_RESTART,
    TYPECODE_CLASS_INFO,
    TYPECODE_CONTINUOUS,
    TYPECODE_SNAPSHOT_CONTROL,
    TYPECODE_SNAPSHOT_RETRIEVE,
    TYPECODE_SNAPSHOT_SETUP,
    FTPClassCode,
    FTPClient,
    FTPDataPoint,
    FTPDevice,
    FTPStream,
    SnapshotHandle,
    SnapshotSetupReply,
    SnapshotState,
    _calculate_msg_size,
    _ftp_status_to_state,
    _next_ftp_task_name,
    _next_snap_task_name,
    _parse_status_update_states,
    build_class_info_request,
    build_continuous_setup,
    build_retrieve_request,
    build_snapshot_control,
    build_snapshot_setup,
    get_ftp_class_info,
    get_snap_class_info,
    parse_class_info_reply,
    parse_continuous_data_reply,
    parse_continuous_first_reply,
    parse_snapshot_data_reply,
    parse_snapshot_setup_reply,
)


# =============================================================================
# Test device fixtures
# =============================================================================

MOUTTMP_SSDN = b"\x00\x00B\x00?!\x00\x00"
MOUTTMP_DI = 27235
MOUTTMP_PI = 12


@pytest.fixture
def mouttmp():
    return FTPDevice(di=MOUTTMP_DI, pi=MOUTTMP_PI, ssdn=MOUTTMP_SSDN)


@pytest.fixture
def mouttmp_4byte():
    return FTPDevice(di=MOUTTMP_DI, pi=MOUTTMP_PI, ssdn=MOUTTMP_SSDN, data_length=4)


# =============================================================================
# FTPDevice tests
# =============================================================================


class TestFTPDevice:
    def test_dipi(self, mouttmp):
        expected = (MOUTTMP_PI << 24) | MOUTTMP_DI
        assert mouttmp.dipi == expected

    def test_ssdn_validation(self):
        with pytest.raises(ValueError, match="SSDN must be 8 bytes"):
            FTPDevice(di=1, pi=0, ssdn=b"\x00" * 7)

    def test_data_length_validation(self):
        with pytest.raises(ValueError, match="data_length must be 2 or 4"):
            FTPDevice(di=1, pi=0, ssdn=b"\x00" * 8, data_length=3)


# =============================================================================
# Class code registry tests
# =============================================================================


class TestClassCodeRegistry:
    def test_ftp_class_16_is_c290(self):
        info = get_ftp_class_info(16)
        assert info is not None
        assert info.max_rate == 1440
        assert "C290" in info.description

    def test_ftp_class_12_is_irm(self):
        info = get_ftp_class_info(12)
        assert info is not None
        assert info.max_rate == 1000

    def test_snap_class_13_is_c290(self):
        info = get_snap_class_info(13)
        assert info is not None
        assert info.max_rate == 90000
        assert info.max_points == 2048
        assert info.has_timestamps is True

    def test_snap_class_19_swift(self):
        info = get_snap_class_info(19)
        assert info is not None
        assert info.max_rate == 800000
        assert info.has_timestamps is False

    def test_unknown_ftp_class(self):
        assert get_ftp_class_info(99) is None

    def test_unknown_snap_class(self):
        assert get_snap_class_info(99) is None

    def test_all_ftp_classes_have_positive_rate(self):
        for code, info in FTP_CLASS_INFO.items():
            assert info.max_rate > 0, f"FTP class {code} has zero max_rate"

    def test_all_snap_classes_have_nonneg_rate(self):
        for code, info in SNAP_CLASS_INFO.items():
            assert info.max_rate >= 0, f"Snap class {code} has negative max_rate"


# =============================================================================
# Packet construction tests
# =============================================================================


class TestBuildClassInfoRequest:
    def test_single_device(self, mouttmp):
        pkt = build_class_info_request([mouttmp])
        assert len(pkt) == 4 + 12  # header(4) + 1 * (dipi(4) + ssdn(8))

        typecode, num_devices = struct.unpack_from("<HH", pkt, 0)
        assert typecode == TYPECODE_CLASS_INFO
        assert num_devices == 1

        dipi = struct.unpack_from("<I", pkt, 4)[0]
        assert dipi == mouttmp.dipi
        assert pkt[8:16] == MOUTTMP_SSDN

    def test_two_devices(self, mouttmp):
        dev2 = FTPDevice(di=100, pi=12, ssdn=b"\x01" * 8)
        pkt = build_class_info_request([mouttmp, dev2])
        assert len(pkt) == 4 + 2 * 12

        _, num_devices = struct.unpack_from("<HH", pkt, 0)
        assert num_devices == 2


class TestBuildContinuousSetup:
    def test_packet_structure(self, mouttmp):
        pkt = build_continuous_setup(
            devices=[mouttmp],
            rate_hz=1440,
            return_period=4,
        )

        offset = 0
        typecode = struct.unpack_from("<H", pkt, offset)[0]
        assert typecode == TYPECODE_CONTINUOUS
        offset += 2

        task_name = struct.unpack_from("<I", pkt, offset)[0]
        assert task_name == 0  # default
        offset += 4

        num_devices = struct.unpack_from("<H", pkt, offset)[0]
        assert num_devices == 1
        offset += 2

        ret_period = struct.unpack_from("<H", pkt, offset)[0]
        assert ret_period == 4
        offset += 2

        msg_size = struct.unpack_from("<H", pkt, offset)[0]
        assert msg_size > 0
        offset += 2

        # ref_word, start, stop, priority, current_time
        offset += 2 * 5
        # reserved (10 bytes)
        offset += 10

        # Per-device block: dipi(4) + offset(4) + ssdn(8) + sample_period(2) + reserved(4) = 22
        dipi = struct.unpack_from("<I", pkt, offset)[0]
        assert dipi == mouttmp.dipi
        offset += 4

        dev_offset = struct.unpack_from("<I", pkt, offset)[0]
        assert dev_offset == 0
        offset += 4

        assert pkt[offset : offset + 8] == MOUTTMP_SSDN
        offset += 8

        sample_period = struct.unpack_from("<H", pkt, offset)[0]
        expected = int(100000 / 1440)
        assert sample_period == expected

    def test_reference_event(self, mouttmp):
        """reference_event is encoded at the correct offset."""
        pkt = build_continuous_setup(devices=[mouttmp], rate_hz=100, reference_event=0x0F)
        # reference_event is at offset 2(typecode)+4(task)+2(ndev)+2(period)+2(msgsize) = 12
        ref_word = struct.unpack_from("<H", pkt, 12)[0]
        assert ref_word == 0x0F

    def test_reference_event_group(self, mouttmp):
        """Group event code has 0x8000 flag."""
        pkt = build_continuous_setup(devices=[mouttmp], rate_hz=100, reference_event=0x8042)
        ref_word = struct.unpack_from("<H", pkt, 12)[0]
        assert ref_word == 0x8042

    def test_task_name(self, mouttmp):
        """task_name is encoded at the correct offset."""
        from pacsys.acnet import rad50

        tn = rad50.encode("FTP001")
        pkt = build_continuous_setup(devices=[mouttmp], rate_hz=100, task_name=tn)
        task_name = struct.unpack_from("<I", pkt, 2)[0]
        assert task_name == tn

    def test_per_device_size(self, mouttmp):
        """Each device adds 22 bytes to the packet."""
        pkt1 = build_continuous_setup(devices=[mouttmp], rate_hz=100)
        dev2 = FTPDevice(di=100, pi=12, ssdn=b"\x01" * 8)
        pkt2 = build_continuous_setup(devices=[mouttmp, dev2], rate_hz=100)
        assert len(pkt2) - len(pkt1) == 22


class TestBuildSnapshotSetup:
    def test_packet_structure(self, mouttmp):
        """Verify typecode 7 packet matches FTPMAN Protocol spec."""
        pkt = build_snapshot_setup(
            devices=[mouttmp],
            rate_hz=1440,
            num_points=2048,
            plot_mode=2,
            priority=0,
        )

        offset = 0
        typecode = struct.unpack_from("<H", pkt, offset)[0]
        assert typecode == TYPECODE_SNAPSHOT_SETUP
        offset += 2

        task_name = struct.unpack_from("<I", pkt, offset)[0]
        assert task_name == 0  # default
        offset += 4

        num_devices = struct.unpack_from("<H", pkt, offset)[0]
        assert num_devices == 1
        offset += 2

        arm_trigger_word = struct.unpack_from("<H", pkt, offset)[0]
        # Default arm_source=2 (ARM_CLOCK_EVENTS), plot_mode=2:
        # (1<<7) | (2<<5) | 2 = 0xC2
        assert arm_trigger_word == 0xC2
        offset += 2

        priority = struct.unpack_from("<H", pkt, offset)[0]
        assert priority == 0
        offset += 2

        rate = struct.unpack_from("<I", pkt, offset)[0]
        assert rate == 1440
        offset += 4

        arm_delay = struct.unpack_from("<I", pkt, offset)[0]
        assert arm_delay == 0
        offset += 4

        # arm events (8B) + sample events (4B)
        offset += 12

        num_points = struct.unpack_from("<I", pkt, offset)[0]
        assert num_points == 2048
        offset += 4

        # Arm device (24B: dipi+offset+ssdn+mask+value) + reserved (8B)
        offset += 32

        # Total header should be 68 bytes
        assert offset == 68

        # Per-device block: 20 bytes
        dipi = struct.unpack_from("<I", pkt, offset)[0]
        assert dipi == mouttmp.dipi
        offset += 4

        dev_offset = struct.unpack_from("<I", pkt, offset)[0]
        assert dev_offset == 0
        offset += 4

        assert pkt[offset : offset + 8] == MOUTTMP_SSDN
        offset += 8

        # 4B reserved
        offset += 4
        assert len(pkt) == 68 + 20  # header + 1 device

    def test_per_device_size(self, mouttmp):
        """Each device adds 20 bytes (per FTPMAN spec)."""
        pkt1 = build_snapshot_setup(devices=[mouttmp], rate_hz=100)
        dev2 = FTPDevice(di=100, pi=12, ssdn=b"\x01" * 8)
        pkt2 = build_snapshot_setup(devices=[mouttmp, dev2], rate_hz=100)
        assert len(pkt2) - len(pkt1) == 20

    def test_arm_trigger_word_external(self):
        """Test arm+trigger word with external sources."""
        from pacsys.acnet.ftp import _build_arm_trigger_word

        # AS=3 (external), AM=2, PM=3 (pre-trigger), TS=3 (external), TM=1
        word = _build_arm_trigger_word(arm_source=3, arm_modifier=2, plot_mode=3, trigger_source=3, trigger_modifier=1)
        assert word & 0x3 == 3  # AS
        assert (word >> 2) & 0x3 == 2  # AM
        assert (word >> 5) & 0x3 == 3  # PM
        assert (word >> 7) & 0x1 == 1  # new protocol
        assert (word >> 8) & 0x3 == 3  # TS
        assert (word >> 10) & 0x3 == 1  # TM


class TestBuildRetrieveRequest:
    def test_structure(self):
        """Verify typecode 8 matches FTPMAN spec: typecode+taskname+item+npts+point."""
        pkt = build_retrieve_request(item_number=1, num_points=512, point_number=-1)
        assert len(pkt) == 14  # 2+4+2+2+4

        typecode = struct.unpack_from("<H", pkt, 0)[0]
        assert typecode == TYPECODE_SNAPSHOT_RETRIEVE

        task_name = struct.unpack_from("<I", pkt, 2)[0]
        assert task_name == 0  # default

        item_num = struct.unpack_from("<H", pkt, 6)[0]
        assert item_num == 1

        num_pts = struct.unpack_from("<H", pkt, 8)[0]
        assert num_pts == 512

        point_num = struct.unpack_from("<I", pkt, 10)[0]
        assert point_num == 0xFFFFFFFF  # -1 as unsigned 32-bit

    def test_sequential_access(self):
        pkt = build_retrieve_request(item_number=3, point_number=-1)
        item = struct.unpack_from("<H", pkt, 6)[0]
        assert item == 3
        pt = struct.unpack_from("<I", pkt, 10)[0]
        assert pt == 0xFFFFFFFF

    def test_random_access(self):
        pkt = build_retrieve_request(item_number=1, point_number=100)
        pt = struct.unpack_from("<I", pkt, 10)[0]
        assert pt == 100

    def test_task_name(self):
        """task_name is encoded at the correct offset."""
        from pacsys.acnet import rad50

        tn = rad50.encode("SNP001")
        pkt = build_retrieve_request(item_number=1, task_name=tn)
        assert struct.unpack_from("<I", pkt, 2)[0] == tn


class TestBuildSnapshotControl:
    def test_restart(self):
        pkt = build_snapshot_control(subtype=1)
        assert len(pkt) == 8  # 2+4+2
        typecode = struct.unpack_from("<H", pkt, 0)[0]
        assert typecode == TYPECODE_SNAPSHOT_CONTROL
        task_name = struct.unpack_from("<I", pkt, 2)[0]
        assert task_name == 0
        sub = struct.unpack_from("<H", pkt, 6)[0]
        assert sub == 1

    def test_reset_pointers(self):
        pkt = build_snapshot_control(subtype=2)
        sub = struct.unpack_from("<H", pkt, 6)[0]
        assert sub == 2


# =============================================================================
# Buffer size calculation tests
# =============================================================================


class TestBufferSize:
    def test_matches_java_formula(self):
        """Buffer size matches Java FTPPool formula exactly.

        Java: msgSize = (numDataBytes + numDevices*2) * rate
               msgSize += 6 * numDevices
               msgSize += msgSize >> 1      // 50% oversize
               msgSize /= returnPeriod * 2  // to words
        """
        # 1 device, 2 data words (= 4 bytes), 1440 Hz, period=3
        result = _calculate_msg_size(1, 2, 1440, 3)
        # Java equivalent: (4 * 1440 + 6) * 1.5 / 6 = 8649 / 6 = 1441
        msg = 2 * 2 * 1440  # num_data_words * 2 * rate = 5760
        msg += 6 * 1  # per-device header = 5766
        msg += msg >> 1  # 50% oversize = 8649
        msg //= 3 * 2  # to words = 1441
        assert result == msg

    def test_capped_at_max(self):
        """Large requests should be capped at MAX_ACNET_MSG_SIZE / 2."""
        result = _calculate_msg_size(100, 100, 10000, 7)
        assert result == MAX_ACNET_MSG_SIZE // 2

    def test_single_15hz_device(self):
        """15 Hz device should produce a small buffer."""
        result = _calculate_msg_size(1, 2, 15, 3)
        assert result < 100  # Very small for 15Hz


# =============================================================================
# Reply parsing tests
# =============================================================================


class TestParseClassInfoReply:
    def test_single_device_success(self):
        # Overall status = 0, then per-device: error=0, ftp=16, snap=13
        data = struct.pack("<H", 0) + struct.pack("<HHH", 0, 16, 13)
        results = parse_class_info_reply(data, 1)
        assert len(results) == 1
        assert results[0] == FTPClassCode(ftp=16, snap=13, error=0)

    def test_device_error(self):
        data = struct.pack("<H", 0) + struct.pack("<HHH", 42, 0, 0)
        results = parse_class_info_reply(data, 1)
        assert results[0].error == 42
        assert results[0].ftp == 0
        assert results[0].snap == 0

    def test_overall_error(self):
        data = struct.pack("<h", -1)  # Negative = error
        with pytest.raises(AcnetError):
            parse_class_info_reply(data, 1)

    def test_positive_status_not_error(self):
        """Positive overall status (warning) should not raise."""
        data = struct.pack("<h", 1) + struct.pack("<hHH", 0, 16, 13)
        results = parse_class_info_reply(data, 1)
        assert results[0].ftp == 16

    def test_truncated(self):
        data = struct.pack("<H", 0)  # no device data
        with pytest.raises(ValueError, match="Truncated"):
            parse_class_info_reply(data, 1)


class TestParseContinuousFirstReply:
    def test_success(self):
        # error=0, reply_type=1, device_status=0
        data = struct.pack("<HH", 0, REPLY_TYPE_SETUP) + struct.pack("<H", 0)
        statuses = parse_continuous_first_reply(data, 1)
        assert statuses == [0]

    def test_error(self):
        data = struct.pack("<hH", -10, REPLY_TYPE_SETUP)
        with pytest.raises(AcnetError):
            parse_continuous_first_reply(data, 1)

    def test_positive_error_not_raised(self):
        """Positive FTP status (e.g. FTP_COLLECTING) should not raise."""
        data = struct.pack("<hH", 4, REPLY_TYPE_SETUP) + struct.pack("<h", 0)
        statuses = parse_continuous_first_reply(data, 1)
        assert statuses == [0]

    def test_wrong_type(self):
        data = struct.pack("<HH", 0, 99)
        with pytest.raises(ValueError, match="Expected reply type"):
            parse_continuous_first_reply(data, 1)


class TestParseContinuousDataReply:
    def test_with_data_points(self):
        """Parse a data reply with 2 data points for 1 device (2-byte values)."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8, data_length=2)

        # Build reply: error(2) + type(2) + reserved(4) + per-device header + data
        # Layout: [header 8B][per_dev 6B][point1 4B][point2 4B]
        header = struct.pack("<HH", 0, REPLY_TYPE_DATA) + b"\x00" * 4

        # index is absolute byte offset to data within the reply buffer
        # Per-device header is 6 bytes (error+index+npts), data follows at offset 14
        data_start = 8 + 6  # 14 bytes from start
        per_dev = struct.pack("<HHH", 0, data_start, 2)

        # 2 points: each is timestamp(2) + value(2) = 4 bytes
        point1 = struct.pack("<Hh", 100, 42)  # ts=100*100us=10ms, val=42
        point2 = struct.pack("<Hh", 200, -5)  # ts=200*100us=20ms, val=-5

        data = header + per_dev + point1 + point2
        result = parse_continuous_data_reply(data, [dev])

        assert 1 in result
        assert len(result[1]) == 2
        assert result[1][0] == FTPDataPoint(timestamp_us=10000, raw_value=42)
        assert result[1][1] == FTPDataPoint(timestamp_us=20000, raw_value=-5)

    def test_device_error_skipped(self):
        """Devices with errors should be skipped."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        header = struct.pack("<HH", 0, REPLY_TYPE_DATA) + b"\x00" * 4
        per_dev = struct.pack("<HHH", 0xBEEF, 0, 0)  # error, skip rest
        data = header + per_dev
        result = parse_continuous_data_reply(data, [dev])
        assert result == {}

    def test_zero_points(self):
        """Zero points should produce empty result."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        header = struct.pack("<HH", 0, REPLY_TYPE_DATA) + b"\x00" * 4
        per_dev = struct.pack("<HHH", 0, 14, 0)  # index=14, npts=0
        data = header + per_dev
        result = parse_continuous_data_reply(data, [dev])
        assert result == {}

    def test_positive_header_error_still_parses(self):
        """Positive header error (warning) should not discard data."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8, data_length=2)
        # Header with error=+4 (FTP_COLLECTING), reply_type=2
        header = struct.pack("<hH", 4, REPLY_TYPE_DATA) + b"\x00" * 4
        data_start = 8 + 6
        per_dev = struct.pack("<hHH", 0, data_start, 1)
        point = struct.pack("<Hh", 50, 42)
        data = header + per_dev + point
        result = parse_continuous_data_reply(data, [dev])
        assert 1 in result
        assert result[1][0].raw_value == 42

    def test_4byte_values(self):
        """Parse 4-byte integer values."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8, data_length=4)
        header = struct.pack("<HH", 0, REPLY_TYPE_DATA) + b"\x00" * 4
        data_start = 8 + 6  # absolute offset to data
        per_dev = struct.pack("<HHH", 0, data_start, 1)
        point = struct.pack("<Hi", 50, 123456)  # ts=50, 4-byte val=123456
        data = header + per_dev + point
        result = parse_continuous_data_reply(data, [dev])
        assert result[1][0] == FTPDataPoint(timestamp_us=5000, raw_value=123456)


class TestParseSnapshotSetupReply:
    def _build_setup_reply(self, error=0, arm_word=0x21, rate=1440, delay=0, npts=2048, devices=None):
        """Build a snapshot setup reply per FTPMAN spec."""
        buf = struct.pack("<h", error)
        buf += struct.pack("<H", arm_word)
        buf += struct.pack("<I", rate)
        buf += struct.pack("<I", delay)
        buf += b"\xff" * 8  # arm events
        buf += struct.pack("<I", npts)
        for dev_err, ref_pt, arm_sec, arm_nsec in devices or [(0, 0, 0, 0)]:
            buf += struct.pack("<h", dev_err)
            buf += struct.pack("<I", ref_pt)
            buf += struct.pack("<I", arm_sec)
            buf += struct.pack("<I", arm_nsec)
            buf += b"\x00" * 4  # reserved
        return buf

    def test_success(self):
        data = self._build_setup_reply(rate=1440, npts=2048)
        reply = parse_snapshot_setup_reply(data, 1)
        assert isinstance(reply, SnapshotSetupReply)
        assert reply.sample_rate_hz == 1440
        assert reply.num_points == 2048
        assert reply.arm_trigger_word == 0x21
        assert reply.per_device_errors == [0]
        assert reply.per_device_ref_points == [0]
        assert reply.per_device_arm_time == [(0, 0)]

    def test_error(self):
        data = self._build_setup_reply(error=-5)
        with pytest.raises(AcnetError):
            parse_snapshot_setup_reply(data, 1)

    def test_positive_error_not_raised(self):
        """Positive status (informational) should not raise."""
        data = self._build_setup_reply(error=1)
        reply = parse_snapshot_setup_reply(data, 1)
        assert reply.sample_rate_hz == 1440

    def test_multiple_devices(self):
        devs = [(0, 100, 1000, 500), (-3, 0, 0, 0)]
        data = self._build_setup_reply(devices=devs)
        reply = parse_snapshot_setup_reply(data, 2)
        assert reply.per_device_errors == [0, -3]
        assert reply.per_device_ref_points == [100, 0]
        assert reply.per_device_arm_time == [(1000, 500), (0, 0)]


class TestParseSnapshotDataReply:
    def test_success_with_timestamps(self):
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        data = struct.pack("<hH", 0, 2)
        data += struct.pack("<Hh", 10, 100)
        data += struct.pack("<Hh", 20, 200)
        points = parse_snapshot_data_reply(data, dev, has_timestamps=True)
        assert len(points) == 2
        assert points[0] == FTPDataPoint(timestamp_us=1000, raw_value=100)
        assert points[1] == FTPDataPoint(timestamp_us=2000, raw_value=200)

    def test_success_without_timestamps(self):
        """Quick Digitizer and Swift classes have no timestamps."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        data = struct.pack("<hH", 0, 3)
        data += struct.pack("<h", 100)
        data += struct.pack("<h", 200)
        data += struct.pack("<h", 300)
        points = parse_snapshot_data_reply(data, dev, has_timestamps=False)
        assert len(points) == 3
        assert all(p.timestamp_us == 0 for p in points)
        assert [p.raw_value for p in points] == [100, 200, 300]

    def test_error(self):
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        data = struct.pack("<hH", -15, 0)
        with pytest.raises(AcnetError):
            parse_snapshot_data_reply(data, dev)

    def test_positive_error_returns_data(self):
        """Positive error (warning) should still parse data."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        data = struct.pack("<hH", 1, 1)
        data += struct.pack("<Hh", 10, 42)
        points = parse_snapshot_data_reply(data, dev)
        assert len(points) == 1
        assert points[0].raw_value == 42

    def test_skip_first_point(self):
        """skip_first_point=True discards the first data point (metadata)."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        data = struct.pack("<hH", 0, 3)  # 3 points
        data += struct.pack("<Hh", 0, 9999)  # metadata point (arm time)
        data += struct.pack("<Hh", 10, 100)  # real data
        data += struct.pack("<Hh", 20, 200)  # real data
        points = parse_snapshot_data_reply(data, dev, skip_first_point=True)
        assert len(points) == 2
        assert points[0].raw_value == 100
        assert points[1].raw_value == 200

    def test_skip_first_point_false(self):
        """skip_first_point=False keeps all points (default)."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        data = struct.pack("<hH", 0, 2)
        data += struct.pack("<Hh", 10, 100)
        data += struct.pack("<Hh", 20, 200)
        points = parse_snapshot_data_reply(data, dev, skip_first_point=False)
        assert len(points) == 2

    def test_skip_first_point_single_point(self):
        """skip_first_point with only 1 point returns empty."""
        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        data = struct.pack("<hH", 0, 1)
        data += struct.pack("<Hh", 0, 9999)
        points = parse_snapshot_data_reply(data, dev, skip_first_point=True)
        assert len(points) == 0


# =============================================================================
# FTPStream tests
# =============================================================================


class TestFTPStream:
    def test_readings_yields_batches(self):
        """FTPStream.readings() should yield parsed data batches."""
        import queue as q

        dev = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        ctx = MagicMock()
        reply_q = q.Queue()

        stream = FTPStream(ctx=ctx, devices=[dev], reply_queue=reply_q, setup_statuses=[0])

        # Enqueue a data reply: header(8) + per_dev(6) + point(4)
        header = struct.pack("<HH", 0, REPLY_TYPE_DATA) + b"\x00" * 4
        data_start = 8 + 6  # absolute byte offset to data
        per_dev = struct.pack("<HHH", 0, data_start, 1)
        point = struct.pack("<Hh", 50, 42)
        reply_data = header + per_dev + point

        reply_q.put((0, reply_data, False))
        reply_q.put(None)  # sentinel to stop

        batches = list(stream.readings(timeout=0.5))
        assert len(batches) == 1
        assert 1 in batches[0]

    def test_stop_cancels_context(self):
        import queue as q

        ctx = MagicMock()
        stream = FTPStream(ctx=ctx, devices=[], reply_queue=q.Queue(), setup_statuses=[])
        stream.stop()
        ctx.cancel.assert_called_once()
        assert stream.stopped

    def test_context_manager(self):
        import queue as q

        ctx = MagicMock()
        stream = FTPStream(ctx=ctx, devices=[], reply_queue=q.Queue(), setup_statuses=[])
        with stream:
            pass
        ctx.cancel.assert_called_once()


# =============================================================================
# FTPClient tests (mocked connection)
# =============================================================================


class TestFTPClientClassCodes:
    def test_get_class_codes_success(self, mouttmp):
        conn = MagicMock()

        # Mock send_request to simulate reply
        def fake_request_single(node, task, data, reply_handler, timeout):
            reply = MagicMock()
            reply.status = 0
            reply.data = struct.pack("<H", 0) + struct.pack("<HHH", 0, 16, 13)
            reply.last = True
            reply_handler(reply)
            return MagicMock()

        conn.request_single = fake_request_single

        client = FTPClient(conn)
        result = client.get_class_codes(node=3018, device=mouttmp)
        assert result.ftp == 16
        assert result.snap == 13
        assert result.error == 0

    def test_get_class_codes_acnet_error(self, mouttmp):
        conn = MagicMock()

        def fake_request_single(node, task, data, reply_handler, timeout):
            reply = MagicMock()
            reply.status = -6  # Negative ACNET status = error
            reply.data = b""
            reply.last = True
            reply_handler(reply)
            return MagicMock()

        conn.request_single = fake_request_single

        client = FTPClient(conn)
        with pytest.raises(AcnetError):
            client.get_class_codes(node=3018, device=mouttmp)

    def test_get_class_codes_positive_status_ok(self, mouttmp):
        """Positive ACNET status (warning/informational) should not raise."""
        conn = MagicMock()

        def fake_request_single(node, task, data, reply_handler, timeout):
            reply = MagicMock()
            reply.status = 1  # Positive = informational, not error
            reply.data = struct.pack("<H", 0) + struct.pack("<HHH", 0, 16, 13)
            reply.last = True
            reply_handler(reply)
            return MagicMock()

        conn.request_single = fake_request_single

        client = FTPClient(conn)
        result = client.get_class_codes(node=3018, device=mouttmp)
        assert result.ftp == 16


class TestFTPClientContinuous:
    def test_start_continuous_success(self, mouttmp):
        conn = MagicMock()
        conn.raw_handle = 0x12345678

        def fake_request_multiple(node, task, data, reply_handler, timeout):
            # Send setup reply
            setup_reply = MagicMock()
            setup_reply.status = 0
            setup_reply.data = struct.pack("<HH", 0, REPLY_TYPE_SETUP) + struct.pack("<H", 0)
            setup_reply.last = False
            reply_handler(setup_reply)
            return MagicMock()

        conn.request_multiple = fake_request_multiple

        client = FTPClient(conn)
        stream = client.start_continuous(node=3018, devices=[mouttmp], rate_hz=1440)
        assert isinstance(stream, FTPStream)
        assert stream.setup_statuses == [0]
        stream.stop()

    def test_start_continuous_setup_error(self, mouttmp):
        conn = MagicMock()
        conn.raw_handle = 0

        def fake_request_multiple(node, task, data, reply_handler, timeout):
            reply = MagicMock()
            reply.status = -1  # Negative ACNET status = error
            reply.data = b""
            reply.last = False
            reply_handler(reply)
            return MagicMock()

        conn.request_multiple = fake_request_multiple

        client = FTPClient(conn)
        with pytest.raises(AcnetError):
            client.start_continuous(node=3018, devices=[mouttmp], rate_hz=1440)


# =============================================================================
# FTP error codes tests
# =============================================================================


class TestFTPErrorCodes:
    def test_ftp_facility_exists(self):
        from pacsys.acnet.errors import FACILITY_FTP

        assert FACILITY_FTP == 15

    def test_ftp_errors_have_correct_facility(self):
        from pacsys.acnet.errors import (
            FTP_BADEV,
            FTP_COLLECTING,
            FTP_FE_OUTOFMEM,
            FTP_INVSSDN,
            FTP_INVTYP,
            FTP_PEND,
            FTP_WAIT_DELAY,
            FTP_WAIT_EVENT,
            parse_error,
        )

        for code in [
            FTP_COLLECTING,
            FTP_WAIT_DELAY,
            FTP_WAIT_EVENT,
            FTP_PEND,
            FTP_INVTYP,
            FTP_INVSSDN,
            FTP_FE_OUTOFMEM,
            FTP_BADEV,
        ]:
            facility, _ = parse_error(code)
            assert facility == 15, f"Expected facility 15 for code {code}"

    def test_ftp_pend_is_positive(self):
        from pacsys.acnet.errors import FTP_PEND, parse_error

        _, err_num = parse_error(FTP_PEND)
        assert err_num > 0  # Warning (positive)

    def test_ftp_invtyp_is_negative(self):
        from pacsys.acnet.errors import FTP_INVTYP, parse_error

        _, err_num = parse_error(FTP_INVTYP)
        assert err_num < 0  # Error (negative)


# =============================================================================
# Task name generation tests
# =============================================================================


class TestTaskNameGeneration:
    def test_ftp_task_names_are_unique(self):
        """Each call to _next_ftp_task_name returns a different RAD50 value."""
        names = [_next_ftp_task_name() for _ in range(3)]
        assert len(set(names)) == 3

    def test_snap_task_names_are_unique(self):
        names = [_next_snap_task_name() for _ in range(3)]
        assert len(set(names)) == 3

    def test_ftp_task_name_decodes_to_ftp_prefix(self):
        from pacsys.acnet.rad50 import decode_stripped

        name = _next_ftp_task_name()
        decoded = decode_stripped(name)
        assert decoded.startswith("FTP")

    def test_snap_task_name_decodes_to_snp_prefix(self):
        from pacsys.acnet.rad50 import decode_stripped

        name = _next_snap_task_name()
        decoded = decode_stripped(name)
        assert decoded.startswith("SNP")


# =============================================================================
# Snapshot control constants tests
# =============================================================================


class TestSnapshotControlConstants:
    def test_control_packet_with_task_name(self):
        from pacsys.acnet import rad50

        tn = rad50.encode("SNP001")
        pkt = build_snapshot_control(subtype=SNAPSHOT_CONTROL_RESTART, task_name=tn)
        assert struct.unpack_from("<I", pkt, 2)[0] == tn


# =============================================================================
# SnapshotHandle tests
# =============================================================================


class TestSnapshotHandle:
    """Tests for SnapshotHandle (class info, retrieval limits)."""

    def _make_handle(self, snap_class_code=None, per_device_errors=None, devices=None):
        """Create a SnapshotHandle with mocked connection."""
        import queue as q

        conn = MagicMock()
        ctx = MagicMock()
        if devices is None:
            devices = [FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)]
        if per_device_errors is None:
            per_device_errors = [0] * len(devices)
        setup_reply = SnapshotSetupReply(
            arm_trigger_word=0xC1,
            sample_rate_hz=1440,
            arm_delay=0,
            arm_events=b"\xff" * 8,
            num_points=2048,
            per_device_errors=per_device_errors,
            per_device_ref_points=[0] * len(devices),
            per_device_arm_time=[(0, 0)] * len(devices),
        )
        return SnapshotHandle(
            connection=conn,
            node=3018,
            ctx=ctx,
            devices=devices,
            setup_reply=setup_reply,
            reply_queue=q.Queue(),
            snap_class_code=snap_class_code,
        )

    def test_snap_class_info_populated(self):
        """snap_class_code=13 populates snap_class_info."""
        handle = self._make_handle(snap_class_code=13)
        try:
            assert handle.snap_class_info is not None
            assert handle.snap_class_info.code == 13
        finally:
            handle.cancel()

    def test_snap_class_info_none_without_code(self):
        handle = self._make_handle(snap_class_code=None)
        try:
            assert handle.snap_class_info is None
        finally:
            handle.cancel()

    def test_retrieval_max_enforced(self):
        """Requesting more than retrieval_max raises ValueError."""
        handle = self._make_handle(snap_class_code=12)  # retrieval_max=512
        try:
            with pytest.raises(ValueError, match="retrieval_max"):
                handle.retrieve(num_points=1000)
        finally:
            handle.cancel()

    def test_retrieval_max_dae_class(self):
        """DAE classes have retrieval_max=4096."""
        handle = self._make_handle(snap_class_code=22)  # DAE 1 Hz, max=4096
        try:
            # Should NOT raise for 4096
            # (will fail at the network level since conn is mocked, but no ValueError)
            with pytest.raises(Exception, match="(?!retrieval_max)"):
                handle.retrieve(num_points=4096)
        finally:
            handle.cancel()


# =============================================================================
# SnapshotState and state tracking tests
# =============================================================================


class TestSnapshotState:
    """Tests for SnapshotState enum and helper functions."""

    def test_state_ordering(self):
        """States have expected progression order."""
        assert SnapshotState.PENDING < SnapshotState.WAIT_EVENT
        assert SnapshotState.WAIT_EVENT < SnapshotState.WAIT_DELAY
        assert SnapshotState.WAIT_DELAY < SnapshotState.COLLECTING
        assert SnapshotState.COLLECTING < SnapshotState.READY

    def test_ftp_status_to_state_mapping(self):
        assert _ftp_status_to_state(FTP_PEND) == SnapshotState.PENDING
        assert _ftp_status_to_state(FTP_WAIT_EVENT) == SnapshotState.WAIT_EVENT
        assert _ftp_status_to_state(FTP_WAIT_DELAY) == SnapshotState.WAIT_DELAY
        assert _ftp_status_to_state(FTP_COLLECTING) == SnapshotState.COLLECTING
        assert _ftp_status_to_state(0) == SnapshotState.READY

    def test_ftp_status_to_state_first_reply_quirk(self):
        """On first reply, status==0 means PENDING (CAMAC quirk)."""
        assert _ftp_status_to_state(0, is_first_reply=True) == SnapshotState.PENDING
        assert _ftp_status_to_state(0, is_first_reply=False) == SnapshotState.READY

    def test_ftp_status_to_state_negative_is_error(self):
        assert _ftp_status_to_state(-241) == SnapshotState.ERROR
        assert _ftp_status_to_state(-1) == SnapshotState.ERROR

    def test_ftp_status_to_state_unknown_positive_is_error(self):
        """Unknown positive status code maps to ERROR."""
        assert _ftp_status_to_state(9999) == SnapshotState.ERROR


class TestParseStatusUpdateStates:
    """Tests for _parse_status_update_states."""

    @staticmethod
    def _build_status_reply(overall_error, per_device_errors):
        """Build a status update reply with the given per-device error codes."""
        buf = struct.pack("<h", overall_error)
        buf += struct.pack("<H", 0xC1)  # arm_trigger_word
        buf += struct.pack("<I", 1440)  # rate
        buf += struct.pack("<I", 0)  # delay
        buf += b"\xff" * 8  # arm events
        buf += struct.pack("<I", 2048)  # num_points
        for err in per_device_errors:
            buf += struct.pack("<h", err)
            buf += b"\x00" * 16  # ref_point + arm_sec + arm_nsec + reserved
        return buf

    def test_single_device_pending(self):
        data = self._build_status_reply(0, [FTP_PEND])
        states = _parse_status_update_states(data, 1)
        assert states == [FTP_PEND]

    def test_single_device_ready(self):
        data = self._build_status_reply(0, [0])
        states = _parse_status_update_states(data, 1)
        assert states == [0]

    def test_two_devices_mixed(self):
        data = self._build_status_reply(0, [FTP_COLLECTING, 0])
        states = _parse_status_update_states(data, 2)
        assert states == [FTP_COLLECTING, 0]

    def test_overall_error_propagated(self):
        """Negative overall error is returned for all devices."""
        data = struct.pack("<h", -241)
        # Pad to 24B minimum header
        data += b"\x00" * 22
        states = _parse_status_update_states(data, 3)
        assert states == [-241, -241, -241]

    def test_too_short_returns_empty(self):
        assert _parse_status_update_states(b"\x00", 1) == []
        assert _parse_status_update_states(b"\x00" * 10, 1) == []


class TestSnapshotStateTracking:
    """Tests for SnapshotHandle state tracking and monitor thread."""

    @staticmethod
    def _build_status_reply(overall_error, per_device_errors):
        """Build a status update reply."""
        buf = struct.pack("<h", overall_error)
        buf += struct.pack("<H", 0xC1)
        buf += struct.pack("<I", 1440)
        buf += struct.pack("<I", 0)
        buf += b"\xff" * 8
        buf += struct.pack("<I", 2048)
        for err in per_device_errors:
            buf += struct.pack("<h", err)
            buf += b"\x00" * 16
        return buf

    def _make_handle(self, per_device_errors=None, devices=None):
        """Create a SnapshotHandle with controllable reply_queue."""
        import queue as q

        conn = MagicMock()
        ctx = MagicMock()
        if devices is None:
            devices = [FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)]
        if per_device_errors is None:
            per_device_errors = [FTP_PEND] * len(devices)
        setup_reply = SnapshotSetupReply(
            arm_trigger_word=0xC1,
            sample_rate_hz=1440,
            arm_delay=0,
            arm_events=b"\xff" * 8,
            num_points=2048,
            per_device_errors=per_device_errors,
            per_device_ref_points=[0] * len(devices),
            per_device_arm_time=[(0, 0)] * len(devices),
        )
        reply_queue = q.Queue()
        handle = SnapshotHandle(
            connection=conn,
            node=3018,
            ctx=ctx,
            devices=devices,
            setup_reply=setup_reply,
            reply_queue=reply_queue,
            snap_class_code=None,
        )
        return handle, reply_queue

    def test_initial_state_pending(self):
        """Setup reply with FTP_PEND → initial state is PENDING."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])
        try:
            assert handle.state == SnapshotState.PENDING
            assert handle.device_states[1] == SnapshotState.PENDING
            assert not handle.is_ready
        finally:
            handle.cancel()

    def test_initial_state_zero_is_pending(self):
        """Setup reply with per_device_error=0 → PENDING (CAMAC quirk)."""
        handle, rq = self._make_handle(per_device_errors=[0])
        try:
            assert handle.state == SnapshotState.PENDING
            assert not handle.is_ready
        finally:
            handle.cancel()

    def test_monitor_transitions_to_ready(self):
        """Monitor thread updates state from PENDING → READY on FTP_OK (0)."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])
        try:
            # Send a status update with FTP_OK (0) for the device
            data = self._build_status_reply(0, [0])
            rq.put((0, data, False))

            assert handle.wait(timeout=2.0)
            assert handle.state == SnapshotState.READY
            assert handle.is_ready
            assert handle.device_states[1] == SnapshotState.READY
        finally:
            handle.cancel()

    def test_monitor_state_progression(self):
        """States progress: PENDING → WAIT_EVENT → COLLECTING → READY."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])
        try:
            # WAIT_EVENT
            data = self._build_status_reply(0, [FTP_WAIT_EVENT])
            rq.put((0, data, False))
            import time

            time.sleep(0.1)
            assert handle.device_states[1] == SnapshotState.WAIT_EVENT

            # COLLECTING
            data = self._build_status_reply(0, [FTP_COLLECTING])
            rq.put((0, data, False))
            time.sleep(0.1)
            assert handle.device_states[1] == SnapshotState.COLLECTING

            # READY
            data = self._build_status_reply(0, [0])
            rq.put((0, data, False))
            assert handle.wait(timeout=2.0)
            assert handle.state == SnapshotState.READY
        finally:
            handle.cancel()

    def test_monitor_error_state(self):
        """Negative ACNET status marks devices as ERROR and raises on wait()."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])
        try:
            # Negative ACNET header status
            rq.put((-241, b"", False))
            with pytest.raises(AcnetError):
                handle.wait(timeout=2.0)
            assert handle.state == SnapshotState.ERROR
        finally:
            handle.cancel()

    def test_monitor_per_device_error(self):
        """Negative per-device error in status update → ERROR state."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])
        try:
            data = self._build_status_reply(0, [-241])
            rq.put((0, data, False))
            with pytest.raises(AcnetError):
                handle.wait(timeout=2.0)
        finally:
            handle.cancel()

    def test_two_devices_one_ready_one_pending(self):
        """Aggregate state reflects least-progressed device."""
        dev1 = FTPDevice(di=1, pi=12, ssdn=b"\x00" * 8)
        dev2 = FTPDevice(di=2, pi=12, ssdn=b"\x00" * 8)
        handle, rq = self._make_handle(
            per_device_errors=[FTP_PEND, FTP_PEND],
            devices=[dev1, dev2],
        )
        try:
            # First device becomes ready, second still pending
            data = self._build_status_reply(0, [0, FTP_COLLECTING])
            rq.put((0, data, False))
            import time

            time.sleep(0.1)
            assert handle.device_states[1] == SnapshotState.READY
            assert handle.device_states[2] == SnapshotState.COLLECTING
            assert handle.state == SnapshotState.COLLECTING
            assert not handle.is_ready

            # Second device becomes ready
            data = self._build_status_reply(0, [0, 0])
            rq.put((0, data, False))
            assert handle.wait(timeout=2.0)
            assert handle.state == SnapshotState.READY
            assert handle.is_ready
        finally:
            handle.cancel()

    def test_wait_timeout(self):
        """wait() returns False on timeout."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])
        try:
            assert handle.wait(timeout=0.1) is False
        finally:
            handle.cancel()

    def test_cancel_stops_monitor_thread(self):
        """cancel() terminates the monitor thread."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])
        handle.cancel()
        assert not handle._monitor_thread.is_alive()

    def test_context_manager_cleanup(self):
        """__exit__ calls cancel and stops the monitor thread."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])
        with handle:
            assert handle._monitor_thread.is_alive()
        assert not handle._monitor_thread.is_alive()

    def test_ready_state_not_downgraded(self):
        """Once READY, device state is not changed by subsequent updates."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])
        try:
            # Mark as ready
            data = self._build_status_reply(0, [0])
            rq.put((0, data, False))
            assert handle.wait(timeout=2.0)

            # Send a COLLECTING update — should be ignored
            data = self._build_status_reply(0, [FTP_COLLECTING])
            rq.put((0, data, False))
            import time

            time.sleep(0.1)
            assert handle.device_states[1] == SnapshotState.READY
        finally:
            handle.cancel()

    def test_restart_resets_states(self):
        """restart() resets device states to PENDING so wait() can be reused."""
        handle, rq = self._make_handle(per_device_errors=[FTP_PEND])

        # Mark as ready
        data = self._build_status_reply(0, [0])
        rq.put((0, data, False))
        assert handle.wait(timeout=2.0)
        assert handle.is_ready

        # Mock the restart network call
        def fake_request_single(node, task, data, reply_handler, timeout):
            reply = MagicMock()
            reply.status = 0
            reply.data = b""
            reply_handler(reply)

        handle._connection.request_single = fake_request_single
        handle.restart()

        assert handle.state == SnapshotState.PENDING
        assert not handle.is_ready

        # New cycle: device becomes ready again
        data = self._build_status_reply(0, [0])
        rq.put((0, data, False))
        assert handle.wait(timeout=2.0)
        assert handle.is_ready
        handle.cancel()
