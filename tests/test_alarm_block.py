"""Tests for alarm block parsing and manipulation."""

import struct

import pytest

from pacsys.alarm_block import (
    AlarmFlags,
    AnalogAlarm,
    DataLength,
    DataType,
    DigitalAlarm,
    FTD,
    LimitType,
)


class TestFTD:
    def test_periodic_from_word(self):
        # 60 ticks = 1 Hz
        ftd = FTD.from_word(60)
        assert ftd.is_periodic
        assert ftd.period_ticks == 60
        assert ftd.rate_hz == pytest.approx(1.0)

    def test_periodic_to_word(self):
        ftd = FTD.periodic_hz(1.0)
        assert ftd.to_word() == 60

    def test_event_from_word(self):
        # 0x800F = sample on event $0F with no delay
        ftd = FTD.from_word(0x800F)
        assert not ftd.is_periodic
        assert ftd.clock_event == 0x0F
        assert ftd.delay_10ms == 0

    def test_event_with_delay(self):
        # Event $0F with 50ms delay (5 * 10ms)
        ftd = FTD.on_event(0x0F, delay_ms=50)
        assert not ftd.is_periodic
        assert ftd.clock_event == 0x0F
        assert ftd.delay_10ms == 5
        word = ftd.to_word()
        assert word == 0x850F  # 0x8000 | (5 << 8) | 0x0F

    def test_default_ftd(self):
        ftd = FTD.default()
        assert ftd.is_periodic
        assert ftd.period_ticks == 0
        assert ftd.to_word() == 0


class TestAnalogAlarm:
    def test_parse_from_bytes(self):
        """Parse analog alarm from raw bytes (ACNET network order)."""
        # Build 20-byte block: flags with MIN_MAX (K=2), 2-byte data (Q=1)
        # Flags: BYPASS=1, K=2 (bits 8-9), Q=1 (bits 5-6) = 0x0221
        flags = AlarmFlags.BYPASS | (2 << 8) | (1 << 5)
        data = struct.pack(
            "<H4s4sBBH6s",
            flags,
            struct.pack("<i", 100),  # value1 = min = 100
            struct.pack("<i", 500),  # value2 = max = 500
            0,  # tries_now (swapped in network order)
            3,  # tries_needed (swapped in network order)
            60,  # FTD = 1 Hz periodic
            b"\x00\x00\x01\x00\x00\x00",  # fe_data with data_type=SIGNED_INT
        )
        alarm = AnalogAlarm.from_bytes(data)

        assert alarm.is_active
        assert alarm.limit_type == LimitType.MIN_MAX
        assert alarm.data_length == DataLength.BYTES_2
        assert alarm.tries_needed == 3
        assert alarm.tries_now == 0
        assert alarm.ftd.rate_hz == pytest.approx(1.0)
        # Raw values (from_bytes doesn't have structured data)
        assert alarm._min_value_raw == 100
        assert alarm._max_value_raw == 500

    def test_raw_min_max_values_int(self):
        """Test raw value packing/unpacking with signed int."""
        alarm = AnalogAlarm()
        alarm.data_type = DataType.SIGNED_INT
        alarm.data_length = DataLength.BYTES_2
        alarm.limit_type = LimitType.MIN_MAX

        alarm._min_value_raw = -100
        alarm._max_value_raw = 200

        assert alarm._min_value_raw == -100
        assert alarm._max_value_raw == 200

    def test_raw_min_max_values_float(self):
        """Test raw value packing/unpacking with float."""
        alarm = AnalogAlarm()
        alarm.data_type = DataType.FLOAT
        alarm.limit_type = LimitType.MIN_MAX

        alarm._min_value_raw = -10.5
        alarm._max_value_raw = 100.25

        assert alarm._min_value_raw == pytest.approx(-10.5)
        assert alarm._max_value_raw == pytest.approx(100.25)

    def test_round_trip(self):
        """Test serialize/deserialize preserves raw values."""
        alarm = AnalogAlarm()
        alarm.is_active = True
        alarm.abort = True
        alarm.abort_inhibit = False
        alarm.data_type = DataType.FLOAT
        alarm.limit_type = LimitType.MIN_MAX
        alarm._min_value_raw = 0.0
        alarm._max_value_raw = 100.0
        alarm.tries_needed = 3
        alarm.ftd = FTD.on_event(0x0F, delay_ms=100)

        data = alarm.to_bytes()
        parsed = AnalogAlarm.from_bytes(data)

        assert parsed.is_active
        assert parsed.abort
        assert not parsed.abort_inhibit
        assert parsed.limit_type == LimitType.MIN_MAX
        assert parsed._min_value_raw == pytest.approx(0.0)
        assert parsed._max_value_raw == pytest.approx(100.0)
        assert parsed.tries_needed == 3
        assert not parsed.ftd.is_periodic
        assert parsed.ftd.clock_event == 0x0F


class TestDigitalAlarm:
    def test_parse_from_bytes(self):
        """Parse digital alarm from raw bytes (ACNET network order)."""
        # Flags: BYPASS=1, DIGITAL=1 (bit 7), Q=1 (bits 5-6) = 0x00A1
        flags = AlarmFlags.BYPASS | AlarmFlags.DIGITAL | (1 << 5)
        data = struct.pack(
            "<H4s4sBBH6s",
            flags,
            struct.pack("<I", 0x00FF),  # value1 = nominal
            struct.pack("<I", 0xFF00),  # value2 = mask
            1,  # tries_now (swapped)
            2,  # tries_needed (swapped)
            0,  # FTD = default
            b"\x00" * 6,
        )
        alarm = DigitalAlarm.from_bytes(data)

        assert alarm.is_active
        assert alarm.is_digital
        assert alarm.data_length == DataLength.BYTES_2
        assert alarm.tries_needed == 2
        assert alarm.tries_now == 1
        assert alarm.nominal == 0x00FF
        assert alarm.mask == 0xFF00

    def test_round_trip(self):
        alarm = DigitalAlarm()
        alarm.is_active = True
        alarm.data_length = DataLength.BYTES_2
        alarm.nominal = 0x1234
        alarm.mask = 0xFFFF
        alarm.tries_needed = 2

        data = alarm.to_bytes()
        parsed = DigitalAlarm.from_bytes(data)

        assert parsed.is_active
        assert parsed.nominal == 0x1234
        assert parsed.mask == 0xFFFF
        assert parsed.tries_needed == 2


class TestFlagsIntegrity:
    """Ensure flag manipulation doesn't corrupt other bits."""

    def test_toggle_bypass_preserves_flags(self):
        alarm = AnalogAlarm()
        alarm.flags = 0xFFFF  # all bits set
        alarm.bypass = True  # clear BYPASS bit
        assert alarm.flags == 0xFFFE  # only bit 0 cleared

    def test_set_data_length_preserves_flags(self):
        alarm = AnalogAlarm()
        alarm.flags = 0xFFFF
        alarm.data_length = DataLength.BYTES_1  # Q=0
        # Bits 5-6 should be 0, others unchanged
        assert alarm.flags == 0xFF9F


class TestEngineeringUnits:
    """Test engineering unit properties for modify() context."""

    def test_minimum_maximum_without_structured(self):
        """Without structured data, engineering unit properties return None."""
        alarm = AnalogAlarm()
        alarm.data_type = DataType.FLOAT
        alarm._min_value_raw = 10.0  # raw value
        alarm._max_value_raw = 100.0

        assert alarm.minimum is None  # no structured data
        assert alarm.maximum is None

    def test_minimum_maximum_with_structured(self):
        """With structured data attached, engineering unit properties work."""
        alarm = AnalogAlarm()
        alarm._structured = {"minimum": 5.0, "maximum": 50.0}

        assert alarm.minimum == 5.0
        assert alarm.maximum == 50.0

        alarm.minimum = 10.0
        alarm.maximum = 100.0
        assert alarm._structured["minimum"] == 10.0
        assert alarm._structured["maximum"] == 100.0

    def test_setting_without_structured_raises(self):
        """Setting engineering units without structured data raises."""
        alarm = AnalogAlarm()
        with pytest.raises(ValueError, match="No structured data"):
            alarm.minimum = 10.0

    def test_flag_setters_sync_structured(self):
        """Flag setters should update _structured when present."""
        alarm = AnalogAlarm()
        alarm._structured = {"alarm_enable": True, "abort": False, "abort_inhibit": False}

        alarm.bypass = True  # sets is_active = False
        assert alarm._structured["alarm_enable"] is False

        alarm.abort = True
        assert alarm._structured["abort"] is True

        alarm.abort_inhibit = True
        assert alarm._structured["abort_inhibit"] is True


class TestModifyContext:
    """Test _AlarmModifyContext read-modify-write pattern."""

    def test_modify_reads_both_raw_and_structured(self, fake_backend):
        """modify() should fetch both raw and structured data."""
        # Setup raw alarm block
        alarm_data = AnalogAlarm()
        alarm_data.is_active = True
        alarm_data.data_type = DataType.FLOAT
        alarm_data._min_value_raw = 10.0
        alarm_data._max_value_raw = 100.0
        raw_bytes = alarm_data.to_bytes()

        # Setup structured response (engineering units - could differ due to transform)
        structured = {
            "minimum": 32.0,  # Fahrenheit (if raw was Celsius)
            "maximum": 212.0,
            "alarm_enable": True,
            "alarm_status": False,
            "abort": False,
            "abort_inhibit": False,
            "tries_needed": 1,
            "tries_now": 0,
        }

        # Note: modify() requests with @I event suffix
        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", raw_bytes)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", structured)

        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            # Engineering unit values (from structured)
            assert alarm.minimum == pytest.approx(32.0)
            assert alarm.maximum == pytest.approx(212.0)

    def test_modify_detects_eng_unit_change(self, fake_backend):
        """Changing engineering units should trigger structured write."""
        alarm_data = AnalogAlarm()
        alarm_data.data_type = DataType.FLOAT
        raw_bytes = alarm_data.to_bytes()

        structured = {
            "minimum": 0.0,
            "maximum": 100.0,
            "alarm_enable": True,
            "alarm_status": False,
            "abort": False,
            "abort_inhibit": False,
            "tries_needed": 1,
            "tries_now": 0,
        }

        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", raw_bytes)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", structured)

        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.maximum = 200.0  # change in engineering units

        # Should have written structured, not raw
        writes = fake_backend.writes
        assert len(writes) == 1
        drf, value = writes[0]
        assert "ANALOG" in drf
        assert isinstance(value, dict)  # structured write
        assert value["maximum"] == 200.0

    def test_modify_detects_ftd_change_uses_raw(self, fake_backend):
        """Changing FTD should trigger raw write."""
        alarm_data = AnalogAlarm()
        alarm_data.ftd = FTD.periodic_hz(1.0)
        raw_bytes = alarm_data.to_bytes()

        structured = {
            "minimum": 0.0,
            "maximum": 100.0,
            "alarm_enable": True,
            "alarm_status": False,
            "abort": False,
            "abort_inhibit": False,
            "tries_needed": 1,
            "tries_now": 0,
        }

        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", raw_bytes)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", structured)

        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.ftd = FTD.periodic_hz(10.0)  # raw-only field

        # Should have written raw, not structured
        writes = fake_backend.writes
        assert len(writes) == 1
        drf, value = writes[0]
        assert ".RAW" in drf
        assert isinstance(value, bytes)

    def test_modify_no_change_no_write(self, fake_backend):
        """No changes should result in no writes."""
        alarm_data = AnalogAlarm()
        raw_bytes = alarm_data.to_bytes()

        structured = {
            "minimum": 0.0,
            "maximum": 100.0,
            "alarm_enable": False,
            "alarm_status": False,
            "abort": False,
            "abort_inhibit": False,
            "tries_needed": 1,
            "tries_now": 0,
        }

        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", raw_bytes)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", structured)

        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as _alarm:
            pass  # no changes

        assert len(fake_backend.writes) == 0


@pytest.fixture
def fake_backend():
    """Provide a FakeBackend for testing."""
    from pacsys.testing import FakeBackend

    return FakeBackend()
