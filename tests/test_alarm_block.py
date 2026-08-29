"""Tests for alarm block parsing and manipulation."""

import struct

import pytest

from pacsys.alarm_block import (
    FTD,
    AlarmFlags,
    AnalogAlarm,
    DataLength,
    DataType,
    DigitalAlarm,
    LimitType,
)
from pacsys.errors import DeviceError
from pacsys.types import ValueType


def _analog_structured(**overrides):
    value = {
        "minimum": 0.0,
        "maximum": 100.0,
        "alarm_enable": True,
        "alarm_status": False,
        "abort": False,
        "abort_inhibit": False,
        "tries_needed": 1,
        "tries_now": 0,
    }
    value.update(overrides)
    return value


def _digital_structured(**overrides):
    value = {
        "nominal": 0x1234,
        "mask": 0xFFFF,
        "alarm_enable": True,
        "alarm_status": False,
        "abort": False,
        "abort_inhibit": False,
        "tries_needed": 1,
        "tries_now": 0,
    }
    value.update(overrides)
    return value


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

    @pytest.mark.parametrize(
        ("make", "match"),
        [
            (lambda: FTD.on_event(0x1FF), "clock_event"),
            (lambda: FTD.on_event(0x0F, delay_ms=-10), "delay_ms"),
            (lambda: FTD.on_event(0x0F, delay_ms=5000), "delay_ms"),
            (lambda: FTD.periodic_hz(0), "hz must be positive"),
            (lambda: FTD.periodic_hz(120), "not exceed 60"),
            (lambda: FTD.periodic_ticks(0x8000), "period_ticks"),
            (lambda: FTD(is_periodic=False, delay_10ms=0x80), "delay_10ms"),
        ],
    )
    def test_out_of_range_values_rejected(self, make, match):
        """Values that the 16-bit encoding would silently mask into a different FTD must raise."""
        with pytest.raises(ValueError, match=match):
            make()

    def test_boundary_values_round_trip(self):
        assert FTD.on_event(0xFF, delay_ms=1270).to_word() == 0xFFFF
        assert FTD.periodic_ticks(0x7FFF).to_word() == 0x7FFF
        assert FTD.periodic_hz(60.0).period_ticks == 1
        assert FTD.from_word(0xFFFF) == FTD.on_event(0xFF, delay_ms=1270)

    def test_to_word_rejects_mutated_field(self):
        ftd = FTD.on_event(0x0F)
        ftd.clock_event = 0x1FF
        with pytest.raises(ValueError, match="clock_event"):
            ftd.to_word()


class TestAnalogAlarm:
    def test_parse_from_bytes(self):
        """Parse analog alarm from raw bytes (ACNET network order)."""
        # Build 20-byte block: flags with MIN_MAX (K=2), 2-byte data (Q=1)
        # Flags: ENABLE=1, K=2 (bits 8-9), Q=1 (bits 5-6) = 0x0221
        flags = AlarmFlags.ENABLE | (2 << 8) | (1 << 5)
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
        alarm.flags |= AlarmFlags.ABORT
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
        # Flags: ENABLE=1, DIGITAL=1 (bit 7), Q=1 (bits 5-6) = 0x00A1
        flags = AlarmFlags.ENABLE | AlarmFlags.DIGITAL | (1 << 5)
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
        alarm.bypass = True  # clear the ENABLE bit
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

    def test_writable_flag_setters_sync_structured(self):
        """Writable flag setters should update _structured when present."""
        alarm = AnalogAlarm()
        alarm._structured = {"alarm_enable": True, "abort": False, "abort_inhibit": False}

        alarm.bypass = True  # sets is_active = False
        assert alarm._structured["alarm_enable"] is False

        alarm.abort_inhibit = True
        assert alarm._structured["abort_inhibit"] is True

    def test_abort_is_read_only(self):
        alarm = AnalogAlarm()

        with pytest.raises(AttributeError):
            alarm.abort = True


class TestAlarmSegments:
    @pytest.mark.parametrize("alarm_cls", [AnalogAlarm, DigitalAlarm])
    def test_structured_read_rejects_nonzero_segment_before_io(self, alarm_cls, fake_backend):
        with pytest.raises(ValueError, match="supports only segment=0"):
            alarm_cls.read("Z:TEST", backend=fake_backend, segment=1)

        assert fake_backend.reads == []
        assert fake_backend.writes == []

    @pytest.mark.parametrize("alarm_cls", [AnalogAlarm, DigitalAlarm])
    def test_modify_rejects_nonzero_segment_immediately(self, alarm_cls, fake_backend):
        with pytest.raises(ValueError, match="supports only segment=0"):
            alarm_cls.modify("Z:TEST", backend=fake_backend, segment=1)

        assert fake_backend.reads == []
        assert fake_backend.writes == []

    @pytest.mark.parametrize(
        ("alarm", "prop"),
        [(AnalogAlarm(), "ANALOG"), (DigitalAlarm(), "DIGITAL")],
    )
    def test_raw_write_supports_nonzero_segment(self, alarm, prop, fake_backend):
        alarm.write("Z:TEST", backend=fake_backend, segment=2)
        assert fake_backend.writes[0][0] == f"Z:TEST.{prop}{{40:20}}.RAW@N"

    def test_raw_write_preserves_abort_flag(self, fake_backend):
        alarm = AnalogAlarm(flags=AlarmFlags.ABORT)

        alarm.write("Z:TEST", backend=fake_backend)

        _, value = fake_backend.writes[0]
        assert isinstance(value, bytes)
        assert AnalogAlarm.from_bytes(value).abort


class TestAlarmReadStatus:
    @pytest.mark.parametrize(("alarm_cls", "prop"), [(AnalogAlarm, "ANALOG"), (DigitalAlarm, "DIGITAL")])
    def test_read_rejects_warning_without_value(self, alarm_cls, prop, fake_backend):
        fake_backend.set_error(f"Z:TEST.{prop}{{0:20}}.RAW@I", 1, "DPM_PEND")

        with pytest.raises(DeviceError) as exc_info:
            alarm_cls.read("Z:TEST", backend=fake_backend)

        assert exc_info.value.error_code == 1

    @pytest.mark.parametrize(("alarm_cls", "prop"), [(AnalogAlarm, "ANALOG"), (DigitalAlarm, "DIGITAL")])
    def test_modify_rejects_warning_without_value(self, alarm_cls, prop, fake_backend):
        fake_backend.set_error(f"Z:TEST.{prop}{{0:20}}.RAW@I", 1, "DPM_PEND")

        with pytest.raises(DeviceError) as exc_info:
            with alarm_cls.modify("Z:TEST", backend=fake_backend):
                pass

        assert exc_info.value.error_code == 1


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
        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", raw_bytes, value_type=ValueType.RAW)
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

        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", raw_bytes, value_type=ValueType.RAW)
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
        assert "abort" not in value

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

        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", raw_bytes, value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", structured)

        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.ftd = FTD.periodic_hz(10.0)  # raw-only field

        # Should have written raw, not structured
        writes = fake_backend.writes
        assert len(writes) == 1
        drf, value = writes[0]
        assert ".RAW" in drf
        assert isinstance(value, bytes)

    @pytest.mark.parametrize(("field", "changed"), [("value1", 20.0), ("value2", 200.0)])
    def test_modify_analog_raw_value_change_uses_raw(self, fake_backend, field, changed):
        alarm_data = AnalogAlarm()
        alarm_data.data_type = DataType.FLOAT
        alarm_data.value1 = 10.0
        alarm_data.value2 = 100.0
        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", alarm_data.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", _analog_structured())

        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            setattr(alarm, field, changed)

        assert len(fake_backend.writes) == 1
        drf, value = fake_backend.writes[0]
        assert drf == "Z:TEST.ANALOG{0:20}.RAW@N"
        assert isinstance(value, bytes)
        assert getattr(AnalogAlarm.from_bytes(value), field) == pytest.approx(changed)

    @pytest.mark.parametrize(("raw_field", "eng_field"), [("value1", "minimum"), ("value2", "maximum")])
    def test_modify_rejects_same_analog_limit_in_raw_and_engineering_units(self, fake_backend, raw_field, eng_field):
        alarm_data = AnalogAlarm()
        alarm_data.data_type = DataType.FLOAT
        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", alarm_data.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", _analog_structured())

        with pytest.raises(ValueError, match="Cannot change both"):
            with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
                setattr(alarm, raw_field, 20.0)
                setattr(alarm, eng_field, 40.0)

        assert fake_backend.writes == []

    @pytest.mark.parametrize("field", ["data_length", "data_type"])
    def test_modify_rejects_engineering_limits_with_format_change(self, fake_backend, field):
        alarm_data = AnalogAlarm()
        alarm_data.data_type = DataType.SIGNED_INT
        alarm_data.data_length = DataLength.BYTES_2
        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", alarm_data.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", _analog_structured())

        with pytest.raises(ValueError, match="engineering limits"):
            with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
                alarm.maximum = 200.0
                setattr(
                    alarm,
                    field,
                    DataLength.BYTES_4 if field == "data_length" else DataType.FLOAT,
                )

        assert fake_backend.writes == []

    @pytest.mark.parametrize(("field", "changed"), [("nominal", 0xABCD), ("mask", 0x00FF)])
    def test_modify_digital_value_change_uses_current_block(self, fake_backend, field, changed):
        alarm_data = DigitalAlarm()
        alarm_data.flags |= AlarmFlags.DIGITAL
        alarm_data.data_length = DataLength.BYTES_2
        alarm_data.nominal = 0x1234
        alarm_data.mask = 0xFFFF

        fake_backend.set_reading("Z:TEST.DIGITAL{0:20}.RAW@I", alarm_data.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_digital_alarm("Z:TEST.DIGITAL@I", _digital_structured())

        with DigitalAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            setattr(alarm, field, changed)

        assert len(fake_backend.writes) == 1
        drf, value = fake_backend.writes[0]
        assert drf == "Z:TEST.DIGITAL@N"
        assert isinstance(value, dict)
        assert value[field] == changed

    def test_modify_analog_tries_needed_uses_current_block(self, fake_backend):
        alarm_data = AnalogAlarm(tries_needed=1)
        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", alarm_data.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", _analog_structured())

        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.tries_needed = 5

        _, value = fake_backend.writes[0]
        assert isinstance(value, dict)
        assert value["tries_needed"] == 5

    def test_modify_digital_tries_needed_uses_current_block(self, fake_backend):
        alarm_data = DigitalAlarm(tries_needed=1)
        fake_backend.set_reading("Z:TEST.DIGITAL{0:20}.RAW@I", alarm_data.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_digital_alarm("Z:TEST.DIGITAL@I", _digital_structured())

        with DigitalAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.tries_needed = 4

        _, value = fake_backend.writes[0]
        assert isinstance(value, dict)
        assert value["tries_needed"] == 4

    def test_modify_without_structured_data_falls_back_to_raw(self, fake_backend):
        alarm_data = DigitalAlarm()
        alarm_data.data_length = DataLength.BYTES_2
        alarm_data.nominal = 0x1234
        fake_backend.set_reading("Z:TEST.DIGITAL{0:20}.RAW@I", alarm_data.to_bytes(), value_type=ValueType.RAW)

        with DigitalAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.nominal = 0xABCD

        assert len(fake_backend.writes) == 1
        drf, value = fake_backend.writes[0]
        assert drf == "Z:TEST.DIGITAL{0:20}.RAW@N"
        assert isinstance(value, bytes)
        assert DigitalAlarm.from_bytes(value).nominal == 0xABCD

    def test_modify_combined_digital_changes_survive_fresh_raw_patch(self, fake_backend):
        alarm_data = DigitalAlarm(tries_needed=1)
        alarm_data.flags |= AlarmFlags.DIGITAL
        alarm_data.data_length = DataLength.BYTES_2
        alarm_data.nominal = 0x1234
        alarm_data.mask = 0xFFFF
        alarm_data.ftd = FTD.periodic_hz(1.0)
        raw_read_drf = "Z:TEST.DIGITAL{0:20}.RAW@I"
        fake_backend.set_reading(raw_read_drf, alarm_data.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_digital_alarm("Z:TEST.DIGITAL@I", _digital_structured())

        with DigitalAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.nominal = 0xABCD
            alarm.mask = 0x00FF
            alarm.tries_needed = 4
            alarm.ftd = FTD.periodic_hz(10.0)

        assert len(fake_backend.writes) == 2
        _, structured = fake_backend.writes[0]
        assert isinstance(structured, dict)
        assert structured["nominal"] == 0xABCD
        assert structured["mask"] == 0x00FF
        assert structured["tries_needed"] == 4

        raw_write_drf, raw_value = fake_backend.writes[1]
        assert raw_write_drf == "Z:TEST.DIGITAL{0:20}.RAW@N"
        assert isinstance(raw_value, bytes)
        patched = DigitalAlarm.from_bytes(raw_value)
        assert patched.nominal == 0xABCD
        assert patched.mask == 0x00FF
        assert patched.tries_needed == 4
        assert patched.ftd == FTD.periodic_hz(10.0)

    def test_modify_combined_analog_keeps_fresh_transformed_limits(self, fake_backend):
        initial = AnalogAlarm(tries_needed=1)
        initial.data_type = DataType.FLOAT
        initial._min_value_raw = 10.0
        initial._max_value_raw = 100.0
        raw_read_drf = "Z:TEST.ANALOG{0:20}.RAW@I"
        fake_backend.set_reading(raw_read_drf, initial.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", _analog_structured(minimum=32.0, maximum=212.0))

        fresh = AnalogAlarm.from_bytes(initial.to_bytes())
        fresh._min_value_raw = 12.0
        fresh._max_value_raw = 104.0

        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.maximum = 220.0
            alarm.ftd = FTD.periodic_hz(10.0)
            fake_backend.set_reading(raw_read_drf, fresh.to_bytes(), value_type=ValueType.RAW)

        raw_write_drf, raw_value = fake_backend.writes[1]
        assert raw_write_drf == "Z:TEST.ANALOG{0:20}.RAW@N"
        assert isinstance(raw_value, bytes)
        patched = AnalogAlarm.from_bytes(raw_value)
        assert patched._min_value_raw == pytest.approx(12.0)
        assert patched._max_value_raw == pytest.approx(104.0)
        assert patched.ftd == FTD.periodic_hz(10.0)

    def test_modify_combined_analog_patches_only_changed_raw_limit(self, fake_backend):
        initial = AnalogAlarm()
        initial.data_type = DataType.FLOAT
        initial.value1 = 10.0
        initial.value2 = 100.0
        raw_read_drf = "Z:TEST.ANALOG{0:20}.RAW@I"
        fake_backend.set_reading(raw_read_drf, initial.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", _analog_structured())

        fresh = AnalogAlarm.from_bytes(initial.to_bytes())
        fresh.value1 = 12.0
        fresh.value2 = 104.0
        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.value1 = 20.0
            alarm.maximum = 220.0
            fake_backend.set_reading(raw_read_drf, fresh.to_bytes(), value_type=ValueType.RAW)

        _, structured = fake_backend.writes[0]
        assert isinstance(structured, dict)
        assert structured["maximum"] == 220.0
        _, raw_value = fake_backend.writes[1]
        assert isinstance(raw_value, bytes)
        patched = AnalogAlarm.from_bytes(raw_value)
        assert patched.value1 == pytest.approx(20.0)
        assert patched.value2 == pytest.approx(104.0)

    def test_modify_combined_change_requires_fresh_raw_bytes(self, fake_backend, monkeypatch):
        initial = AnalogAlarm()
        raw_read_drf = "Z:TEST.ANALOG{0:20}.RAW@I"
        fake_backend.set_reading(raw_read_drf, initial.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", _analog_structured())

        with pytest.raises(TypeError, match="Expected bytes, got float"):
            with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
                alarm.maximum = 200.0
                alarm.ftd = FTD.periodic_hz(10.0)
                monkeypatch.setattr(fake_backend, "read", lambda *_args, **_kwargs: 1.0)

        assert len(fake_backend.writes) == 1
        assert isinstance(fake_backend.writes[0][1], dict)

    def test_modify_combined_change_preserves_direct_abort_flag(self, fake_backend):
        initial = AnalogAlarm()
        initial.data_type = DataType.FLOAT
        raw_read_drf = "Z:TEST.ANALOG{0:20}.RAW@I"
        fake_backend.set_reading(raw_read_drf, initial.to_bytes(), value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", _analog_structured())

        fresh = AnalogAlarm.from_bytes(initial.to_bytes())
        fresh.value2 = 104.0
        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as alarm:
            alarm.flags |= AlarmFlags.ABORT
            alarm.maximum = 220.0
            fake_backend.set_reading(raw_read_drf, fresh.to_bytes(), value_type=ValueType.RAW)

        assert len(fake_backend.writes) == 2
        _, raw_value = fake_backend.writes[1]
        assert isinstance(raw_value, bytes)
        patched = AnalogAlarm.from_bytes(raw_value)
        assert patched.abort
        assert patched.value2 == pytest.approx(104.0)

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

        fake_backend.set_reading("Z:TEST.ANALOG{0:20}.RAW@I", raw_bytes, value_type=ValueType.RAW)
        fake_backend.set_analog_alarm("Z:TEST.ANALOG@I", structured)

        with AnalogAlarm.modify("Z:TEST", backend=fake_backend) as _alarm:
            pass  # no changes

        assert len(fake_backend.writes) == 0


@pytest.fixture
def fake_backend():
    """Provide a FakeBackend for testing."""
    from pacsys.testing import FakeBackend

    return FakeBackend()
