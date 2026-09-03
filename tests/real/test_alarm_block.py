"""
Integration tests for alarm block helper.

Tests that AlarmBlock parsing matches the structured data returned
by reading .ANALOG and .DIGITAL properties directly.
"""

import pytest

from pacsys.alarm_block import AlarmFlags, AnalogAlarm, DigitalAlarm
from pacsys.drf_utils import get_device_name
from pacsys.errors import DeviceError
from pacsys.types import ValueType

from .devices import (
    ANALOG_ALARM_DEVICE,
    DIGITAL_ALARM_DEVICE,
    SCALAR_DEVICE,
    SCALAR_DEVICE_2,
    TIMEOUT_READ,
    requires_dpm_http,
)

# Devices to test - use DRF strings, extract device name with parser
ANALOG_TEST_DEVICES = [
    ANALOG_ALARM_DEVICE,  # N@H801 -> N:H801
    SCALAR_DEVICE,  # M:OUTTMP
    SCALAR_DEVICE_2,  # G:AMANDA
    "Z:ACLTST",  # Test device with IEEE float passthrough
]

DIGITAL_TEST_DEVICES = [
    DIGITAL_ALARM_DEVICE,  # N$H801 -> N:H801
    SCALAR_DEVICE_2,  # G:AMANDA
]

# Device known to NOT have digital alarm - used to verify DeviceError
NO_DIGITAL_ALARM_DEVICE = SCALAR_DEVICE  # M:OUTTMP


def _read_analog_alarm(backend, device: str) -> AnalogAlarm:
    """Read analog alarm, skip test if device has no analog alarm."""
    try:
        return AnalogAlarm.read(device, backend)
    except DeviceError as e:
        pytest.skip(f"{device} has no analog alarm: {e}")


def _read_digital_alarm(backend, device: str) -> DigitalAlarm:
    """Read digital alarm, skip test if device has no digital alarm."""
    try:
        return DigitalAlarm.read(device, backend)
    except DeviceError as e:
        pytest.skip(f"{device} has no digital alarm: {e}")


@requires_dpm_http
class TestAnalogAlarm:
    """Tests for reading and cross-checking analog alarm blocks."""

    @pytest.mark.parametrize("drf", ANALOG_TEST_DEVICES)
    def test_read_and_cross_check(self, dpm_http_backend_cls, drf):
        """Read analog alarm via helper and cross-check with structured read."""
        device = get_device_name(drf)
        alarm = _read_analog_alarm(dpm_http_backend_cls, device)

        # Basic sanity
        assert isinstance(alarm, AnalogAlarm)
        assert len(alarm.to_bytes()) == 20
        assert not alarm.is_digital, "Analog alarm should have AD bit = 0"

        # Cross-check with structured read
        structured = dpm_http_backend_cls.get(f"{device}.ANALOG", timeout=TIMEOUT_READ)
        assert structured.ok, f"Structured read failed: {structured.message}"
        assert structured.value_type == ValueType.ANALOG_ALARM

        s = structured.value
        print(f"\n{device}:")
        print(f"  helper: {alarm}")
        print(f"  structured: {s}")

        # Verify all comparable fields match
        assert alarm.tries_needed == s["tries_needed"], "tries_needed mismatch"
        assert alarm.tries_now == s["tries_now"], "tries_now mismatch"
        assert alarm.is_bad == s["alarm_status"], "alarm_status mismatch"
        assert alarm.is_active == s["alarm_enable"], "alarm_enable mismatch"
        assert alarm.abort == s["abort"], "abort mismatch"
        assert alarm.abort_inhibit == s["abort_inhibit"], "abort_inhibit mismatch"

        # Compare min/max in engineering units (now fetched automatically);
        # alarm._structured comes from the same structured read as s, so a
        # regression to None must not silently skip the cross-check
        assert (alarm.minimum is None) == ("minimum" not in s), "minimum None mismatch"
        if alarm.minimum is not None:
            assert alarm.minimum == pytest.approx(s["minimum"], rel=1e-4), "minimum mismatch"
            assert alarm.maximum == pytest.approx(s["maximum"], rel=1e-4), "maximum mismatch"
            print(f"  eng units: min={alarm.minimum}, max={alarm.maximum}")

    @pytest.mark.parametrize("drf", ANALOG_TEST_DEVICES)
    def test_round_trip(self, dpm_http_backend_cls, drf):
        """Read, serialize, re-parse - should be identical."""
        device = get_device_name(drf)
        alarm = _read_analog_alarm(dpm_http_backend_cls, device)

        data = alarm.to_bytes()
        parsed = AnalogAlarm.from_bytes(data)

        assert parsed.flags == alarm.flags
        assert parsed.value1_raw == alarm.value1_raw
        assert parsed.value2_raw == alarm.value2_raw
        assert parsed.tries_needed == alarm.tries_needed
        assert parsed.tries_now == alarm.tries_now
        assert parsed.ftd.to_word() == alarm.ftd.to_word()
        assert parsed.fe_data == alarm.fe_data

    def test_flag_interpretation(self, dpm_http_backend_cls):
        """Test flag bit accessors are consistent with raw flags."""
        device = get_device_name(ANALOG_ALARM_DEVICE)
        alarm = _read_analog_alarm(dpm_http_backend_cls, device)

        print(f"\nFlag breakdown for {device}:")
        print(f"  raw flags: 0x{alarm.flags:04X}")
        print(f"  ENABLE={bool(alarm.flags & AlarmFlags.ENABLE)} -> is_active={alarm.is_active}")
        print(f"  BAD={bool(alarm.flags & AlarmFlags.BAD)} -> is_bad={alarm.is_bad}")
        print(f"  ABORT={bool(alarm.flags & AlarmFlags.ABORT)} -> abort={alarm.abort}")
        print(f"  ABORT_INHIBIT={bool(alarm.flags & AlarmFlags.ABORT_INHIBIT)} -> abort_inhibit={alarm.abort_inhibit}")
        print(f"  HIGH={bool(alarm.flags & AlarmFlags.HIGH)} -> is_high={alarm.is_high}")
        print(f"  LOW={bool(alarm.flags & AlarmFlags.LOW)} -> is_low={alarm.is_low}")

        # Verify accessors match raw flag bits
        assert alarm.is_active == bool(alarm.flags & AlarmFlags.ENABLE)
        assert alarm.is_bad == bool(alarm.flags & AlarmFlags.BAD)
        assert alarm.abort == bool(alarm.flags & AlarmFlags.ABORT)


@requires_dpm_http
class TestDigitalAlarm:
    """Tests for reading and cross-checking digital alarm blocks."""

    @pytest.mark.parametrize("drf", DIGITAL_TEST_DEVICES)
    def test_read_and_cross_check(self, dpm_http_backend_cls, drf):
        """Read digital alarm via helper and cross-check with structured read."""
        device = get_device_name(drf)
        alarm = _read_digital_alarm(dpm_http_backend_cls, device)

        # Basic sanity
        assert isinstance(alarm, DigitalAlarm)
        assert len(alarm.to_bytes()) == 20
        # G:AMANDA's digital block is stored in the DB with the AD bit clear (flags 0xC020)
        if device != get_device_name(SCALAR_DEVICE_2):
            assert alarm.is_digital, f"{device} digital alarm has AD bit = 0"

        # Cross-check with structured read
        structured = dpm_http_backend_cls.get(f"{device}.DIGITAL", timeout=TIMEOUT_READ)
        assert structured.ok, f"Structured read failed: {structured.message}"
        assert structured.value_type == ValueType.DIGITAL_ALARM

        s = structured.value
        print(f"\n{device}:")
        print(f"  helper: {alarm}")
        print(f"  structured: {s}")

        # Verify all comparable fields match
        assert alarm.tries_needed == s["tries_needed"], "tries_needed mismatch"
        assert alarm.tries_now == s["tries_now"], "tries_now mismatch"
        assert alarm.is_bad == s["alarm_status"], "alarm_status mismatch"
        assert alarm.is_active == s["alarm_enable"], "alarm_enable mismatch"
        assert alarm.abort == s["abort"], "abort mismatch"
        assert alarm.abort_inhibit == s["abort_inhibit"], "abort_inhibit mismatch"
        assert alarm.nominal == s["nominal"], "nominal mismatch"
        assert alarm.mask == s["mask"], "mask mismatch"

    @pytest.mark.parametrize("drf", DIGITAL_TEST_DEVICES)
    def test_round_trip(self, dpm_http_backend_cls, drf):
        """Read, serialize, re-parse - should be identical."""
        device = get_device_name(drf)
        alarm = _read_digital_alarm(dpm_http_backend_cls, device)

        data = alarm.to_bytes()
        parsed = DigitalAlarm.from_bytes(data)

        assert parsed.flags == alarm.flags
        assert parsed.value1_raw == alarm.value1_raw
        assert parsed.value2_raw == alarm.value2_raw
        assert parsed.tries_needed == alarm.tries_needed
        assert parsed.ftd.to_word() == alarm.ftd.to_word()

    def test_flag_interpretation(self, dpm_http_backend_cls):
        """Test flag bit accessors are consistent with raw flags."""
        device = get_device_name(DIGITAL_ALARM_DEVICE)
        alarm = _read_digital_alarm(dpm_http_backend_cls, device)

        print(f"\nFlag breakdown for {device}:")
        print(f"  raw flags: 0x{alarm.flags:04X}")
        print(f"  ENABLE={bool(alarm.flags & AlarmFlags.ENABLE)} -> is_active={alarm.is_active}")
        print(f"  BAD={bool(alarm.flags & AlarmFlags.BAD)} -> is_bad={alarm.is_bad}")
        print(f"  DIGITAL={bool(alarm.flags & AlarmFlags.DIGITAL)} -> is_digital={alarm.is_digital}")

        assert alarm.is_active == bool(alarm.flags & AlarmFlags.ENABLE)
        assert alarm.is_bad == bool(alarm.flags & AlarmFlags.BAD)
        assert alarm.is_digital == bool(alarm.flags & AlarmFlags.DIGITAL)

    def test_no_digital_alarm_raises_device_error(self, dpm_http_backend_cls):
        """Verify DeviceError is raised for device without digital alarm."""
        device = get_device_name(NO_DIGITAL_ALARM_DEVICE)
        with pytest.raises(DeviceError) as exc_info:
            DigitalAlarm.read(device, dpm_http_backend_cls)

        # DBM_NOPROP: facility=16, error=-13
        assert exc_info.value.facility_code == 16
        assert exc_info.value.error_code == -13
        print(f"\n{device}: {exc_info.value}")


def _flags_word(reading) -> int:
    """FLAGS arrives as a float word from DPM/DMQ/gRPC and as hex text (e.g. 'C220') from ACL."""
    assert reading.ok, f"FLAGS read failed: {reading.message}"
    if reading.value_type == ValueType.TEXT:
        return int(reading.value, 16)
    assert reading.value_type == ValueType.SCALAR
    assert float(reading.value).is_integer(), f"FLAGS not integral: {reading.value!r}"
    return int(reading.value)


class TestFlagsField:
    """`.ANALOG.FLAGS` / `.DIGITAL.FLAGS` must equal the flags word of the raw 20-byte block."""

    @pytest.mark.parametrize(
        "drf,prop,alarm_cls",
        [
            (ANALOG_ALARM_DEVICE, "ANALOG", AnalogAlarm),
            (SCALAR_DEVICE_2, "ANALOG", AnalogAlarm),
            (DIGITAL_ALARM_DEVICE, "DIGITAL", DigitalAlarm),
            (SCALAR_DEVICE_2, "DIGITAL", DigitalAlarm),
        ],
    )
    def test_flags_matches_raw_block(self, read_backend_cls, drf, prop, alarm_cls):
        device = get_device_name(drf)
        try:
            alarm = alarm_cls.read(device, read_backend_cls)
        except DeviceError as e:
            pytest.skip(f"{device} has no {prop.lower()} alarm: {e}")

        reading = read_backend_cls.get(f"{device}.{prop}.FLAGS", timeout=TIMEOUT_READ)
        flags = _flags_word(reading)
        print(f"\n{device}.{prop}.FLAGS = {reading.value!r} -> 0x{flags:04X}, raw block flags 0x{alarm.flags:04X}")

        assert 0 <= flags <= 0xFFFF
        assert flags == alarm.flags, "FLAGS field disagrees with raw block flags word"

        # Independent cross-check against server-side structured interpretation
        structured = read_backend_cls.get(f"{device}.{prop}", timeout=TIMEOUT_READ)
        if structured.ok and structured.value_type in (ValueType.ANALOG_ALARM, ValueType.DIGITAL_ALARM):
            s = structured.value
            assert bool(flags & AlarmFlags.ENABLE) == s["alarm_enable"], "ENABLE bit mismatch"
            assert bool(flags & AlarmFlags.BAD) == s["alarm_status"], "BAD bit mismatch"
            assert bool(flags & AlarmFlags.ABORT) == s["abort"], "ABORT bit mismatch"
            assert bool(flags & AlarmFlags.ABORT_INHIBIT) == s["abort_inhibit"], "ABORT_INHIBIT bit mismatch"

    @requires_dpm_http
    @pytest.mark.parametrize("prop", ["ANALOG", "DIGITAL"])
    def test_raw_block_bytes_match_dpm(self, read_backend_cls, dpm_http_backend_cls, prop):
        """Every backend must return the alarm block in the same (little-endian wire) byte order as DPM."""
        drf = f"{get_device_name(ANALOG_ALARM_DEVICE)}.{prop}{{0:20}}.RAW"
        expected = dpm_http_backend_cls.read(drf, timeout=TIMEOUT_READ)
        actual = read_backend_cls.read(drf, timeout=TIMEOUT_READ)
        assert isinstance(actual, bytes) and len(actual) == 20
        assert actual == expected, f"{drf}: {actual.hex()} != DPM {expected.hex()}"


@requires_dpm_http
class TestFTD:
    """Tests for FTD field interpretation."""

    def test_ftd_interpretation(self, dpm_http_backend_cls):
        """Read FTD from real device and verify interpretation."""
        device = get_device_name(ANALOG_ALARM_DEVICE)
        alarm = _read_analog_alarm(dpm_http_backend_cls, device)
        ftd = alarm.ftd
        ftd_word = ftd.to_word()

        print(f"\nFTD for {device}: 0x{ftd_word:04X}")

        if ftd_word == 0x0000:
            print("  -> Use device default")
            assert ftd.is_periodic
            assert ftd.period_ticks == 0
        elif ftd_word & 0x8000:
            print(f"  -> Event ${ftd.clock_event:02X} + {ftd.delay_10ms * 10}ms delay")
            assert not ftd.is_periodic
        else:
            print(f"  -> {ftd.rate_hz:.2f} Hz periodic ({ftd.period_ticks} ticks)")
            assert ftd.is_periodic
