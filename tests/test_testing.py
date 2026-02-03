"""
Tests for pacsys.testing module (FakeBackend).

Tests cover:
- Configuration methods (set_reading, set_error, set_write_result)
- Inspection methods (reads, writes, was_read, was_written, get_written_value)
- Backend interface (read, get, get_many, write, write_many)
- Reset functionality
- Context manager support
- Integration with Device objects
"""

import pytest
from datetime import datetime
import numpy as np

from pacsys.acnet.errors import ERR_NOPROP, ERR_RETRY, FACILITY_DBM
from pacsys.testing import FakeBackend
from pacsys.types import ValueType
from pacsys.errors import DeviceError
from pacsys.device import Device, ScalarDevice, ArrayDevice


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSetReading:
    """Tests for set_reading() configuration."""

    def test_set_reading_basic(self):
        """set_reading configures a simple scalar value."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)

        assert fake.read("M:OUTTMP") == 72.5

    def test_set_reading_with_value_type(self):
        """set_reading configures value type correctly."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5, value_type=ValueType.SCALAR)

        reading = fake.get("M:OUTTMP")
        assert reading.value_type == ValueType.SCALAR

    def test_set_reading_with_array_value(self):
        """set_reading handles array values."""
        fake = FakeBackend()
        arr = np.array([1.0, 2.0, 3.0])
        fake.set_reading("B:HS23T[0:3]", arr, value_type=ValueType.SCALAR_ARRAY)

        result = fake.read("B:HS23T[0:3]")
        np.testing.assert_array_equal(result, arr)

    def test_set_reading_with_units(self):
        """set_reading configures units in metadata."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5, units="degF")

        reading = fake.get("M:OUTTMP")
        assert reading.meta.units == "degF"

    def test_set_reading_with_description(self):
        """set_reading configures description in metadata."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5, description="Outside temperature")

        reading = fake.get("M:OUTTMP")
        assert reading.meta.description == "Outside temperature"

    def test_set_reading_with_timestamp(self):
        """set_reading configures timestamp."""
        fake = FakeBackend()
        ts = datetime(2025, 1, 15, 12, 0, 0)
        fake.set_reading("M:OUTTMP", 72.5, timestamp=ts)

        reading = fake.get("M:OUTTMP")
        assert reading.timestamp == ts

    def test_set_reading_with_cycle(self):
        """set_reading configures cycle number."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5, cycle=42)

        reading = fake.get("M:OUTTMP")
        assert reading.cycle == 42

    def test_set_reading_overwrites_error(self):
        """set_reading removes any configured error for same DRF."""
        fake = FakeBackend()
        fake.set_error("M:OUTTMP", -42, "Device error")
        fake.set_reading("M:OUTTMP", 72.5)

        # Should succeed, not raise
        assert fake.read("M:OUTTMP") == 72.5

    def test_set_reading_string_value(self):
        """set_reading handles string values."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP.DESCRIPTION", "Temperature sensor", value_type=ValueType.TEXT)

        assert fake.read("M:OUTTMP.DESCRIPTION") == "Temperature sensor"


class TestValueTypes:
    """Tests for all ValueType enum values in FakeBackend."""

    def test_scalar_value_type(self):
        """SCALAR value type works correctly."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5, value_type=ValueType.SCALAR)

        reading = fake.get("M:OUTTMP")

        assert reading.ok
        assert reading.value == 72.5
        assert reading.value_type == ValueType.SCALAR

    def test_scalar_array_value_type(self):
        """SCALAR_ARRAY value type works correctly."""
        fake = FakeBackend()
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fake.set_reading("B:HS23T[0:5]", arr, value_type=ValueType.SCALAR_ARRAY)

        reading = fake.get("B:HS23T[0:5]")

        assert reading.ok
        assert reading.value_type == ValueType.SCALAR_ARRAY
        np.testing.assert_array_equal(reading.value, arr)

    def test_raw_value_type(self):
        """RAW value type works correctly with bytes."""
        fake = FakeBackend()
        raw_data = b"\x00\x01\x02\x03\xff\xfe\xfd"
        fake.set_reading("D:RAWDEV", raw_data, value_type=ValueType.RAW)

        reading = fake.get("D:RAWDEV")

        assert reading.ok
        assert reading.value_type == ValueType.RAW
        assert reading.value == raw_data
        assert isinstance(reading.value, bytes)

    def test_text_value_type(self):
        """TEXT value type works correctly."""
        fake = FakeBackend()
        fake.set_reading("M:DESC", "Temperature sensor", value_type=ValueType.TEXT)

        reading = fake.get("M:DESC")

        assert reading.ok
        assert reading.value_type == ValueType.TEXT
        assert reading.value == "Temperature sensor"
        assert isinstance(reading.value, str)

    def test_text_array_value_type(self):
        """TEXT_ARRAY value type works correctly with list of strings."""
        fake = FakeBackend()
        text_list = ["Line 1", "Line 2", "Line 3"]
        fake.set_reading("D:TEXTARR", text_list, value_type=ValueType.TEXT_ARRAY)

        reading = fake.get("D:TEXTARR")

        assert reading.ok
        assert reading.value_type == ValueType.TEXT_ARRAY
        assert reading.value == text_list
        assert isinstance(reading.value, list)
        assert all(isinstance(item, str) for item in reading.value)

    def test_analog_alarm_value_type(self):
        """ANALOG_ALARM value type works correctly with dict."""
        fake = FakeBackend()
        alarm_data = {
            "minimum": -10.0,
            "maximum": 100.0,
            "nominal": 72.5,
            "tolerance": 5.0,
            "status": "OK",
            "alarm_enable": True,
            "alarm_flags": 0,
        }
        fake.set_reading("M:OUTTMP.ANALOG", alarm_data, value_type=ValueType.ANALOG_ALARM)

        reading = fake.get("M:OUTTMP.ANALOG")

        assert reading.ok
        assert reading.value_type == ValueType.ANALOG_ALARM
        assert reading.value == alarm_data
        assert isinstance(reading.value, dict)
        assert reading.value["minimum"] == -10.0
        assert reading.value["maximum"] == 100.0

    def test_digital_alarm_value_type(self):
        """DIGITAL_ALARM value type works correctly with dict."""
        fake = FakeBackend()
        alarm_data = {
            "nom_mask": 0xFF,
            "alarm_mask": 0x0F,
            "status": 0x00,
            "alarm_enable": True,
            "alarm_text": ["OFF", "ON", "FAULT", "UNKNOWN"],
        }
        fake.set_reading("D:STATUS.DIGITAL", alarm_data, value_type=ValueType.DIGITAL_ALARM)

        reading = fake.get("D:STATUS.DIGITAL")

        assert reading.ok
        assert reading.value_type == ValueType.DIGITAL_ALARM
        assert reading.value == alarm_data
        assert isinstance(reading.value, dict)
        assert reading.value["nom_mask"] == 0xFF

    def test_basic_status_value_type(self):
        """BASIC_STATUS value type works correctly with dict."""
        fake = FakeBackend()
        status_data = {
            "on": True,
            "ready": True,
            "remote": True,
            "positive": False,
            "ramp": False,
        }
        fake.set_reading("D:MAGNET.STATUS", status_data, value_type=ValueType.BASIC_STATUS)

        reading = fake.get("D:MAGNET.STATUS")

        assert reading.ok
        assert reading.value_type == ValueType.BASIC_STATUS
        assert reading.value == status_data
        assert isinstance(reading.value, dict)
        assert reading.value["on"] is True
        assert reading.value["ready"] is True

    def test_read_returns_raw_value(self):
        """read() returns raw value for all types."""
        fake = FakeBackend()

        # RAW bytes
        raw_data = b"\x00\x01\x02"
        fake.set_reading("D:RAW", raw_data, value_type=ValueType.RAW)
        assert fake.read("D:RAW") == raw_data

        # TEXT_ARRAY
        text_list = ["a", "b", "c"]
        fake.set_reading("D:TARR", text_list, value_type=ValueType.TEXT_ARRAY)
        assert fake.read("D:TARR") == text_list

        # ANALOG_ALARM dict
        alarm = {"minimum": 0, "maximum": 100}
        fake.set_reading("D:ANA", alarm, value_type=ValueType.ANALOG_ALARM)
        assert fake.read("D:ANA") == alarm

    def test_get_many_mixed_value_types(self):
        """get_many() handles mixed value types correctly."""
        fake = FakeBackend()

        # Configure different types
        fake.set_reading("D:SCALAR", 42.0, value_type=ValueType.SCALAR)
        fake.set_reading("D:ARR", np.array([1, 2, 3]), value_type=ValueType.SCALAR_ARRAY)
        fake.set_reading("D:RAW", b"\xff\x00", value_type=ValueType.RAW)
        fake.set_reading("D:TEXT", "hello", value_type=ValueType.TEXT)
        fake.set_reading("D:TARR", ["a", "b"], value_type=ValueType.TEXT_ARRAY)

        readings = fake.get_many(["D:SCALAR", "D:ARR", "D:RAW", "D:TEXT", "D:TARR"])

        assert len(readings) == 5
        assert readings[0].value_type == ValueType.SCALAR
        assert readings[0].value == 42.0
        assert readings[1].value_type == ValueType.SCALAR_ARRAY
        np.testing.assert_array_equal(readings[1].value, [1, 2, 3])
        assert readings[2].value_type == ValueType.RAW
        assert readings[2].value == b"\xff\x00"
        assert readings[3].value_type == ValueType.TEXT
        assert readings[3].value == "hello"
        assert readings[4].value_type == ValueType.TEXT_ARRAY
        assert readings[4].value == ["a", "b"]

    def test_emit_reading_with_all_value_types(self):
        """emit_reading works with all value types in streaming."""
        fake = FakeBackend()
        collected = []

        handle = fake.subscribe(
            ["D:RAW", "D:TARR", "D:ANA", "D:DIG", "D:STAT"],
            callback=lambda r, h: collected.append(r),
        )

        # Emit all types
        fake.emit_reading("D:RAW", b"\x01\x02\x03", value_type=ValueType.RAW)
        fake.emit_reading("D:TARR", ["x", "y"], value_type=ValueType.TEXT_ARRAY)
        fake.emit_reading("D:ANA", {"min": 0}, value_type=ValueType.ANALOG_ALARM)
        fake.emit_reading("D:DIG", {"mask": 0xFF}, value_type=ValueType.DIGITAL_ALARM)
        fake.emit_reading("D:STAT", {"on": True}, value_type=ValueType.BASIC_STATUS)

        handle.stop()

        assert len(collected) == 5
        assert collected[0].value_type == ValueType.RAW
        assert collected[0].value == b"\x01\x02\x03"
        assert collected[1].value_type == ValueType.TEXT_ARRAY
        assert collected[1].value == ["x", "y"]
        assert collected[2].value_type == ValueType.ANALOG_ALARM
        assert collected[2].value == {"min": 0}
        assert collected[3].value_type == ValueType.DIGITAL_ALARM
        assert collected[3].value == {"mask": 0xFF}
        assert collected[4].value_type == ValueType.BASIC_STATUS
        assert collected[4].value == {"on": True}


class TestSetError:
    """Tests for set_error() configuration."""

    def test_set_error_basic(self):
        """set_error configures an error for a device."""
        fake = FakeBackend()
        fake.set_error("M:BADDEV", -42, "Device not found")

        with pytest.raises(DeviceError) as exc_info:
            fake.read("M:BADDEV")

        assert exc_info.value.error_code == -42
        assert "Device not found" in str(exc_info.value)

    def test_set_error_in_get(self):
        """set_error returns error Reading via get()."""
        fake = FakeBackend()
        fake.set_error("M:BADDEV", -42, "Device not found")

        reading = fake.get("M:BADDEV")
        assert reading.is_error
        assert reading.error_code == -42
        assert reading.message == "Device not found"

    def test_set_error_overwrites_reading(self):
        """set_error removes any configured reading for same DRF."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.set_error("M:OUTTMP", -1, "Error")

        with pytest.raises(DeviceError):
            fake.read("M:OUTTMP")


class TestSetWriteResult:
    """Tests for set_write_result() configuration."""

    def test_set_write_result_success(self):
        """set_write_result configures successful write."""
        fake = FakeBackend()
        fake.set_write_result("M:OUTTMP", success=True)

        result = fake.write("M:OUTTMP", 72.5)
        assert result.success
        assert result.error_code == 0

    def test_set_write_result_failure(self):
        """set_write_result configures failed write."""
        fake = FakeBackend()
        fake.set_write_result("M:OUTTMP", success=False, message="Write failed")

        result = fake.write("M:OUTTMP", 72.5)
        assert not result.success
        assert result.error_code == -1
        assert result.message == "Write failed"

    def test_set_write_result_custom_status(self):
        """set_write_result allows custom status code."""
        fake = FakeBackend()
        fake.set_write_result("M:OUTTMP", error_code=-99)

        result = fake.write("M:OUTTMP", 72.5)
        assert result.error_code == -99

    def test_default_write_succeeds(self):
        """Write succeeds by default when no result configured."""
        fake = FakeBackend()

        result = fake.write("M:OUTTMP", 72.5)
        assert result.success


# ─────────────────────────────────────────────────────────────────────────────
# Inspection Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReadsProperty:
    """Tests for reads property."""

    def test_reads_records_read(self):
        """reads records device reads in order."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.set_reading("G:AMANDA", 1.0)

        fake.read("M:OUTTMP")
        fake.read("G:AMANDA")

        assert fake.reads == ["M:OUTTMP", "G:AMANDA"]

    def test_reads_records_get(self):
        """reads records get() calls."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)

        fake.get("M:OUTTMP")

        assert fake.reads == ["M:OUTTMP"]

    def test_reads_records_get_many(self):
        """reads records get_many() calls."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.set_reading("G:AMANDA", 1.0)

        fake.get_many(["M:OUTTMP", "G:AMANDA"])

        assert fake.reads == ["M:OUTTMP", "G:AMANDA"]

    def test_reads_records_failed_reads(self):
        """reads records even failed read attempts."""
        fake = FakeBackend()

        try:
            fake.read("M:OUTTMP")
        except DeviceError:
            pass

        assert fake.reads == ["M:OUTTMP"]

    def test_reads_returns_copy(self):
        """reads returns a copy, not the original list."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.read("M:OUTTMP")

        reads = fake.reads
        reads.append("FAKE")  # Modify the copy

        assert fake.reads == ["M:OUTTMP"]  # Original unchanged


class TestWritesProperty:
    """Tests for writes property."""

    def test_writes_records_write(self):
        """writes records device writes."""
        fake = FakeBackend()
        fake.write("M:OUTTMP", 72.5)

        assert fake.writes == [("M:OUTTMP", 72.5)]

    def test_writes_records_write_many(self):
        """writes records write_many() calls."""
        fake = FakeBackend()
        fake.write_many([("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)])

        assert fake.writes == [("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)]

    def test_writes_preserves_order(self):
        """writes preserves write order."""
        fake = FakeBackend()
        fake.write("A:DEV", 1.0)
        fake.write("B:DEV", 2.0)
        fake.write("A:DEV", 3.0)

        assert fake.writes == [("A:DEV", 1.0), ("B:DEV", 2.0), ("A:DEV", 3.0)]


class TestWasRead:
    """Tests for was_read() method."""

    def test_was_read_true(self):
        """was_read returns True for read devices."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.read("M:OUTTMP")

        assert fake.was_read("M:OUTTMP")

    def test_was_read_false(self):
        """was_read returns False for unread devices."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)

        assert not fake.was_read("M:OUTTMP")


class TestWasWritten:
    """Tests for was_written() method."""

    def test_was_written_true(self):
        """was_written returns True for written devices."""
        fake = FakeBackend()
        fake.write("M:OUTTMP", 72.5)

        assert fake.was_written("M:OUTTMP")

    def test_was_written_false(self):
        """was_written returns False for unwritten devices."""
        fake = FakeBackend()

        assert not fake.was_written("M:OUTTMP")


class TestGetWrittenValue:
    """Tests for get_written_value() method."""

    def test_get_written_value(self):
        """get_written_value returns last written value."""
        fake = FakeBackend()
        fake.write("M:OUTTMP", 72.5)

        assert fake.get_written_value("M:OUTTMP") == 72.5

    def test_get_written_value_multiple_writes(self):
        """get_written_value returns last value when written multiple times."""
        fake = FakeBackend()
        fake.write("M:OUTTMP", 70.0)
        fake.write("M:OUTTMP", 72.5)
        fake.write("M:OUTTMP", 75.0)

        assert fake.get_written_value("M:OUTTMP") == 75.0

    def test_get_written_value_none(self):
        """get_written_value returns None for unwritten devices."""
        fake = FakeBackend()

        assert fake.get_written_value("M:OUTTMP") is None


# ─────────────────────────────────────────────────────────────────────────────
# Reset Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReset:
    """Tests for reset() method."""

    def test_reset_clears_readings(self):
        """reset clears configured readings."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.reset()

        with pytest.raises(DeviceError):
            fake.read("M:OUTTMP")

    def test_reset_clears_errors(self):
        """reset clears configured errors."""
        fake = FakeBackend()
        fake.set_error("M:OUTTMP", -42, "Error")
        fake.reset()

        # Now returns "no reading configured" error, not the specific error
        reading = fake.get("M:OUTTMP")
        assert reading.is_error
        assert "No reading configured" in reading.message

    def test_reset_clears_read_history(self):
        """reset clears read history."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.read("M:OUTTMP")
        fake.reset()

        assert fake.reads == []

    def test_reset_clears_write_history(self):
        """reset clears write history."""
        fake = FakeBackend()
        fake.write("M:OUTTMP", 72.5)
        fake.reset()

        assert fake.writes == []

    def test_reset_clears_write_results(self):
        """reset clears configured write results."""
        fake = FakeBackend()
        fake.set_write_result("M:OUTTMP", success=False)
        fake.reset()

        # Default success behavior restored
        result = fake.write("M:OUTTMP", 72.5)
        assert result.success


# ─────────────────────────────────────────────────────────────────────────────
# Backend Interface Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReadMethod:
    """Tests for read() method."""

    def test_read_unconfigured_raises(self):
        """read() raises DeviceError for unconfigured device."""
        fake = FakeBackend()

        with pytest.raises(DeviceError) as exc_info:
            fake.read("M:OUTTMP")

        assert exc_info.value.drf == "M:OUTTMP"
        assert exc_info.value.error_code == ERR_RETRY
        assert "No reading configured" in exc_info.value.message

    def test_read_known_device_wrong_property_raises_noprop(self):
        """read() raises DBM_NOPROP when device is known but property isn't configured."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)  # configures READING property

        with pytest.raises(DeviceError) as exc_info:
            fake.read("M:OUTTMP.SETTING")

        assert exc_info.value.facility_code == FACILITY_DBM
        assert exc_info.value.error_code == ERR_NOPROP
        assert "No such property" in exc_info.value.message

    def test_read_configured_value(self):
        """read() returns configured value."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)

        assert fake.read("M:OUTTMP") == 72.5

    def test_read_accepts_timeout(self):
        """read() accepts timeout parameter (ignored)."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)

        # Should not raise
        assert fake.read("M:OUTTMP", timeout=5.0) == 72.5


class TestGetMethod:
    """Tests for get() method."""

    def test_get_unconfigured_returns_error_reading(self):
        """get() returns error Reading for unconfigured device."""
        fake = FakeBackend()

        reading = fake.get("M:OUTTMP")

        assert reading.is_error
        assert reading.error_code == ERR_RETRY
        assert "No reading configured" in reading.message

    def test_get_known_device_wrong_property_returns_noprop(self):
        """get() returns DBM_NOPROP when device is known but property isn't configured."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)  # configures READING property

        reading = fake.get("M:OUTTMP.SETTING")

        assert reading.is_error
        assert reading.facility_code == FACILITY_DBM
        assert reading.error_code == ERR_NOPROP
        assert "No such property" in reading.message

    def test_get_configured_reading(self):
        """get() returns configured Reading."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5, units="degF")

        reading = fake.get("M:OUTTMP")

        assert reading.ok
        assert reading.value == 72.5
        assert reading.meta.units == "degF"

    def test_get_configured_error(self):
        """get() returns error Reading for configured error."""
        fake = FakeBackend()
        fake.set_error("M:BADDEV", -42, "Device not found")

        reading = fake.get("M:BADDEV")

        assert reading.is_error
        assert reading.error_code == -42
        assert reading.message == "Device not found"


class TestGetManyMethod:
    """Tests for get_many() method."""

    def test_get_many_all_configured(self):
        """get_many() returns all configured readings."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.set_reading("G:AMANDA", 1.0)

        readings = fake.get_many(["M:OUTTMP", "G:AMANDA"])

        assert len(readings) == 2
        assert readings[0].value == 72.5
        assert readings[1].value == 1.0

    def test_get_many_mixed_success_error(self):
        """get_many() handles mix of success and error."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.set_error("M:BADDEV", -42, "Error")

        readings = fake.get_many(["M:OUTTMP", "M:BADDEV"])

        assert readings[0].ok
        assert readings[1].is_error

    def test_get_many_preserves_order(self):
        """get_many() returns readings in input order."""
        fake = FakeBackend()
        fake.set_reading("A:DEV", 1.0)
        fake.set_reading("B:DEV", 2.0)
        fake.set_reading("C:DEV", 3.0)

        readings = fake.get_many(["C:DEV", "A:DEV", "B:DEV"])

        assert readings[0].drf == "C:DEV"
        assert readings[1].drf == "A:DEV"
        assert readings[2].drf == "B:DEV"


class TestWriteMethod:
    """Tests for write() method."""

    def test_write_records_value(self):
        """write() records the written value."""
        fake = FakeBackend()
        fake.write("M:OUTTMP", 72.5)

        assert fake.writes == [("M:OUTTMP", 72.5)]

    def test_write_returns_configured_result(self):
        """write() returns configured WriteResult."""
        fake = FakeBackend()
        fake.set_write_result("M:OUTTMP", success=False, message="Failed")

        result = fake.write("M:OUTTMP", 72.5)

        assert not result.success
        assert result.message == "Failed"

    def test_write_accepts_all_parameters(self):
        """write() accepts optional timeout parameter."""
        fake = FakeBackend()

        # Should not raise
        result = fake.write("M:OUTTMP", 72.5, timeout=5.0)
        assert result.success


class TestWriteManyMethod:
    """Tests for write_many() method."""

    def test_write_many_records_all(self):
        """write_many() records all writes."""
        fake = FakeBackend()
        fake.write_many([("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)])

        assert fake.writes == [("M:OUTTMP", 72.5), ("G:AMANDA", 1.0)]

    def test_write_many_returns_results(self):
        """write_many() returns list of WriteResults."""
        fake = FakeBackend()
        fake.set_write_result("M:OUTTMP", success=True)
        fake.set_write_result("M:BADDEV", success=False)

        results = fake.write_many([("M:OUTTMP", 72.5), ("M:BADDEV", 1.0)])

        assert results[0].success
        assert not results[1].success


class TestCloseMethod:
    """Tests for close() method."""

    def test_close(self):
        """close() marks backend as closed."""
        fake = FakeBackend()
        fake.close()
        assert fake._closed is True

    def test_close_multiple_times(self):
        """close() can be called multiple times."""
        fake = FakeBackend()
        fake.close()
        fake.close()  # Should not raise


# ─────────────────────────────────────────────────────────────────────────────
# Context Manager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager(self):
        """FakeBackend works as context manager."""
        with FakeBackend() as fake:
            fake.set_reading("M:OUTTMP", 72.5)
            assert fake.read("M:OUTTMP") == 72.5

    def test_context_manager_closes(self):
        """Context manager closes backend on exit."""
        fake = FakeBackend()
        with fake:
            pass
        assert fake._closed is True


# ─────────────────────────────────────────────────────────────────────────────
# DRF Normalization Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDRFNormalization:
    """Tests for FakeBackend DRF normalization.

    Keys are device identity (name + property + field).  Events and ranges
    are stripped so all access patterns hit the same device state.
    """

    def test_short_form_matches_full_read_drf(self):
        """set_reading('M:OUTTMP') matches read('M:OUTTMP.READING@I')."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        assert fake.read("M:OUTTMP.READING@I") == 72.5

    def test_full_form_matches_short_read(self):
        """set_reading('M:OUTTMP.READING@I') matches read('M:OUTTMP')."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP.READING@I", 72.5)
        assert fake.read("M:OUTTMP") == 72.5

    def test_setting_normalization(self):
        """set_write_result('M_OUTTMP') matches write('M:OUTTMP.SETTING@N')."""
        fake = FakeBackend()
        fake.set_write_result("M_OUTTMP", success=False, message="Blocked")
        result = fake.write("M:OUTTMP.SETTING@N", 72.5)
        assert not result.success

    def test_all_events_same_device(self):
        """@I, @N, @p,1000 all reference the same device state."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        assert fake.read("M:OUTTMP.READING@I") == 72.5
        assert fake.read("M:OUTTMP.READING@p,1000") == 72.5
        assert fake.read("M:OUTTMP.READING@N") == 72.5

    def test_error_normalization(self):
        """set_error short form matches Device.read() full form."""
        fake = FakeBackend()
        fake.set_error("M:BADDEV", -42, "Not found")
        with pytest.raises(DeviceError) as exc_info:
            fake.read("M:BADDEV.READING@I")
        assert exc_info.value.error_code == -42

    def test_was_read_normalizes(self):
        """was_read normalizes both stored and queried DRFs."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.read("M:OUTTMP.READING@I")
        assert fake.was_read("M:OUTTMP")

    def test_was_written_normalizes(self):
        """was_written normalizes both stored and queried DRFs."""
        fake = FakeBackend()
        fake.write("M:OUTTMP.SETTING@N", 72.5)
        assert fake.was_written("M_OUTTMP")

    def test_get_written_value_normalizes(self):
        """get_written_value normalizes DRF for lookup."""
        fake = FakeBackend()
        fake.write("M:OUTTMP.SETTING@N", 72.5)
        assert fake.get_written_value("M_OUTTMP") == 72.5

    def test_set_reading_clears_normalized_error(self):
        """set_reading removes error even when DRF forms differ."""
        fake = FakeBackend()
        fake.set_error("M:OUTTMP.READING@I", -1, "Error")
        fake.set_reading("M:OUTTMP", 72.5)
        assert fake.read("M:OUTTMP") == 72.5

    def test_range_stripped_from_key(self):
        """Ranges are stripped from key — stored value is the full device array."""
        fake = FakeBackend()
        fake.set_reading("B:HS23T", np.array([10, 20, 30, 40, 50]))
        # Read with same range or different range
        np.testing.assert_array_equal(fake.read("B:HS23T[0:2]"), [10, 20, 30])
        np.testing.assert_array_equal(fake.read("B:HS23T[1:3]"), [20, 30, 40])

    def test_range_set_and_read_back(self):
        """set_reading with range stores value; read with same range returns it."""
        fake = FakeBackend()
        arr = np.array([1.0, 2.0, 3.0])
        fake.set_reading("B:HS23T[0:2]", arr)
        # Range is stripped from key, so reading without range returns full value
        np.testing.assert_array_equal(fake.read("B:HS23T"), arr)


class TestWriteUpdatesState:
    """Tests for write() updating device readable state."""

    def test_write_then_read(self):
        """Successful write updates readable state for same property."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP.SETTING", 50.0)
        fake.write("M:OUTTMP.SETTING@N", 72.5)
        assert fake.read("M:OUTTMP.SETTING@I") == 72.5

    def test_write_creates_state(self):
        """Write to unconfigured device creates readable state."""
        fake = FakeBackend()
        fake.write("M:OUTTMP.SETTING@N", 72.5)
        assert fake.read("M:OUTTMP.SETTING") == 72.5

    def test_failed_write_does_not_update_state(self):
        """Failed write (configured result) does not change state."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP.SETTING", 50.0)
        fake.set_write_result("M:OUTTMP.SETTING", success=False)
        fake.write("M:OUTTMP.SETTING@N", 72.5)
        assert fake.read("M:OUTTMP.SETTING") == 50.0

    def test_write_clears_error(self):
        """Successful write clears a configured error for that device+property."""
        fake = FakeBackend()
        fake.set_error("M:OUTTMP.SETTING", -1, "Offline")
        fake.write("M:OUTTMP.SETTING@N", 72.5)
        assert fake.read("M:OUTTMP.SETTING") == 72.5

    def test_write_preserves_metadata(self):
        """Write preserves existing metadata (units, description)."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP.SETTING", 50.0, units="degF", description="Temp")
        fake.write("M:OUTTMP.SETTING@N", 72.5)
        reading = fake.get("M:OUTTMP.SETTING")
        assert reading.value == 72.5
        assert reading.meta.units == "degF"
        assert reading.meta.description == "Temp"

    def test_ranged_write_slice_assigns(self):
        """Write to a ranged DRF updates only that slice of the stored array."""
        fake = FakeBackend()
        fake.set_reading("B:HS23T", np.array([10, 20, 30, 40, 50]))
        fake.write("B:HS23T.READING[1:2]@N", np.array([77, 88]))
        np.testing.assert_array_equal(fake.read("B:HS23T"), [10, 77, 88, 40, 50])

    def test_ranged_write_single_element(self):
        """Write to a single-element range updates only that index."""
        fake = FakeBackend()
        fake.set_reading("B:HS23T", np.array([10, 20, 30]))
        fake.write("B:HS23T.READING[0]@N", 99)
        np.testing.assert_array_equal(fake.read("B:HS23T"), [99, 20, 30])


# ─────────────────────────────────────────────────────────────────────────────
# Device Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeviceIntegration:
    """Tests for FakeBackend integration with Device objects.

    DRF normalization means users can configure FakeBackend with short-form
    DRFs (e.g., "M:OUTTMP") even though Device.read() internally builds
    fully-qualified forms (e.g., "M:OUTTMP.READING@I").
    """

    def test_device_with_fake_backend(self):
        """Device can use FakeBackend with short-form DRF."""
        fake = FakeBackend()
        dev = Device("M:OUTTMP", backend=fake)
        fake.set_reading("M:OUTTMP", 72.5)

        assert dev.read() == 72.5

    def test_scalar_device_with_fake_backend(self):
        """ScalarDevice can use FakeBackend."""
        fake = FakeBackend()
        dev = ScalarDevice("M:OUTTMP", backend=fake)
        fake.set_reading("M:OUTTMP", 72.5)

        value = dev.read()

        assert isinstance(value, float)
        assert value == 72.5

    def test_array_device_with_fake_backend(self):
        """ArrayDevice can use FakeBackend."""
        fake = FakeBackend()
        dev = ArrayDevice("B:HS23T[0:3]", backend=fake)
        fake.set_reading("B:HS23T[0:3]", np.array([1.0, 2.0, 3.0]))

        values = dev.read()

        assert isinstance(values, np.ndarray)
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])

    def test_device_error_handling(self):
        """Device properly raises DeviceError from FakeBackend."""
        fake = FakeBackend()
        dev = Device("M:BADDEV", backend=fake)
        fake.set_error("M:BADDEV", -42, "Device not found")

        with pytest.raises(DeviceError) as exc_info:
            dev.read()

        assert exc_info.value.error_code == -42

    def test_device_with_backend_method(self):
        """Device.with_backend() works with FakeBackend."""
        fake = FakeBackend()
        dev = Device("M:OUTTMP")
        fake.set_reading("M:OUTTMP", 72.5)

        dev_with_fake = dev.with_backend(fake)

        assert dev_with_fake.read() == 72.5

    def test_device_read_records_history(self):
        """Device reads are recorded in FakeBackend history."""
        fake = FakeBackend()
        dev = Device("M:OUTTMP", backend=fake)
        fake.set_reading("M:OUTTMP", 72.5)

        dev.read()

        # was_read normalizes, so short form works even though
        # Device sent "M:OUTTMP.READING@I" to the backend
        assert fake.was_read("M:OUTTMP")


# ─────────────────────────────────────────────────────────────────────────────
# Usage Pattern Examples (also serve as documentation tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestUsagePatterns:
    """Tests demonstrating common usage patterns."""

    def test_basic_usage(self):
        """Basic usage: configure and read."""
        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        fake.set_error("M:BADDEV", -42, "Device not found")

        # Test success case
        assert fake.read("M:OUTTMP") == 72.5

        # Test error case
        reading = fake.get("M:BADDEV")
        assert reading.is_error
        assert reading.error_code == -42

    def test_write_tracking(self):
        """Write tracking pattern."""
        fake = FakeBackend()
        fake.write("M:OUTTMP", 72.5)

        assert fake.writes == [("M:OUTTMP", 72.5)]

    def test_dependency_injection(self):
        """Dependency injection pattern."""

        class TemperatureMonitor:
            def __init__(self, backend):
                self.backend = backend

            def get_temp(self):
                return self.backend.read("M:OUTTMP")

        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)
        monitor = TemperatureMonitor(fake)

        assert monitor.get_temp() == 72.5

    def test_verify_interactions(self):
        """Verify interactions pattern."""

        class DeviceController:
            def __init__(self, backend):
                self.backend = backend

            def set_and_verify(self, drf, value):
                self.backend.write(drf, value)
                return self.backend.read(drf)

        fake = FakeBackend()
        fake.set_reading("M:OUTTMP", 72.5)

        controller = DeviceController(fake)
        controller.set_and_verify("M:OUTTMP", 72.5)

        # Verify the interactions
        assert fake.was_written("M:OUTTMP")
        assert fake.get_written_value("M:OUTTMP") == 72.5
        assert fake.was_read("M:OUTTMP")

    def test_reset_between_tests(self):
        """Reset between tests pattern."""
        fake = FakeBackend()

        # First test scenario
        fake.set_reading("M:OUTTMP", 72.5)
        fake.read("M:OUTTMP")
        assert fake.reads == ["M:OUTTMP"]

        # Reset for second test scenario
        fake.reset()
        assert fake.reads == []

        # Second test scenario
        fake.set_error("M:OUTTMP", -1, "Different error")
        reading = fake.get("M:OUTTMP")
        assert reading.is_error


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSubscribe:
    """Tests for subscribe() method."""

    def test_subscribe_returns_handle(self):
        """subscribe() returns a FakeSubscriptionHandle."""
        from pacsys.testing import FakeSubscriptionHandle

        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP@p,1000"])

        assert isinstance(handle, FakeSubscriptionHandle)
        handle.stop()

    def test_subscribe_handle_properties(self):
        """Subscription handle has correct initial properties."""
        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP", "G:AMANDA"])

        assert handle.ref_ids == [0, 1]
        assert handle.stopped is False
        assert handle.exc is None
        handle.stop()

    def test_subscribe_context_manager(self):
        """Subscription handle works as context manager."""
        fake = FakeBackend()

        with fake.subscribe(["M:OUTTMP"]) as handle:
            assert not handle.stopped

        assert handle.stopped


class TestEmitReading:
    """Tests for emit_reading() method."""

    def test_emit_reading_iterator_mode(self):
        """emit_reading delivers to iterator mode subscription."""
        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP@p,1000"])

        fake.emit_reading("M:OUTTMP@p,1000", 72.5)

        readings = list(handle.readings(timeout=0.1))
        assert len(readings) == 1
        reading, h = readings[0]
        assert reading.value == 72.5
        assert h is handle
        handle.stop()

    def test_emit_reading_callback_mode(self):
        """emit_reading delivers to callback mode subscription."""
        collected = []

        def callback(reading, handle):
            collected.append(reading)

        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"], callback=callback)

        fake.emit_reading("M:OUTTMP", 72.5)
        fake.emit_reading("M:OUTTMP", 73.0)

        assert len(collected) == 2
        assert collected[0].value == 72.5
        assert collected[1].value == 73.0
        handle.stop()

    def test_emit_reading_with_metadata(self):
        """emit_reading includes metadata in reading."""
        fake = FakeBackend()
        ts = datetime(2025, 1, 15, 12, 0, 0)

        with fake.subscribe(["M:OUTTMP"]) as handle:
            fake.emit_reading(
                "M:OUTTMP",
                72.5,
                value_type=ValueType.SCALAR,
                units="degF",
                description="Test device",
                timestamp=ts,
                cycle=42,
            )

            for reading, _ in handle.readings(timeout=0.1):
                assert reading.value == 72.5
                assert reading.value_type == ValueType.SCALAR
                assert reading.meta.units == "degF"
                assert reading.meta.description == "Test device"
                assert reading.timestamp == ts
                assert reading.cycle == 42
                break

    def test_emit_reading_only_matching_subscriptions(self):
        """emit_reading only delivers to subscriptions with matching DRF."""
        collected1 = []
        collected2 = []

        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"], callback=lambda r, h: collected1.append(r))
        h2 = fake.subscribe(["G:AMANDA"], callback=lambda r, h: collected2.append(r))

        fake.emit_reading("M:OUTTMP", 72.5)

        assert len(collected1) == 1
        assert len(collected2) == 0
        h1.stop()
        h2.stop()

    def test_emit_reading_multiple_subscriptions_same_drf(self):
        """emit_reading delivers to all subscriptions for same DRF."""
        collected1 = []
        collected2 = []

        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"], callback=lambda r, h: collected1.append(r))
        h2 = fake.subscribe(["M:OUTTMP"], callback=lambda r, h: collected2.append(r))

        fake.emit_reading("M:OUTTMP", 72.5)

        assert len(collected1) == 1
        assert len(collected2) == 1
        h1.stop()
        h2.stop()

    def test_emit_reading_stopped_subscription_ignored(self):
        """emit_reading does not deliver to stopped subscriptions."""
        collected = []

        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"], callback=lambda r, h: collected.append(r))

        fake.emit_reading("M:OUTTMP", 72.5)
        handle.stop()
        fake.emit_reading("M:OUTTMP", 73.0)

        assert len(collected) == 1


class TestEmitError:
    """Tests for emit_error() method."""

    def test_emit_error_sets_exc(self):
        """emit_error sets the exc property on subscriptions."""
        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"])

        fake.emit_error(ConnectionError("Simulated disconnect"))

        assert handle.exc is not None
        assert isinstance(handle.exc, ConnectionError)
        handle.stop()

    def test_emit_error_with_on_error_callback(self):
        """emit_error calls on_error callback."""
        errors = []

        def on_error(exc, handle):
            errors.append((exc, handle))

        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"], on_error=on_error)

        err = ConnectionError("Simulated disconnect")
        fake.emit_error(err)

        assert len(errors) == 1
        assert errors[0][0] is err
        assert errors[0][1] is handle
        handle.stop()

    def test_emit_error_raises_in_iterator(self):
        """emit_error causes exception to be raised in readings() iterator."""
        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"])

        fake.emit_error(ConnectionError("Simulated disconnect"))

        with pytest.raises(ConnectionError):
            for _ in handle.readings(timeout=0.1):
                pass


class TestSubscriptionStop:
    """Tests for subscription stop() method."""

    def test_stop_sets_stopped(self):
        """stop() sets stopped property to True."""
        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"])

        assert not handle.stopped
        handle.stop()
        assert handle.stopped

    def test_stop_removes_from_backend(self):
        """stop() removes subscription from backend tracking."""
        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"])

        assert len(fake._subscriptions) == 1
        handle.stop()
        assert len(fake._subscriptions) == 0

    def test_stop_idempotent(self):
        """stop() can be called multiple times safely."""
        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"])

        handle.stop()
        handle.stop()  # Should not raise

        assert handle.stopped


class TestStopStreaming:
    """Tests for stop_streaming() method."""

    def test_stop_streaming_stops_all(self):
        """stop_streaming() stops all active subscriptions."""
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])

        fake.stop_streaming()

        assert h1.stopped
        assert h2.stopped
        assert len(fake._subscriptions) == 0

    def test_stop_streaming_empty(self):
        """stop_streaming() works with no subscriptions."""
        fake = FakeBackend()
        fake.stop_streaming()  # Should not raise


class TestRemoveMethod:
    """Tests for remove() method."""

    def test_remove_stops_subscription(self):
        """remove() stops the given subscription."""
        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"])

        fake.remove(handle)

        assert handle.stopped


class TestCloseStopsSubscriptions:
    """Tests for close() stopping subscriptions."""

    def test_close_stops_all_subscriptions(self):
        """close() stops all active subscriptions."""
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])

        fake.close()

        assert h1.stopped
        assert h2.stopped


class TestResetClearsSubscriptions:
    """Tests for reset() clearing subscriptions."""

    def test_reset_stops_subscriptions(self):
        """reset() stops all subscriptions."""
        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"])

        fake.reset()

        assert handle.stopped
        assert len(fake._subscriptions) == 0


class TestStreamingUsagePatterns:
    """Tests demonstrating streaming usage patterns."""

    def test_iterator_pattern(self):
        """Iterator pattern for consuming streaming data."""
        fake = FakeBackend()

        with fake.subscribe(["M:OUTTMP@p,1000"]) as sub:
            # Emit some readings
            for i in range(3):
                fake.emit_reading("M:OUTTMP@p,1000", 70.0 + i)

            # Consume them
            values = []
            for reading, handle in sub.readings(timeout=0.1):
                values.append(reading.value)

            assert values == [70.0, 71.0, 72.0]

    def test_callback_pattern(self):
        """Callback pattern for processing streaming data."""
        readings = []

        def process_reading(reading, handle):
            readings.append(reading.value)
            if len(readings) >= 3:
                handle.stop()

        fake = FakeBackend()
        handle = fake.subscribe(["M:OUTTMP"], callback=process_reading)

        for i in range(5):
            fake.emit_reading("M:OUTTMP", 70.0 + i)

        # Should have stopped after 3 readings
        assert len(readings) == 3
        assert handle.stopped

    def test_error_handling_pattern(self):
        """Error handling pattern with on_error callback."""
        errors = []
        readings = []

        fake = FakeBackend()
        handle = fake.subscribe(
            ["M:OUTTMP"],
            callback=lambda r, h: readings.append(r),
            on_error=lambda e, h: errors.append(e),
        )

        # Some successful readings
        fake.emit_reading("M:OUTTMP", 72.5)
        fake.emit_reading("M:OUTTMP", 73.0)

        # Then an error
        fake.emit_error(ConnectionError("Lost connection"))

        assert len(readings) == 2
        assert len(errors) == 1
        assert isinstance(errors[0], ConnectionError)
        handle.stop()

    def test_multiple_devices_pattern(self):
        """Multiple devices in single subscription."""
        collected = []

        fake = FakeBackend()
        handle = fake.subscribe(
            ["M:OUTTMP", "G:AMANDA"],
            callback=lambda r, h: collected.append((r.drf, r.value)),
        )

        fake.emit_reading("M:OUTTMP", 72.5)
        fake.emit_reading("G:AMANDA", 1.234)
        fake.emit_reading("M:OUTTMP", 73.0)

        assert len(collected) == 3
        assert collected[0] == ("M:OUTTMP", 72.5)
        assert collected[1] == ("G:AMANDA", 1.234)
        assert collected[2] == ("M:OUTTMP", 73.0)
        handle.stop()
