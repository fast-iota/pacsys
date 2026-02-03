"""
Tests for pacsys.types module.
"""

import pytest
from datetime import datetime
import numpy as np

from pacsys.types import (
    BackendCapability,
    ValueType,
    DeviceMeta,
    Reading,
    WriteResult,
)
from pacsys.errors import DeviceError, AuthenticationError


class TestValueType:
    """Tests for ValueType enum."""

    def test_all_values_exist(self):
        assert ValueType.SCALAR.value == "scalar"
        assert ValueType.SCALAR_ARRAY.value == "scalarArr"
        assert ValueType.RAW.value == "raw"
        assert ValueType.TEXT.value == "text"
        assert ValueType.TEXT_ARRAY.value == "textArr"
        assert ValueType.ANALOG_ALARM.value == "anaAlarm"
        assert ValueType.DIGITAL_ALARM.value == "digAlarm"
        assert ValueType.BASIC_STATUS.value == "basicStatus"

    def test_value_type_count(self):
        assert len(ValueType) == 8

    def test_from_string(self):
        assert ValueType("scalar") == ValueType.SCALAR
        assert ValueType("scalarArr") == ValueType.SCALAR_ARRAY


class TestBackendCapability:
    """Tests for BackendCapability flag."""

    def test_flag_combinations(self):
        caps = BackendCapability.READ | BackendCapability.WRITE
        assert BackendCapability.READ in caps
        assert BackendCapability.WRITE in caps
        assert BackendCapability.STREAM not in caps


class TestDeviceMeta:
    """Tests for DeviceMeta dataclass."""

    def test_creation_with_all_fields(self):
        meta = DeviceMeta(
            device_index=12345,
            name="M:OUTTMP",
            description="Outdoor temperature",
            units="degF",
            format_hint=2,
        )
        assert meta.device_index == 12345
        assert meta.name == "M:OUTTMP"
        assert meta.description == "Outdoor temperature"
        assert meta.units == "degF"
        assert meta.format_hint == 2

    def test_creation_with_defaults(self):
        meta = DeviceMeta(device_index=1, name="D:TEST", description="Test device")
        assert meta.device_index == 1
        assert meta.name == "D:TEST"
        assert meta.description == "Test device"
        assert meta.units is None
        assert meta.format_hint is None

    def test_immutability(self):
        meta = DeviceMeta(device_index=1, name="D:TEST", description="Test")
        with pytest.raises(AttributeError):
            meta.name = "CHANGED"  # type: ignore
        with pytest.raises(AttributeError):
            meta.units = "meters"  # type: ignore


class TestReading:
    """Tests for Reading dataclass."""

    def test_creation_minimal(self):
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR)
        assert reading.drf == "M:OUTTMP"
        assert reading.value_type == ValueType.SCALAR
        assert reading.tag is None
        assert reading.facility_code == 0
        assert reading.error_code == 0
        assert reading.value is None
        assert reading.message is None
        assert reading.timestamp is None
        assert reading.cycle == 0
        assert reading.meta is None

    def test_creation_with_all_fields(self):
        ts = datetime.now()
        meta = DeviceMeta(device_index=1, name="M:OUTTMP", description="Temp")
        reading = Reading(
            drf="M:OUTTMP@p,1000",
            value_type=ValueType.SCALAR,
            tag=42,
            facility_code=17,
            error_code=0,
            value=72.5,
            message=None,
            timestamp=ts,
            cycle=100,
            meta=meta,
        )
        assert reading.drf == "M:OUTTMP@p,1000"
        assert reading.value_type == ValueType.SCALAR
        assert reading.tag == 42
        assert reading.facility_code == 17
        assert reading.error_code == 0
        assert reading.value == 72.5
        assert reading.timestamp == ts
        assert reading.cycle == 100
        assert reading.meta == meta

    def test_immutability(self):
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5)
        with pytest.raises(AttributeError):
            reading.value = 80.0  # type: ignore
        with pytest.raises(AttributeError):
            reading.error_code = -1  # type: ignore

    @pytest.mark.parametrize(
        "error_code,is_success,is_warning,is_error",
        [
            (0, True, False, False),  # success
            (1, False, True, False),  # warning
            (100, False, True, False),  # warning (high)
            (-1, False, False, True),  # error
            (-42, False, False, True),  # error (other)
        ],
    )
    def test_status_flags(self, error_code, is_success, is_warning, is_error):
        """Reading status flags reflect error_code correctly."""
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, error_code=error_code)
        assert reading.is_success is is_success
        assert reading.is_warning is is_warning
        assert reading.is_error is is_error

    @pytest.mark.parametrize(
        "error_code,value,expected_ok",
        [
            (0, 72.5, True),  # success + value = ok
            (0, None, False),  # success + no value = not ok
            (1, 72.5, True),  # warning + value = ok
            (1, None, False),  # warning + no value = not ok
            (-1, 72.5, False),  # error + value = not ok
            (-1, None, False),  # error + no value = not ok
        ],
    )
    def test_ok_property(self, error_code, value, expected_ok):
        """Reading.ok requires non-negative error_code AND value is not None."""
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, error_code=error_code, value=value)
        assert reading.ok is expected_ok

    def test_name_from_meta(self):
        meta = DeviceMeta(device_index=1, name="M:OUTTMP", description="Temp")
        reading = Reading(drf="M:OUTTMP@p,1000", value_type=ValueType.SCALAR, meta=meta)
        assert reading.name == "M:OUTTMP"

    @pytest.mark.parametrize(
        "drf,expected_name",
        [
            ("M:OUTTMP", "M:OUTTMP"),  # simple
            ("M:OUTTMP@p,1000", "M:OUTTMP"),  # with event
            ("B:HS23T[0:10]", "B:HS23T"),  # with range
            ("B:HS23T[0:10]@p,1000", "B:HS23T"),  # with range and event
        ],
    )
    def test_name_from_drf(self, drf, expected_name):
        """Reading.name extracts device name from DRF."""
        reading = Reading(drf=drf, value_type=ValueType.SCALAR)
        assert reading.name == expected_name

    def test_units_from_meta(self):
        meta = DeviceMeta(device_index=1, name="M:OUTTMP", description="Temp", units="degF")
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, meta=meta)
        assert reading.units == "degF"

    def test_units_without_meta(self):
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR)
        assert reading.units is None

    @pytest.mark.parametrize(
        "vtype,value",
        [
            (ValueType.SCALAR, 72.5),
            (ValueType.SCALAR_ARRAY, np.array([1.0, 2.0, 3.0])),
            (ValueType.TEXT, "Outdoor temperature"),
            (ValueType.RAW, b"\x01\x02"),
        ],
    )
    def test_different_value_types(self, vtype, value):
        reading = Reading(drf="M:OUTTMP", value_type=vtype, value=value)
        if vtype == ValueType.SCALAR_ARRAY:
            assert np.array_equal(reading.value, value)
        else:
            assert reading.value == value
        assert reading.value_type == vtype


class TestWriteResult:
    """Tests for WriteResult dataclass."""

    def test_creation_minimal(self):
        result = WriteResult(drf="M:OUTTMP")
        assert result.drf == "M:OUTTMP"
        assert result.facility_code == 0
        assert result.error_code == 0
        assert result.message is None

    def test_creation_with_all_fields(self):
        result = WriteResult(
            drf="M:OUTTMP",
            facility_code=17,
            error_code=-1,
            message="Write failed",
        )
        assert result.drf == "M:OUTTMP"
        assert result.facility_code == 17
        assert result.error_code == -1
        assert result.message == "Write failed"

    def test_immutability(self):
        result = WriteResult(drf="M:OUTTMP")
        with pytest.raises(AttributeError):
            result.error_code = -1  # type: ignore

    @pytest.mark.parametrize(
        "error_code,expected_success",
        [
            (0, True),  # success
            (-1, False),  # error
            (1, False),  # warning (only error_code == 0 is success)
        ],
    )
    def test_success_property(self, error_code, expected_success):
        """WriteResult.success is True only when error_code == 0."""
        result = WriteResult(drf="M:OUTTMP", error_code=error_code)
        assert result.success is expected_success


class TestDeviceError:
    """Tests for DeviceError exception."""

    def test_creation_with_message(self):
        err = DeviceError(
            drf="M:OUTTMP",
            facility_code=17,
            error_code=-26,
            message="Device not found",
        )
        assert err.drf == "M:OUTTMP"
        assert err.facility_code == 17
        assert err.error_code == -26
        assert err.message == "Device not found"
        assert "M:OUTTMP" in str(err)
        assert "Device not found" in str(err)

    def test_creation_without_message(self):
        err = DeviceError(drf="M:OUTTMP", facility_code=17, error_code=-26)
        assert err.drf == "M:OUTTMP"
        assert err.facility_code == 17
        assert err.error_code == -26
        assert err.message is None
        assert "M:OUTTMP" in str(err)
        assert "error" in str(err)

    def test_can_raise_and_catch(self):
        with pytest.raises(DeviceError) as exc_info:
            raise DeviceError(drf="M:OUTTMP", facility_code=17, error_code=-26)
        assert exc_info.value.drf == "M:OUTTMP"


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_creation(self):
        err = AuthenticationError("Kerberos ticket expired")
        assert err.message == "Kerberos ticket expired"
        assert str(err) == "Kerberos ticket expired"

    def test_can_raise_and_catch(self):
        with pytest.raises(AuthenticationError) as exc_info:
            raise AuthenticationError("Authentication required for write")
        assert "Authentication required" in exc_info.value.message
