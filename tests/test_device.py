"""
Tests for Device API.

Tests cover:
- Device creation with valid/invalid DRF strings
- Property accessors (drf, name, has_event, is_periodic)
- Immutability (with_event, with_range, with_backend return new Device)
- Typed device subclasses (ScalarDevice, ArrayDevice, TextDevice)
- Read/get delegation to backend
- Property-specific read methods (setting, status, analog_alarm, etc.)
- Write methods (write, control, on/off/reset shortcuts)
- Alarm setters (set_analog_alarm, set_digital_alarm)
- Verify flow (write+verify, check_first, control+verify)
- Field validation
"""

import pytest
from unittest import mock
import numpy as np

from pacsys.device import Device, ScalarDevice, ArrayDevice, TextDevice
from pacsys.errors import DeviceError
from pacsys.types import Reading, ValueType, WriteResult, BasicControl
from pacsys.verify import Verify


# Fixtures


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = mock.MagicMock()
    backend.read.return_value = 72.5
    backend.get.return_value = Reading(
        drf="M:OUTTMP",
        value_type=ValueType.SCALAR,
        value=72.5,
        error_code=0,
    )
    backend.write.return_value = WriteResult(drf="M:OUTTMP.SETTING@N", error_code=0)
    return backend


# Device Creation Tests


class TestDeviceCreation:
    """Tests for Device creation and DRF validation."""

    def test_create_simple_device(self):
        dev = Device("M:OUTTMP")
        assert dev.name == "M:OUTTMP"

    def test_create_device_with_property(self):
        dev = Device("M:OUTTMP.READING")
        assert dev.name == "M:OUTTMP"

    def test_create_device_with_range(self):
        dev = Device("B:HS23T[0:10]")
        assert dev.name == "B:HS23T"
        assert dev.request.range is not None
        assert dev.request.range.low == 0
        assert dev.request.range.high == 10

    def test_create_device_with_event(self):
        dev = Device("M:OUTTMP@p,1000")
        assert dev.name == "M:OUTTMP"
        assert dev.has_event
        assert dev.is_periodic

    def test_create_device_with_full_drf(self):
        dev = Device("B:HS23T.SETTING[0:10]@P,500")
        assert dev.name == "B:HS23T"
        assert dev.has_event
        assert dev.is_periodic
        assert dev.request.range.low == 0
        assert dev.request.range.high == 10

    def test_create_invalid_drf_raises_valueerror(self):
        with pytest.raises(ValueError):
            Device("X")  # Too short

    def test_create_device_with_invalid_event_raises_valueerror(self):
        with pytest.raises(ValueError):
            Device("M:OUTTMP@Z")  # Invalid event type


# Property Accessor Tests


class TestDeviceProperties:
    """Tests for Device property accessors."""

    def test_drf_returns_canonical_form(self):
        dev = Device("M:OUTTMP")
        assert "M:OUTTMP" in dev.drf

    def test_name_returns_device_name(self):
        dev = Device("M:OUTTMP.READING[0:10]@p,1000")
        assert dev.name == "M:OUTTMP"

    def test_request_returns_datarequest(self):
        dev = Device("M:OUTTMP")
        req = dev.request
        assert req.device == "M:OUTTMP"

    def test_has_event_false_for_default(self):
        dev = Device("M:OUTTMP")
        assert not dev.has_event

    def test_has_event_true_for_explicit(self):
        dev = Device("M:OUTTMP@p,1000")
        assert dev.has_event

    def test_is_periodic_true_for_p_event(self):
        dev = Device("M:OUTTMP@p,1000")
        assert dev.is_periodic

    def test_is_periodic_true_for_q_event(self):
        dev = Device("M:OUTTMP@Q,1000")
        assert dev.is_periodic

    def test_is_periodic_false_for_immediate(self):
        dev = Device("M:OUTTMP@I")
        assert not dev.is_periodic

    def test_is_periodic_false_for_clock_event(self):
        dev = Device("M:OUTTMP@E,0F")
        assert not dev.is_periodic


# Immutability Tests


class TestDeviceImmutability:
    """Tests for Device immutability - modification methods return new instances."""

    def test_with_event_returns_new_device(self):
        original = Device("M:OUTTMP")
        modified = original.with_event("p,1000")
        assert modified.is_periodic
        assert modified.has_event
        assert not original.has_event
        assert not original.is_periodic
        assert original is not modified

    def test_with_range_returns_new_device(self):
        original = Device("B:HS23T")
        modified = original.with_range(0, 10)
        assert modified.request.range is not None
        assert modified.request.range.low == 0
        assert modified.request.range.high == 10
        assert original.request.range is None
        assert original is not modified

    def test_with_backend_returns_new_device(self, mock_backend):
        original = Device("M:OUTTMP")
        modified = original.with_backend(mock_backend)
        assert modified._backend is mock_backend
        assert original._backend is None
        assert original is not modified

    def test_chained_modifications(self, mock_backend):
        dev = Device("M:OUTTMP").with_event("p,1000").with_backend(mock_backend)
        assert dev.is_periodic
        assert dev._backend is mock_backend

    def test_with_event_replaces_existing_event(self):
        dev = Device("M:OUTTMP@p,1000")
        modified = dev.with_event("E,0F")
        assert not modified.is_periodic
        assert "@E,0F" in modified.drf.upper() or "E,0F" in modified.drf


# Read/Get Delegation Tests


class TestDeviceReadOperations:
    """Tests for Device read and get operations."""

    def test_read_delegates_to_backend(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.read()
        assert result == 72.5
        mock_backend.read.assert_called_once()

    def test_get_delegates_to_backend(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.get()
        assert result.value == 72.5
        mock_backend.get.assert_called_once()

    def test_read_with_timeout(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.read(timeout=5.0)
        called_drf = mock_backend.read.call_args[0][0]
        called_timeout = mock_backend.read.call_args[0][1]
        assert "M:OUTTMP" in called_drf
        assert called_timeout == 5.0

    def test_read_uses_reading_property_and_immediate_event(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.read()
        called_drf = mock_backend.read.call_args[0][0]
        assert ".READING" in called_drf
        assert "@I" in called_drf


# ScalarDevice Tests


class TestScalarDevice:
    """Tests for ScalarDevice."""

    def test_scalar_device_returns_float(self, mock_backend):
        mock_backend.read.return_value = 72.5
        dev = ScalarDevice("M:OUTTMP", backend=mock_backend)
        result = dev.read()
        assert isinstance(result, float)
        assert result == 72.5

    def test_scalar_device_converts_int(self, mock_backend):
        mock_backend.read.return_value = 42
        dev = ScalarDevice("M:OUTTMP", backend=mock_backend)
        result = dev.read()
        assert isinstance(result, float)
        assert result == 42.0

    @pytest.mark.parametrize("bad_value", [np.array([1, 2, 3]), "text"])
    def test_scalar_device_raises_on_wrong_type(self, mock_backend, bad_value):
        mock_backend.read.return_value = bad_value
        dev = ScalarDevice("M:OUTTMP", backend=mock_backend)
        with pytest.raises(TypeError, match="Expected scalar"):
            dev.read()


# ArrayDevice Tests


class TestArrayDevice:
    """Tests for ArrayDevice."""

    def test_array_device_returns_ndarray(self, mock_backend):
        mock_backend.read.return_value = np.array([1.0, 2.0, 3.0])
        dev = ArrayDevice("B:HS23T[0:10]", backend=mock_backend)
        result = dev.read()
        assert isinstance(result, np.ndarray)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_array_device_converts_list(self, mock_backend):
        mock_backend.read.return_value = [1.0, 2.0, 3.0]
        dev = ArrayDevice("B:HS23T[0:10]", backend=mock_backend)
        result = dev.read()
        assert isinstance(result, np.ndarray)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_array_device_converts_tuple(self, mock_backend):
        mock_backend.read.return_value = (1.0, 2.0, 3.0)
        dev = ArrayDevice("B:HS23T[0:10]", backend=mock_backend)
        result = dev.read()
        assert isinstance(result, np.ndarray)
        assert list(result) == [1.0, 2.0, 3.0]

    @pytest.mark.parametrize("bad_value", [72.5, "text"])
    def test_array_device_raises_on_wrong_type(self, mock_backend, bad_value):
        mock_backend.read.return_value = bad_value
        dev = ArrayDevice("B:HS23T[0:10]", backend=mock_backend)
        with pytest.raises(TypeError, match="Expected array"):
            dev.read()


# TextDevice Tests


class TestTextDevice:
    """Tests for TextDevice."""

    def test_text_device_returns_string(self, mock_backend):
        mock_backend.read.return_value = "some text"
        dev = TextDevice("M:OUTTMP.DESCRIPTION", backend=mock_backend)
        result = dev.read()
        assert isinstance(result, str)
        assert result == "some text"

    @pytest.mark.parametrize("bad_value", [72.5, np.array([1, 2, 3])])
    def test_text_device_raises_on_wrong_type(self, mock_backend, bad_value):
        mock_backend.read.return_value = bad_value
        dev = TextDevice("M:OUTTMP.DESCRIPTION", backend=mock_backend)
        with pytest.raises(TypeError, match="Expected string"):
            dev.read()


# Special Methods Tests


class TestDeviceSpecialMethods:
    """Tests for Device special methods (__repr__, __eq__, __hash__)."""

    def test_devices_usable_in_set(self):
        dev1 = Device("M:OUTTMP")
        dev2 = Device("M:OUTTMP")
        dev3 = Device("G:AMANDA")
        s = {dev1, dev2, dev3}
        assert len(s) == 2

    def test_devices_usable_as_dict_keys(self):
        dev1 = Device("M:OUTTMP")
        dev2 = Device("M:OUTTMP")
        d = {dev1: "temperature"}
        assert d[dev2] == "temperature"


# Subclass Preservation Tests


class TestSubclassPreservation:
    """Tests that modification methods preserve the Device subclass."""

    def test_scalar_device_with_event_returns_scalar_device(self, mock_backend):
        dev = ScalarDevice("M:OUTTMP")
        modified = dev.with_event("p,1000")
        assert isinstance(modified, ScalarDevice)

    def test_scalar_device_with_backend_returns_scalar_device(self, mock_backend):
        dev = ScalarDevice("M:OUTTMP")
        modified = dev.with_backend(mock_backend)
        assert isinstance(modified, ScalarDevice)

    def test_array_device_with_range_returns_array_device(self, mock_backend):
        dev = ArrayDevice("B:HS23T")
        modified = dev.with_range(0, 10)
        assert isinstance(modified, ArrayDevice)


# Edge Cases


class TestEdgeCases:
    """Tests for edge cases."""

    def test_with_range_single_element(self):
        dev = Device("B:HS23T")
        modified = dev.with_range(5)
        assert modified.request.range is not None
        assert modified.request.range.low == 5

    def test_device_with_lowercase_drf(self):
        dev = Device("m:outtmp")
        assert dev.name.upper() == "M:OUTTMP"


# ─── New tests for property-specific reads ─────────────────────────────


class TestDeviceReadMethods:
    """Tests for property-specific read methods."""

    def test_read_default_field(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.read()
        drf = mock_backend.read.call_args[0][0]
        assert ".READING" in drf
        assert "@I" in drf

    def test_read_raw_field(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.read(field="raw")
        drf = mock_backend.read.call_args[0][0]
        assert ".READING" in drf
        assert ".RAW" in drf
        assert "@I" in drf

    def test_setting_default(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.setting()
        drf = mock_backend.read.call_args[0][0]
        assert ".SETTING" in drf
        assert "@I" in drf

    def test_setting_raw_field(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.setting(field="raw")
        drf = mock_backend.read.call_args[0][0]
        assert ".SETTING" in drf
        assert ".RAW" in drf

    def test_status_default(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.status()
        drf = mock_backend.read.call_args[0][0]
        assert ".STATUS" in drf
        assert "@I" in drf

    def test_status_on_field(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.status(field="on")
        drf = mock_backend.read.call_args[0][0]
        assert ".STATUS" in drf
        assert ".ON" in drf

    def test_analog_alarm_default(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.analog_alarm()
        drf = mock_backend.read.call_args[0][0]
        assert ".ANALOG" in drf
        assert "@I" in drf

    def test_analog_alarm_min_field(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.analog_alarm(field="min")
        drf = mock_backend.read.call_args[0][0]
        assert ".ANALOG" in drf
        assert ".MIN" in drf

    def test_digital_alarm_default(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.digital_alarm()
        drf = mock_backend.read.call_args[0][0]
        assert ".DIGITAL" in drf
        assert "@I" in drf

    def test_description(self, mock_backend):
        mock_backend.read.return_value = "Outside temperature"
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.description()
        assert result == "Outside temperature"
        drf = mock_backend.read.call_args[0][0]
        assert ".DESCRIPTION" in drf

    def test_read_preserves_range(self, mock_backend):
        dev = Device("B:HS23T[0:10]", backend=mock_backend)
        dev.read()
        drf = mock_backend.read.call_args[0][0]
        assert "[0:10]" in drf
        assert ".READING" in drf

    def test_setting_preserves_range(self, mock_backend):
        dev = Device("B:HS23T[0:10]", backend=mock_backend)
        dev.setting()
        drf = mock_backend.read.call_args[0][0]
        assert "[0:10]" in drf
        assert ".SETTING" in drf


class TestDeviceFieldValidation:
    """Tests that invalid fields raise ValueError."""

    def test_invalid_field_for_reading(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        with pytest.raises(ValueError, match="not allowed"):
            dev.read(field="on")  # ON is a STATUS field, not READING

    def test_invalid_field_for_status(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        with pytest.raises(ValueError, match="not allowed"):
            dev.status(field="primary")  # PRIMARY is a READING field, not STATUS

    def test_invalid_field_for_setting(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        with pytest.raises(ValueError, match="not allowed"):
            dev.setting(field="on")

    def test_unknown_field_name(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        with pytest.raises(ValueError):
            dev.read(field="nonexistent")


# ─── Write method tests ────────────────────────────────────────────────


class TestDeviceWriteMethods:
    """Tests for Device write methods."""

    def test_write_basic(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5)
        assert result.success
        drf = mock_backend.write.call_args[0][0]
        value = mock_backend.write.call_args[0][1]
        assert ".SETTING" in drf
        assert "@N" in drf
        assert value == 72.5

    def test_write_raw_field(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.write(100, field="raw")
        drf = mock_backend.write.call_args[0][0]
        assert ".SETTING" in drf
        assert ".RAW" in drf
        assert "@N" in drf

    def test_control_on(self, mock_backend):
        dev = Device("Z:ACLTST", backend=mock_backend)
        dev.control(BasicControl.ON)
        drf = mock_backend.write.call_args[0][0]
        value = mock_backend.write.call_args[0][1]
        assert ".CONTROL" in drf
        assert "@N" in drf
        assert value == BasicControl.ON

    @pytest.mark.parametrize(
        "method,expected",
        [
            ("on", BasicControl.ON),
            ("off", BasicControl.OFF),
            ("reset", BasicControl.RESET),
            ("positive", BasicControl.POSITIVE),
            ("negative", BasicControl.NEGATIVE),
            ("ramp", BasicControl.RAMP),
            ("dc", BasicControl.DC),
        ],
    )
    def test_control_shortcut(self, mock_backend, method, expected):
        """Control shortcuts delegate to control() with correct command."""
        dev = Device("Z:ACLTST", backend=mock_backend)
        getattr(dev, method)()
        assert mock_backend.write.call_args[0][1] == expected

    def test_write_failed_returns_result(self, mock_backend):
        mock_backend.write.return_value = WriteResult(drf="M:OUTTMP.SETTING@N", error_code=-1, message="Fail")
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5)
        assert not result.success
        assert result.error_code == -1


# ─── Alarm setter tests ────────────────────────────────────────────────


class TestDeviceAlarmSetters:
    """Tests for alarm setter methods."""

    def test_set_analog_alarm(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        settings = {"minimum": 40, "maximum": 80}
        dev.set_analog_alarm(settings)
        drf = mock_backend.write.call_args[0][0]
        value = mock_backend.write.call_args[0][1]
        assert ".ANALOG" in drf
        assert "@N" in drf
        assert value == settings

    def test_set_digital_alarm(self, mock_backend):
        dev = Device("M:OUTTMP", backend=mock_backend)
        settings = {"nominal": 0x01, "mask": 0xFF}
        dev.set_digital_alarm(settings)
        drf = mock_backend.write.call_args[0][0]
        value = mock_backend.write.call_args[0][1]
        assert ".DIGITAL" in drf
        assert "@N" in drf
        assert value == settings


# ─── Verify flow tests ─────────────────────────────────────────────────


class TestDeviceVerify:
    """Tests for write-and-verify flow."""

    def test_write_with_verify_true_reads_back(self, mock_backend):
        """verify=True triggers readback after write."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, retry_delay=0))
        assert result.verified is True
        assert result.readback == 72.5
        assert result.attempts == 1

    def test_write_verify_mismatch(self, mock_backend):
        """Verify fails when readback doesn't match."""
        mock_backend.read.return_value = 99.0
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, retry_delay=0, max_attempts=2))
        assert result.verified is False
        assert result.readback == 99.0
        assert result.attempts == 2

    def test_write_verify_with_tolerance(self, mock_backend):
        """Verify passes within tolerance."""
        mock_backend.read.return_value = 72.6
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, tolerance=0.2))
        assert result.verified is True

    def test_write_check_first_skip(self, mock_backend):
        """check_first skips write when value already matches."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(check_first=True, initial_delay=0))
        assert result.skipped is True
        assert result.verified is True
        mock_backend.write.assert_not_called()

    def test_write_check_first_proceeds_on_mismatch(self, mock_backend):
        """check_first proceeds to write when value differs."""
        # First read returns current (different), then readback returns written value
        mock_backend.read.side_effect = [50.0, 72.5]
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(check_first=True, initial_delay=0, retry_delay=0))
        mock_backend.write.assert_called_once()
        assert result.verified is True
        assert not result.skipped

    def test_write_no_verify_returns_plain_result(self, mock_backend):
        """Without verify, no readback occurs."""
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5)
        assert result.verified is None
        assert result.readback is None
        assert result.skipped is False
        assert mock_backend.read.call_count == 0

    def test_write_verify_false_no_readback(self, mock_backend):
        """verify=False explicitly disables verification."""
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=False)
        assert result.verified is None
        assert mock_backend.read.call_count == 0

    def test_control_with_verify(self, mock_backend):
        """Control commands verify via STATUS read."""
        mock_backend.read.return_value = True
        dev = Device("Z:ACLTST", backend=mock_backend)
        result = dev.on(verify=Verify(initial_delay=0, retry_delay=0))
        assert result.verified is True
        # Should have read STATUS.ON
        read_drf = mock_backend.read.call_args[0][0]
        assert ".STATUS" in read_drf
        assert ".ON" in read_drf

    def test_control_verify_check_first_skip(self, mock_backend):
        """Control check_first skips if status already matches."""
        mock_backend.read.return_value = True  # already on
        dev = Device("Z:ACLTST", backend=mock_backend)
        result = dev.on(verify=Verify(check_first=True, initial_delay=0))
        assert result.skipped is True
        mock_backend.write.assert_not_called()

    def test_control_verify_unmapped_command_raises(self, mock_backend):
        """Verify raises ValueError for commands without a STATUS field mapping."""
        dev = Device("Z:ACLTST", backend=mock_backend)
        with pytest.raises(ValueError, match="no STATUS field mapping"):
            dev.control(99, verify=True)

    def test_control_no_verify_unmapped_command_succeeds(self, mock_backend):
        """Unmapped commands work fine without verify."""
        dev = Device("Z:ACLTST", backend=mock_backend)
        result = dev.control(99)
        assert result.verified is None

    def test_write_failed_skips_verify(self, mock_backend):
        """If the write itself fails, no verify is attempted."""
        mock_backend.write.return_value = WriteResult(drf="M:OUTTMP.SETTING@N", error_code=-1, message="Fail")
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0))
        assert result.error_code == -1
        # verify fields should be from the raw result, not from readback
        assert result.verified is None
        mock_backend.read.assert_not_called()

    def test_verify_context_always(self, mock_backend):
        """Verify context with always=True auto-verifies."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP", backend=mock_backend)
        with Verify(always=True, initial_delay=0, retry_delay=0):
            result = dev.write(72.5)
        assert result.verified is True

    def test_write_with_custom_readback_drf(self, mock_backend):
        """Verify uses custom readback DRF when specified."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(readback="M:OUTTMP.READING@I", initial_delay=0))
        read_drf = mock_backend.read.call_args[0][0]
        assert read_drf == "M:OUTTMP.READING@I"
        assert result.verified is True


class TestDeviceExtra:
    """Tests for DRF extra modifier preservation (e.g., <-FTP)."""

    def test_build_drf_preserves_ftp_extra_on_read(self, mock_backend):
        """read() preserves <-FTP extra in the DRF passed to backend."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP<-FTP", backend=mock_backend)
        dev.read()
        drf = mock_backend.read.call_args[0][0]
        assert drf.endswith("<-FTP")
        assert drf == "M:OUTTMP.READING@I<-FTP"

    def test_build_drf_preserves_ftp_extra_on_write(self, mock_backend):
        """write() preserves <-FTP extra in the DRF passed to backend."""
        dev = Device("M:OUTTMP<-FTP", backend=mock_backend)
        dev.write(72.5)
        drf = mock_backend.write.call_args[0][0]
        assert drf.endswith("<-FTP")
        assert drf == "M:OUTTMP.SETTING@N<-FTP"

    def test_build_drf_preserves_ftp_extra_on_status(self, mock_backend):
        """status() preserves <-FTP extra."""
        mock_backend.read.return_value = True
        dev = Device("M:OUTTMP<-FTP", backend=mock_backend)
        dev.status(field="on")
        drf = mock_backend.read.call_args[0][0]
        assert drf == "M:OUTTMP.STATUS.ON@I<-FTP"

    def test_digital_status_preserves_ftp_extra(self, mock_backend):
        """digital_status() preserves <-FTP extra in all three DRFs."""
        from pacsys.types import Reading, ValueType

        mock_backend.get_many.return_value = [
            Reading(drf="M:OUTTMP.STATUS.BIT_VALUE@I<-FTP", value=0x02, value_type=ValueType.SCALAR),
            Reading(drf="M:OUTTMP.STATUS.BIT_NAMES@I<-FTP", value=["On", "Ready"], value_type=ValueType.TEXT_ARRAY),
            Reading(drf="M:OUTTMP.STATUS.BIT_VALUES@I<-FTP", value=["No", "Yes"], value_type=ValueType.TEXT_ARRAY),
        ]
        dev = Device("M:OUTTMP<-FTP", backend=mock_backend)
        dev.digital_status()
        drfs = mock_backend.get_many.call_args[0][0]
        for drf in drfs:
            assert drf.endswith("<-FTP"), f"{drf} missing <-FTP"

    def test_no_extra_omits_suffix(self, mock_backend):
        """Devices without extra don't get a spurious suffix."""
        mock_backend.read.return_value = 72.5
        dev = Device("M:OUTTMP", backend=mock_backend)
        dev.read()
        drf = mock_backend.read.call_args[0][0]
        assert "<-" not in drf


class TestVerifyReadbackError:
    """Tests for verify readback when DeviceError occurs."""

    def test_readback_device_error_returns_verified_false(self, mock_backend):
        """If readback raises DeviceError, result is verified=False, not exception."""
        mock_backend.read.side_effect = DeviceError("M:OUTTMP", 0, -6, "Timeout")
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, retry_delay=0, max_attempts=2))
        assert result.verified is False
        assert result.readback is None
        assert result.attempts == 2

    def test_readback_error_then_success(self, mock_backend):
        """If first readback fails but second succeeds, result is verified=True."""
        mock_backend.read.side_effect = [
            DeviceError("M:OUTTMP", 0, -6, "Timeout"),
            72.5,
        ]
        dev = Device("M:OUTTMP", backend=mock_backend)
        result = dev.write(72.5, verify=Verify(initial_delay=0, retry_delay=0, max_attempts=3))
        assert result.verified is True
        assert result.readback == 72.5
        assert result.attempts == 2

    def test_control_readback_error_returns_verified_false(self, mock_backend):
        """Control verify handles DeviceError during STATUS readback."""
        mock_backend.read.side_effect = DeviceError("Z:ACLTST", 0, -6, "Timeout")
        dev = Device("Z:ACLTST", backend=mock_backend)
        result = dev.on(verify=Verify(initial_delay=0, retry_delay=0, max_attempts=1))
        assert result.verified is False
