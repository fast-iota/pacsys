"""Tests for scan."""

import logging
from unittest import mock

import numpy as np
import pytest

from pacsys.exp import ScanRestoreError
from pacsys.exp._scan import ScanResult, _build_values, _read_step, scan
from pacsys.testing import FakeBackend
from pacsys.types import Reading, ValueType, WriteResult
from pacsys.verify import Verify


@pytest.fixture
def fake():
    fb = FakeBackend()
    fb.set_reading("Z:ACLTST.SETTING", 0.0)
    fb.set_reading("Z:ACLTST", 0.0)
    fb.set_reading("M:OUTTMP", 72.0)
    fb.set_reading("G:AMANDA", 1.0)
    return fb


class TestBuildValues:
    def test_explicit_values(self):
        assert _build_values([1.0, 2.0, 3.0], None, None, None) == [1.0, 2.0, 3.0]

    def test_linear_range(self):
        vals = _build_values(None, 0.0, 1.0, 3)
        assert vals == pytest.approx([0.0, 0.5, 1.0])

    def test_both_raises(self):
        with pytest.raises(ValueError, match="not both"):
            _build_values([1.0], 0.0, 1.0, 3)

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            _build_values(None, None, None, None)

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="values cannot be empty"):
            _build_values([], None, None, None)

    def test_steps_less_than_2_raises(self):
        with pytest.raises(ValueError, match="steps must be >= 2"):
            _build_values(None, 0.0, 1.0, 1)

    def test_numpy_linspace(self):
        vals = _build_values(np.linspace(0.0, 1.0, 5), None, None, None)
        assert vals == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_numpy_single_zero_not_empty(self):
        # bool(np.array([0.0])) is False -- must not be mis-rejected as empty
        assert _build_values(np.array([0.0]), None, None, None) == [0.0]

    def test_generator_and_empty_iterables(self):
        assert _build_values((v for v in [1.0, 2.0]), None, None, None) == [1.0, 2.0]
        for empty in (np.array([]), (v for v in []), ()):
            with pytest.raises(ValueError, match="values cannot be empty"):
                _build_values(empty, None, None, None)


class TestScan:
    def test_basic_scan(self, fake):
        result = scan(
            write_device="Z:ACLTST",
            read_devices=["M:OUTTMP"],
            values=[0.0, 1.0, 2.0],
            settle=0,
            backend=fake,
        )
        assert isinstance(result, ScanResult)
        assert len(result.set_values) == 3
        assert len(result.readings) == 3
        assert len(result.write_results) == 3
        assert all(wr.ok for wr in result.write_results)

    def test_verification_failure_stops_before_reading(self, fake):
        write_device = mock.Mock()
        write_device.setting.return_value = 42.0
        write_device.write.side_effect = [
            WriteResult(drf="Z:ACLTST.SETTING@N", verified=True, readback=1.0),
            WriteResult(drf="Z:ACLTST.SETTING@N", verified=False, readback=0.0),
            WriteResult(drf="Z:ACLTST.SETTING@N"),
        ]

        with mock.patch("pacsys.device.Device", return_value=write_device):
            result = scan(
                write_device="Z:ACLTST",
                read_devices=["M:OUTTMP"],
                values=[1.0, 2.0, 3.0],
                settle=0,
                verify=Verify(initial_delay=0, retry_delay=0),
                backend=fake,
            )

        assert result.set_values == [1.0, 2.0]
        assert len(result.write_results) == 2
        assert result.write_results[-1].verified is False
        assert len(result.readings) == 1
        assert fake.reads == ["M:OUTTMP"]
        assert result.restored
        assert write_device.write.call_args_list[-1] == mock.call(42.0, timeout=None)

    def test_failed_write_marks_aborted(self, fake):
        fake.set_write_result("Z:ACLTST.SETTING@N", success=False, error_code=-42, message="rejected")
        result = scan(
            write_device="Z:ACLTST",
            read_devices=["M:OUTTMP"],
            values=[0.0, 1.0, 2.0],
            settle=0,
            restore=False,
            backend=fake,
        )
        assert result.aborted
        assert len(result.write_results) == 1
        assert result.readings == []

    def test_restores_original_setting(self, fake):
        fake.set_reading("Z:ACLTST.SETTING", 42.0)
        scan(
            write_device="Z:ACLTST",
            read_devices=["M:OUTTMP"],
            values=[1.0, 2.0],
            settle=0,
            restore=True,
            backend=fake,
        )
        # Last write should restore the original SETTING value (42.0)
        last_write = fake.writes[-1]
        assert last_write[1] == 42.0

    def test_failed_error_cleanup_restore_is_logged(self, fake, caplog):
        write_device = mock.Mock()
        write_device.setting.return_value = 42.0
        write_device.write.side_effect = [
            WriteResult(drf="Z:ACLTST.SETTING@N"),
            WriteResult(drf="Z:ACLTST.SETTING@N", error_code=-1, message="restore failed"),
        ]
        fake.get_many = mock.Mock(side_effect=RuntimeError("read failed"))

        with (
            mock.patch("pacsys.device.Device", return_value=write_device),
            caplog.at_level(logging.ERROR, logger="pacsys.exp._scan"),
            pytest.raises(RuntimeError, match="read failed"),
        ):
            scan(
                write_device="Z:ACLTST",
                read_devices=["M:OUTTMP"],
                values=[1.0],
                settle=0,
                backend=fake,
            )

        assert "Failed to restore Z:ACLTST to 42.0 during error cleanup: restore failed" in caplog.text

    def test_failed_normal_restore_preserves_scan_result(self, fake):
        write_device = mock.Mock()
        write_device.setting.return_value = 42.0
        write_device.write.side_effect = [
            WriteResult(drf="Z:ACLTST.SETTING@N"),
            WriteResult(drf="Z:ACLTST.SETTING@N", error_code=-1, message="restore failed"),
        ]

        with (
            mock.patch("pacsys.device.Device", return_value=write_device),
            pytest.raises(RuntimeError, match="failed to restore") as exc_info,
        ):
            scan(
                write_device="Z:ACLTST",
                read_devices=["M:OUTTMP"],
                values=[1.0],
                settle=0,
                backend=fake,
            )

        assert exc_info.value.result.readings
        assert exc_info.value.result.restored is False
        assert isinstance(exc_info.value, ScanRestoreError)

    def test_no_restore(self, fake):
        fake.set_reading("Z:ACLTST.SETTING", 42.0)
        result = scan(
            write_device="Z:ACLTST",
            read_devices=["M:OUTTMP"],
            values=[1.0],
            settle=0,
            restore=False,
            backend=fake,
        )
        assert not result.restored
        write_values = [v for _, v in fake.writes]
        assert 42.0 not in write_values

    def test_abort_if(self, fake):
        result = scan(
            write_device="Z:ACLTST",
            read_devices=["M:OUTTMP"],
            values=[0.0, 1.0, 2.0, 3.0, 4.0],
            settle=0,
            abort_if=lambda readings: True,
            backend=fake,
        )
        assert result.aborted
        assert len(result.set_values) == 1

    def test_linear_range(self, fake):
        result = scan(
            write_device="Z:ACLTST",
            read_devices=["M:OUTTMP"],
            start=0.0,
            stop=2.0,
            steps=3,
            settle=0,
            backend=fake,
        )
        assert result.set_values == pytest.approx([0.0, 1.0, 2.0])

    def test_multiple_read_devices(self, fake):
        result = scan(
            write_device="Z:ACLTST",
            read_devices=["M:OUTTMP", "G:AMANDA"],
            values=[1.0],
            settle=0,
            backend=fake,
        )
        step = result.readings[0]
        assert len(step) == 2

    def test_readings_per_step(self, fake):
        result = scan(
            write_device="Z:ACLTST",
            read_devices=["M:OUTTMP"],
            values=[1.0],
            settle=0,
            readings_per_step=3,
            backend=fake,
        )
        assert len(result.readings) == 1

    def test_readings_per_step_rejects_numpy_boolean(self):
        reading = Reading(drf="Z:BOOL", value_type=ValueType.SCALAR, value=np.bool_(True))
        backend = mock.Mock()
        backend.get_many.return_value = [reading]

        with pytest.raises(TypeError, match="Z:BOOL"):
            _read_step(backend, ["Z:BOOL"], readings_per_step=2, timeout=None)

    def test_readings_per_step_averages_arrays(self):
        backend = mock.Mock()
        backend.get_many.side_effect = [
            [Reading(drf="Z:ARRAY", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0]))],
            [Reading(drf="Z:ARRAY", value_type=ValueType.SCALAR_ARRAY, value=[3.0, 4.0])],
        ]

        result = _read_step(backend, ["Z:ARRAY"], readings_per_step=2, timeout=None)

        np.testing.assert_array_equal(result["Z:ARRAY"].value, [2.0, 3.0])
        assert result["Z:ARRAY"].value_type == ValueType.SCALAR_ARRAY

    def test_readings_per_step_rejects_mixed_array_shapes(self):
        backend = mock.Mock()
        backend.get_many.side_effect = [
            [Reading(drf="Z:ARRAY", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0]))],
            [Reading(drf="Z:ARRAY", value_type=ValueType.SCALAR_ARRAY, value=np.array([3.0]))],
        ]

        with pytest.raises(ValueError, match="Z:ARRAY"):
            _read_step(backend, ["Z:ARRAY"], readings_per_step=2, timeout=None)

    @pytest.mark.parametrize("value", ["not numeric", {"data": [1.0]}, True])
    def test_readings_per_step_rejects_non_numeric_values(self, value):
        reading = Reading(drf="Z:BAD", value_type=ValueType.SCALAR, value=value)
        backend = mock.Mock()
        backend.get_many.return_value = [reading]

        with pytest.raises(TypeError, match="Z:BAD"):
            _read_step(backend, ["Z:BAD"], readings_per_step=2, timeout=None)

    def test_readings_per_step_zero_raises(self, fake):
        with pytest.raises(ValueError, match="readings_per_step must be >= 1"):
            scan(
                write_device="Z:ACLTST",
                read_devices=["M:OUTTMP"],
                values=[1.0],
                settle=0,
                readings_per_step=0,
                backend=fake,
            )
