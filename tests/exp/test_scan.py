"""Tests for scan."""

import logging
from unittest import mock

import numpy as np
import pytest

from pacsys import Device
from pacsys.exp import ScanRestoreError, scan
from pacsys.exp._scan import ScanResult, _build_values, _read_step
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

    @pytest.mark.parametrize("readings_per_step", [1, 3])
    def test_duplicate_read_devices(self, fake, readings_per_step):
        drfs = ["M:OUTTMP", "G:AMANDA", "M:OUTTMP[0]@I", "M:OUTTMP[1]@I", "M:OUTTMP[0]@P,100", "M:OUTTMP.READING"]
        calls = []

        def get_many(requests, *, timeout):
            calls.append(list(requests))
            return [
                Reading(drf=drf, value_type=ValueType.SCALAR, value=len(calls) * 10 + i)
                for i, drf in enumerate(requests)
            ]

        with mock.patch.object(fake, "get_many", side_effect=get_many):
            result = scan(
                "Z:ACLTST",
                [drfs[0], drfs[0], *drfs[1:], Device(drfs[-1]), drfs[1]],
                values=[1.0, 2.0],
                readings_per_step=readings_per_step,
                settle=0,
                restore=False,
                backend=fake,
            )

        assert calls == [drfs] * (2 * readings_per_step)
        assert result.read_devices == drfs
        assert len(result.readings) == 2
        for step, readings in enumerate(result.readings):
            assert list(readings) == drfs
            first_sample = step * readings_per_step + 1
            mean_sample = first_sample + (readings_per_step - 1) / 2
            assert [r.value for r in readings.values()] == [mean_sample * 10 + i for i in range(len(drfs))]

    def test_verification_failure_stops_before_reading(self, fake):
        write_device = mock.Mock(request=Device("Z:ACLTST").request)
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

    @pytest.mark.parametrize(
        "write_drf, setting_drf, original",
        [
            ("Z:ACLTST", "Z:ACLTST.SETTING", 42.0),
            ("Z_ACLTST", "Z:ACLTST.SETTING", 42.0),
            ("Z:ACLTST.READING[2].PRIMARY", "Z:ACLTST.SETTING[2].PRIMARY", [0.0, 0.0, 42.0]),
            ("Z:ACLTST.SETTING[2].RAW", "Z:ACLTST.SETTING[2].RAW", [0.0, 0.0, 42.0]),
            ("test:pv.VAL", "test:pv.VAL", 42.0),
        ],
    )
    def test_restores_original_setting(self, fake, write_drf, setting_drf, original):
        fake.set_reading(
            setting_drf, original, value_type=ValueType.SCALAR_ARRAY if isinstance(original, list) else ValueType.SCALAR
        )
        result = scan(
            write_device=write_drf,
            read_devices=["M:OUTTMP"],
            values=[1.0, 2.0],
            settle=0,
            restore=True,
            backend=fake,
        )
        assert result.restored
        assert fake.reads[0] == setting_drf + "@I"
        assert fake.writes == [(setting_drf + "@N", value) for value in (1.0, 2.0, 42.0)]

    def test_failed_error_cleanup_restore_is_logged(self, fake, caplog):
        write_device = mock.Mock(request=Device("Z:ACLTST").request)
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

    @pytest.mark.parametrize(
        "failure",
        [
            WriteResult(drf="Z:ACLTST.SETTING@N", error_code=-1, message="restore failed"),
            OSError("restore transport failed"),
        ],
    )
    def test_failed_normal_restore_preserves_scan_result(self, fake, failure):
        write_device = mock.Mock(request=Device("Z:ACLTST").request)
        write_device.setting.return_value = 42.0
        write_device.write.side_effect = [
            WriteResult(drf="Z:ACLTST.SETTING@N"),
            failure,
        ]

        with (
            mock.patch("pacsys.device.Device", return_value=write_device),
            pytest.raises(ScanRestoreError, match="failed to restore") as exc_info,
        ):
            scan(
                write_device="Z:ACLTST",
                read_devices=["M:OUTTMP"],
                values=[1.0],
                settle=0,
                backend=fake,
            )

        assert exc_info.value.result.set_values == [1.0]
        assert exc_info.value.result.readings[0]["M:OUTTMP"].value == 72.0
        assert exc_info.value.result.restored is False
        assert exc_info.value.__cause__ is (failure if isinstance(failure, Exception) else None)

    @staticmethod
    def _ignore_restore_writes(fake):
        """Restore write (42.0) is accepted but never lands, so its verify readback fails."""
        real_write = fake.write

        def write(drf, value, timeout=None):
            return WriteResult(drf=drf) if value == 42.0 else real_write(drf, value, timeout)

        return mock.patch.object(fake, "write", side_effect=write)

    def test_ambient_verify_failed_restore_readback_is_not_restored(self, fake):
        fake.set_reading("Z:ACLTST.SETTING", 42.0)
        with (
            self._ignore_restore_writes(fake),
            Verify(always=True, initial_delay=0, retry_delay=0, max_attempts=1),
            pytest.raises(ScanRestoreError, match=r"failed to restore Z:ACLTST to 42.0: readback=2.0") as exc_info,
        ):
            scan("Z:ACLTST", ["M:OUTTMP"], values=[1.0, 2.0], settle=0, backend=fake)
        result = exc_info.value.result
        assert result.restored is False
        assert [wr.verified for wr in result.write_results] == [True, True]

    def test_ambient_verify_failed_error_cleanup_restore_is_logged(self, fake, caplog):
        fake.set_reading("Z:ACLTST.SETTING", 42.0)
        fake.get_many = mock.Mock(side_effect=RuntimeError("read failed"))
        with (
            self._ignore_restore_writes(fake),
            Verify(always=True, initial_delay=0, retry_delay=0, max_attempts=1),
            caplog.at_level(logging.ERROR, logger="pacsys.exp._scan"),
            pytest.raises(RuntimeError, match="read failed"),
        ):
            scan("Z:ACLTST", ["M:OUTTMP"], values=[1.0], settle=0, backend=fake)
        assert "Failed to restore Z:ACLTST to 42.0 during error cleanup: readback=1.0" in caplog.text

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

    @pytest.mark.parametrize("value", ["not numeric", {"data": [1.0]}, True, np.bool_(True), np.array([True])])
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
