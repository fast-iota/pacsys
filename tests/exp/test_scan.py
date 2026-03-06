"""Tests for scan."""

import pytest

from pacsys.exp._scan import scan, ScanResult, _build_values
from pacsys.testing import FakeBackend


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
