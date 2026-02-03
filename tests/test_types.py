"""
Tests for pacsys.types module.
"""

import pytest

from pacsys.types import (
    ValueType,
    Reading,
    WriteResult,
)


class TestReading:
    """Tests for Reading dataclass."""

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


class TestWriteResult:
    """Tests for WriteResult.success property."""

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
