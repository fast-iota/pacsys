"""Tests for pacsys.errors exception transport."""

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import pytest

from pacsys.errors import DeviceError, ReadError
from pacsys.types import Reading, ValueType


def _roundtrip(exc):
    return pickle.loads(pickle.dumps(exc))


def _mixed_readings() -> list[Reading]:
    return [
        Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5),
        Reading(drf="Z:ACLTST", facility_code=1, error_code=-1, message="failed"),
    ]


def _raise_read_error():
    raise ReadError(_mixed_readings(), "batch failed")


class TestDeviceErrorPickle:
    @pytest.mark.parametrize("message", ["failed", None])
    def test_roundtrip(self, message):
        exc = DeviceError("Z:ACLTST", 1, -1, message)
        exc.extra = "state"
        got = _roundtrip(exc)
        assert type(got) is DeviceError
        assert got.args == exc.args
        assert str(got) == str(exc)
        assert (got.drf, got.facility_code, got.error_code, got.message) == ("Z:ACLTST", 1, -1, message)
        assert got.extra == "state"


class TestReadErrorPickle:
    def test_roundtrip_mixed_readings(self):
        exc = ReadError(_mixed_readings(), "batch failed")
        exc.extra = "state"
        got = _roundtrip(exc)
        assert type(got) is ReadError
        assert got.args == exc.args
        assert str(got) == str(exc) == "batch failed (failed: Z:ACLTST)"
        assert got.readings == exc.readings
        assert [r.ok for r in got.readings] == [True, False]
        assert got.extra == "state"

    def test_process_pool_delivers_original(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            with pytest.raises(ReadError) as ei:
                pool.submit(_raise_read_error).result(timeout=10)
            assert str(ei.value) == "batch failed (failed: Z:ACLTST)"
            assert ei.value.readings == _mixed_readings()
            # Pool survives the exception transport
            assert pool.submit(int, "7").result(timeout=10) == 7
