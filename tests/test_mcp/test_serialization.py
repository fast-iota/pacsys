from datetime import datetime, timezone

import numpy as np

from pacsys.mcp._serialization import reading_to_dict, write_result_to_dict
from pacsys.types import DeviceMeta, Reading, ValueType, WriteResult


def test_scalar_reading():
    r = Reading(
        drf="M:OUTTMP.READING.SCALED@I",
        value_type=ValueType.SCALAR,
        value=72.5,
        error_code=0,
        timestamp=datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc),
        cycle=0,
        meta=DeviceMeta(device_index=0, name="M:OUTTMP", description="Outside temp", units="deg F"),
    )
    d = reading_to_dict(r)
    assert d["ok"] is True
    assert d["name"] == "M:OUTTMP"
    assert d["value"] == 72.5
    assert d["units"] == "deg F"
    assert d["timestamp"] == "2026-03-03T14:00:00+00:00"
    assert d["cycle"] == 0
    assert "error" not in d


def test_error_reading():
    r = Reading(drf="M:BADDEV", facility_code=16, error_code=-42, message="DIO_NO_SUCH - device not found")
    d = reading_to_dict(r)
    assert d["ok"] is False
    assert d["value"] is None
    assert (d["facility_code"], d["error_code"]) == (16, -42)
    assert d["error"] == "DIO_NO_SUCH - device not found"


def test_numpy_array_reading():
    arr = np.array([1.0, 2.0, 3.0])
    r = Reading(
        drf="M:OUTTMP[0:2]",
        value_type=ValueType.SCALAR_ARRAY,
        value=arr,
        error_code=0,
        meta=DeviceMeta(device_index=0, name="M:OUTTMP", description="Outside temp"),
    )
    d = reading_to_dict(r)
    assert d["value"] == [1.0, 2.0, 3.0]


def test_bytes_reading():
    r = Reading(
        drf="M:OUTTMP.READING.RAW",
        value_type=ValueType.RAW,
        value=b"\x01\x02\x03",
        error_code=0,
    )
    d = reading_to_dict(r)
    assert d["value"] == "AQID"  # base64


def test_none_value_reading():
    r = Reading(drf="M:OUTTMP", error_code=0, value=None)
    d = reading_to_dict(r)
    # Reading.ok is False when value is None (no usable data)
    assert d["ok"] is False
    assert d["value"] is None


def test_warning_reading_includes_diagnostic():
    usable = reading_to_dict(
        Reading(
            drf="M:OUTTMP",
            value_type=ValueType.SCALAR,
            value=72.5,
            facility_code=17,
            error_code=2,
            message="stale",
        )
    )
    status_only = reading_to_dict(Reading(drf="Z:NOTFND", facility_code=17, error_code=1, message="DPM_PEND"))

    assert usable["ok"] is True
    assert usable["value"] == 72.5
    assert (usable["facility_code"], usable["error_code"]) == (17, 2)
    assert usable["message"] == "stale"
    assert "error" not in usable
    assert status_only["ok"] is False
    assert (status_only["facility_code"], status_only["error_code"]) == (17, 1)
    assert status_only["error"] == "DPM_PEND"
    assert "message" not in status_only


def test_write_result_success():
    wr = WriteResult(drf="Z:ACLTST.SETTING.SCALED@N", error_code=0)
    d = write_result_to_dict(wr)
    assert d["ok"] is True
    assert d["drf"] == "Z:ACLTST.SETTING.SCALED@N"
    assert "error" not in d


def test_write_result_error():
    denied = write_result_to_dict(
        WriteResult(drf="Z:ACLTST", facility_code=1, error_code=-1, message="Permission denied")
    )
    silent = write_result_to_dict(WriteResult(drf="Z:ACLTST", facility_code=17, error_code=-44))

    assert denied["ok"] is False
    assert (denied["facility_code"], denied["error_code"]) == (1, -1)
    assert denied["message"] == denied["error"] == "Permission denied"
    assert silent["ok"] is False
    assert (silent["facility_code"], silent["error_code"]) == (17, -44)
    assert silent["error"] == "Write failed (facility=17, error=-44)"
    assert "message" not in silent
