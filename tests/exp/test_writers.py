"""Tests for CsvWriter and ParquetWriter."""

import base64
import csv
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from pacsys.exp import CsvWriter, LogWriter, ParquetWriter
from pacsys.types import DeviceMeta, Reading, ValueType

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _reading(drf="M:OUTTMP", value=72.5, **kwargs) -> Reading:
    return Reading(
        drf=drf,
        value_type=ValueType.SCALAR,
        value=value,
        timestamp=TS,
        **kwargs,
    )


class TestCsvWriter:
    def test_writes_header_and_rows(self, tmp_path):
        path = tmp_path / "test.csv"
        writer = CsvWriter(path)
        writer.write_readings([_reading(), _reading(value=73.0)])
        writer.close()

        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["timestamp", "drf", "value", "units", "facility_code", "error_code", "message"]
        assert len(rows) == 3
        assert rows[1][1] == "M:OUTTMP"
        assert rows[1][2] == "72.5"
        assert rows[1][4:] == ["0", "0", ""]

    def test_preserves_reading_status(self, tmp_path):
        path = tmp_path / "test.csv"
        readings = [
            Reading(drf="M:OUTTMP", value_type=ValueType.TEXT, value="", timestamp=TS),
            Reading(drf="M:OUTTMP", facility_code=17, error_code=-1, message="Read failed", timestamp=TS),
            _reading(facility_code=1, error_code=1, message="Warning with data"),
        ]
        writer = CsvWriter(path)
        writer.write_readings(readings)
        writer.close()

        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert [row["value"] for row in rows] == ["", "", "72.5"]
        assert [row["facility_code"] for row in rows] == ["0", "17", "1"]
        assert [row["error_code"] for row in rows] == ["0", "-1", "1"]
        assert [row["message"] for row in rows] == ["", "Read failed", "Warning with data"]

    def test_rows_are_visible_before_close(self, tmp_path):
        path = tmp_path / "test.csv"
        writer = CsvWriter(path)
        try:
            writer.write_readings([_reading()])
            with path.open(newline="") as f:
                assert len(list(csv.reader(f))) == 2
        finally:
            writer.close()

    def test_conversion_failure_does_not_write_batch_prefix(self, tmp_path):
        path = tmp_path / "test.csv"
        good = _reading()
        malformed = Reading(drf="D:ARRAY", value_type=ValueType.SCALAR_ARRAY, value=np.array([[1.0, 2.0]]))
        fixed = Reading(drf="D:ARRAY", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0]))
        writer = CsvWriter(path)
        try:
            with pytest.raises(TypeError, match="one-dimensional"):
                writer.write_readings([good, malformed])
            writer.write_readings([good, fixed])
        finally:
            writer.close()

        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        assert rows == [
            ["timestamp", "drf", "value", "units", "facility_code", "error_code", "message"],
            [TS.isoformat(), "M:OUTTMP", "72.5", "", "0", "0", ""],
            ["", "D:ARRAY", "[1.0, 2.0]", "", "0", "0", ""],
        ]

    def test_csv_array_as_json(self, tmp_path):
        """Scalar arrays are serialized as JSON lists, not Python repr."""
        import numpy as np

        path = tmp_path / "test.csv"
        r = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0, 3.0]), timestamp=TS)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()

        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        assert json.loads(rows[1][2]) == [1.0, 2.0, 3.0]

    @pytest.mark.parametrize(
        "value",
        [[True, False], np.array([True, False], dtype=np.bool_)],
        ids=("list", "ndarray"),
    )
    def test_csv_boolean_array_as_json(self, tmp_path, value):
        path = tmp_path / "test.csv"
        r = Reading(drf="Z:BOOL", value_type=ValueType.SCALAR_ARRAY, value=value, timestamp=TS)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()

        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        assert json.loads(rows[1][2]) == [True, False]

    def test_csv_basic_status_as_json(self, tmp_path):
        """Status dicts are serialized as JSON, not Python repr."""
        path = tmp_path / "test.csv"
        status = {"on": True, "ready": False}
        r = Reading(drf="M:OUTTMP", value_type=ValueType.BASIC_STATUS, value=status, timestamp=TS)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()

        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        assert json.loads(rows[1][2]) == status

    def test_csv_alarm_dict_with_numpy_scalar(self, tmp_path):
        """Alarm dicts with nested numpy scalars serialize as JSON, not TypeError."""
        path = tmp_path / "test.csv"
        alarm = {"min": np.float64(1.5), "max": np.float64(9.0), "enabled": np.bool_(True)}
        r = Reading(drf="M:OUTTMP", value_type=ValueType.ANALOG_ALARM, value=alarm, timestamp=TS)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()

        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        assert json.loads(rows[1][2]) == {"min": 1.5, "max": 9.0, "enabled": True}

    def test_csv_raw_bytes_as_base64(self, tmp_path):
        """Raw bytes are serialized as base64."""
        path = tmp_path / "test.csv"
        raw = b"\x00\x01\x02\xff"
        r = Reading(drf="M:OUTTMP", value_type=ValueType.RAW, value=raw, timestamp=TS)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()

        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        assert base64.b64decode(rows[1][2]) == raw

    def test_implements_protocol(self):
        writer = CsvWriter.__new__(CsvWriter)
        assert isinstance(writer, LogWriter)

    def test_handles_none_timestamp(self, tmp_path):
        path = tmp_path / "test.csv"
        r = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=1.0)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        assert rows[1][0] == ""  # empty timestamp


def _read_parquet(path):
    import pyarrow.parquet as pq

    return pq.read_table(path)


class TestParquetWriter:
    def test_scalar_values(self, tmp_path):
        """Scalar values stored as native float64."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        writer = ParquetWriter(path)
        writer.write_readings([_reading(value=72.5), _reading(value=73.0)])
        writer.close()

        table = _read_parquet(path)
        assert len(table) == 2
        assert table.column("value").to_pylist() == [72.5, 73.0]
        assert table.column("int_value").to_pylist() == [None, None]
        assert table.column("value_array").to_pylist() == [None, None]
        assert table.column("value_text").to_pylist() == [None, None]
        assert table.column("value_type").to_pylist() == ["scalar", "scalar"]

    def test_scalar_int_value(self, tmp_path):
        """Integer scalars are stored as int64, preserving type."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(drf="Z:ACLTST", value_type=ValueType.SCALAR, value=42, timestamp=TS)
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("value").to_pylist() == [None]
        assert table.column("int_value").to_pylist() == [42]

    def test_scalar_bool_value(self, tmp_path):
        """Boolean scalars are stored as int64."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(drf="Z:ACLTST", value_type=ValueType.SCALAR, value=True, timestamp=TS)
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("value").to_pylist() == [None]
        assert table.column("int_value").to_pylist() == [1]

    def test_scalar_numpy_int_and_bool(self, tmp_path):
        """Numpy bool/integer scalars route to int64 column (not float, not TypeError)."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        readings = [
            Reading(drf="Z:ACLTST", value_type=ValueType.SCALAR, value=np.bool_(True), timestamp=TS),
            Reading(drf="Z:ACLTST", value_type=ValueType.SCALAR, value=np.int64(-7), timestamp=TS),
            Reading(drf="Z:ACLTST", value_type=ValueType.SCALAR, value=np.uint32(9), timestamp=TS),
            Reading(drf="Z:ACLTST", value_type=ValueType.SCALAR, value=np.float32(1.5), timestamp=TS),
        ]
        writer = ParquetWriter(path)
        writer.write_readings(readings)
        writer.close()

        table = _read_parquet(path)
        assert table.column("int_value").to_pylist() == [1, -7, 9, None]
        assert table.column("value").to_pylist() == [None, None, None, 1.5]

    def test_scalar_array(self, tmp_path):
        """Scalar arrays stored in value_array as list<float64>."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(
            drf="M:OUTTMP",
            value_type=ValueType.SCALAR_ARRAY,
            value=[1.0, 2.0, 3.0],
            timestamp=TS,
        )
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("value").to_pylist() == [None]
        assert table.column("value_array").to_pylist() == [[1.0, 2.0, 3.0]]

    def test_scalar_array_numpy(self, tmp_path):
        """numpy ndarrays stored in value_array."""
        pytest.importorskip("pyarrow")
        np = pytest.importorskip("numpy")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(
            drf="M:OUTTMP",
            value_type=ValueType.SCALAR_ARRAY,
            value=np.array([10.0, 20.0]),
            timestamp=TS,
        )
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("value_array").to_pylist() == [[10.0, 20.0]]

    @pytest.mark.parametrize(
        "value",
        [[True, False], np.array([True, False], dtype=np.bool_)],
        ids=("list", "ndarray"),
    )
    def test_scalar_array_boolean(self, tmp_path, value):
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(drf="Z:BOOL", value_type=ValueType.SCALAR_ARRAY, value=value, timestamp=TS)
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("value_array").to_pylist() == [[1.0, 0.0]]

    def test_text_value(self, tmp_path):
        """Text values stored as plain string in value_text."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(drf="M:OUTTMP", value_type=ValueType.TEXT, value="hello", timestamp=TS)
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("value").to_pylist() == [None]
        assert table.column("value_text").to_pylist() == ["hello"]

    def test_text_array(self, tmp_path):
        """Text arrays stored as JSON in value_text."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(
            drf="M:OUTTMP",
            value_type=ValueType.TEXT_ARRAY,
            value=["a", "b", "c"],
            timestamp=TS,
        )
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        result = table.column("value_text").to_pylist()[0]
        assert json.loads(result) == ["a", "b", "c"]

    def test_analog_alarm(self, tmp_path):
        """Analog alarm dicts stored as JSON in value_text."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        alarm = {
            "minimum": 0.0,
            "maximum": 100.0,
            "alarm_enable": True,
            "alarm_status": False,
            "abort": False,
            "abort_inhibit": False,
            "tries_needed": 3,
            "tries_now": 0,
        }
        path = tmp_path / "test.parquet"
        r = Reading(
            drf="M:OUTTMP",
            value_type=ValueType.ANALOG_ALARM,
            value=alarm,
            timestamp=TS,
        )
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        result = json.loads(table.column("value_text").to_pylist()[0])
        assert result == alarm
        assert table.column("value_type").to_pylist() == ["anaAlarm"]

    def test_basic_status(self, tmp_path):
        """Basic status dicts stored as JSON in value_text."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        status = {"on": True, "ready": True, "remote": False, "positive": True}
        path = tmp_path / "test.parquet"
        r = Reading(
            drf="M:OUTTMP",
            value_type=ValueType.BASIC_STATUS,
            value=status,
            timestamp=TS,
        )
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        result = json.loads(table.column("value_text").to_pylist()[0])
        assert result == status

    def test_digital_alarm(self, tmp_path):
        """Digital alarm dicts are stored as JSON in value_text."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        alarm = {"nominal": 0xAA, "mask": 0xFF, "alarm_enable": True}
        path = tmp_path / "test.parquet"
        writer = ParquetWriter(path)
        writer.write_readings([Reading(drf="Z:ACLTST", value_type=ValueType.DIGITAL_ALARM, value=alarm, timestamp=TS)])
        writer.close()

        table = _read_parquet(path)
        assert json.loads(table.column("value_text").to_pylist()[0]) == alarm
        assert table.column("value_type").to_pylist() == ["digAlarm"]

    def test_raw_bytes(self, tmp_path):
        """Raw bytes stored as base64 in value_text."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        raw = b"\x00\x01\x02\xff"
        path = tmp_path / "test.parquet"
        r = Reading(drf="M:OUTTMP", value_type=ValueType.RAW, value=raw, timestamp=TS)
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        encoded = table.column("value_text").to_pylist()[0]
        assert base64.b64decode(encoded) == raw

    def test_error_reading(self, tmp_path):
        """Status survives file roundtrips, including warnings with data."""
        pa = pytest.importorskip("pyarrow")
        from pacsys.exp import ParquetWriter

        path = tmp_path / "test.parquet"
        readings = [
            _reading(),
            Reading(drf="M:OUTTMP", facility_code=17, error_code=-66, message="Read failed", timestamp=TS),
            _reading(facility_code=1, error_code=1, message="Warning with data"),
            Reading(drf="M:OUTTMP", facility_code=255, error_code=-66, message="", timestamp=TS),
        ]
        writer = ParquetWriter(path)
        writer.write_readings(readings)
        writer.close()

        table = _read_parquet(path)
        assert table.column("value").to_pylist() == [72.5, None, 72.5, None]
        assert table.column("int_value").to_pylist() == [None] * 4
        assert table.column("value_array").to_pylist() == [None] * 4
        assert table.column("value_text").to_pylist() == [None] * 4
        assert table.column("error_code").to_pylist() == [0, -66, 1, -66]
        assert table.column("facility_code").to_pylist() == [0, 17, 1, 255]
        assert table.column("message").to_pylist() == [None, "Read failed", "Warning with data", ""]
        assert table.schema.field("facility_code").type == pa.int16()
        assert table.schema.field("message").type == pa.string()
        assert table.column_names[-3:] == ["cycle", "facility_code", "message"]

    def test_timestamp_native(self, tmp_path):
        """Timestamps stored as native pyarrow timestamps."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        writer = ParquetWriter(path)
        writer.write_readings([_reading()])
        writer.close()

        table = _read_parquet(path)
        ts_col = table.column("timestamp").to_pylist()
        assert ts_col[0] == TS

    def test_none_timestamp(self, tmp_path):
        """None timestamp stored as null."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=1.0)
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("timestamp").to_pylist() == [None]

    def test_units_and_cycle(self, tmp_path):
        """Units and cycle columns populated correctly."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(
            drf="M:OUTTMP",
            value_type=ValueType.SCALAR,
            value=72.5,
            timestamp=TS,
            cycle=14,
            meta=DeviceMeta(device_index=0, name="M:OUTTMP", description="", units="degF"),
        )
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("units").to_pylist() == ["degF"]
        assert table.column("cycle").to_pylist() == [14]

    def test_incremental_writes(self, tmp_path):
        """Multiple write_readings calls append to same file."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        writer = ParquetWriter(path)
        writer.write_readings([_reading(value=72.0)])
        writer.write_readings([_reading(value=73.0)])
        writer.close()

        table = _read_parquet(path)
        assert len(table) == 2
        assert table.column("value").to_pylist() == [72.0, 73.0]

    def test_empty_close_no_file(self, tmp_path):
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        writer = ParquetWriter(path)
        writer.close()
        assert not path.exists()

    def test_mixed_value_types(self, tmp_path):
        """Different value types in same file route to correct columns."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        readings = [
            Reading(drf="D:SCALAR", value_type=ValueType.SCALAR, value=1.0, timestamp=TS),
            Reading(drf="D:ARRAY", value_type=ValueType.SCALAR_ARRAY, value=[1.0, 2.0], timestamp=TS),
            Reading(drf="D:TEXT", value_type=ValueType.TEXT, value="hi", timestamp=TS),
        ]
        writer = ParquetWriter(path)
        writer.write_readings(readings)
        writer.close()

        table = _read_parquet(path)
        assert len(table) == 3
        vals = table.column("value").to_pylist()
        assert vals == [1.0, None, None]
        arrs = table.column("value_array").to_pylist()
        assert arrs == [None, [1.0, 2.0], None]
        texts = table.column("value_text").to_pylist()
        assert texts == [None, None, "hi"]

    def test_zstd_compression(self, tmp_path):
        """Parquet file uses ZSTD compression."""
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        writer = ParquetWriter(path)
        writer.write_readings([_reading()])
        writer.close()

        meta = pq.read_metadata(path)
        col_meta = meta.row_group(0).column(0)
        assert col_meta.compression == "ZSTD"

    def test_implements_protocol(self):
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        writer = ParquetWriter.__new__(ParquetWriter)
        assert isinstance(writer, LogWriter)


@pytest.mark.parametrize("writer_type", [CsvWriter, ParquetWriter])
@pytest.mark.parametrize("data", [[1.0, 2.0, 3.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], ids=["scalar", "records"])
def test_timed_scalar_array_roundtrip(tmp_path, writer_type, data):
    if writer_type is ParquetWriter:
        pytest.importorskip("pyarrow")
    path = tmp_path / "readings"
    reading = Reading(
        drf="M:OUTTMP",
        value_type=ValueType.TIMED_SCALAR_ARRAY,
        value={"data": np.array(data), "micros": np.array([100, 200, 300], dtype=np.int64)},
        timestamp=TS,
    )
    writer = writer_type(path)
    try:
        writer.write_readings([reading])
    finally:
        writer.close()

    if writer_type is CsvWriter:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        result = json.loads(rows[0]["value"])
    else:
        table = _read_parquet(path)
        assert table.column("value").to_pylist() == [None]
        assert table.column("value_array").to_pylist() == [None]
        assert table.column("value_type").to_pylist() == ["timedScalarArr"]
        result = json.loads(table.column("value_text").to_pylist()[0])
    assert result == {"data": data, "micros": [100, 200, 300]}


@pytest.mark.parametrize("writer_type", [CsvWriter, ParquetWriter])
@pytest.mark.parametrize(
    "value_type,value",
    [
        (ValueType.TIMED_SCALAR_ARRAY, {"data": np.zeros((1, 1, 1)), "micros": np.array([100])}),
        (ValueType.TIMED_SCALAR_ARRAY, {"data": np.array([[1.0]]), "micros": np.array([[100]])}),
        (ValueType.SCALAR_ARRAY, np.array([[1.0]])),
    ],
    ids=["3d-data", "2d-micros", "2d-scalar-array"],
)
def test_writer_rejects_unsupported_array_dimensions(tmp_path, writer_type, value_type, value):
    if writer_type is ParquetWriter:
        pytest.importorskip("pyarrow")
    writer = writer_type(tmp_path / "readings")
    try:
        with pytest.raises(TypeError, match="one-dimensional"):
            writer.write_readings([Reading(drf="M:OUTTMP", value_type=value_type, value=value)])
    finally:
        writer.close()
