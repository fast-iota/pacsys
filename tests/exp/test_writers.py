"""Tests for CsvWriter and ParquetWriter."""

import csv
import json
import base64
from datetime import datetime, timezone

import pytest

from pacsys.exp._writers import CsvWriter, LogWriter
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

        rows = list(csv.reader(open(path)))
        assert rows[0] == ["timestamp", "drf", "value", "units"]
        assert len(rows) == 3
        assert rows[1][1] == "M:OUTTMP"
        assert rows[1][2] == "72.5"

    def test_csv_array_as_json(self, tmp_path):
        """Scalar arrays are serialized as JSON lists, not Python repr."""
        import numpy as np

        path = tmp_path / "test.csv"
        r = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0, 3.0]), timestamp=TS)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()

        rows = list(csv.reader(open(path)))
        assert json.loads(rows[1][2]) == [1.0, 2.0, 3.0]

    def test_csv_basic_status_as_json(self, tmp_path):
        """Status dicts are serialized as JSON, not Python repr."""
        path = tmp_path / "test.csv"
        status = {"on": True, "ready": False}
        r = Reading(drf="M:OUTTMP", value_type=ValueType.BASIC_STATUS, value=status, timestamp=TS)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()

        rows = list(csv.reader(open(path)))
        assert json.loads(rows[1][2]) == status

    def test_csv_raw_bytes_as_base64(self, tmp_path):
        """Raw bytes are serialized as base64."""
        path = tmp_path / "test.csv"
        raw = b"\x00\x01\x02\xff"
        r = Reading(drf="M:OUTTMP", value_type=ValueType.RAW, value=raw, timestamp=TS)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()

        rows = list(csv.reader(open(path)))
        assert base64.b64decode(rows[1][2]) == raw

    def test_implements_protocol(self):
        assert isinstance(CsvWriter, type)
        writer = CsvWriter.__new__(CsvWriter)
        assert isinstance(writer, LogWriter)

    def test_handles_none_timestamp(self, tmp_path):
        path = tmp_path / "test.csv"
        r = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=1.0)
        writer = CsvWriter(path)
        writer.write_readings([r])
        writer.close()
        rows = list(csv.reader(open(path)))
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

    def test_timed_scalar_array(self, tmp_path):
        """Timed scalar arrays (dict with data+micros) stored as JSON in value_text."""
        pytest.importorskip("pyarrow")
        np = pytest.importorskip("numpy")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(
            drf="M:OUTTMP",
            value_type=ValueType.TIMED_SCALAR_ARRAY,
            value={"data": np.array([1.0, 2.0, 3.0]), "micros": np.array([100, 200, 300], dtype=np.int64)},
            timestamp=TS,
        )
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("value").to_pylist() == [None]
        assert table.column("value_array").to_pylist() == [None]
        result = json.loads(table.column("value_text").to_pylist()[0])
        assert result["data"] == [1.0, 2.0, 3.0]
        assert result["micros"] == [100, 200, 300]

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
        """Error readings: no value columns populated, error_code set."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        r = Reading(drf="M:OUTTMP", error_code=-66, timestamp=TS)
        writer = ParquetWriter(path)
        writer.write_readings([r])
        writer.close()

        table = _read_parquet(path)
        assert table.column("value").to_pylist() == [None]
        assert table.column("int_value").to_pylist() == [None]
        assert table.column("value_array").to_pylist() == [None]
        assert table.column("value_text").to_pylist() == [None]
        assert table.column("error_code").to_pylist() == [-66]

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
