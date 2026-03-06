"""Tests for CsvWriter and ParquetWriter."""

import csv
from datetime import datetime, timezone

import pytest

from pacsys.exp._writers import CsvWriter, LogWriter
from pacsys.types import Reading, ValueType


def _reading(drf="M:OUTTMP", value=72.5) -> Reading:
    return Reading(
        drf=drf,
        value_type=ValueType.SCALAR,
        value=value,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
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


class TestParquetWriter:
    def test_writes_parquet_file(self, tmp_path):
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        writer = ParquetWriter(path)
        writer.write_readings([_reading(), _reading(value=73.0)])
        writer.close()

        import pyarrow.parquet as pq

        table = pq.read_table(path)
        assert len(table) == 2
        assert "drf" in table.column_names

    def test_incremental_writes(self, tmp_path):
        """Multiple write_readings calls append to same file."""
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        writer = ParquetWriter(path)
        writer.write_readings([_reading(value=72.0)])
        writer.write_readings([_reading(value=73.0)])
        writer.close()

        import pyarrow.parquet as pq

        table = pq.read_table(path)
        assert len(table) == 2

    def test_empty_close_no_file(self, tmp_path):
        pytest.importorskip("pyarrow")
        from pacsys.exp._writers import ParquetWriter

        path = tmp_path / "test.parquet"
        writer = ParquetWriter(path)
        writer.close()
        assert not path.exists()
