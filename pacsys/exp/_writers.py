"""Log writers for DataLogger: CSV and Parquet."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol, runtime_checkable

from pacsys.types import Reading


@runtime_checkable
class LogWriter(Protocol):
    """Protocol for DataLogger writers."""

    def write_readings(self, readings: list[Reading]) -> None: ...
    def close(self) -> None: ...


class CsvWriter:
    """Write readings to a CSV file.

    Columns: timestamp, drf, value, units
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "drf", "value", "units"])

    def write_readings(self, readings: list[Reading]) -> None:
        for r in readings:
            self._writer.writerow(
                [
                    r.timestamp.isoformat() if r.timestamp else "",
                    r.drf,
                    r.value,
                    r.units or "",
                ]
            )

    def close(self) -> None:
        self._file.close()


class ParquetWriter:
    """Write readings to a Parquet file incrementally. Requires pyarrow."""

    def __init__(self, path: str | Path):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for ParquetWriter. Install with: pip install pyarrow")
        self._path = Path(path)
        self._pa = pa
        self._pq = pq
        self._schema = pa.schema(
            [
                ("timestamp", pa.string()),
                ("drf", pa.string()),
                ("value", pa.string()),
                ("units", pa.string()),
            ]
        )
        self._writer: pq.ParquetWriter | None = None

    def write_readings(self, readings: list[Reading]) -> None:
        if not readings:
            return
        rows: dict[str, list[str]] = {
            "timestamp": [],
            "drf": [],
            "value": [],
            "units": [],
        }
        for r in readings:
            rows["timestamp"].append(r.timestamp.isoformat() if r.timestamp else "")
            rows["drf"].append(r.drf)
            rows["value"].append(str(r.value))
            rows["units"].append(r.units or "")
        batch = self._pa.table(rows, schema=self._schema)
        if self._writer is None:
            self._writer = self._pq.ParquetWriter(self._path, self._schema)
        self._writer.write_table(batch)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
