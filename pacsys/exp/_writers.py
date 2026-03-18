"""Log writers for DataLogger: CSV and Parquet."""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from pacsys.types import Reading, Value, ValueType


@runtime_checkable
class LogWriter(Protocol):
    """Protocol for DataLogger writers."""

    def write_readings(self, readings: list[Reading]) -> None: ...
    def close(self) -> None: ...


_ARRAY_TYPES = frozenset({ValueType.SCALAR_ARRAY})
_JSON_TYPES = frozenset({ValueType.TEXT_ARRAY, ValueType.ANALOG_ALARM, ValueType.DIGITAL_ALARM, ValueType.BASIC_STATUS})


def _arraylike_to_floats(v: Value) -> list[float]:
    """Convert ndarray or list to list[float], avoiding redundant float() on numpy tolist()."""
    if hasattr(v, "tolist"):
        return v.tolist()  # type: ignore[union-attr]
    return [float(x) for x in v]  # type: ignore[arg-type]


def _timed_array_to_json(v: Value) -> str:
    """Serialize a timed scalar array dict {"data": ndarray, "micros": ndarray} to JSON."""
    if not isinstance(v, dict):
        v = {"data": v}
    out = {}
    for k, arr in v.items():
        out[k] = arr.tolist() if hasattr(arr, "tolist") else list(arr)  # type: ignore[union-attr]
    return json.dumps(out)


def _format_value_str(r: Reading) -> str:
    """Format a reading's value as a parseable string (for CSV)."""
    if r.value is None:
        return ""
    if r.value_type == ValueType.SCALAR:
        return str(r.value)
    if r.value_type in _ARRAY_TYPES:
        return json.dumps(_arraylike_to_floats(r.value))
    if r.value_type == ValueType.TIMED_SCALAR_ARRAY:
        return _timed_array_to_json(r.value)
    if r.value_type == ValueType.TEXT:
        return str(r.value)
    if r.value_type == ValueType.RAW:
        raw = r.value if isinstance(r.value, (bytes, bytearray)) else str(r.value).encode()
        return base64.b64encode(raw).decode("ascii")
    if r.value_type in _JSON_TYPES:
        return json.dumps(r.value)
    return str(r.value)


class CsvWriter:
    """Write readings to a CSV file.

    Columns: timestamp, drf, value, units
    Values are serialized as parseable strings: scalars as-is, arrays as JSON
    lists, dicts as JSON objects, raw bytes as base64.
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
                    _format_value_str(r),
                    r.units or "",
                ]
            )

    def close(self) -> None:
        self._file.close()


def _convert_value(r: Reading) -> tuple[float | None, int | None, list[float] | None, str | None]:
    """Route a reading's value to the correct typed column.

    Returns (value, int_value, value_array, value_text).
    """
    if r.value is None or r.value_type is None:
        return None, None, None, None

    if r.value_type == ValueType.SCALAR:
        # int/bool (includes numpy integer via int subclass) → int64 column
        if isinstance(r.value, int):
            return None, int(r.value), None, None
        # float and any numpy floating type → float64 column
        return float(r.value), None, None, None  # type: ignore[arg-type]

    if r.value_type in _ARRAY_TYPES:
        return None, None, _arraylike_to_floats(r.value), None

    if r.value_type == ValueType.TIMED_SCALAR_ARRAY:
        return None, None, None, _timed_array_to_json(r.value)

    if r.value_type == ValueType.TEXT:
        return None, None, None, str(r.value)

    if r.value_type == ValueType.RAW:
        raw = r.value if isinstance(r.value, (bytes, bytearray)) else str(r.value).encode()
        return None, None, None, base64.b64encode(raw).decode("ascii")

    if r.value_type in _JSON_TYPES:
        return None, None, None, json.dumps(r.value)

    # Unknown type — fall back to string
    return None, None, None, str(r.value)


class ParquetWriter:
    """Write readings to a Parquet file with typed columns. Requires pyarrow.

    Schema:
        timestamp   - timestamp[us, UTC]
        drf         - string (dictionary-encoded)
        value_type  - string (dictionary-encoded)
        value       - float64 (float scalars)
        int_value   - int64 (int/bool scalars)
        value_array - list<float64> (arrays)
        value_text  - string (text, JSON, base64)
        error_code  - int16
        units       - string
        cycle       - int64
    """

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
                ("timestamp", pa.timestamp("us", tz="UTC")),
                ("drf", pa.string()),
                ("value_type", pa.string()),
                ("value", pa.float64()),
                ("int_value", pa.int64()),
                ("value_array", pa.list_(pa.float64())),
                ("value_text", pa.string()),
                ("error_code", pa.int16()),
                ("units", pa.string()),
                ("cycle", pa.int64()),
            ]
        )
        self._writer: pq.ParquetWriter | None = None

    def write_readings(self, readings: list[Reading]) -> None:
        if not readings:
            return
        cols: dict[str, list] = {
            "timestamp": [],
            "drf": [],
            "value_type": [],
            "value": [],
            "int_value": [],
            "value_array": [],
            "value_text": [],
            "error_code": [],
            "units": [],
            "cycle": [],
        }
        for r in readings:
            cols["timestamp"].append(r.timestamp)
            cols["drf"].append(r.drf)
            cols["value_type"].append(r.value_type.value if r.value_type else None)
            v, iv, va, vt = _convert_value(r)
            cols["value"].append(v)
            cols["int_value"].append(iv)
            cols["value_array"].append(va)
            cols["value_text"].append(vt)
            cols["error_code"].append(r.error_code)
            cols["units"].append(r.units)
            cols["cycle"].append(r.cycle)
        batch = self._pa.table(cols, schema=self._schema)
        if self._writer is None:
            self._writer = self._pq.ParquetWriter(self._path, self._schema, compression="zstd")
        self._writer.write_table(batch)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
