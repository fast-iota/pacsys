"""Reading/WriteResult to JSON-safe dict conversion for MCP tool output.

These produce compact, human-friendly dicts (with ``ok``, ``name``, ``error``
convenience keys). For round-trippable serialization use ``Reading.to_dict()``
/ ``Reading.from_dict()`` directly.
"""

from pacsys.types import Reading, WriteResult, _value_to_json


def _add_status(d: dict, status: Reading | WriteResult, failure: str) -> None:
    """Add non-success status codes, ``message`` (usable results) or ``error`` (failures)."""
    if status.error_code != 0:
        d["facility_code"] = status.facility_code
        d["error_code"] = status.error_code
    if status.ok:
        if status.message:
            d["message"] = status.message
    else:
        d["error"] = status.message or f"{failure} (facility={status.facility_code}, error={status.error_code})"


def reading_to_dict(reading: Reading) -> dict:
    """Convert a Reading to a JSON-safe dict for MCP tool output."""
    d: dict = {
        "ok": reading.ok,
        "name": reading.name,
        "drf": reading.drf,
        "value": _value_to_json(reading.value),
    }
    if reading.meta and reading.meta.units:
        d["units"] = reading.meta.units
    if reading.timestamp is not None:
        d["timestamp"] = reading.timestamp.isoformat()
    if reading.cycle is not None:
        d["cycle"] = reading.cycle
    _add_status(d, reading, "Read failed")
    return d


def write_result_to_dict(result: WriteResult) -> dict:
    """Convert a WriteResult to a JSON-safe dict for MCP tool output."""
    d: dict = {
        "ok": result.ok,
        "drf": result.drf,
    }
    if result.message:
        d["message"] = result.message
    _add_status(d, result, "Write failed")
    return d
