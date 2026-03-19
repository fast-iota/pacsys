"""Reading/WriteResult to JSON-safe dict conversion for MCP tool output.

These produce compact, human-friendly dicts (with ``ok``, ``name``, ``error``
convenience keys). For round-trippable serialization use ``Reading.to_dict()``
/ ``Reading.from_dict()`` directly.
"""

from pacsys.types import Reading, WriteResult, _value_to_json


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
    if reading.is_error:
        d["error"] = reading.message
    return d


def write_result_to_dict(result: WriteResult) -> dict:
    """Convert a WriteResult to a JSON-safe dict for MCP tool output."""
    d: dict = {
        "ok": result.ok,
        "drf": result.drf,
    }
    if result.message:
        d["message"] = result.message
    if not result.ok:
        d["error"] = result.message
    return d
