"""Reading/WriteResult to JSON-safe dict conversion."""

import base64

from pacsys.types import Reading, WriteResult


def _json_value(value):
    """Convert a pacsys Value to a JSON-serializable Python object."""
    if value is None:
        return None
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value


def reading_to_dict(reading: Reading) -> dict:
    """Convert a Reading to a JSON-safe dict for MCP tool output."""
    d: dict = {
        "ok": reading.ok,
        "name": reading.name,
        "drf": reading.drf,
        "value": _json_value(reading.value),
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
