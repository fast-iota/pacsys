"""
DRF string utilities using proper parsing.

These functions provide safe DRF manipulation using the drf3 parser
instead of fragile string splitting.
"""

from pacsys.drf3 import parse_request
from pacsys.drf3.event import NeverEvent, parse_event


def ensure_immediate_event(drf: str) -> str:
    """Ensure DRF has immediate event (@I) if no event specified.

    Args:
        drf: DRF string (e.g., "M:OUTTMP" or "M:OUTTMP@p,1000")

    Returns:
        DRF with @I appended if no event was present.
        Preserves the original form (doesn't canonicalize).
    """
    request = parse_request(drf)
    if request.event is None or request.event.mode == "U":
        # Append @I to original string (preserves original format)
        return f"{drf}@I"
    return drf


def get_device_name(drf: str) -> str:
    """Extract just the device name from a DRF string.

    Args:
        drf: Full DRF string (e.g., "M:OUTTMP.SETTING[0:10]@p,1000")

    Returns:
        Device name only (e.g., "M:OUTTMP")
    """
    request = parse_request(drf)
    return request.device


def replace_event(drf: str, event_str: str) -> str:
    """Replace or add event in a DRF string.

    Args:
        drf: DRF string
        event_str: New event (e.g., "p,1000", "I", "E,0F")

    Returns:
        DRF with new event
    """
    request = parse_request(drf)
    new_event = parse_event(event_str)
    return request.to_canonical(event=new_event)


def strip_event(drf: str) -> str:
    """Remove event from DRF string.

    Args:
        drf: DRF string with or without event

    Returns:
        DRF without event portion
    """
    request = parse_request(drf)
    # Build DRF manually without event
    out = request.device
    if request.property is not None:
        out += f".{request.property.name}"
    if request.range is not None:
        out += str(request.range)
    if request.field is not None and request.property is not None:
        from pacsys.drf3.field import DEFAULT_FIELD_FOR_PROPERTY

        if DEFAULT_FIELD_FOR_PROPERTY.get(request.property) != request.field:
            out += f".{request.field.name}"
    return out


def has_event(drf: str) -> bool:
    """Check if DRF has an explicit event.

    Args:
        drf: DRF string

    Returns:
        True if DRF has an event specified (not default 'U')
    """
    request = parse_request(drf)
    return request.event is not None and request.event.mode != "U"


def has_explicit_property(drf: str) -> bool:
    """Check if DRF has an explicit property (e.g., .SETTING, .READING).

    Qualifier characters (M_OUTTMP) are NOT considered explicit — only
    dotted property names (.SETTING, .READING, etc.) count.

    Args:
        drf: DRF string

    Returns:
        True if DRF has an explicit property specified
    """
    request = parse_request(drf)
    return request.property_explicit


def is_setting_property(drf: str) -> bool:
    """Check if DRF is for the SETTING property.

    Args:
        drf: DRF string

    Returns:
        True if DRF's property is SETTING (explicit or via qualifier char)
    """
    from pacsys.drf3.property import DRF_PROPERTY

    request = parse_request(drf)
    return request.property == DRF_PROPERTY.SETTING


def prepare_for_write(drf: str) -> str:
    """Prepare a DRF string for write operations.

    Converts properties to their writable counterparts and forces @N (never)
    event — writes should never request data back.

    Property conversions:
        READING     -> SETTING   (read value -> write setpoint)
        STATUS      -> CONTROL   (read status -> write control bits)
        SETTING     -> preserved
        CONTROL     -> preserved
        ANALOG/DIGITAL -> preserved (alarm block writes)

    Args:
        drf: DRF string (any form: "Z:ACLTST", "Z_ACLTST", "Z:ACLTST.READING", etc.)

    Returns:
        DRF ready for write operation (e.g. "Z:ACLTST.SETTING@N")
    """
    from pacsys.drf3.property import DRF_PROPERTY

    request = parse_request(drf)

    # Map read properties to their writable counterparts
    _WRITE_PROPERTY = {
        DRF_PROPERTY.READING: DRF_PROPERTY.SETTING,
        DRF_PROPERTY.STATUS: DRF_PROPERTY.CONTROL,
    }
    new_property = _WRITE_PROPERTY.get(request.property, request.property)

    # Force @N — writes never need a response event
    return request.to_canonical(property=new_property, event=NeverEvent())
