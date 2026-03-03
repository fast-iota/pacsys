"""MCP tool implementations — pure business logic, no MCP dependency."""

import logging

from pacsys.backends import Backend
from pacsys.drf_utils import get_device_name, prepare_for_write
from pacsys.supervised._policies import (
    Policy,
    PolicyDecision,
    RequestContext,
    evaluate_policies,
)

from ._serialization import reading_to_dict, write_result_to_dict

logger = logging.getLogger("pacsys.mcp")


def tool_read_device(backend: Backend, drf: str) -> dict:
    """Read a device value. Returns a JSON-safe dict."""
    try:
        reading = backend.get(drf)
        return reading_to_dict(reading)
    except Exception as e:
        logger.error("read_device drf=%s error=%s", drf, e, exc_info=True)
        return {"ok": False, "name": get_device_name(drf), "drf": drf, "value": None, "error": str(e)}


def tool_write_device(
    backend: Backend,
    drf: str,
    value: float | str | list,
    policies: list[Policy],
    audit_log=None,
) -> dict:
    """Write a device value with policy enforcement. Returns a JSON-safe dict."""
    write_drf = prepare_for_write(drf)
    device_name = get_device_name(drf)

    # Build request context for policy evaluation
    ctx = RequestContext(
        drfs=[write_drf],
        rpc_method="Set",
        peer="mcp-client",
        metadata={},
        values=[(write_drf, value)],
        raw_request=None,
        allowed=frozenset(),  # writes start unapproved
    )

    # Evaluate policies
    if policies:
        decision = evaluate_policies(policies, ctx)
    else:
        decision = PolicyDecision(allowed=True, ctx=ctx)

    # Check if write was approved by any policy
    final_ctx = decision.ctx if decision.ctx is not None else ctx
    if not decision.allowed:
        reason = decision.reason or "Write denied by policy"
        logger.warning("write_device drf=%s device=%s denied reason=%s", write_drf, device_name, reason)
        return {"ok": False, "drf": write_drf, "error": reason}

    unapproved = set(range(len(final_ctx.drfs))) - set(final_ctx.allowed)
    if unapproved:
        if not any(p.allows_writes for p in policies):
            reason = "No policy explicitly allows write operations"
        else:
            reason = f"No write policy approves: {device_name}"
        logger.warning("write_device drf=%s device=%s denied reason=%s", write_drf, device_name, reason)
        return {"ok": False, "drf": write_drf, "error": reason}

    # Execute write
    try:
        result = backend.write(write_drf, value)
        return write_result_to_dict(result)
    except Exception as e:
        logger.error("write_device drf=%s error=%s", write_drf, e, exc_info=True)
        return {"ok": False, "drf": write_drf, "error": str(e)}


def tool_device_info(devdb, name: str) -> dict:
    """Query device metadata from DevDB. Returns a JSON-safe dict."""
    if devdb is None:
        return {"ok": False, "name": name, "error": "DevDB client unavailable"}

    try:
        info_map = devdb.get_device_info([name])
        info = info_map.get(name)
        if info is None:
            return {"ok": False, "name": name, "error": f"Device {name} not found in DevDB"}

        d: dict = {
            "ok": True,
            "name": name,
            "description": info.description,
            "device_index": info.device_index,
        }

        if info.reading:
            d["reading"] = {
                "units": info.reading.primary_units,
                "common_units": info.reading.common_units,
                "min": info.reading.min_val,
                "max": info.reading.max_val,
            }

        if info.setting:
            d["setting"] = {
                "units": info.setting.primary_units,
                "common_units": info.setting.common_units,
                "min": info.setting.min_val,
                "max": info.setting.max_val,
            }

        if info.control:
            d["control_commands"] = [
                {"value": cmd.value, "short_name": cmd.short_name, "long_name": cmd.long_name} for cmd in info.control
            ]

        return d
    except Exception as e:
        logger.error("device_info name=%s error=%s", name, e, exc_info=True)
        return {"ok": False, "name": name, "error": str(e)}
