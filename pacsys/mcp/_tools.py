"""MCP tool implementations — pure business logic, no MCP dependency."""

import logging
import threading

from pacsys.backends import Backend
from pacsys.drf_utils import get_device_name, prepare_for_write
from pacsys.supervised._audit import AuditLog
from pacsys.supervised._policies import (
    Policy,
    PolicyDecision,
    RequestContext,
    evaluate_policies,
)

from ._serialization import reading_to_dict, write_result_to_dict

logger = logging.getLogger("pacsys.mcp")


# Serializes policy check + write so stateful policies see no interleaving
_write_lock = threading.Lock()


def _audit_write(audit_log: AuditLog | None, ctx: RequestContext, decision: PolicyDecision) -> bool:
    """Record a write decision. Returns False if a configured audit log failed."""
    if audit_log is None:
        return True
    try:
        audit_log.log_request(ctx, decision)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("write_device audit failed for drf=%s", ctx.drfs[0])
        return False


def tool_read_device(backend: Backend, drf: str, policies: list[Policy]) -> dict:
    """Read a device value with policy enforcement. Returns a JSON-safe dict."""
    try:
        name = get_device_name(drf)  # single parse: a malformed DRF must not escape as a raw exception
    except ValueError as e:
        logger.warning("read_device rejected malformed drf=%r: %s", drf, e)
        return {"ok": False, "name": drf, "drf": drf, "value": None, "error": str(e)}
    final_drf = drf
    if policies:
        ctx = RequestContext(
            drfs=[drf],
            rpc_method="Read",
            peer="mcp-client",
            metadata={},
            values=[],
            raw_request=None,
            allowed=frozenset({0}),  # reads start approved (match supervised server)
        )
        try:
            decision = evaluate_policies(policies, ctx)
        except ValueError as e:
            logger.exception("read_device policy failed for drf=%s", drf)
            return {"ok": False, "name": name, "drf": drf, "value": None, "error": str(e)}
        if not decision.allowed:
            reason = decision.reason or "Read denied by policy"
            logger.warning("read_device drf=%s denied reason=%s", drf, reason)
            return {"ok": False, "name": name, "drf": drf, "value": None, "error": reason}
        assert decision.ctx is not None
        if 0 not in decision.ctx.allowed:
            reason = "Read denied by policy"
            logger.warning("read_device drf=%s denied reason=%s", drf, reason)
            return {"ok": False, "name": name, "drf": drf, "value": None, "error": reason}
        final_drf = decision.ctx.drfs[0]
    try:
        reading = backend.get(final_drf)
        return reading_to_dict(reading)
    except Exception as e:
        logger.exception("read_device failed for drf=%s", drf)
        return {"ok": False, "name": name, "drf": drf, "value": None, "error": str(e)}


def tool_write_device(
    backend: Backend,
    drf: str,
    value: float | str | list,
    policies: list[Policy],
    audit_log: AuditLog | None = None,
) -> dict:
    """Write a device value with policy enforcement. Returns a JSON-safe dict."""
    with _write_lock:
        return _write_device_locked(backend, drf, value, policies, audit_log)


def _write_device_locked(
    backend: Backend,
    drf: str,
    value: float | str | list,
    policies: list[Policy],
    audit_log: AuditLog | None,
) -> dict:
    try:
        write_drf = prepare_for_write(drf)
        device_name = get_device_name(write_drf)
    except ValueError as e:
        logger.warning("write_device rejected malformed drf=%r: %s", drf, e)
        raw_ctx = RequestContext(
            drfs=[drf], rpc_method="Set", peer="mcp-client", metadata={}, values=[(drf, value)], raw_request=None
        )
        _audit_write(audit_log, raw_ctx, PolicyDecision(allowed=False, reason=str(e)))
        return {"ok": False, "drf": drf, "error": str(e)}

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
    try:
        if policies:
            decision = evaluate_policies(policies, ctx)
        else:
            decision = PolicyDecision(allowed=True, ctx=ctx)
    except ValueError as e:
        logger.exception("write_device policy failed for drf=%s", write_drf)
        _audit_write(audit_log, ctx, PolicyDecision(allowed=False, reason=str(e)))
        return {"ok": False, "drf": write_drf, "error": str(e)}

    # Check if write was approved by any policy
    final_ctx = decision.ctx if decision.ctx is not None else ctx
    if not decision.allowed:
        reason = decision.reason or "Write denied by policy"
        _audit_write(audit_log, ctx, PolicyDecision(allowed=False, reason=reason))
        logger.warning("write_device drf=%s device=%s denied reason=%s", write_drf, device_name, reason)
        return {"ok": False, "drf": write_drf, "error": reason}

    unapproved = set(range(len(final_ctx.drfs))) - set(final_ctx.allowed)
    if unapproved:
        if not any(p.allows_writes for p in policies):
            reason = "No policy explicitly allows write operations"
        else:
            reason = f"No write policy approves: {device_name}"
        _audit_write(audit_log, ctx, PolicyDecision(allowed=False, reason=reason))
        logger.warning("write_device drf=%s device=%s denied reason=%s", write_drf, device_name, reason)
        return {"ok": False, "drf": write_drf, "error": reason}

    # Execute write - an unrecordable write is blocked, not executed
    final_drf, final_value = final_ctx.values[0]
    if not _audit_write(audit_log, final_ctx, PolicyDecision(allowed=True, ctx=final_ctx)):
        return {"ok": False, "drf": final_drf, "error": "Audit log failed; write blocked"}
    try:
        result = backend.write(final_drf, final_value)
        return write_result_to_dict(result)
    except Exception as e:
        logger.exception("write_device failed for drf=%s", final_drf)
        return {"ok": False, "drf": final_drf, "error": str(e)}


def tool_device_info(devdb, name: str) -> dict:
    """Query device metadata from DevDB. Returns a JSON-safe dict."""
    if devdb is None:
        return {"ok": False, "name": name, "error": "DevDB client unavailable"}

    try:
        info = devdb.get_device_info([name])[name]
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
        logger.exception("device_info failed for name=%s", name)
        return {"ok": False, "name": name, "error": str(e)}
