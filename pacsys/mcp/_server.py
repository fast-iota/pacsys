"""FastMCP server setup — tool registration and backend lifecycle."""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial

from anyio.to_thread import run_sync
from mcp.server.fastmcp import FastMCP

from pacsys.backends import Backend
from pacsys.supervised._audit import AuditLog
from pacsys.supervised._policies import Policy

from ._config import MCPConfig, build_policies
from ._tools import tool_device_info, tool_read_device, tool_write_device

logger = logging.getLogger("pacsys.mcp")


@dataclass(frozen=True)
class ServerContext:
    """Resources available during server lifetime."""

    backend: Backend
    devdb: object | None  # DevDBClient or None
    policies: list[Policy]
    audit_log: AuditLog | None = None


@asynccontextmanager
async def _lifespan(server: FastMCP, *, config: MCPConfig):
    """Manage backend and devdb lifecycle."""
    from pacsys.auth import KerberosAuth
    from pacsys.backends.dpm_http import DPMHTTPBackend
    from pacsys.errors import AuthenticationError

    config = config.finalized()
    policies = build_policies(config)
    has_write_policies = any(p.allows_writes for p in policies)

    auth = None
    try:
        auth = KerberosAuth()
        logger.info("Kerberos authenticated as %s", auth.principal)
    except (ImportError, AuthenticationError) as e:
        if has_write_policies:
            raise RuntimeError(f"Write devices configured but Kerberos unavailable: {e}") from e
        logger.warning("No Kerberos ticket — running in read-only mode")

    # Respect PACSYS_* env vars (same as pacsys.configure())
    kwargs: dict = {"timeout": 5.0}
    host = os.environ.get("PACSYS_DPM_HOST")
    if host:
        kwargs["host"] = host
    port = os.environ.get("PACSYS_DPM_PORT")
    if port:
        kwargs["port"] = int(port)
    if auth is not None:
        kwargs["auth"] = auth
    if config.role is not None:
        kwargs["role"] = config.role
    backend = DPMHTTPBackend(**kwargs)

    devdb = None
    try:
        from pacsys.devdb import DEVDB_AVAILABLE, DevDBClient

        if DEVDB_AVAILABLE:
            devdb = DevDBClient()
            logger.info("DevDB client connected")
    except Exception as e:  # noqa: BLE001
        logger.warning("DevDB unavailable: %s", e)

    audit_log = None
    try:
        if config.audit_log is not None:
            audit_log = AuditLog(config.audit_log)
        yield ServerContext(backend=backend, devdb=devdb, policies=policies, audit_log=audit_log)
    finally:
        try:
            backend.close()
        finally:
            try:
                if devdb is not None:
                    devdb.close()
            finally:
                if audit_log is not None:
                    audit_log.close()
        logger.info("MCP server resources cleaned up")


def create_server(config: MCPConfig) -> FastMCP:
    """Create a configured FastMCP server instance."""
    config = config.finalized()
    build_policies(config)  # fail before accepting a client
    lifespan = partial(_lifespan, config=config)
    mcp = FastMCP(
        "pacsys",
        instructions=(
            "ACNET control system device interface. "
            "Use read_device to read values, write_device to set values, "
            "device_info to look up metadata. "
            "Writes require policy approval — denied by default unless configured."
        ),
        lifespan=lifespan,
        port=config.port or 8000,
    )

    @mcp.tool(
        description="Read a device value. Pass a DRF string like 'M:OUTTMP' or 'M:OUTTMP.SETTING' or 'M:OUTTMP[0:9]'."
    )
    async def read_device(drf: str) -> dict:
        ctx: ServerContext = mcp.get_context().request_context.lifespan_context
        return await run_sync(tool_read_device, ctx.backend, drf, ctx.policies)

    @mcp.tool(
        description="Write a value to a device. Requires policy approval. Pass DRF and value (float, string, or list)."
    )
    async def write_device(drf: str, value: float | str | list) -> dict:
        ctx: ServerContext = mcp.get_context().request_context.lifespan_context
        return await run_sync(tool_write_device, ctx.backend, drf, value, ctx.policies, ctx.audit_log)

    @mcp.tool(
        description="Look up device metadata (description, units, limits, control commands) from the device database."
    )
    async def device_info(name: str) -> dict:
        ctx: ServerContext = mcp.get_context().request_context.lifespan_context
        return await run_sync(tool_device_info, ctx.devdb, name)

    return mcp
