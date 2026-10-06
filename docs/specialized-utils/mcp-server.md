# MCP Server

The MCP server exposes pacsys device read/write as tools for AI agents (Claude Code, etc.) via the [Model Context Protocol](https://modelcontextprotocol.io/).

Install with `pip install "pacsys[mcp]"` (uses the official MCP Python SDK).

## Overview

```
[Claude Code] ──MCP stdio──> [pacsys.mcp] ──DPM HTTP──> [ACNET]
                                   │
                              policies + HITL
```

Three tools are exposed:

| Tool | Description |
|------|-------------|
| `read_device(drf)` | Read a device value |
| `write_device(drf, value)` | Write with policy enforcement |
| `device_info(name)` | Look up device metadata from DevDB |

Write safety comes from two layers:

1. **Claude Code's tool permission prompt** — human-in-the-loop approval for each write call
2. **Server-side policy chain** — `DeviceAccessPolicy` → `ValueRangePolicy`

Without a policy config, all writes are denied. Reads are always allowed; the config only expresses write policies. This is the same policy system used by [Supervised Mode](supervised.md); for read restrictions, run the supervised proxy instead.

All three tools first apply the supervised proxy's [DRF gate](supervised.md#access-control-defaults): malformed DRFs, device-index aliases (`0:1234`, `#:1234`), and `#` list directives are rejected with `"ok": false`. Rejected writes are audited like any other denial.

Writes require Kerberos credentials. The server refuses to start if write devices are configured but no Kerberos ticket is available.

---

## Quick Start

### Read-only (no config needed)

Register the server without a policy config:

```bash
claude mcp add --transport stdio --scope project pacsys -- python -m pacsys.mcp
```

### With write access

```bash
claude --mcp-config scripts/.mcp_prod.json
```

Where `scripts/.mcp_prod.json` points to a TOML config:

```json
{
  "mcpServers": {
    "pacsys": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "pacsys.mcp", "--config", "scripts/pacsys-mcp.toml"],
      "env": {}
    }
  }
}
```

---

## Configuration

### TOML config file

```toml
[server]
transport = "stdio"    # "stdio" or "sse"
# port = 8000          # only used with transport = "sse"
# role = "testing"     # DPM role for access control
# audit_log = "audit.jsonl"

[policies]
# Devices allowed for writing (glob patterns).
# Without this list, ALL writes are denied.
write_devices = ["Z:ACLTST", "Z:CUBE_Z"]
# Raw writes to range-limited devices are denied unless explicitly exempted.
allow_raw = ["B:HS*"]

# Value range limits per device pattern.
[policies.value_ranges]
"Z:ACLTST" = [0.0, 100.0]
```

A sample config is provided at `scripts/pacsys-mcp.toml`.

Configuration is validated strictly at startup: unknown keys, malformed policies, and
options that don't fit the transport (such as `port` with `stdio`) stop the server.

When `audit_log` is configured, every write attempt is recorded as JSON Lines with its
DRF, value, decision, and reason. An unwritable audit log stops the server at startup,
and a write that cannot be recorded is blocked.

### CLI flags

CLI flags override values from the config file:

```bash
python -m pacsys.mcp --config pacsys-mcp.toml --transport sse --port 9090 --role testing --debug
```

| Flag | Description |
|------|-------------|
| `--config PATH` | TOML config file |
| `--transport` | `stdio` (default) or `sse` |
| `--port` | Port for SSE transport |
| `--role` | DPM role for access control |
| `--debug` | Enable debug logging |

The SSE port defaults to 8000. When calling `create_server(config)` directly, pass the
transport to `run()`: `server.run()` for stdio or `server.run(transport="sse", port=9090)`
for SSE (`config.port` is not applied).

### Environment variables

The server respects standard pacsys environment variables:

| Variable | Description |
|----------|-------------|
| `PACSYS_DPM_HOST` | DPM proxy hostname (default: `acsys-proxy.fnal.gov`) |
| `PACSYS_DPM_PORT` | DPM proxy port (default: `6802`) |

These can be set in the MCP config JSON:

```json
{
  "mcpServers": {
    "pacsys": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "pacsys.mcp"],
      "env": {
        "PACSYS_DPM_HOST": "my-proxy.fnal.gov"
      }
    }
  }
}
```

---

## Tools

### read_device

Read a device value. Accepts any DRF string that passes the DRF gate.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `drf` | `str` | DRF string (e.g. `M:OUTTMP`, `M:OUTTMP.SETTING`, `M:OUTTMP[0:9]`) |

**Returns:**

```json
{
  "ok": true,
  "name": "M:OUTTMP",
  "drf": "M:OUTTMP",
  "value": 72.5,
  "units": "deg F",
  "timestamp": "2026-03-03T14:00:00+00:00",
  "cycle": 0
}
```

On error:

```json
{
  "ok": false,
  "name": "M:BADDEV",
  "drf": "M:BADDEV",
  "value": null,
  "facility_code": 16,
  "error_code": -42,
  "error": "DIO_NO_SUCH - device not found"
}
```

Any non-zero status adds `facility_code` and `error_code`. A warning with usable data stays `"ok": true` with its text in `message`.

### write_device

Write a value to a device. Requires policy approval.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `drf` | `str` | Device DRF string |
| `value` | `float`, `str`, or `list` | Value to write |

**Returns:**

```json
{"ok": true, "drf": "Z:ACLTST.SETTING@N"}
```

On denial:

```json
{"ok": false, "drf": "Z:ACLTST.SETTING@N", "error": "Value 200.0 for Z:ACLTST outside range [0.0, 100.0]"}
```

A backend write failure also includes the status codes:

```json
{"ok": false, "drf": "Z:ACLTST.SETTING@N", "facility_code": 17, "error_code": -44, "error": "Write failed (facility=17, error=-44)"}
```

### device_info

Look up device metadata from the device database.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Device name (e.g. `M:OUTTMP`) |

**Returns:**

```json
{
  "ok": true,
  "name": "M:OUTTMP",
  "description": "Outside temperature",
  "device_index": 12345,
  "reading": {"units": "deg F", "common_units": "deg F", "min": -40.0, "max": 140.0},
  "setting": {"units": "deg F", "common_units": "deg F", "min": 0.0, "max": 100.0},
  "control_commands": [{"value": 0, "short_name": "OFF", "long_name": "Turn off"}]
}
```

DevDB must be available (requires `grpcio`). If unavailable, returns `{"ok": false, "error": "DevDB client unavailable"}`.

---

## See Also

- [Supervised Mode](supervised.md) — the policy system used by the MCP server
- [Writing Guide](../guide/writing.md) — write operations and authentication
- [DRF Format](../drf.md) — DRF string syntax for device addressing
