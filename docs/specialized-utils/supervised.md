# Supervised Mode

The `SupervisedServer` is a DPM/gRPC server that wraps any Backend, forwarding requests while enforcing policies and logging traffic.

## Overview

```
[gRPC Client] ──DAQ stub──> [SupervisedServer] ──Backend API──> [Any Backend]
                                   │
                              policies + logging
```

Use cases:

- **Testing** -- expose a `FakeBackend` as a real gRPC server for integration tests
- **Digital twins** -- connect to arbitrary data sources, similarly to EPICS soft IOC
- **Access control** -- restrict which operations are allowed, apply value/slew/rate limits, etc.
- **Audit logging** -- log client info, timing, data, and policy decisions
- **Custom logic** -- MCR killswitch, status GUI, etc.

---

## Access Control Defaults

Reads are **allowed by default** — any client can read any device without explicit policy approval.

Writes (Set RPCs) are **denied by default** — every write must be explicitly approved by a `DeviceAccessPolicy` with `mode="allow"` covering the `"set"` (or `"all"`) action. Without such a policy, all writes return `PERMISSION_DENIED`.

This means:
- A server with no policies allows all reads and denies all writes
- `ReadOnlyPolicy` is still useful for explicit intent but is now optional — writes are denied regardless
- Policies like `RateLimitPolicy` or `ValueRangePolicy` do not unlock writes — they only constrain already-approved writes

## Quick Start

```python
from pacsys.testing import FakeBackend
from pacsys.supervised import SupervisedServer, DeviceAccessPolicy
import pacsys

fb = FakeBackend()
fb.set_reading("M:OUTTMP", 72.5)

# Reads work by default; writes require explicit approval
with SupervisedServer(fb, port=50099, policies=[
    DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow"),
]) as srv:
    with pacsys.grpc(host="localhost", port=50099) as client:
        print(client.read("M:OUTTMP"))      # 72.5
        client.write("M:OUTTMP", 80.0)      # OK (M:* approved)
        client.write("Z:SECRET", 1.0)        # PERMISSION_DENIED
```

---

## SupervisedServer

```python
SupervisedServer(backend, port=50051, host="[::]", policies=None, token=None, audit_log=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `Backend \| AsyncBackend` | *(required)* | Any backend instance to proxy |
| `port` | `int` | `50051` | Port to listen on (use `0` for OS-assigned) |
| `host` | `str` | `[::]` | Bind address |
| `policies` | `list[Policy]` | `None` | Policy chain for access control |
| `token` | `str \| None` | `None` | Bearer token for write authentication. When set, clients must pass `JWTAuth(token=...)` with this value or write (`Set`) RPCs are rejected with `UNAUTHENTICATED`. Reads are always open. |
| `audit_log` | `AuditLog \| None` | `None` | Structured audit log instance (see [AuditLog](#auditlog)) |

### Lifecycle

```python
# Context manager (recommended)
with SupervisedServer(backend, port=0) as srv:
    print(srv.port)  # actual port if 0 was used

# Manual start/stop
srv = SupervisedServer(backend, port=50051)
srv.start()
# ... use server ...
srv.stop()

# Blocking mode (main thread, handles SIGINT/SIGTERM)
srv = SupervisedServer(backend, port=50051)
srv.run()  # blocks until signal received
```

| Method | Description |
|--------|-------------|
| `start()` | Start server in background daemon thread |
| `stop()` | Stop server and join thread |
| `wait(timeout)` | Block until server stops |
| `run()` | Start and block until SIGINT/SIGTERM (main thread only) |
| `port` | Actual port (useful when `port=0`) |

---

## Policies

Policies are evaluated as a middleware chain. Each policy can inspect, deny, or modify the request. The first denial short-circuits -- remaining policies are skipped. On allow, each policy may return a modified `RequestContext` that subsequent policies (and the final backend call) will see.

**Default behavior:** Reads are allowed; writes require explicit approval via `DeviceAccessPolicy` with `mode="allow"` covering the `"set"` action (see [Access Control Defaults](#access-control-defaults)).

### ReadOnlyPolicy

Blocks all write (`Set`) operations, allows reads.

```python
from pacsys.supervised import ReadOnlyPolicy

policies = [ReadOnlyPolicy()]
```

### DeviceAccessPolicy

Allow or deny access based on device name patterns. In `mode="allow"`, matching devices are **approved** for the operation (non-matching devices are left unapproved, not denied). In `mode="deny"`, matching devices are blocked outright. The `action` parameter controls which RPC types the policy applies to.

```python
from pacsys.supervised import DeviceAccessPolicy

# Approve writes for M: and G: devices
policies = [DeviceAccessPolicy(patterns=["M:*", "G:*"], action="set", mode="allow")]

# Block specific devices from all operations
policies = [DeviceAccessPolicy(patterns=["Z:SECRET*"], mode="deny")]

# Approve writes for M: devices, deny reads from Z: devices
policies = [
    DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow"),
    DeviceAccessPolicy(patterns=["Z:*"], action="read", mode="deny"),
]

# Regex syntax for more complex matching
policies = [DeviceAccessPolicy(patterns=[r"M:OUT.*", r"G:AMANDA"], action="set", syntax="regex")]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patterns` | `list[str]` | *(required)* | Patterns against device names |
| `mode` | `str` | `"allow"` | `"allow"` = approve matching devices, `"deny"` = block matching devices |
| `action` | `str` | `"all"` | `"all"` = both Read and Set, `"read"` = Read only, `"set"` = Set only |
| `syntax` | `str` | `"glob"` | `"glob"` (fnmatch) or `"regex"` (full-match, case-insensitive) |

**Per-slot approval:** In `mode="allow"`, the policy tracks which request slots (device indices) it approves. Multiple `DeviceAccessPolicy` instances compose — each adds its approved slots. After the full policy chain, any unapproved slots cause `PERMISSION_DENIED`.

### RateLimitPolicy

Sliding window rate limit per client peer address.

```python
from pacsys.supervised import RateLimitPolicy

# Max 100 requests per 60 seconds per client
policies = [RateLimitPolicy(max_requests=100, window_seconds=60)]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_requests` | `int` | *(required)* | Max requests per window |
| `window_seconds` | `float` | `60.0` | Window size in seconds |

### ValueRangePolicy

Deny writes where numeric values fall outside allowed ranges. Non-numeric values and unmatched devices are passed through.

```python
from pacsys.supervised import ValueRangePolicy

# Limit M: devices to [0, 100], G: devices to [-50, 50]
policies = [ValueRangePolicy(limits={"M:*": (0.0, 100.0), "G:*": (-50.0, 50.0)})]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limits` | `dict[str, tuple[float, float]]` | *(required)* | Glob pattern to (min, max) bounds |

### SlewRatePolicy

Enforce maximum step size and/or rate of change per device. Stateful -- tracks the last written value and timestamp. First write to any device is always allowed. Accepts that failed backend writes will leave stale history.

Each device pattern maps to a `SlewLimit(max_step=..., max_rate=...)`. At least one must be set; both can be combined.

```python
from pacsys.supervised import SlewRatePolicy, SlewLimit

# Max 10 units per write from last one (absolute step)
policies = [SlewRatePolicy(limits={"M:*": SlewLimit(max_step=10.0)})]

# Max 5 units/second (rate)
policies = [SlewRatePolicy(limits={"M:*": SlewLimit(max_rate=5.0)})]

# Both: max 10 units per step AND max 5 units/second
policies = [SlewRatePolicy(limits={"M:*": SlewLimit(max_step=10.0, max_rate=5.0)})]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limits` | `dict[str, SlewLimit]` | *(required)* | Glob pattern to slew constraints |

`SlewLimit` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_step` | `float \| None` | `None` | Max absolute change per write |
| `max_rate` | `float \| None` | `None` | Max units/second |

### AuditLog

Structured audit log that writes JSON lines and optionally tagged length-delimited binary protobuf. Not a `Policy` — passed as a separate `audit_log=` parameter to `SupervisedServer`. Logs both allowed and denied requests. Called automatically by the server after each policy decision.

Two modes controlled by `log_responses`:

- `False` (default): one `"in"` JSON entry + request protobuf per RPC.
- `True`: `"in"` entry per request AND `"out"` entry per response protobuf.

```python
from pacsys.supervised import AuditLog, SupervisedServer

# Request-only logging (JSON lines)
audit = AuditLog("audit.jsonl")

# Full request+response logging with binary protobuf capture
audit = AuditLog(
    "audit.jsonl",
    proto_path="audit.binpb",
    log_responses=True,
    flush_interval=50,
)

with SupervisedServer(backend, port=50051, audit_log=audit) as srv:
    srv.wait()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | *(required)* | JSON lines file path |
| `proto_path` | `str \| None` | `None` | Binary protobuf file path to store complete raw packets (optional) |
| `log_responses` | `bool` | `False` | Log outgoing responses too |
| `flush_interval` | `int` | `1` | Flush files every N writes |

**JSON schema — request (`dir: "in"`):**

```json
{"ts": "2026-02-15T14:30:01.123456+00:00", "seq": 42, "dir": "in", "peer": "ipv4:192.168.1.5:43210", "method": "Set", "drfs": ["M:OUTTMP@e,01"], "allowed": true, "reason": null}
```

**JSON schema — response (`dir: "out"`, only when `log_responses=True`):**

```json
{"ts": "2026-02-15T14:30:01.135456+00:00", "seq": 42, "dir": "out", "peer": "ipv4:192.168.1.5:43210", "method": "Set"}
```

**Binary protobuf framing:** `tag_byte + varint_length + serialized_bytes`. Tags identify message type:

| Tag | Message type |
|-----|-------------|
| `0x00` | `ReadRequest` |
| `0x01` | `ReadReply` |
| `0x02` | `SettingRequest` |
| `0x03` | `SettingReply` |

The server calls `close()` automatically on `stop()`.

### Combining Policies

Policies compose naturally -- stack them in order of priority:

```python
from pacsys.supervised import (
    SupervisedServer, DeviceAccessPolicy,
    RateLimitPolicy, ValueRangePolicy, SlewRatePolicy, SlewLimit,
    AuditLog,
)

audit = AuditLog("audit.jsonl", proto_path="audit.binpb", log_responses=True)

policies = [
    DeviceAccessPolicy(patterns=["M:*", "G:*"], action="set", mode="allow"),  # approve writes for M: and G:
    DeviceAccessPolicy(patterns=["Z:*"], mode="deny"),                         # block Z: from all operations
    RateLimitPolicy(max_requests=200, window_seconds=60),                      # throttle per client
    ValueRangePolicy(limits={"M:*": (0.0, 100.0)}),                           # safe range for M:
    SlewRatePolicy(limits={"M:*": SlewLimit(max_step=10.0, max_rate=5.0)}),
]

with SupervisedServer(backend, port=50051, policies=policies, audit_log=audit) as srv:
    srv.wait()
```

### Custom Policies

Subclass `Policy` and implement `check()`:

```python
from pacsys.supervised import Policy, PolicyDecision, RequestContext

class BusinessHoursPolicy(Policy):
    """Only allow access during business hours."""

    def check(self, ctx: RequestContext) -> PolicyDecision:
        from datetime import datetime
        hour = datetime.now().hour
        if 8 <= hour < 17:
            return PolicyDecision(allowed=True)
        return PolicyDecision(allowed=False, reason="Outside business hours (8-17)")
```

`RequestContext` fields:

| Field | Type | Description |
|-------|------|-------------|
| `drfs` | `list[str]` | DRF strings in the request |
| `rpc_method` | `str` | `"Read"` or `"Set"` |
| `peer` | `str` | Client address |
| `metadata` | `dict[str, str]` | gRPC metadata from the call |
| `values` | `list[tuple[str, object]]` | `[(DRF, value), ...]` — preserves order and duplicates (empty for reads) |
| `raw_request` | `object` | Raw protobuf request message |
| `allowed` | `frozenset[int]` | Slot indices approved for this operation (all for reads, empty for sets initially) |

`PolicyDecision` fields:

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | `bool` | Whether the request is allowed |
| `reason` | `str \| None` | Required when denied |
| `ctx` | `RequestContext \| None` | Modified context (None = no change) |

**`allows_writes` property:** Override this property to return `True` if your custom policy explicitly gates write access. The server uses this to generate clearer error messages when writes are denied.

```python
class MyWriteGatePolicy(Policy):
    @property
    def allows_writes(self) -> bool:
        return True  # tells the server this policy gates writes

    def check(self, ctx: RequestContext) -> PolicyDecision:
        ...
```

### Request Modification

Policies can modify the request by returning a new `RequestContext` in the `ctx` field of `PolicyDecision`. Use `dataclasses.replace()` to create modified copies — this preserves all fields including `allowed`.

```python
from dataclasses import replace

class ClampPolicy(Policy):
    """Clamp write values to [0, 100]."""

    def check(self, ctx: RequestContext) -> PolicyDecision:
        if ctx.rpc_method != "Set":
            return PolicyDecision(allowed=True)
        new_values = [
            (drf, max(0.0, min(100.0, val)) if isinstance(val, (int, float)) else val)
            for drf, val in ctx.values
        ]
        return PolicyDecision(allowed=True, ctx=replace(ctx, values=new_values))
```

```python
from dataclasses import replace

class RedirectPolicy(Policy):
    """Redirect devices matching a prefix to a different prefix.

    Example: route T:OUTTMP (test namespace) to M:OUTTMP (production).
    Only rewrites ctx.drfs — the server uses drfs as the authoritative
    target for both reads and writes, so values need not be touched.
    """

    def __init__(self, from_prefix: str, to_prefix: str):
        self._from = from_prefix.upper()
        self._to = to_prefix.upper()

    def _rewrite(self, drf: str) -> str:
        name = get_device_name(drf)
        if name.upper().startswith(self._from):
            return drf.replace(name, self._to + name[len(self._from):], 1)
        return drf

    def check(self, ctx: RequestContext) -> PolicyDecision:
        new_drfs = [self._rewrite(d) for d in ctx.drfs]
        if new_drfs == ctx.drfs:
            return PolicyDecision(allowed=True)
        return PolicyDecision(allowed=True, ctx=replace(ctx, drfs=new_drfs))

# Route T: (test) devices to M: (production)
policies = [RedirectPolicy("T:", "M:")]
```

---

## Logging

All requests are logged to the `pacsys.supervised` logger:

```
INFO  rpc=Read peer=ipv4:127.0.0.1:54321 devices=M:OUTTMP, G:AMANDA decision=allowed
INFO  rpc=Read peer=ipv4:127.0.0.1:54321 elapsed_ms=12.3 items=2
WARN  rpc=Set  peer=ipv4:127.0.0.1:54321 devices=M:OUTTMP decision=denied reason=Write operations disabled
```

Enable debug logging for streaming lifecycle events:

```python
import logging
logging.getLogger("pacsys.supervised").setLevel(logging.DEBUG)
```

Write logs to a rotating set of files (10 MB each, keep 5 backups):

```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "supervised.log", maxBytes=10_000_000, backupCount=5
)
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s"
))
logger = logging.getLogger("pacsys.supervised")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

---

## Using with Async Backends

SupervisedServer also accepts `AsyncBackend` instances from `pacsys.aio`.
When an async backend is provided, the server calls its methods directly
on the gRPC event loop — no executor threads, no callback bridges.

```python
import pacsys.aio as aio
from pacsys.supervised import SupervisedServer

backend = aio.dpm()
with SupervisedServer(backend, port=50051) as srv:
    srv.run()
```

---

## Streaming

The server automatically detects one-shot vs streaming requests based on the DRF event qualifier:

| Event | Behavior |
|-------|----------|
| `@I`, `@N` | One-shot: uses `get_many()`, returns all results |
| Everything else (no event, `@U`, `@P`, `@Q`, `@E`, `@S`) | Streaming: uses `subscribe()`, yields until client disconnects |

Bare DRFs (no event) and `@U` resolve to the device's default event, which is typically `@p,1000` — so they are routed through streaming.

```python
# One-shot (returns immediately)
client.read("M:OUTTMP@I")

# Streaming (continuous updates)
with client.subscribe(["M:OUTTMP@p,1000"]) as stream:
    for reading, _ in stream.readings(timeout=30):
        print(reading.value)
```

---

## See Also

- [gRPC Backend](../backends/grpc.md) -- the client side of the gRPC protocol
- [Writing Guide](../guide/writing.md) -- write operations
