# Supervised Mode

The `SupervisedServer` is a gRPC proxy that wraps any Backend, forwarding requests while enforcing access policies and logging all traffic.

## Overview

```
[gRPC Client] ──DAQ stub──> [SupervisedServer] ──Backend API──> [Any Backend]
                                   │
                              policies + logging
```

Use cases:

- **Testing** -- expose a `FakeBackend` as a real gRPC server for integration tests
- **Access control** -- restrict which devices or operations are allowed
- **Rate limiting** -- throttle requests per client
- **Audit logging** -- all requests are logged with peer info, timing, and policy decisions

---

## Quick Start

```python
from pacsys.testing import FakeBackend
from pacsys.supervised import SupervisedServer, ReadOnlyPolicy
import pacsys

fb = FakeBackend()
fb.set_reading("M:OUTTMP", 72.5)

with SupervisedServer(fb, port=50099, policies=[ReadOnlyPolicy()]) as srv:
    with pacsys.grpc(host="localhost", port=50099) as client:
        print(client.read("M:OUTTMP"))  # 72.5
        client.write("M:OUTTMP", 80.0)  # PERMISSION_DENIED
```

---

## SupervisedServer

```python
SupervisedServer(backend, port=50051, host="[::]", policies=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `Backend` | *(required)* | Any backend instance to proxy |
| `port` | `int` | `50051` | Port to listen on (use `0` for OS-assigned) |
| `host` | `str` | `[::]` | Bind address |
| `policies` | `list[Policy]` | `None` | Policy chain for access control |

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

Policies are evaluated in order. The first denial short-circuits -- remaining policies are skipped.

### ReadOnlyPolicy

Blocks all write (`Set`) operations, allows reads.

```python
from pacsys.supervised import ReadOnlyPolicy

policies = [ReadOnlyPolicy()]
```

### DeviceAccessPolicy

Allow or deny access based on device name glob patterns.

```python
from pacsys.supervised import DeviceAccessPolicy

# Only allow M: and G: devices
policies = [DeviceAccessPolicy(patterns=["M:*", "G:*"], mode="allow")]

# Block specific devices
policies = [DeviceAccessPolicy(patterns=["Z:SECRET*"], mode="deny")]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patterns` | `list[str]` | *(required)* | `fnmatch` patterns against device names |
| `mode` | `str` | `"allow"` | `"allow"` = only matching allowed, `"deny"` = matching blocked |

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

### Combining Policies

Policies compose naturally -- stack them in order of priority:

```python
from pacsys.supervised import (
    SupervisedServer, ReadOnlyPolicy, DeviceAccessPolicy, RateLimitPolicy
)

policies = [
    ReadOnlyPolicy(),                                      # no writes
    DeviceAccessPolicy(patterns=["M:*", "G:*"]),           # only M: and G: devices
    RateLimitPolicy(max_requests=200, window_seconds=60),  # throttle per client
]

with SupervisedServer(backend, port=50051, policies=policies) as srv:
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
| `rpc_method` | `str` | `"Read"`, `"Set"`, or `"Alarms"` |
| `peer` | `str` | Client address |
| `metadata` | `dict[str, str]` | gRPC metadata from the call |

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

---

## Streaming

The server automatically detects one-shot vs streaming requests based on the DRF event qualifier:

| Event | Behavior |
|-------|----------|
| `@I`, `@U`, `@N`, `@Q,...` (or no event) | One-shot: uses `get_many()`, returns all results |
| `@p,...`, `@e,...`, `@S,...` | Streaming: uses `subscribe()`, yields until client disconnects |

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
- [Alarms](alarms.md) -- alarm configuration (not yet supported in supervised mode)
