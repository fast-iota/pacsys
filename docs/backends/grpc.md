# DPM/gRPC

Modern gRPC interface to DPM. Will be the default in the future. Uses Protocol Buffers for serialization.

```mermaid
sequenceDiagram
    participant App as Your App
    participant gRPC as DPM gRPC<br>:50051
    participant DPM as DPM Server

    App->>gRPC: gRPC connect
    Note over App,gRPC: JWT in metadata (for writes)

    App->>gRPC: ReadDevice(drf)
    gRPC->>DPM: Internal lookup
    DPM-->>gRPC: Device value
    gRPC-->>App: ReadResponse

    App->>gRPC: WriteDevice(drf, value)
    gRPC->>DPM: Apply setting
    DPM-->>gRPC: Result
    gRPC-->>App: WriteResponse
```

## Characteristics

- **Strongly typed**: Protobuf schema with clear message types
- **JWT authentication**: Token-based auth for writes
- **Reachability**: Only accessible on controls network
- **Timestamps**: Proto timestamps carry nanosecond precision but are currently truncated to microseconds by Python `datetime`. All timestamps are UTC-aware. The timestamp type may change in the future to preserve full nanosecond fidelity.

## Usage

```python
import pacsys
from pacsys import JWTAuth

# Read-only
with pacsys.grpc() as backend:
    value = backend.read("M:OUTTMP")

# With explicit JWT authentication (or set PACSYS_JWT_TOKEN env var for automatic auth)
auth = JWTAuth(token="eyJ...")
with pacsys.grpc(auth=auth) as backend:
    result = backend.write("M:OUTTMP", 72.5)
```

## Configuration

| Parameter | Default | Environment Variable |
|-----------|---------|---------------------|
| `host` | dce08.fnal.gov | - |
| `port` | 50051 | - |
| `auth` | None | `PACSYS_JWT_TOKEN` |

## Write Permissions (JWT)

JWT tokens are introspected server-side via a Keycloak endpoint. Your token's `realm_access.roles` determine which devices you can write to. Roles are mapped to ACNET console classes (e.g. `MCR`, `ASTA`, ...). The same bitwise check logic is applied as for DPM/HTTP.
