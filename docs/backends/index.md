# Backends

## Overview

PACSys provides multiple backends that talk to various central services. The DPM backends connect to the DPM server, DMQ talks to DataBroker/Bunny OAC/DAE, etc.

```mermaid
flowchart TB
    subgraph App[Your Application]
        P[Python + pacsys]
    end

    subgraph CS[Central Services]
        DPM[DPM]
        ACL[ACL CGI]
        DMQ[DataBroker<br>Bunny OAC/DAE]
        DB[PostgreSQL]
        CLX[CLX nodes]
    end

    FE[Frontends]
    OT[Other tasks]

    P -->|DPM/HTTP<br>TCP:6802<br>SDD| DPM
    P -->|DPM/gRPC<br>gRPC:50051<br>Protobuf| DPM
    P -->|HTTPS:443<br>CGI script| ACL
    P -->|DMQ<br>TCP:5672<br>AMQP| DMQ
    P -->|DevDB<br>gRPC:443/TLS<br>Protobuf| DB
    P -->|ACL-over-SSH<br>SSH:22| CLX

    CS --> FE
    CS --> OT
```

## Backend Comparison

| Feature | DPM/HTTP | DPM/gRPC | DMQ | ACL |
|---------|:--------:|:--------:|:---:|:---:|
| **Read** | ✓ | ✓ | ✓ | ✓ |
| **Write** | ✓ | ✓ | ✓ | - |
| **Stream** | ✓ | ✓ | ✓ | - |
| **Auth** | Kerberos² | JWT | Kerberos³ | None |
| **Permissions** | Role/class | Role | Console class | N/A |
| **Protocol** | TCP/SDD¹ | gRPC/Protobuf | AMQP/SDD¹ | HTTP/CGI |
| **Port** | 6802 | 50051 | 5672 | 443 |
| **Factory** | `pacsys.dpm()` | `pacsys.grpc()` | `pacsys.dmq()` | `pacsys.acl()` |

¹ SDD = Self-Describing Data, the binary encoding used by protocol compiler in DMQ and DPM/HTTP
² DPM/HTTP: Kerberos required for writes only, reads work anonymously
³ DMQ: Kerberos required for ALL operations (reads, writes, streaming)

---

## Implicit Write Conversion

All write-capable backends (DPM/HTTP, DPM/gRPC, DMQ) automatically adjust DRF strings when you call `write()`:

- **READING → SETTING**: `M:OUTTMP` becomes `M:OUTTMP.SETTING@N`
- **STATUS → CONTROL**: `M:OUTTMP.STATUS` becomes `M:OUTTMP.CONTROL@N`
- **Already correct**: SETTING, CONTROL, ANALOG, DIGITAL are preserved (just `@N` is forced)

The `@N` ("never") event tells the server not to send periodic data back - writes are fire-and-confirm, not subscriptions.

---

## Backend recommendations

| Scenario | Backend |
|----------|---------|
| Good-enough read/write | **DPM/HTTP** |
| High-performance with JWT token (ask ACORN when this will work) | **DPM/gRPC** |
| High-performance read-only (works now) | **DPM/gRPC** |
| Fallback if DPM has issues/bugs | **DMQ** |
| Last resort | **ACL** |

---

## Multiple Backends

You can use different backends simultaneously:

```python
import pacsys
from pacsys import KerberosAuth

# Quick read via ACL (no auth needed)
with pacsys.acl() as acl:
    current = acl.read("M:OUTTMP")

# Write via DPM/HTTP with auth
with pacsys.dpm(auth=KerberosAuth(), role="testing") as dpm:
    dpm.write("M:OUTTMP", current + 1.0)

# Or write via DMQ (no role needed, just Kerberos)
with pacsys.dmq(auth=KerberosAuth()) as dmq:
    dmq.write("M:OUTTMP", current + 1.0)

# Stream via DPM/HTTP
with pacsys.dpm() as dpm:
    with dpm.subscribe(["M:OUTTMP@p,1000"]) as stream:
        for reading, _ in stream.readings(timeout=30):
            print(reading.value)

# Or stream via DMQ
with pacsys.dmq(auth=KerberosAuth()) as dmq:
    with dmq.subscribe(["M:OUTTMP@p,1000"]) as stream:
        for reading, _ in stream.readings(timeout=30):
            print(reading.value)
```
