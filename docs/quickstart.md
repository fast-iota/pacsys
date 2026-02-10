# Quick Start Guide

## Prerequisites

1. **Network access** to Fermilab (on-site, VPN, or tunnel)
2. **Python 3.10+**
3. For **write operations**: Valid Kerberos credentials

## Installation

```bash
pip install pacsys
```

### Development Install

```bash
git clone https://github.com/fast-iota/pacsys.git
cd pacsys
pip install -e ".[dev]"
```

---

## Read a Device

```python
import pacsys

# Simple value
temperature = pacsys.read("M:OUTTMP")
print(f"Temperature: {temperature}")

# With metadata
reading = pacsys.get("M:OUTTMP")
print(f"{reading.value} {reading.units}")  # e.g. "72.5 DegF"

# Multiple devices at once
readings = pacsys.get_many(["M:OUTTMP", "G:AMANDA"])
for r in readings:
    print(f"{r.name}: {r.value}")
```

`read()` raises `DeviceError` on failure. `get()` returns a `Reading` object you can inspect with `reading.is_error`.

:material-arrow-right: [Reading Guide](guide/reading.md) - all property types, value types, error handling

---

## Stream Data

```python
import pacsys

with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
    for reading, handle in stream.readings(timeout=30):
        print(f"{reading.name}: {reading.value}")
        if reading.value > 100:
            stream.stop() # will stop iterating on next loop
```

The `@p,1000` means "send data every 1000 milliseconds". For streaming, one of repeating event types must be specified.

:material-arrow-right: [Streaming Guide](guide/streaming.md) - callbacks, CombinedStream, error handling

---

## Write a Value

```python
from pacsys import KerberosAuth
import pacsys

auth = KerberosAuth() # will grab default ticket from whichever library is loaded

with pacsys.dpm(auth=auth, role="testing") as backend:
    result = backend.write("Z:ACLTST", 45.0)
    if result.success:
        print("Write successful")
```

pacsys automatically converts READING to SETTING and STATUS to CONTROL when writing.

:material-arrow-right: [Writing Guide](guide/writing.md) - all value types, batch writes, alarm config, device control

---

## Choosing a Backend

| Backend | Factory | Auth | Read | Write | Stream |
|---------|---------|------|:----:|:-----:|:------:|
| **DPM/HTTP** | `pacsys.dpm()` | Kerberos (writes) | :material-check: | :material-check: | :material-check: |
| **DPM/gRPC** | `pacsys.grpc()` | JWT (writes) | :material-check: | :material-check: | :material-check: |
| **DMQ** | `pacsys.dmq()` | Kerberos (all) | :material-check: | :material-check: | :material-check: |
| **ACL** | `pacsys.acl()` | None | :material-check: | - | - |

```python
# Default (DPM/HTTP, implicit)
value = pacsys.read("M:OUTTMP")

# Explicit backend
from pacsys import KerberosAuth
with pacsys.dmq(auth=KerberosAuth()) as backend:
    value = backend.read("M:OUTTMP")
```

:material-arrow-right: [Backends](backends/index.md) - architecture, configuration, comparison

---

## Cleanup

Resources are automatically cleaned up when the Python process exits. No explicit
`shutdown()` call is needed for normal usage:

```python
value = pacsys.read("M:OUTTMP")
# That's it - connections close automatically at exit
```

Use `shutdown()` only if you need to reset state mid-process (e.g., before
re-configuring with `configure()`).

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PACSYS_DPM_HOST` | DPM proxy hostname | acsys-proxy.fnal.gov |
| `PACSYS_DPM_PORT` | DPM proxy port | 6802 |
| `PACSYS_TIMEOUT` | Default timeout (seconds) | 5.0 |
| `PACSYS_JWT_TOKEN` | JWT token for gRPC auth | - |
| `PACSYS_DMQ_HOST` | RabbitMQ broker host | appsrv2.fnal.gov |
| `PACSYS_DMQ_PORT` | RabbitMQ broker port | 5672 |
| `PACSYS_POOL_SIZE` | DPM connection pool size | 4 |
| `PACSYS_DEVDB_HOST` | DevDB gRPC hostname | localhost |
| `PACSYS_DEVDB_PORT` | DevDB gRPC port | 6802 |

---

## Next Steps

- [Reading Guide](guide/reading.md) - All property types and value types
- [Writing Guide](guide/writing.md) - Writes, control commands, alarm config
- [Streaming Guide](guide/streaming.md) - Continuous data, callbacks, combined streams
- [Device Status](guide/status.md) - Digital status bits and control
- [Device API](guide/device-api.md) - Object-oriented device access
- [DRF Format](drf.md) - Device Request Format reference
- [Backends](backends/index.md) - Backend architecture and comparison
- [API Reference](api.md) - Complete API documentation
