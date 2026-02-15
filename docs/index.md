# PACSys - Pure-Python library for Fermilab's control system

PACSys is a Python library that lets you interact with ACNET (aka ACSys).

High-level features:

- **Read/Write/Stream** any ACNET data types with synchronous or async APIs
- **Multiple backends** to connect to DPM, DMQ, and ACL
- **Full DRF3 parser** for data requests with automatic conversion
- **Utilities** for device database, SSH tunneling, and more
- **Command-line tools** like in EPICS - `acget`, `acput`, `acmonitor`, `acinfo`

Low-level features:

- **Raw ACNET UDP/TCP** - talk like a civilized member of ACNET society
- **FTPMAN for snapshots** - yes, really
- **SSH utilities and ACL-over-SSH** - authenticated command runners, useful for ACL/DABBEL
- **DevDB integration** - use database info for better interpretation of device properties


## Quick Example

```python
import pacsys

# Read immediately
temperature = pacsys.read("M:OUTTMP@I")         #  72.5
print(f"Temperature: {temperature}")

# Stream
with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
    for reading, _ in stream.readings(timeout=30):
        print(f"{reading.name}: {reading.value}")

# Write (tries default auth like default kerberos ticket)
pacsys.write("Z:ACLTST", 72.5)
```

## Backend API
To gain more control over how and where requests go, use backend API.

```python
# Specify DPM/HTTP backend with specific kerberos principal and role
with pacsys.dpm(auth=pacsys.KerberosAuth(), role="testing") as backend:
    temperature = backend.read("M:OUTTMP")
    print(f"Temperature: {temperature}")
    backend.write("Z:ACLTST", 72.5)
```

## Device API

For device-centric workflows, the `Device` class provides an object-oriented interface with DRF validation, typed reads, and write verification:

```python
from pacsys import Device, ScalarDevice, Verify

dev = Device("M:OUTTMP")

# Read different properties
temperature = dev.read()               # READING (scaled)
setpoint = dev.setting()               # SETTING
is_on = dev.status(field="on")         # STATUS field (bool)
desc = dev.description()               # DESCRIPTION (str)

# Full reading with metadata
reading = dev.get()
print(f"{reading.value} {reading.units}")  # e.g. "72.5 DegF"

# Typed devices enforce return types
temp = ScalarDevice("M:OUTTMP")        # read() -> float

# Write with automatic readback verification
result = dev.write(72.5, verify=Verify(tolerance=0.5))

# Control commands
dev.on()
dev.off()
dev.reset()

# Immutable fluent modifications
periodic = dev.with_event("p,1000")
sliced = dev.with_range(0, 10)
```

See the [Device API guide](guide/device-api.md) for full documentation.

## Backends

PACSys connects to services using backends:

| Backend | Protocol | Auth Required |
|---------|----------|---------------|
| **DPM/HTTP** | TCP + binary protocol | Kerberos (for writes) |
| **DPM/gRPC** | TCP + gRPC | JWT token (for writes) |
| **DMQ** | TCP + AMQP + binary protocol | Kerberos (mandatory) |
| **ACL/HTTP** | TCP + HTTP/CGI | None (read-only) |
| **SSH utilities** | TCP + SSH | Kerberos (mandatory) |

See [Backends](backends/index.md) for details.

## Installation

```bash
pip install pacsys
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Detailed examples for reading, writing, and streaming
- [DRF Format](drf.md) - Device addressing syntax (properties, events, ranges)
- [Backends](backends/index.md) - Connection options and architecture diagrams
- [API Reference](api.md) - Complete API documentation
