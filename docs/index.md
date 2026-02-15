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
- **DevDB integration** - formatted interpretation of device properties (auto-used if available)

## Example

```python
import pacsys

# Read immediately
temperature = pacsys.read("M:OUTTMP@I")         #  72.5
print(f"Temperature: {temperature}")

# Stream periodic
with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
    for reading, _ in stream.readings(timeout=30):
        print(f"{reading.name}: {reading.value}")

# Write (uses default auth)
pacsys.write("Z:ACLTST", 72.5)
```

See the [Quickstart](quickstart.md) for more examples.

## Backends

Multiple backends are implemented:

| Backend | Protocol | Auth Required |
|---------|----------|---------------|
| **DPM/HTTP** | TCP + SDD | Kerberos (for writes) |
| **DPM/gRPC** | TCP + gRPC | JWT token (for writes) |
| **DMQ** | TCP + AMQP + SDD | Kerberos (mandatory) |
| **ACL/HTTP** | TCP + HTTP/CGI | None (read-only) |

See [Backends](backends/index.md) for details.

## Installation

```bash
pip install pacsys
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Detailed examples for reading, writing, and streaming
- [DRF Format](drf.md) - Device addressing syntax (properties, events, ranges)
- [Backends](backends/index.md) - More details on connection options
- [User guide](guide/reading.md) - Start reading user guide
