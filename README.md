# pacsys

Pure-Python library for Fermilab's control system.

## About

ACNET (Accelerator Control NETwork) is the control system used at Fermilab's particle accelerators. pacsys provides a simple Python interface to read, write, and stream data from ACNET devices without needing to understand the underlying protocols.

## Features

- **Read/Write** device values with synchronous or async APIs
- **Stream** real-time updates EPICS-style
- **Multiple backends** to connect to DPM (Data Pool Manager) and other services
- **Full DRF3 parser** for data request strings

## Device API (recommended)

```python
from pacsys import Device, ScalarDevice, ArrayDevice, Verify, KerberosAuth

# Create a device — DRF is validated immediately
dev = Device("M:OUTTMP")

# Read different properties
temperature = dev.read()               # READING (scaled value)
setpoint = dev.setting()               # SETTING property
is_on = dev.status(field="on")         # STATUS: single bool
alarm = dev.analog_alarm()             # ANALOG alarm
desc = dev.description()               # DESCRIPTION string

# Full reading with metadata
reading = dev.get()
print(f"{reading.value} {reading.units}")

# Typed devices enforce return types
temp = ScalarDevice("M:OUTTMP")        # read() -> float
arr = ArrayDevice("B:IRMS06[0:10]")    # read() -> np.ndarray

# Write with automatic readback verification
with pacsys.dpm(auth=KerberosAuth(), role="testing") as backend:
    dev = Device("M:OUTTMP", backend=backend)
    result = dev.write(72.5, verify=Verify(tolerance=0.5))
    assert result.verified

# Control commands with shortcuts
dev.on()
dev.off()
dev.reset()

# Immutable — modifications return new instances
periodic_dev = dev.with_event("p,1000")
sliced_dev = dev.with_range(0, 10)
```

## Backend API

```python
import pacsys

# Read a device value
temperature = pacsys.read("M:OUTTMP")
print(f"Temperature: {temperature}")

# Stream real-time data
with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
    for reading, _ in stream.readings(timeout=30):
        print(f"{reading.name}: {reading.value}")

# Write (requires authentication)
from pacsys import KerberosAuth
with pacsys.dpm(auth=KerberosAuth(), role="testing") as backend:
    backend.write("Z:ACLTST", 72.5)
```

## Installation

```bash
pip install pacsys
```

## Backends

| Backend | Protocol | Use Case |
|---------|----------|----------|
| **DPM/HTTP** | TCP + binary | General use (default) |
| **DPM/gRPC** | gRPC + Protobuf | High-performance with JWT auth |
| **DMQ** | RabbitMQ + binary | Legacy AMQP backend separate from DPM |
| **ACL** | HTTP/CGI | Quick read-only access |

## Requirements

- Python 3.9+
- For writes: Kerberos credentials with appropriate role assigned

## Documentation

See the [full documentation](https://fast-iota.github.io/pacsys/) for guides, API reference, and protocol details.
