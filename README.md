<h1 align="center">pacsys</h1>

<p align="center">Pure-Python library for Fermilab's control system.</p>

<p align="center">
  <a href="https://github.com/fast-iota/pacsys/actions/workflows/tests.yml"><img src="https://github.com/fast-iota/pacsys/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://fast-iota.github.io/pacsys/"><img src="https://img.shields.io/badge/docs-available-blue" alt="Documentation"></a>
  <a href="https://github.com/fast-iota/pacsys/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="License: GPL-3.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
</p>

## About

ACNET (Accelerator Control NETwork) is the control system used at Fermilab's particle accelerators. PACSysprovides a simple Python interface to read, write, and stream ACNET data without needing to understand the underlying protocols.

## Features

- **Read/Write/Stream** any ACNET data types with synchronous or async APIs
- **Multiple backends** to connect to DPM, DMQ, and other services
- **Full DRF3 parser** for data requests with automatic conversion
- **Utilities** for device database, SSH tunneling, CLI, and more

## Device API (recommended)

```python
import pacsys
from pacsys import Device, ScalarDevice, ArrayDevice, Verify, KerberosAuth

# Create a device -- DRF is validated immediately
dev = Device("M:OUTTMP")

# Read different properties
temperature = dev.read()               # READING (scaled value)
setpoint = dev.setting()               # SETTING property
is_on = dev.status(field="on")         # STATUS field ON
alarm = dev.analog_alarm()             # ANALOG alarm

# Full reading with metadata
reading = dev.get()
print(f"{reading.value} {reading.units}")  # e.g. "72.5 DegF"

# Write with automatic readback verification
with pacsys.dpm(auth=KerberosAuth(), role="testing") as backend:
    dev = Device("M:OUTTMP", backend=backend)
    result = dev.write(72.5, verify=Verify(tolerance=0.5))
    assert result.verified

# Control commands with shortcuts
dev.on()
dev.off()
dev.reset()

# Device database metadata (scaling, limits, units)
info = dev.info()
print(info.description)                # "Outside temperature"
print(info.reading.common_units)       # "DegF"
print(info.reading.min_val)            # 0.0

# Stream data
with dev.with_event("p,1000").subscribe() as stream:
    for reading, _ in stream.readings(timeout=10):
        print(reading.value)

# Immutable -- modifications return new instances
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

# Stream with callback dispatch mode
# WORKER (default): callbacks on dedicated worker thread, protects reactor
# DIRECT: callbacks inline on reactor thread (lower latency)
with pacsys.dpm(dispatch_mode=pacsys.DispatchMode.DIRECT) as backend:
    handle = backend.subscribe(
        ["M:OUTTMP@p,1000"],
        callback=lambda r, h: print(r.value),
    )
    import time; time.sleep(10)
    handle.stop()

# Write (requires authentication)
with pacsys.dpm(auth=pacsys.KerberosAuth(), role="testing") as backend:
    backend.write("Z:ACLTST", 72.5)
```

## Async API

Native async support for asyncio applications. Same API surface, no background threads.

```python
import pacsys.aio as aio

# Module-level API (mirrors pacsys.read, pacsys.get, etc.)
value = await aio.read("M:OUTTMP")
reading = await aio.get("M:OUTTMP")

# Explicit async backend
async with aio.dpm(auth=pacsys.KerberosAuth()) as backend:
    await backend.write("Z:ACLTST", 72.5)

# Async streaming
async with await backend.subscribe(["M:OUTTMP@p,1000"]) as stream:
    async for reading, handle in stream.readings(timeout=30):
        print(f"{reading.name}: {reading.value}")

# AsyncDevice
from pacsys.aio import AsyncDevice

dev = AsyncDevice("M:OUTTMP", backend=backend)
temp = await dev.read()
await dev.on()
```

## SSH Utilities

Port tunneling, SFTP, and interactive processes over multi-hop SSH.

```python
import pacsys

# Execute commands with automatic Kerberos auth
with pacsys.ssh(["jump.fnal.gov", "target.fnal.gov"]) as ssh:
    result = ssh.exec("hostname")
    print(result.stdout) # target

# ACL can be run on the fly - beam switch, DB, etc.
with pacsys.ssh("clx01.fnal.gov") as ssh:
    result = ssh.acl("read M:OUTTMP") # "M:OUTTMP       =  72.500 DegF"
```

## CLI Tools

pacsys includes EPICS-style command-line tools:

```bash
# Read devices
acget M:OUTTMP Z:ACLTST
acget --format json M:OUTTMP

# Write devices (requires authentication)
acput M:OUTTMP 72.5
acput -a kerberos --verify --tolerance 0.5 M:OUTTMP 72.5

# Monitor (streaming)
acmonitor M:OUTTMP
acmonitor -n 10 M:OUTTMP@p,500

# Device info
acinfo -v M:OUTTMP
```

All tools are also available as `pacsys-get`, `pacsys-put`, `pacsys-monitor`, `pacsys-info`.

## Installation

```bash
pip install pacsys
```

## Requirements

- Python 3.10+
- For writes: Kerberos credentials with appropriate role assigned
- For some utilities: Must run on the controls network or have SSH access to it

## Documentation

See the [full documentation](https://fast-iota.github.io/pacsys/) for guides, API reference, and protocol details.
