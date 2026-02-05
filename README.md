<h1 align="center">pacsys</h1>

<p align="center">Pure-Python library for Fermilab's control system.</p>

<p align="center">
  <a href="https://github.com/fast-iota/pacsys/actions/workflows/tests.yml"><img src="https://github.com/fast-iota/pacsys/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://fast-iota.github.io/pacsys/"><img src="https://img.shields.io/badge/docs-available-blue" alt="Documentation"></a>
  <a href="https://github.com/fast-iota/pacsys/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="License: GPL-3.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
</p>

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

# Create a device -- DRF is validated immediately
dev = Device("M:OUTTMP")

# Read different properties
temperature = dev.read()               # READING (scaled value)
setpoint = dev.setting()               # SETTING property
is_on = dev.status(field="on")         # STATUS field ON
alarm = dev.analog_alarm()             # ANALOG alarm

# Full reading with metadata
reading = dev.get()
print(f"{reading.value} {reading.units}")

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

# Write (requires authentication)
from pacsys import KerberosAuth
with pacsys.dpm(auth=KerberosAuth(), role="testing") as backend:
    backend.write("Z:ACLTST", 72.5)
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

## Installation

```bash
pip install pacsys
```

## Requirements

- Python 3.10+
- For writes: Kerberos credentials with appropriate role assigned

## Documentation

See the [full documentation](https://fast-iota.github.io/pacsys/) for guides, API reference, and protocol details.
