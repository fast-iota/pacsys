# Writing to Devices

Write operations allow setting scalars, arrays, raw bytes, alarm configuration, and device control.

!!! warning "Authorization Required"
    Writing to devices requires authentication and proper permissions.
    Writes without auth raise `AuthenticationError`.

---

## Simple API

The quickest way to write — uses the global backend:

```python
import pacsys

# Console class writes — just need a Kerberos ticket (kinit)
pacsys.write("Z:ACLTST", 45.0)

# Role-based writes — configure role first
pacsys.configure(role="testing")
pacsys.write("Z:ACLTST", 45.0)

# Batch
results = pacsys.write_many([
    ("Z:ACLTST", 45.0),
    ("G:AMANDA", 1.0),
])
```

The global DPM/HTTP backend auto-creates lazy Kerberos auth — credentials are only validated on the first write. For explicit auth, use `pacsys.configure(auth=pacsys.KerberosAuth())`.

---

## Authentication Setup

### DPM/HTTP - Kerberos

```python
import pacsys

auth = pacsys.KerberosAuth()   # Default principal from `kinit`

# Specific principal (e.g., multiple tickets in cache)
auth = pacsys.KerberosAuth(name="operator@FNAL.GOV")

with pacsys.dpm(auth=auth, role="testing") as backend:
    result = backend.write("Z:ACLTST", 45.0)
```

`auth` is required. `role` is optional — console class writes work without it, but role-based device access requires it (e.g., `role="testing"`).

### DPM/gRPC - JWT

```python
import pacsys

auth = pacsys.JWTAuth(token="eyJ...")   # Or set PACSYS_JWT_TOKEN env var

with pacsys.grpc(auth=auth) as backend:
    result = backend.write("Z:ACLTST", 45.0)
```

### DMQ - Kerberos

```python
import pacsys

auth = pacsys.KerberosAuth()

with pacsys.dmq(auth=auth) as backend:
    result = backend.write("Z:ACLTST", 45.0)
```

DMQ requires Kerberos for all operations (reads too), but no role is needed for writes.

---

## Writing Values

### Scalar

```python
result = backend.write("Z:ACLTST", 45.0)

if result.success:
    print("Write successful")
else:
    print(f"Write failed: [{result.facility_code},{result.error_code}] {result.message}")
```

`WriteResult` has:

| Field | Description |
|-------|-------------|
| `drf` | The DRF that was written |
| `success` | `True` if `error_code == 0` |
| `facility_code` | ACNET facility code |
| `error_code` | 0 = success, <0 = error |
| `message` | Error message (if failed) |

### String

```python
result = backend.write("Z:STRINGTEST", "hello")
```

### Raw Bytes

Write unscaled binary data directly:

```python
# DEC F_float representation of 45.0
result = backend.write("Z:ACLTST.SETTING.RAW", b"\x34\x43\x00\x00")
```

Use the `.RAW` field suffix to bypass scaling transformations.

### Array

```python
import numpy as np

result = backend.write("Z:ACLTS1[0:10]", np.array([1.0, 2.0, 3.0]))
result = backend.write("Z:ACLTS1[0:10]", [1.0, 2.0, 3.0])   # list also works
```

DPM/HTTP accepts one-dimensional homogeneous arrays: all numeric values or all strings.
Mixed text/numeric arrays and multidimensional NumPy arrays raise `TypeError` before any
protocol message is sent.

---

## Batch Writes

Write multiple devices in a single operation:

```python
results = backend.write_many([
    ("Z:ACLTST", 45.0),
    ("G:AMANDA", 1.0),
])

for result in results:
    if not result.success:
        print(f"Failed: {result.drf} - {result.message}")
```

All devices are written in the same request if possible. Results are returned in the same order as the input list.

---

## Implicit Property Conversion

When writing, PACSys automatically converts read properties to their writable counterparts and forces the `@N` (never) event:

| Input | Wire request | Why |
|-------|-------------|-----|
| `Z:ACLTST` | `Z:ACLTST.SETTING@N` | READING → SETTING |
| `Z:ACLTST.READING` | `Z:ACLTST.SETTING@N` | READING → SETTING |
| `Z_ACLTST` | `Z:ACLTST.SETTING@N` | `_` qualifier = SETTING |
| `Z:ACLTST@p,1000` | `Z:ACLTST.SETTING@N` | Event replaced with @N |
| `Z|ACLTST` | `Z:ACLTST.CONTROL@N` | `|` qualifier = STATUS → CONTROL |
| `Z&ACLTST` | `Z:ACLTST.CONTROL@N` | `&` qualifier = CONTROL |
| `Z@ACLTST.MAX` | `Z:ACLTST.ANALOG.MAX@N` | ANALOG alarm field |

The `@N` event tells the server this is a fire-and-confirm operation, not a subscription.

---

## Device Control (on/off/reset)

Use `BasicControl` enum values to send control commands:

```python
from pacsys import BasicControl

# Turn device on/off
backend.write("Z&ACLTST", BasicControl.ON)
backend.write("Z|ACLTST", BasicControl.ON)  # STATUS → CONTROL auto-conversion
backend.write("Z|ACLTST", BasicControl.OFF)

# Other control commands
backend.write("Z|ACLTST", BasicControl.RESET)
backend.write("Z|ACLTST", BasicControl.POSITIVE)
backend.write("Z|ACLTST", BasicControl.NEGATIVE)
backend.write("Z|ACLTST", BasicControl.RAMP)
backend.write("Z|ACLTST", BasicControl.DC)
```

!!! note "Control Commands Are Sequential"
    Each `BasicControl` value is a single command. To toggle on/off and set polarity, issue separate writes. There is no batch control command in the protocol.

See [Device Status](status.md) for reading back status after control writes.

---

## Alarm Configuration Writes

### Individual Fields

Write a single alarm field:

```python
# Analog alarm
backend.write("Z@ACLTST.MAX", 50.0)        # Set maximum limit
backend.write("Z@ACLTST.MIN", 40.0)        # Set minimum limit
backend.write("Z@ACLTST.ALARM_ENABLE", 1)  # Enable alarm
backend.write("Z@ACLTST.ABORT_INHIBIT", 1) # Set abort inhibit (bypass)

# Digital alarm
backend.write("Z$ACLTST.NOM", 0x0001)      # Set nominal bit pattern
backend.write("Z$ACLTST.MASK", 0x00FF)     # Set mask
```

### Dict Shortcut

On any writable backend, you can write multiple alarm fields by passing a dict:

```python
# Analog alarm - set multiple fields at once
backend.write("Z@ACLTST", {
    "minimum": 40.0,
    "maximum": 50.0,
    "alarm_enable": True,
    "abort_inhibit": False,
})

# Digital alarm
backend.write("Z$ACLTST", {
    "nominal": 0x0001,
    "mask": 0x00FF,
    "alarm_enable": True,
})
```

Allowed keys for analog alarms: `minimum`, `maximum`, `alarm_enable`, `abort_inhibit`, `tries_needed`.

Allowed keys for digital alarms: `nominal`, `mask`, `alarm_enable`, `abort_inhibit`, `tries_needed`.

Unknown and read-only keys (`abort`, `alarm_status`, `tries_now`) raise
`ValueError`. Boolean values are converted to 0/1 automatically.

!!! info "Implementation Detail"
    The DPM protocol used in DPM/HTTP has no structured alarm write message. The dict is expanded to individual per-field writes issued sequentially (alarm fields share a hardware block and would overwrite each other in a batch).

### Context Manager (Recommended)

For read-modify-write patterns, use the alarm helpers:

```python
from pacsys.alarm_block import AnalogAlarm

with AnalogAlarm.modify("Z:ACLTST", backend=backend) as alarm:
    alarm.maximum = 50.0
    alarm.minimum = 40.0
    alarm.bypass = False
```

See [Alarm Helpers](../specialized-utils/alarms.md) for the full API.

---

## Write Verification (Device API)

Write verification is available via the [Device API](device-api.md#write-verification) and works with **all** backends:

```python
from pacsys import Device, Verify

dev = Device("Z:ACLTST", backend=backend)
result = dev.write(45.0, verify=Verify(tolerance=0.1))
print(result.confirmed)  # True if the write succeeded and readback matched
```

Note: verification is a `Device.write()` feature, not a backend `write()` feature. Backend `write()` methods do not accept `verify` or `tolerance` parameters.
`WriteResult.success` reports backend write acceptance; `WriteResult.confirmed`
also requires requested verification to succeed.

---

## Error Handling

```python
import pacsys

try:
    result = backend.write("Z:ACLTST", 45.0)
    if not result.success:
        print(f"Write rejected: {result.message}")
except pacsys.AuthenticationError as e:
    print(f"Auth failed: {e}")
```

### Partial Failures in Batch Writes

```python
results = backend.write_many([
    ("Z:ACLTST", 45.0),
    ("Z:NOTFND", 1.0),    # This device doesn't exist
])

# First succeeds, second fails
assert results[0].success
assert not results[1].success
print(f"Failed: {results[1].error_code}")
```

### Common Errors

| Situation | Exception / Result |
|-----------|-------------------|
| No auth configured | `AuthenticationError("not configured for authenticated")` |
| Dict write to non-alarm DRF | `ValueError("Cannot write dict to READING property")` |
| Dict write to STATUS/CONTROL | `ValueError` pointing to `BasicControl` enum |
| Device not writable | `WriteResult.success == False` |

---

## Backend Differences

| Feature | DPM/HTTP | gRPC | DMQ |
|---------|----------|----------|-----|
| Auth type | Kerberos + role | JWT | Kerberos (no role) |
| Alarm dict write | Yes (sequential) | Yes (atomic) | Yes (atomic) |

---

## See Also

- [Reading Devices](reading.md) - Reading values back after writes
- [Device Status](status.md) - Control commands and status verification
- [Alarm Helpers](../specialized-utils/alarms.md) - Structured alarm read-modify-write
- [Backends](../backends/index.md) - Backend architecture and comparison
