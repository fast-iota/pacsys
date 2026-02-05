# Reading Devices

This guide covers all the ways to read data from ACNET devices.

---

## Simple API

The module-level functions use a shared global backend (DPM/HTTP, lazily initialized).

### Single Value

```python
import pacsys

value = pacsys.read("M:OUTTMP")      # Returns float, str, bytes, np.ndarray, or dict
print(f"Temperature: {value}")
```

`read()` raises `DeviceError` on failure. Use `get()` if you want to inspect errors without exceptions.

### With Metadata

```python
reading = pacsys.get("M:OUTTMP")

print(f"Value: {reading.value} {reading.units}")
print(f"Timestamp: {reading.timestamp}")
print(f"Type: {reading.value_type}")

if reading.is_error:
    print(f"Error: [{reading.facility_code},{reading.error_code}] {reading.message}")
```

Using the Device API, you can get a full `Reading` for any property:

```python
from pacsys import Device

dev = Device("M:OUTTMP")
reading = dev.get(prop="setting")
print(f"Setpoint: {reading.value} at {reading.timestamp}")

reading = dev.get(prop="status", field="on")
print(f"On: {reading.value} at {reading.timestamp}")
```

The `Reading` object fields:

| Field | Description |
|-------|-------------|
| `value` | Device value (type depends on property) |
| `value_type` | `ValueType` enum (SCALAR, SCALAR_ARRAY, RAW, TEXT, etc.) |
| `units` | Engineering units string, or None |
| `timestamp` | `datetime` when reading was taken |
| `cycle` | Machine cycle number |
| `is_success` | `error_code == 0` |
| `is_warning` | `error_code > 0` (data may still be usable) |
| `is_error` | `error_code < 0` |
| `ok` | `error_code >= 0` and value is not None |
| `name` | Device name extracted from DRF or metadata |

### Batch Reads

Read multiple devices in a single network round-trip:

```python
readings = pacsys.get_many([
    "M:OUTTMP",
    "G:AMANDA",
    "B:IRMS06[0:10]",
])

for reading in readings:
    if reading.ok:
        print(f"{reading.name}: {reading.value}")
    else:
        print(f"{reading.name}: ERROR - {reading.message}")
```

Results are returned in the same order as the input list. Devices that time out get a Reading with `is_error=True`.

---

## Reading by Property

The second character of the device name (or an explicit `.PROPERTY`) determines what you read.

### Scalar Reading (default)

```python
# These are equivalent
pacsys.read("M:OUTTMP")
pacsys.read("M:OUTTMP.READING")
```

Returns `float`. `value_type = ValueType.SCALAR`.

### Array Reading

```python
import numpy as np

# Array range [start:end]
arr = pacsys.read("B:IRMS06[0:10]")    # np.ndarray, 11 elements
element = pacsys.read("B:IRMS06[0]")   # float, single element
```

Array reads return `np.ndarray`. `value_type = ValueType.SCALAR_ARRAY`.

### Setting (setpoint)

```python
# Read the current setpoint, not the measured value
pacsys.read("Z_ACLTST")              # _ qualifier = SETTING
pacsys.read("Z:ACLTST.SETTING")      # explicit property

# Array setpoint
pacsys.read("Z_ACLTS1[0:10]")
```

Returns `float` or `np.ndarray`. Same value types as reading.

### Raw Bytes

```python
raw = pacsys.read("M:OUTTMP.RAW")    # bytes object
print(f"Raw: {raw.hex()}")
print(f"Length: {len(raw)} bytes")
```

Returns `bytes`. `value_type = ValueType.RAW`. Raw reads return the unscaled binary value before any database transformation.

### Description (text)

```python
desc = pacsys.read("M~OUTTMP")       # ~ qualifier = DESCRIPTION
# "OUTSIDE TEMPERATURE"
```

Returns `str`. `value_type = ValueType.TEXT`.

### Basic Status

```python
status = pacsys.read("N|LGXS")       # | qualifier = STATUS
# {"on": True, "ready": False, "remote": True, "positive": True, "ramp": False}
```

Returns `dict` with five boolean keys. `value_type = ValueType.BASIC_STATUS`.

For richer status information, see [Device Status](status.md).

### Analog Alarm

```python
alarm = pacsys.read("N@H801")        # @ qualifier = ANALOG alarm
# {"minimum": -10.0, "maximum": 10.0, "alarm_enable": True, ...}
```

Returns `dict` with alarm configuration. `value_type = ValueType.ANALOG_ALARM`.

See also [Alarm Helpers](../specialized-utils/alarms.md) for the `AnalogAlarm` class.

### Digital Alarm

```python
alarm = pacsys.read("N$H801")        # $ qualifier = DIGITAL alarm
# {"nominal": 7, "mask": 255, "alarm_enable": True, ...}
```

Returns `dict`. `value_type = ValueType.DIGITAL_ALARM`.

See also [Alarm Helpers](../specialized-utils/alarms.md) for the `DigitalAlarm` class.

---

## Value Type Reference

| DRF Qualifier | Property | Python Type | `ValueType` |
|---------------|----------|-------------|-------------|
| `:` or default | READING | `float` | `SCALAR` |
| `:` with range | READING | `np.ndarray` | `SCALAR_ARRAY` |
| `_` | SETTING | `float` / `np.ndarray` | `SCALAR` / `SCALAR_ARRAY` |
| `.RAW` | RAW | `bytes` | `RAW` |
| `~` | DESCRIPTION | `str` | `TEXT` |
| `\|` | STATUS | `dict` | `BASIC_STATUS` |
| `@` | ANALOG alarm | `dict` | `ANALOG_ALARM` |
| `$` | DIGITAL alarm | `dict` | `DIGITAL_ALARM` |

---

## Explicit Backend

Instead of the global backend, create your own:

```python
import pacsys

with pacsys.dpm() as backend:
    value = backend.read("M:OUTTMP")
    reading = backend.get("M:OUTTMP")
    readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
```

Any backend works the same way:

```python
# ACL (read-only, no auth)
with pacsys.acl() as backend:
    value = backend.read("M:OUTTMP")

# DMQ (requires Kerberos for all operations)
from pacsys import KerberosAuth
with pacsys.dmq(auth=KerberosAuth()) as backend:
    value = backend.read("M:OUTTMP")

# gRPC
with pacsys.grpc() as backend:
    value = backend.read("M:OUTTMP")
```

See [Backends](../backends/index.md) for details on each backend.

---

## Error Handling

```python
from pacsys import DeviceError

# Option 1: exception-based
try:
    value = pacsys.read("Z:NOTFND")
except DeviceError as e:
    print(f"Facility: {e.facility_code}, Error: {e.error_code}")
    print(f"Message: {e.message}")

# Option 2: inspect Reading status
reading = pacsys.get("Z:NOTFND")
if reading.is_error:
    print(f"Error: {reading.message}")
elif reading.is_warning:
    print(f"Warning (data may be stale): {reading.value}")
else:
    print(f"Value: {reading.value}")
```

Common error codes:

| Facility | Error | Meaning |
|----------|-------|---------|
| 17 (DPM) | 1 | `DPM_PEND` - device not found or no data |
| 16 (DBM) | -13 | `DBM_NOPROP` - property doesn't exist for this device |

---

## See Also

- [DRF Format](../drf.md) - Full DRF syntax reference
- [Device API](device-api.md) - Object-oriented device access
- [Device Status](status.md) - Rich digital status reading
- [Alarm Helpers](../specialized-utils/alarms.md) - Structured alarm access
