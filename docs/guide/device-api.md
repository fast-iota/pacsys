# Device API

The Device API provides an object-oriented interface to ACNET devices with DRF3 validation at construction time.

---

## Creating Devices

```python
from pacsys import Device

dev = Device("M:OUTTMP")           # Validates DRF syntax immediately
dev = Device("M:OUTTMP@p,1000")    # With periodic event
dev = Device("B:IRMS06[0:10]")     # With array range
```

Invalid DRF strings raise `ValueError` at construction, not at read time.

### Typed Devices

Subclasses enforce value types on `read()`:

```python
from pacsys import ScalarDevice, ArrayDevice, TextDevice

temp = ScalarDevice("M:OUTTMP")      # read() returns float
arr = ArrayDevice("B:IRMS06[0:10]")  # read() returns np.ndarray
desc = TextDevice("M~OUTTMP")        # read() returns str
```

If the actual value doesn't match the expected type, `read()` raises `TypeError`.

---

## Reading Properties

Each property has a dedicated read method. All use `@I` (immediate) event.

```python
dev = Device("M:OUTTMP")

# READING property (default)
value = dev.read()                     # Scaled value
raw = dev.read(field="raw")            # Raw value
primary = dev.read(field="primary")    # Primary units

# SETTING property
setpoint = dev.setting()
raw_set = dev.setting(field="raw")

# STATUS property
status = dev.status()                  # Full status dict
is_on = dev.status(field="on")         # Single bool
is_ready = dev.status(field="ready")

# ANALOG / DIGITAL alarm
alarm = dev.analog_alarm()
dalarm = dev.digital_alarm()
alarm_min = dev.analog_alarm(field="min")

# DESCRIPTION
desc = dev.description()               # Returns str

# Full metadata read (uses device's base DRF)
reading = dev.get()
print(f"{reading.value} {reading.timestamp}")

# Full metadata for a specific property
reading = dev.get(prop="setting")
reading = dev.get(prop="status", field="on")
print(f"{reading.value} at {reading.timestamp}")
```

The `prop` argument accepts any property name: `'reading'`, `'setting'`, `'status'`, `'analog'`, `'digital'`, `'description'`. When omitted, `get()` uses the device's base DRF.

Invalid fields raise `ValueError`:

```python
dev.read(field="on")      # ValueError: 'on' not allowed for READING
```

---

## Device Metadata

Fetch device information from DevDB (scaling parameters, limits, control commands, status bit definitions):

```python
dev = Device("M:OUTTMP")
info = dev.info()

print(info.description)                # "OUTSIDE TEMPERATURE"
print(info.reading.common_units)       # "DegF"
print(info.reading.p_index)            # Primary transform index
```

`info()` returns a `DeviceInfoResult` with fields: `device_index`, `description`, `reading` (`PropertyInfo`), `setting` (`PropertyInfo`), `control`, `status_bits`. Results are cached.

!!! note "Requires DevDB"
    DevDB connects to `ad-services.fnal.gov/services.devdb` by default. Override with `pacsys.configure(devdb_host=...)` or the `PACSYS_DEVDB_HOST` environment variable.

---

## Writing

### Setting Values

```python
dev = Device("M:OUTTMP", backend=backend)

# Write to SETTING property
result = dev.write(72.5)
result = dev.write(100, field="raw")

assert result.success
```

### Control Commands

```python
from pacsys import BasicControl

dev = Device("Z:ACLTST", backend=backend)

# Using control() directly
result = dev.control(BasicControl.ON)

# Using shortcuts
dev.on()
dev.off()
dev.reset()
dev.positive()
dev.negative()
dev.ramp()
dev.dc()
dev.local()
dev.remote()
dev.trip()
```

### Alarm Settings

```python
dev.set_analog_alarm({"minimum": 40, "maximum": 80})
dev.set_digital_alarm({"nominal": 0x01, "mask": 0xFF})
```

---

## Write Verification

The `Verify` dataclass configures automatic readback verification after writes.

### Basic Usage

```python
from pacsys import Verify

dev = Device("M:OUTTMP", backend=backend)

# Verify with default settings (3 attempts, 0.3s initial delay)
result = dev.write(72.5, verify=True)
assert result.verified    # True if readback matched

# Custom verify settings
v = Verify(tolerance=0.5, max_attempts=5)
result = dev.write(72.5, verify=v)

# Disable verification explicitly
result = dev.write(72.5, verify=False)
```

### Check First (Skip Redundant Writes)

```python
v = Verify(check_first=True, tolerance=0.1)
result = dev.write(72.5, verify=v)
if result.skipped:
    print("Value was already correct, write skipped")
```

### Context Manager

Set verification defaults for a block of code:

```python
with Verify(always=True, tolerance=0.1):
    # All writes in this block auto-verify
    dev.write(72.5)       # verify=None → uses context
    dev.write(80.0)       # also auto-verified

    dev.write(90.0, verify=False)  # explicitly disabled
```

### Control Verification

Control commands verify by reading the corresponding STATUS field:

```python
result = dev.on(verify=True)
# Reads STATUS.ON after writing CONTROL, confirms it's True
assert result.verified
```

### WriteResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `drf` | `str` | DRF that was written |
| `success` | `bool` | True if error_code == 0 |
| `facility_code` | `int` | ACNET facility code |
| `error_code` | `int` | 0 = success, <0 = error |
| `message` | `str \| None` | Error message (if failed) |
| `verified` | `bool \| None` | True=matched, False=failed, None=no verify |
| `readback` | `float \| str \| bytes \| ... \| None` | Last readback value |
| `skipped` | `bool` | True if check_first found value correct |
| `attempts` | `int` | Number of readback attempts made |

---

## Streaming

Subscribe to a device for continuous updates using `subscribe()`. The device must have an event (set at construction or via `event=`).

### Iterator Mode

```python
dev = Device("M:OUTTMP@p,1000")

with dev.subscribe() as stream:
    for reading, handle in stream.readings(timeout=30):
        print(f"{reading.value}")
        if reading.value > 100:
            stream.stop()
```

### Callback Mode

```python
dev = Device("M:OUTTMP")

handle = dev.subscribe(
    callback=lambda r, h: print(r.value),
    event="p,1000",
)
import time; time.sleep(10)
handle.stop()
```

### Property and Field

```python
dev = Device("M:OUTTMP@p,1000")

# Stream a specific property
with dev.subscribe(prop="setting") as stream:
    for reading, _ in stream.readings(timeout=10):
        print(f"Setpoint: {reading.value}")

# Stream a specific field
with dev.subscribe(prop="status", field="on") as stream:
    for reading, _ in stream.readings(timeout=10):
        print(f"On: {reading.value}")
```

### Event Override

```python
dev = Device("M:OUTTMP")  # no event

# Provide event at subscribe time
with dev.subscribe(event="p,500") as stream:
    for reading, _ in stream.readings(timeout=10):
        print(reading.value)
```

`subscribe()` raises `ValueError` if no event is available (neither on the device nor via `event=`).

---

## Attributes

| Property | Type | Description |
|----------|------|-------------|
| `drf` | `str` | Canonical DRF string |
| `name` | `str` | Device name only (no property/range/event) |
| `request` | `DataRequest` | Parsed DRF3 request |
| `has_event` | `bool` | True if explicit event specified |
| `is_periodic` | `bool` | True if periodic event |

```python
dev = Device("M:OUTTMP@p,1000")
dev.drf          # "M:OUTTMP.READING@p,1000"
dev.name         # "M:OUTTMP"
dev.has_event    # True
dev.is_periodic  # True
```

---

## Fluent Modifications

Device objects are immutable. Modification methods return new instances:

### Change Event

```python
dev = Device("M:OUTTMP")
periodic = dev.with_event("p,1000")     # M:OUTTMP.READING@p,1000
immediate = dev.with_event("I")         # M:OUTTMP.READING@I
```

### Change Range

```python
dev = Device("B:IRMS06")
sliced = dev.with_range(0, 10)          # B:IRMS06.READING[0:10]
from_5 = dev.with_range(start=5)        # B:IRMS06.READING[5:]
single = dev.with_range(at=5)           # B:IRMS06.READING[5]
full   = dev.with_range()               # B:IRMS06.READING[:]
```

### Bind to Backend

```python
import pacsys

with pacsys.dpm() as backend:
    dev = Device("M:OUTTMP").with_backend(backend)
    value = dev.read()   # Uses the explicit backend
```

Without `with_backend()`, devices use the global backend (initialized on first use).

### Subclass Preservation

Fluent methods preserve the subclass:

```python
temp = ScalarDevice("M:OUTTMP")
periodic = temp.with_event("p,1000")
type(periodic)   # ScalarDevice (not Device)
```

---

## Digital Status

Fetch full bit-level digital status:

```python
dev = Device("Z:ACLTST")
status = dev.digital_status()

print(status)
# Z:ACLTST status=0x02
#   On:       No
#   Ready:    Yes

# Inspect individual bits
status["Ready"].is_set    # True
status.on                 # False
```

This reads `STATUS.BIT_VALUE`, `STATUS.BIT_NAMES`, and `STATUS.BIT_VALUES` from the backend and constructs a `DigitalStatus` object.

See [Device Status](status.md) for the full DigitalStatus API.

---

## Equality and Hashing

Devices compare by canonical DRF string:

```python
Device("M:OUTTMP") == Device("M:OUTTMP")          # True
Device("M:OUTTMP") == Device("M:OUTTMP@p,1000")   # False

# Can be used as dict keys or in sets
devices = {Device("M:OUTTMP"), Device("G:AMANDA")}
```

---

## See Also

- [Reading Devices](reading.md) - Reading patterns and value types
- [Streaming Guide](streaming.md) - Backend-level streaming, CombinedStream, error handling
- [Device Status](status.md) - DigitalStatus and control commands
- [DRF Format](../drf.md) - DRF syntax reference
