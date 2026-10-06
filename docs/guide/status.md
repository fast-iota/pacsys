# Status and Control

ACNET devices have digital status bits (on/off, ready/tripped, polarity, etc.) and accept control commands to change their state.

---

## Reading Status

### Quick Status (dict)

The simplest way to read status is a basic status read:

```python
import pacsys

status = pacsys.read("Z|ACLTST")   # | qualifier = STATUS
# {"on": True, "ready": False, "remote": True, "positive": True, "ramp": False}
```

Depending on the source, status contains boolean attributes such as `on`, `ready`, `remote`, `positive`, and `ramp`, or display-name keys with text values. Unavailable attributes may be omitted.

### Full Status - DigitalStatus

For richer information (per-bit labels, display values, any number of bits), use `Device.digital_status()`:

```python
from pacsys import Device

dev = Device("Z:ACLTST")
status = dev.digital_status()

print(status)
# Z:ACLTST status=0x02
#   On:       No
#   Ready:    Yes
#   Polarity: Minus
#   ...
```

Synchronous `Device.digital_status()` uses DevDB definitions plus a `BIT_VALUE` read when definitions are available; otherwise it reads `BIT_VALUE`, `BIT_NAMES`, and `BIT_VALUES`. `AsyncDevice.digital_status()` uses the three-sub-property path.

For synchronous `Device.digital_status(timeout=T)`, the timeout applies separately to the DevDB lookup and the status read, not as a shared deadline.

### DigitalStatus API

```python
# Lookup by name (case-insensitive)
bit = status["Ready"]
print(f"{bit.name}: {bit.value} (position {bit.position}, is_set={bit.is_set})")

# Lookup by stored position (meaning depends on constructor)
bit = status[0]

# Safe lookup (returns None if not found)
bit = status.get("Ready")

# Containment check
if "Ready" in status:
    print("Has Ready bit")

# Iteration
for bit in status:
    print(f"{bit.name}: {bit.value}")

# Dict export
d = status.to_dict()
# {"On": "No", "Ready": "Yes", "Polarity": "Minus", ...}
```

### Convenience Attributes

`DigitalStatus` exposes the five standard attributes as `bool | None`:

```python
status.on         # True/False/None
status.ready
status.remote
status.positive
status.ramp
```

These are `None` when the attribute is absent or not recognized by name.

### StatusBit

Each entry in `status.bits` is a frozen `StatusBit`:

| Field | Type | Description |
|-------|------|-------------|
| `position` | `int` | Bit index (see below) |
| `name` | `str` | Label from source data |
| `value` | `str` | Display text ("Yes", "On", "Minus", etc.) |
| `is_set` | `bool` | Whether the bit or attribute is set |

`bool(bit)` returns `is_set`.

`position` is the physical bit index only for `from_bit_arrays()` and DevDB per-bit definitions. DevDB basic attributes are mask predicates (possibly multi-bit or inverted), and dict-based status (`from_status_dict()`, `from_reading()`) uses synthetic positions. For text values, `is_set` is a heuristic: unrecognized text counts as set.

---

## Constructing DigitalStatus

### From Bit Arrays (any backend)

```python
from pacsys.digital_status import DigitalStatus

readings = backend.get_many([
    "Z:ACLTST.STATUS.BIT_VALUE@I",
    "Z:ACLTST.STATUS.BIT_NAMES@I",
    "Z:ACLTST.STATUS.BIT_VALUES@I",
])

status = DigitalStatus.from_bit_arrays(
    device="Z:ACLTST",
    raw_value=int(readings[0].value),
    bit_names=readings[1].value,
    bit_values=readings[2].value,
)
```

### From a BasicStatus Reading

```python
from pacsys.digital_status import DigitalStatus

reading = backend.get("Z|ACLTST")
status = DigitalStatus.from_reading(reading)

# Or from a raw dict
status = DigitalStatus.from_status_dict("Z:ACLTST", {"on": True, "ready": False})
```

Without `raw_value=`, dict-based `status.raw_value` is a synthetic encoding, not the hardware word.

---

## Control Commands

To change a device's state, write `BasicControl` enum values:

```python
from pacsys import BasicControl, KerberosAuth
import pacsys

with pacsys.dpm(auth=KerberosAuth(), role="testing") as backend:
    backend.write("Z|ACLTST", BasicControl.ON)
    backend.write("Z|ACLTST", BasicControl.OFF)
```

### Available Commands

| Command | Effect |
|---------|--------|
| `BasicControl.ON` | Turn device on |
| `BasicControl.OFF` | Turn device off |
| `BasicControl.POSITIVE` | Set positive polarity |
| `BasicControl.NEGATIVE` | Set negative polarity |
| `BasicControl.RAMP` | Set ramp mode |
| `BasicControl.DC` | Set DC mode |
| `BasicControl.RESET` | Reset device |

### DRF for Control

Both of these are equivalent - STATUS is automatically converted to CONTROL for writes:

```python
backend.write("Z|ACLTST", BasicControl.ON)    # | = STATUS → CONTROL
backend.write("Z&ACLTST", BasicControl.ON)    # & = CONTROL directly
```

### Verify Control Effect

Read back status after a control command to verify it took effect:

```python
from pacsys import Device, BasicControl

dev = Device("Z:ACLTST", backend=backend)

backend.write("Z|ACLTST", BasicControl.ON)
status = dev.digital_status()
assert status.on is True

backend.write("Z|ACLTST", BasicControl.OFF)
status = dev.digital_status()
assert status.on is False
```

!!! note "No Batch Control"
    Neither the DPM nor gRPC protocol supports sending multiple control commands in a single message. Issue separate writes for each command.

---

## Writing Status as Dict - Not Supported

You cannot write a dict to STATUS or CONTROL properties:

```python
# This raises ValueError
backend.write("Z|ACLTST", {"on": True, "ready": False})
# ValueError: Cannot write a dict to STATUS property.
#   Use BasicControl enum values instead: backend.write("Z|ACLTST", BasicControl.ON)
```

Use `BasicControl` commands instead.

---

## See Also

- [Writing to Devices](writing.md) - General write operations
- [Reading Devices](reading.md) - Reading status as dict
- [Alarm Helpers](../specialized-utils/alarms.md) - Alarm configuration (separate from status)
