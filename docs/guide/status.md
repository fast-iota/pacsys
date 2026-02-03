# Device Status and Control

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

This returns a dict with five boolean keys: `on`, `ready`, `remote`, `positive`, `ramp`.

### Full Status — DigitalStatus

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

This fetches three sub-properties (`BIT_VALUE`, `BIT_NAMES`, `BIT_VALUES`) and constructs a `DigitalStatus` object.

### DigitalStatus API

```python
# Lookup by name (case-insensitive)
bit = status["Ready"]
print(f"{bit.name}: {bit.value} (bit {bit.position}, set={bit.is_set})")

# Lookup by bit position
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

### Legacy Attributes

`DigitalStatus` exposes the five standard attributes as `bool | None`:

```python
status.on         # True/False/None
status.ready
status.remote
status.positive
status.ramp
```

These are `None` if the bit doesn't exist for the device.

### StatusBit

Each bit in `status.bits` is a frozen `StatusBit`:

| Field | Type | Description |
|-------|------|-------------|
| `position` | `int` | Bit index (0-31) |
| `name` | `str` | Label from database |
| `value` | `str` | Display text ("Yes", "On", "Minus", etc.) |
| `is_set` | `bool` | Raw bit value |

`bool(bit)` returns `is_set`.

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

Both of these are equivalent — STATUS is automatically converted to CONTROL for writes:

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

## Writing Status as Dict — Not Supported

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

- [Writing to Devices](writing.md) — General write operations
- [Reading Devices](reading.md) — Reading status as dict
- [Alarm Helpers](../specialized-utils/alarms.md) — Alarm configuration (separate from status)
