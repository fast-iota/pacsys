# Alarm Helpers

The `AnalogAlarm` and `DigitalAlarm` classes provide convenient interfaces for reading and modifying ACNET alarm blocks. On the backend, they read/write `.RAW` byte arrays and fields as necessary - there is no magic.

!!! info "Reference"
    For low-level alarm block structure details, see the [MOOC Property Documentation](https://www-bd.fnal.gov/controls/micro_p/mooc_project/prop.html).

## Overview

| Type | Monitors | Alarms When |
|------|----------|-------------|
| **Analog** | Numeric values | Value outside min/max range |
| **Digital** | Bit patterns | `(reading & mask) != (nominal & mask)` |

```python
from pacsys.alarm_block import AnalogAlarm, DigitalAlarm

# Quick read
analog = AnalogAlarm.read("Z:ACLTST")
digital = DigitalAlarm.read("Z:ACLTST")

print(f"Analog: {analog.minimum} to {analog.maximum}")
print(f"Digital: nominal=0x{digital.nominal:X}, mask=0x{digital.mask:X}")
```

---

## Context Manager (Recommended)

The `modify()` context manager handles read-modify-write automatically:

```python
from pacsys.alarm_block import AnalogAlarm, DigitalAlarm

# Analog: set temperature limits (engineering units)
with AnalogAlarm.modify("Z:ACLTST") as alarm:
    alarm.minimum = 32.0   # Engineering units (e.g., Fahrenheit)
    alarm.maximum = 100.0
    alarm.bypass = False

# Digital: require bit 0 set
with DigitalAlarm.modify("Z:ACLTST") as alarm:
    alarm.nominal = 0x0001
    alarm.mask = 0x0001
    alarm.bypass = False
```

Alarm state is read on context entrance and changes are written on context exit; nothing is written if no changes were made or an exception occurs.

### Engineering Units

Both `read()` and `modify()` also fetch server-scaled engineering ('common') limits. If those are unavailable, `minimum`/`maximum` are `None` and setting them raises `ValueError`. Reading `.RAW` directly returns unscaled bytes.

```python
alarm = AnalogAlarm.read("M:OUTTMP")
print(f"Limits: {alarm.minimum} to {alarm.maximum}")  # in F
```

Since we don't know how to convert raw to common on the client, writing can get complicated. The context manager automatically determines optimal write strategy based on what fields you changed.

### With Explicit Backend

```python
from pacsys import KerberosAuth
import pacsys

auth = KerberosAuth()
with pacsys.dpm(auth=auth, role="testing") as backend:
    with AnalogAlarm.modify("Z:ACLTST", backend=backend) as alarm:
        alarm.bypass = True
```

---

## Manual Control

As usual, you can use `read()` and `write()` separately:

```python
from pacsys.alarm_block import AnalogAlarm

# Read
alarm = AnalogAlarm.read("Z:ACLTST")
print(f"Current: {alarm}")

# Inspect and modify
alarm.bypass = True
alarm.write("Z:ACLTST")

# do naughty stuff

# Set back
alarm.bypass = False
alarm.write("Z:ACLTST")
```

Note: for manual-style writes, all alarm fields are written every time as one raw block. `write()`
cannot apply engineering-unit `minimum`/`maximum` edits; it raises `ValueError` if they changed
since `read()`. Use `modify()` for limits.

---

## Common Properties

Both alarm types share these properties:

| Property | Type | Description |
|----------|------|-------------|
| `is_active` | `bool` | True if alarm is active (not bypassed) |
| `bypass` | `bool` | True if bypassed (inverse of `is_active`) |
| `is_bad` | `bool` | True if currently in alarm state |
| `abort` | `bool` | Read-only abort flag (AB bit) |
| `abort_inhibit` | `bool` | Abort inhibit flag (AI bit) |
| `abort_enabled` | `bool` | True if alarm can trigger abort |
| `tries_needed` | `int` | Consecutive bad readings before alarm |
| `tries_now` | `int` | Current consecutive bad count |
| `ftd` | `FTD` | Sampling configuration |
| `data_length` | `DataLength` | Value byte size (1, 2, or 4) |

---

## Analog Alarm

Monitors numeric values against configurable limits.

### Limit Types

Use `minimum` and `maximum` to set limits in engineering units:

```python
from pacsys.alarm_block import AnalogAlarm

with AnalogAlarm.modify("Z:ACLTST") as alarm:
    alarm.minimum = 0.0    # Engineering units
    alarm.maximum = 100.0
```

### Raw Limit Modes

`minimum`/`maximum` are engineering limits supplied by the server; whether they are available for a given raw limit mode depends on the server and device scaling.

<details markdown>
<summary>Raw alarm block internals</summary>

The raw 20-byte alarm block has a `limit_type` flag (K bits 8-9) that controls how
the two 4-byte value fields are interpreted:

| `limit_type` | K bits | value1 meaning | value2 meaning |
|--------------|--------|----------------|----------------|
| `MIN_MAX` | `0b10` | minimum | maximum |
| `NOM_TOL` | `0b00` | nominal | tolerance |
| `NOM_PCT_TOL` | `0b01` | nominal | percent tolerance |

`value1`/`value2` decode the integer or float stored in the front-end hardware
without applying device scaling. `minimum`/`maximum` contain server-supplied
engineering values when structured scaling succeeds.

Structured analog messages expose only `minimum` and `maximum`. Changing `limit_type`
writes raw metadata; it does not convert either value.

Q bits 5–6 encode lengths 1, 2, and 4 bytes as codes 0, 1, and 2. Reserved Q=3
and K=3 raise `ValueError` on typed access but survive raw parsing and serialization.

</details>

You can still *read* the `limit_type` flag to check which mode a device uses, and
access the raw values via `value1`/`value2`:

```python
from pacsys.alarm_block import AnalogAlarm, LimitType

alarm = AnalogAlarm.read("Z:ACLTST")
if alarm.limit_type == LimitType.NOM_TOL:
    print(f"Nominal (raw units): {alarm.value1}")
    print(f"Tolerance (raw units): {alarm.value2}")
elif alarm.limit_type == LimitType.MIN_MAX:
    print(f"Min (eng units): {alarm.minimum}")
    print(f"Max (eng units): {alarm.maximum}")
elif alarm.limit_type == LimitType.NOM_PCT_TOL:
    print(f"Nominal (raw units): {alarm.value1}")
    print(f"Percent tolerance (raw): {alarm.value2}")
```

Within one `modify()`, a raw `value1`/`value2` edit cannot be combined with the matching
`minimum`/`maximum` edit (or with any limit edit outside `MIN_MAX` mode), and limit edits
cannot be combined with `limit_type`, `data_type`, or `data_length` changes. Conflicts
raise `ValueError` before writing.

### Analog-Specific Properties

| Property | Type | Description |
|----------|------|-------------|
| `limit_type` | `LimitType` | `MIN_MAX`, `NOM_TOL`, or `NOM_PCT_TOL` (raw metadata; see above) |
| `minimum` | `float` or `None` | Minimum in engineering units |
| `maximum` | `float` or `None` | Maximum in engineering units |
| `value1` | `int` or `float` | Raw value 1 (minimum or nominal) |
| `value2` | `int` or `float` | Raw value 2 (maximum, tolerance, or percent tolerance) |
| `is_high` | `bool` | Reading exceeds high limit |
| `is_low` | `bool` | Reading below low limit |
| `data_type` | `DataType` | Value type (signed/unsigned/float) |

---

## Digital Alarm

Monitors bit patterns using mask comparison: `(reading & mask) != (nominal & mask)`

### Mask Examples

```python
from pacsys.alarm_block import DigitalAlarm

with DigitalAlarm.modify("Z:ACLTST") as alarm:
    # Alarm if bit 0 not set
    alarm.nominal = 0x0001
    alarm.mask = 0x0001

    # Alarm if bits 0-3 don't equal 0x05
    alarm.nominal = 0x0005
    alarm.mask = 0x000F

    # Only check bit 7, ignore others
    alarm.nominal = 0x0080
    alarm.mask = 0x0080
```

### Digital-Specific Properties

| Property | Type | Description |
|----------|------|-------------|
| `nominal` | `int` | Expected bit pattern |
| `mask` | `int` | Which bits to check |

---

## FTD (Sampling Configuration)

Controls how often the alarm system samples the device. Equivalent to DRF3 `@p` and `@e` events.

```python
from pacsys.alarm_block import FTD

# Periodic sampling
alarm.ftd = FTD.periodic_hz(1.0)       # 1 Hz
alarm.ftd = FTD.periodic_ticks(60)     # 60Hz ticks (= 1 Hz)

# Event-triggered
alarm.ftd = FTD.on_event(0x0F)              # On TCLK event $0F
alarm.ftd = FTD.on_event(0x0F, delay_ms=100) # With 100ms delay

# Device default (what D80 shows)
alarm.ftd = FTD.default()
```

---

## Alarm Segments

Structured alarm reads and `modify()` support segment 0 only (the server ignores ranges for
structured alarm fields). Access other segments through raw blocks:

```python
raw = backend.read("Z:ACLTST.ANALOG{20:20}.RAW@I")
alarm1 = AnalogAlarm.from_bytes(raw)
alarm1.write("Z:ACLTST", backend=backend, segment=1)
```

---

## Error Handling

```python
from pacsys.alarm_block import AnalogAlarm
from pacsys.errors import DeviceError

try:
    alarm = AnalogAlarm.read("INVALID:DEVICE")
except DeviceError as e:
    print(f"Cannot read alarm: {e}")
    # Common: DBM_NOPROP if device has no analog alarm
```

---

## Complete Example

```python
from pacsys.alarm_block import AnalogAlarm, DigitalAlarm, FTD
from pacsys import KerberosAuth
import pacsys

auth = KerberosAuth()

with pacsys.dpm(auth=auth, role="testing") as backend:
    # Configure analog alarm for temperature (engineering units)
    with AnalogAlarm.modify("Z:TEMP", backend=backend) as alarm:
        alarm.minimum = 32.0   # Engineering units (DPM handles raw conversion)
        alarm.maximum = 100.0
        alarm.ftd = FTD.periodic_hz(0.5)
        alarm.tries_needed = 3
        alarm.bypass = False

    # Configure digital alarm for interlock
    with DigitalAlarm.modify("Z:INTLK", backend=backend) as alarm:
        alarm.nominal = 0x0007  # Bits 0,1,2 must be set
        alarm.mask = 0x0007
        alarm.ftd = FTD.periodic_hz(60.0)
        alarm.tries_needed = 1
        alarm.bypass = False
```

---

## Dict Write Shortcut

Use DPM/HTTP (`pacsys.dpm()`) to write alarm fields as a dict:

```python
backend.write("Z@ACLTST", {"minimum": 40.0, "maximum": 50.0, "alarm_enable": True})
backend.write("Z$ACLTST", {"nominal": 0x0001, "mask": 0x00FF})
```

The dict becomes sequential, non-atomic per-field writes. See [Writing Guide - Alarm Configuration](../guide/writing.md#alarm-configuration-writes) for allowed keys and backend support.

---

## See Also

- [Writing Guide](../guide/writing.md) - General write operations and alarm dict writes
- [DRF Format](../drf.md) - Device request format reference
- [MOOC Property Documentation](https://www-bd.fnal.gov/controls/micro_p/mooc_project/prop.html) - Low-level alarm block details
