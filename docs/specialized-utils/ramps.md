# Corrector Ramp Tables

The `CorrectorRamp` and `BoosterRamp` classes provide convenient interfaces for reading and writing corrector magnet ramp tables. Ramp tables are 64-point arrays of (time, value) pairs stored as raw bytes in the SETTING property.

---

## Overview

Each ramp **slot** contains 64 points. Each point is 4 bytes: a signed int16 value followed by a signed int16 time (little-endian). The total slot size is 256 bytes.

Values are converted between raw int16 and engineering units via overridable transform functions:

```
Forward:  engineering = common_transform(primary_transform(raw))
Inverse:  raw = inverse_primary_transform(inverse_common_transform(engineering))
```

`BoosterRamp` implements these as linear transforms for Booster corrector magnets (engineering units in Amps).

```python
from pacsys import BoosterRamp

ramp = BoosterRamp.read("B:HS23T", slot=0)
print(ramp.values)  # float64 array, Amps
print(ramp.times)   # int16 array, microseconds
```

---

## Context Manager (Recommended)

The `modify()` context manager handles read-modify-write automatically:

```python
from pacsys import BoosterRamp

with BoosterRamp.modify("B:HS23T", slot=1) as ramp:
    ramp.values[0] += 1.0   # bump first point by 1 Amp
```

Ramp state is read on context entrance and changes are written on context exit; nothing is written if no changes were made or an exception occurs.

---

## Manual Read/Write

```python
from pacsys import BoosterRamp

# Read
ramp = BoosterRamp.read("B:HS23T", slot=0)

# Modify
ramp.values[:8] = [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]
ramp.times[:8] = [0, 100, 200, 300, 400, 500, 600, 700]

# Write back
ramp.write("B:HS23T", slot=0)
```

---

## Transforms

| Class | Primary Transform | Common Transform | Combined | Units |
|-------|------------------|-----------------|----------|-------|
| `BoosterRamp` | raw / 3276.8 | primary * 4.0 | raw / 819.2 | Amps |

The quantization step for BoosterRamp is approximately 0.00122 Amps.

---

## Slots

Ramp slots are indexed starting at 0. Each slot occupies 256 bytes in the SETTING property:

| Slot | Byte Offset | DRF Pattern |
|------|------------|-------------|
| 0 | 0 | `SETTING{0:256}.RAW` |
| 1 | 256 | `SETTING{256:256}.RAW` |
| 2 | 512 | `SETTING{512:256}.RAW` |

---

## Custom Machine Types

Subclass `CorrectorRamp` and implement the four transform functions:

```python
from pacsys import CorrectorRamp

class MainInjectorRamp(CorrectorRamp):
    max_value = 500.0   # optional validation bound
    max_time = 10000    # optional validation bound

    @classmethod
    def primary_transform(cls, raw):
        return raw / 1638.4

    @classmethod
    def common_transform(cls, primary):
        return primary * 2.0

    @classmethod
    def inverse_common_transform(cls, common):
        return common / 2.0

    @classmethod
    def inverse_primary_transform(cls, primary):
        return primary * 1638.4

ramp = MainInjectorRamp.read("MI:DEVICE", slot=0)
```

Transforms can also be nonlinear (e.g., polynomial, lookup table). We currently do not (re)implement any standard ACNET transforms - look up them on your own.

---

## Raw Bytes

For low-level access, `from_bytes()` and `to_bytes()` handle the binary encoding:

```python
from pacsys import BoosterRamp

raw = b"\x00" * 256  # 64 zero points
ramp = BoosterRamp.from_bytes(raw)

# Serialize back
raw_out = ramp.to_bytes()
```

---

## Error Handling

```python
from pacsys import BoosterRamp
from pacsys.errors import DeviceError

try:
    ramp = BoosterRamp.read("B:BADDEV", slot=0)
except DeviceError as e:
    print(f"Read failed: {e}")
```

Validation errors (values exceeding `max_value` or `max_time`) are raised during `to_bytes()` / `write()`:

```python
ramp.values[0] = 1500.0  # exceeds BoosterRamp.max_value (1000.0)
ramp.to_bytes()  # raises ValueError
```

---

## Display

```python
ramp = BoosterRamp.read("B:HS23T")
print(repr(ramp))  # BoosterRamp(8/64 active points)
print(ramp)
# BoosterRamp (64 points):
#   [ 0] t=     0us  value=1.2345
#   [ 1] t=   100us  value=2.3456
#   ...
```
