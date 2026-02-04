# Ramp Tables

The `Ramp` (and its subclass `BoosterRamp`) classes provide convenient interface for reading and writing ramp tables. This is a partial reimplementation of Java `RampDevice`.

---

## Overview

Each ramp **slot** contains 64 points stored as raw bytes in the SETTING property. Each point is 4 bytes little-endian: a signed int16 **value** followed by a signed int16 **time (clock ticks)**.

```
byte[0:1] = value (int16 LE)  -- F(t) amplitude
byte[2:3] = time  (int16 LE)  -- delta time (clock ticks)
```

The total slot size is 256 bytes.

### Value Scaling

Values are converted between raw int16 and engineering units via overridable transform functions:

```
Forward:  engineering = common_transform(primary_transform(raw))
Inverse:  raw = inverse_primary_transform(inverse_common_transform(engineering))
```

`BoosterRamp` implements these as linear transforms for Booster corrector magnets (engineering units in Amps).

### Time Scaling

Raw times on the wire are clock ticks. The card's `update_rate_hz` determines the tick period. Times are always presented in **microseconds**:

```
Forward:  time_us = raw_ticks * (1e6 / update_rate_hz)
Inverse:  raw_ticks = round(time_us * update_rate_hz / 1e6)
```

Different card types have different update rates:

| Card Class | Type | Update Rate | Tick Period | Notes |
|-----------|------|-------------|-------------|-------|
| 453 | CAMAC | 720 Hz fixed | 1389 µs | Legacy |
| 465/466 | CAMAC | 1 / 5 / 10 KHz | configurable | Stored at byte offset 22 |
| 473 | CAMAC | 100 KHz fixed | 10 µs | Booster correctors |

```python
from pacsys import BoosterRamp

ramp = BoosterRamp.read("B:HS23T", slot=0)
print(ramp.values)  # float64 array, Amps
print(ramp.times)   # float64 array, microseconds
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
ramp.times[:8] = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000]  # microseconds

# Write back
ramp.write("B:HS23T", slot=0)
```

---

## Transforms

### Value Transforms

| Class | Primary Transform | Common Transform | Combined | Units |
|-------|------------------|-----------------|----------|-------|
| `BoosterRamp` | raw / 3276.8 | primary * 4.0 | raw / 819.2 | Amps |

The quantization step for BoosterRamp is approximately 0.00122 Amps.

### Time Scaling

| Class | `update_rate_hz` | Tick Period | Max Time |
|-------|-----------------|-------------|----------|
| `Ramp` (default) | 10,000 Hz | 100 µs | (none) |
| `BoosterRamp` | 100,000 Hz | 10 µs | 66,660 µs (~one 15 Hz cycle) |

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

Subclass `Ramp` and implement the four transform functions:

```python
from pacsys import Ramp

class MainInjectorRamp(Ramp):
    update_rate_hz = 5000  # 5 KHz card (200 us/tick)
    max_value = 500.0      # optional validation bound (engineering units)
    max_time = 1_000_000   # optional validation bound (microseconds)

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

Transforms can also be nonlinear (e.g., polynomial, lookup table). We currently do not (re)implement any standard ACNET transforms -- look them up on your own.

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
#   [ 0] t=     0.0us  value=1.2345
#   [ 1] t=  2400.0us  value=2.3456
#   ...
```

---

## Ramp Card Hardware

### 465 Class CAMAC Cards (453/465/466) (Waveform Generator)

The 465 class CAMAC ramp card produces a single output waveform in response to a TCLK event. There are 32 defined interrupt levels, each triggered by the OR of up to 8 TCLK events.

For each interrupt level the output waveform is:

```
sf1·m1·F(t) + sf2·m2·g(M1) + sf3·m3·h(M2)
```

Where:

- **sf1, sf2, sf3** -- scale factors (-128 to +127.9)
- **m1, m2, m3** -- raw MDAT readings / 256
- **F(t)** -- interpolated function of time (the ramp table `Ramp` manipulates)
- **g(M1), h(M2)** -- interpolated functions of selected MDAT parameters

Update frequency is 1 / 5 / 10 KHz, with up to 15 slots. See references for more details.


### 473 Class CAMAC Cards (Quad Ramp Controller)

The 473 CAMAC ramp card is used by Booster correctors. It has a fixed 100 KHz update rate (10 µs tick period). The `BoosterRamp` subclass uses this card type.

The output waveform is:

```
output = sf1·f(t) + offset (C473)
output = sf1·f(t) + offset + sf2·g(M1) + sf3·h(M2) (C475)
```

Where:

- **sf1, sf2, and sf3** are constant scale factors having a range of -128.0 to +127.9
- **f(t)** is an interpolated function of time which is initiated by a TCLK event. f(t) defines the
overall shape of the output function
- **offset** is a constant offset having a range of -32768 to +32767
- **M1 and M2** are variable values received via MDAT
- **g(M1) and h(M2)** are interpolated functions of M1 and M2, respectively

The output functions of all four channels share a common trigger. Each channel has an
independent delay, programmable from 0 to 65535 µsec, between the TCLK trigger event and the
start of the output functions.

Note: Although the shortest programmable delay is 0 µsec, at least 30 µsec (C473) or 100 µsec
(C475) must be allowed for the processor to service the trigger interrupt. The C473/C475 will
enforce the minimum delay.

See references for more details and configuration.

### References

- [C465 CAMAC module documentation](http://www-bd.fnal.gov/controls/camac_modules/c465.htm)
- [C465 associated device listing](http://www-bd.fnal.gov/controls/micro_p/camac.doc/465.lis)
- [C473 module](https://www-bd.fnal.gov/controls/camac_modules/c473.pdf)
