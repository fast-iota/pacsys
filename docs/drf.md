# DRF: Data Request Format

DRF (Data Request Format) is the standard way to address data in the Fermilab control system. PACSys includes a full **DRF3 parser** that validates, normalizes, and manipulates DRF strings. All inputs to functions like `read()` are validated and converted to DRF objects before processing.

## Using the DRF Parser

The `DataRequest` class parses any DRF string and provides access to its components:

```python
from pacsys.drf3 import DataRequest

# Parse a DRF string
req = DataRequest("M:OUTTMP.READING[0:10]@p,1000")

# Access components
req.device          # "M:OUTTMP"
req.property        # Property.READING
req.range           # ArrayRange(start=0, end=10)
req.event           # PeriodicEvent(period=1000, ...)

# Get canonical (normalized) form
req.canonical       # "M:OUTTMP.READING[0:10]@p,1000" - we follow Java here instead of writing 1S

# Validation happens at parse time
DataRequest("INVALID!")  # raises ParseError
```

The parser accepts any valid DRF2/DRF3 syntax including property aliases, qualifier shortcuts, and various event formats.

## DRF Utilities

For common operations, `pacsys.drf_utils` provides helper functions that handle parsing internally:

```python
from pacsys.drf_utils import (
    get_device_name,         # Extract just the device name
    ensure_immediate_event,  # Add @I if no event specified
    replace_event,           # Change the event
    strip_event,             # Remove the event
    has_event,               # Check if explicit event exists
    has_explicit_property,   # Check if property was specified
    is_setting_property,     # Check if SETTING property
    prepare_for_write,       # Prepare DRF for write operations
)
```

## Canonical Form

The parser normalizes DRF strings for consistent comparison:

| Rule | Example |
|------|---------|
| Property qualifier → `:` | `M_OUTTMP` → `M:OUTTMP` |
| Aliases → canonical names | `.READ` → `.READING` |
| Default field omitted | `.READING.SCALED` → `.READING` |
| Case: device preserved, rest uppercase | `m:outtmp.read` → `m:OUTTMP.READING` |

```python
from pacsys.drf3 import DataRequest

req = DataRequest("m:outtmp.read@p,1000")
print(req.canonical)  # "m:OUTTMP.READING@p,1000"
```

---

# DRF Reference

## Request Structure

A DRF string combines up to five components:

```
device [ "." property ] [ range ] [ "." field ] [ "@" event ]
```

Only the **device** is required. Examples:

| DRF String | Meaning |
|------------|---------|
| `M:OUTTMP` | Device only (default: READING property, immediate) |
| `M:OUTTMP.SETTING` | Explicit SETTING property |
| `M:OUTTMP[0:99]` | Array range (elements 0-99) |
| `M:OUTTMP@p,1000` | Periodic event (every 1000ms) |
| `M:OUTTMP.READING.SCALED@p,500` | Full specification |

---

## Device Names

Device names are up to 64 characters, starting with a letter (or `0` for index-based addressing).

The **second character** can be a property qualifier shortcut:

| Char | Property | Example |
|------|----------|---------|
| `:` | READING (default) | `M:OUTTMP` |
| `?` | READING | `M?OUTTMP` |
| `_` | SETTING | `M_OUTTMP` |
| `\|` | STATUS | `M\|OUTTMP` |
| `&` | CONTROL | `M&OUTTMP` |
| `@` | ANALOG alarm | `M@OUTTMP` |
| `$` | DIGITAL alarm | `M$OUTTMP` |
| `~` | DESCRIPTION | `M~OUTTMP` |

---

## Properties

Properties specify which aspect of a device to access:

| Property | Canonical | Aliases | Description |
|----------|-----------|---------|-------------|
| Reading | `READING` | READ, PRREAD | Current measured value |
| Setting | `SETTING` | SET, PRSET | Setpoint/commanded value |
| Status | `STATUS` | BASIC_STATUS, STS, PRBSTS | On/off, ready, remote state |
| Control | `CONTROL` | BASIC_CONTROL, CTRL, PRBCTL | Send control commands |
| Analog Alarm | `ANALOG` | ANALOG_ALARM, AA, PRANAB | Analog alarm limits |
| Digital Alarm | `DIGITAL` | DIGITAL_ALARM, DA, PRDABL | Digital alarm configuration |
| Description | `DESCRIPTION` | DESC, PRDESC | Device description text |
| Index | `INDEX` | — | Numeric device index |
| Long Name | `LONG_NAME` | LNGNAM, PRLNAM | Extended device name |

---

## Ranges

Ranges address elements within a data array. Two forms are supported:

### Array Range (preferred)

Uses brackets with element indices:

| Syntax | Meaning |
|--------|---------|
| `[n]` | Single element at index n |
| `[start:end]` | Elements from start to end (inclusive) |
| `[start:]` | From start to end of array |
| `[:end]` | From 0 to end |
| `[]` | Full array |

### Byte Range

Uses braces with byte offset/length (discouraged—requires frontend knowledge):

| Syntax | Meaning |
|--------|---------|
| `{offset}` | Single byte at offset |
| `{offset:length}` | Length bytes starting at offset |
| `{offset:}` | From offset to end |
| `{}` | Full data |

---

## Fields

Fields select specific data flavors within a property:

### Reading/Setting Fields

| Field | Description |
|-------|-------------|
| `SCALED` | Engineering units (default) |
| `PRIMARY` | Primary/volts units |
| `RAW` | Raw binary data |

### Status Fields

| Field | Description |
|-------|-------------|
| `ALL` | All status bits (default) |
| `ON`, `READY`, `REMOTE`, `POSITIVE`, `RAMP` | Individual bits |
| `TEXT`, `EXTENDED_TEXT` | Human-readable status |

### Alarm Fields

| Field | Description |
|-------|-------------|
| `ALL` | Full alarm structure (default) |
| `MIN`, `MAX`, `NOM`, `TOL` | Scaled limits |
| `RAW_MIN`, `RAW_MAX`, `RAW_NOM`, `RAW_TOL` | Raw limits |
| `ALARM_ENABLE`, `ALARM_STATUS` | Enable/status flags |
| `TRIES_NEEDED`, `TRIES_NOW` | Alarm trip counts |

---

## Events

Events control **when** data is acquired:

### Immediate (`@I`)

One-shot read. This is the default for most operations.

```python
pacsys.read("M:OUTTMP")      # Implicit @I
pacsys.read("M:OUTTMP@I")    # Explicit
```

### Periodic (`@P` / `@Q`)

Continuous data at fixed intervals:

| Syntax | Meaning |
|--------|---------|
| `@p,1000` | Every 1000ms |
| `@p,1S` | Every 1 second |
| `@p,500,TRUE` | 500ms, immediate first reading |
| `@q,1000` | Non-continuous (only on change) |

Time units: `S` (seconds), `M` (milliseconds, default), `U` (microseconds)

Frequency units: `H` (Hertz), `K` (kilohertz)

### Clock Event (`@E`)

Triggered by TCLK timing system events:

| Syntax | Meaning |
|--------|---------|
| `@E,0F` | Clock event 0x0F |
| `@E,0F,H` | Hardware trigger only |
| `@E,0F,S` | Software trigger only |
| `@E,0F,E,100` | Either, with 100ms delay |

### State Event (`@S`)

Triggered by state changes on another device:

```
@S,device,value,delay,expression
```

Expressions: `=`, `!=`, `>`, `<`, `>=`, `<=`, `*` (any change)

### Never (`@N`)

Automatically used for settings-only requests (no acquisition).
