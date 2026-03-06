# Data Helpers

The `pacsys.exp` module provides high-level utilities for common accelerator physics workflows: monitoring channels, scanning parameters, logging data to files, and waiting for conditions.

```python
from pacsys.exp import Monitor, read_fresh, watch, scan, DataLogger
from pacsys.exp import CsvWriter, ParquetWriter
```

All functions accept DRF strings or `Device` objects and use the global default backend unless one is explicitly provided.

---

## Monitor

Subscribe to one or more channels and collect readings into ring buffers. Supports both blocking (`collect`) and non-blocking (`start`/`stop`) modes.

### Blocking Collection

```python
from pacsys.exp import Monitor

mon = Monitor(["M:OUTTMP@p,1000", "G:AMANDA@e,8f"])
result = mon.collect(duration=10.0)

print(result.mean("M:OUTTMP@p,1000"))    # Mean value
print(result.std("M:OUTTMP@p,1000"))     # Standard deviation
print(result.counts)                      # {drf: count} per channel
print(result.rate())                      # Readings/sec per channel
```

Alternatively, collect until each channel has a minimum number of readings:

```python
result = mon.collect(count=100, timeout=30.0)
```

### Non-Blocking (Start/Stop)

```python
mon = Monitor(["M:OUTTMP@p,1000"])
mon.start()

# Peek without consuming data
snap = mon.snapshot()

# Swap out buffers and return old data
result = mon.flush()

mon.stop()
```

Or use the context manager:

```python
with Monitor(["M:OUTTMP@p,1000"]) as mon:
    snap = mon.snapshot()
```

### Waiting for New Data

Use `await_next()` to block until a new reading arrives on a specific channel:

```python
with Monitor(["M:OUTTMP@p,1000"]) as mon:
    reading = mon.await_next("M:OUTTMP@p,1000", timeout=5.0)
    print(reading.value)
```

Use `wait_until()` to block until an arbitrary predicate is satisfied:

```python
with Monitor(["M:OUTTMP@p,1000"]) as mon:
    result = mon.wait_until(
        lambda snap: snap.mean("M:OUTTMP@p,1000") > 72.0,
        timeout=30.0,
    )
```

### Change Detection

Use `tags` and `has_new()` for cheap polling without copying buffers:

```python
with Monitor(["M:OUTTMP@p,1000"]) as mon:
    old_tags = mon.tags
    # ... do other work ...
    if mon.has_new(old_tags):
        snap = mon.snapshot()
```

### Constructor Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devices` | `list[DeviceSpec]` | required | DRF strings or `Device` objects |
| `buffer_size` | `int` | `10_000` | Max readings per channel (ring buffer) |
| `backend` | `Backend` | `None` | Backend to use (global default if `None`) |

---

## MonitorResult

Returned by `Monitor.collect()`, `snapshot()`, and `flush()`. Provides statistics, slicing, and export methods.

### Statistics

All stat methods accept an optional `drf` argument. If given, returns a single value; if omitted, returns a dict keyed by DRF.

| Method | Description |
|--------|-------------|
| `mean(drf?)` | Arithmetic mean |
| `std(drf?)` | Standard deviation (population) |
| `median(drf?)` | Median value |
| `min(drf?)` | Minimum value |
| `max(drf?)` | Maximum value |
| `rate(drf?)` | Readings per second |
| `last(n, drf?)` | Last *n* values |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `channels` | `dict[str, ChannelData]` | Per-channel data |
| `counts` | `dict[str, int]` | Reading count per channel |
| `elapsed` | `timedelta \| None` | Duration between start and stop |
| `started` | `datetime \| None` | When collection started |
| `stopped` | `datetime \| None` | When collection stopped |

### Accessing Raw Data

```python
values = result.values("M:OUTTMP@p,1000")          # list[Value]
timestamps = result.timestamps("M:OUTTMP@p,1000")  # list[datetime]
```

### Time Slicing

```python
from datetime import datetime, timezone

start = datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc)
end = datetime(2026, 3, 5, 12, 5, tzinfo=timezone.utc)
sliced = result.slice("M:OUTTMP@p,1000", start=start, end=end)
# Returns a ChannelData with only readings in [start, end]
```

### Export to NumPy

```python
timestamps, values = result.to_numpy("M:OUTTMP@p,1000")
# timestamps: float64 epoch seconds (UTC)
# values: float64
```

### Export to pandas

```python
# Single channel — indexed by timestamp
df = result.to_dataframe("M:OUTTMP@p,1000")

# All channels — flat table with drf column
df = result.to_dataframe()

# Relative timestamps (seconds since collection started)
df = result.to_dataframe("M:OUTTMP@p,1000", relative=True)
```

---

## read_fresh

Wait for one or more fresh readings per channel via a temporary subscription.
Consider it a `pacsys.read()` but with a lot more options and ability to collect
multiple readings.

```python
from pacsys.exp import read_fresh

# Single reading per channel (default)
results = read_fresh(["M:OUTTMP", "G:AMANDA"], default_event="p,1000")
for r in results:
    print(f"{r.drf}: {r.value}")

# Collect 10 readings and average
results = read_fresh(["M:OUTTMP@p,1000"], count=10, timeout=5.0)
print(results[0].mean())     # mean of last 10
print(results[0].std())      # standard deviation
print(results[0].median())   # median
print(results[0].values)     # all collected values
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devices` | `list[DeviceSpec]` | required | DRF strings or `Device` objects |
| `count` | `int` | `1` | Readings to collect per channel |
| `default_event` | `str \| None` | `None` | Event to apply if DRF has none |
| `timeout` | `float` | `5.0` | Max seconds to wait |
| `backend` | `Backend \| None` | `None` | Backend to use |

### FreshResult

Each element in the returned list is a `FreshResult`:

| Property / Method | Returns | Description |
|-------------------|---------|-------------|
| `value` | `Value` | Last reading's value |
| `values` | `list[Value]` | All ok values |
| `reading` | `Reading` | Last Reading object |
| `timestamp` | `datetime` | Last reading's timestamp |
| `timestamps` | `list[datetime]` | All timestamps |
| `length` | `int` | Number of ok values |
| `len(r)` | `int` | Total readings collected |
| `requested_count` | `int` | How many were requested |

### Windowed Statistics

Stats methods accept an optional `n` parameter:

| Call | Window |
|------|--------|
| `r.mean()` | Last `requested_count` values |
| `r.mean(-1)` | All values |
| `r.mean(8)` | Last 8 values (raises if < 8) |

Available: `mean`, `median`, `std`, `min`, `max`.

---

## watch

Block until a condition is met on a streaming channel.

```python
from pacsys.exp import watch

# Wait for temperature to exceed 80
reading = watch(
    "M:OUTTMP@p,1000",
    condition=lambda r: r.ok and r.value > 80.0,
    timeout=60.0,
)
print(f"Crossed threshold at {reading.timestamp}: {reading.value}")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `DeviceSpec` | required | DRF string or `Device` (must include a streaming event) |
| `condition` | `Callable[[Reading], bool]` | required | Predicate; returns `True` to stop |
| `timeout` | `float` | `30.0` | Max seconds to wait |
| `backend` | `Backend \| None` | `None` | Backend to use |

Returns the `Reading` that satisfied the condition. Raises `TimeoutError` if the condition is not met within the timeout.

---

## scan

Ramp a device through a series of values while reading other devices at each step. Automatically restores the original setting value on completion or error.

```python
from pacsys.exp import scan

result = scan(
    write_device="Z:ACLTST",
    read_devices=["M:OUTTMP", "G:AMANDA"],
    start=0.0,
    stop=10.0,
    steps=11,
    settle=0.5,
)

# Inspect results
for sv, step in zip(result.set_values, result.readings):
    print(f"Set {sv}: read {step['M:OUTTMP'].value}")

# Export to DataFrame
df = result.to_dataframe()
```

### Explicit Values

```python
result = scan(
    write_device="Z:ACLTST",
    read_devices=["M:OUTTMP"],
    values=[0.0, 2.5, 5.0, 7.5, 10.0],
)
```

### Averaging and Abort

```python
result = scan(
    write_device="Z:ACLTST",
    read_devices=["M:OUTTMP"],
    start=0.0, stop=10.0, steps=11,
    readings_per_step=5,  # Average 5 readings per step
    abort_if=lambda step: step["M:OUTTMP"].value > 100.0,
)
print(result.aborted)   # True if abort_if triggered
print(result.restored)  # True if original value was restored
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `write_device` | `DeviceSpec` | required | Device to ramp |
| `read_devices` | `list[DeviceSpec]` | required | Devices to read at each step |
| `values` | `list[float]` | `None` | Explicit scan values |
| `start` / `stop` / `steps` | `float` / `float` / `int` | `None` | Linear range (alternative to `values`) |
| `settle` | `float` | `0.5` | Seconds to wait after each write |
| `readings_per_step` | `int` | `1` | Readings to average per step |
| `verify` | `bool \| Verify` | `None` | Verify writes (see [Writing](../guide/writing.md)) |
| `restore` | `bool` | `True` | Restore original setting on completion |
| `abort_if` | `Callable` | `None` | Abort predicate receiving step readings |
| `timeout` | `float \| None` | `None` | Per-operation timeout |
| `backend` | `Backend \| None` | `None` | Backend to use |

### ScanResult

| Property | Type | Description |
|----------|------|-------------|
| `write_device` | `str` | DRF of the ramped device |
| `read_devices` | `list[str]` | DRFs of the read devices |
| `set_values` | `list[float]` | Values that were written |
| `readings` | `list[dict[str, Reading]]` | Per-step readings keyed by DRF |
| `write_results` | `list[WriteResult]` | Per-step write results |
| `aborted` | `bool` | Whether `abort_if` triggered |
| `restored` | `bool` | Whether original value was restored |
| `to_dataframe()` | `DataFrame` | Export with `set_value` + read columns |

---

## DataLogger

Subscribe to channels and log readings to a file via a pluggable writer. Runs in the background with periodic flushing.

```python
from pacsys.exp import DataLogger, CsvWriter

with DataLogger(["M:OUTTMP@p,1000"], writer=CsvWriter("log.csv")) as dl:
    time.sleep(60)  # Log for 60 seconds
```

### Manual Control

```python
dl = DataLogger(
    ["M:OUTTMP@p,1000", "G:AMANDA@e,8f"],
    writer=ParquetWriter("data.parquet"),
    flush_interval=10.0,
)
dl.start()
# ... do work ...
dl.stop()  # Flushes remaining data and closes the writer
```

### Constructor Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devices` | `list[DeviceSpec]` | required | Channels to subscribe to |
| `writer` | `LogWriter` | required | Writer implementation |
| `flush_interval` | `float` | `5.0` | Seconds between flushes |
| `backend` | `Backend \| None` | `None` | Backend to use |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `running` | `bool` | Whether the logger is actively collecting |
| `last_error` | `Exception \| None` | Last write error, if any |

Failed writes are retried up to 3 times before the batch is dropped. Errors are logged and available via `last_error`.

---

## Writers

Writers implement the `LogWriter` protocol:

```python
class LogWriter(Protocol):
    def write_readings(self, readings: list[Reading]) -> None: ...
    def close(self) -> None: ...
```

### CsvWriter

Simple CSV output with columns: `timestamp`, `drf`, `value`, `units`.

```python
from pacsys.exp import CsvWriter

writer = CsvWriter("output.csv")
```

### ParquetWriter

Typed columnar output using Apache Parquet (requires `pyarrow`). Handles scalars, arrays, text, raw bytes, alarms, and status values in separate typed columns.

```python
from pacsys.exp import ParquetWriter

writer = ParquetWriter("output.parquet")
```

**Schema:**

| Column | Type | Content |
|--------|------|---------|
| `timestamp` | `timestamp[us, UTC]` | Reading timestamp |
| `drf` | `string` | DRF string |
| `value_type` | `string` | Value type name |
| `value` | `float64` | Scalar values |
| `value_array` | `list<float64>` | Array values |
| `value_text` | `string` | Text, JSON-encoded alarms/status, base64-encoded raw bytes |
| `error_code` | `int16` | ACNET error code |
| `units` | `string` | Engineering units |
| `cycle` | `int64` | Cycle number |

You can also implement your own writer by conforming to the `LogWriter` protocol.

---

## See Also

- [Reading Guide](../guide/reading.md) - Basic reading operations
- [Streaming Guide](../guide/streaming.md) - Low-level streaming API
- [Writing Guide](../guide/writing.md) - Write operations and verification
- [Device API](../guide/device-api.md) - Object-oriented device interface
