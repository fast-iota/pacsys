# Streaming Data

Streaming lets you receive continuous updates from devices as they change.

---

## Basic Streaming

Subscribe to a device with a periodic event:

```python
import pacsys

with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
    for reading, handle in stream.readings(timeout=30):
        print(f"{reading.name}: {reading.value}")
```

The `@p,1000` means "send data every 1000 milliseconds." The `timeout` is a **total wall-clock timeout** from the first `.readings()` call (not a per-reading idle timeout). See [DRF Events](../drf.md) for all event types.

!!! tip "Always Use Context Manager"
    The `with` statement ensures the subscription is properly cleaned up. Without it, you must call `stream.stop()` manually.

---

## Multiple Devices

Subscribe to several devices in one subscription:

```python
with pacsys.subscribe([
    "M:OUTTMP@p,1000",   # Every second
    "G:AMANDA@p,500",    # Every 500ms
]) as stream:
    for reading, handle in stream.readings(timeout=60):
        print(f"{reading.name}: {reading.value}")
```

Readings arrive interleaved from all devices.

---

## Stopping Early

Stop when a condition is met:

```python
with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
    for reading, handle in stream.readings(timeout=60):
        print(f"Temperature: {reading.value}")
        if reading.value > 100:
            stream.stop()
```

Breaking out of the `for` loop also works - the context manager calls `stop()` on exit.

---

## Callback Mode

For background streaming, use a callback instead of iteration:

```python
import pacsys
import time

def on_reading(reading, handle):
    print(f"{reading.name}: {reading.value}")
    if reading.value > 100:
        handle.stop()

handle = pacsys.subscribe(
    ["M:OUTTMP@p,1000"],
    callback=on_reading,
)

# Do other work while data streams in the background
time.sleep(30)

# Clean up
handle.stop()
pacsys.shutdown()
```

The callback runs on the receiver thread - keep it fast to avoid blocking other readings.

---

## Error Handling

### Iterator Mode

Errors during streaming raise an exception when iterating:

```python
try:
    with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
        for reading, handle in stream.readings(timeout=30):
            print(reading.value)
except Exception as e:
    print(f"Stream error: {e}")
```

Check `handle.exc` to inspect stored exceptions:

```python
with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
    for reading, handle in stream.readings(timeout=5):
        print(reading.value)

    if stream.exc:
        print(f"Stream ended with error: {stream.exc}")
```

### Callback Mode

Use `on_error` to handle connection errors:

```python
def on_error(exc, handle):
    print(f"Connection error: {exc}")

handle = pacsys.subscribe(
    ["M:OUTTMP@p,1000"],
    callback=on_reading,
    on_error=on_error,
)
```

---

## CombinedStream

Combine multiple independent subscriptions into a single iterable:

```python
from pacsys import CombinedStream

with pacsys.dpm() as backend:
    sub1 = backend.subscribe(["M:OUTTMP@p,1000"])
    sub2 = backend.subscribe(["G:AMANDA@p,500"])

    with CombinedStream([sub1, sub2]) as combined:
        for reading, handle in combined.readings(timeout=30):
            print(f"{reading.name}: {reading.value}")
```

Each `subscribe()` call creates its own TCP connection (on DPM/HTTP), so subscriptions are truly independent - stopping one doesn't affect the others.

`CombinedStream` properties:

| Property | Description |
|----------|-------------|
| `stopped` | True when all subscriptions have stopped |
| `exc` | First exception from any subscription |
| `stop()` | Stop all subscriptions |

---

## Common Patterns

### Periodic Sampling

```python
import pacsys
import time

with pacsys.subscribe(["M:OUTTMP@p,5000"]) as stream:
    start = time.time()
    for reading, _ in stream.readings(timeout=60):
        elapsed = time.time() - start
        print(f"[{elapsed:.1f}s] {reading.value}")
```

### Collecting Data

```python
import pacsys

data = []

with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
    for reading, handle in stream.readings(timeout=10):
        data.append({
            "time": reading.timestamp,
            "value": reading.value,
        })

print(f"Collected {len(data)} samples")
```

### TCLK Event Trigger

```python
# Trigger on TCLK event $0F
with pacsys.subscribe(["M:OUTTMP@E,0F"]) as stream:
    for reading, _ in stream.readings(timeout=60):
        print(f"Event triggered: {reading.value}")
```

---

## Explicit Backend

All backends support streaming with the same API:

```python
import pacsys

# DPM/HTTP
with pacsys.dpm() as backend:
    with backend.subscribe(["M:OUTTMP@p,1000"]) as stream:
        for reading, _ in stream.readings(timeout=10):
            print(reading.value)

# DMQ (requires Kerberos)
from pacsys import KerberosAuth
with pacsys.dmq(auth=KerberosAuth()) as backend:
    with backend.subscribe(["M:OUTTMP@p,1000"]) as stream:
        for reading, _ in stream.readings(timeout=10):
            print(reading.value)
```

---

## See Also

- [DRF Format](../drf.md) - Event syntax (`@p`, `@E`, `@S`)
- [Backends](../backends/index.md) - Backend streaming architecture
- [Reading Devices](reading.md) - One-shot reads
