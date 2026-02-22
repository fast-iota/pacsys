# Quick Start Guide

## Prerequisites

1. **Network access** to Fermilab (on-site, VPN, or tunnel)
2. **Python 3.10+**
3. For **write operations**: Valid Kerberos credentials

## Installation

```bash
pip install pacsys
```

### Development Install

```bash
git clone https://github.com/fast-iota/pacsys.git
cd pacsys
pip install -e ".[dev]"
```

---

## Read a Device

=== "Sync"

    ```python
    import pacsys

    # Simple value
    temperature = pacsys.read("M:OUTTMP")
    print(f"Temperature: {temperature}")

    # With metadata
    reading = pacsys.get("M:OUTTMP")
    print(f"{reading.value} {reading.units}")  # e.g. "72.5 DegF"

    # Multiple devices at once
    readings = pacsys.get_many(["M:OUTTMP", "G:AMANDA"])
    for r in readings:
        print(f"{r.name}: {r.value}")
    ```

=== "Async"

    ```python
    import pacsys.aio as pacsys

    # Simple value
    temperature = await pacsys.read("M:OUTTMP")
    print(f"Temperature: {temperature}")

    # With metadata
    reading = await pacsys.get("M:OUTTMP")
    print(f"{reading.value} {reading.units}")  # e.g. "72.5 DegF"

    # Multiple devices at once
    readings = await pacsys.get_many(["M:OUTTMP", "G:AMANDA"])
    for r in readings:
        print(f"{r.name}: {r.value}")
    ```

`read()` raises `DeviceError` on failure. `get()` returns a `Reading` object you can inspect with `reading.is_error`.

:material-arrow-right: [Reading Guide](guide/reading.md) - all property types, value types, error handling

---

## Stream Data

=== "Sync"

    ```python
    import pacsys

    with pacsys.subscribe(["M:OUTTMP@p,1000"]) as stream:
        for reading, handle in stream.readings(timeout=30):
            print(f"{reading.name}: {reading.value}")
            if reading.value > 100:
                stream.stop()
    ```

=== "Async"

    ```python
    import pacsys.aio as pacsys

    stream = await pacsys.subscribe(["M:OUTTMP@p,1000"])
    async for reading, handle in stream.readings(timeout=30):
        print(f"{reading.name}: {reading.value}")
        if reading.value > 100:
            handle.stop()
    ```

The `@p,1000` means "send data every 1000 milliseconds". For streaming, a repeating event type must be specified.

Or use the Device API:

=== "Sync"

    ```python
    import pacsys

    dev = pacsys.Device("M:OUTTMP@p,1000")
    with dev.subscribe() as stream:
        for reading, _ in stream.readings(timeout=30):
            print(reading.value)
    ```

=== "Async"

    ```python
    from pacsys.aio import Device

    dev = Device("M:OUTTMP@p,1000")
    stream = await dev.subscribe()
    async for reading, _ in stream.readings(timeout=30):
        print(reading.value)
    ```

:material-arrow-right: [Streaming Guide](guide/streaming.md) - callbacks, CombinedStream, error handling

---

## Write a Value

=== "Sync"

    ```python
    import pacsys

    auth = pacsys.KerberosAuth()  # requires kinit beforehand

    with pacsys.dpm(auth=auth, role="testing") as backend:
        result = backend.write("Z:ACLTST", 45.0)
        if result.success:
            print("Write successful")
    ```

=== "Async"

    ```python
    from pacsys import KerberosAuth
    import pacsys.aio as aio

    auth = KerberosAuth()  # requires kinit beforehand

    async with aio.dpm(auth=auth, role="testing") as backend:
        result = await backend.write("Z:ACLTST", 45.0)
        if result.success:
            print("Write successful")
    ```

PACSys automatically converts READING to SETTING and STATUS to CONTROL when writing.

:material-arrow-right: [Writing Guide](guide/writing.md) - all value types, batch writes, alarm config, device control

---

## Choosing a Backend

| Backend | Factory | Protocol | Auth | Read | Write | Stream |
|---------|---------|----------|------|:----:|:-----:|:------:|
| **DPM/HTTP** | `pacsys.dpm()` | TCP + SDD | Kerberos (writes) | :material-check: | :material-check: | :material-check: |
| **DPM/gRPC** | `pacsys.grpc()` | TCP + gRPC | JWT (writes) | :material-check: | :material-check: | :material-check: |
| **DMQ** | `pacsys.dmq()` | TCP + AMQP + SDD | Kerberos (all) | :material-check: | :material-check: | :material-check: |
| **ACL/HTTP** | `pacsys.acl()` | TCP + HTTP/CGI | None | :material-check: | - | - |

```python
# Default (DPM/HTTP, implicit)
value = pacsys.read("M:OUTTMP")

# Explicit backend
with pacsys.dmq(auth=pacsys.KerberosAuth()) as backend:
    value = backend.read("M:OUTTMP")
```

Use `configure()` to change the default backend and settings for all top-level calls:

```python
pacsys.configure(backend="grpc", default_timeout=10.0)

# Now all top-level calls use gRPC
value = pacsys.read("M:OUTTMP")
```

`configure()` can be called at any time — if a backend is already running, it will be
automatically shut down and replaced on the next operation. See the full list of options in the [API Reference](api.md).

:material-arrow-right: [Backends](backends/index.md) - architecture, configuration, comparison

---

## Cleanup

Resources are automatically cleaned up when the Python process exits. No explicit
`shutdown()` call is needed for normal usage:

```python
value = pacsys.read("M:OUTTMP")
# That's it - connections close automatically at exit
```

Use `shutdown()` only if you need to explicitly close connections mid-process
(e.g., before exiting a long-running script). `configure()` handles shutdown
automatically when reconfiguring.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PACSYS_DPM_HOST` | DPM proxy hostname | acsys-proxy.fnal.gov |
| `PACSYS_DPM_PORT` | DPM proxy port | 6802 |
| `PACSYS_TIMEOUT` | Default timeout (seconds) | 5.0 |
| `PACSYS_JWT_TOKEN` | JWT token for gRPC auth | - |
| `PACSYS_DMQ_HOST` | RabbitMQ broker host | appsrv2.fnal.gov |
| `PACSYS_DMQ_PORT` | RabbitMQ broker port | 5672 |
| `PACSYS_POOL_SIZE` | DPM connection pool size | 4 |
| `PACSYS_DEVDB_HOST` | DevDB gRPC hostname | ad-services.fnal.gov/services.devdb |
| `PACSYS_DEVDB_PORT` | DevDB gRPC port | 6802 |
| `PACSYS_ACL_URL` | ACL CGI base URL | https://www-bd.fnal.gov/cgi-bin/acl.pl |

---

## Next Steps

- [Reading Guide](guide/reading.md) - All property types and value types
- [Writing Guide](guide/writing.md) - Writes, control commands, alarm config
- [Streaming Guide](guide/streaming.md) - Continuous data, callbacks, combined streams
- [Device Status](guide/status.md) - Digital status bits and control
- [Device API](guide/device-api.md) - Object-oriented device access
- [DRF Format](drf.md) - Device Request Format reference
- [Backends](backends/index.md) - Backend architecture and comparison
- [API Reference](api.md) - Complete API documentation
