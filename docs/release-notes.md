# Release notes

## 0.3.0 (unreleased)

Changes since 0.2.2. Python 3.10 or newer is required.

This release adds typed async devices, monitoring health checks, ACNET UDP support,
direct GETS32/RETDAT front-end reads, and Booster skew-quadrupole ramps. It also improves write verification, subscription
shutdown, and timeout handling. Review the notes before upgrading.

### Upgrading from 0.2.2

#### Installation

Kerberos and SSH dependencies are now optional. Install the extra needed by your application:

```bash
pip install --upgrade pacsys               # DPM/gRPC/ACL reads and gRPC token writes
pip install --upgrade "pacsys[kerberos]"   # Also enables Kerberos writes, DMQ, and SSH
pip install --upgrade "pacsys[all]"        # Kerberos, Parquet, and MCP dependencies
```

DMQ requires Kerberos for reads as well as writes. Missing optional dependencies
raise an installation error when the relevant functionality is used.

#### API replacements

| Removed API | Replacement |
|-------------|-------------|
| `pacsys.ssh(...)` | `pacsys.SSHClient(...)` |
| `pacsys.devdb(...)` | `pacsys.DevDBClient(...)` |
| `pacsys.supervised(...)` | `pacsys.SupervisedServer(...)` |
| `pacsys.acnet.AcnetConnection` | `AcnetConnectionTCP` or `AcnetConnectionUDP`, imported from `pacsys.acnet` |
| `SlewRatePolicy` and `SlewLimit` | No built-in replacement; implement a custom `Policy` if slew limiting is required |

`pacsys.ssh`, `pacsys.devdb`, and `pacsys.supervised` now resolve to their public
submodules. `pacsys.dpm()`, `pacsys.dpm_http()`, `pacsys.grpc()`, `pacsys.dmq()`, and
`pacsys.acl()` remain backend factories.

Replace `AcnetConnection("MYTASK")` with `AcnetConnectionUDP(name="MYTASK")`:
the first positional argument is now `host`. UDP requires a local acnetd;
use `AcnetConnectionTCP` for remote connections.

#### Endpoints and configuration

- **gRPC DAQ:** the default changes from `localhost:23456` to
  `dce08.fnal.gov:50051`. Set `host`/`port` or `PACSYS_GRPC_HOST`/`PACSYS_GRPC_PORT`
  explicitly when using a local proxy or tunnel.
- **DevDB:** the default is now `ad-services.fnal.gov:443` with TLS. Supply a
  hostname without a URL path. For a plaintext tunnel, pass `tls=False` or set
  `PACSYS_DEVDB_TLS=0`. The positional order remains
  `DevDBClient(host, port, timeout, cache_ttl)`; `tls` is keyword-only.
- **DPM factories:** `dpm()`, `dpm_http()`, and `aio.dpm()` use
  `PACSYS_DPM_HOST`, `PACSYS_DPM_PORT`, `PACSYS_POOL_SIZE`, and `PACSYS_TIMEOUT`
  when the corresponding arguments are omitted.
- **Backend binding:** module-level operations honor the backend attached to a
  `Device`. A batch must use devices bound to the same backend, or entirely unbound
  targets. Mixing bound devices with bare DRFs or another backend raises before I/O.
- **Validation:** invalid configuration is rejected before replacing the current
  backend. Malformed ACNET DRFs, invalid ranges, and out-of-range timing values now
  raise instead of being silently accepted or reinterpreted.

#### Writes and policies

`WriteResult.confirmed` is true when the write succeeded and any requested
verification did not fail. It does **not** imply that verification was requested:
`verified=None` means no readback check ran; `verified=True` means readback matched.

Module-level `pacsys.write(..., BasicControl.ON)` targets the CONTROL property.
For `Device` and `AsyncDevice`, use `control()` or shortcuts such as `on()` and
`reset()`; passing a `BasicControl` enum to `write()` raises `TypeError`.

For range-limited devices, `ValueRangePolicy` rejects raw bytes and
RAW/PRIMARY/VOLTS writes unless the device has an explicit `allow_raw` exemption.
All matching value ranges are intersected. CONTROL commands are governed by device
access policies, not numeric ranges. MCP reads follow read policies; invalid MCP
configuration is rejected at startup, and audit failures block writes. See
[supervised mode](specialized-utils/supervised.md) and [MCP configuration](specialized-utils/mcp-server.md).

#### Results and experiment workflows

NumPy arrays in `Reading.value` and `WriteResult.readback` are now read-only.
Constructing either result with an owning array freezes that caller-owned array
in place; views are copied. Use `.copy()` when you need a mutable array.

`DataLogger.dropped_count` counts lost **readings**, and `failed` indicates data loss.
After readings are dropped, `stop()` and normal context-manager exit raise
`RuntimeError`. A failure restoring settings after a completed scan raises
`ScanRestoreError` with the collected result; failures during the scan remain the
primary error.

Raw byte values in `acget --format json` output are now base64 strings.

### Additions and improvements

- **Typed async devices:** `AsyncScalarDevice`, `AsyncArrayDevice`, and
  `AsyncTextDevice`, plus `AsyncDevice.await_next()`. Fluent methods preserve the
  device subclass in both sync and async APIs.
- **Batch reads:** `read_many()` is now available directly on sync and async
  backends. Unusable readings raise `ReadError`, which retains the batch readings.
- **Monitoring:** `Monitor.health()`, stale/recovery callbacks, per-element array
  statistics, and dictionary export. `MonitorResult` supports channel lookup and iteration.
- **Scans:** `scan(values=...)` accepts generators and NumPy arrays.
- **Serialization:** `Reading`, `WriteResult`, and `DeviceMeta` support dictionary
  round trips and value-based equality. Array-valued results retain their types.
- **Errors:** `PacsysError` provides a common base for library-specific exceptions.
- **Direct front-end reads:** low-level `Gets32Client` and `RetdatClient` support
  one-shot and streaming raw reads. See [GETS32 and RETDAT](frontend-read-protocols.md).
- **ACNET and ramps:** FTPMAN snapshots over local UDP, and
  `BoosterSQRamp`/`BoosterSQRampGroup` (thanks M. Balcewicz for reporting)

### Correctness and reliability

- **DPM:** reads and writes enforce end-to-end deadlines. Once sending settings
  starts, connection failures are not retried automatically; missing acknowledgements
  report "outcome unknown" because the setting may already have executed. Repeated
  LOGGER/LOGGERDURATION reads now retrieve the requested window reliably (thanks R. Santucci for reporting).
- **Streaming:** caller-initiated stop suppresses queued reading callbacks, and
  iterators honor timeouts even with queued data. DMQ aborted batches no longer send
  queued settings later.
- **Data handling:** gRPC warnings carrying data remain usable, including in batch
  and logger aggregation. ACL raw reads now produce the correct little-endian bytes.
- **Alarm edits:** server-scaled limits are preserved. Combining engineering limits
  with mode, data type, or data length changes is rejected before writing, as is
  structured access to nonzero alarm segments.

### Known limitations

- Gracefully closing a low-level ACNET connection can leave incoming requests
  marked active. This affects applications serving requests through `handle_requests()`.
- `Monitor.collect(count=...)` can fail prematurely when a finite stream ends while
  its final reading callbacks are still queued.
