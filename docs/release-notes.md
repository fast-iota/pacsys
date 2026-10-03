# Release notes

## 0.3.0 (2026-10-05)

This release adds typed async devices, monitoring health checks, ACNET UDP support, direct GETS32/RETDAT front-end reads, and Booster skew-quadrupole ramps. It also improves write verification, subscription shutdown, and timeout handling.

### Upgrading from 0.2.2

#### Installation

`grpcio >= 1.80.0` and `Paramiko >= 3.2, < 5` are now required. Kerberos and SSH dependencies remain part of the default installation:

```bash
pip install --upgrade pacsys           # Includes Kerberos and SSH dependencies
pip install --upgrade "pacsys[all]"    # + Parquet and MCP dependencies
```

Kerberos credentials are still required for DPM writes, all DMQ operations, and SSH with GSSAPI authentication.

#### API replacements

| Removed API | Replacement |
|-------------|-------------|
| `pacsys.ssh(...)` | `pacsys.SSHClient(...)` |
| `pacsys.devdb(...)` | `pacsys.DevDBClient(...)` |
| `pacsys.supervised(...)` | `pacsys.SupervisedServer(...)` |
| `pacsys.acnet.AcnetConnection` | `AcnetConnectionTCP` or `AcnetConnectionUDP`, imported from `pacsys.acnet` |
| `SlewRatePolicy` and `SlewLimit` | Implement a custom `Policy` if slew limiting is required |

`pacsys.ssh`, `pacsys.devdb`, and `pacsys.supervised` now resolve to public submodules. `pacsys.dpm()`, `pacsys.dpm_http()`, `pacsys.grpc()`, `pacsys.dmq()`, and `pacsys.acl()` remain backend factories.

Replace `AcnetConnection("MYTASK")` with `AcnetConnectionUDP(name="MYTASK")`: the first positional argument is now `host`. UDP requires a local acnetd; use `AcnetConnectionTCP` for remote connections.

`Scaler.from_property_info(prop, input_len)` now requires the raw data width in bytes (1, 2, or 4). `PropertyInfo` does not provide it.

The optional MCP server now requires `mcp >= 2.2, < 3`. `create_server()` now returns an `MCPServer`; direct callers must pass the SSE port to `server.run(transport="sse", port=...)`.

#### Endpoints and configuration

- **gRPC DAQ:** the default is now `dce08.fnal.gov:50051`. Set `host`/`port` or `PACSYS_GRPC_HOST`/`PACSYS_GRPC_PORT` for a local proxy or tunnel.
- **DevDB:** the default is now `ad-services.fnal.gov:443` with TLS. Supply a hostname without a URL path. For plaintext tunnels, pass `tls=False` or set `PACSYS_DEVDB_TLS=0`. The positional order remains `DevDBClient(host, port, timeout, cache_ttl)`; `tls` is keyword-only.
- **DPM factories:** `dpm()`, `dpm_http()`, and `aio.dpm()` use `PACSYS_DPM_HOST`, `PACSYS_DPM_PORT`, `PACSYS_POOL_SIZE`, and `PACSYS_TIMEOUT` for omitted arguments.
- **Backend binding:** module-level operations honor a `Device`'s backend. Batches must contain devices bound to one backend or entirely unbound targets; mixing bound devices with bare DRFs or another backend raises before I/O.
- **Validation:** invalid configuration is rejected before replacing the backend or acquiring credentials. Malformed ACNET DRFs, invalid fields and ranges, and out-of-range timing values now raise. `VOLTS`/`COMMON` normalize to `PRIMARY`/`SCALED`, preserving write units.
- **Device identity:** canonical ACNET names are uppercase, so case variants share identity and fake-backend data. EPICS names retain case. `digital_status()` rejects non-ACNET devices before lookup.

#### Writes and policies

`WriteResult.confirmed` means the write succeeded and any requested verification did not fail. It does **not** imply verification was requested: `verified=None` means no readback check ran; `verified=True` means readback matched.

Module-level `pacsys.write(..., BasicControl.ON)` and `BasicControl` values in mixed `write_ramps()` settings target CONTROL. For `Device` and `AsyncDevice`, use `control()` or shortcuts such as `on()` and `reset()`; `write(BasicControl.ON)` raises `TypeError`.

For range-limited devices, `ValueRangePolicy` rejects raw bytes and RAW/PRIMARY/VOLTS writes without an explicit `allow_raw` exemption. Matching ranges are intersected. CONTROL commands follow device access policies. Invalid MCP configuration fails at startup, and audit failures block writes. MCP tools reject device-index aliases (`0:1234`, `#:1234`). See [supervised mode](specialized-utils/supervised.md) and [MCP configuration](specialized-utils/mcp-server.md).

`acput --verify` preserves the requested setting, basic control, or single alarm field. Unsupported targets, including whole alarm blocks and ACNET RAW fields, reject the entire command before any write.

DMQ scalar integer settings must fit signed 32-bit values; out-of-range values raise before I/O. DPM text writes reject non-Latin-1 characters before connection setup. gRPC writes reject reserved DPM list directives per item, preserving batch result alignment.

#### Results and experiment workflows

- **CSV:** `CsvWriter` appends `facility_code`, `error_code`, and `message` to the existing four columns and always writes UTF-8. Update consumers requiring an exact header or column count.
- **Parquet:** `ParquetWriter` inserts `int_value` (`int64`) after `value` and appends `facility_code` (`int16`) and `message` (`string`) after `cycle`. Integer and boolean scalars are stored in `int_value` with `value` null; coalesce the two columns. Update exact-schema and positional consumers; mixed old/new datasets may need explicit schema unification.
- **Arrays:** NumPy arrays in `Reading.value` and `WriteResult.readback` are read-only. Construction freezes caller-owned owning arrays in place; views are copied. Use `.copy()` for mutable data.
- **Data loss:** `DataLogger.dropped_count` counts lost **readings**, and `failed` indicates data loss. Dropped readings cause `stop()` and normal context-manager exit to raise `RuntimeError`.
- **Scans:** restoration failure after a completed scan raises `ScanRestoreError` with the collected result; errors during the scan remain primary. Resolved read DRFs are deduplicated in first-seen order; use `readings_per_step` for repeated sampling. ACNET write targets must use READING or SETTING; EPICS PVs remain supported.
- **Ramps:** the default is 15 physical slots (indices 0–14); subclasses can override `MAX_SLOTS`.
- **Fake backends:** partial ranged writes fail unless a whole value was seeded with `set_reading()`, even with a configured successful write result. Empty subscriptions raise `ValueError`.
- **Low-level ACNET:** `FTPStream` batches are keyed by position in the device list passed to `start_continuous()`, not by device index. `DPMAcnet` returns array data as NumPy arrays instead of lists.
- **CLI:** `acget --format json` represents raw bytes as base64 strings.

### Additions and improvements

- **Async devices:** `AsyncScalarDevice`, `AsyncArrayDevice`, `AsyncTextDevice`, and `AsyncDevice.await_next()`. Fluent methods preserve subclasses in sync and async APIs.
- **Batch reads:** sync and async backends expose `read_many()`. Unusable readings raise `ReadError`, retaining the batch readings.
- **Monitoring:** `Monitor.health()`, stale/recovery callbacks, per-element array statistics, dictionary export, and result channel lookup and iteration.
- **Scans:** `scan(values=...)` accepts generators and NumPy arrays.
- **Results and errors:** `Reading`, `WriteResult`, and `DeviceMeta` support dictionary round trips and value-based equality, preserving array types. `PacsysError` is a common exception base; `ReadError` and `DeviceError` preserve details across processes.
- **Front-end reads:** `Gets32Client` and `RetdatClient` support one-shot and streaming raw reads. See [GETS32 and RETDAT](frontend-read-protocols.md).
- **ACNET and ramps:** FTPMAN snapshots over local UDP, and `BoosterSQRamp`/`BoosterSQRampGroup` (thanks M. Balcewicz for reporting). `RampGroup` accepts numeric nested lists; summaries count points with nonzero delta time as active.
- **[Ramp serialization](specialized-utils/ramps.md#serialization):** `Ramp` and `RampGroup` support JSON-safe `to_dict()`/`from_dict()` round trips that restore built-in subclasses.
- **Mixed ramp writes:** `write_ramps()` accepts ramps, groups, and `(drf, value)` settings in one backend call, returning results in flattened input order. Batches are not atomic; check each result.
- **[Active-span ramp writes](specialized-utils/ramps.md#active-span-writes-advanced):** `write_mode="active"` sends only the active nonzero subset. Full writes remain the default and are required to clear shortened or empty ramps.

### Correctness and reliability

#### Backends and transport

- **gRPC keepalive:** five-minute pings only during active RPCs avoid excessive-ping disconnects on quiet streams under the default server policy. Silent connection loss can take about five minutes plus ten seconds to detect; ordinary RPC timeouts are unchanged.
- **gRPC reads and values:** one-shot reads normalize omitted and `@U` events to `@I`, preserving explicit non-default events and historical requests. Malformed DRFs reject the batch before the Read RPC. Warnings carrying data remain usable in batch and logger aggregation; uneven logger array records fail only the affected device. Status readings preserve display text, and writes support one-dimensional NumPy string arrays.
- **DPM deadlines and errors:** reads and writes enforce end-to-end deadlines. Once sending settings starts, connection failures are not retried; missing acknowledgements report "outcome unknown" because settings may have executed. Authentication and setup failures preserve server status codes.
- **DPM authentication:** async Kerberos initialization keeps the event loop responsive and stops waiting at the authentication deadline.
- **DPM history and text:** repeated LOGGER/LOGGERDURATION reads retrieve the requested window reliably (thanks R. Santucci for reporting). Device read helpers preserve logger events; historical arrays retain rows and microsecond timestamps. DPM and DMQ preserve non-ASCII units and descriptions.
- **DMQ lifecycle:** startup failures, missing heartbeats, and connection closure surface promptly as errors. Unexpected subscription closures identify affected devices in logs; stopping a subscription preserves the shared connection.
- **DMQ writes:** initialization honors the caller's timeout. Queued writes survive idle cleanup, and aborted batches do not send queued settings later.
- **Low-level ACNET:** handshake and read failures preserve server status codes; setup timeouts consistently raise `AcnetUnavailableError` on Python 3.10. FTP streams and event-armed snapshots tolerate empty heartbeats and preserve front-end rejection status.
- **ACL reads:** raw reads return correct little-endian bytes; basic-status fields parse as booleans and preserve clock events and delays.
- **SSH:** private keys load by type, channel setup and command start are bounded by `timeout`, and failed interactive sessions are cleaned up. ACL script cleanup reports transport failures without masking the original result or error; see [SSH error handling](guide/ssh.md#acl-error-handling).

#### Streaming and cleanup

- Async DPM and fake backend iterator subscriptions report stream failures through both `on_error` and `readings()`. Error notifications are retained even when the reading queue is full.
- Caller-initiated stop suppresses queued reading callbacks, including remaining duplicate DMQ deliveries. Iterators honor timeouts even with queued data.
- `watch`, `read_fresh`, and `DataLogger` continue through recoverable subscription errors. Terminal logger failures set `failed`/`last_error` and raise from `stop()` after writer cleanup.
- Monitor ignores recoverable retry notifications. After terminal stream failure, it continues reporting stale channels and notifies waiting readers.
- gRPC, DPM HTTP, and synchronous ACNET deliver results and errors completed during shutdown to waiting callers. Interrupted cleanup releases pooled connections and cancels requests.
- Supervised streams support timed arrays, basic-status maps, and one-dimensional NumPy string arrays; individual encoding failures do not abort the batch. Shutdown gives pending cleanup time to finish and logs unfinished cleanup. The caller retains ownership of the backend.

#### Devices and data

- **Write verification:** sync and async readback `ReadError` counts as a failed attempt, allowing remaining configured attempts after a successful write.
- **Logging:** CSV and Parquet writers accept historical timed arrays. CSV batch conversion finishes before rows are written, and a batch that fails mid-write is removed, so `DataLogger` retries never duplicate rows.
- **Diagnostics:** CLI monitoring and MCP serialization preserve status diagnostics.
- **Scaling:** transforms 48 and 68 return NaN for fractional powers of negative bases, allowing array processing to continue. Analytic inverse transforms no longer require limits needed only for numerical searches.
- **Alarm edits:** server-scaled limits are preserved, and backend-provided readings stay unchanged even when `modify()` raises. Blocks distinguish nominal and percent tolerance; conflicting engineering-limit and format/mode changes are rejected before writing. `AnalogAlarm.write()` raises `ValueError` for `minimum`/`maximum` edits instead of silently dropping them; use `modify()` for limits.
