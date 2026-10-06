# GETS32 and RETDAT

These utilities read raw device properties directly from a front end over an existing ACNET connection. They do not resolve device names, scale values, interpret status bits, or implement SETS32/SETDAT writes.

Use `pacsys.acnet.gets32.Gets32Client` or `pacsys.acnet.retdat.RetdatClient`. The common `ReadDevice`, `ReadValue`, and `ReadStream` types are available from both modules and from `pacsys.acnet`.

## Addressing and data

A `ReadDevice(di, pi, ssdn, length, offset=0)` specifies a property-device index pair, an eight-byte SSDN in wire order, and a byte range. Requests target one front-end node. All entries use the same acquisition event; their order and duplicates are preserved. GETS32 uses 32-bit length/offset fields; RETDAT uses 16-bit fields.

The protocols do not identify the data type; every entry is returned as raw `ReadValue(status, data)`. `data` is the complete word-aligned wire slot, **including padding for odd lengths**; on some RETDAT systems the pad byte precedes the last data byte, so do not truncate to the requested length. Data accompanying a nonzero device status may be invalid.

## Immediate reads

M:OUTTMP is a two-byte reading on MUONFE:

```python
from pacsys.acnet import AcnetConnectionTCP, Gets32Client, ReadDevice

with AcnetConnectionTCP("acsys-proxy.fnal.gov") as conn:
    node = conn.get_node("MUONFE")
    device = ReadDevice(27235, 12, bytes.fromhex("000042003f210000"), length=2)
    client = Gets32Client(conn)
    reply = client.read(node, [device], timeout=1.0)
    print(reply.header.global_status, reply.values[0].status, reply.values[0].data.hex())
```

The clients also accept `AcnetConnectionUDP`. TCP access requires a proxy that permits GETS32/RETDAT tasks; clx and many central nodes restrict these tasks. On a restricted node, use `AcnetConnectionUDP` with its local acnetd. See [TCP connection restrictions](acnet-protocol.md).

`read()` requests one batch, then cancels the request. Its timeout bounds the wait for data; setup uses the connection's command timeout. Device and GETS32 global statuses are returned in the result; negative ACNET statuses raise `AcnetError`.

## Acquisition events

GETS32 standard event strings are canonicalized before transmission:

| Event | Meaning |
|---|---|
| `I` | Immediate, one reply |
| `P,500` | Periodic, every 500 ms |
| `P,500,false` | Periodic without requesting an initial immediate notification |
| `Q,500,true` | Poll every 500 ms and return on data change |
| `E,02` | Clock event 02, canonicalized to `E,2,E,0` |
| `E,02,H,100` | Hardware clock event with a 100 ms delay |
| `E,02,S,100` | Software clock event with a 100 ms delay |
| `S,V:CLDRST,9,1000,=` | State announcement equal to 9, delayed by 1 second |
| `S,134679,0,0,*` | Any announcement from state device index 134679 |
| `N` | Never; no acquisition data is expected |

Standard clock events are hexadecimal `00`–`FF`; larger values are rejected because the reference frontend parser keeps only the low byte.

State comparisons support `=`, `!=`, `*`, `<`, `>`, `<=`, and `>=`. A state event listens to state announcements; it does not poll basic-status bits. Event support is up to the front end, and its rejections are returned as errors. `U` needs database defaults and is rejected.

For legacy or frontend-specific event strings, construct `Gets32Event(text, classic_ftd, repetitive)`; the text is sent verbatim. Use `classic_ftd=-1` when the event has no classic FTD equivalent.

RETDAT takes an explicit unsigned 16-bit `ftd`: zero is immediate, periodic FTDs are 60 Hz ticks (30 = 500 ms), and `0x8000 | event_number` selects a clock event. `stream()` sets the ACNET multiple-reply flag; `read()` clears it.

## Streams

```python
from pacsys.acnet import AcnetConnectionTCP, Gets32Client, ReadDevice

device = ReadDevice(27235, 12, bytes.fromhex("000042003f210000"), length=2)
with AcnetConnectionTCP("acsys-proxy.fnal.gov") as conn:
    client = Gets32Client(conn)
    node = conn.get_node("MUONFE")
    with client.stream(node, [device], event="P,500") as stream:
        for reply in stream.readings(timeout=5.0):
            print(reply.values)
```

For RETDAT, use `RetdatClient(conn)` and `stream(node, [device], ftd=30)`. For a single clock-triggered acquisition, use GETS32 `read(..., event="E,02")` or RETDAT `read(..., ftd=0x8002)`, allowing enough time for that event to occur.

`readings(timeout=...)` bounds the **whole iteration** from its start and raises `AcnetTimeoutError` on expiry; `timeout=None` waits until the stream ends. Only one iterator may consume a stream at a time.

The reply buffer holds 256 batches by default (`buffer_size`). Overflow cancels acquisition and raises `BufferError` once the buffered batches are consumed. `close()` discards queued batches; always close streams or use a context manager.

## Reply metadata

GETS32 preserves the global status, type/version/order fields, sequence number, and native cycle, collection, and reply time fields in `Gets32Header`. Timestamps normally use epoch milliseconds, but the cycle field can instead contain a UCD sequence number with a zero high word. The client does not guess which is present.

Both result types record the outer `acnet_status` and local callback receipt time as Unix nanoseconds (`received_at_ns`). RETDAT supplies no collection timestamp. The pure `parse_reply()` functions leave transport metadata as `None` unless supplied.

Each module also exposes `build_request()` and `parse_reply()` for direct wire work.

## Message-size limits

Both clients and both `build_request()` functions accept keyword-only `max_request_size` and `max_reply_size`, defaulting to **8,320 bytes** (the conservative DPM frontend default). Limits count GETS32/RETDAT headers, statuses, and padding, but not ACNET/UDP/IP headers. The maximums are **65,484** request and **65,488** reply bytes. Requests whose encoded or expected reply size exceeds a limit are rejected before sending.

```python
# Set limits appropriate for every destination used with this client.
client = Gets32Client(conn, max_request_size=16_384, max_reply_size=60_000)
```

For example, one GETS32 entry of 40,000 raw bytes needs a reply limit of at least **40,036 bytes** (34-byte header plus 2-byte status). Front ends may impose smaller limits, and the client does not reassemble multiple replies; split large arrays into byte-range requests.
