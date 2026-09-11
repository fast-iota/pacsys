# GETS32 and RETDAT

These utilities read raw device properties directly from a front end over an existing ACNET connection. They do not resolve device names, scale values, interpret status bits, or implement SETS32/SETDAT writes.

Use `pacsys.acnet.gets32.Gets32Client` or `pacsys.acnet.retdat.RetdatClient`. The common `ReadDevice`, `ReadValue`, and `ReadStream` types are available from both modules and from `pacsys.acnet`.

## Addressing and data

A `ReadDevice(di, pi, ssdn, length, offset=0)` specifies a property-device index pair, an eight-byte SSDN in wire order, and a byte range. Requests target one front-end node. All entries use the same acquisition event; their order and duplicates are preserved. GETS32 uses 32-bit length/offset fields; RETDAT uses 16-bit fields.

The protocols do not identify the data type. Integers, floating point values, arrays, text, status words, alarm blocks, and device-specific structures are returned as `ReadValue(status, data)` objects. Each `data` contains the complete word-aligned wire slot, **including padding for odd lengths** (the RETDAT protocol notes describe systems where padding can precede the last data byte). Data accompanying a nonzero device status may be invalid; the original bytes and status are preserved.

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

`read()` requests one batch and cancels on success, timeout, or error. Its timeout bounds the wait for data after request setup; the connection's command/ACK timeout still applies to setup. Device and GETS32 global error statuses remain in the result. Negative outer ACNET statuses raise `AcnetError`.

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

Standard clock events are hexadecimal `00`–`FF`. Larger values are rejected because the reference frontend parser keeps only the low byte (`E,100` would become event `00`). Frontend-specific extended events require explicit `Gets32Event(...)` construction and support from the destination.

State comparisons support `=`, `!=`, `*`, `<`, `>`, `<=`, and `>=`. A state event listens to state announcements; it does not poll arbitrary basic-status bits. `Q` is the separate change-only acquisition mode. Support for each event is determined by the front end. There is no client-side polling emulation or protocol fallback. Client surfaces front-end rejections. Note that `U` requires database defaults and is rejected by this low-level utility.

For legacy or frontend-specific event strings, construct `Gets32Event(text, classic_ftd, repetitive)` explicitly. The text is transmitted verbatim, with protocol space padding. Use `classic_ftd=-1` when the event has no complete classic FTD representation. This escape hatch validates wire bounds and ASCII encoding, not the frontend's event grammar.

RETDAT takes an explicit unsigned 16-bit `ftd`. Zero requests immediate data; periodic FTDs specify 60 Hz ticks (30 = 500 ms); `0x8000 | event_number` selects a clock event. Legacy/reserved bits are transmitted unchanged. No GETS32 event string is silently reduced to a RETDAT event. The ACNET multiple-reply flag is set by `stream()` and cleared by `read()`, independently of the FTD.

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

`readings(timeout=...)` validates the timeout immediately and starts its deadline when iteration begins. The timeout bounds the **whole iteration**, including consumer delays and queued data. It raises `AcnetTimeoutError` on expiry; `timeout=None` waits until closure or a terminal reply. Pending heartbeats do not yield data or extend the deadline. Only one iterator may consume a stream at a time.

The reply buffer defaults to 256 batches; `buffer_size` sets its capacity. Overflow cancels acquisition and raises `BufferError` after already-buffered batches are consumed. Other terminal errors likewise follow queued data. `close()` discards queued batches and wakes blocked readers. Always close streams or use a context manager, especially when ending iteration early.

## Reply metadata

GETS32 preserves the global status, type/version/order fields, sequence number, and native cycle, collection, and reply time fields in `Gets32Header`. Timestamps normally use epoch milliseconds, but the cycle field can instead contain a UCD sequence number with a zero high word. The client does not guess which is present.

Both result types record the outer `acnet_status` and local callback receipt time as Unix nanoseconds (`received_at_ns`). RETDAT supplies no collection timestamp. The pure `parse_reply()` functions leave transport metadata as `None` unless supplied.

Each module also exposes `build_request()` and `parse_reply()` for direct wire work.

## Message-size limits

Both clients accept keyword-only `max_request_size` and `max_reply_size`, each defaulting to **8,320 bytes**, the conservative DPM frontend default. These limits include GETS32/RETDAT headers, per-device statuses, and padding, but exclude ACNET, UDP, and IP headers. The same keywords are available on both `build_request()` functions.

```python
# Set limits appropriate for every destination used with this client.
client = Gets32Client(conn, max_request_size=16_384, max_reply_size=60_000)
```

For example, one GETS32 entry containing 40,000 raw bytes needs a reply limit of at least **40,036 bytes** (34-byte reply header and 2-byte device status). Raising the reply limit does not require raising the request limit: the request itself only describes the byte range.

Limits must be positive integers. The supported transport ceilings are **65,484 request bytes** and **65,488 reply bytes**; larger configured limits are rejected at construction or encoding. These account for the local UDP send-with-timeout command and acnetd's network packet limit, respectively. Requests are never sent when their encoded size or expected reply size exceeds the configured limit.

Frontends may impose smaller limits. Raising a client limit does not negotiate support with the destination or assemble multiple replies automatically. Arrays that exceed the destination's capacity need explicit byte-range requests.
