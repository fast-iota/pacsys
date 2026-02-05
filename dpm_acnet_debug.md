# DPM/ACNET Protocol Debug Findings

## Protocol Flow (from reference acsys-python and acnetd source)

### Connection Establishment
1. TCP connect to `acsys-proxy.fnal.gov:6802`
2. Send `RAW\r\n\r\n` handshake
3. Send `CMD_CONNECT` (cmd=1) → get handle in ACK
4. Resolve DPM node: `get_node("DPM06")` → node address 3510

### Open List (persistent streaming channel)
1. Send `SEND_REQUEST` (cmd=18) to `DPMD@DPM06` with `mult=1` (multiple-reply), `timeout=1000ms`
2. ACK returns `reqid` — this is the persistent channel ID
3. DPM sends `OpenList_reply` as a DATA packet on this channel
4. Reference acsys-python calls `_ack_request(reqid)` after registering handler (see below)
5. Channel stays open — all DPM data replies (readings, device info, status) arrive here

### AddToList / StartList / StopList (one-shot commands)
1. Each sent as separate `SEND_REQUEST` (cmd=18) with `mult=0` (single-reply), `timeout=1000ms`
2. ACK returns `reqid`, then a single DATA reply with the response
3. These are independent ACNET requests — not on the open list channel

### Data Flow
- DPM sends readings on the **open list channel** (the persistent `mult=1` request)
- Each reading is a DATA packet routed by `reqid` to the registered handler
- `DeviceInfo_reply`, `Scalar_reply`, `ScalarArray_reply`, `Status_reply` etc. all arrive here

## CMD_REQUEST_ACK (cmd=9) — What It Actually Does

### Reference acsys-python (init_file.py:734)
```python
self.protocol.add_handler(reqid, handler)
await self._ack_request(reqid)  # CMD_REQUEST_ACK with reqid
```
Called after registering handler for `request_stream` (mult=1 requests).

### acnetd server (exttask.cpp:306-329)
```cpp
void ExternalTask::handleRequestAck(RequestAckCommand const *cmd)
{
    RpyInfo* rep = taskPool().rpyPool.rpyInfo(cmd->rpyid());
    if (rep && rep->task().equals(this)) {
        rep->ackIt();                    // sets acked=true
        decrementPendingRequests();      // throttle counter
    }
}
```

### Key Finding: CMD_REQUEST_ACK is for INCOMING requests, not outgoing
- The `rpyid` in CMD_REQUEST_ACK refers to an **incoming request** that this connection is **handling** (replying to)
- It tells acnetd "I've processed this request, decrement my pending counter"
- The reference acsys-python uses it in `request_stream` context because the DPM stream acts as a receiver
- **It is NOT required for outgoing requests to work**

### Auto-ack on first reply (rpyinfo.cpp:21-35)
```cpp
bool RpyInfo::xmitReply(...) {
    if (!beenAcked()) {
        ackIt();
        task().decrementPendingRequests();
        syslog(LOG_WARNING, "un-acked request ...");  // warning only
    }
    ...
}
```
- If CMD_REQUEST_ACK was never sent, first `SEND_REPLY` auto-acks with a log warning
- Data flows normally either way — CMD_REQUEST_ACK is bookkeeping, not gating

### Throttling (taskinfo.cpp:39-51)
- `pendingRequests` counter tracks un-acked incoming requests
- Limit is 256 — after that, new incoming requests get `ACNET_NOREMMEM`
- CMD_REQUEST_ACK decrements this counter
- Only matters for connections receiving many concurrent requests

## Our DPMAcnet Implementation Issues

### Issue 1: `read()` doesn't clear list between calls
- `read()` calls `add_entry(tag, drf)` → `start()` → read → `stop()`
- But entries accumulate — same tag on repeat reads causes DPM errors
- Need to either `clear_list()` between reads or use unique tags

### Issue 2: Intermittent ACK timeouts after rapid connect/disconnect
- After rapid TCP connection churn (e.g., test_acnet_tcp runs first), new DPMAcnet connections sometimes fail
- `_xact` sends SEND_REQUEST but ACK never arrives (5s timeout)
- The TCP read thread is running, socket is connected — but ACK doesn't come back
- This is intermittent and appears load/timing-dependent
- **Not reproducible in isolation** — only after prior connection churn
- Possible causes:
  - acsys-proxy rate limiting or connection pooling artifacts
  - acnetd handle allocation delays under rapid churn
  - TCP socket buffering/ordering issues

### Issue 3: Wrong attempt to use CMD_REQUEST_ACK for outgoing requests
- We attempted to send CMD_REQUEST_ACK with the outgoing `reqid` after `send_request(mult=1)`
- This is incorrect — the server interprets `rpyid` as an incoming request ID
- Server returns `ACNET_NSR` (no such request) since the ID doesn't match any incoming request
- The failed `_xact` for the ACK then corrupts the ACK queue, causing subsequent commands to fail
- **Fix: Do not send CMD_REQUEST_ACK for outgoing requests. It's not needed.**

## TCP Message Types (wire format)
| Type | Name | Description |
|------|------|-------------|
| 0 | PING | Keepalive from server |
| 1 | COMMAND | Client→server command |
| 2 | ACK | Server→client acknowledgement for a command |
| 3 | DATA | Server→client ACNET packet (request, reply, message, cancel) |

## ACNET Command Codes (sent as type=1)
| Code | Name | Description |
|------|------|-------------|
| 1 | CONNECT | Register with acnetd, get handle |
| 3 | DISCONNECT | Close connection |
| 8 | CANCEL | Cancel an outgoing request |
| 9 | REQUEST_ACK | Acknowledge an incoming request (throttle management) |
| 11 | NAME_LOOKUP | Resolve node name → address |
| 18 | SEND_REQUEST_TIMEOUT | Send request with timeout |

## Error Codes Encountered
| Code | Hex | Facility | Error | Name | Meaning |
|------|-----|----------|-------|------|---------|
| -1535 | 0xfa01 | 1 (ACNET) | -6 | ACNET_REQTMO | Request timeout (6.5 min) |
| -7167 | 0xe401 | 1 (ACNET) | -28 | (unnamed) | Unknown ACNET error |
| -6127 | 0xe811 | ? | ? | ? | DPM rejection |
