# Low-Level ACNET

This page documents the low-level ACNET protocol for advanced users who need to communicate directly with frontends or implement custom protocols.

## ACNET Concepts

When discussing ACNET, people generally refer to the entire control system infrastructure - protocols, frontends, and central services. Here we focus specifically on the **ACNET UDP packet protocol**.

**ACNET** is a UDP-based mesh protocol that passes messages between **tasks** running on **nodes**.

A **node** is a computer (physical or VM) that runs either:
- `acnetd` - the ACNET daemon (on central services and some frontends)
- Frontend application code (on VME crates, PLCs, etc.)

Each node has a unique **address** encoded as `trunk:node` (two bytes). All nodes maintain tables mapping node addresses to IP addresses.

A **task** is a named process that can send/receive ACNET messages. Task names are 6-character strings encoded using **RAD50** (a base-40 encoding that packs 6 chars into 32 bits).

## Packet Structure

ACNET packets have an 18-byte header followed by variable-length payload:

| Offset | Size | Endian | Field | Description |
|--------|------|--------|-------|-------------|
| 0 | 2 | LE | flags | Message type and control flags |
| 2 | 2 | LE | status | Status/error code (signed) |
| 4 | 2 | BE | server | Server node (trunk:node) |
| 6 | 2 | BE | client | Client node (trunk:node) |
| 8 | 4 | LE | serverTask | Server task name (RAD50) |
| 12 | 2 | LE | clientTaskId | Client task identifier |
| 14 | 2 | LE | id | Message/request ID |
| 16 | 2 | LE | length | Total packet length |
| 18+ | var | LE | data | Payload |

Note the mixed endianness: node addresses are big-endian, everything else is little-endian.

## Message Types

The `flags` field determines the message type:

| Type | Flag Value | Description |
|------|------------|-------------|
| USM | `0x00` | Unsolicited message (no reply expected) |
| Request | `0x02` | Request (expects reply) |
| Reply | `0x04` | Reply to a request |
| Cancel | `0x0200` | Cancel an outstanding request |

Additional flag bits:
- `0x01` (MLT) - Multiple replies expected/following
- Upper nibble - Reply sequence number (for detecting missed replies)

## Request/Reply Flow

```
Client                          Server
  │                               │
  │──── Request (flags=0x02) ────►│
  │                               │
  │◄─── Reply (flags=0x04) ───────│
  │                               │
```

For multiple-reply requests (MLT flag set):
```
Client                          Server
  │                               │
  │── Request (flags=0x03) ──────►│  (0x02 | 0x01 = request + MLT)
  │                               │
  │◄── Reply (flags=0x05) ────────│  (0x04 | 0x01 = reply + more coming)
  │◄── Reply (flags=0x05) ────────│
  │◄── Reply (flags=0x04) ────────│  (MLT clear = final reply)
  │                               │
```

## Node Addressing

Node addresses are 16-bit values: `(trunk << 8) | node`

Examples:
- `0x0A06` = trunk 10, node 6 (displayed as "A06" in hex)
- `0x09CC` = trunk 9, node 204 (displayed as "9CC")

Trunk 230 (`0xE6`) is reserved for pseudo-nodes (open-access clients).

## RAD50 Encoding

Task names use RAD50, a base-40 encoding that fits 6 characters into 32 bits:

```python
from pacsys.acnet import rad50

# Encode task name
encoded = rad50.encode("DPMD")    # -> 0x000004A3

# Decode back
name = rad50.decode(0x000004A3)   # -> "DPMD  "
```

Character set (40 chars): ` ABCDEFGHIJKLMNOPQRSTUVWXYZ$.%0123456789`

## Status Codes

The `status` field uses facility-error encoding:
- Low byte: facility code (unsigned)
- High byte: error code (signed)
- Negative = failure
- Zero = success
- Positive = conditional success (e.g., pending)

## Using pacsys.acnet

PACSys provides low-level ACNET access via the `pacsys.acnet` module:

```python
from pacsys.acnet import AcnetPacket
from pacsys.acnet.constants import ACNET_PORT

# Parse a raw packet
packet = AcnetPacket.parse(raw_bytes)

if packet.is_reply():
    print(f"Reply from {packet.server_task_name}")
    print(f"Status: {packet.status}")
    print(f"Data: {packet.data.hex()}")
```

### Packet Classes

| Class | Description |
|-------|-------------|
| `AcnetPacket` | Base class |
| `AcnetRequest` | Incoming request |
| `AcnetReply` | Reply to our request |
| `AcnetMessage` | Unsolicited message |
| `AcnetCancel` | Cancel notification |

### Constants

```python
from pacsys.acnet.constants import (
    ACNET_PORT,           # 6801 - UDP port
    ACNET_TCP_PORT,       # 6802 - TCP port (for acnetd)
    ACNET_HEADER_SIZE,    # 18 bytes

    # Message flags
    ACNET_FLG_USM,        # 0x00 - Unsolicited message
    ACNET_FLG_REQ,        # 0x02 - Request
    ACNET_FLG_RPY,        # 0x04 - Reply
    ACNET_FLG_MLT,        # 0x01 - Multiple reply
    ACNET_FLG_CAN,        # 0x0200 - Cancel
)
```

## Wire Format Notes

From the original protocol documentation:

1. **No odd-length packets** - ACNET does not support odd-length payloads
2. **Byte swapping** - Conceptually, data is little-endian with even/odd bytes swapped per word
3. **Multiple packets per datagram** - A single UDP datagram may contain multiple ACNET packets

The byte-swap rule means:
- 2-byte integers appear big-endian on the wire (after swap)
- 4-byte integers have "middle-endian" representation
- Strings like `"MISCBOOT"` appear as `"IMCSOBTO"` on the wire

## Common Tasks

| Task Name | RAD50 | Purpose |
|-----------|-------|---------|
| DPMD | 0x000004A3 | Data Pool Manager daemon |
| RETDAT | 0x193C779C | Return data (frontend) |
| SETDAT | 0x193C779C | Set data (frontend) |

## Further Reading

- ACNET Design Note 22 (internal Fermilab documentation)
- `pacsys/acnet/` source code for implementation details
