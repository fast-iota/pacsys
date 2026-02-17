"""Structured audit log for supervised proxy server.

Writes JSON lines to a structured log and optionally tagged
length-delimited protobuf to a binary file.

Two modes controlled by ``log_responses``:

- ``False`` (default): one ``"in"`` JSON entry + request protobuf per RPC.
- ``True``: ``"in"`` entry per request AND ``"out"`` entry per response,
  with both protobufs written to the binary file.

Binary protobuf framing: ``tag_byte + varint_length + serialized_bytes``.
Tag identifies the message type so the file is self-describing.
"""

import json
import threading
from datetime import datetime, timezone
from typing import Optional

from ._policies import PolicyDecision, RequestContext

TAG_READ_REQUEST = 0x00
TAG_READ_REPLY = 0x01
TAG_SETTING_REQUEST = 0x02
TAG_SETTING_REPLY = 0x03

_REQUEST_TAGS = {"Read": TAG_READ_REQUEST, "Set": TAG_SETTING_REQUEST}
_RESPONSE_TAGS = {"Read": TAG_READ_REPLY, "Set": TAG_SETTING_REPLY}


def _encode_varint(write, value):
    """Encode an integer as a protobuf varint (single write call)."""
    buf = bytearray()
    bits = value & 0x7F
    value >>= 7
    while value:
        buf.append(0x80 | bits)
        bits = value & 0x7F
        value >>= 7
    buf.append(bits)
    write(buf)


class AuditLog:
    """Structured audit log with optional raw protobuf capture.

    Args:
        path: JSON lines file path.
        proto_path: Binary protobuf file path (optional).
        log_responses: Log outgoing responses too (default: False).
        flush_interval: Flush files every N writes (default: 1).
    """

    def __init__(
        self,
        path: str,
        proto_path: Optional[str] = None,
        log_responses: bool = False,
        flush_interval: int = 1,
    ):
        if flush_interval < 1:
            raise ValueError(f"flush_interval must be >= 1, got {flush_interval}")
        self._path = path
        self._proto_path = proto_path
        self._log_responses = log_responses
        self._flush_interval = flush_interval
        self._lock = threading.Lock()
        self._json_file = None
        self._proto_file = None
        self._writes_since_flush = 0
        self._seq = 0

    def log_request(self, ctx: RequestContext, decision: PolicyDecision) -> int:
        """Log incoming request. Returns sequence number for correlation."""
        with self._lock:
            self._seq += 1
            seq = self._seq

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "seq": seq,
                "dir": "in",
                "peer": ctx.peer,
                "method": ctx.rpc_method,
                "drfs": ctx.drfs,
                "allowed": decision.allowed,
                "reason": decision.reason,
            }
            if decision.allowed and decision.ctx is not None and decision.ctx.drfs != ctx.drfs:
                entry["final_drfs"] = decision.ctx.drfs
            self._write_json(entry)

            if self._proto_path is not None:
                serialize = getattr(ctx.raw_request, "SerializeToString", None)
                if serialize is not None:
                    tag = _REQUEST_TAGS.get(ctx.rpc_method)
                    if tag is None:
                        raise ValueError(f"Unknown RPC method for audit tagging: {ctx.rpc_method!r}")
                    self._write_proto(tag, serialize())

            self._maybe_flush()
            return seq

    def log_response(self, seq: int, peer: str, method: str, response_proto) -> None:
        """Log outgoing response. No-op when log_responses is False."""
        if not self._log_responses:
            return

        with self._lock:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "seq": seq,
                "dir": "out",
                "peer": peer,
                "method": method,
            }
            self._write_json(entry)

            if self._proto_path is not None:
                serialize = getattr(response_proto, "SerializeToString", None)
                if serialize is not None:
                    tag = _RESPONSE_TAGS.get(method)
                    if tag is None:
                        raise ValueError(f"Unknown RPC method for audit tagging: {method!r}")
                    self._write_proto(tag, serialize())

            self._maybe_flush()

    def _write_json(self, entry: dict):
        if self._json_file is None:
            self._json_file = open(self._path, "a")  # noqa: SIM115
        self._json_file.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._writes_since_flush += 1

    def _write_proto(self, tag: int, data: bytes):
        if self._proto_file is None:
            assert self._proto_path is not None
            self._proto_file = open(self._proto_path, "ab")  # noqa: SIM115
        self._proto_file.write(bytes((tag,)))
        _encode_varint(self._proto_file.write, len(data))
        self._proto_file.write(data)

    def _maybe_flush(self):
        if self._writes_since_flush >= self._flush_interval:
            if self._json_file is not None:
                self._json_file.flush()
            if self._proto_file is not None:
                self._proto_file.flush()
            self._writes_since_flush = 0

    def close(self):
        """Flush and close both files."""
        with self._lock:
            for f in (self._json_file, self._proto_file):
                if f is not None:
                    try:
                        f.flush()
                    except Exception:
                        pass
                    try:
                        f.close()
                    except Exception:
                        pass
            self._json_file = None
            self._proto_file = None
            self._writes_since_flush = 0
