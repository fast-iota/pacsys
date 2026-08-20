"""Tests for AuditLog — structured JSON + tagged binary protobuf logging."""

import json

import pytest

from pacsys.supervised._audit import (
    TAG_READ_REPLY,
    TAG_READ_REQUEST,
    TAG_SETTING_REPLY,
    TAG_SETTING_REQUEST,
    AuditLog,
    _encode_varint,
)
from pacsys.supervised._policies import PolicyDecision, RequestContext


class _FakeProto:
    """Minimal object that quacks like a protobuf message."""

    def __init__(self, data: bytes):
        self._data = data

    def SerializeToString(self):  # noqa: N802 -- protobuf method name
        return self._data


def _ctx(
    drfs=None,
    rpc_method="Read",
    peer="ipv4:127.0.0.1:9999",
    values=None,
    raw_request=None,
):
    return RequestContext(
        drfs=drfs or ["M:OUTTMP"],
        rpc_method=rpc_method,
        peer=peer,
        metadata={},
        values=values or [],
        raw_request=raw_request,
    )


def _read_jsonl(path):
    """Read JSON lines file, return list of dicts."""
    lines = path.read_text().strip().split("\n")
    return [json.loads(line) for line in lines]


# ── JSON output ──────────────────────────────────────────────────────────


class TestAuditLogJSON:
    def test_request_only_mode(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path))
        ctx = _ctx(raw_request=_FakeProto(b"\x01"))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        seq = audit.log_request(ctx, decision)
        audit.close()

        entries = _read_jsonl(path)
        assert len(entries) == 1
        e = entries[0]
        assert e["seq"] == seq
        assert e["dir"] == "in"
        assert e["peer"] == "ipv4:127.0.0.1:9999"
        assert e["method"] == "Read"
        assert e["drfs"] == ["M:OUTTMP"]
        assert e["allowed"] is True
        assert e["reason"] is None

    def test_denied_request_logged(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path))
        ctx = _ctx(rpc_method="Set")
        decision = PolicyDecision(allowed=False, reason="blocked")
        audit.log_request(ctx, decision)
        audit.close()

        entries = _read_jsonl(path)
        assert len(entries) == 1
        assert entries[0]["allowed"] is False
        assert entries[0]["reason"] == "blocked"

    def test_response_noop_when_disabled(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path), log_responses=False)
        ctx = _ctx(raw_request=_FakeProto(b"\x01"))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        seq = audit.log_request(ctx, decision)
        audit.log_response(seq, "ipv4:127.0.0.1:9999", "Read", _FakeProto(b"\x02"))
        audit.close()

        entries = _read_jsonl(path)
        assert len(entries) == 1  # only the request

    def test_response_logged_when_enabled(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path), log_responses=True)
        ctx = _ctx(raw_request=_FakeProto(b"\x01"))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        seq = audit.log_request(ctx, decision)
        audit.log_response(seq, "ipv4:127.0.0.1:9999", "Read", _FakeProto(b"\x02"))
        audit.close()

        entries = _read_jsonl(path)
        assert len(entries) == 2
        assert entries[0]["dir"] == "in"
        assert entries[1]["dir"] == "out"
        assert entries[1]["seq"] == seq
        assert entries[1]["peer"] == "ipv4:127.0.0.1:9999"
        assert entries[1]["method"] == "Read"

    def test_multiple_responses_same_seq(self, tmp_path):
        """Streaming: many out entries share the same seq."""
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path), log_responses=True)
        ctx = _ctx(raw_request=_FakeProto(b"\x01"))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        seq = audit.log_request(ctx, decision)
        for _ in range(3):
            audit.log_response(seq, "ipv4:127.0.0.1:9999", "Read", _FakeProto(b"\x02"))
        audit.close()

        entries = _read_jsonl(path)
        assert len(entries) == 4  # 1 in + 3 out
        out_entries = [e for e in entries if e["dir"] == "out"]
        assert all(e["seq"] == seq for e in out_entries)

    def test_no_final_drfs_when_unchanged(self, tmp_path):
        """When DRFs are not modified, final_drfs is absent."""
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path))
        ctx = _ctx(drfs=["M:OUTTMP"])
        decision = PolicyDecision(allowed=True, ctx=ctx)
        audit.log_request(ctx, decision)
        audit.close()

        entries = _read_jsonl(path)
        assert "final_drfs" not in entries[0]

    def test_no_final_drfs_when_denied(self, tmp_path):
        """Denied requests never have final_drfs (never reach backend)."""
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path))
        ctx = _ctx(drfs=["M:OUTTMP"])
        decision = PolicyDecision(allowed=False, reason="blocked")
        audit.log_request(ctx, decision)
        audit.close()

        entries = _read_jsonl(path)
        assert "final_drfs" not in entries[0]

    def test_seq_increments(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path))
        ctx = _ctx()
        decision = PolicyDecision(allowed=True, ctx=ctx)
        seq1 = audit.log_request(ctx, decision)
        seq2 = audit.log_request(ctx, decision)
        assert seq2 == seq1 + 1
        audit.close()

    def test_timestamp_present(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path))
        ctx = _ctx()
        decision = PolicyDecision(allowed=True, ctx=ctx)
        audit.log_request(ctx, decision)
        audit.close()

        entries = _read_jsonl(path)
        assert "ts" in entries[0]
        # ISO format with timezone
        assert "T" in entries[0]["ts"]


# ── Binary protobuf output ───────────────────────────────────────────────


def _read_tagged_protos(path):
    """Read tagged length-delimited protobuf file, return [(tag, data), ...]."""
    raw = path.read_bytes()
    entries = []
    pos = 0
    while pos < len(raw):
        tag = raw[pos]
        pos += 1
        # Decode varint
        length = 0
        shift = 0
        while True:
            b = raw[pos]
            pos += 1
            length |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        data = raw[pos : pos + length]
        pos += length
        entries.append((tag, data))
    return entries


class TestAuditLogProto:
    def test_request_proto_tagged(self, tmp_path):
        json_path = tmp_path / "audit.jsonl"
        proto_path = tmp_path / "audit.binpb"
        audit = AuditLog(str(json_path), proto_path=str(proto_path))

        payload = b"\x08\x01\x12\x03foo"
        ctx = _ctx(raw_request=_FakeProto(payload))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        audit.log_request(ctx, decision)
        audit.close()

        entries = _read_tagged_protos(proto_path)
        assert len(entries) == 1
        assert entries[0] == (TAG_READ_REQUEST, payload)

    def test_set_request_tag(self, tmp_path):
        json_path = tmp_path / "audit.jsonl"
        proto_path = tmp_path / "audit.binpb"
        audit = AuditLog(str(json_path), proto_path=str(proto_path))

        payload = b"\xab\xcd"
        ctx = _ctx(rpc_method="Set", raw_request=_FakeProto(payload))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        audit.log_request(ctx, decision)
        audit.close()

        entries = _read_tagged_protos(proto_path)
        assert entries[0][0] == TAG_SETTING_REQUEST

    def test_response_proto_tagged(self, tmp_path):
        json_path = tmp_path / "audit.jsonl"
        proto_path = tmp_path / "audit.binpb"
        audit = AuditLog(str(json_path), proto_path=str(proto_path), log_responses=True)

        req_payload = b"\x01"
        resp_payload = b"\x02\x03"
        ctx = _ctx(raw_request=_FakeProto(req_payload))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        seq = audit.log_request(ctx, decision)
        audit.log_response(seq, "peer", "Read", _FakeProto(resp_payload))
        audit.close()

        entries = _read_tagged_protos(proto_path)
        assert len(entries) == 2
        assert entries[0] == (TAG_READ_REQUEST, req_payload)
        assert entries[1] == (TAG_READ_REPLY, resp_payload)

    def test_set_response_tag(self, tmp_path):
        json_path = tmp_path / "audit.jsonl"
        proto_path = tmp_path / "audit.binpb"
        audit = AuditLog(str(json_path), proto_path=str(proto_path), log_responses=True)

        ctx = _ctx(rpc_method="Set", raw_request=_FakeProto(b"\x01"))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        seq = audit.log_request(ctx, decision)
        audit.log_response(seq, "peer", "Set", _FakeProto(b"\x02"))
        audit.close()

        entries = _read_tagged_protos(proto_path)
        assert entries[1][0] == TAG_SETTING_REPLY

    def test_no_proto_file_when_path_not_set(self, tmp_path):
        json_path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(json_path))  # no proto_path
        ctx = _ctx(raw_request=_FakeProto(b"\x01"))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        audit.log_request(ctx, decision)
        audit.close()

        assert not (tmp_path / "audit.binpb").exists()

    def test_non_serializable_skipped_in_proto(self, tmp_path):
        json_path = tmp_path / "audit.jsonl"
        proto_path = tmp_path / "audit.binpb"
        audit = AuditLog(str(json_path), proto_path=str(proto_path))
        ctx = _ctx(raw_request="not a proto")
        decision = PolicyDecision(allowed=True, ctx=ctx)
        audit.log_request(ctx, decision)
        audit.close()

        # JSON still written
        assert json_path.exists()
        # Proto file not created (no serializable data)
        assert not proto_path.exists()

    def test_response_not_written_when_disabled(self, tmp_path):
        json_path = tmp_path / "audit.jsonl"
        proto_path = tmp_path / "audit.binpb"
        audit = AuditLog(str(json_path), proto_path=str(proto_path), log_responses=False)

        ctx = _ctx(raw_request=_FakeProto(b"\x01"))
        decision = PolicyDecision(allowed=True, ctx=ctx)
        seq = audit.log_request(ctx, decision)
        audit.log_response(seq, "peer", "Read", _FakeProto(b"\x02"))
        audit.close()

        entries = _read_tagged_protos(proto_path)
        assert len(entries) == 1  # only request


# ── Flush & lifecycle ────────────────────────────────────────────────────


class TestAuditLogLifecycle:
    def test_flush_interval_batches(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path), flush_interval=3)
        ctx = _ctx()
        decision = PolicyDecision(allowed=True, ctx=ctx)

        audit.log_request(ctx, decision)
        audit.log_request(ctx, decision)
        assert audit._writes_since_flush == 2

        audit.log_request(ctx, decision)
        assert audit._writes_since_flush == 0  # flushed
        audit.close()

    def test_close_flushes_pending(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path), flush_interval=100)
        ctx = _ctx()
        decision = PolicyDecision(allowed=True, ctx=ctx)
        audit.log_request(ctx, decision)
        assert audit._writes_since_flush == 1
        audit.close()

        entries = _read_jsonl(path)
        assert len(entries) == 1

    def test_close_idempotent(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(str(path))
        ctx = _ctx()
        decision = PolicyDecision(allowed=True, ctx=ctx)
        audit.log_request(ctx, decision)
        audit.close()
        audit.close()  # should not raise

    def test_flush_interval_zero_raises(self):
        with pytest.raises(ValueError, match="flush_interval"):
            AuditLog("/dev/null", flush_interval=0)

    def test_flush_interval_default_is_one(self, tmp_path):
        audit = AuditLog(str(tmp_path / "audit.jsonl"))
        assert audit._flush_interval == 1
        audit.close()


# ── _encode_varint ───────────────────────────────────────────────────────


class TestEncodeVarint:
    def test_single_byte(self):
        buf = bytearray()
        _encode_varint(buf.extend, 5)
        assert buf == bytes([5])

    def test_two_bytes(self):
        buf = bytearray()
        _encode_varint(buf.extend, 300)
        # 300 = 0b100101100 -> 0xAC 0x02
        assert buf == bytes([0xAC, 0x02])

    def test_zero(self):
        buf = bytearray()
        _encode_varint(buf.extend, 0)
        assert buf == bytes([0])
