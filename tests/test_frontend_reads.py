"""Wire and lifecycle tests for GETS32/RETDAT, without live services."""

import struct
import threading
import time
from unittest.mock import MagicMock

import pytest

from pacsys.acnet import Gets32Client, Gets32Event, ReadDevice, RetdatClient, gets32, retdat
from pacsys.acnet.constants import ACNET_FLG_MLT, ACNET_FLG_RPY
from pacsys.acnet.errors import ACNET_DISCONNECTED, ACNET_ENDMULT, ACNET_PEND, AcnetError, AcnetTimeoutError
from pacsys.acnet.packet import AcnetReply

DEVICE = ReadDevice(27235, 12, bytes.fromhex("000042003f210000"), 2)
HEADER = struct.pack("<hBBBBIQQQ", 0, 1, 1, 0, 0, 7, 123, 1000, 1001)


def packet(data=b"", status=0, last=False):
    return AcnetReply(ACNET_FLG_RPY | (0 if last else ACNET_FLG_MLT), status, 3018, 1, 0, 0, 1, 18 + len(data), data)


@pytest.mark.parametrize("protocol", [gets32, retdat])
def test_wire_slots_preserve_data_padding_and_status(protocol):
    # Keep both possible padding positions for odd-length frontend data.
    payloads = [b"\x00\xa5", b"\xa5\x00", b"", b"\xff\x80ab"]
    lengths = [1, 1, 0, 4]
    devices = [ReadDevice(i, 12, b"\x00" * 8, size) for i, size in enumerate(lengths)]
    statuses = [0, 1, -2, 0]
    data = b"".join(struct.pack("<h", status) + raw for status, raw in zip(statuses, payloads))
    if protocol is gets32:
        data = HEADER + data
    reply = protocol.parse_reply(data, devices)
    assert [value.data for value in reply.values] == payloads
    assert [value.status for value in reply.values] == statuses
    assert reply.acnet_status is None and reply.received_at_ns is None


def test_retdat_request_layout_and_duplicate_entries():
    device = ReadDevice(0xFABCDE, 0xFF, bytes(range(8)), 3, 0xFFFF)
    entry = bytes.fromhex("debcfaff00010203040506070300ffff")
    assert retdat.build_request([device, device], 0x8002) == bytes.fromhex("0c0002000280") + entry * 2


def test_gets32_request_layout_large_offset_and_odd_event_padding():
    event = Gets32Event("E,2,E,0", 0x8002, True)
    device = ReadDevice(0xABCDEF, 12, bytes(range(8)), 3, 0x12345678)
    expected = (
        bytes.fromhex("0101000bca00010128000000010002800800")
        + b"E,2,E,0 "
        + bytes.fromhex("efcdab0c00010203040506070300000078563412")
    )
    assert gets32.build_request(3018, [device], event) == expected


@pytest.mark.parametrize(
    ("text", "wire", "ftd", "repetitive"),
    [
        ("I", "I", 0, False),
        ("N", "N", 2, False),
        ("p,2H", "P,500,true", 30, True),
        ("Q,1S,F", "Q,1000,false", -1, True),
        ("E,02", "E,2,E,0", 0x8002, True),
        ("e,0f,h,10M", "E,F,H,10", -1, True),
        ("E,2,S,0", "E,2,S,0", -1, True),
        ("E,FF", "E,FF,E,0", 0x80FF, True),
        ("S,V:CLDRST,9,1S,=", "S,V:CLDRST,9,1000,=", -1, True),
        ("S,134679,0,0,*", "S,134679,0,0,*", -1, True),
    ],
)
def test_event_canonicalization(text, wire, ftd, repetitive):
    assert Gets32Event.from_string(text) == Gets32Event(wire, ftd, repetitive)


@pytest.mark.parametrize("operator", ["=", "!=", "*", ">", "<", ">=", "<="])
def test_state_comparisons(operator):
    assert Gets32Event.from_string(f"S,134679,3,100,{operator}").text == f"S,134679,3,100,{operator}"


def test_explicit_legacy_event_and_ftd_are_not_reinterpreted():
    event = Gets32Event("m", 0xFFFF, True)
    request = gets32.build_request(3018, [DEVICE], event)
    assert request[14:20] == b"\xff\xff\x02\x00m "
    assert retdat.build_request([DEVICE], 0xC123)[4:6] == b"\x23\xc1"


@pytest.mark.parametrize(
    "event", ["U", "P,0", "E,100", "E,7FFF", "E,10000", "S,134679,65536,0,*", "S,134679,0,0,BAD", "E,2,H,bad"]
)
def test_invalid_or_unresolved_events(event):
    with pytest.raises(ValueError):
        Gets32Event.from_string(event)


def test_explicit_extended_clock_event_is_preserved():
    event = Gets32Event("E,7FFF,E,0", -1, True)
    assert gets32.build_request(3018, [DEVICE], event)[18:28] == b"E,7FFF,E,0"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"di": -1},
        {"di": 0x1000000},
        {"pi": 256},
        {"length": -1},
        {"offset": 2**32},
        {"ssdn": b"short"},
        {"ssdn": bytearray(8)},
        {"length": True},
    ],
)
def test_invalid_device(kwargs):
    fields = dict(di=1, pi=12, ssdn=bytes(8), length=2)
    fields.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        ReadDevice(**fields)


@pytest.mark.parametrize("protocol", [gets32, retdat])
def test_empty_requests_rejected(protocol):
    with pytest.raises(ValueError, match="empty"):
        build_for(protocol, [])


def test_retdat_rejects_32_bit_address_fields():
    with pytest.raises(ValueError, match="offset"):
        retdat.build_request([ReadDevice(1, 12, bytes(8), 2, 65536)])
    with pytest.raises(ValueError, match="length"):
        retdat.build_request([ReadDevice(1, 12, bytes(8), 65536)])


@pytest.mark.parametrize("protocol", [gets32, retdat])
def test_truncated_and_extra_reply_bytes(protocol):
    payload = (HEADER if protocol is gets32 else b"") + b"\x00\x00\xaa\xbb"
    for size in range(len(payload)):
        with pytest.raises(ValueError):
            protocol.parse_reply(payload[:size], [DEVICE])
    with pytest.raises(ValueError):
        protocol.parse_reply(payload + b"\x00", [DEVICE])


def test_gets32_preserves_all_header_fields_and_cycle_counter():
    header = struct.pack("<hBBBBIQQQ", -99, 1, 1, 0, 0, 0xFFFFFFFF, 123, 0, 0)
    reply = gets32.parse_reply(header + b"\x00\x00\x01\x02", [DEVICE])
    assert reply.header.global_status == -99
    assert reply.header.sequence == 0xFFFFFFFF
    assert reply.header.cycle_timestamp == 123
    assert reply.header.collection_timestamp == 0
    with pytest.raises(ValueError, match="Unsupported"):
        gets32.parse_reply(header[:3] + b"\x02" + header[4:] + b"\x00" * 4, [DEVICE])


@pytest.fixture(params=[gets32, retdat])
def wire(request):
    protocol = request.param
    conn = MagicMock()
    conn.connected = True
    context = MagicMock()
    delivered = []

    def send(node, task, payload, handler, **kwargs):
        conn.handler = handler
        for reply in delivered:
            handler(reply)
        return context

    conn.send_request.side_effect = send
    client = Gets32Client(conn) if protocol is gets32 else RetdatClient(conn)
    options = {"event": "P,500"} if protocol is gets32 else {"ftd": 30}
    prefix = HEADER if protocol is gets32 else b""
    return client, conn, context, delivered, options, prefix


def test_read_is_single_and_cleans_up(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.append(packet(prefix + b"\x00\x00\x7b\x00", last=True))
    reply = client.read(3018, [DEVICE], timeout=0.1)
    assert reply.values[0].data == b"\x7b\x00"
    assert reply.acnet_status == 0 and reply.received_at_ns > 0
    assert conn.send_request.call_args.kwargs["multiple_reply"] is False
    ctx.cancel.assert_called()


def test_one_shot_on_clock_event(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.append(packet(prefix + b"\x00" * 4, last=True))
    opts = {"event": "E,02"} if isinstance(client, Gets32Client) else {"ftd": 0x8002}
    client.read(3018, [DEVICE], **opts)
    assert conn.send_request.call_args.kwargs["multiple_reply"] is False
    if isinstance(client, Gets32Client):
        assert conn.send_request.call_args.args[2][7] == 0
    ctx.cancel.assert_called()


def test_pending_and_final_data(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.extend([packet(status=ACNET_PEND), packet(prefix + b"\x00\x00\x01\x02", status=1, last=True)])
    with client.stream(3018, [DEVICE], **options) as stream:
        rows = list(stream.readings(timeout=0.1))
    assert len(rows) == 1 and rows[0].acnet_status == 1
    assert rows[0].values[0].data == b"\x01\x02"
    ctx.cancel.assert_called()


def test_queued_data_precedes_disconnect(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.extend([packet(prefix + b"\x00" * 4), packet(status=ACNET_DISCONNECTED, last=True)])
    with client.stream(3018, [DEVICE], **options) as stream:
        rows = stream.readings(timeout=0.1)
        assert next(rows).values[0].status == 0
        with pytest.raises(AcnetError) as exc:
            next(rows)
        assert exc.value.status == ACNET_DISCONNECTED
        ctx.cancel.assert_called()


def test_overflow_during_registration_is_reported_and_cancelled(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.extend([packet(prefix + b"\x00" * 4)] * 2)
    with client.stream(3018, [DEVICE], buffer_size=1, **options) as stream:
        ctx.cancel.assert_called()
        rows = stream.readings(timeout=0.1)
        next(rows)
        with pytest.raises(BufferError):
            next(rows)


def test_malformed_data_cancels(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.append(packet(b"\x00"))
    with client.stream(3018, [DEVICE], **options) as stream:
        ctx.cancel.assert_not_called()
        with pytest.raises(ValueError):
            next(stream.readings(timeout=0.1))
        ctx.cancel.assert_called()


def test_timeout_with_queued_data(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.extend([packet(prefix + b"\x00" * 4)] * 2)
    with client.stream(3018, [DEVICE], **options) as stream:
        rows = stream.readings(timeout=0.01)
        next(rows)
        time.sleep(0.02)
        with pytest.raises(AcnetTimeoutError):
            next(rows)
    ctx.cancel.assert_called()


def test_timeout_and_empty_end_cleanup(wire):
    client, conn, ctx, delivered, options, prefix = wire
    with pytest.raises(AcnetTimeoutError):
        client.read(3018, [DEVICE], timeout=0.01)
    ctx.cancel.assert_called()
    ctx.cancel.reset_mock()
    delivered.append(packet(status=ACNET_ENDMULT, last=True))
    with pytest.raises(AcnetError, match="without data"):
        client.read(3018, [DEVICE], timeout=0.1)
    ctx.cancel.assert_called()


def test_close_wakes_waiter_and_suppresses_late_data(wire, monkeypatch):
    client, conn, ctx, delivered, options, prefix = wire
    stream = client.stream(3018, [DEVICE], **options)
    rows = []
    errors = []
    ready = threading.Event()
    original_wait = stream._condition.wait

    def wait(timeout=None):
        # Called with the condition locked, so close cannot pass until wait releases it.
        ready.set()
        return original_wait(timeout)

    def consume():
        try:
            rows.extend(stream.readings(timeout=None))
        except Exception as exc:  # noqa: BLE001 -- assert worker failures in the test thread
            errors.append(exc)

    monkeypatch.setattr(stream._condition, "wait", wait)
    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    try:
        assert ready.wait(0.5)
        stream.close()
        worker.join(0.5)
        assert not worker.is_alive() and rows == [] and errors == []
        conn.handler(packet(prefix + b"\x00" * 4))
        assert list(stream.readings(timeout=0.1)) == []
        ctx.cancel.assert_called()
    finally:
        stream.close()
        with stream._condition:
            stream._condition.notify_all()
        worker.join(0.5)


def test_invalid_inputs_never_send(wire):
    client, conn, ctx, delivered, options, prefix = wire
    for timeout in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            client.read(3018, [DEVICE], timeout=timeout)
    with pytest.raises(ValueError):
        client.stream(3018, [DEVICE], buffer_size=0, **options)
    with pytest.raises(ValueError):
        client.stream(65536, [DEVICE], **options)
    conn.send_request.assert_not_called()


def test_connection_failure_is_not_hidden(wire):
    client, conn, ctx, delivered, options, prefix = wire
    conn.send_request.side_effect = OSError("closed transport")
    with pytest.raises(OSError, match="closed transport"):
        client.read(3018, [DEVICE])


def test_disconnected_connection_rejected_before_send(wire):
    client, conn, ctx, delivered, options, prefix = wire
    conn.connected = False
    with pytest.raises(RuntimeError, match="must be connected"):
        client.read(3018, [DEVICE])
    with pytest.raises(RuntimeError, match="must be connected"):
        client.stream(3018, [DEVICE], **options)
    conn.send_request.assert_not_called()


def test_state_target_does_not_discard_drf_attributes():
    with pytest.raises(ValueError, match="state target"):
        Gets32Event.from_string("S,V:CLDRST.SETTING,9,0,=")


@pytest.mark.parametrize("text", ["", " ", "I\x00", "E,2\n", "é"])
def test_raw_event_requires_printable_nonempty_ascii(text):
    with pytest.raises(ValueError):
        Gets32Event(text, 0, True)


def test_stream_rejects_multiple_consumers_and_can_resume(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.extend([packet(prefix + b"\x00" * 4)] * 2)
    with client.stream(3018, [DEVICE], **options) as stream:
        first = stream.readings(timeout=0.1)
        next(first)
        with pytest.raises(RuntimeError, match="active iterator"):
            next(stream.readings(timeout=0.1))
        first.close()
        resumed = stream.readings(timeout=0.1)
        assert next(resumed).values[0].status == 0
        resumed.close()


def test_readings_validates_before_iteration(wire):
    client, conn, ctx, delivered, options, prefix = wire
    with client.stream(3018, [DEVICE], **options) as stream:
        for timeout in [0, -1, float("nan"), float("inf"), True]:
            with pytest.raises(ValueError):
                stream.readings(timeout=timeout)


def test_unused_iterator_does_not_reserve_stream_or_start_deadline(wire, monkeypatch):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.extend([packet(prefix + b"\x00" * 4)] * 2)
    with client.stream(3018, [DEVICE], **options) as stream:
        monkeypatch.setattr("pacsys.acnet._read.time.monotonic", lambda: 10.0)
        unused = stream.readings(timeout=1)
        active = stream.readings(timeout=1)
        assert next(active).values[0].status == 0
        active.close()
        monkeypatch.setattr("pacsys.acnet._read.time.monotonic", lambda: 20.0)
        assert next(unused).values[0].status == 0
        unused.close()


def test_close_discards_existing_backlog(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.extend([packet(prefix + b"\x00" * 4)] * 2)
    stream = client.stream(3018, [DEVICE], **options)
    stream.close()
    assert list(stream.readings(timeout=0.1)) == []


def test_control_reply_cannot_masquerade_as_data(wire):
    client, conn, ctx, delivered, options, prefix = wire
    delivered.append(packet(status=1))
    with pytest.raises(AcnetError, match="empty non-pending"):
        client.read(3018, [DEVICE], timeout=0.1)
    ctx.cancel.assert_called()


def build_for(protocol, devices, **limits):
    if protocol is gets32:
        return protocol.build_request(3018, devices, **limits)
    return protocol.build_request(devices, **limits)


@pytest.mark.parametrize("protocol", [gets32, retdat])
def test_larger_reply_limit_and_padding_boundary(protocol):
    header_size = 34 if protocol is gets32 else 0
    device = ReadDevice(1, 12, bytes(8), 40_000)
    with pytest.raises(ValueError, match="reply size"):
        build_for(protocol, [device])
    limit = header_size + 2 + device.length
    request = build_for(protocol, [device], max_reply_size=limit)
    assert struct.unpack_from("<I" if protocol is gets32 else "<H", request, 8 if protocol is gets32 else 0)[0] == limit
    with pytest.raises(ValueError, match="reply size"):
        build_for(protocol, [device], max_reply_size=limit - 1)
    # One extra requested byte needs two wire bytes because of word padding.
    odd = ReadDevice(1, 12, bytes(8), device.length + 1)
    with pytest.raises(ValueError, match="reply size"):
        build_for(protocol, [odd], max_reply_size=limit + 1)
    build_for(protocol, [odd], max_reply_size=limit + 2)


@pytest.mark.parametrize("protocol", [gets32, retdat])
def test_larger_request_limit_is_independent_of_reply_limit(protocol):
    devices = [ReadDevice(1, 12, bytes(8), 0)] * 600
    with pytest.raises(ValueError, match="request size"):
        build_for(protocol, devices)
    request = build_for(protocol, devices, max_request_size=16_384)
    assert len(request) > 8320
    build_for(protocol, devices, max_request_size=len(request))
    with pytest.raises(ValueError, match="request size"):
        build_for(protocol, devices, max_request_size=len(request) - 1)


@pytest.mark.parametrize("protocol", [gets32, retdat])
def test_reply_transport_ceiling(protocol):
    overhead = 36 if protocol is gets32 else 2
    device = ReadDevice(1, 12, bytes(8), 65_488 - overhead)
    build_for(protocol, [device], max_reply_size=65_488)
    too_large = ReadDevice(1, 12, bytes(8), device.length + 1)
    with pytest.raises(ValueError, match="reply size"):
        build_for(protocol, [too_large], max_reply_size=65_488)


@pytest.mark.parametrize("protocol", [gets32, retdat])
def test_request_transport_ceiling(protocol):
    count = 3273 if protocol is gets32 else 4092
    devices = [ReadDevice(1, 12, bytes(8), 0)] * count
    assert len(build_for(protocol, devices, max_request_size=65_484)) <= 65_484
    with pytest.raises(ValueError, match="request size"):
        build_for(protocol, devices + devices[:1], max_request_size=65_484)


@pytest.mark.parametrize("protocol,client_class", [(gets32, Gets32Client), (retdat, RetdatClient)])
@pytest.mark.parametrize(
    "name,bad",
    [
        ("max_request_size", 0),
        ("max_request_size", -1),
        ("max_request_size", 65_485),
        ("max_reply_size", 0),
        ("max_reply_size", 65_489),
        ("max_reply_size", None),
        ("max_reply_size", True),
        ("max_reply_size", 8320.0),
    ],
)
def test_invalid_limits_fail_at_construction_and_encoding(protocol, client_class, name, bad):
    conn = MagicMock()
    with pytest.raises((ValueError, TypeError), match=name):
        client_class(conn, **{name: bad})
    with pytest.raises((ValueError, TypeError), match=name):
        build_for(protocol, [DEVICE], **{name: bad})
    conn.send_request.assert_not_called()


def test_client_reply_limit_applies_to_read_and_stream(wire):
    old_client, conn, ctx, delivered, options, prefix = wire
    client = type(old_client)(conn, max_reply_size=40_036)
    device = ReadDevice(1, 12, bytes(8), 40_000)
    raw = b"\x12\x34" * 20_000
    delivered.append(packet(prefix + b"\x00\x00" + raw, last=True))
    assert client.read(3018, [device]).values[0].data == raw
    with client.stream(3018, [device], **options) as stream:
        rows = list(stream.readings(timeout=0.1))
    assert [row.values[0].data for row in rows] == [raw]
    ctx.cancel.assert_called()


@pytest.mark.parametrize("limits,kind", [({"max_request_size": 1}, "request"), ({"max_reply_size": 3}, "reply")])
def test_client_smaller_limits_prevent_io_on_both_paths(wire, limits, kind):
    old_client, conn, ctx, delivered, options, prefix = wire
    client = type(old_client)(conn, **limits)
    with pytest.raises(ValueError, match=kind + " size"):
        client.read(3018, [DEVICE])
    with pytest.raises(ValueError, match=kind + " size"):
        client.stream(3018, [DEVICE], **options)
    conn.send_request.assert_not_called()
