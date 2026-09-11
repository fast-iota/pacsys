"""GETS32/RETDAT through the UDP connection, with in-memory daemon datagrams."""

import asyncio
import struct
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pacsys.acnet import AcnetConnectionUDP, Gets32Client, ReadDevice, RetdatClient, encode, gets32, retdat
from pacsys.acnet.constants import (
    ACNET_FLG_MLT,
    ACNET_FLG_RPY,
    CMD_CANCEL,
    CMD_CONNECT,
    CMD_DISCONNECT,
    CMD_SEND_REQUEST_TIMEOUT,
    INFINITE_TIMEOUT,
)
from pacsys.acnet.errors import ACNET_PEND, AcnetTimeoutError

NODE = 3018
REQUEST_ID = 7
HEADER = struct.pack("<hBBBBIQQQ", 0, 1, 1, 0, 0, 7, 123, 1000, 1001)


def datagram(data=b"", *, status=0, last=False):
    flags = ACNET_FLG_RPY | (0 if last else ACNET_FLG_MLT)
    return (
        struct.pack("<Hh", flags, status)
        + struct.pack(">HH", NODE, 1)
        + struct.pack("<IHHH", 0, 1, REQUEST_ID, 18 + len(data))
        + data
    )


@pytest.fixture(params=[gets32, retdat])
def udp_frontend(request, monkeypatch):
    protocol = request.param
    daemon = SimpleNamespace(replies=[], requests=[], before_ack=False, cancelled=threading.Event())
    endpoints = []

    async def create_endpoint(loop, factory, **kwargs):
        callback = factory()
        transport = MagicMock(spec=asyncio.DatagramTransport)
        transport.get_extra_info.side_effect = lambda key: ("127.0.0.1", 12345) if key == "sockname" else None
        endpoints.append((transport, callback))
        if len(endpoints) == 2:
            transport.sendto.side_effect = send_command
        return transport, callback

    def send_command(data):
        command = struct.unpack_from(">H", data)[0]
        ack = struct.pack(">Hh", 0, 0)
        if command == CMD_CONNECT:
            ack = struct.pack(">HhBI", 1, 0, 1, encode("TEST"))
        elif command == CMD_SEND_REQUEST_TIMEOUT:
            daemon.requests.append(data)
            ack = struct.pack(">HhH", 2, 0, REQUEST_ID)
        elif command == CMD_CANCEL:
            assert struct.unpack_from(">H", data, 10)[0] == REQUEST_ID
            daemon.cancelled.set()
        else:
            assert command == CMD_DISCONNECT
        deliver_ack = lambda: endpoints[1][1].datagram_received(ack, ("127.0.0.1", 6802))
        if not daemon.before_ack:
            deliver_ack()
        if command == CMD_SEND_REQUEST_TIMEOUT:
            for reply in daemon.replies:
                endpoints[0][1].datagram_received(reply, ("127.0.0.1", 6802))
        if daemon.before_ack:
            deliver_ack()

    monkeypatch.setattr(asyncio.BaseEventLoop, "create_datagram_endpoint", create_endpoint)
    with AcnetConnectionUDP(name="TEST") as conn:
        thread = conn._reactor_thread
        client_type = Gets32Client if protocol is gets32 else RetdatClient
        yield client_type(conn, max_reply_size=60_000), daemon, protocol
    assert not thread.is_alive()
    for transport, _ in endpoints:
        transport.close.assert_called_once()


@pytest.mark.parametrize("before_ack", [False, True])
def test_udp_read_mixed_status_and_large_byte_range(udp_frontend, before_ack):
    client, daemon, protocol = udp_frontend
    daemon.before_ack = before_ack
    devices = [ReadDevice(i, 12, bytes(8), size, offset=32) for i, size in enumerate([40_000, 1, 2])]
    raw = bytes(range(250)) * 160
    payload = b"\x00\x00" + raw + struct.pack("<h", -42) + b"x!" + struct.pack("<h", 1) + b"ok"
    prefix = HEADER if protocol is gets32 else b""
    daemon.replies = [datagram(prefix + payload, last=True)]
    reply = client.read(NODE, devices, timeout=0.5)
    assert [(v.status, v.data) for v in reply.values] == [(0, raw), (-42, b"x!"), (1, b"ok")]
    assert reply.acnet_status == 0 and reply.received_at_ns > 0
    assert len(daemon.requests) == 1
    request = daemon.requests[0]
    command, handle, vnode, task, node, multiple, timeout = struct.unpack_from(">H3I2HI", request)
    assert (command, handle, vnode, task, node, multiple, timeout) == (
        CMD_SEND_REQUEST_TIMEOUT,
        encode("TEST"),
        0,
        encode("GETS32" if protocol is gets32 else "RETDAT"),
        NODE,
        0,
        INFINITE_TIMEOUT,
    )
    body = request[22:]
    if protocol is gets32:
        assert struct.unpack_from("<IH", body, 8) == (len(prefix + payload), 3)
        entry_start, entry_format, entry_size = 20, "<I8sII", 20
    else:
        assert struct.unpack_from("<HHH", body) == (len(payload), 3, 0)
        entry_start, entry_format, entry_size = 6, "<I8sHH", 16
    assert [struct.unpack_from(entry_format, body, entry_start + i * entry_size) for i in range(3)] == [
        (d.dipi, d.ssdn, d.length, d.offset) for d in devices
    ]
    assert len(body) == entry_start + 3 * entry_size


@pytest.mark.parametrize("before_ack", [False, True])
def test_udp_stream_pending_batches_and_cancellation(udp_frontend, before_ack):
    client, daemon, protocol = udp_frontend
    daemon.before_ack = before_ack
    prefix = HEADER if protocol is gets32 else b""
    daemon.replies = [
        datagram(status=ACNET_PEND),
        datagram(prefix + b"\x00\x00ab") + datagram(prefix + b"\x00\x00cd"),
    ]
    options = {"event": "P,500"} if protocol is gets32 else {"ftd": 30}
    with client.stream(NODE, [ReadDevice(1, 12, bytes(8), 2)], **options) as stream:
        rows = stream.readings(timeout=0.5)
        try:
            assert [next(rows).values[0].data for _ in range(2)] == [b"ab", b"cd"]
        finally:
            rows.close()
        assert struct.unpack_from(">H", daemon.requests[0], 16)[0] == 1
    assert daemon.cancelled.wait(0.5)
    assert list(stream.readings(timeout=0.5)) == []


def test_udp_read_timeout_cancels_request(udp_frontend):
    client, daemon, protocol = udp_frontend
    with pytest.raises(AcnetTimeoutError):
        client.read(NODE, [ReadDevice(1, 12, bytes(8), 2)], timeout=0.02)
    assert daemon.cancelled.wait(0.5)
