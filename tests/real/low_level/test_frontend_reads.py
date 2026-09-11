"""Raw GETS32/RETDAT reads of M:OUTTMP only; no settings are issued."""

import itertools
import statistics

import pytest

from pacsys.acnet import Gets32Client, ReadDevice, RetdatClient
from pacsys.acnet.errors import AcnetError, AcnetTimeoutError
from tests.real.devices import (
    MOUTTMP_DI,
    MOUTTMP_PI,
    MOUTTMP_SSDN,
    MUONFE_EXPECTED_ADDRESS,
    MUONFE_NODE,
    requires_acnet_tcp,
)

pytestmark = requires_acnet_tcp
DEVICE = ReadDevice(MOUTTMP_DI, MOUTTMP_PI, MOUTTMP_SSDN, 2)


@pytest.fixture(params=[Gets32Client, RetdatClient])
def frontend(request, acnet_tcp_connection):
    node = acnet_tcp_connection.get_node(MUONFE_NODE)
    assert node == MUONFE_EXPECTED_ADDRESS
    return request.param(acnet_tcp_connection), node


def check_reply(reply, count=1):
    assert reply.acnet_status == 0
    assert reply.received_at_ns is not None
    assert len(reply.values) == count
    assert all(value.status == 0 and len(value.data) == 2 for value in reply.values), reply
    if hasattr(reply, "header"):
        assert reply.header.global_status == 0
        assert reply.header.collection_timestamp > 0
        assert reply.header.reply_timestamp > 0


def test_immediate_and_batch(frontend):
    client, node = frontend
    check_reply(client.read(node, [DEVICE]))
    check_reply(client.read(node, [DEVICE, DEVICE]), count=2)


def test_periodic(frontend):
    client, node = frontend
    options = {"event": "P,500"} if isinstance(client, Gets32Client) else {"ftd": 30}
    with client.stream(node, [DEVICE], **options) as stream:
        rows = list(itertools.islice(stream.readings(timeout=4), 6))
    assert len(rows) == 6
    for row in rows:
        check_reply(row)
    received = [row.received_at_ns / 1e9 for row in rows]
    intervals = [b - a for a, b in itertools.pairwise(received[1:])]
    assert 0.35 <= statistics.median(intervals) <= 0.65, intervals
    if isinstance(client, Gets32Client):
        sequences = [row.header.sequence for row in rows]
        assert all(b == (a + 1) % 2**32 for a, b in itertools.pairwise(sequences))


def test_clock_stream(frontend):
    client, node = frontend
    options = {"event": "E,02"} if isinstance(client, Gets32Client) else {"ftd": 0x8002}
    with client.stream(node, [DEVICE], **options) as stream:
        rows = list(itertools.islice(stream.readings(timeout=15), 2))
    assert len(rows) == 2
    for row in rows:
        check_reply(row)
    assert rows[1].received_at_ns - rows[0].received_at_ns > 100_000_000


def test_single_clock_reply(frontend):
    client, node = frontend
    options = {"event": "E,02"} if isinstance(client, Gets32Client) else {"ftd": 0x8002}
    check_reply(client.read(node, [DEVICE], timeout=8, **options))


def test_gets32_change_only_acceptance_or_rejection(acnet_tcp_connection):
    node = acnet_tcp_connection.get_node(MUONFE_NODE)
    client = Gets32Client(acnet_tcp_connection)
    with client.stream(node, [DEVICE], event="Q,200,true") as stream:
        # Probe support only; an initial reply does not establish change-only semantics.
        try:
            row = next(stream.readings(timeout=1))
        except AcnetTimeoutError:
            raise
        except AcnetError as exc:
            # MUONFE rejects Q with an outer -8; preserve that reply instead
            # of silently substituting ordinary periodic acquisition.
            assert exc.status == -8
        else:
            check_reply(row)


def test_gets32_accepts_delayed_software_clock(acnet_tcp_connection):
    node = acnet_tcp_connection.get_node(MUONFE_NODE)
    client = Gets32Client(acnet_tcp_connection)
    reply = client.read(node, [DEVICE], event="E,02,S,100", timeout=8)
    check_reply(reply)
