"""
Low-level tests for AcnetConnectionTCP (direct acnetd commands over TCP).

Tests the TCP protocol path: AcnetConnectionTCP -> acnetd

Run with: pytest tests/real/low_level/test_acnet_tcp.py -v -s
"""

import queue
import time

import pytest

from pacsys.acnet import NodeStats, node_parts
from tests.real.devices import requires_acnet_tcp


@pytest.fixture(autouse=True)
def _settle():
    """Brief pause between tests to let acnetd connections settle."""
    yield
    time.sleep(0.5)


# Known ACNET front-end nodes used for testing.
# address = (trunk << 8) | node
CLX74_NODE = "CLX74"
CLX74_EXPECTED_ADDRESS = (14 << 8) | 74  # 3658

MUONFE_NODE = "MUONFE"
MUONFE_EXPECTED_ADDRESS = (11 << 8) | 202  # 3018


@requires_acnet_tcp
class TestAcnetTCPConnection:
    """Tests for basic connection lifecycle."""

    def test_connect_gets_handle(self, acnet_tcp_connection):
        conn = acnet_tcp_connection
        assert conn.connected
        assert conn.name  # daemon assigns a handle
        assert conn.raw_handle != 0
        print(f"\n  Connected as: {conn.name} ({conn.raw_handle:#x})")


@requires_acnet_tcp
class TestAcnetTCPLookups:
    """Tests for node/name lookups."""

    def test_get_local_node(self, acnet_tcp_connection):
        conn = acnet_tcp_connection
        node = conn.get_local_node()
        assert node > 0
        trunk, n = node >> 8, node & 0xFF
        print(f"\n  Local node: {node} (trunk={trunk}, node={n})")

    def test_get_default_node(self, acnet_tcp_connection):
        conn = acnet_tcp_connection
        node = conn.get_default_node()
        assert node > 0
        trunk, n = node >> 8, node & 0xFF
        print(f"\n  Default node: {node} (trunk={trunk}, node={n})")

    def test_get_name_roundtrip(self, acnet_tcp_connection):
        conn = acnet_tcp_connection
        local = conn.get_local_node()
        name = conn.get_name(local)
        assert name  # non-empty
        resolved = conn.get_node(name)
        assert resolved == local
        print(f"\n  {local} -> {name} -> {resolved}")

    def test_resolve_clx74(self, acnet_tcp_connection):
        """Resolve CLX74 node name to address (trunk=14, node=74 → 3658)."""
        conn = acnet_tcp_connection
        addr = conn.get_node(CLX74_NODE)
        trunk, node = node_parts(addr)
        print(f"\n  Resolved '{CLX74_NODE}' to {addr} (trunk={trunk}, node={node})")
        assert addr == CLX74_EXPECTED_ADDRESS

    def test_resolve_muonfe(self, acnet_tcp_connection):
        """Resolve MUONFE node name to address (trunk=11, node=202 → 3018)."""
        conn = acnet_tcp_connection
        addr = conn.get_node(MUONFE_NODE)
        trunk, node = node_parts(addr)
        print(f"\n  Resolved '{MUONFE_NODE}' to {addr} (trunk={trunk}, node={node})")
        assert addr == MUONFE_EXPECTED_ADDRESS


@requires_acnet_tcp
class TestAcnetTCPStats:
    """Tests for node statistics."""

    def test_get_node_stats(self, acnet_tcp_connection):
        conn = acnet_tcp_connection
        stats = conn.get_node_stats()
        assert isinstance(stats, NodeStats)
        # All counters should be non-negative
        assert stats.usm_received >= 0
        assert stats.requests_received >= 0
        assert stats.replies_received >= 0
        assert stats.usm_sent >= 0
        assert stats.requests_sent >= 0
        assert stats.replies_sent >= 0
        assert stats.request_queue_limit >= 0
        print(f"\n  Stats: {stats}")


@requires_acnet_tcp
class TestAcnetTCPTaskOperations:
    """Tests for task-level operations."""

    def test_get_task_pid(self, acnet_tcp_connection):
        conn = acnet_tcp_connection
        pid = conn.get_task_pid(conn.name)
        assert pid > 0
        print(f"\n  PID for {conn.name}: {pid}")

    def test_rename_task(self, acnet_tcp_connection):
        conn = acnet_tcp_connection
        old_name = conn.name
        conn.rename_task("PYTEST")
        assert conn.name == "PYTEST"
        print(f"\n  Renamed: {old_name} -> {conn.name}")


def _ping(conn, node_name, expected_addr, timeout=10):
    """Send ACNET ping and return the reply."""
    target = conn.get_node(node_name)
    assert target == expected_addr

    reply_q = queue.Queue()
    conn.send_request(
        node=target,
        task="ACNET",
        data=b"\x00\x00",
        reply_handler=lambda reply: reply_q.put(reply),
        multiple_reply=False,
        timeout=timeout * 1000,
    )

    reply = reply_q.get(timeout=timeout)
    trunk, node = node_parts(reply.server)
    print(
        f"\n  Ping {node_name} ({target}):"
        f"\n    status={reply.status}, last={reply.last}"
        f"\n    replier={reply.server} (trunk={trunk}, node={node})"
        f"\n    payload={reply.data!r}"
    )
    return reply


@requires_acnet_tcp
class TestAcnetTCPPing:
    """Tests for ACNET ping (request/reply to remote nodes).

    An ACNET ping sends a request to the "ACNET" task on a remote node with
    payload b'\\x00\\x00'. acnetd nodes echo the payload back; front-end
    nodes reply with status 0 but may return empty payload.
    """

    def test_ping_clx74(self, acnet_tcp_connection):
        """Ping CLX74 (acnetd node) -- must echo payload back."""
        reply = _ping(acnet_tcp_connection, CLX74_NODE, CLX74_EXPECTED_ADDRESS)
        assert reply.status == 0, f"Ping CLX74 failed with status {reply.status}"
        assert reply.data == b"\x00\x00", f"CLX74 should echo payload: {reply.data!r}"
        assert reply.last

    def test_ping_muonfe(self, acnet_tcp_connection):
        """Ping MUONFE (front-end node) -- status 0 is sufficient."""
        reply = _ping(acnet_tcp_connection, MUONFE_NODE, MUONFE_EXPECTED_ADDRESS)
        assert reply.status == 0, f"Ping MUONFE failed with status {reply.status}"
        assert reply.last


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
