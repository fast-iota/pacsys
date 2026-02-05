"""
Live SSH tests against real jump and destination hosts.

Requires:
- Valid Kerberos ticket (kinit)
- Env vars PACSYS_TEST_SSH_JUMP and PACSYS_TEST_SSH_DEST
  (set in tests/real/.env.ssh)

Run:
    source tests/real/.env.ssh && python -m pytest tests/real/test_ssh.py -v -s -o "addopts="
"""

import os
import socket

import pytest

from pacsys.ssh import (
    SSHClient,
    SSHTimeoutError,
)
from .devices import (
    SSH_JUMP_HOST,
    SSH_DEST_HOST,
    requires_ssh,
    kerberos_available,
)


pytestmark = [requires_ssh]


# =============================================================================
# Direct (single-hop) tests against jump host
# =============================================================================


class TestSingleHop:
    """Tests using direct SSH to the jump host (shared connection)."""

    @pytest.fixture(autouse=True, scope="class")
    def ssh(self, request):
        client = SSHClient(SSH_JUMP_HOST)
        request.cls.ssh = client
        yield client
        client.close()

    def test_exec_hostname(self):
        result = self.ssh.exec("hostname")
        assert result.ok
        assert result.stdout.strip()
        print(f"  hostname: {result.stdout.strip()}")

    def test_exec_whoami(self):
        result = self.ssh.exec("whoami")
        assert result.ok
        username = result.stdout.strip()
        assert username  # non-empty
        print(f"  whoami: {username}")

    def test_exec_nonzero_exit(self):
        result = self.ssh.exec("ls /nonexistent_path_xyz_12345")
        assert not result.ok
        assert result.exit_code != 0
        assert result.stderr  # should have error message

    def test_exec_with_stdin(self):
        result = self.ssh.exec("cat", input="hello from pacsys\n")
        assert result.ok
        assert "hello from pacsys" in result.stdout

    def test_exec_many(self):
        results = self.ssh.exec_many(["hostname", "date", "uptime"])
        assert len(results) == 3
        assert all(r.ok for r in results)
        for r in results:
            print(f"  {r.command}: {r.stdout.strip()}")

    def test_exec_stream(self):
        lines = list(self.ssh.exec_stream("echo line1; echo line2; echo line3"))
        assert len(lines) >= 3
        assert "line1" in lines[0]
        assert "line2" in lines[1]
        assert "line3" in lines[2]

    def test_sftp_listdir(self):
        with self.ssh.sftp() as sftp:
            entries = sftp.listdir("/tmp")
            assert isinstance(entries, list)
            print(f"  /tmp has {len(entries)} entries")

    def test_sftp_stat(self):
        with self.ssh.sftp() as sftp:
            info = sftp.stat(".bashrc")
            assert info.st_size > 0

    def test_tunnel_connectivity(self):
        """Verify tunnel can accept local connections."""
        with self.ssh.forward(0, "127.0.0.1", 22) as tunnel:
            assert tunnel.active
            assert tunnel.local_port > 0
            try:
                sock = socket.create_connection(("127.0.0.1", tunnel.local_port), timeout=5.0)
                data = sock.recv(256)
                assert b"SSH" in data
                sock.close()
                print(f"  tunnel port {tunnel.local_port} -> SSH banner OK")
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                pytest.fail(f"Tunnel connection failed: {e}")


class TestSingleHopLifecycle:
    """Tests that need their own SSH connection."""

    def test_exec_timeout(self):
        with SSHClient(SSH_JUMP_HOST) as ssh:
            with pytest.raises(SSHTimeoutError):
                ssh.exec("sleep 30", timeout=1.0)

    def test_lazy_connection(self):
        """Client should not connect until first operation."""
        ssh = SSHClient(SSH_JUMP_HOST)
        assert not ssh.connected
        try:
            result = ssh.exec("echo lazy")
            assert ssh.connected
            assert result.ok
        finally:
            ssh.close()
        assert not ssh.connected


# =============================================================================
# Multi-hop tests: jump -> destination
# =============================================================================


class TestMultiHop:
    """Tests using jump host to reach destination (shared connection)."""

    @pytest.fixture(autouse=True, scope="class")
    def ssh(self, request):
        client = SSHClient([SSH_JUMP_HOST, SSH_DEST_HOST])
        request.cls.ssh = client
        yield client
        client.close()

    def test_exec_hostname_via_jump(self):
        result = self.ssh.exec("hostname")
        assert result.ok
        hostname = result.stdout.strip()
        print(f"  dest hostname: {hostname}")
        assert hostname  # non-empty

    def test_exec_uname(self):
        result = self.ssh.exec("uname -a")
        assert result.ok
        assert "Linux" in result.stdout or "linux" in result.stdout.lower()
        print(f"  uname: {result.stdout.strip()[:80]}")

    def test_exec_many_via_jump(self):
        results = self.ssh.exec_many(["hostname", "whoami", "pwd"])
        assert len(results) == 3
        assert all(r.ok for r in results)

    def test_exec_stream_via_jump(self):
        lines = list(self.ssh.exec_stream("seq 1 5"))
        assert len(lines) == 5
        assert lines[0].strip() == "1"
        assert lines[4].strip() == "5"

    def test_sftp_via_jump(self):
        with self.ssh.sftp() as sftp:
            entries = sftp.listdir("/tmp")
            assert isinstance(entries, list)
            print(f"  dest /tmp has {len(entries)} entries")

    def test_sftp_roundtrip_via_jump(self):
        """Write a file via SFTP and read it back."""
        import uuid

        tag = uuid.uuid4().hex[:8]
        remote_path = f"/tmp/pacsys_test_{tag}.txt"
        content = f"pacsys ssh test {tag}\n"
        local_write = f"/tmp/pacsys_ssh_test_put_{tag}.txt"
        local_read = f"/tmp/pacsys_ssh_test_get_{tag}.txt"

        with open(local_write, "w") as f:
            f.write(content)

        try:
            with self.ssh.sftp() as sftp:
                sftp.put(local_write, remote_path)
                sftp.get(remote_path, local_read)
                sftp.remove(remote_path)

            with open(local_read) as f:
                assert f.read() == content
        finally:
            for p in (local_write, local_read):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_repr_multi_hop(self):
        r = repr(self.ssh)
        assert SSH_JUMP_HOST in r
        assert SSH_DEST_HOST in r
        assert "connected" in r


def test_tunnel_to_dest():
    """Tunnel through jump host to destination's SSH port."""
    with SSHClient(SSH_JUMP_HOST) as ssh:
        with ssh.forward(0, SSH_DEST_HOST, 22) as tunnel:
            assert tunnel.active
            try:
                sock = socket.create_connection(("127.0.0.1", tunnel.local_port), timeout=5.0)
                data = sock.recv(256)
                assert b"SSH" in data
                sock.close()
                print(f"  tunnel to {SSH_DEST_HOST}:22 OK (port {tunnel.local_port})")
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                pytest.fail(f"Tunnel to dest failed: {e}")


# =============================================================================
# Explicit KerberosAuth tests
# =============================================================================


class TestExplicitAuth:
    """Tests with explicit KerberosAuth passed to SSHClient."""

    @pytest.mark.skipif(not kerberos_available(), reason="Kerberos not available")
    def test_explicit_kerberos_auth(self):
        from pacsys.auth import KerberosAuth

        auth = KerberosAuth()
        with SSHClient(SSH_JUMP_HOST, auth=auth) as ssh:
            result = ssh.exec("hostname")
            assert result.ok
            print(f"  authenticated as: {auth.principal}")

    @pytest.mark.skipif(not kerberos_available(), reason="Kerberos not available")
    def test_explicit_auth_multi_hop(self):
        from pacsys.auth import KerberosAuth

        auth = KerberosAuth()
        with SSHClient([SSH_JUMP_HOST, SSH_DEST_HOST], auth=auth) as ssh:
            result = ssh.exec("whoami")
            assert result.ok
            print(f"  whoami on dest: {result.stdout.strip()}")


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    def test_double_close(self):
        ssh = SSHClient(SSH_JUMP_HOST)
        ssh.exec("true")
        ssh.close()
        ssh.close()  # should not raise

    def test_close_before_connect(self):
        ssh = SSHClient(SSH_JUMP_HOST)
        ssh.close()  # should not raise

    def test_connection_reuse(self):
        """Multiple commands should reuse the same transport."""
        with SSHClient(SSH_JUMP_HOST) as ssh:
            r1 = ssh.exec("echo first")
            r2 = ssh.exec("echo second")
            r3 = ssh.exec("echo third")
            assert r1.ok and r2.ok and r3.ok
            assert r1.stdout.strip() == "first"
            assert r3.stdout.strip() == "third"

    def test_large_output(self):
        """Handle commands with large stdout."""
        with SSHClient(SSH_JUMP_HOST) as ssh:
            result = ssh.exec("seq 1 10000")
            assert result.ok
            lines = result.stdout.strip().split("\n")
            assert len(lines) == 10000
            assert lines[0] == "1"
            assert lines[-1] == "10000"
