"""
Live RemoteProcess tests against real SSH hosts.

Requires:
- Valid Kerberos ticket (kinit)
- Env vars PACSYS_TEST_SSH_JUMP and PACSYS_TEST_SSH_DEST
  (set in tests/real/.env.ssh)

Run:
    source tests/real/.env.ssh && PACSYS_TEST_REAL=1 python -m pytest tests/real/test_remote_process.py -v -s
"""

import pytest

from pacsys.ssh import SSHClient, SSHTimeoutError

from .devices import SSH_DEST_HOST, SSH_JUMP_HOST, requires_ssh

pytestmark = [requires_ssh]


class TestRemoteProcessSingleHop:
    """RemoteProcess tests via jump host."""

    @pytest.fixture(autouse=True, scope="class")
    def ssh(self, request):
        client = SSHClient(SSH_JUMP_HOST)
        request.cls.ssh = client
        yield client
        client.close()

    def test_send_line_and_read_until(self):
        with self.ssh.remote_process("cat") as proc:
            proc.send_line("hello world")
            result = proc.read_until(b"\n", timeout=5.0)
            assert result == b"hello world"

    def test_send_bytes(self):
        with self.ssh.remote_process("cat") as proc:
            proc.send_bytes(b"raw bytes\n")
            result = proc.read_until(b"\n", timeout=5.0)
            assert result == b"raw bytes"

    def test_read_until_marker_in_stream(self):
        with self.ssh.remote_process("cat") as proc:
            proc.send_line("aaaMARKERbbb")
            result = proc.read_until(b"MARKER", timeout=5.0)
            assert result == b"aaa"
            # "bbb\n" should still be in buffer
            rest = proc.read_until(b"\n", timeout=5.0)
            assert rest == b"bbb"

    def test_read_for(self):
        with self.ssh.remote_process("echo hello; sleep 0.5; echo world") as proc:
            result = proc.read_for(2.0)
            text = result.decode()
            assert "hello" in text
            assert "world" in text

    def test_read_until_timeout(self):
        with self.ssh.remote_process("cat") as proc:
            with pytest.raises(SSHTimeoutError):
                proc.read_until(b"NEVER_APPEARS", timeout=0.5)

    def test_alive_and_close(self):
        proc = self.ssh.remote_process("cat")
        assert proc.alive
        proc.close()
        assert not proc.alive

    def test_context_manager(self):
        with self.ssh.remote_process("cat") as proc:
            assert proc.alive
        assert not proc.alive

    def test_multiple_exchanges(self):
        with self.ssh.remote_process("cat") as proc:
            for i in range(5):
                proc.send_line(f"line {i}")
                result = proc.read_until(b"\n", timeout=5.0)
                assert result == f"line {i}".encode()

    def test_multiple_processes_on_same_ssh(self):
        """Paramiko multiplexes channels - multiple processes should work."""
        with self.ssh.remote_process("cat") as p1:
            with self.ssh.remote_process("cat") as p2:
                p1.send_line("from p1")
                p2.send_line("from p2")
                r1 = p1.read_until(b"\n", timeout=5.0)
                r2 = p2.read_until(b"\n", timeout=5.0)
                assert r1 == b"from p1"
                assert r2 == b"from p2"


class TestRemoteProcessMultiHop:
    """RemoteProcess through jump -> destination."""

    @pytest.fixture(autouse=True, scope="class")
    def ssh(self, request):
        client = SSHClient([SSH_JUMP_HOST, SSH_DEST_HOST])
        request.cls.ssh = client
        yield client
        client.close()

    def test_send_and_read_via_jump(self):
        with self.ssh.remote_process("cat") as proc:
            proc.send_line("multi-hop test")
            result = proc.read_until(b"\n", timeout=5.0)
            assert result == b"multi-hop test"

    def test_interactive_shell_command(self):
        """Run bc (calculator) as interactive process."""
        with self.ssh.remote_process("bc -q") as proc:
            proc.send_line("2 + 3")
            result = proc.read_until(b"\n", timeout=5.0)
            assert result.strip() == b"5"

            proc.send_line("10 * 20")
            result = proc.read_until(b"\n", timeout=5.0)
            assert result.strip() == b"200"
