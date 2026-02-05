"""
Live ACL-over-SSH tests against real ACNET console hosts.

Requires:
- Valid Kerberos ticket (kinit)
- Env vars PACSYS_TEST_ACL_JUMP and PACSYS_TEST_ACL_DEST
  (set in tests/real/.env.ssh)

Run:
    source tests/real/.env.ssh && python -m pytest tests/real/test_acl_session.py -v -s -o "addopts="
"""

import pytest

from pacsys.errors import ACLError
from pacsys.ssh import SSHClient

from .devices import ACL_DEST_HOST, ACL_JUMP_HOST, requires_acl_ssh


pytestmark = [requires_acl_ssh]


# ---------------------------------------------------------------------------
# One-shot: SSHClient.acl()
# ---------------------------------------------------------------------------


class TestOneShotACL:
    @pytest.fixture(autouse=True, scope="class")
    def ssh(self, request):
        client = SSHClient([ACL_JUMP_HOST, ACL_DEST_HOST])
        request.cls.ssh = client
        yield client
        client.close()

    def test_read_device(self):
        output = self.ssh.acl("read M:OUTTMP")
        print(f"  acl read: {output}")
        assert "M:OUTTMP" in output

    def test_semicolon_multiple_commands(self):
        output = self.ssh.acl("read M:OUTTMP; read G:AMANDA")
        print(f"  multi-cmd: {output}")
        lines = [line for line in output.splitlines() if line.strip()]
        assert len(lines) == 2
        assert "M:OUTTMP" in lines[0]
        assert "G:AMANDA" in lines[1]

    def test_nonexistent_device(self):
        # ACL returns error text inline, not via exit code
        output = self.ssh.acl("read Z:NOTFND")
        print(f"  error output: {output}")
        assert "DBM_NOREC" in output or "NOTFND" in output

    def test_custom_timeout(self):
        output = self.ssh.acl("read M:OUTTMP", timeout=10.0)
        assert "M:OUTTMP" in output

    def test_list_script_mode(self):
        output = self.ssh.acl(["read M:OUTTMP", "read G:AMANDA"])
        print(f"  script mode: {output}")
        assert "M:OUTTMP" in output
        assert "G:AMANDA" in output

    def test_list_single_command(self):
        output = self.ssh.acl(["read M:OUTTMP"])
        print(f"  script single: {output}")
        assert "M:OUTTMP" in output

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.ssh.acl([])


# ---------------------------------------------------------------------------
# Persistent: ACLSession
# ---------------------------------------------------------------------------


class TestACLSession:
    @pytest.fixture(autouse=True, scope="class")
    def ssh(self, request):
        client = SSHClient([ACL_JUMP_HOST, ACL_DEST_HOST])
        request.cls.ssh = client
        yield client
        client.close()

    def test_simple_read(self):
        with self.ssh.acl_session() as acl:
            output = acl.send("read M:OUTTMP")
            print(f"  session read: {output}")
            assert "M:OUTTMP" in output

    def test_variable_within_script(self):
        """Variables work within a single script (semicolon-separated)."""
        with self.ssh.acl_session() as acl:
            output = acl.send("deviceValue = M:OUTTMP ; print deviceValue")
            print(f"  var result: {output}")
            assert output

    def test_multiple_sends(self):
        with self.ssh.acl_session() as acl:
            r1 = acl.send("read M:OUTTMP")
            r2 = acl.send("read G:AMANDA")
            print(f"  send1: {r1}")
            print(f"  send2: {r2}")
            assert "M:OUTTMP" in r1
            assert "G:AMANDA" in r2

    def test_multiple_sessions_on_same_ssh(self):
        """Paramiko multiplexes channels — multiple sessions should work."""
        with self.ssh.acl_session() as acl1:
            with self.ssh.acl_session() as acl2:
                r1 = acl1.send("read M:OUTTMP")
                r2 = acl2.send("read G:AMANDA")
                assert "M:OUTTMP" in r1
                assert "G:AMANDA" in r2

    def test_close_session_not_ssh(self):
        """Closing ACL session should not close the SSH connection."""
        session = self.ssh.acl_session()
        output = session.send("read M:OUTTMP")
        assert "M:OUTTMP" in output
        session.close()
        # SSH should still work
        result = self.ssh.exec("hostname")
        assert result.ok

    def test_send_after_close_raises(self):
        session = self.ssh.acl_session()
        session.close()
        with pytest.raises(ACLError, match="closed"):
            session.send("read M:OUTTMP")
