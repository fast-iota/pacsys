"""Tests for ACL-over-SSH: one-shot SSHClient.acl() and persistent ACLSession."""

import sys
import threading
from unittest.mock import MagicMock, patch

import paramiko
import pytest

from pacsys.acl_session import ACLSession, _strip_acl_output
from pacsys.errors import ACLError
from pacsys.ssh import SSHClient, SSHHop

_ssh_mod = sys.modules["pacsys.ssh"]


@pytest.fixture(autouse=True)
def _mock_getuser():
    with patch("getpass.getuser", return_value="testuser"):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RealTransport = paramiko.Transport


def _make_mock_transport(active=True):
    t = MagicMock(spec=_RealTransport)
    t.is_active.return_value = active
    return t


def _connected_ssh(mock_connect, mock_transport_cls, active=True):
    """Return a connected SSHClient with mocked transport."""
    mock_connect.return_value = MagicMock()
    mock_transport = _make_mock_transport(active)
    mock_transport_cls.return_value = mock_transport
    ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
    return ssh, mock_transport


def _make_exec_channel(stdout=b"", stderr=b"", exit_code=0):
    """Create a mock channel that simulates one-shot command execution."""
    chan = MagicMock()
    chan.status_event = threading.Event()
    chan.status_event.set()

    stdout_returned = [False]
    stderr_returned = [False]

    def recv_ready():
        return not stdout_returned[0] and bool(stdout)

    def recv(size):
        stdout_returned[0] = True
        return stdout

    def recv_stderr_ready():
        return not stderr_returned[0] and bool(stderr)

    def recv_stderr(size):
        stderr_returned[0] = True
        return stderr

    def exit_status_ready():
        return stdout_returned[0] or not stdout

    chan.recv_ready = MagicMock(side_effect=lambda: recv_ready())
    chan.recv = MagicMock(side_effect=lambda size: recv(size))
    chan.recv_stderr_ready = MagicMock(side_effect=lambda: recv_stderr_ready())
    chan.recv_stderr = MagicMock(side_effect=lambda size: recv_stderr(size))
    chan.exit_status_ready = MagicMock(side_effect=lambda: exit_status_ready())
    chan.recv_exit_status.return_value = exit_code
    return chan


def _make_interactive_channel(responses):
    """Create a mock channel for persistent ACL session tests.

    Args:
        responses: List of byte strings. Each call to recv() returns the next one.
                  Must include real ACL prompt format: b"\\nACL> "
    """
    chan = MagicMock()
    chan.status_event = threading.Event()
    chan.status_event.set()
    chan.closed = False

    remaining = list(responses)

    def recv_ready():
        return bool(remaining)

    def recv(size):
        if remaining:
            return remaining.pop(0)
        return b""

    def exit_status_ready():
        return False

    def recv_stderr_ready():
        return False

    chan.recv_ready = MagicMock(side_effect=lambda: recv_ready())
    chan.recv = MagicMock(side_effect=lambda size: recv(size))
    chan.exit_status_ready = MagicMock(side_effect=lambda: exit_status_ready())
    chan.recv_stderr_ready = MagicMock(side_effect=lambda: recv_stderr_ready())
    chan.recv_stderr = MagicMock(side_effect=lambda size: b"")
    return chan


# ---------------------------------------------------------------------------
# _strip_acl_output helper
# ---------------------------------------------------------------------------


class TestStripACLOutput:
    def test_single_command(self):
        text = "\nACL> read M:OUTTMP\n\nM:OUTTMP       =  7.313 DegF\n\nACL> \n"
        assert _strip_acl_output(text) == "M:OUTTMP       =  7.313 DegF"

    def test_multi_command(self):
        text = (
            "\nACL> read M:OUTTMP; read G:AMANDA\n\n"
            "M:OUTTMP       =  7.313 DegF\n\n"
            "G:AMANDA       =  66    oops\n\n"
            "ACL> \n"
        )
        result = _strip_acl_output(text)
        assert "7.313" in result
        assert "66" in result

    def test_empty_output(self):
        text = "\nACL> set x 1\n\nACL> \n"
        assert _strip_acl_output(text) == ""

    def test_no_prompts(self):
        assert _strip_acl_output("just text") == "just text"


# ---------------------------------------------------------------------------
# One-shot: SSHClient.acl()
# ---------------------------------------------------------------------------


class TestSSHClientACL:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_success(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        # Real ACL output includes prompts and echoed command
        stdout = b"\nACL> read M:OUTTMP\n\nM:OUTTMP       =  72.500 DegF\n\nACL> \n"
        chan = _make_exec_channel(stdout=stdout, exit_code=0)
        transport.open_session.return_value = chan

        result = ssh.acl("read M:OUTTMP")
        assert result == "M:OUTTMP       =  72.500 DegF"
        chan.exec_command.assert_called_once_with("acl")
        chan.sendall.assert_called_once_with(b"read M:OUTTMP\n")

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_failure_raises(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_exec_channel(stderr=b"error text\n", exit_code=1)
        transport.open_session.return_value = chan

        with pytest.raises(ACLError, match="error text"):
            ssh.acl("bad command")

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_strips_prompts(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        stdout = b"\nACL> cmd\n\n  result  \n\nACL> \n"
        chan = _make_exec_channel(stdout=stdout, exit_code=0)
        transport.open_session.return_value = chan

        result = ssh.acl("cmd")
        assert result == "result"

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_semicolons(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        stdout = b"\nACL> read M:OUTTMP; read G:AMANDA\n\nout1\nout2\n\nACL> \n"
        chan = _make_exec_channel(stdout=stdout, exit_code=0)
        transport.open_session.return_value = chan

        result = ssh.acl("read M:OUTTMP; read G:AMANDA")
        chan.sendall.assert_called_once_with(b"read M:OUTTMP; read G:AMANDA\n")
        assert "out1" in result
        assert "out2" in result

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_default_timeout(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_exec_channel(exit_code=0)
        transport.open_session.return_value = chan

        ssh.acl("cmd")
        chan.settimeout.assert_called_once_with(30.0)

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_custom_timeout(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_exec_channel(exit_code=0)
        transport.open_session.return_value = chan

        ssh.acl("cmd", timeout=5.0)
        chan.settimeout.assert_called_once_with(5.0)


# ---------------------------------------------------------------------------
# One-shot: SSHClient.acl() with list (script mode)
# ---------------------------------------------------------------------------


class TestSSHClientACLScript:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_list_writes_and_runs_script(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        # Three exec calls: write script, run acl, rm cleanup
        acl_stdout = b"\nACL> read M:OUTTMP\n\nM:OUTTMP       =  72.500 DegF\n\nACL> \n"
        transport.open_session.side_effect = [
            _make_exec_channel(exit_code=0),  # cat > script
            _make_exec_channel(stdout=acl_stdout, exit_code=0),  # acl script
            _make_exec_channel(exit_code=0),  # rm -f script
        ]

        result = ssh.acl(["read M:OUTTMP"])
        assert result == "M:OUTTMP       =  72.500 DegF"
        # Verify three exec calls happened
        assert transport.open_session.call_count == 3

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_list_script_content(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        write_chan = _make_exec_channel(exit_code=0)
        transport.open_session.side_effect = [
            write_chan,  # cat > script
            _make_exec_channel(exit_code=0),  # acl script
            _make_exec_channel(exit_code=0),  # rm
        ]

        ssh.acl(["read M:OUTTMP", "read G:AMANDA"])
        # Verify script content written to stdin
        write_chan.sendall.assert_called_once()
        written = write_chan.sendall.call_args[0][0]
        assert written == b"read M:OUTTMP\nread G:AMANDA\n"

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_list_cleans_up_on_failure(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        transport.open_session.side_effect = [
            _make_exec_channel(exit_code=0),  # cat > script
            _make_exec_channel(stderr=b"error\n", exit_code=1),  # acl fails
            _make_exec_channel(exit_code=0),  # rm still runs
        ]

        with pytest.raises(ACLError, match="ACL script failed"):
            ssh.acl(["bad command"])
        # rm -f should still have been called (3 exec calls total)
        assert transport.open_session.call_count == 3

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_empty_list_raises(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        with pytest.raises(ValueError, match="empty"):
            ssh.acl([])

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_list_script_write_failure(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        transport.open_session.side_effect = [
            _make_exec_channel(stderr=b"Permission denied\n", exit_code=1),  # cat fails
        ]

        with pytest.raises(ACLError, match="Failed to write"):
            ssh.acl(["read M:OUTTMP"])


# ---------------------------------------------------------------------------
# SSHClient.open_channel()
# ---------------------------------------------------------------------------


class TestOpenChannel:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_open_channel_returns_channel(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        mock_chan = MagicMock()
        transport.open_session.return_value = mock_chan

        chan = ssh.open_channel("acl")
        mock_chan.exec_command.assert_called_once_with("acl")
        # stdin should NOT be shut down
        mock_chan.shutdown_write.assert_not_called()
        assert chan is mock_chan

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_open_channel_with_timeout(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        mock_chan = MagicMock()
        transport.open_session.return_value = mock_chan

        ssh.open_channel("acl", timeout=10.0)
        mock_chan.settimeout.assert_called_once_with(10.0)

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_open_channel_inactive_transport(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        ssh._ensure_connected()
        transport.is_active.return_value = False

        from pacsys.ssh import SSHConnectionError

        with pytest.raises(SSHConnectionError, match="no longer active"):
            ssh.open_channel("acl")


# ---------------------------------------------------------------------------
# SSHClient.acl_session() factory
# ---------------------------------------------------------------------------


class TestACLSessionFactory:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_session_returns_session(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ssh.acl_session()
        assert isinstance(session, ACLSession)
        session.close()


# ---------------------------------------------------------------------------
# ACLSession
# ---------------------------------------------------------------------------


class TestACLSession:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_session_waits_for_initial_prompt(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel([b"Welcome to ACL\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        assert not session._closed
        session.close()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_send_returns_output(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel(
            [
                b"\nACL> ",  # initial prompt
                b"read M:OUTTMP\n\nM:OUTTMP       =  72.500 DegF\n\nACL> ",  # response
            ]
        )
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        result = session.send("read M:OUTTMP")
        assert result == "M:OUTTMP       =  72.500 DegF"
        chan.sendall.assert_called_with(b"read M:OUTTMP\n")
        session.close()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_send_strips_echo_and_output(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel(
            [
                b"\nACL> ",
                b"cmd\n\n  result text  \n\nACL> ",
            ]
        )
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        result = session.send("cmd")
        assert result == "result text"
        session.close()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_send_no_output_command(self, mock_connect, mock_transport_cls):
        """Commands like variable assignment produce no output."""
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel(
            [
                b"\nACL> ",
                b"myvar = M:OUTTMP\nACL> ",
            ]
        )
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        result = session.send("myvar = M:OUTTMP")
        assert result == ""
        session.close()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_send_on_closed_session_raises(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        session.close()
        with pytest.raises(ACLError, match="closed"):
            session.send("read M:OUTTMP")

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_context_manager(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        with ACLSession(ssh) as session:
            assert not session._closed
        assert session._closed
        chan.close.assert_called()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_close_does_not_close_ssh_client(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        session.close()
        assert transport.close.call_count == 0

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_double_close_safe(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        session.close()
        session.close()  # should not raise

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_process_exit_raises(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        # Simulate process exit on next send
        chan.recv_ready = MagicMock(return_value=False)
        chan.exit_status_ready = MagicMock(return_value=True)

        with pytest.raises(ACLError, match="exited unexpectedly"):
            session.send("read M:OUTTMP")

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_timeout_waiting_for_prompt(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        # Channel never produces data — will time out
        chan = MagicMock()
        chan.status_event = threading.Event()
        chan.closed = False
        chan.recv_ready = MagicMock(return_value=False)
        chan.recv_stderr_ready = MagicMock(return_value=False)
        chan.exit_status_ready = MagicMock(return_value=False)
        transport.open_session.return_value = chan

        with pytest.raises(ACLError, match="Timed out"):
            ACLSession(ssh, timeout=0.1)

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_multiple_sends(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel(
            [
                b"\nACL> ",
                b"cmd1\n\noutput1\n\nACL> ",
                b"cmd2\n\noutput2\n\nACL> ",
            ]
        )
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        r1 = session.send("cmd1")
        r2 = session.send("cmd2")
        assert r1 == "output1"
        assert r2 == "output2"
        session.close()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_prompt_split_across_chunks(self, mock_connect, mock_transport_cls):
        """\\nACL> prompt may arrive split across multiple recv() calls."""
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel(
            [
                b"\nAC",
                b"L> ",  # initial prompt split
                b"cmd echo\n\nresult\n\nAC",
                b"L> ",  # command response prompt also split
            ]
        )
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        result = session.send("cmd")
        assert result == "result"
        session.close()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_stderr_drained(self, mock_connect, mock_transport_cls):
        """Stderr should be consumed to prevent deadlock."""
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel([b"\nACL> "])
        # Make stderr available during init
        stderr_calls = [True, False]
        chan.recv_stderr_ready = MagicMock(side_effect=lambda: stderr_calls.pop(0) if stderr_calls else False)
        chan.recv_stderr = MagicMock(return_value=b"some warning\n")
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        chan.recv_stderr.assert_called()
        session.close()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_repr(self, mock_connect, mock_transport_cls):
        ssh, transport = _connected_ssh(mock_connect, mock_transport_cls)
        chan = _make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        assert "open" in repr(session)
        session.close()
        assert "closed" in repr(session)
