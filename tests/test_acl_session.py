"""Tests for ACL-over-SSH: one-shot SSHClient.acl() and persistent ACLSession."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from pacsys.acl_session import ACLSession, _strip_acl_output
from pacsys.errors import ACLError

from .ssh_helpers import connected_ssh, make_exec_channel, make_interactive_channel


def _mktemp_channel():
    """Remote mktemp reply: the script path acl() will use."""
    return make_exec_channel(stdout=b"/tmp/pacsys_acl_a1b2c3d4.acl\n", exit_code=0)


@pytest.fixture(autouse=True)
def _mock_getuser():
    with patch("getpass.getuser", return_value="testuser"):
        yield


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
# One-shot: SSHClient.acl() (script mode - always uses _acl_script)
# ---------------------------------------------------------------------------


class TestSSHClientACL:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_success(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        acl_stdout = b"\nACL> read M:OUTTMP\n\nM:OUTTMP       =  72.500 DegF\n\nACL> \n"
        transport.open_session.side_effect = [
            _mktemp_channel(),
            make_exec_channel(exit_code=0),  # cat > script
            make_exec_channel(stdout=acl_stdout, exit_code=0),  # acl script
            make_exec_channel(exit_code=0),  # rm -f script
        ]

        result = ssh.acl("read M:OUTTMP")
        assert result == "M:OUTTMP       =  72.500 DegF"

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_failure_raises(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        transport.open_session.side_effect = [
            _mktemp_channel(),
            make_exec_channel(exit_code=0),  # cat > script
            make_exec_channel(stderr=b"error text\n", exit_code=1),  # acl fails
            make_exec_channel(exit_code=0),  # rm -f
        ]

        with pytest.raises(ACLError, match="error text"):
            ssh.acl("bad command")

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_strips_prompts(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        stdout = b"\nACL> cmd\n\n  result  \n\nACL> \n"
        transport.open_session.side_effect = [
            _mktemp_channel(),
            make_exec_channel(exit_code=0),
            make_exec_channel(stdout=stdout, exit_code=0),
            make_exec_channel(exit_code=0),
        ]

        result = ssh.acl("cmd")
        assert result == "result"

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_semicolons(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        stdout = b"\nACL> read M:OUTTMP; read G:AMANDA\n\nout1\nout2\n\nACL> \n"
        transport.open_session.side_effect = [
            _mktemp_channel(),
            make_exec_channel(exit_code=0),
            make_exec_channel(stdout=stdout, exit_code=0),
            make_exec_channel(exit_code=0),
        ]

        result = ssh.acl("read M:OUTTMP; read G:AMANDA")
        assert "out1" in result
        assert "out2" in result


# ---------------------------------------------------------------------------
# One-shot: SSHClient.acl() with list (script mode)
# ---------------------------------------------------------------------------


class TestSSHClientACLScript:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_list_writes_and_runs_script(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        acl_stdout = b"\nACL> read M:OUTTMP\n\nM:OUTTMP       =  72.500 DegF\n\nACL> \n"
        transport.open_session.side_effect = [
            _mktemp_channel(),
            make_exec_channel(exit_code=0),  # cat > script
            make_exec_channel(stdout=acl_stdout, exit_code=0),  # acl script
            make_exec_channel(exit_code=0),  # rm -f script
        ]

        result = ssh.acl(["read M:OUTTMP"])
        assert result == "M:OUTTMP       =  72.500 DegF"
        assert transport.open_session.call_count == 4

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_list_script_content(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        write_chan = make_exec_channel(exit_code=0)
        transport.open_session.side_effect = [
            _mktemp_channel(),
            write_chan,  # cat > script
            make_exec_channel(exit_code=0),  # acl script
            make_exec_channel(exit_code=0),  # rm
        ]

        ssh.acl(["read M:OUTTMP", "read G:AMANDA"])
        write_chan.sendall.assert_called_once()
        written = write_chan.sendall.call_args[0][0]
        assert written == b"read M:OUTTMP\nread G:AMANDA\n"

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_list_cleans_up_on_failure(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        transport.open_session.side_effect = [
            _mktemp_channel(),
            make_exec_channel(exit_code=0),  # cat > script
            make_exec_channel(stderr=b"error\n", exit_code=1),  # acl fails
            make_exec_channel(exit_code=0),  # rm still runs
        ]

        with pytest.raises(ACLError, match="ACL script failed"):
            ssh.acl(["bad command"])
        assert transport.open_session.call_count == 4

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_empty_list_raises(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        with pytest.raises(ValueError, match="empty"):
            ssh.acl([])

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_acl_list_script_write_failure(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        transport.open_session.side_effect = [
            _mktemp_channel(),
            make_exec_channel(stderr=b"Permission denied\n", exit_code=1),  # cat fails
            make_exec_channel(exit_code=0),  # rm -f still removes the mktemp file
        ]

        with pytest.raises(ACLError, match="Failed to write"):
            ssh.acl(["read M:OUTTMP"])
        assert transport.open_session.call_count == 3


# ---------------------------------------------------------------------------
# SSHClient.open_channel()
# ---------------------------------------------------------------------------


class TestOpenChannel:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_open_channel_returns_channel(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        mock_chan = MagicMock()
        transport.open_session.return_value = mock_chan

        chan = ssh.open_channel("acl")
        mock_chan.exec_command.assert_called_once_with("acl")
        mock_chan.shutdown_write.assert_not_called()
        assert chan is mock_chan

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_open_channel_with_timeout(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        mock_chan = MagicMock()
        transport.open_session.return_value = mock_chan

        ssh.open_channel("acl", timeout=10.0)
        mock_chan.settimeout.assert_called_once_with(10.0)

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_open_channel_inactive_transport(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
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
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"\nACL> "])
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
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"Welcome to ACL\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        assert not session._closed
        session.close()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_send_returns_output(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel(
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
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel(
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
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel(
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
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        session.close()
        with pytest.raises(ACLError, match="closed"):
            session.send("read M:OUTTMP")

    @pytest.mark.parametrize("command", ["read M:OUTTMP\nread G:AMANDA", "read M:OUTTMP\rread G:AMANDA"])
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_send_rejects_line_breaks(self, mock_connect, mock_transport_cls, command):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        with pytest.raises(ValueError, match="line breaks"):
            session.send(command)

        assert not session._closed
        chan.sendall.assert_not_called()
        session.close()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_context_manager(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        with ACLSession(ssh) as session:
            assert not session._closed
        assert session._closed
        chan.close.assert_called()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_close_does_not_close_ssh_client(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        session.close()
        assert transport.close.call_count == 0

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_double_close_safe(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        session.close()
        session.close()  # should not raise

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_process_exit_raises(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        # Simulate process exit on next send
        chan.recv_ready = MagicMock(return_value=False)
        chan.exit_status_ready = MagicMock(return_value=True)

        with pytest.raises(ACLError, match="Process exited"):
            session.send("read M:OUTTMP")

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_timeout_waiting_for_prompt(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
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
    def test_command_timeout_closes_session(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"\nACL> "])
        transport.open_session.return_value = chan

        session = ACLSession(ssh, timeout=0.01)
        with pytest.raises(ACLError, match="Timed out"):
            session.send("slow command")

        assert session._closed
        chan.close.assert_called_once()
        with pytest.raises(ACLError, match="closed"):
            session.send("next command")
        chan.sendall.assert_called_once_with(b"slow command\n")

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_multiple_sends(self, mock_connect, mock_transport_cls):
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel(
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
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel(
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
        ssh, transport = connected_ssh(mock_connect, mock_transport_cls)
        chan = make_interactive_channel([b"\nACL> "])
        stderr_calls = [True, False]
        chan.recv_stderr_ready = MagicMock(side_effect=lambda: stderr_calls.pop(0) if stderr_calls else False)
        chan.recv_stderr = MagicMock(return_value=b"some warning\n")
        transport.open_session.return_value = chan

        session = ACLSession(ssh)
        chan.recv_stderr.assert_called()
        session.close()
