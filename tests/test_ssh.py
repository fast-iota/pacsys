"""Tests for pacsys.ssh - SSH client with multi-hop support."""

import socket
import threading
from unittest.mock import MagicMock, patch

import paramiko
import pytest

from pacsys.ssh import (
    SSHClient,
    SSHConnectionError,
    SSHCommandError,
    SSHHop,
    SSHTimeoutError,
    CommandResult,
    SFTPSession,
    Tunnel,
    _normalize_hops,
)


# ---------------------------------------------------------------------------
# SSHHop validation
# ---------------------------------------------------------------------------


class TestSSHHop:
    def test_empty_hostname_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SSHHop("")

    def test_whitespace_hostname_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SSHHop("   ")

    def test_bad_port_zero(self):
        with pytest.raises(ValueError, match="1-65535"):
            SSHHop("host", port=0)

    def test_bad_port_negative(self):
        with pytest.raises(ValueError, match="1-65535"):
            SSHHop("host", port=-1)

    def test_bad_port_too_large(self):
        with pytest.raises(ValueError, match="1-65535"):
            SSHHop("host", port=70000)

    def test_bad_auth_method(self):
        with pytest.raises(ValueError, match="auth_method"):
            SSHHop("host", auth_method="oauth")

    def test_key_without_filename(self):
        with pytest.raises(ValueError, match="key_filename"):
            SSHHop("host", auth_method="key")

    def test_password_without_value(self):
        with pytest.raises(ValueError, match="password required"):
            SSHHop("host", auth_method="password")

    def test_key_auth_valid(self):
        hop = SSHHop("host", auth_method="key", key_filename="~/.ssh/id_rsa")
        assert hop.key_filename == "~/.ssh/id_rsa"

    def test_password_auth_valid(self):
        hop = SSHHop("host", auth_method="password", password="secret")
        assert hop.password == "secret"

    def test_effective_username_explicit(self):
        hop = SSHHop("host", username="bob")
        assert hop.effective_username == "bob"

    @patch("pacsys.ssh._gssapi_username", return_value="kerbuser")
    def test_effective_username_gssapi(self, mock_gssapi):
        hop = SSHHop("host")  # auth_method="gssapi" by default, no username
        assert hop.effective_username == "kerbuser"

    def test_effective_username_password_fallback(self):
        hop = SSHHop("host", auth_method="password", password="pw")
        # Should use os.getlogin(), not gssapi
        assert hop.effective_username  # just check it returns something


# ---------------------------------------------------------------------------
# _normalize_hops
# ---------------------------------------------------------------------------


class TestNormalizeHops:
    def test_single_string(self):
        hops = _normalize_hops("host.example.com")
        assert len(hops) == 1
        assert hops[0].hostname == "host.example.com"

    def test_single_sshhop(self):
        hop = SSHHop("host.example.com", port=2222)
        hops = _normalize_hops(hop)
        assert len(hops) == 1
        assert hops[0].port == 2222

    def test_list_of_strings(self):
        hops = _normalize_hops(["jump.example.com", "target.example.com"])
        assert len(hops) == 2
        assert hops[0].hostname == "jump.example.com"
        assert hops[1].hostname == "target.example.com"

    def test_mixed_list(self):
        hops = _normalize_hops(["jump.example.com", SSHHop("target.example.com", port=2222)])
        assert len(hops) == 2
        assert hops[1].port == 2222

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one hop"):
            _normalize_hops([])

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match="Expected str or SSHHop"):
            _normalize_hops([123])  # type: ignore


# ---------------------------------------------------------------------------
# CommandResult
# ---------------------------------------------------------------------------


class TestCommandResult:
    def test_ok_zero(self):
        r = CommandResult("ls", 0, "file.txt\n", "")
        assert r.ok is True

    def test_ok_nonzero(self):
        r = CommandResult("ls /bad", 1, "", "not found\n")
        assert r.ok is False


# ---------------------------------------------------------------------------
# SSHClient init and lazy connection
# ---------------------------------------------------------------------------


_RealTransport = paramiko.Transport
_RealChannel = paramiko.Channel


def _make_mock_transport(active=True):
    """Create a mock paramiko.Transport."""
    t = MagicMock(spec=_RealTransport)
    t.is_active.return_value = active
    t.open_channel.return_value = MagicMock(spec=_RealChannel)
    t.open_session.return_value = MagicMock(spec=_RealChannel)
    return t


class TestSSHClientInit:
    @patch("pacsys.ssh.socket.create_connection")
    @patch("pacsys.ssh.paramiko.Transport")
    def test_no_connection_until_operation(self, mock_transport_cls, mock_connect):
        """Client should not connect at init time."""
        ssh = SSHClient("host.example.com")
        assert ssh.connected is False
        mock_connect.assert_not_called()
        mock_transport_cls.assert_not_called()

    def test_gssapi_import_check(self):
        """If gssapi is importable, init should succeed for gssapi hops."""
        # gssapi is installed in test env, so this should not raise
        ssh = SSHClient(SSHHop("host.example.com", auth_method="gssapi"))
        assert len(ssh.hops) == 1

    def test_non_kerberos_auth_with_gssapi_hop_raises(self):
        """Passing non-KerberosAuth for gssapi hop should raise."""
        with pytest.raises(ValueError, match="KerberosAuth"):
            SSHClient("host", auth="not-kerberos-auth")  # type: ignore

    def test_key_hop_no_gssapi_check(self):
        """Key-based hop should not require gssapi validation."""
        ssh = SSHClient(SSHHop("host", auth_method="key", key_filename="/tmp/key"))
        assert ssh.connected is False


# ---------------------------------------------------------------------------
# SSHClient connection chain
# ---------------------------------------------------------------------------


class TestSSHClientConnect:
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_single_hop_connects(self, mock_connect, mock_transport_cls):
        mock_sock = MagicMock()
        mock_connect.return_value = mock_sock
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host.example.com", auth_method="password", password="pw"))
        ssh._ensure_connected()

        assert ssh.connected is True
        mock_connect.assert_called_once_with(("host.example.com", 22), timeout=10.0)
        mock_transport_cls.assert_called_once_with(mock_sock)
        mock_transport.start_client.assert_called_once()
        mock_transport.set_keepalive.assert_called_once_with(30)
        mock_transport.auth_password.assert_called_once()

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_multi_hop_chain(self, mock_connect, mock_transport_cls):
        """Multi-hop should open direct-tcpip channel for second hop."""
        mock_sock = MagicMock()
        mock_connect.return_value = mock_sock

        hop1_transport = _make_mock_transport()
        hop1_channel = MagicMock(spec=paramiko.Channel)
        hop1_transport.open_channel.return_value = hop1_channel

        hop2_transport = _make_mock_transport()
        mock_transport_cls.side_effect = [hop1_transport, hop2_transport]

        ssh = SSHClient(
            [
                SSHHop("jump", auth_method="password", password="pw1"),
                SSHHop("target", auth_method="password", password="pw2"),
            ]
        )
        ssh._ensure_connected()

        assert ssh.connected is True
        assert mock_transport_cls.call_count == 2
        # Second transport built on channel from first
        hop1_transport.open_channel.assert_called_once_with("direct-tcpip", ("target", 22), ("127.0.0.1", 0))
        mock_transport_cls.assert_any_call(hop1_channel)

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_connection_failure_cleans_up(self, mock_connect, mock_transport_cls):
        mock_connect.side_effect = socket.error("Connection refused")

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        with pytest.raises(SSHConnectionError, match="Connection refused"):
            ssh._ensure_connected()

        assert ssh.connected is False

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_auth_failure_cleans_up(self, mock_connect, mock_transport_cls):
        mock_transport = MagicMock()
        mock_transport.auth_password.side_effect = paramiko.AuthenticationException("bad pw")
        mock_transport_cls.return_value = mock_transport
        mock_connect.return_value = MagicMock()

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        with pytest.raises(SSHConnectionError, match="Authentication failed"):
            ssh._ensure_connected()

        assert ssh.connected is False
        # Transport was closed during cleanup (it hadn't been appended to _transports yet)
        mock_transport.close.assert_called_once()


# ---------------------------------------------------------------------------
# SSHClient.exec()
# ---------------------------------------------------------------------------


def _make_exec_channel(stdout=b"", stderr=b"", exit_code=0):
    """Create a mock channel that simulates command execution.

    Returns data on first recv_ready check, then signals exit.
    """
    chan = MagicMock()
    chan.status_event = threading.Event()
    chan.status_event.set()

    stdout_returned = [False]
    stderr_returned = [False]

    def recv_ready():
        if not stdout_returned[0] and stdout:
            return True
        return False

    def recv(size):
        stdout_returned[0] = True
        return stdout

    def recv_stderr_ready():
        if not stderr_returned[0] and stderr:
            return True
        return False

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


class TestSSHClientExec:
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_exec_success(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        chan = _make_exec_channel(stdout=b"hello world\n", exit_code=0)
        mock_transport.open_session.return_value = chan

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        result = ssh.exec("echo hello world")

        assert result.ok
        assert result.stdout == "hello world\n"
        assert result.exit_code == 0
        chan.exec_command.assert_called_once_with("echo hello world")
        chan.shutdown_write.assert_called_once()
        chan.close.assert_called_once()

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_exec_with_input(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        chan = _make_exec_channel(exit_code=0)
        mock_transport.open_session.return_value = chan

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        result = ssh.exec("cat", input="hello")

        chan.sendall.assert_called_once_with(b"hello")
        assert result.ok

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_exec_timeout(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        chan = MagicMock()
        chan.status_event = threading.Event()
        chan.recv_ready.side_effect = socket.timeout("timed out")
        mock_transport.open_session.return_value = chan

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        with pytest.raises(SSHTimeoutError, match="timed out"):
            ssh.exec("sleep 100", timeout=1.0)

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_exec_inactive_transport_raises(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport(active=False)
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        ssh._ensure_connected()
        mock_transport.is_active.return_value = False

        with pytest.raises(SSHConnectionError, match="no longer active"):
            ssh.exec("ls")


# ---------------------------------------------------------------------------
# SSHClient.exec_stream()
# ---------------------------------------------------------------------------


def _make_stream_channel(chunks, stderr=b"", exit_code=0):
    """Create a mock channel for exec_stream tests.

    Args:
        chunks: List of bytes chunks to return sequentially from recv()
        stderr: stderr data
        exit_code: Command exit code
    """
    chan = MagicMock()
    chan.status_event = threading.Event()
    chan.status_event.set()

    remaining = list(chunks)
    pending = [None]  # chunk ready to be recv()'d
    stderr_returned = [False]

    def recv_ready():
        if pending[0] is not None:
            return True
        if remaining:
            pending[0] = remaining.pop(0)
            return True
        return False

    def recv(size):
        data = pending[0]
        pending[0] = None
        return data

    def recv_stderr_ready():
        if not stderr_returned[0] and stderr and not remaining and pending[0] is None:
            return True
        return False

    def recv_stderr(size):
        stderr_returned[0] = True
        return stderr

    def exit_status_ready():
        return not remaining and pending[0] is None

    chan.recv_ready = MagicMock(side_effect=lambda: recv_ready())
    chan.recv = MagicMock(side_effect=lambda size: recv(size))
    chan.recv_stderr_ready = MagicMock(side_effect=lambda: recv_stderr_ready())
    chan.recv_stderr = MagicMock(side_effect=lambda size: recv_stderr(size))
    chan.exit_status_ready = MagicMock(side_effect=lambda: exit_status_ready())
    chan.recv_exit_status.return_value = exit_code
    return chan


class TestSSHClientExecStream:
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_stream_yields_lines(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        chan = _make_stream_channel([b"line1\nline2\n", b"line3\n"], exit_code=0)
        mock_transport.open_session.return_value = chan

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        lines = list(ssh.exec_stream("ls"))

        assert lines == ["line1", "line2", "line3"]

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_stream_nonzero_exit_raises(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        chan = _make_stream_channel([], stderr=b"error msg\n", exit_code=1)
        mock_transport.open_session.return_value = chan

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        with pytest.raises(SSHCommandError, match="error msg"):
            list(ssh.exec_stream("bad_cmd"))


# ---------------------------------------------------------------------------
# SSHClient.exec_many()
# ---------------------------------------------------------------------------


class TestSSHClientExecMany:
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_exec_many_returns_all(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        mock_transport.open_session.side_effect = [
            _make_exec_channel(exit_code=0),
            _make_exec_channel(exit_code=0),
        ]

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        results = ssh.exec_many(["cmd1", "cmd2"])

        assert len(results) == 2
        assert all(r.ok for r in results)


# ---------------------------------------------------------------------------
# SSHClient.forward()
# ---------------------------------------------------------------------------


class TestSSHClientForward:
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_forward_creates_tunnel(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        tunnel = ssh.forward(0, "db.internal", 5432)

        try:
            assert tunnel.active
            assert tunnel.local_port > 0
            assert tunnel.remote_host == "db.internal"
            assert tunnel.remote_port == 5432
        finally:
            tunnel.stop()

        assert not tunnel.active

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_forward_tracked_and_cleaned(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        tunnel = ssh.forward(0, "db.internal", 5432)
        assert len(ssh._tunnels) == 1

        ssh.close()
        assert not tunnel.active
        assert len(ssh._tunnels) == 0


# ---------------------------------------------------------------------------
# SSHClient.sftp()
# ---------------------------------------------------------------------------


class TestSSHClientSFTP:
    @patch("pacsys.ssh.paramiko.SFTPClient.from_transport")
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_sftp_returns_session(self, mock_connect, mock_transport_cls, mock_sftp_from):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        mock_sftp = MagicMock(spec=paramiko.SFTPClient)
        mock_sftp_from.return_value = mock_sftp

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        session = ssh.sftp()

        assert isinstance(session, SFTPSession)
        mock_sftp_from.assert_called_once_with(mock_transport)

    @patch("pacsys.ssh.paramiko.SFTPClient.from_transport")
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_sftp_none_raises(self, mock_connect, mock_transport_cls, mock_sftp_from):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport
        mock_sftp_from.return_value = None

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        with pytest.raises(SSHConnectionError, match="Failed to open SFTP"):
            ssh.sftp()


# ---------------------------------------------------------------------------
# SFTPSession
# ---------------------------------------------------------------------------


class TestSFTPSession:
    def test_context_manager(self):
        mock_sftp = MagicMock(spec=paramiko.SFTPClient)
        with SFTPSession(mock_sftp) as s:
            s.listdir("/tmp")
        mock_sftp.listdir.assert_called_once_with("/tmp")
        mock_sftp.close.assert_called_once()

    def test_operations_delegate(self):
        mock_sftp = MagicMock(spec=paramiko.SFTPClient)
        s = SFTPSession(mock_sftp)

        s.get("/remote/file", "/local/file")
        mock_sftp.get.assert_called_once_with("/remote/file", "/local/file")

        s.put("/local/file", "/remote/file")
        mock_sftp.put.assert_called_once_with("/local/file", "/remote/file")

        s.mkdir("/new_dir", 0o700)
        mock_sftp.mkdir.assert_called_once_with("/new_dir", 0o700)

        s.remove("/old_file")
        mock_sftp.remove.assert_called_once_with("/old_file")

        s.stat("/some_file")
        mock_sftp.stat.assert_called_once_with("/some_file")

        s.close()
        mock_sftp.close.assert_called_once()


# ---------------------------------------------------------------------------
# SSHClient.close()
# ---------------------------------------------------------------------------


class TestSSHClientClose:
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_close_disconnects(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        ssh._ensure_connected()
        assert ssh.connected

        ssh.close()
        assert not ssh.connected
        mock_transport.close.assert_called()

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_double_close_safe(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        ssh._ensure_connected()
        ssh.close()
        ssh.close()  # should not raise

    def test_close_without_connect(self):
        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        ssh.close()  # should not raise


# ---------------------------------------------------------------------------
# SSHClient context manager
# ---------------------------------------------------------------------------


class TestSSHClientContextManager:
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_context_manager(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        with SSHClient(SSHHop("host", auth_method="password", password="pw")) as ssh:
            ssh._ensure_connected()
            assert ssh.connected
        assert not ssh.connected


# ---------------------------------------------------------------------------
# Tunnel
# ---------------------------------------------------------------------------


class TestTunnel:
    def test_stop_idempotent(self):
        mock_transport = _make_mock_transport()
        tunnel = Tunnel(0, "remote", 5432, mock_transport)
        tunnel.stop()
        tunnel.stop()  # should not raise

    def test_context_manager(self):
        mock_transport = _make_mock_transport()
        with Tunnel(0, "remote", 5432, mock_transport) as t:
            assert t.active
        assert not t.active


# ---------------------------------------------------------------------------
# Auth dispatch
# ---------------------------------------------------------------------------


class TestAuthDispatch:
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_gssapi_auth_explicit_username(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", username="user"))
        ssh._ensure_connected()
        mock_transport.auth_gssapi_with_mic.assert_called_once_with("user", "host", gss_deleg_creds=True)

    @patch("pacsys.ssh._gssapi_username", return_value="kerbuser")
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_gssapi_auth_from_principal(self, mock_connect, mock_transport_cls, mock_gssapi):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host"))  # no explicit username
        ssh._ensure_connected()
        mock_transport.auth_gssapi_with_mic.assert_called_once_with("kerbuser", "host", gss_deleg_creds=True)

    @patch("pacsys.ssh.paramiko.RSAKey.from_private_key_file")
    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_key_auth(self, mock_connect, mock_transport_cls, mock_key_load):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        mock_pkey = MagicMock()
        mock_key_load.return_value = mock_pkey

        # Create a temp key file path
        ssh = SSHClient(SSHHop("host", auth_method="key", key_filename="/tmp/test_key", username="user"))
        with patch("pacsys.ssh.Path.exists", return_value=True):
            ssh._ensure_connected()
        mock_transport.auth_publickey.assert_called_once_with("user", mock_pkey)

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_key_missing_file_raises(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="key", key_filename="/nonexistent/key"))
        with pytest.raises(SSHConnectionError, match="Key file not found"):
            ssh._ensure_connected()

    @patch("pacsys.ssh.paramiko.Transport")
    @patch("pacsys.ssh.socket.create_connection")
    def test_password_auth(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="password", password="secret", username="user"))
        ssh._ensure_connected()
        mock_transport.auth_password.assert_called_once_with("user", "secret")
