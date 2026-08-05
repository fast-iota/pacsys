"""Tests for pacsys.ssh - SSH client with multi-hop support."""

import sys
import threading
from unittest.mock import MagicMock, patch

import paramiko
import pytest

from pacsys.ssh import (
    SFTPSession,
    SSHClient,
    SSHCommandError,
    SSHConnectionError,
    SSHHop,
    SSHTimeoutError,
    Tunnel,
    _normalize_hops,
)
from tests.ssh_helpers import make_exec_channel

# pacsys.__init__ defines a function named 'ssh' that shadows the pacsys.ssh
# module. On Python <=3.12, patch("pacsys.ssh.X") resolves via getattr and
# finds the function instead of the module, breaking all mock targets.
# We grab the real module from sys.modules for patch.object() calls.
_ssh_mod = sys.modules["pacsys.ssh"]


@pytest.fixture(autouse=True)
def _mock_getuser():
    """Prevent getpass.getuser() failures in CI (no TTY)."""
    with patch("getpass.getuser", return_value="testuser"):
        yield


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

    def test_effective_username_gssapi(self):
        with patch.object(_ssh_mod, "_gssapi_username", return_value="kerbuser"):
            hop = SSHHop("host")  # auth_method="gssapi" by default, no username
            assert hop.effective_username == "kerbuser"

    def test_effective_username_password_fallback(self):
        hop = SSHHop("host", auth_method="password", password="pw")
        assert hop.effective_username == "testuser"


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
            _normalize_hops([123])  # ty: ignore[invalid-argument-type]


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
    @patch("socket.create_connection")
    @patch("paramiko.Transport")
    def test_no_connection_until_operation(self, mock_transport_cls, mock_connect):
        """Client should not connect at init time."""
        ssh = SSHClient("host.example.com")
        assert ssh.connected is False
        mock_connect.assert_not_called()
        mock_transport_cls.assert_not_called()

    def test_gssapi_import_check(self):
        """If gssapi is importable, init should succeed for gssapi hops."""
        pytest.importorskip("gssapi")
        ssh = SSHClient(SSHHop("host.example.com", auth_method="gssapi"))
        assert len(ssh.hops) == 1

    def test_non_kerberos_auth_with_gssapi_hop_raises(self):
        """Passing non-KerberosAuth for gssapi hop should raise."""
        with pytest.raises(ValueError, match="KerberosAuth"):
            SSHClient("host", auth="not-kerberos-auth")  # ty: ignore[invalid-argument-type]

    def test_key_hop_no_gssapi_check(self):
        """Key-based hop should not require gssapi validation."""
        ssh = SSHClient(SSHHop("host", auth_method="key", key_filename="/tmp/key"))
        assert ssh.connected is False


# ---------------------------------------------------------------------------
# SSHClient connection chain
# ---------------------------------------------------------------------------


class TestSSHClientConnect:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_connection_failure_cleans_up(self, mock_connect, mock_transport_cls):
        mock_connect.side_effect = OSError("Connection refused")

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        with pytest.raises(SSHConnectionError, match="Connection refused"):
            ssh._ensure_connected()

        assert ssh.connected is False

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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


class TestSSHClientExec:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_exec_success(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        chan = make_exec_channel(stdout=b"hello world\n", exit_code=0)
        mock_transport.open_session.return_value = chan

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        result = ssh.exec("echo hello world")

        assert result.ok
        assert result.stdout == "hello world\n"
        assert result.exit_code == 0
        chan.exec_command.assert_called_once_with("echo hello world")
        chan.shutdown_write.assert_called_once()
        chan.close.assert_called_once()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_exec_with_input(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        chan = make_exec_channel(exit_code=0)
        mock_transport.open_session.return_value = chan

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        result = ssh.exec("cat", input="hello")

        chan.sendall.assert_called_once_with(b"hello")
        assert result.ok

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_exec_timeout(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        chan = MagicMock()
        chan.status_event = threading.Event()
        chan.recv_ready.side_effect = TimeoutError("timed out")
        mock_transport.open_session.return_value = chan

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        with pytest.raises(SSHTimeoutError, match="timed out"):
            ssh.exec("sleep 100", timeout=1.0)

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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
        return bool(not stderr_returned[0] and stderr and not remaining and pending[0] is None)

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
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_stream_yields_lines(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        chan = _make_stream_channel([b"line1\nline2\n", b"line3\n"], exit_code=0)
        mock_transport.open_session.return_value = chan

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        lines = list(ssh.exec_stream("ls"))

        assert lines == ["line1", "line2", "line3"]

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_exec_many_returns_all(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        mock_transport.open_session.side_effect = [
            make_exec_channel(exit_code=0),
            make_exec_channel(exit_code=0),
        ]

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        results = ssh.exec_many(["cmd1", "cmd2"])

        assert len(results) == 2
        assert all(r.ok for r in results)


# ---------------------------------------------------------------------------
# SSHClient.forward()
# ---------------------------------------------------------------------------


class TestSSHClientForward:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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
    @patch("paramiko.SFTPClient.from_transport")
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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

    @patch("paramiko.SFTPClient.from_transport")
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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

    def test_close_logs_tunnel_failure_and_continues(self, caplog):
        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        failed_tunnel = MagicMock(local_port=10001)
        failed_tunnel.stop.side_effect = RuntimeError("shutdown failed")
        healthy_tunnel = MagicMock(local_port=10002)
        ssh._tunnels = [failed_tunnel, healthy_tunnel]
        ssh._cleanup_transports = MagicMock()

        ssh.close()

        failed_tunnel.stop.assert_called_once_with()
        healthy_tunnel.stop.assert_called_once_with()
        ssh._cleanup_transports.assert_called_once_with()
        assert not ssh._tunnels
        assert "Failed to stop SSH tunnel on port 10001" in caplog.text


# ---------------------------------------------------------------------------
# SSHClient context manager
# ---------------------------------------------------------------------------


class TestSSHClientContextManager:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
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

    def test_shutdown_error_still_closes_server_and_joins_thread(self):
        mock_transport = _make_mock_transport()
        tunnel = Tunnel(0, "remote", 5432, mock_transport)
        server = tunnel._server
        acceptor_thread = tunnel._acceptor_thread
        assert server is not None
        assert acceptor_thread is not None
        original_shutdown = server.shutdown

        def shutdown_then_fail():
            original_shutdown()
            raise RuntimeError("shutdown failed")

        server.shutdown = shutdown_then_fail

        with pytest.raises(RuntimeError, match="shutdown failed"):
            tunnel.stop()

        assert tunnel._server is None
        assert tunnel._acceptor_thread is None
        assert not acceptor_thread.is_alive()
        assert server.socket.fileno() == -1

    def test_join_error_still_reconciles_state(self):
        tunnel = Tunnel.__new__(Tunnel)
        tunnel.local_port = 10001
        tunnel._stop_event = threading.Event()
        server = MagicMock()
        tunnel._server = server
        tunnel._acceptor_thread = MagicMock()
        tunnel._acceptor_thread.join.side_effect = RuntimeError("join failed")

        with pytest.raises(RuntimeError, match="join failed"):
            tunnel.stop()

        server.server_close.assert_called_once_with()
        assert tunnel._server is None
        assert tunnel._acceptor_thread is None


# ---------------------------------------------------------------------------
# Auth dispatch
# ---------------------------------------------------------------------------


class TestAuthDispatch:
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_gssapi_auth_explicit_username(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", username="user"))
        ssh._ensure_connected()
        mock_transport.auth_gssapi_with_mic.assert_called_once_with("user", "host", gss_deleg_creds=True)

    @patch.object(_ssh_mod, "_gssapi_username", return_value="kerbuser")
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_gssapi_auth_from_principal(self, mock_connect, mock_transport_cls, mock_gssapi):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host"))  # no explicit username
        ssh._ensure_connected()
        mock_transport.auth_gssapi_with_mic.assert_called_once_with("kerbuser", "host", gss_deleg_creds=True)

    @patch("paramiko.RSAKey.from_private_key_file")
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_key_auth(self, mock_connect, mock_transport_cls, mock_key_load):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        mock_pkey = MagicMock()
        mock_key_load.return_value = mock_pkey

        # Create a temp key file path
        ssh = SSHClient(SSHHop("host", auth_method="key", key_filename="/tmp/test_key", username="user"))
        with patch("pathlib.Path.exists", return_value=True):
            ssh._ensure_connected()
        mock_transport.auth_publickey.assert_called_once_with("user", mock_pkey)

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_key_missing_file_raises(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="key", key_filename="/nonexistent/key"))
        with pytest.raises(SSHConnectionError, match="Key file not found"):
            ssh._ensure_connected()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_password_auth(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="password", password="secret", username="user"))
        ssh._ensure_connected()
        mock_transport.auth_password.assert_called_once_with("user", "secret")
