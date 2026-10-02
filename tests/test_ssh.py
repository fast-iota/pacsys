"""Tests for pacsys.ssh - SSH client with multi-hop support."""

import threading
from unittest.mock import MagicMock, patch

import paramiko
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

from pacsys.ssh import (
    CommandResult,
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


@pytest.fixture(autouse=True)
def _mock_getuser():
    """Prevent getpass.getuser() failures in CI (no TTY)."""
    with patch("getpass.getuser", return_value="testuser"):
        yield


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


# ---------------------------------------------------------------------------
# SSHClient init and lazy connection
# ---------------------------------------------------------------------------


_RealTransport = paramiko.Transport
_RealChannel = paramiko.Channel


def _rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _ec_key():
    return ec.generate_private_key(ec.SECP256R1())


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
        hop1_transport.open_channel.assert_called_once_with(
            "direct-tcpip", ("target", 22), ("127.0.0.1", 0), timeout=10.0
        )
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

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_missing_key_file_cleans_up_and_reports_hop(self, mock_connect, mock_transport_cls):
        """SSHConnectionError from _authenticate (missing key) must clean up hop-1 state."""
        hop1_transport = _make_mock_transport()
        hop2_transport = _make_mock_transport()
        mock_transport_cls.side_effect = [hop1_transport, hop2_transport]
        mock_connect.return_value = MagicMock()

        hops = [
            SSHHop("jump", auth_method="password", password="pw"),
            SSHHop("target", auth_method="key", key_filename="/nonexistent/key"),
        ]
        ssh = SSHClient(hops)
        with pytest.raises(SSHConnectionError, match="Key file not found") as exc_info:
            ssh._ensure_connected()

        assert exc_info.value.hop is hops[1]
        assert ssh.connected is False
        assert ssh._transports == []
        assert ssh._channels == []
        hop1_transport.close.assert_called()
        hop2_transport.close.assert_called()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_retry_after_failure_starts_fresh(self, mock_connect, mock_transport_cls):
        """A failed connect must not leave stale transports for the next attempt to chain through."""
        bad_transport = _make_mock_transport()
        bad_transport.auth_password.side_effect = paramiko.AuthenticationException("bad pw")
        good1, good2 = _make_mock_transport(), _make_mock_transport()
        mock_transport_cls.side_effect = [bad_transport, good1, good2]
        mock_connect.return_value = MagicMock()

        ssh = SSHClient(
            [
                SSHHop("jump", auth_method="password", password="pw"),
                SSHHop("target", auth_method="password", password="pw"),
            ]
        )
        with pytest.raises(SSHConnectionError):
            ssh._ensure_connected()
        assert ssh._transports == []

        ssh._ensure_connected()
        assert ssh.connected is True
        # Retry chained hop 2 through the fresh hop-1 transport, not a stale one
        good1.open_channel.assert_called_once()
        bad_transport.open_channel.assert_not_called()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_multi_hop_failure_reports_failing_hop(self, mock_connect, mock_transport_cls):
        """The raised error must name the hop that failed, not hop 0."""
        hop1_transport = _make_mock_transport()
        hop2_transport = _make_mock_transport()
        hop2_transport.auth_password.side_effect = paramiko.AuthenticationException("bad pw")
        mock_transport_cls.side_effect = [hop1_transport, hop2_transport]
        mock_connect.return_value = MagicMock()

        hops = [
            SSHHop("jump", auth_method="password", password="pw1"),
            SSHHop("target", auth_method="password", password="pw2"),
        ]
        ssh = SSHClient(hops)
        with pytest.raises(SSHConnectionError, match="Authentication failed") as exc_info:
            ssh._ensure_connected()
        assert exc_info.value.hop is hops[1]

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_keyboard_interrupt_cleans_up_and_reraises(self, mock_connect, mock_transport_cls):
        mock_transport = _make_mock_transport()
        mock_transport.auth_password.side_effect = KeyboardInterrupt
        mock_transport_cls.return_value = mock_transport
        mock_connect.return_value = MagicMock()

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        with pytest.raises(KeyboardInterrupt):
            ssh._ensure_connected()
        assert ssh._transports == []
        mock_transport.close.assert_called_once()

    def test_effective_username_gss_failure_wrapped(self):
        """Direct effective_username access must raise AuthenticationError, not raw GSSError."""
        from pacsys.errors import AuthenticationError

        class MockGSSError(Exception):
            pass

        class MockGSSAPI:
            class exceptions:  # noqa: N801 -- GSSAPI namespace
                GSSError = MockGSSError

            @staticmethod
            def Credentials(usage=None):  # noqa: N802 -- GSSAPI method name
                raise MockGSSError("no ticket")

        hop = SSHHop("host", auth_method="gssapi")
        with patch.dict("sys.modules", {"gssapi": MockGSSAPI()}):
            with pytest.raises(AuthenticationError, match="No valid Kerberos credentials"):
                _ = hop.effective_username

    def test_gssapi_error_wrapped(self):
        """Raw GSSError from GSSAPI auth is wrapped as SSHConnectionError with the hop."""
        gssapi_exc = pytest.importorskip("gssapi.exceptions")
        hop = SSHHop("host", username="user", auth_method="gssapi")
        ssh = SSHClient(hop)
        transport = _make_mock_transport()
        transport.auth_gssapi_with_mic.side_effect = gssapi_exc.GSSError(851968, 0)
        with pytest.raises(SSHConnectionError, match="GSSAPI authentication failed") as exc_info:
            ssh._authenticate(transport, hop)
        assert exc_info.value.hop is hop

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_socket_closed_if_transport_ctor_fails(self, mock_connect, mock_transport_cls):
        mock_sock = MagicMock()
        mock_connect.return_value = mock_sock
        mock_transport_cls.side_effect = paramiko.SSHException("bad banner")

        ssh = SSHClient(SSHHop("host", auth_method="password", password="pw"))
        with pytest.raises(SSHConnectionError, match="Connection failed"):
            ssh._ensure_connected()
        mock_sock.close.assert_called_once()


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
    def test_exec_timeout_includes_session_open(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        mock_transport.open_session.side_effect = paramiko.SSHException("Timeout opening channel.")
        with SSHClient(SSHHop("host", auth_method="password", password="pw")) as ssh:
            with pytest.raises(SSHTimeoutError, match="timed out"):
                ssh.exec("true", timeout=0.05)
        mock_transport.open_session.assert_called_once_with(timeout=0.05)


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

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_gssapi_delegation_can_be_disabled(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", username="user", delegate_credentials=False))
        ssh._ensure_connected()
        mock_transport.auth_gssapi_with_mic.assert_called_once_with("user", "host", gss_deleg_creds=False)

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_gssapi_auth_username_from_auth_principal(self, mock_connect, mock_transport_cls):
        """SSHClient(auth=KerberosAuth(...)) must log in as that principal, not the default cache one."""
        from pacsys.auth import KerberosAuth

        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport
        auth = MagicMock(spec=KerberosAuth)
        auth.name = None
        auth.principal = "operator@FNAL.GOV"

        with patch("pacsys.ssh._gssapi_username", side_effect=AssertionError("default cache consulted")):
            ssh = SSHClient(SSHHop("host"), auth=auth)
            ssh._ensure_connected()
        mock_transport.auth_gssapi_with_mic.assert_called_once_with("operator", "host", gss_deleg_creds=True)

    @patch("pacsys.ssh._default_principal", return_value="nikita@FNAL.GOV")
    def test_named_auth_must_be_default_principal(self, _mock_default):
        """paramiko can only present the default credential; a different named principal fails at init."""
        from pacsys.auth import KerberosAuth
        from pacsys.errors import AuthenticationError

        auth = MagicMock(spec=KerberosAuth)
        auth.name = "operator@FNAL.GOV"
        auth.principal = "operator@FNAL.GOV"
        with pytest.raises(AuthenticationError, match="default credential-cache principal nikita@FNAL.GOV"):
            SSHClient(SSHHop("host"), auth=auth)

        auth.principal = "nikita@FNAL.GOV"
        auth.name = "nikita@FNAL.GOV"
        assert SSHClient(SSHHop("host"), auth=auth).connected is False

    @patch("pacsys.ssh._gssapi_username", return_value="kerbuser")
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_gssapi_auth_from_principal(self, mock_connect, mock_transport_cls, mock_gssapi):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host"))  # no explicit username
        ssh._ensure_connected()
        mock_transport.auth_gssapi_with_mic.assert_called_once_with("kerbuser", "host", gss_deleg_creds=True)

    @pytest.mark.parametrize(
        "key_factory,fmt,expected_cls",
        [
            (_rsa_key, serialization.PrivateFormat.TraditionalOpenSSL, paramiko.RSAKey),
            (_rsa_key, serialization.PrivateFormat.OpenSSH, paramiko.RSAKey),
            (ed25519.Ed25519PrivateKey.generate, serialization.PrivateFormat.OpenSSH, paramiko.Ed25519Key),
            (_ec_key, serialization.PrivateFormat.TraditionalOpenSSL, paramiko.ECDSAKey),
            (_ec_key, serialization.PrivateFormat.OpenSSH, paramiko.ECDSAKey),
        ],
    )
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_key_auth_detects_type_from_contents(
        self, mock_connect, mock_transport_cls, key_factory, fmt, expected_cls, tmp_path
    ):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport
        key = key_factory()
        key_file = tmp_path / "id_key"  # name carries no type hint
        key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, fmt, serialization.NoEncryption()))

        ssh = SSHClient(SSHHop("host", auth_method="key", key_filename=str(key_file), username="user"))
        ssh._ensure_connected()

        (username, pkey), _ = mock_transport.auth_publickey.call_args
        assert username == "user"
        assert type(pkey) is expected_cls
        expected_pub = key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
        assert f"{pkey.get_name()} {pkey.get_base64()}".encode() == expected_pub
        mock_transport.auth_password.assert_not_called()

    @pytest.mark.parametrize(
        "fmt",
        [serialization.PrivateFormat.OpenSSH, serialization.PrivateFormat.TraditionalOpenSSL],
    )
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_key_encrypted_raises(self, mock_connect, mock_transport_cls, fmt, tmp_path):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport
        key_file = tmp_path / "id_rsa"
        key_file.write_bytes(
            _rsa_key().private_bytes(serialization.Encoding.PEM, fmt, serialization.BestAvailableEncryption(b"pw"))
        )

        ssh = SSHClient(SSHHop("host", auth_method="key", key_filename=str(key_file)))
        with pytest.raises(SSHConnectionError, match="is encrypted") as exc_info:
            ssh._ensure_connected()
        assert str(key_file) in str(exc_info.value)
        mock_transport.auth_publickey.assert_not_called()
        mock_transport.close.assert_called_once()

    @pytest.mark.parametrize(
        "content",
        [
            b"not a key\n",
            b"-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n-----END OPENSSH PRIVATE KEY-----\n",
            # PKCS#8 parses but Paramiko key classes do not accept it
            _rsa_key().private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            ),
        ],
        ids=["garbage", "truncated-openssh", "pkcs8"],
    )
    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_key_invalid_file_cleans_up_and_reports_hop(self, mock_connect, mock_transport_cls, content, tmp_path):
        hop1_transport = _make_mock_transport()
        hop2_transport = _make_mock_transport()
        mock_transport_cls.side_effect = [hop1_transport, hop2_transport]
        mock_connect.return_value = MagicMock()
        key_file = tmp_path / "id_ed25519"
        key_file.write_bytes(content)
        hops: list[str | SSHHop] = [
            SSHHop("jump", auth_method="password", password="pw"),
            SSHHop("target", auth_method="key", key_filename=str(key_file)),
        ]

        ssh = SSHClient(hops)
        with pytest.raises(SSHConnectionError, match="Invalid private key file") as exc_info:
            ssh._ensure_connected()

        assert exc_info.value.hop is hops[1]
        assert str(key_file) in str(exc_info.value)
        assert ssh.connected is False
        assert ssh._transports == []
        assert ssh._channels == []
        hop1_transport.close.assert_called()
        hop2_transport.close.assert_called()
        hop2_transport.auth_publickey.assert_not_called()
        hop2_transport.auth_password.assert_not_called()

    @patch("paramiko.Transport")
    @patch("socket.create_connection")
    def test_password_auth(self, mock_connect, mock_transport_cls):
        mock_connect.return_value = MagicMock()
        mock_transport = _make_mock_transport()
        mock_transport_cls.return_value = mock_transport

        ssh = SSHClient(SSHHop("host", auth_method="password", password="secret", username="user"))
        ssh._ensure_connected()
        mock_transport.auth_password.assert_called_once_with("user", "secret")


# ---------------------------------------------------------------------------
# ACL script execution
# ---------------------------------------------------------------------------


class TestACLScript:
    _MKTEMP = "mktemp --suffix=.acl /tmp/pacsys_acl_XXXXXXXX"

    @staticmethod
    def _client():
        return SSHClient(SSHHop("host", auth_method="key", key_filename="/tmp/key"))

    @staticmethod
    def _result(command, exit_code=0, stdout="", stderr=""):
        return CommandResult(command=command, exit_code=exit_code, stdout=stdout, stderr=stderr)

    @pytest.mark.parametrize(
        ("mktemp_stdout", "path"),
        [
            ("/tmp/pacsys_acl_a1b2c3d4.acl\n", "/tmp/pacsys_acl_a1b2c3d4.acl"),
            ("Welcome to host\n/tmp/pacsys_acl_a1b2c3d4.acl\nlogout\n", "/tmp/pacsys_acl_a1b2c3d4.acl"),  # banners
        ],
    )
    def test_uses_remote_mktemp_and_removes_exact_path(self, mktemp_stdout, path):
        """The script path is the mktemp-shaped line of the output and is removed even when acl fails."""
        import shlex

        from pacsys.errors import ACLError

        ssh = self._client()
        q = shlex.quote(path)

        def fake_exec(command, timeout=None, input=None):
            if command == self._MKTEMP:
                return self._result(command, stdout=mktemp_stdout)
            if command.startswith("acl "):
                return self._result(command, exit_code=1, stderr="boom")
            return self._result(command)

        with patch.object(ssh, "exec", side_effect=fake_exec) as ex:
            with pytest.raises(ACLError, match="boom"):
                ssh.acl("read M:OUTTMP")
        commands = [c.args[0] for c in ex.call_args_list]
        assert commands == [self._MKTEMP, f"cat > {q}", f"acl {q}", f"rm -f {q}"]
        assert ex.call_args_list[1].kwargs["input"] == "read M:OUTTMP\n"

    def test_failure_message_keeps_stdout_reason(self):
        """stderr may be console noise (xprop) while the real ACL error is in stdout."""
        from pacsys.errors import ACLError

        ssh = self._client()

        def fake_exec(command, timeout=None, input=None):
            if command == self._MKTEMP:
                return self._result(command, stdout="/tmp/pacsys_acl_a1b2c3d4.acl\n")
            if command.startswith("acl "):
                return self._result(command, exit_code=1, stdout="... - CLIB_NOPRIV\n", stderr="xprop: no display\n")
            return self._result(command)

        with patch.object(ssh, "exec", side_effect=fake_exec):
            with pytest.raises(ACLError, match="xprop: no display; ... - CLIB_NOPRIV"):
                ssh.acl("set Z:ACLTST 1")

    def test_mktemp_failure_runs_nothing_else(self):
        from pacsys.errors import ACLError

        ssh = self._client()
        with patch.object(ssh, "exec", return_value=self._result(self._MKTEMP, exit_code=1, stderr="ro fs")) as ex:
            with pytest.raises(ACLError, match="Failed to create ACL script file: ro fs"):
                ssh.acl("read M:OUTTMP")
        assert [c.args[0] for c in ex.call_args_list] == [self._MKTEMP]

    @pytest.mark.parametrize(
        "cleanup_error",
        [
            SSHTimeoutError("cleanup timed out"),
            paramiko.ChannelException(1, "cleanup channel rejected"),
            paramiko.SSHException("cleanup session disconnected"),
            EOFError("cleanup transport EOF"),
            OSError("cleanup socket failed"),
        ],
    )
    def test_cleanup_failure_does_not_mask_success(self, caplog, cleanup_error):
        path = "/tmp/pacsys_acl_a1b2c3d4.acl"
        ssh = self._client()
        transport = _make_mock_transport()
        channels = [
            make_exec_channel(stdout=f"{path}\n".encode()),
            make_exec_channel(),
            make_exec_channel(stdout=b"M:OUTTMP = 72.5\n"),
        ]
        transport.open_session.side_effect = [*channels, cleanup_error]
        ssh._transports = [transport]
        ssh._connected = True

        assert ssh.acl("read M:OUTTMP") == "M:OUTTMP = 72.5"
        assert transport.open_session.call_count == 4
        transport.open_session.assert_called_with(timeout=5.0)
        for channel in channels:
            channel.close.assert_called_once()
        assert path in caplog.text
        assert str(cleanup_error) in caplog.text

    @pytest.mark.parametrize("stage", ["write", "acl"])
    def test_cleanup_transport_failure_preserves_operation_error(self, caplog, stage):
        path = "/tmp/pacsys_acl_a1b2c3d4.acl"
        ssh = self._client()
        transport = _make_mock_transport()
        original = SSHTimeoutError("operation timed out")
        cleanup_error = paramiko.ChannelException(1, "cleanup channel rejected")
        responses = [make_exec_channel(stdout=f"{path}\n".encode())]
        if stage == "acl":
            responses.append(make_exec_channel())
        responses.append(original)
        transport.open_session.side_effect = [*responses, cleanup_error]
        ssh._transports = [transport]
        ssh._connected = True

        with pytest.raises(SSHTimeoutError) as exc_info:
            ssh._acl_script(["read M:OUTTMP"], 30.0, str.strip)

        assert exc_info.value is original
        transport.open_session.assert_called_with(timeout=5.0)
        assert transport.open_session.call_count == len(responses) + 1
        assert path in caplog.text
        assert str(cleanup_error) in caplog.text

    @pytest.mark.parametrize(
        "stdout", ["/etc/passwd\n", "/tmp/pacsys_acl_x/../../home/u/.bashrc\n", "/tmp/pacsys_acl_a b$c.acl\n", ""]
    )
    def test_unexpected_mktemp_output_rejected(self, stdout):
        """Only a path of exactly mktemp's shape is ever written, executed, or removed."""
        from pacsys.errors import ACLError

        ssh = self._client()
        with patch.object(ssh, "exec", return_value=self._result(self._MKTEMP, stdout=stdout)) as ex:
            with pytest.raises(ACLError, match="Failed to create ACL script file"):
                ssh.acl("read M:OUTTMP")
        assert ex.call_count == 1
