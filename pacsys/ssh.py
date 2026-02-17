"""
SSH utility for command execution, port tunneling, and SFTP over multi-hop SSH chains.

Uses paramiko + GSSAPI (Kerberos) for authentication. Not a Backend subclass -
this is a general-purpose utility for remote operations on ACNET hosts.

Example:
    from pacsys.ssh import SSHClient

    with SSHClient("target.fnal.gov") as ssh:
        result = ssh.exec("ls /tmp")
        print(result.stdout)

    # Multi-hop
    with SSHClient(["jump.fnal.gov", "target.fnal.gov"]) as ssh:
        result = ssh.exec("hostname")
"""

from __future__ import annotations

import getpass
import logging
import select
import socket
import socketserver
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Union

import paramiko

if TYPE_CHECKING:
    from pacsys.acl_session import ACLSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SSHError(Exception):
    """Base exception for SSH operations."""


class SSHConnectionError(SSHError):
    """Connection or authentication failure."""

    def __init__(self, message: str, hop: Optional[SSHHop] = None):
        self.hop = hop
        hop_info = f" (hop: {hop.hostname}:{hop.port})" if hop else ""
        super().__init__(f"{message}{hop_info}")


class SSHCommandError(SSHError):
    """Command exited with non-zero status."""

    def __init__(self, command: str, exit_code: int, stderr: str):
        self.command = command
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(f"Command {command!r} failed (exit={exit_code}): {stderr.strip()}")


class SSHTimeoutError(SSHError):
    """Operation timed out."""


# ---------------------------------------------------------------------------
# SSHHop
# ---------------------------------------------------------------------------


def _gssapi_username() -> str:
    """Extract username from Kerberos principal"""
    import gssapi

    creds = gssapi.Credentials(usage="initiate")
    principal = str(creds.name)
    return principal.split("@")[0]


@dataclass(frozen=True)
class SSHHop:
    """Configuration for a single SSH hop.

    Args:
        hostname: SSH server hostname (required, non-empty)
        port: SSH port (default 22)
        username: SSH username (default: current OS user)
        auth_method: "gssapi", "key", or "password"
        key_filename: Path to private key (required when auth_method="key")
        password: Password (required when auth_method="password", excluded from repr)
    """

    hostname: str
    port: int = 22
    username: Optional[str] = None
    auth_method: str = "gssapi"
    key_filename: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.hostname or not self.hostname.strip():
            raise ValueError("hostname must be a non-empty string")
        if not isinstance(self.port, int) or self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be 1-65535, got {self.port}")
        if self.auth_method not in ("gssapi", "key", "password"):
            raise ValueError(f"auth_method must be 'gssapi', 'key', or 'password', got {self.auth_method!r}")
        if self.auth_method == "key" and not self.key_filename:
            raise ValueError("key_filename required when auth_method='key'")
        if self.auth_method == "password" and not self.password:
            raise ValueError("password required when auth_method='password'")

    @property
    def effective_username(self) -> str:
        if self.username:
            return self.username
        if self.auth_method == "gssapi":
            return _gssapi_username()
        return getpass.getuser()


# ---------------------------------------------------------------------------
# CommandResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    """Result of a remote command execution."""

    command: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# ---------------------------------------------------------------------------
# Tunnel
# ---------------------------------------------------------------------------


class Tunnel:
    """SSH port forward: local_port -> remote_host:remote_port via SSH transport.

    Use as context manager or call stop() explicitly.
    """

    def __init__(
        self,
        local_port: int,
        remote_host: str,
        remote_port: int,
        transport: paramiko.Transport,
    ):
        self.local_port = local_port
        self.remote_host = remote_host
        self.remote_port = remote_port
        self._transport = transport
        self._stop_event = threading.Event()
        self._server: Optional[socketserver.TCPServer] = None
        self._acceptor_thread: Optional[threading.Thread] = None

        self._start()

    @property
    def active(self) -> bool:
        return not self._stop_event.is_set() and self._server is not None

    def _start(self):
        tunnel = self

        class ForwardHandler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    chan = tunnel._transport.open_channel(
                        "direct-tcpip",
                        (tunnel.remote_host, tunnel.remote_port),
                        self.request.getpeername(),
                    )
                except Exception as e:
                    logger.error("Tunnel channel open failed: %s", e)
                    return

                try:
                    _bidirectional_forward(self.request, chan, tunnel._stop_event)
                finally:
                    chan.close()

        class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = ThreadedTCPServer(("127.0.0.1", self.local_port), ForwardHandler)
        # Update local_port in case 0 was passed (OS-assigned)
        self.local_port = self._server.server_address[1]

        self._acceptor_thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"ssh-tunnel-{self.local_port}",
            daemon=True,
        )
        self._acceptor_thread.start()
        logger.info("Tunnel listening on 127.0.0.1:%d -> %s:%d", self.local_port, self.remote_host, self.remote_port)

    def stop(self):
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._acceptor_thread:
            self._acceptor_thread.join(timeout=3.0)
        logger.info("Tunnel on port %d stopped", self.local_port)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def __repr__(self):
        state = "active" if self.active else "stopped"
        return f"Tunnel(127.0.0.1:{self.local_port} -> {self.remote_host}:{self.remote_port}, {state})"


def _bidirectional_forward(sock: socket.socket, chan: paramiko.Channel, stop_event: threading.Event):
    """Bidirectional data forwarding between a socket and an SSH channel.

    Uses select.select for cross-platform I/O multiplexing. This works on
    Windows because paramiko.Channel.fileno() returns a socket-backed fd
    via paramiko.pipe.WindowsPipe (loopback socket pair), not an os.pipe() fd.
    """
    while not stop_event.is_set():
        r, _, _ = select.select([sock, chan], [], [], 1.0)
        if sock in r:
            data = sock.recv(16384)
            if not data:
                break
            chan.sendall(data)
        if chan in r:
            data = chan.recv(16384)
            if not data:
                break
            sock.sendall(data)


# ---------------------------------------------------------------------------
# SFTPSession
# ---------------------------------------------------------------------------


class SFTPSession:
    """Thin wrapper around paramiko.SFTPClient with context manager support."""

    def __init__(self, sftp_client: paramiko.SFTPClient):
        self._sftp = sftp_client

    def get(self, remote_path: str, local_path: str) -> None:
        self._sftp.get(remote_path, local_path)

    def put(self, local_path: str, remote_path: str) -> None:
        self._sftp.put(local_path, remote_path)

    def listdir(self, path: str = ".") -> list[str]:
        return self._sftp.listdir(path)

    def stat(self, path: str) -> paramiko.SFTPAttributes:
        return self._sftp.stat(path)

    def mkdir(self, path: str, mode: int = 0o755) -> None:
        self._sftp.mkdir(path, mode)

    def remove(self, path: str) -> None:
        self._sftp.remove(path)

    def close(self) -> None:
        self._sftp.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# RemoteProcess
# ---------------------------------------------------------------------------


class RemoteProcess:
    """Persistent interactive process over SSH. Dumb bidirectional pipe.

    Not thread-safe. Context manager. Does NOT own SSHClient.

    Args:
        ssh: Connected SSHClient instance
        command: Command to execute on the remote host
        timeout: Default timeout for read operations in seconds
    """

    _MAX_BUF = 16 * 1024 * 1024  # 16 MB

    def __init__(self, ssh: SSHClient, command: str, *, timeout: float = 30.0):
        self._channel = ssh.open_channel(command, timeout=timeout)
        self._timeout = timeout
        self._buf = b""
        self._closed = False

    def send_line(self, line: str) -> None:
        """Send line + newline."""
        self._channel.sendall(f"{line}\n".encode())

    def send_bytes(self, data: bytes) -> None:
        """Send raw bytes."""
        self._channel.sendall(data)

    def read_until(self, marker: bytes, timeout: float | None = None) -> bytes:
        """Read until marker found in stream, return bytes before marker.

        Marker is consumed from buffer.

        Raises:
            SSHTimeoutError: If timeout expires before marker found
            SSHError: If channel closes before marker found
        """
        t = timeout if timeout is not None else self._timeout
        deadline = time.monotonic() + t

        while True:
            idx = self._buf.find(marker)
            if idx >= 0:
                output = self._buf[:idx]
                self._buf = self._buf[idx + len(marker) :]
                return output

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SSHTimeoutError(
                    f"Timed out waiting for marker {marker!r} after {t}s (buffer tail: {self._buf[-200:]!r})"
                )

            self._drain_stderr()

            if self._channel.recv_ready():
                data = self._channel.recv(65536)
                if not data:
                    raise SSHError(f"Channel closed while waiting for marker {marker!r}")
                self._buf += data
                if len(self._buf) > self._MAX_BUF:
                    raise SSHError(f"Buffer exceeded {self._MAX_BUF} bytes waiting for marker {marker!r}")
            elif self._channel.closed or self._channel.exit_status_ready():
                raise SSHError(
                    f"Process exited while waiting for marker {marker!r} (buffer tail: {self._buf[-200:]!r})"
                )
            else:
                self._channel.status_event.wait(min(0.05, remaining))

    def read_for(self, seconds: float) -> bytes:
        """Read all data arriving within timeout. Returns on idle."""
        deadline = time.monotonic() + seconds

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            self._drain_stderr()

            if self._channel.recv_ready():
                data = self._channel.recv(65536)
                if not data:
                    break
                self._buf += data
                if len(self._buf) > self._MAX_BUF:
                    raise SSHError(f"Buffer exceeded {self._MAX_BUF} bytes in read_for")
            elif self._channel.closed or self._channel.exit_status_ready():
                break
            else:
                self._channel.status_event.wait(min(0.05, remaining))

        result = self._buf
        self._buf = b""
        return result

    def _drain_stderr(self) -> None:
        """Drain stderr to prevent deadlock."""
        while self._channel.recv_stderr_ready():
            self._channel.recv_stderr(65536)

    @property
    def alive(self) -> bool:
        """Process still running."""
        return not self._closed and not self._channel.closed and not self._channel.exit_status_ready()

    def close(self) -> None:
        """Close channel (idempotent)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._channel.close()
        except Exception:
            pass

    def __enter__(self) -> RemoteProcess:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "alive" if self.alive else "closed"
        return f"RemoteProcess({state})"


# ---------------------------------------------------------------------------
# SSHClient
# ---------------------------------------------------------------------------


HopSpec = Union[str, SSHHop]


def _normalize_hops(hops: Union[HopSpec, list[HopSpec]]) -> list[SSHHop]:
    """Normalize hop specifications into a list of SSHHop objects."""
    if isinstance(hops, (str, SSHHop)):
        hops = [hops]
    result = []
    for h in hops:
        if isinstance(h, str):
            result.append(SSHHop(hostname=h))
        elif isinstance(h, SSHHop):
            result.append(h)
        else:
            raise TypeError(f"Expected str or SSHHop, got {type(h).__name__}")
    if not result:
        raise ValueError("At least one hop is required")
    return result


class SSHClient:
    """SSH client supporting multi-hop connections, command execution, tunneling, and SFTP.

    Connection is lazy - established on first operation. Uses paramiko Transport
    with GSSAPI (Kerberos), key, or password authentication per hop.

    Args:
        hops: Target host(s). Accepts a hostname string, SSHHop, or list of either.
              Multiple hops create a chain (jump hosts).
        auth: Optional KerberosAuth for GSSAPI hops. If None and any hop uses
              gssapi auth, credentials are validated at init (fail fast).
        connect_timeout: TCP connection timeout in seconds (default 10.0).

    Example:
        with SSHClient("target.fnal.gov") as ssh:
            result = ssh.exec("hostname")
            print(result.stdout)
    """

    def __init__(
        self,
        hops: Union[HopSpec, list[HopSpec]],
        auth: Optional[object] = None,
        connect_timeout: float = 10.0,
    ):
        self._hops = _normalize_hops(hops)
        self._auth = auth
        self._connect_timeout = connect_timeout

        # Validate GSSAPI availability if any hop needs it
        needs_gssapi = any(h.auth_method == "gssapi" for h in self._hops)
        if needs_gssapi:
            if self._auth is not None:
                from pacsys.auth import KerberosAuth

                if not isinstance(self._auth, KerberosAuth):
                    raise ValueError(f"GSSAPI hops require KerberosAuth, got {type(self._auth).__name__}")
            # Validate GSSAPI is available (fail fast)
            try:
                import gssapi  # noqa: F401
            except (ImportError, OSError) as exc:
                raise ImportError(
                    "gssapi library required for GSSAPI SSH auth. Install with: pip install gssapi"
                ) from exc

        # Lazy connection state (protected by lock)
        self._lock = threading.Lock()
        self._transports: list[paramiko.Transport] = []
        self._channels: list[paramiko.Channel] = []  # intermediate channels for multi-hop
        self._connected = False
        self._tunnels: list[Tunnel] = []

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def hops(self) -> list[SSHHop]:
        return list(self._hops)

    def _ensure_connected(self):
        """Lazy connect with double-check locking."""
        if self._connected:
            return
        with self._lock:
            if self._connected:
                return
            self._connect()

    def _connect(self):
        """Build the transport chain through all hops."""
        current_transport = None
        try:
            for i, hop in enumerate(self._hops):
                if i == 0:
                    sock = socket.create_connection(
                        (hop.hostname, hop.port),
                        timeout=self._connect_timeout,
                    )
                    current_transport = paramiko.Transport(sock)
                else:
                    prev_transport = self._transports[-1]
                    chan = prev_transport.open_channel(
                        "direct-tcpip",
                        (hop.hostname, hop.port),
                        ("127.0.0.1", 0),
                    )
                    self._channels.append(chan)
                    current_transport = paramiko.Transport(chan)

                current_transport.start_client()
                self._authenticate(current_transport, hop)
                current_transport.set_keepalive(30)
                self._transports.append(current_transport)
                current_transport = None  # now owned by _transports

            self._connected = True
            logger.info(
                "SSH connected through %d hop(s): %s",
                len(self._hops),
                " -> ".join(h.hostname for h in self._hops),
            )

        except paramiko.AuthenticationException as e:
            if current_transport:
                try:
                    current_transport.close()
                except Exception:
                    pass
            self._cleanup_transports()
            hop = self._hops[min(len(self._transports), len(self._hops) - 1)]
            raise SSHConnectionError(f"Authentication failed: {e}", hop=hop) from e
        except (socket.error, paramiko.SSHException, OSError) as e:
            if current_transport:
                try:
                    current_transport.close()
                except Exception:
                    pass
            self._cleanup_transports()
            hop = self._hops[min(len(self._transports), len(self._hops) - 1)]
            raise SSHConnectionError(f"Connection failed: {e}", hop=hop) from e

    def _authenticate(self, transport: paramiko.Transport, hop: SSHHop):
        """Authenticate a transport using the hop's auth method."""
        username = hop.effective_username

        if hop.auth_method == "gssapi":
            transport.auth_gssapi_with_mic(username, hop.hostname, gss_deleg_creds=True)

        elif hop.auth_method == "key":
            assert hop.key_filename is not None  # validated in __post_init__
            key_path = Path(hop.key_filename).expanduser()
            if not key_path.exists():
                raise SSHConnectionError(f"Key file not found: {key_path}", hop=hop)
            pkey = paramiko.RSAKey.from_private_key_file(str(key_path))
            transport.auth_publickey(username, pkey)

        elif hop.auth_method == "password":
            transport.auth_password(username, hop.password)

    def _cleanup_transports(self):
        """Close all transports and channels in reverse order."""
        for t in reversed(self._transports):
            try:
                t.close()
            except Exception:
                pass
        for c in reversed(self._channels):
            try:
                c.close()
            except Exception:
                pass
        self._transports.clear()
        self._channels.clear()
        self._connected = False

    @property
    def _final_transport(self) -> paramiko.Transport:
        """The transport for the final (target) hop."""
        self._ensure_connected()
        return self._transports[-1]

    def exec(self, command: str, timeout: Optional[float] = None, input: Optional[str] = None) -> CommandResult:
        """Execute a command on the remote host.

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds (None = no timeout)
            input: Optional stdin data to send

        Returns:
            CommandResult with exit_code, stdout, stderr

        Raises:
            SSHTimeoutError: If timeout is exceeded
            SSHConnectionError: If transport is not active
        """
        transport = self._final_transport
        if not transport.is_active():
            raise SSHConnectionError("Transport is no longer active")

        chan = transport.open_session()
        try:
            chan.exec_command(command)

            if input is not None:
                chan.sendall(input.encode())
            chan.shutdown_write()

            deadline = time.monotonic() + timeout if timeout is not None else None
            stdout_chunks: list[bytes] = []
            stderr_chunks: list[bytes] = []

            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise SSHTimeoutError(f"Command timed out after {timeout}s: {command!r}")
                try:
                    # Read stdout
                    if chan.recv_ready():
                        data = chan.recv(65536)
                        if data:
                            stdout_chunks.append(data)
                    # Read stderr
                    if chan.recv_stderr_ready():
                        data = chan.recv_stderr(65536)
                        if data:
                            stderr_chunks.append(data)
                    # Check if done
                    if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                        break
                    # Small sleep to avoid busy loop
                    if not chan.recv_ready() and not chan.recv_stderr_ready() and not chan.exit_status_ready():
                        chan.status_event.wait(0.1)
                except socket.timeout as e:
                    raise SSHTimeoutError(str(e)) from e

            exit_code = chan.recv_exit_status()
            stdout = b"".join(stdout_chunks).decode(errors="replace")
            stderr = b"".join(stderr_chunks).decode(errors="replace")

            return CommandResult(command=command, exit_code=exit_code, stdout=stdout, stderr=stderr)

        finally:
            chan.close()

    def exec_stream(self, command: str, timeout: Optional[float] = None) -> Iterator[str]:
        """Execute a command and yield stdout lines as they arrive.

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds

        Yields:
            Lines of stdout output

        Raises:
            SSHCommandError: If command exits with non-zero status (after all output)
            SSHTimeoutError: If timeout is exceeded
        """
        transport = self._final_transport
        if not transport.is_active():
            raise SSHConnectionError("Transport is no longer active")

        chan = transport.open_session()
        try:
            chan.exec_command(command)
            chan.shutdown_write()

            deadline = time.monotonic() + timeout if timeout is not None else None
            buf = ""
            stderr_chunks: list[bytes] = []

            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise SSHTimeoutError(f"Command timed out after {timeout}s: {command!r}")

                try:
                    if chan.recv_ready():
                        data = chan.recv(65536).decode(errors="replace")
                        buf += data
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            yield line

                    if chan.recv_stderr_ready():
                        stderr_chunks.append(chan.recv_stderr(65536))

                    if chan.exit_status_ready() and not chan.recv_ready():
                        break

                    if not chan.recv_ready() and not chan.exit_status_ready():
                        chan.status_event.wait(0.1)
                except socket.timeout as e:
                    raise SSHTimeoutError(str(e)) from e

            # Yield remaining buffer
            if buf:
                yield buf

            exit_code = chan.recv_exit_status()
            if exit_code != 0:
                stderr = b"".join(stderr_chunks).decode(errors="replace")
                raise SSHCommandError(command, exit_code, stderr)

        finally:
            chan.close()

    def exec_many(self, commands: list[str], timeout: Optional[float] = None) -> list[CommandResult]:
        """Execute multiple commands sequentially.

        Args:
            commands: List of shell commands
            timeout: Per-command timeout in seconds

        Returns:
            List of CommandResult in same order as input
        """
        return [self.exec(cmd, timeout=timeout) for cmd in commands]

    def forward(self, local_port: int, remote_host: str, remote_port: int) -> Tunnel:
        """Create a local port forward through the SSH connection.

        Args:
            local_port: Local port to listen on (0 for OS-assigned)
            remote_host: Remote host to forward to
            remote_port: Remote port to forward to

        Returns:
            Tunnel object (use as context manager or call stop())
        """
        transport = self._final_transport
        tunnel = Tunnel(local_port, remote_host, remote_port, transport)
        self._tunnels.append(tunnel)
        return tunnel

    def open_channel(self, command: str, timeout: float | None = None) -> paramiko.Channel:
        """Open an SSH channel executing a command, with stdin kept open for interactive use.

        Unlike exec(), the channel's stdin is NOT closed after opening, allowing
        interactive input. The caller is responsible for closing the channel.

        Args:
            command: Command to execute on the remote host
            timeout: Channel timeout in seconds (None = no timeout)

        Returns:
            paramiko.Channel with the command running and stdin open
        """
        transport = self._final_transport
        if not transport.is_active():
            raise SSHConnectionError("Transport is no longer active")

        chan = transport.open_session()
        if timeout is not None:
            chan.settimeout(timeout)
        chan.exec_command(command)
        return chan

    def remote_process(self, command: str, *, timeout: float = 30.0) -> RemoteProcess:
        """Open a persistent interactive process over SSH.

        Args:
            command: Command to execute on the remote host
            timeout: Default timeout for read operations in seconds

        Returns:
            RemoteProcess (use as context manager or call close())
        """
        return RemoteProcess(self, command, timeout=timeout)

    def acl(self, command: str | list[str], timeout: float | None = None) -> str:
        """Execute ACL command(s) and return output text.

        Commands are written to a temp script file on the remote host and
        executed as ``acl /tmp/pacsys_acl_XXXX.acl``.

        Args:
            command: ACL command string, or list of commands
                     (semicolons in a string are treated as one line).
            timeout: Command timeout in seconds (default 30.0)

        Returns:
            Command output with ACL prompts stripped

        Raises:
            ACLError: If the ACL process exits with non-zero status
            ValueError: If command list is empty
        """
        from pacsys.acl_session import _strip_acl_output

        effective_timeout = timeout or 30.0

        if isinstance(command, str):
            command = [command]
        if not command:
            raise ValueError("command list must not be empty")
        return self._acl_script(command, effective_timeout, _strip_acl_output)

    def _acl_script(self, commands: list[str], timeout: float, strip_fn) -> str:
        """Write commands to a temp script on the remote host and run via acl."""
        import uuid

        from pacsys.errors import ACLError

        script = "\n".join(commands) + "\n"
        name = f"/tmp/pacsys_acl_{uuid.uuid4().hex[:8]}.acl"

        # Write script file
        write_result = self.exec(f"cat > {name}", input=script, timeout=timeout)
        if not write_result.ok:
            raise ACLError(f"Failed to write ACL script: {write_result.stderr.strip()}")

        try:
            result = self.exec(f"acl {name}", timeout=timeout)
            # ACL exits non-zero on script errors (bad device, etc.) but
            # still produces useful output. Only raise on real failures.
            if not result.ok and (result.stderr.strip() or not result.stdout.strip()):
                msg = result.stderr.strip() or f"exit code {result.exit_code}"
                raise ACLError(f"ACL script failed: {msg}")
            return strip_fn(result.stdout)
        finally:
            self.exec(f"rm -f {name}", timeout=5.0)

    def acl_session(self, *, timeout: float = 30.0) -> ACLSession:
        """Open a persistent ACL interpreter session.

        The session keeps an ``acl`` process alive over an SSH channel,
        avoiding process startup overhead. Each send() is a separate
        script execution - state does NOT persist between calls.

        Args:
            timeout: Default timeout for prompt detection in seconds

        Returns:
            ACLSession (use as context manager or call close())
        """
        from pacsys.acl_session import ACLSession

        return ACLSession(self, timeout=timeout)

    def sftp(self) -> SFTPSession:
        """Open an SFTP session on the remote host.

        Returns:
            SFTPSession (use as context manager or call close())
        """
        transport = self._final_transport
        sftp_client = paramiko.SFTPClient.from_transport(transport)
        if sftp_client is None:
            raise SSHConnectionError("Failed to open SFTP session")
        return SFTPSession(sftp_client)

    def close(self):
        """Close the SSH connection and all tunnels."""
        # Stop all tunnels first
        for tunnel in self._tunnels:
            try:
                tunnel.stop()
            except Exception:
                pass
        self._tunnels.clear()

        self._cleanup_transports()
        logger.info("SSH client closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        hops = " -> ".join(h.hostname for h in self._hops)
        state = "connected" if self._connected else "disconnected"
        return f"SSHClient({hops}, {state})"


__all__ = [
    "SSHError",
    "SSHConnectionError",
    "SSHCommandError",
    "SSHTimeoutError",
    "SSHHop",
    "CommandResult",
    "Tunnel",
    "SFTPSession",
    "RemoteProcess",
    "SSHClient",
]
