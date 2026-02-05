"""Persistent ACL interpreter session over SSH.

Keeps an ``acl`` process alive on a remote host, allowing stateful
multi-command workflows where variables and symbols persist between calls.

Example:
    with ssh.acl_session() as acl:
        acl.send("value = M:OUTTMP")
        acl.send("if (value > 100) set M:OUTTMP 100; endif")
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pacsys.errors import ACLError

if TYPE_CHECKING:
    from pacsys.ssh import SSHClient

logger = logging.getLogger(__name__)

# Real ACL prompt is "\nACL> " (newline before, space after).
# Anchoring on \n prevents false matches in command output.
_ACL_PROMPT = b"\nACL> "


def _strip_acl_output(text: str) -> str:
    """Strip ACL prompts and echoed commands from one-shot acl output."""
    lines = text.splitlines()
    out = [line for line in lines if not line.startswith("ACL>")]
    return "\n".join(out).strip()


class ACLSession:
    """Persistent ACL interpreter session over SSH.

    Opens an ``acl`` process on a remote host via SSH and keeps it alive
    for stateful interaction. Multiple sessions can coexist on the same
    SSHClient (paramiko multiplexes channels on a single transport).

    Not thread-safe — do not share a single session across threads.
    Use separate sessions per thread instead.

    Args:
        ssh: Connected SSHClient instance
        timeout: Default timeout for prompt detection in seconds

    Usage:
        with ssh.acl_session() as acl:
            acl.send("read M:OUTTMP")

        # Or explicitly:
        acl = ACLSession(ssh_client)
        acl.send("read M:OUTTMP")
        acl.close()
    """

    def __init__(self, ssh: SSHClient, *, timeout: float = 30.0):
        self._ssh = ssh
        self._timeout = timeout
        self._channel = ssh.open_channel("acl", timeout=timeout)
        self._buf = b""
        self._closed = False

        # Wait for initial ACL> prompt to confirm ACL started
        self._wait_for_prompt(timeout)
        logger.debug("ACL session opened")

    def send(self, command: str, timeout: float | None = None) -> str:
        """Send a command to the ACL interpreter and return the output.

        Args:
            command: ACL command string
            timeout: Override default timeout for this command

        Returns:
            Command output with prompts and echoed command stripped

        Raises:
            ACLError: If the session is closed, the process exits, or prompt times out
        """
        if self._closed:
            raise ACLError("ACL session is closed")

        effective_timeout = timeout if timeout is not None else self._timeout

        self._channel.sendall(f"{command}\n".encode())
        raw = self._wait_for_prompt(effective_timeout)

        # Decode and strip echoed command (first line) from output
        text = raw.decode(errors="replace").strip()
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()
        else:
            # Output is only the echoed command — no actual output
            text = ""
        return text

    def close(self) -> None:
        """Close the ACL session (closes the SSH channel, not the SSHClient)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._channel.close()
        except Exception:
            pass
        logger.debug("ACL session closed")

    def _wait_for_prompt(self, timeout: float) -> bytes:
        """Read from channel until ``\\nACL> `` prompt appears.

        Buffers raw bytes to avoid corrupting multi-byte characters split
        across recv() boundaries. Also drains stderr to prevent deadlock.

        Returns:
            Raw bytes received before the prompt marker

        Raises:
            ACLError: If timeout expires or channel closes before prompt
        """
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ACLError(f"Timed out waiting for ACL prompt after {timeout}s (buffer: {self._buf[-200:]!r})")

            # Check if channel is still open
            if self._channel.closed or self._channel.exit_status_ready():
                raise ACLError(f"ACL process exited unexpectedly (buffer: {self._buf[-200:]!r})")

            got_data = False

            # Read stdout
            if self._channel.recv_ready():
                data = self._channel.recv(65536)
                if not data:
                    raise ACLError("ACL channel closed (received empty data)")
                self._buf += data
                got_data = True

            # Drain stderr to prevent deadlock
            if self._channel.recv_stderr_ready():
                self._channel.recv_stderr(65536)
                got_data = True

            if not got_data:
                self._channel.status_event.wait(min(0.05, remaining))

            # Check for prompt in bytes buffer
            idx = self._buf.find(_ACL_PROMPT)
            if idx >= 0:
                output = self._buf[:idx]
                self._buf = self._buf[idx + len(_ACL_PROMPT) :]
                return output

    def __enter__(self) -> ACLSession:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"ACLSession({state})"
