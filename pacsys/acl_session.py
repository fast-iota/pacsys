"""Persistent ACL interpreter session over SSH.

Keeps an ``acl`` process alive on a remote host, avoiding the startup
overhead of launching a new process per command. Each ``send()`` call
is a separate script execution - state (variables, symbols) does NOT
persist between calls. Use semicolons to combine dependent commands
in a single ``send()``.

Example:
    with ssh.acl_session() as acl:
        acl.send("read M:OUTTMP")
        acl.send("value = M:OUTTMP ; if (value > 100) set M:OUTTMP 100; endif")
"""

from __future__ import annotations

import logging
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

    Opens an ``acl`` process on a remote host via SSH and keeps it alive,
    avoiding process startup overhead. Each ``send()`` is a separate script
    execution - state (variables, symbols) does NOT persist between calls.
    Combine dependent commands with semicolons in a single ``send()``.

    Multiple sessions can coexist on the same SSHClient (paramiko multiplexes
    channels on a single transport).

    Not thread-safe - do not share a single session across threads.
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
        from pacsys.ssh import RemoteProcess

        self._proc = RemoteProcess(ssh, "acl", timeout=timeout)
        self._timeout = timeout
        self._closed = False

        # Wait for initial ACL> prompt to confirm ACL started
        try:
            self._proc.read_until(_ACL_PROMPT, timeout=timeout)
        except Exception as e:
            self._proc.close()
            raise ACLError(f"Failed to start ACL session: {e}") from e
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
            ValueError: If command contains a line break
        """
        if self._closed:
            raise ACLError("ACL session is closed")
        if "\n" in command or "\r" in command:
            raise ValueError("ACLSession commands cannot contain line breaks; use semicolons")

        effective_timeout = timeout if timeout is not None else self._timeout

        try:
            self._proc.send_line(command)
            raw = self._proc.read_until(_ACL_PROMPT, timeout=effective_timeout)
        except ACLError:
            self.close()
            raise
        except Exception as e:
            self.close()
            raise ACLError(str(e)) from e

        # Decode and strip echoed command (first line) from output
        text = raw.decode(errors="replace").strip()
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()
        else:
            # Output is only the echoed command - no actual output
            text = ""
        return text

    def close(self) -> None:
        """Close the ACL session (closes the SSH channel, not the SSHClient)."""
        if self._closed:
            return
        self._closed = True
        self._proc.close()
        logger.debug("ACL session closed")

    def __enter__(self) -> ACLSession:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"ACLSession({state})"
