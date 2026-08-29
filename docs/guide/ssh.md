# SSH Utility

The `pacsys.ssh` module provides SSH command execution, port tunneling, and SFTP
over multi-hop SSH chains using paramiko and GSSAPI (Kerberos) authentication.

This is a standalone utility -- not a backend subclass -- useful for running
remote commands, transferring files, and setting up tunnels (e.g., for gRPC).

## Quick Start

```python
import pacsys

# Execute a remote command
with pacsys.SSHClient("target.fnal.gov") as client:
    result = client.exec("hostname")
    print(result.stdout)      # "target.fnal.gov\n"
    print(result.ok)          # True
```

## Multi-Hop Connections

Chain through jump hosts. Each hop can use a different auth method.

```python
# Simple multi-hop (all Kerberos)
with pacsys.SSHClient(["jump.fnal.gov", "target.fnal.gov"]) as client:
    result = client.exec("whoami")

# Mixed auth per hop
from pacsys.ssh import SSHHop

with pacsys.SSHClient([
    SSHHop("jump.fnal.gov"),  # Kerberos (default)
    SSHHop("target.fnal.gov", auth_method="key",
           key_filename="~/.ssh/id_ed25519"),
]) as client:
    result = client.exec("ls /data")
```

## Command Execution

### Single Command

```python
result = client.exec("ls -la /tmp")
print(result.exit_code)  # 0
print(result.stdout)     # file listing
print(result.stderr)     # empty string
print(result.ok)         # True
```

### With Stdin Input

```python
result = client.exec("cat > /tmp/config.txt", input="key=value\n")
```

### With Timeout

```python
from pacsys.ssh import SSHTimeoutError

try:
    result = client.exec("long-running-job", timeout=30.0)
except SSHTimeoutError:
    print("Command timed out")
```

### Streaming Output

Yields stdout lines as they arrive:

```python
for line in client.exec_stream("tail -f /var/log/messages"):
    print(line)
    if "ERROR" in line:
        break
```

Non-zero exit raises `SSHCommandError` after all output is consumed:

```python
from pacsys.ssh import SSHCommandError

try:
    for line in client.exec_stream("failing-command"):
        print(line)
except SSHCommandError as e:
    print(f"Exit code: {e.exit_code}, stderr: {e.stderr}")
```

### Multiple Commands

```python
results = client.exec_many(["hostname", "uptime", "df -h"])
for r in results:
    print(f"{r.command}: exit={r.exit_code}")
```

## Port Forwarding

Create local port forwards through the SSH connection:

```python
# Forward local:23456 -> dce08.fnal.gov:50051 through jump host
with pacsys.SSHClient("jump.fnal.gov") as client:
    with client.forward(23456, "dce08.fnal.gov", 50051) as tunnel:
        print(f"Listening on 127.0.0.1:{tunnel.local_port}")

        # Use with gRPC backend
        with pacsys.grpc(port=tunnel.local_port) as backend:
            value = backend.read("M:OUTTMP")
```

Use `local_port=0` for OS-assigned port:

```python
with client.forward(0, "remote-db", 5432) as tunnel:
    print(f"Assigned port: {tunnel.local_port}")
```

## SFTP

```python
with client.sftp() as sftp:
    # Download
    sftp.get("/remote/data.csv", "/local/data.csv")

    # Upload
    sftp.put("/local/config.ini", "/remote/config.ini")

    # Directory operations
    files = sftp.listdir("/data")
    sftp.mkdir("/data/output")
    sftp.remove("/data/old_file.txt")

    # File info
    info = sftp.stat("/data/important.dat")
    print(info.st_size)
```

## Interactive Processes

`RemoteProcess` is a persistent bidirectional byte pipe over SSH. Use it for
interactive programs that read stdin and write stdout (REPLs, calculators,
custom protocols). It does not decode bytes -- that's the caller's job.

```python
with pacsys.SSHClient("host.fnal.gov") as client:
    with client.remote_process("bc -q") as proc:
        proc.send_line("2 + 3")
        result = proc.read_until(b"\n", timeout=5.0)
        print(result)  # b"5"

        proc.send_line("10 * 20")
        result = proc.read_until(b"\n", timeout=5.0)
        print(result)  # b"200"
```

### Reading Data

**`read_until(marker, timeout)`** -- reads bytes until `marker` is found in the
stream. Returns everything *before* the marker; the marker itself is consumed
from the internal buffer. Useful when the remote process emits a known prompt
or delimiter.

```python
with client.remote_process("my_app") as proc:
    proc.send_line("run_query")
    output = proc.read_until(b"PROMPT> ", timeout=10.0)
    print(output.decode())
```

**`read_for(seconds)`** -- reads everything that arrives within the given
wall-clock duration. Returns accumulated bytes. Useful when there's no
predictable marker.

```python
with client.remote_process("echo hello; sleep 0.5; echo world") as proc:
    data = proc.read_for(2.0)
    print(data)  # b"hello\nworld\n"
```

### Sending Data

```python
proc.send_line("command")        # sends "command\n" encoded as bytes
proc.send_bytes(b"\x00\x01\x02")  # sends raw bytes (no newline)
```

### Lifecycle

```python
# As context manager (recommended)
with client.remote_process("cat") as proc:
    assert proc.alive
    proc.send_line("hello")
    proc.read_until(b"\n")
# channel closed automatically

# Manual lifecycle
proc = client.remote_process("cat")
proc.send_line("hello")
proc.close()  # idempotent
proc.close()  # safe to call again
```

### Multiple Processes

Paramiko multiplexes channels on a single transport, so multiple processes
can coexist on one `SSHClient`:

```python
with client.remote_process("bc -q") as calc:
    with client.remote_process("cat") as echo:
        calc.send_line("6 * 7")
        echo.send_line("hello")
        print(calc.read_until(b"\n"))  # b"42"
        print(echo.read_until(b"\n"))  # b"hello"
```

### Error Handling

| Exception | When |
|-----------|------|
| `SSHTimeoutError` | `read_until` timeout expires before marker found |
| `SSHError` | Channel closes or process exits before marker found |

```python
from pacsys.ssh import SSHTimeoutError, SSHError

with client.remote_process("cat") as proc:
    try:
        proc.read_until(b"NEVER", timeout=1.0)
    except SSHTimeoutError:
        print("Marker not found in time")
```

### Notes

- Not thread-safe -- use separate processes per thread
- Does NOT own the `SSHClient` -- closing the process does not close the SSH connection
- Stderr is drained automatically to prevent deadlock (contents are discarded)
- The `timeout` parameter on the constructor is passed to `open_channel()` and
  used as the default for `read_until()` calls

## ACL over SSH

Execute [ACL](https://www-bd.fnal.gov/issues/wiki/ACL) commands on remote
ACNET console hosts via SSH.

### One-Shot Commands

`SSHClient.acl()` runs a fresh `acl` process per call. Accepts a string or a
list of strings (written to a temp script file):

```python
with pacsys.SSHClient(["jump.fnal.gov", "clx01.fnal.gov"]) as ssh:
    # Single command
    output = ssh.acl("read M:OUTTMP")
    print(output)  # "M:OUTTMP       =  72.500 DegF"

    # Semicolons (treated as one line)
    output = ssh.acl("read M:OUTTMP; read G:AMANDA")

    # List of commands (written to temp script file)
    output = ssh.acl(["read M:OUTTMP", "read G:AMANDA"])
```

### Persistent Sessions

`ACLSession` keeps an `acl` process alive via `RemoteProcess`, avoiding the
startup overhead of launching a new process per command. Each `send()` is a
separate script execution - state (variables, symbols) does **not** persist
between calls. Combine dependent commands with semicolons in a single `send()`:

```python
with pacsys.SSHClient(["jump.fnal.gov", "clx01.fnal.gov"]) as ssh:
    with ssh.acl_session() as acl:
        # Each send() is separate - use semicolons for dependencies
        acl.send("read M:OUTTMP")
        acl.send("value = M:OUTTMP ; if (value > 100) set M:OUTTMP 100; endif")
```

Multiple sessions can coexist on one SSH connection:

```python
with ssh.acl_session() as acl1:
    with ssh.acl_session() as acl2:
        r1 = acl1.send("read M:OUTTMP")
        r2 = acl2.send("read G:AMANDA")
```

### ACL Error Handling

Both `acl()` and `ACLSession.send()` raise `ACLError` on failures:

```python
from pacsys.errors import ACLError

with ssh.acl_session() as acl:
    try:
        acl.send("read Z:NOTFOUND")
    except ACLError as e:
        print(f"ACL error: {e}")
```

Persistent sessions reject commands containing line breaks; combine statements with
semicolons instead. If sending or prompt detection fails, the session closes itself to
prevent delayed output from being attributed to a later command. Create a new session
before retrying.

Sending on a closed session also raises `ACLError`:

```python
session = ssh.acl_session()
session.close()
session.send("read M:OUTTMP")  # raises ACLError("ACL session is closed")
```

## Authentication

### Kerberos (Default)

All hops use GSSAPI (Kerberos) by default. Requires a valid ticket (`kinit`).

```python
# Implicit -- validates GSSAPI availability at init
client = pacsys.SSHClient("host.fnal.gov")

# Explicit -- reuse KerberosAuth from other pacsys operations
from pacsys import KerberosAuth
client = pacsys.SSHClient("host.fnal.gov", auth=KerberosAuth())
```

The login name defaults to the principal of `auth` (or of the default credential cache). paramiko can
only present the default credential-cache principal, so `SSHClient(..., auth=KerberosAuth(name=...))`
raises at init unless that principal is the cache default (`kswitch` first).

Ticket delegation (forwarding your TGT to the remote host) is off by default and is not needed for
multi-hop chains. Without it the remote session has no Kerberos ticket, so enable it per hop when a
remote tool needs one — e.g. `ssh.acl("set ...")` or other Kerberized commands on that host:

```python
SSHHop("host.fnal.gov", delegate_credentials=True)
```

### Key-Based

```python
from pacsys.ssh import SSHHop

client = pacsys.SSHClient(SSHHop(
    "host.fnal.gov",
    auth_method="key",
    key_filename="~/.ssh/id_ed25519",
))
```

### Password

```python
client = pacsys.SSHClient(SSHHop(
    "host.fnal.gov",
    auth_method="password",
    password="secret",  # excluded from repr
))
```

## Error Handling

| Exception | When |
|-----------|------|
| `SSHConnectionError` | Connection or auth failure (includes hop info) |
| `SSHCommandError` | Non-zero exit from `exec_stream()` |
| `SSHTimeoutError` | Command or connection timeout |

`exec()` does *not* raise on non-zero exit -- check `result.ok` instead.

```python
from pacsys.ssh import SSHConnectionError

try:
    with pacsys.SSHClient("unreachable.fnal.gov") as client:
        client.exec("ls")
except SSHConnectionError as e:
    print(f"Failed: {e}")
    print(f"Hop: {e.hop}")
```

## Connection Lifecycle

- **Lazy**: No TCP connection until the first operation
- **Keepalive**: 30-second keepalive on all transports
- **No auto-reconnect**: Create a new client if the connection drops
- **Daemon threads**: Tunnel threads are daemon -- safe in Jupyter
- **Context manager**: `close()` stops all tunnels and disconnects in reverse hop order
