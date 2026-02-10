# ACL

HTTP-based read-only access via the ACL CGI script. ACL is a separate service on the control system (not DPM). No authentication required.

```mermaid
sequenceDiagram
    participant App as Your App
    participant CGI as www-bd.fnal.gov<br>/cgi-bin/acl.pl
    participant ACNET as ACNET

    App->>CGI: HTTPS GET ?acl=read+{dev1}\;read+{dev2}
    CGI->>ACNET: Device queries
    ACNET-->>CGI: Device values
    CGI-->>App: DEVICE = VALUE UNITS (one per line)
```

## Configuration

| Parameter | Default | Environment Variable |
|-----------|---------|---------------------|
| `base_url` | www-bd.fnal.gov | `PACSYS_ACL_URL` |

## Characteristics

- **No authentication**: Anyone can read
- **Read-only**: No write or streaming support
- **Simple**: Just HTTP requests. No writes. No streaming.
- **Slower**: HTTP overhead vs binary protocol

## Usage

```python
import pacsys

with pacsys.acl() as backend:
    value = backend.read("M:OUTTMP")
    reading = backend.get("M:OUTTMP")
    readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
```

## Advanced: Raw ACL Commands

The `execute()` method sends arbitrary ACL command strings directly to the CGI endpoint. The argument is placed verbatim after `?acl=` in the URL. Spaces are `+`, semicolons are `\;`.

```python
with pacsys.acl() as backend:
    # Simple read
    text = backend.execute("read+M:OUTTMP")

    # Batch with device_list + read_list (simultaneous)
    text = backend.execute(
        "device_list/create+devs+devices='M:OUTTMP,G:AMANDA'"
        "\\;read_list/no_name/no_units+device_list=devs"
    )

    # Historical data from logger
    text = backend.execute(
        "logger_get/date_format='utc_seconds'"
        "/start=%222024-01-01+00:00:00%22"
        "/end=%222024-01-01+00:01:00%22+M:OUTTMP"
    )
```

See the [ACL command reference](https://www-bd.fnal.gov/issues/wiki/ACLCommands) for stuff not possible through regular acnet.

## URL Encoding

The ACL CGI only decodes `+`/`%20` (space) and `%27` (quote) from the query string. General `%XX` sequences like `%3A` are **not** decoded - DRF characters (`:`, `[]`, `@`, `.`) must be sent raw. The backend handles this automatically for `read`/`get`/`get_many`.

## Limitations

- **URL length**: `get_many()` builds a single HTTP GET URL with semicolon-separated `read` commands. Most servers enforce an ~8 KB URL limit (~200 simple devices per call). For large batches, use DPM or gRPC instead.
- **No writes or streaming**: Read-only, request/response only.
- **Error handling**: ACL aborts the entire script on the first bad device. `get_many()` detects this and falls back to individual reads so valid devices still return data.

## When to Use

- Quick one-off reads when there are difficulties installing dependencies
- Advanced ACL scripting (`execute()`) for logger queries, device lists, etc.
