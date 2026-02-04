# ACL

HTTP-based read-only access via the ACL CGI script. ACL is a separate service on the control system (not DPM). No authentication required.

```mermaid
sequenceDiagram
    participant App as Your App
    participant CGI as www-ad.fnal.gov<br>/cgi-bin/acl.pl
    participant ACNET as ACNET

    App->>CGI: HTTPS GET ?command=read&drf=...
    CGI->>ACNET: Device query
    ACNET-->>CGI: Device value
    CGI-->>App: Text response
```

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
    readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
```

## Limitations

- **URL length**: `get_many()` builds a single HTTP GET URL for all devices. Most servers enforce an ~8 KB URL limit, which allows roughly 700 simple devices or ~190 complex DRF strings per call. Exceeding this returns an HTTP error for all devices in the batch. For large batches, use DPM or gRPC instead.
- **No writes or streaming**: Read-only, request/response only.

## When to Use

- Quick one-off reads when there are difficulties installing dependencies
