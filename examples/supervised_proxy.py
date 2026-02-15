"""Supervised proxy with async DPM/HTTP backend.

Demonstrates:
- Async DPM/HTTP backend (direct-await, no executor threads)
- Write-only access restricted to an explicit device list
- Reads allowed for all devices
- Structured audit log (JSON lines + binary protobuf)
- Rotating file-based traffic log
- Rate limiting per client
- Value range enforcement
"""

import argparse
import logging
from logging.handlers import RotatingFileHandler

import pacsys.aio as aio
from pacsys import KerberosAuth
from pacsys.supervised import (
    AuditLog,
    Policy,
    PolicyDecision,
    RateLimitPolicy,
    RequestContext,
    SupervisedServer,
    ValueRangePolicy,
)

# -- Devices allowed for writing -----------------------------------------

WRITABLE_DEVICES = [
    "Z:ACLTST",
]

# -- Custom policy --------------------------------------------------------


class WriteDeviceAllowlistPolicy(Policy):
    """Allow reads for everything, restrict writes to an explicit list."""

    def __init__(self, writable: list[str]):
        self._allowed = {d.upper() for d in writable}

    def check(self, ctx: RequestContext) -> PolicyDecision:
        if ctx.rpc_method != "Set":
            return PolicyDecision(allowed=True)
        from pacsys.drf_utils import get_device_name

        for drf in ctx.drfs:
            name = get_device_name(drf).upper()
            if name not in self._allowed:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Device {name} not in write allowlist",
                )
        return PolicyDecision(allowed=True)


# -- Logging config --------------------------------------------------------


def configure_logging():
    # Console: INFO summary
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))

    # File: full traffic log with rotation (10 MB x 5)
    traffic = RotatingFileHandler("supervised_traffic.log", maxBytes=10_000_000, backupCount=5)
    traffic.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s %(message)s"))

    logger = logging.getLogger("pacsys.supervised")
    logger.addHandler(console)
    logger.addHandler(traffic)
    logger.setLevel(logging.INFO)


# -- Main ------------------------------------------------------------------


PROXY_TOKEN = "my-supervised-proxy-token"


def parse_args():
    p = argparse.ArgumentParser(description="Supervised gRPC proxy with async DPM/HTTP backend")
    p.add_argument("--port", type=int, default=50052, help="gRPC listen port (default: 50052, 0 = OS-assigned)")
    p.add_argument("--host", default="[::]", help="bind address (default: [::])")
    p.add_argument("--token", default=PROXY_TOKEN, help="bearer token for client auth")
    return p.parse_args()


def main():
    args = parse_args()
    configure_logging()

    # Async DPM/HTTP backend with Kerberos auth for writes
    backend = aio.dpm(auth=KerberosAuth(), role="testing")

    # Audit log: JSON lines + binary protobuf, log both requests and responses
    audit = AuditLog(
        "supervised_audit.jsonl",
        proto_path="supervised_audit.binpb",
        log_responses=True,
        flush_interval=50,
    )

    policies = [
        # 1. Reads allowed for everything; writes only for listed devices
        WriteDeviceAllowlistPolicy(WRITABLE_DEVICES),
        # 2. Rate limit: 200 requests/min per client
        RateLimitPolicy(max_requests=200, window_seconds=60),
        # 3. Value range enforcement for writable devices
        ValueRangePolicy(limits={"Z:ACLTST": (0.0, 100.0)}),
    ]

    srv = SupervisedServer(
        backend,
        port=args.port,
        host=args.host,
        policies=policies,
        token=args.token,
        audit_log=audit,
    )
    print(f"Starting supervised proxy on {args.host}:{srv._port}...")
    print(f"Writable devices: {', '.join(WRITABLE_DEVICES)}")
    print("Reads allowed for all devices")
    print("Audit log: supervised_audit.jsonl + supervised_audit.binpb")
    print("Ctrl+C to stop")
    srv.run()  # blocks until SIGINT/SIGTERM


if __name__ == "__main__":
    main()
