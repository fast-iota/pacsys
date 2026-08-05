"""Client for the supervised proxy (supervised_proxy.py).

Demonstrates reads, writes, and streaming through a supervised gRPC proxy.
Start the proxy first, then run this script.

Usage:
    python supervised_proxy.py --port 50051 &
    python supervised_proxy_client.py --port 50051
"""

import argparse

# Must match the token configured on the supervised proxy (--token flag).
# The proxy validates this bearer token on every RPC; the real backend
# auth (Kerberos) is handled server-side.
from supervised_proxy import PROXY_TOKEN

import pacsys
from pacsys import JWTAuth


def parse_args():
    p = argparse.ArgumentParser(description="Client for supervised gRPC proxy")
    p.add_argument("--port", type=int, default=50052, help="proxy port (default: 50052)")
    p.add_argument("--host", default="localhost", help="proxy host (default: localhost)")
    p.add_argument("--token", default=PROXY_TOKEN, help="bearer token (must match proxy)")
    return p.parse_args()


def main():
    args = parse_args()

    auth = JWTAuth(token=args.token)
    with pacsys.grpc(host=args.host, port=args.port, auth=auth) as backend:
        # -- Reads (allowed for all devices) -----------------------------------
        print("=== Reads ===")
        value = backend.read("M:OUTTMP")
        print(f"M:OUTTMP = {value}")

        reading = backend.get("G:AMANDA")
        print(f"G:AMANDA = {reading.value} (ok={reading.ok})")

        # -- Batch read --------------------------------------------------------
        print("\n=== Batch read ===")
        readings = backend.get_many(["M:OUTTMP@I", "G:AMANDA@I", "Z:CUBE_X@I", "Z:CUBE@p,1000", "Z:CUBE_Y@e,02"])
        for r in readings:
            print(f"  {r.drf} = {r.value}")

        # -- Write (only allowed for devices in the proxy's allowlist) ---------
        print("\n=== Write (allowlisted device) ===")
        result = backend.write("Z:ACLTST", 42.0)
        print(f"Z:ACLTST write: {result}")

        # -- Write beyond allowed range ---------
        print("\n=== Write (beyond allowed range, expect denied) ===")
        result = backend.write("Z:ACLTST", 101.0)
        print(f"Z:ACLTST write: {result}")

        # -- Write to non-allowlisted device ----------------
        print("\n=== Write (non-allowlisted device, expect denied) ===")
        result = backend.write("Z:ACLTS2", 99.0)
        print(f"Z:ACLTS2 write: {result}")

        # -- Write to allowed and forbidden devices ----------------
        print("\n=== Write to allowed and forbidden devices ===")
        result = backend.write_many([("Z:ACLTS2", 99.0), ("Z:CUBE_Z", 99.0)])
        print(f"Z:ACLTS2 write: {result}")

        # -- Streaming ---------------------------------------------------------
        print("\n=== Streaming (10 readings) ===")
        with backend.subscribe(["M:OUTTMP@p,1000"]) as sub:
            for count, (reading, _handle) in enumerate(sub.readings(timeout=20)):
                print(f"  [{count}] {reading.drf} = {reading.value}")
                if count >= 9:
                    break

        print("\n=== Streaming mixed (10 readings) ===")
        with backend.subscribe(["M:OUTTMP@p,1000", "M:OUTTMP@p,500", "Z:ACLTST@p,1000"]) as sub:
            for count, (reading, _handle) in enumerate(sub.readings(timeout=20)):
                print(f"  [{count}] {reading.drf} = {reading.value}")
                if count >= 9:
                    break


if __name__ == "__main__":
    main()
