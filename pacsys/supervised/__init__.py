"""Supervised mode: gRPC proxy server with logging and policy enforcement.

Wraps any Backend instance and exposes it as a gRPC DAQ service,
forwarding requests while enforcing access policies and logging all traffic.

Example:
    from pacsys.testing import FakeBackend
    from pacsys.supervised import SupervisedServer, ReadOnlyPolicy

    fb = FakeBackend()
    fb.set_reading("M:OUTTMP", 72.5)

    with SupervisedServer(fb, port=50099, policies=[ReadOnlyPolicy()]) as srv:
        import pacsys
        with pacsys.grpc(host="localhost", port=50099) as client:
            print(client.read("M:OUTTMP"))
"""

from ._policies import (
    DeviceAccessPolicy,
    Policy,
    PolicyDecision,
    RateLimitPolicy,
    ReadOnlyPolicy,
    RequestContext,
    SlewLimit,
    SlewRatePolicy,
    ValueRangePolicy,
    evaluate_policies,
)
from ._server import SupervisedServer

__all__ = [
    "SupervisedServer",
    "Policy",
    "PolicyDecision",
    "RequestContext",
    "ReadOnlyPolicy",
    "DeviceAccessPolicy",
    "RateLimitPolicy",
    "ValueRangePolicy",
    "SlewLimit",
    "SlewRatePolicy",
    "evaluate_policies",
]
