"""Tests for supervised mode policy system - pure unit tests, no server needed."""

import time

import pytest

from pacsys.supervised._policies import (
    DeviceAccessPolicy,
    PolicyDecision,
    RateLimitPolicy,
    ReadOnlyPolicy,
    RequestContext,
    evaluate_policies,
)


def _ctx(drfs=None, rpc_method="Read", peer="ipv4:127.0.0.1:9999"):
    return RequestContext(drfs=drfs or ["M:OUTTMP"], rpc_method=rpc_method, peer=peer)


# ── PolicyDecision ────────────────────────────────────────────────────────


class TestPolicyDecision:
    def test_allowed(self):
        d = PolicyDecision(allowed=True)
        assert d.allowed
        assert d.reason is None

    def test_denied_requires_reason(self):
        with pytest.raises(ValueError, match="reason"):
            PolicyDecision(allowed=False)

    def test_denied_with_reason(self):
        d = PolicyDecision(allowed=False, reason="nope")
        assert not d.allowed
        assert d.reason == "nope"


# ── ReadOnlyPolicy ────────────────────────────────────────────────────────


class TestReadOnlyPolicy:
    def test_allows_read(self):
        p = ReadOnlyPolicy()
        assert p.check(_ctx(rpc_method="Read")).allowed

    def test_allows_alarms(self):
        p = ReadOnlyPolicy()
        assert p.check(_ctx(rpc_method="Alarms")).allowed

    def test_blocks_set(self):
        p = ReadOnlyPolicy()
        d = p.check(_ctx(rpc_method="Set"))
        assert not d.allowed
        assert "Write" in d.reason


# ── DeviceAccessPolicy ────────────────────────────────────────────────────


class TestDeviceAccessPolicy:
    def test_allow_mode_permits_matching(self):
        p = DeviceAccessPolicy(patterns=["M:*"], mode="allow")
        assert p.check(_ctx(drfs=["M:OUTTMP"])).allowed

    def test_allow_mode_blocks_non_matching(self):
        p = DeviceAccessPolicy(patterns=["M:*"], mode="allow")
        d = p.check(_ctx(drfs=["G:AMANDA"]))
        assert not d.allowed
        assert "G:AMANDA" in d.reason

    def test_deny_mode_blocks_matching(self):
        p = DeviceAccessPolicy(patterns=["Z:*"], mode="deny")
        d = p.check(_ctx(drfs=["Z:ACLTST"]))
        assert not d.allowed
        assert "denied" in d.reason.lower()

    def test_deny_mode_allows_non_matching(self):
        p = DeviceAccessPolicy(patterns=["Z:*"], mode="deny")
        assert p.check(_ctx(drfs=["M:OUTTMP"])).allowed

    def test_case_insensitive(self):
        p = DeviceAccessPolicy(patterns=["m:*"], mode="allow")
        assert p.check(_ctx(drfs=["M:OUTTMP"])).allowed

    def test_multiple_patterns(self):
        p = DeviceAccessPolicy(patterns=["M:*", "G:*"], mode="allow")
        assert p.check(_ctx(drfs=["M:OUTTMP"])).allowed
        assert p.check(_ctx(drfs=["G:AMANDA"])).allowed
        assert not p.check(_ctx(drfs=["Z:ACLTST"])).allowed

    def test_mixed_drfs_blocks_on_first_failure(self):
        p = DeviceAccessPolicy(patterns=["M:*"], mode="allow")
        d = p.check(_ctx(drfs=["M:OUTTMP", "G:AMANDA"]))
        assert not d.allowed

    def test_empty_patterns_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DeviceAccessPolicy(patterns=[], mode="allow")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            DeviceAccessPolicy(patterns=["M:*"], mode="block")

    def test_drf_with_property_and_event(self):
        p = DeviceAccessPolicy(patterns=["M:*"], mode="allow")
        assert p.check(_ctx(drfs=["M:OUTTMP.SETTING@p,1000"])).allowed


# ── RateLimitPolicy ───────────────────────────────────────────────────────


class TestRateLimitPolicy:
    def test_allows_within_limit(self):
        p = RateLimitPolicy(max_requests=3)
        for _ in range(3):
            assert p.check(_ctx()).allowed

    def test_blocks_over_limit(self):
        p = RateLimitPolicy(max_requests=2)
        assert p.check(_ctx()).allowed
        assert p.check(_ctx()).allowed
        d = p.check(_ctx())
        assert not d.allowed
        assert "Rate limit" in d.reason

    def test_per_peer_isolation(self):
        p = RateLimitPolicy(max_requests=1)
        assert p.check(_ctx(peer="peer_a")).allowed
        assert p.check(_ctx(peer="peer_b")).allowed
        assert not p.check(_ctx(peer="peer_a")).allowed

    def test_window_expiry(self):
        p = RateLimitPolicy(max_requests=1, window_seconds=0.1)
        assert p.check(_ctx()).allowed
        assert not p.check(_ctx()).allowed
        time.sleep(0.15)
        assert p.check(_ctx()).allowed

    def test_zero_max_raises(self):
        with pytest.raises(ValueError, match="max_requests"):
            RateLimitPolicy(max_requests=0)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError, match="window_seconds"):
            RateLimitPolicy(max_requests=10, window_seconds=0)


# ── Chain Evaluation ──────────────────────────────────────────────────────


class TestEvaluatePolicies:
    def test_empty_chain_allows(self):
        assert evaluate_policies([], _ctx()).allowed

    def test_single_allow(self):
        assert evaluate_policies([ReadOnlyPolicy()], _ctx(rpc_method="Read")).allowed

    def test_first_denial_short_circuits(self):
        # ReadOnly blocks Set before DeviceAccess is even checked
        policies = [ReadOnlyPolicy(), DeviceAccessPolicy(patterns=["M:*"], mode="allow")]
        d = evaluate_policies(policies, _ctx(rpc_method="Set"))
        assert not d.allowed
        assert "Write" in d.reason

    def test_all_pass(self):
        policies = [ReadOnlyPolicy(), DeviceAccessPolicy(patterns=["M:*"], mode="allow")]
        assert evaluate_policies(policies, _ctx(rpc_method="Read", drfs=["M:OUTTMP"])).allowed
