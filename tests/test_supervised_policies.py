"""Tests for supervised mode policy system - pure unit tests, no server needed."""

import time

import pytest

from pacsys.supervised._policies import (
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


def _ctx(
    drfs=None,
    rpc_method="Read",
    peer="ipv4:127.0.0.1:9999",
    values=None,
    raw_request=None,
    allowed=None,
):
    return RequestContext(
        drfs=drfs or ["M:OUTTMP"],
        rpc_method=rpc_method,
        peer=peer,
        metadata={},
        values=values or [],
        raw_request=raw_request,
        allowed=allowed if allowed is not None else frozenset(),
    )


# ── RequestContext.allowed ────────────────────────────────────────────────


class TestRequestContextAllowed:
    def test_default_allowed_is_empty(self):
        ctx = _ctx()
        assert ctx.allowed == frozenset()

    def test_allowed_preserved(self):
        ctx = _ctx(allowed=frozenset({0, 2}))
        assert ctx.allowed == frozenset({0, 2})


# ── PolicyDecision ────────────────────────────────────────────────────────


class TestPolicyDecision:
    def test_allowed(self):
        d = PolicyDecision(allowed=True)
        assert d.allowed
        assert d.reason is None
        assert d.ctx is None

    def test_denied_requires_reason(self):
        with pytest.raises(ValueError, match="reason"):
            PolicyDecision(allowed=False)

    def test_denied_with_reason(self):
        d = PolicyDecision(allowed=False, reason="nope")
        assert not d.allowed
        assert d.reason == "nope"

    def test_allowed_with_modified_ctx(self):
        ctx = _ctx(drfs=["M:OUTTMP"])
        d = PolicyDecision(allowed=True, ctx=ctx)
        assert d.allowed
        assert d.ctx is ctx


# ── Policy.allows_writes ──────────────────────────────────────────────────


class TestPolicyAllowsWrites:
    def test_base_default_false(self):
        """All built-in policies that don't gate writes return False."""
        assert ReadOnlyPolicy().allows_writes is False
        assert RateLimitPolicy(max_requests=10).allows_writes is False
        assert ValueRangePolicy(limits={"M:*": (0, 100)}).allows_writes is False
        assert SlewRatePolicy(limits={"M:*": SlewLimit(max_step=10)}).allows_writes is False


# ── ReadOnlyPolicy ────────────────────────────────────────────────────────


class TestReadOnlyPolicy:
    def test_allows_read(self):
        p = ReadOnlyPolicy()
        assert p.check(_ctx(rpc_method="Read")).allowed

    def test_blocks_set(self):
        p = ReadOnlyPolicy()
        d = p.check(_ctx(rpc_method="Set"))
        assert not d.allowed
        assert "Write" in d.reason


# ── DeviceAccessPolicy ────────────────────────────────────────────────────


class TestDeviceAccessPolicy:
    def test_allow_mode_permits_matching(self):
        p = DeviceAccessPolicy(patterns=["M:*"], mode="allow")
        d = p.check(_ctx(drfs=["M:OUTTMP"]))
        assert d.allowed
        assert d.ctx.allowed == frozenset({0})

    def test_allow_mode_passes_through_non_matching(self):
        """Allow mode no longer denies non-matching — just doesn't approve them."""
        p = DeviceAccessPolicy(patterns=["M:*"], mode="allow")
        d = p.check(_ctx(drfs=["G:AMANDA"]))
        assert d.allowed
        assert d.ctx is None or 0 not in d.ctx.allowed

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
        # Z: not matched — passes through but not approved
        d = p.check(_ctx(drfs=["Z:ACLTST"]))
        assert d.allowed
        assert d.ctx is None or 0 not in d.ctx.allowed

    def test_mixed_drfs_approves_matching_only(self):
        """Allow mode approves matching slots, leaves non-matching unapproved."""
        p = DeviceAccessPolicy(patterns=["M:*"], mode="allow")
        d = p.check(_ctx(drfs=["M:OUTTMP", "G:AMANDA"]))
        assert d.allowed
        assert d.ctx.allowed == frozenset({0})

    def test_empty_patterns_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DeviceAccessPolicy(patterns=[], mode="allow")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            DeviceAccessPolicy(patterns=["M:*"], mode="block")

    def test_drf_with_property_and_event(self):
        p = DeviceAccessPolicy(patterns=["M:*"], mode="allow")
        assert p.check(_ctx(drfs=["M:OUTTMP.SETTING@p,1000"])).allowed

    def test_regex_allow(self):
        p = DeviceAccessPolicy(patterns=[r"M:OUT.*"], mode="allow", syntax="regex")
        d = p.check(_ctx(drfs=["M:OUTTMP"]))
        assert d.allowed
        assert d.ctx.allowed == frozenset({0})
        # Non-matching passes through unapproved
        d2 = p.check(_ctx(drfs=["G:AMANDA"]))
        assert d2.allowed

    def test_regex_deny(self):
        p = DeviceAccessPolicy(patterns=[r"Z:ACL.+"], mode="deny", syntax="regex")
        assert not p.check(_ctx(drfs=["Z:ACLTST"])).allowed
        assert p.check(_ctx(drfs=["M:OUTTMP"])).allowed

    def test_regex_case_insensitive(self):
        p = DeviceAccessPolicy(patterns=[r"m:outtmp"], mode="allow", syntax="regex")
        assert p.check(_ctx(drfs=["M:OUTTMP"])).allowed

    def test_regex_fullmatch(self):
        p = DeviceAccessPolicy(patterns=[r"M:OUT"], mode="allow", syntax="regex")
        d = p.check(_ctx(drfs=["M:OUTTMP"]))
        assert d.allowed  # passes through unapproved
        assert d.ctx is None or 0 not in d.ctx.allowed

    def test_invalid_syntax_raises(self):
        with pytest.raises(ValueError, match="syntax"):
            DeviceAccessPolicy(patterns=["M:*"], syntax="wildcard")

    # ── action parameter ──

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            DeviceAccessPolicy(patterns=["M:*"], action="write")

    def test_action_set_skips_reads(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow")
        d = p.check(_ctx(rpc_method="Read", drfs=["Z:NOPE"]))
        assert d.allowed  # not filtered — action doesn't match

    def test_action_read_skips_sets(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="read", mode="deny")
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"]))
        assert d.allowed  # not filtered — action doesn't match

    def test_action_all_applies_to_reads(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="all", mode="deny")
        d = p.check(_ctx(rpc_method="Read", drfs=["M:OUTTMP"]))
        assert not d.allowed

    def test_action_all_applies_to_sets(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="all", mode="deny")
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"]))
        assert not d.allowed

    # ── per-slot write approval ──

    def test_allow_set_approves_matching_slots(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow")
        ctx = _ctx(rpc_method="Set", drfs=["M:OUTTMP", "G:AMANDA"])
        d = p.check(ctx)
        assert d.allowed
        assert d.ctx is not None
        assert d.ctx.allowed == frozenset({0})

    def test_allow_set_approves_all_matching(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow")
        ctx = _ctx(rpc_method="Set", drfs=["M:OUTTMP", "M:OTHER"])
        d = p.check(ctx)
        assert d.ctx.allowed == frozenset({0, 1})

    def test_allow_set_accumulates_across_policies(self):
        """Two composable allow-mode policies for different device groups."""
        p1 = DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow")
        p2 = DeviceAccessPolicy(patterns=["G:*"], action="set", mode="allow")
        ctx = _ctx(rpc_method="Set", drfs=["M:OUTTMP", "G:AMANDA"])
        d1 = p1.check(ctx)
        assert d1.ctx.allowed == frozenset({0})
        d2 = p2.check(d1.ctx)
        assert d2.ctx.allowed == frozenset({0, 1})

    def test_allow_set_no_matches_leaves_allowed_empty(self):
        p = DeviceAccessPolicy(patterns=["Z:*"], action="set", mode="allow")
        ctx = _ctx(rpc_method="Set", drfs=["M:OUTTMP"])
        d = p.check(ctx)
        assert d.allowed
        assert d.ctx is None or d.ctx.allowed == frozenset()

    def test_deny_set_still_short_circuits(self):
        p = DeviceAccessPolicy(patterns=["Z:*"], action="set", mode="deny")
        ctx = _ctx(rpc_method="Set", drfs=["Z:SECRET"])
        d = p.check(ctx)
        assert not d.allowed
        assert "denied" in d.reason.lower()

    # ── allows_writes property ──

    def test_allows_writes_allow_set(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="set", mode="allow")
        assert p.allows_writes is True

    def test_allows_writes_allow_all(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="all", mode="allow")
        assert p.allows_writes is True

    def test_allows_writes_deny_mode(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="set", mode="deny")
        assert p.allows_writes is False

    def test_allows_writes_allow_read(self):
        p = DeviceAccessPolicy(patterns=["M:*"], action="read", mode="allow")
        assert p.allows_writes is False


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


# ── ValueRangePolicy ─────────────────────────────────────────────────────


class TestValueRangePolicy:
    def test_in_range_allowed(self):
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 50.0)]))
        assert d.allowed

    def test_out_of_range_denied(self):
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 150.0)]))
        assert not d.allowed
        assert "outside range" in d.reason

    def test_below_range_denied(self):
        p = ValueRangePolicy(limits={"M:*": (10.0, 100.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 5.0)]))
        assert not d.allowed

    def test_non_numeric_skipped(self):
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", "hello")]))
        assert d.allowed

    def test_reads_pass_through(self):
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        assert p.check(_ctx(rpc_method="Read")).allowed

    def test_glob_matching(self):
        p = ValueRangePolicy(limits={"G:*": (0.0, 10.0)})
        # M: device not matched by pattern -> allowed regardless of value
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 999.0)]))
        assert d.allowed
        # G: device matched -> enforced
        d = p.check(_ctx(rpc_method="Set", drfs=["G:AMANDA"], values=[("G:AMANDA", 999.0)]))
        assert not d.allowed

    def test_boundary_values(self):
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        assert p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 0.0)])).allowed
        assert p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 100.0)])).allowed

    def test_empty_limits_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ValueRangePolicy(limits={})


# ── SlewLimit ─────────────────────────────────────────────────────────────


class TestSlewLimit:
    def test_max_step_only(self):
        s = SlewLimit(max_step=10.0)
        assert s.max_step == 10.0
        assert s.max_rate is None

    def test_max_rate_only(self):
        s = SlewLimit(max_rate=5.0)
        assert s.max_rate == 5.0
        assert s.max_step is None

    def test_both(self):
        s = SlewLimit(max_step=10.0, max_rate=5.0)
        assert s.max_step == 10.0
        assert s.max_rate == 5.0

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SlewLimit()


# ── SlewRatePolicy ───────────────────────────────────────────────────────


class TestSlewRatePolicy:
    def test_first_write_always_allowed(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_rate=10.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 50.0)]))
        assert d.allowed

    def test_within_rate_allowed(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_rate=1000.0)})
        p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 50.0)]))
        time.sleep(0.05)
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 51.0)]))
        assert d.allowed

    def test_exceeds_rate_denied(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_rate=1.0)})
        p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 0.0)]))
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 100.0)]))
        assert not d.allowed
        assert "Slew rate" in d.reason

    def test_max_step_allowed(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_step=10.0)})
        p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 50.0)]))
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 55.0)]))
        assert d.allowed

    def test_max_step_denied(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_step=5.0)})
        p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 50.0)]))
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 70.0)]))
        assert not d.allowed
        assert "Step" in d.reason

    def test_both_limits_rate_denied(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_step=100.0, max_rate=1.0)})
        p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 0.0)]))
        # Step is fine (50 < 100), but rate is not (50/~0s >> 1/s)
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 50.0)]))
        assert not d.allowed
        assert "Slew rate" in d.reason

    def test_both_limits_step_denied(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_step=5.0, max_rate=1000.0)})
        p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 0.0)]))
        time.sleep(0.05)
        # Rate is fine (20/0.05 = 400 < 1000), but step is not (20 > 5)
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 20.0)]))
        assert not d.allowed
        assert "Step" in d.reason

    def test_window_decay(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_rate=10.0)})
        p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 0.0)]))
        time.sleep(0.6)
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 5.0)]))
        assert d.allowed

    def test_denied_does_not_update_history(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_rate=1.0)})
        p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 0.0)]))
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 100.0)]))
        assert not d.allowed
        time.sleep(0.2)
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 0.1)]))
        assert d.allowed

    def test_reads_pass_through(self):
        p = SlewRatePolicy(limits={"M:*": SlewLimit(max_rate=1.0)})
        assert p.check(_ctx(rpc_method="Read")).allowed

    def test_empty_limits_raises(self):
        with pytest.raises(ValueError, match="empty"):
            SlewRatePolicy(limits={})


# ── Chain Evaluation ──────────────────────────────────────────────────────


class TestEvaluatePolicies:
    def test_empty_chain_allows(self):
        d = evaluate_policies([], _ctx())
        assert d.allowed
        assert d.ctx is not None

    def test_single_allow(self):
        d = evaluate_policies([ReadOnlyPolicy()], _ctx(rpc_method="Read"))
        assert d.allowed
        assert d.ctx is not None

    def test_first_denial_short_circuits(self):
        policies = [ReadOnlyPolicy(), DeviceAccessPolicy(patterns=["M:*"], mode="allow")]
        d = evaluate_policies(policies, _ctx(rpc_method="Set"))
        assert not d.allowed
        assert "Write" in d.reason

    def test_all_pass(self):
        policies = [ReadOnlyPolicy(), DeviceAccessPolicy(patterns=["M:*"], mode="allow")]
        d = evaluate_policies(policies, _ctx(rpc_method="Read", drfs=["M:OUTTMP"]))
        assert d.allowed

    def test_modification_chaining(self):
        """Policy A modifies ctx, Policy B sees the modified ctx."""

        class ClampPolicy(Policy):
            def check(self, ctx: RequestContext) -> PolicyDecision:
                if ctx.rpc_method != "Set":
                    return PolicyDecision(allowed=True)
                new_values = [
                    (drf, min(val, 100.0) if isinstance(val, (int, float)) else val) for drf, val in ctx.values
                ]
                from dataclasses import replace

                return PolicyDecision(allowed=True, ctx=replace(ctx, values=new_values))

        class AssertMaxPolicy(Policy):
            """Denies if any value > 100 (should never fire after ClampPolicy)."""

            def check(self, ctx: RequestContext) -> PolicyDecision:
                for _, val in ctx.values:
                    if isinstance(val, (int, float)) and val > 100:
                        return PolicyDecision(allowed=False, reason="too high")
                return PolicyDecision(allowed=True)

        ctx = _ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 200.0)])
        d = evaluate_policies([ClampPolicy(), AssertMaxPolicy()], ctx)
        assert d.allowed
        # Final ctx should have the clamped value
        assert d.ctx.values[0] == ("M:OUTTMP", 100.0)
