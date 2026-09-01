"""Tests for supervised mode policy system - pure unit tests, no server needed."""

from types import SimpleNamespace

import numpy as np
import pytest

from pacsys.supervised import _policies as policies
from pacsys.supervised._policies import (
    DeviceAccessPolicy,
    Policy,
    PolicyDecision,
    RateLimitPolicy,
    ReadOnlyPolicy,
    RequestContext,
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


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    fake_time = SimpleNamespace(**vars(policies.time))
    fake_time.monotonic = lambda: now[0]
    monkeypatch.setattr(policies, "time", fake_time)

    def advance(seconds):
        now[0] += seconds

    return advance


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
        """Allow mode passes through nonmatching requests without approving them."""
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

    def test_acnet_glob_case_insensitive(self):
        p = DeviceAccessPolicy(patterns=["m:*"], mode="allow")
        assert p.check(_ctx(drfs=["M:OUTTMP"])).ctx.allowed == frozenset({0})

    @pytest.mark.parametrize("syntax", ["glob", "regex"])
    @pytest.mark.parametrize("mode", ["allow", "deny"])
    def test_epics_matching_is_case_sensitive(self, syntax, mode):
        p = DeviceAccessPolicy(patterns=["SR:diag:rawbuf"], mode=mode, syntax=syntax)
        exact = p.check(_ctx(drfs=["SR:diag:rawbuf"]))
        different = p.check(_ctx(drfs=["SR:DIAG:RAWBUF"]))
        if mode == "allow":
            assert exact.ctx.allowed == frozenset({0})
            assert different.ctx is None
        else:
            assert not exact.allowed
            assert different.allowed

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

    def test_acnet_regex_case_insensitive(self):
        p = DeviceAccessPolicy(patterns=[r"m:outtmp"], mode="allow", syntax="regex")
        assert p.check(_ctx(drfs=["M:OUTTMP"])).ctx.allowed == frozenset({0})

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

    def test_allow_read_combination_rejected(self):
        # Read allowlists cannot work (reads are allowed by default) — must fail loudly
        with pytest.raises(ValueError, match="no effect"):
            DeviceAccessPolicy(patterns=["M:*"], action="read", mode="allow")


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

    @pytest.mark.parametrize(
        ("first", "second"), [("ipv4:1.2.3.4:1000", "ipv4:1.2.3.4:1001"), ("ipv6:[::1]:5", "ipv6:[::1]:6")]
    )
    def test_reconnect_shares_bucket(self, first, second):
        """A new TCP connection (fresh ephemeral port) must not get a fresh bucket."""
        p = RateLimitPolicy(max_requests=1)
        assert p.check(_ctx(peer=first)).allowed
        assert not p.check(_ctx(peer=second)).allowed

    def test_pruning_respects_long_window(self, clock):
        p = RateLimitPolicy(max_requests=1, window_seconds=7200)
        for i in range(101):
            assert p.check(_ctx(peer=f"peer_{i}")).allowed
        clock(3601.0)
        assert p.check(_ctx(peer="current")).allowed
        assert not p.check(_ctx(peer="peer_0")).allowed  # still inside its window

    def test_stale_peers_pruned(self, clock):
        p = RateLimitPolicy(max_requests=1)
        for i in range(101):
            assert p.check(_ctx(peer=f"peer_{i}")).allowed

        clock(3601.0)
        assert p.check(_ctx(peer="current")).allowed
        assert set(p._timestamps) == {"current"}

    def test_window_expiry(self, clock):
        p = RateLimitPolicy(max_requests=1, window_seconds=0.1)
        assert p.check(_ctx()).allowed
        assert not p.check(_ctx()).allowed
        clock(0.15)
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

    @pytest.mark.parametrize("drf", ["M:OUTTMP.SETTING.RAW", "M:OUTTMP.SETTING.PRIMARY"])
    def test_unscaled_field_write_denied(self, drf):
        """50 raw counts may be a huge engineering value: not comparable to the configured bounds."""
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=[drf], values=[(drf, 50.0)]))
        assert not d.allowed and "not comparable" in d.reason
        exempt = ValueRangePolicy(limits={"M:*": (0.0, 100.0)}, allow_raw=["m:out*"])
        assert exempt.check(_ctx(rpc_method="Set", drfs=[drf], values=[(drf, 30000.0)])).allowed  # unchecked

    def test_control_commands_pass_through(self):
        """CONTROL payloads are command ordinals, not values: limits never apply (gate them by device access)."""
        p = ValueRangePolicy(limits={"M:*": (0.0, 1.0)})
        assert p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP.CONTROL"], values=[("M:OUTTMP.CONTROL", 2.0)])).allowed

    def test_below_range_denied(self):
        p = ValueRangePolicy(limits={"M:*": (10.0, 100.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 5.0)]))
        assert not d.allowed

    def test_non_numeric_denied_for_limited_device(self):
        """Fail closed: values the policy can't interpret must not bypass a configured limit."""
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", "hello")]))
        assert not d.allowed
        assert "Non-numeric" in d.reason

    def test_non_numeric_allowed_for_unlimited_device(self):
        p = ValueRangePolicy(limits={"G:*": (0.0, 100.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", "hello")]))
        assert d.allowed

    def test_control_enum_denied_as_setpoint(self):
        """BasicControl is an IntEnum — its ordinal must never pass a setpoint range check; on the CONTROL
        property it is a command, which value limits do not gate."""
        from pacsys.types import BasicControl

        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})  # ON ordinal (1) is inside this range
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", BasicControl.ON)]))
        assert not d.allowed
        assert "Non-numeric" in d.reason
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP.CONTROL"], values=[("M:OUTTMP.CONTROL", BasicControl.ON)]))
        assert d.allowed

    @pytest.mark.parametrize(
        "value",
        [
            [9999.0],
            "9999",
            np.array([9999.0]),
            np.int32(9999),
            [1.0, 9999.0],  # one bad element in an otherwise-good array
            float("nan"),
        ],
    )
    def test_bypass_encodings_denied(self, value):
        """Regression for policy bypass: every re-encoding of an out-of-range value is denied."""
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", value)]))
        assert not d.allowed

    @pytest.mark.parametrize("value", [b"\x01\x02", bytearray(b"\x01\x02")])
    def test_raw_write_requires_explicit_exemption(self, value):
        ctx = _ctx(
            rpc_method="Set",
            drfs=["B:HS23T.SETTING{0:64}.RAW"],
            values=[("B:HS23T.SETTING{0:64}.RAW", value)],
        )
        assert not ValueRangePolicy(limits={"B:*": (-10.0, 10.0)}).check(ctx).allowed
        assert ValueRangePolicy(limits={"B:*": (-10.0, 10.0)}, allow_raw=["B:HS23T"]).check(ctx).allowed

    def test_in_range_array_allowed(self):
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        for value in ([1.0, 50.0, 100.0], np.array([1.0, 2.0]), np.int32(50)):
            d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", value)]))
            assert d.allowed, f"{value!r} should be allowed"

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

    def test_acnet_limit_matching_is_case_insensitive(self):
        p = ValueRangePolicy(limits={"m:outtmp": (0.0, 1.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 2.0)]))
        assert not d.allowed

    def test_epics_limit_matching_is_case_sensitive(self):
        p = ValueRangePolicy(limits={"SR:diag:rawbuf": (0.0, 1.0)})
        exact = p.check(_ctx(rpc_method="Set", drfs=["SR:diag:rawbuf"], values=[("SR:diag:rawbuf", 2.0)]))
        different = p.check(_ctx(rpc_method="Set", drfs=["SR:DIAG:RAWBUF"], values=[("SR:DIAG:RAWBUF", 2.0)]))
        assert not exact.allowed
        assert different.allowed

    def test_epics_allow_raw_matching_is_case_sensitive(self):
        p = ValueRangePolicy(limits={"SR:*": (0.0, 1.0)}, allow_raw=["SR:diag:rawbuf"])
        exact = p.check(_ctx(rpc_method="Set", drfs=["SR:diag:rawbuf"], values=[("SR:diag:rawbuf", b"raw")]))
        different = p.check(_ctx(rpc_method="Set", drfs=["SR:DIAG:RAWBUF"], values=[("SR:DIAG:RAWBUF", b"raw")]))
        assert exact.allowed
        assert not different.allowed

    def test_all_matching_bounds_are_intersected(self):
        p = ValueRangePolicy(limits={"M:*": (0.0, 1000.0), "M:OUTTMP": (0.0, 1.0)})
        denied = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 500.0)]))
        allowed = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 0.5)]))
        assert not denied.allowed
        assert allowed.allowed

    def test_empty_bound_intersection_denied(self):
        p = ValueRangePolicy(limits={"M:*": (0.0, 10.0), "M:OUTTMP": (20.0, 30.0)})
        d = p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 5.0)]))
        assert not d.allowed
        assert "M:OUTTMP" in d.reason
        assert "Contradictory" in d.reason

    def test_boundary_values(self):
        p = ValueRangePolicy(limits={"M:*": (0.0, 100.0)})
        assert p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 0.0)])).allowed
        assert p.check(_ctx(rpc_method="Set", drfs=["M:OUTTMP"], values=[("M:OUTTMP", 100.0)])).allowed

    def test_empty_limits_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ValueRangePolicy(limits={})


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

    def test_target_rewrite_rejected(self):
        class RetargetPolicy(Policy):
            def check(self, ctx: RequestContext) -> PolicyDecision:
                from dataclasses import replace

                return PolicyDecision(allowed=True, ctx=replace(ctx, drfs=["G:AMANDA"]))

        with pytest.raises(ValueError, match="RetargetPolicy modified ctx.drfs"):
            evaluate_policies([RetargetPolicy()], _ctx(drfs=["M:OUTTMP"]))

    def test_value_drf_reorder_rejected(self):
        class ReorderValuesPolicy(Policy):
            def check(self, ctx: RequestContext) -> PolicyDecision:
                from dataclasses import replace

                return PolicyDecision(allowed=True, ctx=replace(ctx, values=list(reversed(ctx.values))))

        ctx = _ctx(
            rpc_method="Set",
            drfs=["M:OUTTMP", "G:AMANDA"],
            values=[("M:OUTTMP", 1.0), ("G:AMANDA", 2.0)],
        )
        with pytest.raises(ValueError, match="ReorderValuesPolicy modified or reordered value DRFs"):
            evaluate_policies([ReorderValuesPolicy()], ctx)

    def test_in_place_target_mutation_rejected(self):
        class MutatingPolicy(Policy):
            def check(self, ctx: RequestContext) -> PolicyDecision:
                ctx.drfs[0] = "G:AMANDA"
                return PolicyDecision(allowed=True)

        with pytest.raises(ValueError, match="MutatingPolicy modified ctx.drfs"):
            evaluate_policies([MutatingPolicy()], _ctx(drfs=["M:OUTTMP"]))

    def test_out_of_range_allowed_slot_rejected(self):
        class BadAllowedPolicy(Policy):
            def check(self, ctx: RequestContext) -> PolicyDecision:
                from dataclasses import replace

                return PolicyDecision(allowed=True, ctx=replace(ctx, allowed=frozenset({1})))

        with pytest.raises(ValueError, match="BadAllowedPolicy produced out-of-range allowed slots"):
            evaluate_policies([BadAllowedPolicy()], _ctx(drfs=["M:OUTTMP"]))
