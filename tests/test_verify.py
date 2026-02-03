"""Tests for Verify dataclass, context manager, and resolution logic."""

import threading

import numpy as np
import pytest

from pacsys.verify import (
    Verify,
    verify_context,
    get_active_verify,
    resolve_verify,
    values_match,
)


class TestVerifyDefaults:
    def test_default_values(self):
        v = Verify()
        assert v.check_first is False
        assert v.tolerance == 0.0
        assert v.initial_delay == 0.3
        assert v.retry_delay == 0.5
        assert v.max_attempts == 3
        assert v.readback is None
        assert v.always is False

    def test_custom_values(self):
        v = Verify(check_first=True, tolerance=0.5, max_attempts=5, always=True)
        assert v.check_first is True
        assert v.tolerance == 0.5
        assert v.max_attempts == 5
        assert v.always is True

    def test_frozen(self):
        v = Verify()
        with pytest.raises(AttributeError):
            v.tolerance = 1.0

    def test_defaults_classmethod(self):
        v = Verify.defaults(tolerance=0.1, max_attempts=10)
        assert v.tolerance == 0.1
        assert v.max_attempts == 10
        assert v.check_first is False  # still default


class TestVerifyContextManager:
    def test_context_manager_push_pop(self):
        assert get_active_verify() is None
        v = Verify(tolerance=1.0)
        with v:
            assert get_active_verify() is v
        assert get_active_verify() is None

    def test_verify_context_function(self):
        assert get_active_verify() is None
        v = Verify(tolerance=2.0)
        with verify_context(v) as ctx:
            assert ctx is v
            assert get_active_verify() is v
        assert get_active_verify() is None

    def test_nested_contexts(self):
        v1 = Verify(tolerance=1.0)
        v2 = Verify(tolerance=2.0)
        with v1:
            assert get_active_verify() is v1
            with v2:
                assert get_active_verify() is v2
            assert get_active_verify() is v1
        assert get_active_verify() is None

    def test_context_pops_on_exception(self):
        v = Verify()
        try:
            with v:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert get_active_verify() is None


class TestResolveVerify:
    def test_false_returns_none(self):
        assert resolve_verify(False) is None

    def test_false_ignores_context(self):
        with Verify(always=True):
            assert resolve_verify(False) is None

    def test_true_no_context_returns_defaults(self):
        v = resolve_verify(True)
        assert isinstance(v, Verify)
        assert v == Verify()

    def test_true_with_context_returns_context(self):
        ctx = Verify(tolerance=5.0)
        with ctx:
            assert resolve_verify(True) is ctx

    def test_instance_returns_itself(self):
        v = Verify(tolerance=9.0)
        assert resolve_verify(v) is v

    def test_none_no_context_returns_none(self):
        assert resolve_verify(None) is None

    def test_none_with_context_not_always_returns_none(self):
        with Verify(always=False):
            assert resolve_verify(None) is None

    def test_none_with_always_context_returns_context(self):
        ctx = Verify(always=True, tolerance=3.0)
        with ctx:
            result = resolve_verify(None)
            assert result is ctx


class TestThreadIsolation:
    def test_contexts_are_thread_local(self):
        v_main = Verify(tolerance=1.0)
        results = {}

        def thread_fn():
            results["thread_before"] = get_active_verify()
            v_thread = Verify(tolerance=99.0)
            with v_thread:
                results["thread_during"] = get_active_verify()
            results["thread_after"] = get_active_verify()

        with v_main:
            t = threading.Thread(target=thread_fn)
            t.start()
            t.join()
            assert get_active_verify() is v_main

        assert results["thread_before"] is None
        assert results["thread_during"].tolerance == 99.0
        assert results["thread_after"] is None


class TestValuesMatch:
    def test_floats_exact(self):
        assert values_match(1.0, 1.0)
        assert not values_match(1.0, 2.0)

    def test_floats_with_tolerance(self):
        assert values_match(1.0, 1.05, tolerance=0.1)
        assert not values_match(1.0, 1.2, tolerance=0.1)

    def test_ints(self):
        assert values_match(42, 42)
        assert not values_match(42, 43)

    def test_int_float_mix(self):
        assert values_match(1, 1.0)
        assert values_match(1.0, 1, tolerance=0.0)

    def test_bools(self):
        assert values_match(True, True)
        assert values_match(False, False)
        assert not values_match(True, False)

    def test_strings(self):
        assert values_match("abc", "abc")
        assert not values_match("abc", "xyz")

    def test_numpy_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        c = np.array([1.0, 2.0, 4.0])
        assert values_match(a, b)
        assert not values_match(a, c)

    def test_numpy_with_tolerance(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.05, 2.05])
        assert values_match(a, b, tolerance=0.1)
        assert not values_match(a, b, tolerance=0.01)

    def test_numpy_shape_mismatch(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        assert not values_match(a, b)
