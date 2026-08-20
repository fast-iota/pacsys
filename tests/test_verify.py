"""Tests for Verify dataclass, context manager, and resolution logic."""

import threading

import numpy as np
import pytest

from pacsys.verify import (
    Verify,
    get_active_verify,
    resolve_verify,
    values_match,
)


class TestVerifyContextManager:
    def test_context_manager_push_pop(self):
        assert get_active_verify() is None
        v = Verify(tolerance=1.0)
        with v:
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


class TestVerifyValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("tolerance", -0.1),
            ("tolerance", float("nan")),
            ("tolerance", float("inf")),
            ("initial_delay", -0.1),
            ("initial_delay", float("inf")),
            ("retry_delay", -0.1),
            ("retry_delay", float("nan")),
        ],
    )
    def test_rejects_invalid_numeric_value(self, field, value):
        with pytest.raises(ValueError, match=field):
            Verify(**{field: value})

    @pytest.mark.parametrize("field", ["tolerance", "initial_delay", "retry_delay"])
    def test_rejects_boolean_numeric_value(self, field):
        with pytest.raises(TypeError, match=field):
            Verify(**{field: True})

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_nonpositive_attempts(self, value):
        with pytest.raises(ValueError, match="max_attempts"):
            Verify(max_attempts=value)

    @pytest.mark.parametrize("value", [True, 1.5])
    def test_rejects_noninteger_attempts(self, value):
        with pytest.raises(TypeError, match="max_attempts"):
            Verify(max_attempts=value)

    def test_accepts_numpy_integer_attempts(self):
        assert Verify(max_attempts=np.int64(2)).max_attempts == 2


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
        assert not values_match(True, 1)
        assert not values_match(0, False)
        assert values_match(np.bool_(True), True)

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

    def test_numpy_tolerance_is_absolute_only(self):
        assert not values_match(np.array([1e9]), np.array([1e9 + 1]), tolerance=0.0)

    def test_numpy_scalar_uses_numeric_tolerance(self):
        assert values_match(np.int32(5), 6, tolerance=2.0)
        assert not values_match(np.int32(5), 8, tolerance=2.0)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (np.array([True]), np.array([1])),
            ([True], [1]),
        ],
    )
    def test_boolean_arrays_are_type_strict(self, a, b):
        assert not values_match(a, b)

    def test_equal_text_arrays_match_exactly(self):
        assert values_match(np.array(["a", "b"]), np.array(["a", "b"]))
        assert not values_match(np.array(["a", "b"]), np.array(["a", "c"]))

    def test_nan_never_matches(self):
        assert not values_match(float("nan"), float("nan"))
        assert not values_match(np.array([np.nan]), np.array([np.nan]))

    def test_same_signed_infinity_matches(self):
        assert values_match(float("inf"), float("inf"))
        assert values_match(float("-inf"), float("-inf"))
        assert not values_match(float("inf"), float("-inf"))
        assert not values_match(float("inf"), 1.0)

    def test_numpy_shape_mismatch(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        assert not values_match(a, b)

    def test_numpy_broadcast_compatible_shapes_do_not_match(self):
        assert not values_match(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))
