"""Tests for read_fresh."""

import threading
import time
from datetime import datetime, timezone

import pytest

from pacsys.exp._read_fresh import FreshResult, read_fresh
from pacsys.testing import FakeBackend
from pacsys.types import Reading, ValueType


def _make_reading(drf="M:OUTTMP@p,1000", value=1.0, ts_offset=0):
    return Reading(
        drf=drf,
        value=value,
        value_type=ValueType.SCALAR,
        error_code=0,
        timestamp=datetime(2026, 3, 6, 12, 0, ts_offset, tzinfo=timezone.utc),
    )


class TestFreshResult:
    def test_len(self):
        r = FreshResult(drf="X", readings=(_make_reading(),) * 3, requested_count=2)
        assert len(r) == 3

    def test_length_counts_ok_values(self):
        ok = _make_reading(value=1.0)
        err = Reading(drf="M:OUTTMP@p,1000", error_code=-1, message="fail")
        r = FreshResult(drf="X", readings=(ok, err, ok), requested_count=2)
        assert r.length == 2

    def test_value_returns_last(self):
        r = FreshResult(drf="X", readings=(_make_reading(value=1.0), _make_reading(value=2.0)), requested_count=2)
        assert r.value == 2.0

    def test_values_returns_all_ok(self):
        ok1 = _make_reading(value=1.0)
        ok2 = _make_reading(value=2.0)
        err = Reading(drf="M:OUTTMP@p,1000", error_code=-1, message="fail")
        r = FreshResult(drf="X", readings=(ok1, err, ok2), requested_count=2)
        assert r.values == [1.0, 2.0]

    def test_reading_returns_last(self):
        r1 = _make_reading(value=1.0)
        r2 = _make_reading(value=2.0)
        r = FreshResult(drf="X", readings=(r1, r2), requested_count=2)
        assert r.reading is r2

    def test_timestamp_returns_last(self):
        r1 = _make_reading(value=1.0, ts_offset=0)
        r2 = _make_reading(value=2.0, ts_offset=1)
        r = FreshResult(drf="X", readings=(r1, r2), requested_count=2)
        assert r.timestamp == r2.timestamp

    def test_timestamps_returns_all(self):
        r1 = _make_reading(value=1.0, ts_offset=0)
        r2 = _make_reading(value=2.0, ts_offset=1)
        r = FreshResult(drf="X", readings=(r1, r2), requested_count=2)
        assert r.timestamps == [r1.timestamp, r2.timestamp]

    def test_empty_readings_raises_on_value(self):
        r = FreshResult(drf="X", readings=(), requested_count=0)
        with pytest.raises(IndexError):
            _ = r.value


class TestFreshResultStats:
    def _result(self, values, requested_count=None):
        if requested_count is None:
            requested_count = len(values)
        readings = tuple(_make_reading(value=v, ts_offset=i) for i, v in enumerate(values))
        return FreshResult(drf="X", readings=readings, requested_count=requested_count)

    def test_mean_default_uses_requested_count(self):
        # 5 readings, requested 3 -> mean of last 3
        r = self._result([1.0, 2.0, 3.0, 4.0, 5.0], requested_count=3)
        assert r.mean() == pytest.approx(4.0)  # (3+4+5)/3

    def test_mean_all(self):
        r = self._result([1.0, 2.0, 3.0, 4.0, 5.0], requested_count=3)
        assert r.mean(-1) == pytest.approx(3.0)  # (1+2+3+4+5)/5

    def test_mean_explicit_n(self):
        r = self._result([1.0, 2.0, 3.0, 4.0, 5.0], requested_count=3)
        assert r.mean(2) == pytest.approx(4.5)  # (4+5)/2

    def test_mean_n_too_large_raises(self):
        r = self._result([1.0, 2.0], requested_count=2)
        with pytest.raises(ValueError, match="fewer than 5"):
            r.mean(5)

    def test_std(self):
        r = self._result([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert r.std() == pytest.approx(2.0)

    def test_median_odd(self):
        r = self._result([1.0, 3.0, 2.0])
        assert r.median() == 2.0

    def test_median_even(self):
        r = self._result([1.0, 2.0, 3.0, 4.0])
        assert r.median() == 2.5

    def test_min_max(self):
        r = self._result([3.0, 1.0, 4.0, 1.0, 5.0])
        assert r.min() == 1.0
        assert r.max() == 5.0

    def test_n_zero_raises(self):
        r = self._result([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="n must be >= 1"):
            r.mean(0)

    def test_n_negative_raises(self):
        r = self._result([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="n must be >= 1"):
            r.mean(-2)

    def test_stats_skip_error_readings(self):
        ok1 = _make_reading(value=2.0, ts_offset=0)
        err = Reading(drf="M:OUTTMP@p,1000", error_code=-1, message="fail")
        ok2 = _make_reading(value=4.0, ts_offset=2)
        r = FreshResult(drf="X", readings=(ok1, err, ok2), requested_count=2)
        assert r.mean() == pytest.approx(3.0)

    def test_stats_on_non_numeric_raises(self):
        readings = (Reading(drf="X", value="text", value_type=ValueType.TEXT, error_code=0),)
        r = FreshResult(drf="X", readings=readings, requested_count=1)
        with pytest.raises(TypeError):
            r.mean()

    def test_stats_no_values_raises(self):
        r = FreshResult(drf="X", readings=(), requested_count=0)
        with pytest.raises(ValueError, match="No values"):
            r.mean()


@pytest.fixture
def fake():
    return FakeBackend()


class TestReadFresh:
    def test_basic(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(["M:OUTTMP@p,1000"], timeout=1.0, backend=fake)
        assert len(results) == 1
        assert results[0].value == 72.5

    def test_multiple_channels(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)
            fake.emit_reading("G:AMANDA@p,1000", 1.5)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(
            ["M:OUTTMP@p,1000", "G:AMANDA@p,1000"],
            timeout=1.0,
            backend=fake,
        )
        assert len(results) == 2
        assert results[0].value == 72.5
        assert results[1].value == 1.5

    def test_timeout_raises(self, fake):
        with pytest.raises(TimeoutError, match="Timed out"):
            read_fresh(["M:OUTTMP@p,1000"], timeout=0.1, backend=fake)

    def test_default_event_applied(self, fake):
        """default_event is applied to DRFs without events."""

        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP.READING@p,1000", 72.5)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(
            ["M:OUTTMP"],
            default_event="p,1000",
            timeout=1.0,
            backend=fake,
        )
        assert results[0].value == 72.5

    def test_empty_devices_raises(self, fake):
        with pytest.raises(ValueError, match="devices cannot be empty"):
            read_fresh([], backend=fake)

    def test_preserves_order(self, fake):
        def emitter():
            time.sleep(0.02)
            # Emit in reverse order
            fake.emit_reading("B:DEV@p,1000", 2.0)
            fake.emit_reading("A:DEV@p,1000", 1.0)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(
            ["A:DEV@p,1000", "B:DEV@p,1000"],
            timeout=1.0,
            backend=fake,
        )
        assert results[0].value == 1.0
        assert results[1].value == 2.0

    def test_collects_all_readings(self, fake):
        """With count=1, extra readings are still stored."""

        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)
            fake.emit_reading("M:OUTTMP@p,1000", 99.0)
            time.sleep(0.02)  # let second arrive

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(["M:OUTTMP@p,1000"], timeout=1.0, backend=fake)
        assert isinstance(results[0], FreshResult)
        assert results[0].value is not None  # has at least one


class TestReadFreshMultiCount:
    def test_count_collects_multiple(self, fake):
        def emitter():
            time.sleep(0.02)
            for i in range(5):
                fake.emit_reading("M:OUTTMP@p,1000", float(i))
                time.sleep(0.01)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(["M:OUTTMP@p,1000"], count=5, timeout=2.0, backend=fake)
        assert len(results) == 1
        assert isinstance(results[0], FreshResult)
        assert len(results[0]) >= 5
        assert results[0].requested_count == 5

    def test_count_1_default(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(["M:OUTTMP@p,1000"], timeout=1.0, backend=fake)
        assert isinstance(results[0], FreshResult)
        assert results[0].value == 72.5
        assert results[0].requested_count == 1

    def test_count_timeout_raises(self, fake):
        def emitter():
            time.sleep(0.02)
            fake.emit_reading("M:OUTTMP@p,1000", 1.0)  # only 1

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        with pytest.raises(TimeoutError):
            read_fresh(["M:OUTTMP@p,1000"], count=10, timeout=0.2, backend=fake)

    def test_stats_on_fresh_result(self, fake):
        def emitter():
            time.sleep(0.02)
            for v in [10.0, 20.0, 30.0]:
                fake.emit_reading("M:OUTTMP@p,1000", v)
                time.sleep(0.01)

        t = threading.Thread(target=emitter, daemon=True)
        t.start()
        results = read_fresh(["M:OUTTMP@p,1000"], count=3, timeout=2.0, backend=fake)
        assert results[0].mean() == pytest.approx(20.0)

    def test_count_zero_raises(self, fake):
        with pytest.raises(ValueError, match="count must be >= 1"):
            read_fresh(["M:OUTTMP@p,1000"], count=0, backend=fake)
