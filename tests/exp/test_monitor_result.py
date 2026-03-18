"""Tests for MonitorResult and ChannelData."""

import pytest
from datetime import datetime, timezone

from pacsys.exp._monitor import ChannelData, MonitorResult
from pacsys.types import Reading, ValueType


def _reading(drf: str, value: float, ts: datetime | None = None) -> Reading:
    """Helper to create a Reading."""
    return Reading(
        drf=drf,
        value_type=ValueType.SCALAR,
        value=value,
        timestamp=ts or datetime.now(timezone.utc),
    )


class TestChannelData:
    def test_values_extracts_ok_readings(self):
        ch = ChannelData(
            drf="M:OUTTMP",
            readings=(
                _reading("M:OUTTMP", 1.0),
                _reading("M:OUTTMP", 2.0),
            ),
        )
        assert ch.values() == [1.0, 2.0]

    def test_values_skips_error_readings(self):
        err = Reading(drf="M:OUTTMP", error_code=-1, message="fail")
        ch = ChannelData(drf="M:OUTTMP", readings=(_reading("M:OUTTMP", 1.0), err))
        assert ch.values() == [1.0]

    def test_timestamps(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ch = ChannelData(drf="M:OUTTMP", readings=(_reading("M:OUTTMP", 1.0, ts),))
        assert ch.timestamps() == [ts]

    def test_getitem(self):
        r0 = _reading("M:OUTTMP", 1.0)
        r1 = _reading("M:OUTTMP", 2.0)
        ch = ChannelData(drf="M:OUTTMP", readings=(r0, r1))
        assert ch[0] is r0
        assert ch[-1] is r1

    def test_len(self):
        ch = ChannelData(drf="M:OUTTMP", readings=(_reading("M:OUTTMP", 1.0),) * 3)
        assert len(ch) == 3

    def test_iter(self):
        r0 = _reading("M:OUTTMP", 1.0)
        r1 = _reading("M:OUTTMP", 2.0)
        ch = ChannelData(drf="M:OUTTMP", readings=(r0, r1))
        assert list(ch) == [r0, r1]


class TestMonitorResult:
    def _result(self, values_a=(1.0, 2.0, 3.0), values_b=(10.0, 20.0)):
        ch_a = ChannelData("A:DEV", tuple(_reading("A:DEV", v) for v in values_a))
        ch_b = ChannelData("B:DEV", tuple(_reading("B:DEV", v) for v in values_b))
        return MonitorResult(channels={"A:DEV": ch_a, "B:DEV": ch_b})

    def test_mean_single(self):
        r = self._result()
        assert r.mean("A:DEV") == pytest.approx(2.0)

    def test_mean_all(self):
        r = self._result()
        means = r.mean()
        assert means["A:DEV"] == pytest.approx(2.0)
        assert means["B:DEV"] == pytest.approx(15.0)

    def test_std_single(self):
        r = self._result(values_a=(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0))
        assert r.std("A:DEV") == pytest.approx(2.0)

    def test_min_max(self):
        r = self._result()
        assert r.min("A:DEV") == 1.0
        assert r.max("A:DEV") == 3.0

    def test_last(self):
        r = self._result()
        assert r.last(2, "A:DEV") == [2.0, 3.0]

    def test_last_all_channels(self):
        r = self._result()
        lasts = r.last(1)
        assert lasts["A:DEV"] == [3.0]
        assert lasts["B:DEV"] == [20.0]

    def test_values(self):
        r = self._result()
        assert r.values("A:DEV") == [1.0, 2.0, 3.0]

    def test_getitem(self):
        r = self._result()
        ch = r["A:DEV"]
        assert isinstance(ch, ChannelData)
        assert ch.drf == "A:DEV"

    def test_getitem_unknown_raises(self):
        r = self._result()
        with pytest.raises(KeyError, match="No channel"):
            r["X:NOPE"]

    def test_len(self):
        r = self._result()
        assert len(r) == 2

    def test_iter(self):
        r = self._result()
        assert set(r) == {"A:DEV", "B:DEV"}

    def test_contains(self):
        r = self._result()
        assert "A:DEV" in r
        assert "X:NOPE" not in r

    def test_getitem_chaining(self):
        """result[drf][0] returns a Reading."""
        r = self._result()
        reading = r["A:DEV"][0]
        assert reading.value == 1.0

    def test_unknown_channel_raises(self):
        r = self._result()
        with pytest.raises(KeyError, match="No channel"):
            r.mean("X:NOPE")

    def test_empty_channel_raises(self):
        ch = ChannelData("A:DEV", ())
        r = MonitorResult(channels={"A:DEV": ch})
        with pytest.raises(ValueError, match="No readings"):
            r.mean("A:DEV")


class TestMonitorResultToDict:
    def test_to_dict_returns_readings_per_channel(self):
        r1 = _reading("A:DEV", 1.0)
        r2 = _reading("A:DEV", 2.0)
        r3 = _reading("B:DEV", 10.0)
        ch_a = ChannelData("A:DEV", (r1, r2))
        ch_b = ChannelData("B:DEV", (r3,))
        result = MonitorResult(channels={"A:DEV": ch_a, "B:DEV": ch_b})
        d = result.to_dict()
        assert d["A:DEV"] == [r1, r2]
        assert d["B:DEV"] == [r3]

    def test_to_dict_empty_channels(self):
        ch = ChannelData("A:DEV", ())
        result = MonitorResult(channels={"A:DEV": ch})
        assert result.to_dict() == {"A:DEV": []}


class TestMonitorResultSlice:
    def test_slice_by_time_range(self):
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
        t4 = datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc)
        ch = ChannelData(
            "A:DEV",
            (
                _reading("A:DEV", 1.0, t1),
                _reading("A:DEV", 2.0, t2),
                _reading("A:DEV", 3.0, t3),
                _reading("A:DEV", 4.0, t4),
            ),
        )
        r = MonitorResult(channels={"A:DEV": ch})
        sliced = r.slice("A:DEV", start=t2, end=t3)
        assert sliced.values() == [2.0, 3.0]

    def test_slice_open_start(self):
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
        ch = ChannelData(
            "A:DEV",
            (
                _reading("A:DEV", 1.0, t1),
                _reading("A:DEV", 2.0, t2),
                _reading("A:DEV", 3.0, t3),
            ),
        )
        r = MonitorResult(channels={"A:DEV": ch})
        sliced = r.slice("A:DEV", end=t2)
        assert sliced.values() == [1.0, 2.0]

    def test_slice_open_end(self):
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
        ch = ChannelData(
            "A:DEV",
            (
                _reading("A:DEV", 1.0, t1),
                _reading("A:DEV", 2.0, t2),
                _reading("A:DEV", 3.0, t3),
            ),
        )
        r = MonitorResult(channels={"A:DEV": ch})
        sliced = r.slice("A:DEV", start=t2)
        assert sliced.values() == [2.0, 3.0]

    def test_slice_empty_result(self):
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t_far = datetime(2099, 1, 1, tzinfo=timezone.utc)
        ch = ChannelData("A:DEV", (_reading("A:DEV", 1.0, t1),))
        r = MonitorResult(channels={"A:DEV": ch})
        sliced = r.slice("A:DEV", start=t_far)
        assert sliced.values() == []


class TestMonitorResultMedian:
    def _result(self, values=(1.0, 3.0, 2.0)):
        ch = ChannelData("A:DEV", tuple(_reading("A:DEV", v) for v in values))
        return MonitorResult(channels={"A:DEV": ch})

    def test_median_odd(self):
        r = self._result((1.0, 3.0, 2.0))
        assert r.median("A:DEV") == 2.0

    def test_median_even(self):
        r = self._result((1.0, 2.0, 3.0, 4.0))
        assert r.median("A:DEV") == pytest.approx(2.5)

    def test_median_all_channels(self):
        ch_a = ChannelData("A:DEV", tuple(_reading("A:DEV", v) for v in (1.0, 3.0, 2.0)))
        ch_b = ChannelData("B:DEV", tuple(_reading("B:DEV", v) for v in (10.0, 20.0)))
        r = MonitorResult(channels={"A:DEV": ch_a, "B:DEV": ch_b})
        medians = r.median()
        assert medians["A:DEV"] == 2.0
        assert medians["B:DEV"] == pytest.approx(15.0)

    def test_median_empty_raises(self):
        ch = ChannelData("A:DEV", ())
        r = MonitorResult(channels={"A:DEV": ch})
        with pytest.raises(ValueError, match="No readings"):
            r.median("A:DEV")


class TestMonitorResultArrayValues:
    """Stats on array-valued channels (e.g. B:IRM011[0:700])."""

    def _array_result(self):
        np = pytest.importorskip("numpy")
        ch = ChannelData(
            "B:IRM",
            tuple(
                Reading(
                    drf="B:IRM",
                    value_type=ValueType.SCALAR_ARRAY,
                    value=np.array([float(i), float(i + 1), float(i + 2)]),
                    timestamp=datetime.now(timezone.utc),
                )
                for i in range(4)
            ),
        )
        return np, MonitorResult(channels={"B:IRM": ch})

    def test_mean_array(self):
        np, r = self._array_result()
        # values: [0,1,2], [1,2,3], [2,3,4], [3,4,5] → mean = [1.5, 2.5, 3.5]
        np.testing.assert_array_almost_equal(r.mean("B:IRM"), [1.5, 2.5, 3.5])

    def test_std_array(self):
        np, r = self._array_result()
        result = r.std("B:IRM")
        assert result.shape == (3,)
        assert result[0] > 0

    def test_median_array(self):
        np, r = self._array_result()
        # median of [0,1,2,3], [1,2,3,4], [2,3,4,5] → [1.5, 2.5, 3.5]
        np.testing.assert_array_almost_equal(r.median("B:IRM"), [1.5, 2.5, 3.5])

    def test_min_max_array(self):
        np, r = self._array_result()
        np.testing.assert_array_equal(r.min("B:IRM"), [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(r.max("B:IRM"), [3.0, 4.0, 5.0])

    def test_mean_all_mixed(self):
        """mean() across both scalar and array channels."""
        np = pytest.importorskip("numpy")
        ch_scalar = ChannelData("A:DEV", tuple(_reading("A:DEV", v) for v in (1.0, 3.0)))
        ch_array = ChannelData(
            "B:IRM",
            tuple(
                Reading(drf="B:IRM", value_type=ValueType.SCALAR_ARRAY, value=np.array([10.0, 20.0])) for _ in range(2)
            ),
        )
        r = MonitorResult(channels={"A:DEV": ch_scalar, "B:IRM": ch_array})
        means = r.mean()
        assert means["A:DEV"] == pytest.approx(2.0)
        np.testing.assert_array_equal(means["B:IRM"], [10.0, 20.0])


class TestMonitorResultTimedScalarArray:
    """Stats on TIMED_SCALAR_ARRAY channels (logger data with micros)."""

    def _timed_result(self):
        np = pytest.importorskip("numpy")
        ch = ChannelData(
            "L:DEV",
            tuple(
                Reading(
                    drf="L:DEV",
                    value_type=ValueType.TIMED_SCALAR_ARRAY,
                    value={"data": np.array([float(i), float(i + 10)]), "micros": np.array([100, 200])},
                    timestamp=datetime.now(timezone.utc),
                )
                for i in range(4)
            ),
        )
        return np, MonitorResult(channels={"L:DEV": ch})

    def test_mean_timed(self):
        np, r = self._timed_result()
        # data: [0,10], [1,11], [2,12], [3,13] → mean = [1.5, 11.5]
        np.testing.assert_array_almost_equal(r.mean("L:DEV"), [1.5, 11.5])

    def test_min_max_timed(self):
        np, r = self._timed_result()
        np.testing.assert_array_equal(r.min("L:DEV"), [0.0, 10.0])
        np.testing.assert_array_equal(r.max("L:DEV"), [3.0, 13.0])

    def test_to_numpy_timed(self):
        np, r = self._timed_result()
        timestamps, values = r.to_numpy("L:DEV")
        assert values.shape == (4, 2)
        np.testing.assert_array_equal(values[0], [0.0, 10.0])


class TestMonitorResultToNumpy:
    def test_to_numpy_returns_arrays(self):
        np = pytest.importorskip("numpy")
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        ch = ChannelData(
            "A:DEV",
            (
                _reading("A:DEV", 1.0, t1),
                _reading("A:DEV", 2.0, t2),
            ),
        )
        r = MonitorResult(channels={"A:DEV": ch})
        timestamps, values = r.to_numpy("A:DEV")
        np.testing.assert_array_equal(values, [1.0, 2.0])
        assert timestamps.dtype == np.float64
        assert len(timestamps) == 2

    def test_to_numpy_skips_error_readings(self):
        np = pytest.importorskip("numpy")
        err = Reading(drf="A:DEV", error_code=-1, message="fail")
        ch = ChannelData(
            "A:DEV",
            (
                _reading("A:DEV", 1.0),
                err,
                _reading("A:DEV", 3.0),
            ),
        )
        r = MonitorResult(channels={"A:DEV": ch})
        timestamps, values = r.to_numpy("A:DEV")
        np.testing.assert_array_equal(values, [1.0, 3.0])

    def test_to_numpy_empty(self):
        pytest.importorskip("numpy")
        ch = ChannelData("A:DEV", ())
        r = MonitorResult(channels={"A:DEV": ch})
        timestamps, values = r.to_numpy("A:DEV")
        assert len(timestamps) == 0
        assert len(values) == 0

    def test_to_numpy_array_values(self):
        np = pytest.importorskip("numpy")
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        ch = ChannelData(
            "B:IRM",
            (
                Reading(drf="B:IRM", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0]), timestamp=t1),
                Reading(drf="B:IRM", value_type=ValueType.SCALAR_ARRAY, value=np.array([3.0, 4.0]), timestamp=t2),
            ),
        )
        r = MonitorResult(channels={"B:IRM": ch})
        timestamps, values = r.to_numpy("B:IRM")
        assert values.shape == (2, 2)
        np.testing.assert_array_equal(values[0], [1.0, 2.0])
        np.testing.assert_array_equal(values[1], [3.0, 4.0])


class TestMonitorResultDataframeRelative:
    def test_relative_single_channel(self):
        pytest.importorskip("pandas")
        t_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
        ch = ChannelData(
            "A:DEV",
            (
                _reading("A:DEV", 1.0, t1),
                _reading("A:DEV", 2.0, t2),
            ),
        )
        r = MonitorResult(channels={"A:DEV": ch}, started=t_start)
        df = r.to_dataframe("A:DEV", relative=True)
        assert df.index.name == "elapsed_s"
        assert list(df.index) == pytest.approx([1.0, 2.0])

    def test_relative_all_channels(self):
        pytest.importorskip("pandas")
        t_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        ch = ChannelData("A:DEV", (_reading("A:DEV", 1.0, t1),))
        r = MonitorResult(channels={"A:DEV": ch}, started=t_start)
        df = r.to_dataframe(relative=True)
        assert "elapsed_s" in df.columns
        assert df["elapsed_s"].iloc[0] == pytest.approx(1.0)

    def test_relative_no_start_raises(self):
        pytest.importorskip("pandas")
        ch = ChannelData("A:DEV", (_reading("A:DEV", 1.0),))
        r = MonitorResult(channels={"A:DEV": ch}, started=None)
        with pytest.raises(ValueError, match="started"):
            r.to_dataframe("A:DEV", relative=True)
