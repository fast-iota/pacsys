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

    def test_unknown_channel_raises(self):
        r = self._result()
        with pytest.raises(KeyError, match="No channel"):
            r.mean("X:NOPE")

    def test_empty_channel_raises(self):
        ch = ChannelData("A:DEV", ())
        r = MonitorResult(channels={"A:DEV": ch})
        with pytest.raises(ValueError, match="No readings"):
            r.mean("A:DEV")


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
