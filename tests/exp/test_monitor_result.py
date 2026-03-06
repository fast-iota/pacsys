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
