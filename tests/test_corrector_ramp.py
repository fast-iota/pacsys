"""Tests for corrector ramp table manipulation."""

import struct

import numpy as np
import pytest

from pacsys.corrector_ramp import BoosterRamp, CorrectorRamp


def _make_ramp_bytes(pairs: list[tuple[int, int]], n_points: int = 64) -> bytes:
    """Build raw ramp bytes from (raw_value, raw_time) int16 pairs, zero-padded."""
    while len(pairs) < n_points:
        pairs.append((0, 0))
    fmt = f"<{n_points * 2}h"
    flat = []
    for v, t in pairs:
        flat.extend([v, t])
    return struct.pack(fmt, *flat)


class TestFromBytes:
    def test_parse_zeros(self):
        data = b"\x00" * 256
        ramp = BoosterRamp.from_bytes(data)
        assert np.all(ramp.values == 0.0)
        assert np.all(ramp.times == 0)

    def test_parse_known_values(self):
        # raw_value=8192 -> eng = 8192 / 3276.8 * 4.0 = 10.0
        # raw_time=100 -> 100 us
        data = _make_ramp_bytes([(8192, 100), (-8192, 200)])
        ramp = BoosterRamp.from_bytes(data)
        assert ramp.values[0] == pytest.approx(10.0, abs=0.01)
        assert ramp.values[1] == pytest.approx(-10.0, abs=0.01)
        assert ramp.times[0] == 100
        assert ramp.times[1] == 200
        assert np.all(ramp.values[2:] == 0.0)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="requires 256 bytes"):
            BoosterRamp.from_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="requires 256 bytes"):
            BoosterRamp.from_bytes(b"\x00" * 300)

    def test_byte_order(self):
        # Manual little-endian: value=0x0400=1024, time=0x0010=16
        point = struct.pack("<hh", 1024, 16)
        data = point + b"\x00" * (256 - 4)
        ramp = BoosterRamp.from_bytes(data)
        expected_eng = 1024 / 3276.8 * 4.0
        assert ramp.values[0] == pytest.approx(expected_eng, abs=0.001)
        assert ramp.times[0] == 16


class TestToBytes:
    def test_round_trip(self):
        """from_bytes(to_bytes(x)) preserves values within quantization."""
        original_data = _make_ramp_bytes([(8192, 100), (-4096, 50), (16384, 300)])
        ramp = BoosterRamp.from_bytes(original_data)
        round_tripped = BoosterRamp.from_bytes(ramp.to_bytes())
        np.testing.assert_allclose(round_tripped.values, ramp.values, atol=0.002)
        np.testing.assert_array_equal(round_tripped.times, ramp.times)

    def test_known_engineering_to_raw(self):
        """Engineering value 10.0 -> raw 8192 (10.0 / 4.0 * 3276.8 = 8192)."""
        ramp = BoosterRamp(
            values=np.zeros(64),
            times=np.zeros(64, dtype=np.int16),
        )
        ramp.values[0] = 10.0
        ramp.times[0] = 50
        raw = ramp.to_bytes()
        v, t = struct.unpack_from("<hh", raw, 0)
        assert v == 8192
        assert t == 50

    def test_rounding(self):
        """Values that don't map to exact int16 are rounded."""
        ramp = BoosterRamp(
            values=np.zeros(64),
            times=np.zeros(64, dtype=np.int16),
        )
        # 1.0 Amps -> raw = 1.0 / 4.0 * 3276.8 = 819.2 -> rounds to 819
        ramp.values[0] = 1.0
        raw = ramp.to_bytes()
        v, _ = struct.unpack_from("<hh", raw, 0)
        assert v == 819


class TestBoosterRamp:
    def test_transforms(self):
        raw = np.array([3276.8])
        assert BoosterRamp.primary_transform(raw)[0] == pytest.approx(1.0)
        assert BoosterRamp.common_transform(np.array([1.0]))[0] == pytest.approx(4.0)
        assert BoosterRamp.inverse_common_transform(np.array([4.0]))[0] == pytest.approx(1.0)
        assert BoosterRamp.inverse_primary_transform(np.array([1.0]))[0] == pytest.approx(3276.8)

    def test_conversion_forward(self):
        """raw 8192 -> eng 10.0"""
        data = _make_ramp_bytes([(8192, 0)])
        ramp = BoosterRamp.from_bytes(data)
        assert ramp.values[0] == pytest.approx(10.0, abs=0.01)

    def test_conversion_inverse(self):
        """eng 10.0 -> raw 8192"""
        ramp = BoosterRamp(
            values=np.array([10.0] + [0.0] * 63),
            times=np.zeros(64, dtype=np.int16),
        )
        raw = ramp.to_bytes()
        v, _ = struct.unpack_from("<hh", raw, 0)
        assert v == 8192


class TestValidation:
    def test_wrong_values_length(self):
        with pytest.raises(ValueError, match="Expected 64 values"):
            BoosterRamp(values=np.zeros(10), times=np.zeros(64, dtype=np.int16))

    def test_wrong_times_length(self):
        with pytest.raises(ValueError, match="Expected 64 times"):
            BoosterRamp(values=np.zeros(64), times=np.zeros(10, dtype=np.int16))

    def test_max_value_exceeded(self):
        ramp = BoosterRamp(
            values=np.array([1500.0] + [0.0] * 63),
            times=np.zeros(64, dtype=np.int16),
        )
        with pytest.raises(ValueError, match="Ramp values exceed max"):
            ramp.to_bytes()

    def test_max_time_exceeded(self):
        ramp = BoosterRamp(
            values=np.zeros(64),
            times=np.array([7000] + [0] * 63, dtype=np.int16),
        )
        with pytest.raises(ValueError, match="Ramp times exceed max"):
            ramp.to_bytes()


class TestReadWrite:
    def test_read_success(self, fake_backend):
        raw_bytes = _make_ramp_bytes([(8192, 100)])
        fake_backend.set_reading("B:HS23T.SETTING{0:256}.RAW@I", raw_bytes)

        ramp = BoosterRamp.read("B:HS23T", slot=0, backend=fake_backend)
        assert ramp.values[0] == pytest.approx(10.0, abs=0.01)
        assert ramp.times[0] == 100

    def test_read_error_raises(self, fake_backend):
        fake_backend.set_error("B:HS23T.SETTING{0:256}.RAW@I", -1, "Device offline")

        from pacsys.errors import DeviceError

        with pytest.raises(DeviceError):
            BoosterRamp.read("B:HS23T", slot=0, backend=fake_backend)

    def test_read_non_bytes_raises(self, fake_backend):
        fake_backend.set_reading("B:HS23T.SETTING{0:256}.RAW@I", 42.0)

        with pytest.raises(Exception):  # DeviceError from range apply or TypeError
            BoosterRamp.read("B:HS23T", slot=0, backend=fake_backend)

    def test_write_success(self, fake_backend):
        ramp = BoosterRamp(
            values=np.array([10.0] + [0.0] * 63),
            times=np.zeros(64, dtype=np.int16),
        )
        ramp.write("B:HS23T", slot=0, backend=fake_backend)

        assert len(fake_backend.writes) == 1
        drf, value = fake_backend.writes[0]
        assert "SETTING" in drf
        assert ".RAW" in drf
        assert isinstance(value, bytes)
        assert len(value) == 256

    def test_slot_drf(self):
        """Verify DRF uses correct byte offsets for different slots."""
        drf0 = BoosterRamp._make_drf("B:HS23T", slot=0)
        assert "{0:256}" in drf0

        drf1 = BoosterRamp._make_drf("B:HS23T", slot=1)
        assert "{256:256}" in drf1

        drf2 = BoosterRamp._make_drf("B:HS23T", slot=2)
        assert "{512:256}" in drf2


class TestModifyContext:
    def test_modify_reads_and_returns_ramp(self, fake_backend):
        raw_bytes = _make_ramp_bytes([(8192, 100)])
        fake_backend.set_reading("B:HS23T.SETTING{0:256}.RAW@I", raw_bytes)

        with BoosterRamp.modify("B:HS23T", slot=0, backend=fake_backend) as ramp:
            assert isinstance(ramp, BoosterRamp)
            assert ramp.values[0] == pytest.approx(10.0, abs=0.01)

    def test_modify_writes_on_change(self, fake_backend):
        raw_bytes = _make_ramp_bytes([(8192, 100)])
        fake_backend.set_reading("B:HS23T.SETTING{0:256}.RAW@I", raw_bytes)

        with BoosterRamp.modify("B:HS23T", slot=0, backend=fake_backend) as ramp:
            ramp.values[0] = 20.0

        assert len(fake_backend.writes) == 1
        _, value = fake_backend.writes[0]
        assert isinstance(value, bytes)

    def test_modify_no_write_on_no_change(self, fake_backend):
        raw_bytes = _make_ramp_bytes([(8192, 100)])
        fake_backend.set_reading("B:HS23T.SETTING{0:256}.RAW@I", raw_bytes)

        with BoosterRamp.modify("B:HS23T", slot=0, backend=fake_backend) as _ramp:
            pass

        assert len(fake_backend.writes) == 0

    def test_modify_no_write_on_exception(self, fake_backend):
        raw_bytes = _make_ramp_bytes([(8192, 100)])
        fake_backend.set_reading("B:HS23T.SETTING{0:256}.RAW@I", raw_bytes)

        with pytest.raises(RuntimeError):
            with BoosterRamp.modify("B:HS23T", slot=0, backend=fake_backend) as ramp:
                ramp.values[0] = 20.0
                raise RuntimeError("abort")

        assert len(fake_backend.writes) == 0

    def test_modify_error_reading_raises(self, fake_backend):
        fake_backend.set_error("B:HS23T.SETTING{0:256}.RAW@I", -1, "Device offline")

        from pacsys.errors import DeviceError

        with pytest.raises(DeviceError):
            with BoosterRamp.modify("B:HS23T", slot=0, backend=fake_backend) as _ramp:
                pass


class TestReprStr:
    def test_repr_with_active_points(self):
        ramp = BoosterRamp(
            values=np.array([10.0, 5.0] + [0.0] * 62),
            times=np.zeros(64, dtype=np.int16),
        )
        r = repr(ramp)
        assert "BoosterRamp" in r
        assert "2/64" in r

    def test_repr_all_zeros(self):
        ramp = BoosterRamp(values=np.zeros(64), times=np.zeros(64, dtype=np.int16))
        r = repr(ramp)
        assert "0/64" in r

    def test_str_shows_nonzero(self):
        ramp = BoosterRamp(
            values=np.array([10.0] + [0.0] * 63),
            times=np.array([100] + [0] * 63, dtype=np.int16),
        )
        s = str(ramp)
        assert "[ 0]" in s
        assert "10.0000" in s
        assert "100" in s

    def test_str_all_zeros(self):
        ramp = BoosterRamp(values=np.zeros(64), times=np.zeros(64, dtype=np.int16))
        s = str(ramp)
        assert "all zeros" in s


class TestCustomSubclass:
    def test_custom_transforms(self):
        class TestRamp(CorrectorRamp):
            @classmethod
            def primary_transform(cls, raw):
                return raw / 1000.0

            @classmethod
            def common_transform(cls, primary):
                return primary * 2.0

            @classmethod
            def inverse_common_transform(cls, common):
                return common / 2.0

            @classmethod
            def inverse_primary_transform(cls, primary):
                return primary * 1000.0

        # raw 500 -> primary 0.5 -> eng 1.0
        data = _make_ramp_bytes([(500, 0)])
        ramp = TestRamp.from_bytes(data)
        assert ramp.values[0] == pytest.approx(1.0)

    def test_nonlinear_transforms(self):
        """Transform functions can be nonlinear."""

        class LogRamp(CorrectorRamp):
            @classmethod
            def primary_transform(cls, raw):
                return raw.astype(np.float64)

            @classmethod
            def common_transform(cls, primary):
                return primary**2 / 1000.0

            @classmethod
            def inverse_common_transform(cls, common):
                return np.sqrt(common * 1000.0)

            @classmethod
            def inverse_primary_transform(cls, primary):
                return primary

        data = _make_ramp_bytes([(100, 0)])
        ramp = LogRamp.from_bytes(data)
        assert ramp.values[0] == pytest.approx(10.0)  # 100^2 / 1000 = 10
        # Round-trip
        rt = LogRamp.from_bytes(ramp.to_bytes())
        assert rt.values[0] == pytest.approx(10.0, abs=0.1)

    def test_missing_transforms_raises(self):
        """Using CorrectorRamp directly without implementing transforms raises."""
        with pytest.raises(NotImplementedError):
            CorrectorRamp.from_bytes(b"\x00" * 256)


@pytest.fixture
def fake_backend():
    from pacsys.testing import FakeBackend

    return FakeBackend()
