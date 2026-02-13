"""Tests for ramp table manipulation."""

import struct

import numpy as np
import pytest

from pacsys.errors import DeviceError
from pacsys.types import ValueType
from pacsys.ramp import (
    BoosterHVRamp,
    BoosterHVRampGroup,
    BoosterQRamp,
    Ramp,
    RampGroup,
    RecyclerHVSQRamp,
    RecyclerQRamp,
    RecyclerSCRamp,
    RecyclerSRamp,
    read_ramps,
    write_ramps,
)


class _TestRamp(Ramp):
    """Identity-transform ramp for infrastructure tests.

    raw == engineering, raw_ticks == microseconds.
    No validation bounds - tests focus on mechanics, not limits.
    """

    update_rate_hz = 1_000_000  # 1 MHz → 1 us/tick

    @classmethod
    def primary_transform(cls, raw):
        return raw.astype(np.float64)

    @classmethod
    def common_transform(cls, primary):
        return primary

    @classmethod
    def inverse_common_transform(cls, common):
        return common

    @classmethod
    def inverse_primary_transform(cls, primary):
        return primary


class _TestRampGroup(RampGroup):
    base = _TestRamp


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
        ramp = BoosterHVRamp.from_bytes(data)
        assert np.all(ramp.values == 0.0)
        assert np.all(ramp.times == 0)

    def test_parse_known_values(self):
        # raw_value=8192 -> eng = 8192 / 3276.8 * 4.0 = 10.0
        # raw_time=100 ticks -> 100 * 10 = 1000 us at 100 KHz
        data = _make_ramp_bytes([(8192, 100), (-8192, 200)])
        ramp = BoosterHVRamp.from_bytes(data)
        assert ramp.values[0] == 10.0
        assert ramp.values[1] == -10.0
        assert ramp.times[0] == 1_000.0
        assert ramp.times[1] == 2_000.0
        assert np.all(ramp.values[2:] == 0.0)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="requires 256 bytes"):
            BoosterHVRamp.from_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="requires 256 bytes"):
            BoosterHVRamp.from_bytes(b"\x00" * 300)

    def test_byte_order(self):
        # Manual little-endian: value=0x0400=1024, time=0x0010=16
        point = struct.pack("<hh", 1024, 16)
        data = point + b"\x00" * (256 - 4)
        ramp = BoosterHVRamp.from_bytes(data)
        expected_eng = 1024 / 3276.8 * 4.0
        assert ramp.values[0] == expected_eng
        assert ramp.times[0] == 160.0  # 16 ticks * 10 us


class TestToBytes:
    def test_round_trip(self):
        """from_bytes(to_bytes(x)) preserves values within quantization."""
        original_data = _make_ramp_bytes([(8192, 100), (-4096, 50), (16384, 300)])
        ramp = BoosterHVRamp.from_bytes(original_data)
        round_tripped = BoosterHVRamp.from_bytes(ramp.to_bytes())
        np.testing.assert_allclose(round_tripped.values, ramp.values, atol=0.002)
        np.testing.assert_array_equal(round_tripped.times, ramp.times)

    def test_int16_boundary_round_trip(self):
        """Extreme int16 values (-32768, 32767) survive from_bytes -> to_bytes."""
        data = _make_ramp_bytes([(-32768, -32768), (32767, 32767)])
        ramp = _TestRamp.from_bytes(data)
        assert ramp.values[0] == -32768.0
        assert ramp.values[1] == 32767.0
        assert ramp.times[0] == -32768.0
        assert ramp.times[1] == 32767.0
        raw = ramp.to_bytes()
        v0, t0, v1, t1 = struct.unpack_from("<hhhh", raw, 0)
        assert (v0, t0) == (-32768, -32768)
        assert (v1, t1) == (32767, 32767)

    def test_known_engineering_to_raw(self):
        """Engineering value 10.0 -> raw 8192 (10.0 / 4.0 * 3276.8 = 8192)."""
        ramp = BoosterHVRamp(
            values=np.zeros(64),
            times=np.zeros(64),
        )
        ramp.values[0] = 10.0
        ramp.times[0] = 500.0  # 500 us -> 50 ticks at 100 KHz
        raw = ramp.to_bytes()
        v, t = struct.unpack_from("<hh", raw, 0)
        assert v == 8192
        assert t == 50

    def test_rounding(self):
        """Values that don't map to exact int16 are rounded."""
        ramp = BoosterHVRamp(
            values=np.zeros(64),
            times=np.zeros(64),
        )
        # 1.0 Amps -> raw = 1.0 / 4.0 * 3276.8 = 819.2 -> rounds to 819
        ramp.values[0] = 1.0
        raw = ramp.to_bytes()
        v, _ = struct.unpack_from("<hh", raw, 0)
        assert v == 819


_ALL_RAMP_CLASSES = [
    pytest.param(BoosterHVRamp, id="BoosterHV"),
    pytest.param(BoosterQRamp, id="BoosterQ"),
    pytest.param(RecyclerQRamp, id="RecyclerQ"),
    pytest.param(RecyclerSRamp, id="RecyclerS"),
    pytest.param(RecyclerSCRamp, id="RecyclerSC"),
    pytest.param(RecyclerHVSQRamp, id="RecyclerHVSQ"),
]


@pytest.mark.parametrize("ramp_cls", _ALL_RAMP_CLASSES)
class TestScalerRamp:
    """Conversion tests for all concrete Ramp subclasses with scaler."""

    def test_scaler_configured(self, ramp_cls):
        from pacsys.scaling import Scaler

        s = ramp_cls.scaler
        assert isinstance(s, Scaler)
        assert s.unscale(s.scale(1000)) == 1000

    def test_conversion_forward(self, ramp_cls):
        """from_bytes applies scaler correctly."""
        raw = 1000
        expected = ramp_cls.scaler.scale(raw)
        data = _make_ramp_bytes([(raw, 0)])
        ramp = ramp_cls.from_bytes(data)
        assert ramp.values[0] == pytest.approx(expected)

    def test_conversion_inverse(self, ramp_cls):
        """to_bytes inverts scaler correctly."""
        raw = 1000
        eng = ramp_cls.scaler.scale(raw)
        ramp = ramp_cls(
            values=np.array([eng] + [0.0] * 63),
            times=np.zeros(64),
        )
        raw_bytes = ramp.to_bytes()
        v, _ = struct.unpack_from("<hh", raw_bytes, 0)
        assert v == raw


class TestValidation:
    def test_wrong_values_length(self):
        with pytest.raises(ValueError, match="Expected 64 values"):
            BoosterHVRamp(values=np.zeros(10), times=np.zeros(64))

    def test_wrong_times_length(self):
        with pytest.raises(ValueError, match="Expected 64 times"):
            BoosterHVRamp(values=np.zeros(64), times=np.zeros(10))

    def test_max_value_exceeded(self):
        ramp = BoosterHVRamp(
            values=np.array([1500.0] + [0.0] * 63),
            times=np.zeros(64),
        )
        with pytest.raises(ValueError, match="Ramp values exceed max"):
            ramp.to_bytes()

    def test_nan_value_raises(self):
        ramp = BoosterHVRamp(values=np.zeros(64), times=np.zeros(64))
        ramp.values[0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            ramp.to_bytes()

    def test_inf_time_raises(self):
        """Inf time is caught - by max_time if set, by finite check otherwise."""
        ramp = BoosterHVRamp(values=np.zeros(64), times=np.zeros(64))
        ramp.times[0] = float("inf")
        with pytest.raises(ValueError):
            ramp.to_bytes()

    def test_value_overflow_int16_raises(self):
        """Engineering value that maps to raw > 32767 raises on serialization."""
        ramp = BoosterHVRamp(values=np.zeros(64), times=np.zeros(64))
        ramp.values[0] = 50.0  # raw = 50/4*3276.8 = 40960, exceeds int16
        with pytest.raises(ValueError, match="overflow"):
            ramp.to_bytes()

    def test_max_time_exceeded(self):
        # 66_660 us is max for BoosterHVRamp; 70_000 us exceeds it
        ramp = BoosterHVRamp(
            values=np.zeros(64),
            times=np.array([70_000.0] + [0.0] * 63),
        )
        with pytest.raises(ValueError, match="Ramp times exceed max"):
            ramp.to_bytes()

    def test_value_at_max_allowed(self):
        """Exactly max_value (1000.0) passes _validate()."""
        ramp = BoosterHVRamp(
            values=np.array([1000.0] + [0.0] * 63),
            times=np.zeros(64),
        )
        ramp._validate()  # should not raise

    def test_time_at_max_allowed(self):
        """Exactly max_time (66660 us) is accepted."""
        ramp = BoosterHVRamp(
            values=np.zeros(64),
            times=np.array([66_660.0] + [0.0] * 63),
        )
        ramp.to_bytes()  # should not raise

    def test_time_overflow_int16_raises(self):
        """Time that maps to raw ticks > 32767 raises on serialization."""
        ramp = _TestRamp(values=np.zeros(64), times=np.zeros(64))
        ramp.times[0] = 40_000.0  # raw ticks = 40000, exceeds int16 max
        with pytest.raises(ValueError, match="overflow int16"):
            ramp.to_bytes()


class TestReadWrite:
    def test_read_success(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        ramp = _TestRamp.read("B:HS23T", slot=0, backend=fake_backend)
        assert ramp.values[0] == 100.0
        assert ramp.times[0] == 50.0
        assert ramp.device == "B:HS23T"
        assert ramp.slot == 0

    def test_read_nonzero_slot(self, fake_backend):
        """Non-zero slot exercises FakeBackend byte-range slicing."""
        _setup_devices(fake_backend, ["B:HS23T"], slot=2)
        ramp = _TestRamp.read("B:HS23T", slot=2, backend=fake_backend)
        assert ramp.values[0] == 100.0
        assert ramp.device == "B:HS23T"
        assert ramp.slot == 2

    def test_read_error_raises(self, fake_backend):
        fake_backend.set_error("B:HS23T.SETTING.RAW@I", -1, "Device offline")

        with pytest.raises(DeviceError):
            _TestRamp.read("B:HS23T", slot=0, backend=fake_backend)

    def test_read_non_bytes_raises(self, fake_backend):
        fake_backend.set_reading("B:HS23T.SETTING.RAW", 42.0)

        with pytest.raises(DeviceError):
            _TestRamp.read("B:HS23T", slot=0, backend=fake_backend)

    def test_write_success(self, fake_backend):
        ramp = _TestRamp(
            values=np.array([100.0] + [0.0] * 63),
            times=np.zeros(64),
        )
        ramp.write("B:HS23T", slot=0, backend=fake_backend)

        assert len(fake_backend.writes) == 1
        drf, value = fake_backend.writes[0]
        assert "B:HS23T" in drf
        assert "SETTING" in drf
        assert ".RAW" in drf
        assert "{0:256}" in drf  # slot 0
        assert isinstance(value, bytes)
        assert len(value) == 256

    def test_write_nonzero_slot(self, fake_backend):
        """Writing to slot=3 produces correct byte offset in DRF."""
        ramp = _TestRamp(values=np.zeros(64), times=np.zeros(64))
        ramp.write("B:HS23T", slot=3, backend=fake_backend)
        drf, value = fake_backend.writes[0]
        assert "B:HS23T" in drf
        assert "{768:256}" in drf  # slot 3 = 3*256
        assert isinstance(value, bytes)

    def test_write_then_read_returns_updated_value(self, fake_backend):
        """Written ramp data is readable back through FakeBackend."""
        _setup_devices(fake_backend, ["B:HS23T"])
        ramp = _TestRamp.read("B:HS23T", slot=0, backend=fake_backend)
        ramp.values[0] = 20.0
        ramp.write("B:HS23T", slot=0, backend=fake_backend)

        ramp2 = _TestRamp.read("B:HS23T", slot=0, backend=fake_backend)
        assert ramp2.values[0] == 20.0

    def test_write_failure_raises(self, fake_backend):
        fake_backend.set_write_result("B:HS23T.SETTING.RAW", success=False, message="denied")
        ramp = _TestRamp(values=np.zeros(64), times=np.zeros(64))
        with pytest.raises(RuntimeError, match="Failed to write"):
            ramp.write("B:HS23T", slot=0, backend=fake_backend)

    def test_slot_drf(self):
        """Verify DRF uses correct byte offsets for different slots."""
        drf0 = _TestRamp._make_drf("B:HS23T", slot=0)
        assert "{0:256}" in drf0

        drf1 = _TestRamp._make_drf("B:HS23T", slot=1)
        assert "{256:256}" in drf1

        drf2 = _TestRamp._make_drf("B:HS23T", slot=2)
        assert "{512:256}" in drf2


class TestModifyContext:
    def test_modify_reads_and_returns_ramp(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        with _TestRamp.modify("B:HS23T", slot=0, backend=fake_backend) as ramp:
            assert isinstance(ramp, _TestRamp)
            assert ramp.values[0] == 100.0

    def test_modify_writes_on_change(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        with _TestRamp.modify("B:HS23T", slot=0, backend=fake_backend) as ramp:
            ramp.values[0] = 20.0

        assert len(fake_backend.writes) == 1
        drf, value = fake_backend.writes[0]
        assert "B:HS23T" in drf
        assert isinstance(value, bytes)

    def test_modify_no_write_on_no_change(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        with _TestRamp.modify("B:HS23T", slot=0, backend=fake_backend) as _ramp:
            pass

        assert len(fake_backend.writes) == 0

    def test_modify_no_write_on_exception(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        with pytest.raises(RuntimeError):
            with _TestRamp.modify("B:HS23T", slot=0, backend=fake_backend) as ramp:
                ramp.values[0] = 20.0
                raise RuntimeError("abort")

        assert len(fake_backend.writes) == 0

    def test_modify_write_failure_raises(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        fake_backend.set_write_result("B:HS23T.SETTING.RAW", success=False, message="denied")

        with pytest.raises(RuntimeError, match="Failed to write"):
            with _TestRamp.modify("B:HS23T", slot=0, backend=fake_backend) as ramp:
                ramp.values[0] = 20.0

    def test_modify_error_reading_raises(self, fake_backend):
        fake_backend.set_error("B:HS23T.SETTING.RAW@I", -1, "Device offline")
        with pytest.raises(DeviceError):
            with _TestRamp.modify("B:HS23T", slot=0, backend=fake_backend) as _ramp:
                pass

    def test_modify_sub_lsb_change_no_write(self, fake_backend):
        """Sub-LSB change quantizes to same raw bytes - no write."""
        _setup_devices(fake_backend, ["B:HS23T"])
        with _TestRamp.modify("B:HS23T", slot=0, backend=fake_backend) as ramp:
            ramp.values[0] += 0.4  # 100.0 → 100.4, rounds to raw 100
        assert len(fake_backend.writes) == 0

    def test_modify_nonzero_slot(self, fake_backend):
        """modify() at non-zero slot uses correct byte offset for read and write."""
        _setup_devices(fake_backend, ["B:HS23T"], slot=2)
        with _TestRamp.modify("B:HS23T", slot=2, backend=fake_backend) as ramp:
            assert ramp.slot == 2
            ramp.values[0] = 20.0
        assert len(fake_backend.writes) == 1
        drf, _ = fake_backend.writes[0]
        assert "B:HS23T" in drf
        assert "{512:256}" in drf  # slot 2


class TestTimeScaling:
    def test_booster_rate_100khz(self):
        assert BoosterHVRamp.update_rate_hz == 100_000
        assert BoosterHVRamp._tick_us() == 10.0

    def test_time_round_trip(self):
        """Times survive from_bytes -> to_bytes at 100 KHz."""
        data = _make_ramp_bytes([(0, 500)])  # 500 ticks
        ramp = BoosterHVRamp.from_bytes(data)
        assert ramp.times[0] == 5_000.0  # 500 * 10 us
        raw = ramp.to_bytes()
        _, t = struct.unpack_from("<hh", raw, 0)
        assert t == 500

    def test_custom_rate(self):
        """Subclass with different update rate scales times correctly."""

        class SlowRamp(Ramp):
            update_rate_hz = 1_000  # 1 KHz -> 1000 us/tick

            @classmethod
            def primary_transform(cls, raw):
                return raw.astype(np.float64)

            @classmethod
            def common_transform(cls, primary):
                return primary

            @classmethod
            def inverse_common_transform(cls, common):
                return common

            @classmethod
            def inverse_primary_transform(cls, primary):
                return primary

        data = _make_ramp_bytes([(0, 50)])  # 50 ticks
        ramp = SlowRamp.from_bytes(data)
        assert ramp.times[0] == 50_000.0  # 50 * 1000 us
        raw = ramp.to_bytes()
        _, t = struct.unpack_from("<hh", raw, 0)
        assert t == 50


class TestCustomSubclass:
    def test_custom_transforms(self):
        class TestRamp(Ramp):
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
        assert ramp.values[0] == 1.0

    def test_nonlinear_transforms(self):
        """Transform functions can be nonlinear."""

        class LogRamp(Ramp):
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
        assert ramp.values[0] == 10.0  # 100^2 / 1000 = 10
        # Round-trip
        rt = LogRamp.from_bytes(ramp.to_bytes())
        assert rt.values[0] == 10.0

    def test_missing_transforms_raises(self):
        """Using Ramp directly without implementing transforms raises."""
        with pytest.raises(NotImplementedError):
            Ramp.from_bytes(b"\x00" * 256)


# ─────────────────────────────────────────────────────────────────────────────
# Test data: distinct values per device to catch indexing bugs.
# _TestRamp uses identity transforms so (value, time_us) == raw == eng.
# ─────────────────────────────────────────────────────────────────────────────

_DEV_DATA = {
    "B:HS23T": (100, 50),
    "B:HS24T": (200, 75),
    "B:HS25T": (-300, 25),
}


def _setup_devices(fake_backend, devices=None, slot=0):
    """Load test ramp bytes into FakeBackend for given devices.

    Uses the event-specific DRF (@I) so FakeBackend stores under both the
    full key (*.RAW@I) and base key (*.RAW).  Production code reads with @I,
    so the full-key path is exercised.  A buffer large enough to cover the
    slot offset is stored so ranged reads slice correctly.
    """
    devices = devices or list(_DEV_DATA)
    slot_bytes = _TestRamp._slot_bytes()
    for dev in devices:
        val, time = _DEV_DATA[dev]
        ramp_bytes = _make_ramp_bytes([(val, time)])
        # Pad front so the ramp data sits at the correct slot offset
        buf = b"\x00" * (slot * slot_bytes) + ramp_bytes
        fake_backend.set_reading(f"{dev}.SETTING.RAW@I", buf, value_type=ValueType.RAW)


class TestRampDeviceSlot:
    def test_read_sets_device_and_slot(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        ramp = _TestRamp.read("B:HS23T", slot=0, backend=fake_backend)
        assert ramp.device == "B:HS23T"
        assert ramp.slot == 0

    def test_from_bytes_leaves_none(self):
        ramp = _TestRamp.from_bytes(b"\x00" * 256)
        assert ramp.device is None
        assert ramp.slot is None

    def test_write_uses_stored_defaults(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        ramp = _TestRamp.read("B:HS23T", slot=0, backend=fake_backend)
        ramp.write(backend=fake_backend)
        assert len(fake_backend.writes) == 1
        drf, _ = fake_backend.writes[0]
        assert "B:HS23T" in drf
        assert "{0:256}" in drf  # slot 0

    def test_write_raises_no_device(self):
        ramp = _TestRamp(values=np.zeros(64), times=np.zeros(64))
        with pytest.raises(ValueError, match="No device"):
            ramp.write()

    def test_write_raises_no_slot(self):
        ramp = _TestRamp(values=np.zeros(64), times=np.zeros(64))
        ramp.device = "B:HS23T"
        with pytest.raises(ValueError, match="No slot"):
            ramp.write()

    def test_write_allows_overrides(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        ramp = _TestRamp.read("B:HS23T", slot=0, backend=fake_backend)
        ramp.write(device="B:HS24T", slot=1, backend=fake_backend)
        drf, _ = fake_backend.writes[0]
        assert "B:HS24T" in drf
        assert "{256:256}" in drf  # slot 1

    def test_modify_sets_device_and_slot(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        with _TestRamp.modify("B:HS23T", slot=0, backend=fake_backend) as ramp:
            assert ramp.device == "B:HS23T"
            assert ramp.slot == 0

    def test_read_rejects_non_simple_drf(self, fake_backend):
        with pytest.raises(ValueError, match="bare device name"):
            _TestRamp.read("B:HS23T.SETTING", backend=fake_backend)

    def test_write_rejects_non_simple_drf(self, fake_backend):
        ramp = _TestRamp(values=np.zeros(64), times=np.zeros(64))
        with pytest.raises(ValueError, match="bare device name"):
            ramp.write(device="B:HS23T@p,1000", backend=fake_backend)

    def test_modify_rejects_non_simple_drf(self, fake_backend):
        with pytest.raises(ValueError, match="bare device name"):
            with _TestRamp.modify("B:HS23T{0:256}", backend=fake_backend):
                pass

    def test_accepts_qualifier_shorthand(self, fake_backend):
        """Qualifier-based names like B_HS23T are accepted (implicit property)."""
        _setup_devices(fake_backend, ["B:HS23T"])
        ramp = _TestRamp.read("B_HS23T", slot=0, backend=fake_backend)
        assert ramp.device == "B:HS23T"


class TestReadRamps:
    def test_batched_read(self, fake_backend):
        _setup_devices(fake_backend)
        ramps = read_ramps(_TestRamp, ["B:HS23T", "B:HS24T", "B:HS25T"], backend=fake_backend)

        assert len(ramps) == 3
        for ramp, (dev, (val, time)) in zip(ramps, _DEV_DATA.items()):
            assert ramp.device == dev
            assert ramp.slot == 0
            assert ramp.values[0] == val
            assert ramp.times[0] == time

    def test_error_raises(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        fake_backend.set_error("B:HS24T.SETTING.RAW@I", -1, "Device offline")

        with pytest.raises(DeviceError):
            read_ramps(_TestRamp, ["B:HS23T", "B:HS24T"], backend=fake_backend)

    def test_read_many_classmethod(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T", "B:HS24T"])
        ramps = _TestRamp.read_many(["B:HS23T", "B:HS24T"], backend=fake_backend)
        assert len(ramps) == 2
        assert ramps[0].device == "B:HS23T"
        assert ramps[0].slot == 0
        assert ramps[0].values[0] == 100.0
        assert ramps[1].device == "B:HS24T"
        assert ramps[1].values[0] == 200.0

    def test_empty_device_list_raises(self, fake_backend):
        with pytest.raises(ValueError, match="must not be empty"):
            read_ramps(_TestRamp, [], backend=fake_backend)

    def test_non_bytes_raises(self, fake_backend):
        fake_backend.set_reading("B:HS23T.SETTING.RAW", 42.0)
        _setup_devices(fake_backend, ["B:HS24T"])
        with pytest.raises(DeviceError):
            read_ramps(_TestRamp, ["B:HS23T", "B:HS24T"], backend=fake_backend)

    def test_rejects_drf_with_property(self, fake_backend):
        with pytest.raises(ValueError, match="bare device name"):
            read_ramps(_TestRamp, ["B:HS23T.SETTING"], backend=fake_backend)

    def test_rejects_drf_with_event(self, fake_backend):
        with pytest.raises(ValueError, match="bare device name"):
            read_ramps(_TestRamp, ["B:HS23T@p,1000"], backend=fake_backend)

    def test_rejects_drf_with_range(self, fake_backend):
        with pytest.raises(ValueError, match="bare device name"):
            read_ramps(_TestRamp, ["B:HS23T[0:10]"], backend=fake_backend)

    def test_read_from_nonzero_slot(self, fake_backend):
        """Batched read uses the correct slot offset in each DRF."""
        _setup_devices(fake_backend, slot=2)
        ramps = read_ramps(_TestRamp, list(_DEV_DATA), slot=2, backend=fake_backend)
        for ramp, dev in zip(ramps, _DEV_DATA):
            assert ramp.slot == 2
            val, _ = _DEV_DATA[dev]
            assert ramp.values[0] == val


class TestWriteRamps:
    def test_single_ramp(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        ramp = _TestRamp.read("B:HS23T", backend=fake_backend)
        results = write_ramps(ramp, backend=fake_backend)
        assert len(results) == 1
        assert results[0].success

    def test_list_of_ramps(self, fake_backend):
        _setup_devices(fake_backend)
        ramps = _TestRamp.read_many(list(_DEV_DATA), backend=fake_backend)
        results = write_ramps(ramps, backend=fake_backend)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_ramp_group(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        results = write_ramps(group, backend=fake_backend)
        assert len(results) == 3

    def test_mixed_list(self, fake_backend):
        _setup_devices(fake_backend)
        ramp = _TestRamp.read("B:HS23T", backend=fake_backend)
        group = _TestRampGroup.read(["B:HS24T", "B:HS25T"], backend=fake_backend)
        results = write_ramps([ramp, group], backend=fake_backend)
        assert len(results) == 3

    def test_slot_override(self, fake_backend):
        _setup_devices(fake_backend, ["B:HS23T"])
        ramp = _TestRamp.read("B:HS23T", backend=fake_backend)
        write_ramps(ramp, slot=2, backend=fake_backend)
        drf, _ = fake_backend.writes[-1]
        assert "{512:256}" in drf  # slot 2

    def test_writes_all_devices(self, fake_backend):
        """write_ramps produces one write per device."""
        _setup_devices(fake_backend)
        ramps = _TestRamp.read_many(list(_DEV_DATA), backend=fake_backend)
        initial_writes = len(fake_backend.writes)
        write_ramps(ramps, backend=fake_backend)
        assert len(fake_backend.writes) - initial_writes == 3

    def test_correct_bytes_go_to_correct_device(self, fake_backend):
        """Each device's ramp bytes are written to the correct DRF."""
        _setup_devices(fake_backend)
        ramps = _TestRamp.read_many(list(_DEV_DATA), backend=fake_backend)
        write_ramps(ramps, backend=fake_backend)

        writes = fake_backend.writes[-3:]
        for (drf, written_bytes), dev in zip(writes, _DEV_DATA):
            assert dev in drf, f"Expected {dev} in DRF {drf}"
            val, time = _DEV_DATA[dev]
            expected = _make_ramp_bytes([(val, time)])
            assert written_bytes == expected, f"Wrong bytes for {dev}"

    def test_slot_override_all_devices(self, fake_backend):
        """Slot override applies to every device in the batch."""
        _setup_devices(fake_backend)
        ramps = _TestRamp.read_many(list(_DEV_DATA), backend=fake_backend)
        write_ramps(ramps, slot=3, backend=fake_backend)
        for drf, _ in fake_backend.writes[-3:]:
            assert "{768:256}" in drf  # slot 3 = 3*256

    def test_mixed_list_correct_redirection(self, fake_backend):
        """Mixed list[Ramp | RampGroup] writes each device's data to its own DRF."""
        _setup_devices(fake_backend)
        ramp = _TestRamp.read("B:HS23T", backend=fake_backend)
        group = _TestRampGroup.read(["B:HS24T", "B:HS25T"], backend=fake_backend)
        write_ramps([ramp, group], backend=fake_backend)

        writes = fake_backend.writes[-3:]
        devs = ["B:HS23T", "B:HS24T", "B:HS25T"]
        for (drf, written_bytes), dev in zip(writes, devs):
            assert dev in drf
            val, time = _DEV_DATA[dev]
            assert written_bytes == _make_ramp_bytes([(val, time)])

    def test_missing_device_raises(self, fake_backend):
        ramp = _TestRamp(values=np.zeros(64), times=np.zeros(64))
        ramp.slot = 0
        with pytest.raises(ValueError, match="no device"):
            write_ramps(ramp, backend=fake_backend)

    def test_missing_slot_raises(self, fake_backend):
        ramp = _TestRamp(values=np.zeros(64), times=np.zeros(64))
        ramp.device = "B:HS23T"
        with pytest.raises(ValueError, match="No slot"):
            write_ramps(ramp, backend=fake_backend)


class TestRampGroup:
    def test_shapes(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        assert group.values.shape == (64, 3)
        assert group.times.shape == (64, 3)

    def test_device_indexing_returns_view(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        ramp = group["B:HS23T"]
        assert ramp.device == "B:HS23T"
        assert ramp.slot == 0
        assert ramp.values[0] == 100.0

    def test_view_mutation_propagates_to_group(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        ramp = group["B:HS23T"]
        ramp.values[0] = 99.0
        assert group.values[0, 0] == 99.0

    def test_group_mutation_propagates_to_view(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        ramp = group["B:HS24T"]
        group.values[0, 1] = 42.0
        assert ramp.values[0] == 42.0

    def test_2d_broadcast_all(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        old_vals = group.values.copy()
        group.values += 0.5
        np.testing.assert_allclose(group.values, old_vals + 0.5)

    def test_2d_row_broadcast(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        old_row = group.values[5].copy()
        group.values[5] += 0.5
        np.testing.assert_allclose(group.values[5], old_row + 0.5)

    def test_len_iter_contains(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        assert len(group) == 3
        assert list(group) == list(_DEV_DATA)
        assert "B:HS23T" in group
        assert "B:NOPE" not in group

    def test_write_correct_bytes_per_device(self, fake_backend):
        """RampGroup.write() sends column i's bytes to device i's DRF."""
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        group.write(backend=fake_backend)

        writes = fake_backend.writes[-3:]
        for (drf, written_bytes), dev in zip(writes, _DEV_DATA):
            assert dev in drf
            val, time = _DEV_DATA[dev]
            assert written_bytes == _make_ramp_bytes([(val, time)])

    def test_write_device_override(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        new_devs = ["B:X1", "B:X2", "B:X3"]
        group.write(devices=new_devs, backend=fake_backend)
        written_drfs = [drf for drf, _ in fake_backend.writes[-3:]]
        for drf, dev in zip(written_drfs, new_devs):
            assert dev in drf

    def test_write_slot_override(self, fake_backend):
        """RampGroup.write(slot=N) writes all devices to slot N."""
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        group.write(slot=4, backend=fake_backend)
        for drf, _ in fake_backend.writes[-3:]:
            assert "{1024:256}" in drf  # slot 4 = 4*256

    def test_write_shape_mismatch_raises(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        with pytest.raises(ValueError, match="Expected 3 devices"):
            group.write(devices=["B:X1", "B:X2"], backend=fake_backend)

    def test_getitem_unknown_device_raises(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        with pytest.raises(KeyError):
            group["B:NONEXISTENT"]

    def test_constructor_validates_shape(self):
        with pytest.raises(ValueError, match="Expected values shape"):
            _TestRampGroup(
                devices=["A", "B"],
                values=np.zeros((64, 3)),  # wrong: 3 columns for 2 devices
                times=np.zeros((64, 2)),
            )

    def test_constructor_rejects_duplicate_devices(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _TestRampGroup(
                devices=["A", "A"],
                values=np.zeros((64, 2)),
                times=np.zeros((64, 2)),
            )

    def test_write_empty_device_list(self, fake_backend):
        _setup_devices(fake_backend)
        group = _TestRampGroup.read(list(_DEV_DATA), backend=fake_backend)
        with pytest.raises(ValueError, match="Expected 3 devices, got 0"):
            group.write(devices=[], backend=fake_backend)


class TestRampGroupModify:
    def test_reads_on_enter_writes_on_exit(self, fake_backend):
        _setup_devices(fake_backend)
        with _TestRampGroup.modify(list(_DEV_DATA), backend=fake_backend) as group:
            assert group.values.shape == (64, 3)
            group.values[0, 0] = 20.0  # change B:HS23T

        # Only 1 device changed, only 1 write
        assert len(fake_backend.writes) == 1
        drf, written_bytes = fake_backend.writes[0]
        assert "B:HS23T" in drf
        v, _ = struct.unpack_from("<hh", written_bytes, 0)
        assert v == 20  # identity transform: eng == raw

    def test_modify_writes_correct_bytes_to_correct_device(self, fake_backend):
        """When two devices change, each gets its own modified bytes."""
        _setup_devices(fake_backend)
        with _TestRampGroup.modify(list(_DEV_DATA), backend=fake_backend) as group:
            group.values[0, 0] = 20.0  # B:HS23T
            group.values[0, 2] = -20.0  # B:HS25T

        assert len(fake_backend.writes) == 2
        write_map = {drf: data for drf, data in fake_backend.writes}

        hs23_drf = next(d for d in write_map if "B:HS23T" in d)
        v, _ = struct.unpack_from("<hh", write_map[hs23_drf], 0)
        assert v == 20

        hs25_drf = next(d for d in write_map if "B:HS25T" in d)
        v, _ = struct.unpack_from("<hh", write_map[hs25_drf], 0)
        assert v == -20

    def test_no_write_if_unchanged(self, fake_backend):
        _setup_devices(fake_backend)
        with _TestRampGroup.modify(list(_DEV_DATA), backend=fake_backend) as _group:
            pass
        assert len(fake_backend.writes) == 0

    def test_no_write_on_exception(self, fake_backend):
        _setup_devices(fake_backend)
        with pytest.raises(RuntimeError):
            with _TestRampGroup.modify(list(_DEV_DATA), backend=fake_backend) as group:
                group.values[0, 0] = 20.0
                raise RuntimeError("abort")
        assert len(fake_backend.writes) == 0

    def test_partial_failure_raises(self, fake_backend):
        _setup_devices(fake_backend)
        fake_backend.set_write_result("B:HS24T.SETTING.RAW", success=False, message="write denied")
        with pytest.raises(RuntimeError, match="Partial write failure"):
            with _TestRampGroup.modify(list(_DEV_DATA), backend=fake_backend) as group:
                group.values[0] += 1.0  # change all devices

    def test_modify_nonzero_slot(self, fake_backend):
        """modify() at slot=2 reads and writes using the correct byte offset."""
        _setup_devices(fake_backend, slot=2)
        with _TestRampGroup.modify(list(_DEV_DATA), slot=2, backend=fake_backend) as group:
            group.values[0, 0] = 20.0  # change B:HS23T

        assert len(fake_backend.writes) == 1
        drf, _ = fake_backend.writes[0]
        assert "B:HS23T" in drf
        assert "{512:256}" in drf  # slot 2 = 2*256

    def test_modify_skips_unchanged_device(self, fake_backend):
        """B:HS24T (column 1) is not modified, so no write for it."""
        _setup_devices(fake_backend)
        with _TestRampGroup.modify(list(_DEV_DATA), backend=fake_backend) as group:
            group.values[0, 0] = 20.0  # B:HS23T changed
            group.values[0, 2] = -20.0  # B:HS25T changed
            # B:HS24T (column 1) untouched
        assert len(fake_backend.writes) == 2
        written_devs = {drf for drf, _ in fake_backend.writes}
        assert not any("B:HS24T" in d for d in written_devs)


class TestBoosterHVRampGroup:
    def test_read_returns_correct_type(self, fake_backend):
        for dev in ["B:HS23T", "B:HS24T"]:
            fake_backend.set_reading(f"{dev}.SETTING.RAW@I", b"\x00" * 256, value_type=ValueType.RAW)
        group = BoosterHVRampGroup.read(["B:HS23T", "B:HS24T"], backend=fake_backend)
        assert isinstance(group, BoosterHVRampGroup)
        assert isinstance(group["B:HS23T"], BoosterHVRamp)


@pytest.fixture
def fake_backend():
    from pacsys.testing import FakeBackend

    return FakeBackend()
