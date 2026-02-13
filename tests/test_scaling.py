"""Tests for pacsys.scaling - client-side scaling transforms."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from pacsys.scaling import (
    Scaler,
    ScalingError,
    _check_signed,
    _check_unsigned,
    _common_scale,
    _common_unscale,
    _int_to_float,
    _primary_scale,
    _primary_unscale,
    _sign_extend,
)


# ---- Helpers -----------------------------------------------------------------


def _float_bits(f: float) -> int:
    """Convert float to its IEEE 754 32-bit int representation."""
    return struct.unpack(">I", struct.pack(">f", f))[0]


def _approx(a, b, rel=1e-4, abs_tol=1e-6):
    """Check approximate equality."""
    return pytest.approx(b, rel=rel, abs=abs_tol) == a


# ---- Low-level helpers -------------------------------------------------------


class TestHelpers:
    def test_sign_extend(self):
        assert _sign_extend(0xFF, 1) == -1
        assert _sign_extend(0x7F, 1) == 127
        assert _sign_extend(0xFFFF, 2) == -1
        assert _sign_extend(0x7FFF, 2) == 32767
        assert _sign_extend(0xFFFFFFFF, 4) == -1
        assert _sign_extend(0x7FFFFFFF, 4) == 2147483647

    def test_sign_extend_invalid(self):
        with pytest.raises(ScalingError, match="Invalid input_len"):
            _sign_extend(0, 3)

    def test_check_signed(self):
        assert _check_signed(127, 1) == 127
        assert _check_signed(-128, 1) == -128
        with pytest.raises(ScalingError, match="Overflow"):
            _check_signed(128, 1)
        with pytest.raises(ScalingError, match="Overflow"):
            _check_signed(32768, 2)

    def test_check_unsigned(self):
        assert _check_unsigned(255, 1) == 255
        with pytest.raises(ScalingError, match="Overflow"):
            _check_unsigned(256, 1)
        with pytest.raises(ScalingError, match="Overflow"):
            _check_unsigned(-1, 2)


# ---- Primary scale (raw -> primary) -----------------------------------------


class TestPrimaryScale:
    """Test each primary transform (forward: raw -> primary)."""

    def test_p0_dac_3200(self):
        assert _primary_scale(3200, 0, 2) == pytest.approx(1.0)
        assert _primary_scale(-3200, 0, 2) == pytest.approx(-1.0)

    def test_p2_dac_3276(self):
        assert _primary_scale(3277, 2, 2) == pytest.approx(1.0006, rel=1e-3)

    def test_p4_dac_6553(self):
        assert _primary_scale(6554, 4, 2) == pytest.approx(1.0001, rel=1e-3)

    def test_p6_dac_13107(self):
        assert _primary_scale(13107, 6, 2) == pytest.approx(1.0, rel=1e-3)

    def test_p8_offset(self):
        assert _primary_scale(0, 8, 2) == 32768.0
        assert _primary_scale(1000, 8, 2) == 33768.0

    def test_p10_identity(self):
        assert _primary_scale(42, 10, 2) == 42.0
        assert _primary_scale(-100, 10, 2) == -100.0

    def test_p12_divide_320(self):
        assert _primary_scale(320, 12, 2) == pytest.approx(1.0)

    def test_p14_bcd(self):
        # mantissa=100, exponent=2 -> 100 * 10^2 = 10000
        raw = 100 | (2 << 10)
        assert _primary_scale(raw, 14, 2) == pytest.approx(10000.0)

    def test_p16_ieee_float(self):
        raw = _float_bits(3.14)
        assert _primary_scale(raw, 16, 4) == pytest.approx(3.14, rel=1e-5)

    def test_p18_multiply(self):
        assert _primary_scale(1000, 18, 2) == pytest.approx(1.0406, rel=1e-4)

    def test_p20_unsigned_correction(self):
        assert _primary_scale(100, 20, 2) == 100.0
        # Negative 16-bit value -> add 65536
        assert _primary_scale(-1, 20, 2) == 65535.0
        # Negative 8-bit -> add 256
        assert _primary_scale(-1, 20, 1) == 255.0

    def test_p22_dec_float(self):
        # Word-swap then float/4. Test with known bits.
        ieee_val = 4.0  # so DEC float = IEEE * 4 = 16.0 -> float/4 = 4.0
        bits = _float_bits(ieee_val)
        # Word-swap the bits for input
        swapped = ((bits & 0xFFFF) << 16) | ((bits >> 16) & 0xFFFF)
        assert _primary_scale(swapped, 22, 4) == pytest.approx(1.0, rel=1e-5)

    def test_p24_ieee_68000(self):
        # Word-swap then reinterpret as float
        val = 2.5
        bits = _float_bits(val)
        swapped = ((bits & 0xFFFF) << 16) | ((bits >> 16) & 0xFFFF)
        assert _primary_scale(swapped, 24, 4) == pytest.approx(2.5, rel=1e-5)

    def test_p26_pbar(self):
        # ((x >> 8) & 0xFF) / 82.1865 - 0.310269935
        raw = 0x8000  # high byte = 0x80 = 128
        result = 128 / 82.1865 - 0.310269935
        assert _primary_scale(raw, 26, 2) == pytest.approx(result, rel=1e-4)

    def test_p28_word_swap(self):
        # Word-swap as signed 32-bit
        raw = 0x00010002  # -> swapped: 0x00020001
        assert _primary_scale(raw, 28, 4) == pytest.approx(0x00020001, rel=1e-9)

    def test_p30_low_byte_signed(self):
        assert _primary_scale(0x0005, 30, 2) == 5.0
        assert _primary_scale(0x00FF, 30, 2) == -1.0

    def test_p32_high_byte_signed(self):
        assert _primary_scale(0x0500, 32, 2) == 5.0
        assert _primary_scale(0xFF00, 32, 2) == -1.0

    def test_p34_low_byte_unsigned(self):
        assert _primary_scale(0x00FF, 34, 2) == 255.0

    def test_p36_high_byte_unsigned(self):
        assert _primary_scale(0xFF00, 36, 2) == 255.0

    def test_p38_pbar_lo(self):
        raw = 0x80  # 128
        result = 128 / 82.1865 - 0.310269935
        assert _primary_scale(raw, 38, 1) == pytest.approx(result, rel=1e-4)

    def test_p40_divide_256(self):
        assert _primary_scale(256, 40, 2) == pytest.approx(1.0)

    def test_p42_camac_unsigned(self):
        assert _primary_scale(6554, 42, 2) == pytest.approx(1.0001, rel=1e-3)

    def test_p44_bcd7(self):
        # 1234567 in BCD: 0x01234567
        raw = 0x01234567
        assert _primary_scale(raw, 44, 4) == pytest.approx(1234567.0)

    def test_p46_unsigned_i4(self):
        assert _primary_scale(100, 46, 4) == 100.0
        assert _primary_scale(-1, 46, 4) == 4294967295.0

    def test_p48_ieee_div_036(self):
        raw = _float_bits(0.036)
        assert _primary_scale(raw, 48, 4) == pytest.approx(1.0, rel=1e-4)

    def test_p50_ieee_clamped(self):
        raw = _float_bits(5.0)
        assert _primary_scale(raw, 50, 4) == pytest.approx(5.0)
        raw = _float_bits(20.0)
        assert _primary_scale(raw, 50, 4) == 10.235

    def test_p52_byte_swap_word(self):
        raw = 0x0102  # 2 bytes: swap -> 0x0201 = 513
        result = _primary_scale(raw, 52, 2)
        assert result == pytest.approx(513.0)

    def test_p52_byte_swap_long(self):
        raw = 0x01020304
        # Reversed: 0x04030201
        result = _primary_scale(raw, 52, 4)
        expected = struct.unpack(">i", bytes([4, 3, 2, 1]))[0]
        assert result == pytest.approx(float(expected))

    def test_p54_plc(self):
        assert _primary_scale(0, 54, 2) == pytest.approx(4.0)

    def test_p54_invalid(self):
        with pytest.raises(ScalingError):
            _primary_scale(0x8000, 54, 2)

    def test_p56_unsigned_word(self):
        # (unsigned - 32768) / 3276.8
        assert _primary_scale(32768, 56, 2) == pytest.approx(0.0)

    def test_p58_unsigned_div256(self):
        assert _primary_scale(256, 58, 4) == pytest.approx(1.0)

    def test_p60_ieee_mul500(self):
        raw = _float_bits(0.002)
        assert _primary_scale(raw, 60, 4) == pytest.approx(1.0, rel=1e-4)

    def test_p62_divide_6400(self):
        assert _primary_scale(6400, 62, 2) == pytest.approx(1.0)

    def test_p64_scaled_to_1(self):
        # 1-byte: 127/128 ≈ 0.992 (128 sign-extends to -128)
        assert _primary_scale(64, 64, 1) == pytest.approx(0.5)
        # 2-byte: 32767/32768 ≈ 1.0 (32768 sign-extends to -32768)
        assert _primary_scale(16384, 64, 2) == pytest.approx(0.5)

    def test_p66_word_positive(self):
        assert _primary_scale(3200, 66, 2) == pytest.approx(1.0)

    def test_p66_invalid(self):
        with pytest.raises(ScalingError):
            _primary_scale(0, 66, 2)

    def test_p68_raises(self):
        with pytest.raises(ScalingError, match="Alternate"):
            _primary_scale(0, 68, 2)

    def test_p70_divide_1000(self):
        assert _primary_scale(1000, 70, 2) == pytest.approx(1.0)

    def test_p72_unsigned_word_3200(self):
        # (unsigned - 32768) / 3200
        assert _primary_scale(32768, 72, 2) == pytest.approx(0.0)

    def test_p74_multiply(self):
        assert _primary_scale(1000, 74, 2) == pytest.approx(0.64088, rel=1e-4)

    def test_p76_word_swap_unsigned32(self):
        raw = 0x00010000  # word-swap -> 0x00000001 = 1
        assert _primary_scale(raw, 76, 4) == pytest.approx(1.0)

    def test_p78_ieee_clamped_0_5(self):
        raw = _float_bits(3.0)
        assert _primary_scale(raw, 78, 4) == pytest.approx(3.0)
        raw = _float_bits(6.0)
        assert _primary_scale(raw, 78, 4) == 5.0

    def test_p80_ieee_clamped_0_10(self):
        raw = _float_bits(7.0)
        assert _primary_scale(raw, 80, 4) == pytest.approx(7.0)
        raw = _float_bits(12.0)
        assert _primary_scale(raw, 80, 4) == 10.0

    def test_p82_12bit(self):
        assert _primary_scale(4095, 82, 2) == pytest.approx(10.0, rel=1e-3)

    def test_p82_invalid(self):
        with pytest.raises(ScalingError):
            _primary_scale(0xF000, 82, 2)

    def test_p84_byte_reverse_ieee(self):
        val = 1.5
        bits = _float_bits(val)
        # Byte-reverse for input
        b = bits.to_bytes(4, "big")
        reversed_bits = int.from_bytes(b[::-1], "big")
        assert _primary_scale(reversed_bits, 84, 4) == pytest.approx(1.5, rel=1e-5)

    def test_invalid_p_index(self):
        with pytest.raises(ScalingError, match="not found"):
            _primary_scale(0, 99, 2)


# ---- Primary unscale (primary -> raw) ----------------------------------------


class TestPrimaryUnscale:
    """Test primary inverse transforms."""

    @pytest.mark.parametrize(
        "p_index",
        [0, 2, 4, 6, 8, 10, 12, 70],
    )
    def test_roundtrip_simple(self, p_index):
        """Round-trip for simple division/multiply/offset transforms."""
        raw = 1000
        primary = _primary_scale(raw, p_index, 2)
        recovered = _primary_unscale(primary, p_index, 2)
        assert recovered == raw

    def test_roundtrip_p18(self):
        """p18 uses different fwd/rev constants (0.0010406 vs 0.001040625), ±1 is OK."""
        raw = 1000
        primary = _primary_scale(raw, 18, 2)
        recovered = _primary_unscale(primary, 18, 2)
        assert abs(recovered - raw) <= 1

    def test_roundtrip_p40(self):
        """p40 unscale checks overflow against size=1, so use small value."""
        raw = 25  # 25/256 < 1.0 -> 25
        primary = _primary_scale(raw, 40, 2)
        recovered = _primary_unscale(primary, 40, 2)
        assert recovered == raw

    def test_roundtrip_p62(self):
        raw = 1000
        primary = _primary_scale(raw, 62, 2)
        recovered = _primary_unscale(primary, 62, 2)
        assert recovered == raw

    def test_roundtrip_p16_ieee(self):
        val = 3.14
        bits = _float_bits(val)
        primary = _primary_scale(bits, 16, 4)
        raw = _primary_unscale(primary, 16, 4)
        assert _int_to_float(raw) == pytest.approx(val, rel=1e-5)

    def test_roundtrip_p64(self):
        raw = 16384  # 0.5 in ±1.0 range for 16-bit
        primary = _primary_scale(raw, 64, 2)
        recovered = _primary_unscale(primary, 64, 2)
        assert recovered == raw

    def test_roundtrip_p52_word(self):
        raw = 0x0102
        primary = _primary_scale(raw, 52, 2)
        recovered = _primary_unscale(primary, 52, 2)
        assert recovered == raw

    def test_roundtrip_p44_bcd7(self):
        raw = 0x01234567
        primary = _primary_scale(raw, 44, 4)
        recovered = _primary_unscale(primary, 44, 4)
        assert recovered == raw

    def test_p14_roundtrip(self):
        raw = 100 | (2 << 10)  # mantissa=100, exp=2 -> 10000
        primary = _primary_scale(raw, 14, 2)
        assert primary == pytest.approx(10000.0)
        recovered = _primary_unscale(primary, 14, 2)
        # The BCD encode adds 0xC000 bits, so check mantissa and exponent
        assert _primary_scale(recovered, 14, 2) == pytest.approx(primary, rel=1e-4)


# ---- Primary round-trip (for all transforms that support it) ----------------


class TestPrimaryRoundTrip:
    """Verify unscale(scale(x)) ≈ x for primary transforms."""

    @pytest.mark.parametrize(
        "p_index,raw,input_len",
        [
            (0, 1000, 2),
            (2, 1000, 2),
            (4, 1000, 2),
            (6, 1000, 2),
            (8, 1000, 2),
            (10, 42, 2),
            (12, 100, 2),
            (16, _float_bits(2.5), 4),
            (30, 50, 2),
            (34, 100, 2),
            (40, 25, 2),
            (50, _float_bits(5.0), 4),
            (56, 32768, 2),
            (60, _float_bits(0.01), 4),
            (64, 16000, 2),
            (70, 1000, 2),
            (78, _float_bits(3.0), 4),
            (80, _float_bits(7.0), 4),
            (82, 2000, 2),
        ],
    )
    def test_roundtrip(self, p_index, raw, input_len):
        primary = _primary_scale(raw, p_index, input_len)
        recovered = _primary_unscale(primary, p_index, input_len)
        # For IEEE float transforms, compare via float conversion
        if p_index in (16, 50, 60, 78, 80):
            assert _int_to_float(recovered) == pytest.approx(_int_to_float(raw), rel=1e-5)
        else:
            assert recovered == raw

    def test_roundtrip_p18_approx(self):
        """p18 has mismatched fwd/rev constants, allow ±1."""
        primary = _primary_scale(1000, 18, 2)
        recovered = _primary_unscale(primary, 18, 2)
        assert abs(recovered - 1000) <= 1


# ---- Common scale (primary -> common) ----------------------------------------


class TestCommonScale:
    """Test common transforms (forward: primary -> common)."""

    def test_c0_identity(self):
        assert _common_scale(42.0, 0, ()) == 42.0

    def test_c2_linear(self):
        c = (100.0, 10.0, 5.0)
        # (100*x/10) + 5 = 10x + 5
        assert _common_scale(3.0, 2, c) == pytest.approx(35.0)

    def test_c2_div_by_zero(self):
        with pytest.raises(ScalingError, match="zero"):
            _common_scale(1.0, 2, (1.0, 0.0, 0.0))

    def test_c4_offset_divide(self):
        c = (10.0, 5.0)
        # (x - 10) / 5
        assert _common_scale(20.0, 4, c) == pytest.approx(2.0)

    def test_c6_ratio(self):
        c = (3.0, 2.0)
        assert _common_scale(4.0, 6, c) == pytest.approx(6.0)

    def test_c8_rational(self):
        c = (1.0, 0.0, 1.0, 2.0)
        # C4 + C1*X/(C3+C2*X) = 2 + x/(1+0) = 2+x
        assert _common_scale(3.0, 8, c) == pytest.approx(5.0)

    def test_c10_reciprocal(self):
        c = (1.0, 10.0, 5.0)
        # 5 + 10/(1*x) = 5 + 10/x
        assert _common_scale(2.0, 10, c) == pytest.approx(10.0)

    def test_c12_polynomial(self):
        # C1*x^4 + C2*x^3 + C3*x^2 + C4*x + C5
        c = (0.0, 0.0, 0.0, 2.0, 1.0)  # 2x + 1
        assert _common_scale(3.0, 12, c) == pytest.approx(7.0)

    def test_c14_exp_poly(self):
        c = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # exp(0) - 0 = 1
        assert _common_scale(0.0, 14, c) == pytest.approx(1.0)

    def test_c16_double_exp(self):
        c = (1.0, 1.0, 1.0, 1.0)
        # C2*exp(-x/C1) + C4*exp(-x/C3) = exp(-x) + exp(-x) = 2*exp(-x)
        assert _common_scale(0.0, 16, c) == pytest.approx(2.0)

    def test_c18_double_exp_offsets(self):
        c = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        # C3*exp(C2*(x+C1)) + C6*exp(C5*(x+C4)) = exp(0) + exp(0) = 2
        assert _common_scale(0.0, 18, c) == pytest.approx(2.0)

    def test_c20_log_rational(self):
        c = (0.0, 1.0, 0.0)
        # log10(x)/(0+1)^2 + 0 = log10(x)
        assert _common_scale(100.0, 20, c) == pytest.approx(2.0)

    def test_c22_power_of_10(self):
        c = (1.0, 1.0)
        # 1.0 * 10^(x/1) = 10^x
        assert _common_scale(2.0, 22, c) == pytest.approx(100.0)

    def test_c24_piecewise(self):
        c = (5.0, 2.0, 1.0, 0.0, 0.1, 0.0)
        # x < 5: 2*(1*x+0) = 2x
        assert _common_scale(3.0, 24, c) == pytest.approx(6.0)
        # x >= 5: 2*exp(0.1*x+0)
        assert _common_scale(10.0, 24, c) == pytest.approx(2 * math.exp(1.0))

    def test_c26_5th_degree(self):
        c = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # x
        assert _common_scale(5.0, 26, c) == pytest.approx(5.0)

    def test_c28_rational_offset(self):
        c = (0.0, 1.0, 10.0, 5.0)
        # 10/(1+0) + 5 = 15
        assert _common_scale(1.0, 28, c) == pytest.approx(15.0)

    def test_c30_piecewise_cubic(self):
        c = (2.0, 0.0, 0.0, 1.0, 0.0, 99.0)
        # x < 2: C6 = 99
        assert _common_scale(1.0, 30, c) == pytest.approx(99.0)
        # x >= 2: C5 + C4*x = x
        assert _common_scale(3.0, 30, c) == pytest.approx(3.0)

    def test_c32_natural_log(self):
        c = (1.0, 1.0, 0.0, 0.0)
        # 1*ln(1*x+0) + 0 = ln(x)
        assert _common_scale(math.e, 32, c) == pytest.approx(1.0)

    def test_c34_rational_linear(self):
        c = (1.0, 0.0, 0.0, 1.0)
        # (0+x)/(1+0) = x
        assert _common_scale(7.0, 34, c) == pytest.approx(7.0)

    def test_c36_sqrt(self):
        c = (0.0, 1.0, 0.0)
        # sqrt(x+0) + 0
        assert _common_scale(9.0, 36, c) == pytest.approx(3.0)

    def test_c38_vapor_pressure_below(self):
        c = (1.0, 1.0, 0.0, 0.0, 0.0, 100.0)
        # x <= C6=100 -> 760000
        assert _common_scale(50.0, 38, c) == pytest.approx(760000.0)

    def test_c40_linear(self):
        c = (10.0, 2.0, 1.0, 0.0, 0.0, 0.0)
        assert _common_scale(4.0, 40, c) == pytest.approx(21.0)

    def test_c42_piecewise(self):
        c = (5.0, 1.0, 0.0, 0.0, 0.1, 0.0)
        # x < 5: x^2
        assert _common_scale(3.0, 42, c) == pytest.approx(9.0)

    def test_c44_double_exp(self):
        c = (5.0, 1.0, 0.0, 2.0, 0.0)
        # x < 5: 1*exp(0) = 1
        assert _common_scale(3.0, 44, c) == pytest.approx(1.0)

    def test_c46_gaussian_exp(self):
        c = (5.0, 1.0, 0.0, 0.0, 2.0, 0.0)
        # x < 5: 1*exp(0) = 1
        assert _common_scale(3.0, 46, c) == pytest.approx(1.0)

    def test_c50_arccosine(self):
        c = (1.0, 1.0)
        # acos(x/1) = acos(x)
        assert _common_scale(0.0, 50, c) == pytest.approx(math.pi / 2)

    def test_c52_piecewise_exp(self):
        c = (5.0, 0.0, 0.0, 0.0, 0.0)
        # x < 5: exp(0) = 1
        assert _common_scale(3.0, 52, c) == pytest.approx(1.0)

    def test_c54_piecewise_exp_quad(self):
        c = (5.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # x < 5: exp(0) = 1
        assert _common_scale(3.0, 54, c) == pytest.approx(1.0)

    def test_c56_raises(self):
        with pytest.raises(ScalingError, match="interpolation table"):
            _common_scale(1.0, 56, (1.0, 0.0, 10.0))

    def test_c58_raises(self):
        with pytest.raises(ScalingError, match="interpolation table"):
            _common_scale(1.0, 58, (1.0, 0.0, 10.0))

    def test_c60_raises(self):
        with pytest.raises(ScalingError, match="Alternate"):
            _common_scale(1.0, 60, (0.0, 0.0, 0.0))

    def test_c62_log_power10(self):
        c = (1.0, 1.0, 0.0)
        # 1*(0 + 10^(x/1)) = 10^x
        assert _common_scale(2.0, 62, c) == pytest.approx(100.0)

    def test_c64_n2_vpt(self):
        # N2 vapor pressure thermo, x=0 -> 77.4
        assert _common_scale(0.0, 64, (0.0,)) == pytest.approx(77.4)

    def test_c64_he_vpt(self):
        assert _common_scale(0.0, 64, (1.0,)) == pytest.approx(4.2)

    def test_c66_exp_base2(self):
        c = (1.0, 1.0, 0.0, 0.0)
        # 1*2^(1*(x+0)) + 0 = 2^x
        assert _common_scale(3.0, 66, c) == pytest.approx(8.0)

    def test_c68_complex_log(self):
        c = (1.0, 1.0, 0.0, 0.0, 1.0, 1.0)
        # C6*(C2*ln(C1*x+C4)+C3*x)^C5 = 1*(1*ln(x))^1 = ln(x)
        assert _common_scale(math.e, 68, c) == pytest.approx(1.0)

    def test_c70_triple_exp(self):
        c = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        # All exp(-0/1) = 1, so 1+1+1+4 = 7
        assert _common_scale(0.0, 70, c) == pytest.approx(7.0)

    def test_c74_rational_quad(self):
        c = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        # (1+0+0)/(1+0+0) = 1
        assert _common_scale(5.0, 74, c) == pytest.approx(1.0)

    def test_c76_piecewise_power(self):
        c = (5.0, 1.0, 2.0, 1.0, 0.1, 0.0)
        # x < 5: 1*x^2 = x^2
        assert _common_scale(3.0, 76, c) == pytest.approx(9.0)

    def test_c78_power10_scale(self):
        c = (1.0, 1.0, 0.0, 0.0)
        # 1*10^(x+0) + 0 = 10^x
        assert _common_scale(2.0, 78, c) == pytest.approx(100.0)

    def test_c80_identity(self):
        assert _common_scale(42.0, 80, ()) == 42.0

    def test_c82_log10_scale(self):
        c = (1.0, 1.0, 0.0, 0.0)
        # 1*log10(1*x+0) + 0 = log10(x)
        assert _common_scale(100.0, 82, c) == pytest.approx(2.0)

    def test_c84_raises(self):
        with pytest.raises(ScalingError, match="Alternate"):
            _common_scale(1.0, 84, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_c86_piecewise_lin_log_exp(self):
        c = (2.0, 8.0, -1.0, 10.0, -0.1, 1.0)
        # x < 2: -1*x + 10 = 10-x
        assert _common_scale(1.0, 86, c) == pytest.approx(9.0)
        # x > 8: exp(-0.1*x + 1)
        assert _common_scale(10.0, 86, c) == pytest.approx(math.exp(0.0))

    def test_c88_pade(self):
        c = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # (1+0+0)/(1+0+0+0) = 1
        assert _common_scale(5.0, 88, c) == pytest.approx(1.0)

    def test_c90_raises(self):
        with pytest.raises(ScalingError, match="multifunction"):
            _common_scale(1.0, 90, (1.0, 0.0, 10.0))

    def test_c201_raises(self):
        with pytest.raises(ScalingError, match="sub-transform"):
            _common_scale(1.0, 201, tuple(range(8)))

    def test_invalid_c_index(self):
        with pytest.raises(ScalingError, match="not found"):
            _common_scale(1.0, 999, ())


# ---- Common unscale (common -> primary) round-trips --------------------------


class TestCommonRoundTrip:
    """Test common transform round-trips: unscale(scale(x)) ≈ x."""

    @pytest.mark.parametrize(
        "c_index,constants,primary",
        [
            (0, (), 42.0),
            (2, (10.0, 2.0, 1.0), 3.0),
            (4, (5.0, 2.0), 8.0),
            (6, (3.0, 2.0), 4.0),
            (8, (1.0, 0.5, 2.0, 1.0), 3.0),
            (10, (2.0, 10.0, 5.0), 1.0),
            (28, (1.0, 2.0, 10.0, 3.0), 2.0),
            (32, (1.0, 2.0, 0.0, 0.0), 3.0),
            (34, (1.0, 0.0, 0.0, 1.0), 7.0),
            (36, (0.0, 2.0, 0.0), 4.0),
            (40, (10.0, 2.0, 1.0, 0.0, 0.0, 0.0), 3.0),
            (50, (1.0, 10.0), 5.0),
            (62, (1.0, 1.0, 0.0), 2.0),
            (66, (1.0, 1.0, 0.0, 0.0), 3.0),
            (78, (1.0, 1.0, 0.0, 0.0), 2.0),
            (80, (), 42.0),
        ],
    )
    def test_analytical_roundtrip(self, c_index, constants, primary):
        """Round-trip for common transforms with analytical inverses."""
        common = _common_scale(primary, c_index, constants)
        recovered = _common_unscale(common, c_index, constants, 10)  # p_index=10 (identity)
        assert recovered == pytest.approx(primary, rel=1e-4, abs=1e-6)

    def test_c86_roundtrip(self):
        """Round-trip for piecewise lin/log/exp (c_index=86, decreasing function)."""
        # Decreasing function: linear part slopes down, exp part decays
        c = (2.0, 8.0, -1.0, 10.0, -0.1, 1.0)
        for primary in [1.0, 5.0, 10.0]:
            common = _common_scale(primary, 86, c)
            recovered = _common_unscale(common, 86, c, 10)
            assert recovered == pytest.approx(primary, rel=1e-3)


class TestCommonBinarySearchRoundTrip:
    """Round-trip tests for common transforms that use binary search inverse."""

    def test_c12_binary_search(self):
        """4th-degree poly (monotonic x^3+2x+1) binary search."""
        c = (0.0, 1.0, 0.0, 2.0, 1.0)
        common = _common_scale(3.0, 12, c)
        recovered = _common_unscale(common, 12, c, 10)
        assert recovered == pytest.approx(3.0, rel=1e-3)

    def test_c14_binary_search(self):
        """exp(poly) binary search with narrow bounds."""
        c = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # exp(x)
        common = _common_scale(2.0, 14, c)
        recovered = _common_unscale(common, 14, c, 0)  # p_index=0 for [-10.24, 10.235]
        assert recovered == pytest.approx(2.0, rel=1e-3)

    def test_c26_binary_search(self):
        """5th-degree poly (identity) binary search."""
        c = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # just x
        common = _common_scale(5.0, 26, c)
        recovered = _common_unscale(common, 26, c, 10)
        assert recovered == pytest.approx(5.0, rel=1e-3)

    def test_c88_binary_search(self):
        """Pade approximant (identity) binary search."""
        c = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)  # x/1 = x
        common = _common_scale(3.0, 88, c)
        recovered = _common_unscale(common, 88, c, 10)
        assert recovered == pytest.approx(3.0, rel=1e-3)

    def test_c64_vpt_roundtrip(self):
        """Vapor pressure thermo (N2) binary search inverse."""
        primary = 2.0
        common = _common_scale(primary, 64, (0.0,))
        recovered = _common_unscale(common, 64, (0.0,), 10)
        assert recovered == pytest.approx(primary, rel=1e-3)

    def test_c12_quadratic_special_case(self):
        """4th-degree poly with C1=C2=0 uses quadratic formula instead of binary search."""
        c = (0.0, 0.0, 1.0, 0.0, 0.0)  # x^2
        common = _common_scale(3.0, 12, c)
        recovered = _common_unscale(common, 12, c, 10)
        assert recovered == pytest.approx(3.0, rel=1e-4)


# ---- Scaler dataclass --------------------------------------------------------


class TestScaler:
    def test_basic_pipeline(self):
        # Identity primary (p_index=10), linear common (c_index=2)
        s = Scaler(p_index=10, c_index=2, constants=(2.0, 1.0, 0.0), input_len=2)
        # primary = float(raw), common = 2*primary/1 + 0 = 2*raw
        assert s.scale(100) == pytest.approx(200.0)

    def test_unscale_pipeline(self):
        s = Scaler(p_index=10, c_index=2, constants=(2.0, 1.0, 0.0), input_len=2)
        raw = s.unscale(200.0)
        assert raw == 100

    def test_roundtrip(self):
        s = Scaler(p_index=2, c_index=2, constants=(10.0, 1.0, 0.0), input_len=2)
        raw = 1000
        value = s.scale(raw)
        recovered = s.unscale(value)
        assert recovered == raw

    def test_individual_stages(self):
        s = Scaler(p_index=10, c_index=2, constants=(2.0, 1.0, 0.0), input_len=2)
        primary = s.raw_to_primary(50)
        assert primary == 50.0
        common = s.primary_to_common(50.0)
        assert common == pytest.approx(100.0)
        primary_back = s.common_to_primary(100.0)
        assert primary_back == pytest.approx(50.0)
        raw_back = s.primary_to_raw(50.0)
        assert raw_back == 50

    def test_identity_scaler(self):
        s = Scaler(p_index=10, c_index=0, constants=(), input_len=2)
        assert s.scale(42) == 42.0
        assert s.unscale(42.0) == 42

    def test_frozen(self):
        s = Scaler(p_index=10, c_index=0, constants=(), input_len=2)
        with pytest.raises(AttributeError):
            s.p_index = 5  # type: ignore[misc]

    def test_invalid_input_len(self):
        with pytest.raises(ValueError, match="input_len must be 1, 2, or 4"):
            Scaler(p_index=10, c_index=0, constants=(), input_len=3)


class TestScalerNumpy:
    def test_scale_array(self):
        s = Scaler(p_index=10, c_index=0, constants=(), input_len=2)
        raw = np.array([1, 2, 3, 4, 5])
        result = s.scale(raw)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_unscale_array(self):
        s = Scaler(p_index=10, c_index=0, constants=(), input_len=2)
        values = np.array([1.0, 2.0, 3.0])
        result = s.unscale(values)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_array_matches_scalar(self):
        s = Scaler(p_index=2, c_index=2, constants=(10.0, 1.0, 0.0), input_len=2)
        raw_values = [100, 200, 300, 500, 1000]
        scalar_results = [s.scale(r) for r in raw_values]
        array_results = s.scale(np.array(raw_values))
        np.testing.assert_array_almost_equal(array_results, scalar_results)

    def test_raw_to_primary_array(self):
        s = Scaler(p_index=10, c_index=0, constants=(), input_len=2)
        result = s.raw_to_primary(np.array([10, 20]))
        np.testing.assert_array_almost_equal(result, [10.0, 20.0])

    def test_common_to_primary_array(self):
        s = Scaler(p_index=10, c_index=2, constants=(2.0, 1.0, 0.0), input_len=2)
        result = s.common_to_primary(np.array([100.0, 200.0]))
        np.testing.assert_array_almost_equal(result, [50.0, 100.0])


class TestScalerFromPropertyInfo:
    def test_from_property_info(self):
        from pacsys.devdb import PropertyInfo

        prop = PropertyInfo(
            primary_units="V",
            common_units="degF",
            min_val=-10.0,
            max_val=10.0,
            p_index=2,
            c_index=2,
            coeff=(100.0, 1.0, 0.0),
            is_step_motor=False,
            is_destructive_read=False,
            is_fe_scaling=False,
            is_contr_setting=False,
            is_knobbable=False,
        )
        s = Scaler.from_property_info(prop, input_len=2)
        assert s.p_index == 2
        assert s.c_index == 2
        assert s.constants == (100.0, 1.0, 0.0)
        assert s.input_len == 2

    def test_default_input_len(self):
        from pacsys.devdb import PropertyInfo

        prop = PropertyInfo(
            primary_units=None,
            common_units=None,
            min_val=0.0,
            max_val=0.0,
            p_index=10,
            c_index=0,
            coeff=(),
            is_step_motor=False,
            is_destructive_read=False,
            is_fe_scaling=False,
            is_contr_setting=False,
            is_knobbable=False,
        )
        s = Scaler.from_property_info(prop)
        assert s.input_len == 2


# ---- Full pipeline round-trip ------------------------------------------------


class TestFullPipeline:
    """End-to-end round-trip: raw -> scale -> unscale -> raw."""

    @pytest.mark.parametrize(
        "p_index,c_index,constants,raw,input_len",
        [
            # Identity primary + identity common
            (10, 0, (), 42, 2),
            # DAC primary + linear common
            (2, 2, (10.0, 1.0, 0.0), 1000, 2),
            # DAC primary + identity common
            (0, 0, (), 1000, 2),
            # Divide-320 primary + ratio common
            (12, 6, (3.0, 2.0), 100, 2),
            # Identity primary + offset-divide common
            (10, 4, (5.0, 2.0), 15, 2),
        ],
    )
    def test_full_roundtrip(self, p_index, c_index, constants, raw, input_len):
        s = Scaler(p_index=p_index, c_index=c_index, constants=constants, input_len=input_len)
        value = s.scale(raw)
        recovered = s.unscale(value)
        assert recovered == raw


# ---- Edge cases and error paths ----------------------------------------------


class TestEdgeCases:
    def test_zero_raw(self):
        s = Scaler(p_index=10, c_index=0, constants=(), input_len=2)
        assert s.scale(0) == 0.0

    def test_negative_raw(self):
        s = Scaler(p_index=10, c_index=0, constants=(), input_len=2)
        assert s.scale(-100) == -100.0

    def test_common_div_by_zero_c4(self):
        with pytest.raises(ScalingError):
            _common_scale(1.0, 4, (0.0, 0.0))

    def test_common_div_by_zero_c8(self):
        # C3 + C2*X = 0
        with pytest.raises(ScalingError):
            _common_scale(0.0, 8, (1.0, 0.0, 0.0, 0.0))

    def test_common_insufficient_constants(self):
        with pytest.raises(ScalingError, match="Insufficient"):
            _common_scale(1.0, 2, (1.0,))

    def test_primary_invalid_length(self):
        with pytest.raises(ScalingError, match="Invalid input_len"):
            _primary_scale(0, 0, 3)
