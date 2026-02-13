"""Client-side scaling for ACNET devices.

Tested/ported from Java DPM scaling transforms.

Implements primary transforms (raw -> primary) and common transforms
(primary -> common/engineering units).  Table-lookup and multifunction
transforms (indices 56, 58, 90, 201) are not supported and raise clear errors.

Usage:
    from pacsys.scaling import Scaler

    s = Scaler(p_index=2, c_index=2, constants=(100.0, 1.0, 0.0), input_len=2)
    value = s.scale(1000)       # raw int -> engineering units
    raw = s.unscale(value)      # engineering units -> raw int
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pacsys.devdb import PropertyInfo

try:
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


class ScalingError(Exception):
    """Raised when client-side scaling fails (maps to DIO_SCALEFAIL)."""


# ---- Constants ---------------------------------------------------------------

_FLT_MAX = 1.7014117e38
_TOL_DIVISOR = 131072.0
_MAX_SEARCH_ITER = 1000
_MAX_BISECT_ITER = 30

# Binary search limits indexed by p_index // 2 (from Common.java lowlim/uprlim)
_LOWLIM = (
    -10.24,
    -10.0,
    -5.0,
    -2.5,
    0.0,  # p0..p8
    -4.295e9,
    0.0,
    100.0,
    -_FLT_MAX,
    0.0,  # p10..p18
    0.0,
    -4.295e9,
    -_FLT_MAX,
    -0.310269935,
    -4.295e9,  # p20..p28
    -128.0,
    -128.0,
    0.0,
    0.0,
    -0.310269935,  # p30..p38
    -8388608.0,
    0.0,
    0.0,
    0.0,
    -6.125e36,  # p40..p48
    0.0,
    -_FLT_MAX,
    4.0,
    -10.0,
    0.0,  # p50..p58
    -_FLT_MAX / 500.0,
    0.0,
    -1.0,
    0.0,
    0.0,  # p60..p68
    -10.24,
    -10.24,
    0.0,
    0.0,
    0.0,  # p70..p78
    0.0,  # p80
)

_UPRLIM = (
    10.235,
    9.995,
    4.998,
    2.499,
    65536.0,  # p0..p8
    4.295e9,
    102.35,
    10.0e9,
    _FLT_MAX,
    25.0,  # p10..p18
    65535.0,
    4.295e9,
    _FLT_MAX,
    3.10269935,
    4.295e9,  # p20..p28
    127.0,
    127.0,
    255.0,
    255.0,
    3.10269935,  # p30..p38
    8388607.996,
    10.0,
    9999999.0,
    4294967295.0,
    6.125e36,  # p40..p48
    10.0,
    _FLT_MAX,
    20.0,
    10.0,
    256.0,  # p50..p58
    _FLT_MAX / 500.0,
    10.24,
    1.0,
    10.235,
    0.0,  # p60..p68
    10.235,
    10.235,
    21.0,
    4294967295.0,
    5.0,  # p70..p78
    10.0,  # p80
)

_UNSUPPORTED_TABLE = "Requires interpolation table data. See Common.java in DPM"
_UNSUPPORTED_MFC = "Requires multifunction table data. See Common.java mfcTransform() in DPM"
_UNSUPPORTED_SUB = "Only valid as sub-transform of index 90 (multifunction). See Common.java in DPM"


# ---- Low-level helpers -------------------------------------------------------


def _int_to_float(x: int) -> float:
    """Reinterpret 32-bit int as IEEE 754 float (Java Float.intBitsToFloat)."""
    return struct.unpack(">f", struct.pack(">I", x & 0xFFFFFFFF))[0]


def _float_to_int(f: float) -> int:
    """Reinterpret IEEE 754 float as 32-bit signed int (Java Float.floatToIntBits)."""
    return struct.unpack(">i", struct.pack(">f", f))[0]


def _sign_extend(data: int, length: int) -> int:
    """Sign-extend raw data based on byte length (Java cast semantics)."""
    if length == 1:
        v = data & 0xFF
        return v - 0x100 if v & 0x80 else v
    if length == 2:
        v = data & 0xFFFF
        return v - 0x10000 if v & 0x8000 else v
    if length == 4:
        v = data & 0xFFFFFFFF
        return v - 0x100000000 if v & 0x80000000 else v
    raise ScalingError(f"Invalid input_len: {length}")


def _to_signed32(v: int) -> int:
    """Interpret a Python int as a signed 32-bit value."""
    v = v & 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def _check_signed(value: int, size: int) -> int:
    if size == 1 and (value < -128 or value > 127):
        raise ScalingError("Overflow")
    if size == 2 and (value < -32768 or value > 32767):
        raise ScalingError("Overflow")
    return value


def _check_unsigned(value: int, size: int) -> int:
    if size == 1 and (value < 0 or value > 255):
        raise ScalingError("Overflow")
    if size == 2 and (value < 0 or value > 65535):
        raise ScalingError("Overflow")
    return value


def _java_int_cast(v: float) -> int:
    """Emulate Java ``(int) doubleValue`` narrowing conversion (JLS 5.1.3).

    Java saturates at Integer.MIN/MAX for out-of-range doubles and
    returns 0 for NaN.  Python's ``int()`` raises or returns big ints.
    """
    if math.isnan(v):
        return 0
    if v >= 2147483647.0:
        return 2147483647
    if v <= -2147483648.0:
        return -2147483648
    return int(v)


# ---- Primary transform: raw -> primary -----------------


def _primary_scale(raw: int, p_index: int, input_len: int) -> float:
    """Scale raw integer to primary units."""
    x = _sign_extend(raw, input_len)

    if p_index == 0:
        return x / 3200.0
    if p_index == 2:
        return x / 3276.8
    if p_index == 4:
        return x / 6553.6
    if p_index == 6:
        return x / 13107.2
    if p_index == 8:
        return x + 32768.0
    if p_index == 10:
        return float(x)
    if p_index == 12:
        return x / 320.0
    if p_index == 14:
        # BCD-like: mantissa * 10^exponent
        mantissa = float(x & 0x03FF)
        exponent = (x & 0x1C00) >> 10
        for _ in range(exponent):
            mantissa *= 10.0
        return mantissa
    if p_index == 16:
        return _int_to_float(x)
    if p_index == 18:
        return x * 0.0010406
    if p_index == 20:
        # Unsigned correction
        flt = float(x)
        if flt >= 0.0:
            return flt
        return flt + (256.0 if input_len == 1 else 65536.0)
    if p_index == 22:
        # DEC/VAX float: word-swap then IEEE float / 4
        x1 = ((x & 0xFFFF0000) >> 16) & 0x0000FFFF
        x2 = ((x & 0x0000FFFF) << 16) & 0xFFFF0000
        return _int_to_float(x1 | x2) / 4.0
    if p_index == 24:
        # IEEE float 68000 word ordering (word-swap)
        x1 = ((x & 0xFFFF0000) >> 16) & 0x0000FFFF
        x2 = ((x & 0x0000FFFF) << 16) & 0xFFFF0000
        return _int_to_float(x1 | x2)
    if p_index == 26:
        return ((x >> 8) & 0xFF) / 82.1865 - 0.310269935
    if p_index == 28:
        # 68000 longword: word-swap as signed 32-bit
        result = ((x >> 16) & 0x0000FFFF) | ((x & 0x0000FFFF) << 16)
        return float(_to_signed32(result))
    if p_index == 30:
        # Low byte signed
        return float(_sign_extend(x, 1))
    if p_index == 32:
        # High byte signed
        return float(_sign_extend(x >> 8, 1))
    if p_index == 34:
        return float(x & 0xFF)
    if p_index == 36:
        return float((x >> 8) & 0xFF)
    if p_index == 38:
        return (x & 0xFF) / 82.1865 - 0.310269935
    if p_index == 40:
        return x / 256.0
    if p_index == 42:
        return (x & 0xFFFF) / 6553.6
    if p_index == 44:
        # 7-digit BCD decode
        flt = 0.0
        val = x
        for _ in range(7):
            val <<= 4
            digit = (val >> 28) & 0xF
            flt = flt * 10.0 + digit
        return flt
    if p_index == 46:
        # Unsigned I*4
        if x >= 0:
            return float(x)
        return float(x + 4294967296)
    if p_index == 48:
        return _int_to_float(x) / 0.036
    if p_index == 50:
        # IEEE float clamped to +/- 10.24
        flt = _int_to_float(x)
        return max(-10.24, min(10.235, flt))
    if p_index == 52:
        # Byte-swap
        if input_len == 2:
            y = (((x & 0xFF00) >> 8) & 0x00FF) | ((x << 8) & 0xFF00)
            return float(_sign_extend(y, 2))
        if input_len == 4:
            y = (x << 24) & 0xFF000000
            y |= ((x & 0x0000FF00) << 8) & 0x00FF0000
            y |= ((x & 0x00FF0000) >> 8) & 0x0000FF00
            y |= ((x & 0xFF000000) >> 24) & 0x000000FF
            return float(_to_signed32(y))
        return 0.0
    if p_index == 54:
        if input_len != 2 or (raw & 0x8000) != 0:
            raise ScalingError("Corrupt data")
        return x * 0.0004882961516 + 4.0
    if p_index == 56:
        if input_len != 2:
            raise ScalingError("Must be two byte quantity")
        unsigned_x = raw & 0x0000FFFF
        return (float(unsigned_x) - 32768.0) / 3276.8
    if p_index == 58:
        unsigned_x = raw & 0xFFFFFFFF
        return unsigned_x / 256.0
    if p_index == 60:
        return 500.0 * _int_to_float(x)
    if p_index == 62:
        return x / 6400.0
    if p_index == 64:
        if input_len == 1:
            return x / 128.0
        if input_len == 2:
            return x / 32768.0
        return x / 2147483648.0
    if p_index == 66:
        if input_len != 2:
            raise ScalingError("Must be two byte quantity")
        if x <= 0:
            raise ScalingError("Corrupt data")
        return x / 3200.0
    if p_index == 68:
        raise ScalingError("Alternate Scaling Required")
    if p_index == 70:
        return x / 1000.0
    if p_index == 72:
        if input_len != 2:
            raise ScalingError("Must be two byte quantity")
        unsigned_x = raw & 0x0000FFFF
        return (float(unsigned_x) - 32768.0) / 3200.0
    if p_index == 74:
        if input_len != 2:
            raise ScalingError("Data length error")
        return x * 0.00064088
    if p_index == 76:
        if input_len != 4:
            raise ScalingError("Data length error")
        x1 = ((x & 0xFFFF0000) >> 16) & 0x0000FFFF
        x2 = ((x & 0x0000FFFF) << 16) & 0xFFFF0000
        y = x1 | x2
        return float(y if y >= 0 else y + 0x100000000)
    if p_index == 78:
        flt = _int_to_float(x)
        return max(0.0, min(5.0, flt))
    if p_index == 80:
        flt = _int_to_float(x)
        return max(0.0, min(10.0, flt))
    if p_index == 82:
        if input_len != 2:
            raise ScalingError("Must be two byte quantity")
        if (x & 0xF000) != 0:
            raise ScalingError("Corrupted input data")
        return x / 409.5
    if p_index == 84:
        if input_len != 4:
            raise ScalingError("Must be four byte quantity")
        # Byte-reverse: abcd -> dcba
        y = ((x & 0xFF000000) >> 24) & 0x000000FF
        y |= ((x & 0x00FF0000) >> 8) & 0x0000FF00
        y |= ((x & 0x0000FF00) << 8) & 0x00FF0000
        y |= ((x & 0x000000FF) << 24) & 0xFF000000
        return _int_to_float(y)

    raise ScalingError(f"Primary transform {p_index} not found")


# ---- Primary transform: primary -> raw ---------------


def _primary_unscale(value: float, p_index: int, input_len: int) -> int:
    """Unscale primary value to raw integer."""
    if p_index == 0:
        return _check_signed(int(value * 3200.0), input_len)
    if p_index == 2:
        return _check_signed(int(value * 3276.8), input_len)
    if p_index == 4:
        return _check_signed(int(value * 6553.6), input_len)
    if p_index == 6:
        return _check_signed(int(value * 13107.2), input_len)
    if p_index == 8:
        return _check_signed(int(value - 32768.0), input_len)
    if p_index == 10:
        return _check_signed(int(value), input_len)
    if p_index == 12:
        return _check_signed(int(value * 320.0), input_len)
    if p_index == 14:
        # BCD encode
        mantis = value
        expon = 0
        while mantis > 1023.0:
            mantis /= 10.0
            expon += 1
        state = int(mantis)
        state |= ((expon & 0xFFFF) << 10) & 0xFFFF
        state |= 0xC000
        return state
    if p_index == 16:
        return _float_to_int(float(value))
    if p_index == 18:
        return _check_signed(int(value / 0.001040625), input_len)
    if p_index == 20:
        if value < 0.0:
            raise ScalingError("Corrupt data")
        if input_len == 1:
            if value > 255.0:
                raise ScalingError("Corrupt data")
        elif input_len == 2:
            if value > 65535.0:
                raise ScalingError("Corrupt data")
        elif input_len == 4:
            raise ScalingError("Bad scale")
        return _java_int_cast(value)
    if p_index == 22:
        # DEC float encode: IEEE float * 4 then word-swap
        state = _float_to_int(float(value * 4.0))
        x1 = ((state & 0xFFFF0000) >> 16) & 0x0000FFFF
        x2 = ((state & 0x0000FFFF) << 16) & 0xFFFF0000
        return _to_signed32(x1 | x2)
    if p_index == 24:
        # 68000 IEEE float: float-to-int then word-swap
        state = _float_to_int(float(value))
        x1 = ((state & 0xFFFF0000) >> 16) & 0x0000FFFF
        x2 = ((state & 0x0000FFFF) << 16) & 0xFFFF0000
        return _to_signed32(x1 | x2)
    if p_index == 26:
        mantis = (value + 0.310269935) * 82.1865 * 256.0
        return int(abs(mantis))
    if p_index == 28:
        # 68000 longword: word-swap
        state = int(value)
        x1 = ((state & 0xFFFF0000) >> 16) & 0x0000FFFF
        x2 = ((state & 0x0000FFFF) << 16) & 0xFFFF0000
        return _to_signed32(x1 | x2)
    if p_index == 30:
        return _check_signed(int(value), 1)
    if p_index == 32:
        state = _check_signed(int(value), 1)
        return ((state & 0xFF) << 8) & 0xFF00
    if p_index == 34:
        return _check_unsigned(int(abs(value)), 1)
    if p_index == 36:
        state = _check_signed(int(abs(value)) & 0xFFFF, 1)
        return ((state & 0xFF) << 8) & 0xFF00
    if p_index == 38:
        state = int(abs((value + 0.310269935) * 82.1865))
        return state & 0xFF
    if p_index == 40:
        return _check_signed(int(value * 256.0), 1)
    if p_index == 42:
        xx = value * 6553.6
        if xx > 32768.0:
            xx -= 65536.0
        return int(xx) & 0x0000FFFF
    if p_index == 44:
        # 7-digit BCD encode
        temp = int(value)
        state = 0
        for ii in range(7, 0, -1):
            hbit = temp // int(10**ii)
            if hbit != 0:
                temp = temp - hbit * int(10**ii)
                state += hbit * int(16**ii)
        return state + temp
    if p_index == 46:
        if value < 0.0:
            raise ScalingError("Corrupt data")
        unsigned_state = _java_int_cast(value)
        if unsigned_state < 0:
            unsigned_state = -unsigned_state
        return unsigned_state
    if p_index == 48:
        return _float_to_int(float(value * 0.036))
    if p_index == 50:
        if value < -10.24 or value > 10.235:
            raise ScalingError("Out of valid range")
        return _float_to_int(float(value))
    if p_index == 52:
        state = _check_signed(int(value), input_len)
        y = 0
        if input_len == 2:
            y = (((state & 0xFF00) >> 8) & 0x00FF) | ((state << 8) & 0xFF00)
        elif input_len == 4:
            y = (state << 24) & 0xFF000000
            y |= ((state & 0x0000FF00) << 8) & 0x00FF0000
            y |= ((state & 0x00FF0000) >> 8) & 0x0000FF00
            y |= ((state & 0xFF000000) >> 24) & 0x000000FF
        return _to_signed32(y & 0xFFFFFFFF)
    if p_index == 54:
        if input_len != 2 or value < 4.0 or value > 20.0:
            raise ScalingError("Value out of range")
        state = _check_unsigned(int((value - 4.0) / 0.0004882961516), input_len)
        if (state & 0x8000) != 0:
            raise ScalingError("Value out of range")
        return state
    if p_index == 56:
        if input_len != 2:
            raise ScalingError("Data size invalid")
        return _check_unsigned(int(value * 3276.8 + 32768.0), input_len)
    if p_index == 58:
        unsigned_state = _java_int_cast(value * 256.0)
        if unsigned_state < 0:
            unsigned_state = -unsigned_state
        return unsigned_state
    if p_index == 60:
        return _float_to_int(float(value / 500.0))
    if p_index == 62:
        xx = value * 6400.0
        if xx > 32768.0:
            xx -= 65536.0
        return int(xx) & 0x0000FFFF
    if p_index == 64:
        if value < -1.0 or value > 1.0:
            raise ScalingError("Value out of range")
        if input_len == 1:
            return int(value * 128.0)
        if input_len == 2:
            return int(value * 32768.0)
        return _java_int_cast(value * 2147483648.0)
    if p_index == 66:
        if value <= 0.0:
            raise ScalingError("Value out of range")
        return int(value * 3200.0)
    if p_index == 68:
        return int(value)
    if p_index == 70:
        return _check_signed(int(value * 1000.0), input_len)
    if p_index == 72:
        if input_len != 2:
            raise ScalingError("Data length invalid")
        return _check_unsigned(int(value * 3200.0 + 32768.0), input_len)
    if p_index == 74:
        if input_len != 2 or value < -21.0 or value > 21.0:
            raise ScalingError("Data range invalid")
        return _check_unsigned(int(value / 0.00064088), input_len)
    if p_index == 76:
        if input_len != 4:
            raise ScalingError("Data length invalid")
        xx = value
        if xx >= float(0x100000000) or xx < 0:
            raise ScalingError("Data range invalid")
        if xx >= float(0x80000000):
            state = int(xx - float(0x80000000)) | 0x80000000
        else:
            state = int(xx)
        x1 = ((state & 0xFFFF0000) >> 16) & 0x0000FFFF
        x2 = ((state & 0x0000FFFF) << 16) & 0xFFFF0000
        return _to_signed32(x1 | x2)
    if p_index == 78:
        if value < 0 or value > 5.0:
            raise ScalingError("Out of valid range")
        return _float_to_int(float(value))
    if p_index == 80:
        if value < 0 or value > 10.0:
            raise ScalingError("Out of valid range")
        return _float_to_int(float(value))
    if p_index == 82:
        if input_len != 2:
            raise ScalingError("Data length invalid")
        if value < 0 or value > 10.0:
            raise ScalingError("Out of valid range")
        return int(value * 409.5)
    if p_index == 84:
        if input_len != 4:
            raise ScalingError("Data length invalid")
        state = _float_to_int(float(value))
        # Byte-reverse: abcd -> dcba
        x2 = ((state & 0xFF000000) >> 24) & 0x000000FF
        x2 |= ((state & 0x00FF0000) >> 8) & 0x0000FF00
        x2 |= ((state & 0x0000FF00) << 8) & 0x00FF0000
        x2 |= ((state & 0x000000FF) << 24) & 0xFF000000
        return _to_signed32(x2)

    raise ScalingError(f"Primary transform {p_index} not found")


# ---- Common transform: primary -> common ---------------


def _common_scale(data: float, c_index: int, constants: tuple[float, ...]) -> float:
    """Scale primary value to common (engineering) units."""
    c = constants
    x = data

    if c_index == 0 or c_index == 80:
        return x

    if c_index == 2:
        if len(c) < 3:
            raise ScalingError("Insufficient constants")
        if c[1] == 0.0:
            raise ScalingError("Invalid constant C2 (zero)")
        return (c[0] * x / c[1]) + c[2]

    if c_index == 4:
        if len(c) < 2:
            raise ScalingError("Insufficient constants")
        if c[1] == 0.0:
            raise ScalingError("Invalid constant C2 (zero)")
        return (x - c[0]) / c[1]

    if c_index == 6:
        if len(c) < 2:
            raise ScalingError("Insufficient constants")
        if c[1] == 0.0:
            raise ScalingError("Invalid constant C2 (zero)")
        return c[0] * x / c[1]

    if c_index == 8:
        # X' = C4 + (C1*X)/(C3 + C2*X)
        if len(c) < 4:
            raise ScalingError("Insufficient constants")
        denom = c[2] + c[1] * x
        if denom == 0.0:
            raise ScalingError("Division by zero")
        return c[3] + (c[0] * x) / denom

    if c_index == 10:
        # X' = C3 + C2/(C1*X)
        if len(c) < 3:
            raise ScalingError("Insufficient constants")
        denom = c[0] * x
        if denom == 0.0:
            raise ScalingError("Division by zero")
        return c[2] + c[1] / denom

    if c_index == 12:
        # 4th-degree poly (Horner): C1*x^4 + C2*x^3 + C3*x^2 + C4*x + C5
        if len(c) < 5:
            raise ScalingError("Insufficient constants")
        result = 0.0
        for i in range(5):
            result = c[i] + result * x
        return result

    if c_index == 14:
        # exp(4th-degree poly) - C6
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        poly = 0.0
        for i in range(5):
            poly = c[i] + poly * x
        return math.exp(poly) - c[5]

    if c_index == 16:
        # C2*exp(-X/C1) + C4*exp(-X/C3)
        if len(c) < 4:
            raise ScalingError("Insufficient constants")
        if c[0] == 0.0 or c[2] == 0.0:
            raise ScalingError("Zero C1 or C3 constant")
        return c[1] * math.exp(-x / c[0]) + c[3] * math.exp(-x / c[2])

    if c_index == 18:
        # C3*exp(C2*(X+C1)) + C6*exp(C5*(X+C4))
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        return c[2] * math.exp(c[1] * (x + c[0])) + c[5] * math.exp(c[4] * (x + c[3]))

    if c_index == 20:
        # log10(X)/(C1*log10(X)+C2)^2 + C3
        if len(c) < 3:
            raise ScalingError("Insufficient constants")
        log10x = math.log10(x)
        denom = (c[0] * log10x + c[1]) ** 2
        if denom == 0.0:
            raise ScalingError("Division by zero")
        return log10x / denom + c[2]

    if c_index == 22:
        # C2 * 10^(X/C1)
        if len(c) < 2:
            raise ScalingError("Insufficient constants")
        if c[0] == 0.0:
            raise ScalingError("Zero C1 constant")
        return c[1] * (10.0 ** (x / c[0]))

    if c_index == 24:
        # Piecewise: X<C1 -> C2*(C3*X+C4), else C2*exp(C5*X+C6)
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        if x < c[0]:
            return c[1] * (c[2] * x + c[3])
        return c[1] * math.exp(c[4] * x + c[5])

    if c_index == 26:
        # 5th-degree poly (Horner)
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        result = 0.0
        for i in range(6):
            result = c[i] + result * x
        return result

    if c_index == 28:
        # C3/(C2+C1*X) + C4
        if len(c) < 4:
            raise ScalingError("Insufficient constants")
        denom = c[1] + c[0] * x
        if denom == 0.0:
            raise ScalingError("Division by zero")
        return c[2] / denom + c[3]

    if c_index == 30:
        # X<C1 -> C6, else C5 + C4*X + C3*X^2 + C2*X^3
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        if x < c[0]:
            return c[5]
        return c[4] + c[3] * x + c[2] * x**2 + c[1] * x**3

    if c_index == 32:
        # C2*ln(C1*X+C4) + C3
        if len(c) < 4:
            raise ScalingError("Insufficient constants")
        return c[1] * math.log(c[0] * x + c[3]) + c[2]

    if c_index == 34:
        # (C2+C1*X)/(C4+C3*X)
        if len(c) < 4:
            raise ScalingError("Insufficient constants")
        denom = c[3] + c[2] * x
        if denom == 0.0:
            raise ScalingError("Division by zero")
        return (c[1] + c[0] * x) / denom

    if c_index == 36:
        # C2*sqrt(X+C1) + C3
        if len(c) < 3:
            raise ScalingError("Insufficient constants")
        arg = x + c[0]
        if arg < 0.0:
            raise ScalingError("Square root of negative")
        return c[1] * math.sqrt(arg) + c[2]

    if c_index == 38:
        # Vapor pressure: 10^(C1+C2*X+C3*exp(X)+C4/X+C5/X^2) if X>C6, else 760000
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        if x > c[5]:
            if x == 0.0:
                raise ScalingError("Division by zero")
            if c[2] != 0:
                exponent = c[0] + c[1] * x + c[2] * math.exp(x) + c[3] / x + c[4] / x**2
            else:
                exponent = c[0] + c[1] * x + c[3] / x + c[4] / x**2
            return 10.0**exponent
        return 760000.0

    if c_index == 40:
        # Same as 2: (C1*X/C2) + C3 (C4-C6 are metadata)
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        if c[1] == 0.0:
            raise ScalingError("Zero C2 constant")
        return (c[0] * x / c[1]) + c[2]

    if c_index == 42:
        # X<C1 -> C2*X^2+C3*X+C4, else C2*exp(C5*X+C6)
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        if x < c[0]:
            return c[1] * x**2 + c[2] * x + c[3]
        return c[1] * math.exp(c[4] * x + c[5])

    if c_index == 44:
        # X<C1 -> C2*exp(C3*X), else C4*exp(C5*X)
        if len(c) < 5:
            raise ScalingError("Insufficient constants")
        if x < c[0]:
            return c[1] * math.exp(c[2] * x)
        return c[3] * math.exp(c[4] * x)

    if c_index == 46:
        # X<C1 -> C2*exp(C3*X^2+C4*X), else C5*exp(C6*X)
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        if x < c[0]:
            return c[1] * math.exp(c[2] * x**2 + c[3] * x)
        return c[4] * math.exp(c[5] * x)

    if c_index == 48:
        # C1*C2^(1/X)*X^C3
        if len(c) < 3:
            raise ScalingError("Insufficient constants")
        return c[0] * (c[1] ** (1.0 / x)) * (x ** c[2])

    if c_index == 50:
        # C1*acos(X/C2)
        if len(c) < 2:
            raise ScalingError("Insufficient constants")
        return c[0] * math.acos(x / c[1])

    if c_index == 52:
        # X<C1 -> exp(C2*X+C3), else exp(C4*X+C5)
        if len(c) < 5:
            raise ScalingError("Insufficient constants")
        if x < c[0]:
            return math.exp(c[1] * x + c[2])
        return math.exp(c[3] * x + c[4])

    if c_index == 54:
        # X<C1 -> exp(C2*X^2+C3*X+C4), else exp(C5*X+C6)
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        if x < c[0]:
            return math.exp(c[1] * x**2 + c[2] * x + c[3])
        return math.exp(c[4] * x + c[5])

    if c_index == 56:
        raise ScalingError(_UNSUPPORTED_TABLE)

    if c_index == 58:
        raise ScalingError(_UNSUPPORTED_TABLE)

    if c_index == 60:
        raise ScalingError("Alternate Scaling Required")

    if c_index == 62:
        # C2*(C3 + 10^(X/C1))
        if len(c) < 3:
            raise ScalingError("Insufficient constants")
        if c[0] == 0.0:
            raise ScalingError("Zero C1 constant")
        return c[1] * (c[2] + 10.0 ** (x / c[0]))

    if c_index == 64:
        # Vapor pressure thermo (hardcoded polynomials)
        if len(c) < 1:
            raise ScalingError("Insufficient constants")
        kind = int(c[0])
        if kind == 0:  # N2
            if x <= 4.825:
                return 77.4 + 13.463 * x - 8.2465 * x**2 + 3.6896 * x**3 - 0.7824 * x**4 + 0.0608 * x**5
            return (
                -0.5451 - 428.2927 * x - 210274.8806 * x**2 + 129913.0267 * x**3 - 26741.8467 * x**4 + 1834.8285 * x**5
            )
        if kind == 1:  # He
            if x <= 0.9148:
                return 4.2 + 1.443 * x - 0.40755 * x**2 + 0.3005 * x**3 - 1.5166 * x**4 + 1.3716 * x**5
            return 4.271 + 2.89 * x - 4.11 * x**2 + 3.27 * x**3 - 1.26 * x**4 + 0.202 * x**5
        raise ScalingError("Invalid constant C1 for VPT")

    if c_index == 66:
        # C1*2^(C2*(X+C3)) + C4
        if len(c) < 4:
            raise ScalingError("Insufficient constants")
        return c[0] * (2.0 ** (c[1] * (x + c[2]))) + c[3]

    if c_index == 68:
        # C6*(C2*ln(C1*X+C4) + C3*X)^C5
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        arg = c[0] * x + c[3]
        if c[0] != 0.0 and arg <= 0.0:
            return 0.0
        return c[5] * (c[1] * math.log(arg) + c[2] * x) ** c[4]

    if c_index == 70:
        # C1*exp(-X/C2) + C3*exp(-X/C4) + C5*exp(-X/C6) + 4
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        return c[0] * math.exp(-x / c[1]) + c[2] * math.exp(-x / c[3]) + c[4] * math.exp(-x / c[5]) + 4

    if c_index == 72:
        # C1*10^(C2+C3*log10(X)+C4*log10(X)^2+C5*log10(X)^3) + C6
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        logx = math.log10(x)
        return c[0] * (10.0 ** (c[1] + c[2] * logx + c[3] * logx**2 + c[4] * logx**3)) + c[5]

    if c_index == 74:
        # (C1+C2*X+C3*X^2)/(C4+C5*X+C6*X^2)
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        return (c[0] + c[1] * x + c[2] * x * x) / (c[3] + c[4] * x + c[5] * x * x)

    if c_index == 76:
        # X<C1 -> C2*X^C3, else C4*exp(C5*X+C6)
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        if x < c[0]:
            # math.pow raises ValueError for negative base with fractional exp;
            # Java Math.pow returns NaN - propagate as NaN to match.
            try:
                return c[1] * math.pow(x, c[2])
            except ValueError:
                return math.nan
        return c[3] * math.exp(c[4] * x + c[5])

    if c_index == 78:
        # C1*10^(C2*X+C3) + C4
        if len(c) < 4:
            raise ScalingError("Insufficient constants")
        return c[0] * (10 ** (c[1] * x + c[2])) + c[3]

    if c_index == 82:
        # C2*log10(C1*X+C4) + C3
        if len(c) < 4:
            raise ScalingError("Insufficient constants")
        return c[1] * math.log10(c[0] * x + c[3]) + c[2]

    if c_index == 84:
        raise ScalingError("Alternate Scaling Required")

    if c_index == 86:
        # Piecewise lin/log/exp
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        if x < c[0]:
            return c[2] * x + c[3]
        if x > c[1]:
            return math.exp(c[4] * x + c[5])
        # Logarithmic interpolation between c[0] and c[1]
        x0, x1 = c[0], c[1]
        y0 = c[2] * c[0] + c[3]
        y1 = math.exp(c[4] * c[1] + c[5])
        log_result = math.log(y1 / y0) * (x - x0) / (x1 - x0) + math.log(y0)
        return math.exp(log_result)

    if c_index == 88:
        # Pade approximant: (C1+C2*X+C3*X^2)/(1+C4*X+C5*X^2+C6*X^3)
        if len(c) < 6:
            raise ScalingError("Insufficient constants")
        return (c[0] + x * (c[1] + x * c[2])) / (1.0 + x * (c[3] + x * (c[4] + x * c[5])))

    if c_index == 90:
        raise ScalingError(_UNSUPPORTED_MFC)

    if c_index == 201:
        raise ScalingError(_UNSUPPORTED_SUB)

    raise ScalingError(f"Common transform {c_index} not found")


# ---- Binary search / bisection --------------------------------


def _binary_search(
    target: float,
    c_index: int,
    constants: tuple[float, ...],
    pulow: float,
    puupr: float,
    tol: float,
) -> float:
    """Binary search for common transform inverse (port of java binarySearch)."""
    if tol > 0.0001:
        tol = 0.0001

    def _f(primary: float) -> float:
        # Java returns Infinity/NaN on overflow/domain-error; Python raises.
        try:
            return _common_scale(primary, c_index, constants)
        except OverflowError:
            return math.inf
        except (ValueError, ZeroDivisionError):
            return math.nan

    cuupr = _f(puupr) - target
    culow = _f(pulow) - target

    # Phase 1: find sign-change interval
    for _ in range(_MAX_SEARCH_ITER):
        if (cuupr > 0 and culow > 0) or (cuupr < 0 and culow < 0):
            pumid = (pulow + puupr) / 2.0
            cumid = _f(pumid) - target

            if abs(puupr) < _FLT_MAX / 2.0 and abs(pulow) < _FLT_MAX / 2.0:
                if abs(puupr - pulow) <= tol:
                    raise ScalingError("Interval not found")

            if pumid == pulow or pumid == puupr:
                raise ScalingError("Interval not found")

            if (cuupr > 0 and cumid < 0) or (cuupr < 0 and cumid > 0):
                puupr = pumid
                cuupr = cumid
                break

            d1, d2, d3 = abs(cumid), abs(culow), abs(cuupr)
            if d3 > d1 and d3 >= d2:
                puupr = pumid
                cuupr = cumid
            elif d2 > d1 and d2 >= d3:
                pulow = pumid
                culow = cumid
            else:
                raise ScalingError("Interval not found")
        else:
            break

    # Phase 2: bisect to convergence
    for _ in range(_MAX_SEARCH_ITER):
        pumid = (pulow + puupr) / 2.0

        if abs(puupr) < _FLT_MAX / 2.0 or abs(pulow) < _FLT_MAX / 2.0:
            if abs(puupr - pulow) <= tol:
                return pumid

        if pumid == pulow or pumid == puupr:
            return pumid

        cuupr_val = _f(puupr)
        cumid_val = _f(pumid)

        if cumid_val == target:
            return pumid
        elif cumid_val > target and cuupr_val >= target:
            puupr = pumid
        elif cumid_val < target and cuupr_val <= target:
            puupr = pumid
        elif cumid_val < target and cuupr_val >= target:
            pulow = pumid
        elif cumid_val > target and cuupr_val <= target:
            pulow = pumid

    raise ScalingError("Binary search did not converge")


def _root_bisection(
    target: float,
    c_index: int,
    constants: tuple[float, ...],
    x1: float,
    x2: float,
    xacc: float,
) -> float:
    """Root bisection for vapor pressure inverse (port of java rootBisection)."""

    def _f(primary: float) -> float:
        try:
            return _common_scale(primary, c_index, constants)
        except OverflowError:
            return math.inf
        except (ValueError, ZeroDivisionError):
            return math.nan

    f = _f(x1) - target
    fmid = _f(x2) - target

    if f * fmid >= 0.0:
        raise ScalingError("Value not bracketed")

    if f < 0.0:
        rtb = x1
        dx = x2 - x1
    else:
        rtb = x2
        dx = x1 - x2

    for _ in range(_MAX_BISECT_ITER):
        dx *= 0.5
        xmid = rtb + dx
        fmid = _f(xmid) - target
        if fmid <= 0.0:
            rtb = xmid
        if abs(dx) < xacc or fmid == 0.0:
            return rtb

    return rtb


# ---- Common transform inverse: common -> primary -----


def _common_unscale(
    xx: float,
    c_index: int,
    constants: tuple[float, ...],
    p_index: int,
) -> float:
    """Unscale common (engineering) value to primary units."""
    c = constants
    lim_idx = p_index // 2

    # Default primary-unit limits for binary search
    if lim_idx < len(_LOWLIM):
        pulow = _LOWLIM[lim_idx]
        puupr = _UPRLIM[lim_idx]
    else:
        pulow = 0.0
        puupr = 0.0

    if c_index == 0 or c_index == 80:
        return xx

    if c_index in (2, 40):
        return (xx - c[2]) * c[1] / c[0]

    if c_index == 4:
        return xx * c[1] + c[0]

    if c_index == 6:
        return xx * c[1] / c[0]

    if c_index == 8:
        # X = C3*(X'-C4) / (C1 - C2*(X'-C4))
        denom = c[0] - c[1] * (xx - c[3])
        if denom == 0:
            raise ScalingError("Division by zero")
        return c[2] * (xx - c[3]) / denom

    if c_index == 10:
        denom = (xx - c[2]) * c[0]
        if denom == 0:
            raise ScalingError("Division by zero")
        return c[1] / denom

    if c_index == 12:
        # Quadratic special case, else binary search
        if c[0] == 0.0 and c[1] == 0.0 and c[2] != 0.0:
            aa, bb, cc = c[2], c[3], c[4] - xx
            discr = bb * bb - 4.0 * aa * cc
            if discr < 0:
                raise ScalingError("Negative discriminant")
            return (-bb + math.sqrt(discr)) / (2.0 * aa)
        tol = puupr / _TOL_DIVISOR
        return _binary_search(xx, c_index, constants, pulow, puupr, tol)

    if c_index == 20:
        tmpx = xx if c[2] == 0.0 else xx - c[2]
        a = -4 * tmpx * c[0] * c[1] + 1
        if a < 0 or tmpx == 0:
            raise ScalingError("Domain error")
        b = ((1 - 2 * tmpx * c[0] * c[1] + math.sqrt(a)) / (2 * tmpx * c[0] ** 2)) / 0.43429
        return math.exp(b)

    if c_index == 28:
        denom = c[0] * (xx - c[3])
        if denom == 0:
            raise ScalingError("Division by zero")
        return c[2] / denom - c[1] / c[0]

    if c_index == 32:
        a = (xx - c[2]) / c[1]
        return math.exp(a) / c[0] - c[3]

    if c_index == 34:
        denom = c[0] - c[2] * xx
        if denom == 0:
            raise ScalingError("Division by zero")
        return (c[3] * xx - c[1]) / denom

    if c_index == 36:
        return ((xx - c[2]) / c[1]) ** 2 - c[0]

    if c_index == 38:
        # Vapor pressure inverse
        if xx == 7.6e5:
            return c[5]
        if xx > 7.6e5 or xx <= 0:
            raise ScalingError("Value out of range")
        if c[2] == 0 and c[3] == 0 and c[4] == 0 and c[1] != 0:
            return (math.log10(xx) - c[0]) / c[1]
        # Binary search / root bisection
        if p_index == 16:
            pl = 1.00001 * c[5]
            pu = 15.0
        else:
            pl = 0.00001 if c[5] == 0 else 1.00001 * c[5]
            pu = puupr
        tol = pu / _TOL_DIVISOR
        return _root_bisection(xx, c_index, constants, pl, pu, tol)

    if c_index == 50:
        return c[1] * math.cos(xx / c[0])

    if c_index == 56:
        raise ScalingError(_UNSUPPORTED_TABLE)

    if c_index == 58:
        raise ScalingError(_UNSUPPORTED_TABLE)

    if c_index == 60:
        return xx

    if c_index == 62:
        if c[0] == 0.0 or c[1] == 0.0:
            raise ScalingError("Zero constant")
        tmpx = xx / c[1] - c[2]
        if tmpx < 0:
            raise ScalingError("Logarithm of negative value")
        return c[0] * math.log10(tmpx)

    if c_index == 64:
        # Vapor pressure thermo: binary search with fixed limits
        pl, pu = -1.75489, 5.05
        tol = (pu - pl) / _TOL_DIVISOR
        upper_bound = _common_scale(pu, c_index, constants)
        if xx < 0 or xx > upper_bound:
            raise ScalingError("Value out of range")
        return _binary_search(xx, c_index, constants, pl, pu, tol)

    if c_index == 66:
        return (math.log((xx - c[3]) / c[0]) / 0.6931471805599453 - c[2]) / c[1]

    if c_index == 68:
        # Binary search with overridden upper limit
        pu = 0.0
        tol = (pu - pulow) / _TOL_DIVISOR
        upper_bound = _common_scale(pulow, c_index, constants)
        if xx < 0 or xx > upper_bound:
            raise ScalingError("Value out of range")
        return _binary_search(xx, c_index, constants, pulow, pu, tol)

    if c_index == 74:
        a = xx * c[5] - c[2]
        b = xx * c[4] - c[1]
        g = xx * c[3] - c[0]
        d = b * b - 4 * a * g
        if d < 0:
            raise ScalingError("Negative discriminant")
        d = math.sqrt(d)
        tmpx = (-b + d) / (2 * a)
        if pulow <= tmpx <= puupr:
            state = tmpx
        else:
            state = (-b - d) / (2 * a)
        if state < pulow or state > puupr:
            raise ScalingError("Value out of range")
        return state

    if c_index == 76:
        a = c[1] * math.pow(c[0], c[2])  # breakpoint y-value
        b = c[3] * c[4]  # slope sign of right branch
        if c[1] > 0 and b > 0:
            if xx < a:
                return math.pow(xx / c[1], 1.0 / c[2])
            return (math.log(xx / c[3]) - c[5]) / c[4]
        if c[1] < 0 and b < 0:
            if xx >= a:
                return math.pow(xx / c[1], 1.0 / c[2])
            return (math.log(xx / c[3]) - c[5]) / c[4]
        if c[1] > 0 and b < 0:
            if xx < a:
                return math.pow(xx / c[1], 1.0 / c[2])
            raise ScalingError("Value out of range")
        # c[1] < 0 and b > 0
        if xx >= a:
            return math.pow(xx / c[1], 1.0 / c[2])
        raise ScalingError("Value out of range")

    if c_index == 78:
        return (math.log10((xx - c[3]) / c[0]) - c[2]) / c[1]

    if c_index == 86:
        x0, x1 = c[0], c[1]
        y0 = c[2] * c[0] + c[3]
        y1 = math.exp(c[4] * c[1] + c[5])
        if xx > y0:
            return (xx - c[3]) / c[2]
        if xx < y1:
            return (math.log(xx) - c[5]) / c[4]
        return (x1 - x0) * math.log(xx / y0) / math.log(y1 / y0) + x0

    if c_index == 90:
        raise ScalingError(_UNSUPPORTED_MFC)

    if c_index == 201:
        raise ScalingError(_UNSUPPORTED_SUB)

    # Default: binary search
    tol = puupr / _TOL_DIVISOR
    return _binary_search(xx, c_index, constants, pulow, puupr, tol)


# ---- Scaler dataclass --------------------------------------------------------


@dataclass(frozen=True)
class Scaler:
    """Client-side scaler for ACNET devices.

    Combines primary transform (raw bytes -> primary units) and common
    transform (primary -> engineering units) into a single pipeline.

    Args:
        p_index: Primary transform index (0-84, even)
        c_index: Common transform index (0-90, even)
        constants: Common transform constants (C1..C6)
        input_len: Raw data width in bytes (1, 2, or 4)
    """

    p_index: int
    c_index: int
    constants: tuple[float, ...]
    input_len: int

    def __post_init__(self) -> None:
        if self.input_len not in (1, 2, 4):
            raise ValueError(f"input_len must be 1, 2, or 4, got {self.input_len}")

    def scale(self, raw):  # type: (int | float | object) -> float | object
        """Full pipeline: raw integer -> engineering units.

        Accepts int, float, or numpy array.
        """
        if _HAS_NUMPY and isinstance(raw, _np.ndarray):
            return _np.vectorize(self._scale_one)(raw)
        return self._scale_one(int(raw))

    def unscale(self, value):  # type: (int | float | object) -> int | object
        """Full pipeline: engineering units -> raw integer.

        Accepts int, float, or numpy array.
        """
        if _HAS_NUMPY and isinstance(value, _np.ndarray):
            return _np.vectorize(self._unscale_one)(value)
        return self._unscale_one(float(value))

    def raw_to_primary(self, raw):  # type: (int | float | object) -> float | object
        """Raw integer -> primary units."""
        if _HAS_NUMPY and isinstance(raw, _np.ndarray):
            return _np.vectorize(lambda r: _primary_scale(int(r), self.p_index, self.input_len))(raw)
        return _primary_scale(int(raw), self.p_index, self.input_len)

    def primary_to_common(self, primary):  # type: (float | object) -> float | object
        """Primary units -> common (engineering) units."""
        if _HAS_NUMPY and isinstance(primary, _np.ndarray):
            return _np.vectorize(lambda p: _common_scale(float(p), self.c_index, self.constants))(primary)
        return _common_scale(float(primary), self.c_index, self.constants)

    def common_to_primary(self, value):  # type: (float | object) -> float | object
        """Common (engineering) units -> primary units."""
        if _HAS_NUMPY and isinstance(value, _np.ndarray):
            return _np.vectorize(lambda v: _common_unscale(float(v), self.c_index, self.constants, self.p_index))(value)
        return _common_unscale(float(value), self.c_index, self.constants, self.p_index)

    def primary_to_raw(self, primary):  # type: (float | object) -> int | object
        """Primary units -> raw integer."""
        if _HAS_NUMPY and isinstance(primary, _np.ndarray):
            return _np.vectorize(lambda p: _primary_unscale(float(p), self.p_index, self.input_len))(primary)
        return _primary_unscale(float(primary), self.p_index, self.input_len)

    def _scale_one(self, raw: int) -> float:
        primary = _primary_scale(raw, self.p_index, self.input_len)
        return _common_scale(primary, self.c_index, self.constants)

    def _unscale_one(self, value: float) -> int:
        primary = _common_unscale(value, self.c_index, self.constants, self.p_index)
        return _primary_unscale(primary, self.p_index, self.input_len)

    @classmethod
    def from_property_info(cls, prop: PropertyInfo, input_len: int = 2) -> Scaler:
        """Create a Scaler from DevDB PropertyInfo.

        Args:
            prop: PropertyInfo from DevDB (has p_index, c_index, coeff)
            input_len: Raw data width in bytes (default: 2). DevDB doesn't
                       store this, so it must be provided separately.
        """
        return cls(
            p_index=prop.p_index,
            c_index=prop.c_index,
            constants=prop.coeff,
            input_len=input_len,
        )
