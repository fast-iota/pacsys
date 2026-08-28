"""Cursor-based SDD decoding: must be bit-for-bit identical to the byte-iterator path."""

import math
import random

import numpy as np
import pytest

from pacsys.dpm_protocol import (
    DeviceInfo_reply,
    ProtocolError,
    Scalar_reply,
    ScalarArray_reply,
    Text_reply,
    TimedScalarArray_reply,
    _Cursor,
    unmarshal_reply,
)


def _array_msg(n: int, timed: bool = False) -> bytes:
    r = TimedScalarArray_reply() if timed else ScalarArray_reply()
    r.ref_id, r.timestamp, r.cycle, r.status = 7, 1_724_800_000_123, 15, 0
    vals = [random.uniform(-1e6, 1e6) for _ in range(n)]
    if n >= 4:
        vals[:4] = [math.nan, math.inf, -math.inf, -0.0]
    r.data = vals
    if timed:
        r.micros = [i * 1000 for i in range(n)]
    return bytes(r.marshal())


@pytest.mark.parametrize("n", [0, 1, 4, 100, 487, 10_000])
@pytest.mark.parametrize("timed", [False, True])
def test_double_array_parity(n, timed):
    msg = _array_msg(n, timed)
    old = unmarshal_reply(iter(msg))
    new = unmarshal_reply(_Cursor(msg))
    assert type(new) is type(old)
    assert isinstance(new.data, np.ndarray)
    assert new.data.dtype == np.dtype("=f8") and new.data.base is None  # owning, native-endian
    assert np.array_equal(np.array(old.data), new.data, equal_nan=True)
    if n >= 4:
        assert np.signbit(new.data[3])
    if timed:
        assert new.micros == old.micros
    assert (new.ref_id, new.timestamp, new.cycle, new.status) == (old.ref_id, old.timestamp, old.cycle, old.status)
    if n < 4:  # arrays with NaN are never equal to a separately decoded copy, same as before
        assert new == old


def test_scalar_and_text_parity():
    s = Scalar_reply()
    s.ref_id, s.timestamp, s.cycle, s.status, s.data = 7, 1_724_800_000_123, 15, 0, 72.5
    t = Text_reply()
    t.ref_id, t.timestamp, t.cycle, t.status, t.data = 7, 1_724_800_000_123, 15, 0, "M:OUTTMP outdoor temperature"
    d = DeviceInfo_reply()
    d.ref_id, d.di, d.name, d.description, d.units = 3, 1234, "M:OUTTMP", "Outdoor temp", "degF"
    for reply in (s, t, d):
        msg = bytes(reply.marshal())
        assert vars(unmarshal_reply(_Cursor(msg))) == vars(unmarshal_reply(iter(msg)))


@pytest.mark.parametrize("cut", [1, 5, 40])
def test_truncated_array_raises_protocol_error(cut):
    msg = _array_msg(100)
    with pytest.raises(ProtocolError):
        unmarshal_reply(_Cursor(msg[:-cut]))


def test_bad_double_tag_raises_protocol_error():
    msg = bytearray(_array_msg(10))
    # Corrupt the tag byte of the last double (9-byte records at the end of the message)
    msg[-9] = 0x29
    with pytest.raises(ProtocolError, match="expected tag for double"):
        unmarshal_reply(_Cursor(bytes(msg)))


def test_array_reply_equality_with_ndarray_data():
    r = ScalarArray_reply()
    r.data = [float(i) for i in range(50)]  # no NaN: equality must hold across separate decodes
    msg = bytes(r.marshal())
    a, b = unmarshal_reply(_Cursor(msg)), unmarshal_reply(_Cursor(msg))
    assert a == b
    b.data = b.data.copy()
    b.data[7] += 1.0
    assert a != b
