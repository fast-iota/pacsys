"""
Tests for pacsys.types module.
"""

import threading
import time
from datetime import datetime, timezone

import numpy as np
import pytest

from pacsys.testing import FakeBackend
from pacsys.types import (
    CombinedStream,
    DeviceMeta,
    Reading,
    ValueType,
    WriteResult,
)


class TestReading:
    """Tests for Reading dataclass."""

    @pytest.mark.parametrize(
        ("error_code", "is_success", "is_warning", "is_error"),
        [
            (0, True, False, False),  # success
            (1, False, True, False),  # warning
            (100, False, True, False),  # warning (high)
            (-1, False, False, True),  # error
            (-42, False, False, True),  # error (other)
        ],
    )
    def test_status_flags(self, error_code, is_success, is_warning, is_error):
        """Reading status flags reflect error_code correctly."""
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, error_code=error_code)
        assert reading.is_success is is_success
        assert reading.is_warning is is_warning
        assert reading.is_error is is_error

    @pytest.mark.parametrize(
        ("error_code", "value", "expected_ok"),
        [
            (0, 72.5, True),  # success + value = ok
            (0, None, False),  # success + no value = not ok
            (1, 72.5, True),  # warning + value = ok
            (1, None, False),  # warning + no value = not ok
            (-1, 72.5, False),  # error + value = not ok
            (-1, None, False),  # error + no value = not ok
        ],
    )
    def test_ok_property(self, error_code, value, expected_ok):
        """Reading.ok requires non-negative error_code AND value is not None."""
        reading = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, error_code=error_code, value=value)
        assert reading.ok is expected_ok

    @pytest.mark.parametrize(
        ("drf", "expected_name"),
        [
            ("M:OUTTMP", "M:OUTTMP"),  # simple
            ("M:OUTTMP@p,1000", "M:OUTTMP"),  # with event
            ("B:HS23T[0:10]", "B:HS23T"),  # with range
            ("B:HS23T[0:10]@p,1000", "B:HS23T"),  # with range and event
        ],
    )
    def test_name_from_drf(self, drf, expected_name):
        """Reading.name extracts device name from DRF."""
        reading = Reading(drf=drf, value_type=ValueType.SCALAR)
        assert reading.name == expected_name


class TestReadingEquality:
    def test_equal_scalar(self):
        a = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5)
        b = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5)
        assert a == b

    def test_not_equal_different_value(self):
        a = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5)
        b = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=73.0)
        assert a != b

    def test_equal_numpy_arrays(self):
        a = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0]))
        b = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0]))
        assert a == b

    def test_not_equal_numpy_arrays(self):
        a = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0]))
        b = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 3.0]))
        assert a != b

    def test_equal_timed_scalar_array(self):
        va = {"data": np.array([1.0]), "micros": np.array([0])}
        vb = {"data": np.array([1.0]), "micros": np.array([0])}
        a = Reading(drf="M:OUTTMP", value_type=ValueType.TIMED_SCALAR_ARRAY, value=va)
        b = Reading(drf="M:OUTTMP", value_type=ValueType.TIMED_SCALAR_ARRAY, value=vb)
        assert a == b

    def test_not_equal_to_non_reading(self):
        a = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=1.0)
        assert a != "not a reading"

    def test_hash_includes_value(self):
        a = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5)
        b = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=73.0)
        assert hash(a) != hash(b)

    def test_hash_numpy_array(self):
        a = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0]))
        b = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 3.0]))
        assert hash(a) != hash(b)
        # equal arrays hash the same
        c = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1.0, 2.0]))
        assert hash(a) == hash(c)

    def test_owning_array_frozen_in_place(self):
        # Ownership transfer: the passed-in owning array becomes read-only, no copy
        arr = np.array([1.0, 2.0])
        r = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=arr)
        assert r.value is arr
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 9.0

    def test_view_replaced_by_stable_copy(self):
        # Freezing a view can't freeze its base - a copy keeps the hash stable
        buf = np.arange(10.0)
        r = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=buf[0:5])
        s = {r}
        buf[0] = 99.0
        assert r in s
        # base stays writable and the reading's copy is unaffected
        np.testing.assert_array_equal(r.value, [0.0, 1.0, 2.0, 3.0, 4.0])


class TestCombinedStream:
    """Tests for CombinedStream."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            CombinedStream([])

    def test_context_manager_stops_all(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])

        with CombinedStream([h1, h2]) as cs:
            assert not cs.stopped

        assert h1.stopped
        assert h2.stopped

    def test_stopped_property(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])
        cs = CombinedStream([h1, h2])

        assert not cs.stopped
        h1.stop()
        assert not cs.stopped
        h2.stop()
        assert cs.stopped

    def test_exc_property(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])
        cs = CombinedStream([h1, h2])

        assert cs.exc is None
        err = RuntimeError("boom")
        fake.emit_error(err)
        assert cs.exc is err

    # --- non-blocking mode (timeout=0) ---

    def test_nonblocking_empty(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        results = list(CombinedStream([h1]).readings(timeout=0))
        assert results == []
        h1.stop()

    def test_nonblocking_single_sub(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        fake.emit_reading("M:OUTTMP", 1.0)
        fake.emit_reading("M:OUTTMP", 2.0)

        results = list(CombinedStream([h1]).readings(timeout=0))
        assert len(results) == 2
        assert results[0][0].value == 1.0
        assert results[1][0].value == 2.0
        h1.stop()

    def test_nonblocking_sorts_by_timestamp(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])

        t_early = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t_late = datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        fake.emit_reading("G:AMANDA", 10.0, timestamp=t_late)
        fake.emit_reading("M:OUTTMP", 20.0, timestamp=t_early)

        results = list(CombinedStream([h1, h2]).readings(timeout=0))
        assert len(results) == 2
        assert results[0][0].value == 20.0  # earlier timestamp first
        assert results[1][0].value == 10.0
        h1.stop()
        h2.stop()

    def test_nonblocking_drains_stopped_subs(self):
        # Stopped handles may still hold buffered readings; nonblocking mode
        # must drain them (FakeSubscriptionHandle discards its buffer on stop,
        # so this regression needs the real BufferedSubscriptionHandle).
        from pacsys.backends._subscription import BufferedSubscriptionHandle

        h = BufferedSubscriptionHandle()
        for i in range(3):
            h._dispatch(Reading(drf="M:OUTTMP", value=float(i), value_type=ValueType.SCALAR))
        h._signal_stop()

        results = list(CombinedStream([h]).readings(timeout=0))
        assert [r.value for r, _ in results] == [0.0, 1.0, 2.0]

    def test_nonblocking_raises_on_error(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        err = RuntimeError("boom")
        fake.emit_error(err)

        with pytest.raises(RuntimeError, match="boom"):
            list(CombinedStream([h1]).readings(timeout=0))

    def test_nonblocking_returns_correct_handle(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])
        fake.emit_reading("M:OUTTMP", 1.0)
        fake.emit_reading("G:AMANDA", 2.0)

        results = list(CombinedStream([h1, h2]).readings(timeout=0))
        handles = {r[0].drf: r[1] for r in results}
        assert handles["M:OUTTMP"] is h1
        assert handles["G:AMANDA"] is h2
        h1.stop()
        h2.stop()

    # --- blocking mode ---

    def test_blocking_receives_readings(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])

        def emit_then_stop():
            fake.emit_reading("M:OUTTMP", 42.0)
            h1.stop()

        threading.Timer(0.05, emit_then_stop).start()

        results = list(CombinedStream([h1]).readings(timeout=2))
        assert len(results) == 1
        assert results[0][0].value == 42.0

    def test_blocking_multiple_subs(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])
        cs = CombinedStream([h1, h2])

        def emit_then_stop():
            fake.emit_reading("M:OUTTMP", 1.0)
            fake.emit_reading("G:AMANDA", 2.0)
            h1.stop()
            h2.stop()

        threading.Timer(0.05, emit_then_stop).start()

        results = list(cs.readings(timeout=2))
        values = sorted(r[0].value for r in results)
        assert values == [1.0, 2.0]

    def test_blocking_sorts_by_timestamp(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        h2 = fake.subscribe(["G:AMANDA"])

        t_early = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t_late = datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        # Emit both readings first so they're buffered in the queues,
        # then stop - this avoids timing-dependent batch boundaries.
        fake.emit_reading("G:AMANDA", 10.0, timestamp=t_late)
        fake.emit_reading("M:OUTTMP", 20.0, timestamp=t_early)

        def stop_later():
            time.sleep(0.05)
            h1.stop()
            h2.stop()

        threading.Thread(target=stop_later, daemon=True).start()

        results = list(CombinedStream([h1, h2]).readings(timeout=2))
        assert len(results) == 2
        assert results[0][0].value == 20.0  # earlier timestamp first
        assert results[1][0].value == 10.0

    def test_blocking_timeout_expires(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])

        results = list(CombinedStream([h1]).readings(timeout=0.2))
        assert results == []
        h1.stop()

    def test_blocking_error_propagation(self):
        """Error set before iteration starts is raised by feeder."""
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        fake.emit_error(RuntimeError("fail"))

        with pytest.raises(RuntimeError, match="fail"):
            list(CombinedStream([h1]).readings(timeout=2))

    def test_blocking_error_available_after_stop(self):
        """Error set mid-stream is accessible via .exc after readings end."""
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        err = RuntimeError("late")
        cs = CombinedStream([h1])

        def error_then_stop():
            fake.emit_error(err)

        threading.Timer(0.05, error_then_stop).start()

        list(cs.readings(timeout=2))
        assert cs.exc is err

    def test_blocking_none_timeout_stops_when_all_done(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])

        def emit_then_stop():
            fake.emit_reading("M:OUTTMP", 99.0)
            h1.stop()

        threading.Timer(0.05, emit_then_stop).start()

        results = list(CombinedStream([h1]).readings(timeout=None))
        assert len(results) == 1
        assert results[0][0].value == 99.0

    def test_stop_terminates_blocking_readings(self):
        fake = FakeBackend()
        h1 = fake.subscribe(["M:OUTTMP"])
        cs = CombinedStream([h1])

        threading.Timer(0.1, cs.stop).start()

        results = list(cs.readings(timeout=5))
        assert results == []


class TestReadingToDict:
    def test_scalar_round_trip(self):
        r = Reading(
            drf="M:OUTTMP",
            value_type=ValueType.SCALAR,
            value=72.5,
            timestamp=datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc),
            cycle=5,
            meta=DeviceMeta(device_index=123, name="M:OUTTMP", description="Outside temp", units="deg F"),
        )
        d = r.to_dict()
        r2 = Reading.from_dict(d)
        assert r2.drf == r.drf
        assert r2.value == 72.5
        assert r2.value_type == ValueType.SCALAR
        assert r2.timestamp == r.timestamp
        assert r2.cycle == 5
        assert r2.meta.name == "M:OUTTMP"
        assert r2.meta.units == "deg F"

    def test_array_round_trip(self):
        arr = np.array([1.0, 2.0, 3.0])
        r = Reading(drf="M:OUTTMP[0:2]", value_type=ValueType.SCALAR_ARRAY, value=arr)
        d = r.to_dict()
        r2 = Reading.from_dict(d)
        np.testing.assert_array_equal(r2.value, arr)

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, bool])
    def test_array_dtype_round_trip(self, dtype):
        # Full equality requires dtype preservation, not just value equality
        r = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1, 0, 3], dtype=dtype))
        r2 = Reading.from_dict(r.to_dict())
        assert r2 == r

    def test_legacy_dict_without_dtype(self):
        # Dicts serialized before value_dtype existed must still deserialize
        r = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR_ARRAY, value=np.array([1, 2], dtype=np.int16))
        d = r.to_dict()
        d.pop("value_dtype")
        r2 = Reading.from_dict(d)
        np.testing.assert_array_equal(r2.value, [1, 2])

    def test_bytes_round_trip(self):
        r = Reading(drf="M:OUTTMP.RAW", value_type=ValueType.RAW, value=b"\x01\x02\x03")
        d = r.to_dict()
        r2 = Reading.from_dict(d)
        assert r2.value == b"\x01\x02\x03"

    def test_error_reading_round_trip(self):
        r = Reading(drf="M:BAD", error_code=-42, message="DIO_NO_SUCH")
        d = r.to_dict()
        r2 = Reading.from_dict(d)
        assert r2.is_error
        assert r2.error_code == -42
        assert r2.message == "DIO_NO_SUCH"
        assert r2.value is None

    def test_json_safe(self):
        import json

        r = Reading(
            drf="M:OUTTMP",
            value_type=ValueType.SCALAR,
            value=np.float64(72.5),
            meta=DeviceMeta(device_index=0, name="M:OUTTMP", description="test"),
        )
        s = json.dumps(r.to_dict())
        r2 = Reading.from_dict(json.loads(s))
        assert r2.value == 72.5

    def test_timed_scalar_array_round_trip(self):
        import json

        value = {"data": np.array([1.0, 2.0]), "micros": np.array([0, 1000])}
        r = Reading(drf="M:OUTTMP", value_type=ValueType.TIMED_SCALAR_ARRAY, value=value)
        d = r.to_dict()
        # Must be JSON-safe (nested ndarrays converted to lists)
        s = json.dumps(d)
        r2 = Reading.from_dict(json.loads(s))
        np.testing.assert_array_equal(r2.value["data"], value["data"])
        np.testing.assert_array_equal(r2.value["micros"], value["micros"])
        # Per-key dtypes preserved (data float64, micros int64)
        assert r2 == r

    def test_scalar_dict_value_no_dtype_key(self):
        # BASIC_STATUS-style dicts hold plain scalars - no value_dtype emitted
        r = Reading(drf="M:OUTTMP.STATUS", value_type=ValueType.BASIC_STATUS, value={"on": True, "ready": False})
        d = r.to_dict()
        assert "value_dtype" not in d
        assert Reading.from_dict(d) == r

    def test_omits_none_fields(self):
        r = Reading(drf="M:OUTTMP", error_code=0)
        d = r.to_dict()
        assert "value" not in d
        assert "timestamp" not in d
        assert "cycle" not in d
        assert "meta" not in d
        assert "value_type" not in d


class TestValueToJson:
    def test_numpy_bool(self):
        from pacsys.types import _value_to_json

        result = _value_to_json(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_bytearray_base64(self):
        import base64

        from pacsys.types import _value_to_json

        assert _value_to_json(bytearray(b"\x01\x02")) == base64.b64encode(b"\x01\x02").decode("ascii")

    def test_bytearray_infers_raw(self):
        from pacsys.types import _infer_serialization_type

        assert _infer_serialization_type(bytearray(b"\x01")) == ValueType.RAW


class TestWriteResultToDict:
    def test_success_round_trip(self):
        wr = WriteResult(drf="Z:ACLTST", error_code=0)
        d = wr.to_dict()
        wr2 = WriteResult.from_dict(d)
        assert wr2.ok
        assert wr2.drf == "Z:ACLTST"

    def test_error_round_trip(self):
        wr = WriteResult(drf="Z:ACLTST", error_code=-1, message="Permission denied")
        d = wr.to_dict()
        wr2 = WriteResult.from_dict(d)
        assert not wr2.ok
        assert wr2.message == "Permission denied"

    def test_verification_fields_round_trip(self):
        wr = WriteResult(drf="Z:ACLTST", verified=True, skipped=True, attempts=3)
        d = wr.to_dict()
        wr2 = WriteResult.from_dict(d)
        assert wr2.verified is True
        assert wr2.skipped is True
        assert wr2.attempts == 3

    def test_json_safe(self):
        import json

        wr = WriteResult(drf="Z:ACLTST", error_code=0, message="ok")
        s = json.dumps(wr.to_dict())
        wr2 = WriteResult.from_dict(json.loads(s))
        assert wr2.ok

    def test_array_readback_round_trip(self):
        import json

        wr = WriteResult(drf="Z:ACLTST", verified=True, readback=np.array([1, 2], dtype=np.int16), attempts=1)
        wr2 = WriteResult.from_dict(json.loads(json.dumps(wr.to_dict())))
        assert isinstance(wr2.readback, np.ndarray)
        assert wr2 == wr

    def test_bytes_readback_round_trip(self):
        import json

        wr = WriteResult(drf="Z:ACLTST", verified=True, readback=b"\x01\x02")
        wr2 = WriteResult.from_dict(json.loads(json.dumps(wr.to_dict())))
        assert wr2.readback == b"\x01\x02"
        assert wr2 == wr

    def test_array_readback_eq_hash(self):
        wr = WriteResult(drf="Z:ACLTST", verified=True, readback=np.array([1.0, 2.0]))
        wr2 = WriteResult(drf="Z:ACLTST", verified=True, readback=np.array([1.0, 2.0]))
        assert wr == wr2
        assert hash(wr) == hash(wr2)
        assert wr != WriteResult(drf="Z:ACLTST", verified=True, readback=np.array([1.0, 9.0]))


class TestDeviceMetaToDict:
    def test_round_trip(self):
        m = DeviceMeta(device_index=123, name="M:OUTTMP", description="Outside temp", units="deg F", format_hint=2)
        d = m.to_dict()
        m2 = DeviceMeta.from_dict(d)
        assert m2 == m

    def test_optional_fields_omitted(self):
        m = DeviceMeta(device_index=0, name="X", description="test")
        d = m.to_dict()
        assert "units" not in d
        assert "format_hint" not in d
        m2 = DeviceMeta.from_dict(d)
        assert m2 == m
