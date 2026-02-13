"""
Unit tests for _DpmStreamCore.stream().

Tests the pure-async DPM streaming protocol logic in isolation,
using MockAsyncDPMConnection to inject canned reply sequences.
Follows the TestDaqCoreStream pattern from test_grpc_backend.py.
"""

import asyncio

from pacsys.backends.dpm_http import _DpmStreamCore
from pacsys.dpm_connection import DPMConnectionError
from pacsys.types import ValueType

from tests.devices import (
    TEMP_DEVICE,
    TEMP_DEVICE_2,
    TEMP_VALUE,
    AMANDA_VALUE,
    TEXT_VALUE,
    TEXT_ARRAY_VALUES,
    RAW_BYTES,
    MockAsyncDPMConnection,
    make_add_to_list_reply,
    make_analog_alarm_reply,
    make_basic_status_reply,
    make_device_info,
    make_list_status_reply,
    make_raw_reply,
    make_scalar_array_reply,
    make_scalar_reply,
    make_start_list,
    make_status_reply,
    make_text_array_reply,
    make_text_reply,
)


class TestDpmStreamCore:
    """Tests for _DpmStreamCore.stream() protocol logic."""

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    @staticmethod
    def _core(conn):
        return _DpmStreamCore(conn)

    # -- Scalar reply dispatched -------------------------------------------

    def test_scalar_reply_dispatched(self):
        """Single scalar reply produces a dispatched Reading."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name=TEMP_DEVICE, ref_id=1),
                make_scalar_reply(value=TEMP_VALUE, ref_id=1),
            ]
        )
        dispatched, errors = [], []

        self._run(
            self._core(conn).stream(
                drfs=[f"{TEMP_DEVICE}@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=errors.append,
            )
        )

        assert len(dispatched) == 1
        r = dispatched[0]
        assert r.value == TEMP_VALUE
        assert r.value_type == ValueType.SCALAR
        assert r.drf == f"{TEMP_DEVICE}@p,1000"
        assert r.meta is not None
        assert r.meta.name == TEMP_DEVICE

    # -- Multiple devices --------------------------------------------------

    def test_multiple_devices(self):
        """N devices produce N readings with correct DRF-to-value mapping."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name=TEMP_DEVICE, ref_id=1),
                make_device_info(name=TEMP_DEVICE_2, ref_id=2),
                make_scalar_reply(value=TEMP_VALUE, ref_id=1),
                make_scalar_reply(value=AMANDA_VALUE, ref_id=2),
            ]
        )
        dispatched, errors = [], []

        self._run(
            self._core(conn).stream(
                drfs=[f"{TEMP_DEVICE}@p,1000", f"{TEMP_DEVICE_2}@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=errors.append,
            )
        )

        assert len(dispatched) == 2
        assert dispatched[0].value == TEMP_VALUE
        assert dispatched[0].drf == f"{TEMP_DEVICE}@p,1000"
        assert dispatched[1].value == AMANDA_VALUE
        assert dispatched[1].drf == f"{TEMP_DEVICE_2}@p,1000"

    # -- DeviceInfo populates meta -----------------------------------------

    def test_device_info_populates_meta(self):
        """DeviceInfo_reply sets Reading.meta with name/description/units."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name=TEMP_DEVICE, ref_id=1, description="Outdoor temp", units="degF"),
                make_scalar_reply(ref_id=1),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 1
        meta = dispatched[0].meta
        assert meta is not None
        assert meta.name == TEMP_DEVICE
        assert meta.description == "Outdoor temp"
        assert meta.units == "degF"

    # -- AddToList error dispatches error reading --------------------------

    def test_add_to_list_error_dispatches_error_reading(self):
        """AddToList with non-zero status dispatches an error Reading."""
        error_status = 0x0002_002A  # some error code
        conn = MockAsyncDPMConnection(
            replies=[
                make_add_to_list_reply(ref_id=1, status=error_status),
                make_start_list(),
            ]
        )
        dispatched, errors = [], []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=errors.append,
            )
        )

        assert len(dispatched) == 1
        r = dispatched[0]
        assert not r.ok
        assert r.drf == TEMP_DEVICE

    # -- Heartbeat filtered ------------------------------------------------

    def test_heartbeat_filtered(self):
        """ListStatus_reply (heartbeat) does not produce a dispatch."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_list_status_reply(),
                make_list_status_reply(),
                make_device_info(ref_id=1),
                make_scalar_reply(ref_id=1),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        # Only the scalar reply should be dispatched, not heartbeats
        assert len(dispatched) == 1
        assert dispatched[0].value == TEMP_VALUE

    # -- StartList failure logged ------------------------------------------

    def test_start_list_failure_continues(self):
        """StartList with error status is logged but recv loop continues."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(status=99),
                make_device_info(ref_id=1),
                make_scalar_reply(ref_id=1),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        # Stream should continue after bad StartList
        assert len(dispatched) == 1

    # -- Connection error calls error_fn -----------------------------------

    def test_connection_error_calls_error_fn(self):
        """DPMConnectionError from recv_message() calls error_fn."""
        # Empty replies → MockAsyncDPMConnection raises DPMConnectionError
        conn = MockAsyncDPMConnection(replies=[])
        dispatched, errors = [], []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=errors.append,
            )
        )

        assert not dispatched
        assert len(errors) == 1
        assert isinstance(errors[0], DPMConnectionError)

    # -- CancelledError clean exit -----------------------------------------

    def test_cancelled_error_clean_exit(self):
        """CancelledError produces no error_fn or dispatch_fn calls."""

        class _CancellingConn(MockAsyncDPMConnection):
            async def recv_message(self):
                raise asyncio.CancelledError

        conn = _CancellingConn(replies=[])
        dispatched, errors = [], []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=errors.append,
            )
        )

        assert not dispatched
        assert not errors

    # -- stop_check exits loop ---------------------------------------------

    def test_stop_check_exits_loop(self):
        """stop_check returning True after N dispatches exits cleanly."""
        call_count = [0]

        def stop_after_one():
            return call_count[0] >= 1

        def counting_dispatch(reading):
            call_count[0] += 1

        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_scalar_reply(ref_id=1),
                make_scalar_reply(ref_id=1),  # should not be reached
                make_scalar_reply(ref_id=1),
            ]
        )
        errors = []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=counting_dispatch,
                stop_check=stop_after_one,
                error_fn=errors.append,
            )
        )

        assert call_count[0] == 1
        assert not errors

    # -- Unknown ref_id ignored --------------------------------------------

    def test_unknown_ref_id_ignored(self):
        """Reply with unknown ref_id is logged but not dispatched."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_scalar_reply(ref_id=99, value=1.0),  # unknown ref_id
                make_scalar_reply(ref_id=1, value=TEMP_VALUE),  # known
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 1
        assert dispatched[0].value == TEMP_VALUE

    # -- Mixed reply types -------------------------------------------------

    def test_mixed_reply_types(self):
        """Scalar + Text + Raw in one stream produce correct value types."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name="D:SCALAR", ref_id=1),
                make_device_info(name="D:TEXT", ref_id=2),
                make_device_info(name="D:RAW", ref_id=3),
                make_scalar_reply(value=42.0, ref_id=1),
                make_text_reply(text=TEXT_VALUE, ref_id=2),
                make_raw_reply(data=RAW_BYTES, ref_id=3),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=["D:SCALAR@p,1000", "D:TEXT@p,1000", "D:RAW@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 3
        assert dispatched[0].value_type == ValueType.SCALAR
        assert dispatched[0].value == 42.0
        assert dispatched[1].value_type == ValueType.TEXT
        assert dispatched[1].value == TEXT_VALUE
        assert dispatched[2].value_type == ValueType.RAW
        assert dispatched[2].value == RAW_BYTES

    # -- Status_reply dispatches error reading -----------------------------

    def test_status_reply_dispatches_error_reading(self):
        """Status_reply with non-zero status produces error Reading with correct codes."""
        from pacsys.acnet.errors import make_error

        error_status = make_error(1, -42)
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name=TEMP_DEVICE, ref_id=1),
                make_status_reply(status=error_status, ref_id=1),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 1
        r = dispatched[0]
        assert r.is_error
        assert r.error_code == -42
        assert r.facility_code == 1
        assert r.value is None
        assert r.drf == TEMP_DEVICE

    # -- ScalarArray_reply dispatched --------------------------------------

    def test_scalar_array_reply_dispatched(self):
        """ScalarArray_reply produces SCALAR_ARRAY reading with correct array."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name="B:HS23T", ref_id=1),
                make_scalar_array_reply(values=values, ref_id=1),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=["B:HS23T[0:4]@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 1
        r = dispatched[0]
        assert r.value_type == ValueType.SCALAR_ARRAY
        assert list(r.value) == values
        assert r.drf == "B:HS23T[0:4]@p,1000"

    # -- TextArray_reply dispatched ----------------------------------------

    def test_text_array_reply_dispatched(self):
        """TextArray_reply produces TEXT_ARRAY reading with correct list."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name="D:TARR", ref_id=1),
                make_text_array_reply(texts=TEXT_ARRAY_VALUES, ref_id=1),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=["D:TARR@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 1
        r = dispatched[0]
        assert r.value_type == ValueType.TEXT_ARRAY
        assert r.value == TEXT_ARRAY_VALUES
        assert r.drf == "D:TARR@p,1000"

    # -- AnalogAlarm_reply dispatched --------------------------------------

    def test_analog_alarm_reply_dispatched(self):
        """AnalogAlarm_reply produces ANALOG_ALARM reading with correct dict."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name=TEMP_DEVICE, ref_id=1),
                make_analog_alarm_reply(ref_id=1, minimum=-10.0, maximum=100.0),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=[f"{TEMP_DEVICE}.ANALOG_ALARM@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 1
        r = dispatched[0]
        assert r.value_type == ValueType.ANALOG_ALARM
        assert r.ok
        assert r.value["minimum"] == -10.0
        assert r.value["maximum"] == 100.0
        assert r.value["alarm_enable"] is True
        assert r.value["alarm_status"] is False

    # -- BasicStatus_reply dispatched --------------------------------------

    def test_basic_status_reply_dispatched(self):
        """BasicStatus_reply produces BASIC_STATUS reading with present fields."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name=TEMP_DEVICE, ref_id=1),
                make_basic_status_reply(ref_id=1, on=True, ready=False, remote=True),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=[f"{TEMP_DEVICE}.STATUS@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 1
        r = dispatched[0]
        assert r.value_type == ValueType.BASIC_STATUS
        assert r.ok
        assert r.value["on"] is True
        assert r.value["ready"] is False
        assert r.value["remote"] is True
        # positive and ramp were not set (None) → omitted from dict
        assert "positive" not in r.value
        assert "ramp" not in r.value

    # -- Mixed valid and invalid devices -----------------------------------

    def test_mixed_valid_and_invalid_devices(self):
        """One AddToList error + one successful data reply → 1 error + 1 success."""
        from pacsys.acnet.errors import make_error

        error_status = make_error(1, -42)
        conn = MockAsyncDPMConnection(
            replies=[
                make_add_to_list_reply(ref_id=1, status=error_status),
                make_add_to_list_reply(ref_id=2, status=0),
                make_start_list(),
                make_device_info(name=TEMP_DEVICE_2, ref_id=2),
                make_scalar_reply(value=AMANDA_VALUE, ref_id=2),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE, f"{TEMP_DEVICE_2}@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 2
        # Error reading from AddToList failure
        err_reading = [r for r in dispatched if r.is_error]
        ok_reading = [r for r in dispatched if r.ok]
        assert len(err_reading) == 1
        assert err_reading[0].drf == TEMP_DEVICE
        assert err_reading[0].error_code == -42
        assert len(ok_reading) == 1
        assert ok_reading[0].value == AMANDA_VALUE

    # -- Connection error mid-stream ---------------------------------------

    def test_connection_error_mid_stream(self):
        """Some successful dispatches then DPMConnectionError → readings + error_fn."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name=TEMP_DEVICE, ref_id=1),
                make_scalar_reply(value=TEMP_VALUE, ref_id=1),
                # MockAsyncDPMConnection raises DPMConnectionError when exhausted
            ]
        )
        dispatched, errors = [], []

        self._run(
            self._core(conn).stream(
                drfs=[f"{TEMP_DEVICE}@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=errors.append,
            )
        )

        assert len(dispatched) == 1
        assert dispatched[0].value == TEMP_VALUE
        assert len(errors) == 1
        assert isinstance(errors[0], DPMConnectionError)

    # -- Multiple data replies same ref_id ---------------------------------

    def test_multiple_data_replies_same_ref_id(self):
        """Periodic stream: same ref_id dispatched N times with correct DRF/meta."""
        conn = MockAsyncDPMConnection(
            replies=[
                make_start_list(),
                make_device_info(name=TEMP_DEVICE, ref_id=1, description="Sensor"),
                make_scalar_reply(value=70.0, ref_id=1),
                make_scalar_reply(value=71.0, ref_id=1),
                make_scalar_reply(value=72.0, ref_id=1),
            ]
        )
        dispatched = []

        self._run(
            self._core(conn).stream(
                drfs=[f"{TEMP_DEVICE}@p,1000"],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=lambda e: None,
            )
        )

        assert len(dispatched) == 3
        assert [r.value for r in dispatched] == [70.0, 71.0, 72.0]
        # All readings share the same DRF and meta
        for r in dispatched:
            assert r.drf == f"{TEMP_DEVICE}@p,1000"
            assert r.meta is not None
            assert r.meta.name == TEMP_DEVICE
            assert r.meta.description == "Sensor"

    # -- IncompleteReadError calls error_fn --------------------------------

    def test_incomplete_read_calls_error_fn(self):
        """asyncio.IncompleteReadError from recv → error_fn called."""

        class _IncompleteReadConn(MockAsyncDPMConnection):
            async def recv_message(self):
                raise asyncio.IncompleteReadError(partial=b"\x00", expected=4)

        conn = _IncompleteReadConn(replies=[])
        dispatched, errors = [], []

        self._run(
            self._core(conn).stream(
                drfs=[TEMP_DEVICE],
                dispatch_fn=dispatched.append,
                stop_check=lambda: False,
                error_fn=errors.append,
            )
        )

        assert not dispatched
        assert len(errors) == 1
        assert isinstance(errors[0], asyncio.IncompleteReadError)
