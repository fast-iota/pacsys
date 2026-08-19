import pytest

from pacsys.drf3 import (
    ARRAY_RANGE,
    DRF_FIELD,
    DRF_PROPERTY,
    ClockEvent,
    DefaultEvent,
    ImmediateEvent,
    PeriodicEvent,
    get_qualified_device,
    parse_device,
    parse_event,
    parse_range,
    parse_request,
)
from pacsys.drf_utils import ensure_immediate_event, prepare_for_write

_NO_RANGE = parse_range(None)


@pytest.mark.parametrize(
    ("drf", "expected_parts", "expected_canonical", "expected_qualified"),
    [
        (
            "N:I2B1RI",
            ("N:I2B1RI", DRF_PROPERTY.READING, _NO_RANGE, DRF_FIELD.SCALED, DefaultEvent()),
            "N:I2B1RI.READING",
            "N:I2B1RI",
        ),
        (
            "N_I2B1RI",
            ("N:I2B1RI", DRF_PROPERTY.SETTING, _NO_RANGE, DRF_FIELD.SCALED, DefaultEvent()),
            "N:I2B1RI.SETTING",
            "N_I2B1RI",
        ),
        (
            "N|I2B1RI",
            ("N:I2B1RI", DRF_PROPERTY.STATUS, _NO_RANGE, None, DefaultEvent()),
            "N:I2B1RI.STATUS",
            "N|I2B1RI",
        ),
        (
            "N:I2B1RI@p,500",
            ("N:I2B1RI", DRF_PROPERTY.READING, _NO_RANGE, DRF_FIELD.SCALED, PeriodicEvent("p,500", "P")),
            "N:I2B1RI.READING@p,500",
            "N:I2B1RI@p,500",
        ),
        (
            "N_I2B1RI@p,500",
            ("N:I2B1RI", DRF_PROPERTY.SETTING, _NO_RANGE, DRF_FIELD.SCALED, PeriodicEvent("p,500", "P")),
            "N:I2B1RI.SETTING@p,500",
            "N_I2B1RI@p,500",
        ),
        (
            "N:I2B1RI[:]@p,500",
            (
                "N:I2B1RI",
                DRF_PROPERTY.READING,
                ARRAY_RANGE("full", None, None),
                DRF_FIELD.SCALED,
                PeriodicEvent("p,500", "P"),
            ),
            "N:I2B1RI.READING[:]@p,500",
            "N:I2B1RI[:]@p,500",
        ),
        (
            "N:I2B1RI[]@p,500",
            (
                "N:I2B1RI",
                DRF_PROPERTY.READING,
                ARRAY_RANGE("full", None, None),
                DRF_FIELD.SCALED,
                PeriodicEvent("p,500", "P"),
            ),
            "N:I2B1RI.READING[:]@p,500",
            "N:I2B1RI[:]@p,500",
        ),
        (
            "N:I2B1RI[:2048]@I",
            (
                "N:I2B1RI",
                DRF_PROPERTY.READING,
                ARRAY_RANGE("std", None, 2048),
                DRF_FIELD.SCALED,
                ImmediateEvent("I", "I"),
            ),
            "N:I2B1RI.READING[:2048]@I",
            "N:I2B1RI[:2048]@I",
        ),
        (
            "N:I2B1RI.SETTING[50:]@I",
            (
                "N:I2B1RI",
                DRF_PROPERTY.SETTING,
                ARRAY_RANGE("std", 50, None),
                DRF_FIELD.SCALED,
                ImmediateEvent("I", "I"),
            ),
            "N:I2B1RI.SETTING[50:]@I",
            "N_I2B1RI[50:]@I",
        ),
        (
            "N_I2B1RI.SETTING[50:]@I",
            (
                "N:I2B1RI",
                DRF_PROPERTY.SETTING,
                ARRAY_RANGE("std", 50, None),
                DRF_FIELD.SCALED,
                ImmediateEvent("I", "I"),
            ),
            "N:I2B1RI.SETTING[50:]@I",
            "N_I2B1RI[50:]@I",
        ),
        (
            "N_I2B1RI.SETTING[50]@e,AE,e,1000",
            (
                "N:I2B1RI",
                DRF_PROPERTY.SETTING,
                ARRAY_RANGE("single", 50, None),
                DRF_FIELD.SCALED,
                ClockEvent("e,AE,e,1000", "E"),
            ),
            "N:I2B1RI.SETTING[50]@e,AE,e,1000",
            "N_I2B1RI[50]@e,AE,e,1000",
        ),
        (
            "N_I2B1RI.SETTING[50].RAW@e,AE,e,1000",
            (
                "N:I2B1RI",
                DRF_PROPERTY.SETTING,
                ARRAY_RANGE("single", 50, None),
                DRF_FIELD.RAW,
                ClockEvent("e,AE,e,1000", "E"),
            ),
            "N:I2B1RI.SETTING[50].RAW@e,AE,e,1000",
            "N_I2B1RI[50].RAW@e,AE,e,1000",
        ),
        (
            "Z:CACHE[50:]",
            ("Z:CACHE", DRF_PROPERTY.READING, ARRAY_RANGE("std", 50, None), DRF_FIELD.SCALED, DefaultEvent()),
            "Z:CACHE.READING[50:]",
            "Z:CACHE[50:]",
        ),
        (
            "E:TRTGTD@e,AE,e,1000",
            ("E:TRTGTD", DRF_PROPERTY.READING, _NO_RANGE, DRF_FIELD.SCALED, ClockEvent("e,AE,e,1000", "E")),
            "E:TRTGTD.READING@e,AE,e,1000",
            "E:TRTGTD@e,AE,e,1000",
        ),
        # Periodic with Hz unit suffix (100H = 100 Hz)
        (
            "M:OUTTMP@p,100H",
            ("M:OUTTMP", DRF_PROPERTY.READING, _NO_RANGE, DRF_FIELD.SCALED, PeriodicEvent("p,100H", "P")),
            "M:OUTTMP.READING@p,100H",
            "M:OUTTMP@p,100H",
        ),
        # Periodic with seconds unit suffix (2S = 2 seconds = 2000ms)
        (
            "M:OUTTMP@p,2S",
            ("M:OUTTMP", DRF_PROPERTY.READING, _NO_RANGE, DRF_FIELD.SCALED, PeriodicEvent("p,2S", "P")),
            "M:OUTTMP.READING@p,2S",
            "M:OUTTMP@p,2S",
        ),
    ],
)
def test_drf_parse(drf, expected_parts, expected_canonical, expected_qualified):
    result = parse_request(drf)
    assert result.parts[:-1] == expected_parts[:-1]
    assert type(result.event) is type(expected_parts[-1])
    assert vars(result.event) == vars(expected_parts[-1])
    assert result.to_canonical() == expected_canonical
    assert result.to_qualified() == expected_qualified


@pytest.mark.parametrize(
    ("drf", "expected_canonical"),
    [
        ("N:I2B1RI", "N:I2B1RI"),
        ("N_I2B1RI", "N:I2B1RI"),
        # Qualifier characters ^, #, ! must be recognized as ACNET names
        ("M^OUTTMP", "M:OUTTMP"),
        ("M#OUTTMP", "M:OUTTMP"),
        ("M!OUTTMP", "M:OUTTMP"),
    ],
)
def test_drf_device_parse(drf, expected_canonical):
    dev = parse_device(drf)
    assert dev.canonical_string == expected_canonical
    assert dev.is_acnet


def test_parse_device_epics_passthrough():
    # Non-ACNET names pass through verbatim; full DRFs are not device names
    dev = parse_device("N_I2B1RI@p,1000")
    assert dev.canonical_string == "N_I2B1RI@p,1000"
    assert not dev.is_acnet


def test_get_qualified_device():
    assert get_qualified_device("N:I2B1RI", DRF_PROPERTY.SETTING) == "N_I2B1RI"


def test_get_qualified_device_rejects_epics():
    with pytest.raises(ValueError, match="non-ACNET"):
        get_qualified_device("LINAC:CAV1:PHASE", DRF_PROPERTY.SETTING)


@pytest.mark.parametrize(
    "drf",
    [
        "M:OUTTMP.READING.RAW[0:2]",  # field before range
        "M:OUTTMP.",
        "M:OUTTMP@",
        "M:OUTTMP..READING",
        "M:OUTTMP[-1:5]",  # negative range rejected, must not become device name
        "M:OUTTMP{bad}",  # malformed byte range, brace group closes the DRF
        "Z:ACLTST{-1:20}",
        "M:OUTTMP{0:2O}.RAW",  # letter O in length, followed by a field
        "M:OUTTMP{bad}@I",  # followed by an event
    ],
)
def test_parse_request_rejects_acnet_near_miss(drf):
    with pytest.raises(ValueError, match="Invalid ACNET DRF"):
        parse_request(drf)


@pytest.mark.parametrize(
    "drf",
    [
        "INVALID!",
        "LINAC:CAV1:PHASE",
        "temperature:water",
        "SR:C01-MG:G02A{HFCor:FM1}Fld:SP",
        "XF:31IDA-OP{Tbl-Ax:X1}Mtr",
        "X:31IDA-OP{Tbl-Ax:X1}Mtr",  # ACNET-shaped prefix but braces -> EPICS
    ],
)
def test_parse_request_epics_fallback(drf):
    req = parse_request(drf)
    assert req.device == drf
    assert not req.is_acnet


def test_parse_request_is_acnet_flag():
    assert parse_request("M:OUTTMP.READING[0:2].RAW").is_acnet
    assert parse_request("Z:ACLTST<-REDIR:N@UALL").is_acnet


@pytest.mark.parametrize(
    ("drf", "fields"),
    [
        ("pv:name.VAL", ("VAL", None)),  # common EPICS record fields must parse
        ("pv:name.val", ("val", None)),  # case preserved
        ("pv:name.value.alarm", ("value", "alarm")),
        ("XF:31IDA-OP{Tbl-Ax:X1}Mtr.RBV", ("RBV", None)),
        ("pv:name[0:5]", (None, None)),
    ],
)
def test_parse_request_epics_suffix_verbatim(drf, fields):
    """EPICS dot-suffixes are raw server-side field slots, never ACNET properties."""
    req = parse_request(drf)
    assert not req.is_acnet
    assert req.epics_fields == fields
    assert req.to_canonical() == drf  # round-trip, no synthesized .READING


def test_epics_to_canonical_rejects_acnet_overrides():
    req = parse_request("LINAC:CAV1:PHASE")
    with pytest.raises(ValueError, match="non-ACNET"):
        req.to_canonical(property=DRF_PROPERTY.SETTING)


def test_to_qualified_rejects_epics():
    with pytest.raises(ValueError, match="non-ACNET"):
        parse_request("LINAC:CAV1:PHASE").to_qualified()
    with pytest.raises(ValueError, match="non-ACNET"):
        parse_request("LINAC:CAV1:PHASE").name_as(DRF_PROPERTY.SETTING)


@pytest.mark.parametrize(
    ("drf", "expected"),
    [
        ("M:OUTTMP", "M:OUTTMP.READING@I"),
        ("B:HS23T[0:10]", "B:HS23T.READING[0:10]@I"),
        ("M@UTEST", "M:UTEST.ANALOG@I"),
        ("M@UTEST@U", "M:UTEST.ANALOG@I"),
        ("M:OUTTMP@p,1000", "M:OUTTMP@p,1000"),
        ("M:OUTTMP@p,100H", "M:OUTTMP@p,100H"),
        ("M:OUTTMP@E,0F", "M:OUTTMP@E,0F"),
        ("M:OUTTMP@I", "M:OUTTMP@I"),
        ("M:OUTTMP<-FTP", "M:OUTTMP.READING@I<-FTP"),
        ("Z:ACLTST<-REDIR:N@UALL", "Z:ACLTST.READING@I<-REDIR:N@UALL"),
        ("M:OUTTMP<-LOGGER", "M:OUTTMP<-LOGGER"),
        ("M:OUTTMP@p,100H<-FTP", "M:OUTTMP@p,100H<-FTP"),
        # EPICS: append @I (server would otherwise subscribe), never inject .READING
        ("SR:BPM:01:X", "SR:BPM:01:X@I"),
        ("pv:name.VAL", "pv:name.VAL@I"),
        ("LINAC:CAV1:PHASE@p,1000", "LINAC:CAV1:PHASE@p,1000"),
    ],
)
def test_ensure_immediate_event(drf, expected):
    assert ensure_immediate_event(drf) == expected


@pytest.mark.parametrize(
    ("event", "event_type"),
    [
        ("u", DefaultEvent),
        ("i", ImmediateEvent),
    ],
)
def test_parse_simple_event_is_case_insensitive(event, event_type):
    assert isinstance(parse_event(event), event_type)


@pytest.mark.parametrize("event", ["Ujunk", "Ifoo"])
def test_parse_simple_event_rejects_trailing_text(event):
    with pytest.raises(ValueError, match=f"Invalid event: {event}"):
        parse_event(event)


@pytest.mark.parametrize(
    ("raw", "expected_ms"),
    [
        ("500", 500),  # default = ms
        ("1000M", 1000),  # explicit ms
        ("2S", 2000),  # seconds
        ("500U", 1),  # 500 us -> 0.5ms -> java_round = 1ms
        ("1500U", 2),  # 1500 us -> 1.5ms -> java_round = 2ms
        ("1U", 0),  # 1 us -> round(0.001) = 0ms
        ("100H", 10),  # 100 Hz = 10ms
        ("10H", 100),  # 10 Hz = 100ms
        ("60H", 17),  # 60 Hz -> round(16.667) = 17ms
        ("1K", 1),  # 1 kHz = 1ms
        ("3K", 0),  # 3 kHz -> round(0.333) = 0ms
        ("0H", 0),  # zero is always 0ms
    ],
)
def test_parse_time_freq(raw, expected_ms):
    from pacsys.drf3.event import _parse_time_freq

    assert _parse_time_freq(raw) == expected_ms


@pytest.mark.parametrize(
    ("drf", "expected"),
    [
        ("M:OUTTMP", "M:OUTTMP.SETTING@N"),
        ("M:OUTTMP.READING.RAW", "M:OUTTMP.SETTING.RAW@N"),
        ("M:OUTTMP.STATUS", "M:OUTTMP.CONTROL@N"),
        ("M:OUTTMP.STATUS.ON", "M:OUTTMP.CONTROL@N"),
        ("M_OUTTMP", "M:OUTTMP.SETTING@N"),
        # EPICS: write the PV itself - no property mapping, just @N
        ("SR:BPM:01:X", "SR:BPM:01:X@N"),
        ("pv:name.VAL", "pv:name.VAL@N"),
    ],
)
def test_prepare_for_write(drf, expected):
    assert prepare_for_write(drf) == expected


@pytest.mark.parametrize(
    ("drf", "expected"),
    [
        ("M:OUTTMP", False),  # parser default-fills SCALED
        ("M:OUTTMP.READING", False),  # explicit property, no field
        ("M:OUTTMP.READING.RAW", True),
        ("M:OUTTMP.RAW", True),  # bare field reinterpreted from property slot
        ("M:OUTTMP.STATUS.ON", True),
        ("Z|ACLTST", False),  # qualifier sets property, not field
    ],
)
def test_field_explicit(drf, expected):
    assert parse_request(drf).field_explicit is expected
