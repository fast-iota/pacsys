import pytest

from pacsys.drf_utils import ensure_immediate_event
from pacsys.drf3 import (
    ARRAY_RANGE,
    ClockEvent,
    DRF_FIELD,
    DRF_PROPERTY,
    DefaultEvent,
    ImmediateEvent,
    PeriodicEvent,
    get_qualified_device,
    parse_device,
    parse_range,
    parse_request,
)

_NO_RANGE = parse_range(None)


@pytest.mark.parametrize(
    "drf,expected_parts,expected_canonical,expected_qualified",
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
    assert result.parts == expected_parts
    assert result.to_canonical() == expected_canonical
    assert result.to_qualified() == expected_qualified


@pytest.mark.parametrize(
    "drf,expected_canonical",
    [
        ("N:I2B1RI", "N:I2B1RI"),
        ("N_I2B1RI", "N:I2B1RI"),
        ("N:I2B1RI@p,1000", "N:I2B1RI@p,1000"),
        ("N_I2B1RI@p,1000", "N:I2B1RI@p,1000"),
        # Qualifier characters ^, #, ! must be recognized as ACNET names
        ("M^OUTTMP", "M:OUTTMP"),
        ("M#OUTTMP", "M:OUTTMP"),
        ("M!OUTTMP", "M:OUTTMP"),
    ],
)
def test_drf_device_parse(drf, expected_canonical):
    assert parse_device(drf).canonical_string == expected_canonical


def test_get_qualified_device():
    assert get_qualified_device("N:I2B1RI", DRF_PROPERTY.SETTING) == "N_I2B1RI"


@pytest.mark.parametrize(
    "drf,expected",
    [
        ("M:OUTTMP", "M:OUTTMP@I"),
        ("B:HS23T[0:10]", "B:HS23T[0:10]@I"),
        ("M:OUTTMP@p,1000", "M:OUTTMP@p,1000"),
        ("M:OUTTMP@p,100H", "M:OUTTMP@p,100H"),
        ("M:OUTTMP@E,0F", "M:OUTTMP@E,0F"),
        ("M:OUTTMP@I", "M:OUTTMP@I"),
        ("M:OUTTMP<-FTP", "M:OUTTMP@I<-FTP"),
        ("M:OUTTMP@p,100H<-FTP", "M:OUTTMP@p,100H<-FTP"),
    ],
)
def test_ensure_immediate_event(drf, expected):
    assert ensure_immediate_event(drf) == expected


@pytest.mark.parametrize(
    "raw,expected_ms",
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
