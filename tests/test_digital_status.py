"""Tests for DigitalStatus value object."""

import pytest

from pacsys.digital_status import DigitalStatus, StatusBit
from pacsys.types import Reading, ValueType


# --- Test data from resources/digital_status_control_notes.txt ---

# Z:ACLTST: raw=2 (0b10), bits: On(0), Ready(1), unused(2), Polarity(3)
ACLTST_NAMES = ["On", "Ready", "", "Polarity"]
ACLTST_VALUES = ["No", "Yes", "", "Minus"]
ACLTST_RAW = 2

# G:AMANDA: raw=9 (0b1001), 6 bits defined
AMANDA_NAMES = ["Henk On/Off", "Ready???", "Remote Henk", "Polarity", " test 2", "testtest"]
AMANDA_VALUES = ["On", "Nope", "L", "Bi", " good", "GOOD"]
AMANDA_RAW = 9


class TestStatusBit:
    def test_bool_true(self):
        bit = StatusBit(position=1, name="Ready", value="Yes", is_set=True)
        assert bool(bit) is True

    def test_bool_false(self):
        bit = StatusBit(position=0, name="On", value="No", is_set=False)
        assert bool(bit) is False

    def test_frozen(self):
        bit = StatusBit(position=0, name="On", value="Off", is_set=False)
        with pytest.raises(AttributeError):
            bit.name = "Changed"


class TestFromBitArrays:
    def test_acltst(self):
        status = DigitalStatus.from_bit_arrays(
            "Z:ACLTST",
            ACLTST_RAW,
            ACLTST_NAMES,
            ACLTST_VALUES,
        )
        assert status.device == "Z:ACLTST"
        assert status.raw_value == 2
        # 3 defined bits (bit 2 is empty → skipped)
        assert len(status) == 3

    def test_bit_positions(self):
        status = DigitalStatus.from_bit_arrays(
            "Z:ACLTST",
            ACLTST_RAW,
            ACLTST_NAMES,
            ACLTST_VALUES,
        )
        assert status.bits[0].position == 0
        assert status.bits[0].name == "On"
        assert status.bits[0].is_set is False  # bit 0 of 0b10 = 0

        assert status.bits[1].position == 1
        assert status.bits[1].name == "Ready"
        assert status.bits[1].is_set is True  # bit 1 of 0b10 = 1

        assert status.bits[2].position == 3
        assert status.bits[2].name == "Polarity"
        assert status.bits[2].is_set is False  # bit 3 of 0b10 = 0

    def test_bit_values_text(self):
        status = DigitalStatus.from_bit_arrays(
            "Z:ACLTST",
            ACLTST_RAW,
            ACLTST_NAMES,
            ACLTST_VALUES,
        )
        assert status.bits[0].value == "No"
        assert status.bits[1].value == "Yes"
        assert status.bits[2].value == "Minus"

    def test_amanda_six_bits(self):
        status = DigitalStatus.from_bit_arrays(
            "G:AMANDA",
            AMANDA_RAW,
            AMANDA_NAMES,
            AMANDA_VALUES,
        )
        assert len(status) == 6
        assert status.raw_value == 9  # 0b1001

        # bit 0 set, bit 3 set
        assert status.bits[0].is_set is True  # bit 0 of 0b1001
        assert status.bits[1].is_set is False  # bit 1
        assert status.bits[2].is_set is False  # bit 2
        assert status.bits[3].is_set is True  # bit 3

    def test_skips_empty_bits(self):
        names = ["A", "", "", "D"]
        values = ["val_a", "", "", "val_d"]
        status = DigitalStatus.from_bit_arrays("TEST", 0b1001, names, values)
        assert len(status) == 2
        assert status.bits[0].position == 0
        assert status.bits[1].position == 3

    def test_mismatched_lengths_pads(self):
        """bit_names shorter than bit_values — should not crash."""
        status = DigitalStatus.from_bit_arrays(
            "TEST",
            0b11,
            ["A"],
            ["val_a", "val_b"],
        )
        assert len(status) == 2
        assert status.bits[1].name == ""
        assert status.bits[1].value == "val_b"


class TestLegacyInference:
    def test_on_inferred_from_name(self):
        status = DigitalStatus.from_bit_arrays(
            "Z:ACLTST",
            ACLTST_RAW,
            ACLTST_NAMES,
            ACLTST_VALUES,
        )
        assert status.on is False  # bit 0 not set
        assert status.ready is True  # bit 1 set

    def test_polarity_inferred(self):
        status = DigitalStatus.from_bit_arrays(
            "Z:ACLTST",
            ACLTST_RAW,
            ACLTST_NAMES,
            ACLTST_VALUES,
        )
        assert status.positive is False  # "Polarity" matches positive pattern

    def test_non_matching_names_stay_none(self):
        status = DigitalStatus.from_bit_arrays(
            "TEST",
            0xFF,
            ["FooBar", "Baz"],
            ["X", "Y"],
        )
        assert status.on is None
        assert status.ready is None
        assert status.remote is None


class TestFromStatusDict:
    def test_legacy_bool_format(self):
        """PC/DMQ format: lowercase keys, bool values."""
        d = {"on": True, "ready": False}
        status = DigitalStatus.from_status_dict("Z:TEST", d)
        assert status.on is True
        assert status.ready is False
        assert status.remote is None
        assert len(status) == 2

    def test_legacy_all_five(self):
        d = {"on": True, "ready": True, "remote": False, "positive": True, "ramp": False}
        status = DigitalStatus.from_status_dict("Z:TEST", d)
        assert status.on is True
        assert status.ready is True
        assert status.remote is False
        assert status.positive is True
        assert status.ramp is False
        assert len(status) == 5

    def test_grpc_string_format(self):
        """gRPC format: display-name keys, text values."""
        d = {"On": "No", "Ready": "Yes", "Polarity": "Minus"}
        status = DigitalStatus.from_status_dict("Z:TEST", d)
        assert len(status) == 3
        assert status.bits[0].name == "On"
        assert status.bits[0].value == "No"
        assert status.on is False  # "No" → not set
        assert status.ready is True  # "Yes" → set

    def test_raw_value_passthrough(self):
        d = {"on": True}
        status = DigitalStatus.from_status_dict("T", d, raw_value=0xAB)
        assert status.raw_value == 0xAB

    def test_raw_value_reconstructed(self):
        d = {"on": True, "ready": False}
        status = DigitalStatus.from_status_dict("T", d)
        # on=True at position 0 → raw bit 0 set
        assert status.raw_value == 1

    def test_empty_dict(self):
        status = DigitalStatus.from_status_dict("T", {})
        assert len(status) == 0
        assert status.raw_value == 0


class TestFromReading:
    def test_basic_status_reading(self):
        reading = Reading(
            drf="Z:ACLTST.STATUS",
            value_type=ValueType.BASIC_STATUS,
            value={"on": True, "ready": False},
        )
        status = DigitalStatus.from_reading(reading)
        assert status.on is True
        assert status.ready is False

    def test_wrong_value_type_raises(self):
        reading = Reading(
            drf="Z:ACLTST",
            value_type=ValueType.SCALAR,
            value=42.0,
        )
        with pytest.raises(ValueError, match="Expected BASIC_STATUS"):
            DigitalStatus.from_reading(reading)

    def test_none_value_raises(self):
        reading = Reading(
            drf="Z:ACLTST.STATUS",
            value_type=ValueType.BASIC_STATUS,
            value=None,
        )
        with pytest.raises(ValueError, match="no value"):
            DigitalStatus.from_reading(reading)


class TestLookup:
    @pytest.fixture()
    def status(self):
        return DigitalStatus.from_bit_arrays(
            "Z:ACLTST",
            ACLTST_RAW,
            ACLTST_NAMES,
            ACLTST_VALUES,
        )

    def test_lookup_by_name(self, status):
        bit = status["Ready"]
        assert bit.name == "Ready"
        assert bit.is_set is True

    def test_lookup_by_name_case_insensitive(self, status):
        bit = status["ready"]
        assert bit.name == "Ready"

    def test_lookup_by_position(self, status):
        bit = status[0]
        assert bit.name == "On"

    def test_lookup_missing_name_raises(self, status):
        with pytest.raises(KeyError):
            status["Nonexistent"]

    def test_lookup_missing_position_raises(self, status):
        with pytest.raises(IndexError):
            status[2]  # bit 2 is undefined for Z:ACLTST

    def test_get_with_default(self, status):
        assert status.get("Nonexistent") is None
        assert status.get(99) is None
        assert status.get("Ready") is not None

    def test_contains(self, status):
        assert "Ready" in status
        assert "Nonexistent" not in status
        assert 0 in status
        assert 2 not in status


class TestIteration:
    def test_iter(self):
        status = DigitalStatus.from_bit_arrays(
            "Z:ACLTST",
            ACLTST_RAW,
            ACLTST_NAMES,
            ACLTST_VALUES,
        )
        names = [b.name for b in status]
        assert names == ["On", "Ready", "Polarity"]

    def test_len(self):
        status = DigitalStatus.from_bit_arrays(
            "G:AMANDA",
            AMANDA_RAW,
            AMANDA_NAMES,
            AMANDA_VALUES,
        )
        assert len(status) == 6


class TestDisplay:
    def test_str_format(self):
        status = DigitalStatus.from_bit_arrays(
            "Z:ACLTST",
            ACLTST_RAW,
            ACLTST_NAMES,
            ACLTST_VALUES,
        )
        text = str(status)
        assert "Z:ACLTST" in text
        assert "0x02" in text
        assert "On:" in text
        assert "Ready:" in text
        assert "Polarity:" in text
        assert "No" in text
        assert "Yes" in text
        assert "Minus" in text

    def test_to_dict(self):
        status = DigitalStatus.from_bit_arrays(
            "Z:ACLTST",
            ACLTST_RAW,
            ACLTST_NAMES,
            ACLTST_VALUES,
        )
        d = status.to_dict()
        assert d == {"On": "No", "Ready": "Yes", "Polarity": "Minus"}


class TestEdgeCases:
    def test_all_bits_set(self):
        names = ["A", "B", "C", "D"]
        values = ["1", "1", "1", "1"]
        status = DigitalStatus.from_bit_arrays("T", 0xF, names, values)
        assert all(b.is_set for b in status)

    def test_no_bits_set(self):
        names = ["A", "B"]
        values = ["0", "0"]
        status = DigitalStatus.from_bit_arrays("T", 0, names, values)
        assert not any(b.is_set for b in status)

    def test_32bit_value(self):
        names = [f"bit{i}" for i in range(32)]
        values = [f"v{i}" for i in range(32)]
        status = DigitalStatus.from_bit_arrays("T", 0xDEADBEEF, names, values)
        assert len(status) == 32
        assert status.raw_value == 0xDEADBEEF
        assert status[0].is_set is True  # bit 0 of 0xDEADBEEF = 1
        assert status[1].is_set is True  # bit 1 = 1
        assert status[4].is_set is False  # bit 4 = 0

    def test_frozen(self):
        status = DigitalStatus.from_bit_arrays("T", 0, ["A"], ["B"])
        with pytest.raises(AttributeError):
            status.raw_value = 99
