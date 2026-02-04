"""Tests for DevDB client, dataclasses, cache, and Device API integration."""

import time
from unittest import mock

import pytest

from pacsys.devdb import (
    DevDBClient,
    DeviceInfoResult,
    PropertyInfo,
    StatusBitDef,
    ExtStatusBitDef,
    ControlCommandDef,
    _TTLCache,
    _convert_device_info,
    _convert_property,
    _convert_status_bit,
    _convert_ext_status_bit,
    _convert_control_cmd,
)
from pacsys.digital_status import DigitalStatus
from pacsys.errors import DeviceError
from pacsys.types import Reading, ValueType


# ─── Test data (matches Z:ACLTST from real DevDB) ────────────────────────────

ACLTST_STATUS_BITS = (
    StatusBitDef(
        mask=1,
        match=1,
        invert=False,
        short_name="On",
        long_name="On/Off",
        true_str="On",
        false_str="Off",
        true_color=1,
        true_char="",
        false_color=2,
        false_char="",
    ),
    StatusBitDef(
        mask=2,
        match=2,
        invert=False,
        short_name="Ready",
        long_name="Ready/Tripped",
        true_str="Ready",
        false_str="Tripped",
        true_color=1,
        true_char="",
        false_color=2,
        false_char="",
    ),
    StatusBitDef(
        mask=4,
        match=4,
        invert=False,
        short_name="Remote",
        long_name="Remote/Local",
        true_str="Remote",
        false_str="Local",
        true_color=1,
        true_char="",
        false_color=2,
        false_char="",
    ),
    StatusBitDef(
        mask=8,
        match=8,
        invert=False,
        short_name="Polarity",
        long_name="Polarity",
        true_str="Plus",
        false_str="Minus",
        true_color=1,
        true_char="",
        false_color=2,
        false_char="",
    ),
)

ACLTST_EXT_STATUS_BITS = (ExtStatusBitDef(bit_no=0, description="Power", name0="Off", name1="On", color0=2, color1=1),)

ACLTST_CONTROLS = (
    ControlCommandDef(value=0, short_name="Reset", long_name="Reset device"),
    ControlCommandDef(value=1, short_name="On", long_name="Turn on"),
    ControlCommandDef(value=2, short_name="Off", long_name="Turn off"),
)


def _make_info(**overrides) -> DeviceInfoResult:
    """Helper to build a DeviceInfoResult with sensible defaults."""
    defaults = dict(
        device_index=140013,
        description="ACL test device!",
        reading=PropertyInfo(
            primary_units="cnt",
            common_units="blip",
            min_val=-500.0,
            max_val=500.0,
            p_index=1,
            c_index=1,
            coeff=(0.0, 1.0),
            is_step_motor=False,
            is_destructive_read=False,
            is_fe_scaling=False,
            is_contr_setting=False,
            is_knobbable=True,
        ),
        setting=PropertyInfo(
            primary_units="cnt",
            common_units="blip",
            min_val=-500.0,
            max_val=500.0,
            p_index=1,
            c_index=1,
            coeff=(0.0, 1.0),
            is_step_motor=False,
            is_destructive_read=False,
            is_fe_scaling=False,
            is_contr_setting=False,
            is_knobbable=True,
        ),
        control=ACLTST_CONTROLS,
        status_bits=ACLTST_STATUS_BITS,
        ext_status_bits=ACLTST_EXT_STATUS_BITS,
    )
    defaults.update(overrides)
    return DeviceInfoResult(**defaults)


# ─── TTL Cache Tests ─────────────────────────────────────────────────────────


class TestTTLCache:
    def test_put_and_get(self):
        cache = _TTLCache(ttl=60.0)
        cache.put("key", "value")
        assert cache.get("key") == "value"

    def test_get_missing_returns_none(self):
        cache = _TTLCache(ttl=60.0)
        assert cache.get("missing") is None

    def test_expiry(self):
        cache = _TTLCache(ttl=0.01)
        cache.put("key", "value")
        time.sleep(0.02)
        assert cache.get("key") is None

    def test_clear_all(self):
        cache = _TTLCache(ttl=60.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_clear_specific_key(self):
        cache = _TTLCache(ttl=60.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear("a")
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_overwrite(self):
        cache = _TTLCache(ttl=60.0)
        cache.put("key", "old")
        cache.put("key", "new")
        assert cache.get("key") == "new"


# ─── Proto Conversion Tests ──────────────────────────────────────────────────


class TestProtoConversion:
    """Tests for proto-to-dataclass conversion using mock proto objects."""

    def test_convert_property(self):
        proto = mock.MagicMock()
        proto.HasField.side_effect = lambda f: f in ("primary_units", "common_units")
        proto.primary_units = "cnt"
        proto.common_units = "blip"
        proto.min_val = -500.0
        proto.max_val = 500.0
        proto.p_index = 1
        proto.c_index = 1
        proto.coeff = [0.0, 1.0]
        proto.is_step_motor = False
        proto.is_fe_scaling = False
        proto.is_knobbable = True

        result = _convert_property(proto)
        assert isinstance(result, PropertyInfo)
        assert result.primary_units == "cnt"
        assert result.common_units == "blip"
        assert result.coeff == (0.0, 1.0)
        assert result.is_knobbable is True

    def test_convert_property_no_units(self):
        proto = mock.MagicMock()
        proto.HasField.return_value = False
        proto.min_val = 0.0
        proto.max_val = 100.0
        proto.p_index = 0
        proto.c_index = 0
        proto.coeff = []
        proto.is_step_motor = False
        proto.is_fe_scaling = False
        proto.is_knobbable = False

        result = _convert_property(proto)
        assert result.primary_units is None
        assert result.common_units is None

    def test_convert_status_bit(self):
        proto = mock.MagicMock()
        proto.mask_val = 1
        proto.match_val = 1
        proto.invert = False
        proto.short_name = "On"
        proto.long_name = "On/Off"
        proto.true_str = "On"
        proto.false_str = "Off"
        proto.true_color = 1
        proto.false_color = 2

        result = _convert_status_bit(proto)
        assert isinstance(result, StatusBitDef)
        assert result.mask == 1
        assert result.short_name == "On"
        assert result.true_str == "On"

    def test_convert_ext_status_bit(self):
        proto = mock.MagicMock()
        proto.bit_no = 0
        proto.description = "Power"
        proto.name0 = "Off"
        proto.name1 = "On"
        proto.color0 = 2
        proto.color1 = 1

        result = _convert_ext_status_bit(proto)
        assert isinstance(result, ExtStatusBitDef)
        assert result.bit_no == 0
        assert result.name1 == "On"

    def test_convert_control_cmd(self):
        proto = mock.MagicMock()
        proto.value = 1
        proto.short_name = "On"
        proto.long_name = "Turn on"

        result = _convert_control_cmd(proto)
        assert isinstance(result, ControlCommandDef)
        assert result.value == 1
        assert result.short_name == "On"

    def test_convert_device_info_full(self):
        """Full device info with all optional fields present."""
        status_bit = mock.MagicMock()
        status_bit.mask_val = 1
        status_bit.match_val = 1
        status_bit.invert = False
        status_bit.short_name = "On"
        status_bit.long_name = "On/Off"
        status_bit.true_str = "On"
        status_bit.false_str = "Off"
        status_bit.true_color = 1
        status_bit.false_color = 2

        ext_bit = mock.MagicMock()
        ext_bit.bit_no = 0
        ext_bit.description = "Power"
        ext_bit.name0 = "Off"
        ext_bit.name1 = "On"
        ext_bit.color0 = 2
        ext_bit.color1 = 1

        cmd = mock.MagicMock()
        cmd.value = 1
        cmd.short_name = "On"
        cmd.long_name = "Turn on"

        reading = mock.MagicMock()
        reading.HasField.side_effect = lambda f: f in ("primary_units",)
        reading.primary_units = "cnt"
        reading.common_units = "blip"
        reading.min_val = 0.0
        reading.max_val = 100.0
        reading.p_index = 0
        reading.c_index = 0
        reading.coeff = []
        reading.is_step_motor = False
        reading.is_fe_scaling = False
        reading.is_knobbable = True

        proto = mock.MagicMock()
        proto.device_index = 140013
        proto.description = "ACL test"
        proto.HasField.side_effect = lambda f: True
        proto.reading = reading
        proto.setting = reading
        proto.control.cmds = [cmd]
        proto.status.bits = [status_bit]
        proto.status.ext_bits = [ext_bit]

        result = _convert_device_info(proto)
        assert isinstance(result, DeviceInfoResult)
        assert result.device_index == 140013
        assert result.reading is not None
        assert result.setting is not None
        assert result.control is not None
        assert len(result.control) == 1
        assert result.status_bits is not None
        assert len(result.status_bits) == 1
        assert result.ext_status_bits is not None

    def test_convert_device_info_minimal(self):
        """Device info with no optional fields."""
        proto = mock.MagicMock()
        proto.device_index = 1
        proto.description = "Minimal"
        proto.HasField.return_value = False

        result = _convert_device_info(proto)
        assert result.reading is None
        assert result.setting is None
        assert result.control is None
        assert result.status_bits is None
        assert result.ext_status_bits is None


# ─── DigitalStatus.from_devdb_bits Tests ─────────────────────────────────────


class TestDigitalStatusFromDevdbBits:
    def test_basic_evaluation(self):
        """Test mask/match/invert evaluation with raw_value=0b10 (Ready set)."""
        status = DigitalStatus.from_devdb_bits(
            "Z:ACLTST",
            raw_value=0b0010,
            bit_defs=ACLTST_STATUS_BITS,
        )
        assert len(status) == 4
        assert status["On"].is_set is False
        assert status["On"].value == "Off"
        assert status["Ready"].is_set is True
        assert status["Ready"].value == "Ready"
        assert status["Remote"].is_set is False
        assert status["Remote"].value == "Local"
        assert status["Polarity"].is_set is False
        assert status["Polarity"].value == "Minus"

    def test_all_active(self):
        """All bits active."""
        status = DigitalStatus.from_devdb_bits(
            "Z:TEST",
            raw_value=0b1111,
            bit_defs=ACLTST_STATUS_BITS,
        )
        for bit in status:
            assert bit.is_set is True

    def test_none_active(self):
        """No bits active."""
        status = DigitalStatus.from_devdb_bits(
            "Z:TEST",
            raw_value=0b0000,
            bit_defs=ACLTST_STATUS_BITS,
        )
        for bit in status:
            assert bit.is_set is False

    def test_invert_logic(self):
        """Test that invert=True flips the evaluation."""
        inverted_bit = StatusBitDef(
            mask=1,
            match=1,
            invert=True,
            short_name="Inv",
            long_name="Inverted",
            true_str="True",
            false_str="False",
            true_color=1,
            true_char="",
            false_color=2,
            false_char="",
        )
        # raw has bit 0 set, but invert=True flips the result
        status = DigitalStatus.from_devdb_bits("T", raw_value=1, bit_defs=(inverted_bit,))
        assert status[0].is_set is False  # (1 & 1 == 1) ^ True = False
        assert status[0].value == "False"

        # raw has bit 0 clear
        status = DigitalStatus.from_devdb_bits("T", raw_value=0, bit_defs=(inverted_bit,))
        assert status[0].is_set is True  # (0 & 1 == 1) is False, ^ True = True
        assert status[0].value == "True"

    def test_legacy_inference(self):
        """Legacy on/ready/etc. fields are inferred from DevDB bit names."""
        status = DigitalStatus.from_devdb_bits(
            "Z:ACLTST",
            raw_value=0b0010,
            bit_defs=ACLTST_STATUS_BITS,
        )
        assert status.on is False
        assert status.ready is True
        assert status.remote is False
        assert status.positive is False

    def test_with_ext_bits(self):
        """ext_bit_defs are included in status bits."""
        status = DigitalStatus.from_devdb_bits(
            "Z:ACLTST",
            raw_value=0b0010,
            bit_defs=ACLTST_STATUS_BITS,
            ext_bit_defs=ACLTST_EXT_STATUS_BITS,
        )
        assert len(status) == 5  # 4 standard + 1 extended
        ext_bit = status.bits[-1]
        assert ext_bit.position == 0
        assert ext_bit.name == "Power"
        assert ext_bit.value == "Off"  # bit 0 is not set in 0b0010
        assert ext_bit.is_set is False

    def test_with_ext_bits_active(self):
        """ext_bit_defs evaluate correctly when bit is set."""
        status = DigitalStatus.from_devdb_bits(
            "Z:ACLTST",
            raw_value=0b0011,  # bit 0 set
            bit_defs=ACLTST_STATUS_BITS,
            ext_bit_defs=ACLTST_EXT_STATUS_BITS,
        )
        ext_bit = status.bits[-1]
        assert ext_bit.value == "On"  # bit 0 is set
        assert ext_bit.is_set is True


# ─── DevDBClient Construction Tests ──────────────────────────────────────────


class TestDevDBClientConstruction:
    def test_import_guard(self):
        """DevDBClient requires gRPC."""
        # We can construct it since grpc is available in test environment
        # Just verify the class exists and has expected attributes
        assert hasattr(DevDBClient, "get_device_info")
        assert hasattr(DevDBClient, "get_alarm_info")
        assert hasattr(DevDBClient, "get_alarm_text")
        assert hasattr(DevDBClient, "close")
        assert hasattr(DevDBClient, "clear_cache")

    def test_closed_client_raises(self):
        """Operations on a closed client raise RuntimeError."""
        client = DevDBClient(host="localhost", port=99999)
        client.close()
        with pytest.raises(RuntimeError, match="closed"):
            client.get_device_info(["Z:ACLTST"])

    def test_context_manager(self):
        """Context manager closes the client."""
        with DevDBClient(host="localhost", port=99999) as client:
            assert not client._closed
        assert client._closed

    def test_env_var_defaults(self, monkeypatch):
        """Environment variables are used for defaults."""
        monkeypatch.setenv("PACSYS_DEVDB_HOST", "devdb.example.com")
        monkeypatch.setenv("PACSYS_DEVDB_PORT", "12345")
        client = DevDBClient()
        assert client._host == "devdb.example.com"
        assert client._port == 12345
        client.close()

    def test_explicit_overrides_env(self, monkeypatch):
        """Explicit parameters override env vars."""
        monkeypatch.setenv("PACSYS_DEVDB_HOST", "env-host")
        monkeypatch.setenv("PACSYS_DEVDB_PORT", "11111")
        client = DevDBClient(host="explicit-host", port=22222)
        assert client._host == "explicit-host"
        assert client._port == 22222
        client.close()


# ─── DevDBClient get_device_info with Mock ───────────────────────────────────


class TestDevDBClientGetDeviceInfo:
    def _make_client_with_mock_stub(self):
        """Create a DevDBClient with a mocked gRPC stub."""
        client = DevDBClient(host="localhost", port=99999)
        client._stub = mock.MagicMock()
        return client

    def test_single_device_success(self):
        client = self._make_client_with_mock_stub()

        # Build mock reply
        entry = mock.MagicMock()
        entry.name = "Z:ACLTST"
        entry.WhichOneof.return_value = "device"

        device_info = entry.device
        device_info.device_index = 140013
        device_info.description = "ACL test device!"
        device_info.HasField.return_value = False  # no optional fields

        reply = mock.MagicMock()
        reply.set = [entry]
        client._stub.getDeviceInfo.return_value = reply

        result = client.get_device_info(["Z:ACLTST"])
        assert "Z:ACLTST" in result
        assert result["Z:ACLTST"].device_index == 140013
        assert result["Z:ACLTST"].description == "ACL test device!"
        client.close()

    def test_error_device_raises(self):
        client = self._make_client_with_mock_stub()

        entry = mock.MagicMock()
        entry.name = "X:BOGUS"
        entry.WhichOneof.return_value = "errMsg"
        entry.errMsg = "Device not found"

        reply = mock.MagicMock()
        reply.set = [entry]
        client._stub.getDeviceInfo.return_value = reply

        with pytest.raises(DeviceError, match="Device not found"):
            client.get_device_info(["X:BOGUS"])
        client.close()

    def test_cache_hit_avoids_rpc(self):
        client = self._make_client_with_mock_stub()

        # Build mock reply
        entry = mock.MagicMock()
        entry.name = "Z:ACLTST"
        entry.WhichOneof.return_value = "device"
        entry.device.device_index = 140013
        entry.device.description = "test"
        entry.device.HasField.return_value = False

        reply = mock.MagicMock()
        reply.set = [entry]
        client._stub.getDeviceInfo.return_value = reply

        # First call hits RPC
        client.get_device_info(["Z:ACLTST"])
        assert client._stub.getDeviceInfo.call_count == 1

        # Second call hits cache
        result = client.get_device_info(["Z:ACLTST"])
        assert client._stub.getDeviceInfo.call_count == 1
        assert "Z:ACLTST" in result
        client.close()

    def test_clear_cache_forces_rpc(self):
        client = self._make_client_with_mock_stub()

        entry = mock.MagicMock()
        entry.name = "Z:ACLTST"
        entry.WhichOneof.return_value = "device"
        entry.device.device_index = 140013
        entry.device.description = "test"
        entry.device.HasField.return_value = False

        reply = mock.MagicMock()
        reply.set = [entry]
        client._stub.getDeviceInfo.return_value = reply

        client.get_device_info(["Z:ACLTST"])
        client.clear_cache("Z:ACLTST")
        client.get_device_info(["Z:ACLTST"])
        assert client._stub.getDeviceInfo.call_count == 2
        client.close()

    def test_partial_cache_hit(self):
        """Mixed cached and uncached devices in single request."""
        client = self._make_client_with_mock_stub()

        # First query caches Z:ACLTST
        entry1 = mock.MagicMock()
        entry1.name = "Z:ACLTST"
        entry1.WhichOneof.return_value = "device"
        entry1.device.device_index = 140013
        entry1.device.description = "test"
        entry1.device.HasField.return_value = False

        reply1 = mock.MagicMock()
        reply1.set = [entry1]
        client._stub.getDeviceInfo.return_value = reply1
        client.get_device_info(["Z:ACLTST"])

        # Second query has Z:ACLTST cached, M:OUTTMP uncached
        entry2 = mock.MagicMock()
        entry2.name = "M:OUTTMP"
        entry2.WhichOneof.return_value = "device"
        entry2.device.device_index = 99
        entry2.device.description = "temp"
        entry2.device.HasField.return_value = False

        reply2 = mock.MagicMock()
        reply2.set = [entry2]
        client._stub.getDeviceInfo.return_value = reply2

        result = client.get_device_info(["Z:ACLTST", "M:OUTTMP"])
        assert "Z:ACLTST" in result
        assert "M:OUTTMP" in result

        # Second RPC should only request M:OUTTMP
        call_args = client._stub.getDeviceInfo.call_args[0][0]
        assert list(call_args.device) == ["M:OUTTMP"]
        client.close()


# ─── Device API Integration Tests ────────────────────────────────────────────


class TestDeviceInfoIntegration:
    def test_info_raises_without_devdb(self):
        """Device.info() raises RuntimeError when DevDB is not configured."""
        from pacsys.device import Device

        dev = Device("Z:ACLTST", backend=mock.MagicMock())
        with mock.patch("pacsys.device.Device._get_devdb", return_value=None):
            with pytest.raises(RuntimeError, match="DevDB not available"):
                dev.info()

    def test_info_delegates_to_devdb(self):
        """Device.info() queries DevDB and returns DeviceInfoResult."""
        from pacsys.device import Device

        expected = _make_info()
        mock_devdb = mock.MagicMock()
        mock_devdb.get_device_info.return_value = {"Z:ACLTST": expected}

        dev = Device("Z:ACLTST", backend=mock.MagicMock())
        with mock.patch("pacsys.device.Device._get_devdb", return_value=mock_devdb):
            result = dev.info()
        assert result is expected
        mock_devdb.get_device_info.assert_called_once_with(["Z:ACLTST"], timeout=None)


class TestDeviceDigitalStatusDevDB:
    def test_devdb_accelerated_path(self):
        """digital_status() uses 1-read path when DevDB has status_bits."""
        from pacsys.device import Device

        info = _make_info()
        mock_devdb = mock.MagicMock()
        mock_devdb.get_device_info.return_value = {"Z:ACLTST": info}

        mock_backend = mock.MagicMock()
        mock_backend.get.return_value = Reading(
            drf="Z:ACLTST.STATUS.BIT_VALUE@I",
            value_type=ValueType.SCALAR,
            value=2,  # 0b10 = Ready
            error_code=0,
        )

        dev = Device("Z:ACLTST", backend=mock_backend)
        with mock.patch("pacsys.device.Device._get_devdb", return_value=mock_devdb):
            status = dev.digital_status()

        # Should use get() (single read) not get_many() (batch)
        mock_backend.get.assert_called_once()
        mock_backend.get_many.assert_not_called()

        assert status.device == "Z:ACLTST"
        assert status.raw_value == 2
        assert status["On"].is_set is False
        assert status["Ready"].is_set is True

    def test_fallback_to_3read_when_no_devdb(self):
        """digital_status() falls back to 3-read path without DevDB."""
        from pacsys.device import Device

        mock_backend = mock.MagicMock()
        mock_backend.get_many.return_value = [
            Reading(drf="Z:ACLTST.STATUS.BIT_VALUE@I", value=2, value_type=ValueType.SCALAR),
            Reading(drf="Z:ACLTST.STATUS.BIT_NAMES@I", value=["On", "Ready"], value_type=ValueType.TEXT_ARRAY),
            Reading(drf="Z:ACLTST.STATUS.BIT_VALUES@I", value=["Off", "Yes"], value_type=ValueType.TEXT_ARRAY),
        ]

        dev = Device("Z:ACLTST", backend=mock_backend)
        with mock.patch("pacsys.device.Device._get_devdb", return_value=None):
            status = dev.digital_status()

        mock_backend.get_many.assert_called_once()
        assert status.raw_value == 2

    def test_fallback_on_devdb_error(self):
        """digital_status() falls back to 3-read path when DevDB raises."""
        from pacsys.device import Device

        mock_devdb = mock.MagicMock()
        mock_devdb.get_device_info.side_effect = Exception("DevDB down")

        mock_backend = mock.MagicMock()
        mock_backend.get_many.return_value = [
            Reading(drf="Z:ACLTST.STATUS.BIT_VALUE@I", value=2, value_type=ValueType.SCALAR),
            Reading(drf="Z:ACLTST.STATUS.BIT_NAMES@I", value=["On", "Ready"], value_type=ValueType.TEXT_ARRAY),
            Reading(drf="Z:ACLTST.STATUS.BIT_VALUES@I", value=["Off", "Yes"], value_type=ValueType.TEXT_ARRAY),
        ]

        dev = Device("Z:ACLTST", backend=mock_backend)
        with mock.patch("pacsys.device.Device._get_devdb", return_value=mock_devdb):
            status = dev.digital_status()

        # Should have fallen back to 3-read path
        mock_backend.get_many.assert_called_once()
        assert status.raw_value == 2

    def test_fallback_when_no_status_bits(self):
        """digital_status() falls back if DevDB has no status_bits for device."""
        from pacsys.device import Device

        info = _make_info(status_bits=None, ext_status_bits=None)
        mock_devdb = mock.MagicMock()
        mock_devdb.get_device_info.return_value = {"Z:ACLTST": info}

        mock_backend = mock.MagicMock()
        mock_backend.get_many.return_value = [
            Reading(drf="Z:ACLTST.STATUS.BIT_VALUE@I", value=2, value_type=ValueType.SCALAR),
            Reading(drf="Z:ACLTST.STATUS.BIT_NAMES@I", value=["On", "Ready"], value_type=ValueType.TEXT_ARRAY),
            Reading(drf="Z:ACLTST.STATUS.BIT_VALUES@I", value=["Off", "Yes"], value_type=ValueType.TEXT_ARRAY),
        ]

        dev = Device("Z:ACLTST", backend=mock_backend)
        with mock.patch("pacsys.device.Device._get_devdb", return_value=mock_devdb):
            dev.digital_status()

        mock_backend.get_many.assert_called_once()

    def test_backend_error_propagates_with_devdb(self):
        """Backend errors propagate (not swallowed) when using DevDB-accelerated path."""
        from pacsys.device import Device

        info = _make_info()
        mock_devdb = mock.MagicMock()
        mock_devdb.get_device_info.return_value = {"Z:ACLTST": info}

        mock_backend = mock.MagicMock()
        mock_backend.get.return_value = Reading(
            drf="Z:ACLTST.STATUS.BIT_VALUE@I",
            value_type=ValueType.SCALAR,
            value=None,
            error_code=-6,
            message="Timeout",
        )

        dev = Device("Z:ACLTST", backend=mock_backend)
        with mock.patch("pacsys.device.Device._get_devdb", return_value=mock_devdb):
            with pytest.raises(DeviceError, match="Timeout"):
                dev.digital_status()

        # Should NOT fall back to 3-read path
        mock_backend.get_many.assert_not_called()


# ─── Cache Key Normalization Tests ────────────────────────────────────────────


class TestCacheKeyNormalization:
    def test_case_insensitive_cache_hit(self):
        """Cache lookup is case-insensitive."""
        client = DevDBClient(host="localhost", port=99999)
        client._stub = mock.MagicMock()

        entry = mock.MagicMock()
        entry.name = "Z:ACLTST"
        entry.WhichOneof.return_value = "device"
        entry.device.device_index = 140013
        entry.device.description = "test"
        entry.device.HasField.return_value = False

        reply = mock.MagicMock()
        reply.set = [entry]
        client._stub.getDeviceInfo.return_value = reply

        # Query with uppercase
        client.get_device_info(["Z:ACLTST"])

        # Query with lowercase should hit cache
        result = client.get_device_info(["z:acltst"])
        assert "z:acltst" in result
        assert client._stub.getDeviceInfo.call_count == 1  # Only 1 RPC
        client.close()

    def test_clear_cache_case_insensitive(self):
        """clear_cache() works case-insensitively."""
        client = DevDBClient(host="localhost", port=99999)
        client._stub = mock.MagicMock()

        entry = mock.MagicMock()
        entry.name = "Z:ACLTST"
        entry.WhichOneof.return_value = "device"
        entry.device.device_index = 140013
        entry.device.description = "test"
        entry.device.HasField.return_value = False

        reply = mock.MagicMock()
        reply.set = [entry]
        client._stub.getDeviceInfo.return_value = reply

        client.get_device_info(["Z:ACLTST"])
        client.clear_cache("z:acltst")  # lowercase clear
        client.get_device_info(["Z:ACLTST"])  # should miss cache
        assert client._stub.getDeviceInfo.call_count == 2
        client.close()


# ─── Cache Max Size Tests ─────────────────────────────────────────────────────


class TestCacheMaxSize:
    def test_evicts_oldest_on_overflow(self):
        cache = _TTLCache(ttl=60.0, max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # should evict "a" (oldest)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4

    def test_max_size_respected(self):
        cache = _TTLCache(ttl=60.0, max_size=2)
        for i in range(100):
            cache.put(f"key{i}", i)
        # Should not grow beyond max_size + 1 (due to eviction after put)
        with cache._lock:
            assert len(cache._data) <= 2


# ─── Dataclass Immutability Tests ─────────────────────────────────────────────


class TestDataclassImmutability:
    def test_device_info_result_frozen(self):
        info = _make_info()
        with pytest.raises(AttributeError):
            info.description = "changed"  # type: ignore[misc]

    def test_property_info_frozen(self):
        prop = PropertyInfo(
            primary_units="cnt",
            common_units="blip",
            min_val=0.0,
            max_val=100.0,
            p_index=0,
            c_index=0,
            coeff=(),
            is_step_motor=False,
            is_destructive_read=False,
            is_fe_scaling=False,
            is_contr_setting=False,
            is_knobbable=False,
        )
        with pytest.raises(AttributeError):
            prop.min_val = 999.0  # type: ignore[misc]

    def test_status_bit_def_frozen(self):
        with pytest.raises(AttributeError):
            ACLTST_STATUS_BITS[0].mask = 99  # type: ignore[misc]

    def test_control_command_def_frozen(self):
        with pytest.raises(AttributeError):
            ACLTST_CONTROLS[0].value = 99  # type: ignore[misc]
