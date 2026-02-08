"""
Unit tests for ACLBackend.

Tests cover:
- Backend initialization and capabilities
- ACL output line parsing (DEVICE = VALUE UNITS format)
- Error detection (ACL error codes, ! prefix)
- URL construction (single and batch with \\; separator)
- Single device read/get
- Multiple device read with batch fallback
- HTTP error handling
- Timeout handling
"""

from unittest import mock

import httpx
import pytest

from pacsys.backends.acl import (
    ACLBackend,
    _acl_read_command,
    _is_basic_status_request,
    _parse_acl_line,
    _parse_raw_hex,
    _is_error_response,
)
from pacsys.errors import DeviceError
from pacsys.types import Reading, ValueType
from tests.conftest import MockACLResponse


class TestACLBackendInit:
    """Tests for ACLBackend input validation."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"base_url": ""}, "base_url cannot be empty"),
            ({"timeout": 0}, "timeout must be positive"),
            ({"timeout": -1.0}, "timeout must be positive"),
        ],
    )
    def test_invalid_init_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ACLBackend(**kwargs)


class TestParseACLLine:
    """Tests for _parse_acl_line - parses full ACL output lines."""

    @pytest.mark.parametrize(
        "line,expected_value,expected_type",
        [
            # Scalar with device name and units
            ("M:OUTTMP       =  12.34 DegF", 12.34, ValueType.SCALAR),
            ("G:AMANDA       =  66", 66.0, ValueType.SCALAR),
            # Alarm fields (no_name/no_units ignored by ACL)
            ("Z:ACLTST alarm maximum = 50 blip", 50.0, ValueType.SCALAR),
            ("Z:ACLTST alarm minimum = -4.007 DegF", -4.007, ValueType.SCALAR),
            # Description (text after =)
            (
                "M:OUTTMP = Outdoor temperature (F)",
                "Outdoor temperature (F)",
                ValueType.TEXT,
            ),
            # Bare numeric (no = sign, e.g. from no_name/no_units)
            ("  12.68", 12.68, ValueType.SCALAR),
            ("42", 42.0, ValueType.SCALAR),
            ("-3.14", -3.14, ValueType.SCALAR),
            ("1.23e-4", pytest.approx(1.23e-4), ValueType.SCALAR),
            # Array (all-numeric tokens)
            ("45  2.2  2  102.81933", [45, 2.2, 2, 102.81933], ValueType.SCALAR_ARRAY),
            # Array with units (all-but-last numeric)
            ("45  2.2  3.0 blip", [45.0, 2.2, 3.0], ValueType.SCALAR_ARRAY),
            # Pure text
            ("Hello World", "Hello World", ValueType.TEXT),
        ],
    )
    def test_parse_acl_line(self, line, expected_value, expected_type):
        value, vtype = _parse_acl_line(line)
        assert value == expected_value
        assert vtype == expected_type


class TestIsErrorResponse:
    """Tests for error detection."""

    def test_exclamation_prefix(self):
        is_error, msg = _is_error_response("! Device not found")
        assert is_error is True
        assert msg == "Device not found"

    def test_acl_error_code_pattern(self):
        line = "Invalid device name (Z:BAD) in read device command at line 0 - DIO_NO_SUCH"
        is_error, msg = _is_error_response(line)
        assert is_error is True
        assert "DIO_NO_SUCH" in msg

    def test_clib_error_code(self):
        line = "Invalid read value variable (G:AMANDA) specified in read_device command at line 1 - CLIB_SYNTAX"
        is_error, msg = _is_error_response(line)
        assert is_error is True

    def test_normal_reading_not_error(self):
        is_error, msg = _is_error_response("M:OUTTMP       =  12.34 DegF")
        assert is_error is False
        assert msg is None

    def test_description_with_dash_not_error(self):
        """A description containing ' - ' should not be a false positive."""
        is_error, msg = _is_error_response("M:FOO = Temperature - external sensor")
        assert is_error is False


class TestAclReadCommand:
    """Tests for _acl_read_command - DRF to ACL command + qualifier mapping."""

    def test_plain_device(self):
        cmd, drf, quals = _acl_read_command("M:OUTTMP")
        assert cmd == "read"
        assert "OUTTMP" in drf
        assert quals == ""

    def test_raw_field(self):
        cmd, drf, quals = _acl_read_command("M:OUTTMP.RAW")
        assert cmd == "read"
        assert ".RAW" in drf
        assert "/raw" in quals

    def test_clock_event(self):
        cmd, drf, quals = _acl_read_command("M:OUTTMP@e,02")
        assert cmd == "read/pendwait"
        assert "/event='e,02'" in quals
        assert "@" not in drf

    def test_clock_event_with_delay(self):
        cmd, drf, quals = _acl_read_command("M:OUTTMP@e,02,e,500")
        assert cmd == "read/pendwait"
        assert "/event='e,02,e,500'" in quals
        assert "@" not in drf

    def test_raw_with_clock_event(self):
        cmd, drf, quals = _acl_read_command("M:OUTTMP.RAW@e,0f")
        assert cmd == "read/pendwait"
        assert "/raw" in quals
        assert "/event='e,0f'" in quals
        assert ".RAW" in drf
        assert "@" not in drf

    def test_immediate_event_stripped(self):
        cmd, drf, quals = _acl_read_command("M:OUTTMP@I")
        assert cmd == "read"
        assert quals == ""
        assert "@" not in drf

    def test_qualifier_canonicalized(self):
        """DRF qualifier chars are expanded to explicit property names."""
        cmd, drf, quals = _acl_read_command("M~OUTTMP")
        assert cmd == "read"
        assert drf == "M:OUTTMP.DESCRIPTION"

    def test_periodic_event_raises(self):
        with pytest.raises(DeviceError, match="periodic"):
            _acl_read_command("M:OUTTMP@p,100")

    def test_periodic_q_event_raises(self):
        with pytest.raises(DeviceError, match="periodic"):
            _acl_read_command("M:OUTTMP@q,500")


class TestParseRawHex:
    """Tests for _parse_raw_hex - parses ACL /raw hex output."""

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("M:OUTTMP = 0x42900000", bytes.fromhex("42900000")),
            ("M:OUTTMP = 0x4290 0x0000", bytes.fromhex("42900000")),
            ("M:OUTTMP = 42900000", bytes.fromhex("42900000")),
            ("M:OUTTMP = 42 90 00 00", bytes.fromhex("42900000")),
            ("0x42900000", bytes.fromhex("42900000")),
            ("M:OUTTMP = 0xA", bytes.fromhex("0a")),
            ("M:OUTTMP = ", b""),
        ],
    )
    def test_parse_raw_hex(self, line, expected):
        assert _parse_raw_hex(line) == expected

    def test_no_hex_data_raises(self):
        with pytest.raises(ValueError, match="No hex data"):
            _parse_raw_hex("M:OUTTMP = not hex at all")


class TestBuildURL:
    """Tests for URL construction."""

    def test_single_device_url(self):
        backend = ACLBackend()
        url = backend._build_url(["M:OUTTMP"])
        assert "?acl=read+M:OUTTMP" in url
        assert "\\;" not in url  # no semicolons for single device
        backend.close()

    def test_batch_url_uses_semicolons(self):
        backend = ACLBackend()
        url = backend._build_url(["M:OUTTMP", "G:AMANDA"])
        assert "read+M:OUTTMP\\;read+G:AMANDA" in url
        backend.close()

    def test_raw_device_url_has_qualifier(self):
        backend = ACLBackend()
        url = backend._build_url(["M:OUTTMP.RAW"])
        assert "read+M:OUTTMP.RAW/raw" in url
        backend.close()

    def test_batch_mixed_raw_and_scaled(self):
        backend = ACLBackend()
        url = backend._build_url(["M:OUTTMP.RAW", "G:AMANDA"])
        assert "read+M:OUTTMP.RAW/raw" in url
        assert "read+G:AMANDA" in url
        backend.close()

    def test_clock_event_url(self):
        backend = ACLBackend()
        url = backend._build_url(["M:OUTTMP@e,02"])
        assert "read/pendwait+M:OUTTMP.READING/event='e,02'" in url
        backend.close()


class TestSingleDeviceRead:
    """Tests for single device read/get operations."""

    def test_read_scalar_success(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("M:OUTTMP       =  72.5 DegF")
            backend = ACLBackend()
            try:
                value = backend.read("M:OUTTMP")
                assert value == 72.5
            finally:
                backend.close()

    def test_read_text_success(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("M:OUTTMP = Outdoor Temperature")
            backend = ACLBackend()
            try:
                value = backend.read("M:OUTTMP.DESCRIPTION")
                assert value == "Outdoor Temperature"
            finally:
                backend.close()

    def test_get_returns_reading(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("M:OUTTMP       =  72.5 DegF")
            backend = ACLBackend()
            try:
                reading = backend.get("M:OUTTMP")
                assert isinstance(reading, Reading)
                assert reading.value == 72.5
                assert reading.value_type == ValueType.SCALAR
                assert reading.ok
            finally:
                backend.close()

    def test_read_error_raises_device_error(self):
        """ACL error triggers fallback which also errors → DeviceError."""
        with mock.patch("httpx.Client.get") as mock_get:
            # Both batch and individual read return the error
            mock_get.return_value = MockACLResponse("Invalid device name (M:BADDEV) in read command - DIO_NO_SUCH")
            backend = ACLBackend()
            try:
                with pytest.raises(DeviceError):
                    backend.read("M:BADDEV")
            finally:
                backend.close()

    def test_get_error_returns_error_reading(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("Invalid device name (M:BADDEV) in read command - DIO_NO_SUCH")
            backend = ACLBackend()
            try:
                reading = backend.get("M:BADDEV")
                assert reading.is_error
                assert "DIO_NO_SUCH" in reading.message
            finally:
                backend.close()


class TestRawRead:
    """Tests for .RAW field reading."""

    def test_get_raw_returns_bytes(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("M:OUTTMP = 0x42900000")
            with ACLBackend() as backend:
                reading = backend.get("M:OUTTMP.RAW")
                assert reading.value == bytes.fromhex("42900000")
                assert reading.value_type == ValueType.RAW
                assert reading.ok

    def test_batch_mixed_raw_and_scaled(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("M:OUTTMP = 0x42900000\nG:AMANDA       =  66")
            with ACLBackend() as backend:
                readings = backend.get_many(["M:OUTTMP.RAW", "G:AMANDA"])
                assert readings[0].value == bytes.fromhex("42900000")
                assert readings[0].value_type == ValueType.RAW
                assert readings[1].value == 66.0
                assert readings[1].value_type == ValueType.SCALAR

    def test_raw_fallback_individual(self):
        """Raw reads work through the individual-fallback path."""
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = [
                # Batch: error triggers fallback
                MockACLResponse("Invalid device - DIO_NO_SUCH"),
                # Individual: M:OUTTMP.RAW succeeds
                MockACLResponse("M:OUTTMP = 0xDEADBEEF"),
                # Individual: Z:BAD fails
                MockACLResponse("Invalid device name (Z:BAD) - DIO_NO_SUCH"),
            ]
            with ACLBackend() as backend:
                readings = backend.get_many(["M:OUTTMP.RAW", "Z:BAD"])
                assert readings[0].value == bytes.fromhex("DEADBEEF")
                assert readings[0].value_type == ValueType.RAW
                assert readings[1].is_error


class TestMultipleDeviceRead:
    """Tests for get_many - batch and fallback behavior."""

    def test_batch_success(self):
        """All devices succeed in a single batch request."""
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("M:OUTTMP       =  72.5 DegF\nG:AMANDA       =  66")
            backend = ACLBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
                assert len(readings) == 2
                assert readings[0].value == 72.5
                assert readings[1].value == 66.0
            finally:
                backend.close()

    def test_fallback_on_batch_error(self):
        """Bad device in batch triggers individual fallback."""
        with mock.patch("httpx.Client.get") as mock_get:
            # First call (batch) returns error
            # Subsequent calls (individual) return per-device results
            mock_get.side_effect = [
                # Batch: ACL aborts on bad device
                MockACLResponse("Invalid device name (Z:BAD) - DIO_NO_SUCH"),
                # Individual: M:OUTTMP succeeds
                MockACLResponse("M:OUTTMP       =  72.5 DegF"),
                # Individual: Z:BAD fails
                MockACLResponse("Invalid device name (Z:BAD) - DIO_NO_SUCH"),
                # Individual: G:AMANDA succeeds
                MockACLResponse("G:AMANDA       =  66"),
            ]
            backend = ACLBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "Z:BAD", "G:AMANDA"])
                assert len(readings) == 3
                assert readings[0].ok
                assert readings[0].value == 72.5
                assert readings[1].is_error
                assert "DIO_NO_SUCH" in readings[1].message
                assert readings[2].ok
                assert readings[2].value == 66.0
            finally:
                backend.close()

    def test_fallback_on_line_count_mismatch(self):
        """Fewer lines than DRFs triggers individual fallback."""
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = [
                # Batch: only 1 line for 2 devices
                MockACLResponse("M:OUTTMP       =  72.5 DegF"),
                # Individual reads
                MockACLResponse("M:OUTTMP       =  72.5 DegF"),
                MockACLResponse("G:AMANDA       =  66"),
            ]
            backend = ACLBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
                assert len(readings) == 2
                assert readings[0].ok
                assert readings[1].ok
            finally:
                backend.close()


class TestHTTPErrors:
    """Tests for HTTP error handling."""

    def test_http_error_returns_error_readings(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("", status_code=503)
            backend = ACLBackend()
            try:
                readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
                assert len(readings) == 2
                assert all(r.is_error for r in readings)
                assert all("HTTP 503" in r.message for r in readings)
            finally:
                backend.close()

    def test_connection_error(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            backend = ACLBackend()
            try:
                with pytest.raises(DeviceError, match="ACL request failed"):
                    backend.read("M:OUTTMP")
            finally:
                backend.close()

    def test_timeout_error(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = httpx.ReadTimeout("timed out")
            backend = ACLBackend()
            try:
                with pytest.raises(DeviceError, match="timed out"):
                    backend.read("M:OUTTMP")
            finally:
                backend.close()


class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager_closes(self):
        with ACLBackend() as backend:
            assert not backend._closed
        assert backend._closed

    def test_context_manager_on_exception(self):
        try:
            with ACLBackend() as backend:
                raise ValueError("test error")
        except ValueError:
            pass
        assert backend._closed


class TestIsBasicStatusRequest:
    """Tests for _is_basic_status_request - detects bare STATUS DRFs."""

    @pytest.mark.parametrize(
        "drf,expected",
        [
            ("N|LGXS", True),  # qualifier char
            ("N:LGXS.STATUS", True),  # explicit property
            ("Z|ACLTST", True),
            ("N:LGXS.STATUS.ON", False),  # specific field
            ("N:LGXS.STATUS.RAW", False),
            ("N:LGXS.STATUS.TEXT", False),
            ("M:OUTTMP", False),  # READING property
            ("M_OUTTMP", False),  # SETTING property
        ],
    )
    def test_is_basic_status_request(self, drf, expected):
        assert _is_basic_status_request(drf) is expected

    def test_invalid_drf_returns_false(self):
        assert _is_basic_status_request("X") is False


class TestBasicStatusRead:
    """Tests for basic status reading via individual field queries."""

    def test_all_fields_present(self):
        """All 5 status fields return True/False → full dict."""
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = [
                MockACLResponse("N:LGXS is on = False"),
                MockACLResponse("N:LGXS is ready = False"),
                MockACLResponse("N:LGXS is remote = True"),
                MockACLResponse("N:LGXS is positive = True"),
                MockACLResponse("N:LGXS is ramping = False"),
            ]
            with ACLBackend() as backend:
                reading = backend.get("N|LGXS")
                assert reading.ok
                assert reading.value_type == ValueType.BASIC_STATUS
                assert reading.value == {
                    "on": False,
                    "ready": False,
                    "remote": True,
                    "positive": True,
                    "ramp": False,
                }

    def test_missing_attribute_omitted(self):
        """DIO_NOATT for remote → key omitted from dict (matches DPM)."""
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = [
                MockACLResponse("Z:ACLTST is on = False"),
                MockACLResponse("Z:ACLTST is ready = True"),
                MockACLResponse("Error determining status text in read device command at line 1 - Z:ACLTST DIO_NOATT"),
                MockACLResponse("Z:ACLTST is positive = False"),
                MockACLResponse("Z:ACLTST is ramping = False"),
            ]
            with ACLBackend() as backend:
                reading = backend.get("Z:ACLTST.STATUS")
                assert reading.ok
                assert reading.value_type == ValueType.BASIC_STATUS
                assert "remote" not in reading.value
                assert reading.value == {"on": False, "ready": True, "positive": False, "ramp": False}

    def test_nonexistent_device_returns_error(self):
        """First non-NOATT error (DBM_NOREC) immediately fails the whole read."""
        error_line = "Invalid device name (Z:NOTFND) in read device command at line 1 - DBM_NOREC"
        with mock.patch("httpx.Client.get") as mock_get:
            # Only 1 response needed - first field error aborts
            mock_get.return_value = MockACLResponse(error_line)
            with ACLBackend() as backend:
                reading = backend.get("Z:NOTFND.STATUS")
                assert reading.is_error
                assert "DBM_NOREC" in reading.message

    def test_non_noatt_error_mid_loop_fails(self):
        """A non-NOATT error on any field fails the whole status read."""
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = [
                MockACLResponse("N:LGXS is on = True"),
                MockACLResponse("N:LGXS is ready = True"),
                # Unexpected error on REMOTE - should fail immediately
                MockACLResponse("Error in read device command - DIO_NOSCALE"),
            ]
            with ACLBackend() as backend:
                reading = backend.get("N|LGXS")
                assert reading.is_error
                assert "DIO_NOSCALE" in reading.message

    def test_http_error_returns_error_reading(self):
        """HTTP error during status field read returns error Reading, not exception."""
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = [
                MockACLResponse("N:LGXS is on = False"),
                MockACLResponse("", status_code=500),
            ]
            with ACLBackend() as backend:
                reading = backend.get("N|LGXS")
                assert reading.is_error
                assert "HTTP 500" in reading.message

    def test_get_many_mixes_status_and_normal(self):
        """get_many routes status DRFs through per-field reads, others through batch."""
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = [
                # Batch for normal DRF (recursive get_many runs first)
                MockACLResponse("M:OUTTMP       =  72.5 DegF"),
                # 5 status field reads for N|LGXS
                MockACLResponse("N:LGXS is on = False"),
                MockACLResponse("N:LGXS is ready = False"),
                MockACLResponse("N:LGXS is remote = True"),
                MockACLResponse("N:LGXS is positive = True"),
                MockACLResponse("Error - N:LGXS DIO_NOATT"),
            ]
            with ACLBackend() as backend:
                readings = backend.get_many(["N|LGXS", "M:OUTTMP"])
                assert len(readings) == 2
                assert readings[0].value_type == ValueType.BASIC_STATUS
                assert readings[0].value == {"on": False, "ready": False, "remote": True, "positive": True}
                assert readings[1].value == 72.5


class TestWriteNotSupported:
    """Tests for write operations."""

    def test_write_raises_not_implemented(self):
        backend = ACLBackend()
        try:
            with pytest.raises(NotImplementedError):
                backend.write("M:OUTTMP", 72.5)
        finally:
            backend.close()


class TestOperationAfterClose:
    """Tests for operations after close."""

    def test_read_after_close_raises(self):
        backend = ACLBackend()
        backend.close()
        with pytest.raises(RuntimeError, match="Backend is closed"):
            backend.read("M:OUTTMP")


class TestTimeout:
    """Tests for timeout handling."""

    def test_custom_timeout_passed_to_client(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("M:OUTTMP = 72.5")
            backend = ACLBackend(timeout=3.0)
            try:
                backend.read("M:OUTTMP")
                assert mock_get.call_args.kwargs["timeout"] == 3.0
            finally:
                backend.close()

    def test_per_call_timeout_overrides_default(self):
        with mock.patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MockACLResponse("M:OUTTMP = 72.5")
            backend = ACLBackend(timeout=10.0)
            try:
                backend.read("M:OUTTMP", timeout=2.0)
                assert mock_get.call_args.kwargs["timeout"] == 2.0
            finally:
                backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
