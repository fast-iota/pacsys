"""Tests for pacsys.cli._common — shared CLI infrastructure."""

import json
from datetime import datetime
from unittest import mock

import numpy as np
import pytest

from pacsys.types import Reading, WriteResult, ValueType, DeviceMeta


class TestParseSlice:
    """parse_slice converts Python slice syntax strings to slice objects."""

    def test_single_index(self):
        from pacsys.cli._common import parse_slice

        assert parse_slice("5") == slice(5, 6)

    def test_negative_single_index(self):
        from pacsys.cli._common import parse_slice

        assert parse_slice("-1") == slice(-1, None)

    def test_start_stop(self):
        from pacsys.cli._common import parse_slice

        assert parse_slice("0:10") == slice(0, 10)

    def test_start_only(self):
        from pacsys.cli._common import parse_slice

        assert parse_slice("3:") == slice(3, None)

    def test_stop_only(self):
        from pacsys.cli._common import parse_slice

        assert parse_slice(":5") == slice(None, 5)

    def test_full_slice(self):
        from pacsys.cli._common import parse_slice

        assert parse_slice("0:10:2") == slice(0, 10, 2)

    def test_step_only(self):
        from pacsys.cli._common import parse_slice

        assert parse_slice("::2") == slice(None, None, 2)

    def test_negative_start(self):
        from pacsys.cli._common import parse_slice

        assert parse_slice("-5:") == slice(-5, None)

    def test_invalid_raises(self):
        from pacsys.cli._common import parse_slice

        with pytest.raises(ValueError):
            parse_slice("abc")

    def test_too_many_colons_raises(self):
        from pacsys.cli._common import parse_slice

        with pytest.raises(ValueError):
            parse_slice("1:2:3:4")


class TestParseValue:
    """parse_value converts CLI value strings to Python values."""

    def test_float(self):
        from pacsys.cli._common import parse_value

        assert parse_value("72.5") == 72.5

    def test_integer_becomes_float(self):
        from pacsys.cli._common import parse_value

        assert parse_value("42") == 42.0
        assert isinstance(parse_value("42"), float)

    def test_string_passthrough(self):
        from pacsys.cli._common import parse_value

        assert parse_value("hello") == "hello"

    def test_comma_separated_floats(self):
        from pacsys.cli._common import parse_value

        result = parse_value("1,2,3")
        assert result == [1.0, 2.0, 3.0]

    def test_comma_separated_mixed_is_string(self):
        from pacsys.cli._common import parse_value

        # Non-numeric comma values are treated as plain strings, not arrays
        result = parse_value("1,hello,3")
        assert result == "1,hello,3"

    def test_negative_float(self):
        from pacsys.cli._common import parse_value

        assert parse_value("-3.14") == -3.14


class TestFormatValue:
    """format_value formats values with optional Python format spec."""

    def test_scalar_default(self):
        from pacsys.cli._common import format_value

        assert format_value(72.5, None) == "72.5"

    def test_scalar_format_spec(self):
        from pacsys.cli._common import format_value

        assert format_value(72.5, ".3f") == "72.500"

    def test_scientific(self):
        from pacsys.cli._common import format_value

        result = format_value(1234.0, "e")
        assert "e+" in result.lower()

    def test_numpy_array(self):
        from pacsys.cli._common import format_value

        arr = np.array([1.0, 2.0, 3.0])
        result = format_value(arr, None)
        assert "1.0" in result
        assert "2.0" in result
        assert "3.0" in result

    def test_numpy_array_with_format(self):
        from pacsys.cli._common import format_value

        arr = np.array([1.0, 2.0])
        result = format_value(arr, ".2f")
        assert "1.00" in result
        assert "2.00" in result

    def test_string_value(self):
        from pacsys.cli._common import format_value

        assert format_value("hello", None) == "hello"

    def test_list_value(self):
        from pacsys.cli._common import format_value

        result = format_value([1.0, 2.0, 3.0], None)
        assert "1.0" in result
        assert "2.0" in result
        assert "3.0" in result

    def test_integer_value(self):
        from pacsys.cli._common import format_value

        assert format_value(42, None) == "42"

    def test_hex_format(self):
        from pacsys.cli._common import format_value

        assert format_value(255, "x") == "ff"


class TestFormatReading:
    """format_reading formats a Reading for text/terse/json output."""

    def _make_reading(self, value=72.5, error_code=0, drf="M:OUTTMP", units="degF", ts=None, msg=None):
        meta = DeviceMeta(device_index=0, name="M:OUTTMP", description="Outside temp", units=units)
        return Reading(
            drf=drf,
            value_type=ValueType.SCALAR,
            value=value,
            error_code=error_code,
            message=msg,
            timestamp=ts or datetime(2025, 6, 15, 12, 0, 0),
            meta=meta,
        )

    def test_text_format(self):
        from pacsys.cli._common import format_reading

        r = self._make_reading()
        result = format_reading(r, fmt="text", number_format=None, array_slice=None)
        assert "M:OUTTMP" in result
        assert "72.5" in result
        assert "degF" in result

    def test_text_timestamp_includes_date(self):
        from pacsys.cli._common import format_reading

        r = self._make_reading()
        result = format_reading(r, fmt="text", number_format=None, array_slice=None)
        assert "2025-06-15 12:00:00" in result

    def test_terse_format(self):
        from pacsys.cli._common import format_reading

        r = self._make_reading()
        result = format_reading(r, fmt="terse", number_format=None, array_slice=None)
        assert "72.5" in result
        # Terse should not have device name prefix
        assert not result.startswith("M:OUTTMP")
        # Terse should be bare value only — no units
        assert "degF" not in result

    def test_json_format(self):
        from pacsys.cli._common import format_reading

        r = self._make_reading()
        result = format_reading(r, fmt="json", number_format=None, array_slice=None)
        data = json.loads(result)
        assert data["device"] == "M:OUTTMP"
        assert data["ok"] is True
        assert data["value"] == 72.5
        assert data["units"] == "degF"

    def test_error_reading_text(self):
        from pacsys.cli._common import format_reading

        r = self._make_reading(value=None, error_code=-1, msg="DIO_NOATT")
        result = format_reading(r, fmt="text", number_format=None, array_slice=None)
        assert "M:OUTTMP" in result
        assert "DIO_NOATT" in result

    def test_error_reading_json(self):
        from pacsys.cli._common import format_reading

        r = self._make_reading(value=None, error_code=-1, msg="DIO_NOATT")
        result = format_reading(r, fmt="json", number_format=None, array_slice=None)
        data = json.loads(result)
        assert data["ok"] is False
        assert "DIO_NOATT" in data.get("error", "")

    def test_array_slicing(self):
        from pacsys.cli._common import format_reading

        arr = np.arange(10, dtype=float)
        r = self._make_reading(value=arr, units=None)
        result = format_reading(r, fmt="terse", number_format=None, array_slice=slice(0, 3))
        # Should only show first 3 elements
        assert "0.0" in result
        assert "1.0" in result
        assert "2.0" in result

    def test_number_format(self):
        from pacsys.cli._common import format_reading

        r = self._make_reading()
        result = format_reading(r, fmt="terse", number_format=".3f", array_slice=None)
        assert "72.500" in result

    def test_error_reading_terse(self):
        from pacsys.cli._common import format_reading

        r = self._make_reading(value=None, error_code=-1, msg="DIO_NOATT")
        result = format_reading(r, fmt="terse", number_format=None, array_slice=None)
        assert "DIO_NOATT" in result

    def test_text_epoch_timestamp(self):
        from pacsys.cli._common import format_reading

        ts = datetime(2025, 6, 15, 12, 0, 0)
        r = self._make_reading(ts=ts)
        result = format_reading(r, fmt="text", number_format=None, array_slice=None, timestamp_format="epoch")
        expected = f"{ts.timestamp():.3f}"
        assert expected in result

    def test_text_relative_timestamp(self):
        from pacsys.cli._common import format_reading

        ts = datetime(2025, 6, 15, 12, 0, 5)
        ref = datetime(2025, 6, 15, 12, 0, 0).timestamp()
        r = self._make_reading(ts=ts)
        result = format_reading(
            r, fmt="text", number_format=None, array_slice=None, timestamp_format="relative", reference_time=ref
        )
        assert "5.000" in result

    def test_json_epoch_timestamp(self):
        from pacsys.cli._common import format_reading

        ts = datetime(2025, 6, 15, 12, 0, 0)
        r = self._make_reading(ts=ts)
        result = format_reading(r, fmt="json", number_format=None, array_slice=None, timestamp_format="epoch")
        data = json.loads(result)
        assert isinstance(data["timestamp"], float)
        assert data["timestamp"] == ts.timestamp()

    def test_json_relative_timestamp(self):
        from pacsys.cli._common import format_reading

        ts = datetime(2025, 6, 15, 12, 0, 5)
        ref = datetime(2025, 6, 15, 12, 0, 0).timestamp()
        r = self._make_reading(ts=ts)
        result = format_reading(
            r, fmt="json", number_format=None, array_slice=None, timestamp_format="relative", reference_time=ref
        )
        data = json.loads(result)
        assert isinstance(data["timestamp"], float)
        assert abs(data["timestamp"] - 5.0) < 0.01

    def test_iso_is_default(self):
        from pacsys.cli._common import format_reading

        r = self._make_reading()
        result = format_reading(r, fmt="text", number_format=None, array_slice=None)
        assert "2025-06-15 12:00:00" in result


class TestFormatWriteResult:
    """format_write_result formats WriteResult for text/json output."""

    def _make_result(self, ok=True, drf="Z:ACLTST", verified=None, readback=None, msg=None):
        return WriteResult(
            drf=drf,
            error_code=0 if ok else -1,
            message=msg if not ok else None,
            verified=verified,
            readback=readback,
        )

    def test_text_ok(self):
        from pacsys.cli._common import format_write_result

        r = self._make_result()
        result = format_write_result(r, fmt="text")
        assert "Z:ACLTST" in result
        assert "ok" in result.lower()

    def test_text_error(self):
        from pacsys.cli._common import format_write_result

        r = self._make_result(ok=False, msg="WRITE_FAILED")
        result = format_write_result(r, fmt="text")
        assert "Z:ACLTST" in result
        assert "WRITE_FAILED" in result

    def test_json_ok(self):
        from pacsys.cli._common import format_write_result

        r = self._make_result()
        result = format_write_result(r, fmt="json")
        data = json.loads(result)
        assert data["device"] == "Z:ACLTST"
        assert data["ok"] is True

    def test_json_error(self):
        from pacsys.cli._common import format_write_result

        r = self._make_result(ok=False, msg="WRITE_FAILED")
        result = format_write_result(r, fmt="json")
        data = json.loads(result)
        assert data["ok"] is False
        assert "WRITE_FAILED" in data.get("error", "")

    def test_text_verified(self):
        from pacsys.cli._common import format_write_result

        r = self._make_result(verified=True, readback=72.5)
        result = format_write_result(r, fmt="text")
        assert "verified" in result.lower() or "72.5" in result

    def test_terse_ok(self):
        from pacsys.cli._common import format_write_result

        r = self._make_result()
        result = format_write_result(r, fmt="terse")
        assert "ok" in result.lower()

    def test_terse_verify_failed(self):
        from pacsys.cli._common import format_write_result

        r = self._make_result(verified=False, readback=99.0)
        result = format_write_result(r, fmt="terse")
        assert "VERIFY FAILED" in result
        assert "99.0" in result


class TestMakeBackend:
    """make_backend creates correct backend type based on args."""

    def _make_args(self, backend="dpm", host=None, port=None, timeout=5.0, auth=None, role=None):
        args = mock.MagicMock()
        args.backend = backend
        args.host = host
        args.port = port
        args.timeout = timeout
        args.auth = auth
        args.role = role
        return args

    @mock.patch("pacsys.dpm")
    def test_dpm_backend(self, mock_dpm):
        from pacsys.cli._common import make_backend

        args = self._make_args(backend="dpm")
        make_backend(args)
        mock_dpm.assert_called_once()

    @mock.patch("pacsys.grpc")
    def test_grpc_backend(self, mock_grpc):
        from pacsys.cli._common import make_backend

        args = self._make_args(backend="grpc")
        make_backend(args)
        mock_grpc.assert_called_once()

    @mock.patch("pacsys.dmq")
    def test_dmq_backend(self, mock_dmq):
        from pacsys.cli._common import make_backend

        args = self._make_args(backend="dmq")
        make_backend(args)
        mock_dmq.assert_called_once()

    @mock.patch("pacsys.acl")
    def test_acl_backend(self, mock_acl):
        from pacsys.cli._common import make_backend

        args = self._make_args(backend="acl")
        make_backend(args)
        mock_acl.assert_called_once()

    @mock.patch("pacsys.dpm")
    def test_passes_timeout(self, mock_dpm):
        from pacsys.cli._common import make_backend

        args = self._make_args(backend="dpm", timeout=10.0)
        make_backend(args)
        _, kwargs = mock_dpm.call_args
        assert kwargs["timeout"] == 10.0

    @mock.patch("pacsys.dpm")
    def test_passes_host_port(self, mock_dpm):
        from pacsys.cli._common import make_backend

        args = self._make_args(backend="dpm", host="myhost", port=1234)
        make_backend(args)
        _, kwargs = mock_dpm.call_args
        assert kwargs["host"] == "myhost"
        assert kwargs["port"] == 1234


class TestBaseParser:
    """base_parser creates ArgumentParser with common flags."""

    def test_has_backend_flag(self):
        from pacsys.cli._common import base_parser

        parser = base_parser("test")
        args = parser.parse_args(["-b", "grpc"])
        assert args.backend == "grpc"

    def test_defaults(self):
        from pacsys.cli._common import base_parser

        parser = base_parser("test")
        args = parser.parse_args([])
        assert args.backend == "dpm"
        assert args.timeout == 5.0
        assert args.output_format == "text"
        assert args.terse is False
        assert args.verbose is False

    def test_timeout_flag(self):
        from pacsys.cli._common import base_parser

        parser = base_parser("test")
        args = parser.parse_args(["--timeout", "10.0"])
        assert args.timeout == 10.0

    def test_format_flag(self):
        from pacsys.cli._common import base_parser

        parser = base_parser("test")
        args = parser.parse_args(["--format", "json"])
        assert args.output_format == "json"

    def test_terse_flag(self):
        from pacsys.cli._common import base_parser

        parser = base_parser("test")
        args = parser.parse_args(["-t"])
        assert args.terse is True


class TestJsonSafe:
    """_json_safe converts numpy types to Python native for JSON."""

    def test_numpy_float(self):
        from pacsys.cli._common import _json_safe

        assert _json_safe(np.float64(1.5)) == 1.5
        assert isinstance(_json_safe(np.float64(1.5)), float)

    def test_numpy_int(self):
        from pacsys.cli._common import _json_safe

        assert _json_safe(np.int64(42)) == 42
        assert isinstance(_json_safe(np.int64(42)), int)

    def test_numpy_array(self):
        from pacsys.cli._common import _json_safe

        arr = np.array([1.0, 2.0, 3.0])
        result = _json_safe(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_passthrough(self):
        from pacsys.cli._common import _json_safe

        assert _json_safe(42) == 42
        assert _json_safe("hello") == "hello"
        assert _json_safe([1, 2]) == [1, 2]


class TestResolveAuth:
    """_resolve_auth maps auth strings to Auth objects."""

    def test_none_returns_none(self):
        from pacsys.cli._common import _resolve_auth

        assert _resolve_auth(None) is None

    @mock.patch("pacsys.KerberosAuth")
    def test_kerberos(self, mock_kerb):
        from pacsys.cli._common import _resolve_auth

        result = _resolve_auth("kerberos")
        mock_kerb.assert_called_once()
        assert result is mock_kerb.return_value

    @mock.patch("pacsys.JWTAuth")
    def test_jwt(self, mock_jwt):
        from pacsys.cli._common import _resolve_auth

        result = _resolve_auth("jwt")
        mock_jwt.from_env.assert_called_once()
        assert result is mock_jwt.from_env.return_value

    def test_unknown_raises(self):
        from pacsys.cli._common import _resolve_auth

        with pytest.raises(ValueError):
            _resolve_auth("bogus")
