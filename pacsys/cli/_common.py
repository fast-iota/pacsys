"""Shared CLI infrastructure for acget/acput/acmonitor/acinfo."""

import argparse
import json
import signal
import sys
from typing import Any, Callable, Optional, Union

import pacsys
from pacsys.drf_utils import get_device_name
from pacsys.types import BasicControl, Reading, WriteResult

# Exit codes
EXIT_OK = 0
EXIT_DEVICE_ERROR = 1
EXIT_USAGE_ERROR = 2


def base_parser(description: str) -> argparse.ArgumentParser:
    """Create ArgumentParser with common flags shared by all CLI tools."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-b", "--backend", choices=("dpm", "grpc", "dmq", "acl"), default="dpm", help="backend type")
    parser.add_argument("-H", "--host", default=None, help="backend host")
    parser.add_argument("-P", "--port", type=int, default=None, help="backend port")
    parser.add_argument("--timeout", type=float, default=5.0, help="operation timeout in seconds (default: 5.0)")
    parser.add_argument("-a", "--auth", choices=("kerberos", "jwt"), default=None, help="authentication method")
    parser.add_argument("--role", default=None, help="role for authenticated operations")
    parser.add_argument(
        "--format", dest="output_format", choices=("text", "json"), default="text", help="output format"
    )
    parser.add_argument("-t", "--terse", action="store_true", help="terse output (bare values)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    return parser


def make_backend(args):
    """Create backend via pacsys factory functions based on parsed args."""
    auth = _resolve_auth(getattr(args, "auth", None))
    kwargs: dict[str, Any] = {"timeout": args.timeout}
    if args.host is not None:
        kwargs["host"] = args.host
    if args.port is not None:
        kwargs["port"] = args.port
    if auth is not None:
        kwargs["auth"] = auth
    role = getattr(args, "role", None)
    if role is not None:
        kwargs["role"] = role

    backend_type = args.backend
    if backend_type == "dpm":
        return pacsys.dpm(**kwargs)
    elif backend_type == "grpc":
        kwargs.pop("role", None)
        return pacsys.grpc(**kwargs)
    elif backend_type == "dmq":
        kwargs.pop("role", None)
        if "auth" not in kwargs:
            kwargs["auth"] = pacsys.KerberosAuth()
        return pacsys.dmq(**kwargs)
    elif backend_type == "acl":
        # ACL only accepts timeout (and optionally base_url)
        acl_kwargs: dict[str, Any] = {"timeout": args.timeout}
        return pacsys.acl(**acl_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type!r}")


def _resolve_auth(auth_str: Optional[str]):
    """Resolve auth string to Auth object."""
    if auth_str is None:
        return None
    if auth_str == "kerberos":
        return pacsys.KerberosAuth()
    if auth_str == "jwt":
        return pacsys.JWTAuth.from_env()
    raise ValueError(f"Unknown auth type: {auth_str!r}")


def parse_slice(s: str) -> slice:
    """Parse Python slice syntax string.

    "5"     -> slice(5, 6)     single index
    "-1"    -> slice(-1, None) negative single index
    "0:10"  -> slice(0, 10)
    "::2"   -> slice(None, None, 2)
    "-5:"   -> slice(-5, None)
    """
    parts = s.split(":")
    if len(parts) == 1:
        # Single index
        try:
            idx = int(s)
        except ValueError:
            raise ValueError(f"Invalid slice: {s!r}")
        if idx < 0:
            return slice(idx, None)
        return slice(idx, idx + 1)
    if len(parts) > 3:
        raise ValueError(f"Invalid slice: {s!r}")

    def _parse_part(p: str) -> Optional[int]:
        p = p.strip()
        if p == "":
            return None
        try:
            return int(p)
        except ValueError:
            raise ValueError(f"Invalid slice: {s!r}")

    parsed = [_parse_part(p) for p in parts]
    if len(parsed) == 2:
        return slice(parsed[0], parsed[1])
    return slice(parsed[0], parsed[1], parsed[2])


_BASIC_CONTROL_NAMES = {m.name.lower(): m for m in BasicControl}


def parse_value(s: str) -> Union[float, str, list, BasicControl]:
    """Parse CLI value string.

    Control names (on/off/reset/...) -> BasicControl enum (case-insensitive).
    Comma-separated all-numeric -> list of floats (array write).
    Otherwise try float, fallback to string (preserves commas in text).
    """
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return s
    ctrl = _BASIC_CONTROL_NAMES.get(s.lower())
    if ctrl is not None:
        return ctrl
    try:
        return float(s)
    except ValueError:
        return s


def format_value(value: Any, number_format: Optional[str]) -> str:
    """Format a value for display.

    numpy arrays and lists: space-joined elements, each formatted if spec given.
    Scalars: format() if spec. Strings: str().
    """
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ndarray):
        fmt = number_format or "g"
        return " ".join(format(v, fmt) for v in value)
    if isinstance(value, list):
        if number_format:
            return " ".join(format(v, number_format) for v in value)
        return " ".join(format(v, "g") if isinstance(v, float) else str(v) for v in value)
    if isinstance(value, str):
        return value
    if isinstance(value, float) and value.is_integer() and number_format:
        value = int(value)
    if number_format:
        return format(value, number_format)
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _format_timestamp(ts, timestamp_format: str, reference_time: Optional[float]) -> str:
    """Format a timestamp according to the given format."""
    if ts is None:
        return ""
    if timestamp_format == "epoch":
        return f"{ts.timestamp():.3f}"
    if timestamp_format == "relative" and reference_time is not None:
        return f"{ts.timestamp() - reference_time:.3f}"
    # iso (default)
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_reading(
    reading: Reading,
    *,
    fmt: str,
    number_format: Optional[str],
    array_slice: Optional[slice],
    timestamp_format: str = "iso",
    reference_time: Optional[float] = None,
) -> str:
    """Format a Reading for output.

    fmt="terse": bare value (or error message).
    fmt="json": JSON dict with device/ok/value/units/timestamp.
    fmt="text": columns with device name | value+units | timestamp.

    timestamp_format: "iso" (default), "epoch" (unix seconds), "relative" (seconds since reference_time).
    reference_time: epoch float used as origin for "relative" mode.
    """
    device = reading.name

    if not reading.ok:
        error_msg = reading.message or f"error {reading.error_code}"
        if fmt == "json":
            return json.dumps({"device": device, "ok": False, "error": error_msg})
        if fmt == "terse":
            return f"[ERROR] {error_msg}"
        return f"{device:<28s} [ERROR] {error_msg}"

    # Get displayable value
    val = reading.value
    np = sys.modules.get("numpy")
    if array_slice is not None and np is not None and isinstance(val, np.ndarray):
        val = val[array_slice]

    formatted = format_value(val, number_format)
    units = reading.units

    if fmt == "json":
        d: dict[str, Any] = {
            "device": device,
            "ok": True,
            "value": _json_safe(val),
        }
        d["units"] = units or None
        ts = reading.timestamp
        if timestamp_format == "epoch":
            d["timestamp"] = ts.timestamp() if ts else None
        elif timestamp_format == "relative" and reference_time is not None:
            d["timestamp"] = ts.timestamp() - reference_time if ts else None
        else:
            d["timestamp"] = ts.isoformat() if ts else None
        return json.dumps(d)

    if fmt == "terse":
        return formatted

    # text format
    val_str = f"{formatted} {units}" if units else formatted
    ts_str = _format_timestamp(reading.timestamp, timestamp_format, reference_time)
    return f"{device}  {val_str}  {ts_str}"


def format_write_result(result: WriteResult, *, fmt: str) -> str:
    """Format a WriteResult for output."""
    device = get_device_name(result.drf)

    if fmt == "json":
        d: dict[str, Any] = {"device": device, "ok": result.ok}
        if not result.ok:
            d["error"] = result.message or f"error {result.error_code}"
        if result.verified is not None:
            d["verified"] = result.verified
        if result.readback is not None:
            d["readback"] = _json_safe(result.readback)
        return json.dumps(d)

    if fmt == "terse":
        if not result.ok:
            return f"FAILED: {result.message or result.error_code}"
        if result.verified is True:
            return "ok (verified)"
        if result.verified is False:
            return f"VERIFY FAILED (readback: {result.readback})"
        return "ok"

    # text format
    if not result.ok:
        return f"{device}  FAILED: {result.message or result.error_code}"
    parts = [device, " ok"]
    if result.verified is True:
        readback_str = f" (verified, readback={result.readback})" if result.readback is not None else " (verified)"
        parts.append(readback_str)
    elif result.verified is False:
        readback_str = (
            f" (verify FAILED, readback={result.readback})" if result.readback is not None else " (verify FAILED)"
        )
        parts.append(readback_str)
    return "".join(parts)


def _json_safe(value: Any) -> Any:
    """Convert numpy types to Python native for JSON serialization."""
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    return value


def install_signal_handlers(cleanup_fn: Callable[[], None]) -> None:
    """Install SIGINT/SIGTERM handlers that call cleanup_fn then exit."""

    def _handler(signum, frame):
        cleanup_fn()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
