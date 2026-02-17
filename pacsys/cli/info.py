"""acinfo / pacsys-info -- Display device metadata and status."""

import json
import sys
from dataclasses import dataclass
from typing import Any

from pacsys.cli._common import (
    EXIT_DEVICE_ERROR,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    base_parser,
    format_value,
    make_backend,
)
from pacsys.device import Device
from pacsys.drf_utils import get_device_name


@dataclass(frozen=True)
class _DeviceProps:
    """Which properties a device supports, from DevDB. All True = unknown/query everything."""

    has_reading: bool = True
    has_setting: bool = True
    has_status: bool = True
    has_analog_alarm: bool = True
    has_digital_alarm: bool = True
    device_index: int | None = None
    ext_status_bits: tuple | None = None  # ExtStatusBitDef tuples for bit names


def _is_noprop(msg: str | Exception | None) -> bool:
    """True if the error indicates the property doesn't exist (expected, not a real failure)."""
    return msg is not None and "DBM_NOPROP" in str(msg)


def _noprop_str(err: Exception | str | None, result=None) -> str:
    """Format a NOPROP error as 'DRF: message' without doubled suffixes."""
    from pacsys.errors import DeviceError

    if isinstance(err, DeviceError):
        return f"{err.drf}: {err.message}"
    if err is not None:
        return str(err)
    # Non-ok Reading with NOPROP message
    if result is not None:
        return f"{result.drf}: {result.message}"
    return "DBM_NOPROP"


def _section(label: str, fn) -> tuple[Exception | None, Any]:
    """Call fn(), return (exception, result). On exception, return (exc, None)."""
    try:
        result = fn()
        return None, result
    except Exception as e:
        return e, None


def _format_reading_compact(reading, number_format: str | None = None) -> str:
    """Format a Reading for info display: value units (TYPE)."""
    if not reading.ok:
        return f"[ERROR] {reading.message or f'error {reading.error_code}'}"
    val_str = format_value(reading.value, number_format)
    parts = [val_str]
    if reading.units:
        parts.append(reading.units)
    parts.append(f"({reading.value_type.name})")
    return " ".join(parts)


def _format_status_compact(status) -> str:
    """Format status dict as comma-separated active flags."""
    if isinstance(status, dict):
        labels = {
            "on": ("ON", "OFF"),
            "ready": ("READY", "TRIPPED"),
            "remote": ("REMOTE", "LOCAL"),
            "positive": ("POSITIVE", "NEGATIVE"),
            "ramp": ("RAMP", "DC"),
        }
        parts = []
        for key, (true_label, false_label) in labels.items():
            val = status.get(key)
            if val is None:
                continue
            parts.append(true_label if val else false_label)
        return ", ".join(parts) if parts else str(status)
    return str(status)


def _format_analog_alarm_compact(alarm, number_format: str | None = None) -> str:
    """Format analog alarm as key=value pairs."""
    if isinstance(alarm, dict):
        parts = []
        units = alarm.get("units", "")
        for k, v in alarm.items():
            if k == "units":
                continue
            _np = sys.modules.get("numpy")
            _num = (int, float, _np.integer, _np.floating) if _np is not None else (int, float)
            val_str = format_value(v, number_format) if isinstance(v, _num) else str(v)
            parts.append(f"{k}={val_str}")
        result = ", ".join(parts)
        if units:
            result += f" {units}"
        return result
    return str(alarm)


def _format_digital_status_compact(ds) -> str:
    """Format digital status as binary bitfield."""
    if not ds.bits:
        return "(no bits)"
    max_pos = max(b.position for b in ds.bits)
    bitfield = ["0"] * (max_pos + 1)
    for b in ds.bits:
        bitfield[b.position] = "1" if b.is_set else "0"
    # Display MSB-first
    binary = "".join(reversed(bitfield))
    return f"{binary}  ({len(ds.bits)} bits)"


def _format_digital_status_verbose(ds) -> str:
    """Format digital status with per-bit detail."""
    w = max((b.position for b in ds.bits), default=0)
    w = len(str(w))  # digit width for alignment
    lines = []
    for b in ds.bits:
        lines.append(f"    Bit {b.position:>{w}}:  {'1' if b.is_set else '0'}  {b.name}: {b.value}")
    return "\n".join(lines)


_DIGITAL_ALARM_BITMASK_KEYS = {"nominal", "mask"}


def _format_digital_alarm_compact(alarm) -> str:
    """Format digital alarm: bitmasks as binary, everything else as plain values."""
    if isinstance(alarm, dict):
        parts = []
        for k, v in alarm.items():
            if k in _DIGITAL_ALARM_BITMASK_KEYS and isinstance(v, int):
                parts.append(f"{k}={v:08b}")
            elif isinstance(v, bool):
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)
    return str(alarm)


def _format_digital_alarm_verbose(alarm, ext_status_bits=None) -> tuple[str, str]:
    """Format digital alarm verbose. Returns (summary, per-bit detail)."""
    if not isinstance(alarm, dict):
        return str(alarm), ""
    nominal = alarm.get("nominal", 0)
    mask = alarm.get("mask", 0)
    if not isinstance(nominal, int) or not isinstance(mask, int):
        return _format_digital_alarm_compact(alarm), ""
    # Summary from non-bitmask fields
    summary_parts = []
    for k, v in alarm.items():
        if k not in _DIGITAL_ALARM_BITMASK_KEYS:
            summary_parts.append(f"{k}={v}")
    summary = ", ".join(summary_parts) if summary_parts else f"nominal={nominal:08b}, mask={mask:08b}"
    # Build bit_no -> name lookup from DevDB ext_status_bits
    bit_names: dict[int, str] = {}
    if ext_status_bits:
        for ebd in ext_status_bits:
            bit_names[ebd.bit_no] = ebd.description or ebd.name1 or f"bit{ebd.bit_no}"
    max_bits = max(nominal.bit_length(), mask.bit_length(), 1)
    w = len(str(max_bits - 1))  # digit width for alignment
    lines = []
    for i in range(max_bits):
        n_bit = (nominal >> i) & 1
        m_bit = (mask >> i) & 1
        monitored = "monitored" if m_bit else "not monitored"
        name = bit_names.get(i)
        label = f"  {name}" if name else ""
        lines.append(f"    Bit {i:>{w}}:{label}  nominal={n_bit}  mask={m_bit}  ({monitored})")
    return summary, "\n".join(lines)


def _display_text(
    dev, name: str, *, verbose: bool, number_format: str | None, props: _DeviceProps
) -> tuple[list[str], bool]:
    """Build text output lines for one device. Returns (lines, has_error)."""
    di = props.device_index
    lines = [f"{name} (di={di})" if di is not None else name]
    has_error = False

    # Description
    err, desc = _section("Description", lambda: dev.description())
    if err:
        lines.append(f"  Description:      [ERROR] {err}")
        has_error = True
    else:
        lines.append(f"  Description:      {desc}")

    # Reading
    if props.has_reading:
        err, reading = _section("Reading", lambda: dev.get(prop="reading"))
        # Backfill device_index from backend if DevDB wasn't available
        if di is None and reading is not None and reading.ok and reading.meta:
            meta_di = reading.meta.device_index
            if meta_di:
                lines[0] = f"{name} (di={meta_di})"
        if _is_noprop(err) or (reading is not None and not reading.ok and _is_noprop(reading.message)):
            if verbose:
                lines.append(f"  Reading:          {_noprop_str(err, reading)}")
        elif err:
            lines.append(f"  Reading:          [ERROR] {err}")
            has_error = True
        else:
            lines.append(f"  Reading:          {_format_reading_compact(reading, number_format)}")
            if not reading.ok:
                has_error = True

    # Setting
    if props.has_setting:
        err, setting = _section("Setting", lambda: dev.get(prop="setting"))
        if _is_noprop(err) or (setting is not None and not setting.ok and _is_noprop(setting.message)):
            if verbose:
                lines.append(f"  Setting:          {_noprop_str(err, setting)}")
        elif err:
            lines.append(f"  Setting:          [ERROR] {err}")
            has_error = True
        else:
            lines.append(f"  Setting:          {_format_reading_compact(setting, number_format)}")
            if not setting.ok:
                has_error = True

    # Analog alarm
    if props.has_analog_alarm:
        err, alarm = _section("Analog alarm", lambda: dev.analog_alarm())
        if _is_noprop(err):
            if verbose:
                lines.append(f"  Analog alarm:     {_noprop_str(err)}")
        elif err:
            lines.append(f"  Analog alarm:     [ERROR] {err}")
            has_error = True
        else:
            lines.append(f"  Analog alarm:     {_format_analog_alarm_compact(alarm, number_format)}")

    # Status
    if props.has_status:
        err, status = _section("Status", lambda: dev.status())
        if _is_noprop(err):
            if verbose:
                lines.append(f"  Status:           {_noprop_str(err)}")
        elif err:
            lines.append(f"  Status:           [ERROR] {err}")
            has_error = True
        else:
            lines.append(f"  Status:           {_format_status_compact(status)}")

    # Digital status
    if props.has_status:
        err, ds = _section("Digital status", lambda: dev.digital_status())
        if _is_noprop(err):
            if verbose:
                lines.append(f"  Digital status:   {_noprop_str(err)}")
        elif err:
            lines.append(f"  Digital status:   [ERROR] {err}")
            has_error = True
        elif verbose:
            lines.append("  Digital status:")
            lines.append(_format_digital_status_verbose(ds))
        else:
            lines.append(f"  Digital status:   {_format_digital_status_compact(ds)}")

    # Digital alarm
    if props.has_digital_alarm:
        err, da = _section("Digital alarm", lambda: dev.digital_alarm())
        if _is_noprop(err):
            if verbose:
                lines.append(f"  Digital alarm:    {_noprop_str(err)}")
        elif err:
            lines.append(f"  Digital alarm:    [ERROR] {err}")
            has_error = True
        elif verbose:
            summary, bits = _format_digital_alarm_verbose(da, props.ext_status_bits)
            lines.append(f"  Digital alarm:    {summary}")
            if bits:
                lines.append(bits)
        else:
            lines.append(f"  Digital alarm:    {_format_digital_alarm_compact(da)}")

    return lines, has_error


def _reading_to_json(reading) -> dict:
    """Convert a Reading to a JSON-safe dict."""
    if not reading.ok:
        return {"error": reading.message or f"error {reading.error_code}"}
    val = reading.value
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(val, np.ndarray):
            val = val.tolist()
        elif isinstance(val, np.integer):
            val = int(val)
        elif isinstance(val, np.floating):
            val = float(val)
    return {
        "value": val,
        "units": reading.units or None,
        "type": reading.value_type.name,
    }


def _digital_status_to_json(ds) -> list[dict]:
    """Convert DigitalStatus to a list of bit dicts."""
    return [{"bit": b.position, "value": b.is_set, "label": b.name, "text": b.value} for b in ds.bits]


def _build_json(dev, name: str, *, props: _DeviceProps) -> tuple[dict, bool]:
    """Build JSON dict for one device. Returns (dict, has_error)."""
    d: dict = {"device": name}
    if props.device_index is not None:
        d["device_index"] = props.device_index
    has_error = False

    err, desc = _section("description", lambda: dev.description())
    d["description"] = desc if not err else {"error": str(err)}
    has_error = has_error or bool(err)

    if props.has_reading:
        err, reading = _section("reading", lambda: dev.get(prop="reading"))
        # Backfill device_index from backend if DevDB wasn't available
        if "device_index" not in d and reading is not None and reading.ok and reading.meta:
            meta_di = reading.meta.device_index
            if meta_di:
                d["device_index"] = meta_di
        noprop = _is_noprop(err) or (reading is not None and not reading.ok and _is_noprop(reading.message))
        if not noprop:
            d["reading"] = _reading_to_json(reading) if not err else {"error": str(err)}
            has_error = has_error or bool(err) or (reading is not None and not reading.ok)

    if props.has_setting:
        err, setting = _section("setting", lambda: dev.get(prop="setting"))
        noprop = _is_noprop(err) or (setting is not None and not setting.ok and _is_noprop(setting.message))
        if not noprop:
            d["setting"] = _reading_to_json(setting) if not err else {"error": str(err)}
            has_error = has_error or bool(err) or (setting is not None and not setting.ok)

    if props.has_analog_alarm:
        err, alarm = _section("analog_alarm", lambda: dev.analog_alarm())
        if not _is_noprop(err):
            d["analog_alarm"] = alarm if not err else {"error": str(err)}
            has_error = has_error or bool(err)

    if props.has_status:
        err, status = _section("status", lambda: dev.status())
        if not _is_noprop(err):
            d["status"] = status if not err else {"error": str(err)}
            has_error = has_error or bool(err)

    if props.has_status:
        err, ds = _section("digital_status", lambda: dev.digital_status())
        if not _is_noprop(err):
            d["digital_status"] = _digital_status_to_json(ds) if not err else {"error": str(err)}
            has_error = has_error or bool(err)

    if props.has_digital_alarm:
        err, da = _section("digital_alarm", lambda: dev.digital_alarm())
        if not _is_noprop(err):
            d["digital_alarm"] = da if not err else {"error": str(err)}
            has_error = has_error or bool(err)

    return d, has_error


def _get_devdb():
    """Get DevDB client, or None if unavailable."""
    import pacsys

    devdb = pacsys._get_global_devdb()
    if devdb is None:
        print("Warning: DevDB not configured (set PACSYS_DEVDB_HOST to enable)", file=sys.stderr)
        return None
    try:
        devdb.get_device_info(["Z:NO_OP"], timeout=2.0)
        return devdb
    except Exception:
        print(
            f"Warning: DevDB unreachable at {devdb._host}:{devdb._port} — some metadata will be unavailable",
            file=sys.stderr,
        )
        return None


def _query_device_props(devdb, name: str) -> _DeviceProps:
    """Query DevDB to determine which properties a device supports.

    DevDB reliably tells us about reading/setting/status. Alarm properties
    (ANALOG/DIGITAL) are always attempted — NOPROP suppression handles the rest.
    """
    if devdb is None:
        return _DeviceProps()  # all True = query everything

    try:
        info_map = devdb.get_device_info([name], timeout=2.0)
        info = info_map.get(name)
    except Exception:
        return _DeviceProps()

    if info is None:
        return _DeviceProps()

    return _DeviceProps(
        has_reading=info.reading is not None,
        has_setting=info.setting is not None,
        has_status=info.status_bits is not None,
        device_index=info.device_index,
        ext_status_bits=info.ext_status_bits,
    )


def main() -> int:
    parser = base_parser("Display ACNET device metadata and status")
    parser.add_argument("devices", nargs="+", metavar="DEVICE", help="device name(s) or DRF string(s)")
    parser.add_argument("-f", "--number-format", default=None, help="Python format spec for numeric values")
    args = parser.parse_args()

    try:
        backend = make_backend(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        devdb = _get_devdb()

        is_json = args.output_format == "json"
        first = True
        has_error = False

        for drf in args.devices:
            name = get_device_name(drf)
            dev = Device(name, backend=backend)
            props = _query_device_props(devdb, name)

            if is_json:
                data, err = _build_json(dev, name, props=props)
                print(json.dumps(data))
            else:
                if not first:
                    print()  # blank line between devices
                lines, err = _display_text(
                    dev, name, verbose=args.verbose, number_format=args.number_format, props=props
                )
                print("\n".join(lines))
            if err:
                has_error = True
            first = False
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    finally:
        backend.close()

    return EXIT_DEVICE_ERROR if has_error else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
