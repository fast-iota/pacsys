"""acinfo / pacsys-info -- Display device metadata and status."""

import json
import sys

import numpy as np

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


def _section(label: str, fn, *, verbose: bool = False, number_format: str | None = None) -> tuple[str | None, object]:
    """Call fn(), return (error_msg, result). On exception, return (msg, None)."""
    try:
        result = fn()
        return None, result
    except Exception as e:
        return str(e), None


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
            val_str = format_value(v, number_format) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
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
    lines = []
    for b in ds.bits:
        lines.append(f"    Bit {b.position}:  {'1' if b.is_set else '0'}  {b.name}: {b.value}")
    return "\n".join(lines)


def _format_digital_alarm_compact(alarm) -> str:
    """Format digital alarm as nominal/mask."""
    if isinstance(alarm, dict):
        parts = []
        for k, v in alarm.items():
            if isinstance(v, int):
                parts.append(f"{k}={v:08b}")
            else:
                parts.append(f"{k}={v}")
        return "  ".join(parts)
    return str(alarm)


def _format_digital_alarm_verbose(alarm) -> str:
    """Format digital alarm with per-bit detail."""
    if not isinstance(alarm, dict):
        return str(alarm)
    nominal = alarm.get("nominal", 0)
    mask = alarm.get("mask", 0)
    if not isinstance(nominal, int) or not isinstance(mask, int):
        return _format_digital_alarm_compact(alarm)
    max_bits = max(nominal.bit_length(), mask.bit_length(), 1)
    lines = []
    for i in range(max_bits):
        n_bit = (nominal >> i) & 1
        m_bit = (mask >> i) & 1
        monitored = "monitored" if m_bit else "not monitored"
        lines.append(f"    Bit {i}:  nominal={n_bit}  mask={m_bit}  ({monitored})")
    return "\n".join(lines)


def _display_text(dev, name: str, *, verbose: bool, number_format: str | None) -> tuple[list[str], bool]:
    """Build text output lines for one device. Returns (lines, has_error)."""
    lines = [name]
    has_error = False

    # Description
    err, desc = _section("Description", lambda: dev.description())
    if err:
        lines.append(f"  Description:      [ERROR] {err}")
        has_error = True
    else:
        lines.append(f"  Description:      {desc}")

    # Reading
    err, reading = _section("Reading", lambda: dev.get(prop="reading"))
    if err:
        lines.append(f"  Reading:          [ERROR] {err}")
        has_error = True
    else:
        lines.append(f"  Reading:          {_format_reading_compact(reading, number_format)}")
        if not reading.ok:
            has_error = True

    # Setting
    err, setting = _section("Setting", lambda: dev.get(prop="setting"))
    if err:
        lines.append(f"  Setting:          [ERROR] {err}")
        has_error = True
    else:
        lines.append(f"  Setting:          {_format_reading_compact(setting, number_format)}")
        if not setting.ok:
            has_error = True

    # Analog alarm
    err, alarm = _section("Analog alarm", lambda: dev.analog_alarm())
    if err:
        lines.append(f"  Analog alarm:     [ERROR] {err}")
        has_error = True
    else:
        lines.append(f"  Analog alarm:     {_format_analog_alarm_compact(alarm, number_format)}")

    # Status
    err, status = _section("Status", lambda: dev.status())
    if err:
        lines.append(f"  Status:           [ERROR] {err}")
        has_error = True
    else:
        lines.append(f"  Status:           {_format_status_compact(status)}")

    # Digital status
    err, ds = _section("Digital status", lambda: dev.digital_status())
    if err:
        lines.append(f"  Digital status:   [ERROR] {err}")
        has_error = True
    elif verbose:
        lines.append("  Digital status:")
        lines.append(_format_digital_status_verbose(ds))
    else:
        lines.append(f"  Digital status:   {_format_digital_status_compact(ds)}")

    # Digital alarm
    err, da = _section("Digital alarm", lambda: dev.digital_alarm())
    if err:
        lines.append(f"  Digital alarm:    [ERROR] {err}")
        has_error = True
    elif verbose:
        lines.append("  Digital alarm:")
        lines.append(_format_digital_alarm_verbose(da))
    else:
        lines.append(f"  Digital alarm:    {_format_digital_alarm_compact(da)}")

    return lines, has_error


def _reading_to_json(reading) -> dict:
    """Convert a Reading to a JSON-safe dict."""
    if not reading.ok:
        return {"error": reading.message or f"error {reading.error_code}"}
    val = reading.value
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


def _build_json(dev, name: str) -> tuple[dict, bool]:
    """Build JSON dict for one device. Returns (dict, has_error)."""
    d: dict = {"device": name}
    has_error = False

    err, desc = _section("description", lambda: dev.description())
    d["description"] = desc if not err else {"error": err}
    has_error = has_error or bool(err)

    err, reading = _section("reading", lambda: dev.get(prop="reading"))
    d["reading"] = _reading_to_json(reading) if not err else {"error": err}
    has_error = has_error or bool(err) or (reading is not None and not reading.ok)

    err, setting = _section("setting", lambda: dev.get(prop="setting"))
    d["setting"] = _reading_to_json(setting) if not err else {"error": err}
    has_error = has_error or bool(err) or (setting is not None and not setting.ok)

    err, alarm = _section("analog_alarm", lambda: dev.analog_alarm())
    d["analog_alarm"] = alarm if not err else {"error": err}
    has_error = has_error or bool(err)

    err, status = _section("status", lambda: dev.status())
    d["status"] = status if not err else {"error": err}
    has_error = has_error or bool(err)

    err, ds = _section("digital_status", lambda: dev.digital_status())
    d["digital_status"] = _digital_status_to_json(ds) if not err else {"error": err}
    has_error = has_error or bool(err)

    err, da = _section("digital_alarm", lambda: dev.digital_alarm())
    d["digital_alarm"] = da if not err else {"error": err}
    has_error = has_error or bool(err)

    return d, has_error


def _check_devdb(args) -> None:
    """Probe DevDB and print a warning if unreachable."""
    import pacsys

    devdb = pacsys._get_global_devdb()
    if devdb is None:
        print("Warning: DevDB not configured (set PACSYS_DEVDB_HOST to enable)", file=sys.stderr)
        return
    host, port = devdb._host, devdb._port
    try:
        devdb.get_device_info(["Z:NO_OP"], timeout=2.0)
    except Exception:
        print(f"Warning: DevDB unreachable at {host}:{port} — some metadata will be unavailable", file=sys.stderr)


def main() -> int:
    parser = base_parser("Display ACNET device metadata and status")
    parser.add_argument("devices", nargs="+", metavar="DEVICE", help="device name(s) or DRF string(s)")
    parser.add_argument("-f", "--number-format", default=None, help="Python format spec for numeric values")
    args = parser.parse_args()

    try:
        backend = make_backend(args)
    except Exception as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        # Check DevDB availability and warn if unreachable
        _check_devdb(args)

        is_json = args.output_format == "json"
        first = True
        has_error = False

        for drf in args.devices:
            name = get_device_name(drf)
            dev = Device(name, backend=backend)

            if is_json:
                data, err = _build_json(dev, name)
                print(json.dumps(data))
            else:
                if not first:
                    print()  # blank line between devices
                lines, err = _display_text(dev, name, verbose=args.verbose, number_format=args.number_format)
                print("\n".join(lines))
            if err:
                has_error = True
            first = False
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    finally:
        backend.close()

    return EXIT_DEVICE_ERROR if has_error else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
