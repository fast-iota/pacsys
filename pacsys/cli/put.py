"""acput / pacsys-put -- Write device values."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from pacsys.cli._common import (
    EXIT_DEVICE_ERROR,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    base_parser,
    format_write_result,
    make_backend,
    parse_value,
)
from pacsys.drf3 import parse_request
from pacsys.drf_utils import prepare_for_control
from pacsys.types import BasicControl

from .._device_base import _WritePlan
from ..device import Device
from ..drf3.field import DRF_FIELD
from ..drf3.property import DRF_PROPERTY
from ..types import Value

if TYPE_CHECKING:
    from ..verify import Verify


def _plan_verified(drf: str, value: Value, verify: Verify) -> tuple[Device, _WritePlan]:
    """Plan a verified write to the DRF's own writable target (Device.write() always targets SETTING)."""
    dev = Device(drf)
    if isinstance(value, BasicControl):
        return dev, dev._plan_control(value, verify)
    req = dev.request
    # parse_value never yields bytes, so a RAW readback cannot match (EPICS .RAW is a PV suffix, not this field)
    if req.is_acnet and req.field == DRF_FIELD.RAW:
        raise ValueError(f"--verify cannot check RAW fields (CLI values are not bytes): {drf}")
    if not req.is_acnet or req.property in (DRF_PROPERTY.READING, DRF_PROPERTY.SETTING):
        return dev, dev._plan_write(value, None, verify)
    # A whole alarm block reads back as a dict, which no CLI value can match
    if req.property in (DRF_PROPERTY.ANALOG, DRF_PROPERTY.DIGITAL) and req.field not in (None, DRF_FIELD.ALL):
        prop, field = req.property, req.field
        return dev, _WritePlan(dev._build_drf(prop, field, "N"), value, value, verify, dev._build_drf(prop, field, "I"))
    raise ValueError(f"--verify supports SETTING, basic control, and single ANALOG/DIGITAL alarm fields, not {drf}")


def main() -> int:
    parser = base_parser("Write ACNET device values")
    parser.add_argument("pairs", nargs="+", metavar="DEVICE VALUE", help="alternating device/value pairs")
    parser.add_argument("--verify", action="store_true", help="read back after write to confirm")
    parser.add_argument("--tolerance", type=float, default=None, help="numeric tolerance (implies --verify)")
    parser.add_argument("--retries", type=int, default=3, help="verify retry count")
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        print("Error: arguments must be alternating DEVICE VALUE pairs", file=sys.stderr)
        return EXIT_USAGE_ERROR

    if args.retries < 1:
        print("Error: --retries must be at least 1", file=sys.stderr)
        return EXIT_USAGE_ERROR

    # Parse device/value pairs
    settings = []
    for i in range(0, len(args.pairs), 2):
        drf = args.pairs[i]
        try:
            parse_request(drf)
            value = parse_value(args.pairs[i + 1])
            # BasicControl values target CONTROL property regardless of DRF form
            if isinstance(value, BasicControl):
                drf = prepare_for_control(drf)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return EXIT_USAGE_ERROR
        settings.append((drf, value))

    fmt = "terse" if args.terse else args.output_format
    use_verify = args.verify or args.tolerance is not None

    # Plan every verified write before connecting so no pair is written if a later one is unsupported
    plans: list[tuple[Device, _WritePlan]] = []
    if use_verify:
        from ..verify import Verify

        try:
            verify_cfg = Verify(
                tolerance=args.tolerance if args.tolerance is not None else 0.0,
                max_attempts=args.retries,
            )
            plans = [_plan_verified(drf, value, verify_cfg) for drf, value in settings]
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return EXIT_USAGE_ERROR

    try:
        backend = make_backend(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"Connection error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    has_error = False
    try:
        if use_verify:
            for dev, plan in plans:
                result = dev.with_backend(backend)._execute(plan, args.timeout)
                print(format_write_result(result, fmt=fmt))
                if not result.confirmed:
                    has_error = True
        elif len(settings) == 1:
            drf, value = settings[0]
            result = backend.write(drf, value, timeout=args.timeout)
            print(format_write_result(result, fmt=fmt))
            if not result.ok:
                has_error = True
        else:
            results = backend.write_many(settings, timeout=args.timeout)
            for result in results:
                print(format_write_result(result, fmt=fmt))
                if not result.ok:
                    has_error = True
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_DEVICE_ERROR
    finally:
        backend.close()

    return EXIT_DEVICE_ERROR if has_error else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
