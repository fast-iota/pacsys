"""acput / pacsys-put -- Write device values."""

import sys

from pacsys.cli._common import (
    EXIT_DEVICE_ERROR,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    base_parser,
    format_write_result,
    make_backend,
    parse_value,
)


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
        value = parse_value(args.pairs[i + 1])
        settings.append((drf, value))

    fmt = "terse" if args.terse else args.output_format
    use_verify = args.verify or args.tolerance is not None

    try:
        backend = make_backend(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    has_error = False
    try:
        if use_verify:
            from pacsys.device import Device
            from pacsys.verify import Verify

            verify_cfg = Verify(
                tolerance=args.tolerance if args.tolerance is not None else 0.0,
                max_attempts=args.retries,
            )
            for drf, value in settings:
                dev = Device(drf, backend=backend)
                result = dev.write(value, verify=verify_cfg, timeout=args.timeout)
                print(format_write_result(result, fmt=fmt))
                if not result.ok:
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
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    finally:
        backend.close()

    return EXIT_DEVICE_ERROR if has_error else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
