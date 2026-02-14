"""acget / pacsys-get -- Read device values."""

import sys

from pacsys.cli._common import (
    EXIT_DEVICE_ERROR,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    base_parser,
    format_reading,
    make_backend,
    parse_slice,
)


def main() -> int:
    parser = base_parser("Read ACNET device values")
    parser.add_argument("devices", nargs="+", metavar="DEVICE", help="DRF device string(s)")
    parser.add_argument("-f", "--number-format", default=None, help="Python format spec")
    parser.add_argument("-r", "--range", dest="array_range", default=None, help="array slice")
    args = parser.parse_args()

    # Parse array slice
    array_slice = None
    if args.array_range:
        try:
            array_slice = parse_slice(args.array_range)
        except ValueError as e:
            print(f"Invalid range: {e}", file=sys.stderr)
            return EXIT_USAGE_ERROR

    fmt = "terse" if args.terse else args.output_format
    has_error = False

    try:
        backend = make_backend(args)
    except Exception as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        if len(args.devices) == 1:
            reading = backend.get(args.devices[0], timeout=args.timeout)
            print(
                format_reading(
                    reading,
                    fmt=fmt,
                    number_format=args.number_format,
                    array_slice=array_slice,
                )
            )
            if reading.is_error:
                has_error = True
        else:
            readings = backend.get_many(args.devices, timeout=args.timeout)
            for reading in readings:
                print(
                    format_reading(
                        reading,
                        fmt=fmt,
                        number_format=args.number_format,
                        array_slice=array_slice,
                    )
                )
                if reading.is_error:
                    has_error = True
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    finally:
        backend.close()

    return EXIT_DEVICE_ERROR if has_error else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
