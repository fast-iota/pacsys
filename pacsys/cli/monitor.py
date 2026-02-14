"""acmonitor / pacsys-monitor -- Stream device readings."""

import signal
import sys
import time
from collections import defaultdict

from pacsys.cli._common import (
    EXIT_DEVICE_ERROR,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    base_parser,
    format_reading,
    make_backend,
    parse_slice,
)
from pacsys.drf3 import parse_request


def _ensure_event(drf: str) -> str:
    """Append @p,1000 if no event specified."""
    req = parse_request(drf)
    if req.event is None or req.event.mode == "U":
        return f"{drf}@p,1000"
    return drf


def main() -> int:
    parser = base_parser("Monitor ACNET device readings (streaming)")
    parser.add_argument("devices", nargs="+", metavar="DEVICE")
    parser.add_argument("-f", "--number-format", default=None)
    parser.add_argument("-r", "--range", dest="array_range", default=None)
    parser.add_argument("-n", "--count", type=int, default=None)
    parser.add_argument("-s", "--timestamp-format", choices=["iso", "epoch", "relative"], default="iso")
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

    # Resolve DRFs with default event
    try:
        drfs = [_ensure_event(d) for d in args.devices]
    except ValueError as e:
        print(f"Invalid DRF: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        backend = make_backend(args)
    except Exception as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    # SIGTERM should trigger clean shutdown just like Ctrl+C
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    counts: dict[str, int] = defaultdict(int)
    total = 0
    has_error = False
    t0 = time.monotonic()
    reference_time: float | None = None

    try:
        with backend.subscribe(drfs) as handle:
            for reading, _handle in handle.readings(timeout=None):
                if args.timestamp_format == "relative" and reference_time is None and reading.timestamp:
                    reference_time = reading.timestamp.timestamp()
                line = format_reading(
                    reading,
                    fmt=fmt,
                    number_format=args.number_format,
                    array_slice=array_slice,
                    timestamp_format=args.timestamp_format,
                    reference_time=reference_time,
                )
                print(line, flush=True)
                counts[reading.name] += 1
                total += 1
                if not reading.ok:
                    has_error = True
                if args.count is not None and total >= args.count:
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_DEVICE_ERROR
    finally:
        backend.close()

    # Print summary to stderr
    elapsed = time.monotonic() - t0
    parts = [f"{n} readings from {dev}" for dev, n in counts.items()]
    summary = ", ".join(parts)
    print(f"--- {summary} (total in {elapsed:.1f}s) ---", file=sys.stderr)

    return EXIT_DEVICE_ERROR if has_error else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
