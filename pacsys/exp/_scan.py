"""scan: ramp one device, read others at each step."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Callable, TYPE_CHECKING

from pacsys.types import DeviceSpec, Reading, Value, WriteResult
from pacsys.exp._resolve import resolve_drf, resolve_backend

if TYPE_CHECKING:
    from pacsys.backends import Backend
    from pacsys.verify import Verify

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanResult:
    """Result of a parameter scan."""

    write_device: str
    read_devices: list[str]
    set_values: list[float]
    readings: list[dict[str, Reading]]
    write_results: list[WriteResult]
    aborted: bool = False
    restored: bool = False

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        rows = []
        for i, sv in enumerate(self.set_values):
            row: dict[str, object] = {"set_value": sv}
            if i < len(self.readings):
                for drf, reading in self.readings[i].items():
                    row[drf] = reading.value
            rows.append(row)
        return pd.DataFrame(rows)


def scan(
    write_device: DeviceSpec,
    read_devices: list[DeviceSpec],
    *,
    values: list[float] | None = None,
    start: float | None = None,
    stop: float | None = None,
    steps: int | None = None,
    settle: float = 0.5,
    readings_per_step: int = 1,
    verify: bool | Verify | None = None,
    restore: bool = True,
    abort_if: Callable[[dict[str, Reading]], bool] | None = None,
    timeout: float | None = None,
    backend: Backend | None = None,
) -> ScanResult:
    """Ramp write_device through values, read read_devices at each step.

    Provide either `values` (explicit list) or `start`/`stop`/`steps`
    (linear range). Exactly one mode must be used.
    """
    write_drf = resolve_drf(write_device)
    read_drfs = [resolve_drf(d) for d in read_devices]
    be = resolve_backend(backend)

    if readings_per_step < 1:
        raise ValueError("readings_per_step must be >= 1")

    scan_values = _build_values(values, start, stop, steps)

    from pacsys.device import Device

    write_dev = Device(write_drf, backend=be)

    # Save original SETTING value for restore (not READING!)
    original: Value | None = None
    if restore:
        original = write_dev.setting(timeout=timeout)

    all_readings: list[dict[str, Reading]] = []
    all_write_results: list[WriteResult] = []
    aborted = False

    try:
        for sv in scan_values:
            wr = write_dev.write(sv, verify=verify, timeout=timeout)
            all_write_results.append(wr)

            if not wr.ok:
                logger.warning("Write failed at value %s: %s", sv, wr.message)
                break

            if settle > 0:
                time.sleep(settle)

            step_readings = _read_step(be, read_drfs, readings_per_step, timeout)
            all_readings.append(step_readings)

            if abort_if is not None and abort_if(step_readings):
                aborted = True
                break
    except BaseException:
        # Restore on exception — log failure but don't mask the original
        if restore and original is not None:
            try:
                write_dev.write(original, timeout=timeout)
            except Exception:
                logger.exception("Failed to restore %s to %s during error cleanup", write_drf, original)
        raise
    else:
        # Normal completion — restore and raise on failure
        restored = False
        if restore and original is not None:
            restore_result = write_dev.write(original, timeout=timeout)
            restored = restore_result.ok
            if not restored:
                raise RuntimeError(
                    f"Scan completed but failed to restore {write_drf} to {original}: {restore_result.message}"
                )

    return ScanResult(
        write_device=write_drf,
        read_devices=read_drfs,
        set_values=[sv for sv, _ in zip(scan_values, all_write_results)],
        readings=all_readings,
        write_results=all_write_results,
        aborted=aborted,
        restored=restored,
    )


def _build_values(
    values: list[float] | None,
    start: float | None,
    stop: float | None,
    steps: int | None,
) -> list[float]:
    """Build scan value sequence from explicit list or linear range."""
    has_explicit = values is not None
    has_range = any(x is not None for x in (start, stop, steps))

    if has_explicit and has_range:
        raise ValueError("Provide either 'values' or 'start/stop/steps', not both")
    if not has_explicit and not has_range:
        raise ValueError("Provide either 'values' or 'start/stop/steps'")

    if has_explicit:
        if not values:
            raise ValueError("values cannot be empty")
        return list(values)

    if start is None or stop is None or steps is None:
        raise ValueError("All of start, stop, steps must be provided for linear range")
    if steps < 2:
        raise ValueError("steps must be >= 2")
    step_size = (stop - start) / (steps - 1)
    return [start + i * step_size for i in range(steps)]


def _read_step(
    backend: Backend,
    read_drfs: list[str],
    readings_per_step: int,
    timeout: float | None,
) -> dict[str, Reading]:
    """Read all devices, optionally multiple times and average."""
    if readings_per_step == 1:
        readings = backend.get_many(read_drfs, timeout=timeout)
        return dict(zip(read_drfs, readings))

    accumulated: dict[str, list[Reading]] = {drf: [] for drf in read_drfs}
    for _ in range(readings_per_step):
        readings = backend.get_many(read_drfs, timeout=timeout)
        for drf, r in zip(read_drfs, readings):
            accumulated[drf].append(r)

    result: dict[str, Reading] = {}
    for drf, rs in accumulated.items():
        ok_readings = []
        for r in rs:
            if r.ok and not isinstance(r.value, bool) and type(r.value).__name__ != "bool_":
                try:
                    float(r.value)  # type: ignore[arg-type]
                    ok_readings.append(r)
                except (TypeError, ValueError):
                    pass
        if ok_readings:
            numeric = [float(r.value) for r in ok_readings]
            avg = sum(numeric) / len(numeric)
            result[drf] = replace(ok_readings[-1], value=avg)
        else:
            result[drf] = rs[-1]
    return result
