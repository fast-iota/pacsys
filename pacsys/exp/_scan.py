"""scan: ramp one device, read others at each step."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from pacsys.exp._resolve import resolve_backend, resolve_drf
from pacsys.exp._values import numeric_value

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pacsys.backends import Backend
    from pacsys.types import DeviceSpec, Reading, Value, WriteResult
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
    aborted: bool = False  # abort_if fired or a write failed/unconfirmed before the last value
    restored: bool = False

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas") from None

        rows = []
        for i, sv in enumerate(self.set_values):
            row: dict[str, object] = {"set_value": sv}
            if i < len(self.readings):
                for drf, reading in self.readings[i].items():
                    row[drf] = reading.value
            rows.append(row)
        return pd.DataFrame(rows)


class ScanRestoreError(RuntimeError):
    """A completed scan whose original setting could not be restored."""

    def __init__(self, message: str, result: ScanResult):
        super().__init__(message)
        self.result = result


def scan(
    write_device: DeviceSpec,
    read_devices: list[DeviceSpec],
    *,
    values: Iterable[float] | None = None,
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

    Provide either `values` (any iterable of setpoints, e.g. list or
    np.linspace) or `start`/`stop`/`steps` (linear range). Exactly one
    mode must be used.

    With ``readings_per_step > 1`` OK readings are averaged per step (arrays
    element-wise); non-numeric values or mismatched array shapes raise.

    Raises:
        ScanRestoreError: If the scan completes but restoring the original
            setting fails. The collected data is available on ``result``.
        TypeError, ValueError: If ``readings_per_step > 1`` and a read
            device returns values that cannot be averaged.
    """
    write_drf = resolve_drf(write_device)
    read_drfs = [resolve_drf(d) for d in read_devices]
    be = resolve_backend(backend)

    if readings_per_step < 1:
        raise ValueError("readings_per_step must be >= 1")

    scan_values = _build_values(values, start, stop, steps)

    from pacsys.device import Device

    write_dev = Device(write_drf, backend=be)

    # Read the original setting so it can be restored.
    original: Value | None = None
    if restore:
        original = write_dev.setting(timeout=timeout)

    all_readings: list[dict[str, Reading]] = []
    all_write_results: list[WriteResult] = []
    aborted = False
    restored = False
    restore_error: str | None = None

    try:
        for sv in scan_values:
            wr = write_dev.write(sv, verify=verify, timeout=timeout)
            all_write_results.append(wr)

            if not wr.confirmed:
                if wr.ok:
                    detail = wr.message or f"readback={wr.readback!r}"
                    logger.warning("Write verification failed at value %s: %s", sv, detail)
                else:
                    logger.warning("Write failed at value %s: %s", sv, wr.message)
                aborted = True
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
                restore_result = write_dev.write(original, timeout=timeout)
                if not restore_result.ok:
                    detail = restore_result.message or f"error_code={restore_result.error_code}"
                    logger.error("Failed to restore %s to %s during error cleanup: %s", write_drf, original, detail)
            except Exception:
                logger.exception("Failed to restore %s to %s during error cleanup", write_drf, original)
        raise
    else:
        # Normal completion — restore and raise on failure
        if restore and original is not None:
            restore_result = write_dev.write(original, timeout=timeout)
            restored = restore_result.ok
            if not restored:
                restore_error = (
                    f"Scan completed but failed to restore {write_drf} to {original}: {restore_result.message}"
                )

    result = ScanResult(
        write_device=write_drf,
        read_devices=read_drfs,
        set_values=[sv for sv, _ in zip(scan_values, all_write_results, strict=False)],
        readings=all_readings,
        write_results=all_write_results,
        aborted=aborted,
        restored=restored,
    )
    if restore_error is not None:
        raise ScanRestoreError(restore_error, result)
    return result


def _build_values(
    values: Iterable[float] | None,
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
        # Materialize first: ndarray truthiness is ambiguous/element-based
        values = list(values)
        if not values:
            raise ValueError("values cannot be empty")
        return values

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
        return dict(zip(read_drfs, readings, strict=True))

    accumulated: dict[str, list[Reading]] = {drf: [] for drf in read_drfs}
    for _ in range(readings_per_step):
        readings = backend.get_many(read_drfs, timeout=timeout)
        for drf, r in zip(read_drfs, readings, strict=True):
            accumulated[drf].append(r)

    result: dict[str, Reading] = {}
    for drf, rs in accumulated.items():
        ok_readings = [r for r in rs if r.ok]
        if not ok_readings:
            result[drf] = rs[-1]
            continue

        values = [r.value for r in ok_readings]
        array_like = [isinstance(value, (np.ndarray, list, tuple)) for value in values]
        if any(array_like):
            if not all(array_like):
                raise TypeError(f"Cannot average mixed scalar and array readings for {drf}")
            arrays = [np.asarray(value) for value in values]
            if any(array.dtype.kind not in "iufc" for array in arrays):
                raise TypeError(f"Cannot average non-numeric array readings for {drf}")
            try:
                avg = np.mean(np.stack(arrays), axis=0)
            except ValueError as e:
                raise ValueError(f"Cannot average array readings for {drf}: {e}") from None
        else:
            try:
                numeric_values = [numeric_value(value) for value in values]
            except (TypeError, ValueError) as e:
                raise TypeError(f"Cannot average non-numeric readings for {drf}: {e}") from None
            avg = sum(numeric_values) / len(numeric_values)
        result[drf] = replace(ok_readings[-1], value=avg)
    return result
