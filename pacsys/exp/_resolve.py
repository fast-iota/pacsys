"""Shared DRF/backend resolution for exp utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pacsys.types import DeviceSpec

if TYPE_CHECKING:
    from pacsys.backends import Backend

from pacsys.device import Device


def resolve_drf(device: DeviceSpec) -> str:
    """Convert DeviceSpec to DRF string."""
    if isinstance(device, str):
        return device
    if isinstance(device, Device):
        return device.drf
    raise TypeError(f"Expected str or Device, got {type(device).__name__}")


def resolve_backend(backend: Backend | None = None) -> Backend:
    """Return provided backend or the global default."""
    if backend is not None:
        return backend
    from pacsys import _get_global_backend

    return _get_global_backend()
