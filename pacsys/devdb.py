"""
DevDB gRPC client for querying device metadata from the master PostgreSQL database.

DevDB is a metadata service (not a data acquisition backend). It provides device
information like scaling parameters, control commands, and status bit definitions.

Usage:
    import pacsys

    with pacsys.devdb(host="localhost", port=45678) as db:
        info = db.get_device_info(["Z:ACLTST", "M:OUTTMP"])
        print(info["Z:ACLTST"].description)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from pacsys.errors import DeviceError

logger = logging.getLogger(__name__)

_import_error = ""
try:
    import grpc
    from pacsys._proto.controls.service.DevDB.v1 import DevDB_pb2, DevDB_pb2_grpc

    DEVDB_AVAILABLE = True
except (ImportError, TypeError) as e:
    DEVDB_AVAILABLE = False
    grpc = None  # type: ignore[assignment]
    DevDB_pb2 = None  # type: ignore[assignment]
    DevDB_pb2_grpc = None  # type: ignore[assignment]
    _import_error = str(e)


# ─── Result Dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PropertyInfo:
    """Scaling and metadata for a device property (reading or setting)."""

    primary_units: str | None
    common_units: str | None
    min_val: float
    max_val: float
    p_index: int
    c_index: int
    coeff: tuple[float, ...]
    is_step_motor: bool
    is_destructive_read: bool
    is_fe_scaling: bool
    is_contr_setting: bool
    is_knobbable: bool


@dataclass(frozen=True)
class StatusBitDef:
    """One status bit definition from DevDB (not a runtime value).

    Uses mask/match/invert for proper bit evaluation:
      is_active = ((raw_value & mask) == match) ^ invert
    """

    mask: int
    match: int
    invert: bool
    short_name: str
    long_name: str
    true_str: str
    false_str: str
    true_color: int
    true_char: str
    false_color: int
    false_char: str


@dataclass(frozen=True)
class ExtStatusBitDef:
    """Extended status bit definition from DevDB."""

    bit_no: int
    description: str
    name0: str
    name1: str
    color0: int
    color1: int


@dataclass(frozen=True)
class ControlCommandDef:
    """A control command definition from DevDB."""

    value: int
    short_name: str
    long_name: str


@dataclass(frozen=True)
class AlarmBlockInfo:
    """Alarm block information from DevDB."""

    di: int
    pi: int
    status: int
    min_or_nom: int
    max_or_tol: int
    tries_needed: int
    tries_now: int
    clock_event_no: int
    subfunction_code: int
    specific_data: str
    segment: int


@dataclass(frozen=True)
class AlarmInfo:
    """Alarm information for a device from DevDB."""

    device_name: str
    alarm_block: AlarmBlockInfo
    analog_alarm_text_id: int | None
    digital_alarm_condition: int | None
    digital_alarm_mask: int | None
    digital_alarm_text_id: int | None


@dataclass(frozen=True)
class AlarmText:
    """Alarm text entry from DevDB."""

    alarm_text_id: int
    length: int
    priority: int
    hand_code: int
    sound_id: int
    speech_id: int
    spare: int
    text: str
    url: str


@dataclass(frozen=True)
class DeviceInfoResult:
    """Complete device metadata from DevDB."""

    device_index: int
    description: str
    reading: PropertyInfo | None
    setting: PropertyInfo | None
    control: tuple[ControlCommandDef, ...] | None
    status_bits: tuple[StatusBitDef, ...] | None
    ext_status_bits: tuple[ExtStatusBitDef, ...] | None


# ─── Proto-to-dataclass conversion ───────────────────────────────────────────


def _convert_property(proto_prop) -> PropertyInfo:
    return PropertyInfo(
        primary_units=proto_prop.primary_units if proto_prop.HasField("primary_units") else None,
        common_units=proto_prop.common_units if proto_prop.HasField("common_units") else None,
        min_val=proto_prop.min_val,
        max_val=proto_prop.max_val,
        p_index=proto_prop.p_index,
        c_index=proto_prop.c_index,
        coeff=tuple(proto_prop.coeff),
        is_step_motor=proto_prop.is_step_motor,
        is_destructive_read=proto_prop.is_destructive_read,
        is_fe_scaling=proto_prop.is_fe_scaling,
        is_contr_setting=proto_prop.is_contr_setting,
        is_knobbable=proto_prop.is_knobbable,
    )


def _convert_status_bit(proto_bit) -> StatusBitDef:
    return StatusBitDef(
        mask=proto_bit.mask_val,
        match=proto_bit.match_val,
        invert=proto_bit.invert,
        short_name=proto_bit.short_name,
        long_name=proto_bit.long_name,
        true_str=proto_bit.true_str,
        false_str=proto_bit.false_str,
        true_color=proto_bit.true_color,
        true_char=proto_bit.true_char,
        false_color=proto_bit.false_color,
        false_char=proto_bit.false_char,
    )


def _convert_ext_status_bit(proto_bit) -> ExtStatusBitDef:
    return ExtStatusBitDef(
        bit_no=proto_bit.bit_no,
        description=proto_bit.description,
        name0=proto_bit.name0,
        name1=proto_bit.name1,
        color0=proto_bit.color0,
        color1=proto_bit.color1,
    )


def _convert_control_cmd(proto_cmd) -> ControlCommandDef:
    return ControlCommandDef(
        value=proto_cmd.value,
        short_name=proto_cmd.short_name,
        long_name=proto_cmd.long_name,
    )


def _convert_device_info(proto_info) -> DeviceInfoResult:
    reading = _convert_property(proto_info.reading) if proto_info.HasField("reading") else None
    setting = _convert_property(proto_info.setting) if proto_info.HasField("setting") else None

    control = None
    if proto_info.HasField("control"):
        control = tuple(_convert_control_cmd(c) for c in proto_info.control.cmds)

    status_bits = None
    ext_status_bits = None
    if proto_info.HasField("status"):
        status_bits = tuple(_convert_status_bit(b) for b in proto_info.status.bits)
        ext_status_bits = tuple(_convert_ext_status_bit(b) for b in proto_info.status.ext_bits)

    return DeviceInfoResult(
        device_index=proto_info.device_index,
        description=proto_info.description,
        reading=reading,
        setting=setting,
        control=control,
        status_bits=status_bits,
        ext_status_bits=ext_status_bits,
    )


def _convert_alarm_block(proto_block) -> AlarmBlockInfo:
    return AlarmBlockInfo(
        di=proto_block.di,
        pi=proto_block.pi,
        status=proto_block.status,
        min_or_nom=proto_block.min_or_nom,
        max_or_tol=proto_block.max_or_tol,
        tries_needed=proto_block.tries_needed,
        tries_now=proto_block.tries_now,
        clock_event_no=proto_block.clock_event_no,
        subfunction_code=proto_block.subfunction_code,
        specific_data=proto_block.specific_data,
        segment=proto_block.segment,
    )


def _convert_alarm_info(proto_alarm) -> AlarmInfo:
    analog_text_id = None
    if proto_alarm.HasField("device_analog_alarm"):
        analog_text_id = proto_alarm.device_analog_alarm.alarm_text_id

    dig_condition = None
    dig_mask = None
    dig_text_id = None
    if proto_alarm.HasField("device_digital_alarm"):
        dig_condition = proto_alarm.device_digital_alarm.condition
        dig_mask = proto_alarm.device_digital_alarm.mask
        dig_text_id = proto_alarm.device_digital_alarm.alarm_text_id

    return AlarmInfo(
        device_name=proto_alarm.device_name,
        alarm_block=_convert_alarm_block(proto_alarm.alarm_block),
        analog_alarm_text_id=analog_text_id,
        digital_alarm_condition=dig_condition,
        digital_alarm_mask=dig_mask,
        digital_alarm_text_id=dig_text_id,
    )


def _convert_alarm_text(proto_text) -> AlarmText:
    return AlarmText(
        alarm_text_id=proto_text.alarm_text_id,
        length=proto_text.length,
        priority=proto_text.priority,
        hand_code=proto_text.hand_code,
        sound_id=proto_text.sound_id,
        speech_id=proto_text.speech_id,
        spare=proto_text.spare,
        text=proto_text.text,
        url=proto_text.url,
    )


# ─── TTL Cache ────────────────────────────────────────────────────────────────


class _TTLCache:
    """Thread-safe TTL cache with max size eviction."""

    def __init__(self, ttl: float, max_size: int = 10000):
        self._ttl = ttl
        self._max_size = max_size
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()

    def get(self, key: str) -> object | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: object) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (time.monotonic(), value)
            if len(self._data) > self._max_size:
                self._evict_expired()
            if len(self._data) > self._max_size:
                self._evict_oldest()

    def _evict_expired(self) -> None:
        """Remove all expired entries. Caller must hold lock."""
        now = time.monotonic()
        expired = [k for k, (ts, _) in self._data.items() if now - ts > self._ttl]
        for k in expired:
            del self._data[k]

    def _evict_oldest(self) -> None:
        """Remove the oldest entry (front of OrderedDict). Caller must hold lock."""
        self._data.popitem(last=False)

    def clear(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._data.clear()
            else:
                self._data.pop(key, None)


# ─── DevDB Client ────────────────────────────────────────────────────────────


class DevDBClient:
    """Synchronous gRPC client for DevDB device metadata queries.

    Args:
        host: DevDB gRPC server hostname (default: from PACSYS_DEVDB_HOST or localhost)
        port: DevDB gRPC server port (default: from PACSYS_DEVDB_PORT or 6802)
        timeout: RPC timeout in seconds (default: 5.0)
        cache_ttl: TTL for cached results in seconds (default: 3600.0)

    Raises:
        ImportError: If grpc package is not available
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: float | None = None,
        cache_ttl: float = 3600.0,
    ):
        if not DEVDB_AVAILABLE:
            raise ImportError(f"gRPC not available for DevDB: {_import_error}")

        self._host = host or os.environ.get("PACSYS_DEVDB_HOST", "localhost")
        self._port = port or int(os.environ.get("PACSYS_DEVDB_PORT", "6802"))
        self._timeout = timeout or 5.0
        self._cache = _TTLCache(cache_ttl)
        self._closed = False

        target = f"{self._host}:{self._port}"
        self._channel = grpc.insecure_channel(target)
        self._stub = DevDB_pb2_grpc.DevDBStub(self._channel)
        logger.debug("DevDB client connected to %s", target)

    def get_device_info(self, names: list[str], timeout: float | None = None) -> dict[str, DeviceInfoResult]:
        """Query device metadata for one or more devices.

        Args:
            names: Device names (e.g., ["Z:ACLTST", "M:OUTTMP"])
            timeout: gRPC timeout in seconds (default: client's configured timeout).

        Returns:
            Dict mapping device name to DeviceInfoResult.

        Raises:
            DeviceError: If DevDB returns an error for a device.
            grpc.RpcError: On gRPC transport failure.
        """
        self._check_closed()

        # Check cache first, collect uncached names (normalize keys for case-insensitive lookup)
        result: dict[str, DeviceInfoResult] = {}
        uncached: list[str] = []
        for name in names:
            cached = self._cache.get(f"info:{name.upper()}")
            if cached is not None:
                result[name] = cached  # type: ignore[assignment]
            else:
                uncached.append(name)

        if not uncached:
            return result

        request = DevDB_pb2.DeviceList(device=uncached)
        reply = self._stub.getDeviceInfo(request, timeout=timeout or self._timeout)

        for entry in reply.set:
            which = entry.WhichOneof("result")
            if which == "device":
                info = _convert_device_info(entry.device)
                result[entry.name] = info
                self._cache.put(f"info:{entry.name.upper()}", info)
            elif which == "errMsg":
                raise DeviceError(entry.name, 0, -1, entry.errMsg)

        return result

    def get_alarm_info(self, names: list[str]) -> list[AlarmInfo]:
        """Query alarm information for devices.

        Args:
            names: Device names

        Returns:
            List of AlarmInfo entries.

        Raises:
            grpc.RpcError: On gRPC transport failure.
        """
        self._check_closed()
        request = DevDB_pb2.DeviceList(device=names)
        reply = self._stub.getAllAlarmInfo(request, timeout=self._timeout)
        return [_convert_alarm_info(a) for a in reply.alarm_info]

    def get_alarm_text(self, ids: list[int]) -> list[AlarmText]:
        """Query alarm text entries by ID.

        Args:
            ids: Alarm text IDs

        Returns:
            List of AlarmText entries.

        Raises:
            grpc.RpcError: On gRPC transport failure.
        """
        self._check_closed()
        request = DevDB_pb2.AlarmTextIdList(alarm_text_id=ids)
        reply = self._stub.getAlarmText(request, timeout=self._timeout)
        return [_convert_alarm_text(t) for t in reply.device_alarm_text]

    def clear_cache(self, device: str | None = None) -> None:
        """Clear cached results.

        Args:
            device: If given, clear only this device's cache. Otherwise clear all.
        """
        if device is not None:
            self._cache.clear(f"info:{device.upper()}")
        else:
            self._cache.clear()

    def close(self) -> None:
        """Close the gRPC channel."""
        if not self._closed:
            self._closed = True
            self._channel.close()
            logger.debug("DevDB client closed")

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("DevDBClient is closed")

    def __enter__(self) -> DevDBClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"DevDBClient({self._host}:{self._port})"


__all__ = [
    "DevDBClient",
    "DeviceInfoResult",
    "PropertyInfo",
    "StatusBitDef",
    "ExtStatusBitDef",
    "ControlCommandDef",
    "AlarmBlockInfo",
    "AlarmInfo",
    "AlarmText",
    "DEVDB_AVAILABLE",
]
