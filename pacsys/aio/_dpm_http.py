"""Async DPM HTTP backend - connection pooling over _AsyncDpmCore."""

import asyncio
import logging

from pacsys.aio._backends import AsyncBackend
from pacsys.aio._subscription import AsyncSubscriptionHandle, _callback_feeder
from pacsys.auth import KerberosAuth
from pacsys.backends._dpm_core import _AsyncDpmCore
from pacsys.backends.dpm_http import _value_to_setting
from pacsys.drf_utils import prepare_for_write
from pacsys.errors import AuthenticationError, DeviceError
from pacsys.types import (
    BackendCapability,
    ErrorCallback,
    Reading,
    ReadingCallback,
    Value,
    WriteResult,
)

logger = logging.getLogger(__name__)

# Alarm field maps — must match sync DPMHTTPBackend exactly
_ANALOG_ALARM_FIELDS = {
    "minimum": "MIN",
    "maximum": "MAX",
    "alarm_enable": "ALARM_ENABLE",
    "abort_inhibit": "ABORT_INHIBIT",
    "tries_needed": "TRIES_NEEDED",
}

_DIGITAL_ALARM_FIELDS = {
    "mask": "MASK",
    "nominal": "NOM",
    "alarm_enable": "ALARM_ENABLE",
    "abort_inhibit": "ABORT_INHIBIT",
    "tries_needed": "TRIES_NEEDED",
}


def _expand_alarm_dict(drf: str, alarm_dict: dict) -> list[tuple[str, Value]]:
    """Expand an alarm dict into per-field (drf.FIELD, value) pairs."""
    from pacsys.drf3 import parse_request
    from pacsys.drf3.property import DRF_PROPERTY
    from pacsys.drf_utils import get_device_name

    prop = parse_request(drf).property
    if prop == DRF_PROPERTY.ANALOG:
        field_map = _ANALOG_ALARM_FIELDS
        prop_name = "ANALOG"
    elif prop == DRF_PROPERTY.DIGITAL:
        field_map = _DIGITAL_ALARM_FIELDS
        prop_name = "DIGITAL"
    elif prop in (DRF_PROPERTY.STATUS, DRF_PROPERTY.CONTROL):
        raise ValueError(
            f"Cannot write a dict to {prop.name} property. "
            f'Use BasicControl enum values instead: backend.write("{drf}", BasicControl.ON)'
        )
    else:
        raise ValueError(f"Cannot write dict to {prop.name} property (DRF: {drf})")

    writable = set(field_map) | {"abort", "alarm_status", "tries_now"}
    bad_keys = set(alarm_dict) - writable
    if bad_keys:
        raise ValueError(f"Unknown {prop_name} alarm keys: {bad_keys}")

    base = get_device_name(drf)
    pairs: list[tuple[str, Value]] = []
    for key, field_name in field_map.items():
        if key not in alarm_dict:
            continue
        val = alarm_dict[key]
        if isinstance(val, bool):
            val = 1 if val else 0
        pairs.append((f"{base}.{prop_name}.{field_name}", val))
    return pairs


class AsyncDPMHTTPBackend(AsyncBackend):
    """Async DPM HTTP backend with connection pooling.

    Uses asyncio.Queue of _AsyncDpmCore instances for reads.
    Creates dedicated cores for writes (authenticated) and streaming.
    """

    def __init__(
        self,
        host: str = "acsys-proxy.fnal.gov",
        port: int = 6802,
        pool_size: int = 4,
        timeout: float = 5.0,
        auth: KerberosAuth | None = None,
        role: str | None = None,
    ):
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        self._host = host
        self._port = port
        self._pool_size = pool_size
        self._timeout = timeout
        self._auth = auth
        self._role = role
        self._closed = False
        self._pool: asyncio.Queue[_AsyncDpmCore] = asyncio.Queue(maxsize=pool_size)
        self._pool_count = 0
        self._pool_lock = asyncio.Lock()
        self._handles: list[AsyncSubscriptionHandle] = []

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("Backend is closed")

    async def _create_core(self) -> _AsyncDpmCore:
        """Create and connect a new core."""
        core = _AsyncDpmCore(
            host=self._host,
            port=self._port,
            timeout=self._timeout,
            auth=self._auth,
            role=self._role,
        )
        await core.connect()
        return core

    async def _borrow_core(self) -> _AsyncDpmCore:
        """Borrow a core from the read pool, creating if needed."""
        self._check_closed()
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            pass
        async with self._pool_lock:
            if self._pool_count < self._pool_size:
                core = await self._create_core()
                self._pool_count += 1
                return core
        # Pool is full — wait with timeout to avoid permanent hangs
        try:
            return await asyncio.wait_for(self._pool.get(), timeout=self._timeout)
        except asyncio.TimeoutError:
            raise RuntimeError("Connection pool exhausted (all cores busy)")

    async def _release_core(self, core: _AsyncDpmCore) -> None:
        """Return a core to the pool."""
        try:
            self._pool.put_nowait(core)
        except asyncio.QueueFull:
            await core.close()

    async def _discard_core(self, core: _AsyncDpmCore) -> None:
        """Close and discard a core (on error), freeing a pool slot."""
        try:
            await core.close()
        except Exception:
            pass
        async with self._pool_lock:
            self._pool_count = max(0, self._pool_count - 1)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def capabilities(self) -> BackendCapability:
        caps = BackendCapability.READ | BackendCapability.STREAM | BackendCapability.BATCH
        if isinstance(self._auth, KerberosAuth):
            caps |= BackendCapability.WRITE | BackendCapability.AUTH_KERBEROS
        return caps

    @property
    def authenticated(self) -> bool:
        return self._auth is not None

    @property
    def principal(self) -> str | None:
        return self._auth.principal if self._auth else None

    # ── Read ──────────────────────────────────────────────────────────────

    async def read(self, drf: str, timeout: float | None = None) -> Value:
        reading = await self.get(drf, timeout=timeout)
        if not reading.ok:
            raise DeviceError(
                drf=reading.drf,
                facility_code=reading.facility_code,
                error_code=reading.error_code,
                message=reading.message or "Read failed",
            )
        assert reading.value is not None
        return reading.value

    async def get(self, drf: str, timeout: float | None = None) -> Reading:
        readings = await self.get_many([drf], timeout=timeout)
        return readings[0]

    async def get_many(self, drfs: list[str], timeout: float | None = None) -> list[Reading]:
        if not drfs:
            return []
        effective_timeout = timeout if timeout is not None else self._timeout
        core = await self._borrow_core()
        try:
            result = await core.read_many(drfs, effective_timeout)
            # read_many closes its connection on repeating-event DRFs or failures —
            # such cores must be discarded, not re-pooled.
            if core.connected:
                await self._release_core(core)
            else:
                await self._discard_core(core)
            return result
        except BaseException:
            await self._discard_core(core)
            raise

    # ── Write ─────────────────────────────────────────────────────────────

    async def write(self, drf: str, value: Value, timeout: float | None = None) -> WriteResult:
        # Alarm dict expansion (same pattern as sync DPMHTTPBackend)
        if isinstance(value, dict):
            from pacsys.acnet.errors import ERR_OK

            pairs = _expand_alarm_dict(drf, value)
            if not pairs:
                return WriteResult(drf=drf, error_code=ERR_OK)
            for field_drf, field_val in pairs:
                results = await self.write_many([(field_drf, field_val)], timeout=timeout)
                if not results[0].success:
                    r = results[0]
                    return WriteResult(
                        drf=drf,
                        facility_code=r.facility_code,
                        error_code=r.error_code,
                        message=r.message,
                    )
            return WriteResult(drf=drf, error_code=ERR_OK)

        results = await self.write_many([(drf, value)], timeout=timeout)
        return results[0]

    async def write_many(
        self,
        settings: list[tuple[str, Value]],
        timeout: float | None = None,
    ) -> list[WriteResult]:
        if not settings:
            return []
        self._check_closed()
        if not isinstance(self._auth, KerberosAuth):
            raise AuthenticationError("Backend not configured for authenticated operations. Pass auth=KerberosAuth().")
        effective_timeout = timeout if timeout is not None else self._timeout
        prepared = [(prepare_for_write(drf), value) for drf, value in settings]
        setting_payloads = [_value_to_setting(i, value) for i, (_, value) in enumerate(settings, 1)]

        # Dedicated core for writes (fresh authenticated connection)
        core = await self._create_core()
        try:
            result = await core.write_many(
                prepared,
                timeout=effective_timeout,
                setting_payloads=setting_payloads,
            )
            await core.close()
            return result
        except BaseException:
            try:
                await core.close()
            except Exception:
                pass
            raise

    # ── Streaming ─────────────────────────────────────────────────────────

    async def subscribe(
        self,
        drfs: list[str],
        callback: ReadingCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> AsyncSubscriptionHandle:
        self._check_closed()
        if not drfs:
            raise ValueError("drfs cannot be empty")
        core = await self._create_core()
        handle = AsyncSubscriptionHandle(remover=self.remove)
        handle._drfs = drfs
        handle._task = asyncio.ensure_future(
            core.stream(drfs, handle._dispatch, handle._is_stopped, handle._signal_error)
        )
        if callback:
            handle._callback_task = asyncio.ensure_future(_callback_feeder(handle, callback, on_error))
        handle._core = core
        self._handles.append(handle)
        return handle

    async def remove(self, handle) -> None:
        if isinstance(handle, AsyncSubscriptionHandle):
            await handle.stop()
            if handle._core is not None:
                try:
                    await handle._core.close()
                except Exception:
                    pass
            if handle in self._handles:
                self._handles.remove(handle)

    async def stop_streaming(self) -> None:
        for h in list(self._handles):
            await h.stop()
            if hasattr(h, "_core") and hasattr(h._core, "close"):
                try:
                    await h._core.close()
                except Exception:
                    pass
        self._handles.clear()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.stop_streaming()
        while not self._pool.empty():
            try:
                core = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                if core is not None:
                    await core.close()
            except Exception:
                pass
        self._pool_count = 0
