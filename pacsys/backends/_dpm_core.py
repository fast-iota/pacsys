"""Async DPM core - unified read/write/stream over _AsyncDPMConnection."""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from pacsys.acnet.errors import (
    ERR_OK,
    ERR_RETRY,
    ERR_TIMEOUT,
    FACILITY_ACNET,
    parse_error,
    status_message,
)
from pacsys.auth import KerberosAuth
from pacsys.dpm_connection import DPMConnectionError
from pacsys.drf_utils import ensure_immediate_event
from pacsys.errors import AuthenticationError, ReadError
from pacsys.types import (
    DeviceMeta,
    Reading,
    Value,
    ValueType,
    WriteResult,
)
from pacsys.dpm_protocol import (
    AddToList_reply,
    AddToList_request,
    ApplySettings_reply,
    ApplySettings_request,
    Authenticate_reply,
    Authenticate_request,
    ClearList_request,
    DeviceInfo_reply,
    EnableSettings_request,
    ListStatus_reply,
    RawSetting_struct,
    ScaledSetting_struct,
    StartList_reply,
    StartList_request,
    Status_reply,
    StopList_request,
    TextSetting_struct,
)

# Reuse pure helpers from sync backend
from pacsys.backends.dpm_http import (
    _reply_to_reading,
    _device_info_to_meta,
    _AsyncDPMConnection,
)

logger = logging.getLogger(__name__)


def _value_to_setting(
    ref_id: int,
    value: Value,
) -> tuple[Optional[RawSetting_struct], Optional[ScaledSetting_struct], Optional[TextSetting_struct]]:
    """Convert a value to the appropriate setting struct."""
    raw_setting = None
    scaled_setting = None
    text_setting = None

    if isinstance(value, bytes):
        raw_setting = RawSetting_struct()
        raw_setting.ref_id = ref_id
        raw_setting.data = value
    elif isinstance(value, str):
        text_setting = TextSetting_struct()
        text_setting.ref_id = ref_id
        text_setting.data = [value]
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) > 0 and isinstance(value[0], str):  # type: ignore[arg-type]
            text_setting = TextSetting_struct()
            text_setting.ref_id = ref_id
            text_setting.data = list(value)
        else:
            scaled_setting = ScaledSetting_struct()
            scaled_setting.ref_id = ref_id
            scaled_setting.data = [float(v) for v in value]
    elif isinstance(value, dict):
        raise TypeError("write_many() does not support alarm dicts; use write() instead")
    else:
        scaled_setting = ScaledSetting_struct()
        scaled_setting.ref_id = ref_id
        scaled_setting.data = [float(value)]

    return raw_setting, scaled_setting, text_setting


class _AsyncDpmCore:
    """Unified async core for DPM reads, writes, and streaming.

    One _AsyncDPMConnection per core. Owns connection lifecycle.
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        auth: Optional[KerberosAuth] = None,
        role: Optional[str] = None,
    ):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._auth = auth
        self._role = role
        self._conn: Optional[_AsyncDPMConnection] = None
        self._settings_enabled = False
        self._mic: Optional[bytes] = None
        self._mic_message: Optional[bytes] = None

    async def connect(self) -> None:
        self._conn = _AsyncDPMConnection(self._host, self._port)
        await self._conn.connect()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def list_id(self) -> int:
        assert self._conn is not None
        return self._conn.list_id  # type: ignore[return-value]  # guaranteed after connect()

    # ── Authentication ────────────────────────────────────────────────────

    async def authenticate(self) -> None:
        """Kerberos GSSAPI authentication over the DPM connection."""
        import gssapi

        assert self._conn is not None
        if self._auth is None:
            raise AuthenticationError("KerberosAuth required for authentication")

        # Phase 1: request service name
        auth_req = Authenticate_request()
        auth_req.list_id = self._conn.list_id
        auth_req.token = b""
        await self._conn.send_message(auth_req)

        reply = await asyncio.wait_for(self._conn.recv_message(), timeout=self._timeout)
        if not isinstance(reply, Authenticate_reply):
            raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        raw_service_name = reply.serviceName
        if not raw_service_name:
            raise AuthenticationError("Server did not provide a service name")

        gss_name = raw_service_name.translate({ord("@"): "/", ord("\\"): None}) + "@FNAL.GOV"
        logger.debug(f"DPM service name: {gss_name}")

        # Phase 2: GSSAPI context
        service_name = gssapi.Name(gss_name, gssapi.NameType.kerberos_principal)
        creds = self._auth._get_credentials()
        ctx = gssapi.SecurityContext(
            name=service_name,
            usage="initiate",
            creds=creds,
            flags=[  # type: ignore[arg-type]
                gssapi.RequirementFlag.replay_detection,
                gssapi.RequirementFlag.integrity,
                gssapi.RequirementFlag.out_of_sequence_detection,
            ],
            mech=gssapi.MechType.kerberos,
        )

        token = ctx.step()

        auth_req = Authenticate_request()
        auth_req.list_id = self._conn.list_id
        auth_req.token = bytes(token) if token else b""
        await self._conn.send_message(auth_req)

        reply = await asyncio.wait_for(self._conn.recv_message(), timeout=self._timeout)
        if not isinstance(reply, Authenticate_reply):
            raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        if hasattr(reply, "token") and reply.token and not ctx.complete:
            token = ctx.step(reply.token)
            if token:
                auth_req = Authenticate_request()
                auth_req.list_id = self._conn.list_id
                auth_req.token = bytes(token)
                await self._conn.send_message(auth_req)

                reply = await asyncio.wait_for(self._conn.recv_message(), timeout=self._timeout)
                if not isinstance(reply, Authenticate_reply):
                    raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        if not ctx.complete:
            raise AuthenticationError("Kerberos authentication incomplete")

        message = b"1234"
        mic = ctx.get_signature(message)
        self._mic = bytes(mic)
        self._mic_message = message
        logger.debug(f"Kerberos authentication complete for {self._auth.principal}")

    async def enable_settings(self) -> None:
        """Enable settings on the connection after authentication."""
        assert self._conn is not None
        if self._mic is None:
            raise AuthenticationError("Must authenticate before enabling settings")

        enable_req = EnableSettings_request()
        enable_req.list_id = self._conn.list_id
        enable_req.MIC = self._mic
        enable_req.message = self._mic_message

        await self._conn.send_message(enable_req)

        while True:
            reply = await asyncio.wait_for(self._conn.recv_message(), timeout=self._timeout)
            if isinstance(reply, ListStatus_reply):
                continue
            if isinstance(reply, Status_reply):
                if reply.status != 0:
                    facility, error = parse_error(reply.status)
                    raise AuthenticationError(
                        f"EnableSettings failed: facility={facility}, error={error} (DPM_PRIV = privilege denied)"
                    )
                self._settings_enabled = True
                return
            raise AuthenticationError(f"Expected Status_reply, got {type(reply).__name__}")

    # ── Read ──────────────────────────────────────────────────────────────

    async def read_many(self, drfs: list[str], timeout: float) -> list[Reading]:
        """Read multiple devices in a single batch."""
        assert self._conn is not None
        deadline = time.monotonic() + timeout

        prepared_drfs = [ensure_immediate_event(drf) for drf in drfs]
        list_id = self._conn.list_id

        device_infos: dict[int, DeviceInfo_reply] = {}
        data_replies: dict[int, object] = {}
        add_errors: dict[int, AddToList_reply] = {}
        received_count = 0
        expected_count = len(drfs)
        conn_broken = False
        transport_error: Optional[BaseException] = None

        # Batch AddToList + StartList
        setup_msgs = []
        for i, drf in enumerate(prepared_drfs):
            add_req = AddToList_request()
            add_req.list_id = list_id
            add_req.ref_id = i + 1
            add_req.drf_request = drf
            setup_msgs.append(add_req)

        start_req = StartList_request()
        start_req.list_id = list_id
        setup_msgs.append(start_req)
        await self._conn.send_messages_batch(setup_msgs)

        try:
            while received_count < expected_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    reply = await asyncio.wait_for(self._conn.recv_message(), timeout=min(remaining, 2.0))
                except asyncio.TimeoutError:
                    if time.monotonic() >= deadline:
                        break
                    continue

                if isinstance(reply, AddToList_reply):
                    if reply.status != 0:
                        add_errors[reply.ref_id] = reply
                        received_count += 1
                elif isinstance(reply, DeviceInfo_reply):
                    device_infos[reply.ref_id] = reply
                elif isinstance(reply, StartList_reply):
                    if reply.status != 0:
                        logger.warning(f"StartList returned status {reply.status}")
                        break
                elif isinstance(reply, ListStatus_reply):
                    pass
                elif isinstance(reply, Status_reply):
                    if reply.ref_id not in data_replies:
                        data_replies[reply.ref_id] = reply
                        received_count += 1
                elif hasattr(reply, "ref_id"):
                    if reply.ref_id not in data_replies:
                        data_replies[reply.ref_id] = reply
                        received_count += 1
        except (BrokenPipeError, ConnectionResetError, OSError, asyncio.IncompleteReadError, DPMConnectionError) as e:
            conn_broken = True
            transport_error = e
        finally:
            if not conn_broken:
                try:
                    stop_req = StopList_request()
                    stop_req.list_id = list_id
                    clear_req = ClearList_request()
                    clear_req.list_id = list_id
                    await self._conn.send_messages_batch([stop_req, clear_req])
                except Exception:
                    pass  # connection may be dead

        # Assemble readings
        readings: list[Reading] = []
        has_timeout = False

        for i, original_drf in enumerate(drfs):
            ref_id = i + 1
            info = device_infos.get(ref_id)
            reply = data_replies.get(ref_id)
            add_err = add_errors.get(ref_id)
            meta = _device_info_to_meta(info) if info else None

            if add_err is not None:
                facility, error = parse_error(add_err.status)
                readings.append(
                    Reading(
                        drf=original_drf,
                        value_type=ValueType.SCALAR,
                        facility_code=facility,
                        error_code=error,
                        value=None,
                        message=status_message(facility, error) or f"AddToList failed (status={add_err.status})",
                        timestamp=None,
                        cycle=0,
                        meta=meta,
                    )
                )
            elif reply is None:
                has_timeout = True
                ec = ERR_RETRY if transport_error is not None else ERR_TIMEOUT
                msg = f"Connection error: {transport_error}" if transport_error is not None else "Request timeout"
                readings.append(
                    Reading(
                        drf=original_drf,
                        value_type=ValueType.SCALAR,
                        facility_code=FACILITY_ACNET,
                        error_code=ec,
                        value=None,
                        message=msg,
                        timestamp=None,
                        cycle=0,
                        meta=meta,
                    )
                )
            else:
                readings.append(_reply_to_reading(reply, original_drf, meta))

        if transport_error is not None or has_timeout:
            raise ReadError(readings, str(transport_error or "Request timeout")) from transport_error

        return readings

    # ── Write ─────────────────────────────────────────────────────────────

    async def write_many(
        self,
        settings: list[tuple[str, Value]],
        role: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> list[WriteResult]:
        """Write multiple devices."""
        assert self._conn is not None
        effective_timeout = timeout if timeout is not None else self._timeout
        deadline = time.monotonic() + effective_timeout

        if not self._settings_enabled:
            await self.authenticate()
            await self.enable_settings()

        role = role or self._role
        list_id = self._conn.list_id
        add_errors: dict[int, int] = {}

        # Batch: StopList + ClearList + optional ROLE + AddToList*N + StartList
        setup_msgs: list = []

        stop_req = StopList_request()
        stop_req.list_id = list_id
        setup_msgs.append(stop_req)

        clear_req = ClearList_request()
        clear_req.list_id = list_id
        setup_msgs.append(clear_req)

        if role is not None:
            role_req = AddToList_request()
            role_req.list_id = list_id
            role_req.ref_id = 0
            role_req.drf_request = f"#ROLE:{role}"
            setup_msgs.append(role_req)

        for i, (drf, _) in enumerate(settings):
            add_req = AddToList_request()
            add_req.list_id = list_id
            add_req.ref_id = i + 1
            add_req.drf_request = drf
            setup_msgs.append(add_req)

        start_req = StartList_request()
        start_req.list_id = list_id
        setup_msgs.append(start_req)

        await self._conn.send_messages_batch(setup_msgs)

        # Phase 1: Wait for device infos
        received_infos = 0
        expected_count = len(settings)

        while received_infos < expected_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                reply = await asyncio.wait_for(self._conn.recv_message(), timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    break
                continue

            if isinstance(reply, ListStatus_reply):
                pass
            elif isinstance(reply, AddToList_reply):
                if reply.status != 0 and reply.ref_id > 0:
                    add_errors[reply.ref_id] = reply.status
                    received_infos += 1
            elif isinstance(reply, DeviceInfo_reply):
                received_infos += 1
            elif isinstance(reply, StartList_reply):
                if reply.status != 0:
                    logger.warning(f"StartList returned status {reply.status}")
                    return self._build_write_results(settings, None, add_errors)
            elif isinstance(reply, Status_reply):
                received_infos += 1

        # Phase 2: Build and send ApplySettings
        apply_req = ApplySettings_request()
        apply_req.user_name = self._auth.principal if self._auth else ""
        apply_req.list_id = list_id

        raw_settings = []
        scaled_settings = []
        text_settings = []

        for i, (_, value) in enumerate(settings):
            ref_id = i + 1
            raw, scaled, text = _value_to_setting(ref_id, value)
            if raw:
                raw_settings.append(raw)
            if scaled:
                scaled_settings.append(scaled)
            if text:
                text_settings.append(text)

        if raw_settings:
            apply_req.raw_array = raw_settings  # type: ignore[unresolved-attribute]
        if scaled_settings:
            apply_req.scaled_array = scaled_settings  # type: ignore[unresolved-attribute]
        if text_settings:
            apply_req.text_array = text_settings  # type: ignore[unresolved-attribute]

        await self._conn.send_message(apply_req)

        # Phase 3: Wait for ApplySettings reply
        apply_reply = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                reply = await asyncio.wait_for(self._conn.recv_message(), timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    break
                continue

            if isinstance(reply, ApplySettings_reply):
                apply_reply = reply
                break
            elif isinstance(reply, ListStatus_reply):
                pass

        return self._build_write_results(settings, apply_reply, add_errors)

    def _build_write_results(
        self,
        settings: list[tuple[str, Value]],
        apply_reply: Optional[ApplySettings_reply],
        add_errors: dict[int, int],
    ) -> list[WriteResult]:
        """Convert ApplySettings_reply + add_errors into WriteResult list."""
        # Build ref_id → status map from SettingStatus_struct list
        status_map: dict[int, int] = {}
        if apply_reply is not None:
            for status_struct in apply_reply.status:
                status_map[status_struct.ref_id] = status_struct.status

        results: list[WriteResult] = []
        for i, (drf, _) in enumerate(settings):
            ref_id = i + 1
            if ref_id in add_errors:
                facility, error = parse_error(add_errors[ref_id])
                results.append(
                    WriteResult(
                        drf=drf,
                        facility_code=facility,
                        error_code=error,
                        message=status_message(facility, error) or "AddToList failed",
                    )
                )
            elif ref_id in status_map:
                facility, error = parse_error(status_map[ref_id])
                results.append(
                    WriteResult(
                        drf=drf,
                        facility_code=facility,
                        error_code=error,
                        message=status_message(facility, error) if error != ERR_OK else None,
                    )
                )
            else:
                results.append(
                    WriteResult(
                        drf=drf,
                        facility_code=FACILITY_ACNET,
                        error_code=ERR_TIMEOUT,
                        message="No reply from server",
                    )
                )
        return results

    # ── Stream ────────────────────────────────────────────────────────────

    async def stream(self, drfs: list[str], dispatch_fn, stop_check, error_fn) -> None:
        """Long-running streaming recv loop."""
        assert self._conn is not None
        metas: dict[int, DeviceMeta] = {}
        drf_map: dict[int, str] = {}

        try:
            list_id = self._conn.list_id

            setup_msgs = []
            for i, drf in enumerate(drfs):
                ref_id = i + 1
                drf_map[ref_id] = drf
                add_req = AddToList_request()
                add_req.list_id = list_id
                add_req.ref_id = ref_id
                add_req.drf_request = drf
                setup_msgs.append(add_req)

            start_req = StartList_request()
            start_req.list_id = list_id
            setup_msgs.append(start_req)
            await self._conn.send_messages_batch(setup_msgs)

            while not stop_check():
                reply = await self._conn.recv_message()

                if isinstance(reply, AddToList_reply):
                    if reply.status != 0:
                        drf = drf_map.get(reply.ref_id)
                        if drf is not None:
                            facility, error = parse_error(reply.status)
                            reading = Reading(
                                drf=drf,
                                value_type=ValueType.SCALAR,
                                facility_code=facility,
                                error_code=error,
                                value=None,
                                message=status_message(facility, error) or f"AddToList failed (status={reply.status})",
                                timestamp=None,
                                cycle=0,
                                meta=None,
                            )
                            dispatch_fn(reading)
                    continue

                if isinstance(reply, StartList_reply):
                    if reply.status != 0:
                        logger.warning(f"StartList returned status {reply.status}")
                        error_fn(DPMConnectionError(f"StartList failed (status={reply.status})"))
                        return
                    continue

                if isinstance(reply, ListStatus_reply):
                    continue

                if isinstance(reply, DeviceInfo_reply):
                    metas[reply.ref_id] = _device_info_to_meta(reply)
                    continue

                if hasattr(reply, "ref_id"):
                    ref_id = reply.ref_id
                    drf = drf_map.get(ref_id)
                    if drf is None:
                        logger.warning(f"Data for unknown ref_id={ref_id}")
                        continue
                    meta = metas.get(ref_id)
                    reading = _reply_to_reading(reply, drf, meta)
                    dispatch_fn(reading)

        except asyncio.CancelledError:
            pass
        except (asyncio.IncompleteReadError, DPMConnectionError, OSError) as e:
            if not stop_check():
                error_fn(e)
        except Exception as e:
            if not stop_check():
                logger.error(f"Unexpected streaming error: {e}")
                error_fn(e)
