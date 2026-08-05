"""Async DPM core - unified read/write/stream over _AsyncDPMConnection."""

import asyncio
import logging
import time

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

# Reuse pure helpers from sync backend
from pacsys.backends.dpm_http import (
    _aggregate_logger_chunks,
    _AsyncDPMConnection,
    _device_info_to_meta,
    _is_logger_drf,
    _reply_to_reading,
    _SettingPayload,
    _value_to_setting,
)
from pacsys.dpm_connection import DPMConnectionError
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
    ScalarArray_reply,
    StartList_reply,
    StartList_request,
    Status_reply,
    StopList_request,
    TimedScalarArray_reply,
)
from pacsys.drf_utils import ensure_immediate_event, is_immediate_only
from pacsys.errors import AuthenticationError, ReadError
from pacsys.types import (
    DeviceMeta,
    Reading,
    Value,
    ValueType,
    WriteResult,
)

logger = logging.getLogger(__name__)


class _AsyncDpmCore:
    """Unified async core for DPM reads, writes, and streaming.

    One _AsyncDPMConnection per core. Owns connection lifecycle.
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        auth: KerberosAuth | None = None,
        role: str | None = None,
    ):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._auth = auth
        self._role = role
        self._conn: _AsyncDPMConnection | None = None
        self._settings_enabled = False
        self._mic: bytes | None = None
        self._mic_message: bytes | None = None

    async def connect(self) -> None:
        conn = _AsyncDPMConnection(self._host, self._port)
        await conn.connect()
        self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    @property
    def list_id(self) -> int:
        if self._conn is None:
            raise DPMConnectionError("DPM core is not connected")
        return self._conn.list_id

    # ── Authentication ────────────────────────────────────────────────────

    async def authenticate(self) -> None:
        """Kerberos GSSAPI authentication over the DPM connection."""
        try:
            import gssapi
            from gssapi import exceptions as gssapi_exceptions
        except ImportError:
            raise ImportError(
                "gssapi library required for Kerberos authentication. Install with: pip install pacsys[kerberos]"
            ) from None

        assert self._conn is not None
        if self._auth is None:
            raise AuthenticationError("KerberosAuth required for authentication")

        # Phase 1: request service name
        auth_req = Authenticate_request()
        auth_req.list_id = self.list_id
        auth_req.token = b""
        await self._conn.send_message(auth_req)

        reply = await self._conn.recv_message(timeout=self._timeout)
        if not isinstance(reply, Authenticate_reply):
            raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        raw_service_name = reply.serviceName
        if not raw_service_name:
            raise AuthenticationError("Server did not provide a service name")

        gss_name = raw_service_name.translate({ord("@"): "/", ord("\\"): None}) + "@FNAL.GOV"
        logger.debug("DPM service name: %s", gss_name)

        # Phase 2: GSSAPI context
        try:
            service_name = gssapi.Name(gss_name, gssapi.NameType.kerberos_principal)
            creds = self._auth._get_credentials()
            ctx = gssapi.SecurityContext(
                name=service_name,
                usage="initiate",
                creds=creds,
                flags=(
                    gssapi.RequirementFlag.replay_detection
                    | gssapi.RequirementFlag.integrity
                    | gssapi.RequirementFlag.out_of_sequence_detection
                ),
                mech=gssapi.MechType.kerberos,
            )

            token = ctx.step()
        except gssapi_exceptions.GSSError as e:
            raise AuthenticationError(f"Kerberos authentication failed for {gss_name}: {e}") from e

        auth_req = Authenticate_request()
        auth_req.list_id = self.list_id
        auth_req.token = bytes(token) if token else b""
        await self._conn.send_message(auth_req)

        reply = await self._conn.recv_message(timeout=self._timeout)
        if not isinstance(reply, Authenticate_reply):
            raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        if hasattr(reply, "token") and reply.token and not ctx.complete:
            try:
                token = ctx.step(reply.token)
            except gssapi_exceptions.GSSError as e:
                raise AuthenticationError(f"Kerberos authentication failed for {gss_name}: {e}") from e
            if token:
                auth_req = Authenticate_request()
                auth_req.list_id = self.list_id
                auth_req.token = bytes(token)
                await self._conn.send_message(auth_req)

                reply = await self._conn.recv_message(timeout=self._timeout)
                if not isinstance(reply, Authenticate_reply):
                    raise AuthenticationError(f"Expected Authenticate_reply, got {type(reply).__name__}")

        if not ctx.complete:
            raise AuthenticationError("Kerberos authentication incomplete")

        message = b"1234"
        try:
            mic = ctx.get_signature(message)
        except gssapi_exceptions.GSSError as e:
            raise AuthenticationError(f"Kerberos authentication failed for {gss_name}: {e}") from e
        self._mic = bytes(mic)
        self._mic_message = message
        logger.debug("Kerberos authentication complete for %s", self._auth.principal)

    async def enable_settings(self) -> None:
        """Enable settings on the connection after authentication."""
        assert self._conn is not None
        if self._mic is None or self._mic_message is None:
            raise AuthenticationError("Must authenticate before enabling settings")

        enable_req = EnableSettings_request()
        enable_req.list_id = self.list_id
        enable_req.MIC = self._mic
        enable_req.message = self._mic_message

        await self._conn.send_message(enable_req)

        while True:
            reply = await self._conn.recv_message(timeout=self._timeout)
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
        list_id = self.list_id

        # Logger DRFs arrive in 487-point chunks with a final empty chunk.
        logger_refs: set[int] = set()
        for i, drf in enumerate(prepared_drfs):
            if _is_logger_drf(drf):
                logger_refs.add(i + 1)

        device_infos: dict[int, DeviceInfo_reply] = {}
        data_replies: dict[int, object] = {}
        logger_chunks: dict[int, list] = {}
        logger_complete: set[int] = set()
        add_errors: dict[int, AddToList_reply] = {}
        received_count = 0
        expected_count = len(drfs)
        job_error: int | None = None  # ref-0 Status_reply = job start failure
        conn_broken = False
        transport_error: BaseException | None = None

        # Repeating events (@p/@e/...) keep producing replies after the first —
        # a core that carried one must be closed, not re-pooled, or stale replies
        # get attributed to the next borrower's refs.
        reuse_safe = all((i + 1) in logger_refs or is_immediate_only(d) for i, d in enumerate(prepared_drfs))

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
                    reply = await self._conn.recv_message(timeout=min(remaining, 2.0))
                except asyncio.TimeoutError:
                    if time.monotonic() >= deadline:
                        break
                    continue

                if isinstance(reply, AddToList_reply):
                    if reply.status != 0 and 1 <= reply.ref_id <= expected_count and reply.ref_id not in add_errors:
                        add_errors[reply.ref_id] = reply
                        received_count += 1
                elif isinstance(reply, DeviceInfo_reply):
                    if 1 <= reply.ref_id <= expected_count:
                        device_infos[reply.ref_id] = reply
                elif isinstance(reply, StartList_reply):
                    if reply.status != 0:
                        logger.warning(
                            "StartList returned status %d (devices: %s)",
                            reply.status,
                            ", ".join(drfs[:5]) + (f" and {len(drfs) - 5} more" if len(drfs) > 5 else ""),
                        )
                        break
                elif isinstance(reply, ListStatus_reply):
                    pass
                elif isinstance(reply, Status_reply):
                    ref_id = reply.ref_id
                    if ref_id == 0:
                        # On the TCP transport StartList_reply.status is hardwired OK;
                        # a ref-0 Status_reply is the real job-start-failure signal.
                        if reply.status != 0 and job_error is None:
                            job_error = reply.status
                    elif ref_id in logger_refs:
                        # Error for a logger DRF — record as an error chunk
                        if ref_id not in logger_complete:
                            logger_chunks.setdefault(ref_id, []).append(reply)
                            logger_complete.add(ref_id)
                            received_count += 1
                    elif 1 <= ref_id <= expected_count and ref_id not in data_replies:
                        data_replies[ref_id] = reply
                        received_count += 1
                elif hasattr(reply, "ref_id"):
                    ref_id = reply.ref_id
                    if not (1 <= ref_id <= expected_count):
                        pass  # stale/unknown ref — never count toward expected_count
                    elif ref_id in logger_refs:
                        is_empty = (
                            isinstance(reply, (TimedScalarArray_reply, ScalarArray_reply)) and len(reply.data) == 0
                        )
                        if is_empty:
                            if ref_id not in logger_complete:
                                if hasattr(reply, "status") and reply.status != 0:
                                    # Error terminator — accumulate so _aggregate_logger_chunks surfaces the error
                                    logger_chunks.setdefault(ref_id, []).append(reply)
                                logger_complete.add(ref_id)
                                received_count += 1
                        else:
                            logger_chunks.setdefault(ref_id, []).append(reply)
                    elif ref_id not in data_replies:
                        data_replies[ref_id] = reply
                        received_count += 1
        except (BrokenPipeError, ConnectionResetError, OSError, asyncio.IncompleteReadError, DPMConnectionError) as e:
            conn_broken = True
            transport_error = e
        finally:
            if not conn_broken:
                if job_error is not None or received_count < expected_count:
                    await self.close()
                else:
                    try:
                        stop_req = StopList_request()
                        stop_req.list_id = list_id
                        clear_req = ClearList_request()
                        clear_req.list_id = list_id
                        await self._conn.send_messages_batch([stop_req, clear_req])
                    except OSError:
                        # Match sync: a failed StopList send means unknown connection
                        # state — close so the core is not re-pooled dirty.
                        await self.close()
                    except Exception:
                        await self.close()
                        raise
                    else:
                        if not reuse_safe:
                            await self.close()
            else:
                await self.close()

        # Assemble readings
        readings: list[Reading] = []
        has_timeout = False

        for i, original_drf in enumerate(drfs):
            ref_id = i + 1
            info = device_infos.get(ref_id)
            reply = data_replies.get(ref_id)
            chunks = logger_chunks.get(ref_id)
            add_err = add_errors.get(ref_id)
            meta = _device_info_to_meta(info) if info else None

            if add_err is not None:
                facility, error = parse_error(add_err.status)
                readings.append(
                    Reading(
                        drf=original_drf,
                        facility_code=facility,
                        error_code=error,
                        value=None,
                        message=status_message(facility, error) or f"AddToList failed (status={add_err.status})",
                        timestamp=None,
                        cycle=0,
                        meta=meta,
                    )
                )
            elif ref_id in logger_refs:
                if ref_id in logger_complete and chunks:
                    readings.append(_aggregate_logger_chunks(chunks, original_drf, meta))
                elif ref_id in logger_complete:
                    readings.append(
                        Reading(
                            drf=original_drf,
                            value_type=ValueType.TIMED_SCALAR_ARRAY,
                            value={"data": np.array([], dtype=float), "micros": np.array([], dtype=np.int64)},
                            timestamp=None,
                            meta=meta,
                        )
                    )
                else:
                    has_timeout = True
                    if job_error is not None:
                        fc, ec = parse_error(job_error)
                        msg = status_message(fc, ec) or f"DPM job start failed (status={job_error})"
                    else:
                        fc = FACILITY_ACNET
                        ec = ERR_RETRY if transport_error is not None else ERR_TIMEOUT
                        msg = (
                            f"Connection error: {transport_error}"
                            if transport_error is not None
                            else "Logger response incomplete"
                        )
                    readings.append(
                        Reading(
                            drf=original_drf,
                            facility_code=fc,
                            error_code=ec,
                            value=None,
                            message=msg,
                            timestamp=None,
                            cycle=0,
                            meta=meta,
                        )
                    )
            elif reply is None:
                has_timeout = True
                if job_error is not None:
                    fc, ec = parse_error(job_error)
                    msg = status_message(fc, ec) or f"DPM job start failed (status={job_error})"
                else:
                    fc = FACILITY_ACNET
                    ec = ERR_RETRY if transport_error is not None else ERR_TIMEOUT
                    msg = f"Connection error: {transport_error}" if transport_error is not None else "Request timeout"
                readings.append(
                    Reading(
                        drf=original_drf,
                        facility_code=fc,
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
            if job_error is not None:
                fc, ec = parse_error(job_error)
                raise ReadError(readings, f"DPM job start failed: {status_message(fc, ec) or job_error}")
            raise ReadError(readings, str(transport_error or "Request timeout")) from transport_error

        return readings

    # ── Write ─────────────────────────────────────────────────────────────

    async def write_many(
        self,
        settings: list[tuple[str, Value]],
        role: str | None = None,
        timeout: float | None = None,
        setting_payloads: list[_SettingPayload] | None = None,
    ) -> list[WriteResult]:
        """Write multiple devices."""
        assert self._conn is not None
        effective_timeout = timeout if timeout is not None else self._timeout
        deadline = time.monotonic() + effective_timeout

        if setting_payloads is None:
            setting_payloads = [_value_to_setting(i, value) for i, (_, value) in enumerate(settings, 1)]

        if not self._settings_enabled:
            await self.authenticate()
            await self.enable_settings()

        role = role or self._role
        list_id = self.list_id
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
        received_start_list_reply = False
        seen_refs: set[int] = set()  # count each ref at most once

        while received_infos < expected_count or not received_start_list_reply:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                reply = await self._conn.recv_message(timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    break
                continue

            if isinstance(reply, ListStatus_reply):
                pass
            elif isinstance(reply, AddToList_reply):
                if reply.status != 0 and 1 <= reply.ref_id <= expected_count and reply.ref_id not in seen_refs:
                    add_errors[reply.ref_id] = reply.status
                    seen_refs.add(reply.ref_id)
                    received_infos += 1
            elif isinstance(reply, DeviceInfo_reply):
                if 1 <= reply.ref_id <= expected_count and reply.ref_id not in seen_refs:
                    seen_refs.add(reply.ref_id)
                    received_infos += 1
            elif isinstance(reply, StartList_reply):
                received_start_list_reply = True
                if reply.status != 0:
                    write_drfs = [drf for drf, _ in settings]
                    logger.warning(
                        "StartList returned status %d (devices: %s)",
                        reply.status,
                        ", ".join(write_drfs[:5]) + (f" and {len(write_drfs) - 5} more" if len(write_drfs) > 5 else ""),
                    )
                    return self._build_write_results(settings, None, add_errors)
            elif isinstance(reply, Status_reply):
                ref_id = reply.ref_id
                if ref_id == 0:
                    # Job-start failure: StartList_reply.status is hardwired OK on the
                    # TCP transport — this is the real signal. Surface via add_errors[0].
                    if reply.status != 0:
                        add_errors[0] = reply.status
                        return self._build_write_results(settings, None, add_errors)
                elif 1 <= ref_id <= expected_count and ref_id not in seen_refs:
                    if reply.status != 0:
                        add_errors[ref_id] = reply.status
                    seen_refs.add(ref_id)
                    received_infos += 1

        if received_infos < expected_count or not received_start_list_reply:
            write_drfs = [drf for drf, _ in settings]
            drf_summary = ", ".join(write_drfs[:5]) + (
                f" and {len(write_drfs) - 5} more" if len(write_drfs) > 5 else ""
            )
            logger.warning(
                "Write setup timed out: received %d/%d device infos, StartList_reply=%s (devices: %s)",
                received_infos,
                expected_count,
                received_start_list_reply,
                drf_summary,
            )
            return self._build_write_results(settings, None, add_errors)

        # Phase 2: Build and send ApplySettings
        apply_req = ApplySettings_request()
        apply_req.user_name = self._auth.principal if self._auth else ""
        apply_req.list_id = list_id

        raw_settings = []
        scaled_settings = []
        text_settings = []

        for raw, scaled, text in setting_payloads:
            if raw:
                raw_settings.append(raw)
            if scaled:
                scaled_settings.append(scaled)
            if text:
                text_settings.append(text)

        if raw_settings:
            setattr(apply_req, "raw_array", raw_settings)
        if scaled_settings:
            setattr(apply_req, "scaled_array", scaled_settings)
        if text_settings:
            setattr(apply_req, "text_array", text_settings)

        await self._conn.send_message(apply_req)

        # Phase 3: Wait for ApplySettings reply
        apply_reply = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                reply = await self._conn.recv_message(timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    break
                continue

            if isinstance(reply, ApplySettings_reply):
                apply_reply = reply
                break
            if isinstance(reply, ListStatus_reply):
                pass

        return self._build_write_results(settings, apply_reply, add_errors)

    def _build_write_results(
        self,
        settings: list[tuple[str, Value]],
        apply_reply: ApplySettings_reply | None,
        add_errors: dict[int, int],
    ) -> list[WriteResult]:
        """Convert ApplySettings_reply + add_errors into WriteResult list."""
        # Build ref_id → status map from SettingStatus_struct list
        status_map: dict[int, int] = {}
        if apply_reply is not None:
            for status_struct in apply_reply.status:
                status_map[status_struct.ref_id] = status_struct.status

        global_err = status_map.get(0)
        job_err = add_errors.get(0)  # ref-0 setup status = job start failure

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
            elif global_err is not None and global_err != 0:
                facility, error = parse_error(global_err)
                results.append(
                    WriteResult(
                        drf=drf,
                        facility_code=facility,
                        error_code=error,
                        message=status_message(facility, error) or f"Global error {global_err}",
                    )
                )
            elif job_err is not None:
                facility, error = parse_error(job_err)
                results.append(
                    WriteResult(
                        drf=drf,
                        facility_code=facility,
                        error_code=error,
                        message=status_message(facility, error) or f"DPM job start failed (status={job_err})",
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
            list_id = self.list_id

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
                        drf_summary = ", ".join(drfs[:5]) + (f" and {len(drfs) - 5} more" if len(drfs) > 5 else "")
                        logger.warning("StartList returned status %d (devices: %s)", reply.status, drf_summary)
                        error_fn(
                            DPMConnectionError(f"StartList failed (status={reply.status}, devices: {drf_summary})")
                        )
                        return
                    continue

                if isinstance(reply, ListStatus_reply):
                    continue

                if isinstance(reply, DeviceInfo_reply):
                    metas[reply.ref_id] = _device_info_to_meta(reply)
                    continue

                if isinstance(reply, Status_reply) and reply.ref_id == 0:
                    if reply.status != 0:
                        facility, error = parse_error(reply.status)
                        message = status_message(facility, error) or f"status={reply.status}"
                        error_fn(DPMConnectionError(f"DPM job start failed: {message}"))
                        return
                    continue

                if hasattr(reply, "ref_id"):
                    ref_id = reply.ref_id
                    drf = drf_map.get(ref_id)
                    if drf is None:
                        logger.warning("Data for unknown ref_id=%s", ref_id)
                        continue
                    meta = metas.get(ref_id)
                    reading = _reply_to_reading(reply, drf, meta)
                    dispatch_fn(reading)

        except asyncio.CancelledError:
            pass
        except (asyncio.IncompleteReadError, DPMConnectionError, OSError) as e:
            if not stop_check():
                drf_summary = ", ".join(drfs) if len(drfs) <= 5 else f"{', '.join(drfs[:5])} and {len(drfs) - 5} more"
                wrapped = DPMConnectionError(f"{e} (devices: {drf_summary})")
                wrapped.__cause__ = e
                error_fn(wrapped)
        except Exception as e:  # noqa: BLE001
            if not stop_check():
                drf_summary = ", ".join(drfs) if len(drfs) <= 5 else f"{', '.join(drfs[:5])} and {len(drfs) - 5} more"
                logger.error("Unexpected streaming error: %s (devices: %s)", e, drf_summary)
                error_fn(e)
