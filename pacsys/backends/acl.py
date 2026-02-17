"""
ACL Backend implementation.

Provides read-only access to ACNET devices via the ACL CGI endpoint.
This is the simplest backend - just HTTP GET requests.

Usage:
    from pacsys.backends.acl import ACLBackend

    with ACLBackend() as backend:
        temp = backend.read("M:OUTTMP")
        reading = backend.get("M:OUTTMP")
        readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
"""

import logging
import os
import re
import urllib.parse
from datetime import datetime
from typing import Optional

import httpx

from pacsys.acnet.errors import ERR_OK, ERR_RETRY, ERR_TIMEOUT
from pacsys.backends import Backend
from pacsys.drf3 import parse_request
from pacsys.drf3.event import DefaultEvent
from pacsys.drf3.field import DRF_FIELD
from pacsys.drf3.property import DRF_PROPERTY
from pacsys.errors import DeviceError, ReadError
from pacsys.types import (
    BackendCapability,
    Reading,
    Value,
    ValueType,
)

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_BASE_URL = "https://www-bd.fnal.gov/cgi-bin/acl.pl"
DEFAULT_TIMEOUT = 5.0

# Escaped semicolon for separating ACL commands in CGI URLs.
# ACL Usage ref: "semicolons used to separate ACL commands should also be
# escaped by a backslash"
_ACL_CMD_SEP = "\\;"

# ACL error codes: DIO_NO_SUCH, CLIB_SYNTAX, DIO_NOSCALE, etc.
_ACL_ERROR_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")

# Properties whose qualifier char (:) ACL understands natively.
# All other properties (_, |, &, @, $, ~, ^, #, !) must be canonicalized
# to explicit property names (e.g. M_OUTTMP → M:OUTTMP.SETTING).
_ACL_NATIVE_PROPERTIES = frozenset({DRF_PROPERTY.READING})

# ── Basic status via per-field reads ──────────────────────────────────
#
# ACNET basic status and digital status share the same raw status integer
# but interpret it through different lenses:
#
#   Basic status  – 5 semantic attributes (on/ready/remote/positive/ramp),
#                   each with a per-device bit mask stored in the ACNET
#                   database (accdb.basic_status_scaling).  The masks are
#                   NOT fixed across devices (e.g. "ramp" is often 0x200
#                   but this is device-dependent).
#
#   Digital status – per-bit descriptions from a separate DB table
#                    (accdb.digital_status), giving device-specific names
#                    like "Shutter Status" or "Beam Switch".
#
# Because the masks are device-specific, we cannot parse .STATUS.RAW
# ourselves.  Instead we ask ACL for each boolean field individually
# (e.g. "read N:LGXS.STATUS.ON") and let the server apply the correct
# DB scaling.  Fields the device lacks (DIO_NOATT) are omitted from the
# resulting dict, matching DPM's behavior.
#
_BASIC_STATUS_FIELDS = ("ON", "READY", "REMOTE", "POSITIVE", "RAMP")
_BASIC_STATUS_KEYS = ("on", "ready", "remote", "positive", "ramp")


def _is_basic_status_request(drf: str) -> bool:
    """Check if DRF requests basic status (STATUS property, default field)."""
    try:
        req = parse_request(drf)
        return req.property == DRF_PROPERTY.STATUS and req.field in (None, DRF_FIELD.ALL)
    except (ValueError, TypeError):
        return False


def _is_raw_field(drf: str) -> bool:
    """Check if a DRF string requests the RAW field."""
    try:
        return parse_request(drf).field == DRF_FIELD.RAW
    except (ValueError, TypeError):
        return False


def _acl_read_command(drf: str) -> tuple[str, str, str]:
    """Build ACL read command, cleaned DRF, and device-level qualifiers.

    ACL syntax is ``read+DEVICE/qualifier1/qualifier2`` - all qualifiers
    go **after** the device name.  Placing ``/ftd=`` or ``/event=`` before
    the device produces ``CLIB_SYNTAX``.

    Maps DRF components to ACL qualifiers:

    - ``.RAW`` field → ``/raw`` device qualifier (hex output)
    - ``@e,XX`` event → ``/event='e,XX'`` device qualifier + ``/pendwait`` command qualifier
    - ``@I`` / default → stripped (ACL always reads immediately)
    - ``@p,NNN`` → raises DeviceError (streaming not supported)

    Note: ``/pendwait`` is a **command** qualifier (before device), while
    ``/raw`` and ``/event=`` are **device** qualifiers (after device).
    Without ``/pendwait``, clock event reads return cached data instead of
    waiting for the event to fire.

    Returns:
        ``(command, cleaned_drf, qualifiers)``
        e.g. ``("read/pendwait", "M:OUTTMP.RAW", "/raw/event='e,02'")``
    """
    req = parse_request(drf)
    command = "read"
    qualifiers = ""

    if req.field == DRF_FIELD.RAW:
        qualifiers += "/raw"

    event = req.event
    if event is not None and event.mode in ("P", "Q"):
        raise DeviceError(
            drf=drf,
            facility_code=0,
            error_code=ERR_RETRY,
            message=(
                f"ACL does not support periodic/streaming events ({drf}). Use a streaming backend (DPM, gRPC, DMQ)."
            ),
        )

    if event is not None and event.mode == "E":
        command = "read/pendwait"
        qualifiers += f"/event='{event.raw_string}'"

    # Canonicalize only when needed: expand qualifier chars (~, |, @, $)
    # that ACL doesn't understand, or strip non-default events.
    has_event_to_strip = event is not None and event.mode not in ("U",)
    has_exotic_qualifier = req.property not in _ACL_NATIVE_PROPERTIES
    if has_event_to_strip or has_exotic_qualifier:
        clean_drf = req.to_canonical(event=DefaultEvent())
    else:
        clean_drf = drf

    return command, clean_drf, qualifiers


def _parse_raw_hex(text: str) -> bytes:
    """Parse ACL ``/raw`` hex output into bytes.

    ACL ``/raw`` qualifier returns raw data in hex, e.g.::

        M:OUTTMP = 0x42900000
        M:OUTTMP = 0x4290 0x0000
        M:OUTTMP = 42 90 00 00
    """
    text = text.strip()
    if "=" in text:
        _, _, text = text.partition("=")
        text = text.strip()

    if not text:
        return b""

    # Collect hex digits from tokens, stopping at first non-hex token
    hex_parts: list[str] = []
    for token in text.split():
        clean = token.lower().removeprefix("0x")
        if clean and all(c in "0123456789abcdef" for c in clean):
            hex_parts.append(clean)
        else:
            break  # units text or other non-hex suffix

    hex_str = "".join(hex_parts)
    if not hex_str:
        raise ValueError(f"No hex data found in ACL /raw output: {text!r}")

    # Pad to even length for bytes.fromhex
    if len(hex_str) % 2:
        hex_str = "0" + hex_str

    return bytes.fromhex(hex_str)


def _parse_response_line(drf: str, line: str) -> tuple[Value, ValueType]:
    """Parse a single ACL response line, choosing raw or text parsing."""
    if _is_raw_field(drf):
        return _parse_raw_hex(line), ValueType.RAW
    return _parse_acl_line(line)


def _parse_acl_line(text: str) -> tuple[Value, ValueType]:
    """Parse a single line of ACL output into a value and type.

    ACL output format is typically: ``DEVICE = VALUE [UNITS]``
    For alarm/description fields the format varies but always uses '='.
    Lines without '=' (e.g. bare numeric from no_name/no_units) are also handled.
    """
    text = text.strip()

    # Extract value part after '='
    if "=" in text:
        _, _, raw = text.partition("=")
        raw = raw.strip()
    else:
        raw = text

    if not raw:
        return text, ValueType.TEXT

    # 1. Try whole string as float (e.g. "12.34")
    try:
        return float(raw), ValueType.SCALAR
    except ValueError:
        pass

    tokens = raw.split()

    # 2. Try all tokens as floats → array (e.g. "45 2.2 3.0")
    if len(tokens) > 1:
        try:
            return [float(t) for t in tokens], ValueType.SCALAR_ARRAY
        except ValueError:
            pass

    # 3. Try all-but-last as floats → array + units (e.g. "45 2.2 3.0 blip")
    if len(tokens) > 2:
        try:
            return [float(t) for t in tokens[:-1]], ValueType.SCALAR_ARRAY
        except ValueError:
            pass

    # 4. Try first token as float → scalar + units (e.g. "12.34 DegF")
    try:
        return float(tokens[0]), ValueType.SCALAR
    except ValueError:
        pass

    # 5. Text
    return raw, ValueType.TEXT


def _is_error_response(text: str) -> tuple[bool, Optional[str]]:
    """Check if an ACL response line indicates an error.

    ACL errors look like::

        ! error message
        Invalid device name (...) ... - DIO_NO_SUCH
        Error reading device ... - DIO_NOSCALE
    """
    text = text.strip()

    if text.startswith("!"):
        return True, text[1:].strip()

    # ACL errors end with " - ERROR_CODE" or " - DEVICE ERROR_CODE"
    if " - " in text:
        tail = text.rsplit(" - ", 1)[-1].strip()
        # Bare error code (e.g. "DIO_NO_SUCH") or device-prefixed (e.g. "Z:ACLTST DIO_NOATT")
        error_code = tail.rsplit(None, 1)[-1] if " " in tail else tail
        if _ACL_ERROR_CODE_RE.match(error_code):
            return True, text

    return False, None


class ACLBackend(Backend):
    """
    ACL backend for HTTP-based device reads (read-only).

    Uses the ACL CGI endpoint for simple device access.
    No streaming, no authentication, no writes.

    Capabilities:
        - READ: Always enabled
        - WRITE: No (read-only backend)
        - STREAM: No (HTTP is request/response only)
        - AUTH: No (public endpoint)
        - BATCH: Yes (get_many supported)

    Example:
        with ACLBackend() as backend:
            temp = backend.read("M:OUTTMP")
            reading = backend.get("M:OUTTMP")
            readings = backend.get_many(["M:OUTTMP", "G:AMANDA"])
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
    ):
        """
        Initialize ACL backend.

        Args:
            base_url: ACL CGI URL (default: $PACSYS_ACL_URL or https://www-bd.fnal.gov/cgi-bin/acl.pl)
            timeout: HTTP request timeout in seconds (default: 5.0)
            verify_ssl: Verify SSL certificates (disable for local proxies)

        Raises:
            ValueError: If parameters are invalid
        """
        effective_url = base_url if base_url is not None else os.environ.get("PACSYS_ACL_URL", DEFAULT_BASE_URL)
        effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        if not effective_url:
            raise ValueError("base_url cannot be empty")
        if effective_timeout <= 0:
            raise ValueError(f"timeout must be positive, got {effective_timeout}")

        self._base_url = effective_url
        self._timeout = effective_timeout
        self._closed = False
        self._client = httpx.Client(verify=verify_ssl, timeout=effective_timeout)

        logger.debug(f"ACLBackend initialized: base_url={effective_url}, timeout={effective_timeout}")

    @property
    def capabilities(self) -> BackendCapability:
        """ACL only supports READ and BATCH."""
        return BackendCapability.READ | BackendCapability.BATCH

    @property
    def base_url(self) -> str:
        """ACL CGI base URL."""
        return self._base_url

    @property
    def timeout(self) -> float:
        """Default request timeout."""
        return self._timeout

    def _build_url(self, drfs: list[str]) -> str:
        """Build ACL CGI URL for one or more devices.

        ACL ``read`` takes exactly one device; multiple devices are sent as
        separate ``read`` commands joined by ``\\;`` (escaped semicolon).

        Single:   ``?acl=read+DEVICE/qualifier``
        Batch:    ``?acl=read+DEV1/q1\\;read+DEV2/q2``

        Qualifiers (``/raw``, ``/ftd=evtXX``) must come **after** the device
        name - ACL rejects them before the device (CLIB_SYNTAX).
        """
        # The ACL CGI only decodes spaces (+/%20) and quotes (%27) from the
        # query string - general %XX sequences like %3A are NOT decoded.
        # DRF characters (colons, brackets, etc.) must be sent raw.
        commands = []
        for drf in drfs:
            cmd, clean_drf, qualifiers = _acl_read_command(drf)
            quoted = urllib.parse.quote(clean_drf, safe=":[]@,.$|~")
            commands.append(f"{cmd}+{quoted}{qualifiers}")
        return f"{self._base_url}?acl={_ACL_CMD_SEP.join(commands)}"

    def execute(self, acl_command: str, timeout: Optional[float] = None) -> str:
        """Execute a raw ACL command string and return the text output.

        The *acl_command* is placed verbatim after ``?acl=`` in the CGI URL.
        Spaces should be ``+``, semicolons escaped as ``\\;``.

        Example::

            backend.execute("read+M:OUTTMP")
            backend.execute(
                "device_list/create+devs+devices='M:OUTTMP,G:AMANDA'"
                "\\\\;read_list/no_name/no_units+device_list=devs"
            )
        """
        if self._closed:
            raise RuntimeError("Backend is closed")
        effective_timeout = timeout if timeout is not None else self._timeout
        url = f"{self._base_url}?acl={acl_command}"
        return self._fetch(url, effective_timeout)

    def _fetch(self, url: str, timeout: float) -> str:
        """Fetch URL content.

        Raises:
            DeviceError: If HTTP request fails
        """
        try:
            response = self._client.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            raise DeviceError(
                drf="",
                facility_code=0,
                error_code=ERR_RETRY,
                message=f"ACL request failed ({url}): HTTP {e.response.status_code}",
            ) from e
        except httpx.TimeoutException as e:
            raise DeviceError(
                drf="",
                facility_code=0,
                error_code=ERR_TIMEOUT,
                message=f"ACL request timed out after {timeout}s ({self._base_url})",
            ) from e
        except httpx.TransportError as e:
            raise DeviceError(
                drf="",
                facility_code=0,
                error_code=ERR_RETRY,
                message=f"ACL request failed ({self._base_url}): {e}",
            ) from e

    def read(self, drf: str, timeout: Optional[float] = None) -> Value:
        """Read a single device value via HTTP.

        Raises:
            RuntimeError: If backend is closed
            DeviceError: If the read fails
        """
        reading = self.get(drf, timeout=timeout)

        if not reading.ok:
            raise DeviceError(
                drf=reading.drf,
                facility_code=reading.facility_code,
                error_code=reading.error_code,
                message=reading.message or f"Read failed with status {reading.error_code}",
            )

        assert reading.value is not None
        return reading.value

    def get(self, drf: str, timeout: Optional[float] = None) -> Reading:
        """Read a single device with metadata via HTTP."""
        readings = self.get_many([drf], timeout=timeout)
        return readings[0]

    def get_many(self, drfs: list[str], timeout: Optional[float] = None) -> list[Reading]:
        """Read multiple devices via HTTP.

        Sends all reads in a single request using semicolon-separated ACL
        commands.  If the batch fails (e.g. one bad device aborts the whole
        script), falls back to issuing one HTTP request per device so that
        valid devices still return data and only the bad ones get errors.

        Basic-status DRFs (e.g. ``N|LGXS``, ``Z:ACLTST.STATUS``) are routed
        through per-field reads automatically.

        .. todo:: When all DRFs share the same property, use ``device_list``
           + ``read_list`` for a true simultaneous batch read instead of
           sequential ``read`` commands.
        """
        if self._closed:
            raise RuntimeError("Backend is closed")

        if not drfs:
            return []

        effective_timeout = timeout if timeout is not None else self._timeout

        # Route basic-status DRFs through per-field reads (5 HTTP requests each).
        # Remaining DRFs go through the normal batch path (recursive, safe
        # because the recursive call contains no status DRFs).
        status_indices = {i for i, d in enumerate(drfs) if _is_basic_status_request(d)}
        if status_indices:
            normal_drfs = [d for i, d in enumerate(drfs) if i not in status_indices]
            normal_iter = iter(self.get_many(normal_drfs, timeout=timeout) if normal_drfs else [])
            return [
                self._get_basic_status(drf, effective_timeout) if i in status_indices else next(normal_iter)
                for i, drf in enumerate(drfs)
            ]

        url = self._build_url(drfs)
        logger.debug(f"ACL batch request: {url}")

        try:
            response_text = self._fetch(url, effective_timeout)
        except DeviceError as e:
            if e.error_code == ERR_TIMEOUT:
                # Server is unresponsive - individual reads would each timeout too.
                readings = [
                    Reading(
                        drf=drf,
                        value_type=ValueType.SCALAR,
                        facility_code=e.facility_code,
                        error_code=e.error_code,
                        message=e.message,
                        timestamp=datetime.now(),
                    )
                    for drf in drfs
                ]
                raise ReadError(readings, e.message or "ACL request timeout") from e
            # HTTP/URL error (e.g. 400 from one bad DRF) - fall back to
            # individual reads to isolate which devices actually failed.
            logger.debug("ACL batch HTTP error, falling back to individual reads: %s", e.message)
            return self._get_many_individual(drfs, effective_timeout)

        logger.debug(f"ACL batch response: {response_text[:200]}")

        lines = response_text.strip().splitlines()

        # ACL aborts the whole script on the first bad device, so if line
        # count doesn't match or any line is an error, fall back to
        # individual reads to isolate the failure(s).
        if len(lines) != len(drfs) or any(_is_error_response(line)[0] for line in lines):
            logger.debug(
                "ACL batch error/mismatch (%d lines for %d drfs), falling back to individual reads",
                len(lines),
                len(drfs),
            )
            return self._get_many_individual(drfs, effective_timeout)

        # Happy path: one line per device, in order
        readings: list[Reading] = []
        now = datetime.now()
        for drf, line in zip(drfs, lines):
            try:
                value, value_type = _parse_response_line(drf, line)
            except ValueError as exc:
                readings.append(
                    Reading(drf=drf, value_type=ValueType.RAW, error_code=ERR_RETRY, message=str(exc), timestamp=now)
                )
                continue
            readings.append(Reading(drf=drf, value_type=value_type, value=value, error_code=ERR_OK, timestamp=now))
        return readings

    def _get_many_individual(self, drfs: list[str], timeout: float) -> list[Reading]:
        """Fallback: read each device individually to isolate errors."""
        readings: list[Reading] = []
        now = datetime.now()
        for drf in drfs:
            url = self._build_url([drf])
            try:
                response_text = self._fetch(url, timeout)
                lines = response_text.strip().splitlines()
                if not lines:
                    readings.append(
                        Reading(
                            drf=drf,
                            value_type=ValueType.SCALAR,
                            error_code=ERR_RETRY,
                            message="Empty response from ACL",
                            timestamp=now,
                        )
                    )
                    continue
                line = lines[0]
                is_error, error_msg = _is_error_response(line)
                if is_error:
                    readings.append(
                        Reading(
                            drf=drf,
                            value_type=ValueType.SCALAR,
                            facility_code=0,
                            error_code=ERR_RETRY,
                            message=error_msg,
                            timestamp=now,
                        )
                    )
                else:
                    try:
                        value, value_type = _parse_response_line(drf, line)
                    except ValueError as exc:
                        readings.append(
                            Reading(
                                drf=drf,
                                value_type=ValueType.RAW,
                                error_code=ERR_RETRY,
                                message=str(exc),
                                timestamp=now,
                            )
                        )
                        continue
                    readings.append(
                        Reading(drf=drf, value_type=value_type, value=value, error_code=ERR_OK, timestamp=now)
                    )
            except DeviceError as e:
                readings.append(
                    Reading(
                        drf=drf,
                        value_type=ValueType.SCALAR,
                        facility_code=e.facility_code,
                        error_code=e.error_code,
                        message=e.message,
                        timestamp=now,
                    )
                )
        return readings

    def _get_basic_status(self, drf: str, timeout: float) -> Reading:
        """Build a BASIC_STATUS dict by reading individual status fields.

        Issues one HTTP request per field (ON, READY, REMOTE, POSITIVE, RAMP).
        Fields that don't exist on the device (DIO_NOATT) are omitted from
        the dict, matching DPM's behavior.  Any other ACL error (e.g.
        nonexistent device → DBM_NOREC) immediately fails the whole read.
        """
        device = parse_request(drf).device
        now = datetime.now()
        status: dict[str, bool] = {}

        for key, field in zip(_BASIC_STATUS_KEYS, _BASIC_STATUS_FIELDS):
            url = self._build_url([f"{device}.STATUS.{field}"])
            try:
                lines = self._fetch(url, timeout).strip().splitlines()
            except DeviceError as e:
                return Reading(
                    drf=drf,
                    value_type=ValueType.BASIC_STATUS,
                    facility_code=e.facility_code,
                    error_code=e.error_code,
                    message=e.message,
                    timestamp=now,
                )
            if not lines:
                return Reading(
                    drf=drf,
                    value_type=ValueType.BASIC_STATUS,
                    error_code=ERR_RETRY,
                    message=f"Empty response from ACL for {field}",
                    timestamp=now,
                )
            line = lines[0]

            # DIO_NOATT means the device lacks this attribute - omit the
            # key, matching DPM behavior.  The response format includes the
            # device name before the error code ("- Z:ACLTST DIO_NOATT"),
            # so we check the line directly rather than relying on
            # _is_error_response (whose regex expects a bare error code).
            if "DIO_NOATT" in line:
                continue

            is_err, msg = _is_error_response(line)
            if is_err:
                return Reading(
                    drf=drf, value_type=ValueType.BASIC_STATUS, error_code=ERR_RETRY, message=msg, timestamp=now
                )

            if "= True" in line:
                status[key] = True
            elif "= False" in line:
                status[key] = False

        return Reading(drf=drf, value_type=ValueType.BASIC_STATUS, value=status, error_code=ERR_OK, timestamp=now)

    def close(self) -> None:
        """Close the backend and underlying HTTP client."""
        self._closed = True
        self._client.close()
        logger.debug("ACLBackend closed")

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"ACLBackend({self._base_url}, timeout={self._timeout}, {status})"


__all__ = ["ACLBackend"]
