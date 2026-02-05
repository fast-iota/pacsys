"""
DPM connection via ACNET protocol.

This module provides DPM access using AcnetConnectionTCP to communicate
with DPM servers via the ACNET protocol.

This is an alternative to the direct HTTP-based DPMConnection.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from pacsys.dpm_protocol import (
    AddToList_reply,
    AddToList_request,
    AnalogAlarm_reply,
    BasicStatus_reply,
    ClearList_request,
    DeviceInfo_reply,
    DigitalAlarm_reply,
    ListStatus_reply,
    OpenList_reply,
    OpenList_request,
    Raw_reply,
    Scalar_reply,
    ScalarArray_reply,
    StartList_reply,
    StartList_request,
    Status_reply,
    StopList_request,
    Text_reply,
    TextArray_reply,
    TimedScalarArray_reply,
    unmarshal_reply,
)

from .connection_sync import ACSYS_PROXY_HOST, AcnetConnectionTCP  # noqa: E402

logger = logging.getLogger(__name__)

# DPM task name for service discovery
DPM_TASK = "DPMD"
DPM_MCAST = "MCAST"


@dataclass
class DPMReading:
    """Data reading from DPM."""

    ref_id: int
    timestamp: int = 0
    cycle: int = 0
    status: int = 0
    data: object = None
    meta: Optional[dict] = None
    micros: object = None  # Per-sample timestamps from TimedScalarArray (int64 list)


class DPMError(Exception):
    """DPM-specific error."""

    def __init__(self, status: int, message: str = ""):
        self.status = status
        self.message = message
        super().__init__(f"DPM error {status}: {message}" if message else f"DPM error {status}")


class DPMAcnet:
    """
    DPM connection via ACNET protocol.

    Uses AcnetConnectionTCP to communicate with DPM servers, matching
    the approach used by the acsys-python.

    Example usage:
        with DPMAcnet() as dpm:
            dpm.add_entry(1, "M:OUTTMP")
            dpm.start()

            for reading in dpm.readings(timeout=5.0):
                print(f"Tag {reading.ref_id}: {reading.data}")
    """

    def __init__(self, host: str = ACSYS_PROXY_HOST, dpm_node: Optional[str] = None, *, trace: bool = False):
        """
        Initialize DPM ACNET connection.

        Args:
            host: ACNET proxy hostname (default: acsys-proxy.fnal.gov)
            dpm_node: Specific DPM node to use (if None, uses service discovery)
            trace: Enable packet-level tracing on the ACNET connection
        """
        self._host = host
        self._desired_node = dpm_node
        self._trace = trace

        # ACNET connection
        self._con: Optional[AcnetConnectionTCP] = None

        # DPM state
        self._dpm_task: Optional[str] = None
        self._dpm_node: Optional[int] = None
        self._list_id: Optional[int] = None
        self._active = False

        # Request tracking
        self._dev_list: dict[int, str] = {}  # tag -> drf
        self._meta: dict[int, dict] = {}  # ref_id -> metadata

        # Reply handling (bounded to prevent OOM on slow consumers)
        self._reply_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._request_ctx = None

        # Lock for state
        self._lock = threading.Lock()

    @property
    def list_id(self) -> Optional[int]:
        """Get the current list ID."""
        return self._list_id

    @property
    def handle(self) -> str:
        """Get the ACNET handle."""
        return self._con.handle if self._con else ""

    def connect(self):
        """Connect to ACNET and DPM."""
        # Create ACNET connection
        self._con = AcnetConnectionTCP(host=self._host, trace=self._trace)
        self._con.connect()

        # Find DPM and open list
        self._find_dpm()
        self._open_list()

        logger.info(f"Connected to DPM at {self._dpm_task}, list_id={self._list_id}")

    def close(self):
        """Close the connection."""
        if self._request_ctx:
            try:
                self._request_ctx.cancel()
            except Exception:
                pass

        if self._con:
            try:
                self._con.close()
            except Exception:
                pass
            self._con = None

        logger.info("Closed DPM ACNET connection")

    def _find_dpm(self):
        """Find an available DPM server."""
        assert self._con is not None, "not connected"
        if self._desired_node:
            # Use specified node
            self._dpm_node = self._con.get_node(self._desired_node)
            logger.debug(f"Using specified DPM node: {self._desired_node} ({self._dpm_node})")
        else:
            # Use known DPM node (DPM06 is known to work)
            # Service discovery via MCAST can cause issues with multiple ACKs
            self._dpm_node = self._con.get_node("DPM06")
            logger.debug(f"Using DPM06 node: {self._dpm_node}")

    def _open_list(self):
        """Open a new DPM list."""
        msg = OpenList_request()
        data = bytes(msg.marshal())

        result_event = threading.Event()
        result = {"list_id": None, "error": None}

        def handle_reply(reply):
            try:
                resp = unmarshal_reply(iter(reply.data))
                if isinstance(resp, OpenList_reply):
                    result["list_id"] = resp.list_id
                else:
                    # Store for later processing
                    self._handle_dpm_reply(resp)
            except Exception as e:
                result["error"] = str(e)
            result_event.set()

        # Send as multiple-reply request (stays open for data)
        assert self._con is not None, "not connected"
        assert self._dpm_node is not None, "DPM node not found"
        self._request_ctx = self._con.request_multiple(
            node=self._dpm_node,
            task=DPM_TASK,
            data=data,
            reply_handler=handle_reply,
            timeout=0,  # No timeout
        )

        # Wait for OpenList reply
        if not result_event.wait(timeout=5.0):
            raise DPMError(-1, "Timeout waiting for OpenList reply")

        if result["error"]:
            raise DPMError(-1, f"OpenList failed: {result['error']}")

        self._list_id = result["list_id"]

    def _send_request(self, msg, timeout: float = 5.0) -> object:
        """Send a single-reply request to DPM."""
        data = bytes(msg.marshal())

        result_event = threading.Event()
        result = {"reply": None, "error": None}

        def handle_reply(reply):
            try:
                result["reply"] = unmarshal_reply(iter(reply.data))
            except Exception as e:
                result["error"] = str(e)
            result_event.set()

        assert self._con is not None, "not connected"
        assert self._dpm_node is not None, "DPM node not found"
        ctx = self._con.request_single(
            node=self._dpm_node,
            task=DPM_TASK,
            data=data,
            reply_handler=handle_reply,
            timeout=int(timeout * 1000),
        )

        if not result_event.wait(timeout=timeout + 1.0):
            ctx.cancel()
            raise DPMError(-1, "Timeout waiting for reply")

        if result["error"]:
            raise DPMError(-1, result["error"])

        return result["reply"]

    def add_entry(self, tag: int, drf: str):
        """
        Add a device request to the list.

        Args:
            tag: Integer tag to identify this request in replies
            drf: DRF3 format device request string
        """
        if not isinstance(tag, int):
            raise ValueError("tag must be an integer")
        if not isinstance(drf, str):
            raise ValueError("drf must be a string")

        msg = AddToList_request()
        msg.list_id = self._list_id
        msg.ref_id = tag
        msg.drf_request = drf

        reply = self._send_request(msg)

        if isinstance(reply, AddToList_reply):
            if reply.status < 0:
                raise DPMError(reply.status, f"AddToList failed for {drf}")

        # Track the request
        if not drf.startswith("#"):
            self._dev_list[tag] = drf

        logger.debug(f"Added entry tag={tag}, drf={drf}")

    def start(self, model: str | None = None):
        """Start data acquisition."""
        msg = StartList_request()
        msg.list_id = self._list_id
        if model:
            msg.model = model

        reply = self._send_request(msg)

        if isinstance(reply, StartList_reply):
            if reply.status < 0:
                raise DPMError(reply.status, "StartList failed")

        self._active = True
        logger.debug("Started DPM acquisition")

    def stop(self):
        """Stop data acquisition."""
        if not self._active:
            return

        msg = StopList_request()
        msg.list_id = self._list_id

        try:
            self._send_request(msg)
        except Exception as e:
            logger.warning(f"Error stopping list: {e}")

        self._active = False
        logger.debug("Stopped DPM acquisition")

    def clear_list(self):
        """Clear all entries from the list."""
        msg = ClearList_request()
        msg.list_id = self._list_id

        try:
            self._send_request(msg)
        except Exception as e:
            logger.warning(f"Error clearing list: {e}")

        self._dev_list.clear()
        self._meta.clear()
        logger.debug("Cleared DPM list")

    def _handle_dpm_reply(self, msg):
        """Handle a DPM reply message."""
        if isinstance(msg, ListStatus_reply):
            return

        if isinstance(msg, DeviceInfo_reply):
            self._meta[msg.ref_id] = {
                "di": msg.di,
                "name": msg.name,
                "desc": msg.description,
                "units": getattr(msg, "units", None),
                "format_hint": getattr(msg, "format_hint", None),
            }
            return

        if isinstance(msg, Status_reply):
            reading = DPMReading(
                ref_id=msg.ref_id,
                status=msg.status,
                meta=self._meta.get(msg.ref_id),
            )
            try:
                self._reply_queue.put_nowait(reading)
            except queue.Full:
                pass  # drop newest on overflow
            return

        if isinstance(
            msg,
            (
                Scalar_reply,
                ScalarArray_reply,
                Raw_reply,
                Text_reply,
                TextArray_reply,
                TimedScalarArray_reply,
            ),
        ):
            micros = None
            if isinstance(msg, TimedScalarArray_reply) and hasattr(msg, "micros") and msg.micros:
                micros = msg.micros
            reading = DPMReading(
                ref_id=msg.ref_id,
                timestamp=msg.timestamp,
                cycle=msg.cycle,
                status=msg.status,
                data=msg.data if hasattr(msg, "data") else None,
                meta=self._meta.get(msg.ref_id),
                micros=micros,
            )
            try:
                self._reply_queue.put_nowait(reading)
            except queue.Full:
                pass  # drop newest on overflow
            return

        if isinstance(msg, (AnalogAlarm_reply, DigitalAlarm_reply, BasicStatus_reply)):
            reading = DPMReading(
                ref_id=msg.ref_id,
                timestamp=msg.timestamp,
                data=msg.__dict__,
                meta=self._meta.get(msg.ref_id),
            )
            try:
                self._reply_queue.put_nowait(reading)
            except queue.Full:
                pass  # drop newest on overflow
            return

    def readings(self, timeout: float | None = None):
        """
        Generator that yields readings from DPM.

        Args:
            timeout: Maximum time to wait for next reading (None = forever)

        Yields:
            DPMReading objects
        """
        while True:
            try:
                yield self._reply_queue.get(timeout=timeout)
            except queue.Empty:
                return

    def read(self, drf: str, timeout: float = 5.0) -> DPMReading:
        """
        Read a device value synchronously.

        Args:
            drf: DRF format device request (e.g., "M:OUTTMP")
            timeout: Read timeout in seconds

        Returns:
            DPMReading with the device value
        """
        # Add @I for immediate if no event specified
        if "@" not in drf:
            drf = f"{drf}@I"

        # Use a unique tag
        tag = hash(drf) & 0x7FFFFFFF

        self.add_entry(tag, drf)

        was_active = self._active
        if not was_active:
            self.start()

        try:
            start = time.time()
            for reading in self.readings(timeout=timeout):
                if reading.ref_id == tag:
                    if reading.status < 0:
                        raise DPMError(reading.status, f"Error reading {drf}")
                    return reading
                if time.time() - start > timeout:
                    break

            raise TimeoutError(f"Timeout reading {drf}")

        finally:
            if not was_active:
                self.stop()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
