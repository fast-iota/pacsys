"""DataLogger: subscribe and log readings to file via pluggable writers."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from pacsys.types import DeviceSpec, Reading, SubscriptionHandle
from pacsys.exp._resolve import resolve_drf, resolve_backend

if TYPE_CHECKING:
    from pacsys.backends import Backend
    from pacsys.exp._writers import LogWriter

logger = logging.getLogger(__name__)


class DataLogger:
    """Subscribe to channels and log readings via a pluggable writer.

    Usage:
        with DataLogger(drfs, writer=CsvWriter("log.csv")) as dl:
            time.sleep(60)
    """

    def __init__(
        self,
        devices: list[DeviceSpec],
        writer: LogWriter,
        *,
        flush_interval: float = 5.0,
        backend: Backend | None = None,
    ):
        if not devices:
            raise ValueError("devices cannot be empty")
        if flush_interval <= 0:
            raise ValueError("flush_interval must be > 0")
        self._drfs = [resolve_drf(d) for d in devices]
        self._writer = writer
        self._flush_interval = flush_interval
        self._backend = backend
        self._lock = threading.Lock()
        self._buffer: list[Reading] = []
        self._handle: SubscriptionHandle | None = None
        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stopped = False
        self._closed = False
        self._last_error: Exception | None = None

    @property
    def running(self) -> bool:
        return self._handle is not None and not self._handle.stopped and not self._stopped

    @property
    def last_error(self) -> Exception | None:
        """Last write error, or None if no errors occurred."""
        return self._last_error

    def start(self) -> None:
        """Start logging."""
        if self._closed:
            raise RuntimeError("DataLogger has been stopped and writer is closed; create a new instance")
        if self.running:
            raise RuntimeError("DataLogger is already running")
        self._stopped = False
        self._last_error = None
        self._stop_event.clear()
        be = resolve_backend(self._backend)
        self._handle = be.subscribe(self._drfs, callback=self._on_reading)
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop(self) -> None:
        """Stop logging and flush remaining data."""
        self._stopped = True
        self._stop_event.set()
        if self._handle is not None:
            self._handle.stop()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
        self._flush_now()
        self._writer.close()
        self._closed = True

    def _on_reading(self, reading: Reading, handle: SubscriptionHandle) -> None:
        if self._stopped:
            return
        with self._lock:
            self._buffer.append(reading)

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._flush_interval):
            self._flush_now()

    def _flush_now(self) -> None:
        with self._lock:
            batch = self._buffer
            self._buffer = []
        if batch:
            try:
                self._writer.write_readings(batch)
            except Exception as exc:
                # Re-buffer failed batch so data is not lost
                with self._lock:
                    self._buffer = batch + self._buffer
                self._last_error = exc
                logger.exception("Error writing readings")

    def __enter__(self) -> DataLogger:
        self.start()
        return self

    def __exit__(self, *args) -> bool:
        self.stop()
        return False
