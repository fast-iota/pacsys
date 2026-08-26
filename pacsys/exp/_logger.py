"""DataLogger: subscribe and log readings to file via pluggable writers."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from pacsys.exp._resolve import resolve_backend, resolve_drf

if TYPE_CHECKING:
    from pacsys.backends import Backend
    from pacsys.exp._writers import LogWriter
    from pacsys.types import DeviceSpec, Reading, SubscriptionHandle

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
        self._lifecycle_lock = threading.Lock()
        self._writer_lock = threading.Lock()
        self._buffer: list[Reading] = []
        self._handle: SubscriptionHandle | None = None
        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stopped = False
        self._closed = False
        self._last_error: Exception | None = None
        self._retry_count: int = 0
        self._dropped_count: int = 0
        self._max_retries: int = 3

    @property
    def running(self) -> bool:
        return self._handle is not None and not self._handle.stopped and not self._stopped

    @property
    def last_error(self) -> Exception | None:
        """Last write error, or None if no errors occurred."""
        return self._last_error

    @property
    def dropped_count(self) -> int:
        """Readings lost after exhausting write retries; sticky until the next start()."""
        return self._dropped_count

    @property
    def failed(self) -> bool:
        """True once any batch has been dropped. Logging continues; stop() raises."""
        return self._dropped_count > 0

    def start(self) -> None:
        """Start logging."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("DataLogger has been stopped and writer is closed; create a new instance")
            if self.running or (self._flush_thread is not None and self._flush_thread.is_alive()):
                raise RuntimeError("DataLogger is already running")
            with self._lock:
                self._stopped = False
                self._last_error = None
                self._dropped_count = 0
            self._stop_event.clear()
            be = resolve_backend(self._backend)
            try:
                self._handle = be.subscribe(self._drfs, callback=self._on_reading)
                self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
                self._flush_thread.start()
            except BaseException:
                if self._handle is not None:
                    try:
                        self._handle.stop()
                    except Exception:  # noqa: BLE001
                        logger.exception("Error stopping subscription after DataLogger start failure")
                self._handle = None
                self._flush_thread = None
                self._stop_event.set()
                with self._lock:
                    self._stopped = True
                raise

    def stop(self) -> None:
        """Stop logging and flush remaining data."""
        with self._lifecycle_lock:
            if self._closed:
                return
            with self._lock:
                self._stopped = True
            self._stop_event.set()
            handle_error = None
            if self._handle is not None:
                try:
                    self._handle.stop()
                except Exception as exc:  # noqa: BLE001
                    handle_error = exc
                    with self._lock:
                        self._last_error = exc
                    logger.exception("Error stopping DataLogger subscription")
            if self._flush_thread is not None:
                if self._flush_thread is threading.current_thread():
                    error = RuntimeError("DataLogger cannot be stopped from its flush worker")
                    with self._lock:
                        self._last_error = error
                    raise error
                self._flush_thread.join(timeout=5.0)
                if self._flush_thread.is_alive():
                    error = RuntimeError("DataLogger flush worker did not stop; writer left open")
                    with self._lock:
                        self._last_error = error
                    logger.error("%s", error)
                    raise error

            while True:
                with self._lock:
                    if not self._buffer:
                        break
                self._flush_now()

            with self._lock:
                dropped = self._dropped_count

            if handle_error is not None:
                raise RuntimeError("DataLogger subscription did not stop; writer left open") from handle_error

            try:
                with self._writer_lock:
                    self._writer.close()
            except Exception as exc:
                with self._lock:
                    self._last_error = exc
                logger.exception("Error closing data logger writer")
                raise

            self._closed = True
            self._handle = None
            self._flush_thread = None
            if dropped:
                error = RuntimeError(f"Dropped {dropped} readings during DataLogger run (see last_error)")
                raise error from self._last_error

    def _on_reading(self, reading: Reading, handle: SubscriptionHandle) -> None:
        with self._lock:
            if self._stopped:
                return
            self._buffer.append(reading)

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._flush_interval):
            self._flush_now()

    # TODO: distinguish conversion/validation failures (bad reading — quarantine it)
    # from transient I/O failures (retryable) so one poison reading cannot drop the batch
    def _flush_now(self) -> None:
        """Flush one buffered batch."""
        with self._lock:
            batch = self._buffer
            self._buffer = []
        if batch:
            try:
                with self._writer_lock:
                    self._writer.write_readings(batch)
                with self._lock:
                    self._retry_count = 0
            except Exception as exc:
                with self._lock:
                    self._retry_count += 1
                    self._last_error = exc
                    attempt = self._retry_count
                    if attempt < self._max_retries:
                        self._buffer = batch + self._buffer
                if attempt < self._max_retries:
                    logger.exception("Error writing readings (attempt %d/%d)", attempt, self._max_retries)
                else:
                    logger.error(
                        "Dropping %d readings after %d failed attempts: %s",
                        len(batch),
                        self._max_retries,
                        exc,
                    )
                    with self._lock:
                        self._retry_count = 0
                        self._dropped_count += len(batch)
                    return

    def __enter__(self) -> DataLogger:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            if exc_type is None:
                raise
            logger.exception("DataLogger shutdown failed while handling another exception")
        return False
