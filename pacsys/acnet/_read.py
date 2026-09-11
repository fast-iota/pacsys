"""Shared raw read descriptors and request lifecycle for GETS32/RETDAT."""

from __future__ import annotations

import logging
import math
import struct
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from .connection_sync import AcnetConnectionTCP, AcnetConnectionUDP, AcnetRequestContext
from .errors import ACNET_ENDMULT, ACNET_PEND, AcnetError, AcnetTimeoutError
from .packet import AcnetReply

logger = logging.getLogger(__name__)
Connection = AcnetConnectionTCP | AcnetConnectionUDP
T = TypeVar("T")

# Even-sized payloads: local IPv4 UDP command (22 bytes), and acnetd's
# 65534-byte IP packet minus IPv4/UDP/ACNET headers (20/8/18 bytes).
_MAX_REQUEST_SIZE = 65_484
_MAX_REPLY_SIZE = 65_488


def _uint(value: int, maximum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be 0..{maximum}, got {value}")


@dataclass(frozen=True)
class ReadDevice:
    """Explicit property address. Length and offset are bytes; SSDN is wire order.

    The low 24 DI bits are preserved, including any legacy frontend flag bits.
    No database lookup, scaling, or property interpretation is performed.
    """

    di: int
    pi: int
    ssdn: bytes
    length: int
    offset: int = 0

    def __post_init__(self) -> None:
        _uint(self.di, 0xFFFFFF, "di")
        _uint(self.pi, 0xFF, "pi")
        _uint(self.length, 0xFFFFFFFF, "length")
        _uint(self.offset, 0xFFFFFFFF, "offset")
        if not isinstance(self.ssdn, bytes):
            raise TypeError("ssdn must be bytes in wire order")
        if len(self.ssdn) != 8:
            raise ValueError("ssdn must contain exactly 8 bytes")

    @property
    def dipi(self) -> int:
        return (self.pi << 24) | self.di


@dataclass(frozen=True)
class ReadValue:
    """Per-entry status and untouched wire data, including odd-length padding.

    Data layout belongs to the device: integers, floats, text, arrays, status,
    alarm blocks, and custom structures all remain bytes. Error slots are kept.
    """

    status: int
    data: bytes


def _devices(devices: Sequence[ReadDevice], maximum: int) -> tuple[ReadDevice, ...]:
    result = tuple(devices)
    if not result:
        raise ValueError("devices must not be empty")
    _uint(len(result), 0xFFFF, "device count")
    for device in result:
        if not isinstance(device, ReadDevice):
            raise TypeError("devices must contain ReadDevice objects")
        _uint(device.length, maximum, "length")
        _uint(device.offset, maximum, "offset")
    return result


def _reply_size(devices: Sequence[ReadDevice]) -> int:
    return sum(2 + device.length + (device.length & 1) for device in devices)


def _message_limits(max_request_size: int, max_reply_size: int) -> None:
    for value, maximum, name in (
        (max_request_size, _MAX_REQUEST_SIZE, "max_request_size"),
        (max_reply_size, _MAX_REPLY_SIZE, "max_reply_size"),
    ):
        _uint(value, maximum, name)
        if value == 0:
            raise ValueError(f"{name} must be positive")


def _message_size(size: int, limit: int, name: str) -> None:
    if size > limit:
        raise ValueError(f"{name} size {size} exceeds configured limit {limit}; split the request or raise the limit")


def _parse_values(data: bytes, devices: Sequence[ReadDevice], offset: int) -> tuple[ReadValue, ...]:
    expected = offset + _reply_size(devices)
    if len(data) != expected:
        raise ValueError(f"Read reply size {len(data)} does not match expected {expected}")
    values = []
    for device in devices:
        status = struct.unpack_from("<h", data, offset)[0]
        end = offset + 2 + device.length + (device.length & 1)
        values.append(ReadValue(status, data[offset + 2 : end]))
        offset = end
    return tuple(values)


def _timeout(timeout: float | None) -> None:
    if timeout is not None and (isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be finite and positive, or None")


class ReadStream(Generic[T]):
    """One ACNET request. Close it explicitly or use it as a context manager.

    The supplied connection remains caller-owned. Replies are buffered without
    blocking its reactor. A finite readings timeout bounds the entire iteration,
    including time spent consuming queued replies; pending heartbeats do not reset it.
    """

    def __init__(
        self,
        connection: Connection,
        node: int,
        task: str,
        payload: bytes,
        decode: Callable[[AcnetReply, int], T],
        *,
        repetitive: bool,
        buffer_size: int,
    ):
        _uint(node, 0xFFFF, "node")
        _uint(buffer_size, 0x7FFFFFFF, "buffer_size")
        if buffer_size == 0:
            raise ValueError("buffer_size must be positive")
        if not connection.connected:
            raise RuntimeError("ACNET connection must be connected before creating a read request")
        self._node = node
        self._task = task
        self._decode = decode
        self._buffer_size = buffer_size
        self._condition = threading.Condition()
        self._queue: deque[tuple[AcnetReply, int]] = deque()
        self._error: Exception | None = None
        self._ended = False
        self._closed = False
        self._iterating = False
        self._context: AcnetRequestContext | None = None
        try:
            self._context = connection.send_request(
                node, task, payload, self._on_reply, multiple_reply=repetitive, timeout=0
            )
            # A reply (including overflow) can arrive before send_request returns.
            if self._error is not None:
                self._cancel_request()
        except BaseException:
            logger.exception("%s request setup failed for node %s", task, node)
            try:
                self.close()
            except Exception:
                logger.exception("%s setup cleanup failed for node %s", task, node)
            raise

    def _cancel_request(self) -> None:
        if self._context is not None:
            self._context.cancel()

    def _on_reply(self, reply: AcnetReply) -> None:
        with self._condition:
            if self._closed or self._ended:
                return
            if reply.status < 0:
                self._error = AcnetError(reply.status, f"{self._task} read from node {self._node} failed")
            elif reply.data:
                if len(self._queue) >= self._buffer_size:
                    self._error = BufferError(f"{self._task} reply buffer overflow at node {self._node}")
                else:
                    self._queue.append((reply, time.time_ns()))
            elif reply.status != ACNET_PEND and not reply.last:
                self._error = AcnetError(reply.status, f"{self._task} received an empty non-pending reply")
            if self._error is not None:
                logger.error("%s", self._error)
            self._ended = reply.last or self._error is not None
            self._condition.notify_all()
        if self._error is not None:
            self._cancel_request()

    def readings(self, timeout: float | None = 1.0) -> Iterator[T]:
        """Yield complete batches; raise on timeout, transport error, or malformed data.

        Queued batches precede a terminal error. Explicit close discards queued
        batches and wakes readers. Only one iterator may consume a request at a time.
        """
        _timeout(timeout)
        return self._readings(timeout)

    def _readings(self, timeout: float | None) -> Iterator[T]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._iterating:
                raise RuntimeError("ReadStream already has an active iterator")
            self._iterating = True
        try:
            while True:
                with self._condition:
                    while True:
                        if self._closed:
                            return
                        remaining = None if deadline is None else deadline - time.monotonic()
                        if remaining is not None and remaining <= 0:
                            logger.error("%s read timed out at node %s", self._task, self._node)
                            raise AcnetTimeoutError(int(timeout * 1000) if timeout is not None else 0)
                        if self._queue:
                            reply, received_at_ns = self._queue.popleft()
                            break
                        if self._error is not None:
                            raise self._error
                        if self._ended:
                            return
                        self._condition.wait(remaining)
                try:
                    batch = self._decode(reply, received_at_ns)
                except Exception:
                    logger.exception("Malformed %s reply from node %s", self._task, self._node)
                    self.close()
                    raise
                with self._condition:
                    if self._closed:
                        return
                yield batch
        finally:
            with self._condition:
                self._iterating = False

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._queue.clear()
            self._condition.notify_all()
        self._cancel_request()

    def __enter__(self) -> ReadStream[T]:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception:
            logger.exception("Failed to cancel %s request at node %s", self._task, self._node)
            if exc_type is None:
                raise


def _first(stream: ReadStream[T], timeout: float) -> T:
    with stream:
        readings = stream.readings(timeout)
        try:
            return next(readings)
        except StopIteration:
            raise AcnetError(ACNET_ENDMULT, "Read request ended without data") from None
        finally:
            readings.close()
