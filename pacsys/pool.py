"""
Thread-safe connection pool for DPM connections.

Provides borrow/release semantics. Supports context manager usage for safe
resource management.

NOTE: DPM is a stateful protocol where the list_id is tied to the specific TCP
connection. Transparent retry-on-stale is NOT safe - if a connection fails, the
caller must rebuild their request with the new list_id from the fresh connection.
The pool returns raw DPMConnection objects; callers handle reconnection at the
operation level.

Usage:
    # Create pool
    pool = ConnectionPool(host="acsys-proxy.fnal.gov", port=6802)

    # Context manager (preferred)
    with pool.connection() as conn:
        # use conn...
        pass

    # Cleanup
    pool.close()
"""

import logging
import threading
from contextlib import contextmanager
from typing import Optional

from pacsys.dpm_connection import DPMConnection

logger = logging.getLogger(__name__)

# Default pool settings
DEFAULT_POOL_SIZE = 4
DEFAULT_WAIT_TIMEOUT = 30.0


class PoolClosedError(Exception):
    """Raised when attempting to borrow from a closed pool."""

    def __init__(self, message: str = "Pool is closed"):
        self.message = message
        super().__init__(message)


class PoolExhaustedError(Exception):
    """Raised when pool is exhausted and wait times out."""

    def __init__(self, message: str = "Pool exhausted, no connections available"):
        self.message = message
        super().__init__(message)


class ConnectionPool:
    """
    Thread-safe connection pool for DPM connections.

    Manages a pool of DPMConnection objects with borrow/release semantics.
    Connections are created lazily on first borrow. Stale connections are
    handled via retry-on-failure pattern.

    Thread Safety:
        All methods are thread-safe. Multiple threads can borrow and release
        connections concurrently.

    Lifecycle:
        - Pool starts empty, connections created on demand up to pool_size
        - Use close() or context manager to release all resources
        - After close(), pool cannot be reused

    Example:
        with ConnectionPool() as pool:
            with pool.connection() as conn:
                conn.send_message(msg)
                reply = conn.recv_message()
    """

    def __init__(
        self,
        host: str = "acsys-proxy.fnal.gov",
        port: int = 6802,
        pool_size: int = DEFAULT_POOL_SIZE,
        timeout: float = 5.0,
    ):
        """
        Initialize connection pool.

        Args:
            host: DPM server hostname
            port: DPM server port
            pool_size: Maximum number of connections in the pool
            timeout: Default socket timeout for connections (seconds)

        Raises:
            ValueError: If parameters are invalid
        """
        if not host:
            raise ValueError("host cannot be empty")
        if port <= 0 or port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {port}")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        self._host = host
        self._port = port
        self._pool_size = pool_size
        self._timeout = timeout

        # Pool state
        self._available: list[DPMConnection] = []
        self._in_use: set[DPMConnection] = set()
        self._pending_creates = 0  # Slots reserved for connections being created
        self._closed = False

        # Synchronization
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        logger.debug(f"ConnectionPool created: host={host}, port={port}, pool_size={pool_size}, timeout={timeout}")

    @property
    def host(self) -> str:
        """DPM server hostname."""
        return self._host

    @property
    def port(self) -> int:
        """DPM server port."""
        return self._port

    @property
    def pool_size(self) -> int:
        """Maximum number of connections allowed."""
        return self._pool_size

    @property
    def timeout(self) -> float:
        """Default socket timeout for connections."""
        return self._timeout

    @property
    def available_count(self) -> int:
        """Number of available connections in pool."""
        with self._lock:
            return len(self._available)

    @property
    def in_use_count(self) -> int:
        """Number of connections currently borrowed."""
        with self._lock:
            return len(self._in_use)

    @property
    def total_count(self) -> int:
        """Total number of connections (available + in use + pending)."""
        with self._lock:
            return len(self._available) + len(self._in_use) + self._pending_creates

    @property
    def closed(self) -> bool:
        """True if pool has been closed."""
        with self._lock:
            return self._closed

    def borrow(self, wait_timeout: Optional[float] = DEFAULT_WAIT_TIMEOUT) -> DPMConnection:
        """
        Borrow a connection from the pool.

        Returns an available connection or creates a new one if the pool
        is not at capacity. If the pool is exhausted (all connections in use),
        blocks until a connection becomes available.

        Args:
            wait_timeout: Maximum time to wait if pool is exhausted (seconds).
                          None means wait indefinitely. Default is 30 seconds.

        Returns:
            A DPMConnection ready for use

        Raises:
            PoolClosedError: If pool is closed
            PoolExhaustedError: If wait_timeout expires before a connection
                                becomes available
            DPMConnectionError: If connection creation fails
        """
        need_create = False

        with self._condition:
            if self._closed:
                raise PoolClosedError()

            # Try to get an available connection or reserve a slot for creation
            while not self._available:
                # Can we create a new connection? Count includes pending creates
                active_count = len(self._in_use) + self._pending_creates
                if active_count < self._pool_size:
                    # Reserve a slot for connection creation (done outside lock)
                    self._pending_creates += 1
                    need_create = True
                    break

                # Pool exhausted, wait for a connection to be released
                logger.debug("Pool exhausted, waiting for available connection")
                if not self._condition.wait(timeout=wait_timeout):
                    raise PoolExhaustedError(
                        f"No DPM connection to {self._host}:{self._port} available after {wait_timeout}s "
                        f"(pool_size={self._pool_size}, all in use)"
                    )

                # Check if pool was closed while waiting
                if self._closed:
                    raise PoolClosedError()

            if not need_create:
                # Get connection from available pool
                conn = self._available.pop()
                self._in_use.add(conn)
                logger.debug(f"Borrowed connection, available={len(self._available)}, in_use={len(self._in_use)}")
                return conn

        # Create connection OUTSIDE the lock to avoid blocking other threads
        try:
            conn = self._create_connection()
        except Exception:
            # Release the reservation on failure
            with self._condition:
                self._pending_creates -= 1
                self._condition.notify()  # Wake waiting threads
            raise

        # Add the new connection to in_use set
        with self._condition:
            if self._closed:
                # Pool was closed while we were creating the connection
                self._pending_creates -= 1
                self._close_connection(conn)
                raise PoolClosedError()

            self._pending_creates -= 1
            self._in_use.add(conn)
            total = len(self._available) + len(self._in_use)
            logger.debug(f"Created new connection, total={total}")
            return conn

    def _create_connection(self) -> DPMConnection:
        """
        Create a new DPM connection.

        Note: This method performs blocking I/O and should NOT be called
        while holding the pool lock. Callers must reserve a slot first.

        Returns:
            A connected DPMConnection

        Raises:
            DPMConnectionError: If connection fails
        """
        conn = DPMConnection(
            host=self._host,
            port=self._port,
            timeout=self._timeout,
        )
        conn.connect()
        return conn

    def release(self, conn: DPMConnection) -> None:
        """
        Return a connection to the pool.

        The connection is made available for future borrows. If the pool
        is closed, the connection is closed instead.

        Args:
            conn: The connection to release (must have been borrowed)

        Note:
            Safe to call with a connection not from this pool (no-op).
            Safe to call multiple times with same connection (no-op after first).
        """
        with self._condition:
            # Ignore if connection not from this pool
            if conn not in self._in_use:
                logger.debug("Release called for connection not in use (ignoring)")
                return

            self._in_use.discard(conn)

            if self._closed:
                # Pool is closed, close the connection
                self._close_connection(conn)
            elif conn.connected:
                # Return to available pool
                self._available.append(conn)
                logger.debug(f"Released connection, available={len(self._available)}, in_use={len(self._in_use)}")
            else:
                # Connection is dead, don't return to pool
                logger.debug("Released dead connection (discarded)")

            # Notify waiters that a connection may be available
            self._condition.notify()

    def discard(self, conn: DPMConnection) -> None:
        """
        Discard a connection without returning it to the pool.

        Use this when a connection is known to be bad (e.g., after
        a BrokenPipe error). The connection is closed and removed.

        Args:
            conn: The connection to discard
        """
        with self._condition:
            if conn in self._in_use:
                self._in_use.discard(conn)
                self._close_connection(conn)
                logger.debug(f"Discarded connection, in_use={len(self._in_use)}")
                # Notify waiters - a slot is now available for new connection
                self._condition.notify()
            elif conn in self._available:
                self._available.remove(conn)
                self._close_connection(conn)
                logger.debug("Discarded available connection")

    def _close_connection(self, conn: DPMConnection) -> None:
        """Close a connection safely."""
        try:
            conn.close()
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")

    @contextmanager
    def connection(self, wait_timeout: Optional[float] = DEFAULT_WAIT_TIMEOUT):
        """
        Context manager for borrowing a connection.

        Automatically releases the connection when the context exits.
        On connection errors (BrokenPipe, ConnectionReset, etc.), the
        connection is discarded instead of returned to the pool.

        Args:
            wait_timeout: Maximum time to wait if pool is exhausted

        Yields:
            A DPMConnection ready for use

        Example:
            with pool.connection() as conn:
                conn.send_message(msg)
                reply = conn.recv_message()
        """
        conn = self.borrow(wait_timeout=wait_timeout)
        broken = False
        try:
            yield conn
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            broken = True
            raise
        finally:
            if broken:
                self.discard(conn)
            else:
                self.release(conn)

    def close(self) -> None:
        """
        Close the pool and all connections.

        After close(), the pool cannot be reused. Any threads waiting
        on borrow() will receive PoolClosedError.

        Safe to call multiple times.
        """
        with self._condition:
            if self._closed:
                return

            self._closed = True

            # Close all available connections
            for conn in self._available:
                self._close_connection(conn)
            self._available.clear()

            # Close all in-use connections
            for conn in list(self._in_use):
                self._close_connection(conn)
            self._in_use.clear()

            # Wake up all waiters so they get PoolClosedError
            self._condition.notify_all()

            logger.info("ConnectionPool closed")

    def __enter__(self) -> "ConnectionPool":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager - closes the pool."""
        self.close()
        return False

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"ConnectionPool({self._host}:{self._port}, pool_size={self._pool_size}, {status})"
