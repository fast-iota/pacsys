"""
Unit tests for ConnectionPool.

Tests cover:
- Basic borrow/release
- Pool exhaustion (blocks until available)
- Concurrent access from multiple threads
- Pool close behavior
- Context manager usage
"""

import socket
import struct
import threading
import time
import pytest
from unittest import mock

from pacsys.pool import (
    ConnectionPool,
    PoolClosedError,
    PoolExhaustedError,
)
from pacsys.dpm_connection import (
    DPMConnection,
    DPMConnectionError,
)
from pacsys.dpm_protocol import OpenList_reply


def create_openlist_frame(list_id: int = 1) -> bytes:
    """Create a valid OpenList reply frame for mocking."""
    reply = OpenList_reply()
    reply.list_id = list_id
    reply_data = bytes(reply.marshal())
    return struct.pack(">I", len(reply_data)) + reply_data


# Capture the real socket class before any patching
_REAL_SOCKET_CLASS = socket.socket


def create_mock_socket(list_id: int = 1):
    """Create a mock socket that returns OpenList reply on connect."""
    frame = create_openlist_frame(list_id)
    mock_socket = mock.Mock(spec=_REAL_SOCKET_CLASS)
    mock_socket.recv.return_value = frame
    return mock_socket


class TestConnectionPoolInit:
    """Tests for ConnectionPool initialization."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"host": ""}, "host cannot be empty"),
            ({"port": 0}, "port must be between"),
            ({"port": -1}, "port must be between"),
            ({"port": 65536}, "port must be between"),
            ({"pool_size": 0}, "pool_size must be positive"),
            ({"pool_size": -1}, "pool_size must be positive"),
            ({"timeout": 0}, "timeout must be positive"),
            ({"timeout": -1.0}, "timeout must be positive"),
        ],
    )
    def test_invalid_init_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ConnectionPool(**kwargs)


class TestBasicBorrowRelease:
    """Tests for basic borrow/release operations."""

    def test_borrow_creates_connection(self):
        """Test that borrow creates a new connection when pool is empty."""
        pool = ConnectionPool(pool_size=2)

        with mock.patch("socket.socket", return_value=create_mock_socket(list_id=42)):
            conn = pool.borrow()
            assert conn is not None
            assert conn.connected
            assert conn.list_id == 42
            assert pool.in_use_count == 1
            assert pool.available_count == 0

            pool.release(conn)

    def test_release_returns_to_pool(self):
        """Test that release makes connection available again."""
        pool = ConnectionPool(pool_size=2)

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            conn = pool.borrow()
            assert pool.in_use_count == 1
            assert pool.available_count == 0

            pool.release(conn)
            assert pool.in_use_count == 0
            assert pool.available_count == 1

    def test_borrow_reuses_released_connection(self):
        """Test that borrow returns a previously released connection."""
        pool = ConnectionPool(pool_size=2)

        with mock.patch("socket.socket", return_value=create_mock_socket(list_id=99)):
            conn1 = pool.borrow()
            pool.release(conn1)

            # Second borrow should get the same connection
            conn2 = pool.borrow()
            assert conn2 is conn1
            assert pool.total_count == 1

            pool.release(conn2)

    def test_multiple_borrows_creates_multiple_connections(self):
        """Test that multiple concurrent borrows create multiple connections."""
        pool = ConnectionPool(pool_size=4)

        list_ids = iter([1, 2, 3, 4])

        def mock_socket_factory(*args, **kwargs):
            return create_mock_socket(list_id=next(list_ids))

        with mock.patch("socket.socket", side_effect=mock_socket_factory):
            conn1 = pool.borrow()
            conn2 = pool.borrow()
            conn3 = pool.borrow()

            assert pool.in_use_count == 3
            assert pool.available_count == 0
            assert pool.total_count == 3

            # All connections should be different
            assert conn1 is not conn2
            assert conn2 is not conn3
            assert conn1 is not conn3

            pool.release(conn1)
            pool.release(conn2)
            pool.release(conn3)

    def test_release_ignores_unknown_connection(self):
        """Test that releasing a connection not from the pool is safe."""
        pool = ConnectionPool()

        # Create a connection outside the pool
        with mock.patch("socket.socket", return_value=create_mock_socket()):
            external_conn = DPMConnection()
            external_conn.connect()

            # Releasing should not raise or affect pool state
            pool.release(external_conn)
            assert pool.in_use_count == 0
            assert pool.available_count == 0

            external_conn.close()

    def test_double_release_is_safe(self):
        """Test that releasing same connection twice is safe."""
        pool = ConnectionPool()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            conn = pool.borrow()
            pool.release(conn)
            pool.release(conn)  # Should not raise
            assert pool.available_count == 1


class TestPoolExhaustion:
    """Tests for pool exhaustion behavior."""

    def test_pool_exhausted_blocks(self):
        """Test that exhausted pool blocks until connection available."""
        pool = ConnectionPool(pool_size=1)
        results = []
        waiting = threading.Event()

        with mock.patch("socket.socket", return_value=create_mock_socket(list_id=1)):
            # Borrow the only connection
            conn1 = pool.borrow()

            # Try to borrow in another thread (should block)
            def borrow_in_thread():
                waiting.set()  # Signal we're about to block
                conn2 = pool.borrow(wait_timeout=5.0)
                results.append(conn2)

            thread = threading.Thread(target=borrow_in_thread)
            thread.start()

            # Wait for thread to signal it's about to block
            waiting.wait(timeout=2.0)
            assert len(results) == 0  # Still blocked

            # Release the connection
            pool.release(conn1)

            # Thread should complete
            thread.join(timeout=2.0)
            assert len(results) == 1
            assert results[0] is conn1  # Should get the released connection

            pool.release(results[0])

    def test_pool_exhausted_timeout(self):
        """Test that exhausted pool raises PoolExhaustedError on timeout."""
        pool = ConnectionPool(pool_size=1)

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            # Borrow the only connection
            conn1 = pool.borrow()

            # Second borrow should timeout
            with pytest.raises(PoolExhaustedError, match="No DPM connection"):
                pool.borrow(wait_timeout=0.1)

            pool.release(conn1)

    def test_pool_exhausted_indefinite_wait(self):
        """Test that wait_timeout=None waits indefinitely."""
        pool = ConnectionPool(pool_size=1)
        results = []
        waiting = threading.Event()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            conn1 = pool.borrow()

            def borrow_indefinite():
                waiting.set()  # Signal we're about to block
                conn2 = pool.borrow(wait_timeout=None)
                results.append(conn2)

            thread = threading.Thread(target=borrow_indefinite)
            thread.start()

            # Wait for thread to signal it's about to block
            waiting.wait(timeout=2.0)

            # Release should unblock the waiting thread
            pool.release(conn1)
            thread.join(timeout=2.0)

            assert len(results) == 1
            pool.release(results[0])


class TestConcurrentAccess:
    """Tests for concurrent access from multiple threads."""

    def test_concurrent_borrow_release(self):
        """Test concurrent borrow/release from multiple threads."""
        pool = ConnectionPool(pool_size=4)
        list_id_counter = [0]
        lock = threading.Lock()

        def mock_socket_factory(*args, **kwargs):
            with lock:
                list_id_counter[0] += 1
                return create_mock_socket(list_id=list_id_counter[0])

        errors = []
        operations = []

        def worker(worker_id: int, iterations: int):
            for i in range(iterations):
                try:
                    conn = pool.borrow(wait_timeout=5.0)
                    operations.append((worker_id, i, "borrow", conn.list_id))
                    time.sleep(0.01)  # Simulate work
                    pool.release(conn)
                    operations.append((worker_id, i, "release"))
                except Exception as e:
                    errors.append((worker_id, i, e))

        with mock.patch("socket.socket", side_effect=mock_socket_factory):
            threads = [threading.Thread(target=worker, args=(i, 10)) for i in range(8)]

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)

            assert len(errors) == 0, f"Errors occurred: {errors}"
            # Should have had many successful operations
            assert len(operations) > 100

    def test_concurrent_mixed_operations(self):
        """Test concurrent borrow, release, and discard operations."""
        pool = ConnectionPool(pool_size=4)
        list_id_counter = [0]
        lock = threading.Lock()

        def mock_socket_factory(*args, **kwargs):
            with lock:
                list_id_counter[0] += 1
                return create_mock_socket(list_id=list_id_counter[0])

        errors = []

        def borrow_release_worker(worker_id: int):
            for _ in range(5):
                try:
                    conn = pool.borrow(wait_timeout=5.0)
                    time.sleep(0.01)
                    pool.release(conn)
                except PoolClosedError:
                    pass  # Expected if pool closes
                except Exception as e:
                    errors.append((worker_id, "borrow_release", e))

        def borrow_discard_worker(worker_id: int):
            for _ in range(5):
                try:
                    conn = pool.borrow(wait_timeout=5.0)
                    time.sleep(0.01)
                    pool.discard(conn)  # Discard instead of release
                except PoolClosedError:
                    pass
                except Exception as e:
                    errors.append((worker_id, "borrow_discard", e))

        with mock.patch("socket.socket", side_effect=mock_socket_factory):
            threads = []
            for i in range(4):
                threads.append(threading.Thread(target=borrow_release_worker, args=(i,)))
                threads.append(threading.Thread(target=borrow_discard_worker, args=(i + 4,)))

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)

            assert len(errors) == 0, f"Errors occurred: {errors}"


class TestPoolClose:
    """Tests for pool close behavior."""

    def test_close_closes_all_connections(self):
        """Test that close() closes all connections in the pool."""
        pool = ConnectionPool(pool_size=4)
        mock_sockets = []

        def mock_socket_factory(*args, **kwargs):
            sock = create_mock_socket(list_id=len(mock_sockets) + 1)
            mock_sockets.append(sock)
            return sock

        with mock.patch("socket.socket", side_effect=mock_socket_factory):
            # Create some connections
            conn1 = pool.borrow()
            _ = pool.borrow()  # conn2 - in use
            pool.release(conn1)  # One available

            assert pool.available_count == 1
            assert pool.in_use_count == 1

            pool.close()

            assert pool.closed
            assert pool.available_count == 0
            assert pool.in_use_count == 0

    def test_close_rejects_new_borrows(self):
        """Test that closed pool rejects new borrow attempts."""
        pool = ConnectionPool()
        pool.close()

        with pytest.raises(PoolClosedError, match="Pool is closed"):
            pool.borrow()

    def test_close_wakes_waiting_threads(self):
        """Test that close() wakes threads waiting on borrow()."""
        pool = ConnectionPool(pool_size=1)
        errors = []
        waiting = threading.Event()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            # Exhaust the pool
            _ = pool.borrow()

            def waiting_borrow():
                waiting.set()  # Signal we're about to block
                try:
                    pool.borrow(wait_timeout=10.0)
                except PoolClosedError:
                    errors.append("PoolClosedError")
                except Exception as e:
                    errors.append(f"Unexpected: {e}")

            thread = threading.Thread(target=waiting_borrow)
            thread.start()

            # Wait for thread to signal it's about to block
            waiting.wait(timeout=2.0)

            # Close should wake the waiting thread
            pool.close()
            thread.join(timeout=2.0)

            assert errors == ["PoolClosedError"]

    def test_close_multiple_times_safe(self):
        """Test that close() can be called multiple times safely."""
        pool = ConnectionPool()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            pool.borrow()

        pool.close()
        pool.close()  # Should not raise
        pool.close()  # Should not raise

        assert pool.closed

    def test_release_after_close_closes_connection(self):
        """Test that releasing a connection after pool close closes it."""
        pool = ConnectionPool()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            conn = pool.borrow()
            pool.close()

            # Connection should be closed, release is no-op
            pool.release(conn)
            assert pool.available_count == 0


class TestContextManager:
    """Tests for context manager usage."""

    def test_pool_context_manager(self):
        """Test pool as context manager."""
        with mock.patch("socket.socket", return_value=create_mock_socket()):
            with ConnectionPool() as pool:
                conn = pool.borrow()
                assert conn.connected
                pool.release(conn)

            assert pool.closed

    def test_connection_context_manager(self):
        """Test connection() context manager."""
        pool = ConnectionPool()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            with pool.connection() as conn:
                assert conn.connected
                assert pool.in_use_count == 1

            # Connection should be released
            assert pool.in_use_count == 0
            assert pool.available_count == 1

        pool.close()

    def test_connection_context_manager_on_exception(self):
        """Test that connection is released even on exception."""
        pool = ConnectionPool()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            with pytest.raises(ValueError):
                with pool.connection() as _:
                    assert pool.in_use_count == 1
                    raise ValueError("test error")

            # Connection should still be released
            assert pool.in_use_count == 0
            assert pool.available_count == 1

        pool.close()

    def test_connection_context_manager_discards_on_broken_pipe(self):
        """Test that broken connections are discarded, not returned to pool."""
        pool = ConnectionPool()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            with pytest.raises(BrokenPipeError):
                with pool.connection() as _:
                    assert pool.in_use_count == 1
                    raise BrokenPipeError("pipe broke")

            assert pool.in_use_count == 0
            assert pool.available_count == 0  # discarded, not released

        pool.close()

    def test_connection_context_manager_discards_on_connection_reset(self):
        """Test that reset connections are discarded, not returned to pool."""
        pool = ConnectionPool()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            with pytest.raises(ConnectionResetError):
                with pool.connection() as _:
                    raise ConnectionResetError("reset")

            assert pool.in_use_count == 0
            assert pool.available_count == 0

        pool.close()

    def test_connection_context_manager_discards_on_oserror(self):
        """Test that OSError causes discard (covers socket.error etc)."""
        pool = ConnectionPool()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            with pytest.raises(OSError):
                with pool.connection() as _:
                    raise OSError("socket gone")

            assert pool.in_use_count == 0
            assert pool.available_count == 0

        pool.close()

    def test_nested_context_managers(self):
        """Test nested pool and connection context managers."""
        list_id_counter = [0]

        def mock_socket_factory(*args, **kwargs):
            list_id_counter[0] += 1
            return create_mock_socket(list_id=list_id_counter[0])

        with mock.patch("socket.socket", side_effect=mock_socket_factory):
            with ConnectionPool(pool_size=2) as pool:
                with pool.connection() as conn1:
                    with pool.connection() as conn2:
                        assert pool.in_use_count == 2
                        assert conn1 is not conn2

                    assert pool.in_use_count == 1

                assert pool.in_use_count == 0

            assert pool.closed


class TestDiscard:
    """Tests for discard operation."""

    def test_discard_removes_connection(self):
        """Test that discard removes connection from pool."""
        pool = ConnectionPool(pool_size=2)

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            conn = pool.borrow()
            assert pool.in_use_count == 1

            pool.discard(conn)
            assert pool.in_use_count == 0
            assert pool.available_count == 0

    def test_discard_available_connection(self):
        """Test discarding an available (not borrowed) connection."""
        pool = ConnectionPool(pool_size=2)

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            conn = pool.borrow()
            pool.release(conn)
            assert pool.available_count == 1

            pool.discard(conn)
            assert pool.available_count == 0

    def test_discard_frees_slot_for_new_connection(self):
        """Test that discard frees a slot for new connections."""
        pool = ConnectionPool(pool_size=1)
        call_count = [0]

        def mock_socket_factory(*args, **kwargs):
            call_count[0] += 1
            return create_mock_socket(list_id=call_count[0])

        with mock.patch("socket.socket", side_effect=mock_socket_factory):
            conn1 = pool.borrow()
            assert conn1.list_id == 1

            pool.discard(conn1)

            # Should be able to borrow a new connection
            conn2 = pool.borrow()
            assert conn2.list_id == 2
            assert conn2 is not conn1

            pool.release(conn2)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_release_dead_connection(self):
        """Test that releasing a dead connection does not return it to pool."""
        pool = ConnectionPool()

        with mock.patch("socket.socket", return_value=create_mock_socket()):
            conn = pool.borrow()
            conn.close()  # Make connection dead

            pool.release(conn)
            # Dead connection should not be returned to available pool
            assert pool.available_count == 0

    def test_borrow_when_creation_fails(self):
        """Test borrow behavior when connection creation fails."""
        pool = ConnectionPool(pool_size=2)

        with mock.patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.return_value.connect.side_effect = socket.error("Connection refused")

            with pytest.raises(DPMConnectionError, match="Failed to connect"):
                pool.borrow()

            # Pool should still be usable
            assert pool.in_use_count == 0
            assert pool.available_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
