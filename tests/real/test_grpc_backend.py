"""
Integration tests for GRPCBackend (gRPC-specific behavior).

Common read/error/value-type tests are in test_backend_shared.py.
Common streaming tests are in test_backend_shared.py.

This file contains gRPC-specific tests:
- Multiple/concurrent reads

Requires:
- gRPC server accessible at localhost:23456 (tunnel to dce08.fnal.gov:50051)
- grpcio package installed

Run with: pytest tests/real/test_grpc_backend.py -v -s
"""

import time
import threading

import pytest

try:
    from pacsys.backends.grpc_backend import GRPCBackend, GRPC_AVAILABLE
except ImportError:
    GRPC_AVAILABLE = False

if not GRPC_AVAILABLE:
    pytest.skip("grpc not available", allow_module_level=True)

from .devices import (
    requires_grpc,
    assert_fast_response,
    TIMEOUT_READ,
    TIMEOUT_THREAD_JOIN,
)


# =============================================================================
# Multiple Reads Tests
# =============================================================================


@requires_grpc
class TestGRPCBackendMultipleReads:
    """Tests for sequential and concurrent reads."""

    def test_multiple_sequential_reads(self, grpc_backend):
        """Multiple reads work correctly in sequence."""
        total_start = time.time()
        for i in range(3):
            start = time.time()
            value = grpc_backend.read("M:OUTTMP", timeout=TIMEOUT_READ)
            elapsed = time.time() - start
            assert_fast_response(elapsed, f"read #{i + 1}")
            assert isinstance(value, (int, float))
        total_elapsed = time.time() - total_start
        print(f"\n  3 sequential reads completed in {total_elapsed * 1000:.0f}ms")

    def test_concurrent_reads(self):
        """Concurrent reads work correctly."""
        results = []
        errors = []
        timings = []

        def do_read(backend, device):
            try:
                start = time.time()
                results.append(backend.read(device, timeout=TIMEOUT_READ))
                timings.append(time.time() - start)
            except Exception as e:
                errors.append(e)

        start = time.time()
        with GRPCBackend() as backend:
            threads = [threading.Thread(target=do_read, args=(backend, "M:OUTTMP")) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=TIMEOUT_THREAD_JOIN)
        elapsed = time.time() - start

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 4

        for i, t in enumerate(timings):
            assert_fast_response(t, f"concurrent read #{i + 1}")
        print(
            f"\n  4 concurrent reads completed in {elapsed * 1000:.0f}ms (max individual: {max(timings) * 1000:.0f}ms)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
