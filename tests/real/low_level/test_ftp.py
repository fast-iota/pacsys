"""
Integration tests for FTPMAN protocol (Fast Time Plot & Snapshot).

Tests against live ACNET front-end nodes. Requires acnetd TCP connection.

Run with: pytest tests/real/low_level/test_ftp.py -v -s -o "addopts="
"""

import time

import pytest

from pacsys.acnet.errors import (
    FTP_COLLECTING,
    FTP_PEND,
    FTP_WAIT_DELAY,
    FTP_WAIT_EVENT,
    ftp_status_message,
    parse_error,
)
from pacsys.acnet.ftp import (
    FTPClient,
    FTPDevice,
    SnapshotState,
    get_ftp_class_info,
    get_snap_class_info,
)
from tests.real.devices import requires_acnet_tcp

# M:OUTTMP on MUONFE -- known FTP class 16 (C290), snap class 13 (C290)
MOUTTMP_DI = 27235
MOUTTMP_PI = 12
MOUTTMP_SSDN = b"\x00\x00B\x00?!\x00\x00"
MUONFE_NODE = "MUONFE"
MUONFE_EXPECTED_ADDRESS = (11 << 8) | 202  # 3018


def make_ftp_device(di=MOUTTMP_DI, pi=MOUTTMP_PI, ssdn=MOUTTMP_SSDN, **kwargs):
    """Create an FTPDevice with configurable SSDN (defaults to M:OUTTMP)."""
    return FTPDevice(di=di, pi=pi, ssdn=ssdn, **kwargs)


@pytest.fixture(autouse=True)
def _settle():
    """Brief pause between tests to let acnetd connections settle."""
    yield
    time.sleep(0.5)


@pytest.fixture
def mouttmp():
    return make_ftp_device()


@requires_acnet_tcp
class TestFTPClassCodes:
    """Test FTP class code queries against live front-ends."""

    def test_mouttmp_class_codes(self, acnet_tcp_connection, mouttmp):
        """M:OUTTMP on MUONFE should have ftp_class=16, snap_class=13."""
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)
        assert node == MUONFE_EXPECTED_ADDRESS

        ftp = FTPClient(conn)
        result = ftp.get_class_codes(node=node, device=mouttmp)

        print(f"\n  M:OUTTMP class codes: ftp={result.ftp}, snap={result.snap}, error={result.error}")
        assert result.error == 0
        assert result.ftp == 16
        assert result.snap == 13

        # Verify registry info matches
        ftp_info = get_ftp_class_info(result.ftp)
        assert ftp_info is not None
        assert ftp_info.max_rate == 1440

        snap_info = get_snap_class_info(result.snap)
        assert snap_info is not None
        assert snap_info.max_rate == 90000


@requires_acnet_tcp
class TestFTPContinuous:
    """Test continuous FTP streaming against live front-ends."""

    def test_stream_mouttmp(self, acnet_tcp_connection, mouttmp):
        """Stream M:OUTTMP for a short period and verify data arrives."""
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)

        ftp = FTPClient(conn)

        with ftp.start_continuous(
            node=node,
            devices=[mouttmp],
            rate_hz=1440,
            return_period=4,
            timeout=10.0,
        ) as stream:
            assert stream.setup_statuses == [0]

            total_points = 0
            batches = 0
            deadline = time.time() + 3.0

            for batch in stream.readings(timeout=1.0):
                for di, points in batch.items():
                    total_points += len(points)
                batches += 1

                if time.time() > deadline:
                    stream.stop()
                    break

        print(f"\n  Received {total_points} points in {batches} batches")
        # At 1440 Hz over ~3s we expect hundreds of points
        # But first few replies may have 0 points, so be lenient
        assert total_points > 10, f"Expected >10 data points, got {total_points}"

    def test_stream_stop(self, acnet_tcp_connection, mouttmp):
        """Verify clean cancellation of FTP stream."""
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)

        ftp = FTPClient(conn)

        stream = ftp.start_continuous(
            node=node,
            devices=[mouttmp],
            rate_hz=720,
            timeout=10.0,
        )

        # Get at least one batch
        for batch in stream.readings(timeout=2.0):
            break

        stream.stop()
        assert stream.stopped
        print("\n  Stream stopped cleanly")


@requires_acnet_tcp
class TestSnapClassCodes:
    """Test snapshot class code queries against live front-ends."""

    def test_mouttmp_snap_class(self, acnet_tcp_connection, mouttmp):
        """M:OUTTMP on MUONFE should have snap_class=13 (C290 MADC)."""
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)

        ftp = FTPClient(conn)
        result = ftp.get_class_codes(node=node, device=mouttmp)

        print(f"\n  Snap class code: {result.snap}, error={result.error}")
        assert result.error == 0
        assert result.snap == 13

        snap_info = get_snap_class_info(result.snap)
        assert snap_info is not None
        assert snap_info.has_timestamps is True
        assert snap_info.max_rate == 90000
        assert snap_info.max_points == 2048


@requires_acnet_tcp
class TestSnapshotCapture:
    """Test snapshot arm, collection, and retrieval against live front-ends."""

    # Expected per-device statuses after snapshot setup (positive = informational)
    _ACCEPTABLE_SETUP_STATUSES = {FTP_PEND, FTP_WAIT_EVENT, FTP_WAIT_DELAY, FTP_COLLECTING}

    def test_immediate_post_trigger(self, acnet_tcp_connection, mouttmp):
        """Arm immediately, collect 100 points post-trigger, retrieve."""
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)

        ftp = FTPClient(conn)

        with ftp.start_snapshot(
            node=node,
            devices=[mouttmp],
            rate_hz=1440,
            num_points=100,
            # arm_source=2 (clock) with all-0xFF events = immediate arm,
            # matching Java SnapShotPool (never sends ARM_IMMEDIATELY=1)
            plot_mode=2,  # post-trigger
            timeout=10.0,
        ) as snap:
            reply = snap.setup_reply
            assert len(reply.per_device_errors) == 1

            # Validate per-device status
            dev_status = reply.per_device_errors[0]
            fac, err = parse_error(dev_status)
            msg = ftp_status_message(dev_status)
            print(
                f"\n  Setup reply: rate={reply.sample_rate_hz} Hz, "
                f"npts={reply.num_points}, dev_status=[{fac} {err}] ({msg})"
            )

            assert dev_status >= 0, f"Device setup failed: [{fac} {err}] ({msg})"
            assert dev_status == 0 or dev_status in self._ACCEPTABLE_SETUP_STATUSES, (
                f"Unexpected device status: [{fac} {err}] ({msg})"
            )

            # Verify setup reply fields
            assert reply.sample_rate_hz == 1440
            assert reply.num_points == 100

            # With immediate arm at 1440 Hz, 100 points takes ~70ms.
            # Wait generously for collection to finish.
            time.sleep(2.0)

            # Retrieve data (snap class 13 has timestamps)
            points = snap.retrieve(
                device_index=0,
                num_points=100,
                has_timestamps=True,
                timeout=5.0,
            )

            print(f"  Retrieved {len(points)} points")
            assert len(points) == 100, f"Expected 100 points, got {len(points)}"

            # Verify points have plausible timestamps (monotonically increasing)
            for i, pt in enumerate(points[:5]):
                print(f"    ts={pt.timestamp_us} us, raw={pt.raw_value}")
            for i in range(1, len(points)):
                assert points[i].timestamp_us >= points[i - 1].timestamp_us, (
                    f"Timestamps not monotonic at index {i}: {points[i - 1].timestamp_us} -> {points[i].timestamp_us}"
                )

    def test_snapshot_cancel(self, acnet_tcp_connection, mouttmp):
        """Verify clean cancellation of a snapshot."""
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)

        ftp = FTPClient(conn)

        snap = ftp.start_snapshot(
            node=node,
            devices=[mouttmp],
            rate_hz=720,
            num_points=100,
            plot_mode=2,
            timeout=10.0,
        )
        assert len(snap.setup_reply.per_device_errors) == 1

        snap.cancel()
        assert snap._cancelled
        print("\n  Snapshot cancelled cleanly")


@requires_acnet_tcp
class TestSnapshotStateMachine:
    """Verify the snapshot state machine tracks FE status updates end-to-end."""

    def test_clock_event_02_state_transitions(self, acnet_tcp_connection, mouttmp):
        """Arm on TCLK 0x02 (~5 s cycle), 50 Hz for 2 s, verify state transitions.

        Expected progression:
          PENDING → WAIT_EVENT → (WAIT_DELAY) → COLLECTING → READY

        Not every state is guaranteed to be visible (the FE only sends
        updates every ~1-2 s and the event may fire between polls), but
        we must see at least one intermediate state before READY.
        """
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)
        ftp = FTPClient(conn)

        # Each byte is a literal clock event number (0xFF = unused slot)
        arm_events = b"\x02" + b"\xff" * 7

        with ftp.start_snapshot(
            node=node,
            devices=[mouttmp],
            rate_hz=50,
            num_points=100,  # 50 Hz × 2 s
            arm_source=2,  # clock event
            plot_mode=2,  # post-trigger
            arm_events=arm_events,
            snap_class_code=13,  # C290 MADC (M:OUTTMP on MUONFE)
            timeout=10.0,
        ) as snap:
            # --- Track transitions ---
            observed = []
            prev = None

            def record(state):
                nonlocal prev
                if state != prev:
                    observed.append((state, time.monotonic()))
                    prev = state

            record(snap.state)

            # Poll at 200 ms until ready or 15 s hard deadline.
            # The worst case: we arrive just after an 0x02 event and must
            # wait almost a full supercycle (~5.3 s) + 2 s collection.
            deadline = time.monotonic() + 15.0
            while not snap.is_ready and time.monotonic() < deadline:
                time.sleep(0.2)
                record(snap.state)

            # Use the wait() API as a final check (should return immediately
            # if the poll loop already saw READY; raises on device error).
            ready = snap.wait(timeout=2.0)
            record(snap.state)

            # --- Print timeline ---
            t0 = observed[0][1]
            print()
            for state, ts in observed:
                print(f"  +{ts - t0:6.2f}s  {state.name}")

            # --- Assertions ---
            assert ready, f"Snapshot did not become ready. Final: {snap.state.name}"

            state_names = [s.name for s, _ in observed]
            assert state_names[-1] == "READY"

            # Must have seen at least one state before READY (the FE always
            # goes through PENDING / WAIT_EVENT before data is collected).
            assert len(observed) >= 2, f"Expected at least one intermediate state before READY, saw only: {state_names}"

            # WAIT_EVENT is the most expected intermediate state when arming
            # on a clock event.  Accept PENDING too (setup ack arrives
            # before the first status update).
            intermediates = {s for s, _ in observed} - {SnapshotState.READY}
            assert intermediates, f"No intermediate states observed, only: {state_names}"

            # --- Retrieve and verify data ---
            print(f"  Setup: rate={snap.setup_reply.sample_rate_hz} Hz, npts={snap.setup_reply.num_points}")

            # Retrieve using the actual num_points from the setup reply
            # (FE may return 0 if asked for more than captured)
            points = snap.retrieve(
                device_index=0,
                num_points=snap.setup_reply.num_points,
                timeout=5.0,
            )

            print(f"  Retrieved {len(points)} points")
            for pt in points[:5]:
                print(f"    ts={pt.timestamp_us} µs  raw={pt.raw_value}")

            # num_points minus 1 skipped metadata point
            expected_min = snap.setup_reply.num_points - 10
            assert len(points) >= expected_min, f"Expected ~{snap.setup_reply.num_points - 1} points, got {len(points)}"

            # Timestamps should be monotonically non-decreasing
            for i in range(1, len(points)):
                assert points[i].timestamp_us >= points[i - 1].timestamp_us, (
                    f"Timestamps not monotonic at [{i}]: {points[i - 1].timestamp_us} → {points[i].timestamp_us}"
                )


@requires_acnet_tcp
class TestSnapshotRestart:
    """Test snapshot restart/re-arm cycle (typecode 5)."""

    def test_restart_post_trigger(self, acnet_tcp_connection, mouttmp):
        """Arm immediately, retrieve, restart, retrieve again.

        Exercises the full capture-restart-capture cycle:
          setup (typecode 7) → wait → retrieve (typecode 8)
          → restart (typecode 5) → wait → retrieve again
        """
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)
        ftp = FTPClient(conn)

        with ftp.start_snapshot(
            node=node,
            devices=[mouttmp],
            rate_hz=1440,
            num_points=100,
            snap_class_code=13,
            timeout=10.0,
        ) as snap:
            # --- First capture ---
            assert snap.wait(timeout=10.0), f"First capture not ready: {snap.state.name}"
            points1 = snap.retrieve(device_index=0, timeout=5.0)
            print(f"\n  Capture 1: {len(points1)} points")
            assert len(points1) >= 90

            # --- Restart (re-arm) ---
            snap.restart(timeout=5.0)
            assert snap.state == SnapshotState.PENDING, f"Expected PENDING after restart, got {snap.state.name}"
            assert not snap.is_ready

            # --- Second capture ---
            assert snap.wait(timeout=10.0), f"Second capture not ready: {snap.state.name}"
            points2 = snap.retrieve(device_index=0, timeout=5.0)
            print(f"  Capture 2: {len(points2)} points")
            assert len(points2) >= 90

            # Both captures should have data but from different moments
            # (timestamps will differ because the FE re-armed)
            print(f"  Cap1 first ts={points1[0].timestamp_us}, Cap2 first ts={points2[0].timestamp_us}")


@requires_acnet_tcp
class TestSnapshotPreTrigger:
    """Test pre-trigger (plot_mode=3) snapshot capture."""

    def test_pre_trigger_clock_event(self, acnet_tcp_connection, mouttmp):
        """Pre-trigger capture: most data collected before the arm event.

        plot_mode=3 (DIGITIZE_BEFORE): the FE continuously fills a circular
        buffer.  When the arm event fires, it collects arm_delay more points
        then stops.  The buffer contains data from before the event.

        Java SnapShotPool sets arm_delay = pointsAfterArmEvent for pre-trigger.
        """
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)
        ftp = FTPClient(conn)

        # Arm on TCLK 0x02, 50 Hz, 100 points total.
        # arm_delay=25 means collect 25 points after arm event; 75 before.
        arm_events = b"\x02" + b"\xff" * 7
        with ftp.start_snapshot(
            node=node,
            devices=[mouttmp],
            rate_hz=50,
            num_points=100,
            arm_source=2,
            plot_mode=3,  # pre-trigger
            arm_delay=25,  # points after arm event (not microseconds)
            arm_events=arm_events,
            snap_class_code=13,
            timeout=10.0,
        ) as snap:
            # Wait for collection -- may need a full supercycle (~5s) + buffer time
            ready = snap.wait(timeout=20.0)
            print(f"\n  State: {snap.state.name}, ready={ready}")
            assert ready, f"Pre-trigger snapshot not ready: {snap.state.name}"

            reply = snap.setup_reply
            print(f"  Setup: rate={reply.sample_rate_hz} Hz, npts={reply.num_points}")

            points = snap.retrieve(device_index=0, timeout=5.0)
            print(f"  Retrieved {len(points)} points")
            for pt in points[:3]:
                print(f"    ts={pt.timestamp_us} µs  raw={pt.raw_value}")

            # Pre-trigger: count depends on how full the circular buffer was
            # when the arm event fired.  We're guaranteed at least arm_delay
            # (25) post-arm points; the rest depends on timing.
            assert len(points) >= 25, f"Expected at least arm_delay points, got {len(points)}"

            # Timestamps should be monotonically non-decreasing, but
            # pre-trigger data arming on 0x02 spans the supercycle boundary
            # (~5s wrap), so allow exactly one timestamp wrap.
            wraps = 0
            for i in range(1, len(points)):
                if points[i].timestamp_us < points[i - 1].timestamp_us:
                    wraps += 1
                    continue
                assert points[i].timestamp_us >= points[i - 1].timestamp_us, (
                    f"Timestamps not monotonic at [{i}]: {points[i - 1].timestamp_us} → {points[i].timestamp_us}"
                )
            assert wraps <= 1, f"Expected at most 1 supercycle wrap, got {wraps}"


@requires_acnet_tcp
class TestSnapshotSequentialRetrieval:
    """Test multi-chunk sequential retrieval (point_number=-1)."""

    def test_sequential_2048_points(self, acnet_tcp_connection, mouttmp):
        """Capture 2048 points and retrieve in 512-point chunks.

        Class 13 (C290 MADC) has retrieval_max=512, so a full 2048-point
        capture requires 4+ sequential retrieve calls with point_number=-1.
        The first chunk has 1 metadata point (skip_first_point); subsequent
        chunks do not.
        """
        conn = acnet_tcp_connection
        node = conn.get_node(MUONFE_NODE)
        ftp = FTPClient(conn)

        with ftp.start_snapshot(
            node=node,
            devices=[mouttmp],
            rate_hz=1440,
            num_points=2048,
            snap_class_code=13,
            timeout=10.0,
        ) as snap:
            assert snap.wait(timeout=10.0), f"Snapshot not ready: {snap.state.name}"

            # Retrieve in 512-point chunks using sequential access
            all_points = []
            chunk_num = 0
            while True:
                chunk_num += 1
                # First chunk: skip metadata point. Subsequent: don't skip.
                skip = chunk_num == 1
                points = snap.retrieve(
                    device_index=0,
                    num_points=512,
                    point_number=-1,  # sequential
                    skip_first_point=skip,
                    timeout=5.0,
                )
                print(f"  Chunk {chunk_num}: {len(points)} points (skip_first={skip})")
                if not points:
                    break
                all_points.extend(points)
                # Safety: don't loop forever
                if chunk_num >= 10:
                    break

            print(f"\n  Total: {len(all_points)} points in {chunk_num} chunks")
            # 2048 points minus 1 skipped metadata = 2047
            assert len(all_points) >= 2040, f"Expected ~2047 points, got {len(all_points)}"

            # Verify timestamp monotonicity across chunks, allowing one
            # TCLK supercycle wrap (~5s boundary) since 2048 points at
            # 1440 Hz spans ~1.4s which can cross a 0x02 event.
            wraps = 0
            for i in range(1, len(all_points)):
                if all_points[i].timestamp_us < all_points[i - 1].timestamp_us:
                    wraps += 1
                    continue
                assert all_points[i].timestamp_us >= all_points[i - 1].timestamp_us, (
                    f"Timestamps not monotonic at [{i}]: "
                    f"{all_points[i - 1].timestamp_us} → {all_points[i].timestamp_us}"
                )
            assert wraps <= 1, f"Expected at most 1 supercycle wrap, got {wraps}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
