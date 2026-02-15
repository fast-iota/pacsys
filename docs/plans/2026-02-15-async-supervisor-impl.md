# Async Supervisor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `SupervisedServer` accept `AsyncBackend` and call its methods directly (no `asyncio.to_thread`), while retaining full backward compatibility with sync `Backend`.

**Architecture:** Dual-mode runtime check. `_DAQServicer` sets `self._async` flag at init. Each RPC branches at backend call sites only; policies, logging, conversion, and error handling are shared. New `AsyncFakeBackend` enables testing the async path.

**Tech Stack:** Python asyncio, gRPC aio, `pacsys.aio.AsyncBackend`, `pacsys.aio.AsyncSubscriptionHandle`

---

### Task 1: AsyncFakeBackend

Implement `AsyncFakeBackend` in `pacsys/testing.py` so we have an async backend for TDD.

**Files:**
- Modify: `pacsys/testing.py` (add class after `FakeBackend`, update `__all__`)

**Step 1: Write the failing test**

Create `tests/test_async_fake_backend.py`:

```python
"""Tests for AsyncFakeBackend."""

import asyncio
import pytest
from pacsys.testing import AsyncFakeBackend


class TestAsyncFakeBackendRead:
    def test_read_value(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        reading = asyncio.run(fb.get("M:OUTTMP"))
        assert reading.ok
        assert reading.value == pytest.approx(72.5)

    def test_read_error(self):
        fb = AsyncFakeBackend()
        fb.set_error("M:BADDEV", -42, "Device not found")
        reading = asyncio.run(fb.get("M:BADDEV"))
        assert reading.is_error
        assert reading.error_code == -42

    def test_get_many(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        fb.set_reading("G:AMANDA", 42.0)
        readings = asyncio.run(fb.get_many(["M:OUTTMP", "G:AMANDA"]))
        assert len(readings) == 2
        assert readings[0].value == pytest.approx(72.5)
        assert readings[1].value == pytest.approx(42.0)


class TestAsyncFakeBackendWrite:
    def test_write(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        result = asyncio.run(fb.write("M:OUTTMP", 80.0))
        assert result.success
        assert fb.was_written("M:OUTTMP")

    def test_write_many(self):
        fb = AsyncFakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)
        fb.set_reading("G:AMANDA", 42.0)
        results = asyncio.run(fb.write_many([("M:OUTTMP", 80.0), ("G:AMANDA", 50.0)]))
        assert len(results) == 2
        assert all(r.success for r in results)


class TestAsyncFakeBackendStreaming:
    def test_subscribe_and_emit(self):
        async def _run():
            fb = AsyncFakeBackend()
            fb.set_reading("M:OUTTMP", 72.5)
            handle = await fb.subscribe(["M:OUTTMP"])

            # Emit from another task after a short delay
            async def _emit():
                await asyncio.sleep(0.05)
                fb.emit_reading("M:OUTTMP", 73.0)
                fb.emit_reading("M:OUTTMP", 74.0)
                await asyncio.sleep(0.05)
                await handle.stop()

            asyncio.ensure_future(_emit())
            readings = []
            async for reading, _ in handle.readings(timeout=2.0):
                readings.append(reading)
            assert len(readings) == 2
            assert readings[0].value == pytest.approx(73.0)
            assert readings[1].value == pytest.approx(74.0)

        asyncio.run(_run())

    def test_close(self):
        async def _run():
            fb = AsyncFakeBackend()
            fb.set_reading("M:OUTTMP", 72.5)
            await fb.close()
            with pytest.raises(RuntimeError):
                await fb.get("M:OUTTMP")
        asyncio.run(_run())
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_async_fake_backend.py -v -x 2>&1 | tail -10`
Expected: ImportError — `AsyncFakeBackend` doesn't exist yet.

**Step 3: Implement AsyncFakeBackend**

Add to `pacsys/testing.py` after the `FakeBackend` class (after line 987). This is a thin async wrapper
that delegates to a `FakeBackend` instance for all state management. `FakeBackend` operations are all
in-memory and non-blocking, so wrapping them in `async def` is safe.

```python
from pacsys.aio._backends import AsyncBackend as _AsyncBackend
from pacsys.aio._subscription import AsyncSubscriptionHandle


class AsyncFakeBackend(_AsyncBackend):
    """Async fake backend for testing. Wraps FakeBackend for state management."""

    def __init__(self):
        self._sync = FakeBackend(dispatch_mode=DispatchMode.DIRECT)
        self._closed = False
        self._handles: list[AsyncSubscriptionHandle] = []

    # -- Delegate state setup to sync FakeBackend --

    def set_reading(self, drf, value, **kwargs):
        self._sync.set_reading(drf, value, **kwargs)

    def set_error(self, drf, error_code, message):
        self._sync.set_error(drf, error_code, message)

    def reset(self):
        self._sync.reset()
        self._handles.clear()

    def was_read(self, drf):
        return self._sync.was_read(drf)

    def was_written(self, drf):
        return self._sync.was_written(drf)

    @property
    def reads(self):
        return self._sync.reads

    @property
    def writes(self):
        return self._sync.writes

    # -- AsyncBackend interface --

    @property
    def capabilities(self) -> BackendCapability:
        return self._sync.capabilities

    async def read(self, drf, timeout=None):
        self._check_closed()
        return self._sync.read(drf, timeout=timeout)

    async def get(self, drf, timeout=None):
        self._check_closed()
        return self._sync.get(drf, timeout=timeout)

    async def get_many(self, drfs, timeout=None):
        self._check_closed()
        return self._sync.get_many(drfs, timeout=timeout)

    async def write(self, drf, value, timeout=None):
        self._check_closed()
        return self._sync.write(drf, value, timeout=timeout)

    async def write_many(self, settings, timeout=None):
        self._check_closed()
        return self._sync.write_many(settings, timeout=timeout)

    async def subscribe(self, drfs, callback=None, on_error=None):
        self._check_closed()
        handle = AsyncSubscriptionHandle()
        self._handles.append(handle)
        # Register a sync callback that dispatches to the async handle
        def _on_reading(reading, _sync_handle):
            handle._dispatch(reading)

        self._sync.subscribe(drfs, callback=_on_reading)
        return handle

    def emit_reading(self, drf, value, **kwargs):
        """Emit a reading — delegates to sync FakeBackend."""
        self._sync.emit_reading(drf, value, **kwargs)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        for h in self._handles:
            await h.stop()
        self._handles.clear()
        self._sync.close()

    def _check_closed(self):
        if self._closed:
            raise RuntimeError("Backend is closed")
```

Update `__all__` at the bottom of `testing.py`:
```python
__all__ = ["FakeBackend", "FakeSubscriptionHandle", "AsyncFakeBackend"]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_async_fake_backend.py -v -x 2>&1 | tail -15`
Expected: All pass.

**Step 5: Run existing tests to verify no regression**

Run: `python -m pytest tests/test_supervised.py tests/test_supervised_policies.py -v -x 2>&1 | tail -15`
Expected: All pass.

**Step 6: Lint and commit**

```bash
ruff check --fix -q pacsys/testing.py tests/test_async_fake_backend.py
ruff check pacsys/testing.py tests/test_async_fake_backend.py
ruff format -q pacsys/testing.py tests/test_async_fake_backend.py
git add pacsys/testing.py tests/test_async_fake_backend.py
git commit -m "add AsyncFakeBackend for testing async supervisor path"
```

---

### Task 2: Dual-mode `_DAQServicer` — oneshot reads

Make the servicer accept `AsyncBackend` and use direct `await` for oneshot reads.

**Files:**
- Modify: `pacsys/supervised/_server.py:1-35` (imports + `__init__`), `_server.py:87-94` (oneshot read)
- Modify: `pacsys/supervised/_server.py:238-249` (`SupervisedServer.__init__`)

**Step 1: Write the failing test**

Add to `tests/test_supervised.py`:

```python
from pacsys.testing import AsyncFakeBackend


@pytest.fixture(scope="module")
def async_backend():
    fb = AsyncFakeBackend()
    _seed_backend(fb)
    return fb


@pytest.fixture(autouse=True)
def reset_async_backend(request):
    if "async_server" in request.fixturenames:
        fb = request.getfixturevalue("async_backend")
        fb.reset()
        _seed_backend(fb)


@pytest.fixture(scope="module")
def async_server(async_backend):
    """Start a SupervisedServer with AsyncFakeBackend on an OS-assigned port."""
    srv = SupervisedServer(async_backend, port=0)
    srv.start()
    yield srv
    srv.stop()


class TestAsyncOneshotRead:
    def test_single_device(self, async_server):
        with _make_channel(async_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1
            reply = replies[0]
            assert reply.WhichOneof("value") == "readings"
            reading = reply.readings.reading[0]
            assert reading.data.scalar == pytest.approx(72.5)

    def test_multiple_devices(self, async_server):
        with _make_channel(async_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")
            request.drf.append("G:AMANDA@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 2
            values = {}
            for r in replies:
                rd = r.readings.reading[0]
                values[r.index] = rd.data.scalar
            assert values[0] == pytest.approx(72.5)
            assert values[1] == pytest.approx(42.0)

    def test_error_reading(self, async_backend, async_server):
        async_backend.set_error("M:BADDEV", -42, "Device not found")
        with _make_channel(async_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:BADDEV@I")

            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1
            reply = replies[0]
            assert reply.WhichOneof("value") == "status"
            assert reply.status.status_code == -42
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_supervised.py::TestAsyncOneshotRead -v -x 2>&1 | tail -15`
Expected: TypeError — `SupervisedServer` rejects non-`Backend` instance.

**Step 3: Implement dual-mode support**

Modify `pacsys/supervised/_server.py`:

1. **Imports** (top of file, add):
```python
from pacsys.aio._backends import AsyncBackend
```

2. **`_DAQServicer.__init__`** (line 32-34, replace):
```python
def __init__(self, backend: Backend | AsyncBackend, policies: list[Policy]):
    self._backend = backend
    self._policies = policies
    self._async = isinstance(backend, AsyncBackend)
```

3. **Read RPC, oneshot path** (line 90, replace the single line):
```python
if self._async:
    readings = await self._backend.get_many(final_drfs)
else:
    readings = await asyncio.to_thread(self._backend.get_many, final_drfs)
```

4. **`SupervisedServer.__init__`** (lines 238-249, update type check):
```python
def __init__(
    self,
    backend: Backend | AsyncBackend,
    port: int = 50051,
    host: str = "[::]",
    policies: Optional[list[Policy]] = None,
):
    if not isinstance(backend, (Backend, AsyncBackend)):
        raise TypeError(f"backend must be a Backend or AsyncBackend instance, got {type(backend).__name__}")
    ...
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_supervised.py::TestAsyncOneshotRead tests/test_supervised.py::TestOneshotRead -v -x 2>&1 | tail -15`
Expected: All pass (both sync and async oneshot reads).

**Step 5: Lint and commit**

```bash
ruff check --fix -q pacsys/supervised/_server.py tests/test_supervised.py
ruff check pacsys/supervised/_server.py tests/test_supervised.py
ruff format -q pacsys/supervised/_server.py tests/test_supervised.py
git add pacsys/supervised/_server.py tests/test_supervised.py
git commit -m "supervisor: accept AsyncBackend for direct-await oneshot reads"
```

---

### Task 3: Dual-mode `_DAQServicer` — writes

Add async branch for the Set RPC.

**Files:**
- Modify: `pacsys/supervised/_server.py:187` (write_many call)

**Step 1: Write the failing test**

Add to `tests/test_supervised.py`:

```python
class TestAsyncSet:
    def test_single_write(self, async_backend, async_server):
        with _make_channel(async_server) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 80.0
            request.setting.append(setting)

            reply = stub.Set(request, timeout=5.0)
            assert len(reply.status) == 1
            assert reply.status[0].status_code == 0
            assert async_backend.was_written("M:OUTTMP")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_supervised.py::TestAsyncSet -v -x 2>&1 | tail -10`
Expected: Fails — `to_thread` tries to call async method from executor thread.

**Step 3: Implement async write path**

In `_server.py`, replace line 187:

```python
# was: results = await asyncio.to_thread(self._backend.write_many, backend_settings)
if self._async:
    results = await self._backend.write_many(backend_settings)
else:
    results = await asyncio.to_thread(self._backend.write_many, backend_settings)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_supervised.py::TestAsyncSet tests/test_supervised.py::TestSet -v -x 2>&1 | tail -10`
Expected: All pass.

**Step 5: Lint and commit**

```bash
ruff check --fix -q pacsys/supervised/_server.py tests/test_supervised.py
ruff check pacsys/supervised/_server.py tests/test_supervised.py
ruff format -q pacsys/supervised/_server.py tests/test_supervised.py
git add pacsys/supervised/_server.py tests/test_supervised.py
git commit -m "supervisor: async write path for Set RPC"
```

---

### Task 4: Dual-mode `_DAQServicer` — streaming

Replace the queue bridge + `call_soon_threadsafe` with direct `async for` when using an async backend.

**Files:**
- Modify: `pacsys/supervised/_server.py:96-128` (streaming path in Read RPC)

**Step 1: Write the failing test**

Add to `tests/test_supervised.py`:

```python
class TestAsyncStreamingRead:
    def test_streaming_read(self, async_backend):
        with SupervisedServer(async_backend, port=0) as srv:
            with _make_channel(srv) as ch:
                stub = DAQ_pb2_grpc.DAQStub(ch)
                request = DAQ_pb2.ReadingList()
                request.drf.append("M:OUTTMP@p,1000")

                def emit():
                    time.sleep(0.3)
                    for i in range(3):
                        async_backend.emit_reading("M:OUTTMP@p,1000", 70.0 + i)
                        time.sleep(0.05)

                emitter = threading.Thread(target=emit, daemon=True)
                emitter.start()

                replies = []
                for reply in stub.Read(request, timeout=3.0):
                    replies.append(reply)
                    if len(replies) >= 3:
                        break

                assert len(replies) == 3
                values = [r.readings.reading[0].data.scalar for r in replies]
                assert values == [pytest.approx(70.0), pytest.approx(71.0), pytest.approx(72.0)]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_supervised.py::TestAsyncStreamingRead -v -x 2>&1 | tail -10`
Expected: Fails — `to_thread(async_backend.subscribe, ...)` won't work correctly.

**Step 3: Implement async streaming path**

In `_server.py`, replace the streaming `else` branch (lines 96-128) with a dual-mode block:

```python
else:
    # Streaming path
    queue: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
    item_count = 0
    drf_index = {drf: i for i, drf in enumerate(final_drfs)}

    if self._async:
        # Async backend: consume handle.readings() directly
        handle = await self._backend.subscribe(final_drfs)
        try:
            async for reading, _ in handle.readings(timeout=1.0):
                if context.cancelled():
                    break
                idx = drf_index.get(reading.drf, 0)
                yield reading_to_proto_reply(reading, idx)
                item_count += 1
        except asyncio.TimeoutError:
            if not context.cancelled():
                raise
        finally:
            await handle.stop()
            logger.debug("stream peer=%s event=stopped items=%d", peer, item_count)
    else:
        # Sync backend: bridge via queue + call_soon_threadsafe
        loop = asyncio.get_running_loop()

        def _enqueue(reading):
            try:
                queue.put_nowait(reading)
            except asyncio.QueueFull:
                pass

        def on_reading(reading, handle):
            try:
                loop.call_soon_threadsafe(_enqueue, reading)
            except RuntimeError:
                pass

        logger.debug("stream peer=%s event=started items=%d", peer, len(final_drfs))
        handle = await asyncio.to_thread(self._backend.subscribe, final_drfs, on_reading)
        try:
            while not context.cancelled():
                try:
                    reading = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                idx = drf_index.get(reading.drf, 0)
                yield reading_to_proto_reply(reading, idx)
                item_count += 1
        finally:
            await asyncio.to_thread(handle.stop)
            logger.debug("stream peer=%s event=stopped items=%d", peer, item_count)
```

Note: the async path uses `handle.readings(timeout=1.0)` which re-raises `TimeoutError` on inactivity. We catch it and re-check `context.cancelled()`. The `AsyncSubscriptionHandle.readings()` re-raises timeout only when the handle is NOT stopped, so normal stop via `handle.stop()` won't raise.

**Step 4: Run tests**

Run: `python -m pytest tests/test_supervised.py::TestAsyncStreamingRead tests/test_supervised.py::TestStreamingRead -v -x 2>&1 | tail -10`
Expected: Both pass.

**Step 5: Lint and commit**

```bash
ruff check --fix -q pacsys/supervised/_server.py tests/test_supervised.py
ruff check pacsys/supervised/_server.py tests/test_supervised.py
ruff format -q pacsys/supervised/_server.py tests/test_supervised.py
git add pacsys/supervised/_server.py tests/test_supervised.py
git commit -m "supervisor: async streaming path using handle.readings() directly"
```

---

### Task 5: Async policy enforcement tests

Verify policies work identically with async backends.

**Files:**
- Modify: `tests/test_supervised.py`

**Step 1: Write tests**

```python
@pytest.fixture(scope="module")
def async_policy_backend():
    fb = AsyncFakeBackend()
    _seed_backend(fb)
    return fb


@pytest.fixture(autouse=True)
def reset_async_policy_backend(request):
    if "async_server_with_policy" in request.fixturenames:
        fb = request.getfixturevalue("async_policy_backend")
        fb.reset()
        _seed_backend(fb)


@pytest.fixture(scope="module")
def async_server_with_policy(async_policy_backend):
    srv = SupervisedServer(async_policy_backend, port=0, policies=[ReadOnlyPolicy()])
    srv.start()
    yield srv
    srv.stop()


class TestAsyncPolicyEnforcement:
    def test_read_allowed_with_readonly_policy(self, async_server_with_policy):
        with _make_channel(async_server_with_policy) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.ReadingList()
            request.drf.append("M:OUTTMP@I")
            replies = list(stub.Read(request, timeout=5.0))
            assert len(replies) == 1

    def test_set_blocked_with_readonly_policy(self, async_server_with_policy):
        with _make_channel(async_server_with_policy) as ch:
            stub = DAQ_pb2_grpc.DAQStub(ch)
            request = DAQ_pb2.SettingList()
            setting = DAQ_pb2.Setting()
            setting.device = "M:OUTTMP"
            setting.value.scalar = 80.0
            request.setting.append(setting)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Set(request, timeout=5.0)
            assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_supervised.py::TestAsyncPolicyEnforcement tests/test_supervised.py::TestPolicyEnforcement -v -x 2>&1 | tail -10`
Expected: All pass.

**Step 3: Lint and commit**

```bash
ruff check --fix -q tests/test_supervised.py && ruff check tests/test_supervised.py
ruff format -q tests/test_supervised.py
git add tests/test_supervised.py
git commit -m "test async policy enforcement via SupervisedServer"
```

---

### Task 6: Full regression + cleanup

Run all tests, update docs.

**Files:**
- Run: full test suite
- Modify: `docs/specialized-utils/supervised.md` (mention async backend support)

**Step 1: Run full unit test suite**

Run: `python -m pytest tests/ -v -x 2>&1 | tail -30`
Expected: All pass.

**Step 2: Run type checker**

Run: `ty check pacsys/supervised/ 2>&1 | tail -20`
Expected: No new errors.

**Step 3: Update supervised docs**

Add a short section to `docs/specialized-utils/supervised.md` showing async backend usage:

```markdown
## Using with Async Backends

SupervisedServer also accepts `AsyncBackend` instances from `pacsys.aio`.
When an async backend is provided, the server calls its methods directly
on the gRPC event loop — no executor threads, no callback bridges.

```python
import pacsys.aio as aio
from pacsys.supervised import SupervisedServer

backend = aio.dpm()
with SupervisedServer(backend, port=50051) as srv:
    srv.run()
```
```

**Step 4: Lint and commit**

```bash
ruff check --fix -q pacsys/ tests/ && ruff check pacsys/ tests/
ruff format -q pacsys/ tests/
git add -A
git commit -m "async supervisor: docs and full regression pass"
```
