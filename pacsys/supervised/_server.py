"""gRPC proxy server that forwards requests to any Backend with policy enforcement."""

import asyncio
import logging
import signal
import threading
import time
from typing import Optional

import grpc
from grpc import aio as grpc_aio

from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2_grpc
from pacsys.backends import Backend
from pacsys.backends.grpc_backend import _proto_value_to_python
from pacsys.drf_utils import get_device_name
from pacsys.errors import AuthenticationError

from ._conversions import reading_to_proto_reply, write_result_to_proto_status
from ._event_classify import all_oneshot
from ._policies import Policy, PolicyDecision, RequestContext, evaluate_policies

logger = logging.getLogger("pacsys.supervised")

# Bounded queue prevents OOM if client is slower than backend (#1)
_STREAM_QUEUE_MAXSIZE = 10_000


class _DAQServicer(DAQ_pb2_grpc.DAQServicer):
    """DAQ service implementation that proxies to a Backend."""

    def __init__(self, backend: Backend, policies: list[Policy]):
        self._backend = backend
        self._policies = policies

    def _check_policies(self, drfs: list[str], rpc_method: str, context) -> Optional[PolicyDecision]:
        """Run policy chain. Returns denial decision or None if allowed."""
        if not self._policies:
            return None
        peer = context.peer() or "unknown"
        # Extract metadata as dict
        metadata = {}
        invocation_metadata = context.invocation_metadata()
        if invocation_metadata:
            metadata = {k: v for k, v in invocation_metadata}
        ctx = RequestContext(drfs=drfs, rpc_method=rpc_method, peer=peer, metadata=metadata)
        decision = evaluate_policies(self._policies, ctx)
        if not decision.allowed:
            return decision
        return None

    async def Read(self, request, context):
        drfs = list(request.drf)
        if not drfs:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty DRF list")
            return

        peer = context.peer() or "unknown"
        devices = ", ".join(get_device_name(d) for d in drfs[:5])
        if len(drfs) > 5:
            devices += f" (+{len(drfs) - 5} more)"

        # Policy check
        denial = self._check_policies(drfs, "Read", context)
        if denial is not None:
            logger.warning("rpc=Read peer=%s devices=%s decision=denied reason=%s", peer, devices, denial.reason)
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(denial.reason)
            return

        logger.info("rpc=Read peer=%s devices=%s decision=allowed", peer, devices)
        start = time.monotonic()

        try:
            if all_oneshot(drfs):
                # One-shot: use get_many
                readings = await asyncio.to_thread(self._backend.get_many, drfs)
                for i, reading in enumerate(readings):
                    yield reading_to_proto_reply(reading, i)
                elapsed = (time.monotonic() - start) * 1000
                logger.info("rpc=Read peer=%s elapsed_ms=%.1f items=%d", peer, elapsed, len(readings))
            else:
                # Streaming: use subscribe with callback bridging to async queue
                queue: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
                loop = asyncio.get_running_loop()
                item_count = 0
                # Build index map for O(1) lookup (#4)
                drf_index = {drf: i for i, drf in enumerate(drfs)}

                def on_reading(reading, handle):
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, reading)
                    except RuntimeError:
                        pass  # loop closed during shutdown (#3)
                    except asyncio.QueueFull:
                        pass  # backpressure: drop newest under overload (#1)

                logger.debug("stream peer=%s event=started items=%d", peer, len(drfs))
                handle = await asyncio.to_thread(self._backend.subscribe, drfs, on_reading)
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

        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
        except NotImplementedError as e:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(str(e) or "Backend does not support this operation")
        except AuthenticationError as e:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details(str(e))
        except Exception as e:
            logger.error("rpc=Read peer=%s error=%s", peer, e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Backend error: {e}")

    async def Set(self, request, context):
        from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2

        settings_proto = list(request.setting)
        if not settings_proto:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty settings list")
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]

        drfs = [s.device for s in settings_proto]
        peer = context.peer() or "unknown"
        devices = ", ".join(get_device_name(d) for d in drfs[:5])
        if len(drfs) > 5:
            devices += f" (+{len(drfs) - 5} more)"

        # Policy check
        denial = self._check_policies(drfs, "Set", context)
        if denial is not None:
            logger.warning("rpc=Set peer=%s devices=%s decision=denied reason=%s", peer, devices, denial.reason)
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(denial.reason)
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]

        logger.info("rpc=Set peer=%s devices=%s decision=allowed", peer, devices)
        start = time.monotonic()

        try:
            # Convert proto settings to backend format
            backend_settings = []
            for s in settings_proto:
                value, _ = _proto_value_to_python(s.value)
                backend_settings.append((s.device, value))

            results = await asyncio.to_thread(self._backend.write_many, backend_settings)
            reply = DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]
            for result in results:
                reply.status.append(write_result_to_proto_status(result))
            elapsed = (time.monotonic() - start) * 1000
            logger.info("rpc=Set peer=%s elapsed_ms=%.1f items=%d", peer, elapsed, len(results))
            return reply

        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]
        except NotImplementedError as e:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(str(e) or "Backend does not support this operation")
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]
        except AuthenticationError as e:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details(str(e))
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]
        except Exception as e:
            logger.error("rpc=Set peer=%s error=%s", peer, e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Backend error: {e}")
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]

    async def Alarms(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Alarms not supported - Backend ABC has no alarms method")
        from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2

        return DAQ_pb2.AlarmsReply()  # type: ignore[unresolved-attribute]


class SupervisedServer:
    """gRPC proxy server with logging and policy enforcement.

    Wraps any Backend and exposes the DAQ gRPC service, forwarding
    requests while enforcing policies and logging all traffic.

    Args:
        backend: Backend instance to proxy requests to
        port: Port to listen on (default: 50051)
        host: Host to bind (default: "[::] " for all interfaces)
        policies: Optional list of Policy instances for access control

    Example:
        from pacsys.testing import FakeBackend
        from pacsys.supervised import SupervisedServer, ReadOnlyPolicy

        fb = FakeBackend()
        fb.set_reading("M:OUTTMP", 72.5)

        with SupervisedServer(fb, port=50099, policies=[ReadOnlyPolicy()]) as srv:
            # Clients can now connect to localhost:50099
            srv.wait()
    """

    def __init__(
        self,
        backend: Backend,
        port: int = 50051,
        host: str = "[::]",
        policies: Optional[list[Policy]] = None,
    ):
        if not isinstance(backend, Backend):
            raise TypeError(f"backend must be a Backend instance, got {type(backend).__name__}")
        if port < 0 or port > 65535:
            raise ValueError(f"port must be 0-65535, got {port}")

        self._backend = backend
        self._port = port
        self._host = host
        self._policies = list(policies) if policies else []
        self._server: Optional[grpc_aio.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = threading.Event()
        self._start_error: Optional[BaseException] = None  # (#10) propagate root cause

    @property
    def port(self) -> int:
        return self._port

    @property
    def host(self) -> str:
        return self._host

    async def _serve(self):
        """Run the gRPC server on this event loop."""
        server = grpc_aio.server()
        servicer = _DAQServicer(self._backend, self._policies)
        DAQ_pb2_grpc.add_DAQServicer_to_server(servicer, server)

        bind_address = f"{self._host}:{self._port}"
        added_port = server.add_insecure_port(bind_address)
        if added_port == 0:
            raise RuntimeError(f"Failed to bind to {bind_address}")
        self._port = added_port  # Update in case port 0 was used (OS-assigned)

        await server.start()
        self._server = server
        logger.info("SupervisedServer started on %s:%d", self._host, self._port)
        self._started.set()

        try:
            await server.wait_for_termination()
        except asyncio.CancelledError:
            pass

    def _run_loop(self):
        """Thread target: create event loop and run the server."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _run():
            await self._serve()

        try:
            loop.run_until_complete(_run())
        except Exception as e:
            # (#10) Store error so start() can propagate the root cause
            self._start_error = e
            self._started.set()  # unblock start() immediately
            logger.error("Server loop error: %s", e)
        finally:
            loop.close()

    def start(self) -> None:
        """Start the server in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Server is already running")

        self._start_error = None
        self._started.clear()
        self._thread = threading.Thread(target=self._run_loop, name="SupervisedServer", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=10.0):
            raise RuntimeError("Server failed to start within 10 seconds")
        # (#10) Re-raise root cause if server thread failed during startup
        if self._start_error is not None:
            raise RuntimeError(f"Server failed to start: {self._start_error}") from self._start_error
        logger.debug("SupervisedServer background thread started")

    def stop(self) -> None:
        """Stop the server gracefully."""
        server = self._server
        loop = self._loop
        if server is not None and loop is not None:

            async def _shutdown():
                await server.stop(grace=2.0)

            try:
                fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
                fut.result(timeout=5.0)
            except Exception as e:
                logger.debug("Error during server shutdown: %s", e)

            self._server = None

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            # (#5) Only clear references if thread actually finished
            if not self._thread.is_alive():
                self._thread = None
                self._loop = None
            else:
                logger.warning("Server thread did not stop within 5s, resources may be leaked")

        logger.info("SupervisedServer stopped")

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until the server stops or timeout."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def run(self) -> None:
        """Start the server and block until interrupted (SIGINT/SIGTERM).

        Must be called from the main thread (signal handlers require it).
        """
        self.start()
        # (#7) Set up signal handlers; if this fails, clean up the started server
        stop_event = threading.Event()

        def _on_signal(signum, frame):
            logger.info("Received signal %s, shutting down...", signal.Signals(signum).name)
            stop_event.set()

        try:
            old_sigint = signal.signal(signal.SIGINT, _on_signal)
            old_sigterm = signal.signal(signal.SIGTERM, _on_signal)
        except ValueError:
            # signal.signal() only works from main thread
            self.stop()
            raise ValueError("run() must be called from the main thread") from None

        try:
            stop_event.wait()
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            self.stop()

    def __enter__(self) -> "SupervisedServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def __repr__(self) -> str:
        running = self._server is not None
        status = "running" if running else "stopped"
        n_policies = len(self._policies)
        return f"SupervisedServer({self._host}:{self._port}, {status}, policies={n_policies})"
