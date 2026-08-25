"""gRPC proxy server that forwards requests to any Backend with policy enforcement."""

import asyncio
import hmac
import logging
import signal
import threading
import time

import grpc
from grpc import aio as grpc_aio

from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2_grpc
from pacsys.aio._backends import AsyncBackend
from pacsys.backends import Backend
from pacsys.backends.grpc_backend import _proto_value_to_python
from pacsys.drf_utils import get_device_name
from pacsys.errors import AuthenticationError
from pacsys.types import Reading, Value

from ._audit import AuditLog
from ._conversions import reading_to_proto_reply, write_result_to_proto_status
from ._event_classify import all_oneshot
from ._policies import Policy, PolicyDecision, RequestContext, evaluate_policies

logger = logging.getLogger("pacsys.supervised")

# Bounded queue prevents OOM if client is slower than backend
_STREAM_QUEUE_MAXSIZE = 100_000


class _DAQServicer(DAQ_pb2_grpc.DAQServicer):
    """DAQ service implementation that proxies to a Backend."""

    def __init__(
        self,
        backend: Backend | AsyncBackend,
        policies: list[Policy],
        token: str | None = None,
        audit_log: AuditLog | None = None,
    ):
        self._backend = backend
        self._policies = policies
        self._token = token
        self._audit = audit_log

    def _check_token(self, context) -> bool:
        """Validate bearer token from gRPC metadata. Returns True if ok."""
        if self._token is None:
            return True
        md = context.invocation_metadata() or []
        for key, value in md:
            if key == "authorization":
                if hmac.compare_digest(value, f"Bearer {self._token}"):
                    return True
        peer = context.peer() or "unknown"
        logger.warning("auth peer=%s decision=denied reason=invalid or missing token", peer)
        context.set_code(grpc.StatusCode.UNAUTHENTICATED)
        context.set_details("Invalid or missing bearer token")
        return False

    def _audit_request(self, ctx: RequestContext, decision: PolicyDecision) -> int | None:
        """Best-effort audit log of incoming request. Returns seq or None."""
        if self._audit is None:
            return None
        try:
            return self._audit.log_request(ctx, decision)
        except Exception:
            logger.exception("audit log_request failed")
            return None

    def _audit_response(self, seq: int | None, peer: str, method: str, proto) -> None:
        """Best-effort audit log of outgoing response."""
        if seq is None or self._audit is None:
            return
        try:
            self._audit.log_response(seq, peer, method, proto)
        except Exception:
            logger.exception("audit log_response failed")

    def _check_policies(
        self,
        drfs: list[str],
        rpc_method: str,
        context,
        *,
        values: list[tuple[str, Value]] | None = None,
        raw_request=None,
    ) -> tuple[RequestContext, PolicyDecision]:
        """Run policy chain. Returns (original_ctx, decision)."""
        peer = context.peer() or "unknown"
        metadata = {}
        invocation_metadata = context.invocation_metadata()
        if invocation_metadata:
            metadata = dict(invocation_metadata)
        n = len(drfs)
        initial_allowed = frozenset(range(n)) if rpc_method == "Read" else frozenset()
        ctx = RequestContext(
            drfs=drfs,
            rpc_method=rpc_method,
            peer=peer,
            metadata=metadata,
            values=values or [],
            raw_request=raw_request,
            allowed=initial_allowed,
        )
        if not self._policies:
            return ctx, PolicyDecision(allowed=True, ctx=ctx)
        return ctx, evaluate_policies(self._policies, ctx)

    def _check_unapproved(self, drfs, decision, peer, rpc_method, context) -> bool:
        """Check for unapproved slots after policy chain. Returns True if denied."""
        assert decision.ctx is not None
        unapproved = set(range(len(drfs))) - set(decision.ctx.allowed)
        if not unapproved:
            return False
        names = ", ".join(get_device_name(drfs[i]) for i in sorted(unapproved))
        if not any(p.allows_writes for p in self._policies):
            reason = "No policy explicitly allows write operations"
        else:
            reason = f"No write policy approves: {names}"
        logger.warning("rpc=%s peer=%s devices=%s decision=denied reason=%s", rpc_method, peer, names, reason)
        context.set_code(grpc.StatusCode.PERMISSION_DENIED)
        context.set_details(reason)
        return True

    async def Read(self, request, context):  # noqa: N802 -- gRPC method name
        drfs = list(request.drf)
        if not drfs:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty DRF list")
            return

        peer = context.peer() or "unknown"
        devices = ", ".join(get_device_name(d) for d in drfs[:5])
        if len(drfs) > 5:
            devices += f" (+{len(drfs) - 5} more)"

        try:
            req_ctx, decision = self._check_policies(drfs, "Read", context, raw_request=request)
        except Exception as e:
            logger.exception("rpc=Read peer=%s policy check failed", peer)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Policy error: {e}")
            return

        seq = self._audit_request(req_ctx, decision)

        if not decision.allowed:
            logger.warning("rpc=Read peer=%s devices=%s decision=denied reason=%s", peer, devices, decision.reason)
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(decision.reason)
            return

        if self._check_unapproved(drfs, decision, peer, "Read", context):
            return

        logger.info("rpc=Read peer=%s devices=%s decision=allowed", peer, devices)
        assert decision.ctx is not None
        start = time.monotonic()

        try:
            if all_oneshot(drfs):
                if isinstance(self._backend, AsyncBackend):
                    readings = await self._backend.get_many(drfs)
                else:
                    readings = await asyncio.to_thread(self._backend.get_many, drfs)
                for i, reading in enumerate(readings):
                    reply_proto = reading_to_proto_reply(reading, i)
                    self._audit_response(seq, peer, "Read", reply_proto)
                    yield reply_proto
                elapsed = (time.monotonic() - start) * 1000
                logger.info("rpc=Read peer=%s elapsed_ms=%.1f items=%d", peer, elapsed, len(readings))
            else:
                item_count = 0
                # Multimap: backend DRF → original request positions
                drf_indices: dict[str, list[int]] = {}
                for i, drf in enumerate(drfs):
                    drf_indices.setdefault(drf, []).append(i)

                if isinstance(self._backend, AsyncBackend):
                    logger.debug("stream peer=%s event=started items=%d", peer, len(drfs))
                    handle = await self._backend.subscribe(drfs)
                    try:
                        while not context.cancelled():
                            async for reading, _ in handle.readings(timeout=1.0):
                                if context.cancelled():
                                    break
                                indices = drf_indices.get(reading.drf)
                                if indices is None:
                                    raise ValueError(f"Backend returned unexpected DRF {reading.drf!r}")
                                for idx in indices:
                                    reply_proto = reading_to_proto_reply(reading, idx)
                                    self._audit_response(seq, peer, "Read", reply_proto)
                                    yield reply_proto
                                    item_count += 1
                            if handle.stopped:
                                break
                    finally:
                        await handle.stop()
                        logger.debug("stream peer=%s event=stopped items=%d", peer, item_count)
                else:
                    queue: asyncio.Queue[Reading | object] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
                    loop = asyncio.get_running_loop()
                    wake = object()  # terminal-state check marker

                    def _enqueue(item):
                        try:
                            queue.put_nowait(item)
                        except asyncio.QueueFull:
                            if item is wake:
                                return  # timeout path re-checks handle state
                            logger.warning("stream peer=%s queue full, dropping reading for %s", peer, item.drf)

                    def on_reading(reading, handle):
                        try:
                            loop.call_soon_threadsafe(_enqueue, reading)
                        except RuntimeError:
                            pass  # loop closed during shutdown

                    def on_error(exc, h):
                        # Transient (retryable) errors leave the handle running;
                        # only a stopped handle is terminal. Fatal errors set
                        # stopped before this dispatch.
                        if not h.stopped:
                            return
                        try:
                            loop.call_soon_threadsafe(_enqueue, wake)
                        except RuntimeError:
                            pass

                    logger.debug("stream peer=%s event=started items=%d", peer, len(drfs))
                    handle = await asyncio.to_thread(self._backend.subscribe, drfs, on_reading, on_error)
                    try:
                        while not context.cancelled():
                            try:
                                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                            except asyncio.TimeoutError:
                                item = wake  # periodic terminal-state check
                            if not isinstance(item, Reading):
                                if handle.stopped:
                                    if handle.exc is not None:
                                        raise handle.exc
                                    break  # graceful backend stop -> end stream
                                continue
                            reading = item
                            indices = drf_indices.get(reading.drf)
                            if indices is None:
                                raise ValueError(f"Backend returned unexpected DRF {reading.drf!r}")
                            for idx in indices:
                                reply_proto = reading_to_proto_reply(reading, idx)
                                self._audit_response(seq, peer, "Read", reply_proto)
                                yield reply_proto
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
            logger.exception("rpc=Read peer=%s backend error", peer)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Backend error: {e}")

    async def Set(self, request, context):  # noqa: N802 -- gRPC method name
        from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2

        if not self._check_token(context):
            return DAQ_pb2.SettingReply()

        settings_proto = list(request.setting)
        if not settings_proto:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty settings list")
            return DAQ_pb2.SettingReply()

        drfs = [s.device for s in settings_proto]
        peer = context.peer() or "unknown"
        devices = ", ".join(get_device_name(d) for d in drfs[:5])
        if len(drfs) > 5:
            devices += f" (+{len(drfs) - 5} more)"

        try:
            values: list[tuple[str, Value]] = []
            for s in settings_proto:
                value, _ = _proto_value_to_python(s.value)
                values.append((s.device, value))
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return DAQ_pb2.SettingReply()

        try:
            req_ctx, decision = self._check_policies(drfs, "Set", context, values=values, raw_request=request)
        except Exception as e:
            logger.exception("rpc=Set peer=%s policy check failed", peer)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Policy error: {e}")
            return DAQ_pb2.SettingReply()

        seq = self._audit_request(req_ctx, decision)

        if not decision.allowed:
            logger.warning("rpc=Set peer=%s devices=%s decision=denied reason=%s", peer, devices, decision.reason)
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(decision.reason)
            return DAQ_pb2.SettingReply()

        if self._check_unapproved(drfs, decision, peer, "Set", context):
            return DAQ_pb2.SettingReply()

        logger.info("rpc=Set peer=%s devices=%s decision=allowed", peer, devices)
        start = time.monotonic()

        try:
            assert decision.ctx is not None
            backend_settings = list(decision.ctx.values)

            if isinstance(self._backend, AsyncBackend):
                results = await self._backend.write_many(backend_settings)
            else:
                results = await asyncio.to_thread(self._backend.write_many, backend_settings)
            reply = DAQ_pb2.SettingReply()
            for result in results:
                reply.status.append(write_result_to_proto_status(result))
            elapsed = (time.monotonic() - start) * 1000
            logger.info("rpc=Set peer=%s elapsed_ms=%.1f items=%d", peer, elapsed, len(results))
            self._audit_response(seq, peer, "Set", reply)
            return reply

        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            reply = DAQ_pb2.SettingReply()
            self._audit_response(seq, peer, "Set", reply)
            return reply
        except NotImplementedError as e:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(str(e) or "Backend does not support this operation")
            reply = DAQ_pb2.SettingReply()
            self._audit_response(seq, peer, "Set", reply)
            return reply
        except AuthenticationError as e:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details(str(e))
            reply = DAQ_pb2.SettingReply()
            self._audit_response(seq, peer, "Set", reply)
            return reply
        except Exception as e:
            logger.exception("rpc=Set peer=%s backend error", peer)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Backend error: {e}")
            reply = DAQ_pb2.SettingReply()
            self._audit_response(seq, peer, "Set", reply)
            return reply


class SupervisedServer:
    """gRPC proxy server with logging and policy enforcement.

    Wraps any Backend and exposes the DAQ gRPC service, forwarding
    requests while enforcing policies and logging all traffic.

    Args:
        backend: Backend instance to proxy requests to
        port: Port to listen on (default: 50051)
        host: Host to bind (default: "[::]" for all interfaces)
        policies: Optional list of Policy instances for access control
        token: Optional bearer token for write authentication.
            When set, clients must send ``JWTAuth(token=...)`` with this
            value or write (Set) RPCs are rejected with UNAUTHENTICATED.
            Reads are always open.

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
        backend: Backend | AsyncBackend,
        port: int = 50051,
        host: str = "[::]",
        policies: list[Policy] | None = None,
        token: str | None = None,
        audit_log: AuditLog | None = None,
    ):
        if not isinstance(backend, (Backend, AsyncBackend)):
            raise TypeError(f"backend must be a Backend or AsyncBackend instance, got {type(backend).__name__}")
        if port < 0 or port > 65535:
            raise ValueError(f"port must be 0-65535, got {port}")

        self._backend = backend
        self._port = port
        self._host = host
        self._policies = list(policies) if policies else []
        self._token = token
        self._audit_log = audit_log
        self._server: grpc_aio.Server | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_requested: asyncio.Event | None = None
        self._started = threading.Event()
        self._start_error: BaseException | None = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def host(self) -> str:
        return self._host

    async def _serve(self):
        """Run the gRPC server on this event loop."""
        server = grpc_aio.server()
        stop_requested = asyncio.Event()
        self._stop_requested = stop_requested
        servicer = _DAQServicer(self._backend, self._policies, token=self._token, audit_log=self._audit_log)
        DAQ_pb2_grpc.add_DAQServicer_to_server(servicer, server)

        bind_address = f"{self._host}:{self._port}"
        added_port = server.add_insecure_port(bind_address)
        if added_port == 0:
            raise RuntimeError(f"Failed to bind to {bind_address}")
        self._port = added_port

        await server.start()
        self._server = server
        logger.info("SupervisedServer started on %s:%d", self._host, self._port)
        self._started.set()

        try:
            await stop_requested.wait()
        finally:
            await server.stop(grace=0)

    def _run_loop(self):
        """Thread target: create event loop and run the server."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _run():
            await self._serve()

        try:
            loop.run_until_complete(_run())
        except Exception as e:  # noqa: BLE001
            self._start_error = e
            self._started.set()
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
        if self._start_error is not None:
            raise RuntimeError(f"Server failed to start: {self._start_error}") from self._start_error
        logger.debug("SupervisedServer background thread started")

    def stop(self) -> None:
        """Stop the server."""
        loop = self._loop
        stop_requested = self._stop_requested
        if stop_requested is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(stop_requested.set)
            except RuntimeError:
                pass

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if not self._thread.is_alive():
                self._thread = None
                self._loop = None
                self._server = None
                self._stop_requested = None
            else:
                logger.warning("Server thread did not stop within 5s, resources may be leaked")

        if self._audit_log is not None:
            self._audit_log.close()

        logger.info("SupervisedServer stopped")

    def wait(self, timeout: float | None = None) -> None:
        """Block until the server stops or timeout."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def run(self) -> None:
        """Start the server and block until interrupted (SIGINT/SIGTERM).

        Must be called from the main thread (signal handlers require it).
        """
        self.start()
        stop_event = threading.Event()

        def _on_signal(signum, frame):
            logger.info("Received signal %s, shutting down...", signal.Signals(signum).name)
            stop_event.set()

        try:
            old_sigint = signal.signal(signal.SIGINT, _on_signal)
            old_sigterm = signal.signal(signal.SIGTERM, _on_signal)
        except ValueError:
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
