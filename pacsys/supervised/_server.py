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
from pacsys.aio._backends import AsyncBackend
from pacsys.backends import Backend
from pacsys.backends.grpc_backend import _proto_value_to_python
from pacsys.drf_utils import get_device_name
from pacsys.errors import AuthenticationError

from ._audit import AuditLog
from ._conversions import reading_to_proto_reply, write_result_to_proto_status
from ._event_classify import all_oneshot
from ._policies import Policy, PolicyDecision, RequestContext, evaluate_policies

logger = logging.getLogger("pacsys.supervised")

# Bounded queue prevents OOM if client is slower than backend
_STREAM_QUEUE_MAXSIZE = 100_000


def _reorder_map(original: list[str], modified: list[str]) -> list[int] | None:
    """Map original positions to modified positions for index-correct responses.

    Returns None if lists are identical (common fast path).
    Returns map where map[orig_i] = mod_i, so results[map[i]] is the result
    for original position i.
    """
    if original == modified:
        return None
    used: set[int] = set()
    result = []
    for orig_drf in original:
        for mod_i, mod_drf in enumerate(modified):
            if mod_i not in used and mod_drf == orig_drf:
                result.append(mod_i)
                used.add(mod_i)
                break
        else:
            raise ValueError(f"Policy removed DRF {orig_drf!r} — filtering is not supported")
    return result


class _DAQServicer(DAQ_pb2_grpc.DAQServicer):
    """DAQ service implementation that proxies to a Backend."""

    def __init__(
        self,
        backend: Backend | AsyncBackend,
        policies: list[Policy],
        token: Optional[str] = None,
        audit_log: Optional[AuditLog] = None,
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
                if value == f"Bearer {self._token}":
                    return True
        peer = context.peer() or "unknown"
        logger.warning("auth peer=%s decision=denied reason=invalid or missing token", peer)
        context.set_code(grpc.StatusCode.UNAUTHENTICATED)
        context.set_details("Invalid or missing bearer token")
        return False

    def _audit_request(self, ctx: RequestContext, decision: PolicyDecision) -> Optional[int]:
        """Best-effort audit log of incoming request. Returns seq or None."""
        if self._audit is None:
            return None
        try:
            return self._audit.log_request(ctx, decision)
        except Exception:
            logger.error("audit log_request failed", exc_info=True)
            return None

    def _audit_response(self, seq: Optional[int], peer: str, method: str, proto) -> None:
        """Best-effort audit log of outgoing response."""
        if seq is None or self._audit is None:
            return
        try:
            self._audit.log_response(seq, peer, method, proto)
        except Exception:
            logger.error("audit log_response failed", exc_info=True)

    def _check_policies(
        self, drfs: list[str], rpc_method: str, context, *, values=None, raw_request=None
    ) -> tuple[RequestContext, PolicyDecision]:
        """Run policy chain. Returns (original_ctx, decision)."""
        peer = context.peer() or "unknown"
        metadata = {}
        invocation_metadata = context.invocation_metadata()
        if invocation_metadata:
            metadata = {k: v for k, v in invocation_metadata}
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

        try:
            req_ctx, decision = self._check_policies(drfs, "Read", context, raw_request=request)
        except Exception as e:
            logger.error("rpc=Read peer=%s policy error=%s", peer, e, exc_info=True)
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
        final_drfs = decision.ctx.drfs
        start = time.monotonic()

        try:
            if all_oneshot(final_drfs):
                if isinstance(self._backend, AsyncBackend):
                    readings = await self._backend.get_many(final_drfs)
                else:
                    readings = await asyncio.to_thread(self._backend.get_many, final_drfs)
                rmap = _reorder_map(drfs, final_drfs)
                for i in range(len(drfs)):
                    reading = readings[rmap[i]] if rmap else readings[i]
                    reply_proto = reading_to_proto_reply(reading, i)
                    self._audit_response(seq, peer, "Read", reply_proto)
                    yield reply_proto
                elapsed = (time.monotonic() - start) * 1000
                logger.info("rpc=Read peer=%s elapsed_ms=%.1f items=%d", peer, elapsed, len(readings))
            else:
                item_count = 0
                # Multimap: same DRF at multiple positions → fan out readings
                # Use original request positions so clients see correct indices
                drf_indices: dict[str, list[int]] = {}
                for i, drf in enumerate(drfs):
                    drf_indices.setdefault(drf, []).append(i)

                if isinstance(self._backend, AsyncBackend):
                    logger.debug("stream peer=%s event=started items=%d", peer, len(final_drfs))
                    handle = await self._backend.subscribe(final_drfs)
                    try:
                        while not context.cancelled():
                            try:
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
                                else:
                                    break  # generator returned normally (handle stopped)
                            except asyncio.TimeoutError:
                                continue
                    finally:
                        await handle.stop()
                        logger.debug("stream peer=%s event=stopped items=%d", peer, item_count)
                else:
                    queue: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
                    loop = asyncio.get_running_loop()

                    def _enqueue(reading):
                        try:
                            queue.put_nowait(reading)
                        except asyncio.QueueFull:
                            logger.warning("stream peer=%s queue full, dropping reading for %s", peer, reading.drf)

                    def on_reading(reading, handle):
                        try:
                            loop.call_soon_threadsafe(_enqueue, reading)
                        except RuntimeError:
                            pass  # loop closed during shutdown (#3)

                    logger.debug("stream peer=%s event=started items=%d", peer, len(final_drfs))
                    handle = await asyncio.to_thread(self._backend.subscribe, final_drfs, on_reading)
                    try:
                        while not context.cancelled():
                            try:
                                reading = await asyncio.wait_for(queue.get(), timeout=1.0)
                            except asyncio.TimeoutError:
                                continue
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
            logger.error("rpc=Read peer=%s error=%s", peer, e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Backend error: {e}")

    async def Set(self, request, context):
        from pacsys._proto.controls.service.DAQ.v1 import DAQ_pb2

        if not self._check_token(context):
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]

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

        try:
            values = []
            for s in settings_proto:
                value, _ = _proto_value_to_python(s.value)
                values.append((s.device, value))
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]

        try:
            req_ctx, decision = self._check_policies(drfs, "Set", context, values=values, raw_request=request)
        except Exception as e:
            logger.error("rpc=Set peer=%s policy error=%s", peer, e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Policy error: {e}")
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]

        seq = self._audit_request(req_ctx, decision)

        if not decision.allowed:
            logger.warning("rpc=Set peer=%s devices=%s decision=denied reason=%s", peer, devices, decision.reason)
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(decision.reason)
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]

        if self._check_unapproved(drfs, decision, peer, "Set", context):
            return DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]

        logger.info("rpc=Set peer=%s devices=%s decision=allowed", peer, devices)
        start = time.monotonic()

        try:
            assert decision.ctx is not None
            if len(decision.ctx.drfs) != len(decision.ctx.values):
                raise ValueError(
                    f"Policy produced mismatched drfs/values: "
                    f"{len(decision.ctx.drfs)} drfs vs {len(decision.ctx.values)} values"
                )
            backend_settings = [(drf, val) for drf, (_, val) in zip(decision.ctx.drfs, decision.ctx.values)]

            if isinstance(self._backend, AsyncBackend):
                results = await self._backend.write_many(backend_settings)
            else:
                results = await asyncio.to_thread(self._backend.write_many, backend_settings)
            rmap = _reorder_map(drfs, decision.ctx.drfs)
            reply = DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]
            for i in range(len(drfs)):
                result = results[rmap[i]] if rmap else results[i]
                reply.status.append(write_result_to_proto_status(result))
            elapsed = (time.monotonic() - start) * 1000
            logger.info("rpc=Set peer=%s elapsed_ms=%.1f items=%d", peer, elapsed, len(results))
            self._audit_response(seq, peer, "Set", reply)
            return reply

        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            reply = DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]
            self._audit_response(seq, peer, "Set", reply)
            return reply
        except NotImplementedError as e:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(str(e) or "Backend does not support this operation")
            reply = DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]
            self._audit_response(seq, peer, "Set", reply)
            return reply
        except AuthenticationError as e:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details(str(e))
            reply = DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]
            self._audit_response(seq, peer, "Set", reply)
            return reply
        except Exception as e:
            logger.error("rpc=Set peer=%s error=%s", peer, e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Backend error: {e}")
            reply = DAQ_pb2.SettingReply()  # type: ignore[unresolved-attribute]
            self._audit_response(seq, peer, "Set", reply)
            return reply


class SupervisedServer:
    """gRPC proxy server with logging and policy enforcement.

    Wraps any Backend and exposes the DAQ gRPC service, forwarding
    requests while enforcing policies and logging all traffic.

    Args:
        backend: Backend instance to proxy requests to
        port: Port to listen on (default: 50051)
        host: Host to bind (default: "[::] " for all interfaces)
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
        policies: Optional[list[Policy]] = None,
        token: Optional[str] = None,
        audit_log: Optional[AuditLog] = None,
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
        self._server: Optional[grpc_aio.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = threading.Event()
        self._start_error: Optional[BaseException] = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def host(self) -> str:
        return self._host

    async def _serve(self):
        """Run the gRPC server on this event loop."""
        server = grpc_aio.server()
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
        server = self._server
        loop = self._loop
        if server is not None and loop is not None:
            # Fire-and-forget: server.stop(0) causes wait_for_termination()
            # to return, ending the server thread.  We cannot await the
            # future because the loop closes before it can be resolved.
            # grace=0 is intentional: grace>0 deadlocks (the loop closes
            # before the grace period expires, orphaning the future).
            try:
                asyncio.run_coroutine_threadsafe(server.stop(grace=0), loop)
            except RuntimeError:
                pass
            self._server = None

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if not self._thread.is_alive():
                self._thread = None
                self._loop = None
            else:
                logger.warning("Server thread did not stop within 5s, resources may be leaked")

        if self._audit_log is not None:
            self._audit_log.close()

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
