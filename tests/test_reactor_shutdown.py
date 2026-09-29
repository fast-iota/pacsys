"""Shutdown must publish coroutine outcomes to synchronous callers."""

import asyncio

import pytest

from pacsys.acnet import AcnetConnectionTCP
from pacsys.backends.dpm_http import DPMHTTPBackend
from pacsys.backends.grpc_backend import GRPC_AVAILABLE, GRPCBackend


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            GRPCBackend,
            id="grpc",
            marks=pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpc not available"),
        ),
        pytest.param(DPMHTTPBackend, id="dpm-http"),
        pytest.param(AcnetConnectionTCP, id="acnet"),
    ],
)
@pytest.mark.parametrize("outcome", ["result", "error", "cancel"])
def test_reactor_shutdown_publishes_outcome(factory, outcome):
    backend = factory()
    backend._start_reactor()
    loop = backend._loop
    thread = backend._reactor_thread
    assert loop is not None
    assert thread is not None
    sentinel = object()
    error = ValueError("coroutine failed during shutdown")

    async def stop_and_complete():
        asyncio.get_running_loop().stop()
        if outcome == "error":
            raise error
        if outcome == "cancel":
            await asyncio.Future()
        return sentinel

    try:
        future = asyncio.run_coroutine_threadsafe(stop_and_complete(), loop)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert loop.is_closed()
        assert future.done(), "reactor discarded coroutine completion"
        if outcome == "error":
            assert future.exception() is error
        elif outcome == "cancel":
            assert future.cancelled()
        else:
            assert future.result() is sentinel
    finally:
        if thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
        assert not thread.is_alive()
        # The reactor already closed its loop; close only the facade resources.
        backend._loop = None
        backend._reactor_thread = None
        backend.close()
