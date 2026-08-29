import importlib
from types import SimpleNamespace
from unittest import mock

import pytest

from pacsys.mcp._config import MCPConfig
from pacsys.mcp._server import ServerContext, _lifespan, create_server


class _FakeFastMCP:
    def __init__(self, name, *, instructions, lifespan, port):
        self.name = name
        self.instructions = instructions
        self.lifespan = lifespan
        self.port = port
        self.tools = {}
        self.context = None

    def tool(self, *, description):
        def register(func):
            self.tools[func.__name__] = func
            return func

        return register

    def get_context(self):
        request_context = SimpleNamespace(lifespan_context=self.context)
        return SimpleNamespace(request_context=request_context)


def test_create_server_registers_context_bound_tools():
    with mock.patch("pacsys.mcp._server.FastMCP", _FakeFastMCP):
        server = create_server(MCPConfig())

    assert set(server.tools) == {"read_device", "write_device", "device_info"}
    backend = object()
    policies = []
    devdb = object()
    server.context = ServerContext(backend=backend, devdb=devdb, policies=policies)

    with (
        mock.patch("pacsys.mcp._server.tool_read_device", return_value={"read": True}) as read,
        mock.patch("pacsys.mcp._server.tool_write_device", return_value={"write": True}) as write,
        mock.patch("pacsys.mcp._server.tool_device_info", return_value={"info": True}) as info,
    ):
        assert server.tools["read_device"]("M:OUTTMP") == {"read": True}
        assert server.tools["write_device"]("Z:ACLTST", 1.0) == {"write": True}
        assert server.tools["device_info"]("M:OUTTMP") == {"info": True}

    read.assert_called_once_with(backend, "M:OUTTMP", policies)
    write.assert_called_once_with(backend, "Z:ACLTST", 1.0, policies, None)
    info.assert_called_once_with(devdb, "M:OUTTMP")


def test_create_server_wires_sse_port():
    with mock.patch("pacsys.mcp._server.FastMCP", _FakeFastMCP):
        server = create_server(MCPConfig(transport="sse", port=9090))

    assert server.port == 9090


@pytest.mark.asyncio
async def test_lifespan_closes_backend_devdb_and_audit(monkeypatch, tmp_path):
    backend = mock.MagicMock()
    devdb = mock.MagicMock()
    auth = SimpleNamespace(principal="user@EXAMPLE")
    backend_factory = mock.MagicMock(return_value=backend)
    devdb_module = importlib.import_module("pacsys.devdb")

    monkeypatch.delenv("PACSYS_DPM_HOST", raising=False)
    monkeypatch.delenv("PACSYS_DPM_PORT", raising=False)
    # Import dpm_http (via its patch) BEFORE patching KerberosAuth: a first import inside the
    # patch window would bind dpm_http.KerberosAuth to the lambda for the rest of the session.
    monkeypatch.setattr("pacsys.backends.dpm_http.DPMHTTPBackend", backend_factory)
    monkeypatch.setattr("pacsys.auth.KerberosAuth", lambda: auth)
    monkeypatch.setattr(devdb_module, "DEVDB_AVAILABLE", True)
    monkeypatch.setattr(devdb_module, "DevDBClient", mock.MagicMock(return_value=devdb))

    audit_path = tmp_path / "mcp-audit.jsonl"
    async with _lifespan(None, config=MCPConfig(role="testing", audit_log=str(audit_path))) as context:
        assert context.backend is backend
        assert context.devdb is devdb
        assert context.policies == []
        assert context.audit_log is not None

    backend_factory.assert_called_once_with(timeout=5.0, auth=auth, role="testing")
    backend.close.assert_called_once_with()
    devdb.close.assert_called_once_with()
    assert audit_path.exists()
    assert context.audit_log._json_file is None
