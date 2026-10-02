import asyncio
import importlib
import json
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
from mcp import Client

from pacsys.mcp._config import MCPConfig
from pacsys.mcp._server import _lifespan, create_server
from pacsys.testing import FakeBackend


@pytest.fixture(params=["legacy", "auto"])
def mode(request):
    return request.param


@pytest.fixture
def resources(monkeypatch):
    backend = FakeBackend()
    backend.set_reading("M:OUTTMP", 72.5, units="deg F")
    backend.set_reading("G:AMANDA", 1.23)
    backend.set_reading("Z:ACLTST.SETTING", 10.0)
    backend.set_error("M:BADDEV", -42, "Device not found")
    devdb = mock.MagicMock()
    devdb.get_device_info.return_value = {
        "M:OUTTMP": SimpleNamespace(
            description="Outside temperature", device_index=12345, reading=None, setting=None, control=None
        )
    }
    monkeypatch.delenv("PACSYS_DPM_HOST", raising=False)
    monkeypatch.delenv("PACSYS_DPM_PORT", raising=False)
    # Import dpm_http before replacing KerberosAuth so its module retains the real class.
    monkeypatch.setattr("pacsys.backends.dpm_http.DPMHTTPBackend", lambda **kwargs: backend)
    monkeypatch.setattr("pacsys.auth.KerberosAuth", lambda: SimpleNamespace(principal="user@EXAMPLE"))
    devdb_module = importlib.import_module("pacsys.devdb")
    monkeypatch.setattr(devdb_module, "DEVDB_AVAILABLE", True)
    monkeypatch.setattr(devdb_module, "DevDBClient", lambda: devdb)
    yield backend, devdb
    backend.close()


def payload(result):
    assert result.is_error is False
    assert result.structured_content is None
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_server_discovery_reads_and_cleanup(resources, mode):
    backend, devdb = resources
    async with Client(create_server(MCPConfig()), mode=mode, read_timeout_seconds=2) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(tools) == {"read_device", "write_device", "device_info"}
        for name, fields in {
            "read_device": ["drf"],
            "write_device": ["drf", "value"],
            "device_info": ["name"],
        }.items():
            assert tools[name].input_schema["required"] == fields
            assert set(tools[name].input_schema["properties"]) == set(fields)

        reading = payload(await client.call_tool("read_device", {"drf": "M:OUTTMP"}))
        assert reading["ok"] is True
        assert reading["value"] == 72.5
        assert reading["units"] == "deg F"
        assert reading["drf"] == "M:OUTTMP"
        assert backend.reads == ["M:OUTTMP"]

        info = payload(await client.call_tool("device_info", {"name": "M:OUTTMP"}))
        assert info == {"ok": True, "name": "M:OUTTMP", "description": "Outside temperature", "device_index": 12345}
        devdb.get_device_info.assert_called_once_with(["M:OUTTMP"])
        devdb.close.assert_not_called()

    devdb.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="Backend is closed"):
        backend.get("M:OUTTMP")


@pytest.mark.asyncio
async def test_server_errors_and_default_write_denial(resources, mode):
    backend, _ = resources
    async with Client(create_server(MCPConfig()), mode=mode, read_timeout_seconds=2) as client:
        failed = payload(await client.call_tool("read_device", {"drf": "M:BADDEV"}))
        assert failed["ok"] is False
        assert failed["error_code"] == -42
        assert failed["error"] == "Device not found"
        malformed = payload(await client.call_tool("read_device", {"drf": ""}))
        assert malformed["ok"] is False
        assert malformed["error"]
        denied = payload(await client.call_tool("write_device", {"drf": "Z:ACLTST", "value": 15.0}))
        assert denied["ok"] is False
        assert "No policy explicitly allows" in denied["error"]
        assert backend.writes == []

        invalid_arguments = await client.call_tool("read_device", {})
        assert invalid_arguments.is_error is True
        assert "drf" in invalid_arguments.content[0].text
        assert backend.reads == ["M:BADDEV"]


@pytest.mark.asyncio
async def test_server_write_policy_and_audit(resources, mode, tmp_path):
    backend, _ = resources
    audit_path = tmp_path / "audit.jsonl"
    config = MCPConfig(
        write_devices=["Z:ACLTST"],
        value_ranges={"Z:ACLTST": (0.0, 100.0)},
        audit_log=str(audit_path),
    )
    async with Client(create_server(config), mode=mode, read_timeout_seconds=2) as client:
        allowed = payload(await client.call_tool("write_device", {"drf": "Z:ACLTST", "value": 15.0}))
        assert allowed == {"ok": True, "drf": "Z:ACLTST.SETTING@N"}
        assert backend.get_written_value("Z:ACLTST.SETTING") == 15.0
        reading = payload(await client.call_tool("read_device", {"drf": "Z:ACLTST.SETTING"}))
        assert reading["value"] == 15.0
        denied = payload(await client.call_tool("write_device", {"drf": "Z:ACLTST", "value": 200.0}))
        assert denied["ok"] is False
        assert "outside range" in denied["error"]
        assert len(backend.writes) == 1

    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [entry["allowed"] for entry in entries] == [True, False]
    assert [entry["values"] for entry in entries] == [
        [{"drf": "Z:ACLTST.SETTING@N", "value": 15.0}],
        [{"drf": "Z:ACLTST.SETTING@N", "value": 200.0}],
    ]
    assert "outside range" in entries[1]["reason"]


@pytest.mark.asyncio
async def test_tool_calls_overlap(resources, mode, monkeypatch):
    backend, _ = resources
    entered = threading.Barrier(2, timeout=1.0)
    get = backend.get

    def slow_get(drf, timeout=None):
        entered.wait()
        return get(drf, timeout)

    monkeypatch.setattr(backend, "get", slow_get)
    async with Client(create_server(MCPConfig()), mode=mode, read_timeout_seconds=2) as client:
        results = await asyncio.gather(
            client.call_tool("read_device", {"drf": "M:OUTTMP"}),
            client.call_tool("read_device", {"drf": "G:AMANDA"}),
        )
    assert [payload(result)["value"] for result in results] == [72.5, 1.23]


@pytest.mark.parametrize(
    ("config_text", "args", "run_kwargs"),
    [
        ("", [], {}),
        ('[server]\ntransport = "sse"\n', [], {"transport": "sse", "port": 8000}),
        ('[server]\ntransport = "sse"\nport = 9090\n', [], {"transport": "sse", "port": 9090}),
        (
            '[server]\ntransport = "sse"\nport = 9090\n',
            ["--port", "9091"],
            {"transport": "sse", "port": 9091},
        ),
        ("", ["--transport", "sse", "--port", "9092"], {"transport": "sse", "port": 9092}),
    ],
)
def test_cli_passes_transport_options_to_run(monkeypatch, tmp_path, config_text, args, run_kwargs):
    from pacsys.mcp.__main__ import main

    config_path = tmp_path / "mcp.toml"
    config_path.write_text(config_text)
    monkeypatch.setattr("sys.argv", ["pacsys.mcp", "--config", str(config_path), *args])
    with mock.patch("pacsys.mcp.__main__.create_server", wraps=create_server) as factory:
        with mock.patch("pacsys.mcp._server.MCPServer.run") as run:
            main()
    run.assert_called_once_with(**run_kwargs)
    assert factory.call_args.args[0].port == run_kwargs.get("port")


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
