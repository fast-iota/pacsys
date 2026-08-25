"""
pacsys.acnet - ACNET communication layer.

This module provides a Python implementation of the ACNET protocol
used for communication with the Fermilab accelerator control system.

Connection types:
- AcnetConnectionTCP: TCP connection to remote daemon (sync, threaded reactor)
- AcnetConnectionUDP: UDP connection to the local daemon (sync, threaded reactor)
- AsyncAcnetConnectionTCP: TCP connection to remote daemon (async, pure asyncio)
- AsyncAcnetConnectionUDP: UDP connection to the local daemon (async, pure asyncio)

Example (UDP - official local daemon interface):
    from pacsys.acnet import AcnetConnectionUDP

    with AcnetConnectionUDP(name="MYTASK") as conn:
        def handle_reply(reply):
            print(f"Got reply: status={reply.status}")

        conn.send_request(
            node=conn.get_node("CLXSRV"),
            task="DPM",
            data=b"request data",
            reply_handler=handle_reply
        )

Example (TCP - sync via acsys-proxy):
    from pacsys.acnet import AcnetConnectionTCP

    with AcnetConnectionTCP("acsys-proxy.fnal.gov") as conn:
        def handle_reply(reply):
            print(f"Got reply: status={reply.status}")

        conn.send_request(
            node=conn.get_node("CLXSRV"),
            task="DPM",
            data=b"request data",
            reply_handler=handle_reply
        )

Example (TCP - async via acsys-proxy):
    from pacsys.acnet import AsyncAcnetConnectionTCP

    async with AsyncAcnetConnectionTCP("acsys-proxy.fnal.gov") as conn:
        def handle_reply(reply):
            print(f"Got reply: status={reply.status}")

        await conn.send_request(
            node=await conn.get_node("CLXSRV"),
            task="DPM",
            data=b"request data",
            reply_handler=handle_reply
        )
"""

import importlib as _importlib
import typing as _typing

if _typing.TYPE_CHECKING:
    from .async_connection import (
        AsyncAcnetConnectionBase,
        AsyncAcnetConnectionTCP,
        AsyncAcnetConnectionUDP,
        AsyncRequestContext,
    )
    from .connection_sync import (
        ACSYS_PROXY_HOST,
        AcnetConnectionTCP,
        AcnetConnectionUDP,
        AcnetRequestContext,
        NodeStats,
    )
    from .constants import ACNET_CLIENT_PORT, ACNET_HEADER_SIZE, ACNET_PORT, ACNET_TCP_PORT, DEFAULT_TIMEOUT
    from .dpm_acnet import DPMAcnet, DPMError, DPMReading
    from .errors import (
        ACNET_CANCELLED,
        ACNET_DISCONNECTED,
        ACNET_ENDMULT,
        ACNET_NO_NODE,
        ACNET_NO_TASK,
        ACNET_OK,
        ACNET_PEND,
        ACNET_REQTMO,
        ACNET_SUCCESS,
        AcnetError,
        AcnetNodeError,
        AcnetRequestRejectedError,
        AcnetTaskError,
        AcnetTimeoutError,
        AcnetUnavailableError,
    )
    from .ftp import (
        FTPClassCode,
        FTPClassInfo,
        FTPClient,
        FTPDataPoint,
        FTPDevice,
        FTPStream,
        SnapClassInfo,
        SnapshotHandle,
        SnapshotState,
        get_ftp_class_info,
        get_snap_class_info,
    )
    from .ftp_spec import (
        ClockSample,
        ClockTrigger,
        DeviceTrigger,
        ExternalSample,
        ExternalTrigger,
        FTPSpec,
        PeriodicSample,
        ReArmSpec,
        SnapshotSpec,
        StateTrigger,
        parse_ftp_event,
    )
    from .packet import (
        AcnetCancel,
        AcnetMessage,
        AcnetPacket,
        AcnetReply,
        AcnetRequest,
        ReplyId,
        RequestId,
        node_parts,
        node_value,
    )
    from .rad50 import decode, decode_stripped, encode


_LAZY_IMPORTS = {
    # Connections
    "AcnetConnectionTCP": ".connection_sync",
    "AcnetConnectionUDP": ".connection_sync",
    "AsyncAcnetConnectionBase": ".async_connection",
    "AsyncAcnetConnectionTCP": ".async_connection",
    "AsyncAcnetConnectionUDP": ".async_connection",
    "AcnetRequestContext": ".connection_sync",
    "AsyncRequestContext": ".async_connection",
    "NodeStats": ".connection_sync",
    "ACSYS_PROXY_HOST": ".connection_sync",
    # DPM
    "DPMError": ".dpm_acnet",
    "DPMReading": ".dpm_acnet",
    "DPMAcnet": ".dpm_acnet",
    # Packets
    "AcnetPacket": ".packet",
    "AcnetReply": ".packet",
    "AcnetRequest": ".packet",
    "AcnetMessage": ".packet",
    "AcnetCancel": ".packet",
    "RequestId": ".packet",
    "ReplyId": ".packet",
    "node_value": ".packet",
    "node_parts": ".packet",
    # RAD50
    "encode": ".rad50",
    "decode": ".rad50",
    "decode_stripped": ".rad50",
    # Errors
    "AcnetError": ".errors",
    "AcnetUnavailableError": ".errors",
    "AcnetTimeoutError": ".errors",
    "AcnetNodeError": ".errors",
    "AcnetRequestRejectedError": ".errors",
    "AcnetTaskError": ".errors",
    "ACNET_OK": ".errors",
    "ACNET_SUCCESS": ".errors",
    "ACNET_PEND": ".errors",
    "ACNET_ENDMULT": ".errors",
    "ACNET_REQTMO": ".errors",
    "ACNET_CANCELLED": ".errors",
    "ACNET_DISCONNECTED": ".errors",
    "ACNET_NO_NODE": ".errors",
    "ACNET_NO_TASK": ".errors",
    # FTP
    "FTPClient": ".ftp",
    "FTPStream": ".ftp",
    "SnapshotHandle": ".ftp",
    "SnapshotState": ".ftp",
    "FTPDevice": ".ftp",
    "FTPDataPoint": ".ftp",
    "FTPClassCode": ".ftp",
    "FTPClassInfo": ".ftp",
    "SnapClassInfo": ".ftp",
    "get_ftp_class_info": ".ftp",
    "get_snap_class_info": ".ftp",
    # FTP spec
    "FTPSpec": ".ftp_spec",
    "SnapshotSpec": ".ftp_spec",
    "parse_ftp_event": ".ftp_spec",
    "ClockTrigger": ".ftp_spec",
    "DeviceTrigger": ".ftp_spec",
    "ExternalTrigger": ".ftp_spec",
    "StateTrigger": ".ftp_spec",
    "PeriodicSample": ".ftp_spec",
    "ClockSample": ".ftp_spec",
    "ExternalSample": ".ftp_spec",
    "ReArmSpec": ".ftp_spec",
    # Constants
    "ACNET_PORT": ".constants",
    "ACNET_CLIENT_PORT": ".constants",
    "ACNET_TCP_PORT": ".constants",
    "ACNET_HEADER_SIZE": ".constants",
    "DEFAULT_TIMEOUT": ".constants",
}

_LAZY_SUBMODULES = frozenset(
    {
        "async_connection",
        "connection_sync",
        "constants",
        "dpm_acnet",
        "errors",
        "ftp",
        "ftp_spec",
        "packet",
        "rad50",
    }
)

__all__ = [
    # Connections
    "AcnetConnectionTCP",
    "AcnetConnectionUDP",
    "AsyncAcnetConnectionBase",
    "AsyncAcnetConnectionTCP",
    "AsyncAcnetConnectionUDP",
    "AcnetRequestContext",
    "AsyncRequestContext",
    "NodeStats",
    "ACSYS_PROXY_HOST",
    # DPM (via ACNET)
    "DPMError",
    "DPMReading",
    "DPMAcnet",
    # Packets
    "AcnetPacket",
    "AcnetReply",
    "AcnetRequest",
    "AcnetMessage",
    "AcnetCancel",
    "RequestId",
    "ReplyId",
    "node_value",
    "node_parts",
    # RAD50
    "encode",
    "decode",
    "decode_stripped",
    # Errors
    "AcnetError",
    "AcnetUnavailableError",
    "AcnetTimeoutError",
    "AcnetNodeError",
    "AcnetRequestRejectedError",
    "AcnetTaskError",
    "ACNET_OK",
    "ACNET_SUCCESS",
    "ACNET_PEND",
    "ACNET_ENDMULT",
    "ACNET_REQTMO",
    "ACNET_CANCELLED",
    "ACNET_DISCONNECTED",
    "ACNET_NO_NODE",
    "ACNET_NO_TASK",
    # FTP (Fast Time Plot)
    "FTPClient",
    "FTPStream",
    "SnapshotHandle",
    "SnapshotState",
    "FTPDevice",
    "FTPDataPoint",
    "FTPClassCode",
    "FTPClassInfo",
    "SnapClassInfo",
    "get_ftp_class_info",
    "get_snap_class_info",
    # FTP Spec Parser
    "FTPSpec",
    "SnapshotSpec",
    "parse_ftp_event",
    "ClockTrigger",
    "DeviceTrigger",
    "ExternalTrigger",
    "StateTrigger",
    "PeriodicSample",
    "ClockSample",
    "ExternalSample",
    "ReArmSpec",
    # Constants
    "ACNET_PORT",
    "ACNET_CLIENT_PORT",
    "ACNET_TCP_PORT",
    "ACNET_HEADER_SIZE",
    "DEFAULT_TIMEOUT",
]


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is not None:
        module = _importlib.import_module(module_name, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _LAZY_SUBMODULES)
