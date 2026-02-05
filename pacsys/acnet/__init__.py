"""
pacsys.acnet - ACNET communication layer.

This module provides a Python implementation of the ACNET protocol
used for communication with the Fermilab accelerator control system.

Connection types:
- AcnetConnection: UDP connection to local ACNET daemon (sync)
- AcnetConnectionTCP: TCP connection to remote daemon (sync, threaded reactor)
- AcnetConnectionUDP: UDP connection to remote daemon (sync, threaded reactor)
- AsyncAcnetConnectionTCP: TCP connection to remote daemon (async, pure asyncio)
- AsyncAcnetConnectionUDP: UDP connection to remote daemon (async, pure asyncio)

Example (UDP - local daemon):
    from pacsys.acnet import AcnetConnection

    with AcnetConnection("MYTASK") as conn:
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
        async def handle_reply(reply):
            print(f"Got reply: status={reply.status}")

        await conn.send_request(
            node=await conn.get_node("CLXSRV"),
            task="DPM",
            data=b"request data",
            reply_handler=handle_reply
        )
"""

from .connection import AcnetConnection
from .connection_dpm import (
    DPM_PROXY_HOST,
    DPM_PROXY_PORT,
    DPMConnection,
    DPMError,
    DPMReading,
)
from .dpm_acnet import DPMAcnet
from .async_connection import (
    AsyncAcnetConnectionBase,
    AsyncAcnetConnectionTCP,
    AsyncAcnetConnectionUDP,
    AsyncRequestContext,
)
from .connection_sync import ACSYS_PROXY_HOST, AcnetConnectionTCP, AcnetConnectionUDP, AcnetRequestContext, NodeStats
from .constants import (
    ACNET_HEADER_SIZE,
    ACNET_PORT,
    ACNET_TCP_PORT,
    DEFAULT_TIMEOUT,
)
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
from .rad50 import decode, decode_stripped, encode

__all__ = [
    # Connections
    "AcnetConnection",
    "AcnetConnectionTCP",
    "AcnetConnectionUDP",
    "AsyncAcnetConnectionBase",
    "AsyncAcnetConnectionTCP",
    "AsyncAcnetConnectionUDP",
    "AcnetRequestContext",
    "AsyncRequestContext",
    "NodeStats",
    "ACSYS_PROXY_HOST",
    # DPM Connection (direct HTTP)
    "DPMConnection",
    "DPMError",
    "DPMReading",
    "DPM_PROXY_HOST",
    "DPM_PROXY_PORT",
    # DPM Connection (via ACNET)
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
    # Constants
    "ACNET_PORT",
    "ACNET_TCP_PORT",
    "ACNET_HEADER_SIZE",
    "DEFAULT_TIMEOUT",
]
