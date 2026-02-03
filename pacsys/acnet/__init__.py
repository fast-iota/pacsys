"""
pacsys.acnet - Thread-based ACNET communication layer.

This module provides a Python implementation of the ACNET protocol
used for communication with the Fermilab accelerator control system.

Two connection types are available:
- AcnetConnection: UDP connection to local ACNET daemon
- AcnetConnectionTCP: TCP connection to remote daemon (via acsys-proxy)

Key Components:
- AcnetConnection: UDP communication with local ACNET daemon
- AcnetConnectionTCP: TCP communication with remote ACNET daemon
- AcnetPacket: Packet parsing for requests, replies, and messages
- rad50: RAD50 character encoding/decoding
- errors: ACNET status codes and exceptions
- constants: Protocol constants

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

Example (TCP - remote via acsys-proxy):
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
from .connection_tcp import ACSYS_PROXY_HOST, AcnetConnectionTCP, AcnetRequestContext
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
from .rad50 import decode, decode_stripped, encode

__all__ = [
    # Connections
    "AcnetConnection",
    "AcnetConnectionTCP",
    "AcnetRequestContext",
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
    # Constants
    "ACNET_PORT",
    "ACNET_TCP_PORT",
    "ACNET_HEADER_SIZE",
    "DEFAULT_TIMEOUT",
]
