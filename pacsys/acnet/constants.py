"""
ACNET protocol constants.

These constants define the ACNET network protocol parameters including
ports, header sizes, and message type flags.

Command names follow official acnetd CommandList enum naming.
"""

# Network ports
ACNET_PORT = 6801  # Primary ACNET UDP port
ACNET_TCP_PORT = 6802  # TCP port for daemon

# Packet structure
ACNET_HEADER_SIZE = 18  # Fixed header size in bytes
MAX_ACNET_MESSAGE_SIZE = (8 * 1024) + 128  # Maximum message size

# Message type flags (bits 0-3 of flags field)
ACNET_FLG_TYPE = 0x000E  # Mask for message type
ACNET_FLG_USM = 0x0000  # Unsolicited message
ACNET_FLG_REQ = 0x0002  # Request message
ACNET_FLG_RPY = 0x0004  # Reply message
ACNET_FLG_CAN = 0x0200  # Cancel flag
ACNET_FLG_MLT = 0x0001  # Multiple reply flag

# Reply flags
REPLY_NORMAL = 0x00  # Normal reply
REPLY_ENDMULT = 0x02  # End of multiple replies

# acnetd daemon commands (official names from acnetd CommandList enum)
CMD_KEEPALIVE = 0               # cmdKeepAlive - ping/keepalive
CMD_CONNECT = 1                 # cmdConnect - connect with 8-bit task ID
CMD_RENAME_TASK = 2             # cmdRenameTask - rename task
CMD_DISCONNECT = 3              # cmdDisconnect - disconnect from daemon
CMD_SEND = 4                    # cmdSend - send unsolicited message
CMD_SEND_REQUEST = 5            # cmdSendRequest - send request (no timeout)
CMD_RECEIVE_REQUESTS = 6        # cmdReceiveRequests - start receiving requests
CMD_SEND_REPLY = 7              # cmdSendReply - send reply to request
CMD_CANCEL = 8                  # cmdCancel - cancel outstanding request
CMD_REQUEST_ACK = 9             # cmdRequestAck - acknowledge request received
CMD_ADD_NODE = 10               # cmdAddNode - add node to daemon table
CMD_NAME_LOOKUP = 11            # cmdNameLookup - get trunk/node by name
CMD_NODE_LOOKUP = 12            # cmdNodeLookup - get name by trunk/node
CMD_LOCAL_NODE = 13             # cmdLocalNode - get local node value
CMD_TASK_PID = 14               # cmdTaskPid - get task PID
CMD_NODE_STATS = 15             # cmdNodeStats - get node statistics
CMD_CONNECT_EXT = 16            # cmdConnectExt - connect with 16-bit task ID
CMD_DISCONNECT_SINGLE = 17      # cmdDisconnectSingle - disconnect single connection
CMD_SEND_REQUEST_TIMEOUT = 18   # cmdSendRequestWithTimeout - send request with timeout
CMD_IGNORE_REQUEST = 19         # cmdIgnoreRequest - ignore incoming request
CMD_BLOCK_REQUESTS = 20         # cmdBlockRequests - block/stop receiving requests
CMD_TCP_CONNECT = 21            # cmdTcpConnect - TCP connect
CMD_DEFAULT_NODE = 22           # cmdDefaultNode - get default node
CMD_TCP_CONNECT_EXT = 23        # cmdTcpConnectExt - TCP connect extended

# acnetd acknowledgement codes (official names from acnetd AckList enum)
ACK_ACK = 0                     # ackAck - generic acknowledgement
ACK_CONNECT = 1                 # ackConnect - connect response
ACK_SEND_REQUEST = 2            # ackSendRequest - send request response
ACK_SEND_REPLY = 3              # ackSendReply - send reply response
ACK_NAME_LOOKUP = 4             # ackNameLookup - name lookup response
ACK_NODE_LOOKUP = 5             # ackNodeLookup - node lookup response
ACK_TASK_PID = 6                # ackTaskPid - task PID response
ACK_NODE_STATS = 7              # ackNodeStats - node stats response
ACK_CONNECT_EXT = 16            # ackConnectExt - extended connect response

# Timeouts (milliseconds)
DEFAULT_TIMEOUT = 5000  # Default request timeout
HEARTBEAT_TIMEOUT = 10000  # Connection monitor interval
RECONNECT_DELAY = 2000  # Delay before reconnect attempt

# Buffer sizes
CMD_BUFFER_SIZE = 16  # Command buffer size
SEND_BUFFER_SIZE = 256 * 1024  # Socket send buffer
RECV_BUFFER_SIZE = 2048 * 1024  # Socket receive buffer
