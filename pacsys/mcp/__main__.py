"""CLI entry point: python -m pacsys.mcp"""

import argparse
import logging

from ._config import load_config
from ._server import create_server


def main():
    parser = argparse.ArgumentParser(description="pacsys MCP server")
    parser.add_argument("--config", help="Path to TOML config file")
    parser.add_argument("--transport", choices=["stdio", "sse"], default=None, help="Transport (default: stdio)")
    parser.add_argument("--port", type=int, default=None, help="Port for SSE transport")
    parser.add_argument("--role", default=None, help="DPM role for access control")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config = load_config(args.config)

    # CLI args override config file
    if args.transport is not None:
        config.transport = args.transport
    if args.port is not None:
        config.port = args.port
    if args.role is not None:
        config.role = args.role

    server = create_server(config)
    server.run(transport=config.transport)  # ty: ignore[invalid-argument-type]


if __name__ == "__main__":
    main()
