"""``python -m atr_results_mcp`` — serve over stdio (default) or streamable HTTP.

    python -m atr_results_mcp                       # stdio, for a local MCP client
    python -m atr_results_mcp --http --port 8012    # behind nginx on tei (#156)
    python -m atr_results_mcp --call queue          # one answer on stdout, no MCP
    python -m atr_results_mcp --call job '{"job_id": "2026…"}'

``--call`` is the debugging path: it runs a tool through the same transport and
prints the JSON, so a broken answer can be looked at without an MCP client.
"""
from __future__ import annotations

import argparse
import json
import sys

from .remote import SshTransport
from .server import TOOL_NAMES, build_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atr_results_mcp")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--path", default="/mcp",
                        help="public mount path behind a proxy, e.g. /mcp/atr-results/mcp")
    parser.add_argument("--call", metavar="TOOL", choices=TOOL_NAMES,
                        help="run one tool and print its answer")
    parser.add_argument("args", nargs="?", default="{}", help="JSON arguments for --call")
    opts = parser.parse_args(argv)

    if opts.call:
        result = SshTransport().call(opts.call, json.loads(opts.args))
        json.dump(result, sys.stdout, indent=1, ensure_ascii=False)
        sys.stdout.write("\n")
        return 1 if "error" in result else 0

    server = build_server()
    if opts.http:
        server.run(transport="streamable-http", host=opts.host, port=opts.port,
                   streamable_http_path=opts.path)
    else:
        server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
