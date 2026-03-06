"""Entry point: python -m akkadian_mcp"""

import argparse
import asyncio
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Akkadian MCP Server")
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root directory",
    )
    args = parser.parse_args()

    from .server import run_server

    asyncio.run(run_server(args.working_dir.resolve()))


if __name__ == "__main__":
    main()
