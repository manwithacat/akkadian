"""Akkadian MCP Server — dynamic tool registration with optional extras."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from . import __version__
from .state import get_state, has_extra

logger = logging.getLogger(__name__)


def _bootstrap_tool() -> Tool:
    return Tool(
        name="bootstrap",
        description="Project setup and configuration: init, status, update_config",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["init", "status", "update_config"],
                    "description": "Operation to perform",
                },
                "config": {
                    "type": "object",
                    "description": "Config updates (for update_config)",
                },
            },
            "required": ["operation"],
        },
    )


def _kaggle_tool() -> Tool:
    return Tool(
        name="kaggle",
        description=(
            "Kaggle operations: status, list_kernels, submissions, push_kernel, download_output"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "status",
                        "list_kernels",
                        "submissions",
                        "push_kernel",
                        "download_output",
                    ],
                    "description": "Operation to perform",
                },
                "slug": {"type": "string", "description": "Kernel slug"},
                "competition": {"type": "string", "description": "Competition slug"},
                "path": {
                    "type": "string",
                    "description": "Directory path (for push_kernel, download_output)",
                },
            },
            "required": ["operation"],
        },
    )


def _knowledge_tool() -> Tool:
    return Tool(
        name="knowledge",
        description=(
            "Knowledge graph: status, search, entity, neighbourhood, list, "
            "record_score, add_entity, add_relation"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "status",
                        "search",
                        "entity",
                        "neighbourhood",
                        "list",
                        "record_score",
                        "add_entity",
                        "add_relation",
                    ],
                },
                "query": {"type": "string"},
                "id": {"type": "string"},
                "type": {"type": "string"},
                "depth": {"type": "integer"},
                "entity_id": {"type": "string"},
                "metric": {"type": "string"},
                "value": {"type": "number"},
                "source": {"type": "string"},
                "name": {"type": "string"},
                "metadata": {"type": "object"},
                "source_id": {"type": "string"},
                "target_id": {"type": "string"},
                "relation_type": {"type": "string"},
            },
            "required": ["operation"],
        },
    )


def _mlflow_tool() -> Tool:
    return Tool(
        name="mlflow",
        description=(
            "MLflow tracking: setup, experiments, runs, compare, best, ingest_artifact, suggest"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "setup",
                        "experiments",
                        "runs",
                        "compare",
                        "best",
                        "ingest_artifact",
                        "suggest",
                    ],
                },
                "experiment_name": {"type": "string"},
                "run_id": {"type": "string"},
                "run_ids": {"type": "array", "items": {"type": "string"}},
                "metric": {"type": "string"},
                "path": {"type": "string"},
                "filter": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["operation"],
        },
    )


def _optuna_tool() -> Tool:
    return Tool(
        name="optuna",
        description=(
            "Optuna hyperparameter tuning: create_study, studies, trials, best, "
            "importance, suggest_space, prune_config"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "create_study",
                        "studies",
                        "trials",
                        "best",
                        "importance",
                        "suggest_space",
                        "prune_config",
                    ],
                },
                "study_name": {"type": "string"},
                "direction": {"type": "string", "enum": ["minimize", "maximize"]},
                "sampler": {"type": "string"},
                "n_trials": {"type": "integer"},
                "filter": {"type": "string"},
            },
            "required": ["operation"],
        },
    )


def build_tools() -> list[Tool]:
    """Build tool list based on available extras."""
    tools = [_bootstrap_tool(), _kaggle_tool(), _knowledge_tool()]
    if has_extra("mlflow"):
        tools.append(_mlflow_tool())
    if has_extra("optuna"):
        tools.append(_optuna_tool())
    return tools


async def _dispatch(name: str, arguments: dict) -> str:
    """Route tool calls to handlers."""
    if name == "bootstrap":
        from .handlers.bootstrap import handle_bootstrap

        return handle_bootstrap(arguments)
    elif name == "kaggle":
        from .handlers.kaggle import handle_kaggle

        return handle_kaggle(arguments)
    elif name == "knowledge":
        from .handlers.knowledge import handle_knowledge

        return handle_knowledge(arguments)
    elif name == "mlflow":
        from .handlers.mlflow import handle_mlflow

        return handle_mlflow(arguments)
    elif name == "optuna":
        from .handlers.optuna import handle_optuna

        return handle_optuna(arguments)
    else:
        return f"Unknown tool: {name}"


async def run_server(project_root: Path) -> None:
    """Initialize and run the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    logger.info("Starting Akkadian MCP Server v%s", __version__)
    logger.info("Project root: %s", project_root)

    state = get_state()
    state.init(project_root)

    extras = []
    if has_extra("mlflow"):
        extras.append("mlflow")
    if has_extra("optuna"):
        extras.append("optuna")
    logger.info("Available extras: %s", extras or "none")

    server = Server("akkadian", version=__version__)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return build_tools()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = await _dispatch(name, arguments or {})
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.exception("Tool error: %s", name)
            return [TextContent(type="text", text=f"Error: {e}")]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
