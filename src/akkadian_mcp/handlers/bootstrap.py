"""Bootstrap handler — project initialization and configuration."""

from __future__ import annotations

import json

from ..state import get_state, has_extra


def handle_bootstrap(arguments: dict) -> str:
    """Dispatch bootstrap operations."""
    op = arguments.get("operation", "init")
    ops = {
        "init": _init,
        "status": _status,
        "update_config": _update_config,
    }
    handler = ops.get(op)
    if handler is None:
        return f"Unknown bootstrap operation: {op}. Available: {', '.join(ops)}"
    return handler(arguments)


def _init(arguments: dict) -> str:
    """Initialize or re-initialize the project."""
    state = get_state()
    project_root = state.project_root
    akkadian_dir = project_root / ".akkadian"

    created = not akkadian_dir.exists()
    if state.knowledge_graph is None:
        state.init(project_root)

    extras = []
    if has_extra("mlflow"):
        extras.append("mlflow")
    if has_extra("optuna"):
        extras.append("optuna")

    config = state.get_config()

    lines = []
    if created:
        lines.append(f"Project initialized at {project_root}")
    else:
        lines.append(f"Project ready at {project_root}")

    lines.append(f"Config: {akkadian_dir / 'config.json'}")
    lines.append(f"Knowledge DB: {akkadian_dir / 'knowledge.db'}")
    lines.append(f"Extras available: {', '.join(extras) or 'none'}")

    if not config.get("competition"):
        lines.append("")
        lines.append("Next: set your competition with bootstrap update_config:")
        lines.append("  competition: your-competition-slug")
        lines.append("  kaggle_username: your-username")
        lines.append('  goal_metric: e.g. "bleu", "accuracy", "rmse"')
        lines.append('  goal_direction: "maximize" or "minimize"')

    return "\n".join(lines)


def _status(arguments: dict) -> str:
    """Show current project configuration and health."""
    state = get_state()
    config = state.get_config()

    lines = ["Project Status:", f"  Root: {state.project_root}"]

    for key, value in config.items():
        lines.append(f"  {key}: {value or '(not set)'}")

    if state.knowledge_graph:
        stats = state.knowledge_graph.get_stats()
        lines.append(
            f"  Knowledge graph: {stats['entities']} entities, {stats['relations']} relations"
        )

    lines.append(f"  MLflow available: {has_extra('mlflow')}")
    lines.append(f"  Optuna available: {has_extra('optuna')}")

    return "\n".join(lines)


def _update_config(arguments: dict) -> str:
    """Update project configuration."""
    state = get_state()
    updates = arguments.get("config", {})

    if isinstance(updates, str):
        try:
            updates = json.loads(updates)
        except json.JSONDecodeError:
            return f"Invalid config JSON: {updates}"

    if not updates:
        return "No config updates provided. Pass a 'config' object."

    config = state.update_config(updates)
    lines = ["Config updated:"]
    for key, value in config.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)
