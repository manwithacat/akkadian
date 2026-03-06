"""Centralized server state with feature detection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .knowledge_graph.store import KnowledgeGraph


def has_extra(package: str) -> bool:
    """Check if an optional dependency is installed."""
    return importlib.util.find_spec(package) is not None


class ServerState:
    """Single source of truth for MCP server state."""

    def __init__(self) -> None:
        self.project_root: Path = Path.cwd()
        self.knowledge_graph: KnowledgeGraph | None = None
        self._config_path: Path | None = None

    def init(self, project_root: Path) -> None:
        """Initialize state for a project directory."""
        self.project_root = project_root.resolve()

        akkadian_dir = self.project_root / ".akkadian"
        akkadian_dir.mkdir(parents=True, exist_ok=True)

        db_path = akkadian_dir / "knowledge.db"
        self.knowledge_graph = KnowledgeGraph(str(db_path))

        self._config_path = akkadian_dir / "config.json"
        if not self._config_path.exists():
            self._write_config(self._default_config())

    def reset(self) -> None:
        """Clean up resources."""
        if self.knowledge_graph:
            self.knowledge_graph.close()
            self.knowledge_graph = None

    def get_config(self) -> dict:
        """Read project config."""
        if self._config_path and self._config_path.exists():
            with open(self._config_path) as f:
                return json.load(f)
        return self._default_config()

    def update_config(self, updates: dict) -> dict:
        """Merge updates into project config."""
        config = self.get_config()
        config.update(updates)
        self._write_config(config)
        return config

    def _write_config(self, config: dict) -> None:
        if self._config_path:
            with open(self._config_path, "w") as f:
                json.dump(config, f, indent=2)

    def _default_config(self) -> dict:
        from . import __version__

        return {
            "version": __version__,
            "competition": "",
            "platform": "kaggle",
            "kaggle_username": "",
            "goal_metric": "",
            "goal_direction": "maximize",
        }


_state = ServerState()


def get_state() -> ServerState:
    return _state
