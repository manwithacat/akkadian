"""Seed the knowledge graph from project structure."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import time

from .models import Entity, Relation
from .store import KnowledgeGraph

logger = logging.getLogger(__name__)


def seed_if_needed(kg: KnowledgeGraph, project_root: Path) -> None:
    """Seed the graph from project structure if empty."""
    stats = kg.get_stats()
    if stats["entities"] > 0:
        return

    logger.info("Seeding knowledge graph from project structure...")
    _seed_models(kg, project_root)
    _seed_kernels(kg, project_root)

    stats = kg.get_stats()
    logger.info("Seeded %d entities, %d relations", stats["entities"], stats["relations"])


def _seed_models(kg: KnowledgeGraph, project_root: Path) -> None:
    """Discover models from models/ directory."""
    models_dir = project_root / "models"
    if not models_dir.exists():
        return

    now = time()
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        config_path = model_dir / "config.json"
        if not config_path.exists():
            config_path = model_dir / "final" / "config.json"

        metadata: dict = {"path": str(model_dir)}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                metadata["model_type"] = config.get("model_type", "")
                metadata["architectures"] = config.get("architectures", [])
            except (json.JSONDecodeError, OSError):
                pass

        kg.upsert_entity(
            Entity(
                id=f"model:{model_dir.name}",
                name=model_dir.name,
                metadata=metadata,
                created_at=now,
                updated_at=now,
            )
        )


def _seed_kernels(kg: KnowledgeGraph, project_root: Path) -> None:
    """Discover kernels from kernel-metadata.json files."""
    now = time()

    for meta_path in project_root.rglob("kernel-metadata.json"):
        if any(part.startswith(".") for part in meta_path.relative_to(project_root).parts):
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        slug = meta.get("id", "").split("/")[-1] if meta.get("id") else meta_path.parent.name
        metadata = {
            "path": str(meta_path.parent),
            "title": meta.get("title", slug),
            "language": meta.get("language", "python"),
        }

        model_sources = meta.get("model_sources", [])
        if model_sources:
            metadata["model_sources"] = model_sources

        kg.upsert_entity(
            Entity(
                id=f"kernel:{slug}",
                name=slug,
                metadata=metadata,
                created_at=now,
                updated_at=now,
            )
        )

        for ms in model_sources:
            parts = ms.split("/")
            if len(parts) >= 2:
                model_id = f"model:{parts[1]}"
                if kg.get_entity(model_id):
                    kg.add_relation(
                        Relation(
                            source_id=f"kernel:{slug}",
                            target_id=model_id,
                            relation_type="uses_model",
                            created_at=now,
                        )
                    )
