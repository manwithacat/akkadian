"""Data models for the knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass
class Entity:
    """A node in the knowledge graph.

    ID convention uses prefixes for type inference:
      model:my-model-v1
      kernel:my-kernel-v1
      dataset:my-dataset
      experiment:exp-v1
      score:model-bleu
    """

    id: str
    name: str
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    @property
    def entity_type(self) -> str:
        """Infer type from ID prefix."""
        prefix = self.id.split(":", 1)[0] if ":" in self.id else "unknown"
        return TYPE_PREFIXES.get(f"{prefix}:", "unknown")


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    relation_type: str
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time)


TYPE_PREFIXES = {
    "model:": "model",
    "kernel:": "kernel",
    "dataset:": "dataset",
    "experiment:": "experiment",
    "submission:": "submission",
    "score:": "score",
    "config:": "config",
    "finding:": "finding",
    "error:": "error",
}

RELATION_TYPES = {
    "trained_on": "Model was trained on dataset",
    "submitted_as": "Model was submitted as kernel",
    "uses_model": "Kernel uses a model",
    "derived_from": "Model derived from base model",
    "produced_by": "Score produced by experiment",
    "depends_on": "Generic dependency",
    "supersedes": "Newer version supersedes older",
    "related_to": "Generic relation",
}
