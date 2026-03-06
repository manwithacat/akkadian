"""Knowledge graph query handlers."""

from __future__ import annotations

import json
from time import time

from ..knowledge_graph.models import Entity, Relation
from ..state import get_state


def handle_knowledge(arguments: dict) -> str:
    """Dispatch knowledge operations."""
    op = arguments.get("operation", "status")
    ops = {
        "status": _graph_status,
        "search": _search,
        "entity": _get_entity,
        "neighbourhood": _neighbourhood,
        "list": _list_type,
        "record_score": _record_score,
        "add_entity": _add_entity,
        "add_relation": _add_relation,
    }
    handler = ops.get(op)
    if handler is None:
        return f"Unknown knowledge operation: {op}. Available: {', '.join(ops)}"
    return handler(arguments)


def _graph_status(arguments: dict) -> str:
    kg = get_state().knowledge_graph
    stats = kg.get_stats()
    lines = [
        f"Knowledge Graph: {stats['entities']} entities, {stats['relations']} relations",
        "Entity types:",
    ]
    for etype, count in stats["types"].items():
        lines.append(f"  {etype}: {count}")
    return "\n".join(lines)


def _search(arguments: dict) -> str:
    query = arguments.get("query", "")
    if not query:
        return "Provide a 'query' parameter to search."
    kg = get_state().knowledge_graph
    results = kg.search(query)
    if not results:
        return f"No results for '{query}'."
    lines = [f"Search results for '{query}':"]
    for e in results:
        lines.append(f"  [{e.entity_type}] {e.id}: {e.name}")
        if e.metadata:
            summary = {k: v for k, v in e.metadata.items() if k != "path"}
            if summary:
                lines.append(f"    {json.dumps(summary, default=str)}")
    return "\n".join(lines)


def _get_entity(arguments: dict) -> str:
    entity_id = arguments.get("id", "")
    if not entity_id:
        return "Provide an 'id' parameter."
    kg = get_state().knowledge_graph
    entity = kg.get_entity(entity_id)
    if entity is None:
        return f"Entity not found: {entity_id}"
    relations = kg.get_relations(entity_id)
    lines = [
        f"Entity: {entity.id}",
        f"  Name: {entity.name}",
        f"  Type: {entity.entity_type}",
        f"  Metadata: {json.dumps(entity.metadata, indent=2, default=str)}",
    ]
    if relations:
        lines.append("  Relations:")
        for r in relations:
            if r.source_id == entity_id:
                lines.append(f"    -> {r.relation_type} -> {r.target_id}")
            else:
                lines.append(f"    <- {r.relation_type} <- {r.source_id}")
    return "\n".join(lines)


def _neighbourhood(arguments: dict) -> str:
    entity_id = arguments.get("id", "")
    depth = int(arguments.get("depth", 1))
    if not entity_id:
        return "Provide an 'id' parameter."
    kg = get_state().knowledge_graph
    result = kg.get_neighbourhood(entity_id, depth=depth)
    lines = [f"Neighbourhood of {entity_id} (depth={depth}):"]
    lines.append(f"  {len(result['entities'])} entities, {len(result['relations'])} relations")
    for e in result["entities"]:
        lines.append(f"  [{e.entity_type}] {e.id}: {e.name}")
    return "\n".join(lines)


def _list_type(arguments: dict) -> str:
    entity_type = arguments.get("type", "")
    if not entity_type:
        return "Provide a 'type' parameter (model, kernel, dataset, experiment, finding, score)."
    kg = get_state().knowledge_graph
    entities = kg.list_entities(entity_type)
    if not entities:
        return f"No {entity_type} entities found."
    lines = [f"{entity_type} entities ({len(entities)}):"]
    for e in entities:
        lines.append(f"  {e.id}: {e.name}")
    return "\n".join(lines)


def _record_score(arguments: dict) -> str:
    entity_id = arguments.get("entity_id", "")
    metric = arguments.get("metric", "")
    value = arguments.get("value")
    source = arguments.get("source", "local")
    if not entity_id or not metric or value is None:
        return "Required: entity_id, metric, value"
    kg = get_state().knowledge_graph
    kg.record_score(entity_id, metric, float(value), source)
    return f"Recorded {metric}={value} for {entity_id}"


def _add_entity(arguments: dict) -> str:
    entity_id = arguments.get("id", "")
    name = arguments.get("name", "")
    metadata = arguments.get("metadata", {})
    if not entity_id or not name:
        return "Required: id (prefixed, e.g. 'model:my-model'), name"
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return f"Invalid metadata JSON: {metadata}"
    kg = get_state().knowledge_graph
    now = time()
    kg.upsert_entity(
        Entity(
            id=entity_id,
            name=name,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )
    )
    return f"Entity upserted: {entity_id}"


def _add_relation(arguments: dict) -> str:
    source = arguments.get("source_id", "")
    target = arguments.get("target_id", "")
    rel_type = arguments.get("relation_type", "")
    if not source or not target or not rel_type:
        return "Required: source_id, target_id, relation_type"
    kg = get_state().knowledge_graph
    kg.add_relation(
        Relation(
            source_id=source,
            target_id=target,
            relation_type=rel_type,
            created_at=time(),
        )
    )
    return f"Relation added: {source} -> {rel_type} -> {target}"
