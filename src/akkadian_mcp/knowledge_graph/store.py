"""SQLite-backed knowledge graph store."""

from __future__ import annotations

import json
import sqlite3
from time import time

from .models import Entity, Relation


class KnowledgeGraph:
    """SQLite knowledge graph with WAL mode for safe concurrent access."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._persistent_conn: sqlite3.Connection | None = None

        if self._is_memory:
            self._persistent_conn = self._create_connection()
            self._init_schema(self._persistent_conn)
        else:
            conn = self._create_connection()
            try:
                self._init_schema(conn)
            finally:
                conn.close()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if not self._is_memory:
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _conn(self) -> sqlite3.Connection:
        if self._is_memory and self._persistent_conn:
            return self._persistent_conn
        return self._create_connection()

    def _close(self, conn: sqlite3.Connection) -> None:
        if not self._is_memory:
            conn.close()

    def close(self) -> None:
        if self._persistent_conn:
            self._persistent_conn.close()
            self._persistent_conn = None

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS relations (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, relation_type),
                FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
            CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
        """)

    def upsert_entity(self, entity: Entity) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO entities (id, name, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     name = excluded.name,
                     metadata = excluded.metadata,
                     updated_at = excluded.updated_at""",
                (
                    entity.id,
                    entity.name,
                    json.dumps(entity.metadata),
                    entity.created_at,
                    entity.updated_at,
                ),
            )
            conn.commit()
        finally:
            self._close(conn)

    def get_entity(self, entity_id: str) -> Entity | None:
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if row is None:
                return None
            return Entity(
                id=row["id"],
                name=row["name"],
                metadata=json.loads(row["metadata"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        finally:
            self._close(conn)

    def list_entities(self, entity_type: str | None = None) -> list[Entity]:
        conn = self._conn()
        try:
            if entity_type:
                rows = conn.execute(
                    "SELECT * FROM entities WHERE id LIKE ? ORDER BY updated_at DESC",
                    (f"{entity_type}:%",),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM entities ORDER BY updated_at DESC").fetchall()
            return [
                Entity(
                    id=r["id"],
                    name=r["name"],
                    metadata=json.loads(r["metadata"]),
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]
        finally:
            self._close(conn)

    def delete_entity(self, entity_id: str) -> bool:
        conn = self._conn()
        try:
            cursor = conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            self._close(conn)

    def add_relation(self, relation: Relation) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO relations
                   (source_id, target_id, relation_type, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    relation.source_id,
                    relation.target_id,
                    relation.relation_type,
                    json.dumps(relation.metadata),
                    relation.created_at,
                ),
            )
            conn.commit()
        finally:
            self._close(conn)

    def get_relations(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: str | None = None,
    ) -> list[Relation]:
        conn = self._conn()
        try:
            results = []
            type_filter = "AND relation_type = ?" if relation_type else ""
            base_params: list = [entity_id]
            if relation_type:
                base_params.append(relation_type)

            if direction in ("out", "both"):
                rows = conn.execute(
                    f"SELECT * FROM relations WHERE source_id = ? {type_filter}",
                    base_params,
                ).fetchall()
                results.extend(self._rows_to_relations(rows))

            if direction in ("in", "both"):
                rows = conn.execute(
                    f"SELECT * FROM relations WHERE target_id = ? {type_filter}",
                    base_params,
                ).fetchall()
                results.extend(self._rows_to_relations(rows))

            return results
        finally:
            self._close(conn)

    def _rows_to_relations(self, rows: list) -> list[Relation]:
        return [
            Relation(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation_type=r["relation_type"],
                metadata=json.loads(r["metadata"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_neighbourhood(self, entity_id: str, depth: int = 1) -> dict:
        """Get entities within N hops using iterative BFS."""
        visited: set[str] = {entity_id}
        all_relations: list[Relation] = []
        frontier = {entity_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for eid in frontier:
                rels = self.get_relations(eid)
                for rel in rels:
                    all_relations.append(rel)
                    neighbor = rel.target_id if rel.source_id == eid else rel.source_id
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier

        entities = [self.get_entity(eid) for eid in visited]
        return {
            "center": entity_id,
            "entities": [e for e in entities if e is not None],
            "relations": all_relations,
        }

    def search(self, query: str) -> list[Entity]:
        """Full-text search across entity names and metadata."""
        conn = self._conn()
        try:
            pattern = f"%{query}%"
            rows = conn.execute(
                """SELECT * FROM entities
                   WHERE name LIKE ? OR metadata LIKE ?
                   ORDER BY updated_at DESC LIMIT 20""",
                (pattern, pattern),
            ).fetchall()
            return [
                Entity(
                    id=r["id"],
                    name=r["name"],
                    metadata=json.loads(r["metadata"]),
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]
        finally:
            self._close(conn)

    def get_stats(self) -> dict:
        """Get graph statistics."""
        conn = self._conn()
        try:
            entity_count = conn.execute("SELECT COUNT(*) as c FROM entities").fetchone()["c"]
            relation_count = conn.execute("SELECT COUNT(*) as c FROM relations").fetchone()["c"]

            type_counts = {}
            for row in conn.execute(
                """SELECT substr(id, 1, instr(id, ':') - 1) as etype, COUNT(*) as c
                   FROM entities WHERE id LIKE '%:%'
                   GROUP BY etype ORDER BY c DESC"""
            ).fetchall():
                type_counts[row["etype"]] = row["c"]

            return {"entities": entity_count, "relations": relation_count, "types": type_counts}
        finally:
            self._close(conn)

    def record_score(
        self,
        entity_id: str,
        metric: str,
        value: float,
        source: str | None = None,
    ) -> None:
        """Record a metric score for an entity."""
        now = time()
        score_id = f"score:{entity_id.split(':', 1)[-1]}-{metric}"
        self.upsert_entity(
            Entity(
                id=score_id,
                name=f"{metric} for {entity_id}",
                metadata={"metric": metric, "value": value, "source": source or "local"},
                created_at=now,
                updated_at=now,
            )
        )
        self.add_relation(
            Relation(
                source_id=entity_id,
                target_id=score_id,
                relation_type="produced_by",
                created_at=now,
            )
        )
