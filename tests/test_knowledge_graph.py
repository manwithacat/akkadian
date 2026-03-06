import pytest
from akkadian_mcp.knowledge_graph.models import Entity, Relation
from akkadian_mcp.knowledge_graph.store import KnowledgeGraph


@pytest.fixture
def kg():
    g = KnowledgeGraph(":memory:")
    yield g
    g.close()


def test_upsert_and_get_entity(kg):
    e = Entity(id="model:test-v1", name="Test Model")
    kg.upsert_entity(e)
    result = kg.get_entity("model:test-v1")
    assert result is not None
    assert result.name == "Test Model"


def test_get_missing_entity(kg):
    assert kg.get_entity("model:nonexistent") is None


def test_list_entities_by_type(kg):
    kg.upsert_entity(Entity(id="model:a", name="A"))
    kg.upsert_entity(Entity(id="model:b", name="B"))
    kg.upsert_entity(Entity(id="kernel:c", name="C"))
    assert len(kg.list_entities("model")) == 2
    assert len(kg.list_entities("kernel")) == 1


def test_entity_type_from_prefix():
    assert Entity(id="model:test", name="T").entity_type == "model"
    assert Entity(id="kernel:test", name="T").entity_type == "kernel"


def test_add_and_get_relations(kg):
    kg.upsert_entity(Entity(id="kernel:k1", name="K1"))
    kg.upsert_entity(Entity(id="model:m1", name="M1"))
    kg.add_relation(
        Relation(source_id="kernel:k1", target_id="model:m1", relation_type="uses_model")
    )
    rels = kg.get_relations("kernel:k1")
    assert len(rels) == 1
    assert rels[0].relation_type == "uses_model"


def test_search(kg):
    kg.upsert_entity(Entity(id="model:byt5-large", name="ByT5 Large Model"))
    results = kg.search("byt5")
    assert len(results) == 1
    assert results[0].id == "model:byt5-large"


def test_neighbourhood(kg):
    kg.upsert_entity(Entity(id="model:m1", name="M1"))
    kg.upsert_entity(Entity(id="kernel:k1", name="K1"))
    kg.upsert_entity(Entity(id="score:s1", name="S1"))
    kg.add_relation(
        Relation(source_id="kernel:k1", target_id="model:m1", relation_type="uses_model")
    )
    kg.add_relation(
        Relation(source_id="score:s1", target_id="kernel:k1", relation_type="produced_by")
    )
    hood = kg.get_neighbourhood("kernel:k1", depth=1)
    assert len(hood["entities"]) == 3


def test_record_score(kg):
    kg.upsert_entity(Entity(id="model:m1", name="M1"))
    kg.record_score("model:m1", "bleu", 42.5, "local")
    scores = kg.list_entities("score")
    assert len(scores) == 1
    assert scores[0].metadata["value"] == 42.5


def test_get_stats(kg):
    kg.upsert_entity(Entity(id="model:m1", name="M1"))
    kg.upsert_entity(Entity(id="kernel:k1", name="K1"))
    stats = kg.get_stats()
    assert stats["entities"] == 2
    assert stats["types"]["model"] == 1


def test_delete_entity(kg):
    kg.upsert_entity(Entity(id="model:m1", name="M1"))
    assert kg.delete_entity("model:m1")
    assert kg.get_entity("model:m1") is None
