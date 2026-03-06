import pytest

from akkadian_mcp.handlers.knowledge import handle_knowledge
from akkadian_mcp.state import ServerState


@pytest.fixture
def initialized(tmp_path):
    state = ServerState()
    state.init(tmp_path)
    import akkadian_mcp.state as state_mod

    old = state_mod._state
    state_mod._state = state
    yield tmp_path
    state.reset()
    state_mod._state = old


def test_status(initialized):
    result = handle_knowledge({"operation": "status"})
    assert "entities" in result.lower()


def test_add_and_search(initialized):
    handle_knowledge(
        {
            "operation": "add_entity",
            "id": "model:test-v1",
            "name": "Test Model v1",
        }
    )
    result = handle_knowledge({"operation": "search", "query": "test"})
    assert "test-v1" in result.lower()


def test_record_and_list_scores(initialized):
    handle_knowledge(
        {
            "operation": "add_entity",
            "id": "model:m1",
            "name": "Model 1",
        }
    )
    handle_knowledge(
        {
            "operation": "record_score",
            "entity_id": "model:m1",
            "metric": "bleu",
            "value": 42.5,
            "source": "local",
        }
    )
    result = handle_knowledge({"operation": "list", "type": "score"})
    assert "bleu" in result.lower()
