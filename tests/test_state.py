import pytest
from pathlib import Path
from akkadian_mcp.state import ServerState, has_extra


@pytest.fixture
def state(tmp_path):
    s = ServerState()
    s.init(tmp_path)
    yield s
    s.reset()


def test_init_creates_akkadian_dir(state, tmp_path):
    assert (tmp_path / ".akkadian").is_dir()
    assert (tmp_path / ".akkadian" / "knowledge.db").exists()


def test_knowledge_graph_accessible(state):
    assert state.knowledge_graph is not None
    stats = state.knowledge_graph.get_stats()
    assert stats["entities"] >= 0


def test_project_config_default(state):
    config = state.get_config()
    assert config["version"] == "0.2.0"


def test_update_config(state):
    state.update_config({"competition": "my-comp", "platform": "kaggle"})
    config = state.get_config()
    assert config["competition"] == "my-comp"


def test_has_extra():
    assert has_extra("json")
    assert not has_extra("nonexistent_package_xyz")
