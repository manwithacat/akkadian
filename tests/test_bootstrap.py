import json

import pytest

from akkadian_mcp.handlers.bootstrap import handle_bootstrap
from akkadian_mcp.state import ServerState


@pytest.fixture
def initialized(tmp_path):
    state = ServerState()
    state.init(tmp_path)
    # Patch the module-level singleton for this test
    import akkadian_mcp.state as state_mod

    old = state_mod._state
    state_mod._state = state
    yield tmp_path
    state.reset()
    state_mod._state = old


def test_init_creates_project(initialized):
    result = handle_bootstrap({"operation": "init"})
    assert "ready" in result.lower() or "initialized" in result.lower()
    assert (initialized / ".akkadian" / "config.json").exists()


def test_status_shows_config(initialized):
    result = handle_bootstrap({"operation": "status"})
    assert "version" in result
    assert "platform" in result


def test_update_config(initialized):
    result = handle_bootstrap(
        {
            "operation": "update_config",
            "config": {"competition": "titanic", "goal_metric": "accuracy"},
        }
    )
    assert "titanic" in result
    config = json.loads((initialized / ".akkadian" / "config.json").read_text())
    assert config["competition"] == "titanic"


def test_init_idempotent(initialized):
    handle_bootstrap({"operation": "init"})
    handle_bootstrap(
        {
            "operation": "update_config",
            "config": {"competition": "titanic"},
        }
    )
    handle_bootstrap({"operation": "init"})
    config = json.loads((initialized / ".akkadian" / "config.json").read_text())
    assert config["competition"] == "titanic"
