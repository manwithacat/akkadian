import pytest

optuna = pytest.importorskip("optuna")

from akkadian_mcp.handlers.optuna import handle_optuna  # noqa: E402
from akkadian_mcp.state import ServerState  # noqa: E402


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


def test_create_study(initialized):
    result = handle_optuna(
        {
            "operation": "create_study",
            "study_name": "test-study",
            "direction": "minimize",
        }
    )
    assert "test-study" in result


def test_studies_lists(initialized):
    handle_optuna({"operation": "create_study", "study_name": "s1"})
    result = handle_optuna({"operation": "studies"})
    assert "s1" in result


def test_best_no_trials(initialized):
    handle_optuna({"operation": "create_study", "study_name": "empty"})
    result = handle_optuna({"operation": "best", "study_name": "empty"})
    assert "no" in result.lower()


def test_trials_empty(initialized):
    handle_optuna({"operation": "create_study", "study_name": "empty2"})
    result = handle_optuna({"operation": "trials", "study_name": "empty2"})
    assert "no trials" in result.lower()
