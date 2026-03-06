import pytest

mlflow = pytest.importorskip("mlflow")

from akkadian_mcp.state import ServerState
from akkadian_mcp.handlers.mlflow import handle_mlflow


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


def test_setup_creates_experiment(initialized):
    result = handle_mlflow(
        {
            "operation": "setup",
            "experiment_name": "test-experiment",
        }
    )
    assert "test-experiment" in result


def test_experiments_lists(initialized):
    handle_mlflow({"operation": "setup", "experiment_name": "exp1"})
    result = handle_mlflow({"operation": "experiments"})
    assert "exp1" in result


def test_best_no_runs(initialized):
    handle_mlflow({"operation": "setup", "experiment_name": "empty"})
    result = handle_mlflow(
        {
            "operation": "best",
            "experiment_name": "empty",
            "metric": "loss",
        }
    )
    assert "no runs" in result.lower()
