from unittest.mock import MagicMock, patch

import pytest

from akkadian_mcp.handlers.kaggle import handle_kaggle
from akkadian_mcp.state import ServerState


@pytest.fixture
def initialized(tmp_path):
    state = ServerState()
    state.init(tmp_path)
    state.update_config({"kaggle_username": "testuser", "competition": "titanic"})
    import akkadian_mcp.state as state_mod

    old = state_mod._state
    state_mod._state = state
    yield tmp_path
    state.reset()
    state_mod._state = old


def test_unknown_operation(initialized):
    result = handle_kaggle({"operation": "invalid"})
    assert "unknown" in result.lower()


@patch("akkadian_mcp.handlers.kaggle.subprocess")
def test_status_with_slug(mock_subprocess, initialized):
    mock_subprocess.run.return_value = MagicMock(stdout="running", stderr="", returncode=0)
    result = handle_kaggle({"operation": "status", "slug": "my-kernel"})
    assert "running" in result
    call_args = mock_subprocess.run.call_args[0][0]
    assert "testuser/my-kernel" in " ".join(call_args)


@patch("akkadian_mcp.handlers.kaggle.subprocess")
def test_submissions_uses_config_competition(mock_subprocess, initialized):
    mock_subprocess.run.return_value = MagicMock(
        stdout="score,date\n0.95,2026-01-01", stderr="", returncode=0
    )
    handle_kaggle({"operation": "submissions"})
    call_args = mock_subprocess.run.call_args[0][0]
    assert "titanic" in " ".join(call_args)


def test_status_no_slug_no_kernels(initialized):
    result = handle_kaggle({"operation": "status"})
    assert "no" in result.lower()
