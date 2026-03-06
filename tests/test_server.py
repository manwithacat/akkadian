from unittest.mock import patch

from akkadian_mcp.server import build_tools


def test_core_tools_always_present():
    tools = build_tools()
    names = {t.name for t in tools}
    assert "bootstrap" in names
    assert "kaggle" in names
    assert "knowledge" in names


def test_mlflow_tool_absent_without_extra():
    with patch("akkadian_mcp.server.has_extra", lambda pkg: False):
        tools = build_tools()
        assert "mlflow" not in {t.name for t in tools}


def test_mlflow_tool_present_with_extra():
    with patch("akkadian_mcp.server.has_extra", lambda pkg: pkg == "mlflow"):
        tools = build_tools()
        assert "mlflow" in {t.name for t in tools}


def test_optuna_tool_absent_without_extra():
    with patch("akkadian_mcp.server.has_extra", lambda pkg: False):
        tools = build_tools()
        assert "optuna" not in {t.name for t in tools}


def test_optuna_tool_present_with_extra():
    with patch("akkadian_mcp.server.has_extra", lambda pkg: pkg == "optuna"):
        tools = build_tools()
        assert "optuna" in {t.name for t in tools}
