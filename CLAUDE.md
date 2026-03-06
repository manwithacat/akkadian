# Akkadian — Agent Instructions

## What This Is

Akkadian is an MCP server for ML competition workflows. It provides tools for Kaggle interaction, experiment tracking (MLflow), hyperparameter tuning (Optuna), and a knowledge graph for tracking artifacts.

## Running Tests

```bash
pytest                           # All tests
pytest tests/test_server.py -v   # Specific file
pytest -k "mlflow" -v            # By keyword
```

## Package Structure

- `src/akkadian_mcp/server.py` — MCP server, tool registration
- `src/akkadian_mcp/state.py` — Singleton state, config, feature detection
- `src/akkadian_mcp/handlers/` — One file per tool (bootstrap, kaggle, knowledge, mlflow, optuna)
- `src/akkadian_mcp/knowledge_graph/` — SQLite-backed entity/relation store
- `tests/` — Pytest tests, one per handler

## Key Patterns

- Tools use operation enums — one MCP tool per handler, operations dispatched internally
- Optional extras detected via `importlib.util.find_spec()` — tools only registered if deps available
- State is a module-level singleton (`get_state()`)
- Knowledge graph uses ID prefix convention: `model:`, `kernel:`, `dataset:`, `score:`, etc.
- Config stored in `.akkadian/config.json`

## Adding a New Handler

1. Create `src/akkadian_mcp/handlers/newhandler.py` with `handle_newhandler(arguments: dict) -> str`
2. Add tool definition in `server.py` (`_newtool_tool()` function)
3. Add dispatch case in `_dispatch()`
4. Add to `build_tools()` (with `has_extra()` guard if optional)
5. Write tests in `tests/test_newhandler.py`
