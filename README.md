# Akkadian

MCP server for ML competition workflows. Provides AI-agent-friendly tooling for Kaggle competitions with integrated experiment tracking (MLflow) and hyperparameter tuning (Optuna).

## Install

```bash
pip install akkadian                    # Core (Kaggle + knowledge graph)
pip install akkadian[mlflow]            # + MLflow tracking
pip install akkadian[optuna]            # + Optuna tuning
pip install akkadian[all]               # Everything
```

## Quick Start

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "akkadian": {
      "type": "stdio",
      "command": "akkadian-mcp",
      "args": ["--working-dir", "."]
    }
  }
}
```

Then use the `bootstrap init` tool to configure your project.

## Tools

| Tool | Operations | Requires |
|------|-----------|----------|
| `bootstrap` | init, status, update_config | core |
| `kaggle` | status, list_kernels, submissions, push_kernel, download_output | core |
| `knowledge` | status, search, entity, neighbourhood, list, record_score, add_entity, add_relation | core |
| `mlflow` | setup, experiments, runs, compare, best, ingest_artifact, suggest | `akkadian[mlflow]` |
| `optuna` | create_study, studies, trials, best, importance, suggest_space, prune_config | `akkadian[optuna]` |

## How It Works

Akkadian stores all state in a `.akkadian/` directory in your project root:

- `config.json` — project configuration (competition, username, goal metric)
- `knowledge.db` — SQLite knowledge graph tracking models, kernels, datasets, scores
- `mlflow.db` — MLflow experiment tracking (if installed)
- `optuna.db` — Optuna study storage (if installed)

The agent adapts to your workflow — it observes, tracks artifacts, and suggests strategies without enforcing a specific process.

## Development

```bash
git clone https://github.com/manwithacat/akkadian.git
cd akkadian
pip install -e ".[dev,all]"
pytest
```

## License

MIT
