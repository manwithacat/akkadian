"""MLflow handler — experiment tracking, artifact ingestion, suggestions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..state import get_state

logger = logging.getLogger(__name__)


def _get_tracking_uri() -> str:
    state = get_state()
    db_path = state.project_root / ".akkadian" / "mlflow.db"
    return f"sqlite:///{db_path}"


def _ensure_tracking():
    import mlflow

    mlflow.set_tracking_uri(_get_tracking_uri())


def handle_mlflow(arguments: dict) -> str:
    """Dispatch MLflow operations."""
    op = arguments.get("operation", "experiments")
    ops = {
        "setup": _setup,
        "experiments": _experiments,
        "runs": _runs,
        "compare": _compare,
        "best": _best,
        "ingest_artifact": _ingest_artifact,
        "suggest": _suggest,
    }
    handler = ops.get(op)
    if handler is None:
        return f"Unknown mlflow operation: {op}. Available: {', '.join(ops)}"
    return handler(arguments)


def _setup(arguments: dict) -> str:
    import mlflow

    _ensure_tracking()
    name = arguments.get("experiment_name", "default")
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        exp_id = mlflow.create_experiment(name)
        return f"Created experiment '{name}' (id: {exp_id})"
    return f"Experiment '{name}' exists (id: {experiment.experiment_id})"


def _experiments(arguments: dict) -> str:
    import mlflow

    _ensure_tracking()
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    if not experiments:
        return "No experiments found. Use mlflow setup to create one."
    lines = ["Experiments:"]
    for exp in experiments:
        run_count = len(client.search_runs(exp.experiment_id))
        lines.append(f"  {exp.name} (id: {exp.experiment_id}, runs: {run_count})")
    return "\n".join(lines)


def _runs(arguments: dict) -> str:
    import mlflow

    _ensure_tracking()
    client = mlflow.tracking.MlflowClient()
    experiment_name = arguments.get("experiment_name")
    filter_string = arguments.get("filter", "")
    max_results = int(arguments.get("max_results", 20))

    experiment_ids = []
    if experiment_name:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return f"Experiment '{experiment_name}' not found."
        experiment_ids = [exp.experiment_id]
    else:
        experiment_ids = [e.experiment_id for e in client.search_experiments()]

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        max_results=max_results,
        order_by=["start_time DESC"],
    )
    if not runs:
        return "No runs found."
    lines = [f"Runs ({len(runs)}):"]
    for run in runs:
        params_str = ", ".join(f"{k}={v}" for k, v in list(run.data.params.items())[:5])
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in list(run.data.metrics.items())[:5])
        lines.append(f"  {run.info.run_id[:8]} | {run.info.status} | {params_str}")
        if metrics_str:
            lines.append(f"    metrics: {metrics_str}")
    return "\n".join(lines)


def _compare(arguments: dict) -> str:
    import mlflow

    _ensure_tracking()
    client = mlflow.tracking.MlflowClient()
    run_ids = arguments.get("run_ids", [])
    if len(run_ids) < 2:
        return "Provide at least 2 run_ids to compare."

    runs = []
    for rid in run_ids:
        try:
            runs.append(client.get_run(rid))
        except Exception:
            return f"Run not found: {rid}"

    all_params = set()
    all_metrics = set()
    for run in runs:
        all_params.update(run.data.params.keys())
        all_metrics.update(run.data.metrics.keys())

    lines = ["Run Comparison:", ""]
    lines.append("Params:")
    for param in sorted(all_params):
        values = [run.data.params.get(param, "-") for run in runs]
        lines.append(f"  {param}: {' | '.join(values)}")
    lines.append("")
    lines.append("Metrics:")
    for metric in sorted(all_metrics):
        values = []
        for run in runs:
            v = run.data.metrics.get(metric)
            values.append(f"{v:.4f}" if v is not None else "-")
        lines.append(f"  {metric}: {' | '.join(values)}")
    return "\n".join(lines)


def _best(arguments: dict) -> str:
    import mlflow

    _ensure_tracking()
    client = mlflow.tracking.MlflowClient()
    experiment_name = arguments.get("experiment_name", "default")
    metric = arguments.get("metric", "")
    if not metric:
        return "Provide 'metric' to find best run."
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return f"Experiment '{experiment_name}' not found."
    state = get_state()
    direction = state.get_config().get("goal_direction", "maximize")
    order = f"metrics.{metric} {'DESC' if direction == 'maximize' else 'ASC'}"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[order],
        max_results=1,
    )
    if not runs:
        return f"No runs found in experiment '{experiment_name}'."
    best = runs[0]
    lines = [
        f"Best run for {metric} ({direction}):",
        f"  Run ID: {best.info.run_id}",
        f"  {metric}: {best.data.metrics.get(metric, 'N/A')}",
        f"  Params: {json.dumps(best.data.params, indent=2)}",
        "  All metrics: {}".format(
            json.dumps(
                {k: round(v, 4) for k, v in best.data.metrics.items()},
                indent=2,
            )
        ),
    ]
    return "\n".join(lines)


def _ingest_artifact(arguments: dict) -> str:
    import mlflow

    _ensure_tracking()
    path = arguments.get("path", "")
    if not path:
        return "Provide 'path' to directory with training artifacts."
    artifact_dir = Path(path)
    if not artifact_dir.exists():
        return f"Directory not found: {path}"
    experiment_name = arguments.get("experiment_name", "default")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=artifact_dir.name) as run:
        for config_file in artifact_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    config = json.load(f)
                if isinstance(config, dict):
                    for k, v in config.items():
                        if isinstance(v, (str, int, float, bool)):
                            mlflow.log_param(f"{config_file.stem}.{k}", v)
            except (json.JSONDecodeError, OSError):
                pass
        mlflow.log_artifacts(str(artifact_dir))
        lines = [
            f"Artifacts ingested to run {run.info.run_id}",
            f"  Experiment: {experiment_name}",
            f"  Source: {artifact_dir}",
            f"  Files logged: {sum(1 for _ in artifact_dir.rglob('*') if _.is_file())}",
        ]
    return "\n".join(lines)


def _suggest(arguments: dict) -> str:
    import mlflow

    _ensure_tracking()
    client = mlflow.tracking.MlflowClient()
    experiment_name = arguments.get("experiment_name", "default")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return f"Experiment '{experiment_name}' not found."
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=50,
    )
    if not runs:
        return "No runs to analyze. Run some experiments first."

    lines = [f"Analysis of {len(runs)} runs in '{experiment_name}':"]

    all_metrics: dict[str, list[float]] = {}
    for run in runs:
        for k, v in run.data.metrics.items():
            all_metrics.setdefault(k, []).append(v)

    for k, values in sorted(all_metrics.items()):
        lines.append(
            f"  {k}: min={min(values):.4f}, max={max(values):.4f}, "
            f"mean={sum(values) / len(values):.4f}, n={len(values)}"
        )

    all_params: dict[str, set[str]] = {}
    for run in runs:
        for k, v in run.data.params.items():
            all_params.setdefault(k, set()).add(v)

    varied = {k: v for k, v in all_params.items() if len(v) > 1}
    if varied:
        lines.append("")
        lines.append("Parameters that varied across runs:")
        for k, values in sorted(varied.items()):
            lines.append(f"  {k}: {', '.join(sorted(values))}")

    fixed = {k: v for k, v in all_params.items() if len(v) == 1}
    if fixed:
        lines.append("")
        lines.append("Fixed parameters (consider varying):")
        for k, values in sorted(fixed.items()):
            lines.append(f"  {k}: {next(iter(values))}")

    return "\n".join(lines)
