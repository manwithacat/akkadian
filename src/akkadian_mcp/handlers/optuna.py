"""Optuna handler — study management, analysis, search space suggestions."""

from __future__ import annotations

import json
import logging

from ..state import get_state

logger = logging.getLogger(__name__)


def _get_storage_url() -> str:
    state = get_state()
    db_path = state.project_root / ".akkadian" / "optuna.db"
    return f"sqlite:///{db_path}"


def handle_optuna(arguments: dict) -> str:
    """Dispatch Optuna operations."""
    op = arguments.get("operation", "studies")
    ops = {
        "create_study": _create_study,
        "studies": _studies,
        "trials": _trials,
        "best": _best,
        "importance": _importance,
        "suggest_space": _suggest_space,
        "prune_config": _prune_config,
    }
    handler = ops.get(op)
    if handler is None:
        return f"Unknown optuna operation: {op}. Available: {', '.join(ops)}"
    return handler(arguments)


def _create_study(arguments: dict) -> str:
    import optuna

    name = arguments.get("study_name", "")
    if not name:
        return "Provide 'study_name'."
    direction = arguments.get("direction", "minimize")
    sampler_name = arguments.get("sampler", "tpe")
    sampler_map = {
        "tpe": optuna.samplers.TPESampler,
        "random": optuna.samplers.RandomSampler,
        "cmaes": optuna.samplers.CmaEsSampler,
    }
    sampler_cls = sampler_map.get(sampler_name, optuna.samplers.TPESampler)
    study = optuna.create_study(
        study_name=name,
        direction=direction,
        storage=_get_storage_url(),
        sampler=sampler_cls(),
        load_if_exists=True,
    )
    return (
        f"Study '{name}' ready\n"
        f"  Direction: {direction}\n"
        f"  Sampler: {sampler_name}\n"
        f"  Trials: {len(study.trials)}\n"
        f"  Storage: {_get_storage_url()}"
    )


def _studies(arguments: dict) -> str:
    import optuna

    try:
        summaries = optuna.study.get_all_study_summaries(storage=_get_storage_url())
    except Exception:
        return "No studies found. Use optuna create_study first."
    if not summaries:
        return "No studies found."
    lines = ["Studies:"]
    for s in summaries:
        best_str = ""
        if s.best_trial is not None:
            best_str = f", best={s.best_trial.value:.4f}"
        lines.append(f"  {s.study_name}: {s.n_trials} trials{best_str} [{s.direction.name}]")
    return "\n".join(lines)


def _trials(arguments: dict) -> str:
    import optuna

    name = arguments.get("study_name", "")
    if not name:
        return "Provide 'study_name'."
    try:
        study = optuna.load_study(study_name=name, storage=_get_storage_url())
    except KeyError:
        return f"Study '{name}' not found."
    trials = study.trials
    if not trials:
        return f"No trials in study '{name}'."
    lines = [f"Trials for '{name}' ({len(trials)}):"]
    for t in trials[-20:]:
        params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
        value_str = f"{t.value:.4f}" if t.value is not None else "N/A"
        lines.append(f"  #{t.number}: {value_str} [{t.state.name}] {params_str}")
    return "\n".join(lines)


def _best(arguments: dict) -> str:
    import optuna

    name = arguments.get("study_name", "")
    if not name:
        return "Provide 'study_name'."
    try:
        study = optuna.load_study(study_name=name, storage=_get_storage_url())
    except KeyError:
        return f"Study '{name}' not found."
    if not study.trials:
        return f"No completed trials in study '{name}'."
    try:
        best = study.best_trial
    except ValueError:
        return f"No completed trials in study '{name}'."
    lines = [
        f"Best trial for '{name}':",
        f"  Trial #{best.number}",
        f"  Value: {best.value:.4f}",
        f"  Params: {json.dumps(best.params, indent=2, default=str)}",
    ]
    return "\n".join(lines)


def _importance(arguments: dict) -> str:
    import optuna

    name = arguments.get("study_name", "")
    if not name:
        return "Provide 'study_name'."
    try:
        study = optuna.load_study(study_name=name, storage=_get_storage_url())
    except KeyError:
        return f"Study '{name}' not found."
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        return "Need at least 2 completed trials for importance analysis."
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        return f"Cannot compute importance: {e}"
    lines = [f"Parameter importance for '{name}':"]
    for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        lines.append(f"  {param}: {imp:.3f} {bar}")
    return "\n".join(lines)


def _suggest_space(arguments: dict) -> str:
    import optuna

    name = arguments.get("study_name", "")
    if not name:
        return "Provide 'study_name'."
    try:
        study = optuna.load_study(study_name=name, storage=_get_storage_url())
    except KeyError:
        return f"Study '{name}' not found."
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 5:
        return f"Need at least 5 completed trials for suggestions (have {len(completed)})."

    lines = [f"Search space analysis for '{name}' ({len(completed)} trials):"]

    values = [t.value for t in completed if t.value is not None]
    if not values:
        return "No trial values to analyze."

    is_minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    threshold = (
        sorted(values)[len(values) // 4]
        if is_minimize
        else sorted(values, reverse=True)[len(values) // 4]
    )

    top_trials = [
        t
        for t in completed
        if t.value is not None and (t.value <= threshold if is_minimize else t.value >= threshold)
    ]

    all_params: dict[str, list] = {}
    for t in completed:
        for k, v in t.params.items():
            all_params.setdefault(k, []).append(v)

    top_params: dict[str, list] = {}
    for t in top_trials:
        for k, v in t.params.items():
            top_params.setdefault(k, []).append(v)

    lines.append("")
    lines.append("Suggested refinements (based on top 25% of trials):")

    for param in sorted(all_params.keys()):
        all_vals = all_params[param]
        top_vals = top_params.get(param, [])
        if not top_vals:
            continue
        if isinstance(all_vals[0], (int, float)):
            all_min, all_max = min(all_vals), max(all_vals)
            top_min, top_max = min(top_vals), max(top_vals)
            lines.append(f"  {param}:")
            lines.append(f"    Current range: [{all_min}, {all_max}]")
            lines.append(f"    Top trials range: [{top_min}, {top_max}]")
            if top_max - top_min < (all_max - all_min) * 0.5:
                lines.append(f"    -> Consider narrowing to [{top_min}, {top_max}]")
        else:
            from collections import Counter

            top_counts = Counter(top_vals)
            lines.append(f"  {param}:")
            lines.append(f"    Top choices: {dict(top_counts.most_common(5))}")

    return "\n".join(lines)


def _prune_config(arguments: dict) -> str:
    import optuna

    name = arguments.get("study_name", "")
    if not name:
        return "Provide 'study_name'."
    try:
        study = optuna.load_study(study_name=name, storage=_get_storage_url())
    except KeyError:
        return f"Study '{name}' not found."
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    lines = [
        f"Pruning analysis for '{name}':",
        f"  Completed: {len(completed)}",
        f"  Pruned: {len(pruned)}",
        f"  Prune rate: {len(pruned) / max(len(study.trials), 1) * 100:.1f}%",
    ]
    if len(pruned) > len(completed):
        lines.append("  Warning: High prune rate — pruner may be too aggressive")
        lines.append("  Suggestion: increase n_warmup_steps or n_startup_trials")
    elif len(pruned) == 0 and len(completed) > 10:
        lines.append("  Suggestion: enable pruning to save compute")
        lines.append("  Recommended: MedianPruner(n_startup_trials=5, n_warmup_steps=10)")

    config = {
        "pruner": "MedianPruner",
        "n_startup_trials": max(5, len(completed) // 5),
        "n_warmup_steps": 10,
    }
    lines.append(f"\n  Recommended config: {json.dumps(config, indent=2)}")
    return "\n".join(lines)
