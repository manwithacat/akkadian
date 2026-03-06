"""Kaggle operation handlers — generalized, reads config for defaults."""

from __future__ import annotations

import subprocess
from time import time

from ..knowledge_graph.models import Entity
from ..state import get_state


def handle_kaggle(arguments: dict) -> str:
    """Dispatch kaggle operations."""
    op = arguments.get("operation", "status")
    ops = {
        "status": _kernel_status,
        "list_kernels": _list_kernels,
        "submissions": _submissions,
        "push_kernel": _push_kernel,
        "download_output": _download_output,
    }
    handler = ops.get(op)
    if handler is None:
        return f"Unknown kaggle operation: {op}. Available: {', '.join(ops)}"
    return handler(arguments)


def _get_username() -> str:
    return get_state().get_config().get("kaggle_username", "")


def _get_competition() -> str:
    return get_state().get_config().get("competition", "")


def _kernel_status(arguments: dict) -> str:
    slug = arguments.get("slug", "")
    if not slug:
        state = get_state()
        if state.knowledge_graph:
            kernels = state.knowledge_graph.list_entities("kernel")
            if kernels:
                lines = ["Recent kernels:"]
                for k in kernels[:10]:
                    lines.append(f"  {k.name} — {k.metadata.get('title', '')}")
                lines.append("\nProvide slug to check status.")
                return "\n".join(lines)
        return "No slug provided and no kernels tracked."

    if "/" not in slug:
        username = _get_username()
        if username:
            slug = f"{username}/{slug}"
        else:
            return "Slug needs username prefix (user/kernel) or set kaggle_username in config."

    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", slug],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip() or result.stderr.strip()

        state = get_state()
        if state.knowledge_graph:
            kernel_name = slug.split("/")[-1]
            entity = state.knowledge_graph.get_entity(f"kernel:{kernel_name}")
            if entity:
                entity.metadata["last_status"] = output
                entity.updated_at = time()
                state.knowledge_graph.upsert_entity(entity)

        return output
    except subprocess.TimeoutExpired:
        return "Kaggle CLI timed out"
    except FileNotFoundError:
        return "kaggle CLI not found. Install: pip install kaggle"


def _list_kernels(arguments: dict) -> str:
    state = get_state()
    if not state.knowledge_graph:
        return "Knowledge graph not initialized."
    kernels = state.knowledge_graph.list_entities("kernel")
    if not kernels:
        return "No kernels tracked. Push a kernel to populate."
    lines = ["Tracked kernels:"]
    for k in kernels:
        status = k.metadata.get("last_status", "unknown")
        title = k.metadata.get("title", k.name)
        lines.append(f"  {k.name}: {title} [{status}]")
    return "\n".join(lines)


def _submissions(arguments: dict) -> str:
    competition = arguments.get("competition") or _get_competition()
    if not competition:
        return "No competition set. Use bootstrap update_config to set competition slug."
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", competition, "--csv"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip() or "No submissions found."
    except subprocess.TimeoutExpired:
        return "Kaggle CLI timed out"
    except FileNotFoundError:
        return "kaggle CLI not found"


def _push_kernel(arguments: dict) -> str:
    path = arguments.get("path", "")
    if not path:
        return "Provide 'path' to the kernel directory containing kernel-metadata.json."
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "Kaggle push timed out"
    except FileNotFoundError:
        return "kaggle CLI not found"


def _download_output(arguments: dict) -> str:
    slug = arguments.get("slug", "")
    path = arguments.get("path", "")
    if not slug:
        return "Provide 'slug' of the kernel."
    if "/" not in slug:
        username = _get_username()
        if username:
            slug = f"{username}/{slug}"
    if not path:
        state = get_state()
        path = str(state.project_root / "outputs" / slug.split("/")[-1])
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "output", slug, "-p", path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "Download timed out"
    except FileNotFoundError:
        return "kaggle CLI not found"
