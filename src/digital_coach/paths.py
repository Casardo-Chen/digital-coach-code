"""Repository paths. Override with environment variables when the corpus lives elsewhere."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    raw = os.getenv("DIGITAL_COACH_DATA", "").strip()
    return Path(raw).expanduser() if raw else REPO_ROOT / "data"


def results_dir() -> Path:
    raw = os.getenv("DIGITAL_COACH_RESULTS", "").strip()
    return Path(raw).expanduser() if raw else REPO_ROOT / "results"


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def tasks_dir() -> Path:
    return REPO_ROOT / "tasks"


def eval_output_dir(session_id: str) -> Path:
    return results_dir() / "eval" / session_id


def figures_dir() -> Path:
    return results_dir() / "figures"


def live_sessions_dir() -> Path:
    """Human-AI sessions exported by the realtime prototype (not committed)."""
    return results_dir() / "sessions"


def resolve_session_dir(session_id: str, task: str | None = None) -> Path:
    """Find a unified session folder by id, or by task plus folder name."""
    direct = sessions_dir() / session_id
    if (direct / "turns.json").is_file():
        return direct

    root = sessions_dir()
    if root.is_dir() and task:
        for candidate in sorted(p for p in root.iterdir() if p.is_dir()):
            meta_path = candidate / "meta.json"
            if not meta_path.is_file() or not (candidate / "turns.json").is_file():
                continue
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(meta.get("task") or "") == str(task) and (
                candidate.name == session_id or str(meta.get("session_id") or "") == session_id
            ):
                return candidate

    raise FileNotFoundError(
        f"No session folder with turns.json for session={session_id!r} task={task!r} under {root}"
    )
