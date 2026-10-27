"""Repository paths. Session data comes from the Hugging Face Hub (see ``dataset.py``)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    raw = os.getenv("DIGITAL_COACH_RESULTS", "").strip()
    return Path(raw).expanduser() if raw else REPO_ROOT / "results"


def tasks_dir() -> Path:
    return REPO_ROOT / "tasks"


def eval_output_dir(task: str, session: str) -> Path:
    return results_dir() / "eval" / task / session


def annotation_output_dir() -> Path:
    return results_dir() / "annotation"


def live_sessions_dir() -> Path:
    """Human-AI sessions exported by the realtime prototype."""
    return results_dir() / "sessions"
