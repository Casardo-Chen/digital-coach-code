"""Load DigitalCoach sessions from the Hugging Face Hub.

The Hub dataset stores one folder per human-human session:
``<task>/<session>/{meta.json, turns.json, screen.mp4}``. Files are cached by
``huggingface_hub``, so repeated loads do not re-download.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_DATASET = "berkeley-hci/digital-coach"


def dataset_id() -> str:
    return (os.getenv("DIGITAL_COACH_DATASET") or DEFAULT_DATASET).strip()


def _download(filename: str) -> Path:
    return Path(hf_hub_download(dataset_id(), filename, repo_type="dataset"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_session(task: str, session: str, *, with_video: bool = False) -> dict[str, Any]:
    """Return ``{"meta", "turns", "video_path"}`` for one session.

    ``video_path`` is ``None`` unless ``with_video`` is set; the screen recording is large.
    """
    meta = _read_json(_download(f"{task}/{session}/meta.json"))
    turns = _read_json(_download(f"{task}/{session}/turns.json"))
    meta.setdefault("source", "human-human")
    video_path = _download(f"{task}/{session}/screen.mp4") if with_video else None
    return {"meta": meta, "turns": turns, "video_path": video_path}


def load_sessions() -> list[dict[str, Any]]:
    """Download every session's metadata and turns (no video)."""
    root = Path(
        snapshot_download(
            dataset_id(),
            repo_type="dataset",
            allow_patterns=["*/*/meta.json", "*/*/turns.json"],
        )
    )
    sessions = []
    for meta_path in sorted(root.glob("*/*/meta.json")):
        turns_path = meta_path.with_name("turns.json")
        if not turns_path.is_file():
            continue
        meta = _read_json(meta_path)
        meta.setdefault("source", "human-human")
        sessions.append({"meta": meta, "turns": _read_json(turns_path), "video_path": None})
    return sessions


def load_turns() -> list[dict[str, Any]]:
    """One row per turn across all sessions, with session fields repeated."""
    rows = []
    for session in load_sessions():
        meta = session["meta"]
        participants = meta.get("participants") or {}
        for turn in session["turns"]:
            rows.append(
                {
                    "task": meta.get("task"),
                    "session": meta.get("session"),
                    "software": meta.get("software"),
                    "source": meta.get("source"),
                    "coach_id": participants.get("coach_id"),
                    "learner_id": participants.get("learner_id"),
                    **turn,
                }
            )
    return rows
