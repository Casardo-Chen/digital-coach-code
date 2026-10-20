"""Load coaching sessions as a Hugging Face dataset, one row per turn.

Local placeholder sessions live in ``data/sessions``. Set ``DIGITAL_COACH_DATASET``
to a Hub dataset id to load a published corpus instead. This module does not upload data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from digital_coach.paths import sessions_dir
from digital_coach.schema import read_session


def iter_turn_rows(root: Path | None = None) -> Iterator[dict[str, Any]]:
    base = root or sessions_dir()
    if not base.is_dir():
        return
    for session_path in sorted(p for p in base.iterdir() if p.is_dir()):
        if not (session_path / "meta.json").is_file() or not (session_path / "turns.json").is_file():
            continue
        meta, turns = read_session(session_path)
        participants = meta.get("participants") or {}
        model = meta.get("model") or {}
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            yield {
                "session_id": str(meta.get("session_id") or session_path.name),
                "source": str(meta.get("source") or ""),
                "task": str(meta.get("task") or ""),
                "software": str(meta.get("software") or ""),
                "coach_id": str(participants.get("coach_id") or ""),
                "learner_id": str(participants.get("learner_id") or ""),
                "coach_type": str(participants.get("coach_type") or ""),
                "model_provider": str(model.get("provider") or ""),
                "model_name": str(model.get("model") or ""),
                "call_id": str(meta.get("call_id") or ""),
                "turn_id": int(turn.get("turn_id") or 0),
                "start": str(turn.get("start") or ""),
                "end": "" if turn.get("end") in (None, "") else str(turn.get("end")),
                "speaker": str(turn.get("speaker") or ""),
                "text": str(turn.get("text") or ""),
                "dialogue_acts": [str(x) for x in (turn.get("dialogue_acts") or [])],
                "coaching_methods": [str(x) for x in (turn.get("coaching_methods") or [])],
            }


def load_sessions(dataset_id: str | None = None, sessions_path: Path | None = None):
    """Return a ``datasets.Dataset``.

    Pass a Hub id, or set ``DIGITAL_COACH_DATASET``. Otherwise rows are built from
    ``data/sessions`` (or ``sessions_path``).
    """
    from datasets import Dataset, load_dataset

    hub_id = (dataset_id or os.getenv("DIGITAL_COACH_DATASET") or "").strip()
    if hub_id:
        loaded = load_dataset(hub_id)
        if hasattr(loaded, "keys") and "train" in loaded:
            return loaded["train"]
        return loaded
    rows = list(iter_turn_rows(sessions_path))
    return Dataset.from_list(rows)
