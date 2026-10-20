"""One session schema for human-human and human-AI coaching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DIALOGUE_ACTS = (
    "Info Request",
    "Answer",
    "Action Directive",
    "Inform",
    "Opinion",
    "Backchannel",
    "Not Enough Info",
)

COACHING_METHODS = (
    "Direct Instruction",
    "Plan",
    "Confirmation",
    "Diagnosis",
    "Explanation",
    "Tip",
    "Clarification",
    "Reflection",
    "Articulation",
    "Exploration",
    "Non-Instructive",
)

SPEAKERS = ("Coach", "Learner")
SOURCES = ("human-human", "human-ai")
COACH_TYPES = ("human", "model")

TASK_PREFIX_TO_SOFTWARE = {
    "01": "Figma",
    "02": "Blender",
    "03": "Onshape",
    "04": "FL Studio",
    "05": "Excel",
}


def software_for_task(task_id: str) -> str:
    text = str(task_id or "").strip()
    if len(text) >= 2:
        return TASK_PREFIX_TO_SOFTWARE.get(text[:2], "software tool")
    return "software tool"


def empty_turn(
    turn_id: int,
    *,
    start: str,
    speaker: str,
    text: str,
    end: str | None = None,
    dialogue_acts: list[str] | None = None,
    coaching_methods: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "turn_id": int(turn_id),
        "start": start,
        "end": end,
        "speaker": speaker,
        "text": text,
        "dialogue_acts": list(dialogue_acts or []),
        "coaching_methods": list(coaching_methods or []),
    }


def session_meta(
    *,
    session_id: str,
    source: str,
    task: str,
    coach_id: str,
    learner_id: str,
    coach_type: str,
    software: str | None = None,
    model: dict[str, str] | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if coach_type not in COACH_TYPES:
        raise ValueError(f"coach_type must be one of {COACH_TYPES}, got {coach_type!r}")
    meta: dict[str, Any] = {
        "session_id": session_id,
        "source": source,
        "task": task,
        "software": software or software_for_task(task),
        "participants": {
            "coach_id": coach_id,
            "learner_id": learner_id,
            "coach_type": coach_type,
        },
    }
    if model:
        meta["model"] = model
    if call_id:
        meta["call_id"] = call_id
    return meta


def normalize_turn(raw: dict[str, Any], turn_id: int) -> dict[str, Any]:
    speaker = str(raw.get("speaker") or "").strip()
    if speaker.lower() == "coach":
        speaker = "Coach"
    elif speaker.lower() == "learner":
        speaker = "Learner"
    end = raw.get("end")
    return empty_turn(
        turn_id,
        start=str(raw.get("start") or "00:00:00.000"),
        end=None if end in (None, "") else str(end),
        speaker=speaker,
        text=str(raw.get("text") or "").strip(),
        dialogue_acts=[str(x) for x in (raw.get("dialogue_acts") or [])],
        coaching_methods=[str(x) for x in (raw.get("coaching_methods") or [])],
    )


def write_session(directory: Path, meta: dict[str, Any], turns: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (directory / "turns.json").write_text(
        json.dumps(turns, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return directory


def read_session(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    turns = json.loads((directory / "turns.json").read_text(encoding="utf-8"))
    if not isinstance(turns, list):
        raise ValueError(f"turns.json must be a list: {directory}")
    return meta, turns
