"""Label sets and the session layout shared by human-human and human-AI sessions.

A session is a folder with ``meta.json`` and ``turns.json``. ``meta.json`` holds
``task``, ``session``, ``software``, ``source`` (``human-human`` or ``human-ai``), and
``participants``. Each turn is
``{turn_id, start, end, speaker, text, dialogue_acts, coaching_methods}``.
"""

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

TASK_PREFIX_TO_SOFTWARE = {
    "01": "Figma",
    "02": "Blender",
    "03": "Onshape",
    "04": "FL Studio",
    "05": "Excel",
}


def software_for_task(task_id: str | None) -> str:
    return TASK_PREFIX_TO_SOFTWARE.get(str(task_id or "")[:2], "software tool")


def write_session(directory: Path, meta: dict[str, Any], turns: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, doc in (("meta.json", meta), ("turns.json", turns)):
        (directory / name).write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return directory


def read_session(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    turns = json.loads((directory / "turns.json").read_text(encoding="utf-8"))
    return meta, turns
