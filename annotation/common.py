"""Shared I/O for the annotation scripts: load turns, call the OpenRouter model, write results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from digital_coach.dataset import load_session
from digital_coach.llm import chat_json_text, text_model
from digital_coach.paths import annotation_output_dir

load_dotenv()


def add_session_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-t", "--task", default="0101", help="Task id in the HF dataset.")
    parser.add_argument("-s", "--session", default="C1L1", help="Session id in the HF dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Annotate a local turns.json instead (e.g. a human-AI session under results/sessions).",
    )
    parser.add_argument("--model", default=None, help="OpenRouter model (default: OPENROUTER_MODEL).")


def load_input_turns(args: argparse.Namespace) -> tuple[list[dict], str]:
    """Return ``(turns, output_stem)``."""
    if args.input is not None:
        path = args.input.expanduser().resolve()
        return json.loads(path.read_text(encoding="utf-8")), path.parent.name
    return load_session(args.task, args.session)["turns"], f"{args.task}_{args.session}"


def call_json_model(system: str, prompt: str, model: str | None) -> list[dict]:
    print(f"Sending to OpenRouter ({text_model(model)})...")
    raw = chat_json_text(system=system, user=prompt, model=model)
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def write_output(turns: list[dict], stem: str, kind: str) -> Path:
    out_path = annotation_output_dir() / f"{stem}_{kind}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(turns, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return out_path
