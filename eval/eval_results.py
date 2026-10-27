#!/usr/bin/env python3
"""
Score ``eval_session.py`` outputs with CLAIR (Lin et al., EMNLP 2023).

Each input is ``{condition: {task, session, ...}, turns: [...]}`` holding generated coach
turns. References are the human coach turns of the same session, loaded from the Hugging
Face dataset and matched by ``turn_id``. Scores are also grouped by the reference turn's
first coaching method.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from openai import OpenAI

from digital_coach.dataset import load_session
from digital_coach.llm import DEFAULT_TEXT_MODEL, openrouter_client
from digital_coach.paths import results_dir
from eval.clair_metrics import aggregate_clair_mean_by_method_bucket, aggregate_clair_metrics

load_dotenv()


def pair_with_references(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per generated turn: ``{turn_id, reference, prediction, method_bucket}``."""
    cond = doc.get("condition") or {}
    task, session = str(cond.get("task") or ""), str(cond.get("session") or "")
    if not task or not session:
        raise ValueError("condition.task and condition.session are required")
    refs = {int(t["turn_id"]): t for t in load_session(task, session)["turns"]}
    rows = []
    for gen in doc.get("turns") or []:
        ref = refs.get(int(gen.get("turn_id", -1)))
        if ref is None:
            continue
        methods = ref.get("coaching_methods") or []
        rows.append(
            {
                "turn_id": int(ref["turn_id"]),
                "start": ref.get("start"),
                "reference": str(ref.get("text") or ""),
                "prediction": str(gen.get("text") or ""),
                "method_bucket": str(methods[0]) if methods else "UNKNOWN",
            }
        )
    return rows


def score_rows(
    rows: List[Dict[str, Any]],
    *,
    client: Optional[OpenAI],
    model: str,
    max_retries: int,
    max_completion_tokens: int,
) -> Dict[str, Any]:
    refs = [r["reference"] for r in rows]
    hyps = [r["prediction"] for r in rows]
    if client is None:
        metrics = {"clair_mean": 0.0, "clair_num_pairs": float(len(rows))}
        records = [{"score": 0.0, "reason": "dry_run", "skipped_llm": True} for _ in rows]
    else:
        metrics, records = aggregate_clair_metrics(
            refs,
            hyps,
            client=client,
            llm_model=model,
            max_retries=max_retries,
            max_completion_tokens=max_completion_tokens,
        )
    return {
        "metrics": metrics,
        "clair_by_method_bucket": aggregate_clair_mean_by_method_bucket(
            records, [r["method_bucket"] for r in rows]
        ),
        "per_turn": [{**row, "clair": rec} for row, rec in zip(rows, records)],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Score eval_session.py outputs with CLAIR.")
    p.add_argument("inputs", type=Path, nargs="+", help="eval_session.py output JSON file(s).")
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: results/eval/clair_<timestamp>.json.",
    )
    p.add_argument(
        "--clair-llm-model",
        default=DEFAULT_TEXT_MODEL,
        help="OpenRouter judge model (scores 0-100).",
    )
    p.add_argument("--clair-max-retries", type=int, default=3)
    p.add_argument("--clair-max-completion-tokens", type=int, default=65536)
    p.add_argument("--dry-run", action="store_true", help="Pair turns without calling the judge.")
    args = p.parse_args()

    client = None if args.dry_run else openrouter_client()
    runs = {}
    for path in args.inputs:
        path = path.expanduser().resolve()
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = pair_with_references(doc)
        result = score_rows(
            rows,
            client=client,
            model=args.clair_llm_model,
            max_retries=args.clair_max_retries,
            max_completion_tokens=args.clair_max_completion_tokens,
        )
        runs[path.stem] = {"source": str(path), "condition": doc.get("condition"), **result}
        m = result["metrics"]
        print(f"{path.stem}: CLAIR {m['clair_mean']:.1f} over {int(m['clair_num_pairs'])} turns")

    out_path = args.output_json
    if out_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir() / "eval" / f"clair_{ts}.json"
    out_doc = {
        "scoring": {"judge_model": args.clair_llm_model, "dry_run": bool(args.dry_run)},
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
