"""
Shared helpers for stratified sample JSON (turns_sample / eval_sample).

On disk, ``samples`` is an object mapping string id -> turn row, each row including
an integer ``id`` field matching the key. Legacy list-shaped ``samples`` is still
accepted when reading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def samples_payload_to_ordered_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return sample dicts in stable id order; ensures each row has an ``id``."""
    raw = payload.get("samples")
    if isinstance(raw, list):
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            row = dict(item)
            if "id" not in row:
                row["id"] = i
            out.append(row)
        return out
    if isinstance(raw, dict):
        def _key(k: str) -> tuple:
            try:
                return (0, int(str(k)))
            except ValueError:
                return (1, str(k))

        out = []
        for k in sorted(raw.keys(), key=_key):
            item = raw[k]
            if not isinstance(item, dict):
                continue
            row = dict(item)
            if "id" not in row:
                try:
                    row["id"] = int(str(k))
                except ValueError:
                    row["id"] = str(k)
            out.append(row)
        return out
    raise ValueError("payload['samples'] must be a list or a dict of sample objects")


def resolve_sample_json_path(raw: Any, *, repo_root: Optional[Path] = None) -> Path:
    p = Path(str(raw).strip()).expanduser()
    if p.is_file():
        return p.resolve()
    if repo_root is not None:
        alt = (repo_root / "eval" / p.name).resolve()
        if alt.is_file():
            return alt
        alt2 = (repo_root / p).resolve()
        if alt2.is_file():
            return alt2
    raise FileNotFoundError(f"Sample JSON not found: {raw}")


def build_per_sample_from_prediction_doc(
    doc: Dict[str, Any],
    *,
    repo_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Build the same per-sample row list eval_sample used to store under
    ``models.*.per_sample`` from a slim run doc: ``condition`` + ``prediction`` only.
    """
    pred = doc.get("prediction")
    if not isinstance(pred, dict):
        raise ValueError("Expected top-level 'prediction' object mapping sample id -> text")
    cond = doc.get("condition") or {}
    raw_path = cond.get("sample_json")
    if not raw_path:
        raise ValueError("Expected condition.sample_json for prediction-only eval JSON")
    path = resolve_sample_json_path(raw_path, repo_root=repo_root)
    sample_payload = json.loads(path.read_text(encoding="utf-8"))
    ordered = samples_payload_to_ordered_list(sample_payload)
    rows: List[Dict[str, Any]] = []
    for s in ordered:
        sid = s.get("id", len(rows))
        key = str(sid)
        ptxt = pred.get(key)
        if ptxt is None and isinstance(sid, int):
            ptxt = pred.get(sid)
        rows.append(
            {
                "task": str(s.get("task", "")).strip(),
                "session": str(s.get("session", "")).strip(),
                "turn_id": int(s.get("turn_id", -1)),
                "reference": str(s.get("text", "") or "").strip(),
                "prediction": str(ptxt or "").strip(),
                "error": None,
                "method_bucket": str(s.get("method_bucket") or "").strip() or "UNKNOWN",
            }
        )
    return rows
