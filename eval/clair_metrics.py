#!/usr/bin/env python3
"""
CLAIR-style LLM judgment for whether a candidate "caption set" matches a reference set
for the same underlying image (Lin et al., EMNLP 2023).

Paper: https://aclanthology.org/2023.emnlp-main.841v2.pdf

Coach-turn eval: the **reference is always one human utterance** (whitespace-normalized to a
single caption string; we do **not** split the reference on newlines into multiple atomic
captions). The **candidate** is treated the same way: one model utterance as one caption.

**Batching:** ``aggregate_clair_metrics`` sends **one** chat completion per VLM condition
(all aligned reference/candidate pairs in that list), and expects a single JSON object with
a ``results`` array holding one ``{index, score, reason}`` per turn (same order as input indices).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _safe_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def normalize_single_caption(text: str) -> str:
    """Collapse internal newlines/whitespace to a single caption line (no atomic splitting)."""
    s = " ".join(str(text or "").split()).strip()
    return s if s else ""


def _reference_set_block(reference_text: str) -> str:
    """One human reference caption only (CLAIR reference set with a single item)."""
    s = normalize_single_caption(reference_text)
    if not s:
        return "- (empty)"
    return f"- {s}"


def _candidate_set_block(candidate_text: str) -> str:
    """One model caption as a single candidate-set item (same normalization as reference)."""
    return _reference_set_block(candidate_text)


def build_clair_user_message(reference_text: str, candidate_text: str) -> str:
    """Single-pair CLAIR prompt (for documentation / ad-hoc use). Batched eval uses ``build_clair_batch_messages``."""
    ref_block = _reference_set_block(reference_text)
    cand_block = _candidate_set_block(candidate_text)
    return (
        "You are trying to tell if a candidate coach utterance is describing the same image as a reference set of captions.\n"
        "as a reference set of captions.\n"
        "Candidate set:\n"
        f"{cand_block}\n"
        "Reference set:\n"
        f"{ref_block}\n"
        "\n"
        "On a precise scale from 0 to 100, how likely is it that the candidate set is describing "
        "the same image as the reference set?\n"
        '(JSON format, with a key "score", value between 0 and 100, and a key "reason" with a string value.)'
    )


def _coerce_score(raw: Any) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, v))


def _coerce_index(raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def _clair_batch_chat_completion(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_retries: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Like ``_json_chat_completion`` but allows a large completion budget for batched ``results``."""
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_completion_tokens > 0:
            kwargs["max_completion_tokens"] = max_completion_tokens
        try:
            response = client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs2 = {k: v for k, v in kwargs.items() if k != "max_completion_tokens"}
            if max_completion_tokens > 0:
                kwargs2["max_tokens"] = max_completion_tokens
            response = client.chat.completions.create(**kwargs2)
        raw = (response.choices[0].message.content or "").strip()
        raw = _strip_code_fences(raw)
        try:
            parsed = _safe_json_loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object (dict), got {type(parsed)}")
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Your last response could not be parsed as valid JSON, or the schema was wrong. "
                        "Return ONLY valid JSON matching the requested schema, with no additional text."
                    ),
                },
            ]
    raise RuntimeError(f"Failed to obtain valid batched CLAIR JSON after {max_retries} retries: {last_err}")


def build_clair_batch_user_content(*, n: int, items_payload: list[dict[str, Any]]) -> str:
    body = json.dumps({"items": items_payload}, ensure_ascii=False)
    return (
        f"You will judge {n} coach dialogue turns for CLAIR-style agreement with the same underlying "
        '"screen moment" / image.\n\n'
        "Each item has:\n"
        '- "index": integer from 0 to '
        f"{n - 1}\n"
        '- "reference": one human coach utterance (single caption; do not treat line breaks inside it as separate facts)\n'
        '- "candidate": one model coach utterance (same)\n\n'
        "For **each** item independently: on a scale 0–100, how likely is it that the candidate is describing "
        "the **same** on-screen situation as the reference?\n\n"
        "Rules:\n"
        "- If candidate is empty, score 0 with a brief reason.\n"
        "- If reference is empty, score 0 with a brief reason.\n\n"
        "Input JSON:\n"
        f"{body}\n\n"
        "Return ONLY valid JSON of this exact shape:\n"
        '{"results": [{"index": <int>, "score": <number 0-100>, "reason": "<string>"}, ...]}\n\n'
        f"Requirements: `results` must contain exactly {n} objects; every index from 0 to {n - 1} inclusive "
        "must appear exactly once."
    )


def _parse_batch_results(n: int, parsed: dict[str, Any]) -> list[dict[str, Any]]:
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("Top-level key 'results' must be a JSON array")

    by_index: dict[int, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        idx = _coerce_index(row.get("index"))
        if 0 <= idx < n:
            by_index[idx] = row

    if len(by_index) != n:
        missing = [i for i in range(n) if i not in by_index]
        raise ValueError(
            f"Expected {n} distinct valid indices in results, got {len(by_index)}; "
            f"missing (first 20): {missing[:20]}"
        )

    records: list[dict[str, Any]] = []
    for i in range(n):
        row = by_index[i]
        score = _coerce_score(row.get("score"))
        reason = str(row.get("reason") or "").strip() or "(no reason given)"
        records.append({"score": score, "reason": reason, "skipped_llm": False})
    return records


def aggregate_clair_metrics(
    references: List[str],
    candidates: List[str],
    *,
    client: OpenAI,
    llm_model: str,
    max_retries: int,
    max_completion_tokens: int = 65536,
    raw_batch_out: Optional[dict[str, Any]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    One batched chat completion for the whole list: model returns ``{"results": [...]}`` with
    ``len(results) == len(references)``, aligned by ``index``.

    ``raw_batch_out``: if provided, cleared and filled with ``{"parsed": <full dict from model>}`` on success.
    """
    if len(references) != len(candidates):
        raise ValueError("references/candidates length mismatch")

    n = len(references)
    if n == 0:
        return {"clair_mean": 0.0, "clair_num_pairs": 0.0}, []

    norms: list[tuple[str, str]] = [
        (normalize_single_caption(r), normalize_single_caption(c)) for r, c in zip(references, candidates)
    ]

    # Short-circuit: nothing for the model to judge
    if all(not c for _, c in norms):
        records = [
            {"score": 0.0, "reason": "empty candidate; no API call", "skipped_llm": True} for _ in range(n)
        ]
        if raw_batch_out is not None:
            raw_batch_out.clear()
            raw_batch_out["parsed"] = {
                "results": [{"index": i, "score": 0.0, "reason": r["reason"]} for i, r in enumerate(records)]
            }
        return (
            {"clair_mean": 0.0, "clair_num_pairs": float(n)},
            records,
        )

    items_payload = [{"index": i, "reference": r, "candidate": c} for i, (r, c) in enumerate(norms)]
    user_content = build_clair_batch_user_content(n=n, items_payload=items_payload)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You score caption agreement for coaching turns over a screen-recording task. "
                "Reply with only the requested JSON object."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    parsed = _clair_batch_chat_completion(
        client=client,
        model=llm_model,
        messages=messages,
        temperature=0.0,
        max_retries=max_retries,
        max_completion_tokens=max_completion_tokens,
    )
    records = _parse_batch_results(n, parsed)

    # Post-process: empty candidate/reference should be 0 / skipped_llm True (deterministic, overrides model)
    for i, (r_s, c_s) in enumerate(norms):
        if not c_s:
            records[i] = {"score": 0.0, "reason": "empty candidate; not scored by batch rule", "skipped_llm": True}
        elif not r_s:
            records[i] = {"score": 0.0, "reason": "empty reference; not scored by batch rule", "skipped_llm": True}

    if raw_batch_out is not None:
        raw_batch_out.clear()
        raw_batch_out["parsed"] = parsed

    total_score = sum(float(r["score"]) for r in records)
    mean_score = total_score / n
    metrics = {
        "clair_mean": mean_score,
        "clair_num_pairs": float(n),
    }
    return metrics, records


def aggregate_clair_mean_by_method_bucket(
    clair_records: List[Dict[str, Any]],
    method_buckets: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Macro mean CLAIR (0–100) per method_bucket label."""
    from collections import defaultdict

    acc: dict[str, list[float]] = defaultdict(list)
    for rec, bucket in zip(clair_records, method_buckets):
        b = bucket.strip() or "UNKNOWN"
        acc[b].append(float(rec.get("score", 0.0)))
    out: Dict[str, Dict[str, Any]] = {}
    for b in sorted(acc.keys()):
        xs = acc[b]
        out[b] = {"clair_mean": sum(xs) / len(xs), "n": len(xs)}
    return out
