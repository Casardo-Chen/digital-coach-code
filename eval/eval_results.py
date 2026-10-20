#!/usr/bin/env python3
"""
Score eval_sample.py JSON outputs with BLEU, METEOR, ROUGE-L, BERTScore F1, and CLAIR
(same-image caption agreement; Lin et al., EMNLP 2023).

Does not compute content-unit precision/recall. Input schema: either legacy
``models.<label>.per_sample`` rows, or slim eval_sample output with ``condition`` +
``prediction`` (sample ids -> text) plus ``condition.sample_json`` for references.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from digital_coach.llm import DEFAULT_TEXT_MODEL, openrouter_client

import bert_score  # type: ignore
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from openai import OpenAI
from rouge_score import rouge_scorer

from eval.clair_metrics import aggregate_clair_mean_by_method_bucket, aggregate_clair_metrics
from eval.sample_schema import build_per_sample_from_prediction_doc

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_BLEU_SMOOTH = SmoothingFunction().method1
_NLTK_LOCK = threading.Lock()
_WORD_TOKENIZER = RegexpTokenizer(r"\w+")


def _configure_ssl_cert_bundle_for_nltk() -> None:
    try:
        import certifi

        bundle = certifi.where()
        if bundle and os.path.isfile(bundle):
            os.environ.setdefault("SSL_CERT_FILE", bundle)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    except ImportError:
        pass


def _ensure_nltk_resources() -> None:
    _configure_ssl_cert_bundle_for_nltk()
    with _NLTK_LOCK:
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)


def _tokens(s: str) -> List[str]:
    return _WORD_TOKENIZER.tokenize(str(s or "").lower())


def metric_bleu(reference: str, hypothesis: str) -> float:
    r = _tokens(reference)
    h = _tokens(hypothesis)
    if not h or not r:
        return 0.0
    return float(sentence_bleu([r], h, smoothing_function=_BLEU_SMOOTH))


def metric_meteor(reference: str, hypothesis: str) -> float:
    r = _tokens(reference)
    h = _tokens(hypothesis)
    if not r and not h:
        return 1.0
    if not r or not h:
        return 0.0
    return float(meteor_score([r], h))


def metric_rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref = str(reference or "").strip()
    hyp = str(hypothesis or "").strip()
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    return float(_ROUGE.score(ref, hyp)["rougeL"].fmeasure)


def aggregate_text_metrics(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    if len(refs) != len(hyps):
        raise ValueError("refs/hyps length mismatch")
    n = len(refs)
    if n == 0:
        return {
            "bleu_mean": 0.0,
            "meteor_mean": 0.0,
            "rouge_l_mean": 0.0,
            "bertscore_f1_mean": 0.0,
        }

    bleu_acc = meteor_acc = rouge_acc = 0.0
    for r, h in zip(refs, hyps):
        bleu_acc += metric_bleu(r, h)
        meteor_acc += metric_meteor(r, h)
        rouge_acc += metric_rouge_l_f1(r, h)

    hyps_list = [h if h.strip() else " " for h in hyps]
    refs_list = [r if r.strip() else " " for r in refs]
    _p, _r, f1 = bert_score.score(hyps_list, refs_list, lang="en", verbose=False)
    bert_mean = float(f1.mean().item())

    return {
        "bleu_mean": bleu_acc / n,
        "meteor_mean": meteor_acc / n,
        "rouge_l_mean": rouge_acc / n,
        "bertscore_f1_mean": bert_mean,
    }


def _per_sample_bertscore_f1(refs: List[str], hyps: List[str]) -> List[float]:
    if not refs:
        return []
    hyps_list = [h if h.strip() else " " for h in hyps]
    refs_list = [r if r.strip() else " " for r in refs]
    _p, _r, f1 = bert_score.score(hyps_list, refs_list, lang="en", verbose=False)
    return [float(f1[i].item()) for i in range(len(refs))]


def load_eval_sample_output(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    if isinstance(data.get("prediction"), dict) and not data.get("models"):
        rows = build_per_sample_from_prediction_doc(data, repo_root=_REPO_ROOT)
        label = path.stem
        return {
            "condition": data.get("condition"),
            "sample_metadata": None,
            "models": {label: {"per_sample": rows}},
        }
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"Expected non-empty 'models' or 'prediction' object in {path}")
    return data


def score_one_model(
    per_sample: List[Dict[str, Any]],
    *,
    llm_client: Optional[OpenAI],
    clair_llm_model: str,
    clair_max_retries: int,
    clair_max_completion_tokens: int,
    dry_run: bool,
    skip_clair: bool,
) -> Tuple[Dict[str, float], Dict[str, Any], List[Dict[str, Any]]]:
    refs = [str(r.get("reference", "") or "") for r in per_sample]
    hyps = [str(r.get("prediction", "") or "") for r in per_sample]
    method_buckets = [str(r.get("method_bucket") or "").strip() or "UNKNOWN" for r in per_sample]

    text_m = aggregate_text_metrics(refs, hyps)
    bert_each = _per_sample_bertscore_f1(refs, hyps)

    if dry_run or skip_clair or llm_client is None:
        clair_m = {"clair_mean": 0.0, "clair_num_pairs": float(len(refs))}
        clair_records = [
            {
                "score": 0.0,
                "reason": "dry_run" if dry_run else ("skip_clair" if skip_clair else "no_client"),
                "skipped_llm": True,
            }
            for _ in range(len(refs))
        ]
    else:
        clair_m, clair_records = aggregate_clair_metrics(
            refs,
            hyps,
            client=llm_client,
            llm_model=clair_llm_model,
            max_retries=clair_max_retries,
            max_completion_tokens=clair_max_completion_tokens,
        )

    clair_by_bucket = aggregate_clair_mean_by_method_bucket(clair_records, method_buckets)

    scored_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(per_sample):
        r = copy.deepcopy(row)
        r["scores"] = {
            "bleu": metric_bleu(refs[i], hyps[i]),
            "meteor": metric_meteor(refs[i], hyps[i]),
            "rouge_l": metric_rouge_l_f1(refs[i], hyps[i]),
            "bertscore_f1": bert_each[i] if i < len(bert_each) else 0.0,
            "clair": {
                "score": float(clair_records[i].get("score", 0.0)),
                "reason": str(clair_records[i].get("reason") or ""),
                "skipped_llm": bool(clair_records[i].get("skipped_llm", False)),
            },
        }
        scored_rows.append(r)

    metrics = {**text_m, **clair_m}
    return metrics, clair_by_bucket, scored_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Score eval_sample.py JSON (BLEU, METEOR, ROUGE-L, BERTScore, CLAIR).")
    p.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to eval_sample.py output JSON.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write scored JSON here (default: results/eval/<input_stem>_scored_<ts>.json).",
    )
    p.add_argument(
        "--clair-llm-model",
        type=str,
        default=DEFAULT_TEXT_MODEL,
        help="OpenRouter model for batched CLAIR judgment (0–100).",
    )
    p.add_argument("--clair-max-retries", type=int, default=3)
    p.add_argument(
        "--clair-max-completion-tokens",
        type=int,
        default=65536,
        help="Max completion tokens for the batched CLAIR JSON response.",
    )
    p.add_argument(
        "--skip-clair",
        action="store_true",
        help="Skip CLAIR (no OpenAI judge call); per-sample clair scores are zero with skipped_llm=true.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip CLAIR API (same as production code path for empty judge); text metrics still computed.",
    )
    args = p.parse_args()

    in_path = args.input_json.expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    _ensure_nltk_resources()
    payload = load_eval_sample_output(in_path)

    out_path = args.output_json
    if out_path is None:
        out_dir = _REPO_ROOT / "results" / "eval"
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{in_path.stem}_scored_{ts}.json"
    else:
        out_path = out_path.expanduser().resolve()
        if out_path.exists():
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")

    scoring_condition = {
        "clair_llm_model": args.clair_llm_model,
        "clair_max_retries": int(args.clair_max_retries),
        "clair_max_completion_tokens": int(args.clair_max_completion_tokens),
        "skip_clair": bool(args.skip_clair),
        "dry_run": bool(args.dry_run),
    }

    need_clair_client = not (args.dry_run or args.skip_clair)
    llm_client: Optional[OpenAI] = None
    if need_clair_client:
        llm_client = openrouter_client()

    models_in = payload["models"]
    assert isinstance(models_in, dict)
    export_models: Dict[str, Any] = {}
    metrics_by_model: Dict[str, Dict[str, float]] = {}

    for model_label in sorted(models_in.keys()):
        block = models_in[model_label]
        if not isinstance(block, dict):
            continue
        per_sample = block.get("per_sample")
        if not isinstance(per_sample, list):
            raise ValueError(f"models[{model_label!r}].per_sample must be a list")

        metrics, clair_by_bucket, scored_rows = score_one_model(
            per_sample,
            llm_client=llm_client,
            clair_llm_model=args.clair_llm_model,
            clair_max_retries=int(args.clair_max_retries),
            clair_max_completion_tokens=int(args.clair_max_completion_tokens),
            dry_run=bool(args.dry_run),
            skip_clair=bool(args.skip_clair),
        )

        metrics_by_model[model_label] = metrics
        export_models[model_label] = {
            "metrics": metrics,
            "clair_by_method_bucket": clair_by_bucket,
            "per_sample": scored_rows,
        }

    out_doc = {
        "source": {"input_json": str(in_path)},
        "scoring_condition": scoring_condition,
        "generation_condition": payload.get("condition"),
        "sample_metadata": payload.get("sample_metadata"),
        "metrics_by_model": metrics_by_model,
        "models": export_models,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
