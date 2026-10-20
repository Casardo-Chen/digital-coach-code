#!/usr/bin/env python3
"""
Pipeline for evaluating AI coaching agents on static recordings.

Author: Meng Chen
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import argparse
import base64
import math
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
import sys
import tempfile

import moviepy

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from openai import OpenAI

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

from digital_coach.llm import openrouter_client as _shared_openrouter_client
from digital_coach.paths import eval_output_dir, resolve_session_dir
from eval.prompt_templates import (
    COACHING_METHODS_DEFINITION_AND_EXAMPLES,
    VLM_SYSTEM_INSTRUCTION_TEMPLATE,
    VLM_SYSTEM_INSTRUCTION_TEMPLATE_ORACLE,
    VLM_SYSTEM_INSTRUCTION_TEMPLATE_TEACHER,
    render_prompt_template,
)

_VLM_PROMPT_STYLES = frozenset({"default", "teacher", "oracle"})
_VLM_INPUT_MODALITIES = frozenset({"text-only", "text-image", "image-only", "text-video"})
_GEMINI3_THINKING_LEVELS = frozenset({"low", "medium", "high"})
_VLM_GENERATION_TEMPERATURE = 1.0
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _normalize_vlm_modality(value: str) -> str:
    """Canonical modality uses hyphens; accept legacy underscores (e.g. from older JSON)."""
    return (value or "text-image").strip().lower().replace("_", "-")

# Session inputs live under data/sessions/<id>/. Ablation JSON is written to results/eval/<id>/.

# task id prefix -> software tool
TASK_PREFIX_TO_SOFTWARE_TOOL: Dict[str, str] = {
    "01": "Figma",
    "02": "Blender",
    "03": "Onshape",
    "04": "FL studio",
    "05": "Excel",
}


def software_tool_for_task(task_id: str) -> str:
    task_text = str(task_id or "").strip()
    if len(task_text) >= 2:
        return TASK_PREFIX_TO_SOFTWARE_TOOL.get(task_text[:2], "software tool")
    return "software tool"


def video_filename_for_turns_splitted(turns_basename: str) -> str:
    """
    Pick media file for a turns JSON name.
    """
    return "screen.mp4"

def _sanitize_for_filename(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "model"


def vlm_output_filename_for_model(turns_basename: str, model_label: str) -> str:
    """
    Always emit canonical VLM output names:
    turns_vlm_<model>_<modality>_<prompt>_ctx<context>_fps<fps>.json
    """
    _ = turns_basename  # Kept for call-site compatibility.
    safe_label = _sanitize_for_filename(model_label)
    return f"turns_vlm_{safe_label}.json"


def _is_coach_speaker(speaker: str) -> bool:
    """Heuristic: handle minor variations in speaker labels."""
    s = str(speaker).strip().lower()
    return s in {"coach"}


def _is_learner_speaker(speaker: str) -> bool:
    s = str(speaker).strip().lower()
    return s in {"learner"}


def _timestamp_to_seconds(ts: str) -> float:
    """
    Convert timestamps like "00:00:07.050" or "00:07.050" to seconds.
    """
    t = str(ts).strip()
    if not t:
        return 0.0
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(t)


def _speaker_label_for_prompt(speaker: str) -> str:
    if _is_coach_speaker(speaker):
        return "Coach"
    if _is_learner_speaker(speaker):
        return "Student"
    return str(speaker).strip() or "Unknown"


_VLM_FRAME_MAX_LONG_EDGE_DEFAULT = 640
"""Default longest-edge cap for video frames (JPEG) before sending to VL APIs."""

_QWEN_VLM_REFERENCE_MAX_LONG_EDGE = 1024
"""
Qwen VL: downscale task reference images (final.png / final.gif) so the longest edge is
at most this value (~HF long-video / vision preprocessing guidance); avoids huge
native-resolution embeds that blow up vision tokens or provider limits.
"""


def _is_qwen_vlm_model(model: str) -> bool:
    return (model or "").strip().lower().startswith("qwen")


def _supports_native_video_vlm(model: str) -> bool:
    """Models that support the ``text-video`` modality (native clip and/or frame fallback)."""
    return _is_gemini_openai_model(model) or _is_qwen_vlm_model(model)


def _uses_native_video_url_part(model: str) -> bool:
    """
    True when the chat API accepts an OpenAI-style ``video_url`` part.

    Google's Gemini OpenAI-compatible endpoint (AI Studio) rejects ``video_url``; those
  models use JPEG frames sampled from the same pre-coach MP4 segment instead.
    """
    return _is_qwen_vlm_model(model)


def _vlm_video_max_bytes() -> int:
    """Max encoded MP4 segment size for text-video (base64 inflates payload)."""
    raw = (os.getenv("VLM_VIDEO_MAX_BYTES") or str(20 * 1024 * 1024)).strip()
    try:
        v = int(raw)
    except ValueError:
        v = 20 * 1024 * 1024
    return max(256 * 1024, min(v, 100 * 1024 * 1024))


def _vlm_reference_image_max_long_edge(model: str) -> Optional[int]:
    """If set, reference final images are downscaled (JPEG) before embedding."""
    if _is_qwen_vlm_model(model):
        return _QWEN_VLM_REFERENCE_MAX_LONG_EDGE
    return None


def _vlm_image_url_detail_for_model(model: str) -> Optional[str]:
    """
    Optional OpenAI ``image_url.detail`` (``low`` / ``high`` / ``auto``) for vision payloads.
    Qwen routes on OpenRouter sometimes 400 with default/auto; ``low`` reduces vision tokens.
    Override with env ``QWEN_VLM_IMAGE_DETAIL`` (empty to omit the field entirely).
    """
    if not _is_qwen_vlm_model(model):
        return None
    raw = (os.getenv("QWEN_VLM_IMAGE_DETAIL") or "low").strip().lower()
    if not raw or raw in {"none", "off", "omit"}:
        return None
    return raw if raw in ("low", "high", "auto") else "low"


def _qwen_max_context_frames() -> int:
    """
    Max number of video context frames sent per coach turn for Qwen on OpenRouter.
    Long windows (e.g. 30s × 1fps ≈ 30 JPEGs) often 413/400/time out; subsample uniformly.
    Override with env ``QWEN_VLM_MAX_CONTEXT_FRAMES`` (integer, clamped 1..256).
    """
    raw = (os.getenv("QWEN_VLM_MAX_CONTEXT_FRAMES") or "12").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 12
    return max(1, min(v, 256))


def _uniform_subsample_pairs(
    frames: List[Any],
    frame_times: List[float],
    cap: int,
) -> Tuple[List[Any], List[float]]:
    n = len(frames)
    if n <= cap or cap < 1:
        return frames, frame_times
    if cap == 1:
        j = n // 2
        return [frames[j]], [frame_times[j]]
    idxs = sorted({min(n - 1, int(round(i * (n - 1) / (cap - 1)))) for i in range(cap)})
    return [frames[i] for i in idxs], [frame_times[i] for i in idxs]


def _maybe_cap_vlm_context_frames(
    frames: List[Any],
    frame_times: List[float],
    *,
    vlm_model: str,
) -> Tuple[List[Any], List[float]]:
    if not _is_qwen_vlm_model(vlm_model):
        return frames, frame_times
    return _uniform_subsample_pairs(frames, frame_times, _qwen_max_context_frames())


def _frames_to_base64_jpeg(
    frames: Iterable[Any], *, max_dim: int = _VLM_FRAME_MAX_LONG_EDGE_DEFAULT, quality: int = 80
) -> List[str]:
    """
    Convert numpy frames to base64-encoded JPEGs.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for VLM frame encoding (PIL.Image import failed).")

    b64_list: List[str] = []
    for frame in frames:
        img = Image.fromarray(frame)
        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
            img = img.resize(new_size, resample=Image.BILINEAR)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return b64_list


def _is_gemini_openai_model(model: str) -> bool:
    """Model ids routed to Google's OpenAI-compatible Gemini endpoints (AI Studio API key)."""
    m = (model or "").strip().lower()
    return m.startswith("gemini") or m.startswith("google/gemini")


def _is_gemini3_model_id(model: str) -> bool:
    """True for Gemini 3 model ids (support thinking_level / reasoning_effort)."""
    m = (model or "").strip().lower()
    if m.startswith("google/"):
        m = m[len("google/") :]
    return m.startswith("gemini-3")


def _is_openai_gpt_series_model(model: str) -> bool:
    """True for OpenAI Chat Completions ids in the gpt-* family (e.g. gpt-5.4)."""
    return (model or "").strip().lower().startswith("gpt-")


def _gemini3_thinking_level_str() -> str:
    """
    Thinking depth for Gemini 3 on OpenAI-compatible Chat Completions (reasoning_effort).
    Override with GEMINI_THINKING_LEVEL=low|medium|high.
    """
    raw = (os.getenv("GEMINI_THINKING_LEVEL") or "medium").strip().lower()
    return raw if raw in _GEMINI3_THINKING_LEVELS else "medium"


def _is_openai_native_platform_model(model: str) -> bool:
    """
    True when ``model`` is sent to the official OpenAI API (api.openai.com), i.e. supports Batch API.

    Gemini and OpenRouter open-model slugs (Qwen, Llama, etc.) use other base URLs and are not
    batchable via OpenAI Batch.
    """
    if _is_gemini_openai_model(model):
        return False
    m = (model or "").strip().lower()
    non_openai_chat_prefixes = ("qwen", "meta-llama", "anthropic/")
    if any(m.startswith(prefix) for prefix in non_openai_chat_prefixes):
        return False
    return True


def _openai_native_client() -> OpenAI:
    """Official OpenAI client (required for Batch API)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


def _openrouter_optional_headers() -> Dict[str, str]:
    """Optional OpenRouter attribution (leaderboards); see https://openrouter.ai/docs/app-attribution."""
    headers: Dict[str, str] = {}
    referer = (os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_SITE_URL") or "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = (os.getenv("OPENROUTER_TITLE") or os.getenv("OPENROUTER_APP_TITLE") or "").strip()
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def _openrouter_client() -> OpenAI:
    """OpenAI SDK pointed at OpenRouter (shared client)."""
    return _shared_openrouter_client()


def _is_openrouter_open_model_slug(model: str) -> bool:
    """
    Model ids that use OpenRouter without an explicit ``--openrouter`` flag.

    Includes Open-weight slugs (Qwen, Meta-Llama) and provider-prefixed OpenRouter ids
    such as ``anthropic/claude-…``.
    """
    m = (model or "").strip().lower()
    if not m:
        return False
    return (
        m.startswith("qwen")
        or m.startswith("meta-llama")
        or m.startswith("anthropic/")
    )


def _openai_compatible_client_for_model(model: str, *, use_openrouter: bool = False) -> Any:
    """
    Return an OpenAI SDK client for chat completions.

    Supported backends:

    - **OpenAI** (``OPENAI_API_KEY``): native ``gpt-*`` and other models on api.openai.com.
    - **Gemini** (``GOOGLE_API_KEY``): ``gemini*`` / ``google/gemini*`` via Google's OpenAI-compatible
      endpoint (generativelanguage.googleapis.com).
    - **OpenRouter** (``OPENROUTER_API_KEY``): when ``use_openrouter`` is True, or when the model id
      is routed there automatically (``qwen`` / ``meta-llama`` / ``anthropic/`` prefixes, e.g.
      ``qwen/qwen2.5-vl-72b-instruct``, ``anthropic/claude-sonnet-4.6``).
    """
    if use_openrouter or _is_openrouter_open_model_slug(model):
        print("Using OpenRouter")
        client = _openrouter_client()
        return client

    if _is_gemini_openai_model(model):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment.")
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


def _openai_vlm_chat_completion_kwargs(
    *,
    messages: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    """Keyword arguments for ``chat.completions.create`` / Batch body (same shape)."""
    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        # "temperature": _VLM_GENERATION_TEMPERATURE,
    }
    # if _is_gemini3_model_id(model):
    #     create_kwargs["reasoning_effort"] = _gemini3_thinking_level_str()
    # elif _is_openai_gpt_series_model(model):
    #     create_kwargs["reasoning_effort"] = "medium"
    return create_kwargs


def _chat_completion_error_payload(response: Any) -> Optional[Any]:
    """OpenRouter may return a top-level ``error`` dict when ``choices`` is null."""
    err = getattr(response, "error", None)
    if err is not None:
        return err
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump().get("error")
        except Exception:  # pragma: no cover
            return None
    return None


def _format_chat_completion_error_for_raise(err: Any) -> str:
    if err is None:
        return ""
    if hasattr(err, "model_dump"):
        try:
            return json.dumps(err.model_dump(), ensure_ascii=False, default=str)
        except Exception:  # pragma: no cover
            return repr(err)
    if isinstance(err, dict):
        return json.dumps(err, ensure_ascii=False, default=str)
    return str(err)


def _assistant_text_from_chat_completion(response: Any, *, model: str) -> str:
    """
    Normalize ``chat.completions.create`` output to a single string.
    Some providers (e.g. OpenRouter or refusals) return no ``choices`` or empty ``content``.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        err_payload = _chat_completion_error_payload(response)
        if err_payload is not None:
            detail = _format_chat_completion_error_for_raise(err_payload)
            raise RuntimeError(
                f"Chat completions API error for model={model!r} (no choices). {detail}"
            )
        rid = getattr(response, "id", None)
        extra = ""
        if hasattr(response, "model_dump"):
            try:
                extra = json.dumps(response.model_dump(), ensure_ascii=False, default=str)[:12000]
            except Exception:  # pragma: no cover
                extra = repr(response)[:12000]
        raise RuntimeError(
            f"Chat completion returned no choices for model={model!r} "
            f"(response_id={rid!r}). Raw response (truncated): {extra}"
        )

    msg = getattr(choices[0], "message", None)
    if msg is None:
        raise RuntimeError(f"Chat completion first choice has no message for model={model!r}: {choices[0]!r}")

    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    refusal = getattr(msg, "refusal", None)
    if refusal:
        return str(refusal).strip()

    if hasattr(msg, "model_dump"):
        dumped = msg.model_dump()
        c = dumped.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
        r = dumped.get("refusal")
        if r:
            return str(r).strip()

    finish = getattr(choices[0], "finish_reason", None)
    raise RuntimeError(
        f"Chat completion returned empty assistant text for model={model!r} "
        f"(finish_reason={finish!r}, content={content!r})."
    )


def openai_vlm_chat_completion(
    *,
    messages: List[Dict[str, Any]],
    model: str = "gpt-5.4",
    use_openrouter: bool = False,
    client: Optional[Any] = None,
) -> str:
    """
    Multimodal chat completion: prior transcript as alternating user/assistant messages;
    the last user turn may include text plus image_url parts.

    Sets ``reasoning_effort`` for Gemini 3 (see ``GEMINI_THINKING_LEVEL``) and for
    OpenAI ``gpt-*`` models (fixed ``medium``).

    Pass a shared ``client`` from ``_openai_compatible_client_for_model`` when making many
    calls for the same model so connections are reused (one client per model/run).
    """
    resolved = client if client is not None else _openai_compatible_client_for_model(
        model, use_openrouter=use_openrouter
    )
    create_kwargs = _openai_vlm_chat_completion_kwargs(messages=messages, model=model)
    response = resolved.chat.completions.create(**create_kwargs)
    return _assistant_text_from_chat_completion(response, model=model)


def _openai_batch_chat_completion_jsonl_line(
    custom_id: str,
    messages: List[Dict[str, Any]],
    *,
    model: str,
) -> str:
    body = _openai_vlm_chat_completion_kwargs(messages=messages, model=model)
    row = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    return json.dumps(row, ensure_ascii=False)


def _jsonl_line_utf8_byte_len(line: str) -> int:
    return len(line.encode("utf-8")) + 1


def _chunk_jsonl_lines_by_byte_budget(
    lines: List[str],
    max_bytes: int,
) -> List[List[str]]:
    """
    Greedy pack JSONL lines into chunks so each chunk's on-disk UTF-8 size stays <= max_bytes.
    """
    if not lines:
        return []
    chunks: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for line in lines:
        nbytes = _jsonl_line_utf8_byte_len(line)
        if nbytes > max_bytes:
            raise RuntimeError(
                f"A single batch request JSONL line is ~{nbytes / (1024 * 1024):.1f} MB "
                f"(limit {max_bytes // (1024 * 1024)} MB). "
                "Reduce frames (--vlm-fps), context window, or image payload for that turn."
            )
        if current and current_bytes + nbytes > max_bytes:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += nbytes
    if current:
        chunks.append(current)
    return chunks


def _run_one_openai_batch_jsonl(
    client: OpenAI,
    lines: List[str],
    *,
    model: str,
    poll_interval_s: float,
    chunk_label: str,
) -> Dict[str, str]:
    """Upload one JSONL file, wait for batch completion, return custom_id -> assistant text."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        encoding="utf-8",
        newline="\n",
        delete=False,
    ) as tmp:
        for line in lines:
            tmp.write(line + "\n")
        tmp_path = Path(tmp.name)

    try:
        with tmp_path.open("rb") as fh:
            batch_file = client.files.create(file=fh, purpose="batch")

        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": "eval_session.py", "model": model},
        )
        batch_id = batch.id
        print(
            f"[openai-batch] {chunk_label} created batch_id={batch_id} requests={len(lines)}"
        )

        terminal = frozenset({"completed", "failed", "expired", "cancelled"})
        while batch.status not in terminal:
            time.sleep(max(1.0, float(poll_interval_s)))
            batch = client.batches.retrieve(batch_id)
            counts = getattr(batch, "request_counts", None)
            if counts is not None:
                c = getattr(counts, "completed", None)
                t = getattr(counts, "total", None)
                if c is not None and t is not None:
                    print(f"[openai-batch] {chunk_label} status={batch.status} completed={c}/{t}")
            else:
                print(f"[openai-batch] {chunk_label} status={batch.status}")

        if batch.status != "completed":
            err = getattr(batch, "errors", None)
            raise RuntimeError(
                f"OpenAI batch {chunk_label} ended with status={batch.status!r} errors={err!r}"
            )

        out_id = getattr(batch, "output_file_id", None)
        if not out_id:
            raise RuntimeError(f"Batch {chunk_label} completed but output_file_id is missing.")

        out_text = client.files.content(out_id).text
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    results: Dict[str, str] = {}
    for raw_line in out_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        obj = json.loads(raw_line)
        cid = str(obj.get("custom_id", ""))
        err = obj.get("error")
        if err:
            raise RuntimeError(f"Batch line error custom_id={cid!r}: {err}")
        resp = obj.get("response") or {}
        if int(resp.get("status_code", 0)) != 200:
            raise RuntimeError(f"Batch HTTP error custom_id={cid!r}: {resp!r}")
        body = resp.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"Batch missing choices custom_id={cid!r}: {body!r}")
        msg = (choices[0].get("message") or {}).get("content")
        results[cid] = (str(msg) if msg is not None else "").strip()

    return results


def run_openai_batch_vlm_chat_completions(
    items: List[Tuple[str, List[Dict[str, Any]]]],
    *,
    model: str,
    poll_interval_s: float,
) -> Dict[str, str]:
    """
    OpenAI Batch job(s) for many chat completions. Returns ``custom_id`` -> assistant text.

    Splits into multiple batch uploads when the JSONL would exceed the API input size limit
    (default 190 MB, overridable via ``OPENAI_BATCH_MAX_INPUT_BYTES``).

    Output line order is undefined; callers must use ``custom_id`` for alignment.
    """
    if not items:
        return {}
    if not _is_openai_native_platform_model(model):
        raise ValueError(
            f"OpenAI Batch API only applies to native OpenAI models; got {model!r}."
        )

    max_bytes = int(os.getenv("OPENAI_BATCH_MAX_INPUT_BYTES", str(190 * 1024 * 1024)))
    lines = [
        _openai_batch_chat_completion_jsonl_line(cid, msgs, model=model)
        for cid, msgs in items
    ]
    total_bytes = sum(_jsonl_line_utf8_byte_len(line) for line in lines)
    chunks = _chunk_jsonl_lines_by_byte_budget(lines, max_bytes)

    if len(chunks) > 1:
        print(
            f"[openai-batch] splitting {len(items)} requests (~{total_bytes / (1024 * 1024):.1f} MB JSONL) "
            f"into {len(chunks)} batch jobs (limit {max_bytes // (1024 * 1024)} MB each)"
        )

    client = _openai_native_client()
    merged: Dict[str, str] = {}
    for i, chunk_lines in enumerate(chunks, start=1):
        chunk_label = f"part {i}/{len(chunks)}" if len(chunks) > 1 else "batch"
        part = _run_one_openai_batch_jsonl(
            client,
            chunk_lines,
            model=model,
            poll_interval_s=poll_interval_s,
            chunk_label=chunk_label,
        )
        overlap = set(merged) & set(part)
        if overlap:
            raise RuntimeError(f"Duplicate custom_ids across batch chunks: {sorted(overlap)[:8]}")
        merged.update(part)

    missing = [cid for cid, _ in items if cid not in merged]
    if missing:
        raise RuntimeError(f"Batch output missing custom_ids ({len(missing)}): {missing[:8]}...")
    return merged


def _vlm_system_instruction(
    *,
    software_tool: str,
    task_name: str,
    task_description: str,
    final_image_note: str,
    prompt_style: str,
) -> str:
    template = VLM_SYSTEM_INSTRUCTION_TEMPLATE
    if prompt_style == "teacher":
        template = VLM_SYSTEM_INSTRUCTION_TEMPLATE_TEACHER
    elif prompt_style == "oracle":
        template = VLM_SYSTEM_INSTRUCTION_TEMPLATE_ORACLE
    task = task_description.strip() or None
    task_block = (task or "").strip()
    if final_image_note:
        task_block = (
            f"{task_block}\n\n{final_image_note}".strip()
            if task_block
            else final_image_note.strip()
        )
    return render_prompt_template(
        template,
        {
            "{{SOFTWARE_TOOL}}": software_tool,
            "{{TASK_NAME}}": task_name,
            "{{TASK_DESCRIPTION}}": task_block,
            "{{COACHING_METHODS_DEFINITION_AND_EXAMPLES}}": COACHING_METHODS_DEFINITION_AND_EXAMPLES,
        },
    )


def _load_task_prompt_context(task_id: str) -> Tuple[str, str, Optional[Path]]:
    task_text = str(task_id or "").strip()
    task_name = f"Task {task_text}" if task_text else "Task"
    task_dir = _REPO_ROOT / "tasks" / task_text
    task_md_path = task_dir / "task.md"
    final_gif_path = task_dir / "final.gif"
    final_png_path = task_dir / "final.png"
    if final_gif_path.is_file():
        final_reference_path: Optional[Path] = final_gif_path
    elif final_png_path.is_file():
        final_reference_path = final_png_path
    else:
        final_reference_path = None

    task_description = ""
    if task_md_path.is_file():
        task_description = task_md_path.read_text(encoding="utf-8").strip()

    return task_name, task_description, final_reference_path


def _reference_image_content_part(
    image_path: Optional[Path], *, max_long_edge: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    if image_path is None or not image_path.is_file():
        return None

    if max_long_edge is None:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        mime, _ = mimetypes.guess_type(str(image_path))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
        }

    if Image is None:
        raise RuntimeError("Pillow is required to resize reference images (PIL.Image import failed).")

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if max(im.size) > int(max_long_edge):
            scale = int(max_long_edge) / max(im.size)
            new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
            im = im.resize(new_size, resample=Image.BILINEAR)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=85, optimize=True)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
    }


def _extract_oracle_coaching_method_tag(turn: Dict[str, Any]) -> str:
    """
    Best-effort extraction of human reference coaching method tag(s) for a turn.
    """
    candidate_keys = (
        "coaching_methods",
    )

    def _normalize(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            items = [str(v).strip() for v in value if str(v).strip()]
            return ", ".join(items)
        return ""

    for key in candidate_keys:
        val = _normalize(turn.get(key))
        if val:
            return val

    annotations = turn.get("annotations")
    if isinstance(annotations, dict):
        for key in candidate_keys:
            val = _normalize(annotations.get(key))
            if val:
                return val

    return ""


def _clear_turn_annotations_for_export(turn: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove copied human annotation labels from exported VLM turns.
    """
    out_turn = dict(turn)
    out_turn["dialogue_acts"] = []
    out_turn["coaching_methods"] = []

    return out_turn


def _utterances_in_pre_coach_window(
    previous_turns: List[Dict[str, Any]],
    *,
    coach_start_seconds: float,
    window_seconds: float,
) -> List[Dict[str, Any]]:
    """Turns whose speech interval overlaps [coach_start - W, coach_start)."""
    ws = max(0.0, float(coach_start_seconds) - float(window_seconds))
    we = float(coach_start_seconds)
    selected: List[Dict[str, Any]] = []
    for u in previous_turns:
        u0 = _timestamp_to_seconds(u.get("start", "0"))
        u1 = _timestamp_to_seconds(u.get("end", "0"))
        if u1 > ws and u0 < we:
            selected.append(u)
    selected.sort(
        key=lambda t: (_timestamp_to_seconds(t.get("start", "0")), int(t.get("turn_id", 0)))
    )
    return selected


def _merge_consecutive_openai_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] = (
                str(out[-1]["content"]).strip() + "\n" + str(m["content"]).strip()
            )
        else:
            out.append(dict(m))
    return out


def _openai_chat_history_from_turns(utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map transcript to OpenAI roles: Student -> user, Coach -> assistant."""
    out: List[Dict[str, Any]] = []
    for u in utterances:
        sp = u.get("speaker", "")
        text = str(u.get("text", "")).strip()
        if not text:
            continue
        if _is_coach_speaker(sp):
            out.append({"role": "assistant", "content": text})
        elif _is_learner_speaker(sp):
            out.append({"role": "user", "content": text})
        else:
            label = _speaker_label_for_prompt(str(sp))
            out.append({"role": "user", "content": f"[{label}] {text}"})
    return _merge_consecutive_openai_messages(out)


def _summarize_openai_content_parts(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a prompt-printable version of OpenAI multimodal parts without huge payloads."""
    out: List[Dict[str, Any]] = []
    for p in parts:
        part_type = p.get("type")
        if part_type == "image_url":
            image_url = dict(p.get("image_url", {}))
            url = str(image_url.get("url", ""))
            if url.startswith("data:") and ";base64," in url:
                prefix, b64 = url.split(";base64,", 1)
                image_url["url"] = f"{prefix};base64,<omitted:{len(b64)} chars>"
            out.append({"type": "image_url", "image_url": image_url})
            continue
        if part_type == "video_url":
            video_url = dict(p.get("video_url", {}))
            url = str(video_url.get("url", ""))
            if url.startswith("data:") and ";base64," in url:
                prefix, b64 = url.split(";base64,", 1)
                video_url["url"] = f"{prefix};base64,<omitted:{len(b64)} chars>"
            out.append({"type": "video_url", "video_url": video_url})
            continue
        out.append(p)
    return out


def _summarize_openai_messages_for_print(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summarized: List[Dict[str, Any]] = []
    for m in messages:
        item = dict(m)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = _summarize_openai_content_parts(content)
        summarized.append(item)
    return summarized


def _build_openai_messages_for_turn(
    *,
    system_instruction: str,
    chat_history: List[Dict[str, Any]],
    modality: str,
    context_window_seconds: float,
    reference_image_part: Optional[Dict[str, Any]],
    image_b64_list: List[str],
    video_mp4_bytes: Optional[bytes] = None,
    prompt_style: str,
    oracle_method_tag: str = "",
    image_url_detail: Optional[str] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_instruction}]
    if modality in {"text-only", "text-image", "text-video"}:
        if chat_history:
            messages.append(
                {
                    "role": "user",
                    "content": f"Recent ({float(context_window_seconds):.0f}s) dialog history (ordered earliest to most recent):",
                }
            )
            messages.extend(chat_history)
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "(No prior chat history in this context window.)",
                }
            )

    ref_part = reference_image_part
    if ref_part is not None and image_url_detail:
        iu = dict(ref_part.get("image_url") or {})
        iu["detail"] = image_url_detail
        ref_part = {"type": "image_url", "image_url": iu}

    if ref_part is not None:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reference final output image:"},
                    ref_part,
                ],
            }
        )
    
    if modality in {"text-image", "image-only"} and image_b64_list:
        frame_parts: List[Dict[str, Any]] = []
        for b64 in image_b64_list:
            url_obj: Dict[str, Any] = {"url": f"data:image/jpeg;base64,{b64}"}
            if image_url_detail:
                url_obj["detail"] = image_url_detail
            frame_parts.append({"type": "image_url", "image_url": url_obj})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Recent ({float(context_window_seconds):.0f}s) learner screen "
                            "(images ordered earliest to most recent):"
                        ),
                    },
                    *frame_parts,
                ],
            }
        )

    if modality == "text-video" and video_mp4_bytes:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Recent ({float(context_window_seconds):.0f}s) learner screen recording "
                            "(video clip immediately before this coach turn):"
                        ),
                    },
                    _video_url_content_part_from_mp4_bytes(video_mp4_bytes),
                ],
            }
        )
    elif modality == "text-video" and image_b64_list:
        frame_parts: List[Dict[str, Any]] = []
        for b64 in image_b64_list:
            url_obj = {"url": f"data:image/jpeg;base64,{b64}"}
            if image_url_detail:
                url_obj["detail"] = image_url_detail
            frame_parts.append({"type": "image_url", "image_url": url_obj})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Recent ({float(context_window_seconds):.0f}s) learner screen recording "
                            "(frames from the clip immediately before this coach turn, "
                            "ordered earliest to most recent):"
                        ),
                    },
                    *frame_parts,
                ],
            }
        )

    if prompt_style == "default":
        messages.append(
            {
                "role": "user",
                "content": (
                    "You are going to speak as the coach to continue the conversation."
                ),
            }
        )
    elif prompt_style == "teacher":
        messages.append(
            {
                "role": "user",
                "content": (
                    "You are going to speak as the coach to continue the conversation. Choose the most appropriate coaching method(s) and give your coaching utterance. ONLY return the coaching utterance, no coaching method tags."
                ),
            }
        )
    
    elif prompt_style == "oracle":
        oracle_tag = oracle_method_tag.strip() or "unknown"
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your next coach utterance should using the following coaching method(s): {oracle_tag}"
                ),
            }
        )


    return messages


def _print_final_prompt_openai(
    *,
    model: str,
    turn_id: int,
    messages: List[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": _VLM_GENERATION_TEMPERATURE,
        "messages": _summarize_openai_messages_for_print(messages),
    }
    if _is_gemini3_model_id(model):
        payload["reasoning_effort"] = _gemini3_thinking_level_str()
    elif _is_openai_gpt_series_model(model):
        payload["reasoning_effort"] = "medium"
    print(f"=== Final prompt payload (OpenAI-compatible) turn_id={turn_id} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("=== End final prompt payload ===")


def _extract_context_frames(
    *,
    clip: Any,
    coach_start_seconds: float,
    context_window_seconds: float,
    fps: float,
) -> Tuple[List[Any], List[float]]:
    """
    Sample frames evenly over [coach_start - W, coach_start), aligned with the text window.
    """
    duration = float(getattr(clip, "duration", 0.0) or 0.0)
    if duration <= 0:
        duration = 1e9

    turn_start = max(0.0, float(coach_start_seconds))
    w = max(1e-3, float(context_window_seconds))
    earliest_context = max(0.0, turn_start - w)
    latest_before_start = max(0.0, min(turn_start - 1e-3, duration - 1e-3))

    if latest_before_start <= earliest_context + 1e-6:
        times = [earliest_context]
    else:
        effective_window = max(0.0, latest_before_start - earliest_context)
        # Use floor so fps=1 yields at most one frame per full second available.
        n = max(1, int(math.floor(effective_window * float(fps))))
        times = [
            earliest_context
            + (latest_before_start - earliest_context) * (i / max(1, n - 1))
            for i in range(n)
        ]

    frame_times = [min(max(earliest_context, t), latest_before_start) for t in times]
    frames = [clip.get_frame(t) for t in frame_times]
    return frames, frame_times


def _extract_frames_from_segment(
    clip: Any,
    *,
    seg_start: float,
    seg_end: float,
    fps: float,
) -> Tuple[List[Any], List[float]]:
    """Sample frames evenly over absolute timeline ``[seg_start, seg_end]`` in ``clip``."""
    seg_start_f = max(0.0, float(seg_start))
    seg_end_f = max(seg_start_f + 1e-3, float(seg_end))
    effective_window = seg_end_f - seg_start_f
    n = max(1, int(math.floor(effective_window * float(fps))))
    if n == 1:
        times = [seg_start_f]
    else:
        times = [
            seg_start_f + effective_window * (i / max(1, n - 1))
            for i in range(n)
        ]
    frame_times = [min(max(seg_start_f, t), seg_end_f - 1e-6) for t in times]
    frames = [clip.get_frame(t) for t in frame_times]
    return frames, frame_times


_GEMINI_TEXT_VIDEO_MAX_FRAMES = 24


def _maybe_cap_gemini_text_video_frames(
    frames: List[Any],
    frame_times: List[float],
) -> Tuple[List[Any], List[float]]:
    return _uniform_subsample_pairs(frames, frame_times, _GEMINI_TEXT_VIDEO_MAX_FRAMES)


def _pre_turn_video_segment_bounds(
    *,
    clip: Any,
    start_seconds: float,
    window_seconds: float,
) -> Tuple[float, float]:
    """Timeline [seg_start, seg_end] for the pre-coach video segment used in text-video prompts."""
    duration = float(getattr(clip, "duration", 0.0) or 0.0)
    if duration <= 0:
        duration = max(float(start_seconds), 1.0)
    turn_start = max(0.0, float(start_seconds))
    seg_start = max(0.0, turn_start - float(window_seconds))
    seg_end = min(duration, max(seg_start + 1e-3, turn_start))
    return seg_start, seg_end


def _subclip_video_segment(clip: Any, seg_start: float, seg_end: float) -> Any:
    """MoviePy 1.x ``subclip`` / 2.x ``subclipped``."""
    if hasattr(clip, "subclipped"):
        return clip.subclipped(seg_start, seg_end)
    return clip.subclip(seg_start, seg_end)


def _write_video_segment_mp4(segment: Any, path: Path) -> None:
    kwargs: Dict[str, Any] = {"codec": "libx264", "audio": False}
    for extra in ({"logger": None}, {"verbose": False}, {}):
        try:
            segment.write_videofile(str(path), **kwargs, **extra)
            return
        except TypeError:
            continue
    segment.write_videofile(str(path), **kwargs)


def _extract_video_segment_mp4_bytes(
    clip: Any,
    *,
    seg_start: float,
    seg_end: float,
) -> bytes:
    """Cut ``[seg_start, seg_end]`` from an open MoviePy clip and return MP4 bytes."""
    segment = _subclip_video_segment(clip, seg_start, seg_end)
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _write_video_segment_mp4(segment, tmp_path)
        data = tmp_path.read_bytes()
        max_b = _vlm_video_max_bytes()
        if len(data) > max_b:
            raise ValueError(
                f"Encoded video segment [{seg_start:.1f}s, {seg_end:.1f}s] is {len(data)} bytes "
                f"(max {max_b}). Shorten --vlm-context-seconds or raise VLM_VIDEO_MAX_BYTES."
            )
        return data
    finally:
        try:
            segment.close()
        except Exception:  # pragma: no cover
            pass
        tmp_path.unlink(missing_ok=True)


def _video_url_content_part_from_mp4_bytes(mp4_bytes: bytes) -> Dict[str, Any]:
    """OpenAI-compatible ``video_url`` part (e.g. OpenRouter Qwen)."""
    b64 = base64.b64encode(mp4_bytes).decode("ascii")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{b64}"},
    }


def _load_turns(turns_path: Path) -> List[Dict[str, Any]]:
    turns = json.loads(turns_path.read_text(encoding="utf-8"))
    if not isinstance(turns, list):
        raise ValueError(f"Expected turns JSON list at {turns_path}")
    normalized: List[Dict[str, Any]] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        if "turn_id" not in t or "start" not in t or "end" not in t:
            continue
        normalized.append(t)
    normalized.sort(key=lambda u: (_timestamp_to_seconds(u.get("start", "0")), int(u.get("turn_id", 0))))
    return normalized


def _effective_vlm_modality(
    vlm_input_modality: str,
    vlm_model: str,
    gemini_video_input: bool,
) -> str:
    modality = _normalize_vlm_modality(vlm_input_modality)
    if gemini_video_input and str(vlm_model).strip().lower().startswith("gemini"):
        modality = "text-video"
    return modality


def print_sample_input_chats(
    *,
    turns_path: Path,
    video_path: Path,
    context_window_seconds: float,
    vlm_fps: float,
    vlm_input_modality: str,
    vlm_model: str,
    gemini_video_input: bool,
    sample_turn_ids: List[int],
    vlm_prompt_style: str = "default",
) -> None:
    """
    Print one sample conversation (OpenAI chat_history + final user text) per requested coach turn_id.
    Does not call any API. For text-image / image-only / text-video, opens the video when the file exists.
    """
    turns = _load_turns(turns_path)
    modality = _effective_vlm_modality(vlm_input_modality, vlm_model, gemini_video_input)
    if modality == "text-video" and not _supports_native_video_vlm(vlm_model):
        print(
            "[sample chats] text-video requires a Gemini or Qwen model id; "
            "showing transcript layout only (no native clip)."
        )

    clip: Any = None
    try:
        for tid in sample_turn_ids:
            turn_index: Optional[int] = None
            for i, t in enumerate(turns):
                if not _is_coach_speaker(t.get("speaker", "")):
                    continue
                if int(t.get("turn_id", -1)) == int(tid):
                    turn_index = i
                    break

            if turn_index is None:
                print(f"=== Sample chats: no coach turn with turn_id={tid} in {turns_path.name} ===\n")
                continue

            turn = turns[turn_index]
            turn_id = int(turn.get("turn_id"))
            start_seconds = _timestamp_to_seconds(turn.get("start", "0"))
            end_seconds = _timestamp_to_seconds(turn.get("end", "0"))

            previous_turns = turns[:turn_index]
            last_utterances = _utterances_in_pre_coach_window(
                previous_turns,
                coach_start_seconds=start_seconds,
                window_seconds=context_window_seconds,
            )
            chat_history = _openai_chat_history_from_turns(last_utterances)
            # if chat_history and chat_history[0]["role"] == "assistant":
            #     chat_history = [
            #         {
            #             "role": "user",
            #             "content": "(No prior student line in this context window.)",
            #         },
            #         *chat_history,
            #     ]

            frame_times: Optional[List[float]] = None
            video_start: Optional[float] = None
            video_end: Optional[float] = None
            image_note: Dict[str, Any] = {}
            sample_video_bytes: Optional[bytes] = None
            sample_image_b64_list: List[str] = []

            if modality in {"text-image", "image-only"}:
                if video_path.is_file():
                    if clip is None:
                        clip = moviepy.VideoFileClip(str(video_path))
                    _, frame_times = _extract_context_frames(
                        clip=clip,
                        coach_start_seconds=start_seconds,
                        context_window_seconds=context_window_seconds,
                        fps=vlm_fps,
                    )
                    image_note = {
                        "kind": "image_url",
                        "count": len(frame_times or []),
                        "frame_times_s_from_video_start": frame_times or [],
                    }
                else:
                    print(f"[sample chats] turn_id={tid}: video missing, no frame times ({video_path})")
            elif modality == "text-video":
                if video_path.is_file():
                    if clip is None:
                        clip = moviepy.VideoFileClip(str(video_path))
                    video_start, video_end = _pre_turn_video_segment_bounds(
                        clip=clip,
                        start_seconds=start_seconds,
                        window_seconds=float(context_window_seconds),
                    )
                    if _uses_native_video_url_part(vlm_model):
                        sample_video_bytes = _extract_video_segment_mp4_bytes(
                            clip, seg_start=video_start, seg_end=video_end
                        )
                        image_note = {
                            "kind": "video_url",
                            "segment_s": [video_start, video_end],
                            "mp4_bytes": len(sample_video_bytes),
                        }
                    else:
                        frames, frame_times = _extract_frames_from_segment(
                            clip,
                            seg_start=video_start,
                            seg_end=video_end,
                            fps=vlm_fps,
                        )
                        frames, frame_times = _maybe_cap_gemini_text_video_frames(
                            frames, frame_times
                        )
                        sample_image_b64_list = _frames_to_base64_jpeg(frames)
                        image_note = {
                            "kind": "video_frames_jpeg",
                            "segment_s": [video_start, video_end],
                            "frame_count": len(frames),
                            "frame_times_s": frame_times,
                        }
                else:
                    print(f"[sample chats] turn_id={tid}: video missing ({video_path})")

            openai_messages = _build_openai_messages_for_turn(
                system_instruction=_vlm_system_instruction(
                    software_tool="software tool",
                    task_name="Task",
                    task_description="",
                    final_image_note="",
                    prompt_style=str(vlm_prompt_style or "default").strip().lower(),
                ),
                chat_history=(chat_history if modality != "image-only" else []),
                modality=modality,
                context_window_seconds=float(context_window_seconds),
                reference_image_part=None,
                image_b64_list=sample_image_b64_list,
                video_mp4_bytes=sample_video_bytes,
                prompt_style=str(vlm_prompt_style or "default").strip().lower(),
                oracle_method_tag=_extract_oracle_coaching_method_tag(turn),
                image_url_detail=_vlm_image_url_detail_for_model(vlm_model),
            )

            print(f"=== Sample input: coach turn_id={turn_id} | modality={modality} ===")
            print("last_utterances_in_window (raw turns):")
            print(json.dumps(last_utterances, ensure_ascii=False, indent=2))
            print("\nchat_history (OpenAI: prior transcript → user/assistant messages):")
            print(json.dumps(chat_history, ensure_ascii=False, indent=2))
            print("\nopenai_messages (final prompt payload body):")
            print(json.dumps(_summarize_openai_messages_for_print(openai_messages), ensure_ascii=False, indent=2))
            if image_note:
                print("\nmultimodal_attachment_summary:")
                print(json.dumps(image_note, ensure_ascii=False, indent=2))
            print()
    finally:
        if clip is not None:
            clip.close()


def _vlm_openai_messages_for_coach_turn_index(
    *,
    turns: List[Dict[str, Any]],
    turn_index: int,
    clip: Any,
    system_instruction: str,
    reference_image_part: Optional[Dict[str, Any]],
    context_window_seconds: float,
    vlm_fps: float,
    modality: str,
    style: str,
    vlm_model: str,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Build OpenAI Chat Completions ``messages`` for one coach turn. Returns ``(turn_id, messages)``.
    """
    turn = turns[turn_index]
    turn_id = int(turn.get("turn_id"))
    start_seconds = _timestamp_to_seconds(turn.get("start", "0"))

    previous_turns = turns[:turn_index]
    last_utterances = _utterances_in_pre_coach_window(
        previous_turns,
        coach_start_seconds=start_seconds,
        window_seconds=context_window_seconds,
    )
    chat_history = _openai_chat_history_from_turns(last_utterances)
    if chat_history and chat_history[0]["role"] == "assistant":
        chat_history = [
            {
                "role": "user",
                "content": "(No prior student line in this context window.)",
            },
            *chat_history,
        ]

    image_b64_list: List[str] = []
    video_mp4_bytes: Optional[bytes] = None

    if modality in {"text-image", "image-only"}:
        frames, frame_times = _extract_context_frames(
            clip=clip,
            coach_start_seconds=start_seconds,
            context_window_seconds=context_window_seconds,
            fps=vlm_fps,
        )
        frames, _frame_times = _maybe_cap_vlm_context_frames(
            frames, frame_times, vlm_model=vlm_model
        )
        image_b64_list = _frames_to_base64_jpeg(frames)
    elif modality == "text-video":
        if not _supports_native_video_vlm(vlm_model):
            raise ValueError(
                f"text-video modality requires a Gemini or Qwen model id; got {vlm_model!r}."
            )
        seg_start, seg_end = _pre_turn_video_segment_bounds(
            clip=clip,
            start_seconds=start_seconds,
            window_seconds=float(context_window_seconds),
        )
        if _uses_native_video_url_part(vlm_model):
            video_mp4_bytes = _extract_video_segment_mp4_bytes(
                clip, seg_start=seg_start, seg_end=seg_end
            )
        else:
            frames, frame_times = _extract_frames_from_segment(
                clip,
                seg_start=seg_start,
                seg_end=seg_end,
                fps=vlm_fps,
            )
            frames, frame_times = _maybe_cap_gemini_text_video_frames(frames, frame_times)
            image_b64_list = _frames_to_base64_jpeg(frames)

    detail = _vlm_image_url_detail_for_model(vlm_model)
    openai_messages = _build_openai_messages_for_turn(
        system_instruction=system_instruction,
        chat_history=(chat_history if modality in {"text-image", "text-video"} else []),
        modality=modality,
        context_window_seconds=float(context_window_seconds),
        reference_image_part=reference_image_part,
        image_b64_list=image_b64_list,
        video_mp4_bytes=video_mp4_bytes,
        prompt_style=style,
        oracle_method_tag=_extract_oracle_coaching_method_tag(turn),
        image_url_detail=detail,
    )

    return turn_id, openai_messages


def vlm_generate_coach_utterance_for_turn_index(
    *,
    turns: List[Dict[str, Any]],
    turn_index: int,
    clip: Any,
    system_instruction: str,
    reference_image_part: Optional[Dict[str, Any]],
    vlm_model: str,
    context_window_seconds: float,
    vlm_fps: float,
    modality: str,
    style: str,
    print_final_prompt: bool,
    use_openrouter: bool = False,
    client: Optional[Any] = None,
) -> str:
    """
    Generate one coach utterance for the coach turn at ``turn_index`` in ``turns`` (full transcript order).
    Caller supplies an open MoviePy clip and pre-built system instruction / reference image part.
    """
    turn_id, openai_messages = _vlm_openai_messages_for_coach_turn_index(
        turns=turns,
        turn_index=turn_index,
        clip=clip,
        system_instruction=system_instruction,
        reference_image_part=reference_image_part,
        context_window_seconds=context_window_seconds,
        vlm_fps=vlm_fps,
        modality=modality,
        style=style,
        vlm_model=vlm_model,
    )
    if print_final_prompt:
        _print_final_prompt_openai(
            model=vlm_model,
            turn_id=turn_id,
            messages=openai_messages,
        )
    raw = openai_vlm_chat_completion(
        messages=openai_messages,
        model=vlm_model,
        use_openrouter=use_openrouter,
        client=client,
    )
    return str(raw).strip()


def run_vlm_eval(
    *,
    turns_path: Path,
    video_path: Path,
    software_tool: str,
    task_name: str,
    task_description: str = "",
    reference_final_image: Optional[Path] = None,
    vlm_model: str = "gpt-5.4",
    context_window_seconds: float = 10.0,
    vlm_fps: float = 1.0,
    max_coach_turns: Optional[int] = None,
    vlm_input_modality: str = "text-image",
    vlm_prompt_style: str = "default",
    gemini_video_input: bool = False,
    print_final_prompt: bool = False,
    use_openai_batch: bool = False,
    openai_batch_poll_interval_s: float = 15.0,
    use_openrouter: bool = False,
) -> List[Dict[str, Any]]:
    turns = _load_turns(turns_path)
    if not turns:
        raise ValueError(f"No turns found in {turns_path}")

    coach_indices = [i for i, t in enumerate(turns) if _is_coach_speaker(t.get("speaker", ""))]
    if not coach_indices:
        raise ValueError(f"No coach utterances found in {turns_path} (speaker not recognized as Coach).")

    if _normalize_vlm_modality(vlm_input_modality) not in _VLM_INPUT_MODALITIES:
        raise ValueError(
            f"Invalid vlm_input_modality {vlm_input_modality!r}; "
            f"expected one of {sorted(_VLM_INPUT_MODALITIES)}"
        )
    modality = _effective_vlm_modality(vlm_input_modality, vlm_model, gemini_video_input)
    if modality == "text-video" and not _supports_native_video_vlm(vlm_model):
        raise ValueError(
            f"text-video modality requires a Gemini or Qwen model id; got {vlm_model!r}."
        )
    if modality == "text-video" and use_openai_batch:
        raise ValueError("text-video is not supported with --openai-batch.")

    style = (vlm_prompt_style or "default").strip().lower()
    if style not in _VLM_PROMPT_STYLES:
        raise ValueError(
            f"Invalid vlm_prompt_style {vlm_prompt_style!r}; "
            f"expected one of {sorted(_VLM_PROMPT_STYLES)}"
        )

    system_instruction = _vlm_system_instruction(
        software_tool=software_tool,
        task_name=task_name,
        task_description=task_description,
        final_image_note=(
            "A reference image of the expected final output is available and attached in the user message."
            if reference_final_image is not None
            else ""
        ),
        prompt_style=style,
    )
    reference_image_part = _reference_image_content_part(
        reference_final_image,
        max_long_edge=_vlm_reference_image_max_long_edge(vlm_model),
    )

    if _is_qwen_vlm_model(vlm_model) and modality in {"text-image", "image-only"}:
        print(
            f"Qwen VL: at most {_qwen_max_context_frames()} context frames per turn "
            "(uniform subsample over the window). "
            "Override with env QWEN_VLM_MAX_CONTEXT_FRAMES."
        )
    if modality == "text-video":
        if _uses_native_video_url_part(vlm_model):
            print(
                f"text-video: native MP4 clip per turn (max {_vlm_video_max_bytes()} bytes); "
                "override with VLM_VIDEO_MAX_BYTES."
            )
        else:
            print(
                "text-video: Gemini AI Studio does not support video_url; "
                f"sending up to {_GEMINI_TEXT_VIDEO_MAX_FRAMES} JPEG frames sampled from "
                "the pre-coach clip instead."
            )

    results: List[Dict[str, Any]] = []

    if use_openai_batch:
        if use_openrouter or _is_openrouter_open_model_slug(vlm_model):
            raise ValueError("OpenRouter routing is incompatible with --openai-batch (use direct OpenAI models).")
        if not _is_openai_native_platform_model(vlm_model):
            raise ValueError(
                "--openai-batch only applies to native OpenAI models on api.openai.com "
                "(not Gemini or OpenRouter Qwen/Llama slugs). "
                f"Got model={vlm_model!r}."
            )
        limited_indices = list(coach_indices)
        if max_coach_turns is not None:
            limited_indices = limited_indices[: int(max_coach_turns)]

        clip = moviepy.VideoFileClip(str(video_path))
        try:
            batch_items: List[Tuple[str, List[Dict[str, Any]]]] = []
            meta: List[Tuple[int, Dict[str, Any]]] = []
            for ci, turn_index in enumerate(limited_indices):
                turn = turns[turn_index]
                turn_id, openai_messages = _vlm_openai_messages_for_coach_turn_index(
                    turns=turns,
                    turn_index=turn_index,
                    clip=clip,
                    system_instruction=system_instruction,
                    reference_image_part=reference_image_part,
                    context_window_seconds=context_window_seconds,
                    vlm_fps=vlm_fps,
                    modality=modality,
                    style=style,
                    vlm_model=vlm_model,
                )
                custom_id = f"tid{turn_id}_i{ci}"
                batch_items.append((custom_id, openai_messages))
                meta.append((turn_index, turn))
                if print_final_prompt:
                    _print_final_prompt_openai(
                        model=vlm_model,
                        turn_id=turn_id,
                        messages=openai_messages,
                    )

            batch_out = run_openai_batch_vlm_chat_completions(
                batch_items,
                model=vlm_model,
                poll_interval_s=float(openai_batch_poll_interval_s),
            )

            for ci in range(len(meta)):
                _turn_index, turn = meta[ci]
                custom_id = batch_items[ci][0]
                turn_id = int(turn.get("turn_id"))
                generated = batch_out[custom_id]
                out_turn = _clear_turn_annotations_for_export(turn)
                out_turn["speaker"] = "Coach"
                out_turn["text"] = generated
                results.append(out_turn)
                print(f"[{ci+1}/{len(batch_items)}] turn_id={turn_id} -> {generated[:60]!r}")
        finally:
            clip.close()

        return results

    vlm_client = _openai_compatible_client_for_model(vlm_model, use_openrouter=use_openrouter)

    clip = moviepy.VideoFileClip(str(video_path))
    try:
        for ci, turn_index in enumerate(coach_indices):
            if max_coach_turns is not None and ci >= max_coach_turns:
                break

            turn = turns[turn_index]
            turn_id = int(turn.get("turn_id"))

            generated = vlm_generate_coach_utterance_for_turn_index(
                turns=turns,
                turn_index=turn_index,
                clip=clip,
                system_instruction=system_instruction,
                reference_image_part=reference_image_part,
                vlm_model=vlm_model,
                context_window_seconds=context_window_seconds,
                vlm_fps=vlm_fps,
                modality=modality,
                style=style,
                print_final_prompt=print_final_prompt,
                use_openrouter=use_openrouter,
                client=vlm_client,
            )

            out_turn = _clear_turn_annotations_for_export(turn)
            out_turn["speaker"] = "Coach"
            out_turn["text"] = generated
            results.append(out_turn)
            print(f"[{ci+1}/{len(coach_indices)}] turn_id={turn_id} -> {generated[:60]!r}")
    finally:
        clip.close()

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="VLM coaching utterance generation.")
    p.add_argument(
        "-t",
        "--task",
        default="0101",
        help="task id",
    )
    p.add_argument(
        "-s",
        "--session",
        default="hh_0101_example",
        help="Session folder under data/sessions (default: hh_0101_example).",
    )
    p.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./turns_vlm_gpt-5.4.json",
        help="Not used; outputs are vlms/<task>/<session>/turns_vlm_<model>.json or turns_vlm_<model>_<N>.json.",
    )
    p.add_argument(
        "--video",
        type=str,
        default="screen.mp4",
        help="Video filename under .../<session>/media/ (e.g. screen.mp4 or screen-2.mp4).",
    )
    p.add_argument(
        "--turns",
        type=str,
        default="turns.json",
        help="Turns filename inside the session folder (default: turns.json).",
    )
    p.add_argument(
        "--vlm_models",
        nargs="+",
        default=["google/gemini-3.1-pro-preview"],
        help='VLM model names to evaluate.',
    )
    p.add_argument(
        "--gemini_video_input",
        action="store_true",
        help="Deprecated alias: for Gemini models, same as --vlm-input-modality text-video.",
    )
    p.add_argument(
        "--vlm-context-seconds",
        type=float,
        default=30.0,
        metavar="SEC",
        help="Align transcript + visuals to the same pre-coach window: last SEC seconds before the coach turn (e.g. 10, 20, 30).",
    )
    p.add_argument(
        "--vlm-input-modality",
        choices=("text-only", "text-image", "image-only", "text-video"),
        default="text-image",
        help=(
            "Ablation: transcript only; transcript + JPEG frames; images only; or transcript + "
            "pre-coach video (Qwen: native MP4 via video_url; Gemini: JPEG frames from that clip)."
        ),
    )
    p.add_argument(
        "--vlm-prompt-style",
        choices=tuple(sorted(_VLM_PROMPT_STYLES)),
        default="default",
        help="Ablation: default system prompt vs. teacher-emphasis system prompt.",
    )
    p.add_argument(
        "--vlm-fps",
        type=float,
        default=1.0,
        help="Frames/sec to sample over the pre-turn context window (text-image/image-only).",
    )
    p.add_argument(
        "--vlm-result-tag",
        type=str,
        default="",
        help="If set, JSON model key becomes '<model>(<tag>)' so ablations do not overwrite (e.g. ctx20-teacher).",
    )
    p.add_argument(
        "--max-coach-turns",
        type=int,
        default=0,
        help="Limit coach turns per run (0 = no limit).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview selected turns/models/output files without API calls or file writes.",
    )
    p.add_argument(
        "--print-system-prompt",
        action="store_true",
        help="Print the rendered system prompt used for this run.",
    )
    p.add_argument(
        "--print-final-prompt",
        action="store_true",
        help="Print the final model payload (messages/history + final user content) before each call.",
    )
    p.add_argument(
        "--openai-batch",
        action="store_true",
        help=(
            "For native OpenAI models only: submit all coach-turn chat completions as one OpenAI Batch job "
            "(lower cost, higher batch rate limits; completion within up to 24 hours). "
            "Not used for Gemini or OpenRouter (Qwen/Llama) models."
        ),
    )
    p.add_argument(
        "--openrouter",
        action="store_true",
        help=(
            "Route VLM chat completions through OpenRouter (https://openrouter.ai/api/v1). "
            "Models whose ids start with ``qwen``, ``meta-llama``, or ``anthropic/`` use OpenRouter automatically "
            "even without this flag. "
            "Requires OPENROUTER_API_KEY; optional OPENROUTER_HTTP_REFERER and OPENROUTER_TITLE for attribution. "
            "For ``qwen/*`` text-image runs, context frames are subsampled (default cap 12 per turn; "
            "set env QWEN_VLM_MAX_CONTEXT_FRAMES to change) and task reference images are downscaled; "
            "set QWEN_VLM_IMAGE_DETAIL to low|high|auto|omit to tune image_url.detail (default low)."
        ),
    )
    p.add_argument(
        "--openai-batch-poll-interval",
        type=float,
        default=15.0,
        metavar="SEC",
        help="Seconds between OpenAI batch status polls when using --openai-batch (default: 15).",
    )
    p.add_argument(
        "--print-sample-chats",
        action="store_true",
        help=(
            "Print sample prior-window chat_history (OpenAI format) and final user text "
            "for selected coach turn_ids (see --sample-chat-turn-ids). No API calls."
        ),
    )
    p.add_argument(
        "--sample-chat-turn-ids",
        type=int,
        nargs="+",
        default=[5, 10],
        metavar="TURN_ID",
        help="Coach turn_id values for --print-sample-chats (default: 5 10).",
    )
    args = p.parse_args()
    effective_fps = float(args.vlm_fps)
    if effective_fps <= 0:
        raise ValueError("--vlm-fps must be > 0.")

    openrouter_effective = bool(args.openrouter) or any(
        _is_openrouter_open_model_slug(m) for m in args.vlm_models
    )
    if openrouter_effective and args.openai_batch:
        raise ValueError("OpenRouter routing cannot be combined with --openai-batch.")

    if openrouter_effective and not args.dry_run:
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouter-routed models "
                "(``--openrouter`` or ids starting with qwen, meta-llama, anthropic/) "
                "unless --dry-run."
            )

    if args.openai_batch and not args.dry_run:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required when using --openai-batch.")
        for m in args.vlm_models:
            if not _is_openai_native_platform_model(m):
                raise ValueError(
                    f"--openai-batch only supports native OpenAI models; {m!r} is routed to another provider."
                )

    session_dir = resolve_session_dir(args.session, task=args.task)
    session_processed = session_dir
    session_vlms = eval_output_dir(session_dir.name)
    media_dir = session_dir / "media" if (session_dir / "media").is_dir() else session_dir
    software_tool = software_tool_for_task(args.task)
    task_name, task_description, final_reference_path = _load_task_prompt_context(args.task)
    resolved_style = (args.vlm_prompt_style or "default").strip().lower()
    rendered_system_prompt = _vlm_system_instruction(
        software_tool=software_tool,
        task_name=task_name,
        task_description=task_description,
        final_image_note=(
            "A reference image of the expected final output is available and attached in the user message."
            if final_reference_path is not None
            else ""
        ),
        prompt_style=resolved_style,
    )
    if args.print_system_prompt:
        print("=== Rendered System Prompt ===")
        print(rendered_system_prompt)
        print("=== End System Prompt ===")
    default_video = "screen.mp4"
    video_arg = args.video.strip()

    turns_paths = [session_processed / args.turns.strip()]

    def video_name_for_turns_file(turns_path: Path) -> str:
        if len(turns_paths) > 1:
            return video_filename_for_turns_splitted(turns_path.name)
        if video_arg != default_video:
            return video_arg
        return video_filename_for_turns_splitted(turns_path.name)

    for turns_path in turns_paths:
        video_name = video_name_for_turns_file(turns_path)
        video_path = media_dir / video_name
        print(f"Turns: {turns_path.name} | Video: {video_name}")

        if args.print_sample_chats:
            if turns_path.is_file():
                print_sample_input_chats(
                    turns_path=turns_path,
                    video_path=video_path,
                    context_window_seconds=float(args.vlm_context_seconds),
                    vlm_fps=effective_fps,
                    vlm_input_modality=args.vlm_input_modality,
                    vlm_model=(args.vlm_models[0] if args.vlm_models else "gpt-5"),
                    gemini_video_input=args.gemini_video_input,
                    sample_turn_ids=list(args.sample_chat_turn_ids),
                    vlm_prompt_style=args.vlm_prompt_style,
                )
            else:
                print(f"[sample chats] Missing turns file: {turns_path}")

        max_turns = args.max_coach_turns if args.max_coach_turns > 0 else None
        tag = (args.vlm_result_tag or "").strip()
        
        def model_output_label(model_name: str) -> str:
            # Filename segments separated by '_'; modality uses hyphens (text-image, …).
            effective_modality = _effective_vlm_modality(
                args.vlm_input_modality,
                model_name,
                args.gemini_video_input,
            )
            style_part = str(args.vlm_prompt_style or "default").strip().lower()
            modality_part = effective_modality
            context_part = f"ctx{int(float(args.vlm_context_seconds))}"
            sample_part = f"fps{str(float(effective_fps)).rstrip('0').rstrip('.')}"
            parts = [model_name, modality_part, style_part, context_part, sample_part]
            if tag:
                parts.append(tag)
            return "_".join(parts)

        if args.dry_run:
            if not turns_path.is_file():
                print(f"[DRY RUN] Missing turns file: {turns_path}")
                continue
            if not video_path.is_file():
                print(f"[DRY RUN] Missing video file: {video_path}")
                continue

            turns = _load_turns(turns_path)
            coach_turns = [t for t in turns if _is_coach_speaker(t.get("speaker", ""))]
            coach_count = len(coach_turns)
            if max_turns is not None:
                coach_count = min(coach_count, max_turns)
            print(
                f"[DRY RUN] Would evaluate {coach_count} coach turns from {turns_path.name} "
                f"using tool={software_tool}, modality={args.vlm_input_modality}, "
                f"context={args.vlm_context_seconds}s, "
                f"fps={effective_fps}"
                f"{'; OpenAI Batch API (one job per model)' if args.openai_batch else ''}"
                f"{'; OpenRouter' if openrouter_effective else ''}."
            )
            for model_name in args.vlm_models:
                label = model_output_label(model_name)
                model_out_path = session_vlms / vlm_output_filename_for_model(turns_path.name, label)
                print(
                    f"[DRY RUN] model={model_name} -> {model_out_path} "
                    f"(JSON shape: {{condition, turns}})"
                )
            continue

        with ThreadPoolExecutor(max_workers=len(args.vlm_models)) as executor:
            futures = {}
            for model_name in args.vlm_models:
                print(f"Submitting eval for model={model_name}")
                futures[executor.submit(
                    run_vlm_eval,
                    turns_path=turns_path,
                    video_path=video_path,
                    software_tool=software_tool,
                    task_name=task_name,
                    task_description=task_description,
                    reference_final_image=final_reference_path,
                    vlm_model=model_name,
                    context_window_seconds=float(args.vlm_context_seconds),
                    vlm_fps=effective_fps,
                    max_coach_turns=max_turns,
                    vlm_input_modality=args.vlm_input_modality,
                    vlm_prompt_style=args.vlm_prompt_style,
                    gemini_video_input=args.gemini_video_input,
                    print_final_prompt=args.print_final_prompt,
                    use_openai_batch=args.openai_batch,
                    openai_batch_poll_interval_s=float(args.openai_batch_poll_interval),
                    use_openrouter=bool(args.openrouter)
                    or _is_openrouter_open_model_slug(model_name),
                )] = model_name

            for future in as_completed(futures):
                model_name = futures[future]
                model_results = future.result()
                print(f"Completed eval for model={model_name}")
                label = model_output_label(model_name)
                model_out_path = session_vlms / vlm_output_filename_for_model(turns_path.name, label)
                model_rows = sorted(
                    model_results,
                    key=lambda r: (
                        _timestamp_to_seconds(str(r.get("start", "0"))),
                        int(r.get("turn_id", 0)),
                    ),
                )
                tag_part = (args.vlm_result_tag or "").strip()
                prompt_condition = str(args.vlm_prompt_style or "default").strip().lower()
                if tag_part:
                    prompt_condition = f"{prompt_condition}|tag={tag_part}"
                effective_modality = _effective_vlm_modality(
                    args.vlm_input_modality,
                    model_name,
                    args.gemini_video_input,
                )
                output_doc: Dict[str, Any] = {
                    "condition": {
                        "input_context": float(args.vlm_context_seconds),
                        "sample_rate": float(effective_fps),
                        "fps": effective_fps,
                        "input_modality": effective_modality,
                        "prompt_condition": prompt_condition,
                        "model": model_name,
                        "openai_batch": bool(args.openai_batch),
                        "openrouter": bool(args.openrouter)
                        or _is_openrouter_open_model_slug(model_name),
                        "software_tool": software_tool,
                        "task": args.task,
                        "session": args.session,
                        "turns_source": turns_path.name,
                    },
                    "turns": model_rows,
                }
                model_out_path.parent.mkdir(parents=True, exist_ok=True)
                model_out_path.write_text(
                    json.dumps(output_doc, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"Wrote {model_out_path.name}")


if __name__ == "__main__":
    main()