import asyncio
import base64
import json
import logging
import os
import re
import warnings
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vision_agents.core import Agent, AgentLauncher, User
from vision_agents.plugins import gemini, getstream
from vision_agents.plugins.openai import ChatCompletionsVLM

load_dotenv()


def _stream_timeout_seconds() -> float:
    raw = os.getenv("STREAM_TIMEOUT", "30")
    try:
        value = float(raw)
    except ValueError:
        value = 30.0
    return max(value, 6.0)


# GetStream SDK default is 6s; agent authenticate() often needs longer under load.
_STREAM_TIMEOUT = _stream_timeout_seconds()
os.environ.setdefault("STREAM_TIMEOUT", str(_STREAM_TIMEOUT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digital_coach_api")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="dataclasses_json.core"
)

_STREAM_RETRY_EXCEPTIONS = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)

ModelProvider = Literal["gemini", "openai", "qwen", "openrouter"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_TASKS_DIR = Path(__file__).resolve().parent / "web" / "src" / "tasks"
_DEFAULT_TASK_ID = "0101"
_TASK_SOFTWARE_BY_PREFIX: dict[str, str] = {
    "01": "Figma",
    "02": "Blender",
    "03": "Onshape",
    "04": "FL Studio",
    "05": "Excel",
}
_REFERENCE_IMAGE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("final.png", "image/png"),
    ("final.gif", "image/gif"),
    ("final.jpg", "image/jpeg"),
    ("final.jpeg", "image/jpeg"),
    ("final.webp", "image/webp"),
)


def _task_reference_data_url(task_id: str | None) -> str | None:
    """Data URL for the task's reference outcome image (e.g. final.png), if present."""
    if not task_id or task_id == "default":
        return None
    safe = "".join(ch for ch in task_id if ch.isalnum() or ch in "_-")
    if not safe or safe != task_id:
        return None
    folder = _TASKS_DIR / safe
    if not folder.is_dir():
        return None
    for filename, mime in _REFERENCE_IMAGE_CANDIDATES:
        path = folder / filename
        if path.is_file():
            b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{b64}"
    return None


def _safe_task_id(task_id: str | None) -> str | None:
    if not task_id or task_id == "default":
        return None
    safe = "".join(ch for ch in task_id if ch.isalnum() or ch in "_-")
    if not safe or safe != task_id:
        return None
    return safe


def _load_task_markdown(task_id: str | None) -> str | None:
    safe = _safe_task_id(task_id)
    if not safe:
        return None
    path = _TASKS_DIR / safe / "task.md"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _extract_task_section(markdown: str, section_title: str) -> str:
    escaped = re.escape(section_title)
    pattern = rf"^#{{2,6}}\s+{escaped}\s*\n([\s\S]*?)(?=\n#{{1,6}}\s+|$)"
    match = re.search(pattern, markdown, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _build_task_description_block(markdown: str) -> str:
    sections: list[str] = []
    for title, label in (
        ("Task Description", "Task Description"),
        ("Learning Objectives", "Learning Objectives"),
        ("Intermediate Steps", "Intermediate Steps"),
        ("Final Outcome", "Final Outcome"),
    ):
        body = _extract_task_section(markdown, title)
        if body:
            sections.append(f"{label}:\n{body}")
    return "\n\n".join(sections)


def _infer_software_tool(task_id: str | None, task_name: str = "") -> str:
    prefix = (task_id or "")[:2]
    if prefix in _TASK_SOFTWARE_BY_PREFIX:
        return _TASK_SOFTWARE_BY_PREFIX[prefix]
    lowered = task_name.lower()
    for tool in ("figma", "blender", "onshape", "fl studio", "excel"):
        if tool in lowered:
            if tool == "fl studio":
                return "FL Studio"
            if tool == "onshape":
                return "Onshape"
            return tool.title()
    return DEFAULT_SOFTWARE_TOOL


def _render_instructions(*, software_tool: str, task_name: str, task_body: str) -> str:
    task_block = f"## Task\nYou are working on {task_name}.\n\n{task_body}"
    return (
        VLM_SYSTEM_INSTRUCTION_TEMPLATE.replace("{{SOFTWARE_TOOL}}", software_tool)
        .replace(
            "{{COACHING_METHODS_DEFINITION_AND_EXAMPLES}}",
            COACHING_METHODS_DEFINITION_AND_EXAMPLES,
        )
        .replace("{{TASK_DESCRIPTION}}", task_block)
    )


def build_instructions_for_task_id(task_id: str | None) -> str:
    markdown = _load_task_markdown(task_id)
    if not markdown:
        if task_id and task_id != _DEFAULT_TASK_ID:
            return build_instructions_for_task_id(_DEFAULT_TASK_ID)
        return _render_instructions(
            software_tool=DEFAULT_SOFTWARE_TOOL,
            task_name=DEFAULT_TASK_NAME,
            task_body=DEFAULT_TASK_DESCRIPTION,
        )
    task_name = _extract_task_section(markdown, "Task Name") or task_id or DEFAULT_TASK_NAME
    task_body = _build_task_description_block(markdown) or DEFAULT_TASK_DESCRIPTION
    return _render_instructions(
        software_tool=_infer_software_tool(task_id, task_name),
        task_name=task_name,
        task_body=task_body,
    )


def task_metadata_for_id(task_id: str | None) -> dict[str, str]:
    markdown = _load_task_markdown(task_id)
    if not markdown:
        return {
            "task_name": DEFAULT_TASK_NAME,
            "task_description": DEFAULT_TASK_DESCRIPTION,
        }
    return {
        "task_name": _extract_task_section(markdown, "Task Name") or task_id or DEFAULT_TASK_NAME,
        "task_description": _build_task_description_block(markdown) or DEFAULT_TASK_DESCRIPTION,
    }


def runtime_config_payload() -> dict:
    payload = asdict(runtime_config)
    payload.update(task_metadata_for_id(runtime_config.task_id))
    return payload


def apply_task_id(task_id: str | None) -> None:
    runtime_config.task_id = task_id
    runtime_config.instructions = build_instructions_for_task_id(task_id)


COACHING_METHODS_DEFINITION_AND_EXAMPLES = """\
#### Direct Instruction
The coach tells the learner the concrete next action to take, often in a procedural or step-by-step way.
- Example: "Click Sketch, then select this edge."

#### Plan
The coach states a subgoal or a sub-task without specifying the exact action.
- Example: "Then we will start making the magic potion bottle."

#### Confirmation
The coach gives positive affirmation on the learner's action, result, or understanding.
- Example: "Yeah, you can just select this."

#### Diagnosis
The coach provides neutral to negative feedback on the learner's current task progress or identifies a problem.
- Example: "Can you make it a little bit more squarish?"

#### Explanation
The coach explains how or why something works, why something happened, or why a certain operation is appropriate.
- Example: "Extrude works here because it turns a 2D profile into a 3D solid."

#### Tip
The coach gives practical advice, a best practice, an alternative approach, or advanced knowledge that is broadly useful beyond the immediate step. 
- Example: "It's often easier to model one side first and then mirror it."

#### Clarification
The coach asks for missing information about the learner's goal, intended result, or source of confusion before giving further guidance. The main purpose is to understand the learner's situation.
- Example: "Are you on the resizing part?"

#### Reflection
The coach asks the learner to evaluate their result, mistake, or choice. The main purpose is to make the learner interpret what happened and why.
- Example: "If you drag on this, do you see any issues?"

#### Articulation
The coach asks the learner to verbalize their plan, reasoning, or predicted next step before acting. The main purpose is to externalize learner thinking.
- Example: "What do you think the next step should be?"

#### Exploration
The coach intentionally leaves room for learner choice, experimentation, practice themselves, or trying alternatives.
- Example: "Try a couple of options and see which one you like better."
"""


VLM_SYSTEM_INSTRUCTION_TEMPLATE = """\
You are an expert coach guiding a learner in a real-time software coaching session using {{SOFTWARE_TOOL}}.
Your goal is not only to complete the current tutorial, but to help the learner develop skill so that they can solve similar problems on their own in the future.

You are going to speak as the coach to continue the conversation.
You can choose the appropriate coaching method(s) defined in the following section to use in your response.
You are looking at the learner's recent screen and you have your recent conversation with the learner.
The cursor is where the learner is currently interacting with the software.
If there are any visual pink color screen annotations, those are your previous annotations to the learner's screen. You can use them to guide your response.
Your response should sound like something a human coach would actually say aloud in real-time during the session.

## Generation Rules
- Give one concise sentence as your response.
- Treat the chat history as a live conversation (Learner = user, Coach = assistant).
- Avoid overly verbose and overly formal language.
- Avoid any special characters or markdown formatting like lists, bold, italic, etc.
- Avoid transcript artifacts such as ellipses, dashes.
- Do not mention being an AI, model, assistant, or system.

## Output Rules
- Return ONLY the coach utterance as plain text.
- No JSON, no speaker prefix, no markdown.

{{TASK_DESCRIPTION}}"""


DEFAULT_SOFTWARE_TOOL = "Figma"
DEFAULT_TASK_NAME = "Create a magic potion illustration"
DEFAULT_TASK_DESCRIPTION = """\
Task Description:
Build a magic potion bottle by combining basic geometric shapes, manipulating vector paths, and organizing layers into a functional frame.

Learning Objectives:
- Construct complex objects.
- Execute precise object alignment using "Align Horizontal Centers" and smart guides.
- Manipulate geometry via Vector Edit Mode to transform basic circles into fluid shapes.
- Manage project hierarchy by grouping layers and renaming them for clarity.
- Utilize layer stacking order to simulate depth.

Intermediate Steps:
- Create the bottle.
- Fill the bottle with magic potion.
- Top it off with a cork.
- Add final details.

Final Outcome:
Match the provided reference image of the magic potion illustration.
"""


def build_default_instructions() -> str:
    return build_instructions_for_task_id(_DEFAULT_TASK_ID)


_active_call_id: str | None = None


def _llm_logging_enabled() -> bool:
    return os.getenv("LLM_LOG_ENABLED", "true").lower() not in ("0", "false", "no")


def _llm_log_include_images() -> bool:
    return os.getenv("LLM_LOG_INCLUDE_IMAGES", "").lower() in ("1", "true", "yes")


def _llm_export_dir() -> Path:
    raw = os.getenv("LLM_EXPORT_DIR", "llm_exports")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _summarize_data_url(url: str) -> str:
    if url.startswith("data:") and ";base64," in url:
        mime = url[5 : url.index(";base64,")]
        b64_len = len(url.split(";base64,", 1)[1])
        return f"<data-url mime={mime} base64_chars={b64_len}>"
    if len(url) > 160:
        return f"{url[:80]}…({len(url)} chars)…{url[-40:]}"
    return url


def _sanitize_message_content(content: object) -> object:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    include_images = _llm_log_include_images()
    sanitized: list[object] = []
    for part in content:
        if not isinstance(part, dict):
            sanitized.append(part)
            continue
        if part.get("type") == "image_url" and not include_images:
            url = (part.get("image_url") or {}).get("url") or ""
            sanitized.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _summarize_data_url(str(url))},
                }
            )
        else:
            sanitized.append(part)
    return sanitized


def _sanitize_messages_for_log(messages: list[dict]) -> list[dict]:
    return [
        {
            "role": msg.get("role"),
            "content": _sanitize_message_content(msg.get("content")),
        }
        for msg in messages
    ]


def _save_llm_exchange(
    *,
    call_id: str | None,
    model: str,
    messages: list[dict],
    response_text: str,
    trigger: str | None,
) -> Path | None:
    if not _llm_logging_enabled():
        return None
    safe_call_id = _normalize_call_id(call_id) if call_id else "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    inference_id = uuid4().hex[:12]
    sanitized_messages = _sanitize_messages_for_log(messages)

    document = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "trigger": trigger,
        "message_count": len(sanitized_messages),
        "messages": sanitized_messages,
        "response_text": response_text,
    }

    payload = json.dumps(document, indent=2, ensure_ascii=False)
    print(f"\n--- LLM chat completion ({safe_call_id}) ---\n{payload}\n", flush=True)
    logger.info(
        "LLM chat completion call_id=%s model=%s messages=%s response_len=%s",
        safe_call_id,
        model,
        len(sanitized_messages),
        len(response_text),
    )

    call_dir = _llm_export_dir() / safe_call_id
    call_dir.mkdir(parents=True, exist_ok=True)
    turn_path = call_dir / f"{stamp}_{inference_id}.json"
    turn_path.write_text(payload, encoding="utf-8")

    session_path = call_dir / "session.json"
    if session_path.is_file():
        session_doc = json.loads(session_path.read_text(encoding="utf-8"))
    else:
        session_doc = {
            "call_id": safe_call_id,
            "created_at": document["saved_at"],
            "turns": [],
        }
    session_doc["updated_at"] = document["saved_at"]
    session_doc["turns"].append(document)
    session_path.write_text(
        json.dumps(session_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved LLM exchange call_id=%s turn=%s session=%s", safe_call_id, turn_path, session_path)
    return turn_path


_GREET_TRIGGER_PREFIX = "Greet the user, confirm you can observe their screen share"
_PROACTIVE_TRIGGER_SNIPPET = "proactively provide one short coaching suggestion"


def _is_automated_coach_trigger(trigger: str | None) -> bool:
    """True for initial greet and periodic proactive nudges (no learner utterance to record)."""
    if not trigger:
        return True
    t = trigger.strip()
    if t.startswith(_GREET_TRIGGER_PREFIX):
        return True
    if _PROACTIVE_TRIGGER_SNIPPET in t:
        return True
    return False


def _parse_iso_utc(value: str) -> datetime:
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    return datetime.fromisoformat(v).astimezone(timezone.utc)


def _format_transcript_offset(offset: timedelta) -> str:
    """Wall-clock style offset HH:MM:SS.mmm from session start."""
    ms = max(0, int(offset.total_seconds() * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, frac_ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{frac_ms:03d}"


def _session_chat_export_dir() -> Path:
    raw = os.getenv("SESSION_CHAT_EXPORT_DIR", "session_chat")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _export_session_chat_transcript(call_id: str) -> Path | None:
    """Write a compact coach/learner transcript when the agent session ends (from llm_exports session.json)."""
    safe = _normalize_call_id(call_id)
    session_path = _llm_export_dir() / safe / "session.json"
    if not session_path.is_file():
        logger.info("Session chat export skipped (no session.json) call_id=%s", safe)
        return None
    try:
        session_doc = json.loads(session_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Session chat export failed reading session.json call_id=%s err=%s", safe, exc)
        return None

    turns = session_doc.get("turns") or []
    if not turns:
        return None

    created_raw = session_doc.get("created_at") or turns[0].get("saved_at")
    try:
        t0 = _parse_iso_utc(str(created_raw))
    except (ValueError, TypeError):
        logger.warning("Session chat export skipped (bad created_at) call_id=%s", safe)
        return None

    rows_work: list[dict] = []
    for turn in turns:
        saved_raw = turn.get("saved_at")
        if not saved_raw:
            continue
        try:
            t_coach = _parse_iso_utc(str(saved_raw))
        except (ValueError, TypeError):
            continue

        coach_off = t_coach - t0
        coach_ms = int(coach_off.total_seconds() * 1000)
        resp = (turn.get("response_text") or "").strip()
        if not resp:
            continue

        trigger = turn.get("trigger")
        if not _is_automated_coach_trigger(trigger):
            learner_text = (trigger or "").strip()
            if learner_text:
                learner_ms = max(0, coach_ms - 1)
                learner_off = timedelta(milliseconds=learner_ms)
                rows_work.append(
                    {
                        "start": _format_transcript_offset(learner_off),
                        "speaker": "Learner",
                        "text": learner_text,
                        "_t": learner_ms,
                        "_tie": 0,
                    }
                )

        rows_work.append(
            {
                "start": _format_transcript_offset(coach_off),
                "speaker": "Coach",
                "text": resp,
                "_t": coach_ms,
                "_tie": 1,
            }
        )

    if not rows_work:
        return None

    rows_work.sort(key=lambda r: (r["_t"], r["_tie"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = _session_chat_export_dir() / f"{safe}_{stamp}.json"
    transcript = [
        {
            "turn_id": i,
            "start": r["start"],
            "speaker": r["speaker"],
            "text": r["text"],
        }
        for i, r in enumerate(rows_work, start=1)
    ]
    document = {
        "call_id": safe,
        "session_created_at": created_raw,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "transcript": transcript,
    }
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Exported session chat transcript call_id=%s path=%s turns=%s", safe, out_path, len(transcript))
    return out_path


class ChatCompletionsLoggingMixin:
    """Logs each chat-completions request/response to stdout and llm_exports/*.json."""

    _pending_llm_messages: list[dict] | None = None
    _pending_llm_trigger: str | None = None

    async def _build_model_request(self) -> list[dict]:
        messages = await super()._build_model_request()  # type: ignore[misc]
        self._pending_llm_messages = messages
        return messages

    async def simple_response(self, text: str, participant=None):
        self._pending_llm_trigger = (text or "")[:500] or None
        result = await super().simple_response(text, participant)  # type: ignore[misc]
        messages = self._pending_llm_messages or []
        self._pending_llm_messages = None
        response_text = getattr(result, "text", "") or ""
        trigger = self._pending_llm_trigger
        _save_llm_exchange(
            call_id=_active_call_id,
            model=getattr(self, "model", ""),
            messages=messages,
            response_text=response_text,
            trigger=trigger,
        )
        self._pending_llm_trigger = None
        # Append to the live session-chat store so the frontend can poll and display in real-time.
        if _active_call_id:
            safe = _normalize_call_id(_active_call_id)
            now_iso = datetime.now(timezone.utc).isoformat()
            if safe not in _session_chat_store:
                _session_chat_store[safe] = []
            if not _is_automated_coach_trigger(trigger):
                learner_text = (trigger or "").strip()
                if learner_text:
                    _session_chat_store[safe].append(
                        {"speaker": "Learner", "text": learner_text, "timestamp": now_iso}
                    )
            if response_text.strip():
                _session_chat_store[safe].append(
                    {"speaker": "Coach", "text": response_text.strip(), "timestamp": now_iso}
                )
        return result


class RecentHistoryMixin:
    """Limit prior chat messages in each VLM request to the frame-buffer time window."""

    def __init__(self, *args, frame_buffer_seconds: int = 10, **kwargs):
        super().__init__(*args, frame_buffer_seconds=frame_buffer_seconds, **kwargs)
        self._chat_history_seconds = frame_buffer_seconds

    async def _build_model_request(self) -> list[dict]:
        conv = self._conversation
        if conv is None:
            return await super()._build_model_request()  # type: ignore[misc]

        cutoff = datetime.now() - timedelta(seconds=self._chat_history_seconds)
        all_messages = conv.messages
        recent = [
            m
            for m in all_messages
            if m.timestamp is None or m.timestamp >= cutoff
        ]
        if len(recent) == len(all_messages):
            return await super()._build_model_request()  # type: ignore[misc]

        conv.messages = recent
        try:
            return await super()._build_model_request()  # type: ignore[misc]
        finally:
            conv.messages = all_messages


class ReferenceAwareChatCompletionsVLM(RecentHistoryMixin, ChatCompletionsVLM):
    """Prepends the task reference still (final.png) for chat-completions VLM; not used on Gemini Live."""

    def __init__(self, *args, reference_image_data_url: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._reference_image_data_url = reference_image_data_url

    async def _build_model_request(self) -> list[dict]:
        messages: list[dict] = await super()._build_model_request()
        if not self._reference_image_data_url:
            return messages
        insert_at = 1 if messages and messages[0].get("role") == "system" else 0
        ref_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Reference outcome image: this is the target end state for the current task. "
                        "The other images in this request are live screen frames from the session."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": self._reference_image_data_url},
                },
            ],
        }
        messages.insert(insert_at, ref_message)
        return messages


class LoggingChatCompletionsVLM(
    ChatCompletionsLoggingMixin, RecentHistoryMixin, ChatCompletionsVLM
):
    pass


class LoggingReferenceAwareChatCompletionsVLM(
    ChatCompletionsLoggingMixin, ReferenceAwareChatCompletionsVLM
):
    pass


@dataclass
class AgentRuntimeConfig:
    # Default: OpenRouter chat-completions VLM (not Gemini Live / native realtime).
    provider: ModelProvider = "openrouter"
    model: str = "google/gemini-3.1-pro-preview"
    fps: int = 1
    # OpenRouter VLM: seconds of frames buffered for chat-completions (see visionagents.ai video agents).
    frame_buffer_seconds: int = 10
    proactive_enabled: bool = True
    proactive_fire_rate_seconds: int = 30
    instructions: str = field(default_factory=build_default_instructions)
    # Matches web/src/tasks/<task_id>/; used to attach final.png (etc.) to chat-completions VLM only.
    task_id: str | None = _DEFAULT_TASK_ID


runtime_config = AgentRuntimeConfig()
apply_task_id(_DEFAULT_TASK_ID)
runtime_config_lock = asyncio.Lock()

# In-memory cache of the last exported session-chat transcript per call_id.
# Populated by _export_session_chat_transcript so the frontend can fetch it.
_session_chat_store: dict[str, list[dict]] = {}


def _build_llm_and_voice_from_runtime():
    """Returns (llm, stt, tts). STT/TTS are only set for the OpenRouter VLM pipeline."""
    if runtime_config.provider == "gemini":
        return gemini.Realtime(model=runtime_config.model, fps=runtime_config.fps), None, None
    if runtime_config.provider == "openai":
        try:
            from vision_agents.plugins import openai as openai_plugin
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai provider is not installed. "
                "Install extras first, for example: uv add 'vision-agents[openai]'"
            ) from exc
        return openai_plugin.Realtime(model=runtime_config.model, fps=runtime_config.fps), None, None
    if runtime_config.provider == "qwen":
        try:
            from vision_agents.plugins import qwen
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "qwen provider is not installed. "
                "Install extras first, for example: uv add 'vision-agents[qwen]'"
            ) from exc
        return qwen.Realtime(model=runtime_config.model, fps=runtime_config.fps), None, None
    if runtime_config.provider == "openrouter":
        # OpenRouter uses the chat-completions VLM path (frame buffer + STT/TTS), not native realtime.
        from vision_agents.plugins import deepgram, elevenlabs

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY must be set for OpenRouter VLM. "
                "See https://visionagents.ai/integrations/llm/openrouter"
            )
        common_kwargs = dict(
            model=runtime_config.model,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            fps=runtime_config.fps,
            frame_buffer_seconds=runtime_config.frame_buffer_seconds,
        )
        ref_url = _task_reference_data_url(runtime_config.task_id)
        if ref_url:
            llm = LoggingReferenceAwareChatCompletionsVLM(
                reference_image_data_url=ref_url, **common_kwargs
            )
        else:
            llm = LoggingChatCompletionsVLM(**common_kwargs)
        return llm, deepgram.STT(), elevenlabs.TTS()
    raise RuntimeError(f"Unsupported provider: {runtime_config.provider}")


async def create_agent(**kwargs) -> Agent:
    logger.info(
        "Creating agent with provider=%s model=%s fps=%s frame_buffer_seconds=%s "
        "proactive_enabled=%s proactive_fire_rate_seconds=%s",
        runtime_config.provider,
        runtime_config.model,
        runtime_config.fps,
        runtime_config.frame_buffer_seconds,
        runtime_config.proactive_enabled,
        runtime_config.proactive_fire_rate_seconds,
    )
    llm, stt, tts = _build_llm_and_voice_from_runtime()
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Digital Coach", id="agent"),
        instructions=runtime_config.instructions,
        llm=llm,
        stt=stt,
        tts=tts,
    )


async def _with_stream_retries(operation, *, label: str, attempts: int = 3):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except _STREAM_RETRY_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < attempts:
                delay = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "%s failed (attempt %s/%s): %s; retrying in %ss",
                    label,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("%s failed after %s attempts: %s", label, attempts, exc)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed with no exception")


async def _ensure_stream_agent_user() -> None:
    """Upsert the coach agent user so join_call authenticate() is fast and reliable."""
    from getstream import AsyncStream

    client = AsyncStream(
        timeout=_STREAM_TIMEOUT,
        user_agent="digital-coach-api",
    )
    try:
        await _with_stream_retries(
            lambda: client.create_user(name="Digital Coach", id="agent"),
            label="Stream create_user(agent)",
        )
        logger.info("Stream agent user ready (timeout=%ss)", _STREAM_TIMEOUT)
    finally:
        await client.aclose()


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    global _active_call_id
    logger.info("Agent joining call type=%s id=%s", call_type, call_id)
    try:
        call = await _with_stream_retries(
            lambda: agent.create_call(call_type, call_id),
            label=f"agent.create_call({call_type},{call_id})",
        )
    except Exception:
        logger.exception("Agent failed to create/join call type=%s id=%s", call_type, call_id)
        raise
    _active_call_id = call_id
    try:
        async with agent.join(call):
            response = await agent.simple_response(
                "Greet the user, confirm you can observe their screen share, and start coaching."
            )
            logger.info("Initial agent response sent for call id=%s response=%s", call_id, response)
            try:
                # Keep the session alive until the launcher requests close/cancellation.
                loop = asyncio.get_running_loop()
                next_proactive_due_at = loop.time() + runtime_config.proactive_fire_rate_seconds
                while True:
                    await asyncio.sleep(1)
                    if not runtime_config.proactive_enabled:
                        next_proactive_due_at = loop.time() + runtime_config.proactive_fire_rate_seconds
                        continue
                    now = loop.time()
                    if now < next_proactive_due_at:
                        continue
                    try:
                        proactive_response = await asyncio.wait_for(
                            agent.simple_response(
                                "Proactively generate the best possible next coaching utterance based on the latest screen context and transcript. "                            ),
                            timeout=5,
                        )
                        logger.info(
                            "Proactive coaching response sent for call id=%s response=%s",
                            call_id,
                            proactive_response,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Proactive coaching timed out for call id=%s (continuing loop)",
                            call_id,
                        )
                    except Exception:
                        logger.exception(
                            "Proactive coaching failed for call id=%s (continuing loop)",
                            call_id,
                        )
                    finally:
                        next_proactive_due_at = loop.time() + runtime_config.proactive_fire_rate_seconds
            except asyncio.CancelledError:
                logger.info("Agent session cancelled for call id=%s", call_id)
                raise
    finally:
        try:
            _export_session_chat_transcript(call_id)
        except Exception:
            logger.exception("Session chat export failed for call id=%s", call_id)
        _active_call_id = None


launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
stream_client = getstream.Client(timeout=_STREAM_TIMEOUT)


class StartSessionRequest(BaseModel):
    call_id: str | None = None
    call_type: str = Field(default="default")
    task_id: str | None = None
    user_id: str | None = None
    user_name: str = Field(default="Learner")


class AgentConfigRequest(BaseModel):
    provider: ModelProvider
    model: str
    fps: int = Field(default=1, ge=1, le=15)
    frame_buffer_seconds: int = Field(default=10, ge=1, le=120)
    proactive_enabled: bool = True
    proactive_fire_rate_seconds: int = Field(default=30, ge=5, le=300)
    instructions: str
    task_id: str | None = None


class TranscriptEntryIn(BaseModel):
    id: str
    kind: str = "event"
    text: str = ""
    speaker: str | None = None
    role: str | None = None
    # Unix timestamp in milliseconds (int) preferred; string labels accepted for
    # backward-compat but stored as-is.
    timestamp: int | str | None = None


class SaveChatRequest(BaseModel):
    session_id: str | None = None
    task_id: str | None = None
    user: dict | None = None
    # Participant ID supplied from the login page.
    user_id: str | None = None
    # Unix timestamp (ms) when the frontend triggered the save.
    saved_at: int | None = None
    entries: list[TranscriptEntryIn]


def _normalize_call_id(raw: str | None) -> str:
    if raw:
        cleaned = "".join(ch for ch in raw.lower() if ch.isalnum() or ch in "_-")
        if cleaned:
            return cleaned
    return f"screen-{uuid4().hex[:10]}"


def _chat_export_dir() -> Path:
    raw = os.getenv("CHAT_EXPORT_DIR", "chat_exports")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _turn_label_for_entry(entry: dict) -> str:
    kind = entry.get("kind") or "event"
    if kind != "message":
        return "system"
    role = (entry.get("role") or "").strip().lower()
    if role == "coach":
        return "agent"
    if role == "learner":
        return "human"
    return "other"


@asynccontextmanager
async def lifespan(_: FastAPI):
    await launcher.start()
    try:
        await _ensure_stream_agent_user()
    except Exception:
        logger.exception(
            "Failed to pre-register Stream agent user at startup; call joins will retry"
        )
    try:
        yield
    finally:
        await launcher.stop()


app = FastAPI(title="Digital Coach API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/agent/config")
async def get_agent_config() -> dict:
    return runtime_config_payload()


@app.post("/api/agent/config")
async def set_agent_config(payload: AgentConfigRequest) -> dict:
    if payload.provider == "openrouter" and payload.model.strip() == "":
        raise HTTPException(status_code=400, detail="OpenRouter requires a non-empty model id.")
    async with runtime_config_lock:
        runtime_config.provider = payload.provider
        runtime_config.model = payload.model
        runtime_config.fps = payload.fps
        runtime_config.frame_buffer_seconds = payload.frame_buffer_seconds
        runtime_config.proactive_enabled = payload.proactive_enabled
        runtime_config.proactive_fire_rate_seconds = payload.proactive_fire_rate_seconds
        if payload.task_id:
            apply_task_id(payload.task_id)
        else:
            runtime_config.task_id = None
            runtime_config.instructions = payload.instructions
    logger.info(
        "Updated agent config provider=%s model=%s fps=%s frame_buffer_seconds=%s "
        "proactive_enabled=%s proactive_fire_rate_seconds=%s task_id=%s",
        runtime_config.provider,
        runtime_config.model,
        runtime_config.fps,
        runtime_config.frame_buffer_seconds,
        runtime_config.proactive_enabled,
        runtime_config.proactive_fire_rate_seconds,
        runtime_config.task_id,
    )
    return {"updated": True, "config": runtime_config_payload()}


@app.post("/api/session/start")
async def start_session(payload: StartSessionRequest) -> dict:
    call_id = _normalize_call_id(payload.call_id)
    requested_call_type = payload.call_type or "default"
    allowed_call_types = {"default", "audio_room", "development", "livestream"}
    if requested_call_type in allowed_call_types:
        call_type = requested_call_type
    else:
        logger.warning(
            "Unsupported call_type=%s requested (task_id=%s). Falling back to default.",
            requested_call_type,
            payload.task_id,
        )
        call_type = "default"
    user_id = payload.user_id or f"user-{uuid4().hex[:10]}"

    try:
        async with runtime_config_lock:
            if payload.task_id:
                apply_task_id(payload.task_id)
        logger.info(
            "Starting session for user=%s call_type=%s call_id=%s",
            payload.user_name,
            call_type,
            call_id,
        )
        stream_client.create_user(name=payload.user_name, id=user_id)
        token = stream_client.create_call_token(
            user_id=user_id,
            call_cids=[f"{call_type}:{call_id}"],
            expiration=3600,
        )
        session = await launcher.start_session(call_id=call_id, call_type=call_type)
        logger.info("Session started id=%s call_type=%s call_id=%s", session.id, call_type, call_id)
    except Exception as exc:
        logger.exception("Failed to start session for call id=%s", call_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "apiKey": stream_client.api_key,
        "token": token,
        "callId": call_id,
        "callType": call_type,
        "sessionId": session.id,
        "user": {"id": user_id, "name": payload.user_name},
        "agentConfig": runtime_config_payload(),
    }


@app.post("/api/session/{call_id}/chat")
async def save_chat_transcript(call_id: str, payload: SaveChatRequest) -> dict:
    """Persist session transcript (coach / learner turns and events) as JSON."""
    safe_call_id = _normalize_call_id(call_id)
    saved_at_ms: int = payload.saved_at or int(datetime.now(timezone.utc).timestamp() * 1000)
    out_dir = _chat_export_dir()

    # File name: {userId}-{unixTimestampMs}.json
    safe_user_id = re.sub(r"[^\w\-]", "_", payload.user_id or "session")
    filename = f"{safe_user_id}-{saved_at_ms}.json"
    out_path = out_dir / filename

    rows: list[dict] = []
    for item in payload.entries:
        row = item.model_dump()
        row["turn"] = _turn_label_for_entry(row)
        rows.append(row)

    document = {
        "saved_at": saved_at_ms,
        "call_id": safe_call_id,
        "session_id": payload.session_id,
        "task_id": payload.task_id,
        "user_id": payload.user_id,
        "user": payload.user,
        "entry_count": len(rows),
        "entries": rows,
    }
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Saved chat transcript call_id=%s path=%s entries=%s",
        safe_call_id,
        out_path,
        len(rows),
    )
    return {"saved": True, "path": str(out_path), "filename": filename, "entry_count": len(rows)}


@app.get("/api/session/{call_id}/session-chat")
async def get_session_chat(call_id: str) -> dict:
    """Return the exported session-chat transcript for a completed call (populated on agent session end)."""
    safe = _normalize_call_id(call_id)
    transcript = _session_chat_store.get(safe)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Session chat not available yet")
    return {"call_id": safe, "transcript": transcript}


@app.post("/api/session/{call_id}/{session_id}/close")
async def close_session(call_id: str, session_id: str) -> dict:
    logger.info("Closing session id=%s call_id=%s", session_id, call_id)
    try:
        await launcher.request_close_session(call_id=call_id, session_id=session_id)
        return {"closed": True}
    except Exception as exc:
        # During rapid teardown, SFU participant state can disappear before
        # closure RPC completes. Treat it as best-effort close.
        logger.warning(
            "Session close best-effort fallback for id=%s call_id=%s error=%s",
            session_id,
            call_id,
            exc,
        )
        return {"closed": False, "bestEffort": True, "detail": str(exc)}


if __name__ == "__main__":
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8000")))
    uvicorn.run("backend_api:app", host=host, port=port, reload=False)
