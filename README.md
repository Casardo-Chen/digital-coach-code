# Digital Coach

Research code for human and AI coaching: a shared session format, CLAIR scoring with input ablations, dialogue-act and coaching-method annotation, and the realtime coach prototype.

LLM calls use OpenRouter (`OPENROUTER_API_KEY`). Copy `.env.example` to `.env` and fill in the key. Do not commit `.env`.

## Layout

- `data/sessions/` — one folder per session (`meta.json` + `turns.json`). Human–human and human–AI sessions use the same turn fields. Two short examples are committed; full recordings stay out of git.
- `results/` — generated eval JSON, figures, and live session exports. Ignored by git except this folder’s README.
- `src/digital_coach/` — session schema, paths, OpenRouter client, and a Hugging Face dataset loader (`load_sessions`).
- `eval/` — coaching-utterance generation with input ablations, then CLAIR.
- `annotation/` — dialogue acts and coaching methods.
- `tasks/` — task writeups and reference images.
- `prototype/` — realtime human–AI coach (FastAPI + React). It reads `tasks/` and keeps OpenRouter as the default provider.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[eval,plots]"
```

The realtime app also needs `pip install -e ".[prototype]"` plus `npm install` in `prototype/web`.

## Data

```python
from digital_coach.dataset import load_sessions

ds = load_sessions()  # local data/sessions
```

Set `DIGITAL_COACH_DATASET` to a Hugging Face dataset id to load a published corpus instead. This repo does not upload data.

Session fields: `session_id`, `source` (`human-human` or `human-ai`), `task`, `software`, `participants.coach_id`, `participants.learner_id`, `participants.coach_type` (`human` or `model`). Each turn has `turn_id`, `start`, `end` (optional), `speaker`, `text`, `dialogue_acts`, and `coaching_methods`.
