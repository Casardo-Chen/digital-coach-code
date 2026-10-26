<p align="center">
  <img src="docs/assets/banner.svg" alt="DigitalCoach — Communication and Grounding Gaps in Human and Agentic Computer Use Coaching" width="100%">
</p>
<p align="center">
<b>
  <a href="https://arxiv.org/abs/2606.31980">DigitalCoach: Communication and Grounding Gaps in Human and Agentic Computer Use Coaching</a>
  EMNLP 2026 <span style="color: red; font-style: italic;">(Oral)</span>
</b>
</p>
<p align="center">
  <a href="https://meng-chen.com">Meng Chen</a><sup>1</sup>,
  <a href="https://anya-ji.github.io">Anya Ji</a><sup>1</sup>,
  <a href="https://patrickthwu.com">Tsung-Han Wu</a><sup>1</sup>,
  <a href="https://tmaringgele.dev">Tobias Maringgele</a><sup>2</sup>,
  <a href="https://dchan.cc">David M. Chan</a><sup>1</sup>,
  <a href="https://www.alanesuhr.com">Alane Suhr</a><sup>1†</sup>,
  <a href="https://amypavel.com">Amy Pavel</a><sup>1†</sup>
</p>
<p align="center">
  <sub><sup>1</sup>UC Berkeley &nbsp; <sup>2</sup>TU Munich &nbsp; <sup>†</sup>Equal supervision</sub><br>
</p>
<p align="center">Evaluating VLMs on real-time, long-horizon computer use task teaching.</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.31980"><img src="https://img.shields.io/badge/arXiv-2606.31980-b31b1b" alt="arXiv"></a>
  <a href="https://arxiv.org/abs/2606.31980"><img src="https://img.shields.io/badge/EMNLP-2026%20Oral-111111" alt="EMNLP 2026"></a>
  <a href="https://huggingface.co/datasets/berkeley-hci/digital-coach"><img src="https://img.shields.io/badge/data-Hugging%20Face-ff9d00" alt="Hugging Face dataset"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

<p align="center">
  <a href="#what-it-is">What it is</a> &middot;
  <a href="#data">Data</a> &middot;
  <a href="#eval">Eval</a> &middot;
  <a href="#annotation">Annotation</a> &middot;
  <a href="#prototype">Prototype</a>
</p>

---

```bash
pip install -e .
```

The two core operations — **load** a session, **score** a coach utterance:

```bash
# Load human–human and human–AI turns in one schema
python -c "from digital_coach.dataset import load_sessions; print(load_sessions())"

# Preview an input ablation (text only, no screen video required)
python eval/eval_session.py -s hh_0101_example --vlm-input-modality text-only --dry-run
```

> **Same schema for both sides.** A session is `meta.json` + `turns.json`. `source` is `human-human` or `human-ai`. Coach turns carry `dialogue_acts` and `coaching_methods`.

Full recordings stay out of git. Set `DIGITAL_COACH_DATASET=berkeley-hci/digital-coach` to load the released corpus from Hugging Face instead of `data/sessions`.

## What it is

This repo is the code behind that comparison:

- dialogue act and coaching method annotation
- CLAIR scoring of generated coach utterances, with input ablations
- a real-time web prototype for VLM coach that calls models through OpenRouter

## Data

Human–human folders (`DigitalCoach/<task>/<session>/`) and human–AI exports (`transcript` JSON) both become:

```text
data/sessions/<session_id>/
  meta.json
  turns.json
```

`meta.json` records `session_id`, `source`, `task`, `software`, and `participants` (`coach_id`, `learner_id`, `coach_type` of `human` or `model`). A model coach also stores `model.provider` (`openrouter`) and `model.model`.

Each turn is `{turn_id, start, end, speaker, text, dialogue_acts, coaching_methods}`. `end` may be null for a live session. `speaker` is `Coach` or `Learner`.

Two short examples are committed: `hh_0101_example` and `hai_0101_example`.

```python
from digital_coach.dataset import load_sessions

local = load_sessions()  # data/sessions, one row per turn
hub = load_sessions("berkeley-hci/digital-coach")
```

## Eval

CLAIR (Chan et al., EMNLP 2023) asks an LLM judge (gpt-5.1) to evaluate the similarity between two utterances given the same multimodal input. Scores are 0–100. The judge goes through OpenRouter (`OPENROUTER_API_KEY`).

Input ablations change what the generator sees before that score:

| Flag | Values | What changes |
| --- | --- | --- |
| `--vlm-input-modality` | `text-only`, `text-image`, `image-only`, `text-video` | Transcript, frames, or video |
| `--vlm-context-seconds` | `1`, `10`, `30`, `60` | How much of the session precedes the coach turn |
| `--vlm-prompt-style` | `default`, `teacher`, `oracle` | System prompt. `oracle` reveals the human coaching method |

```bash
cp .env.example .env   # set OPENROUTER_API_KEY

python eval/eval_session.py \
  -s hh_0101_example \
  --vlm-input-modality text-image \
  --vlm-context-seconds 30 \
  --vlm-prompt-style oracle
```

Image and video conditions need `screen.mp4` in the session folder (or `media/`). Text-only does not. Generations land in `results/eval/<session_id>/`.

## Annotation

Coach and learner turns are labeled with dialogue acts. Coach turns are also labeled with coaching methods.

**Dialogue acts.** Info Request, Answer, Action Directive, Inform, Opinion, Backchannel, Not Enough Info.

**Coaching methods.** Direct Instruction, Plan, Confirmation, Diagnosis, Explanation, Tip, Clarification, Reflection, Articulation, Exploration, Non-Instructive.

Definitions and examples live in [`annotation/dialogue_acts.py`](annotation/dialogue_acts.py) and [`annotation/coaching_methods.py`](annotation/coaching_methods.py).

## Prototype

`prototype/` is the live human–AI coach: a React learner app and a FastAPI agent. Media goes over GetStream. The default model path is OpenRouter, with Deepgram for speech recognition and ElevenLabs for speech synthesis.

```bash
pip install -e ".[prototype]"
cd prototype/web && npm install && npm run dev
# in another shell, from the repo root:
python prototype/backend_api.py
```

Task prompts are the shared `tasks/` tree. When a call ends, the transcript is written in the same `meta.json` / `turns.json` shape as the recorded sessions, under `results/sessions/`.

Also set `STREAM_API_KEY`, `STREAM_API_SECRET`, `DEEPGRAM_API_KEY`, and `ELEVENLABS_API_KEY`. See `.env.example`.

## Citation

```bibtex
@inproceedings{chen2026digitalcoach,
  title={DigitalCoach: Communication and Grounding Gaps in Human and Agentic Computer Use Coaching},
  author={Chen, Meng and Ji, Anya and Wu, Tsung-Han and Maringgele, Tobias and Chan, David M. and Suhr, Alane and Pavel, Amy},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```

## License

MIT
