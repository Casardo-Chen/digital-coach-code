import json
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from google import genai
from google.genai import types as genai_types

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import box_root

BOX_DIGITAL_COACH = box_root.BOX_DIGITAL_COACH
PROCESSED_DATA_FOLDER = BOX_DIGITAL_COACH / "processed"

DIALOGUE_ACTS_PROMPT = """\
### Info Request
The utterance asks the addressee to provide information, confirm a state, explain something. It may be a yes/no question, wh-question, or clarification question. If the utterance asks for information rather than prompting an action, it is Info Request.

- Examples:
  - "What tool should I use for rounding this corner?"
  - "Are you trying to just go back to the normal?"
  - "What do you want to do with them?"
  - "You see that menu, right?"

- Non-Examples:
  - "So, since the thickness is 0.5, and the height is 2.5, I would say 2 for the length?"
    (Action Directive: question form, but functionally prompts an action.)
  - "Open the mixer."
    (Action Directive: directly tells the addressee to do something.)
  - "The mixer is already open."
    (Inform: provides information.)

### Answer
The utterance answers a previous Info Request by supplying the requested information. An answer can be short or long, but it must provide content that addresses the prior request.

- Examples:
  - "Yes." (If the previous utterance is a Info Request.)
  - "Because the polygon is not closed."
  - "No, I don't see that menu."

- Non-Examples:
  - "Okay."
    (Backchannel: signals receipt rather than answering contentfully.)
  - "I think the polygon is not closed."
    (Opinion: if said independently rather than as a response.)
  - "Try increasing the gain."
    (Action Directive: proposes an action.)

### Action Directive
The utterance guides the addressee toward an action. This includes direct commands, polite action prompts, option offers.
It can be phrased imperatively, politely (Let's...), softly (Maybe...), collaboratively (We could...), or as a question (Can you...?), as long as the function is to get the learner to consider or perform that action.

- Examples:
  - "You can just exit."
  - "Click on the top menu."
  - "Can you open the piano roll?"
  - "Maybe try using a softer synth here."
  - "We could group these elements into a card."
  - "We probably want to make this kind of brownish."
  - "What we want to do is start adding effects to it"
  
- Non-Examples:
  - "Can you see the mixer?"
    (Info Request: asks for information rather than prompting action.)
  - "The mixer is open."
    (Inform: provides information.)
  - "The piano roll is open now."
    (Inform: describes the current state.)

### Inform
The utterance states objective or factual information about the task, interface, current state or general domain knowledge.
- Examples:
  - "In this case, you were cutting out all of the highs."
  - "That's an example of one effect."
  - "The tempo is set to 120 BPM."

- Non-Examples:
  - "I think the button is too small."
    (Opinion: expresses a subjective judgment.)
  - "Why is the button red?"
    (Info Request: seeks information.)
  - "If you just hit the spacebar."
    (Action Directive: proposes the next action to take.)

### Opinion
The utterance expresses a subjective judgement, preference, or interpretation about the task or result.

- Examples:
  - "I think the red button is too small."
  - "The alignment looks better now."
  - "The melody feels too repetitive."
  - "I think that's fine as well."

- Non-Examples:
  - "The button is red."
    (Inform: states an objective fact.)
  - "This is, already a setter bracket."
    (Inform: states an objective screen state rather than a subjective critique.)
  - "Maybe make the button larger."
    (Action Directive: proposes an action.)

### Backchannel
The utterance signals attention, listening, receipt, agreement, understanding, or continued participation without adding substantial new propositional content.

- Examples:
  - "There you go."
  - "Mhm."
  - "Yeah."
  - "Got it."
  - "Good. Now..." 
    (The first part is a Backchannel.)

- Non-Examples:
  - "Yes, it is clipping."
    (Answer: responds with information if the previous utterance is a Info Request.)
  - "The bug is in line 24."
    (Inform: gives objective information.)
  - "Try moving it left."
    (Action Directive: proposes an action.)

### Not Enough Info
The utterance is too incomplete, cut off, vague, or ambiguous to classify reliably. As long as there is enough information to classify, use the other label. This label should not appear together with other labels.

- Examples:
  - "If you just—"
  - "Maybe the—"
  - "Wait, can you—"

- Non-Examples:
  - "Maybe use a low-pass filter here."
    (Action Directive: complete enough to classify.)
  - "And then you would want to sum..."
    (Action Directive: even it is cut off, it is clear enought to see it is an action directive.)
  - "Okay, wait. I think..."
    (Backchannel: first part is clearly a backchannel, the second part is unfinished but we already have a clear tag for it.)
  - "Hmm, maybe like..."
    (Backchannel: signals receipt rather than answering contentfully.)
"""

SYSTEM_PROMPT = f"""\
You are a linguistics expert annotating utterances from task-oriented instructional dialogue.

Your job is to read each utterance (both from Coach and Learner) in context and decide what the speaker is trying to do with it.

Choose all applicable labels from the schema below. Use the full transcript for context.

## Annotation Schema
{DIALOGUE_ACTS_PROMPT}

## Instructions

Return ONLY valid JSON as an array of objects with:
- "turn_id": integer
- "text": string (the orignal text of the turn)
- "dialogue_acts": array of all applicable labels from the schema above

Do not include markdown fences or extra text.
"""


def build_prompt(turns: list[dict]) -> str:
    transcript_lines = []
    for t in turns:
        transcript_lines.append(
            f"[Turn {t['turn_id']}] {t['speaker']}: {t['text']}"
        )
    full_transcript = "\n".join(transcript_lines)

    return f"""\
## Full Transcript

{full_transcript}
"""


def annotate(
    turns: list[dict], provider: str = "gpt", model: str | None = None
) -> list[dict]:
    prompt = build_prompt(turns)

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        model_name = model or "gemini-3.1-pro-preview"
        client = genai.Client(api_key=api_key)
        print(f"Sending to Gemini ({model_name})...")
        response = client.models.generate_content(
            model=model_name,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            config=genai_types.GenerateContentConfig(temperature=0.2),
        )
        raw = (response.text or "").strip()
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        model_name = model or "gpt-5.4"
        client = OpenAI(api_key=api_key)
        print(f"Sending to GPT ({model_name})...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    return json.loads(raw)


def apply_dialogue_acts(turns: list[dict], annotations: list[dict]) -> list[dict]:
    by_id = {int(a["turn_id"]): a for a in annotations if "turn_id" in a}

    updated: list[dict] = []
    for turn in turns:
        t = dict(turn)
        ann = by_id.get(int(t["turn_id"]))
        if ann and isinstance(ann.get("dialogue_acts"), list) and ann["dialogue_acts"]:
            t["dialogue_acts"] = ann["dialogue_acts"]
        else:
            t["dialogue_acts"] = ["Not Enough Information"]
        updated.append(t)
    return updated


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate turns.json with dialogue_acts and write turns_annotated.json."
    )
    parser.add_argument(
        "-t",
        "--task",
        default="0101",
        help='Task folder under processed (default: "0101").',
    )
    parser.add_argument(
        "-s",
        "--session",
        default="C1L1",
        help='Session folder under processed/<task> (default: "C1L1").',
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=["gpt", "gemini"],
        default="gpt",
        help='Model provider to use for annotation (default: "gpt").',
    )
    parser.add_argument(
        "--model",
        default=None,
        help='Model name override. Defaults: "gpt-5.4" for GPT, "gemini-3.1-pro-preview" for Gemini.',
    )
    args = parser.parse_args()

    session_dir = PROCESSED_DATA_FOLDER / args.task / args.session
    in_path = session_dir / "turns_splitted.json"
    provider_suffix = "gpt" if args.provider == "gpt" else "gemini"
    out_path = (
        _REPO_ROOT
        / "data_analysis"
        / "annotations"
        / f"{args.task}_{args.session}_dialogue_{provider_suffix}.json"
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    from annotation.annotate_results import load_turns

    turns = load_turns(in_path)
    annotations = annotate(turns, provider=args.provider, model=args.model)
    updated_turns = apply_dialogue_acts(turns, annotations)
    out_path.write_text(
        json.dumps(updated_turns, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
