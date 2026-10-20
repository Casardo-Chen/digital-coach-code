import os
import re
import csv
import json
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

COACHING_METHODS = """\
#### Direct Instruction
The coach tells the learner the concrete next action to take, often in a procedural or step-by-step way. The main purpose is to move the learner forward immediately by specifying what to do.

- Examples:
  - "Click Sketch, then select this edge."
  - "Use the Fillet tool on that corner."
  - "Open the mixer and lower the drum track by three decibels."

- Non-Examples:
  - "Let’s fix the base shape first."
    (Plan: states the next subgoal, not the exact executable operation.)
  - "Fillet rounds sharp corners."
    (Explanation: explains what the tool does.)

#### Plan
The coach states a subgoal or a sub-task without specifying the exact action. This includes narrowing the problem, chunking the task, or planning upcoming subgoals.

- Examples:
  - "Let’s fix the base shape first, then worry about texture."
  - "Then we will start making the magic potion bottle."
  - "Break it into two parts: first the drum pattern, then the melody."

- Non-Examples:
  - "Click this button first."
    (Direct Instruction: gives the exact action.)
  - "It usually helps to build the rough form before details."
    (Tip: gives general advice rather than an in-the-moment plan.)

#### Confirmation
The coach gives positive affirmation on the learner's action, result, or understanding. The main purpose is to affirm correctness or progress.

- Examples:
  - "Yep"
  - "Yeah, you can just select this"
  - "Exactly, that’s the tool you want."


- Non-Examples:
  - "The sketch is still open."
    (Diagnosis: identifies a problem.)
  - "Click that edge next."
    (Direct Instruction: gives the next action.)
  - "What do you think went wrong?"
    (Reflection: asks the learner to evaluate.)

#### Diagnosis
The coach provides neutral to negative feedback on the learner's current task progress or identifies a problem. The main purpose is to improve the learner's current state or result.

- Examples:
  - "The sketch isn’t closed, so the software can’t make a solid."
  - "To the right a little bit."
  - "So now it is one shape, and we can change the color"
  - "Can you just make it a little bit more squarish?"

- Non-Examples:
  - "Close the sketch before extruding."
    (Direct Instruction: tells the learner what to do.)
  - "A closed sketch is required for extrusion."
    (Explanation: gives general reasoning.)
  - "Yes, now it works."
    (Confirmation: affirms correctness.)

#### Explanation
The coach explains how or why something works, why something happened, or why a certain operation is appropriate. The main purpose is to build understanding.

- Examples:
  - "Extrude works here because it turns a 2D profile into a 3D solid."
  - "Smart Animate matches layers with the same names across frames."
  - "But this one we want to stay still, so uncheck dynamic."

- Non-Examples:
  - "Use Extrude here."
    (Direct Instruction: directs an action.)
  - "It's often easier to model one side and mirror it."
    (Tip: gives reusable practical advice.)

#### Tip
The coach gives practical advice, a best practice, an alternative approach, or advanced knowledge that is broadly useful beyond the immediate step. The main purpose is to share reusable know-how rather than explain a mechanism.

- Examples:
  - "It's often easier to model one side first and then mirror it."
  - "You could also duplicate this frame instead of rebuilding it."
  - "Or click elsewhere on the artboard"
  - "A good habit is to name layers clearly before animating."

- Non-Examples:
  - "Same process."
    (Direct Instruction: gives a concrete next action.)
  - "Mirroring works because it duplicates geometry symmetrically."
    (Explanation: explains how it works.)
  - "Let’s do the base shape first and details later."
    (Plan: organizes the next subgoal.)
  - "I would prefer you typing on the top where you see that fx thing"
    

#### Clarification
The coach asks for missing information about the learner’s goal, intended result, or source of confusion before giving further guidance. The main purpose is to understand the learner's situation.

- Examples:
  - "What are you trying to make this shape look like?"
  - "Which part feels confusing right now?"
  - "What do you want the output to be here?"

- Non-Examples:
  - "What do you think went wrong?"
    (Reflection: asks the learner to evaluate.)
  - "What are you thinking of trying next?"
    (Articulation: asks the learner to verbalize reasoning or plan.)
  - "Use the text animation preset."
    (Direct Instruction: gives the next step.)

#### Reflection
The coach asks the learner to evaluate their result, mistake, or choice. The main purpose is to make the learner interpret what happened and why.

- Examples:
  - "Looking at the result, what do you think went wrong?"
  - "Why do you think the audio is clipping here?"
  - "Which of these two layouts feels clearer, and why?"

- Non-Examples:
  - "What are you trying to build?"
    (Clarification: asks about goal.)
  - "What are you thinking of trying next?"
    (Articulation: asks the learner to state reasoning or plan.)
  - "Try lowering the gain."
    (Direct Instruction: tells the learner what to do.)

#### Articulation
The coach asks the learner to verbalize their reasoning, decision process, or intended next step. The main purpose is to make the learner state what they are thinking, rather than to evaluate what already happened.

- Examples:
  - "What are you thinking of trying next?"
  - "Why did you choose that tool?"
  - "Can you walk me through your reasoning?"

- Non-Examples:
  - "What are you trying to build?"
    (Clarification: asks about the learner’s goal or intended outcome.)
  - "What do you think went wrong here?"
    (Reflection: asks the learner to evaluate an outcome.)
  - "Use the Move tool next."
    (Direct Instruction: supplies the next step.)

#### Exploration
The coach intentionally leaves room for learner choice, practice themselves, or trying alternatives, without strongly steering toward one specific answer. The main purpose is to preserve learner agency and practice opportunities.

- Examples:
  - "There are a couple of options and see which one you like better."
  - "I'm gonna see if you can figure it out, and then I'll help you if you get stuck."
  - "There are several ways to build this; pick one that makes sense to you."

- Non-Examples:
  - "Use the dissolve transition."
    (Direct Instruction: specifies the action.)
  - "Which option is better, and why?"
    (Reflection: asks the learner to evaluate.)

#### Non-Instructive
The utterance is not meaningfully about the software task or learner progress, or it contains not enough information to classify.

- Examples:
  - "I don't know how to do this."
  - "Your internet seems a little unstable."
  - "Haha, I'm just kidding."
  - "If you just-"
  - "Okay. How did you make…"
  - "Wait, can you-"

- Non-Examples:
  - "What are you trying to build?"
    (Clarification: asks about the task goal.)
  - "Yeah, so, and then, so we want to do that."
    (Confirmation: confirms the learner's action.)
  - "Let’s finish the layout before we style it."
    (Plan: structures the task.)
"""


SYSTEM_PROMPT = f"""\
You are an expert annotator for coaching interactions.

Your job is to read each coach turn in context and decide which coaching method labels are used.

Choose all applicable labels from the schema below. Use the full transcript for context (to understand what the coach is \
responding to), but only annotate Coach turns.

## Annotation Schema

{COACHING_METHODS}

## Instructions

For each coach turn, return a JSON array of objects with these fields:
- "turn_id": the turn number (integer)
- "coaching_methods": an array of all applicable labels from the annotation classes

Return ONLY valid JSON (an array of objects). No markdown fencing, no extra text.
"""


def build_prompt(turns: list[dict]) -> str:
    """Build transcript payload for coach-turn annotation."""

    # Full transcript for context (all turns, both speakers)
    transcript_lines = []
    for t in turns:
        transcript_lines.append(
            f"[Turn {t['turn_id']}] | {t['speaker']}: {t['text']}"
        )
    full_transcript = "\n".join(transcript_lines)

    # Coach turns to annotate
    coach_lines = []
    for t in turns:
        if t["speaker"] == "Coach":
            coach_lines.append(
                f"[Turn {t['turn_id']}] | {t['text']}"
            )
    coach_text = "\n".join(coach_lines)

    return f"""\
## Full Transcript (for context)

{full_transcript}

## Coach Turns to Annotate

{coach_text}
"""


def annotate(
    turns: list[dict], provider: str = "gpt", model: str | None = None
) -> list[dict]:
    """Send turns to the selected model provider for annotation."""
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


def apply_coaching_methods(turns: list[dict], annotations: list[dict]) -> list[dict]:
    by_id = {int(a["turn_id"]): a for a in annotations if "turn_id" in a}

    updated: list[dict] = []
    for turn in turns:
        t = dict(turn)
        if t.get("speaker") != "Coach":
            t["coaching_methods"] = []
            updated.append(t)
            continue

        ann = by_id.get(int(t["turn_id"]))
        methods = ann.get("coaching_methods") if isinstance(ann, dict) else None
        if isinstance(methods, list):
            t["coaching_methods"] = methods
        else:
            t["coaching_methods"] = []
        updated.append(t)
    return updated


def save_annotations(
    annotations: list[dict], json_path: str | Path, csv_path: str | Path | None = None
):
    """Save annotations as JSON and CSV."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {json_path}")

    if csv_path is None:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["turn_id", "coaching_methods"]
        )
        writer.writeheader()
        for row in annotations:
            row_copy = dict(row)
            row_copy["coaching_methods"] = "; ".join(row_copy["coaching_methods"])
            writer.writerow(row_copy)
    print(f"Saved CSV: {csv_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate coach turns in processed turns_merged.json."
    )
    parser.add_argument(
        "-t",
        "--task",
        default="0101",
        help='Task folder under processed (default: "data_visualization").',
    )
    parser.add_argument(
        "-s",
        "--session",
        default="C1L1",
        help='Session folder under processed/<task> (default: "example").',
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

    def splitted_input_paths(dir_path: Path) -> list[Path]:
        """turns_splitted.json first (if present), then turns_splitted-<n>.json by numeric order."""
        paths: list[Path] = []
        base = dir_path / "turns_splitted.json"
        if base.exists():
            paths.append(base)
        numbered: list[tuple[int, Path]] = []
        for p in dir_path.iterdir():
            if not p.is_file():
                continue
            m = re.match(r"^turns_splitted-(\d+)\.json$", p.name)
            if m:
                numbered.append((int(m.group(1)), p))
        numbered.sort(key=lambda x: x[0])
        paths.extend(p for _, p in numbered)
        return paths

    in_paths = splitted_input_paths(session_dir)

    if not in_paths:
        raise FileNotFoundError(
            f"Input not found in {session_dir}. "
            "Expected turns_splitted.json and/or turns_splitted-<n>.json (e.g. turns_splitted-1.json)."
        )

    provider_suffix = "gpt" if args.provider == "gpt" else "gemini"

    for in_path in in_paths:
        m = re.match(r"^turns_splitted-(\d+)\.json$", in_path.name)
        if in_path.name == "turns_splitted.json":
            out_name = f"{args.task}_{args.session}_coaching_{provider_suffix}.json"
        elif m:
            out_name = (
                f"{args.task}_{args.session}_coaching_{provider_suffix}-{m.group(1)}.json"
            )
        else:
            out_name = f"{args.task}_{args.session}_coaching_{provider_suffix}.json"
        out_path = _REPO_ROOT / "data_analysis" / "annotations" / out_name
        from annotation.annotate_results import load_turns

        turns = load_turns(in_path)
        annotations = annotate(turns, provider=args.provider, model=args.model)
        updated_turns = apply_coaching_methods(turns, annotations)
        save_annotations(updated_turns, out_path)


if __name__ == "__main__":
    main()
