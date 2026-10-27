"""Coaching-method annotation for Coach turns via OpenRouter."""

import argparse

from annotation.common import add_session_args, call_json_model, load_input_turns, write_output

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


def annotate(turns: list[dict], model: str | None = None) -> list[dict]:
    return call_json_model(SYSTEM_PROMPT, build_prompt(turns), model)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Label Coach turns with coaching methods.")
    add_session_args(parser)
    args = parser.parse_args()

    turns, stem = load_input_turns(args)
    updated = apply_coaching_methods(turns, annotate(turns, model=args.model))
    write_output(updated, stem, "coaching_methods")


if __name__ == "__main__":
    main()
