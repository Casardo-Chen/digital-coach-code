"""Dialogue-act annotation for every turn (Coach and Learner) via OpenRouter."""

import argparse

from annotation.common import add_session_args, call_json_model, load_input_turns, write_output

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


def annotate(turns: list[dict], model: str | None = None) -> list[dict]:
    return call_json_model(SYSTEM_PROMPT, build_prompt(turns), model)


def apply_dialogue_acts(turns: list[dict], annotations: list[dict]) -> list[dict]:
    by_id = {int(a["turn_id"]): a for a in annotations if "turn_id" in a}

    updated: list[dict] = []
    for turn in turns:
        t = dict(turn)
        ann = by_id.get(int(t["turn_id"]))
        if ann and isinstance(ann.get("dialogue_acts"), list) and ann["dialogue_acts"]:
            t["dialogue_acts"] = ann["dialogue_acts"]
        else:
            t["dialogue_acts"] = ["Not Enough Info"]
        updated.append(t)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Label each turn with dialogue acts.")
    add_session_args(parser)
    args = parser.parse_args()

    turns, stem = load_input_turns(args)
    updated = apply_dialogue_acts(turns, annotate(turns, model=args.model))
    write_output(updated, stem, "dialogue_acts")


if __name__ == "__main__":
    main()
