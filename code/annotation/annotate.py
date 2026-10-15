"""Semi-automated MIDA annotation pipeline.

Takes a raw transcript (Timestamp, Speaker, Line) plus the per-game vote_outcome
JSON (which records start/end roles, voting outcome, and any night-action notes)
and asks an LLM playing the role of "Game Analyst for AI Deception Detection"
to assign per-utterance veracity labels (TRUE / FALSE / NEUTRAL) plus a short
explanation, following the rulebook reproduced in code/prompts/annotation_rules.txt.

Output format matches the public MIDA labels:
    Timestamp,Speaker,Line,true_category,Truthfulness,Explanation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

PROMPT_PATH = CODE_ROOT / "prompts" / "annotation_rules.txt"


def load_vote_outcome(vote_path: str | os.PathLike, game_key: str) -> dict:
    data = json.loads(Path(vote_path).read_text(encoding="utf-8"))
    if game_key not in data:
        raise KeyError(f"{game_key} not found in {vote_path}")
    return data[game_key]


def build_prompt(transcript_csv: str | os.PathLike, vote_outcome: dict) -> str:
    """Assemble the full annotation prompt: rulebook + per-game inputs.

    The rulebook (which contains the role description, labeling rules, and
    suggested workflow) is concatenated with the per-game JSON (game state +
    nightActions) and the dialogue transcript.
    """
    rules = PROMPT_PATH.read_text(encoding="utf-8")
    transcript = Path(transcript_csv).read_text(encoding="utf-8")
    return (
        rules
        + "\n\n"
        + "Per-Game Inputs\n===============\n\n"
        + "Game State (JSON, includes nightActions):\n"
        + json.dumps(vote_outcome, ensure_ascii=False, indent=2)
        + "\n\nDialogue Transcript (CSV):\n"
        + transcript
        + "\n\nReturn the labelled CSV only, with the exact header "
        "'Timestamp,Speaker,Line,true_category,Truthfulness,Explanation'."
    )


def call_annotator(prompt: str, model: str = "gemini-2.5-pro") -> str:
    """Calls an OpenAI-compatible API. For Gemini-2.5-Pro use a Gemini-compatible
    OPENAI_BASE_URL or your own provider endpoint."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a Game Analyst for AI Deception Detection. "
                        "Follow the supplied rulebook exactly and return only the "
                        "labelled CSV in the requested format."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcript", required=True, help="Per-game transcript CSV.")
    p.add_argument("--vote-outcome", required=True, help="Path to per-recording vote_outcome JSON.")
    p.add_argument("--game-key", required=True, help="e.g. 'Game1'.")
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--out", required=True, help="Output labelled CSV path.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    vote = load_vote_outcome(args.vote_outcome, args.game_key)
    prompt = build_prompt(args.transcript, vote)
    csv_text = call_annotator(prompt, model=args.model)
    Path(args.out).write_text(csv_text, encoding="utf-8")
    print(f"wrote {args.out}")
