"""Benchmark runner for closed-source MLLMs on MIDA.

Supports any OpenAI-compatible chat-completions API (OpenAI, Azure OpenAI,
Anthropic via openai-compatible proxy, Gemini via openai-compatible proxy, etc.).
Reads the API key strictly from the OPENAI_API_KEY environment variable - never
hard-code keys.

Usage:
    export OPENAI_API_KEY=...
    python run_closed_source.py \\
        --data-root data \\
        --source ego4d \\
        --model gpt-4o-mini \\
        --frame-dir frames/ego4d \\
        --out predictions/ego4d_gpt4o_mini.csv

The optional --frame-dir / --frame-pattern arguments enable visual input by
attaching one frame per utterance; omit them for a text-only run.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# `openai` is the only required external dependency for this script.
from openai import OpenAI

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from utils.dataset import load_split, Game, Utterance  # noqa: E402
from utils.parsing import parse_json_response  # noqa: E402
from eval.metrics import strategy_metrics, veracity_metrics  # noqa: E402

PROMPT_DIR = CODE_ROOT / "prompts"
GAME_RULES = (PROMPT_DIR / "game_rules.txt").read_text(encoding="utf-8")
MAIN_TASK_TEMPLATE = (PROMPT_DIR / "main_task.txt").read_text(encoding="utf-8")


def _build_prompt(history: str, utt: Utterance) -> str:
    return MAIN_TASK_TEMPLATE.format(
        rules=GAME_RULES,
        history=history,
        timestamp=utt.timestamp,
        speaker=utt.speaker,
        statement=utt.line,
    )


def _encode_image(path: Path) -> str | None:
    if not path.is_file():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _timestamp_to_seconds(timestamp: str) -> int | None:
    """Parse 'mm:ss' or 'hh:mm:ss' into total seconds."""
    parts = timestamp.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass
    return None


def _frame_path_for(
    utt: Utterance, frame_dir: Path | None, frame_pattern: str, part: int
) -> Path | None:
    """Build the per-utterance frame path given the current Ego4D video-part index.

    Ego4D Werewolf game sessions are stored as 1-3 contiguous video segments;
    timestamps reset between segments. The runner tracks the current `part`
    (1-indexed) for each game and feeds it into the frame_pattern so that the
    correct folder is selected (e.g. <game_id>_1/, <game_id>_2/).
    """
    if frame_dir is None:
        return None
    parts = utt.timestamp.split(":")[-2:]
    try:
        minutes, seconds = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None
    try:
        formatted = frame_pattern.format(
            game_id=utt.game_id,
            mm=minutes,
            ss=seconds,
            timestamp=utt.timestamp,
            part=part,
        )
    except KeyError:
        return None
    return frame_dir / formatted


def _call_model(
    client: OpenAI, model: str, prompt: str, image: Path | None, temperature: float | None
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image is not None:
        b64 = _encode_image(image)
        if b64:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
    kwargs: dict[str, Any] = {}
    # GPT-5 family only accepts the provider's default sampling settings.
    if temperature is not None and not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Respond ONLY with valid JSON."},
                {"role": "user", "content": content},
            ],
            **kwargs,
        )
    except Exception as e:  # network / rate-limit / unsupported parameter
        return {"_error": str(e)}
    parsed = parse_json_response(resp.choices[0].message.content)
    if parsed is None:
        return {"_error": "json_parse_failed"}
    return parsed


def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")
    base_url = os.environ.get("OPENAI_BASE_URL")  # optional
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    games: list[Game] = load_split(args.data_root, args.source)
    frame_dir = Path(args.frame_dir) if args.frame_dir else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "game_id", "timestamp", "speaker", "line",
        "true_category", "Truthfulness",
        *(f"pred_{k}" for k in (
            "identity_declaration", "accusation", "defense",
            "evidence", "interrogation", "call_for_action",
            "veracity", "reasoning",
        )),
        "error",
    ]

    pred_strategy: list[dict[str, Any]] = []
    pred_veracity: list[str] = []
    gold_categories: list[str] = []
    gold_veracity: list[str] = []
    n_errors = 0

    with out_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for game in games:
            history: list[str] = []
            current_part = 1
            prev_seconds: int | None = None
            for utt in game.utterances:
                # Detect Ego4D part-segment switch: when the running timestamp
                # jumps backwards within the same game, the recording has moved
                # from part N to part N+1.
                cur_seconds = _timestamp_to_seconds(utt.timestamp)
                if (
                    prev_seconds is not None
                    and cur_seconds is not None
                    and cur_seconds < prev_seconds
                ):
                    current_part += 1
                prev_seconds = cur_seconds if cur_seconds is not None else prev_seconds

                prompt = _build_prompt("\n".join(history), utt)
                image = _frame_path_for(utt, frame_dir, args.frame_pattern, current_part)
                resp = _call_model(client, args.model, prompt, image, args.temperature)
                row = {
                    "game_id": game.game_id,
                    "timestamp": utt.timestamp,
                    "speaker": utt.speaker,
                    "line": utt.line,
                    "true_category": utt.true_category,
                    "Truthfulness": utt.truthfulness,
                    "error": resp.get("_error", ""),
                }
                for k in (
                    "identity_declaration", "accusation", "defense",
                    "evidence", "interrogation", "call_for_action",
                    "veracity", "reasoning",
                ):
                    row[f"pred_{k}"] = resp.get(k, "")
                writer.writerow(row)

                # As in the paper, failed API calls / unparsable responses are
                # excluded from scoring and reported as `n_errors`.
                if "_error" in resp:
                    n_errors += 1
                else:
                    pred_strategy.append({k: resp.get(k, 0) for k in (
                        "identity_declaration", "accusation", "defense",
                        "evidence", "interrogation", "call_for_action",
                    )})
                    pred_veracity.append(str(resp.get("veracity", "Neutral")))
                    gold_categories.append(utt.true_category)
                    gold_veracity.append(utt.truthfulness)

                history.append(f"[{utt.timestamp}] {utt.speaker}: {utt.line}")

    # Metrics summary
    strat = strategy_metrics(gold_categories, pred_strategy)
    vera = veracity_metrics(gold_veracity, pred_veracity)
    summary = {"strategy": strat, "veracity": vera,
               "n_scored": len(gold_veracity), "n_errors": n_errors}
    summary_path = out_path.with_suffix(".metrics.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--source", default="ego4d", choices=["ego4d", "youtube"])
    p.add_argument("--model", default="gpt-4o-mini",
                   help="Model id passed to the chat completions API.")
    p.add_argument("--frame-dir", default=None,
                   help="Optional directory containing per-game video frames.")
    p.add_argument("--frame-pattern", default="{game_id}_{part}/frame_00_{mm:02d}_{ss:02d}.jpg",
                   help=("Path pattern under --frame-dir. Supported placeholders: "
                         "{game_id}, {part} (1-indexed Ego4D segment), {mm}, {ss}, {timestamp}. "
                         "The runner increments {part} automatically when the timestamp "
                         "wraps backwards within a single game (Ego4D records games as 1-3 "
                         "contiguous video segments). For YouTube data, pass a pattern that "
                         "omits {part} (e.g. '{game_id}/frame_00_{mm:02d}_{ss:02d}.jpg')."))
    p.add_argument("--temperature", type=float, default=None,
                   help="Optional sampling temperature; omitted from the request when not "
                        "set (and always for gpt-5* models, which only accept the default).")
    p.add_argument("--out", default="predictions/closed_source.csv")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
