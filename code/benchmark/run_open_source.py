"""Benchmark runner for open-source MLLMs on MIDA.

Runs zero-shot inference with HuggingFace Transformers. Supports any chat model
that exposes a transformers `AutoModelForCausalLM` or vision-language model that
follows the standard chat-template / image-input convention used by Qwen2-VL,
Qwen2.5-VL, InternVL, etc.

Usage (text-only):
    python run_open_source.py \\
        --data-root data --source ego4d \\
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --out predictions/ego4d_qwen25vl_text.csv

Usage (with one frame per utterance):
    python run_open_source.py ... --frame-dir frames/ego4d
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from utils.dataset import Game, Utterance, load_split  # noqa: E402
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


def _timestamp_to_seconds(timestamp: str) -> int | None:
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
    """See run_closed_source._frame_path_for for the multi-segment logic."""
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


def _load_model(model_id: str):
    """Load a chat-capable HF model. Supports both vision-language and text-only.
    Vision pipelines are detected by the presence of an `AutoProcessor` with image
    inputs; otherwise falls back to text-only with `AutoTokenizer`.
    """
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
    )

    # Try the vision-language path first.
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = cls.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            return ("vlm", model, processor)
        except (ValueError, KeyError, OSError):
            continue
    print(f"[run_open_source] {model_id}: no vision-language class matched; "
          "loading as a text-only model (frames will be ignored).", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    return ("text", model, tokenizer)


def _generate_vlm(model, processor, prompt: str, image_path: Path | None, gen_kwargs: dict) -> str:
    from PIL import Image
    images = [Image.open(image_path).convert("RGB")] if image_path and image_path.is_file() else None
    user_content = [{"type": "text", "text": prompt}]
    if images is not None:
        user_content.insert(0, {"type": "image"})
    messages = [
        {"role": "system", "content": "Respond ONLY with valid JSON."},
        {"role": "user", "content": user_content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=images, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **gen_kwargs)
    decoded = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return decoded


def _generate_text(model, tokenizer, prompt: str, gen_kwargs: dict) -> str:
    messages = [
        {"role": "system", "content": "Respond ONLY with valid JSON."},
        {"role": "user", "content": prompt},
    ]
    # enable_thinking is read by the Qwen3 chat template (the paper ran Qwen3 in
    # thinking mode) and ignored by other templates.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run(args: argparse.Namespace) -> None:
    kind, model, helper = _load_model(args.model_id)
    gen_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if not args.use_generation_config:
        gen_kwargs["do_sample"] = False  # greedy decoding

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

    with out_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for game in games:
            history: list[str] = []
            current_part = 1
            prev_seconds: int | None = None
            for utt in game.utterances:
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
                if kind == "vlm":
                    raw = _generate_vlm(model, helper, prompt, image, gen_kwargs)
                else:
                    raw = _generate_text(model, helper, prompt, gen_kwargs)
                # Unparsable outputs are scored with the default prediction
                # (all strategies FALSE, veracity Neutral).
                resp = parse_json_response(raw) or {"_error": "json_parse_failed"}

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

                pred_strategy.append({k: resp.get(k, 0) for k in (
                    "identity_declaration", "accusation", "defense",
                    "evidence", "interrogation", "call_for_action",
                )})
                pred_veracity.append(str(resp.get("veracity", "Neutral")))
                gold_categories.append(utt.true_category)
                gold_veracity.append(utt.truthfulness)

                history.append(f"[{utt.timestamp}] {utt.speaker}: {utt.line}")

    summary = {
        "strategy": strategy_metrics(gold_categories, pred_strategy),
        "veracity": veracity_metrics(gold_veracity, pred_veracity),
    }
    out_path.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--source", default="ego4d", choices=["ego4d", "youtube"])
    p.add_argument("--model-id", required=True,
                   help="HuggingFace model id, e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--frame-dir", default=None)
    p.add_argument("--frame-pattern", default="{game_id}_{part}/frame_00_{mm:02d}_{ss:02d}.jpg",
                   help=("Path pattern under --frame-dir. Supported placeholders: "
                         "{game_id}, {part} (1-indexed Ego4D segment, auto-incremented on "
                         "timestamp wrap-around), {mm}, {ss}, {timestamp}. "
                         "For YouTube, use a pattern that omits {part}."))
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="Maximum number of generated tokens (reasoning models such as "
                        "Qwen3 need a larger budget, e.g. 32768).")
    p.add_argument("--use-generation-config", action="store_true",
                   help="Decode with the model's own generation_config instead of "
                        "greedy decoding.")
    p.add_argument("--out", default="predictions/open_source.csv")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
