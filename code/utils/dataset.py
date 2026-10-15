"""Dataset loading utilities for MIDA.

Loads per-game labeled CSVs from the MIDA-Ego4D and MIDA-YouTube splits and
yields per-utterance dicts compatible with the benchmark and annotation scripts.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass
class Utterance:
    timestamp: str
    speaker: str
    line: str
    true_category: str  # "No Strategy" or comma-separated persuasive categories
    truthfulness: str   # "TRUE" / "FALSE" / "Neutral"
    explanation: str = ""
    game_id: str = ""
    source: str = ""    # "ego4d" or "youtube"


@dataclass
class Game:
    game_id: str
    source: str
    utterances: list[Utterance] = field(default_factory=list)
    vote_outcome: dict | None = None  # contents of the per-game block in vote_outcome JSON


def _read_labeled_csv(path: str, game_id: str, source: str) -> list[Utterance]:
    rows: list[Utterance] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Utterance(
                    timestamp=r.get("Timestamp", "").strip(),
                    speaker=r.get("Speaker", "").strip(),
                    line=r.get("Line", "").strip(),
                    true_category=r.get("true_category", "").strip(),
                    truthfulness=r.get("Truthfulness", "").strip(),
                    explanation=r.get("Explanation", "").strip(),
                    game_id=game_id,
                    source=source,
                )
            )
    return rows


_EGO4D_NAME_RE = re.compile(r"labeled_ego4d_([a-f0-9-]+)_(Game\d+)\.csv$")
_YOUTUBE_NAME_RE = re.compile(r"youtube_(.+)_(Game\d+)\.csv$")


def _ego4d_game_id(filename: str) -> str:
    m = _EGO4D_NAME_RE.search(filename)
    if not m:
        return os.path.splitext(filename)[0]
    return f"{m.group(1)}_{m.group(2)}"


def _youtube_game_id(filename: str) -> str:
    m = _YOUTUBE_NAME_RE.search(filename)
    if not m:
        return os.path.splitext(filename)[0]
    return f"{m.group(1)}_{m.group(2)}"


def _load_vote_outcomes(metadata_dir: str) -> dict[str, dict]:
    """vote_outcome JSON files are keyed by recording id; the file content is a
    dict from "GameN" -> per-game info. Flatten to "<recording>_<GameN>" -> info."""
    out: dict[str, dict] = {}
    for path in glob.glob(os.path.join(metadata_dir, "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        recording = os.path.splitext(os.path.basename(path))[0]
        for game_key, info in data.items():
            out[f"{recording}_{game_key}"] = info
    return out


def load_split(data_root: str, source: str) -> list[Game]:
    """Load all games for one source ("ego4d" or "youtube").

    Expected layout under data_root:
      data_root/<source>/labels/*.csv
      data_root/<source>/metadata/vote_outcome/*.json
    """
    src_dir = os.path.join(data_root, source)
    label_dir = os.path.join(src_dir, "labels")
    meta_dir = os.path.join(src_dir, "metadata", "vote_outcome")

    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Labels directory not found: {label_dir}")

    vote_lookup = _load_vote_outcomes(meta_dir) if os.path.isdir(meta_dir) else {}

    games: list[Game] = []
    for csv_path in sorted(glob.glob(os.path.join(label_dir, "*.csv"))):
        fname = os.path.basename(csv_path)
        game_id = _ego4d_game_id(fname) if source == "ego4d" else _youtube_game_id(fname)
        utts = _read_labeled_csv(csv_path, game_id, source)
        games.append(Game(game_id=game_id, source=source, utterances=utts, vote_outcome=vote_lookup.get(game_id)))
    return games


def iter_utterances(games: Iterable[Game]) -> Iterator[Utterance]:
    for g in games:
        yield from g.utterances


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--source", default="ego4d", choices=["ego4d", "youtube"])
    args = p.parse_args()

    games = load_split(args.data_root, args.source)
    total = sum(len(g.utterances) for g in games)
    print(f"{args.source}: {len(games)} games, {total} utterances")
