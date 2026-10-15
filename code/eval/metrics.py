"""Metrics for the MIDA benchmark.

Two metric families:
  - Persuasive Strategy Classification (per-category F1, average F1, joint accuracy).
  - Veracity / Deception Assessment (overall accuracy, binary accuracy on TRUE/FALSE,
    macro-averaged precision/recall/F1, per-class F1 for TRUE/FALSE/Neutral).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

from sklearn.metrics import f1_score, precision_score, recall_score


PERSUASIVE_CATEGORIES = (
    "identity_declaration",
    "accusation",
    "defense",
    "evidence",
    "interrogation",
    "call_for_action",
)

VERACITY_LABELS = ("TRUE", "FALSE", "Neutral")


def _normalize_bool_label(x) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(bool(x))
    s = str(x).strip().lower()
    return int(s in {"true", "1", "yes"})


def _normalize_veracity(x) -> str:
    s = str(x).strip().lower()
    if s in {"true", "t"}:
        return "TRUE"
    if s in {"false", "f"}:
        return "FALSE"
    return "Neutral"


def _category_to_binary_vector(true_category: str) -> dict[str, int]:
    """Convert a gold label into 6 binary indicators.

    Gold labels are multi-label: "No Strategy", a single category, or a
    comma-separated list such as "Defense, Evidence".
    """
    out = {c: 0 for c in PERSUASIVE_CATEGORIES}
    for part in str(true_category or "").replace(";", ",").split(","):
        cat = part.strip().strip('"').strip().lower().replace(" ", "_")
        if cat in out:
            out[cat] = 1
    return out


def strategy_metrics(
    gold_categories: Sequence[str],
    predicted: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Compute per-category F1, average F1, and joint accuracy.

    gold_categories: gold "true_category" string per utterance ("No Strategy" or one of the 6).
    predicted: list of dicts with TRUE/FALSE under each of the 6 persuasive keys.
    """
    if len(gold_categories) != len(predicted):
        raise ValueError("gold and predicted must have the same length")

    per_cat: dict[str, float] = {}
    joint_correct = 0
    n = len(gold_categories)

    gold_vectors = [_category_to_binary_vector(c) for c in gold_categories]

    for cat in PERSUASIVE_CATEGORIES:
        y_true = [v[cat] for v in gold_vectors]
        y_pred = [_normalize_bool_label(p.get(cat, 0)) for p in predicted]
        per_cat[cat] = float(f1_score(y_true, y_pred, zero_division=0)) * 100.0

    avg_f1 = sum(per_cat.values()) / len(per_cat)

    for gold_vec, pred in zip(gold_vectors, predicted):
        all_match = all(
            _normalize_bool_label(pred.get(cat, 0)) == gold_vec[cat]
            for cat in PERSUASIVE_CATEGORIES
        )
        if all_match:
            joint_correct += 1

    out = {f"f1_{c}": v for c, v in per_cat.items()}
    out["avg_f1"] = avg_f1
    out["joint_accuracy"] = (joint_correct / n) * 100.0 if n else 0.0
    return out


def veracity_metrics(
    gold_veracity: Sequence[str],
    predicted_veracity: Sequence[str],
) -> dict[str, float]:
    """Accuracy, binary accuracy on TRUE/FALSE only, macro P/R/F1, per-class F1."""
    if len(gold_veracity) != len(predicted_veracity):
        raise ValueError("gold and predicted must have the same length")

    y_true = [_normalize_veracity(x) for x in gold_veracity]
    y_pred = [_normalize_veracity(x) for x in predicted_veracity]

    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = (correct / n) * 100.0 if n else 0.0

    binary_idx = [i for i, t in enumerate(y_true) if t in {"TRUE", "FALSE"}]
    if binary_idx:
        bin_correct = sum(1 for i in binary_idx if y_true[i] == y_pred[i])
        binary_accuracy = (bin_correct / len(binary_idx)) * 100.0
    else:
        binary_accuracy = 0.0

    macro_p = float(precision_score(y_true, y_pred, labels=list(VERACITY_LABELS), average="macro", zero_division=0)) * 100.0
    macro_r = float(recall_score(y_true, y_pred, labels=list(VERACITY_LABELS), average="macro", zero_division=0)) * 100.0
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(VERACITY_LABELS), average="macro", zero_division=0)) * 100.0

    f1_per = f1_score(y_true, y_pred, labels=list(VERACITY_LABELS), average=None, zero_division=0)
    out = {
        "accuracy": accuracy,
        "binary_accuracy": binary_accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }
    for label, val in zip(VERACITY_LABELS, f1_per):
        out[f"f1_{label.lower()}"] = float(val) * 100.0
    return out


def label_distribution(gold_veracity: Iterable[str]) -> dict[str, int]:
    return dict(Counter(_normalize_veracity(x) for x in gold_veracity))


if __name__ == "__main__":
    # Tiny self-test
    gold = ["TRUE", "FALSE", "Neutral", "TRUE"]
    pred = ["TRUE", "Neutral", "Neutral", "TRUE"]
    print(veracity_metrics(gold, pred))
