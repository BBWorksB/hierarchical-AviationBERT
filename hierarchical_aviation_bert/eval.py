"""Metrics for hierarchical ADREP classification.

Computes everything reported in Table 1 of the paper:
    Macro-F1, Micro-F1, Accuracy, Minority-class recall (MinRec),
    Taxonomy Compliance Rate (TCR).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)


def minority_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_counts: np.ndarray,
    tail_fraction: float = 0.3,
) -> float:
    """Mean recall over the bottom `tail_fraction` of classes by support.

    Class ordering is by `class_counts` ascending (rarest first).
    """
    rank = np.argsort(class_counts)
    k = max(1, int(round(tail_fraction * len(rank))))
    minority_labels = set(rank[:k].tolist())
    mask = np.array([y in minority_labels for y in y_true])
    if mask.sum() == 0:
        return float("nan")
    return float(recall_score(
        y_true[mask],
        y_pred[mask],
        labels=sorted(minority_labels),
        average="macro",
        zero_division=0,
    ))


def taxonomy_compliance_rate(
    y_pred_child: np.ndarray,
    y_pred_parent: Optional[np.ndarray],
    child_to_parent: np.ndarray,
) -> float:
    """Fraction of predictions where the predicted child's parent matches the
    predicted parent (FlatHead has no parent head → returns NaN)."""
    if y_pred_parent is None:
        return float("nan")
    implied_parents = child_to_parent[y_pred_child]
    return float((implied_parents == y_pred_parent).mean())


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_child: np.ndarray,
    y_pred_parent: Optional[np.ndarray],
    class_counts: np.ndarray,
    child_to_parent: np.ndarray,
    child_names: Sequence[str],
) -> Dict[str, float]:
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_child, average="macro", zero_division=0,
    )
    mic_p, mic_r, mic_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_child, average="micro", zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred_child)
    minrec = minority_recall(y_true, y_pred_child, class_counts)
    tcr = taxonomy_compliance_rate(y_pred_child, y_pred_parent, child_to_parent)

    per_class_f1 = f1_score(
        y_true, y_pred_child,
        labels=list(range(len(child_names))),
        average=None, zero_division=0,
    )

    return {
        "macro_f1": float(mac_f1),
        "micro_f1": float(mic_f1),
        "accuracy": float(acc),
        "minority_recall": float(minrec),
        "tcr": float(tcr),
        "macro_precision": float(mac_p),
        "macro_recall": float(mac_r),
        "per_class_f1": {c: float(f) for c, f in zip(child_names, per_class_f1)},
    }
