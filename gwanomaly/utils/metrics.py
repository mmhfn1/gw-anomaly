"""
metrics.py
===========

Evaluation utilities shared across detection/classification stages.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: Optional[Dict[int, str]] = None) -> str:
    """
    Per-class precision/recall/F1, computed from scratch (no sklearn
    dependency assumed elsewhere in this package). Useful given the
    heavy class imbalance typical here (BACKGROUND vastly outnumbers
    BNS/NSBH/BBH/BURST), where overall accuracy alone is misleading.
    """
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    lines = [f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]

    for c in classes:
        name = label_names.get(c, str(c)) if label_names else str(c)
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        lines.append(f"{name:<12}{precision:>10.3f}{recall:>10.3f}{f1:>10.3f}{support:>10d}")

    overall_acc = float((y_true == y_pred).mean())
    lines.append(f"\noverall accuracy: {overall_acc:.3f}  (n={len(y_true)})")
    return "\n".join(lines)


def far_from_threshold(
    background_scores: np.ndarray,
    threshold: float,
    background_duration_seconds: float,
) -> float:
    """
    Estimate false-alarm rate (FAR, in Hz) implied by a given detection
    threshold, from the fraction of background scores exceeding it and
    the total background time those scores were drawn from.

    FAR = (# background scores above threshold) / (total background time)

    This is the standard quantity LIGO/Virgo alerts are reported with
    (e.g. "FAR = 1 per 1000 years") — report candidate significance this
    way rather than a raw SNR/score value alone, since SNR thresholds
    don't have a fixed, comparable meaning across different detectors,
    pipelines, or data-taking eras.
    """
    n_above = int((background_scores > threshold).sum())
    if background_duration_seconds <= 0:
        raise ValueError("background_duration_seconds must be positive")
    return n_above / background_duration_seconds
