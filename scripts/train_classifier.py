#!/usr/bin/env python3
"""
train_classifier.py
======================

Trains the CNN source-type classifier (BACKGROUND/BNS/NSBH/BBH/BURST) on a
dataset built by build_dataset.py. Handles the natural class imbalance
(BACKGROUND windows vastly outnumber labelled events in real GWOSC data)
via inverse-frequency class weighting.

Usage
-----
    python scripts/train_classifier.py --dataset data/gwtc1_dataset.npz \\
        --out models/cnn_classifier.pt --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gwanomaly.classification.cnn_classifier import CNNClassifierConfig, CNNClassifierTrainer
from gwanomaly.data.dataset import IDX_TO_LABEL
from gwanomaly.utils.metrics import classification_report


def compute_class_weights(y: np.ndarray, n_classes: int) -> list:
    """Inverse-frequency weighting, normalised so weights average to 1.0,
    to counteract BACKGROUND >> {BNS,NSBH,BBH,BURST} imbalance without
    distorting the overall loss scale."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for absent classes
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    X_q, y = data["qimages"], data["labels"]
    n_classes = len(IDX_TO_LABEL)

    n_val = max(1, int(len(X_q) * args.val_fraction))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X_q))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X_q[train_idx], y[train_idx]
    X_val, y_val = X_q[val_idx], y[val_idx]

    print(f"train: {len(X_train)} windows, val: {len(X_val)} windows")
    print(f"train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    class_weights = compute_class_weights(y_train, n_classes)
    print(f"class weights (inverse-frequency): "
          f"{ {IDX_TO_LABEL[i]: round(w, 2) for i, w in enumerate(class_weights)} }")

    cfg = CNNClassifierConfig(
        input_shape=X_q.shape[1:],
        n_classes=n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        class_weights=class_weights,
    )
    trainer = CNNClassifierTrainer(cfg, device=args.device)
    print(f"training on device: {trainer.device}")
    trainer.fit(X_train, y_train, X_val, y_val, verbose=True)

    preds = trainer.predict(X_val)
    print("\n" + classification_report(y_val, preds, IDX_TO_LABEL))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.out)
    print(f"\nsaved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
