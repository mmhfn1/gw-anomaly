#!/usr/bin/env python3
"""
train_autoencoder.py
======================

Trains the convolutional autoencoder anomaly detector on the BACKGROUND
windows of a dataset built by build_dataset.py, then calibrates its
detection threshold and reports separation between background and
event/anomaly windows.

Usage
-----
    python scripts/train_autoencoder.py --dataset data/gwtc1_dataset.npz \\
        --out models/autoencoder.pt --epochs 100

Note on compute: this is full-scale training code intended for GPU. On
CPU, expect this to be slow for production-sized datasets/epoch counts —
drop --epochs and dataset size for a quick CPU smoke test (see
notebooks/demo.ipynb for a tiny end-to-end example sized for CPU).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gwanomaly.data.dataset import LABEL_TO_IDX
from gwanomaly.detection.autoencoder import AutoencoderConfig, AutoencoderDetector
from gwanomaly.utils.metrics import far_from_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help=".npz produced by build_dataset.py")
    parser.add_argument("--out", required=True, help="output checkpoint path (.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu'; auto-detected if not set")
    parser.add_argument("--segment-seconds", type=float, default=32.0, help="for FAR reporting")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    X_q, y = data["qimages"], data["labels"]

    bg_mask = y == LABEL_TO_IDX["BACKGROUND"]
    event_mask = ~bg_mask
    X_bg, X_event = X_q[bg_mask], X_q[event_mask]
    print(f"background windows: {len(X_bg)}, event windows: {len(X_event)}")

    if len(X_bg) < 20:
        print(
            "WARNING: fewer than 20 background windows. The autoencoder "
            "needs a substantial background-only training set (hundreds to "
            "thousands of windows for real use) to learn what 'normal' "
            "looks like; results on a tiny set are not meaningful beyond "
            "confirming the code runs."
        )

    n_val = max(1, int(len(X_bg) * args.val_fraction))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X_bg))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, X_val = X_bg[train_idx], X_bg[val_idx]

    cfg = AutoencoderConfig(
        input_shape=X_q.shape[1:],
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threshold_percentile=args.threshold_percentile,
    )
    detector = AutoencoderDetector(cfg, device=args.device)
    print(f"training on device: {detector.device}")
    detector.fit(X_train, X_val, verbose=True)

    detector.calibrate(X_val)
    print(f"\ncalibrated threshold (p{args.threshold_percentile}): {detector.threshold:.5f}")

    bg_scores = detector.score(X_val)
    far = far_from_threshold(bg_scores, detector.threshold, len(X_val) * args.segment_seconds)
    if far > 0:
        days = 1 / far / 86400
        unit, val = ("days", days) if days >= 1 else ("hours", days * 24) if days * 24 >= 1 else ("seconds", days * 86400)
        print(f"implied background FAR at threshold: {far:.3e} Hz (~1 per {val:.2f} {unit})")
    else:
        print("implied background FAR at threshold: 0 (no exceedances in val set)")

    if len(X_event) > 0:
        event_scores = detector.score(X_event)
        detection_rate = detector.predict(X_event).mean()
        print(f"\nevent/anomaly windows: mean score {event_scores.mean():.5f} "
              f"(background val mean: {bg_scores.mean():.5f})")
        print(f"detection rate on event windows at this threshold: {detection_rate:.1%}")
        if event_scores.mean() <= bg_scores.mean() * 1.1:
            print(
                "\nWARNING: event scores are not meaningfully higher than "
                "background scores. This usually means either (a) the "
                "training set is too small/synthetic for the model to learn "
                "a useful background representation, or (b) the events in "
                "this dataset are too subtle relative to background "
                "variance at this duration/SNR. Inspect score distributions "
                "before trusting this detector on real candidates."
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    detector.save(args.out)
    print(f"\nsaved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
