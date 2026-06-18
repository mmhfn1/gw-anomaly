#!/usr/bin/env python3
"""
build_dataset.py
==================

Builds a labelled, preprocessed dataset (Q-images + whitened strain +
labels + PE targets) from GWOSC, saving it as a single .npz for downstream
training scripts.

Usage
-----
    # Real GWOSC data (requires outbound network access to gwosc.org):
    python scripts/build_dataset.py --source gwosc --catalog GWTC-1-confident \\
        --out data/gwtc1_dataset.npz

    # Synthetic stand-in (works anywhere, e.g. for testing this repo itself
    # without network access to GWOSC):
    python scripts/build_dataset.py --source synthetic --n-background 200 \\
        --n-events 100 --out data/synthetic_dataset.npz

See gwanomaly/data/synthetic.py for why the synthetic path exists and its
limitations as a stand-in for real strain data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gwanomaly.data.catalogue import CatalogueBuilder
from gwanomaly.data.dataset import (
    LABEL_TO_IDX,
    StrainSegmentDataset,
    build_background_dataset,
    build_event_dataset,
)
from gwanomaly.data.gwosc_client import GWOSCClient
from gwanomaly.data.synthetic import ChirpParams, make_synthetic_segment
from gwanomaly.preprocessing.pipeline import PreprocessConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _default_bandpass_high(sample_rate: int, requested: float = None) -> float:
    """
    PreprocessConfig's dataclass default (1024 Hz) assumes the standard
    4096 Hz GWOSC strain rate. Scripts/tests that use a different sample
    rate (e.g. a lower rate for faster synthetic demos) need a
    correspondingly scaled bandpass_high, or PreprocessingPipeline's own
    validation will reject it (PreprocessingPipeline requires bandpass_high
    <= 60% of Nyquist, since GWpy's bandpass() pads the stopband edge to
    min(bandpass_high*1.5, nyquist) internally). If the caller passed an
    explicit value, honour it as-is; otherwise scale proportionally to
    sample_rate.
    """
    if requested is not None:
        return requested
    nyquist = sample_rate / 2.0
    return min(1024.0, 0.55 * nyquist)


def build_from_gwosc(args) -> StrainSegmentDataset:
    client = GWOSCClient(cache_dir=args.cache_dir, detectors=tuple(args.detectors))
    catalogue = CatalogueBuilder(catalogs=(args.catalog,))
    events = catalogue.build()
    logger.info("Found %d events with usable mass/source-type labels", len(events))

    bandpass_high = _default_bandpass_high(client.sample_rate, args.bandpass_high)
    dataset = StrainSegmentDataset(
        PreprocessConfig(qimage_shape=tuple(args.qimage_shape), bandpass_high=bandpass_high)
    )
    build_event_dataset(client, events, dataset=dataset)
    logger.info("Added %d event windows", len(dataset))

    quiet_ranges = catalogue.find_quiet_gps_ranges(
        search_start=args.bg_gps_start, search_end=args.bg_gps_end
    )
    for det in args.detectors:
        build_background_dataset(
            client, det, quiet_ranges, dataset=dataset, max_segments=args.n_background
        )
    logger.info("Total windows after adding background: %d", len(dataset))
    return dataset


def build_from_synthetic(args) -> StrainSegmentDataset:
    """
    Synthetic fallback for environments without GWOSC network access (this
    sandbox included). NOT physically rigorous — see
    gwanomaly/data/synthetic.py docstring. Useful for exercising this
    repo's preprocessing/detection/classification code paths end-to-end.
    """
    bandpass_high = _default_bandpass_high(args.sample_rate, args.bandpass_high)
    dataset = StrainSegmentDataset(
        PreprocessConfig(qimage_shape=tuple(args.qimage_shape), bandpass_high=bandpass_high)
    )
    rng = np.random.default_rng(args.seed)

    for i in range(args.n_background):
        seg = make_synthetic_segment(
            detector="H1", duration=args.duration, sample_rate=args.sample_rate, seed=int(rng.integers(1e9))
        )
        dataset.add_segment(seg, label="BACKGROUND")

    source_types = ["BNS", "NSBH", "BBH"]
    mass_ranges = {"BNS": (1.0, 2.5), "NSBH": (1.0, 2.5), "BBH": (5.0, 60.0)}
    for i in range(args.n_events):
        source = source_types[i % len(source_types)]
        lo, hi = mass_ranges[source]
        m1 = float(rng.uniform(lo, hi))
        m2 = float(rng.uniform(lo, hi)) if source != "NSBH" else float(rng.uniform(5.0, 60.0))
        chirp_mass = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

        seg = make_synthetic_segment(
            detector="H1",
            duration=args.duration,
            sample_rate=args.sample_rate,
            inject=ChirpParams(chirp_mass=chirp_mass, amplitude=float(rng.uniform(2e-20, 8e-20))),
            event_name=f"SYNTH-{i:04d}",
            seed=int(rng.integers(1e9)),
        )
        dataset.add_segment(
            seg,
            label=source,
            params={"mass_1": m1, "mass_2": m2, "distance_mpc": float(rng.uniform(100, 2000))},
        )

    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["gwosc", "synthetic"], required=True)
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--qimage-shape", nargs=2, type=int, default=[128, 128])
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    parser.add_argument(
        "--bandpass-high", type=float, default=None,
        help="Upper bandpass frequency (Hz). Defaults to a value safely "
             "below Nyquist for the chosen sample rate if not set.",
    )

    # gwosc-source args
    parser.add_argument("--catalog", default="GWTC-1-confident")
    parser.add_argument("--cache-dir", default="./gwosc_cache")
    parser.add_argument("--bg-gps-start", type=float, default=1126051217.0)  # O1 start
    parser.add_argument("--bg-gps-end", type=float, default=1137254417.0)    # O1 end
    parser.add_argument("--n-background", type=int, default=200)

    # synthetic-source args
    parser.add_argument("--n-events", type=int, default=100)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--sample-rate", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.source == "gwosc":
        dataset = build_from_gwosc(args)
    else:
        logger.warning(
            "Using SYNTHETIC data, not real GWOSC strain. This is a stand-in "
            "for environments without network access to gwosc.org (see "
            "gwanomaly/data/synthetic.py). Re-run with --source gwosc on a "
            "machine with internet access for real training data."
        )
        dataset = build_from_synthetic(args)

    X_q, X_s, y, meta = dataset.to_arrays()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        qimages=X_q,
        strains=X_s,
        labels=y,
        meta=np.array(meta, dtype=object),
        label_to_idx=np.array(list(LABEL_TO_IDX.items()), dtype=object),
    )
    logger.info("Saved dataset with %d windows to %s", len(dataset), out_path)


if __name__ == "__main__":
    main()
