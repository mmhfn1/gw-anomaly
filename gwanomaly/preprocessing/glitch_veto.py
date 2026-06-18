"""
glitch_veto.py
================

Detector coincidence + glitch-veto utilities.

Real LIGO/Virgo analyses use official data-quality flags (CAT1/CAT2/CAT3
vetoes, DQSEGDB) to exclude times with known instrumental problems. Those
flags are themselves fetched from GWOSC/DQSEGDB and are out of scope to
reimplement here, so this module focuses on two things that are fully
self-contained and useful regardless of flag availability:

1. A simple statistical glitch veto: flag windows where whitened-strain
   excess power in a short sliding window exceeds a threshold in a way
   that's inconsistent with Gaussian noise (a fast proxy for "something
   loud and short happened here that isn't necessarily an astrophysical
   signal" — i.e. a candidate instrumental glitch).
2. Detector coincidence checking: for multi-detector analyses, require
   that a candidate trigger time lines up (within light-travel time
   between sites) across at least two detectors before being treated as
   astrophysically interesting, which is standard practice for
   suppressing single-detector glitches.

For production use, prefer combining this with real DQSEGDB flags via
`gwpy.segments.DataQualityFlag.query()`, which requires the same GWOSC
network access as the strain fetch itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Light-travel time between LIGO Hanford-Livingston, Hanford-Virgo,
# Livingston-Virgo (ms), used as coincidence windows.
LIGHT_TRAVEL_TIME_MS = {
    ("H1", "L1"): 10.0,
    ("H1", "V1"): 27.3,
    ("L1", "V1"): 26.4,
}


@dataclass
class GlitchVetoConfig:
    window_seconds: float = 0.1
    z_threshold: float = 6.0  # sliding-window RMS z-score threshold
    min_separation_seconds: float = 0.5  # merge flags closer than this


def apply_glitch_vetoes(
    whitened_strain: np.ndarray,
    sample_rate: int,
    config: GlitchVetoConfig = None,
) -> List[Tuple[float, float]]:
    """
    Flag candidate glitch windows in a whitened strain timeseries via
    sliding-window RMS z-score. Returns a list of (start_s, end_s) flagged
    intervals, relative to the start of the array.

    This is intentionally simple (no ML) so it's cheap to run as a
    pre-filter before the autoencoder/classifier ever see the data, and is
    transparent about what it's flagging (loud transients), since false
    positives here just mean "send to the anomaly detector anyway" rather
    than silently dropping data.
    """
    config = config or GlitchVetoConfig()
    win_n = max(int(config.window_seconds * sample_rate), 1)

    # Sliding RMS via convolution of squared signal
    sq = whitened_strain ** 2
    kernel = np.ones(win_n) / win_n
    rms = np.sqrt(np.convolve(sq, kernel, mode="same"))

    mu, sigma = rms.mean(), rms.std() + 1e-12
    z = (rms - mu) / sigma

    flagged_mask = z > config.z_threshold
    intervals = _mask_to_intervals(flagged_mask, sample_rate)
    intervals = _merge_close_intervals(intervals, config.min_separation_seconds)
    return intervals


def _mask_to_intervals(mask: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
    intervals = []
    in_run = False
    start_idx = 0
    for i, flagged in enumerate(mask):
        if flagged and not in_run:
            in_run, start_idx = True, i
        elif not flagged and in_run:
            in_run = False
            intervals.append((start_idx / sample_rate, i / sample_rate))
    if in_run:
        intervals.append((start_idx / sample_rate, len(mask) / sample_rate))
    return intervals


def _merge_close_intervals(
    intervals: List[Tuple[float, float]], min_sep: float
) -> List[Tuple[float, float]]:
    if not intervals:
        return intervals
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= min_sep:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def check_coincidence(
    trigger_times: dict,  # detector -> trigger GPS time
    tolerance_ms: float = 15.0,
) -> bool:
    """
    Given trigger times from two or more detectors, check whether they're
    consistent with a single astrophysical source (i.e. within light-travel
    time + timing tolerance of each other). Used to suppress single-IFO
    glitches from being passed to the classifier as candidate events.
    """
    if len(trigger_times) < 2:
        return False

    dets = list(trigger_times.keys())
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            d1, d2 = dets[i], dets[j]
            pair = tuple(sorted((d1, d2)))
            max_dt_ms = LIGHT_TRAVEL_TIME_MS.get(pair, 30.0) + tolerance_ms
            dt_ms = abs(trigger_times[d1] - trigger_times[d2]) * 1000.0
            if dt_ms > max_dt_ms:
                return False
    return True
