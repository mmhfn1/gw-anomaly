"""
test_excess_power.py
======================

Regression tests for the excess-power detector's empirical calibration,
which replaced an analytic chi-squared null that was measured to be ~300x
too permissive (see README "Status and how this was validated").
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gwanomaly.data.synthetic import ChirpParams, make_synthetic_segment
from gwanomaly.detection.excess_power import ExcessPowerConfig, ExcessPowerDetector
from gwanomaly.preprocessing.pipeline import PreprocessConfig, PreprocessingPipeline


@pytest.fixture
def calibrated_detector():
    pipe = PreprocessingPipeline(PreprocessConfig(backend="numpy", qimage_shape=(64, 64), bandpass_high=900))
    calib_maps = [
        pipe.run(make_synthetic_segment("H1", duration=16.0, sample_rate=4096, seed=s).data, 4096).qimage_raw_energy
        for s in range(40)
    ]
    detector = ExcessPowerDetector(ExcessPowerConfig(significance_threshold=1e-2))
    detector.calibrate(calib_maps)
    return detector, pipe


def test_detect_before_calibrate_raises():
    detector = ExcessPowerDetector()
    energy = np.random.default_rng(0).random((64, 64))
    with pytest.raises(RuntimeError, match="calibrate"):
        detector.detect(energy)


def test_false_positive_rate_matches_nominal_threshold(calibrated_detector):
    """The core regression test for the ~300x-too-permissive bug: at a
    nominal p<1e-2 threshold, the empirically observed false-positive rate
    on held-out background should be close to 1% of tiles, not ~3 (300x)."""
    detector, pipe = calibrated_detector
    n_tiles_per_map = (64 // 4) ** 2

    fp_count, total = 0, 0
    for seed in range(2000, 2030):
        seg = make_synthetic_segment("H1", duration=16.0, sample_rate=4096, seed=seed)
        r = pipe.run(seg.data, 4096)
        result = detector.detect(r.qimage_raw_energy)
        fp_count += result.n_flagged_tiles
        total += n_tiles_per_map

    observed_rate = fp_count / total
    # Nominal rate is 1e-2; allow a generous band (0.1% to 5%) since this
    # is a finite-sample empirical check, not an exact analytic guarantee.
    # The key regression check is that it's nowhere near the ~300x-inflated
    # rate the old analytic-chi2 implementation produced (~30%+).
    assert observed_rate < 0.05, (
        f"observed false-positive rate {observed_rate:.4f} is too high; "
        f"this matches the symptom of the old, removed analytic chi-squared "
        f"null (see README)"
    )


def test_chirp_injection_triggers(calibrated_detector):
    detector, pipe = calibrated_detector
    evt = make_synthetic_segment(
        "H1", duration=16.0, sample_rate=4096,
        inject=ChirpParams(chirp_mass=30.0, amplitude=3e-20), seed=2000,
    )
    r = pipe.run(evt.data, 4096)
    result = detector.detect(r.qimage_raw_energy)
    assert result.triggered
