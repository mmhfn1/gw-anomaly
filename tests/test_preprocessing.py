"""
test_preprocessing.py
=======================

Regression tests for bugs found and fixed while building this pipeline:
float32 underflow at real strain scale, negative-energy NaN propagation,
and Nyquist-boundary config validation. Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gwanomaly.data.synthetic import ChirpParams, make_synthetic_segment
from gwanomaly.preprocessing.pipeline import PreprocessConfig, PreprocessingPipeline


@pytest.mark.parametrize("backend", ["numpy", "gwpy"])
def test_no_nan_at_real_strain_scale(backend):
    """Regression test for the float32-underflow bug: real strain
    amplitudes (~1e-19) must not produce NaN/Inf anywhere downstream."""
    seg = make_synthetic_segment("H1", duration=8.0, sample_rate=2048, seed=1)
    pipe = PreprocessingPipeline(PreprocessConfig(backend=backend, qimage_shape=(32, 32), bandpass_high=550))
    result = pipe.run(seg.data, 2048)

    assert not np.isnan(result.whitened).any()
    assert not np.isnan(result.qimage).any()
    assert not np.isnan(result.qimage_raw_energy).any()
    assert not np.isinf(result.qimage).any()


@pytest.mark.parametrize("backend", ["numpy", "gwpy"])
def test_raw_energy_is_nonnegative(backend):
    """Regression test for the negative-energy bug: GWpy's q_transform()
    can emit small negatives at the noise floor, which must be clipped
    before being exposed via qimage_raw_energy (ExcessPowerDetector and
    log1p normalisation both assume non-negative energy)."""
    seg = make_synthetic_segment("H1", duration=8.0, sample_rate=2048, seed=2)
    pipe = PreprocessingPipeline(PreprocessConfig(backend=backend, qimage_shape=(32, 32), bandpass_high=550))
    result = pipe.run(seg.data, 2048)

    assert (result.qimage_raw_energy >= 0).all()


def test_bandpass_too_close_to_nyquist_raises():
    """Regression test: bandpass_high too close to Nyquist must raise a
    clear ValueError up front, rather than failing deep inside scipy's
    filter design with a cryptic message."""
    seg = make_synthetic_segment("H1", duration=8.0, sample_rate=2048, seed=3)
    pipe = PreprocessingPipeline(PreprocessConfig(backend="numpy", bandpass_high=1024.0))
    with pytest.raises(ValueError, match="Nyquist"):
        pipe.run(seg.data, 2048)


def test_whiten_fftlength_too_long_raises():
    seg = make_synthetic_segment("H1", duration=2.0, sample_rate=2048, seed=4)
    pipe = PreprocessingPipeline(
        PreprocessConfig(backend="numpy", bandpass_high=550, whiten_fftlength=4.0)
    )
    with pytest.raises(ValueError, match="fftlength"):
        pipe.run(seg.data, 2048)


def test_chirp_injection_increases_qimage_energy():
    """Sanity check that the synthetic chirp injection actually produces
    detectable excess energy relative to background, at a reasonably
    boosted amplitude (this is the precondition for any detector test
    downstream to be meaningful)."""
    pipe = PreprocessingPipeline(PreprocessConfig(backend="numpy", qimage_shape=(64, 64), bandpass_high=900))

    bg = make_synthetic_segment("H1", duration=16.0, sample_rate=4096, seed=5)
    evt = make_synthetic_segment(
        "H1", duration=16.0, sample_rate=4096,
        inject=ChirpParams(chirp_mass=30.0, amplitude=3e-20), seed=5,
    )

    r_bg = pipe.run(bg.data, 4096)
    r_evt = pipe.run(evt.data, 4096)

    assert r_evt.qimage_raw_energy.max() > r_bg.qimage_raw_energy.max()
