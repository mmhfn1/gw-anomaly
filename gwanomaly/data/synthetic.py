"""
synthetic.py
=============

Synthetic strain generator used ONLY as a stand-in for live GWOSC fetches —
for unit tests, CI, and any environment without outbound network access to
gwosc.org (this sandbox included).

It is NOT a physically rigorous waveform model. It produces:
  - coloured Gaussian noise shaped roughly like the LIGO/Virgo design ASD
    (steep wall below ~20 Hz, 1/f-ish mid-band, rising again at high freq)
  - optional injected chirp signals (linear frequency-swept sinusoid with
    an amplitude envelope) standing in for a CBC inspiral-merger-ringdown,
    parameterised loosely by "chirp mass" so the classifier has something
    structured to learn from

Use `gwanomaly.data.gwosc_client.GWOSCClient` for real data. Swap this in
only when that client can't reach the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gwanomaly.data.gwosc_client import StrainSegment


def _design_asd_shape(freqs: np.ndarray) -> np.ndarray:
    """Rough proxy for the LIGO design-sensitivity ASD shape (not real)."""
    f = np.clip(freqs, 1.0, None)
    low_wall = (20.0 / f) ** 4
    seismic = (10.0 / f) ** 8
    mid = 1.0 + (f / 200.0) ** 2 * 0.3
    shot = (f / 1000.0) ** 1.5
    shape = np.sqrt(low_wall + seismic + mid + shot)
    return shape * 1e-23  # arbitrary strain-like scale


def colored_noise(duration: float, sample_rate: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate Gaussian noise coloured by a rough detector-ASD-like shape."""
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    white = rng.normal(size=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    shape = _design_asd_shape(freqs)
    white_f = np.fft.rfft(white)
    colored_f = white_f * shape * np.sqrt(sample_rate / 2.0)
    colored = np.fft.irfft(colored_f, n=n)
    return colored.astype(np.float32)


@dataclass
class ChirpParams:
    chirp_mass: float  # solar masses, controls sweep rate / duration
    f_start: float = 30.0
    f_end: float = 350.0
    merger_time_frac: float = 0.75  # where in the segment merger lands
    amplitude: float = 6e-21


def inject_chirp(
    background: np.ndarray,
    sample_rate: int,
    params: ChirpParams,
) -> np.ndarray:
    """
    Add a toy chirp (frequency-swept sinusoid with rising-then-decaying
    envelope) to a background noise array, standing in for a CBC signal.
    Frequency sweep rate is loosely tied to `chirp_mass` (lower mass =
    faster late-time sweep, mimicking real inspiral scaling).
    """
    n = len(background)
    t = np.arange(n) / sample_rate
    merger_t = params.merger_time_frac * (n / sample_rate)

    tau = merger_t - t
    tau = np.clip(tau, 1e-3, None)
    mass_scale = np.clip(params.chirp_mass / 30.0, 0.2, 5.0)
    freq = params.f_start + (params.f_end - params.f_start) * (
        1.0 - (tau / tau.max()) ** (1.0 / mass_scale)
    )
    freq = np.clip(freq, params.f_start, params.f_end * 1.2)

    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    envelope = np.exp(-((t - merger_t) ** 2) / (2 * (0.15 / mass_scale) ** 2))
    envelope = np.where(t < merger_t, envelope, envelope * np.exp(-(t - merger_t) * 25))

    chirp = params.amplitude * envelope * np.sin(phase)
    return (background + chirp).astype(np.float32)


def make_synthetic_segment(
    detector: str,
    duration: float = 32.0,
    sample_rate: int = 4096,
    inject: Optional[ChirpParams] = None,
    event_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> StrainSegment:
    """Build one synthetic StrainSegment, optionally with a chirp injected."""
    noise = colored_noise(duration, sample_rate, seed=seed)
    data = inject_chirp(noise, sample_rate, inject) if inject is not None else noise
    return StrainSegment(
        detector=detector,
        gps_start=1000000000.0,
        duration=duration,
        sample_rate=sample_rate,
        data=data,
        event_name=event_name,
    )
