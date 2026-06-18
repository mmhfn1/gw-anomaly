"""
matched_filter.py
===================

Matched-filter detection against a bank of CBC (compact binary coalescence)
template waveforms, the standard model-based approach LIGO/Virgo pipelines
(PyCBC, GstLAL) use for known signal morphologies (BBH/BNS/NSBH inspiral-
merger-ringdown).

This wraps PyCBC's waveform generation + matched filter SNR time series.
PyCBC is a heavier dependency (lalsimulation under the hood) and is the
right tool here rather than reimplementing matched filtering from scratch,
since template generation and the SNR normalisation conventions are easy
to get subtly wrong.

If PyCBC isn't installed, `MatchedFilterDetector.available` is False and
calling `detect()` raises a clear ImportError rather than silently no-op-ing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import pycbc
    from pycbc.waveform import get_td_waveform
    from pycbc.filter import matched_filter, sigma
    from pycbc.psd import welch as pycbc_welch, interpolate as pycbc_interpolate
    from pycbc.types import TimeSeries as PyCBCTimeSeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False


@dataclass
class TemplateBankSpec:
    """Simple grid spec for a CBC template bank; for production use, prefer
    PyCBC's `pycbc_geom_nonspinbank` / `pycbc_brute_bank` for a properly
    placed (minimal-mismatch) bank rather than a naive grid."""
    mass1_range: Tuple[float, float] = (1.0, 50.0)
    mass2_range: Tuple[float, float] = (1.0, 50.0)
    n_mass1: int = 10
    n_mass2: int = 10
    approximant: str = "IMRPhenomD"
    f_lower: float = 20.0


@dataclass
class MatchedFilterConfig:
    sample_rate: int = 4096
    snr_threshold: float = 8.0  # standard single-detector SNR trigger threshold
    template_bank: TemplateBankSpec = field(default_factory=TemplateBankSpec)
    psd_segment_seconds: float = 4.0


class MatchedFilterDetector:
    """
    Builds a (naive grid) template bank and matched-filters a strain
    segment against every template, returning the peak SNR and best-fit
    (mass1, mass2) per detector segment.

    For real analyses you'd want a geometrically placed bank (far fewer
    templates for the same coverage) and FAR estimation via time-slides
    against background — both are noted as extension points below.
    """

    def __init__(self, config: Optional[MatchedFilterConfig] = None):
        self.config = config or MatchedFilterConfig()
        self._templates: Optional[List[Tuple[float, float, "PyCBCTimeSeries"]]] = None

    @property
    def available(self) -> bool:
        return PYCBC_AVAILABLE

    def _require_pycbc(self):
        if not PYCBC_AVAILABLE:
            raise ImportError(
                "PyCBC is required for MatchedFilterDetector. Install with "
                "`pip install pycbc` (heavier dependency incl. lalsimulation; "
                "best done in a dedicated conda/venv, see PyCBC install docs)."
            )

    def build_template_bank(self) -> None:
        """Generate time-domain waveforms for a naive (mass1, mass2) grid.
        Real banks should be built with proper metric-based placement to
        avoid wasting templates on near-degenerate mass combinations."""
        self._require_pycbc()
        spec = self.config.template_bank
        m1_grid = np.linspace(*spec.mass1_range, spec.n_mass1)
        m2_grid = np.linspace(*spec.mass2_range, spec.n_mass2)

        templates = []
        for m1 in m1_grid:
            for m2 in m2_grid:
                if m2 > m1:
                    continue  # avoid duplicate (m1,m2)/(m2,m1) pairs
                try:
                    hp, _ = get_td_waveform(
                        approximant=spec.approximant,
                        mass1=float(m1),
                        mass2=float(m2),
                        delta_t=1.0 / self.config.sample_rate,
                        f_lower=spec.f_lower,
                    )
                    templates.append((float(m1), float(m2), hp))
                except Exception:
                    continue  # some mass combos may fail waveform generation
        self._templates = templates

    def detect(
        self,
        strain: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """
        Matched-filter a strain array against the template bank.

        Returns
        -------
        dict with keys: peak_snr, peak_time_s, best_mass1, best_mass2,
        is_trigger (peak_snr > snr_threshold)
        """
        self._require_pycbc()
        if self._templates is None:
            self.build_template_bank()

        ts = PyCBCTimeSeries(strain.astype(np.float64), delta_t=1.0 / sample_rate)

        # Estimate PSD from the segment itself (Welch); for production,
        # estimate PSD from a longer off-source stretch instead, so the
        # PSD isn't contaminated by the signal you're trying to detect.
        psd = pycbc_welch(ts, seg_len=int(self.config.psd_segment_seconds * sample_rate))
        psd = pycbc_interpolate(psd, ts.delta_f)

        best = {"peak_snr": 0.0, "peak_time_s": None, "best_mass1": None, "best_mass2": None}

        for m1, m2, template in self._templates:
            template.resize(len(ts))
            try:
                snr = matched_filter(template, ts, psd=psd, low_frequency_cutoff=self.config.template_bank.f_lower)
                snr = snr.crop(4, 4)  # drop edge artefacts from filter settle-in
            except Exception:
                continue

            peak_idx = np.argmax(np.abs(snr.numpy()))
            peak_val = float(np.abs(snr.numpy())[peak_idx])

            if peak_val > best["peak_snr"]:
                best.update(
                    peak_snr=peak_val,
                    peak_time_s=float(snr.sample_times[peak_idx]),
                    best_mass1=m1,
                    best_mass2=m2,
                )

        best["is_trigger"] = best["peak_snr"] > self.config.snr_threshold
        return best
