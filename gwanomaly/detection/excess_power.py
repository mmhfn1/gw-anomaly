"""
excess_power.py
=================

Excess-power (burst) detection: a model-independent method that looks for
short-duration, time-frequency-localised excess energy in the whitened
strain without assuming any particular waveform morphology — complementary
to matched filtering (which assumes a CBC template) and useful for
catching unmodelled transients (e.g. supernovae, cosmic strings, or
"we don't know what this is yet" signals).

This implements a simplified single-detector version of the excess-power
idea used by burst pipelines like cWB / Omicron: tile the Q-transform
energy map into time-frequency cells, sum energy per tile, and flag tiles
whose energy is statistically inconsistent with background noise.

Why empirical calibration instead of an analytic chi-squared null
-------------------------------------------------------------------
An earlier version of this detector assumed each tile's energy sum follows
a chi-squared distribution with degrees-of-freedom = 2 * n_pixels (treating
each Q-transform pixel as an independent sum-of-two-Gaussians). That
assumption breaks down badly in practice: adjacent frequency rows in a
constant-Q transform are highly correlated (empirically ~0.88 lag-1
correlation in this implementation, since neighbouring log-spaced
frequency bins use near-overlapping wavelets), so a 4x4 tile has far fewer
effective independent degrees of freedom than its analytic pixel count.
Using the analytic chi-squared null produced a measured false-positive
rate roughly 300x higher than the nominal threshold implied (~80 flagged
tiles across 2560 background tiles at a nominal p<1e-4, vs an expected
~0.26) — confirmed by direct simulation. Real burst pipelines (cWB, etc.)
sidestep exactly this problem with empirical background estimation (e.g.
time-slides), and this module does the same: `calibrate()` builds an
empirical null distribution of tile-energy sums from background-only data,
and `detect()` reports p-values against that empirical distribution
rather than an ill-fitting analytic one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ExcessPowerConfig:
    tile_freq_bins: int = 4     # how many Q-image frequency rows per tile
    tile_time_bins: int = 4     # how many Q-image time columns per tile
    significance_threshold: float = 1e-3  # empirical-p-value threshold for a tile to be flagged
    min_tiles_for_trigger: int = 1


@dataclass
class ExcessPowerResult:
    triggered: bool
    n_flagged_tiles: int
    most_significant_pvalue: float
    flagged_tile_indices: List[Tuple[int, int]]  # (freq_tile_idx, time_tile_idx)


class ExcessPowerDetector:
    """
    Operates directly on the *unnormalised* Q-transform energy map, i.e.
    `PreprocessResult.qimage_raw_energy` from `PreprocessingPipeline.run()`
    — NOT `PreprocessResult.qimage`, which has log1p + z-score normalisation
    applied and is no longer a non-negative physical energy.

    Must be calibrated on background-only energy maps before use:

    >>> detector = ExcessPowerDetector(ExcessPowerConfig())
    >>> detector.calibrate(list_of_background_raw_energy_maps)
    >>> result = detector.detect(candidate_raw_energy_map)
    """

    def __init__(self, config: ExcessPowerConfig = None):
        self.config = config or ExcessPowerConfig()
        self._background_tile_sums: Optional[np.ndarray] = None  # sorted, for empirical CDF

    def calibrate(self, background_energy_maps: List[np.ndarray]) -> None:
        """
        Build an empirical null distribution of tile-energy sums from a
        set of background-only (no-event) raw energy maps. The more
        background maps supplied, the finer the smallest achievable
        p-value (with N background tiles total, the smallest non-zero
        empirical p-value is ~1/N) — for a `significance_threshold` of
        1e-3 you want at least several thousand background tiles, i.e.
        several dozen background segments at typical qimage_shape/tile
        settings.
        """
        all_sums = [self._tile_sums(m) for m in background_energy_maps]
        self._background_tile_sums = np.sort(np.concatenate(all_sums))

    def _tile_sums(self, energy_map: np.ndarray) -> np.ndarray:
        cfg = self.config
        F, T = energy_map.shape
        nf, nt = cfg.tile_freq_bins, cfg.tile_time_bins
        sums = []
        for fi in range(0, F - nf + 1, nf):
            for ti in range(0, T - nt + 1, nt):
                sums.append(energy_map[fi:fi + nf, ti:ti + nt].sum())
        return np.array(sums)

    def _empirical_pvalue(self, value: float) -> float:
        """P(background tile sum >= value), via empirical CDF with a
        Laplace-style correction so a value at/above the max background
        sample still gets a finite, non-zero p-value (1/(N+1)) instead of
        a hard zero, which would otherwise make `most_significant_pvalue`
        uninformative for ranking very loud outliers against each other."""
        bg = self._background_tile_sums
        n = len(bg)
        n_geq = n - np.searchsorted(bg, value, side="left")
        return (n_geq + 1) / (n + 1)

    def detect(self, energy_map: np.ndarray) -> ExcessPowerResult:
        """
        Parameters
        ----------
        energy_map : (F, T) array of non-negative Q-transform energy
            (i.e. |q-coefficient|^2, NOT log-scaled/z-scored).
        """
        if self._background_tile_sums is None:
            raise RuntimeError(
                "ExcessPowerDetector.calibrate() must be called with "
                "background-only energy maps before detect()."
            )

        cfg = self.config
        F, T = energy_map.shape
        nf, nt = cfg.tile_freq_bins, cfg.tile_time_bins

        flagged = []
        pvalues = []

        for fi in range(0, F - nf + 1, nf):
            for ti in range(0, T - nt + 1, nt):
                tile_sum = energy_map[fi:fi + nf, ti:ti + nt].sum()
                p = self._empirical_pvalue(tile_sum)
                pvalues.append(p)
                if p < cfg.significance_threshold:
                    flagged.append((fi // nf, ti // nt))

        most_sig = float(min(pvalues)) if pvalues else 1.0
        triggered = len(flagged) >= cfg.min_tiles_for_trigger

        return ExcessPowerResult(
            triggered=triggered,
            n_flagged_tiles=len(flagged),
            most_significant_pvalue=most_sig,
            flagged_tile_indices=flagged,
        )
