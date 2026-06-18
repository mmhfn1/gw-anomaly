"""
dataset.py
===========

Turns raw `StrainSegment`s (from GWOSC or the synthetic generator) into
fixed-length, preprocessed windows ready for the autoencoder / classifier,
and wraps them in a PyTorch Dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from gwanomaly.data.catalogue import EventRecord
from gwanomaly.data.gwosc_client import GWOSCClient, StrainSegment
from gwanomaly.preprocessing.pipeline import PreprocessConfig, PreprocessingPipeline


LABEL_TO_IDX = {"BACKGROUND": 0, "BNS": 1, "NSBH": 2, "BBH": 3, "BURST": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


@dataclass
class PreprocessedWindow:
    qimage: np.ndarray          # (freq_bins, time_bins) Q-transform, log1p + z-score normalised
    qimage_raw_energy: np.ndarray  # (freq_bins, time_bins) unnormalised energy, for ExcessPowerDetector
    whitened_strain: np.ndarray  # 1D whitened/bandpassed timeseries
    label: str
    detector: str
    event_name: Optional[str]
    params: Optional[dict] = None  # chirp_mass / mass_1 / mass_2 / distance, if known


class StrainSegmentDataset:
    """
    Builds and stores `PreprocessedWindow`s from a list of `StrainSegment`s,
    applying the shared preprocessing pipeline so every window the
    detection/classification models see has been whitened, bandpassed, and
    Q-transformed the same way.
    """

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        self.pipeline = PreprocessingPipeline(preprocess_config or PreprocessConfig())
        self.windows: List[PreprocessedWindow] = []

    def add_segment(
        self,
        segment: StrainSegment,
        label: str,
        params: Optional[dict] = None,
    ) -> PreprocessedWindow:
        result = self.pipeline.run(segment.data, segment.sample_rate)
        window = PreprocessedWindow(
            qimage=result.qimage,
            qimage_raw_energy=result.qimage_raw_energy,
            whitened_strain=result.whitened,
            label=label,
            detector=segment.detector,
            event_name=segment.event_name,
            params=params,
        )
        self.windows.append(window)
        return window

    def to_arrays(self):
        """Stack into (X_qimage, X_strain, y, meta) arrays for training."""
        X_q = np.stack([w.qimage for w in self.windows])
        X_s = np.stack([w.whitened_strain for w in self.windows])
        y = np.array([LABEL_TO_IDX[w.label] for w in self.windows], dtype=np.int64)
        meta = [w.params for w in self.windows]
        return X_q, X_s, y, meta

    def __len__(self):
        return len(self.windows)


# ----------------------------------------------------------------------
# Convenience builders
# ----------------------------------------------------------------------

def build_event_dataset(
    client: GWOSCClient,
    events: Sequence[EventRecord],
    dataset: Optional[StrainSegmentDataset] = None,
    window_before: float = 16.0,
    window_after: float = 16.0,
) -> StrainSegmentDataset:
    """Fetch + preprocess a labelled window for each known event."""
    dataset = dataset or StrainSegmentDataset()
    for record in events:
        label = record.source_type or "BURST"
        segments = client.fetch_event_segments(
            record.name, window_before=window_before, window_after=window_after
        )
        for det, seg in segments.items():
            dataset.add_segment(
                seg,
                label=label,
                params={
                    "mass_1": record.mass_1,
                    "mass_2": record.mass_2,
                    "distance_mpc": record.distance_mpc,
                },
            )
    return dataset


def build_background_dataset(
    client: GWOSCClient,
    detector: str,
    quiet_ranges: Sequence[tuple],
    dataset: Optional[StrainSegmentDataset] = None,
    segment_duration: float = 32.0,
    max_segments: Optional[int] = None,
) -> StrainSegmentDataset:
    """Fetch + preprocess "BACKGROUND" (no event) windows for the autoencoder."""
    dataset = dataset or StrainSegmentDataset()
    segments = client.fetch_background_segments(
        detector, quiet_ranges, segment_duration=segment_duration
    )
    if max_segments:
        segments = segments[:max_segments]
    for seg in segments:
        dataset.add_segment(seg, label="BACKGROUND")
    return dataset


try:
    import torch
    from torch.utils.data import Dataset

    class TorchStrainDataset(Dataset):
        """PyTorch-facing view over a `StrainSegmentDataset`."""

        def __init__(self, dataset: StrainSegmentDataset, mode: str = "qimage"):
            assert mode in ("qimage", "strain", "both")
            self.dataset = dataset
            self.mode = mode

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            w = self.dataset.windows[idx]
            y = LABEL_TO_IDX[w.label]
            if self.mode == "qimage":
                x = torch.from_numpy(w.qimage).float().unsqueeze(0)  # (1, F, T)
            elif self.mode == "strain":
                x = torch.from_numpy(w.whitened_strain).float().unsqueeze(0)  # (1, N)
            else:
                x = (
                    torch.from_numpy(w.qimage).float().unsqueeze(0),
                    torch.from_numpy(w.whitened_strain).float().unsqueeze(0),
                )
            return x, y

except ImportError:  # torch optional at this layer
    TorchStrainDataset = None  # type: ignore
