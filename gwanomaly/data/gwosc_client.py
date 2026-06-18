"""
gwosc_client.py
================

Thin, well-documented wrapper around the `gwosc` and `gwpy` packages for
pulling open strain data directly from GWOSC (https://gwosc.org), with no
authentication required.

Two kinds of data come out of here:

1. Event-centred segments  - short (e.g. 32s) strain windows around a known
   confident GW event, for use as positive training examples.
2. Background segments     - strain windows that do NOT overlap any catalogued
   event, used as "clean" data to train the anomaly detector (autoencoder)
   and as negative examples for the classifier.

Network note
------------
This module makes real outbound calls to gwosc.org / gw-openscience.org.
It will not run inside network-restricted sandboxes (e.g. this one) — see
`gwanomaly.data.synthetic` for a drop-in synthetic substitute used for local
testing and the demo notebook.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DETECTORS = ("H1", "L1", "V1")
DEFAULT_SAMPLE_RATE = 4096  # Hz, standard GWOSC strain channel rate


@dataclass
class StrainSegment:
    """Container for a single-detector strain timeseries pulled from GWOSC."""

    detector: str
    gps_start: float
    duration: float
    sample_rate: int
    data: np.ndarray  # shape (duration * sample_rate,)
    event_name: Optional[str] = None  # None => background / non-event segment

    @property
    def gps_end(self) -> float:
        return self.gps_start + self.duration

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "gps_start": self.gps_start,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "event_name": self.event_name,
        }


class GWOSCClient:
    """
    Wrapper around `gwosc` (catalogue/segment queries) and `gwpy.timeseries`
    (actual strain fetch) for building training and inference datasets.

    Parameters
    ----------
    cache_dir : str or Path
        Local directory to cache downloaded .hdf5/.gwf strain files so
        repeated runs don't re-hit the GWOSC API.
    detectors : sequence of str
        Which detectors to pull by default, e.g. ("H1", "L1", "V1").
    sample_rate : int
        Target sample rate in Hz. GWOSC serves strain at 4096 Hz (full) or
        16384 Hz depending on era/dataset; this client resamples down to
        `sample_rate` if needed.
    """

    def __init__(
        self,
        cache_dir: str | Path = "./gwosc_cache",
        detectors: Sequence[str] = DEFAULT_DETECTORS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.detectors = tuple(detectors)
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Catalogue discovery
    # ------------------------------------------------------------------
    def list_events(self, catalog: str = "GWTC-1-confident") -> List[str]:
        """
        Return all event names in a given GWOSC catalogue.

        Common catalogue names: 'GWTC-1-confident', 'GWTC-2.1-confident',
        'GWTC-3-confident'. See https://gwosc.org/eventapi/html/ for the
        full list, which changes as new observing runs are released.
        """
        from gwosc import datasets

        events = datasets.find_datasets(type="events", catalog=catalog)
        logger.info("Found %d events in catalogue %s", len(events), catalog)
        return events

    def event_gps(self, event_name: str) -> float:
        """GPS time of merger for a named event, e.g. 'GW150914'."""
        from gwosc import datasets

        return datasets.event_gps(event_name)

    def event_detectors(self, event_name: str) -> List[str]:
        """Which detectors had usable data for this event."""
        from gwosc import datasets

        return datasets.event_detectors(event_name)

    # ------------------------------------------------------------------
    # Strain fetch
    # ------------------------------------------------------------------
    def fetch_segment(
        self,
        detector: str,
        gps_start: float,
        duration: float,
        event_name: Optional[str] = None,
        sample_rate: Optional[int] = None,
        retries: int = 5,
        backoff_s: float = 10.0,
    ) -> StrainSegment:
        """
        Fetch a single contiguous strain segment for one detector via GWpy.

        Uses `TimeSeries.fetch_open_data`, which talks to GWOSC directly
        (no auth, no API key). Resamples to `self.sample_rate` if the
        native rate differs.
        """
        from gwpy.timeseries import TimeSeries

        sample_rate = sample_rate or self.sample_rate
        gps_end = gps_start + duration

        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                ts = TimeSeries.fetch_open_data(
                    detector, gps_start, gps_end, cache=True, verbose=False
                )
                break
            except Exception as exc:  # network / 404 / no-data-at-time
                last_exc = exc
                logger.warning(
                    "Fetch attempt %d/%d failed for %s [%s, %s]: %s",
                    attempt, retries, detector, gps_start, gps_end, exc,
                )
                time.sleep(backoff_s * attempt)
        else:
            raise RuntimeError(
                f"Failed to fetch {detector} segment after {retries} attempts"
            ) from last_exc

        if int(ts.sample_rate.value) != sample_rate:
            ts = ts.resample(sample_rate)

        return StrainSegment(
            detector=detector,
            gps_start=gps_start,
            duration=duration,
            sample_rate=sample_rate,
            data=ts.value.astype(np.float32),
            event_name=event_name,
        )

    def fetch_event_segments(
        self,
        event_name: str,
        window_before: float = 16.0,
        window_after: float = 16.0,
        detectors: Optional[Sequence[str]] = None,
    ) -> dict:
        """
        Fetch a strain segment around a named event's merger time, for each
        detector that observed it.

        Returns
        -------
        dict mapping detector -> StrainSegment
        """
        gps = self.event_gps(event_name)
        available = set(self.event_detectors(event_name))
        targets = [d for d in (detectors or self.detectors) if d in available]

        out = {}
        for det in targets:
            seg = self.fetch_segment(
                detector=det,
                gps_start=gps - window_before,
                duration=window_before + window_after,
                event_name=event_name,
            )
            out[det] = seg
        return out

    def fetch_background_segments(
        self,
        detector: str,
        gps_ranges: Iterable[tuple[float, float]],
        segment_duration: float = 32.0,
        stride: Optional[float] = None,
    ) -> List[StrainSegment]:
        """
        Slice a set of known-quiet GPS time ranges into fixed-length
        background segments (no catalogued event inside).

        `gps_ranges` should come from `find_quiet_gps_ranges` in
        `gwanomaly.data.catalogue`, which checks candidate windows against
        the event catalogue before returning them.
        """
        stride = stride or segment_duration
        segments = []
        for range_start, range_end in gps_ranges:
            t = range_start
            while t + segment_duration <= range_end:
                try:
                    seg = self.fetch_segment(
                        detector=detector,
                        gps_start=t,
                        duration=segment_duration,
                        event_name=None,
                    )
                    segments.append(seg)
                except Exception as exc:
                    logger.warning("Skipping background segment at %s: %s", t, exc)
                t += stride
        return segments
