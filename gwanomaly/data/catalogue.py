"""
catalogue.py
=============

Helpers built on top of `gwosc.datasets` for:
  - pulling structured metadata for catalogued events (GWTC-1/2/3) to use
    as classification labels and parameter-estimation targets
  - finding GPS time ranges that do NOT overlap any catalogued event, so
    they're safe to use as "background" (anomaly-free) training data

This is the layer that turns the raw event list into something you can
build a labelled dataset from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Rough merger-source labels for well-known GWTC events, since GWOSC's API
# gives masses/distances but not a categorical label directly. Boundary
# convention follows LIGO/Virgo population papers: NS upper mass ~3 Msun.
NS_MASS_CUTOFF = 3.0


@dataclass
class EventRecord:
    name: str
    gps: float
    detectors: List[str]
    mass_1: Optional[float] = None
    mass_2: Optional[float] = None
    distance_mpc: Optional[float] = None
    source_type: Optional[str] = None  # BBH / BNS / NSBH, derived from masses


def classify_by_mass(mass_1: float, mass_2: float) -> str:
    """Heuristic source-type label from component masses (solar masses)."""
    m1_is_ns = mass_1 < NS_MASS_CUTOFF
    m2_is_ns = mass_2 < NS_MASS_CUTOFF
    if m1_is_ns and m2_is_ns:
        return "BNS"
    if m1_is_ns or m2_is_ns:
        return "NSBH"
    return "BBH"


class CatalogueBuilder:
    """
    Builds a list of `EventRecord`s from one or more GWOSC catalogues, with
    derived source-type labels, and finds quiet GPS ranges for background
    sampling.
    """

    def __init__(self, catalogs: Tuple[str, ...] = ("GWTC-1-confident", "GWTC-2.1-confident", "GWTC-3-confident")):
        self.catalogs = catalogs
        self._events: List[EventRecord] = []

    def build(self) -> List[EventRecord]:
        from gwosc import datasets

        records = []
        for catalog in self.catalogs:
            try:
                names = datasets.find_datasets(type="events", catalog=catalog)
            except Exception as exc:
                logger.warning("Could not list catalogue %s: %s", catalog, exc)
                continue

            for name in names:
                try:
                    gps = datasets.event_gps(name)
                    dets = datasets.event_detectors(name)
                    meta = self._fetch_event_parameters(name)
                    source_type = None
                    if meta.get("mass_1") is not None and meta.get("mass_2") is not None:
                        source_type = classify_by_mass(meta["mass_1"], meta["mass_2"])

                    records.append(
                        EventRecord(
                            name=name,
                            gps=gps,
                            detectors=dets,
                            mass_1=meta.get("mass_1"),
                            mass_2=meta.get("mass_2"),
                            distance_mpc=meta.get("distance_mpc"),
                            source_type=source_type,
                        )
                    )
                except Exception as exc:
                    logger.warning("Skipping event %s: %s", name, exc)

        self._events = records
        return records

    @staticmethod
    def _fetch_event_parameters(event_name: str) -> dict:
        """
        Pull point-estimate source parameters for an event from the GWOSC
        event API (median values from the published PE samples), used both
        as classification labels and as regression targets for the
        parameter-estimation head.
        """
        from gwosc.api import fetch_event_json

        try:
            payload = fetch_event_json(event_name)
        except Exception:
            return {}

        # The JSON has one entry per published "version" of the event;
        # take the most recent / highest-versioned one.
        versions = payload.get("events", {})
        if not versions:
            return {}
        latest = list(versions.values())[-1]
        params = latest.get("parameters", {}) or {}
        # Field names follow GWOSC's event-parameter schema.
        best = {}
        for key in params:
            best = params[key]
            break  # first parameter set is typically the recommended one

        return {
            "mass_1": best.get("mass_1_source"),
            "mass_2": best.get("mass_2_source"),
            "distance_mpc": best.get("luminosity_distance"),
        }

    def find_quiet_gps_ranges(
        self,
        search_start: float,
        search_end: float,
        buffer_s: float = 600.0,
    ) -> List[Tuple[float, float]]:
        """
        Given a GPS search window, return sub-ranges that exclude a buffer
        around every catalogued event's merger time. Used to source
        background segments for the autoencoder and negative classifier
        examples.

        Parameters
        ----------
        buffer_s : float
            Seconds excluded on either side of each event GPS time.
        """
        if not self._events:
            self.build()

        event_times = sorted(
            e.gps for e in self._events if search_start <= e.gps <= search_end
        )

        ranges = []
        cursor = search_start
        for t in event_times:
            excl_start, excl_end = t - buffer_s, t + buffer_s
            if cursor < excl_start:
                ranges.append((cursor, excl_start))
            cursor = max(cursor, excl_end)
        if cursor < search_end:
            ranges.append((cursor, search_end))

        return ranges
