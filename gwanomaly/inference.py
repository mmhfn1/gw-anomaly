"""
inference.py
==============

End-to-end inference: given a raw strain segment, run it through
preprocessing, both detection methods (autoencoder anomaly score +
matched-filter/excess-power triggers), and — if flagged — classification
and parameter estimation. This is the "production" code path each script
exercises a piece of; this module is what you'd actually call from a
real-time or batch analysis driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from gwanomaly.classification.cnn_classifier import CNNClassifierTrainer
from gwanomaly.data.dataset import IDX_TO_LABEL
from gwanomaly.detection.autoencoder import AutoencoderDetector
from gwanomaly.detection.excess_power import ExcessPowerDetector
from gwanomaly.preprocessing.pipeline import PreprocessingPipeline


@dataclass
class CandidateResult:
    is_anomaly: bool
    anomaly_score: float
    anomaly_threshold: float
    excess_power_triggered: bool
    excess_power_min_pvalue: float
    predicted_label: Optional[str] = None
    label_probabilities: Optional[Dict[str, float]] = None


class InferencePipeline:
    """
    Wires together PreprocessingPipeline -> AutoencoderDetector +
    ExcessPowerDetector -> CNNClassifierTrainer, mirroring the four-stage
    architecture (Ingest -> Preprocess -> Detect -> Classify) end to end.

    Detection runs unconditionally; classification only runs on windows
    where at least one detector flags something, since the classifier was
    trained to discriminate event TYPES (BNS/NSBH/BBH/BURST) and is not
    necessarily a reliable background-vs-not detector by itself — the
    detection stage is what's calibrated for that distinction.
    """

    def __init__(
        self,
        preprocessing: PreprocessingPipeline,
        autoencoder: AutoencoderDetector,
        excess_power: ExcessPowerDetector,
        classifier: Optional[CNNClassifierTrainer] = None,
    ):
        self.preprocessing = preprocessing
        self.autoencoder = autoencoder
        self.excess_power = excess_power
        self.classifier = classifier

    def run(self, strain: np.ndarray, sample_rate: int) -> CandidateResult:
        result = self.preprocessing.run(strain, sample_rate)

        ae_score = float(self.autoencoder.score(result.qimage[None])[0])
        is_anomaly = ae_score > self.autoencoder.threshold

        ep_result = self.excess_power.detect(result.qimage_raw_energy)

        flagged = is_anomaly or ep_result.triggered
        predicted_label, probs = None, None

        if flagged and self.classifier is not None:
            proba = self.classifier.predict_proba(result.qimage[None])[0]
            predicted_label = IDX_TO_LABEL[int(proba.argmax())]
            probs = {IDX_TO_LABEL[i]: float(p) for i, p in enumerate(proba)}

        return CandidateResult(
            is_anomaly=is_anomaly,
            anomaly_score=ae_score,
            anomaly_threshold=float(self.autoencoder.threshold),
            excess_power_triggered=ep_result.triggered,
            excess_power_min_pvalue=ep_result.most_significant_pvalue,
            predicted_label=predicted_label,
            label_probabilities=probs,
        )
