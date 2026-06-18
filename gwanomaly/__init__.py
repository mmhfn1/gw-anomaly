"""
gwanomaly
=========

A modular pipeline for ingesting GWOSC gravitational-wave strain data,
preprocessing it, detecting anomalies (candidate GW signals or glitches),
and classifying detected events by source type with parameter estimation.

Stages
------
1. data           - GWOSC ingestion (gwosc + GWpy), catalogue access, dataset building
2. preprocessing  - whitening, bandpass, Q-transform/spectrogram, glitch vetoes
3. detection       - autoencoder anomaly detector, matched filter (PyCBC), excess power
4. classification  - CNN (Q-image) / LSTM (strain) source classifier, regression PE head
"""

__version__ = "0.1.0"
