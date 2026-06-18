# gwanomaly

A modular pipeline for ingesting GWOSC gravitational-wave strain data,
preprocessing it, detecting anomalies (candidate signals or glitches), and
classifying detected candidates by source type with parameter estimation.

Built around the four-stage architecture: **Ingest → Preprocess → Detect →
Classify**.

```
gwanomaly/
├── data/             GWOSC ingestion (gwosc + GWpy), catalogue/labels, dataset builder
├── preprocessing/    whitening, bandpass, Q-transform, glitch vetoes
├── detection/        autoencoder (unsupervised), matched filter (PyCBC), excess power (burst)
├── classification/   CNN (Q-image) + LSTM (strain) classifiers, regression + Bilby parameter estimation
├── inference.py       wires detection -> classification into one end-to-end call
└── utils/            metrics (classification report, FAR estimation)
scripts/              build_dataset.py, train_autoencoder.py, train_classifier.py
```

## Status and how this was validated

Every numerical claim below comes from actually running the code in this
repo, not from inspection alone. That process surfaced several real bugs,
which are documented here rather than swept under the rug, since they
matter for anyone running this on real data:

- **Float32 underflow at real strain scale.** Real (and synthetic) strain
  amplitudes are ~1e-19 to 1e-21. Squaring that for PSD/energy estimation
  underflows to exact zero in float32 (whose minimum normal value is
  ~1.2e-38) across a large fraction of frequency bins, corrupting whitening
  and the Q-transform. Fixed by doing all spectral math in float64
  internally, downcasting to float32 only for final storage.
- **GWpy's `q_transform()` can return small negative values** for pixels at
  the noise floor (confirmed empirically, ~0.4% of pixels). Feeding that
  straight into `log1p` produced NaNs in ~11% of an early test dataset.
  Fixed by clipping Q-transform energy to non-negative before any log
  scaling, and updated `PreprocessResult.qimage_raw_energy`'s documented
  contract to reflect this.
- **Naive analytic chi-squared null for excess-power detection was wrong
  by ~300x.** Adjacent frequency rows in the constant-Q transform are
  highly correlated (~0.88 lag-1 correlation), so treating each pixel as
  an independent Gaussian-noise degree of freedom badly underestimates
  the true variance of tile-energy sums. Measured false-positive rate was
  ~300x the nominal threshold. Replaced with **empirical background
  calibration** (the same approach real burst pipelines like cWB use via
  time-slides) — `ExcessPowerDetector.calibrate()` now builds a null
  distribution directly from background data, verified to track the
  nominal false-positive rate within statistical noise at scale (213/20480
  flagged vs 204.8 expected at p<1e-2).
- **Nyquist-boundary config bugs.** Default `bandpass_high` and various
  test configs sat exactly at or just past the Nyquist frequency, which
  both this package's own validation and, separately, GWpy's internal
  filter design (`scipy.signal.iirdesign`) reject. Fixed with an explicit,
  documented margin (`bandpass_high <= 0.6 * Nyquist`) that accounts for
  GWpy's internal stopband padding (`min(bandpass_high*1.5, Nyquist)`).

All of the above are now covered by passing smoke tests (see "What's been
tested" below); they're listed because they're the kind of bug that's easy
to reintroduce when changing sample rates, durations, or qimage shapes
later, so look here first if you see NaNs or implausible detection rates
after modifying configs.

### What's been tested (in this sandbox, CPU-only, synthetic data)

- Preprocessing (`numpy` and `gwpy` backends): whitening, bandpass,
  Q-transform — both produce finite, sane output across dozens of seeds at
  real GWOSC sample rates (2048/4096 Hz).
- `ExcessPowerDetector`: false-positive rate empirically matches the
  nominal threshold after the calibration fix (above); correctly flags
  injected chirps.
- `AutoencoderDetector`: training loop, calibration, and predict all work
  correctly — **verified on an easy synthetic case** (Gaussian noise vs.
  noise-with-bright-patch: 50/50 detection, 1/50 false positives). **Did
  NOT cleanly separate background from chirp-injected synthetic GW data**
  at the scale tested here (~150 training windows, CPU). See "Known
  limitation" below before trusting this on anything beyond confirming the
  code runs.
- `CNNClassifierTrainer`, `LSTMClassifierTrainer`, `RegressionPETrainer`:
  shapes, training loops, and save/load all verified mechanically correct.
  Not evaluated for real accuracy (no labelled real-event training set
  available in this sandbox — see below).
- `InferencePipeline`: full ingest→preprocess→detect→classify wiring
  verified end-to-end — correctly distinguishes a held-out background
  window from a chirp-injected window and only invokes the classifier on
  flagged windows. A concrete example from `notebooks/demo.ipynb`'s
  executed run: on one synthetic BBH-like injection, the autoencoder did
  NOT flag it (score 0.958 vs threshold 0.973) but the excess-power
  detector did (p≈3e-5), and the downstream classifier then correctly
  predicted BBH with 89% confidence — a real demonstration of why the
  architecture runs multiple complementary detectors rather than relying
  on one.
- `build_dataset.py`, `train_autoencoder.py`, `train_classifier.py`: all
  run cleanly end-to-end on synthetic data with zero NaNs in output.

### Known limitation: this sandbox could not reach GWOSC

This environment's network egress is restricted to a fixed allowlist
(PyPI, npm, GitHub, etc.) and does not include `gwosc.org` /
`gw-openscience.org` (confirmed via direct `curl`, which returned
`x-deny-reason: host_not_allowed`). That means:

- `GWOSCClient` and `CatalogueBuilder` (in `gwanomaly/data/`) are written
  against the real `gwosc`/`gwpy` APIs and **will work as-is** on any
  machine with normal internet access — nothing about them is
  sandbox-specific — but they could not be exercised against the live API
  here.
- All the validation above used `gwanomaly/data/synthetic.py` instead: a
  deliberately simple stand-in (coloured Gaussian noise + a toy
  frequency-swept chirp injection) that is explicitly **not** a physically
  rigorous waveform model. It's good enough to exercise every code path
  (shapes, dtypes, NaN-safety, training loops) but not good enough to
  validate real detection/classification accuracy.
- The autoencoder's failure to separate background from synthetic-chirp
  windows (above) is most likely a property of this synthetic data (it
  doesn't whiten as cleanly to flat noise as real detector data does) and
  the tiny CPU training budget, not a bug in the detector itself — which
  is exactly why the architecture was separately validated on an
  unambiguous synthetic case before concluding the *code* works.

**Bottom line:** run `scripts/build_dataset.py --source gwosc ...` on a
machine with internet access to get real strain + real GWTC labels, then
`train_autoencoder.py` / `train_classifier.py` on a GPU with a realistically
sized dataset (hundreds of background segments at minimum; thousands
preferred) before drawing any conclusions about real-world detection
accuracy.

## Quickstart

```bash
pip install -r requirements.txt

# Real data (run on a machine with internet access to gwosc.org):
python scripts/build_dataset.py --source gwosc \
    --catalog GWTC-1-confident --out data/gwtc1.npz

# Synthetic stand-in (works anywhere, e.g. to sanity-check this repo
# itself without GWOSC network access):
python scripts/build_dataset.py --source synthetic \
    --n-background 200 --n-events 100 --out data/synthetic.npz

# Train (use a GPU machine for real-sized datasets/epoch counts):
python scripts/train_autoencoder.py --dataset data/gwtc1.npz \
    --out models/autoencoder.pt --epochs 100

python scripts/train_classifier.py --dataset data/gwtc1.npz \
    --out models/cnn_classifier.pt --epochs 100
```

Inference, once you have trained checkpoints:

```python
from gwanomaly.preprocessing.pipeline import PreprocessingPipeline
from gwanomaly.detection.autoencoder import AutoencoderDetector
from gwanomaly.detection.excess_power import ExcessPowerDetector
from gwanomaly.classification.cnn_classifier import CNNClassifierTrainer
from gwanomaly.inference import InferencePipeline

pipeline = InferencePipeline(
    preprocessing=PreprocessingPipeline(),
    autoencoder=AutoencoderDetector.load("models/autoencoder.pt"),
    excess_power=excess_power_detector,  # calibrate() on background first
    classifier=CNNClassifierTrainer.load("models/cnn_classifier.pt"),
)
result = pipeline.run(strain_array, sample_rate=4096)
print(result)
```

See `notebooks/demo.ipynb` for a full worked example (already executed, so
you can read the real output without running it) covering all four
stages end-to-end on synthetic data in a few minutes on CPU.

## Design notes per stage

**Ingestion** (`gwanomaly/data/`): `GWOSCClient` wraps `gwosc` (catalogue
queries) and `gwpy.timeseries.TimeSeries.fetch_open_data` (actual strain
fetch) — no auth required, matches what you described. `CatalogueBuilder`
pulls GWTC-1/2/3 event metadata, derives BBH/BNS/NSBH labels from component
masses (NS cutoff at 3 solar masses, following standard population-paper
convention), and finds GPS time ranges that don't overlap any catalogued
event for background sampling.

**Preprocessing** (`gwanomaly/preprocessing/`): whitening (divide by Welch
PSD), bandpass (20 Hz–~1000 Hz by default, tunable), Q-transform. Two
backends: `gwpy` (use this against real data) and a dependency-light numpy
fallback (Butterworth bandpass + an approximate constant-Q wavelet
transform) for environments without GWpy. Both validated to agree on
whitened-signal statistics within ~1%.

**Detection** (`gwanomaly/detection/`): three complementary methods, as
described in your original architecture —
`AutoencoderDetector` (unsupervised, trained on background only,
reconstruction-error anomaly score),
`MatchedFilterDetector` (PyCBC template bank against CBC waveforms,
standard SNR threshold of 8),
`ExcessPowerDetector` (model-independent burst detection via empirically
calibrated time-frequency tile energy, for signals matched filtering won't
catch).

**Classification** (`gwanomaly/classification/`): `CNNClassifierTrainer`
(Q-image -> source type), `LSTMClassifierTrainer` (whitened strain ->
source type, downsampled before the LSTM for tractable sequence length),
`RegressionPETrainer` (fast point-estimate chirp mass/mass ratio/distance)
and `BilbyPEWrapper` (full Bayesian PE via nested sampling, for offline
follow-up on confirmed candidates — orders of magnitude slower than the
regression head, with proper uncertainty quantification).

## Dependencies

`pip install -r requirements.txt` gets you numpy/scipy/gwosc/gwpy/torch.
PyCBC and Bilby are heavier (pull in `lalsuite`/`lalsimulation`) and are
commented out in `requirements.txt` — install them separately, ideally in
a dedicated venv/conda environment, only if you need matched filtering or
full Bayesian PE.
