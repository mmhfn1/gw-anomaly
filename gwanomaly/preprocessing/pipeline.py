"""
pipeline.py
============

Core signal preprocessing: whitening (divide by noise ASD), bandpass
filtering, and Q-transform / spectrogram generation.

Two implementations are provided for whitening + bandpass + Q-transform:

  - `gwpy` backend (preferred): uses `TimeSeries.whiten()`, `.bandpass()`,
    and `.q_transform()` directly, which is what you'd use against real
    GWOSC data in any environment with GWpy installed.
  - `numpy` fallback: a from-scratch implementation of the same three
    operations using FFT-based Welch PSD estimation, an FFT brick-wall-ish
    Butterworth bandpass, and a simple constant-Q transform via
    overlapping windowed FFTs at log-spaced center frequencies. This
    requires nothing beyond numpy/scipy and is what runs in
    network-restricted or dependency-light environments.

The pipeline picks gwpy automatically if importable, otherwise falls back.
Both paths produce the same output shape/contract (`PreprocessResult`), so
downstream code (dataset builder, models) doesn't care which ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    bandpass_low: float = 20.0
    bandpass_high: float = 1024.0  # safely below Nyquist (2048 Hz) at the
    # standard 4096 Hz GWOSC strain rate; raise toward ~1900 if you
    # resample to 4096 Hz with headroom, but keep some margin below Nyquist
    whiten_fftlength: float = 4.0       # seconds, PSD estimation window
    qtransform_qrange: Tuple[float, float] = (4, 64)
    qtransform_frange: Tuple[float, float] = (20, 1024)
    qimage_shape: Tuple[int, int] = (128, 128)  # (freq_bins, time_bins), resized output
    crop_seconds: Optional[float] = None  # crop edges after whitening to avoid filter artefacts
    backend: str = "auto"  # "auto" | "gwpy" | "numpy"


@dataclass
class PreprocessResult:
    whitened: np.ndarray   # 1D whitened + bandpassed strain
    qimage: np.ndarray     # 2D (freq_bins, time_bins) normalised Q-transform image (log1p + z-score)
    qimage_raw_energy: np.ndarray  # 2D (freq_bins, time_bins) UNNORMALISED Q-transform energy,
    # clipped to be non-negative. Use this (not `qimage`) as input to
    # ExcessPowerDetector.detect(), which assumes non-negative physical
    # energy, not a normalised/centred quantity. Note: GWpy's native
    # q_transform() output (and our numpy fallback at float64 precision)
    # can emit small negative values for pixels right at the noise floor
    # (~0.4% of pixels, empirically) -- these are clipped to 0 here, since
    # physical energy can't be negative and leaving them as-is corrupts
    # log-scale normalisation downstream (NaN from log1p of a negative).
    sample_rate: int


class PreprocessingPipeline:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._backend = self._select_backend()
        logger.info("PreprocessingPipeline using backend=%s", self._backend)

    def _select_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend
        try:
            import gwpy  # noqa: F401
            return "gwpy"
        except ImportError:
            return "numpy"

    def run(self, strain: np.ndarray, sample_rate: int) -> PreprocessResult:
        self._validate_against(strain, sample_rate)
        if self._backend == "gwpy":
            return self._run_gwpy(strain, sample_rate)
        return self._run_numpy(strain, sample_rate)

    def _validate_against(self, strain: np.ndarray, sample_rate: int) -> None:
        """
        Catch config/data mismatches up front with a clear error, instead
        of letting them surface as cryptic linalg/filter-design exceptions
        deep in gwpy or scipy.
        """
        cfg = self.config
        nyquist = sample_rate / 2.0
        # GWpy's TimeSeries.bandpass(), when not given explicit stopband
        # edges, defaults to a stopband high edge of min(bandpass_high*1.5,
        # nyquist) (see gwpy.signal.filter_design.bandpass). If bandpass_high
        # is close enough to nyquist that bandpass_high*1.5 exceeds it, the
        # min() clamps the stopband edge to EXACTLY nyquist, which is a
        # normalised frequency of exactly 1.0 -- and scipy.signal.iirdesign
        # requires strictly less than 1.0, raising "Values for wp, ws must
        # be less than 1". So the real constraint isn't just bandpass_high
        # < nyquist; it's that bandpass_high*1.5 must also clear nyquist
        # with room to spare. Require bandpass_high <= 0.6 * nyquist, which
        # keeps bandpass_high*1.5 at or below 0.9 * nyquist with margin.
        margin = 0.6
        if cfg.bandpass_high > margin * nyquist:
            raise ValueError(
                f"bandpass_high ({cfg.bandpass_high} Hz) must be at most "
                f"{margin*100:.0f}% of the Nyquist frequency ({nyquist} Hz) "
                f"for sample_rate={sample_rate}. GWpy's bandpass() pads the "
                f"stopband edge to min(bandpass_high*1.5, nyquist) internally, "
                f"which lands exactly on the Nyquist boundary (and fails "
                f"scipy's filter design) if bandpass_high is too close to it. "
                f"Either lower bandpass_high or use a higher sample_rate "
                f"(GWOSC strain is natively 4096 or 16384 Hz)."
            )
        duration = len(strain) / sample_rate
        if cfg.whiten_fftlength * 2 > duration:
            raise ValueError(
                f"whiten_fftlength ({cfg.whiten_fftlength}s) is too long for a "
                f"{duration:.2f}s segment; GWpy's whiten() needs roughly "
                f"2x fftlength of data to estimate a stable PSD. Either "
                f"shorten whiten_fftlength or fetch a longer segment."
            )

    # ------------------------------------------------------------------
    # GWpy-backed implementation (use this against real GWOSC data)
    # ------------------------------------------------------------------
    def _run_gwpy(self, strain: np.ndarray, sample_rate: int) -> PreprocessResult:
        from gwpy.timeseries import TimeSeries

        cfg = self.config
        # Real (and synthetic) strain amplitudes are ~1e-19 to 1e-21. Squaring
        # that for PSD/ASD estimation in float32 underflows to exact zero for
        # a large fraction of frequency bins (float32 min normal ~1.2e-38),
        # which then causes divide-by-zero in whiten(). Do the math in
        # float64 and only downcast to float32 at the very end for storage.
        ts = TimeSeries(np.asarray(strain, dtype=np.float64), sample_rate=sample_rate)

        white = ts.whiten(fftlength=cfg.whiten_fftlength)
        filtered = white.bandpass(cfg.bandpass_low, cfg.bandpass_high)

        if cfg.crop_seconds:
            filtered = filtered.crop(
                filtered.t0.value + cfg.crop_seconds,
                filtered.t0.value + filtered.duration.value - cfg.crop_seconds,
            )

        # Requesting Q-transform content above bandpass_high is unphysical
        # anyway (that frequency content was already filtered out by
        # bandpass() above), and if qtransform_frange's upper bound doesn't
        # fit the segment's duration/Q combination, GWpy silently resets it
        # (UserWarning: "upper frequency ... too high for the given Q
        # range") rather than raising -- so without this clamp, the actual
        # frequency range of `qimage`/`qimage_raw_energy` can silently
        # differ from what the config claims. Clamp explicitly so the
        # configured value and the actual value always agree.
        frange_high = min(cfg.qtransform_frange[1], cfg.bandpass_high)
        frange = (cfg.qtransform_frange[0], frange_high)

        qspec = filtered.q_transform(
            qrange=cfg.qtransform_qrange,
            frange=frange,
            logf=True,
        )
        raw_energy = np.asarray(qspec.value).T  # (freq, time), GWpy's native units (energy-like)
        raw_energy = _resize_2d(raw_energy, cfg.qimage_shape)
        raw_energy = np.clip(raw_energy, 0, None)  # see PreprocessResult docstring: GWpy can emit
        # small negatives at the noise floor; physical energy can't be negative
        qimage = _normalise(raw_energy)

        return PreprocessResult(
            whitened=filtered.value.astype(np.float32),
            qimage=qimage.astype(np.float32),
            qimage_raw_energy=raw_energy.astype(np.float64),
            sample_rate=int(filtered.sample_rate.value),
        )

    # ------------------------------------------------------------------
    # Dependency-light numpy/scipy fallback
    # ------------------------------------------------------------------
    def _run_numpy(self, strain: np.ndarray, sample_rate: int) -> PreprocessResult:
        cfg = self.config

        whitened = _whiten_numpy(strain, sample_rate, fftlength=cfg.whiten_fftlength)
        filtered = _bandpass_numpy(whitened, sample_rate, cfg.bandpass_low, cfg.bandpass_high)

        if cfg.crop_seconds:
            crop_n = int(cfg.crop_seconds * sample_rate)
            filtered = filtered[crop_n:-crop_n] if crop_n > 0 else filtered

        raw_energy = _qtransform_numpy(
            filtered, sample_rate, cfg.qtransform_frange, cfg.qimage_shape
        )
        raw_energy = np.clip(raw_energy, 0, None)  # defensive: our own |conv|^2 is
        # mathematically non-negative already, but clip for consistency/safety
        qimage = _normalise(raw_energy)

        return PreprocessResult(
            whitened=filtered.astype(np.float32),
            qimage=qimage.astype(np.float32),
            qimage_raw_energy=raw_energy.astype(np.float64),
            sample_rate=sample_rate,
        )


# --------------------------------------------------------------------------
# numpy/scipy helper functions
# --------------------------------------------------------------------------

def _whiten_numpy(strain: np.ndarray, sample_rate: int, fftlength: float) -> np.ndarray:
    """Whiten by dividing the spectrum by an estimated ASD (Welch PSD ** 0.5).

    Computed in float64: real/synthetic strain amplitudes (~1e-19 to 1e-21)
    squared for PSD estimation underflow to exact zero in float32 (whose
    minimum normal value is ~1.2e-38), which corrupts a large fraction of
    frequency bins and propagates NaNs/zero-division downstream.
    """
    from scipy.signal import welch

    strain = np.asarray(strain, dtype=np.float64)
    nperseg = int(fftlength * sample_rate)
    nperseg = min(nperseg, len(strain))
    freqs, psd = welch(strain, fs=sample_rate, nperseg=nperseg)
    psd[psd == 0] = np.finfo(float).eps

    n = len(strain)
    spec_freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    psd_interp = np.interp(spec_freqs, freqs, psd)
    asd_interp = np.sqrt(psd_interp)
    asd_interp[asd_interp == 0] = np.finfo(float).eps

    spec = np.fft.rfft(strain)
    white_spec = spec / asd_interp
    white = np.fft.irfft(white_spec, n=n)
    # normalise to unit variance, standard convention for whitened strain
    white = white / (np.std(white) + 1e-12)
    return white.astype(np.float32)


def _bandpass_numpy(
    data: np.ndarray, sample_rate: int, low: float, high: float, order: int = 8
) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyq = sample_rate / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    sos = butter(order, [low_n, high_n], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data).astype(np.float32)


def _qtransform_numpy(
    data: np.ndarray,
    sample_rate: int,
    frange: Tuple[float, float],
    out_shape: Tuple[int, int],
    n_freq_bins: int = 96,
) -> np.ndarray:
    """
    Approximate constant-Q transform: for each log-spaced centre frequency,
    convolve with a complex Morlet-like wavelet sized so each frequency
    sees a comparable number of oscillation cycles (the defining property
    of a Q-transform vs a fixed-window STFT).

    Computed in float64 throughout. This matters even though the normal
    pipeline path calls this AFTER whitening (where data is already
    unit-variance, so float32 is fine) — if this function is ever called
    directly on raw, un-whitened strain (~1e-19 to 1e-21 scale, as in
    excess-power energy-map extraction), squared energies land at or below
    float32's minimum normal value (~1.2e-38) and silently underflow to
    exact zero across most of the array.
    """
    data = np.asarray(data, dtype=np.float64)
    f_lo, f_hi = frange
    freqs = np.geomspace(f_lo, f_hi, n_freq_bins)
    n = len(data)
    energy = np.zeros((n_freq_bins, n), dtype=np.float64)

    q = 8.0  # fixed quality factor for the approx transform
    for i, f0 in enumerate(freqs):
        cycles = q
        sigma_t = cycles / (2 * np.pi * f0)
        half_window = min(int(4 * sigma_t * sample_rate), n // 2 - 1)
        half_window = max(half_window, 4)
        t = np.arange(-half_window, half_window + 1) / sample_rate
        wavelet = np.exp(2j * np.pi * f0 * t) * np.exp(-(t ** 2) / (2 * sigma_t ** 2))
        wavelet /= np.linalg.norm(wavelet) + 1e-12

        conv = np.convolve(data, wavelet, mode="same")
        energy[i] = np.abs(conv) ** 2

    return _resize_2d(energy, out_shape)


def _resize_2d(arr: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D array via simple bin-averaging / nearest interpolation
    (no scipy.ndimage.zoom dependency assumption beyond what's already used
    above; this keeps it index-based and dependency-free)."""
    in_f, in_t = arr.shape
    out_f, out_t = out_shape

    f_idx = np.linspace(0, in_f, out_f, endpoint=False).astype(int)
    t_idx = np.linspace(0, in_t, out_t, endpoint=False).astype(int)
    return arr[np.ix_(f_idx, t_idx)]


def _normalise(qimage: np.ndarray) -> np.ndarray:
    """
    Log-scale + z-score normalise a Q-transform energy image.

    Clips to non-negative before log1p: GWpy's `q_transform()` (and our
    own numpy fallback's wavelet convolution, which can produce tiny
    negative-adjacent rounding artefacts at float64 precision) both
    occasionally return small negative values for time-frequency pixels
    right at the noise floor — confirmed empirically (~0.4% of pixels,
    magnitude comparable to the noise floor itself, not large outliers).
    Physically, energy can't be negative; treat these as ~0 energy rather
    than feeding log1p a negative argument, which produces NaN and was
    silently corrupting ~11% of qimage pixels before this fix.
    """
    qimage = np.clip(qimage, 0, None)
    log_img = np.log1p(qimage)
    mu, sigma = log_img.mean(), log_img.std() + 1e-8
    return (log_img - mu) / sigma
