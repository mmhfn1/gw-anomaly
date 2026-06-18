"""
parameter_estimation.py
=========================

Two complementary approaches to estimating source parameters (chirp mass,
mass ratio, luminosity distance) once a candidate has been classified:

1. RegressionPEHead - a fast point-estimate regressor (small MLP on top of
   CNN/LSTM features, or trained standalone on Q-images) giving an
   immediate, approximate parameter estimate. Useful for low-latency alerts
   and for ranking/triaging candidates, but gives NO uncertainty
   quantification.

2. BilbyPEWrapper - wraps Bilby for full Bayesian parameter estimation via
   nested sampling (dynesty by default), giving posterior distributions
   with proper uncertainty quantification. This is what real LIGO/Virgo PE
   releases use, but it's orders of magnitude slower (minutes to hours per
   event vs milliseconds for the regression head) — appropriate as an
   offline follow-up step on confirmed candidates, not for real-time
   triage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


@dataclass
class PEConfig:
    input_shape: Tuple[int, int] = (128, 128)
    target_names: List[str] = field(default_factory=lambda: ["mass_1", "mass_2", "distance_mpc"])
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 150


if torch is not None:

    class RegressionPEHead(nn.Module):
        """
        Small conv + MLP regressor predicting point estimates for each
        target in `config.target_names`, trained with MSE against
        catalogue-published median values (from
        `gwanomaly.data.catalogue.CatalogueBuilder`).

        Targets should be normalised (e.g. log-scale + z-score) before
        training, since mass (solar masses, O(1-100)) and distance (Mpc,
        O(10-10000)) live on very different scales; predicting raw values
        jointly would let the loss be dominated by distance error.
        """

        def __init__(self, config: PEConfig):
            super().__init__()
            self.config = config
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            )
            h, w = config.input_shape
            h, w = h // 8, w // 8
            flat = 64 * max(h, 1) * max(w, 1)

            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, config.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_dim, len(config.target_names)),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.head(self.conv(x))

else:
    RegressionPEHead = None  # type: ignore


class RegressionPETrainer:
    """Training wrapper for RegressionPEHead, with target normalisation
    baked in so callers work in physical units (solar masses, Mpc)."""

    def __init__(self, config: Optional[PEConfig] = None, device: Optional[str] = None):
        if torch is None:
            raise ImportError("PyTorch is required for RegressionPEHead.")
        self.config = config or PEConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RegressionPEHead(self.config).to(self.device)
        self._target_mean: Optional[np.ndarray] = None
        self._target_std: Optional[np.ndarray] = None

    def _targets_to_array(self, targets: List[Dict[str, float]]) -> np.ndarray:
        return np.array(
            [[t[name] for name in self.config.target_names] for t in targets],
            dtype=np.float32,
        )

    def fit(
        self,
        X_qimage: np.ndarray,
        targets: List[Dict[str, float]],
        verbose: bool = True,
    ) -> List[float]:
        cfg = self.config
        y = self._targets_to_array(targets)
        # log-scale (targets are strictly positive: masses, distance) then
        # z-score, so the loss isn't dominated by whichever target has the
        # largest raw numeric range (distance in Mpc vs mass in Msun)
        y_log = np.log(np.clip(y, 1e-6, None))
        self._target_mean = y_log.mean(axis=0)
        self._target_std = y_log.std(axis=0) + 1e-8
        y_norm = (y_log - self._target_mean) / self._target_std

        X = torch.from_numpy(X_qimage).float().unsqueeze(1)
        y_t = torch.from_numpy(y_norm).float()
        n = X.shape[0]
        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        history = []

        self.model.train()
        for epoch in range(cfg.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                xb, yb = X[idx].to(self.device), y_t[idx].to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.shape[0]
            epoch_loss /= n
            history.append(epoch_loss)
            if verbose and (epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1):
                print(f"epoch {epoch+1}/{cfg.epochs}  mse={epoch_loss:.4f}")
        return history

    @torch.no_grad()
    def predict(self, X_qimage: np.ndarray) -> List[Dict[str, float]]:
        """Returns point estimates in physical units (solar masses, Mpc)."""
        self.model.eval()
        x = torch.from_numpy(X_qimage).float().unsqueeze(1).to(self.device)
        pred_norm = self.model(x).cpu().numpy()
        pred_log = pred_norm * self._target_std + self._target_mean
        pred = np.exp(pred_log)
        return [
            {name: float(val) for name, val in zip(self.config.target_names, row)}
            for row in pred
        ]


class BilbyPEWrapper:
    """
    Thin wrapper around Bilby for full Bayesian parameter estimation via
    nested sampling. This is a heavyweight, offline step — appropriate for
    confirmed candidates, not real-time triage (expect minutes-to-hours
    per event depending on sampler settings and waveform approximant).

    Bilby itself depends on lalsuite/lalsimulation for waveform generation,
    so this wrapper assumes those are installed in the same environment as
    PyCBC (see `gwanomaly.detection.matched_filter`).
    """

    def __init__(
        self,
        outdir: str = "./bilby_pe_output",
        label: str = "gw_event",
        sampler: str = "dynesty",
        nlive: int = 500,
    ):
        self.outdir = outdir
        self.label = label
        self.sampler = sampler
        self.nlive = nlive

    def run(
        self,
        strain_by_detector: Dict[str, np.ndarray],
        sample_rate: int,
        trigger_time: float,
        prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        duration: float = 4.0,
        waveform_approximant: str = "IMRPhenomD",
    ):
        """
        Run Bilby's standard CBC parameter estimation pipeline on a
        multi-detector strain dict, returning the posterior result object.

        `prior_bounds` lets you override the default broad priors (e.g.
        narrow mass priors around a CNN classifier's point estimate to
        speed up convergence) — pass {"mass_1": (m1_lo, m1_hi), ...}.
        """
        try:
            import bilby
        except ImportError as exc:
            raise ImportError(
                "Bilby is required for BilbyPEWrapper. Install with "
                "`pip install bilby` (also requires lalsuite for waveform "
                "generation)."
            ) from exc

        ifos = bilby.gw.detector.InterferometerList([])
        for det_name, strain in strain_by_detector.items():
            ifo = bilby.gw.detector.get_empty_interferometer(det_name)
            ts = bilby.gw.detector.strain_data.StrainData()
            ts.set_from_array(
                strain, sampling_frequency=sample_rate, start_time=trigger_time - duration / 2
            )
            ifo.strain_data = ts
            ifos.append(ifo)

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sample_rate,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments={"waveform_approximant": waveform_approximant, "reference_frequency": 20.0},
        )

        priors = bilby.gw.prior.BBHPriorDict()
        priors["geocent_time"] = bilby.core.prior.Uniform(
            trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
        )
        if prior_bounds:
            for key, (lo, hi) in prior_bounds.items():
                priors[key] = bilby.core.prior.Uniform(lo, hi, name=key)

        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator
        )

        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=self.sampler,
            nlive=self.nlive,
            outdir=self.outdir,
            label=self.label,
        )
        return result
