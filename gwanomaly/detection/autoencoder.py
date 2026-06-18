"""
autoencoder.py
================

Unsupervised anomaly detector: a convolutional autoencoder trained ONLY on
background (no-event) Q-transform images. At inference time, reconstruction
error on a new Q-image is the anomaly score — background-like inputs
reconstruct well (low error), while genuine signals or glitches (which the
model never saw in training) reconstruct poorly (high error).

This is a full-scale model intended for GPU training on real GWOSC
background data. The architecture below is sized for 128x128 Q-images;
adjust `AutoencoderConfig.input_shape` and the channel schedule if you
change `qimage_shape` in preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None


@dataclass
class AutoencoderConfig:
    input_shape: Tuple[int, int] = (128, 128)
    latent_dim: int = 64
    channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 100
    # Anomaly threshold is set post-hoc from a percentile of background
    # validation-set reconstruction error (see AutoencoderDetector.calibrate).
    threshold_percentile: float = 99.0


if torch is not None:

    class ConvAutoencoder(nn.Module):
        """
        Symmetric conv encoder/decoder. Assumes square, power-of-2-friendly
        input (e.g. 128x128); each encoder stage halves spatial resolution
        via stride-2 conv, each decoder stage doubles it via transposed conv.
        """

        def __init__(self, config: AutoencoderConfig):
            super().__init__()
            self.config = config
            c = config.channels
            in_ch = 1

            encoder_layers = []
            prev_ch = in_ch
            for ch in c:
                encoder_layers += [
                    nn.Conv2d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
                prev_ch = ch
            self.encoder = nn.Sequential(*encoder_layers)

            # Compute flattened spatial size after encoder for the bottleneck FC
            h, w = config.input_shape
            for _ in c:
                h, w = (h + 1) // 2, (w + 1) // 2
            self._enc_spatial = (prev_ch, h, w)
            flat = prev_ch * h * w

            self.fc_enc = nn.Linear(flat, config.latent_dim)
            self.fc_dec = nn.Linear(config.latent_dim, flat)

            decoder_layers = []
            rev_channels = list(reversed(c))
            for i in range(len(rev_channels) - 1):
                decoder_layers += [
                    nn.ConvTranspose2d(
                        rev_channels[i], rev_channels[i + 1],
                        kernel_size=4, stride=2, padding=1,
                    ),
                    nn.BatchNorm2d(rev_channels[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            decoder_layers += [
                nn.ConvTranspose2d(rev_channels[-1], 1, kernel_size=4, stride=2, padding=1),
            ]
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.encoder(x)
            z = z.flatten(1)
            return self.fc_enc(z)

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            c, h, w = self._enc_spatial
            x = self.fc_dec(z)
            x = x.view(-1, c, h, w)
            x = self.decoder(x)
            # Crop/pad to exactly match input_shape in case of off-by-one
            # rounding through the stride-2 stages.
            target_h, target_w = self.config.input_shape
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return x

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.encode(x)
            return self.decode(z)

else:
    ConvAutoencoder = None  # type: ignore


class AutoencoderDetector:
    """
    Training + inference wrapper around `ConvAutoencoder`. Handles the
    train-on-background-only loop, threshold calibration, and per-sample
    anomaly scoring.

    Example
    -------
    >>> detector = AutoencoderDetector(AutoencoderConfig())
    >>> detector.fit(background_qimages_train)          # numpy array (N, H, W)
    >>> detector.calibrate(background_qimages_val)        # sets self.threshold
    >>> scores = detector.score(candidate_qimages)        # (M,) reconstruction error
    >>> is_anomaly = detector.predict(candidate_qimages)   # (M,) bool
    """

    def __init__(self, config: Optional[AutoencoderConfig] = None, device: Optional[str] = None):
        if torch is None:
            raise ImportError(
                "PyTorch is required for AutoencoderDetector. Install with "
                "`pip install torch` (use a CUDA build on your GPU machine)."
            )
        self.config = config or AutoencoderConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvAutoencoder(self.config).to(self.device)
        self.threshold: Optional[float] = None

    def fit(
        self,
        background_qimages: np.ndarray,
        val_qimages: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the autoencoder on background-only Q-images.

        Parameters
        ----------
        background_qimages : (N, H, W) float array, normalised as produced
            by `PreprocessingPipeline` (log-scaled, z-scored).
        val_qimages : optional held-out background set for monitoring.
        """
        cfg = self.config
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()

        x = torch.from_numpy(background_qimages).float().unsqueeze(1)  # (N,1,H,W)
        n = x.shape[0]
        history = []

        self.model.train()
        for epoch in range(cfg.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                batch = x[idx].to(self.device)

                opt.zero_grad()
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * batch.shape[0]

            epoch_loss /= n
            history.append(epoch_loss)

            if verbose and (epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1):
                msg = f"epoch {epoch+1}/{cfg.epochs}  train_mse={epoch_loss:.5f}"
                if val_qimages is not None:
                    val_score = self.score(val_qimages).mean()
                    msg += f"  val_mse={val_score:.5f}"
                print(msg)

        return history

    @torch.no_grad()
    def score(self, qimages: np.ndarray) -> np.ndarray:
        """Per-sample reconstruction error (MSE), higher = more anomalous."""
        self.model.eval()
        x = torch.from_numpy(qimages).float().unsqueeze(1).to(self.device)
        recon = self.model(x)
        err = ((recon - x) ** 2).mean(dim=(1, 2, 3))
        return err.cpu().numpy()

    def calibrate(self, background_qimages: np.ndarray) -> float:
        """
        Set `self.threshold` from a percentile of background reconstruction
        error, so that ~`threshold_percentile`% of clean background falls
        below threshold (i.e. controls the background false-alarm rate).
        """
        scores = self.score(background_qimages)
        self.threshold = float(np.percentile(scores, self.config.threshold_percentile))
        return self.threshold

    def predict(self, qimages: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Call calibrate() on background data before predict().")
        return self.score(qimages) > self.threshold

    def save(self, path: str) -> None:
        torch.save(
            {"state_dict": self.model.state_dict(), "config": self.config, "threshold": self.threshold},
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "AutoencoderDetector":
        checkpoint = torch.load(path, map_location="cpu")
        detector = cls(checkpoint["config"], device=device)
        detector.model.load_state_dict(checkpoint["state_dict"])
        detector.threshold = checkpoint["threshold"]
        return detector
