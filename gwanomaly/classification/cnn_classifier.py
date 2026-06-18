"""
cnn_classifier.py
====================

CNN classifier operating on Q-transform (time-frequency) images, predicting
source type: BACKGROUND / BNS / NSBH / BBH / BURST (see
`gwanomaly.data.dataset.LABEL_TO_IDX` for the canonical label mapping).

Trained as a supervised multi-class classifier on labelled GWTC event
windows + background windows (built via `gwanomaly.data.dataset`). This is
the "Classify" stage of the pipeline, intended to run AFTER an anomaly has
been flagged by the detection stage — i.e. it answers "what kind of signal
is this" given something already deemed interesting.
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
class CNNClassifierConfig:
    input_shape: Tuple[int, int] = (128, 128)
    n_classes: int = 5  # BACKGROUND, BNS, NSBH, BBH, BURST
    channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    class_weights: Optional[List[float]] = None  # set from training-set class
    # frequencies to counteract the natural BACKGROUND >> {BNS,NSBH,BBH,BURST}
    # imbalance (catalogued events are rare relative to background time)


if torch is not None:

    class CNNClassifier(nn.Module):
        def __init__(self, config: CNNClassifierConfig):
            super().__init__()
            self.config = config
            layers = []
            prev_ch = 1
            for ch in config.channels:
                layers += [
                    nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
                prev_ch = ch
            self.conv = nn.Sequential(*layers)

            h, w = config.input_shape
            for _ in config.channels:
                h, w = h // 2, w // 2
            flat = prev_ch * max(h, 1) * max(w, 1)

            self.classifier_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(128, config.n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            features = self.conv(x)
            return self.classifier_head(features)

else:
    CNNClassifier = None  # type: ignore


class CNNClassifierTrainer:
    """Training/inference wrapper, mirroring AutoencoderDetector's interface."""

    def __init__(self, config: Optional[CNNClassifierConfig] = None, device: Optional[str] = None):
        if torch is None:
            raise ImportError("PyTorch is required for CNNClassifier.")
        self.config = config or CNNClassifierConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNClassifier(self.config).to(self.device)

    def fit(
        self,
        X_qimage: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[float]:
        cfg = self.config
        weights = (
            torch.tensor(cfg.class_weights, dtype=torch.float32).to(self.device)
            if cfg.class_weights else None
        )
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        X = torch.from_numpy(X_qimage).float().unsqueeze(1)
        y_t = torch.from_numpy(y).long()
        n = X.shape[0]
        history = []

        self.model.train()
        for epoch in range(cfg.epochs):
            perm = torch.randperm(n)
            epoch_loss, correct = 0.0, 0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                xb, yb = X[idx].to(self.device), y_t[idx].to(self.device)

                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

                epoch_loss += loss.item() * xb.shape[0]
                correct += (logits.argmax(1) == yb).sum().item()

            epoch_loss /= n
            train_acc = correct / n
            history.append(epoch_loss)

            if verbose and (epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1):
                msg = f"epoch {epoch+1}/{cfg.epochs}  loss={epoch_loss:.4f}  train_acc={train_acc:.3f}"
                if X_val is not None and y_val is not None:
                    val_acc = self.evaluate(X_val, y_val)
                    msg += f"  val_acc={val_acc:.3f}"
                print(msg)

        return history

    @torch.no_grad()
    def predict_proba(self, X_qimage: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.from_numpy(X_qimage).float().unsqueeze(1).to(self.device)
        logits = self.model(x)
        return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X_qimage: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_qimage).argmax(axis=1)

    def evaluate(self, X_qimage: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X_qimage)
        return float((preds == y).mean())

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "CNNClassifierTrainer":
        checkpoint = torch.load(path, map_location="cpu")
        trainer = cls(checkpoint["config"], device=device)
        trainer.model.load_state_dict(checkpoint["state_dict"])
        return trainer
