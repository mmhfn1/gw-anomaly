"""
lstm_classifier.py
=====================

LSTM classifier operating directly on the whitened strain timeseries
(rather than the Q-transform image), giving the classification stage a
second, complementary view of the same candidate window. In practice you'd
ensemble this with the CNN (e.g. average softmax probabilities, or train a
small stacking model on both sets of logits) rather than relying on one
view alone — time-domain and time-frequency representations lose different
information.

Input is downsampled before feeding to the LSTM: raw 4096 Hz strain over
even a few seconds is tens of thousands of timesteps, which is impractical
for a plain LSTM (vanishing gradients over long sequences, and very slow).
A configurable strided downsample (or alternatively a few conv1d layers
before the LSTM) keeps sequence length tractable. The default here keeps
it simple (strided downsample); swap in a conv-frontend if accuracy on
real data warrants the extra complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None


@dataclass
class LSTMClassifierConfig:
    input_length: int = 4096  # length of whitened_strain window fed in, before downsampling
    downsample_factor: int = 8  # reduces sequence length to input_length // downsample_factor
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = True
    n_classes: int = 5
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    class_weights: Optional[List[float]] = None


if torch is not None:

    class LSTMClassifier(nn.Module):
        def __init__(self, config: LSTMClassifierConfig):
            super().__init__()
            self.config = config
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            direction_mult = 2 if config.bidirectional else 1
            self.classifier_head = nn.Sequential(
                nn.Linear(config.hidden_size * direction_mult, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(64, config.n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, 1)
            cfg = self.config
            if cfg.downsample_factor > 1:
                x = x[:, ::cfg.downsample_factor, :]
            _, (h_n, _) = self.lstm(x)
            if cfg.bidirectional:
                last = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                last = h_n[-1]
            return self.classifier_head(last)

else:
    LSTMClassifier = None  # type: ignore


class LSTMClassifierTrainer:
    """Training/inference wrapper, same interface shape as CNNClassifierTrainer."""

    def __init__(self, config: Optional[LSTMClassifierConfig] = None, device: Optional[str] = None):
        if torch is None:
            raise ImportError("PyTorch is required for LSTMClassifier.")
        self.config = config or LSTMClassifierConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMClassifier(self.config).to(self.device)

    def fit(
        self,
        X_strain: np.ndarray,  # (N, input_length)
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

        X = torch.from_numpy(X_strain).float().unsqueeze(-1)  # (N, L, 1)
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
    def predict_proba(self, X_strain: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.from_numpy(X_strain).float().unsqueeze(-1).to(self.device)
        logits = self.model(x)
        return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X_strain: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_strain).argmax(axis=1)

    def evaluate(self, X_strain: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X_strain)
        return float((preds == y).mean())

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LSTMClassifierTrainer":
        checkpoint = torch.load(path, map_location="cpu")
        trainer = cls(checkpoint["config"], device=device)
        trainer.model.load_state_dict(checkpoint["state_dict"])
        return trainer
