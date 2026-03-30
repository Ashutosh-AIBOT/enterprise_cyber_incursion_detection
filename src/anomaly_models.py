from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


@dataclass
class AnomalyResult:
    model_name: str
    auc: float


class TorchAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def evaluate_ocsvm(X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> AnomalyResult:
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    model.fit(X_train)
    scores = -model.decision_function(X_test)
    return AnomalyResult("one_class_svm", roc_auc_score(y_test, scores))


def evaluate_lof(X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> AnomalyResult:
    model = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination=0.05)
    model.fit(X_train)
    scores = -model.decision_function(X_test)
    return AnomalyResult("local_outlier_factor", roc_auc_score(y_test, scores))


def evaluate_torch_autoencoder(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 12,
    lr: float = 1e-3,
) -> AnomalyResult:
    if not TORCH_AVAILABLE:
        return AnomalyResult("torch_autoencoder", 0.5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = TorchAutoencoder(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon = model(X_train_t)
        loss = criterion(recon, X_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        recon = model(X_test_t)
        mse = torch.mean((X_test_t - recon) ** 2, dim=1).detach().cpu().numpy()
    return AnomalyResult("torch_autoencoder", roc_auc_score(y_test, mse))


def run_anomaly_suite(X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    results = [
        evaluate_ocsvm(X_train, X_test, y_test),
        evaluate_lof(X_train, X_test, y_test),
        evaluate_torch_autoencoder(X_train, X_test, y_test),
    ]
    return {r.model_name: float(r.auc) for r in results}
