from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks


@dataclass
class SamplingResult:
    X_resampled: np.ndarray
    y_resampled: np.ndarray
    strategy: str


class SamplingManager:
    """Central balancing orchestrator for severe class imbalance."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def apply(self, X: np.ndarray, y: np.ndarray, strategy: str) -> SamplingResult:
        strategy = strategy.lower().strip()

        if strategy == "adasyn":
            sampler = ADASYN(random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(X, y)
        elif strategy == "tomek_links":
            sampler = TomekLinks()
            X_res, y_res = sampler.fit_resample(X, y)
        elif strategy == "borderline_smote":
            sampler = BorderlineSMOTE(random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(X, y)
        elif strategy == "adasyn_tomek":
            X_tmp, y_tmp = ADASYN(random_state=self.random_state).fit_resample(X, y)
            X_res, y_res = TomekLinks().fit_resample(X_tmp, y_tmp)
        elif strategy == "borderline_tomek":
            X_tmp, y_tmp = BorderlineSMOTE(random_state=self.random_state).fit_resample(X, y)
            X_res, y_res = TomekLinks().fit_resample(X_tmp, y_tmp)
        elif strategy == "smote_tomek":
            sampler = SMOTETomek(random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(X, y)
        elif strategy == "none":
            X_res, y_res = X, y
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return SamplingResult(X_resampled=X_res, y_resampled=y_res, strategy=strategy)

    def class_ratio(self, y: np.ndarray) -> Tuple[int, int, float]:
        minority = int((y == 1).sum())
        majority = int((y == 0).sum())
        ratio = float(minority / max(majority, 1))
        return minority, majority, ratio
