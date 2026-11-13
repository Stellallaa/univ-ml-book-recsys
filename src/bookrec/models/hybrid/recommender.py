"""Score-level combinations of the project's content and MF rankers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from bookrec.models.base import BOOK_IDS, Recommender


class HybridRecommender(Recommender):
    """Combine user-wise CB and MF percentile scores.

    ``mode='fixed'`` uses a fixed MF weight. ``switch`` selects CB below a
    history threshold and MF from the threshold onward. ``gradual`` linearly
    moves from CB to MF between ``switch_at`` and ``full_mf_at``.
    """

    MODES = {"fixed", "switch", "gradual"}

    def __init__(self, content_model: Recommender, collaborative_model: Recommender,
                 mode: str = "fixed", mf_weight: float = 0.5,
                 switch_at: int = 3, full_mf_at: int = 10):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if not np.isfinite(mf_weight) or not 0 <= mf_weight <= 1:
            raise ValueError("mf_weight must be between 0 and 1")
        if not isinstance(switch_at, int) or isinstance(switch_at, bool) or switch_at < 1:
            raise ValueError("switch_at must be a positive integer")
        if mode == "gradual" and (
            not isinstance(full_mf_at, int) or isinstance(full_mf_at, bool) or full_mf_at <= switch_at
        ):
            raise ValueError("full_mf_at must be an integer greater than switch_at")
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.mode = mode
        self.mf_weight = float(mf_weight)
        self.switch_at = switch_at
        self.full_mf_at = full_mf_at

    def fit(self, train: pd.DataFrame) -> HybridRecommender:
        self.content_model.fit(train)
        self.collaborative_model.fit(train)
        return self

    def score(self, history: pd.DataFrame) -> pd.DataFrame:
        cb = self.content_model.score(history)
        mf = self.collaborative_model.score(history)
        if not cb.index.equals(mf.index) or not cb.columns.equals(mf.columns):
            raise ValueError("CB and MF scores must have identical user and book indexes")
        if not np.array_equal(cb.columns.to_numpy(), BOOK_IDS):
            raise ValueError("score columns must match BOOK_IDS")
        cb_values = self._percentile(cb.to_numpy(dtype=float))
        mf_values = self._percentile(mf.to_numpy(dtype=float))
        sizes = history.groupby("user_id").size().reindex(cb.index).to_numpy()
        if self.mode == "fixed":
            weights = np.full((len(sizes), 1), self.mf_weight)
        elif self.mode == "switch":
            weights = (sizes >= self.switch_at).astype(float)[:, None]
        else:
            weights = np.clip((sizes - self.switch_at) / (self.full_mf_at - self.switch_at), 0, 1)[:, None]
        combined = cb_values * (1 - weights) + mf_values * weights
        return pd.DataFrame(combined, index=cb.index, columns=cb.columns)

    @staticmethod
    def _percentile(values: np.ndarray) -> np.ndarray:
        if not np.isfinite(values).all():
            raise ValueError("model scores must be finite before fusion")
        if values.ndim != 2:
            raise ValueError("model scores must be a two-dimensional matrix")
        # Percentile ranks put CB (0..1) and unbounded MF scores on one scale.
        return rankdata(values, axis=1, method="average") / values.shape[1]
