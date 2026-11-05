"""모든 추천 모델이 따르는 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# NOTE: Goodbooks-10k의 book_id는 1~10000 연속이다. 모든 모델의 점수표는 이 순서의 열을 쓴다.
BOOK_IDS = np.arange(1, 10_001)


class Recommender(ABC):
    """독자의 평점 이력을 받아 책마다 점수를 매기는 추천 모델."""

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Recommender:
        """학습 평점(user_id, book_id, rating)으로 모델을 학습한다."""

    @abstractmethod
    def score(self, history: pd.DataFrame) -> pd.DataFrame:
        """독자별 평점 이력(user_id, book_id, rating)으로 점수표(행 user_id, 열 BOOK_IDS)를 만든다. 높을수록 추천한다."""

    def recommend(self, history: pd.DataFrame, k: int = 10) -> dict[int, list[int]]:
        """독자별로 이력에 없는 책 중 점수 상위 k권의 book_id를 돌려준다."""
        scores = self.score(history)
        values = scores.to_numpy(dtype=float, copy=True)
        values[scores.index.get_indexer(history["user_id"]), scores.columns.get_indexer(history["book_id"])] = -np.inf
        top = np.argsort(-values, axis=1, kind="stable")[:, :k]
        return dict(zip(scores.index.tolist(), scores.columns.to_numpy()[top].tolist()))


class RatingPredictor(ABC):
    """평점 자체를 예측하는 모델. RMSE 평가에만 쓴다."""

    @abstractmethod
    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """(user_id, book_id) 쌍마다 예상 평점을 돌려준다."""
