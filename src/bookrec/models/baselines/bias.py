"""bias 기준선(A6). 전체 평균에 독자 성향(후하게 주는지)과 책 성향(평이 좋은지)만 더해 별점을 예측한다. RMSE 평가에만 쓴다.

출처: Koren, Y. & Bell, R. (2011). Advances in Collaborative Filtering. Recommender Systems Handbook, Springer. 2.1절
"Baseline predictors"의 b_ui = μ + b_u + b_i와, (합) / (λ + 평점 수)로 b_i·b_u를 따로 구하는 방법(논문은 1회, 여기서는 n_iters회 반복).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bookrec.models.base import RatingPredictor

# NOTE: reg는 성향을 구할 때 분모에 더하는 값으로, "오차 0인 가짜 평점 reg개"를 붙이는 효과다. 클수록 평점이 적은 독자·책의 성향이 0에 가까워진다.
USER_REG = 10.0
BOOK_REG = 10.0
N_ITERS = 10  # NOTE: 책 성향 → 독자 성향을 번갈아 다시 계산하는 횟수


class BiasPredictor(RatingPredictor):
    """예측 별점 = mean + user_bias[독자] + book_bias[책]. 학습에 없던 독자·책의 성향은 0이다."""

    def __init__(self, user_reg: float = USER_REG, book_reg: float = BOOK_REG, n_iters: int = N_ITERS):
        self.user_reg = user_reg
        self.book_reg = book_reg
        self.n_iters = n_iters

    def fit(self, train: pd.DataFrame) -> BiasPredictor:
        """책 성향과 독자 성향을 번갈아 n_iters번 다시 계산한다. 평점이 적을수록 reg 때문에 성향이 0에 가까워진다."""
        self.mean = train["rating"].mean()
        # NOTE: 첫 회차에는 독자 성향을 모두 0으로 본다(빈 Series).
        self.user_bias = pd.Series(dtype=float)
        for _ in range(self.n_iters):
            # NOTE: 한쪽 성향을 고정하면 다른 쪽은 (남은 오차 합) / (평점 수 + reg)가 최적이다(정규화한 최소제곱).
            self.book_bias = self._shrunk_mean(train, "book_id", self.user_bias, "user_id", self.book_reg)
            self.user_bias = self._shrunk_mean(train, "user_id", self.book_bias, "book_id", self.user_reg)
        return self

    def _shrunk_mean(self, train: pd.DataFrame, key: str, other_bias: pd.Series, other_key: str, reg: float) -> pd.Series:
        """상대편 성향을 뺀 오차를 key별로 더해 (평점 수 + reg)로 나눈다."""
        # NOTE: Series.map(Series)는 왼쪽 값(ID)을 오른쪽 Series의 인덱스에서 찾아 그 값으로 바꾼다. 없는 ID는 NaN이 되고 fillna(0.0)가 0으로 채운다.
        residual = train["rating"] - self.mean - train[other_key].map(other_bias).fillna(0.0)
        # NOTE: groupby(열).agg(["sum", "size"])는 key별 오차 합과 평점 수를 한 번에 구한다.
        stats = residual.groupby(train[key]).agg(["sum", "size"])
        return stats["sum"] / (stats["size"] + reg)

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """(user_id, book_id) 쌍마다 예상 별점을 1~5로 잘라 돌려준다."""
        user = pairs["user_id"].map(self.user_bias).fillna(0.0).to_numpy()
        book = pairs["book_id"].map(self.book_bias).fillna(0.0).to_numpy()
        # NOTE: MF(collaborative_filtering/als.py)와 같은 조건으로 비교하려고 별점 범위 밖 예측을 자른다.
        return np.clip(self.mean + user + book, 1, 5)
