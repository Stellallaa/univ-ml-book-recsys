"""독자의 평점 이력과 책 태그 벡터로 추천 점수를 계산한다(B3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from bookrec.models.base import BOOK_IDS, Recommender
from bookrec.models.content_based.features import BookFeatures
from bookrec.models.popularity import PopularityRecommender


class ContentBasedRecommender(Recommender):
    """책 벡터의 평점 가중합을 독자 취향으로 사용한다. 영벡터는 인기순으로 대체한다."""

    def __init__(self, features: BookFeatures, profile_weighting: str = "positive"):
        if profile_weighting not in ("positive", "rating"):
            raise ValueError("profile_weighting은 'positive' 또는 'rating'이어야 합니다.")
        if not np.array_equal(features.book_ids, BOOK_IDS) or features.matrix.shape[0] != len(BOOK_IDS):
            raise ValueError("features의 행은 BOOK_IDS 전체와 같은 순서여야 합니다.")
        if features.matrix.shape[1] == 0 or not np.isfinite(features.matrix.data).all() or (features.matrix.data < 0).any():
            raise ValueError("features는 유한한 비음수 가중치와 하나 이상의 특징을 가져야 합니다.")
        self.features = features
        self.profile_weighting = profile_weighting
        # NOTE: B2 행렬은 이미 정규화되어 있다. 외부 BookFeatures 입력도 같은 점수 범위를 갖도록 복사해 정규화한다.
        self.book_vectors = normalize(features.matrix, norm="l2", copy=True)

    def fit(self, train: pd.DataFrame) -> ContentBasedRecommender:
        """train으로 인기순 대체 추천을 준비한다. 독자 취향은 score에 전달된 이력으로 계산한다."""
        self._validate_ratings(train)
        self.popularity = PopularityRecommender().fit(train)
        self.fallback_scores = self.popularity.counts / max(float(self.popularity.counts.max()), 1.0)
        return self

    def score(self, history: pd.DataFrame) -> pd.DataFrame:
        """독자별 코사인 점수(0~1)를 반환한다. 취향을 만들 수 없으면 train 인기순 점수를 사용한다."""
        if not hasattr(self, "fallback_scores"):
            raise RuntimeError("score 전에 fit(train)을 호출하세요.")
        users, profiles = self._profiles(history)
        if not len(users):
            return pd.DataFrame(index=users, columns=BOOK_IDS, dtype=float)
        # NOTE: normalize는 영벡터를 그대로 둔다. 희소 취향 벡터로 곱한 뒤 최종 점수표만 dense 배열로 만든다.
        scores = (normalize(profiles, norm="l2") @ self.book_vectors.T).toarray()
        fallback = np.asarray(profiles.getnnz(axis=1)).ravel() == 0
        scores[fallback] = self.fallback_scores
        # NOTE: 부동소수점 반올림으로 생기는 1 초과 값을 잘라 hybrid에 전달할 범위를 0~1로 고정한다.
        return pd.DataFrame(np.clip(scores, 0, 1), index=users, columns=BOOK_IDS)

    def profile_summary(self, history: pd.DataFrame) -> pd.DataFrame:
        """독자별 이력 수와 인기순 대체 여부를 반환한다. B4에서 대체 비율을 확인하는 데 쓴다."""
        users, profiles = self._profiles(history)
        sizes = history.groupby("user_id").size().reindex(users, fill_value=0)
        return pd.DataFrame({"history_size": sizes, "fallback": profiles.getnnz(axis=1) == 0}, index=users).rename_axis("user_id")

    def _profiles(self, history: pd.DataFrame) -> tuple[np.ndarray, csr_matrix]:
        self._validate_ratings(history)
        users = np.sort(history["user_id"].unique())
        weights = history["rating"].to_numpy(dtype=float)
        if self.profile_weighting == "positive":
            weights = np.maximum(weights - 3, 0)
        rows = pd.Index(users).get_indexer(history["user_id"])
        cols = pd.Index(BOOK_IDS).get_indexer(history["book_id"])
        # NOTE: positive는 4점=1, 5점=2이며 1~3점은 취향에서 제외한다. rating은 별점 자체를 가중치로 쓴다.
        history_matrix = csr_matrix((weights, (rows, cols)), shape=(len(users), len(BOOK_IDS)))
        profiles = csr_matrix(history_matrix @ self.book_vectors)
        profiles.eliminate_zeros()
        return users, profiles

    @staticmethod
    def _validate_ratings(ratings: pd.DataFrame) -> None:
        if not {"user_id", "book_id", "rating"}.issubset(ratings.columns):
            raise ValueError("평점 표에는 user_id, book_id, rating이 필요합니다.")
        if ratings["user_id"].isna().any() or not ratings["book_id"].isin(BOOK_IDS).all():
            raise ValueError("독자 ID는 결측이 없어야 하고 책 ID는 BOOK_IDS에 있어야 합니다.")
        values = ratings["rating"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 1) | (values > 5)).any():
            raise ValueError("rating은 1~5 범위의 유한한 값이어야 합니다.")
