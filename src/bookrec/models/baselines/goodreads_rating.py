"""Goodreads 평점순 추천 모델(A6 기준선). Goodreads 사이트의 책별 평균 별점(average_rating)이 높은 책을 모든 독자에게 똑같이 추천한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bookrec.models.base import BOOK_IDS, Recommender


class GoodreadsRatingRecommender(Recommender):
    """books의 average_rating을 점수로 쓰는 추천 모델. 학습 평점과 독자의 이력은 보지 않는다."""

    # NOTE: average_rating은 우리 ratings.csv가 아니라 Goodreads 사이트 전체 독자의 집계다(외부 정보를 쓴 기준선).
    def __init__(self, books: pd.DataFrame):
        self.ratings = books.set_index("book_id")["average_rating"].reindex(BOOK_IDS).to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame) -> GoodreadsRatingRecommender:
        """학습할 것이 없어 그대로 돌려준다."""
        return self

    def score(self, history: pd.DataFrame) -> pd.DataFrame:
        """이력에 나온 독자마다 self.ratings를 한 줄씩 복사해 점수표(행 user_id, 열 BOOK_IDS)를 만든다."""
        user_ids = np.sort(history["user_id"].unique())
        return pd.DataFrame(np.tile(self.ratings, (len(user_ids), 1)), index=user_ids, columns=BOOK_IDS)
