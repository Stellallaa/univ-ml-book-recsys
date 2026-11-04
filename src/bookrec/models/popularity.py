"""인기순 추천 모델. 평점이 많이 달린 책을 모든 독자에게 똑같이 추천한다.

모델 구현체의 예시이자, 다른 모델의 성능을 비교하는 기준선(A6)이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bookrec.models.base import BOOK_IDS, Recommender


class PopularityRecommender(Recommender):
    """책별 평점 개수를 점수로 쓰는 추천 모델. 독자의 이력은 보지 않아 모든 독자의 점수가 같다."""

    def fit(self, train: pd.DataFrame) -> PopularityRecommender:
        """train에서 책마다 평점 개수를 세어 self.counts(책1~책10000 순서, 길이 10,000)에 저장한다."""
        # NOTE: train에 한 번도 나오지 않은 책은 0으로 채운다.
        self.counts = train["book_id"].value_counts().reindex(BOOK_IDS, fill_value=0).to_numpy(dtype=float)
        return self

    def score(self, history: pd.DataFrame) -> pd.DataFrame:
        """이력에 나온 독자마다 self.counts를 한 줄씩 복사해 점수표(행 user_id, 열 책1~책10000)를 만든다."""
        user_ids = np.sort(history["user_id"].unique())
        return pd.DataFrame(np.tile(self.counts, (len(user_ids), 1)), index=user_ids, columns=BOOK_IDS)
