"""ALS(Alternating Least Squares)로 학습하는 MF. gradient descent로 학습한 MF(mf.py)와 최적화 방법을 비교하는 데 쓴다(C2).

예측 별점은 mf.py와 같다: 전체 평균 + 독자 bias + 책 bias + (독자 벡터 · 책 벡터)
gradient descent는 모든 파라미터를 조금씩 함께 움직인다. ALS는 한쪽을 고정하면 나머지 쪽이 Ridge regression이 된다는 점을 써서,
① 책 쪽을 고정하고 독자마다 [독자 벡터, 독자 bias]를 닫힌 해로 구하고 ② 독자 쪽을 고정하고 책마다 같은 계산을 하는 것을 번갈아 반복한다.
학습률이 필요 없다. 독자 쪽 계산은 mf.py의 fold-in과 같은 식이다.

참고한 자료
- Y. Zhou, D. Wilkinson, R. Schreiber, R. Pan, "Large-Scale Parallel Collaborative Filtering for the Netflix Prize", AAIM, 2008.
  (번갈아 푸는 방식, 정규화 세기에 평점 수를 곱하는 방식)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bookrec.models.base import BOOK_IDS, RatingPredictor

MIN_RATING, MAX_RATING = 1.0, 5.0


class ALSMatrixFactorization(RatingPredictor):
    """ALS로 학습하는 별점 예측 MF."""

    def __init__(self, n_factors: int = 64, reg: float = 0.05, n_epochs: int = 20, patience: int = 2, seed: int = 42):
        self.n_factors, self.reg, self.n_epochs, self.patience, self.seed = n_factors, reg, n_epochs, patience, seed

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> ALSMatrixFactorization:
        """train으로 학습한다. val을 주면 epoch마다 검증 RMSE를 재고, patience번 연속 나아지지 않으면 멈춰 가장 좋았던 epoch로 되돌린다."""
        rng = np.random.default_rng(self.seed)
        self.user_ids = np.sort(train["user_id"].unique())
        self._user_index = pd.Index(self.user_ids)
        u = self._user_index.get_indexer(train["user_id"])
        i = train["book_id"].to_numpy() - 1
        r = train["rating"].to_numpy(dtype=np.float64)
        self.mu = float(r.mean())
        self.bu, self.bi = np.zeros(len(self.user_ids)), np.zeros(len(BOOK_IDS))
        self.P = rng.normal(0, 0.1, (len(self.user_ids), self.n_factors))
        self.Q = rng.normal(0, 0.1, (len(BOOK_IDS), self.n_factors))
        # NOTE: 독자별·책별로 평점이 이어지도록 정렬 순서를 미리 구해 둔다. argsort(kind="stable")는 정렬했을 때의 행 위치를 돌려준다.
        by_user, by_book = np.argsort(u, kind="stable"), np.argsort(i, kind="stable")

        records, best, bad_epochs = [], None, 0
        for epoch in range(1, self.n_epochs + 1):
            # ① 책 쪽 고정 → 독자 쪽: y = 별점 - 전체 평균 - 책 bias 를 [책 벡터, 1]로 설명
            self.P, self.bu = self._solve(u[by_user], i[by_user], self.Q, r[by_user] - self.mu - self.bi[i[by_user]], self.P, self.bu)
            # ② 독자 쪽 고정 → 책 쪽: y = 별점 - 전체 평균 - 독자 bias 를 [독자 벡터, 1]로 설명
            self.Q, self.bi = self._solve(i[by_book], u[by_book], self.P, r[by_book] - self.mu - self.bu[u[by_book]], self.Q, self.bi)
            record = {"epoch": epoch, "train_rmse": _rmse(self.predict(train), r)}
            if val is not None:
                record["val_rmse"] = _rmse(self.predict(val), val["rating"].to_numpy())
            records.append(record)
            if val is None:
                continue
            if best is None or record["val_rmse"] < best[0]:
                best, bad_epochs = (record["val_rmse"], epoch, self.bu.copy(), self.bi.copy(), self.P.copy(), self.Q.copy()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.history = pd.DataFrame(records)
        self.best_epoch = len(records)
        if best is not None:
            _, self.best_epoch, self.bu, self.bi, self.P, self.Q = best
        return self

    def _solve(self, groups: np.ndarray, other_rows: np.ndarray, other: np.ndarray, y: np.ndarray, vectors: np.ndarray, biases: np.ndarray):
        """groups(정렬된 독자 또는 책 번호)의 구간마다 Ridge regression을 풀어 (벡터, bias)를 새로 구한다. 평점이 없는 행은 그대로 둔다.

        닫힌 해: [벡터, bias] = (X^T X + reg * 평점 수 * I)^-1 X^T y,  X = 상대 쪽의 [벡터, 1]
        정규화에 평점 수를 곱해, 평점 하나마다 reg를 거는 gradient descent(mf.py)와 같은 loss가 되게 한다.
        """
        vectors, biases = vectors.copy(), biases.copy()
        # NOTE: 상대 쪽 벡터 끝에 1을 붙인다(1은 bias 자리). 평점 전체에 대해 한 번에 꺼내면 메모리가 수 GB라, 구간마다 필요한 행만 꺼낸다.
        other = np.hstack([other, np.ones((len(other), 1))])
        # NOTE: np.unique(return_index=True)는 (값, 각 값이 처음 나오는 위치)를 돌려준다. 정렬돼 있어 구간의 시작점이다.
        ids, starts = np.unique(groups, return_index=True)
        ends = np.append(starts[1:], len(groups))
        eye = np.eye(self.n_factors + 1)
        for row, s, e in zip(ids, starts, ends):
            # NOTE: np.linalg.solve(A, b)는 A x = b 를 만족하는 x를 구한다.
            X = other[other_rows[s:e]]
            solved = np.linalg.solve(X.T @ X + self.reg * (e - s) * eye, X.T @ y[s:e])
            vectors[row], biases[row] = solved[:-1], solved[-1]
        return vectors, biases

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """(user_id, book_id) 쌍의 예측 별점(1~5로 자름). 학습에 없던 독자는 전체 평균 + 책 bias로 예측한다."""
        u = self._user_index.get_indexer(pairs["user_id"])
        i = pairs["book_id"].to_numpy() - 1
        known = np.flatnonzero(u >= 0)
        pred = self.mu + self.bi[i]
        # NOTE: 쌍 전체의 벡터를 한 번에 꺼내면 (쌍 수 x 차원) 배열이 수 GB가 된다. 50만 쌍씩 나눠 계산한다.
        for start in range(0, len(known), 500_000):
            part = known[start : start + 500_000]
            pred[part] += self.bu[u[part]] + (self.P[u[part]] * self.Q[i[part]]).sum(axis=1)
        return np.clip(pred, MIN_RATING, MAX_RATING)


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))
