"""Matrix Factorization(MF)으로 별점을 예측하고 추천 점수를 만든다(C1 학습, C3 fold-in).

예측값 = 전체 평균 + 독자 bias + 책 bias + (독자 벡터 · 책 벡터)

- 학습(C1): 수업에서 배운 mini-batch gradient descent로 (정답 - 예측값)^2 + L2 정규화를 줄인다. NumPy로 직접 구현했다.
- neg_ratio(C2): 0이면 평점이 있는 칸만 학습한다(별점 예측용, RMSE가 좋다).
  1 이상이면 평점 1개마다 무작위 책 neg_ratio권을 "정답 0점"으로 함께 학습한다(추천용, top-10 적중률이 좋다).
  평점이 있는 칸만 학습하면 "이 독자가 이 책을 읽을지"는 배우지 못해서, 예측 별점 순으로 추천하면 적중률이 인기순보다 낮다.
- fold-in(C3): 책 벡터·책 bias는 고정하고, 이력에 있는 평점만으로 독자 벡터·독자 bias를 Ridge regression의 닫힌 해로 구한다.
  학습에 없던 신규 독자도 평점 1개만 있으면 점수를 낼 수 있다.

참고한 자료
- Y. Koren, R. Bell, C. Volinsky, "Matrix Factorization Techniques for Recommender Systems", IEEE Computer, 2009.
  (bias를 포함한 예측식과 gradient descent 갱신식)
- Y. Hu, Y. Koren, C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets", ICDM, 2008.
  (평점 없는 칸을 낮은 가중치의 0으로 보는 방식, 책 벡터를 고정하고 독자 벡터를 닫힌 해로 구하는 식)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from bookrec.models.base import BOOK_IDS, RatingPredictor, Recommender

MIN_RATING, MAX_RATING = 1.0, 5.0
N_BOOKS = len(BOOK_IDS)


@dataclass(frozen=True)
class MFConfig:
    """MF 설정. C2에서 값을 바꿔 가며 비교한다."""

    n_factors: int = 32  # 독자·책 벡터의 길이(잠재 차원)
    lr: float = 0.01  # 학습률
    reg: float = 0.05  # L2 정규화 세기(학습)
    n_epochs: int = 30  # 최대 epoch 수
    batch_size: int = 2048  # mini-batch 한 묶음의 평점 수
    neg_ratio: int = 0  # 평점 1개마다 "정답 0점"으로 함께 학습할 무작위 책의 수. 0이면 평점만 학습한다
    patience: int = 2  # 검증 점수가 이 횟수만큼 연속으로 나아지지 않으면 학습을 멈춘다(early stopping)
    fold_in_reg: float = 5.0  # fold-in에서 독자 벡터·독자 bias에 거는 L2 정규화 세기(Ridge의 alpha)
    seed: int = 42


class MatrixFactorization(Recommender, RatingPredictor):
    """bias를 포함한 MF. score는 책 1만 권의 예측값이고, predict는 1~5로 자른 예측 별점이다."""

    def __init__(self, config: MFConfig | None = None, **overrides):
        # NOTE: MatrixFactorization(n_factors=64)처럼 일부 설정만 바꿔 만들 수 있게 한다.
        self.config = MFConfig(**{**asdict(config or MFConfig()), **overrides})

    # ------------------------------------------------------------------ 학습 (C1)

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        monitor: Callable[[MatrixFactorization], float] | None = None,
        verbose: bool = False,
    ) -> MatrixFactorization:
        """train으로 학습한다. epoch별 기록은 self.history(표)에 남고, 학습 곡선은 이 표로 그린다.

        - val을 주면 epoch마다 검증 RMSE(val_rmse)를 기록한다.
        - 검증 점수가 patience번 연속으로 나아지지 않으면 멈추고, 가장 좋았던 epoch의 파라미터로 되돌린다.
          검증 점수는 monitor(모델) → 숫자(클수록 좋음, 기록 이름 monitor)를 주면 그 값이고, 없으면 -val_rmse이다.
        - val도 monitor도 없으면 n_epochs만큼 학습한다.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        # NOTE: 독자 ID를 0부터 시작하는 행 번호로 바꾼다. 책은 BOOK_IDS(1~10000) 순서라 book_id - 1이 행 번호다.
        self.user_ids = np.sort(train["user_id"].unique())
        self._user_index = pd.Index(self.user_ids)
        u = self._user_index.get_indexer(train["user_id"])
        i = train["book_id"].to_numpy() - 1
        r = train["rating"].to_numpy(dtype=np.float64)

        # NOTE: 정답 0점인 칸이 평점의 neg_ratio배 섞이므로, 학습하는 정답 전체의 평균은 평점 평균 / (1 + neg_ratio)이다.
        self.mu = float(r.mean()) / (1 + cfg.neg_ratio)
        self.bu = np.zeros(len(self.user_ids))
        self.bi = np.zeros(N_BOOKS)
        # NOTE: 벡터를 0으로 시작하면 기울기도 0이라 학습되지 않는다. 작은 난수(평균 0, 표준편차 0.1)로 시작한다.
        self.P = rng.normal(0, 0.1, (len(self.user_ids), cfg.n_factors))
        self.Q = rng.normal(0, 0.1, (N_BOOKS, cfg.n_factors))

        records, best, bad_epochs = [], None, 0
        for epoch in range(1, cfg.n_epochs + 1):
            # NOTE: permutation(n)은 0~n-1을 무작위로 섞은 배열이다. epoch마다 평점 순서를 섞어 묶음을 만든다.
            order = rng.permutation(len(r))
            squared_error, n_seen = 0.0, 0
            for start in range(0, len(r), cfg.batch_size):
                batch = order[start : start + cfg.batch_size]
                ub, ib, rb = u[batch], i[batch], r[batch]
                if cfg.neg_ratio:
                    # NOTE: 같은 독자에 무작위 책(rng.integers(0, 책 수, 개수))을 붙이고 정답을 0으로 둔다.
                    #       1만 권 중 독자가 읽은 책은 평균 80권 안팎이라 대부분 안 읽은 책이 뽑힌다. 읽은 책이 뽑히는 경우는 걸러 내지 않는다.
                    n_neg = len(batch) * cfg.neg_ratio
                    ub = np.concatenate([ub, np.repeat(ub, cfg.neg_ratio)])
                    ib = np.concatenate([ib, rng.integers(0, N_BOOKS, n_neg)])
                    rb = np.concatenate([rb, np.zeros(n_neg)])
                squared_error += self._gradient_step(ub, ib, rb)
                n_seen += len(rb)
            # NOTE: train_rmse는 epoch 도중(갱신 직전) 묶음들의 오차로 계산한다. 평점 전체를 다시 예측하는 시간을 아끼기 위해서다.
            record = {"epoch": epoch, "train_rmse": float(np.sqrt(squared_error / n_seen))}
            if val is not None or monitor is not None:
                self._prepare_fold_in()
            if val is not None:
                record["val_rmse"] = _rmse(self.predict(val), val["rating"].to_numpy())
            if monitor is not None:
                record["monitor"] = float(monitor(self))
            records.append(record)
            if verbose:
                print(record)
            if val is None and monitor is None:
                continue
            value = record["monitor"] if monitor is not None else -record["val_rmse"]
            if best is None or value > best[0]:
                # NOTE: 가장 좋았던 시점의 파라미터를 복사해 둔다(copy를 안 하면 다음 epoch의 갱신이 그대로 반영된다).
                best, bad_epochs = (value, epoch, self.bu.copy(), self.bi.copy(), self.P.copy(), self.Q.copy()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break
        self.history = pd.DataFrame(records)
        self.best_epoch = len(records)
        if best is not None:
            _, self.best_epoch, self.bu, self.bi, self.P, self.Q = best
        self._prepare_fold_in()
        return self

    def _gradient_step(self, u: np.ndarray, i: np.ndarray, r: np.ndarray) -> float:
        """정답 한 묶음으로 gradient descent를 한 번 하고, 갱신 전 오차 제곱의 합을 돌려준다.

        정답 하나의 loss = (r - 예측)^2 + reg * (bu^2 + bi^2 + |p|^2 + |q|^2) 이고, err = r - 예측일 때
        bu ← bu + lr * (err - reg * bu),  p ← p + lr * (err * q - reg * p)  (책 쪽 bi, q도 같은 모양)
        """
        cfg = self.config
        p, q = self.P[u], self.Q[i]
        # NOTE: (p * q).sum(axis=1)은 행마다 두 벡터의 내적이다.
        err = r - (self.mu + self.bu[u] + self.bi[i] + (p * q).sum(axis=1))
        # NOTE: 한 묶음에 같은 독자·책이 여러 번 나온다. self.P[u] += ... 는 같은 행에 한 번만 더하므로,
        #       np.add.at(배열, 위치, 값)으로 같은 위치에 해당하는 값을 모두 더한다.
        np.add.at(self.bu, u, cfg.lr * (err - cfg.reg * self.bu[u]))
        np.add.at(self.bi, i, cfg.lr * (err - cfg.reg * self.bi[i]))
        np.add.at(self.P, u, cfg.lr * (err[:, None] * q - cfg.reg * p))
        np.add.at(self.Q, i, cfg.lr * (err[:, None] * p - cfg.reg * q))
        return float((err**2).sum())

    # ------------------------------------------------------------------ 별점 예측

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """(user_id, book_id) 쌍의 예측 별점(1~5로 자름). 학습에 없던 독자는 독자 항 없이 전체 평균 + 책 bias로 예측한다.

        별점 예측(RMSE)에는 neg_ratio=0으로 학습한 모델을 쓴다. neg_ratio가 1 이상이면 0점이 섞여 예측값이 별점보다 낮다.
        """
        u = self._user_index.get_indexer(pairs["user_id"])
        i = pairs["book_id"].to_numpy() - 1
        known = u >= 0
        pred = self.mu + self.bi[i]
        pred[known] += self.bu[u[known]] + (self.P[u[known]] * self.Q[i[known]]).sum(axis=1)
        return np.clip(pred, MIN_RATING, MAX_RATING)

    # ------------------------------------------------------------------ fold-in (C3)

    def _prepare_fold_in(self) -> None:
        """fold-in에서 독자마다 다시 계산하지 않도록, 책 전체에 대한 합을 미리 구해 둔다."""
        # NOTE: 책 벡터 끝에 1을 붙인다(1은 독자 bias 자리). X = [q, 1]이면 [p, bu] · X = p·q + bu 이다.
        self._X = np.hstack([self.Q, np.ones((N_BOOKS, 1))])
        # NOTE: 책 1만 권 모두를 "정답 0점"으로 볼 때의 X^T X 와 X^T y (y = 0 - 전체 평균 - 책 bias)
        self._gram_all = self._X.T @ self._X
        self._target_all = self._X.T @ (-self.mu - self.bi)

    def fold_in(self, history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """이력만으로 독자별 (user_ids, 독자 벡터, 독자 bias)를 구한다. 책 벡터·책 bias는 고정한다.

        독자 한 명에 대해 y = 정답 - 전체 평균 - 책 bias 를 X = [책 벡터, 1]로 설명하는 Ridge regression을 푼다.
            최소화: sum 가중치 * (y - [p, bu]·X)^2 + fold_in_reg * (|p|^2 + bu^2)
            닫힌 해: [p, bu] = (X^T W X + fold_in_reg * I)^-1 X^T W y
        이력의 책은 정답 = 별점, 가중치 1이다. 나머지 책은 학습 때와 같게 정답 0, 가중치 w0 = neg_ratio * 이력 수 / 책 수로 둔다
        (학습 때 독자가 평점 n개면 0점 칸 neg_ratio * n개가 책 1만 권에 고르게 뽑힌 것과 같은 비중). neg_ratio = 0이면 이력의 책만 쓴다.
        """
        cfg = self.config
        users = np.sort(history["user_id"].unique())
        # NOTE: 독자별로 행이 이어지도록 정렬한다. kind="stable"은 같은 독자 안에서 원래 순서를 지킨다.
        history = history.sort_values("user_id", kind="stable")
        i = history["book_id"].to_numpy() - 1
        base = self.mu + self.bi[i]
        rating = history["rating"].to_numpy(dtype=np.float64)
        X = self._X[i]
        # NOTE: np.unique(return_index=True)는 각 값이 처음 나오는 위치를 함께 돌려준다. 정렬했으므로 독자별 구간의 시작점이다.
        starts = np.unique(history["user_id"].to_numpy(), return_index=True)[1]
        ends = np.append(starts[1:], len(i))
        penalty = cfg.fold_in_reg * np.eye(cfg.n_factors + 1)
        solved = np.empty((len(users), cfg.n_factors + 1))
        for row, (s, e) in enumerate(zip(starts, ends)):
            w0 = cfg.neg_ratio * (e - s) / N_BOOKS
            # NOTE: "책 전체를 가중치 w0의 0점으로 본 합"에서 이력의 책만 가중치 1의 실제 별점으로 바꿔 넣는다.
            #       이력의 책 몫: X^T X는 w0 → 1 이므로 (1 - w0)배를 더하고, X^T y는 (별점 - base) - w0 * (0 - base)를 더한다.
            gram = w0 * self._gram_all + (1 - w0) * X[s:e].T @ X[s:e] + penalty
            target = w0 * self._target_all + X[s:e].T @ (rating[s:e] - (1 - w0) * base[s:e])
            # NOTE: np.linalg.solve(A, b)는 A x = b 를 만족하는 x를 구한다(역행렬을 직접 구하는 것보다 빠르고 정확하다).
            solved[row] = np.linalg.solve(gram, target)
        return users, solved[:, :-1], solved[:, -1]

    # ------------------------------------------------------------------ 추천 점수

    def score(self, history: pd.DataFrame) -> pd.DataFrame:
        """이력의 독자마다 책 1만 권의 예측값을 점수표(행 user_id, 열 BOOK_IDS)로 돌려준다.

        학습 때 있던 독자든 신규 독자든 모두 넘어온 이력으로 fold-in해서 독자 벡터를 구한다.
        최종 채점(test)에서는 이력에 val이 추가되는데, 다시 학습하지 않아도 그 평점이 독자 벡터에 반영된다.
        """
        users, p, bu = self.fold_in(history)
        return pd.DataFrame(self.mu + bu[:, None] + self.bi[None, :] + p @ self.Q.T, index=users, columns=BOOK_IDS)

    def score_learned(self, history: pd.DataFrame) -> pd.DataFrame:
        """학습으로 얻은 독자 벡터를 그대로 쓴 점수표. fold-in과 비교하는 실험(C3)용이고, 학습에 있던 독자만 넣을 수 있다."""
        users = np.sort(history["user_id"].unique())
        rows = self._user_index.get_indexer(users)
        if (rows < 0).any():
            raise ValueError("학습에 없던 독자가 있습니다. score(history)를 쓰세요.")
        return pd.DataFrame(self.mu + self.bu[rows][:, None] + self.bi[None, :] + self.P[rows] @ self.Q.T, index=users, columns=BOOK_IDS)


    # ------------------------------------------------------------------ 저장·불러오기

    def save(self, path) -> None:
        """학습된 파라미터와 설정을 .npz 파일 하나로 저장한다. 같은 모델을 여러 노트북에서 다시 학습하지 않기 위해 쓴다."""
        # NOTE: np.savez(경로, 이름=배열, ...)는 여러 배열을 이름과 함께 파일 하나에 저장한다. 설정은 JSON 문자열로 함께 넣는다.
        np.savez(path, mu=self.mu, bu=self.bu, bi=self.bi, P=self.P, Q=self.Q, user_ids=self.user_ids, config=json.dumps(asdict(self.config)))

    @classmethod
    def load(cls, path) -> MatrixFactorization:
        """save로 저장한 파일에서 학습된 모델을 되살린다. epoch별 기록(history)은 저장하지 않는다."""
        # NOTE: np.load(경로)는 저장된 배열을 이름으로 꺼낼 수 있는 객체를 돌려준다. with로 열어 다 읽은 뒤 파일을 닫는다.
        with np.load(path) as saved:
            model = cls(MFConfig(**json.loads(str(saved["config"]))))
            model.mu = float(saved["mu"])
            model.bu, model.bi, model.P, model.Q, model.user_ids = saved["bu"], saved["bi"], saved["P"], saved["Q"], saved["user_ids"]
        model._user_index = pd.Index(model.user_ids)
        model._prepare_fold_in()
        return model


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))
