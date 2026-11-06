"""평점을 신규 독자 실험용 독자와 나머지 독자의 학습·검증·평가 구간으로 나눈다(A4)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bookrec.data.loaders import RAW_DIR, load_ratings

N_COLD_USERS = 5000
SEED = 42
# NOTE: 180 * 0.70 = 125.99...처럼 실수 곱이 내림에서 틀어져 정수 백분율로 둔다.
TRAIN_PERCENT = 70
VAL_PERCENT = 15


@dataclass(frozen=True)
class RatingSplit:
    """분할된 평점 표 다섯 개. 모두 원본 행 순서를 유지한다."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cold_tune: pd.DataFrame
    cold_final: pd.DataFrame


def split_ratings(ratings: pd.DataFrame, n_cold_users: int = N_COLD_USERS, seed: int = SEED) -> RatingSplit:
    """신규 독자 실험용 독자를 먼저 떼고, 나머지는 독자별 행 순서대로 70/15/15로 나눈다."""
    cold = np.random.default_rng(seed).choice(ratings["user_id"].unique(), size=n_cold_users, replace=False)
    half = n_cold_users // 2
    is_cold = ratings["user_id"].isin(cold)
    warm = ratings[~is_cold]
    position = warm.groupby("user_id").cumcount()
    count = warm.groupby("user_id")["user_id"].transform("size")
    train_end = count * TRAIN_PERCENT // 100
    val_end = count * (TRAIN_PERCENT + VAL_PERCENT) // 100
    return RatingSplit(
        train=warm[position < train_end],
        val=warm[(position >= train_end) & (position < val_end)],
        test=warm[position >= val_end],
        cold_tune=ratings[ratings["user_id"].isin(cold[:half])],
        cold_final=ratings[ratings["user_id"].isin(cold[half:])],
    )


def load_split(raw_dir: Path = RAW_DIR, n_cold_users: int = N_COLD_USERS, seed: int = SEED) -> RatingSplit:
    """원본 평점을 읽어 `split_ratings`로 나눈다."""
    return split_ratings(load_ratings(raw_dir), n_cold_users, seed)
