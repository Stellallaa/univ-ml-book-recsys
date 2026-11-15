import numpy as np
import pandas as pd
import pytest

from bookrec.models.base import BOOK_IDS, Recommender
from bookrec.models.hybrid import HybridRecommender


class ArrayRecommender(Recommender):
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def fit(self, train):
        self.fitted = True
        return self

    def score(self, history):
        users = np.sort(history.user_id.unique())
        return pd.DataFrame(np.tile(self.values, (len(users), 1)), index=users, columns=BOOK_IDS)


def _history(sizes):
    return pd.DataFrame([(user, book, 5) for user, size in enumerate(sizes, 1) for book in range(1, size + 1)],
                        columns=["user_id", "book_id", "rating"])


def test_fixed_fuses_percentile_ranks_and_preserves_catalog():
    # Given: CB ranks book 1 first and MF ranks book 2 first.
    cb = ArrayRecommender(np.arange(10_000, 0, -1))
    mf = ArrayRecommender(np.arange(10_000))
    model = HybridRecommender(cb, mf, mode="fixed", mf_weight=0.75)
    history = _history([1])

    # When
    scores = model.score(history)

    # Then
    assert scores.shape == (1, len(BOOK_IDS))
    assert scores.loc[1].idxmax() == 10_000
    assert scores.loc[1, 10_000] == pytest.approx(0.75 * 1 + 0.25 * 0.0001)


def test_switch_and_gradual_modes_follow_history_length():
    # Given: equal-shaped complementary score lists and users with varying histories.
    cb = ArrayRecommender(np.arange(10_000))
    mf = ArrayRecommender(np.arange(10_000, 0, -1))
    history = _history([2, 3, 6])

    # When / Then: switch selects CB below 3 and MF at/above 3.
    switched = HybridRecommender(cb, mf, mode="switch", switch_at=3).score(history)
    assert switched.loc[1].idxmax() == 10_000
    assert switched.loc[2].idxmax() == 1

    # Gradual is 0 MF at 2, 0.25 at 3, and 1 MF at 6.
    gradual = HybridRecommender(cb, mf, mode="gradual", switch_at=2, full_mf_at=6).score(history)
    assert gradual.loc[1].idxmax() == 10_000
    assert gradual.loc[3].idxmax() == 1


def test_invalid_hybrid_configuration_is_rejected():
    cb, mf = ArrayRecommender(np.ones(10_000)), ArrayRecommender(np.ones(10_000))
    with pytest.raises(ValueError):
        HybridRecommender(cb, mf, mf_weight=1.1)
    with pytest.raises(ValueError):
        HybridRecommender(cb, mf, mode="gradual", switch_at=5, full_mf_at=5)
