import numpy as np
import pandas as pd
import pytest

from bookrec.evaluation import evaluate_cold, evaluate_rmse, evaluate_warm
from bookrec.models.base import BOOK_IDS, RatingPredictor, Recommender


class FixedRecommender(Recommender):
    """book_id가 작을수록 점수가 높다. 받은 이력을 seen에 남긴다."""

    def __init__(self):
        self.seen = []

    def fit(self, train):
        return self

    def score(self, history):
        self.seen.append(history)
        user_ids = np.sort(history["user_id"].unique())
        return pd.DataFrame(np.tile(-BOOK_IDS.astype(float), (len(user_ids), 1)), index=user_ids, columns=BOOK_IDS)


class ConstantPredictor(RatingPredictor):
    def predict(self, pairs):
        return np.full(len(pairs), 4.0)


def _ratings(rows):
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])


def test_warm_computes_precision_recall_ndcg_from_hit_ranks():
    # Given: 추천은 1~10번 책, 4점 이상 정답 8권 중 2·5·10위가 적중, 1번 책은 3점이라 정답이 아님
    history = _ratings([(1, 100, 5)])
    answers = _ratings([(1, 1, 3), (1, 2, 4), (1, 5, 5), (1, 10, 4)] + [(1, b, 5) for b in range(20, 25)])

    # When
    result = evaluate_warm(FixedRecommender(), history, answers)

    # Then
    dcg = 1 / np.log2(3) + 1 / np.log2(6) + 1 / np.log2(11)
    idcg = sum(1 / np.log2(r + 1) for r in range(1, 9))
    assert result["precision@10"] == pytest.approx(0.3)
    assert result["recall@10"] == pytest.approx(0.375)
    assert result["ndcg@10"] == pytest.approx(dcg / idcg)
    assert result["ndcg@10"] == pytest.approx(0.33, abs=0.01)


def test_warm_excludes_user_without_answers_from_average():
    # Given: 독자 2는 4점 이상 정답이 없다
    history = _ratings([(1, 100, 5), (2, 100, 5)])
    answers = _ratings([(1, 1, 5), (2, 1, 3)])

    # When
    result = evaluate_warm(FixedRecommender(), history, answers)

    # Then
    assert result["users"] == 1
    assert result["precision@10"] == pytest.approx(0.1)
    assert result["recall@10"] == pytest.approx(1.0)


def test_warm_to_read_answers_exclude_books_in_history():
    # Given: 독자 1은 to_read 중 4번만 이력에 없고, 독자 2는 to_read가 전부 이력에 있다
    history = _ratings([(1, 3, 5), (2, 7, 5)])
    answers = _ratings([(1, 1, 5), (2, 1, 5)])
    to_read = pd.DataFrame({"user_id": [1, 1, 2], "book_id": [3, 4, 7]})

    # When
    result = evaluate_warm(FixedRecommender(), history, answers, to_read=to_read)

    # Then
    assert result["to_read_users"] == 1
    assert result["to_read_hit@10"] == pytest.approx(0.1)


def test_cold_uses_first_ratings_in_file_order_and_same_answers_for_every_size():
    # Given: 정답은 크기와 상관없이 3번째 평점부터 4점 이상(1번, 30번 책)이다
    cold = _ratings([(7, 50, 5), (7, 3, 5), (7, 1, 5), (7, 2, 3), (7, 30, 4)])
    model = FixedRecommender()

    # When
    result = evaluate_cold(model, cold, history_sizes=(1, 2))

    # Then
    assert [seen["book_id"].tolist() for seen in model.seen] == [[50], [50, 3]]
    assert result["precision@10"].tolist() == pytest.approx([0.1, 0.1])
    assert result["recall@10"].tolist() == pytest.approx([0.5, 0.5])
    assert result["users"].tolist() == [1, 1]


def test_rmse_on_known_errors():
    # Given: 오차 1, -1, 0, -2
    ratings = _ratings([(1, 1, 5), (1, 2, 3), (2, 1, 4), (2, 2, 2)])

    # When
    rmse = evaluate_rmse(ConstantPredictor(), ratings)

    # Then
    assert rmse == pytest.approx(np.sqrt(1.5))
