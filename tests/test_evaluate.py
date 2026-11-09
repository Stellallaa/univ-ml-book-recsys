import numpy as np
import pandas as pd
import pytest

from bookrec.data import RatingSplit
from bookrec.evaluation import evaluate as evaluate_module
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


def _ratings(rows=()):
    return pd.DataFrame(list(rows), columns=["user_id", "book_id", "rating"])


def _split(**tables):
    return RatingSplit(**{name: tables.get(name, _ratings()) for name in ["train", "val", "test", "cold_tune", "cold_final"]})


def test_warm_computes_precision_recall_ndcg_from_hit_ranks():
    # Given: 추천은 1~10번 책, 4점 이상 정답 8권 중 2·5·10위가 적중, 1번 책은 3점이라 정답이 아님
    split = _split(
        train=_ratings([(1, 100, 5)]),
        val=_ratings([(1, 1, 3), (1, 2, 4), (1, 5, 5), (1, 10, 4)] + [(1, b, 5) for b in range(20, 25)]),
    )

    # When
    result = evaluate_warm(FixedRecommender(), split, on="val")

    # Then
    dcg = 1 / np.log2(3) + 1 / np.log2(6) + 1 / np.log2(11)
    idcg = sum(1 / np.log2(r + 1) for r in range(1, 9))
    assert result["precision@10"] == pytest.approx(0.3)
    assert result["recall@10"] == pytest.approx(0.375)
    assert result["ndcg@10"] == pytest.approx(dcg / idcg)
    assert result["ndcg@10"] == pytest.approx(0.33, abs=0.01)


def test_warm_excludes_user_without_answers_from_average():
    # Given: 독자 2는 4점 이상 정답이 없다
    split = _split(train=_ratings([(1, 100, 5), (2, 100, 5)]), val=_ratings([(1, 1, 5), (2, 1, 3)]))

    # When
    result = evaluate_warm(FixedRecommender(), split, on="val")

    # Then
    assert result["users"] == 1
    assert result["precision@10"] == pytest.approx(0.1)
    assert result["recall@10"] == pytest.approx(1.0)


def test_warm_to_read_answers_exclude_books_in_history():
    # Given: 독자 1은 to_read 중 4번만 이력에 없고, 독자 2는 to_read가 전부 이력에 있다
    split = _split(train=_ratings([(1, 3, 5), (2, 7, 5)]), val=_ratings([(1, 1, 5), (2, 1, 5)]))
    to_read = pd.DataFrame({"user_id": [1, 1, 2], "book_id": [3, 4, 7]})

    # When
    result = evaluate_warm(FixedRecommender(), split, on="val", to_read=to_read)

    # Then
    assert result["to_read_users"] == 1
    assert result["to_read_hit@10"] == pytest.approx(0.1)


def test_warm_on_test_uses_train_and_val_as_history():
    # Given: val의 1번 책은 이미 읽은 책이고, test의 정답은 2번 책이다
    split = _split(train=_ratings([(1, 100, 5)]), val=_ratings([(1, 1, 5)]), test=_ratings([(1, 2, 5)]))
    model = FixedRecommender()

    # When
    result = evaluate_warm(model, split, on="test")

    # Then: 모델은 train + val을 이력으로 받고, 1번 책이 추천에서 빠져 2번 책이 1위로 적중한다
    assert model.seen[0]["book_id"].tolist() == [100, 1]
    assert result["ndcg@10"] == pytest.approx(1.0)


def test_warm_merges_recommendations_across_batches(monkeypatch):
    # Given: 독자 3명, 한 묶음은 2명
    monkeypatch.setattr(evaluate_module, "BATCH_SIZE", 2)
    split = _split(train=_ratings([(u, 100, 5) for u in (1, 2, 3)]), val=_ratings([(u, 1, 5) for u in (1, 2, 3)]))
    model = FixedRecommender()

    # When
    result = evaluate_warm(model, split, on="val")

    # Then: 모델은 두 번 불리고, 세 명 모두 채점된다
    assert [len(seen["user_id"].unique()) for seen in model.seen] == [2, 1]
    assert result["users"] == 3


def test_warm_n_users_samples_same_users_every_time():
    # Given: 독자 10명
    split = _split(train=_ratings([(u, 100, 5) for u in range(10)]), val=_ratings([(u, 1, 5) for u in range(10)]))
    models = [FixedRecommender(), FixedRecommender()]

    # When
    results = [evaluate_warm(model, split, on="val", n_users=3) for model in models]

    # Then: 3명만 채점하고, 두 번 모두 같은 독자가 뽑힌다
    users = [sorted(model.seen[0]["user_id"].unique()) for model in models]
    assert [result["users"] for result in results] == [3, 3]
    assert users[0] == users[1]


def test_warm_rejects_unknown_target():
    # Given
    split = _split(train=_ratings([(1, 100, 5)]))

    # When / Then: train으로는 채점하지 않는다
    with pytest.raises(ValueError):
        evaluate_warm(FixedRecommender(), split, on="train")


def test_cold_uses_first_ratings_in_file_order_and_same_answers_for_every_size():
    # Given: 정답은 크기와 상관없이 3번째 평점부터 4점 이상(1번, 30번 책)이다
    split = _split(cold_tune=_ratings([(7, 50, 5), (7, 3, 5), (7, 1, 5), (7, 2, 3), (7, 30, 4)]))
    model = FixedRecommender()

    # When
    result = evaluate_cold(model, split, on="tune", history_sizes=(1, 2))

    # Then
    assert [seen["book_id"].tolist() for seen in model.seen] == [[50], [50, 3]]
    assert result["precision@10"].tolist() == pytest.approx([0.1, 0.1])
    assert result["recall@10"].tolist() == pytest.approx([0.5, 0.5])
    assert result["users"].tolist() == [1, 1]


def test_cold_excludes_books_read_before_answers_even_when_not_given():
    # Given: 앞 2개(50번, 2번)가 이미 읽은 책이고 정답은 11번 책 하나다
    split = _split(cold_tune=_ratings([(7, 50, 5), (7, 2, 5), (7, 11, 4)]))
    model = FixedRecommender()

    # When
    result = evaluate_cold(model, split, on="tune", history_sizes=(1, 2))

    # Then: 이력 1개일 때도 모델은 50번만 받지만, 2번 책이 추천에서 빠져 11번 책이 10위로 적중한다
    assert [seen["book_id"].tolist() for seen in model.seen] == [[50], [50, 2]]
    assert result["precision@10"].tolist() == pytest.approx([0.1, 0.1])


def test_cold_to_read_answers_exclude_all_books_read_before_answers():
    # Given: to_read는 3번, 4번 책이고 3번 책은 2번째 평점이다
    split = _split(cold_tune=_ratings([(7, 50, 5), (7, 3, 5), (7, 1, 5)]))
    to_read = pd.DataFrame({"user_id": [7, 7], "book_id": [3, 4]})

    # When
    result = evaluate_cold(FixedRecommender(), split, on="tune", to_read=to_read, history_sizes=(1, 2))

    # Then: 이력 크기와 상관없이 3번 책은 이미 읽은 책이라 4번만 정답이다(1, 2, 4, …위 중 3위 적중)
    assert result["to_read_hit@10"].tolist() == pytest.approx([0.1, 0.1])
    assert result["to_read_users"].tolist() == [1, 1]
    assert "ndcg@10" in result.columns


def test_rmse_on_known_errors():
    # Given: 오차 1, -1, 0, -2
    split = _split(test=_ratings([(1, 1, 5), (1, 2, 3), (2, 1, 4), (2, 2, 2)]))

    # When
    rmse = evaluate_rmse(ConstantPredictor(), split, on="test")

    # Then
    assert rmse == pytest.approx(np.sqrt(1.5))
