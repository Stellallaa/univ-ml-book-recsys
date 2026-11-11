import numpy as np
import pandas as pd
import pytest

from bookrec.data import RatingSplit
from bookrec.evaluation import evaluate_cold, evaluate_warm
from bookrec.models.base import BOOK_IDS
from bookrec.models.content_based.features import build_book_features
from bookrec.models.content_based.recommender import ContentBasedRecommender


def _ratings(rows=()):
    return pd.DataFrame(list(rows), columns=["user_id", "book_id", "rating"])


def _features():
    tags = pd.DataFrame([(1, "fantasy", 1), (2, "fantasy", 1), (3, "mystery", 1), (4, "mystery", 1)], columns=["book_id", "tag_name", "count"])
    return build_book_features(tags, min_df=1)


def test_score_uses_new_user_history_and_preserves_inputs():
    # Given: 학습에는 없는 독자 8과 9가 서로 다른 장르를 좋아한다
    train = _ratings([(1, 4, 5), (2, 4, 4)])
    history = _ratings([(9, 3, 5), (8, 1, 4)])
    original = history.copy(deep=True)
    model = ContentBasedRecommender(_features())

    # When
    fitted = model.fit(train)
    scores = model.score(history)

    # Then
    assert fitted is model
    assert scores.index.tolist() == [8, 9]
    np.testing.assert_array_equal(scores.columns, BOOK_IDS)
    assert scores.loc[8, 2] == 1 and scores.loc[8, 4] == 0
    assert scores.loc[9, 2] == 0 and scores.loc[9, 4] == 1
    pd.testing.assert_frame_equal(history, original)


def test_positive_profile_matches_weighted_cosine_and_ignores_low_ratings():
    # Given: fantasy 가중치 1, mystery 가중치 2, 낮은 별점 책은 제외
    model = ContentBasedRecommender(_features()).fit(_ratings())
    history = _ratings([(7, 1, 4), (7, 3, 5), (7, 2, 1)])

    # When
    scores = model.score(history)

    # Then
    assert scores.loc[7, 2] == pytest.approx(1 / np.sqrt(5))
    assert scores.loc[7, 4] == pytest.approx(2 / np.sqrt(5))
    assert np.isfinite(scores.to_numpy()).all()
    assert ((scores.to_numpy() >= 0) & (scores.to_numpy() <= 1)).all()


def test_rating_profile_includes_all_ratings():
    # Given
    model = ContentBasedRecommender(_features(), profile_weighting="rating").fit(_ratings())

    # When
    scores = model.score(_ratings([(7, 1, 1), (7, 3, 2)]))

    # Then
    assert scores.loc[7, 2] == pytest.approx(1 / np.sqrt(5))
    assert scores.loc[7, 4] == pytest.approx(2 / np.sqrt(5))


def test_fallback_uses_train_only_for_low_ratings_and_missing_features():
    # Given: 책 4가 train에서 두 번, 책 2가 한 번 등장한다
    model = ContentBasedRecommender(_features()).fit(_ratings([(1, 4, 5), (2, 4, 4), (2, 2, 4)]))
    history = _ratings([(7, 1, 2), (8, 100, 5), (9, 1, 5)])

    # When
    scores = model.score(history)
    summary = model.profile_summary(history)

    # Then
    assert scores.loc[7, 4] == scores.loc[8, 4] == 1
    assert scores.loc[7, 2] == scores.loc[8, 2] == 0.5
    assert scores.loc[7, 100] == 0
    assert summary["fallback"].tolist() == [True, True, False]
    assert summary["history_size"].tolist() == [1, 1, 1]


def test_recommend_excludes_history_and_separate_read_books():
    # Given
    model = ContentBasedRecommender(_features()).fit(_ratings())
    history = _ratings([(7, 1, 5)])
    read = _ratings([(7, 1, 5), (7, 2, 5)])

    # When
    default = model.recommend(history, k=1)
    explicit = model.recommend(history, k=1, read=read)

    # Then
    assert default == {7: [2]}
    assert explicit == {7: [3]}


def test_empty_history_and_empty_train_return_finite_scores():
    # Given
    model = ContentBasedRecommender(_features()).fit(_ratings())

    # When
    empty = model.score(_ratings())
    fallback = model.score(_ratings([(7, 1, 1)]))

    # Then
    assert empty.shape == (0, 10000)
    assert not fallback.to_numpy().any()


def test_model_requires_fit_and_valid_configuration():
    # Given
    features = _features()

    # When / Then
    with pytest.raises(RuntimeError):
        ContentBasedRecommender(features).score(_ratings([(7, 1, 5)]))
    with pytest.raises(ValueError):
        ContentBasedRecommender(features, profile_weighting="unknown")
    small = build_book_features(pd.DataFrame([(1, "fantasy", 1)], columns=["book_id", "tag_name", "count"]), min_df=1, book_ids=np.array([1]))
    with pytest.raises(ValueError):
        ContentBasedRecommender(small)


@pytest.mark.parametrize("row", [(7, 0, 5), (7, 10001, 5), (7, 1, np.nan), (7, 1, 6), (None, 1, 5)])
def test_model_rejects_invalid_ratings(row):
    # Given
    model = ContentBasedRecommender(_features()).fit(_ratings())

    # When / Then
    with pytest.raises(ValueError):
        model.score(_ratings([row]))
    with pytest.raises(ValueError):
        model.fit(_ratings([row]))


def test_repeated_fit_replaces_popularity_statistics():
    # Given
    model = ContentBasedRecommender(_features()).fit(_ratings([(1, 4, 5)]))

    # When
    model.fit(_ratings([(1, 2, 5)]))
    scores = model.score(_ratings([(7, 1, 1)]))

    # Then
    assert scores.loc[7, 2] == 1 and scores.loc[7, 4] == 0


def test_model_integrates_with_warm_and_cold_evaluation():
    # Given: 신규 독자는 학습에 없고 1개/2개 이력 모두 같은 3번째 이후 정답을 평가한다
    split = RatingSplit(
        train=_ratings([(1, 1, 5)]), val=_ratings([(1, 2, 5)]), test=_ratings(),
        cold_tune=_ratings([(7, 1, 5), (7, 3, 1), (7, 2, 5)]), cold_final=_ratings(),
    )
    model = ContentBasedRecommender(_features()).fit(split.train)

    # When
    warm = evaluate_warm(model, split, on="val", k=1)
    cold = evaluate_cold(model, split, on="tune", history_sizes=(1, 2), k=1)

    # Then
    assert warm["precision@1"] == 1 and warm["ndcg@1"] == 1
    assert cold["precision@1"].tolist() == [1, 1]
    assert cold["ndcg@1"].tolist() == [1, 1]
