import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from bookrec.models.base import BOOK_IDS
from bookrec.models.collaborative_filtering import MatrixFactorization


def _ratings(rows):
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])


def _two_taste_train():
    """독자 1~20은 책 1~5에 5점·책 6~10에 1점, 독자 21~40은 반대로 준 학습 표. 각자 책 1권씩은 평가하지 않는다."""
    rows = []
    for user in range(1, 41):
        likes_first_group = user <= 20
        for book in range(1, 11):
            if book == (user % 10) + 1:
                continue
            rows.append((user, book, 5 if (book <= 5) == likes_first_group else 1))
    return _ratings(rows)


def _fit(**overrides):
    config = {"n_factors": 4, "lr": 0.05, "reg": 0.01, "n_epochs": 60, "batch_size": 64, "fold_in_reg": 0.1} | overrides
    return MatrixFactorization(**config).fit(_two_taste_train())


def test_score_covers_whole_catalog_for_each_user():
    # Given
    model = _fit(n_epochs=2)
    history = _ratings([(7, 1, 5), (8, 6, 4), (8, 7, 5)])

    # When
    scores = model.score(history)

    # Then
    assert scores.index.tolist() == [7, 8]
    assert scores.columns.tolist() == BOOK_IDS.tolist()
    assert np.isfinite(scores.to_numpy()).all()


def test_fit_lowers_training_error_over_epochs():
    # Given
    model = _fit()

    # When
    history = model.history

    # Then
    assert history["epoch"].tolist() == list(range(1, 61))
    assert history["train_rmse"].iloc[-1] < history["train_rmse"].iloc[0] / 2


def test_predict_recovers_ratings_and_stays_in_rating_range():
    # Given
    train = _two_taste_train()
    model = _fit()

    # When
    predicted = model.predict(train[["user_id", "book_id"]])

    # Then
    assert predicted.min() >= 1 and predicted.max() <= 5
    assert np.sqrt(np.mean((predicted - train["rating"]) ** 2)) < 0.5


def test_predict_fills_in_unrated_books_by_taste():
    # Given: 독자 1은 책 2(자기 취향 묶음)를, 독자 25는 책 6(자기 취향 묶음)을 평가하지 않았다
    model = _fit()

    # When
    predicted = model.predict(pd.DataFrame({"user_id": [1, 25], "book_id": [2, 6]}))

    # Then
    assert (predicted > 4).all()


def test_predict_uses_global_mean_and_book_bias_for_unknown_user():
    # Given
    model = _fit(n_epochs=5)

    # When
    predicted = model.predict(pd.DataFrame({"user_id": [999], "book_id": [3]}))

    # Then
    assert predicted[0] == pytest.approx(np.clip(model.mu + model.bi[2], 1, 5))


def test_score_ranks_books_by_taste_for_new_user_with_one_rating():
    # Given: 학습에 없던 독자 100이 책 1에 5점 하나만 남겼다
    model = _fit()
    history = _ratings([(100, 1, 5)])

    # When
    scores = model.score(history).loc[100]

    # Then: 같은 묶음의 책(2~5)이 반대 묶음의 책(6~10)보다 모두 점수가 높다
    assert scores[[2, 3, 4, 5]].min() > scores[[6, 7, 8, 9, 10]].max()


def test_recommend_differs_by_history_of_new_users():
    # Given
    model = _fit()
    history = _ratings([(100, 1, 5), (100, 6, 1), (200, 1, 1), (200, 6, 5)])

    # When
    recs = model.recommend(history, k=4)

    # Then
    assert sorted(recs[100]) == [2, 3, 4, 5]
    assert sorted(recs[200]) == [7, 8, 9, 10]


def test_fold_in_matches_ridge_regression_on_rated_books():
    # Given
    model = _fit(fold_in_reg=2.0)
    history = _ratings([(100, 1, 5), (100, 2, 4), (100, 6, 1), (100, 7, 2), (100, 9, 1)])
    books = history["book_id"].to_numpy() - 1
    X = np.hstack([model.Q[books], np.ones((len(books), 1))])
    y = history["rating"].to_numpy() - model.mu - model.bi[books]

    # When
    users, vectors, biases = model.fold_in(history)

    # Then: 절편 없이 [책 벡터, 1]을 입력으로 한 Ridge(alpha=fold_in_reg)의 계수와 같다
    expected = Ridge(alpha=2.0, fit_intercept=False).fit(X, y).coef_
    assert users.tolist() == [100]
    assert np.append(vectors[0], biases[0]) == pytest.approx(expected)


def test_score_does_not_depend_on_row_order_of_history():
    # Given
    model = _fit(n_epochs=5)
    history = _ratings([(200, 6, 5), (100, 1, 5), (200, 7, 4), (100, 6, 1)])

    # When
    scores = model.score(history)
    shuffled = model.score(history.iloc[::-1])

    # Then
    assert scores.index.tolist() == [100, 200]
    assert np.allclose(scores.to_numpy(), shuffled.to_numpy())


def test_fit_stops_early_and_keeps_best_epoch_when_validation_worsens():
    # Given: 학습 표와 정반대 별점인 검증 표(학습할수록 검증 오차가 커진다)
    train = _two_taste_train()
    val = train.assign(rating=6 - train["rating"])
    model = MatrixFactorization(n_factors=4, lr=0.05, reg=0.01, n_epochs=30, batch_size=64, patience=2)

    # When
    model.fit(train, val=val)

    # Then
    assert model.best_epoch == 1
    assert len(model.history) == 3
    assert np.sqrt(np.mean((model.predict(val) - val["rating"]) ** 2)) == pytest.approx(model.history["val_rmse"].iloc[0])


def test_fit_uses_monitor_for_early_stopping():
    # Given: epoch 2에서 가장 높고 그 뒤로 떨어지는 검증 점수
    values = iter([0.1, 0.5, 0.4, 0.3, 0.2])
    model = MatrixFactorization(n_factors=4, n_epochs=30, batch_size=64, patience=2)

    # When
    model.fit(_two_taste_train(), monitor=lambda _: next(values))

    # Then
    assert model.best_epoch == 2
    assert model.history["monitor"].tolist() == [0.1, 0.5, 0.4, 0.3]


def test_negative_sampling_lowers_scores_of_books_nobody_rated():
    # Given: 학습 표의 책은 1~10뿐이다
    model = _fit(neg_ratio=2, n_epochs=30)
    history = _ratings([(100, 1, 5)])

    # When
    scores = model.score(history).loc[100]

    # Then: 아무도 평가하지 않은 책(11~10000)은 같은 취향 묶음의 책보다 점수가 낮다
    assert scores[[2, 3, 4, 5]].min() > scores.loc[11:].max()


def test_fit_is_reproducible_with_same_seed():
    # Given
    history = _ratings([(100, 1, 5), (100, 7, 2)])

    # When
    first = _fit(n_epochs=3, neg_ratio=1).score(history)
    second = _fit(n_epochs=3, neg_ratio=1).score(history)

    # Then
    assert np.array_equal(first.to_numpy(), second.to_numpy())


def test_score_learned_rejects_users_not_in_training():
    # Given
    model = _fit(n_epochs=2)

    # When / Then
    with pytest.raises(ValueError):
        model.score_learned(_ratings([(999, 1, 5)]))


def test_load_restores_saved_model_with_same_scores_and_predictions(tmp_path):
    # Given
    model = _fit(n_epochs=5, neg_ratio=1)
    history = _ratings([(100, 1, 5), (100, 7, 2), (3, 2, 4)])
    pairs = pd.DataFrame({"user_id": [1, 25, 999], "book_id": [2, 6, 3]})

    # When
    model.save(tmp_path / "model.npz")
    loaded = MatrixFactorization.load(tmp_path / "model.npz")

    # Then
    assert loaded.config == model.config
    assert np.array_equal(loaded.score(history).to_numpy(), model.score(history).to_numpy())
    assert np.array_equal(loaded.predict(pairs), model.predict(pairs))
