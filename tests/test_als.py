import numpy as np
import pandas as pd
import pytest

from bookrec.models.collaborative_filtering.als import ALSMatrixFactorization


def _two_taste_train():
    """독자 1~20은 책 1~5에 5점·책 6~10에 1점, 독자 21~40은 반대로 준 학습 표. 각자 책 1권씩은 평가하지 않는다."""
    rows = []
    for user in range(1, 41):
        for book in range(1, 11):
            if book != (user % 10) + 1:
                rows.append((user, book, 5 if (book <= 5) == (user <= 20) else 1))
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])


def test_fit_lowers_training_error_and_predicts_in_rating_range():
    # Given
    train = _two_taste_train()

    # When
    model = ALSMatrixFactorization(n_factors=2, reg=0.01, n_epochs=10).fit(train)
    predicted = model.predict(train[["user_id", "book_id"]])

    # Then
    assert model.history["train_rmse"].iloc[-1] < 0.3
    assert predicted.min() >= 1 and predicted.max() <= 5


def test_predict_fills_in_unrated_books_by_taste():
    # Given: 독자 1은 책 2(자기 취향 묶음)를, 독자 25는 책 6(자기 취향 묶음)을 평가하지 않았다
    model = ALSMatrixFactorization(n_factors=2, reg=0.01, n_epochs=10).fit(_two_taste_train())

    # When
    predicted = model.predict(pd.DataFrame({"user_id": [1, 25], "book_id": [2, 6]}))

    # Then
    assert (predicted > 4).all()


def test_predict_uses_global_mean_and_book_bias_for_unknown_user():
    # Given
    model = ALSMatrixFactorization(n_factors=2, n_epochs=3).fit(_two_taste_train())

    # When
    predicted = model.predict(pd.DataFrame({"user_id": [999], "book_id": [3]}))

    # Then
    assert predicted[0] == pytest.approx(np.clip(model.mu + model.bi[2], 1, 5))


def test_fit_stops_early_and_keeps_best_epoch_when_validation_worsens():
    # Given: 학습 표와 정반대 별점인 검증 표(학습할수록 검증 오차가 커진다)
    train = _two_taste_train()
    val = train.assign(rating=6 - train["rating"])

    # When
    model = ALSMatrixFactorization(n_factors=2, reg=0.01, n_epochs=20, patience=2).fit(train, val=val)

    # Then
    assert len(model.history) < 20
    assert model.best_epoch == int(model.history.loc[model.history["val_rmse"].idxmin(), "epoch"])
