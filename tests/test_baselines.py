import numpy as np
import pandas as pd
import pytest

from bookrec.models.base import BOOK_IDS
from bookrec.models.bias import BiasPredictor
from bookrec.models.goodreads_rating import GoodreadsRatingRecommender


def _ratings(rows):
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])


def test_goodreads_rating_recommends_highest_average_rating_except_read():
    # Given
    books = pd.DataFrame({"book_id": BOOK_IDS, "average_rating": 3.0})
    books.loc[books["book_id"].isin([20, 30, 40]), "average_rating"] = [4.0, 4.5, 4.2]
    model = GoodreadsRatingRecommender(books).fit(_ratings([(1, 10, 5)]))

    # When
    recs = model.recommend(_ratings([(7, 30, 5), (8, 10, 1)]), k=2)

    # Then
    assert recs == {7: [40, 20], 8: [30, 40]}


def test_bias_predicts_mean_plus_user_and_book_bias_without_reg():
    # Given: 독자 2는 독자 1보다 1점 후하고, 책 20은 책 10보다 2점 높다
    train = _ratings([(1, 10, 1), (1, 20, 3), (2, 10, 2), (2, 20, 4)])
    model = BiasPredictor(user_reg=0, book_reg=0).fit(train)

    # When
    predicted = model.predict(train[["user_id", "book_id"]])

    # Then
    np.testing.assert_allclose(predicted, train["rating"])


def test_bias_shrinks_toward_mean_with_reg():
    # Given
    train = _ratings([(1, 10, 5), (2, 10, 5), (3, 20, 1), (4, 20, 1)])

    # When
    loose = BiasPredictor(user_reg=0, book_reg=0, n_iters=1).fit(train).predict(train[["user_id", "book_id"]])
    shrunk = BiasPredictor(user_reg=10, book_reg=10, n_iters=1).fit(train).predict(train[["user_id", "book_id"]])

    # Then
    assert np.all(np.abs(shrunk - 3) < np.abs(loose - 3))


def test_bias_predicts_mean_for_unknown_user_and_book():
    # Given
    model = BiasPredictor().fit(_ratings([(1, 10, 5), (2, 20, 1), (2, 10, 4)]))

    # When
    predicted = model.predict(pd.DataFrame({"user_id": [99], "book_id": [999]}))

    # Then
    assert predicted == pytest.approx([10 / 3])
