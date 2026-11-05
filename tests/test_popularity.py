import pandas as pd

from bookrec.models.base import BOOK_IDS
from bookrec.models.popularity import PopularityRecommender


def _ratings(rows):
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])


def test_score_covers_whole_catalog_for_each_user():
    # Given
    model = PopularityRecommender().fit(_ratings([(1, 10, 5), (2, 10, 4), (2, 20, 3)]))
    history = _ratings([(7, 10, 5), (8, 20, 4)])

    # When
    scores = model.score(history)

    # Then
    assert scores.index.tolist() == [7, 8]
    assert scores.columns.tolist() == BOOK_IDS.tolist()
    assert scores.loc[7, 10] == 2 and scores.loc[7, 20] == 1


def test_recommend_excludes_books_in_history():
    # Given
    model = PopularityRecommender().fit(_ratings([(1, 10, 5), (2, 10, 4), (3, 10, 3), (1, 20, 5), (2, 20, 4), (1, 30, 5)]))
    history = _ratings([(7, 10, 5)])

    # When
    recs = model.recommend(history, k=2)

    # Then
    assert recs == {7: [20, 30]}
