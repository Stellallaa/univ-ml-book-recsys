import pandas as pd

from bookrec.models.base import BOOK_IDS
from bookrec.models.baselines.popularity import PopularityRecommender


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


def test_recommend_ignores_read_books_of_users_not_being_recommended():
    # Given: 추천받는 독자는 7, 8이고 read에는 점수표에 없는 독자 5의 책 10이 섞여 있다
    model = PopularityRecommender().fit(_ratings([(1, 10, 5), (2, 10, 4), (1, 20, 5)]))
    history = _ratings([(7, 30, 5), (8, 30, 4)])
    read = _ratings([(5, 10, 5), (7, 30, 5), (8, 30, 4)])

    # When
    recs = model.recommend(history, k=1, read=read)

    # Then: 마지막 행인 독자 8의 책 10이 대신 지워지지 않는다
    assert recs == {7: [10], 8: [10]}
