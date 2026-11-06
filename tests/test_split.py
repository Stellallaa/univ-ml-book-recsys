import pandas as pd

from bookrec.data import split_ratings


def make_ratings(counts):
    """독자별 평점 수를 받아 독자가 번갈아 나오는 평점 표를 만든다. book_id는 독자 안의 순번이다."""
    rows = [(user, book) for book in range(max(counts.values())) for user, n in counts.items() if book < n]
    ratings = pd.DataFrame(rows, columns=["user_id", "book_id"])
    ratings["rating"] = ratings["book_id"] % 5 + 1
    return ratings


def test_split_ratings_divides_each_user_70_15_15_in_row_order():
    # Given: 평점 20개, 10개인 두 독자가 번갈아 나오는 표
    ratings = make_ratings({1: 20, 2: 10})

    # When
    split = split_ratings(ratings, n_cold_users=0)

    # Then: 독자별 앞 70%는 train, 다음 15%는 val, 나머지는 test이고 행 순서가 유지된다
    for part, expected in [(split.train, {1: range(14), 2: range(7)}), (split.val, {1: range(14, 17), 2: [7]}), (split.test, {1: range(17, 20), 2: [8, 9]})]:
        assert part.index.is_monotonic_increasing
        for user, books in expected.items():
            assert part.loc[part["user_id"] == user, "book_id"].tolist() == list(books)


def test_split_ratings_holds_out_cold_users_by_person():
    # Given: 평점 수가 다른 독자 10명
    ratings = make_ratings({user: user + 2 for user in range(10)})

    # When
    split = split_ratings(ratings, n_cold_users=4)

    # Then: 4명이 2명씩 나뉘고, train/val/test에 없으며, 평점은 원본 순서 그대로 전부 남는다
    tune, final = set(split.cold_tune["user_id"]), set(split.cold_final["user_id"])
    warm = set(pd.concat([split.train, split.val, split.test])["user_id"])
    assert len(tune) == len(final) == 2
    assert not tune & final and not (tune | final) & warm
    pd.testing.assert_frame_equal(split.cold_tune, ratings[ratings["user_id"].isin(tune)])
    pd.testing.assert_frame_equal(split.cold_final, ratings[ratings["user_id"].isin(final)])


def test_split_ratings_same_seed_gives_same_cold_users():
    # Given
    ratings = make_ratings({user: 5 for user in range(50)})

    # When
    first, second = split_ratings(ratings, n_cold_users=10, seed=7), split_ratings(ratings, n_cold_users=10, seed=7)

    # Then
    pd.testing.assert_frame_equal(first.cold_tune, second.cold_tune)
    pd.testing.assert_frame_equal(first.cold_final, second.cold_final)


def test_split_ratings_keeps_every_row_once():
    # Given
    ratings = make_ratings({user: user * 3 + 1 for user in range(20)})

    # When
    split = split_ratings(ratings, n_cold_users=6)

    # Then: 다섯 표를 합치면 빠지거나 겹친 행 없이 원본과 같다
    parts = pd.concat([split.train, split.val, split.test, split.cold_tune, split.cold_final])
    pd.testing.assert_frame_equal(parts.sort_index(), ratings)
