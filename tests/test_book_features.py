import numpy as np
import pandas as pd
import pytest
from scipy.sparse import isspmatrix_csr

from bookrec.models.base import BOOK_IDS
from bookrec.models.content_based.features import build_book_features


def _tags(rows):
    return pd.DataFrame(rows, columns=["book_id", "tag_name", "count"])


def test_features_keep_catalog_order_compound_tags_and_missing_books():
    # Given
    tags = _tags([(3, "young-adult", 4), (1, "science-fiction", 3)])
    original = tags.copy(deep=True)

    # When
    features = build_book_features(tags, min_df=1)

    # Then
    assert isspmatrix_csr(features.matrix)
    np.testing.assert_array_equal(features.book_ids, BOOK_IDS)
    assert features.tag_names.tolist() == ["science-fiction", "young-adult"]
    assert features.matrix.shape == (10000, 2)
    assert features.matrix[0, 0] == 1 and features.matrix[2, 1] == 1
    assert features.matrix[1].nnz == 0
    pd.testing.assert_frame_equal(tags, original)


def test_tfidf_matches_hand_calculation_including_zero_document():
    # Given: fantasy는 두 책, mystery는 한 책에 등장하고 책 3은 태그가 없다
    tags = _tags([(1, "fantasy", 4), (1, "mystery", 1), (2, "fantasy", 2)])

    # When
    features = build_book_features(tags, min_df=1, book_ids=np.array([1, 2, 3]))

    # Then: 책 수 N=3, tf=1+log(count), idf=log((1+N)/(1+df))+1
    weighted = np.array([(1 + np.log(4)) * (1 + np.log(4 / 3)), 1 + np.log(4 / 2)])
    np.testing.assert_allclose(features.matrix[0].toarray().ravel(), weighted / np.linalg.norm(weighted))
    np.testing.assert_allclose(features.matrix[1].toarray(), [[1, 0]])
    assert features.matrix[2].nnz == 0


def test_min_df_counts_distinct_books_and_ignores_zero_counts():
    # Given: 중복된 rare는 한 책에만 있고 zero는 가중치가 없다
    tags = _tags([(1, "common", 2), (2, "common", 1), (1, "rare", 2), (1, "rare", 3), (2, "zero", 0)])

    # When
    features = build_book_features(tags, min_df=2, book_ids=np.array([2, 1, 3]))

    # Then
    assert features.book_ids.tolist() == [2, 1, 3]
    assert features.tag_names.tolist() == ["common"]
    np.testing.assert_allclose(features.matrix.toarray(), [[1], [1], [0]])


def test_count_and_binary_weighting_have_expected_lengths():
    # Given
    tags = _tags([(1, "fantasy", 3), (1, "mystery", 4)])

    # When
    counts = build_book_features(tags, weighting="count", min_df=1, book_ids=np.array([1]))
    binary = build_book_features(tags, weighting="binary", min_df=1, book_ids=np.array([1]))

    # Then
    np.testing.assert_allclose(counts.matrix.toarray(), [[0.6, 0.8]])
    np.testing.assert_allclose(binary.matrix.toarray(), [[1 / np.sqrt(2), 1 / np.sqrt(2)]])


def test_similar_books_excludes_self_breaks_ties_and_handles_zero_vector():
    # Given
    tags = _tags([(1, "fantasy", 1), (2, "fantasy", 1), (3, "fantasy", 1), (5, "mystery", 1)])
    features = build_book_features(tags, min_df=1, book_ids=np.array([3, 2, 1, 4, 5]))

    # When
    similar = features.similar_books(1, k=10)
    empty = features.similar_books(4)

    # Then
    assert similar["book_id"].tolist() == [2, 3]
    np.testing.assert_allclose(similar["similarity"], 1)
    assert empty.empty and empty.columns.tolist() == ["book_id", "similarity"]
    with pytest.raises(ValueError):
        features.similar_books(999)
    with pytest.raises(ValueError):
        features.similar_books(1, k=0)


@pytest.mark.parametrize("kwargs", [{"weighting": "unknown"}, {"min_df": 0}, {"min_df": 1.5}, {"book_ids": np.array([1, 1])}])
def test_features_reject_invalid_settings(kwargs):
    # Given
    tags = _tags([(1, "fantasy", 1)])

    # When / Then
    with pytest.raises(ValueError):
        build_book_features(tags, **kwargs)


@pytest.mark.parametrize("rows", [[], [(1, "fantasy", -1)], [(1, "fantasy", np.inf)], [(1, "", 1)], [(1, None, 1)], [(2, "fantasy", 1)]])
def test_features_reject_unusable_input(rows):
    # Given
    tags = _tags(rows)

    # When / Then
    with pytest.raises(ValueError):
        build_book_features(tags, min_df=1, book_ids=np.array([1]))
