import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score

from bookrec.models.content_based.features import BookFeatures
from bookrec.models.content_based.clustering import (
    fit_book_clusters, compare_cluster_counts, choose_cluster_count, cluster_overview, representative_books,
)


def _features():
    return BookFeatures(csr_matrix([[1., 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 0]]),
                        np.array([3, 2, 1, 6, 5, 4, 7]), np.array(["fantasy", "mystery"]))


def test_clustering_separates_known_groups_and_keeps_zero_books_unassigned():
    # Given
    features = _features()
    original = features.matrix.copy()

    # When
    result = fit_book_clusters(features, 2)
    again = fit_book_clusters(features, 2)

    # Then
    assert adjusted_rand_score([0, 0, 0, 1, 1, 1], result.labels[:6]) == 1
    assert result.labels[6] == -1
    np.testing.assert_array_equal(result.labels, again.labels)
    np.testing.assert_array_equal(result.book_ids, features.book_ids)
    assert result.inertia == pytest.approx(0)
    assert (features.matrix != original).nnz == 0


def test_comparison_uses_nonzero_sample_and_returns_expected_silhouette():
    # Given
    features = _features()

    # When
    comparison, models = compare_cluster_counts(features, (2,), sample_size=100)

    # Then
    assert comparison.loc[0, "sample_books"] == 6
    assert comparison.loc[0, "silhouette"] == pytest.approx(1)
    assert comparison.loc[0, "min_books"] == comparison.loc[0, "max_books"] == 3
    assert comparison.loc[0, "unassigned_books"] == 1
    assert models[2].labels[6] == -1


def test_representatives_belong_to_clusters_and_break_ties_by_id():
    # Given
    features = _features()
    result = fit_book_clusters(features, 2)

    # When
    books = representative_books(features, result, n_books=2)
    overview = cluster_overview(features, result, n_tags=1)

    # Then
    labels = dict(zip(result.book_ids, result.labels))
    for cluster, group in books.groupby("cluster"):
        assert all(labels[book] == cluster for book in group.book_id)
        assert group.book_id.tolist() == sorted(group.book_id)
        assert group["rank"].tolist() == [1, 2]
    assert set(books.book_id) == {1, 2, 4, 5}
    assert overview.books.tolist() == [3, 3]
    assert set(overview.distinctive_tags) == {"fantasy", "mystery"}
    assert not books.distance.any()


def test_selection_filters_small_clusters_and_breaks_ties():
    # Given
    comparison = pd.DataFrame({"k": [20, 10, 5], "min_books": [5, 100, 100], "silhouette": [0.9, 0.3, 0.3]})

    # When
    selected = choose_cluster_count(comparison)

    # Then
    assert selected == 5
    with pytest.raises(ValueError):
        choose_cluster_count(comparison, min_books=101)


@pytest.mark.parametrize("k", [1, 6, 2.5, True])
def test_clustering_rejects_invalid_k(k):
    # Given
    features = _features()

    # When / Then
    with pytest.raises(ValueError):
        fit_book_clusters(features, k)


def test_reports_reject_misaligned_book_ids():
    # Given
    features = _features()
    result = fit_book_clusters(features, 2)
    reversed_features = BookFeatures(features.matrix, features.book_ids[::-1], features.tag_names)

    # When / Then
    with pytest.raises(ValueError):
        representative_books(reversed_features, result)
    with pytest.raises(ValueError):
        cluster_overview(reversed_features, result)


def test_comparison_rejects_duplicate_candidates_and_too_small_sample():
    # Given
    features = _features()

    # When / Then
    with pytest.raises(ValueError):
        compare_cluster_counts(features, (2, 2))
    with pytest.raises(ValueError):
        compare_cluster_counts(features, (2,), sample_size=2)
