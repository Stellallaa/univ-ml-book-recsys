"""책 특징 벡터를 k-means로 묶고 대표 책을 고른다(B5)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from bookrec.models.content_based.features import BookFeatures


@dataclass(frozen=True)
class BookClusters:
    """책 ID 순서의 군집 번호와 중심. 영벡터 책의 군집 번호는 -1이다."""

    book_ids: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int


def fit_book_clusters(features: BookFeatures, k: int, seed: int = 42) -> BookClusters:
    """영벡터를 제외하고 k-means를 학습한다. 원래 책 ID·행 순서는 유지한다."""
    matrix = features.matrix
    if matrix.shape != (len(features.book_ids), len(features.tag_names)) or len(np.unique(features.book_ids)) != len(features.book_ids):
        raise ValueError("특징 행렬과 책 ID·태그 이름의 대응을 확인하세요.")
    if not matrix.shape[1] or not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("특징은 유한한 비음수 값이어야 하고 태그 열이 필요합니다.")
    nonzero = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel() > 0
    n_books = int(nonzero.sum())
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not 2 <= k < n_books:
        raise ValueError("k는 2 이상이며 영벡터를 제외한 책 수보다 작은 정수여야 합니다.")
    # NOTE: k-means++는 초기 중심 선택, n_init=10은 초기화를 달리한 10회 중 최소 inertia를 고른다.
    #       Lloyd 알고리즘으로 최대 300회 반복하며, 중심 변화 tol=1e-4를 수렴 기준으로 쓴다.
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, tol=1e-4,
                   algorithm="lloyd", random_state=seed).fit(matrix[nonzero])
    if len(np.unique(model.labels_)) != k:
        raise ValueError("서로 다른 특징 벡터가 부족해 요청한 군집 수를 만들 수 없습니다.")
    labels = np.full(len(features.book_ids), -1, dtype=int)
    labels[nonzero] = model.labels_
    return BookClusters(features.book_ids.copy(), labels, model.cluster_centers_, float(model.inertia_), int(model.n_iter_))


def compare_cluster_counts(
    features: BookFeatures, cluster_counts: tuple[int, ...] = (5, 10, 15, 20),
    seed: int = 42, sample_size: int = 2000,
) -> tuple[pd.DataFrame, dict[int, BookClusters]]:
    """같은 책 표본의 Euclidean silhouette와 전체 학습 inertia·군집 크기를 비교한다."""
    if not cluster_counts or len(set(cluster_counts)) != len(cluster_counts):
        raise ValueError("중복 없는 k 후보가 필요합니다.")
    if not isinstance(sample_size, (int, np.integer)) or isinstance(sample_size, bool) or sample_size < 3:
        raise ValueError("sample_size는 3 이상의 정수여야 합니다.")
    valid = np.flatnonzero(np.asarray(features.matrix.multiply(features.matrix).sum(axis=1)).ravel() > 0)
    # NOTE: k마다 같은 표본을 사용한다. 전체 책의 쌍별 거리 행렬을 만들지 않도록 기본 2,000권으로 제한한다.
    sample = np.sort(np.random.default_rng(seed).choice(valid, size=min(sample_size, len(valid)), replace=False))
    rows, models = [], {}
    for k in cluster_counts:
        result = fit_book_clusters(features, k, seed=seed)
        models[k] = result
        sizes = np.bincount(result.labels[result.labels >= 0], minlength=k)
        sampled_labels = result.labels[sample]
        n_labels = len(np.unique(sampled_labels))
        score = silhouette_score(features.matrix[sample], sampled_labels, metric="euclidean") if 1 < n_labels < len(sample) else np.nan
        rows.append({"k": k, "inertia": result.inertia, "silhouette": score, "sample_books": len(sample),
                     "min_books": int(sizes.min()), "max_books": int(sizes.max()),
                     "unassigned_books": int((result.labels < 0).sum()), "n_iter": result.n_iter})
    return pd.DataFrame(rows), models


def choose_cluster_count(comparison: pd.DataFrame, min_books: int = 100) -> int:
    """최소 책 수를 만족하는 후보 중 silhouette가 가장 높은 k를 선택한다. 동점이면 작은 k를 고른다."""
    if not isinstance(min_books, (int, np.integer)) or isinstance(min_books, bool) or min_books < 1:
        raise ValueError("min_books는 1 이상의 정수여야 합니다.")
    eligible = comparison[(comparison["min_books"] >= min_books) & np.isfinite(comparison["silhouette"])]
    if eligible.empty:
        raise ValueError("최소 군집 크기와 silhouette 조건을 만족하는 후보가 없습니다.")
    return int(eligible.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])


def cluster_overview(features: BookFeatures, clusters: BookClusters, n_tags: int = 8) -> pd.DataFrame:
    """군집 크기, 중심 가중치 상위 태그와 전체 평균보다 두드러진 태그를 반환한다."""
    _check_alignment(features, clusters)
    if not isinstance(n_tags, (int, np.integer)) or isinstance(n_tags, bool) or n_tags < 1:
        raise ValueError("n_tags는 1 이상의 정수여야 합니다.")
    valid = clusters.labels >= 0
    mean = np.asarray(features.matrix[valid].mean(axis=0)).ravel()
    rows = []
    for cluster, center in enumerate(clusters.centers):
        top = np.lexsort((features.tag_names, -center))
        difference = center - mean
        distinctive = np.lexsort((features.tag_names, -difference))
        rows.append({"cluster": cluster, "books": int((clusters.labels == cluster).sum()),
                     "top_tags": " | ".join(features.tag_names[top[center[top] > 0]][:n_tags]),
                     "distinctive_tags": " | ".join(features.tag_names[distinctive[difference[distinctive] > 0]][:n_tags])})
    return pd.DataFrame(rows)


def representative_books(features: BookFeatures, clusters: BookClusters, n_books: int = 5) -> pd.DataFrame:
    """각 군집 안에서 중심까지 Euclidean 거리가 가까운 책을 고른다. 동점은 책 ID 순서다."""
    _check_alignment(features, clusters)
    if not isinstance(n_books, (int, np.integer)) or isinstance(n_books, bool) or n_books < 1:
        raise ValueError("n_books는 1 이상의 정수여야 합니다.")
    rows = []
    for cluster, center in enumerate(clusters.centers):
        positions = np.flatnonzero(clusters.labels == cluster)
        distances = euclidean_distances(features.matrix[positions], center.reshape(1, -1)).ravel()
        order = np.lexsort((features.book_ids[positions], distances))[:n_books]
        for rank, index in enumerate(order, start=1):
            rows.append({"cluster": cluster, "rank": rank, "book_id": int(features.book_ids[positions[index]]),
                         "distance": float(distances[index])})
    return pd.DataFrame(rows, columns=["cluster", "rank", "book_id", "distance"])


def _check_alignment(features: BookFeatures, clusters: BookClusters) -> None:
    if not np.array_equal(features.book_ids, clusters.book_ids) or len(clusters.labels) != len(features.book_ids):
        raise ValueError("군집과 특징 행렬의 책 ID 순서가 다릅니다.")
    if clusters.centers.ndim != 2 or clusters.centers.shape[1] != features.matrix.shape[1]:
        raise ValueError("군집 중심과 태그 열 수가 다릅니다.")
