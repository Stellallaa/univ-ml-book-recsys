"""책 특징 벡터를 k-means로 묶고 대표 책을 고른다(B5)."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from bookrec.models.content_based.features import BookFeatures


@dataclass(frozen=True)
class ClusterSpace:
    """군집 계산 전용 좌표. SVD 축은 원래 태그가 아니며 BookFeatures는 그대로 보존한다."""

    book_ids: np.ndarray
    matrix: csr_matrix | np.ndarray
    n_components: int | None
    explained_variance_ratio: float | None


@dataclass(frozen=True)
class BookClusters:
    """책 ID 순서의 군집 번호와 실제 군집 좌표의 중심. 영벡터 책은 -1이다."""

    book_ids: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int
    space: ClusterSpace | None = None


def prepare_cluster_space(features: BookFeatures, n_components: int | None = None, seed: int = 42) -> ClusterSpace:
    """원본 또는 SVD 후 L2 재정규화 좌표를 만든다. 원본 영벡터는 0으로 유지한다."""
    matrix = features.matrix
    if matrix.shape != (len(features.book_ids), len(features.tag_names)) or len(np.unique(features.book_ids)) != len(features.book_ids):
        raise ValueError("특징 행렬과 책 ID·태그 이름의 대응을 확인하세요.")
    if not matrix.shape[1] or not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("특징은 유한한 비음수 값이어야 하고 태그 열이 필요합니다.")
    if n_components is None:
        return ClusterSpace(features.book_ids.copy(), matrix, None, None)
    nonzero = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel() > 0
    if (not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool)
            or not 1 <= n_components < min(int(nonzero.sum()), matrix.shape[1])):
        raise ValueError("SVD 차원은 1 이상이며 유효 책 수와 태그 수보다 작아야 합니다.")
    reducer = TruncatedSVD(n_components=n_components, n_iter=7, random_state=seed)
    reduced = np.zeros((len(features.book_ids), n_components))
    reduced[nonzero] = normalize(reducer.fit_transform(matrix[nonzero]), norm="l2")
    # NOTE: 음수인 SVD 좌표는 정상이다. 원본 태그 가중치의 비음수 조건을 축소 좌표에 적용하지 않는다.
    return ClusterSpace(features.book_ids.copy(), reduced, int(n_components), float(reducer.explained_variance_ratio_.sum()))


def fit_book_clusters(
    features: BookFeatures, k: int, seed: int = 42, space: ClusterSpace | None = None,
) -> BookClusters:
    """영벡터를 제외하고 k-means를 학습한다. space를 생략하면 기존 원본 공간을 사용한다."""
    if space is None:
        space = prepare_cluster_space(features)
    if not np.array_equal(features.book_ids, space.book_ids) or space.matrix.shape[0] != len(features.book_ids):
        raise ValueError("군집 좌표와 원본 특징의 책 ID 순서가 다릅니다.")
    matrix = space.matrix
    values = matrix.data if issparse(matrix) else matrix
    if not np.isfinite(values).all():
        raise ValueError("군집 좌표는 유한한 값이어야 합니다.")
    nonzero = _nonzero_rows(matrix)
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
    return BookClusters(features.book_ids.copy(), labels, model.cluster_centers_, float(model.inertia_), int(model.n_iter_), space)


def compare_cluster_counts(
    features: BookFeatures, cluster_counts: tuple[int, ...] = (5, 10, 15, 20),
    seed: int = 42, sample_size: int = 2000, space: ClusterSpace | None = None,
) -> tuple[pd.DataFrame, dict[int, BookClusters]]:
    """같은 책 표본의 군집 공간 silhouette와 전체 학습 inertia·군집 크기를 비교한다."""
    if not cluster_counts or len(set(cluster_counts)) != len(cluster_counts):
        raise ValueError("중복 없는 k 후보가 필요합니다.")
    if not isinstance(sample_size, (int, np.integer)) or isinstance(sample_size, bool) or sample_size < 3:
        raise ValueError("sample_size는 3 이상의 정수여야 합니다.")
    if space is None:
        space = prepare_cluster_space(features)
    valid = np.flatnonzero(_nonzero_rows(space.matrix))
    # NOTE: k마다 같은 표본을 사용한다. 전체 책의 쌍별 거리 행렬을 만들지 않도록 기본 2,000권으로 제한한다.
    sample = np.sort(np.random.default_rng(seed).choice(valid, size=min(sample_size, len(valid)), replace=False))
    rows, models = [], {}
    for k in cluster_counts:
        result = fit_book_clusters(features, k, seed=seed, space=space)
        models[k] = result
        sizes = np.bincount(result.labels[result.labels >= 0], minlength=k)
        sampled_labels = result.labels[sample]
        n_labels = len(np.unique(sampled_labels))
        score = silhouette_score(space.matrix[sample], sampled_labels, metric="euclidean") if 1 < n_labels < len(sample) else np.nan
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
    for cluster in range(len(clusters.centers)):
        # NOTE: 태그 해석은 원본 공간의 군집 평균으로 한다. SVD 중심의 축을 태그에 대응시키지 않는다.
        center = np.asarray(features.matrix[clusters.labels == cluster].mean(axis=0)).ravel()
        top = np.lexsort((features.tag_names, -center))
        difference = center - mean
        distinctive = np.lexsort((features.tag_names, -difference))
        rows.append({"cluster": cluster, "books": int((clusters.labels == cluster).sum()),
                     "top_tags": " | ".join(features.tag_names[top[center[top] > 0]][:n_tags]),
                     "distinctive_tags": " | ".join(features.tag_names[distinctive[difference[distinctive] > 0]][:n_tags])})
    return pd.DataFrame(rows)


def representative_books(
    features: BookFeatures, clusters: BookClusters, n_books: int = 5,
    books: pd.DataFrame | None = None, diversify: bool = False,
) -> pd.DataFrame:
    """실제 군집 좌표의 중심 거리로 고른다. diversify=True면 저자·명시적 시리즈 중복을 제한한다.

    부족할 때는 시리즈·저자·제목 제한을 차례로 완화한다. 결과는 거리순이며 동점은 책 ID순이다.
    """
    _check_alignment(features, clusters)
    if not isinstance(n_books, (int, np.integer)) or isinstance(n_books, bool) or n_books < 1:
        raise ValueError("n_books는 1 이상의 정수여야 합니다.")
    matrix = features.matrix if clusters.space is None else clusters.space.matrix
    if diversify:
        if books is None or not {"book_id", "title", "main_author"}.issubset(books.columns) or not books["book_id"].is_unique:
            raise ValueError("다양성 선정에는 중복 없는 책 ID·제목·대표 저자 표가 필요합니다.")
        metadata = books.set_index("book_id")
        if not pd.Index(features.book_ids).isin(metadata.index).all():
            raise ValueError("책 메타데이터에 특징 행렬의 책이 누락됐습니다.")
    rows = []
    for cluster, center in enumerate(clusters.centers):
        positions = np.flatnonzero(clusters.labels == cluster)
        distances = euclidean_distances(matrix[positions], center.reshape(1, -1)).ravel()
        order = np.lexsort((features.book_ids[positions], distances))
        if not diversify:
            chosen = [(index, "distance") for index in order[:n_books]]
        else:
            chosen = []
            authors, series, titles = set(), set(), set()
            used = set()
            # NOTE: 중심과 가까운 순서를 유지하며 먼저 저자·명시적 시리즈 중복을 막는다.
            #       수량이 부족할 때만 시리즈, 저자, 제목 제한을 순서대로 완화한다.
            for rule in ("author_series", "relax_series", "relax_author", "distance"):
                for index in order:
                    if index in used or len(chosen) >= n_books:
                        continue
                    book_id = int(features.book_ids[positions[index]])
                    info = metadata.loc[book_id]
                    title = str(info["title"]).strip().casefold()
                    author = str(info["main_author"]).strip().casefold()
                    match = re.search(r"\(([^()]*)#\s*\d[^()]*\)", title)
                    key = match.group(1).rstrip(" ,;-") if match else None
                    if rule != "distance" and title in titles:
                        continue
                    if rule in ("author_series", "relax_series") and author in authors:
                        continue
                    if rule == "author_series" and key and key in series:
                        continue
                    chosen.append((index, rule))
                    used.add(index)
                    authors.add(author)
                    titles.add(title)
                    if key:
                        series.add(key)
            chosen.sort(key=lambda pair: (distances[pair[0]], features.book_ids[positions[pair[0]]]))
        candidate_ranks = {index: rank for rank, index in enumerate(order, start=1)}
        for rank, (index, rule) in enumerate(chosen, start=1):
            row = {"cluster": cluster, "rank": rank, "book_id": int(features.book_ids[positions[index]]),
                   "distance": float(distances[index])}
            if diversify:
                row.update(candidate_rank=candidate_ranks[index], selection_rule=rule)
            rows.append(row)
    columns = ["cluster", "rank", "book_id", "distance"]
    return pd.DataFrame(rows, columns=columns + (["candidate_rank", "selection_rule"] if diversify else []))


def _check_alignment(features: BookFeatures, clusters: BookClusters) -> None:
    if not np.array_equal(features.book_ids, clusters.book_ids) or len(clusters.labels) != len(features.book_ids):
        raise ValueError("군집과 특징 행렬의 책 ID 순서가 다릅니다.")
    matrix = features.matrix if clusters.space is None else clusters.space.matrix
    if clusters.space is not None and not np.array_equal(features.book_ids, clusters.space.book_ids):
        raise ValueError("군집 좌표와 책 ID 순서가 다릅니다.")
    if clusters.centers.ndim != 2 or clusters.centers.shape[1] != matrix.shape[1]:
        raise ValueError("군집 중심과 군집 좌표의 열 수가 다릅니다.")
    if not np.isfinite(clusters.centers).all() or not np.isin(clusters.labels, np.arange(-1, len(clusters.centers))).all():
        raise ValueError("군집 중심과 군집 번호를 확인하세요.")


def _nonzero_rows(matrix: csr_matrix | np.ndarray) -> np.ndarray:
    squares = matrix.multiply(matrix) if issparse(matrix) else matrix * matrix
    return np.asarray(squares.sum(axis=1)).ravel() > 0


def cluster_diagnostics(
    features: BookFeatures, clusters: BookClusters, sample_book_ids: np.ndarray,
    reference_features: BookFeatures | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """같은 표본의 원본·군집·공통 기준 공간 silhouette를 책별·군집별로 반환한다.

    reference_features는 다른 태그 설정도 공정하게 비교할 고정 기준이다.
    영벡터·미배정 책이 표본에 있으면 조용히 빼지 않고 오류를 내어 평가 대상 변경을 막는다.
    """
    _check_alignment(features, clusters)
    reference = features if reference_features is None else reference_features
    if not np.array_equal(reference.book_ids, features.book_ids):
        raise ValueError("공통 기준 특징과 책 ID 순서가 다릅니다.")
    ids = np.asarray(sample_book_ids)
    if ids.ndim != 1 or len(ids) < 3 or len(np.unique(ids)) != len(ids):
        raise ValueError("표본에는 중복 없는 책 ID가 3개 이상 필요합니다.")
    positions = pd.Index(features.book_ids).get_indexer(ids)
    if (positions < 0).any():
        raise ValueError("표본에 특징 행렬에 없는 책 ID가 있습니다.")
    matrix = features.matrix if clusters.space is None else clusters.space.matrix
    labels = clusters.labels[positions]
    if ((labels < 0).any() or not _nonzero_rows(features.matrix[positions]).all()
            or not _nonzero_rows(reference.matrix[positions]).all() or not _nonzero_rows(matrix[positions]).all()):
        raise ValueError("표본에 영벡터 또는 미배정 책이 있습니다.")
    if not 1 < len(np.unique(labels)) < len(ids):
        raise ValueError("silhouette에는 2개 이상, 표본 수 미만의 군집이 필요합니다.")
    original = silhouette_samples(features.matrix[positions], labels, metric="euclidean")
    reduced = original if clusters.space is None or clusters.space.n_components is None else silhouette_samples(matrix[positions], labels, metric="euclidean")
    common = original if reference is features else silhouette_samples(reference.matrix[positions], labels, metric="euclidean")
    per_book = pd.DataFrame({"book_id": ids, "cluster": labels, "original_silhouette": original,
                             "clustering_silhouette": reduced, "reference_silhouette": common})
    rows = []
    for cluster in range(len(clusters.centers)):
        group = per_book[per_book["cluster"] == cluster]
        row = {"cluster": cluster, "books": int((clusters.labels == cluster).sum()), "sample_books": len(group)}
        for name in ("original", "clustering", "reference"):
            values = group[f"{name}_silhouette"]
            row[f"{name}_silhouette"] = values.mean()
            row[f"{name}_negative_rate"] = float((values < 0).mean()) if len(values) else np.nan
        rows.append(row)
    return per_book, pd.DataFrame(rows)

def load_selected_clusters(
    book_tags: pd.DataFrame | None = None, config_path: str | Path | None = None,
) -> tuple[BookFeatures, BookClusters]:
    """개선 실험의 최종 JSON으로 특징과 군집을 재현한다. 암묵적인 기본 설정 대체는 없다."""
    from bookrec.data import load_book_tags
    from bookrec.models.content_based.tuning import ContentBasedConfig, make_content_based

    path = Path(config_path) if config_path is not None else Path(__file__).resolve().parents[4] / "results/content_based/B5_improvement_selected_config.json"
    saved = json.loads(path.read_text())
    required = {"adopted", "feature_config", "n_components", "k", "svd_seed", "svd_n_iter", "fit_seed",
                "n_init", "max_iter", "tol", "algorithm", "initialization", "metric"}
    if not isinstance(saved, dict) or not required.issubset(saved) or type(saved["adopted"]) is not bool:
        raise ValueError("최종 군집 JSON의 필수 설정·채택 여부를 확인하세요.")
    expected = {"svd_n_iter": 7, "n_init": 10, "max_iter": 300, "tol": 1e-4,
                "algorithm": "lloyd", "initialization": "k-means++", "metric": "euclidean"}
    if any(saved[key] != value for key, value in expected.items()):
        raise ValueError("현재 군집 구현과 JSON의 학습 파라미터가 다릅니다.")
    for key in ("svd_seed", "fit_seed"):
        if type(saved[key]) is not int or not 0 <= saved[key] < 2**32:
            raise ValueError("군집·SVD seed는 0 이상 2**32 미만의 정수여야 합니다.")
    if not isinstance(saved["feature_config"], dict) or set(saved["feature_config"]) != set(ContentBasedConfig.__dataclass_fields__):
        raise ValueError("최종 군집 feature_config의 필드를 확인하세요.")
    features = make_content_based(load_book_tags() if book_tags is None else book_tags,
                                  ContentBasedConfig(**saved["feature_config"])).features
    space = prepare_cluster_space(features, saved["n_components"], seed=saved["svd_seed"])
    return features, fit_book_clusters(features, saved["k"], seed=saved["fit_seed"], space=space)
