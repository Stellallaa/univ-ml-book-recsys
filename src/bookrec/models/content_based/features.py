"""정제한 태그로 책 특징 행렬을 만든다(B2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from bookrec.models.base import BOOK_IDS


@dataclass(frozen=True)
class BookFeatures:
    """행은 book_ids, 열은 tag_names 순서인 L2 정규화 희소 행렬."""

    matrix: csr_matrix
    book_ids: np.ndarray
    tag_names: np.ndarray

    def similar_books(self, book_id: int, k: int = 10) -> pd.DataFrame:
        """자기 자신을 뺀 유사도 상위 k권을 돌려준다. 영벡터인 책은 빈 표를 돌려준다."""
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
            raise ValueError("k는 1 이상의 정수여야 합니다.")
        positions = np.flatnonzero(self.book_ids == book_id)
        if not len(positions):
            raise ValueError(f"특징 행렬에 없는 book_id입니다: {book_id}")
        row = int(positions[0])
        empty = pd.DataFrame({"book_id": pd.Series(dtype=int), "similarity": pd.Series(dtype=float)})
        if self.matrix[row].nnz == 0:
            return empty
        # NOTE: L2 정규화된 두 벡터의 내적은 코사인 유사도다. 1만 x 1만 행렬은 만들지 않는다.
        scores = (self.matrix @ self.matrix[row].T).toarray().ravel()
        candidates = np.flatnonzero((self.book_ids != book_id) & (scores > 0))
        # NOTE: lexsort의 마지막 키가 우선이다. 유사도가 같으면 작은 book_id를 먼저 둔다.
        order = np.lexsort((self.book_ids[candidates], -scores[candidates]))[:k]
        selected = candidates[order]
        return pd.DataFrame({"book_id": self.book_ids[selected], "similarity": scores[selected]})


def build_book_features(
    book_tags: pd.DataFrame,
    weighting: str = "tfidf",
    min_df: int = 2,
    sublinear_tf: bool = True,
    book_ids: np.ndarray | None = None,
) -> BookFeatures:
    """B1 정제 결과(book_id, tag_name, count)를 책 x 태그 행렬로 바꾼다.

    weighting은 tfidf/count/binary이고, 모든 방식에 L2 정규화를 적용한다.
    min_df는 태그를 남길 최소 책 수다. 태그는 단어로 나누거나 이름을 바꾸지 않는다.
    기본 행은 BOOK_IDS 전체이며, 태그가 없는 책은 영벡터로 남긴다.
    """
    if weighting not in ("tfidf", "count", "binary"):
        raise ValueError(f"지원하지 않는 weighting입니다: {weighting!r}")
    if not isinstance(min_df, (int, np.integer)) or isinstance(min_df, bool) or min_df < 1:
        raise ValueError("min_df는 1 이상의 정수여야 합니다.")
    ids = np.array(BOOK_IDS if book_ids is None else book_ids, copy=True)
    if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError("book_ids는 중복 없는 정수 ID의 비어 있지 않은 1차원 배열이어야 합니다.")
    tags = book_tags[["book_id", "tag_name", "count"]].copy()
    if not tags["book_id"].isin(ids).all():
        raise ValueError("book_tags에 book_ids 밖의 책 ID가 있습니다.")
    if not tags["tag_name"].map(lambda name: isinstance(name, str) and bool(name.strip())).all():
        raise ValueError("태그 이름은 비어 있지 않은 문자열이어야 합니다.")
    if tags["count"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("count에 bool은 사용할 수 없습니다.")
    counts = pd.to_numeric(tags["count"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any() or (counts > (2**53 - 1)).any()):
        raise ValueError("count는 0 이상 2**53-1 이하의 유한한 정수여야 합니다.")
    # NOTE: 검사에 사용한 숫자를 실제 집계에도 반영해 문자열 연결과 정수 overflow를 막는다.
    tags["count"] = counts
    # NOTE: 0회 태그는 특징에 기여하지 않는다. 중복 입력은 합산하되, A3 입력은 이미 중복이 없다.
    tags = tags.loc[counts > 0].groupby(["book_id", "tag_name"], as_index=False)["count"].sum()
    if (tags["count"] > (2**53 - 1)).any():
        raise ValueError("중복 합산 count가 정확하게 표현할 수 있는 범위를 넘었습니다.")
    frequency = tags.groupby("tag_name")["book_id"].nunique()
    names = np.array(sorted(frequency[frequency >= min_df].index), dtype=str)
    if not len(names):
        raise ValueError("조건을 만족하는 태그가 없습니다. 입력 또는 min_df를 확인하세요.")
    tags = tags[tags["tag_name"].isin(names)]
    rows = pd.Index(ids).get_indexer(tags["book_id"])
    cols = pd.Index(names).get_indexer(tags["tag_name"])
    matrix = csr_matrix((tags["count"].to_numpy(dtype=float), (rows, cols)), shape=(len(ids), len(names)))
    if weighting == "tfidf":
        # NOTE: smooth_idf=True는 idf=log((1+책 수)/(1+등장 책 수))+1을 쓴다.
        #       sublinear_tf=True는 큰 count를 1+log(count)로 완화한다. norm은 책별 L2 정규화다.
        matrix = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=sublinear_tf).fit_transform(matrix)
    else:
        if weighting == "binary":
            matrix.data[:] = 1.0
        matrix = normalize(matrix, norm="l2", copy=False)
    return BookFeatures(csr_matrix(matrix), ids, names)
