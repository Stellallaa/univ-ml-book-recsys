"""CB 설정을 재현하고 검증 결과로 최종 설정을 고른다(B4)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bookrec.data.tag_filter import filter_book_tags
from bookrec.models.content_based.features import build_book_features
from bookrec.models.content_based.recommender import ContentBasedRecommender


@dataclass(frozen=True)
class ContentBasedConfig:
    """태그 정제·책 특징·독자 가중치 설정. JSON의 config 항목으로 재생성할 수 있다."""

    extended: bool = True
    weighting: str = "tfidf"
    min_df: int = 2
    sublinear_tf: bool = True
    profile_weighting: str = "positive"


def make_content_based(book_tags: pd.DataFrame, config: ContentBasedConfig) -> ContentBasedRecommender:
    """A3 태그와 설정으로 학습 전 CB를 만든다. B1·B2·B3 함수를 그대로 사용한다."""
    if not isinstance(config.extended, bool) or not isinstance(config.sublinear_tf, bool):
        raise ValueError("extended와 sublinear_tf는 bool이어야 합니다.")
    tags = filter_book_tags(book_tags, extended=config.extended)
    features = build_book_features(tags, weighting=config.weighting, min_df=config.min_df, sublinear_tf=config.sublinear_tf)
    return ContentBasedRecommender(features, profile_weighting=config.profile_weighting)


def rank_validation(results: pd.DataFrame) -> pd.DataFrame:
    """cold_tune의 이력 1·3·5·10 평균 NDCG로 정렬한다. 동점이면 warm_val, 설정 ID 순서다.

    한 설정당 네 cold 행과 한 warm 행만 허용한다. 최종 평가나 누락된 결과로는 선택하지 않는다.
    """
    columns = {"candidate_id", "goal", "history_size", "ndcg@10"}
    if not columns.issubset(results.columns) or results.empty:
        raise ValueError("설정 ID·목표·이력 크기·NDCG를 포함한 검증 결과가 필요합니다.")
    if results["candidate_id"].isna().any() or not results["goal"].isin(["warm_val", "cold_tune"]).all():
        raise ValueError("설정 ID는 결측이 없어야 하며 val·cold_tune 결과만 허용합니다.")
    if not np.isfinite(results["ndcg@10"].to_numpy(dtype=float)).all() or not results["ndcg@10"].between(0, 1).all():
        raise ValueError("NDCG는 0~1 범위의 유한한 값이어야 합니다.")
    records = []
    for candidate, group in results.groupby("candidate_id", sort=True):
        cold = group[group["goal"] == "cold_tune"]
        warm = group[group["goal"] == "warm_val"]
        if len(cold) != 4 or set(cold["history_size"].astype(str)) != {"1", "3", "5", "10"}:
            raise ValueError(f"{candidate}: cold 이력 1·3·5·10 결과가 각각 하나씩 필요합니다.")
        if len(warm) != 1 or str(warm["history_size"].iloc[0]) != "train":
            raise ValueError(f"{candidate}: train 이력의 warm_val 결과 하나가 필요합니다.")
        records.append({"candidate_id": candidate, "cold_mean_ndcg@10": cold["ndcg@10"].mean(),
                        "warm_ndcg@10": warm["ndcg@10"].iloc[0]})
    return pd.DataFrame(records).sort_values(
        ["cold_mean_ndcg@10", "warm_ndcg@10", "candidate_id"], ascending=[False, False, True], ignore_index=True
    )
