"""어떤 모델이든 Goal 1~3의 순위 지표와 RMSE를 같은 방식으로 계산한다(A5)."""

import numpy as np
import pandas as pd

from bookrec.data import RatingSplit
from bookrec.models.base import RatingPredictor, Recommender

RELEVANT_RATING = 4
K = 10
SEED = 42
COLD_HISTORY_SIZES = (1, 3, 5, 10)
BATCH_SIZE = 500


def _recommend(model: Recommender, history: pd.DataFrame, read: pd.DataFrame, k: int) -> dict[int, list[int]]:
    """독자를 BATCH_SIZE명씩 나눠 model.recommend(이력, k, 이미 읽은 책)를 부르고 결과를 합친다."""
    users = history["user_id"].unique()
    recs = {}
    for start in range(0, len(users), BATCH_SIZE):
        batch = users[start : start + BATCH_SIZE]
        recs |= model.recommend(history[history["user_id"].isin(batch)], k, read[read["user_id"].isin(batch)])
    return recs


def _rank_metrics(recs: dict[int, list[int]], answers: pd.DataFrame, k: int) -> pd.DataFrame:
    """독자별 추천 목록과 정답(user_id, book_id)으로 독자별 precision·recall·ndcg를 계산한다. 정답이 없는 독자는 NaN이다."""
    users = list(recs)
    # NOTE: Series.explode()는 리스트가 든 칸을 원소마다 한 행으로 펼친다(인덱스 user_id는 원소 수만큼 반복).
    #       rename_axis("user_id")로 인덱스에 이름을 붙이고 reset_index()로 열로 꺼내 (user_id, book_id) 표를 만든다.
    ranked = pd.Series(recs).explode().astype(int).rename("book_id").rename_axis("user_id").reset_index()
    # NOTE: groupby("user_id").cumcount()는 독자 안에서 행마다 0, 1, 2…를 매긴다. 추천 순서대로 펼쳤으므로 +1이 순위다.
    ranked["rank"] = ranked.groupby("user_id").cumcount() + 1
    answers = answers[["user_id", "book_id"]].drop_duplicates()
    # NOTE: merge(on=[...])는 how 기본값이 "inner"라 양쪽에 모두 있는 (독자, 책)만 남는다. 즉 적중한 추천만 남는다.
    hit = ranked.merge(answers, on=["user_id", "book_id"])
    # NOTE: value_counts()는 독자별 행 수를 센다. reindex(users, fill_value=0)는 행을 users 순서로 맞추고 없는 독자는 0으로 채운다.
    n_answers = answers["user_id"].value_counts().reindex(users, fill_value=0)
    hits = hit.groupby("user_id").size().reindex(users, fill_value=0)
    dcg = (1 / np.log2(hit["rank"] + 1)).groupby(hit["user_id"]).sum().reindex(users, fill_value=0.0)
    # NOTE: idcg[n] = 정답 n권이 1~n위를 채웠을 때의 DCG. 1~k위의 1/log2(순위+1)을 np.cumsum으로 누적한 길이 k+1 배열을
    #       미리 만들고(0번 칸은 정답 0권), 독자별 정답 수(clip(upper=k)로 최대 k)를 위치로 써서 꺼낸다.
    idcg = np.concatenate([[0.0], np.cumsum(1 / np.log2(np.arange(2, k + 2)))])[n_answers.clip(upper=k)]
    metrics = pd.DataFrame({"precision": hits / k, "recall": hits / n_answers, "ndcg": dcg / idcg})
    # NOTE: where(조건, axis=0)는 조건이 False인 행(정답 0권 독자)을 NaN으로 바꾼다. mean()과 count()는 NaN을 건너뛴다.
    return metrics.where(n_answers > 0, axis=0)


def _summarize(recs: dict[int, list[int]], answers: pd.DataFrame, read: pd.DataFrame, to_read: pd.DataFrame | None, k: int) -> dict:
    """추천 목록을 정답(4점 이상)과 to_read(이미 읽은 책 제외)로 채점해 독자 평균과 평균 낸 독자 수를 돌려준다."""
    metrics = _rank_metrics(recs, answers, k)
    result = {
        f"precision@{k}": metrics["precision"].mean(),
        f"recall@{k}": metrics["recall"].mean(),
        f"ndcg@{k}": metrics["ndcg"].mean(),
        "users": metrics["precision"].count(),
    }
    if to_read is not None:
        # NOTE: merge(how="left", indicator=True)는 to_read 행을 모두 남기고, read에도 있으면 "both", 없으면 "left_only"를
        #       _merge 열에 적는다. "left_only"만 남겨 이미 읽은 책을 뺀다.
        wanted = to_read[to_read["user_id"].isin(list(recs))].merge(
            read[["user_id", "book_id"]], on=["user_id", "book_id"], how="left", indicator=True
        )
        to_read_hit = _rank_metrics(recs, wanted[wanted["_merge"] == "left_only"], k)["precision"]
        result |= {f"to_read_hit@{k}": to_read_hit.mean(), "to_read_users": to_read_hit.count()}
    return result


def evaluate_warm(
    model: Recommender,
    split: RatingSplit,
    on: str = "val",
    to_read: pd.DataFrame | None = None,
    n_users: int | None = None,
    seed: int = SEED,
    k: int = K,
) -> pd.Series:
    """기존 독자를 on 구간(val 또는 test)으로 채점한다. n_users를 주면 그만큼만 무작위로 뽑는다."""
    if on not in ("val", "test"):
        raise ValueError(f"on은 'val' 또는 'test'여야 합니다: {on!r}")
    answers = getattr(split, on)
    users = np.sort(answers["user_id"].unique())
    if n_users is not None:
        users = np.random.default_rng(seed).choice(users, size=min(n_users, len(users)), replace=False)
    # NOTE: test 시점의 독자는 val의 책도 이미 읽었으므로 이력에 val을 붙인다.
    history = split.train if on == "val" else pd.concat([split.train, split.val])
    history = history[history["user_id"].isin(users)]
    recs = _recommend(model, history, history, k)
    return pd.Series(_summarize(recs, answers[answers["rating"] >= RELEVANT_RATING], history, to_read, k), dtype=object)


def evaluate_cold(
    model: Recommender,
    split: RatingSplit,
    on: str = "tune",
    to_read: pd.DataFrame | None = None,
    history_sizes: tuple[int, ...] = COLD_HISTORY_SIZES,
    k: int = K,
) -> pd.DataFrame:
    """신규 독자(on = "tune" 또는 "final")의 앞 s개 평점을 이력으로 넣어 s마다 채점한다.

    모든 s에서 앞 max(history_sizes)개는 이미 읽은 책으로 추천에서 빼고, 그 뒤의 4점 이상을 정답으로 쓴다.
    """
    if on not in ("tune", "final"):
        raise ValueError(f"on은 'tune' 또는 'final'이어야 합니다: {on!r}")
    cold = getattr(split, f"cold_{on}")
    # NOTE: 파일 순서가 시간순이라 정렬하지 않고 순번을 매긴다.
    position = cold.groupby("user_id").cumcount()
    read = cold[position < max(history_sizes)]
    answers = cold[(position >= max(history_sizes)) & (cold["rating"] >= RELEVANT_RATING)]
    rows = {}
    for size in history_sizes:
        history = cold[position < size]
        rows[size] = _summarize(_recommend(model, history, read, k), answers, read, to_read, k)
    # NOTE: from_dict(orient="index")는 dict의 키(s)를 행, 안쪽 dict의 키(지표)를 열로 둔다.
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("history_size")


def evaluate_rmse(model: RatingPredictor, split: RatingSplit, on: str = "val") -> float:
    """on 구간(val 또는 test)의 실제 별점과 model.predict 예측의 RMSE를 계산한다."""
    if on not in ("val", "test"):
        raise ValueError(f"on은 'val' 또는 'test'여야 합니다: {on!r}")
    ratings = getattr(split, on)
    errors = model.predict(ratings[["user_id", "book_id"]]) - ratings["rating"].to_numpy()
    return float(np.sqrt(np.mean(errors**2)))
