"""어떤 모델이든 Goal 1~3의 순위 지표와 RMSE를 같은 방식으로 계산한다(A5)."""

import numpy as np
import pandas as pd

from bookrec.models.base import RatingPredictor, Recommender

RELEVANT_RATING = 4
K = 10
N_EVAL_USERS = 5000
SEED = 42
COLD_HISTORY_SIZES = (1, 3, 5, 10)
BATCH_SIZE = 500


def _recommend(model: Recommender, history: pd.DataFrame, k: int) -> dict[int, list[int]]:
    """독자를 BATCH_SIZE명씩 나눠 model.recommend를 부르고 결과를 합친다."""
    users = history["user_id"].unique()
    recs = {}
    for start in range(0, len(users), BATCH_SIZE):
        recs |= model.recommend(history[history["user_id"].isin(users[start : start + BATCH_SIZE])], k)
    return recs


def _rank_metrics(recs: dict[int, list[int]], answers: pd.DataFrame, k: int) -> pd.DataFrame:
    """독자별 추천 목록과 정답(user_id, book_id)으로 독자별 precision·recall·ndcg를 계산한다. 정답이 없는 독자는 NaN이다."""
    users = list(recs)
    ranked = pd.Series(recs).explode().astype(int).rename("book_id").rename_axis("user_id").reset_index()
    ranked["rank"] = ranked.groupby("user_id").cumcount() + 1
    answers = answers[["user_id", "book_id"]].drop_duplicates()
    hit = ranked.merge(answers, on=["user_id", "book_id"])
    n_answers = answers["user_id"].value_counts().reindex(users, fill_value=0)
    hits = hit.groupby("user_id").size().reindex(users, fill_value=0)
    dcg = (1 / np.log2(hit["rank"] + 1)).groupby(hit["user_id"]).sum().reindex(users, fill_value=0.0)
    # NOTE: idcg[n] = 정답 n권이 1~n위를 채웠을 때의 DCG.
    idcg = np.concatenate([[0.0], np.cumsum(1 / np.log2(np.arange(2, k + 2)))])[n_answers.clip(upper=k)]
    metrics = pd.DataFrame({"precision": hits / k, "recall": hits / n_answers, "ndcg": dcg / idcg})
    return metrics.where(n_answers > 0, axis=0)


def evaluate_warm(
    model: Recommender,
    history: pd.DataFrame,
    answers: pd.DataFrame,
    to_read: pd.DataFrame | None = None,
    n_users: int = N_EVAL_USERS,
    seed: int = SEED,
    k: int = K,
) -> pd.Series:
    """기존 독자 n_users명을 뽑아 Goal 2(정답 = answers의 4점 이상)와 Goal 3(정답 = 이력에 없는 to_read) 지표를 계산한다."""
    users = np.sort(answers["user_id"].unique())
    users = np.random.default_rng(seed).choice(users, size=min(n_users, len(users)), replace=False)
    history = history[history["user_id"].isin(users)]
    recs = _recommend(model, history, k)
    metrics = _rank_metrics(recs, answers[answers["rating"] >= RELEVANT_RATING], k)
    result = {
        f"precision@{k}": metrics["precision"].mean(),
        f"recall@{k}": metrics["recall"].mean(),
        f"ndcg@{k}": metrics["ndcg"].mean(),
        "users": metrics["precision"].count(),
    }
    if to_read is not None:
        wanted = to_read[to_read["user_id"].isin(users)].merge(
            history[["user_id", "book_id"]], on=["user_id", "book_id"], how="left", indicator=True
        )
        to_read_hit = _rank_metrics(recs, wanted[wanted["_merge"] == "left_only"], k)["precision"]
        result |= {f"to_read_hit@{k}": to_read_hit.mean(), "to_read_users": to_read_hit.count()}
    return pd.Series(result)


def evaluate_cold(
    model: Recommender, cold_ratings: pd.DataFrame, history_sizes: tuple[int, ...] = COLD_HISTORY_SIZES, k: int = K
) -> pd.DataFrame:
    """신규 독자의 앞 s개 평점을 이력으로 넣어 s마다 Goal 1 지표를 계산한다. 정답은 모든 s에서 max(history_sizes)번째 이후 4점 이상이다."""
    # NOTE: 파일 순서가 시간순이라 정렬하지 않고 순번을 매긴다.
    position = cold_ratings.groupby("user_id").cumcount()
    answers = cold_ratings[(position >= max(history_sizes)) & (cold_ratings["rating"] >= RELEVANT_RATING)]
    rows = {}
    for size in history_sizes:
        metrics = _rank_metrics(_recommend(model, cold_ratings[position < size], k), answers, k)
        rows[size] = {
            f"precision@{k}": metrics["precision"].mean(),
            f"recall@{k}": metrics["recall"].mean(),
            "users": metrics["precision"].count(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("history_size")


def evaluate_rmse(model: RatingPredictor, ratings: pd.DataFrame) -> float:
    """ratings(user_id, book_id, rating)의 실제 평점과 model.predict 예측의 RMSE를 계산한다."""
    errors = model.predict(ratings[["user_id", "book_id"]]) - ratings["rating"].to_numpy()
    return float(np.sqrt(np.mean(errors**2)))
