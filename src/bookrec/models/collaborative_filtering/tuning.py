"""MF 설정 비교(C2)에 쓰는 함수와, 고른 최종 설정으로 모델을 만드는 함수."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from bookrec.data import RatingSplit
from bookrec.evaluation import evaluate_warm
from bookrec.models.collaborative_filtering.als import ALSMatrixFactorization
from bookrec.models.collaborative_filtering.mf import MatrixFactorization, MFConfig

RESULT_DIR = Path(__file__).resolve().parents[4] / "results" / "collaborative_filtering"
SELECTED_PATH = RESULT_DIR / "C2_selected_config.json"
# NOTE: data/는 커밋하지 않는 폴더다. 학습된 모델 파일은 여기에만 둔다.
CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "models"
# NOTE: 설정 후보를 비교할 때는 val 독자 5,000명(시드 고정)만 채점해 시간을 줄인다. 최종 점수는 전원으로 잰다.
TUNING_USERS = 5000
PURPOSES = ("rating", "ranking")


def run_experiment(config: MFConfig, split: RatingSplit, purpose: str, n_users: int = TUNING_USERS) -> tuple[dict, MatrixFactorization]:
    """설정 하나로 train을 학습하고 val 점수를 잰다. (기록 한 줄, 학습된 모델)을 돌려준다.

    purpose="rating"은 검증 RMSE가, "ranking"은 검증 NDCG@10이 가장 좋은 epoch에서 멈춘다(early stopping).
    기록에는 설정, 멈춘 epoch, 그 epoch의 val RMSE와 val 추천 지표, 걸린 시간이 들어간다.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose는 'rating' 또는 'ranking'이어야 합니다: {purpose!r}")
    monitor = None
    if purpose == "ranking":
        monitor = lambda model: evaluate_warm(model, split, on="val", n_users=n_users)["ndcg@10"]  # noqa: E731
    started = time.time()
    model = MatrixFactorization(config).fit(split.train, val=split.val, monitor=monitor)
    seconds = time.time() - started
    best = model.history[model.history["epoch"] == model.best_epoch].iloc[0]
    warm = evaluate_warm(model, split, on="val", n_users=n_users)
    record = {
        "purpose": purpose,
        **asdict(config),
        "best_epoch": model.best_epoch,
        "epochs_run": len(model.history),
        "train_rmse": best["train_rmse"],
        "val_rmse": best["val_rmse"],
        "val_precision@10": warm["precision@10"],
        "val_recall@10": warm["recall@10"],
        "val_ndcg@10": warm["ndcg@10"],
        "val_users": warm["users"],
        "train_seconds": round(seconds, 1),
    }
    return record, model


def save_selected(selected: dict[str, dict], path: Path = SELECTED_PATH) -> None:
    """용도별 최종 설정을 JSON으로 저장한다. selected = {"rating": {"config": {...}, ...}, "ranking": {...}}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: indent=2는 사람이 읽기 쉽게 들여쓰고, ensure_ascii=False는 한글을 그대로 저장한다.
    path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")


def load_selected_mf(purpose: str = "ranking", path: Path = SELECTED_PATH) -> MatrixFactorization | ALSMatrixFactorization:
    """C2에서 고른 최종 설정으로 학습 전 MF 모델을 만든다. 쓰는 쪽에서 .fit(split.train)을 부른다.

    - purpose="ranking": 추천(top-10)용. evaluate_warm, evaluate_cold, hybrid에는 이 모델을 쓴다.
    - purpose="rating": 별점 예측용. evaluate_rmse에는 이 모델을 쓴다.
    n_epochs가 튜닝에서 가장 좋았던 epoch로 저장되어 있어, val 없이 fit(train)만 해도 같은 모델이 나온다.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose는 'rating' 또는 'ranking'이어야 합니다: {purpose!r}")
    saved = json.loads(Path(path).read_text())[purpose]
    # NOTE: 별점 예측용은 C2 추가 실험에서 ALS가 더 좋으면 ALS로 저장된다. ALS 모델은 predict만 있고 score는 없다.
    if saved.get("optimizer") == "als":
        return ALSMatrixFactorization(**saved["config"])
    return MatrixFactorization(MFConfig(**saved["config"]))


def result_rows(model_name: str, goal: str, result: dict | pd.DataFrame) -> pd.DataFrame:
    """evaluate_warm의 dict나 evaluate_cold의 표를 (model, goal, history_size, 지표…) 행으로 바꾼다. 결과 CSV 형식을 맞추는 데 쓴다."""
    if isinstance(result, dict):
        table = pd.DataFrame([result]).assign(history_size="all")
    else:
        table = result.reset_index()
    table.insert(0, "goal", goal)
    table.insert(0, "model", model_name)
    front = ["model", "goal", "history_size"]
    return table[front + [column for column in table.columns if column not in front]]


def fit_selected_mf(purpose: str, train: pd.DataFrame, cache_dir: Path = CACHE_DIR, path: Path = SELECTED_PATH):
    """최종 설정으로 train을 학습한 모델을 돌려준다. 추천용 MF는 학습 결과를 data/models/에 저장해 두고 다음부터는 불러온다.

    추천용 최종 설정은 학습에 10분 넘게 걸려서, 여러 노트북과 hybrid에서 매번 다시 학습하지 않기 위한 함수다.
    파일 이름에 설정이 들어 있어 설정이 바뀌면 새로 학습한다. train이 바뀌었으면 data/models/의 파일을 지우고 부른다.
    """
    model = load_selected_mf(purpose, path)
    if not isinstance(model, MatrixFactorization):
        return model.fit(train)
    config = model.config
    cache = Path(cache_dir) / (
        f"mf_{purpose}_f{config.n_factors}_lr{config.lr}_reg{config.reg}_neg{config.neg_ratio}_e{config.n_epochs}_b{config.batch_size}_s{config.seed}.npz"
    )
    if cache.exists():
        loaded = MatrixFactorization.load(cache)
        # NOTE: fold_in_reg는 학습과 무관하게 fold-in 때만 쓰는 값이라, 저장된 책 벡터에 지금 설정의 값을 적용한다.
        loaded.config = config
        return loaded
    model.fit(train)
    cache.parent.mkdir(parents=True, exist_ok=True)
    model.save(cache)
    return model
