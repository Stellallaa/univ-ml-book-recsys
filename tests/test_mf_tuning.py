import json

import numpy as np
import pandas as pd
import pytest

from bookrec.data import RatingSplit
from bookrec.models.collaborative_filtering import MFConfig, fit_selected_mf, load_selected_mf
from bookrec.models.collaborative_filtering.tuning import result_rows, run_experiment, save_selected


def _ratings(rows):
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])


def _small_split():
    """독자 1~20은 책 1~5를, 독자 21~40은 책 6~10을 좋아한다. 독자마다 앞 8권이 train, 뒤 2권이 val이다."""
    train, val = [], []
    for user in range(1, 41):
        likes_first_group = user <= 20
        books = [(book + user) % 10 + 1 for book in range(10)]
        for position, book in enumerate(books):
            row = (user, book, 5 if (book <= 5) == likes_first_group else 1)
            (train if position < 8 else val).append(row)
    empty = _ratings([])
    return RatingSplit(train=_ratings(train), val=_ratings(val), test=empty, cold_tune=empty, cold_final=empty)


def test_run_experiment_records_config_and_validation_scores_for_rating_model():
    # Given
    config = MFConfig(n_factors=4, lr=0.05, reg=0.01, n_epochs=40, batch_size=64, patience=3)

    # When
    record, model = run_experiment(config, _small_split(), "rating")

    # Then
    assert record["purpose"] == "rating" and record["n_factors"] == 4 and record["neg_ratio"] == 0
    assert record["best_epoch"] == model.best_epoch <= record["epochs_run"] <= 40
    assert record["val_rmse"] == pytest.approx(model.history["val_rmse"].min())
    assert 0 <= record["val_ndcg@10"] <= 1


def test_run_experiment_stops_ranking_model_on_validation_ndcg():
    # Given
    config = MFConfig(n_factors=4, lr=0.05, reg=0.01, neg_ratio=1, n_epochs=40, batch_size=64, patience=3)

    # When
    record, model = run_experiment(config, _small_split(), "ranking", n_users=20)

    # Then: 멈춘 epoch의 monitor 값이 기록된 val NDCG@10이고, 그 값이 전체 epoch 중 가장 높다
    assert record["val_users"] <= 20
    assert record["val_ndcg@10"] == pytest.approx(model.history["monitor"].max())
    assert model.history.loc[model.history["epoch"] == record["best_epoch"], "monitor"].iloc[0] == pytest.approx(record["val_ndcg@10"])


def test_run_experiment_rejects_unknown_purpose():
    # Given
    split = _small_split()

    # When / Then
    with pytest.raises(ValueError):
        run_experiment(MFConfig(), split, "hybrid")


def test_load_selected_mf_builds_unfitted_model_from_saved_config(tmp_path):
    # Given
    path = tmp_path / "selected.json"
    ranking = MFConfig(n_factors=16, neg_ratio=2, n_epochs=4, fold_in_reg=10.0)
    rating = MFConfig(n_factors=64, n_epochs=21)
    save_selected({"ranking": {"config": ranking.__dict__}, "rating": {"config": rating.__dict__}}, path)

    # When
    ranking_model = load_selected_mf("ranking", path)
    rating_model = load_selected_mf("rating", path)

    # Then
    assert ranking_model.config == ranking
    assert rating_model.config == rating
    assert not hasattr(ranking_model, "Q")
    assert json.loads(path.read_text())["ranking"]["config"]["fold_in_reg"] == 10.0


def test_load_selected_mf_rejects_unknown_purpose(tmp_path):
    # Given
    path = tmp_path / "selected.json"
    save_selected({"ranking": {"config": MFConfig().__dict__}}, path)

    # When / Then
    with pytest.raises(ValueError):
        load_selected_mf("hybrid", path)


def test_result_rows_formats_warm_dict_and_cold_table_alike():
    # Given
    warm = {"precision@10": 0.1, "ndcg@10": 0.2, "users": 5}
    cold = pd.DataFrame({"precision@10": [0.1, 0.3], "ndcg@10": [0.2, 0.4], "users": [5, 5]}, index=pd.Index([1, 3], name="history_size"))

    # When
    warm_rows = result_rows("mf", "warm_val", warm)
    cold_rows = result_rows("mf", "cold_tune", cold)

    # Then
    assert warm_rows.columns.tolist() == cold_rows.columns.tolist() == ["model", "goal", "history_size", "precision@10", "ndcg@10", "users"]
    assert warm_rows["history_size"].tolist() == ["all"]
    assert cold_rows["history_size"].tolist() == [1, 3] and cold_rows["goal"].tolist() == ["cold_tune", "cold_tune"]


def test_load_selected_mf_builds_als_model_when_saved_optimizer_is_als(tmp_path):
    # Given
    path = tmp_path / "selected.json"
    save_selected({"rating": {"optimizer": "als", "config": {"n_factors": 8, "reg": 0.1, "n_epochs": 5, "patience": 2, "seed": 42}}}, path)

    # When
    model = load_selected_mf("rating", path)

    # Then
    assert type(model).__name__ == "ALSMatrixFactorization"
    assert (model.n_factors, model.reg, model.n_epochs) == (8, 0.1, 5)


def test_fit_selected_mf_reuses_saved_model_and_applies_current_fold_in_reg(tmp_path):
    # Given: 추천용 설정으로 한 번 학습해 저장한 뒤, fold_in_reg만 바꾼 설정 파일
    split = _small_split()
    config = MFConfig(n_factors=4, lr=0.05, n_epochs=3, batch_size=64, neg_ratio=1, fold_in_reg=5.0)
    path = tmp_path / "selected.json"
    save_selected({"ranking": {"config": config.__dict__}}, path)
    first = fit_selected_mf("ranking", split.train, cache_dir=tmp_path, path=path)
    save_selected({"ranking": {"config": {**config.__dict__, "fold_in_reg": 1.0}}}, path)

    # When
    second = fit_selected_mf("ranking", split.train, cache_dir=tmp_path, path=path)

    # Then: 다시 학습하지 않아 책 벡터가 같고, fold_in_reg는 새 값이다
    assert len(list(tmp_path.glob("*.npz"))) == 1
    assert np.array_equal(first.Q, second.Q)
    assert second.config.fold_in_reg == 1.0
