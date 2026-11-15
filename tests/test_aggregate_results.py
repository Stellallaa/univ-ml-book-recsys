import pandas as pd
import pytest

from scripts import aggregate_results
from scripts.aggregate_results import add_improvement


def test_add_improvement_uses_matching_goal_baseline():
    # Given
    table = pd.DataFrame([
        {"model": "popularity", "goal": "cold_tune", "history_size": "1", "precision@10": 0.1, "ndcg@10": 0.2},
        {"model": "hybrid", "goal": "cold_tune", "history_size": "1", "precision@10": 0.15, "ndcg@10": 0.25},
        {"model": "bias", "goal": "rmse_val", "history_size": "all", "rmse": 0.9},
        {"model": "mf", "goal": "rmse_val", "history_size": "all", "rmse": 0.8},
    ])

    # When
    result = add_improvement(table)

    # Then: rank delta is model - popularity; RMSE improvement is baseline - model.
    assert result.loc[result.model == "hybrid", "ndcg_improvement"].item() == pytest.approx(0.05)
    assert result.loc[result.model == "hybrid", "precision_improvement"].item() == pytest.approx(0.05)
    assert result.loc[result.model == "mf", "rmse_improvement"].item() == pytest.approx(0.1)


def test_add_improvement_keeps_missing_baseline_as_nan():
    table = pd.DataFrame([{"model": "hybrid", "goal": "warm_test", "history_size": "all", "ndcg@10": 0.3}])
    result = add_improvement(table)
    assert pd.isna(result.loc[0, "ndcg_improvement"])


def test_collect_returns_empty_when_sources_are_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_results, "ROOT", tmp_path)
    monkeypatch.setattr(aggregate_results, "EVAL", tmp_path / "results" / "evaluation")
    assert aggregate_results.collect("validation").empty


def test_collect_rejects_duplicate_model_goal_rows(tmp_path, monkeypatch):
    evaluation = tmp_path / "results" / "evaluation"
    evaluation.mkdir(parents=True)
    row = "model,precision@10,recall@10,ndcg@10,users\npopularity,0.1,0.1,0.1,10\n"
    (evaluation / "baselines_warm_val.csv").write_text(row, encoding="utf-8")
    (evaluation / "mf_warm_val.csv").write_text(row, encoding="utf-8")
    monkeypatch.setattr(aggregate_results, "ROOT", tmp_path)
    monkeypatch.setattr(aggregate_results, "EVAL", evaluation)
    with pytest.raises(ValueError, match="Duplicate model results"):
        aggregate_results.collect("validation")
