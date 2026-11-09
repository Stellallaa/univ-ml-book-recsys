import numpy as np
import pandas as pd
import pytest

from bookrec.models.content_based.tuning import ContentBasedConfig, make_content_based, rank_validation


def _result(candidate, cold, warm):
    rows = [{"candidate_id": candidate, "goal": "cold_tune", "history_size": str(s), "ndcg@10": n} for s, n in zip([1, 3, 5, 10], cold)]
    return rows + [{"candidate_id": candidate, "goal": "warm_val", "history_size": "train", "ndcg@10": warm}]


def test_ranking_prioritizes_all_cold_sizes_before_warm_score():
    # Given: A는 마지막 이력에서만 높지만 B의 네 이력 평균이 더 높다
    results = pd.DataFrame(_result("A", [0, 0, 0, 1], 1) + _result("B", [0.3] * 4, 0))

    # When
    ranking = rank_validation(results)

    # Then
    assert ranking["candidate_id"].tolist() == ["B", "A"]
    assert ranking["cold_mean_ndcg@10"].tolist() == pytest.approx([0.3, 0.25])


def test_ranking_breaks_ties_by_warm_then_id():
    # Given
    results = pd.DataFrame(_result("C", [0.2] * 4, 0.1) + _result("B", [0.2] * 4, 0.3) + _result("A", [0.2] * 4, 0.3))

    # When
    ranking = rank_validation(results.sample(frac=1, random_state=42))

    # Then
    assert ranking["candidate_id"].tolist() == ["A", "B", "C"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "test", "nan", "range", "warm_history"])
def test_ranking_rejects_incomplete_or_final_results(change):
    # Given
    results = pd.DataFrame(_result("A", [0.2] * 4, 0.1))
    if change == "missing":
        results = results.iloc[1:]
    elif change == "duplicate":
        results = pd.concat([results, results.iloc[:1]])
    elif change == "test":
        results.loc[0, "goal"] = "cold_final"
    elif change == "nan":
        results.loc[0, "ndcg@10"] = np.nan
    elif change == "range":
        results.loc[0, "ndcg@10"] = 1.1
    else:
        results.loc[4, "history_size"] = "test"

    # When / Then
    with pytest.raises(ValueError):
        rank_validation(results)


def test_factory_applies_tag_and_profile_settings_without_changing_input():
    # Given: basic는 favorites를 남기고 extended는 제거한다
    tags = pd.DataFrame([(1, "fantasy", 1), (2, "fantasy", 1), (1, "favorites", 5), (2, "favorites", 5)], columns=["book_id", "tag_name", "count"])
    original = tags.copy(deep=True)
    train = pd.DataFrame([(1, 2, 5)], columns=["user_id", "book_id", "rating"])
    history = pd.DataFrame([(7, 1, 2)], columns=train.columns)

    # When
    basic = make_content_based(tags, ContentBasedConfig(extended=False, profile_weighting="rating")).fit(train)
    extended = make_content_based(tags, ContentBasedConfig()).fit(train)

    # Then
    assert "favorites" in basic.features.tag_names
    assert "favorites" not in extended.features.tag_names
    assert not basic.profile_summary(history).loc[7, "fallback"]
    assert extended.profile_summary(history).loc[7, "fallback"]
    pd.testing.assert_frame_equal(tags, original)


def test_factory_rejects_non_boolean_flags():
    # Given
    tags = pd.DataFrame([(1, "fantasy", 1)], columns=["book_id", "tag_name", "count"])

    # When / Then
    with pytest.raises(ValueError):
        make_content_based(tags, ContentBasedConfig(extended="false"))
