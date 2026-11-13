"""Tune CB/MF score fusion on validation, then optionally score the holdout."""

import argparse
import json
from pathlib import Path

import pandas as pd

from bookrec.data import load_book_tags, load_books, load_split, load_to_read
from bookrec.evaluation import evaluate_cold, evaluate_warm
from bookrec.models.collaborative_filtering import fit_selected_mf
from bookrec.models.content_based.tuning import load_selected_content_based
from bookrec.models.hybrid import HybridRecommender
from bookrec.models.collaborative_filtering.tuning import result_rows
from bookrec.models.baselines.goodreads_rating import GoodreadsRatingRecommender
from bookrec.models.baselines.popularity import PopularityRecommender

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "hybrid"
EVAL_DIR = ROOT / "results" / "evaluation"
SELECTED_PATH = OUT_DIR / "selected_config.json"


def candidates():
    """Small, explicit grid covering the three fusion approaches in the proposal."""
    return [
        (f"fixed-mf-{weight:.2f}", {"mode": "fixed", "mf_weight": weight})
        for weight in (0.25, 0.50, 0.75)
    ] + [
        (f"switch-at-{threshold}", {"mode": "switch", "switch_at": threshold})
        for threshold in (3, 5, 10)
    ] + [
        (f"gradual-{start}-to-{end}", {"mode": "gradual", "switch_at": start, "full_mf_at": end})
        for start, end in ((1, 5), (1, 10), (3, 10))
    ]


def _new_models(train, book_tags):
    cb = load_selected_content_based(book_tags=book_tags).fit(train)
    mf = fit_selected_mf("ranking", train)
    return cb, mf


def tune():
    split, read = load_split(), load_to_read()
    cb, mf = _new_models(split.train, load_book_tags())
    frames, summaries = [], []
    for candidate_id, config in candidates():
        model = HybridRecommender(cb, mf, **config)
        warm = evaluate_warm(model, split, on="val", to_read=read, n_users=1000)
        cold = evaluate_cold(model, split, on="tune", to_read=read)
        rows = pd.concat([
            result_rows("hybrid", "warm_val", warm).assign(candidate_id=candidate_id),
            result_rows("hybrid", "cold_tune", cold).assign(candidate_id=candidate_id),
        ], ignore_index=True)
        frames.append(rows)
        cold_mean = cold["ndcg@10"].mean()
        summaries.append({"candidate_id": candidate_id, **config,
                          "cold_mean_ndcg@10": cold_mean,
                          "warm_ndcg@10": warm["ndcg@10"],
                          "selection_score": (cold_mean + warm["ndcg@10"]) / 2,
                          "warm_users": warm["users"], "cold_users": int(cold["users"].min())})
        print(candidate_id, summaries[-1]["selection_score"], flush=True)

    table = pd.DataFrame(summaries).sort_values(
        ["selection_score", "cold_mean_ndcg@10", "warm_ndcg@10", "candidate_id"],
        ascending=[False, False, False, True], ignore_index=True,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "validation_scores.csv", index=False)
    table.to_csv(OUT_DIR / "candidate_comparison.csv", index=False)
    selected = table.iloc[0].to_dict()
    selected_config = {}
    for key in ("mode", "mf_weight", "switch_at", "full_mf_at"):
        if key not in selected or pd.isna(selected[key]):
            continue
        selected_config[key] = int(selected[key]) if key in {"switch_at", "full_mf_at"} else selected[key]
    SELECTED_PATH.write_text(json.dumps({
        "candidate_id": selected["candidate_id"],
        "config": selected_config,
        "criterion": "mean of warm_val NDCG@10 (1000 users, seed 42) and mean cold_tune NDCG@10 (history sizes 1,3,5,10)",
        "train_source": "split.train", "final_evaluation_performed": False,
        "selection_score": selected["selection_score"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


def final():
    if not SELECTED_PATH.exists():
        raise FileNotFoundError(f"Run validation tuning first: {SELECTED_PATH}")
    selected = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    split, read = load_split(), load_to_read()
    train = pd.concat([split.train, split.val], ignore_index=True)
    cb, mf = _new_models(train, load_book_tags())
    model = HybridRecommender(cb, mf, **selected["config"])
    cb_warm = result_rows("cb", "warm_test", evaluate_warm(cb, split, on="test", to_read=read))
    cb_cold = result_rows("cb", "cold_final", evaluate_cold(cb, split, on="final", to_read=read))

    config = selected["config"]
    threshold = config.get("switch_at") if config.get("mode") == "switch" else None
    test_history = pd.concat([split.train, split.val], ignore_index=True)
    minimum_history = test_history.groupby("user_id").size().reindex(split.test["user_id"].unique()).min()
    mf_warm_path = EVAL_DIR / "mf_warm_test.csv"
    if threshold is not None and minimum_history >= threshold and mf_warm_path.exists():
        # Under a full-MF switch, the hybrid ranking is exactly MF's ranking.
        # Reuse the shared final score file rather than recomputing fold-in for
        # every warm user (the final training source and holdout protocol match).
        warm = pd.read_csv(mf_warm_path)
        if len(warm) != 1 or warm["model"].tolist() != ["mf"]:
            raise ValueError(f"Unexpected MF warm holdout result format: {mf_warm_path}")
        warm["model"] = "hybrid"
        warm["goal"] = "warm_test"
        warm["history_size"] = "all"
    else:
        warm = result_rows("hybrid", "warm_test", evaluate_warm(model, split, on="test", to_read=read))

    if threshold is not None and (EVAL_DIR / "mf_cold_final.csv").exists():
        mf_cold = pd.read_csv(EVAL_DIR / "mf_cold_final.csv")
        if set(mf_cold["history_size"].astype(int)) != {1, 3, 5, 10}:
            raise ValueError("MF cold_final must contain history sizes 1, 3, 5, and 10")
        cb_lookup = cb_cold.set_index("history_size")
        mf_lookup = mf_cold.set_index("history_size")
        hybrid_rows = []
        for size in (1, 3, 5, 10):
            source = cb_lookup if size < threshold else mf_lookup
            row = source.loc[size].to_dict()
            row.update(model="hybrid", goal="cold_final", history_size=size)
            hybrid_rows.append(row)
        cold = pd.DataFrame(hybrid_rows)
    else:
        cold = result_rows("hybrid", "cold_final", evaluate_cold(model, split, on="final", to_read=read))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warm.to_csv(OUT_DIR / "hybrid_warm_test.csv", index=False)
    cold.to_csv(OUT_DIR / "hybrid_cold_final.csv", index=False)
    pd.concat([cb_warm, cb_cold], ignore_index=True).to_csv(OUT_DIR / "cb_final_scores.csv", index=False)
    selected["final_evaluation_performed"] = True
    selected["final_train_source"] = "train + val"
    SELECTED_PATH.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.concat([cb_warm, cb_cold, warm, cold], ignore_index=True).to_string(index=False))


def export_validation_comparison():
    """Score standalone models and the selected hybrid on one shared validation cohort."""
    if not SELECTED_PATH.exists():
        raise FileNotFoundError(f"Run validation tuning first: {SELECTED_PATH}")
    selected = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    split, read = load_split(), load_to_read()
    tags = load_book_tags()
    cb, mf = _new_models(split.train, tags)
    models = {
        "popularity": PopularityRecommender(),
        "goodreads_rating": GoodreadsRatingRecommender(load_books()),
        "cb": cb,
        "mf": mf,
        "hybrid": HybridRecommender(cb, mf, **selected["config"]),
    }
    rows = []
    for name, model in models.items():
        if name in {"popularity", "goodreads_rating"}:
            model.fit(split.train)
        rows.append(result_rows(name, "warm_val", evaluate_warm(
            model, split, on="val", to_read=read, n_users=1000, seed=42)))
        rows.append(result_rows(name, "cold_tune", evaluate_cold(model, split, on="tune", to_read=read)))
        print(f"scored {name}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / "validation_models.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="score the selected setup on test and cold_final")
    parser.add_argument("--export-validation", action="store_true", help="evaluate all selected models on one shared validation cohort")
    args = parser.parse_args()
    if args.final:
        final()
    elif args.export_validation:
        export_validation_comparison()
    else:
        tune()


if __name__ == "__main__":
    main()
