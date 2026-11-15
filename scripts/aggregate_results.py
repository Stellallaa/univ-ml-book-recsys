"""Collect model evaluation CSVs into comparable tables and figures (D4)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bookrec.plot_style import MODEL_COLORS, save, setup

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation"
OUT = ROOT / "results" / "final"


def collect(phase: str) -> pd.DataFrame:
    """Read the requested validation or holdout sources; absent files are skipped."""
    if phase not in {"validation", "holdout"}:
        raise ValueError("phase must be validation or holdout")
    warm_on, cold_on = ("val", "tune") if phase == "validation" else ("test", "final")
    shared_validation = (ROOT / "results" / "hybrid" / "validation_models.csv")
    has_shared_validation = phase == "validation" and shared_validation.exists()
    tables = []
    for source in ("baselines", "mf"):
        for kind, goal in (("warm", f"warm_{warm_on}"),
                           ("cold", f"cold_{cold_on}"),
                           ("rmse", f"rmse_{warm_on}")):
            if has_shared_validation and kind != "rmse":
                continue
            path = EVAL / f"{source}_{goal}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            if "model" not in frame or frame["model"].isna().any():
                raise ValueError(f"{path}: invalid model column")
            if kind == "rmse":
                frame["history_size"] = "all"
            elif "history_size" not in frame:
                frame["history_size"] = "all" if kind == "warm" else None
            frame["goal"] = goal
            tables.append(frame)

    if has_shared_validation:
        shared = pd.read_csv(shared_validation)
        if not shared.empty:
            tables.append(shared)

    cb_path = (ROOT / "results" / "content_based" / "B4_validation_scores.csv" if phase == "validation"
               else ROOT / "results" / "hybrid" / "cb_final_scores.csv")
    if cb_path.exists() and not has_shared_validation:
        cb = pd.read_csv(cb_path)
        if phase == "validation" and "candidate_id" in cb:
            cb = cb[cb["candidate_id"].astype(str) == "C17"].copy()
        if not cb.empty:
            cb["model"] = "cb"
            cb["goal"] = cb["goal"].map({"warm_val": "warm_val", "cold_tune": "cold_tune",
                                         "warm_test": "warm_test", "cold_final": "cold_final"}).fillna(cb["goal"])
            tables.append(cb)

    hybrid_dir = ROOT / "results" / "hybrid"
    hybrid_paths = ([] if has_shared_validation else [hybrid_dir / "validation_scores.csv"] if phase == "validation" else
                    [hybrid_dir / "hybrid_warm_test.csv", hybrid_dir / "hybrid_cold_final.csv"])
    for path in hybrid_paths:
        if path.exists():
            hybrid = pd.read_csv(path)
            if not hybrid.empty:
                if "candidate_id" in hybrid:
                    selected_path = hybrid_dir / "selected_config.json"
                    if selected_path.exists():
                        import json
                        selected = json.loads(selected_path.read_text(encoding="utf-8"))["candidate_id"]
                        hybrid = hybrid[hybrid["candidate_id"] == selected].copy()
                hybrid["model"] = "hybrid"
                tables.append(hybrid)

    if not tables:
        return pd.DataFrame()
    combined = pd.concat(tables, ignore_index=True, sort=False)
    required = {"model", "goal", "history_size"}
    if not required.issubset(combined.columns):
        raise ValueError(f"Collected results are missing columns: {sorted(required - set(combined.columns))}")
    combined["history_size"] = combined["history_size"].fillna("all").astype(str)
    keys = ["model", "goal", "history_size"]
    if combined.duplicated(keys).any():
        dupes = combined.loc[combined.duplicated(keys, keep=False), keys].drop_duplicates()
        raise ValueError(f"Duplicate model results found:\n{dupes.to_string(index=False)}")
    for metric in ("precision@10", "recall@10", "ndcg@10", "to_read_hit@10", "rmse"):
        if metric in combined:
            combined[metric] = pd.to_numeric(combined[metric], errors="coerce")
            invalid = combined[metric].notna() & ~np.isfinite(combined[metric])
            if invalid.any():
                raise ValueError(f"{metric} contains non-finite values")
    return combined.sort_values(keys, ignore_index=True)


def add_improvement(table: pd.DataFrame) -> pd.DataFrame:
    """Add per-goal delta against popularity (ranking) or bias (RMSE)."""
    table = table.copy()
    table["baseline_model"] = np.where(table["goal"].astype(str).str.startswith("rmse_"), "bias", "popularity")
    keys = pd.MultiIndex.from_frame(table[["goal", "history_size"]])
    for metric in ("precision@10", "recall@10", "ndcg@10", "to_read_hit@10"):
        delta_column = metric.replace("@10", "_improvement")
        table[delta_column] = np.nan
        if metric in table:
            reference = table[table["model"] == "popularity"].set_index(["goal", "history_size"])[metric]
            table[delta_column] = table[metric].to_numpy() - reference.reindex(keys).to_numpy()
    table["rmse_improvement"] = np.nan
    if "rmse" in table:
        reference = table[table["model"] == "bias"].set_index(["goal", "history_size"])["rmse"]
        # Positive means lower error than the bias baseline.
        table["rmse_improvement"] = reference.reindex(keys).to_numpy() - table["rmse"].to_numpy()
    return table


def _plot(table: pd.DataFrame, phase: str) -> None:
    setup()
    for goal in ("warm_val", "cold_tune") if phase == "validation" else ("warm_test", "cold_final"):
        subset = table[table["goal"] == goal]
        metric = "ndcg@10"
        if subset.empty or metric not in subset or subset[metric].notna().sum() == 0:
            continue
        if goal.startswith("cold"):
            subset = subset[subset["history_size"].isin(["1", "3", "5", "10"])]
            fig, ax = plt.subplots()
            for model, rows in subset.groupby("model", sort=True):
                rows = rows.set_index("history_size").reindex(["1", "3", "5", "10"])
                ax.plot([1, 3, 5, 10], rows[metric], marker="o", label=model,
                        color=MODEL_COLORS.get(model, MODEL_COLORS["hybrid"] if model == "hybrid" else None))
            ax.set(xlabel="Number of history ratings", ylabel="NDCG@10", title=f"{goal}: model comparison")
            ax.legend()
        else:
            subset = subset[subset["history_size"] == "all"]
            if subset.empty:
                continue
            fig, ax = plt.subplots()
            ordered = subset.sort_values(metric)
            colors = [MODEL_COLORS.get(m, MODEL_COLORS["hybrid"]) for m in ordered["model"]]
            ax.bar(ordered["model"], ordered[metric], color=colors)
            ax.set(ylabel="NDCG@10", title=f"{goal}: model comparison")
            ax.tick_params(axis="x", rotation=20)
        save(fig, f"D4_{goal}_comparison")
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for phase in ("validation", "holdout"):
        table = add_improvement(collect(phase))
        if table.empty:
            print(f"No {phase} results found; skipped")
            continue
        table.to_csv(OUT / f"{phase}_comparison.csv", index=False)
        _plot(table, phase)
        print(f"Saved {len(table)} {phase} rows to {OUT}")


if __name__ == "__main__":
    main()
