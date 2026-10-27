"""Complete flow-measurement sensitivity and targeted random-forest checks.

The reduced flow composite omits SMC1, SMC6, and SMC9. Exact Shapley values
reuse the original scores for the 32 subsets that do not contain flow and
refit only the 32 subsets affected by the reduced flow representation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from construct_level_cv import ROOT, SEED, THEORY_GROUPS, make_model, prepare_data
from domain_shapley import exact_shapley, groups_for_mask
from unified_algorithm_benchmark import build_models


OUT = ROOT / "results" / "phase11_flow_measurement_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT / "reduced_flow_domain_subset_scores.csv"
ORIGINAL_SCORES = ROOT / "results" / "phase7_domain_shapley" / "domain_subset_scores.csv"
ORIGINAL_SHAPLEY = ROOT / "results" / "phase7_domain_shapley" / "domain_shapley_summary.csv"
RF_FULL_SCORES = ROOT / "results" / "phase5_algorithm_benchmark" / "algorithm_scores_25_splits.csv"

REDUCED_ITEMS = ["SMC2", "SMC3", "SMC4", "SMC5", "SMC7", "SMC8"]


def columns_for_mask(mask: int, names: list[str], groups: dict[str, list[str]]) -> list[str]:
    return [column for i, name in enumerate(names) if mask & (1 << i)
            for column in groups[name]]


def evaluate_subset(mask, names, groups, data, outcome, splits):
    columns = columns_for_mask(mask, names, groups)
    rows = []
    for split_id, (train, test) in enumerate(splits, 1):
        model = make_model(columns)
        model.fit(data.iloc[train][columns], outcome.iloc[train])
        prediction = model.predict(data.iloc[test][columns])
        rows.append({
            "subset_mask": mask,
            "subset_groups": "|".join(groups_for_mask(mask, names)),
            "n_groups": int(mask.bit_count()),
            "split_id": split_id,
            "repeat": (split_id - 1) // 5 + 1,
            "fold": (split_id - 1) % 5 + 1,
            "r2": r2_score(outcome.iloc[test], prediction),
            "mae": mean_absolute_error(outcome.iloc[test], prediction),
            "rmse": mean_squared_error(outcome.iloc[test], prediction) ** .5,
        })
    return pd.DataFrame(rows)


def summarize_specification(scores, mask, label):
    part = scores[scores.subset_mask.eq(mask)].sort_values("split_id")
    return {
        "specification": label,
        "r2_mean": part.r2.mean(),
        "r2_sd": part.r2.std(ddof=1),
        "mae_mean": part.mae.mean(),
        "rmse_mean": part.rmse.mean(),
    }


def run_reduced_shapley(data, outcome, splits):
    names = list(THEORY_GROUPS)
    flow_index = names.index("trait_flow")
    flow_bit = 1 << flow_index
    groups = {name: list(columns) for name, columns in THEORY_GROUPS.items()}
    groups["trait_flow"] = ["SMC_reduced"]

    original = pd.read_csv(ORIGINAL_SCORES)
    unaffected = original[original.subset_mask.map(lambda value: not (int(value) & flow_bit))].copy()
    if CHECKPOINT.exists():
        affected = pd.read_csv(CHECKPOINT)
    else:
        affected = pd.DataFrame()
    complete_masks = set(
        affected.groupby("subset_mask").size().loc[lambda s: s.eq(25)].index
    ) if len(affected) else set()

    affected_masks = [mask for mask in range(1 << len(names)) if mask & flow_bit]
    for position, mask in enumerate(affected_masks, 1):
        if mask in complete_masks:
            continue
        print(
            f"Reduced-flow subset {position}/{len(affected_masks)}: "
            f"{groups_for_mask(mask, names)}",
            flush=True,
        )
        if len(affected):
            affected = affected[affected.subset_mask.ne(mask)]
        new_scores = evaluate_subset(mask, names, groups, data, outcome, splits)
        affected = pd.concat([affected, new_scores], ignore_index=True)
        affected.sort_values(["subset_mask", "split_id"]).to_csv(
            CHECKPOINT, index=False, encoding="utf-8-sig"
        )

    scores = pd.concat([unaffected, affected], ignore_index=True)
    scores = scores.sort_values(["subset_mask", "split_id"]).reset_index(drop=True)
    if len(scores) != 64 * 25 or not scores.groupby("subset_mask").size().eq(25).all():
        raise RuntimeError("Reduced-flow subset grid is incomplete")

    shapley, marginal = exact_shapley(scores, names)
    summary = (shapley.groupby("domain", as_index=False)
               .agg(shapley_r2_mean=("shapley_r2", "mean"),
                    shapley_r2_sd=("shapley_r2", "std"),
                    positive_splits=("shapley_r2", lambda x: int((x > 0).sum()))))
    summary = summary.sort_values("shapley_r2_mean", ascending=False)

    full_mask = (1 << len(names)) - 1
    without_flow_mask = full_mask ^ flow_bit
    flow_only_mask = flow_bit
    empty_mask = 0
    flow_shapley = float(
        summary.loc[summary.domain.eq("trait_flow"), "shapley_r2_mean"].iloc[0]
    )
    full = scores[scores.subset_mask.eq(full_mask)].sort_values("split_id")
    without = scores[scores.subset_mask.eq(without_flow_mask)].sort_values("split_id")
    deletion = full.r2.to_numpy() - without.r2.to_numpy()
    empty = scores[scores.subset_mask.eq(empty_mask)].sort_values("split_id")
    full_gain = (full.r2.to_numpy() - empty.r2.to_numpy()).mean()

    key = pd.DataFrame([
        summarize_specification(scores, full_mask, "reduced_flow_full_six_domain"),
        summarize_specification(scores, flow_only_mask, "reduced_flow_only"),
        summarize_specification(scores, without_flow_mask, "model_without_flow"),
    ])
    key["flow_shapley_r2"] = np.nan
    key["flow_deletion_loss_r2"] = np.nan
    key.loc[key.specification.eq("reduced_flow_full_six_domain"), "flow_shapley_r2"] = flow_shapley
    key.loc[key.specification.eq("reduced_flow_full_six_domain"), "flow_deletion_loss_r2"] = deletion.mean()
    key.loc[key.specification.eq("reduced_flow_full_six_domain"), "flow_shapley_share"] = flow_shapley / full_gain

    scores.to_csv(OUT / "reduced_flow_domain_subset_scores_complete.csv", index=False, encoding="utf-8-sig")
    shapley.to_csv(OUT / "reduced_flow_domain_shapley_by_split.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "reduced_flow_domain_shapley_summary.csv", index=False, encoding="utf-8-sig")
    marginal.to_csv(OUT / "reduced_flow_domain_marginal_contributions.csv", index=False, encoding="utf-8-sig")
    key.to_csv(OUT / "reduced_flow_key_metrics.csv", index=False, encoding="utf-8-sig")
    return scores, summary, key


def run_random_forest(data, outcome, splits):
    full_columns = sum(THEORY_GROUPS.values(), [])
    specifications = {
        "random_forest_flow_only": THEORY_GROUPS["trait_flow"],
        "random_forest_without_flow": [c for c in full_columns if c not in THEORY_GROUPS["trait_flow"]],
    }
    rows = []
    for label, columns in specifications.items():
        print(f"Running {label}", flush=True)
        for split_id, (train, test) in enumerate(splits, 1):
            model = build_models(columns)["RandomForest"]
            model.fit(data.iloc[train][columns], outcome.iloc[train])
            prediction = model.predict(data.iloc[test][columns])
            rows.append({
                "specification": label,
                "split_id": split_id,
                "repeat": (split_id - 1) // 5 + 1,
                "fold": (split_id - 1) % 5 + 1,
                "r2": r2_score(outcome.iloc[test], prediction),
                "mae": mean_absolute_error(outcome.iloc[test], prediction),
                "rmse": mean_squared_error(outcome.iloc[test], prediction) ** .5,
            })
    scores = pd.DataFrame(rows)
    original = pd.read_csv(RF_FULL_SCORES)
    full = original[original.model.eq("RandomForest")].copy()
    full["specification"] = "random_forest_full"
    full = full[["specification", "split_id", "repeat", "fold", "r2", "mae", "rmse"]]
    scores = pd.concat([full, scores], ignore_index=True)
    summary = (scores.groupby("specification", as_index=False)
               .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                    mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean")))
    wide = scores.pivot(index="split_id", columns="specification", values="r2")
    deletion = wide.random_forest_full - wide.random_forest_without_flow
    summary["flow_deletion_loss_r2"] = np.nan
    summary.loc[summary.specification.eq("random_forest_full"), "flow_deletion_loss_r2"] = deletion.mean()
    scores.to_csv(OUT / "random_forest_flow_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "random_forest_flow_summary.csv", index=False, encoding="utf-8-sig")
    return scores, summary


def build_comparison(reduced_key):
    original_scores = pd.read_csv(ORIGINAL_SCORES)
    names = list(THEORY_GROUPS)
    flow_bit = 1 << names.index("trait_flow")
    full_mask = (1 << len(names)) - 1
    without_flow_mask = full_mask ^ flow_bit
    original_full = original_scores[original_scores.subset_mask.eq(full_mask)].sort_values("split_id")
    original_without = original_scores[original_scores.subset_mask.eq(without_flow_mask)].sort_values("split_id")
    original_flow_only = original_scores[original_scores.subset_mask.eq(flow_bit)].sort_values("split_id")
    original_shapley = pd.read_csv(ORIGINAL_SHAPLEY)
    original_flow_shapley = float(
        original_shapley.loc[original_shapley.domain.eq("trait_flow"), "shapley_r2_mean"].iloc[0]
    )
    reduced_full = reduced_key[reduced_key.specification.eq("reduced_flow_full_six_domain")].iloc[0]
    reduced_only = reduced_key[reduced_key.specification.eq("reduced_flow_only")].iloc[0]
    reduced_without = reduced_key[reduced_key.specification.eq("model_without_flow")].iloc[0]
    comparison = pd.DataFrame([
        {
            "flow_representation": "original_9_item_composite",
            "full_six_domain_r2": original_full.r2.mean(),
            "flow_only_r2": original_flow_only.r2.mean(),
            "model_without_flow_r2": original_without.r2.mean(),
            "flow_shapley_r2": original_flow_shapley,
            "flow_deletion_loss_r2": (original_full.r2.to_numpy() - original_without.r2.to_numpy()).mean(),
        },
        {
            "flow_representation": "reduced_6_item_composite",
            "full_six_domain_r2": reduced_full.r2_mean,
            "flow_only_r2": reduced_only.r2_mean,
            "model_without_flow_r2": reduced_without.r2_mean,
            "flow_shapley_r2": reduced_full.flow_shapley_r2,
            "flow_deletion_loss_r2": reduced_full.flow_deletion_loss_r2,
        },
    ])
    comparison.to_csv(OUT / "original_vs_reduced_flow_comparison.csv", index=False, encoding="utf-8-sig")
    return comparison


def main():
    data, _ = prepare_data()
    data["SMC_reduced"] = data[REDUCED_ITEMS].mean(axis=1)
    outcome = data.CPSA.astype(float)
    splits = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(data, outcome))
    _, reduced_summary, reduced_key = run_reduced_shapley(data, outcome, splits)
    _, rf_summary = run_random_forest(data, outcome, splits)
    comparison = build_comparison(reduced_key)
    metadata = {
        "seed": SEED,
        "outer_validation": "5 repeats x 5 folds; identical to primary analyses",
        "original_flow_items": [f"SMC{i}" for i in range(1, 10)],
        "reduced_flow_items": REDUCED_ITEMS,
        "omitted_semantically_adjacent_items": ["SMC1", "SMC6", "SMC9"],
        "elastic_net_inner_validation": "5-fold ElasticNetCV within each outer training fold",
        "random_forest_parameters": {
            "n_estimators": 500, "min_samples_leaf": 5,
            "max_features": 0.7, "random_state": SEED,
        },
    }
    (OUT / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pd.ExcelWriter(OUT / "flow_measurement_sensitivity_complete.xlsx") as writer:
        comparison.to_excel(writer, sheet_name="original_vs_reduced", index=False)
        reduced_summary.to_excel(writer, sheet_name="reduced_shapley", index=False)
        rf_summary.to_excel(writer, sheet_name="random_forest", index=False)
    print("\nORIGINAL VS REDUCED FLOW\n", comparison.to_string(index=False))
    print("\nRANDOM FOREST\n", rf_summary.to_string(index=False))


if __name__ == "__main__":
    main()
