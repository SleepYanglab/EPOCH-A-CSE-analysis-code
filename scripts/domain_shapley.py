"""Exact cross-validated Shapley decomposition across six predictor domains.

Every one of the 2^6 domain subsets is evaluated on the same 5x5 repeated
outer folds.  The resulting Shapley values allocate predictive utility averaged
over all possible domain entry orders; they are not causal effects or unique
variance components.
"""
from __future__ import annotations

import json
from math import factorial

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from construct_level_cv import ROOT, SEED, THEORY_GROUPS, make_model, prepare_data

OUT = ROOT / "results" / "phase7_domain_shapley"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT / "domain_subset_scores.csv"


def columns_for_mask(mask, names):
    return [column for i, name in enumerate(names) if mask & (1 << i)
            for column in THEORY_GROUPS[name]]


def groups_for_mask(mask, names):
    return [name for i, name in enumerate(names) if mask & (1 << i)]


def evaluate_subset(mask, names, data, outcome, splits):
    columns = columns_for_mask(mask, names)
    groups = groups_for_mask(mask, names)
    rows = []
    for split_id, (train, test) in enumerate(splits, 1):
        if columns:
            model = make_model(columns)
            model.fit(data.iloc[train][columns], outcome.iloc[train])
            prediction = model.predict(data.iloc[test][columns])
        else:
            model = DummyRegressor(strategy="mean")
            model.fit(np.zeros((len(train), 1)), outcome.iloc[train])
            prediction = model.predict(np.zeros((len(test), 1)))
        rows.append({
            "subset_mask": mask,
            "subset_groups": "|".join(groups) if groups else "empty",
            "n_groups": len(groups),
            "split_id": split_id,
            "repeat": (split_id-1)//5+1,
            "fold": (split_id-1)%5+1,
            "r2": r2_score(outcome.iloc[test], prediction),
            "mae": mean_absolute_error(outcome.iloc[test], prediction),
            "rmse": mean_squared_error(outcome.iloc[test], prediction)**.5,
        })
    return pd.DataFrame(rows)


def exact_shapley(scores, names):
    m = len(names)
    score_index = scores.set_index(["subset_mask", "split_id"])["r2"]
    shapley_rows, marginal_rows = [], []
    for split_id in sorted(scores.split_id.unique()):
        for j, domain in enumerate(names):
            contributions = []
            for mask in range(1 << m):
                if mask & (1 << j):
                    continue
                size = int(mask.bit_count())
                weight = factorial(size)*factorial(m-size-1)/factorial(m)
                delta = (score_index.loc[(mask | (1 << j), split_id)]
                         - score_index.loc[(mask, split_id)])
                contributions.append(weight*delta)
                marginal_rows.append({
                    "split_id": split_id,
                    "domain": domain,
                    "context_mask": mask,
                    "context_size": size,
                    "context_groups": "|".join(groups_for_mask(mask, names)) or "empty",
                    "marginal_delta_r2": delta,
                    "shapley_weight": weight,
                })
            shapley_rows.append({"split_id": split_id, "domain": domain,
                                 "shapley_r2": np.sum(contributions)})
    return pd.DataFrame(shapley_rows), pd.DataFrame(marginal_rows)


def main():
    data, _ = prepare_data()
    outcome = data.CPSA.astype(float)
    names = list(THEORY_GROUPS)
    splits = list(RepeatedKFold(n_splits=5, n_repeats=5,
                               random_state=SEED).split(data, outcome))
    if CHECKPOINT.exists():
        scores = pd.read_csv(CHECKPOINT)
    else:
        scores = pd.DataFrame()
    completed = set(scores.groupby("subset_mask").size().loc[lambda s: s.eq(25)].index) if len(scores) else set()

    for mask in range(1 << len(names)):
        if mask in completed:
            continue
        print(f"Evaluating subset {mask+1}/64: {groups_for_mask(mask, names) or ['empty']} ...", flush=True)
        scores = scores[scores.subset_mask.ne(mask)] if len(scores) else scores
        new_scores = evaluate_subset(mask, names, data, outcome, splits)
        scores = pd.concat([scores, new_scores], ignore_index=True)
        scores.sort_values(["subset_mask", "split_id"]).to_csv(
            CHECKPOINT, index=False, encoding="utf-8-sig")

    scores = scores.sort_values(["subset_mask", "split_id"]).reset_index(drop=True)
    assert scores.groupby("subset_mask").size().eq(25).all()
    shapley, marginal = exact_shapley(scores, names)
    shapley["repeat"] = (shapley.split_id-1)//5+1
    shapley["fold"] = (shapley.split_id-1)%5+1
    summary = (shapley.groupby("domain", as_index=False)
               .agg(shapley_r2_mean=("shapley_r2", "mean"),
                    shapley_r2_sd=("shapley_r2", "std"),
                    positive_splits=("shapley_r2", lambda x: int((x > 0).sum())),
                    negative_splits=("shapley_r2", lambda x: int((x < 0).sum()))))
    context = (marginal.groupby("domain", as_index=False)
               .agg(marginal_delta_mean=("marginal_delta_r2", "mean"),
                    marginal_delta_sd=("marginal_delta_r2", "std"),
                    marginal_delta_min=("marginal_delta_r2", "min"),
                    marginal_delta_max=("marginal_delta_r2", "max"),
                    positive_context_fold_pairs=("marginal_delta_r2", lambda x: int((x > 0).sum())),
                    n_context_fold_pairs=("marginal_delta_r2", "size")))
    empty = scores[scores.subset_mask.eq(0)].set_index("split_id").r2
    full_mask = (1 << len(names))-1
    full = scores[scores.subset_mask.eq(full_mask)].set_index("split_id").r2
    total = (full-empty).rename("full_minus_empty_r2")
    allocated = shapley.groupby("split_id").shapley_r2.sum().rename("sum_shapley_r2")
    efficiency = pd.concat([total, allocated], axis=1).reset_index()
    efficiency["allocation_error"] = efficiency.sum_shapley_r2-efficiency.full_minus_empty_r2
    summary = summary.merge(context, on="domain")
    summary["share_of_mean_full_model_gain"] = (
        summary.shapley_r2_mean/efficiency.full_minus_empty_r2.mean())
    summary = summary.sort_values("shapley_r2_mean", ascending=False)

    subset_summary = (scores.groupby(["subset_mask", "subset_groups", "n_groups"], as_index=False)
                      .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                           mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"))
                      .sort_values("r2_mean", ascending=False))
    shapley.to_csv(OUT/"domain_shapley_by_split.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT/"domain_shapley_summary.csv", index=False, encoding="utf-8-sig")
    marginal.to_csv(OUT/"domain_marginal_contributions_all_contexts.csv", index=False, encoding="utf-8-sig")
    subset_summary.to_csv(OUT/"domain_subset_summary.csv", index=False, encoding="utf-8-sig")
    efficiency.to_csv(OUT/"shapley_efficiency_check.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT/"CSE_domain_shapley.xlsx") as writer:
        summary.to_excel(writer, sheet_name="shapley_summary", index=False)
        shapley.to_excel(writer, sheet_name="shapley_by_split", index=False)
        context.to_excel(writer, sheet_name="context_sensitivity", index=False)
        subset_summary.to_excel(writer, sheet_name="subset_summary", index=False)
        efficiency.to_excel(writer, sheet_name="efficiency_check", index=False)
    metadata = {
        "domains": THEORY_GROUPS,
        "number_of_domain_subsets": 64,
        "outer_validation": "5 repeats x 5 folds; identical splits for all subsets",
        "within_fold_model": "ElasticNetCV with 5-fold inner tuning",
        "interpretation": "Average predictive utility across all entry orders; not a causal or unique-variance decomposition.",
        "mean_full_minus_empty_r2": efficiency.full_minus_empty_r2.mean(),
        "maximum_absolute_efficiency_error": efficiency.allocation_error.abs().max(),
    }
    (OUT/"shapley_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSHAPLEY SUMMARY\n", summary.to_string(index=False))
    print("\nTOP SUBSETS\n", subset_summary.head(10).to_string(index=False))
    print("\nMAX EFFICIENCY ERROR", efficiency.allocation_error.abs().max())


if __name__ == "__main__":
    main()
