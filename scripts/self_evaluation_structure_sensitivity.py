"""Sensitivity analysis for the eight human-trait self-evaluation items.

The questionnaire contains four positively and four negatively worded items
adapted from Yang et al. (2015).  The source framework distinguishes human
nature (HN) from uniquely human (HU) traits rather than treating valence as the
construct boundary.  This script compares those two groupings without changing
the preregistered/main domain definition: all eight items remain in the
self-evaluation domain in the original full model.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from semopy import Model, calc_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from construct_level_cv import (
    FLOW,
    ROOT,
    SEED,
    SELF_EVALUATION,
    THEORY_GROUPS,
    make_model,
    prepare_data,
)
from ordinal_measurement import (
    _patch_semopy_for_current_numpy_scipy,
    ordinal_alpha,
    srmr_from_model,
)
from unified_algorithm_benchmark import build_models

warnings.filterwarnings("ignore", message="Found unknown categories")

OUT = ROOT / "results" / "phase9_self_evaluation_structure_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
N_PERMUTATIONS = 20

POSITIVE = [f"high_self_esteem{i}" for i in range(1, 5)]
NEGATIVE = [f"low_self_esteem{i}" for i in range(1, 5)]

# Item order in the administered questionnaire:
# HN: open-minded, emotional/responsive, superficial (reversed), mechanical/cold (reversed)
# HU: refined/cultured, rational/logical, lacks restraint (reversed), immature (reversed)
HN_ORIGINAL = ["high_self_esteem1", "high_self_esteem2",
               "low_self_esteem3", "low_self_esteem4"]
HU_ORIGINAL = ["high_self_esteem3", "high_self_esteem4",
               "low_self_esteem1", "low_self_esteem2"]
REVERSED_NAMES = {
    "low_self_esteem1": "rev_low_self_esteem1",
    "low_self_esteem2": "rev_low_self_esteem2",
    "low_self_esteem3": "rev_low_self_esteem3",
    "low_self_esteem4": "rev_low_self_esteem4",
}
HN_SCORED = ["high_self_esteem1", "high_self_esteem2",
             "rev_low_self_esteem3", "rev_low_self_esteem4"]
HU_SCORED = ["high_self_esteem3", "high_self_esteem4",
             "rev_low_self_esteem1", "rev_low_self_esteem2"]
POS_SCORED = POSITIVE
NEG_SCORED = list(REVERSED_NAMES.values())


def cronbach_alpha(frame: pd.DataFrame) -> float:
    values = frame.to_numpy(dtype=float)
    k = values.shape[1]
    return float(k / (k - 1) * (1 - values.var(axis=0, ddof=1).sum()
                                / values.sum(axis=1).var(ddof=1)))


def add_scored_variables(data: pd.DataFrame) -> pd.DataFrame:
    scored = data.copy()
    for original, reversed_name in REVERSED_NAMES.items():
        scored[reversed_name] = 6 - scored[original]
    scored["HN_humanness"] = scored[HN_SCORED].sum(axis=1, min_count=4)
    scored["HU_humanness"] = scored[HU_SCORED].sum(axis=1, min_count=4)
    return scored


def cfa_description(kind: str) -> str:
    all_items = POS_SCORED + NEG_SCORED
    ordinal = "DEFINE(ordinal) " + " ".join(all_items)
    if kind == "one_factor":
        return "HUMANNESS =~ " + " + ".join(all_items) + "\n" + ordinal
    if kind == "valence":
        return ("POSITIVE =~ " + " + ".join(POS_SCORED) + "\n"
                + "NEGATIVE_REVERSED =~ " + " + ".join(NEG_SCORED) + "\n"
                + ordinal)
    return ("HN =~ " + " + ".join(HN_SCORED) + "\n"
            + "HU =~ " + " + ".join(HU_SCORED) + "\n" + ordinal)


def fit_measurement_models(scored: pd.DataFrame):
    _patch_semopy_for_current_numpy_scipy()
    items = POS_SCORED + NEG_SCORED
    item_data = scored[items].dropna().astype(int)
    fit_rows, loading_rows, correlations = [], [], {}
    specs = [
        ("one_factor_humanness", "one_factor"),
        ("two_factor_valence", "valence"),
        ("two_factor_HN_HU", "theory"),
    ]
    for name, kind in specs:
        print(f"Fitting measurement model: {name}", flush=True)
        model = Model(cfa_description(kind))
        solution = model.fit(item_data, obj="DWLS")
        stats = calc_stats(model).iloc[0]
        estimates = model.inspect(std_est=True)
        latent = estimates[(estimates.op.eq("~~")) &
                           (estimates.lval != estimates.rval) &
                           estimates.lval.isin(["POSITIVE", "NEGATIVE_REVERSED", "HN", "HU"]) &
                           estimates.rval.isin(["POSITIVE", "NEGATIVE_REVERSED", "HN", "HU"])]
        latent_r = float(latent["Est. Std"].iloc[0]) if len(latent) else np.nan
        fit_rows.append({
            "model": name,
            "estimator": "polychoric_DWLS_not_WLSMV",
            "converged": bool(solution.success),
            "n": len(item_data),
            "chi_square": float(stats["chi2"]),
            "df": int(stats["DoF"]),
            "CFI": float(stats["CFI"]),
            "TLI": float(stats["TLI"]),
            "RMSEA": float(stats["RMSEA"]),
            "SRMR": srmr_from_model(model),
            "latent_correlation": latent_r,
        })
        loadings = estimates[estimates.op.eq("~")].copy()
        for _, row in loadings.iterrows():
            loading_rows.append({
                "model": name,
                "item": row.lval,
                "factor": row.rval,
                "loading_unstandardized": row.Estimate,
                "loading_standardized": row["Est. Std"],
            })
        order = list(model.vars["observed"])
        correlations[name] = pd.DataFrame(model.mx_cov, index=order, columns=order)

    reliability_rows = []
    for grouping, groups in {
        "valence": {"positive": POS_SCORED, "negative_reversed": NEG_SCORED},
        "theory": {"human_nature": HN_SCORED, "human_uniqueness": HU_SCORED},
    }.items():
        correlation = correlations["two_factor_valence" if grouping == "valence"
                                   else "two_factor_HN_HU"]
        for construct, members in groups.items():
            reliability_rows.append({
                "grouping": grouping,
                "construct": construct,
                "items": "|".join(members),
                "raw_alpha": cronbach_alpha(item_data[members]),
                "ordinal_alpha": ordinal_alpha(correlation.loc[members, members].to_numpy()),
            })
    return (pd.DataFrame(fit_rows), pd.DataFrame(loading_rows),
            pd.DataFrame(reliability_rows), correlations)


def descriptive_associations(scored: pd.DataFrame) -> pd.DataFrame:
    constructs = {
        "positive_items_mean": scored[POS_SCORED].mean(axis=1),
        "negative_items_reversed_mean": scored[NEG_SCORED].mean(axis=1),
        "human_nature_HN": scored[HN_SCORED].mean(axis=1),
        "human_uniqueness_HU": scored[HU_SCORED].mean(axis=1),
    }
    return pd.DataFrame([
        {
            "construct": name,
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "spearman_with_CSE": values.corr(scored.CPSA, method="spearman"),
            "pearson_with_CSE": values.corr(scored.CPSA, method="pearson"),
        }
        for name, values in constructs.items()
    ])


def representation_sensitivity(scored: pd.DataFrame, splits) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_columns = sum(THEORY_GROUPS.values(), [])
    composite_columns = [c for c in original_columns if c not in SELF_EVALUATION]
    composite_columns += ["HN_humanness", "HU_humanness"]
    specifications = {
        "full_eight_items": original_columns,
        "full_HN_HU_composites": composite_columns,
        "self_evaluation_eight_items_only": SELF_EVALUATION,
        "self_evaluation_HN_HU_only": ["HN_humanness", "HU_humanness"],
        "flow_plus_eight_items": FLOW + SELF_EVALUATION,
        "flow_plus_HN_HU": FLOW + ["HN_humanness", "HU_humanness"],
    }
    rows = []
    for name, columns in specifications.items():
        print(f"Running representation model: {name}", flush=True)
        for split_id, (train, test) in enumerate(splits, 1):
            model = make_model(columns)
            model.fit(scored.iloc[train][columns], scored.CPSA.iloc[train])
            prediction = model.predict(scored.iloc[test][columns])
            observed = scored.CPSA.iloc[test]
            rows.append({
                "model": name,
                "split_id": split_id,
                "r2": r2_score(observed, prediction),
                "mae": mean_absolute_error(observed, prediction),
                "rmse": mean_squared_error(observed, prediction) ** .5,
            })
    raw = pd.DataFrame(rows)
    summary = (raw.groupby("model", as_index=False)
               .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                    mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean")))
    return raw, summary


def grouped_importance(scored: pd.DataFrame, splits) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sum(THEORY_GROUPS.values(), [])
    groupings = {
        "valence_positive": POSITIVE,
        "valence_negative": NEGATIVE,
        "human_nature_HN": HN_ORIGINAL,
        "human_uniqueness_HU": HU_ORIGINAL,
    }
    models = build_models(columns)
    models = {name: models[name] for name in ["ElasticNet", "RandomForest"]}
    rows = []
    for model_name, model in models.items():
        print(f"Running grouped importance: {model_name}", flush=True)
        for split_id, (train, test) in enumerate(splits, 1):
            model.fit(scored.iloc[train][columns], scored.CPSA.iloc[train])
            test_x = scored.iloc[test][columns].copy()
            test_y = scored.CPSA.iloc[test]
            baseline = r2_score(test_y, model.predict(test_x))
            for group_name, members in groupings.items():
                drops = []
                for permutation_id in range(N_PERMUTATIONS):
                    rng = np.random.default_rng(SEED + split_id * 1000 + permutation_id)
                    order = rng.permutation(len(test_x))
                    permuted = test_x.copy()
                    permuted.loc[:, members] = test_x[members].to_numpy()[order]
                    drops.append(baseline - r2_score(test_y, model.predict(permuted)))
                rows.append({
                    "model": model_name,
                    "split_id": split_id,
                    "grouping": "valence" if group_name.startswith("valence") else "theory",
                    "construct": group_name,
                    "columns": "|".join(members),
                    "baseline_r2": baseline,
                    "importance_r2_drop": np.mean(drops),
                    "permutation_sd": np.std(drops, ddof=1),
                })
    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["model", "grouping", "construct", "columns"], as_index=False)
               .agg(importance_mean=("importance_r2_drop", "mean"),
                    importance_sd=("importance_r2_drop", "std"),
                    importance_min=("importance_r2_drop", "min"),
                    importance_max=("importance_r2_drop", "max"),
                    positive_splits=("importance_r2_drop", lambda x: int((x > 0).sum()))))
    return raw, summary


def main() -> None:
    data, _ = prepare_data()
    scored = add_scored_variables(data)
    splits = list(RepeatedKFold(n_splits=5, n_repeats=5,
                               random_state=SEED).split(scored, scored.CPSA))

    fit, loadings, reliability, correlations = fit_measurement_models(scored)
    associations = descriptive_associations(scored)
    representation_raw, representation_summary = representation_sensitivity(scored, splits)
    importance_raw, importance_summary = grouped_importance(scored, splits)

    fit.to_csv(OUT / "self_evaluation_DWLS_fit.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(OUT / "self_evaluation_DWLS_loadings.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(OUT / "self_evaluation_reliability.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(OUT / "self_evaluation_associations.csv", index=False, encoding="utf-8-sig")
    representation_raw.to_csv(OUT / "representation_scores_25_splits.csv", index=False,
                              encoding="utf-8-sig")
    representation_summary.to_csv(OUT / "representation_summary.csv", index=False,
                                  encoding="utf-8-sig")
    importance_raw.to_csv(OUT / "alternative_group_importance_by_split.csv", index=False,
                          encoding="utf-8-sig")
    importance_summary.to_csv(OUT / "alternative_group_importance_summary.csv", index=False,
                              encoding="utf-8-sig")
    correlations["two_factor_HN_HU"].to_csv(
        OUT / "self_evaluation_polychoric_correlations.csv", encoding="utf-8-sig")

    with pd.ExcelWriter(OUT / "self_evaluation_structure_sensitivity.xlsx") as writer:
        fit.to_excel(writer, sheet_name="measurement_fit", index=False)
        loadings.to_excel(writer, sheet_name="loadings", index=False)
        reliability.to_excel(writer, sheet_name="reliability", index=False)
        associations.to_excel(writer, sheet_name="associations", index=False)
        representation_summary.to_excel(writer, sheet_name="prediction_summary", index=False)
        representation_raw.to_excel(writer, sheet_name="prediction_splits", index=False)
        importance_summary.to_excel(writer, sheet_name="importance_summary", index=False)
        importance_raw.to_excel(writer, sheet_name="importance_splits", index=False)

    metadata = {
        "purpose": "Compare valence grouping with Yang et al. (2015) HN/HU grouping.",
        "HN_items": HN_ORIGINAL,
        "HU_items": HU_ORIGINAL,
        "reverse_scoring": "Negative items were transformed as 6 - response for composites and CFA.",
        "outer_validation": "5 repeats x 5 folds; identical seed and splits as main analyses",
        "permutations_per_group_per_split": N_PERMUTATIONS,
        "interpretation": "Post hoc measurement and prediction sensitivity analysis.",
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nMEASUREMENT FIT\n", fit.to_string(index=False))
    print("\nRELIABILITY\n", reliability.to_string(index=False))
    print("\nASSOCIATIONS\n", associations.to_string(index=False))
    print("\nPREDICTION\n", representation_summary.to_string(index=False))
    print("\nIMPORTANCE\n", importance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
