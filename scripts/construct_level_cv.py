"""Construct-level, repeated-CV analysis of creative self-efficacy.

The primary aim is to compare the out-of-sample predictive information carried
by psychological, behavioral, background, and environmental domains.  Ordered
entry models are retained as one view of overlap, not as the sole research aim.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import t
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message="Found unknown categories")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "EPOCH-A 全国数字行为与环境心理横断面样本-清洗后样本.sav"
OUT = ROOT / "results" / "phase2_construct_cv"
OUT.mkdir(exist_ok=True)
SEED = 20261009

BACKGROUND = [
    "age", "gender", "academic_year", "enrollment_status", "sss", "marital_status",
    "part_time_job", "only_child", "siblings_clean", "birth_order_clean",
    "academic_ranking", "place", "E", "health", "diet", "career_plan",
]
DIGITAL = ["IU", "WVB", "ISB", "AIP", "DDMB"]
ATTENTION = ["MS", "C", "AD", "HD"]  # ADHD excluded: exact AD + HD duplicate
FLOW = ["SMC"]
SELF_EVALUATION = [
    "high_self_esteem1", "high_self_esteem2", "high_self_esteem3", "high_self_esteem4",
    "low_self_esteem1", "low_self_esteem2", "low_self_esteem3", "low_self_esteem4",
]
FAMILY_ENVIRONMENT = [
    "PCOS", "PUP", "PG", "Rnearest_greenery", "comfort", "temperature", "humidity",
    "lighting", "ANWP", "NC",
]
THEORY_GROUPS = {
    "background": BACKGROUND,
    "digital_behavior": DIGITAL,
    "attention_cognitive": ATTENTION,
    "trait_flow": FLOW,
    "self_evaluation": SELF_EVALUATION,
    "family_environment": FAMILY_ENVIRONMENT,
}
NOMINAL = {
    "gender", "academic_year", "enrollment_status", "marital_status", "part_time_job",
    "only_child", "academic_ranking", "place", "career_plan",
}


def prepare_data():
    d, meta = pyreadstat.read_sav(str(DATA))
    # Structural skip: only children logically have one child in family and birth order one.
    d["siblings_clean"] = d["siblings"]
    d["birth_order_clean"] = d["birth_order"]
    only = d["only_child"].eq(1)
    d.loc[only, "siblings_clean"] = 1
    d.loc[only, "birth_order_clean"] = 1
    # Three impossible family-size codes are treated as data-entry errors, not genuine values.
    d.loc[d["siblings_clean"].gt(20), "siblings_clean"] = np.nan
    return d, meta


def make_model(columns):
    cat = [c for c in columns if c in NOMINAL]
    num = [c for c in columns if c not in NOMINAL]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), cat),
    ])
    reg = TransformedTargetRegressor(
        regressor=ElasticNetCV(
            l1_ratio=[.05, .2, .5, .8, .95, 1.0],
            cv=5, max_iter=30000, n_jobs=-1, random_state=SEED,
        ),
        transformer=StandardScaler(),
    )
    return Pipeline([("pre", pre), ("reg", reg)])


def evaluate(columns, d, y, splits, label):
    rows = []
    for split_id, (tr, te) in enumerate(splits, 1):
        model = make_model(columns)
        model.fit(d.iloc[tr][columns], y.iloc[tr])
        pred = model.predict(d.iloc[te][columns])
        rows.append({
            "model": label, "split_id": split_id,
            "repeat": (split_id - 1) // 5 + 1, "fold": (split_id - 1) % 5 + 1,
            "r2": r2_score(y.iloc[te], pred),
            "mae": mean_absolute_error(y.iloc[te], pred),
            "rmse": mean_squared_error(y.iloc[te], pred) ** .5,
        })
    return pd.DataFrame(rows)


def summarize(scores):
    return (scores.groupby("model", as_index=False)
            .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                 mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                 rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std")))


def paired_delta(left, right, label):
    a = left.sort_values("split_id")
    b = right.sort_values("split_id")
    delta = b["r2"].to_numpy() - a["r2"].to_numpy()
    n = len(delta)
    half = t.ppf(.975, n - 1) * delta.std(ddof=1) / np.sqrt(n)
    return {
        "comparison": label, "delta_r2_mean": delta.mean(), "delta_r2_sd": delta.std(ddof=1),
        "descriptive_ci95_low": delta.mean() - half,
        "descriptive_ci95_high": delta.mean() + half,
        "positive_splits": int((delta > 0).sum()), "n_splits": n,
    }


def coefficient_stability(columns, d, y, splits):
    rows = []
    for split_id, (tr, _) in enumerate(splits, 1):
        model = make_model(columns).fit(d.iloc[tr][columns], y.iloc[tr])
        names = model.named_steps["pre"].get_feature_names_out()
        coef = model.named_steps["reg"].regressor_.coef_
        rows.extend({"split_id": split_id, "feature": n, "coefficient": c}
                    for n, c in zip(names, coef))
    raw = pd.DataFrame(rows)
    summary = (raw.groupby("feature", as_index=False)
               .agg(coef_mean=("coefficient", "mean"), coef_sd=("coefficient", "std"),
                    nonzero_rate=("coefficient", lambda s: np.mean(np.abs(s) > 1e-10)),
                    positive_rate=("coefficient", lambda s: np.mean(s > 1e-10)),
                    negative_rate=("coefficient", lambda s: np.mean(s < -1e-10))))
    summary["abs_coef_mean"] = summary["coef_mean"].abs()
    return raw, summary.sort_values("abs_coef_mean", ascending=False)


def main():
    d, _ = prepare_data()
    y = d["CPSA"].astype(float)
    all_columns = sum(THEORY_GROUPS.values(), [])
    models = {
        "A_background": BACKGROUND,
        "B_background_psychology": BACKGROUND + FLOW + ATTENTION + SELF_EVALUATION,
        "C_plus_digital": BACKGROUND + FLOW + ATTENTION + SELF_EVALUATION + DIGITAL,
        "D_full_plus_family_environment": all_columns,
    }

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    splits = list(rkf.split(d, y))
    score_parts = [evaluate(cols, d, y, splits, name) for name, cols in models.items()]
    scores = pd.concat(score_parts, ignore_index=True)
    summary = summarize(scores)
    by_name = {name: scores[scores.model.eq(name)] for name in models}
    deltas = pd.DataFrame([
        paired_delta(by_name["A_background"], by_name["B_background_psychology"], "B - A: psychology increment"),
        paired_delta(by_name["B_background_psychology"], by_name["C_plus_digital"], "C - B: digital increment"),
        paired_delta(by_name["C_plus_digital"], by_name["D_full_plus_family_environment"], "D - C: family/environment increment"),
    ])

    # Leave-one-theoretical-group-out retraining, always on identical outer splits.
    full = by_name["D_full_plus_family_environment"]
    logo_rows = []
    logo_scores = []
    for group, removed in THEORY_GROUPS.items():
        kept = [c for c in all_columns if c not in removed]
        sc = evaluate(kept, d, y, splits, f"without_{group}")
        logo_scores.append(sc)
        # Positive value means performance lost when the group was removed.
        logo_rows.append(paired_delta(sc, full, f"full - without {group}"))
    logo = pd.DataFrame(logo_rows).sort_values("delta_r2_mean", ascending=False)
    logo_fold_scores = pd.concat(logo_scores, ignore_index=True)

    # Conditional directions and selection stability in the full construct-level model.
    coef_raw, coef_summary = coefficient_stability(all_columns, d, y, splits)

    # Adult-only sensitivity, using identical model specification and a fresh repeated CV.
    adults = d[d.age >= 18].reset_index(drop=True)
    y_adult = adults.CPSA.astype(float)
    adult_splits = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(adults, y_adult))
    adult_scores = evaluate(all_columns, adults, y_adult, adult_splits, "D_full_adults_18plus")

    data_checks = pd.DataFrame({
        "check": ["n", "under_18_n", "CPSA_alpha", "SMC_CPSA_Pearson_r", "ISB_CPSA_Pearson_r",
                  "siblings_impossible_n", "SMC_exact_item_sum", "ADHD_exact_AD_plus_HD",
                  "Rnearest_exact_reverse"],
        "value": [len(d), int((d.age < 18).sum()),
                  3 / 2 * (1 - d[["CPSA1", "CPSA2", "CPSA3"]].var(ddof=1).sum() /
                           d[["CPSA1", "CPSA2", "CPSA3"]].sum(axis=1).var(ddof=1)),
                  d.SMC.corr(y), d.ISB.corr(y), int(d.siblings.gt(20).sum()),
                  bool((d.SMC == d[[f"SMC{i}" for i in range(1, 10)]].sum(axis=1)).all()),
                  bool((d.ADHD == d.AD + d.HD).all()),
                  bool((d.Rnearest_greenery == 6 - d.nearest_greenery).all())],
    })

    scores.to_csv(OUT / "hierarchical_repeated_cv_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "hierarchical_model_summary.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(OUT / "hierarchical_paired_delta_r2.csv", index=False, encoding="utf-8-sig")
    logo.to_csv(OUT / "leave_one_group_out_delta_r2.csv", index=False, encoding="utf-8-sig")
    logo_fold_scores.to_csv(OUT / "leave_one_group_out_scores.csv", index=False, encoding="utf-8-sig")
    coef_raw.to_csv(OUT / "conditional_coefficients_splits.csv", index=False, encoding="utf-8-sig")
    coef_summary.to_csv(OUT / "conditional_coefficient_stability.csv", index=False, encoding="utf-8-sig")
    adult_scores.to_csv(OUT / "adult_sensitivity_scores.csv", index=False, encoding="utf-8-sig")
    data_checks.to_csv(OUT / "data_checks.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "CSE_construct_level_repeated_CV.xlsx") as w:
        data_checks.to_excel(w, "data_checks", index=False)
        summary.to_excel(w, "model_summary", index=False)
        deltas.to_excel(w, "paired_increments", index=False)
        scores.to_excel(w, "model_split_scores", index=False)
        logo.to_excel(w, "LOGO_summary", index=False)
        logo_fold_scores.to_excel(w, "LOGO_split_scores", index=False)
        coef_summary.to_excel(w, "coefficient_stability", index=False)
        adult_scores.to_excel(w, "adult_sensitivity", index=False)

    metadata = {"seed": SEED, "outer_cv": "5x5 repeated KFold", "inner_cv": "5-fold ElasticNetCV",
                "outcome": "CPSA", "excluded": ["time", "CPSA1-3", "all scale items",
                "ADHD exact duplicate", "nearest_greenery exact reverse duplicate",
                "raw components where E/PG/ANWP composite used"], "groups": THEORY_GROUPS}
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("MODEL SUMMARY\n", summary.to_string(index=False))
    print("\nPAIRED INCREMENTS\n", deltas.to_string(index=False))
    print("\nLEAVE-ONE-GROUP-OUT\n", logo.to_string(index=False))
    print("\nADULT FULL MODEL\n", summarize(adult_scores).to_string(index=False))
    print("\nTOP CONDITIONAL COEFFICIENTS\n", coef_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
