from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "EPOCH-A 全国数字行为与环境心理横断面样本-清洗后样本.sav"
OUT = ROOT / "results" / "phase1_exploratory"
OUT.mkdir(exist_ok=True)
RNG = 20261009


def load_data():
    df, meta = pyreadstat.read_sav(str(DATA))
    return df, meta


def models(feature_columns):
    nominal = [c for c in ["gender", "academic_year", "enrollment_status", "marital_status",
                            "part_time_job", "only_child", "academic_ranking", "place",
                            "career_plan"] if c in feature_columns]
    numeric = [c for c in feature_columns if c not in nominal]
    linear_preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), numeric),
        ("nominal", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]), nominal),
    ])
    return {
        "ElasticNet": Pipeline([
            ("preprocess", linear_preprocess),
            ("model", TransformedTargetRegressor(
                regressor=ElasticNetCV(l1_ratio=[.05, .2, .5, .8, .95, 1.0], alphas=None,
                                       cv=5, max_iter=20000, random_state=RNG),
                transformer=StandardScaler(),
            )),
        ]),
        "RandomForest": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                             max_features=.7, n_jobs=-1, random_state=RNG)),
        ]),
        "ExtraTrees": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3,
                                           max_features=.8, n_jobs=-1, random_state=RNG)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingRegressor(max_iter=350, learning_rate=.05,
                                                     l2_regularization=1.0,
                                                     random_state=RNG)),
        ]),
    }


def build_groups(columns):
    scale_prefixes = ["PCOS", "PUP", "IU", "WVB", "ISB", "AIP", "DDMB", "MS",
                      "C", "AD", "HD", "SMC"]
    groups = {}
    used = set()
    for prefix in scale_prefixes:
        # Exact total-score column plus numbered items. Avoid prefix collisions (AD vs ADHD).
        members = [c for c in columns if c == prefix or
                   (c.startswith(prefix) and c[len(prefix):].isdigit())]
        if members:
            groups[prefix] = members
            used.update(members)
    # ADHD contains AD and HD and is a higher-order total; group separately as its own column.
    if "ADHD" in columns:
        groups["ADHD_total"] = ["ADHD"]
        used.add("ADHD")
    # Derived environmental indices are kept as distinct factors.
    for c in columns:
        if c not in used:
            groups[c] = [c]
    return groups


def grouped_permutation_cv(estimator, X, y, groups, splits, repeats=5):
    rows = []
    rng = np.random.default_rng(RNG)
    for fold, (tr, te) in enumerate(splits, 1):
        fitted = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        Xte = X.iloc[te].copy()
        yte = y.iloc[te]
        base = r2_score(yte, fitted.predict(Xte))
        for name, cols in groups.items():
            drops = []
            for _ in range(repeats):
                xp = Xte.copy()
                order = rng.permutation(len(xp))
                # Same permutation preserves within-construct relationships while breaking outcome link.
                xp.loc[:, cols] = xp.loc[:, cols].to_numpy()[order]
                drops.append(base - r2_score(yte, fitted.predict(xp)))
            rows.append({"fold": fold, "factor": name, "columns": ", ".join(cols),
                         "importance_r2_drop": float(np.mean(drops)),
                         "repeat_sd": float(np.std(drops, ddof=1))})
    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["factor", "columns"], as_index=False)
               .agg(importance_mean=("importance_r2_drop", "mean"),
                    importance_sd=("importance_r2_drop", "std"),
                    min_fold=("importance_r2_drop", "min"),
                    max_fold=("importance_r2_drop", "max")))
    summary["positive_fold_count"] = raw.groupby("factor")["importance_r2_drop"].apply(lambda s: int((s > 0).sum())).values
    return raw, summary.sort_values("importance_mean", ascending=False)


def main():
    df, meta = load_data()
    target = "CPSA"
    leakage = ["CPSA1", "CPSA2", "CPSA3", target]
    X = df.drop(columns=leakage).copy()
    y = df[target].astype(float)

    # No columns are silently discarded: time and all other predictors remain in the full model.
    constant = X.columns[X.nunique(dropna=False) <= 1].tolist()
    if constant:
        X = X.drop(columns=constant)

    cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
    splits = list(cv.split(X, y))
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    score_rows = []
    fitted_models = models(list(X.columns))
    for name, model in fitted_models.items():
        s = cross_validate(model, X, y, cv=splits, scoring=scoring, n_jobs=1)
        for fold in range(5):
            score_rows.append({"model": name, "fold": fold + 1,
                               "r2": s["test_r2"][fold],
                               "mae": -s["test_mae"][fold],
                               "rmse": -s["test_rmse"][fold]})
    fold_scores = pd.DataFrame(score_rows)
    score_summary = (fold_scores.groupby("model", as_index=False)
                     .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                          mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                          rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"))
                     .sort_values("r2_mean", ascending=False))
    best_name = score_summary.iloc[0]["model"]
    best_model = fitted_models[best_name]

    groups = build_groups(list(X.columns))
    raw_imp, imp = grouped_permutation_cv(best_model, X, y, groups, splits, repeats=5)

    # Direction is descriptive Spearman association, not a causal or model coefficient.
    direction_rows = []
    for factor, cols in groups.items():
        if len(cols) == 1:
            proxy = X[cols[0]]
        else:
            total = factor if factor in cols else None
            proxy = X[total] if total else X[cols].mean(axis=1)
        rho = pd.Series(proxy).corr(y, method="spearman")
        direction_rows.append({"factor": factor, "spearman_rho": rho,
                               "direction": "正向" if rho > 0 else ("负向" if rho < 0 else "无")})
    directions = pd.DataFrame(direction_rows)
    imp = imp.merge(directions, on="factor", how="left")
    imp["rank"] = np.arange(1, len(imp) + 1)

    descriptive = pd.DataFrame({
        "n": [len(df)], "predictor_columns": [X.shape[1]], "target_mean": [y.mean()],
        "target_sd": [y.std(ddof=1)], "target_min": [y.min()], "target_max": [y.max()],
        "target_missing": [y.isna().sum()], "constant_removed": [", ".join(constant)],
    })
    variable_dict = pd.DataFrame({
        "variable": df.columns,
        "label": [meta.column_names_to_labels.get(c) for c in df.columns],
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
        "missing_n": [df[c].isna().sum() for c in df.columns],
        "role": ["结果变量" if c == target else ("结果组成题项（防泄漏排除）" if c in leakage else "自变量") for c in df.columns],
    })

    fold_scores.to_csv(OUT / "model_cv_fold_scores.csv", index=False, encoding="utf-8-sig")
    score_summary.to_csv(OUT / "model_cv_summary.csv", index=False, encoding="utf-8-sig")
    raw_imp.to_csv(OUT / "grouped_permutation_importance_folds.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(OUT / "factor_importance_summary.csv", index=False, encoding="utf-8-sig")
    variable_dict.to_csv(OUT / "variable_dictionary.csv", index=False, encoding="utf-8-sig")
    descriptive.to_csv(OUT / "sample_target_summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "creative_self_efficacy_ml_results.xlsx") as writer:
        descriptive.to_excel(writer, "sample", index=False)
        score_summary.to_excel(writer, "model_comparison", index=False)
        fold_scores.to_excel(writer, "cv_folds", index=False)
        imp.to_excel(writer, "factor_importance", index=False)
        raw_imp.to_excel(writer, "importance_folds", index=False)
        variable_dict.to_excel(writer, "variable_dictionary", index=False)

    sns.set_theme(style="whitegrid")
    top = imp.head(20).sort_values("importance_mean")
    plt.figure(figsize=(10, 8))
    colors = ["#2c7fb8" if x >= 0 else "#d95f0e" for x in top["spearman_rho"]]
    plt.barh(top["factor"], top["importance_mean"], xerr=top["importance_sd"], color=colors, alpha=.85)
    plt.xlabel("Grouped permutation importance (decrease in test R²)")
    plt.ylabel("Factor")
    plt.title("Top predictors of creative self-efficacy\nBlue: positive Spearman association; orange: negative")
    plt.tight_layout()
    plt.savefig(OUT / "top_factor_importance.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    order = score_summary["model"].tolist()
    sns.boxplot(data=fold_scores, x="model", y="r2", order=order, color="#80b1d3")
    sns.stripplot(data=fold_scores, x="model", y="r2", order=order, color="black", size=4)
    plt.ylabel("Held-out R²")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.title("Five-fold cross-validated model performance")
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=220)
    plt.close()

    metadata = {"data_file": DATA.name, "n": len(df), "target": target,
                "excluded_for_leakage": leakage[:-1], "predictors_used": list(X.columns),
                "best_model": best_name, "random_seed": RNG, "cv": "5-fold shuffled KFold"}
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(score_summary.to_string(index=False))
    print("\nBEST", best_name)
    print("\nTOP FACTORS")
    print(imp.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
