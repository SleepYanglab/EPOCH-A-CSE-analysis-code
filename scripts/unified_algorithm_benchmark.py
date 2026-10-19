"""Unified construct-level benchmark across linear and nonlinear regressors.

All algorithms use the same 5x5 repeated outer folds and construct-level
features. Ridge and Elastic Net tune regularization within each outer training
fold; nonlinear challengers use prespecified conservative settings.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVR

from construct_level_cv import ROOT, NOMINAL, SEED, THEORY_GROUPS, prepare_data

warnings.filterwarnings("ignore", message="Found unknown categories")
OUT = ROOT / "results" / "phase5_algorithm_benchmark"
OUT.mkdir(parents=True, exist_ok=True)


def preprocessors(columns):
    categorical = [c for c in columns if c in NOMINAL]
    numeric = [c for c in columns if c not in NOMINAL]
    linear = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), categorical),
    ])
    dense = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    # Additive nonlinear model: a separate spline basis for every numeric
    # construct, without interaction terms.
    gam = ColumnTransformer([
        ("numeric_splines", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return linear, dense, gam


def scaled_target(regressor):
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def build_models(columns):
    linear, dense, gam = preprocessors(columns)
    return {
        "MeanBaseline": Pipeline([
            ("preprocess", dense), ("model", DummyRegressor(strategy="mean")),
        ]),
        "Ridge": Pipeline([
            ("preprocess", linear),
            ("model", scaled_target(RidgeCV(alphas=np.logspace(-3, 3, 49)))),
        ]),
        "ElasticNet": Pipeline([
            ("preprocess", linear),
            ("model", scaled_target(ElasticNetCV(
                l1_ratio=[.05, .2, .5, .8, .95, 1.0],
                cv=5, max_iter=30000, random_state=SEED,
            ))),
        ]),
        "SplineGAM_Ridge": Pipeline([
            ("preprocess", gam),
            ("model", scaled_target(RidgeCV(alphas=np.logspace(-3, 4, 57)))),
        ]),
        "RBF_SVR": Pipeline([
            ("preprocess", linear),
            ("model", scaled_target(SVR(kernel="rbf", C=3.0, gamma="scale", epsilon=.10))),
        ]),
        "RandomForest": Pipeline([
            ("preprocess", dense),
            ("model", RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5, max_features=.7,
                n_jobs=-1, random_state=SEED,
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("preprocess", dense),
            ("model", ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=3, max_features=.8,
                n_jobs=-1, random_state=SEED,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocess", dense),
            ("model", HistGradientBoostingRegressor(
                max_iter=350, learning_rate=.05, max_leaf_nodes=31,
                min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=True, random_state=SEED,
            )),
        ]),
    }


def confidence_interval(values):
    values = np.asarray(values)
    half = t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean()-half, values.mean()+half


def main():
    data, _ = prepare_data()
    y = data.CPSA.astype(float)
    columns = sum(THEORY_GROUPS.values(), [])
    X = data[columns]
    splits = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y))
    models = build_models(columns)

    score_rows, prediction_rows = [], []
    for model_name, model in models.items():
        print(f"Running {model_name} ...", flush=True)
        for split_id, (train, test) in enumerate(splits, 1):
            model.fit(X.iloc[train], y.iloc[train])
            pred = model.predict(X.iloc[test])
            score_rows.append({
                "model": model_name, "split_id": split_id,
                "repeat": (split_id-1)//5+1, "fold": (split_id-1)%5+1,
                "r2": r2_score(y.iloc[test], pred),
                "mae": mean_absolute_error(y.iloc[test], pred),
                "rmse": mean_squared_error(y.iloc[test], pred)**.5,
            })
            prediction_rows.extend({
                "model": model_name, "participant_id": int(i),
                "split_id": split_id, "repeat": (split_id-1)//5+1,
                "observed": float(y.iloc[i]), "predicted": float(p),
            } for i, p in zip(test, pred))

    scores = pd.DataFrame(score_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = (scores.groupby("model", as_index=False)
               .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                    mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                    rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"))
               .sort_values("r2_mean", ascending=False))

    reference = scores[scores.model.eq("ElasticNet")].sort_values("split_id")
    paired = []
    for name in summary.model:
        candidate = scores[scores.model.eq(name)].sort_values("split_id")
        delta_r2 = candidate.r2.to_numpy()-reference.r2.to_numpy()
        # Positive MAE/RMSE improvements mean lower error than Elastic Net.
        improve_mae = reference.mae.to_numpy()-candidate.mae.to_numpy()
        improve_rmse = reference.rmse.to_numpy()-candidate.rmse.to_numpy()
        r2_low, r2_high = confidence_interval(delta_r2)
        mae_low, mae_high = confidence_interval(improve_mae)
        rmse_low, rmse_high = confidence_interval(improve_rmse)
        paired.append({
            "model": name,
            "delta_r2_vs_elastic_net": delta_r2.mean(),
            "delta_r2_sd": delta_r2.std(ddof=1),
            "delta_r2_descriptive_ci95_low": r2_low,
            "delta_r2_descriptive_ci95_high": r2_high,
            "r2_better_splits": int((delta_r2 > 0).sum()),
            "mae_improvement_vs_elastic_net": improve_mae.mean(),
            "mae_descriptive_ci95_low": mae_low,
            "mae_descriptive_ci95_high": mae_high,
            "mae_better_splits": int((improve_mae > 0).sum()),
            "rmse_improvement_vs_elastic_net": improve_rmse.mean(),
            "rmse_descriptive_ci95_low": rmse_low,
            "rmse_descriptive_ci95_high": rmse_high,
            "rmse_better_splits": int((improve_rmse > 0).sum()),
            "n_splits": len(delta_r2),
        })
    paired = pd.DataFrame(paired).sort_values("delta_r2_vs_elastic_net", ascending=False)

    participant = (predictions.groupby(["model", "participant_id"], as_index=False)
                   .agg(observed=("observed", "first"), predicted=("predicted", "mean")))
    calibration_rows = []
    for name, group in participant.groupby("model"):
        if name == "MeanBaseline":
            # Fold-specific training means vary slightly, but a calibration
            # slope for this no-information baseline is not meaningful.
            slope, intercept = np.nan, float(group.observed.mean())
        else:
            slope, intercept = np.polyfit(group.predicted, group.observed, 1)
        calibration_rows.append({
            "model": name, "calibration_intercept": intercept,
            "calibration_slope": slope,
            "participant_average_r2": r2_score(group.observed, group.predicted),
            "participant_average_mae": mean_absolute_error(group.observed, group.predicted),
            "participant_average_rmse": mean_squared_error(group.observed, group.predicted)**.5,
        })
    calibration = pd.DataFrame(calibration_rows).sort_values("participant_average_r2", ascending=False)

    scores.to_csv(OUT/"algorithm_scores_25_splits.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT/"algorithm_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT/"paired_performance_vs_elastic_net.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT/"algorithm_oof_predictions_long.csv", index=False, encoding="utf-8-sig")
    participant.to_csv(OUT/"algorithm_participant_predictions.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(OUT/"algorithm_calibration.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT/"unified_algorithm_benchmark.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        paired.to_excel(writer, sheet_name="paired_vs_elasticnet", index=False)
        calibration.to_excel(writer, sheet_name="calibration", index=False)
        scores.to_excel(writer, sheet_name="split_scores", index=False)
    metadata = {
        "outcome": "CPSA", "features": columns,
        "outer_validation": "5 repeats x 5 folds; identical folds for all algorithms",
        "linear_tuning": "RidgeCV/ElasticNetCV within each outer training fold",
        "nonlinear_parameters": "prespecified; no test-fold tuning",
        "random_seed": SEED,
    }
    (OUT/"benchmark_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSUMMARY\n", summary.to_string(index=False))
    print("\nPAIRED VS ELASTIC NET\n", paired.to_string(index=False))
    print("\nCALIBRATION\n", calibration.to_string(index=False))


if __name__ == "__main__":
    main()
