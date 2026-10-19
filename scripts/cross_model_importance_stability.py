"""Cross-model stability of construct-level grouped permutation importance."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

from construct_level_cv import ROOT, SEED, THEORY_GROUPS, prepare_data
from construct_predictor_profile import CONSTRUCTS
from unified_algorithm_benchmark import build_models

OUT = ROOT / "results" / "phase8_cross_model_stability"
OUT.mkdir(parents=True, exist_ok=True)
N_PERMUTATIONS = 20


def main():
    data, _ = prepare_data()
    outcome = data.CPSA.astype(float)
    columns = sum(THEORY_GROUPS.values(), [])
    assert set(sum(CONSTRUCTS.values(), [])) == set(columns)
    splits = list(RepeatedKFold(n_splits=5, n_repeats=5,
                               random_state=SEED).split(data, outcome))
    all_models = build_models(columns)
    models = {name: all_models[name] for name in ["ElasticNet", "RandomForest"]}
    rows, baseline_rows = [], []

    for model_name, model in models.items():
        print(f"Running {model_name} ...", flush=True)
        for split_id, (train, test) in enumerate(splits, 1):
            model.fit(data.iloc[train][columns], outcome.iloc[train])
            test_x = data.iloc[test][columns].copy()
            test_y = outcome.iloc[test]
            preprocessor = model.named_steps["preprocess"]
            estimator = model.named_steps["model"]
            processed = preprocessor.transform(test_x)
            if hasattr(processed, "toarray"):
                processed = processed.toarray()
            processed = np.asarray(processed)
            feature_names = np.asarray(preprocessor.get_feature_names_out())
            baseline = r2_score(test_y, estimator.predict(processed))
            baseline_rows.append({"model": model_name, "split_id": split_id,
                                  "repeat": (split_id-1)//5+1,
                                  "fold": (split_id-1)%5+1,
                                  "baseline_r2": baseline})
            for construct_index, (construct, members) in enumerate(CONSTRUCTS.items()):
                permuted_batches = []
                member_indices = []
                for member in members:
                    member_indices.extend(np.flatnonzero(
                        (feature_names == f"numeric__{member}") |
                        (feature_names == f"numeric__missingindicator_{member}") |
                        np.char.startswith(feature_names.astype(str), f"categorical__{member}_")
                    ).tolist())
                member_indices = sorted(set(member_indices))
                if not member_indices:
                    raise RuntimeError(f"No transformed columns found for {construct}: {members}")
                for permutation_id in range(N_PERMUTATIONS):
                    # Deterministic seed gives both algorithms exactly the same
                    # row permutation for a construct, split, and repetition.
                    rng = np.random.default_rng(
                        SEED + split_id*100000 + construct_index*100 + permutation_id)
                    order = rng.permutation(len(test_x))
                    permuted = processed.copy()
                    permuted[:, member_indices] = processed[order][:, member_indices]
                    permuted_batches.append(permuted)
                batch_prediction = estimator.predict(np.vstack(permuted_batches)).reshape(
                    N_PERMUTATIONS, len(test_y))
                drops = [baseline-r2_score(test_y, prediction)
                         for prediction in batch_prediction]
                rows.append({
                    "model": model_name,
                    "split_id": split_id,
                    "repeat": (split_id-1)//5+1,
                    "fold": (split_id-1)%5+1,
                    "construct": construct,
                    "columns": ", ".join(members),
                    "importance_r2_drop": np.mean(drops),
                    "within_split_permutation_sd": np.std(drops, ddof=1),
                })

    raw = pd.DataFrame(rows)
    raw["rank_within_split"] = raw.groupby(["model", "split_id"])["importance_r2_drop"].rank(
        ascending=False, method="average")
    profile = (raw.groupby(["model", "construct", "columns"], as_index=False)
               .agg(importance_mean=("importance_r2_drop", "mean"),
                    importance_sd=("importance_r2_drop", "std"),
                    importance_min=("importance_r2_drop", "min"),
                    importance_max=("importance_r2_drop", "max"),
                    positive_splits=("importance_r2_drop", lambda x: int((x > 0).sum())),
                    median_rank=("rank_within_split", "median"),
                    rank_q25=("rank_within_split", lambda x: x.quantile(.25)),
                    rank_q75=("rank_within_split", lambda x: x.quantile(.75)),
                    top5_rate=("rank_within_split", lambda x: np.mean(x <= 5)),
                    top10_rate=("rank_within_split", lambda x: np.mean(x <= 10))))
    profile["mean_importance_rank"] = profile.groupby("model").importance_mean.rank(
        ascending=False, method="average")

    wide_importance = profile.pivot(index="construct", columns="model", values="importance_mean")
    wide_rank = profile.pivot(index="construct", columns="model", values="mean_importance_rank")
    comparison = wide_importance.join(wide_rank, lsuffix="_importance", rsuffix="_rank").reset_index()
    comparison.columns = [str(c).replace("ElasticNet_importance", "elastic_net_importance")
                          .replace("RandomForest_importance", "random_forest_importance")
                          .replace("ElasticNet_rank", "elastic_net_rank")
                          .replace("RandomForest_rank", "random_forest_rank") for c in comparison.columns]
    comparison["absolute_rank_difference"] = (
        comparison.elastic_net_rank-comparison.random_forest_rank).abs()
    comparison["consensus_mean_rank"] = (
        comparison.elastic_net_rank+comparison.random_forest_rank)/2
    comparison = comparison.sort_values("consensus_mean_rank")

    rho, rho_p = spearmanr(comparison.elastic_net_importance,
                           comparison.random_forest_importance)
    tau, tau_p = kendalltau(comparison.elastic_net_rank,
                            comparison.random_forest_rank)
    top10_en = set(comparison.nsmallest(10, "elastic_net_rank").construct)
    top10_rf = set(comparison.nsmallest(10, "random_forest_rank").construct)
    agreement = pd.DataFrame([{
        "spearman_importance_rho": rho,
        "spearman_p_value": rho_p,
        "kendall_rank_tau": tau,
        "kendall_p_value": tau_p,
        "top10_overlap_n": len(top10_en & top10_rf),
        "top10_jaccard": len(top10_en & top10_rf)/len(top10_en | top10_rf),
        "top10_shared_constructs": "|".join(sorted(top10_en & top10_rf)),
    }])
    baseline = pd.DataFrame(baseline_rows)

    raw.to_csv(OUT/"cross_model_importance_by_split.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(OUT/"cross_model_importance_profile.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT/"construct_rank_comparison.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(OUT/"cross_model_rank_agreement.csv", index=False, encoding="utf-8-sig")
    baseline.to_csv(OUT/"model_baseline_by_split.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT/"CSE_cross_model_importance.xlsx") as writer:
        comparison.to_excel(writer, sheet_name="rank_comparison", index=False)
        profile.to_excel(writer, sheet_name="model_profiles", index=False)
        agreement.to_excel(writer, sheet_name="agreement", index=False)
        raw.to_excel(writer, sheet_name="importance_by_split", index=False)
        baseline.to_excel(writer, sheet_name="model_baselines", index=False)
    metadata = {
        "models": list(models),
        "outer_validation": "5 repeats x 5 folds; identical splits",
        "permutations_per_construct_per_split": N_PERMUTATIONS,
        "permutation_rule": "same row permutation applied to all columns within a construct and shared across algorithms",
        "interpretation": "Fixed-model reliance and rank stability, not causal or unique contribution.",
    }
    (OUT/"importance_stability_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nAGREEMENT\n", agreement.to_string(index=False))
    print("\nCONSENSUS TOP 15\n", comparison.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
