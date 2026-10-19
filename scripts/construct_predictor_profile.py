"""Construct-level predictor profile for creative self-efficacy."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

from construct_level_cv import (
    ROOT, prepare_data, make_model, THEORY_GROUPS, SEED
)

warnings.filterwarnings("ignore", message="Found unknown categories")
OUT = ROOT / "results" / "phase4_predictor_profile"
OUT.mkdir(exist_ok=True)

# Each group is a theoretical construct or a standalone background characteristic.
CONSTRUCTS = {
    "trait_flow_SMC": ["SMC"],
    "digital_multitasking_DDMB": ["DDMB"],
    "problematic_internet_use_IU": ["IU"],
    "video_speed_behavior_WVB": ["WVB"],
    "fragmented_information_ISB": ["ISB"],
    "information_overload_AIP": ["AIP"],
    "mind_blank_MS": ["MS"],
    "concentration_difficulty_C": ["C"],
    "attention_deficit_AD": ["AD"],
    "hyperactivity_HD": ["HD"],
    "positive_self_evaluation": [f"high_self_esteem{i}" for i in range(1, 5)],
    "negative_self_evaluation": [f"low_self_esteem{i}" for i in range(1, 5)],
    "parental_control_PCOS": ["PCOS"],
    "parental_phone_use_PUP": ["PUP"],
    "perceived_greenery_PG": ["PG"],
    "green_space_proximity": ["Rnearest_greenery"],
    "indoor_environment": ["comfort", "temperature", "humidity", "lighting"],
    "pollution_ANWP": ["ANWP"],
    "nature_connectedness_NC": ["NC"],
    "age": ["age"],
    "gender": ["gender"],
    "academic_year": ["academic_year"],
    "enrollment_status": ["enrollment_status"],
    "subjective_SES": ["sss"],
    "marital_status": ["marital_status"],
    "part_time_job": ["part_time_job"],
    "only_child": ["only_child"],
    "family_size": ["siblings_clean"],
    "birth_order": ["birth_order_clean"],
    "academic_ranking": ["academic_ranking"],
    "residential_place": ["place"],
    "exercise_E": ["E"],
    "self_rated_health": ["health"],
    "self_rated_diet": ["diet"],
    "career_plan": ["career_plan"],
}


def main():
    d, _ = prepare_data()
    y = d.CPSA.astype(float)
    columns = sum(THEORY_GROUPS.values(), [])
    assert set(sum(CONSTRUCTS.values(), [])) == set(columns)
    splits = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(d, y))
    rng = np.random.default_rng(SEED)
    rows = []
    baseline_rows = []
    for sid, (tr, te) in enumerate(splits, 1):
        model = make_model(columns).fit(d.iloc[tr][columns], y.iloc[tr])
        xt = d.iloc[te][columns].copy()
        yt = y.iloc[te]
        baseline = r2_score(yt, model.predict(xt))
        baseline_rows.append({"split_id": sid, "repeat": (sid-1)//5+1,
                              "fold": (sid-1)%5+1, "baseline_r2": baseline})
        for construct, members in CONSTRUCTS.items():
            drops = []
            for rep in range(20):
                xp = xt.copy()
                order = rng.permutation(len(xp))
                # One shared row permutation for every column in the construct.
                xp.loc[:, members] = xp.loc[:, members].to_numpy()[order]
                drops.append(baseline-r2_score(yt, model.predict(xp)))
            rows.append({"split_id": sid, "construct": construct,
                         "columns": ", ".join(members),
                         "importance_r2_drop": np.mean(drops),
                         "within_split_permutation_sd": np.std(drops, ddof=1)})
    raw = pd.DataFrame(rows)
    profile = (raw.groupby(["construct", "columns"], as_index=False)
               .agg(importance_mean=("importance_r2_drop", "mean"),
                    importance_sd_across_splits=("importance_r2_drop", "std"),
                    importance_min=("importance_r2_drop", "min"),
                    importance_max=("importance_r2_drop", "max"),
                    positive_splits=("importance_r2_drop", lambda s: int((s > 0).sum()))))
    profile["rank"] = profile.importance_mean.rank(ascending=False, method="min").astype(int)

    # Descriptive zero-order direction and full-model conditional coefficient direction.
    desc = []
    coef_path = ROOT / "results" / "phase2_construct_cv" / "conditional_coefficient_stability.csv"
    coefs = pd.read_csv(coef_path)
    for construct, members in CONSTRUCTS.items():
        nominal = any(m in {"gender", "academic_year", "enrollment_status", "marital_status",
                            "part_time_job", "only_child", "academic_ranking", "place",
                            "career_plan"} for m in members)
        proxy = d[members].mean(axis=1)
        rho = proxy.corr(y, method="spearman") if not nominal else np.nan
        matched = coefs[coefs.feature.isin([f"num__{m}" for m in members])]
        if nominal:
            cond = "category-specific"
        elif matched.empty:
            cond = "not available"
        elif (matched.coef_mean > 0).all():
            cond = "positive"
        elif (matched.coef_mean < 0).all():
            cond = "negative"
        else:
            cond = "mixed"
        desc.append({"construct": construct, "spearman_rho": rho,
                     "conditional_direction": cond,
                     "mean_abs_standardized_coefficient": matched.coef_mean.abs().mean() if len(matched) else np.nan})
    profile = profile.merge(pd.DataFrame(desc), on="construct").sort_values("importance_mean", ascending=False)

    domains = pd.read_csv(ROOT/"results"/"phase2_construct_cv"/"leave_one_group_out_delta_r2.csv")
    baseline = pd.DataFrame(baseline_rows)
    raw.to_csv(OUT/"construct_permutation_importance_splits.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(OUT/"CSE_construct_predictor_profile.csv", index=False, encoding="utf-8-sig")
    baseline.to_csv(OUT/"full_model_baseline_splits.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT/"CSE_predictor_profile.xlsx") as w:
        profile.to_excel(w, sheet_name="construct_profile", index=False)
        raw.to_excel(w, sheet_name="importance_splits", index=False)
        domains.to_excel(w, sheet_name="domain_ablation", index=False)
        baseline.to_excel(w, sheet_name="baseline_splits", index=False)
    print(profile.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
