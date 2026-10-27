"""Build the grayscale APA-style figure set for the CSE manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "figures_apa"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "RandomForest": "Random forest", "ExtraTrees": "Extra Trees",
    "ElasticNet": "Elastic Net", "Ridge": "Ridge",
    "HistGradientBoosting": "Histogram gradient boosting",
    "SplineGAM_Ridge": "Spline additive model",
    "RBF_SVR": "RBF support vector regression", "MeanBaseline": "Mean baseline",
}
DOMAIN_LABELS = {
    "trait_flow": "Dispositional flow", "self_evaluation": "Human-attribute self-ratings",
    "digital_behavior": "Digital behavior",
    "attention_cognitive": "Attention-related experiences",
    "family_environment": "Family/environment", "background": "Background",
}
CONSTRUCT_LABELS = {
    "trait_flow_SMC": "Dispositional flow",
    "positive_self_evaluation": "Positive human-attribute ratings",
    "digital_multitasking_DDMB": "Digital media multitasking",
    "hyperactivity_HD": "Hyperactivity",
    "concentration_difficulty_C": "Concentration difficulty",
    "negative_self_evaluation": "Negative human-attribute ratings",
    "residential_place": "Residential area",
    "attention_deficit_AD": "Attention deficit",
    "fragmented_information_ISB": "Fragmented information intake",
    "gender": "Gender", "parental_control_PCOS": "Parental intrusion",
    "exercise_E": "Exercise", "nature_connectedness_NC": "Nature connectedness",
    "problematic_internet_use_IU": "Problematic internet use",
    "mind_blank_MS": "Mind blanking",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 600, "savefig.bbox": "tight",
    })


def panel_label(axis, label: str) -> None:
    axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontsize=10,
              fontweight="bold", va="top")


def save(fig, stem: str) -> None:
    for extension in ("png", "svg", "tiff"):
        kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        fig.savefig(OUT / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def figure1_algorithm_performance() -> None:
    data = pd.read_csv(ROOT / "results/phase5_algorithm_benchmark/algorithm_summary.csv")
    data = data.sort_values("r2_mean", ascending=True)
    labels = [MODEL_LABELS[item] for item in data["model"]]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), sharey=True)
    specs = (
        (axes[0], "r2_mean", "r2_sd", "Mean outer-fold R²", "A"),
        (axes[1], "mae_mean", "mae_sd", "Mean absolute error", "B"),
    )
    for axis, mean, sd, xlabel, letter in specs:
        axis.errorbar(data[mean], y, xerr=data[sd], fmt="o", color="black",
                      ecolor="0.55", elinewidth=0.9, capsize=2.5, markersize=4.5)
        if mean == "r2_mean":
            axis.axvline(0, color="0.65", linewidth=0.7, linestyle="--")
        axis.set_xlabel(xlabel)
        axis.tick_params(axis="y", length=0)
        panel_label(axis, letter)
    axes[0].set_yticks(y, labels)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].set_xlim(-0.05, 0.45)
    axes[1].set_xlim(1.15, 1.68)
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.15, top=0.96, wspace=0.25)
    save(fig, "figure1_algorithm_performance")


def figure2_subset_landscape() -> None:
    data = pd.read_csv(ROOT / "results/phase7_domain_shapley/domain_subset_summary.csv")
    rng = np.random.default_rng(20261009)
    x = data["n_groups"].to_numpy(float) + rng.uniform(-0.12, 0.12, len(data))
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.scatter(x, data["r2_mean"], s=20, facecolor="0.78", edgecolor="none")
    full = "background|digital_behavior|attention_cognitive|trait_flow|self_evaluation|family_environment"
    highlights = {
        "empty": ("Mean baseline", "o", (18, -16), "left"),
        "trait_flow": ("Flow only", "D", (18, -12), "left"),
        full: ("Complete model", "s", (-18, -14), "right"),
    }
    for subset, (label, marker, text_offset, alignment) in highlights.items():
        row = data.loc[data["subset_groups"] == subset].iloc[0]
        axis.scatter(row["n_groups"], row["r2_mean"], marker=marker, s=55,
                     facecolor="black" if subset != "trait_flow" else "white",
                     edgecolor="black", linewidth=1.1, zorder=4)
        axis.annotate(f"{label} ({row['r2_mean']:.3f})",
                      (row["n_groups"], row["r2_mean"]), xytext=text_offset,
                      textcoords="offset points", ha=alignment, va="center", fontsize=9)
    axis.set_xlabel("Number of predictor domains")
    axis.set_ylabel("Mean outer-fold R²")
    axis.set_xticks(range(7)); axis.set_xlim(-0.35, 6.35); axis.set_ylim(-0.04, 0.42)
    axis.axhline(0, color="0.65", linewidth=0.7, linestyle="--")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.97)
    save(fig, "figure2_domain_subset_landscape")


def figure3_domain_contributions() -> None:
    shapley = pd.read_csv(ROOT / "results/phase7_domain_shapley/domain_shapley_summary.csv").set_index("domain")
    deletion = pd.read_csv(ROOT / "results/phase2_construct_cv/leave_one_group_out_delta_r2.csv")
    deletion["domain"] = deletion["comparison"].str.replace("full - without ", "", regex=False)
    deletion = deletion.set_index("domain")
    order = list(shapley.sort_values("shapley_r2_mean", ascending=True).index)
    labels = [DOMAIN_LABELS[item] for item in order]; y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    axes[0].barh(y, shapley.loc[order, "shapley_r2_mean"], color="0.25", height=0.62)
    axes[1].barh(y, deletion.loc[order, "delta_r2_mean"], facecolor="white",
                 edgecolor="black", linewidth=0.9, height=0.62, hatch="///")
    for index, (axis, xlabel, letter) in enumerate(zip(
        axes, ("Exact Shapley allocation, ΔR²", "Deletion-and-retraining loss, ΔR²"), ("A", "B"))):
        axis.axvline(0, color="black", linewidth=0.8); axis.set_xlabel(xlabel)
        axis.tick_params(axis="y", length=0); panel_label(axis, letter)
        values = shapley.loc[order, "shapley_r2_mean"] if index == 0 else deletion.loc[order, "delta_r2_mean"]
        span = max(values.max() - min(0, values.min()), 0.01)
        for yi, value in zip(y, values):
            offset = span * 0.025
            axis.text(value + offset if value >= 0 else value - offset, yi, f"{value:.3f}",
                      va="center", ha="left" if value >= 0 else "right", fontsize=7.5)
    axes[0].set_yticks(y, labels); axes[1].tick_params(axis="y", labelleft=False)
    axes[0].set_xlim(0, 0.27); axes[1].set_xlim(-0.012, 0.19)
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.16, top=0.96, wspace=0.24)
    save(fig, "figure3_domain_contributions")


def figure4_construct_importance() -> None:
    data = pd.read_csv(ROOT / "results/phase8_cross_model_stability/construct_rank_comparison.csv")
    data = data.nsmallest(13, "consensus_mean_rank")
    flow = data.iloc[[0]]; remainder = data.iloc[1:].sort_values("consensus_mean_rank", ascending=False)
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.2), gridspec_kw={"height_ratios": [1.0, 5.2]})
    markers = (("elastic_net_importance", "Elastic Net", "o", "black"),
               ("random_forest_importance", "Random forest", "s", "white"))
    for column, label, marker, face in markers:
        axes[0].scatter(flow[column], [0], marker=marker, s=42, facecolor=face,
                        edgecolor="black", linewidth=0.9, label=label, zorder=3)
    axes[0].set_yticks([0], ["Dispositional flow"]); axes[0].set_xlim(0.38, 0.45)
    axes[0].set_xlabel("Permutation ΔR²"); axes[0].tick_params(axis="y", length=0)
    panel_label(axes[0], "A")
    y = np.arange(len(remainder))
    for column, label, marker, face in markers:
        axes[1].scatter(remainder[column], y, marker=marker, s=34, facecolor=face,
                        edgecolor="black", linewidth=0.9, label=label, zorder=3)
    axes[1].axvline(0, color="0.55", linestyle="--", linewidth=0.8)
    axes[1].set_yticks(y, [CONSTRUCT_LABELS.get(item, item) for item in remainder["construct"]])
    axes[1].set_xlim(-0.006, 0.080); axes[1].set_xlabel("Permutation ΔR²")
    axes[1].tick_params(axis="y", length=0); panel_label(axes[1], "B")
    axes[1].legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.38, right=0.98, bottom=0.10, top=0.97, hspace=0.34)
    save(fig, "figure4_construct_importance")


def figure_s1_calibration() -> None:
    data = pd.read_csv(ROOT / "results/phase5_algorithm_benchmark/algorithm_participant_predictions.csv")
    fig, axis = plt.subplots(figsize=(5.6, 5.0))
    designs = (("ElasticNet", "Elastic Net", "o", "black"),
               ("RandomForest", "Random forest", "s", "white"))
    for model, label, marker, face in designs:
        frame = data.loc[data["model"] == model].copy()
        frame["bin"] = pd.qcut(frame["predicted"], 10, labels=False, duplicates="drop")
        grouped = frame.groupby("bin", as_index=False).agg(predicted=("predicted", "mean"), observed=("observed", "mean"))
        axis.plot(grouped["predicted"], grouped["observed"], color="black", linewidth=0.8,
                  marker=marker, markersize=4.5, markerfacecolor=face,
                  markeredgecolor="black", label=label)
    axis.plot([5, 13], [5, 13], color="0.55", linestyle="--", linewidth=0.9, label="Perfect calibration")
    axis.set_xlim(5, 13); axis.set_ylim(5, 13)
    axis.set_xlabel("Mean predicted CSE"); axis.set_ylabel("Mean observed CSE")
    axis.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.14, top=0.97)
    save(fig, "figure_s1_calibration")


def figure_s2_algorithm_deltas() -> None:
    data = pd.read_csv(ROOT / "results/phase5_algorithm_benchmark/paired_performance_vs_elastic_net.csv")
    data = data.loc[~data["model"].isin(["ElasticNet", "MeanBaseline"])].sort_values("delta_r2_vs_elastic_net")
    labels = [MODEL_LABELS[item] for item in data["model"]]
    low = data["delta_r2_vs_elastic_net"] - data["delta_r2_descriptive_ci95_low"]
    high = data["delta_r2_descriptive_ci95_high"] - data["delta_r2_vs_elastic_net"]
    y = np.arange(len(data)); fig, axis = plt.subplots(figsize=(6.3, 4.0))
    axis.errorbar(data["delta_r2_vs_elastic_net"], y, xerr=np.vstack([low, high]),
                  fmt="o", color="black", ecolor="0.45", capsize=3, markersize=4.5)
    axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_yticks(y, labels); axis.tick_params(axis="y", length=0)
    axis.set_xlabel("Paired difference in outer-fold R² versus Elastic Net")
    fig.subplots_adjust(left=0.39, right=0.98, bottom=0.18, top=0.96)
    save(fig, "figure_s2_algorithm_deltas")


def figure_s4_digital_robustness() -> None:
    draws = pd.read_csv(ROOT / "results/phase3_robustness/bootstrap_draws.csv")
    entry = pd.read_csv(ROOT / "results/phase3_robustness/entry_order_sensitivity.csv")
    semantic = pd.read_csv(ROOT / "results/phase3_robustness/SMC_semantic_overlap_sensitivity.csv").set_index("specification")
    values = pd.Series({
        "After background": float(entry.loc[entry["comparison"] == "digital after background", "delta_r2_mean"].iloc[0]),
        "After psychological\ncharacteristics": float(semantic.loc["digital_increment_original_SMC", "r2_mean"]),
        "After reduced-flow\nspecification": float(semantic.loc["digital_increment_reduced_SMC", "r2_mean"]),
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    axes[0].hist(draws["delta_R2"], bins=35, color="0.75", edgecolor="white")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].axvline(draws["delta_R2"].quantile(0.025), color="black", linewidth=0.8)
    axes[0].axvline(draws["delta_R2"].quantile(0.975), color="black", linewidth=0.8)
    axes[0].set_xlabel("Fixed-prediction bootstrap ΔR²"); axes[0].set_ylabel("Bootstrap frequency")
    panel_label(axes[0], "A")
    positions = np.arange(len(values))
    axes[1].barh(positions, values, color=["0.25", "0.60", "0.85"],
                 edgecolor="black", linewidth=0.7, height=0.58)
    axes[1].set_yticks(positions, values.index); axes[1].set_xlabel("Incremental outer-fold R²")
    axes[1].set_xlim(0, 0.085); axes[1].invert_yaxis(); axes[1].tick_params(axis="y", length=0)
    for index, value in enumerate(values):
        axes[1].text(value + 0.002, index, f"{value:.3f}", ha="left", va="center", fontsize=8)
    panel_label(axes[1], "B")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.17, top=0.95, wspace=0.46)
    save(fig, "figure_s4_digital_robustness")


def figure_s3_measurement_loadings() -> None:
    data = pd.read_csv(ROOT / "results/phase6_ordinal_measurement/ordinal_DWLS_loadings.csv")
    data = data.loc[data["model"] == "two_factor_all_items"].copy()
    data["item_number"] = data["item"].str.extract(r"(\d+)").astype(int)
    data["label"] = np.where(data["factor"] == "CSE", "CSE" + data["item_number"].astype(str),
                             "Flow" + data["item_number"].astype(str))
    data = data.sort_values(["factor", "item_number"], ascending=[False, False]).reset_index(drop=True)
    y = np.arange(len(data)); fig, axis = plt.subplots(figsize=(5.8, 4.7))
    for factor, marker, face, label in (("CSE", "o", "black", "CSE items"),
                                        ("FLOW", "s", "white", "Flow items")):
        frame = data.loc[data["factor"] == factor]
        axis.scatter(frame["loading_standardized"], frame.index, marker=marker, s=38,
                     facecolor=face, edgecolor="black", linewidth=0.9, label=label)
    axis.axvline(0.50, color="0.55", linestyle="--", linewidth=0.8)
    axis.set_yticks(y, data["label"]); axis.set_xlim(0.45, 1.00)
    axis.set_xlabel("Standardized ordinal factor loading"); axis.tick_params(axis="y", length=0)
    axis.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.14, top=0.97)
    save(fig, "figure_s3_measurement_loadings")


def main() -> None:
    style()
    figure1_algorithm_performance(); figure2_subset_landscape()
    figure3_domain_contributions(); figure4_construct_importance()
    figure_s1_calibration(); figure_s2_algorithm_deltas()
    figure_s3_measurement_loadings(); figure_s4_digital_robustness()
    for path in sorted(OUT.glob("*.png")):
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
