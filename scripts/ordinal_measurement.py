"""Ordinal measurement checks for CSE and trait flow.

Likert items are treated as ordered categorical indicators.  semopy estimates
polychoric correlations and fits the CFA with DWLS.  This is an ordinal DWLS
analysis, not lavaan's mean/variance-adjusted WLSMV estimator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from semopy import Model, calc_stats
import semopy.polycorr as polycorr

from construct_level_cv import ROOT, prepare_data

OUT = ROOT / "results" / "phase6_ordinal_measurement"
OUT.mkdir(parents=True, exist_ok=True)
CSE_ITEMS = ["CPSA1", "CPSA2", "CPSA3"]
FLOW_ITEMS = [f"SMC{i}" for i in range(1, 10)]
REDUCED_FLOW_ITEMS = ["SMC2", "SMC3", "SMC4", "SMC5", "SMC7", "SMC8"]


def _patch_semopy_for_current_numpy_scipy():
    """Compatibility replacements for APIs removed from NumPy/SciPy.

    semopy 2.3.11 calls ``np.ma.corrcoef(..., bias=True)`` and
    ``scipy.stats.mvn.mvnun``.  The replacements below implement the same
    correlation initialization and Gaussian rectangle probabilities using
    current public SciPy functions.  No package files are modified.
    """
    polycorr.cor = lambda data: np.corrcoef(np.asarray(data), rowvar=False)

    def univariate_cdf(lower, upper, mean=0, var=1):
        scale = np.sqrt(var)
        return norm.cdf((upper-mean)/scale) - norm.cdf((lower-mean)/scale)

    def bivariate_cdf(lower, upper, corr, means=(0, 0), var=(1, 1)):
        covariance = [[var[0], corr], [corr, var[1]]]
        cdf = lambda point: multivariate_normal.cdf(point, mean=means, cov=covariance)
        probability = (cdf(upper) - cdf([lower[0], upper[1]])
                       - cdf([upper[0], lower[1]]) + cdf(lower))
        return max(float(probability), 1e-12)

    polycorr.univariate_cdf = univariate_cdf
    polycorr.bivariate_cdf = bivariate_cdf


def model_description(kind, flow_items):
    items = CSE_ITEMS + flow_items
    ordinal = "DEFINE(ordinal) " + " ".join(items)
    if kind == "one_factor":
        return "GENERAL =~ " + " + ".join(items) + "\n" + ordinal
    return ("CSE =~ " + " + ".join(CSE_ITEMS) + "\n"
            + "FLOW =~ " + " + ".join(flow_items) + "\n" + ordinal)


def srmr_from_model(model):
    observed = np.asarray(model.mx_cov, dtype=float)
    implied = np.asarray(model.calc_sigma()[0], dtype=float)
    observed_cor = observed / np.sqrt(np.outer(np.diag(observed), np.diag(observed)))
    implied_cor = implied / np.sqrt(np.outer(np.diag(implied), np.diag(implied)))
    residual = observed_cor-implied_cor
    return float(np.sqrt(np.mean(residual[np.triu_indices_from(residual, 1)]**2)))


def ordinal_alpha(correlation):
    k = correlation.shape[0]
    mean_r = correlation[np.triu_indices(k, 1)].mean()
    return float(k*mean_r/(1+(k-1)*mean_r))


def fit_model(data, name, kind, flow_items):
    items = CSE_ITEMS + flow_items
    model = Model(model_description(kind, flow_items))
    solution = model.fit(data[items], obj="DWLS")
    stats = calc_stats(model).iloc[0]
    estimates = model.inspect(std_est=True)
    loadings = estimates[estimates.op.eq("~")].copy()
    loadings.insert(0, "model", name)
    loadings = loadings.rename(columns={"lval": "item", "rval": "factor",
                                       "Estimate": "loading_unstandardized",
                                       "Est. Std": "loading_standardized"})
    latent = estimates[(estimates.op.eq("~~")) & (
        (estimates.lval.eq("CSE") & estimates.rval.eq("FLOW")) |
        (estimates.lval.eq("FLOW") & estimates.rval.eq("CSE"))
    )]
    latent_r = float(latent["Est. Std"].iloc[0]) if len(latent) else np.nan
    row = {
        "model": name,
        "estimator": "polychoric_DWLS_not_WLSMV",
        "converged": bool(solution.success),
        "n": len(data),
        "n_items": len(items),
        "chi_square": float(stats["chi2"]),
        "df": int(stats["DoF"]),
        "p_value": float(stats["chi2 p-value"]),
        "CFI": float(stats["CFI"]),
        "TLI": float(stats["TLI"]),
        "RMSEA": float(stats["RMSEA"]),
        "SRMR": srmr_from_model(model),
        "latent_correlation": latent_r,
    }
    order = list(model.vars["observed"])
    correlation = pd.DataFrame(model.mx_cov, index=order, columns=order)
    return row, loadings, correlation


def reliability_from_loadings(loadings, correlation, model_name):
    rows = []
    for factor, items in [("CSE", CSE_ITEMS), ("FLOW", [c for c in correlation if c.startswith("SMC")])]:
        subset = loadings[(loadings.model.eq(model_name)) & loadings.factor.eq(factor)]
        lam = subset.loading_standardized.astype(float).to_numpy()
        error = 1-lam**2
        omega = (lam.sum()**2)/((lam.sum()**2)+error.sum())
        corr = correlation.loc[items, items].to_numpy()
        rows.append({"model": model_name, "factor": factor, "n_items": len(items),
                     "ordinal_alpha": ordinal_alpha(corr),
                     "omega_from_standardized_loadings": omega,
                     "AVE": np.mean(lam**2)})
    return rows


def htmt_polychoric(correlation, a_items, b_items):
    hetero = np.abs(correlation.loc[a_items, b_items].to_numpy()).mean()
    a = np.abs(correlation.loc[a_items, a_items].to_numpy())
    b = np.abs(correlation.loc[b_items, b_items].to_numpy())
    mono_a = a[np.triu_indices(len(a_items), 1)].mean()
    mono_b = b[np.triu_indices(len(b_items), 1)].mean()
    return float(hetero/np.sqrt(mono_a*mono_b))


def main():
    _patch_semopy_for_current_numpy_scipy()
    data, meta = prepare_data()
    item_data = data[CSE_ITEMS+FLOW_ITEMS].dropna().astype(int)

    specifications = [
        ("one_factor_all_items", "one_factor", FLOW_ITEMS),
        ("two_factor_all_items", "two_factor", FLOW_ITEMS),
        ("two_factor_reduced_flow", "two_factor", REDUCED_FLOW_ITEMS),
    ]
    fit_rows, loading_tables, correlations = [], [], {}
    reliability_rows = []
    for name, kind, flow_items in specifications:
        print(f"Fitting {name} ...", flush=True)
        row, loadings, correlation = fit_model(item_data, name, kind, flow_items)
        fit_rows.append(row)
        loading_tables.append(loadings)
        correlations[name] = correlation
        if kind == "two_factor":
            reliability_rows.extend(reliability_from_loadings(loadings, correlation, name))

    fit = pd.DataFrame(fit_rows)
    loadings = pd.concat(loading_tables, ignore_index=True)
    reliability = pd.DataFrame(reliability_rows)
    full_cor = correlations["two_factor_all_items"]
    discriminant = pd.DataFrame([{
        "model": "two_factor_all_items",
        "latent_correlation": fit.loc[fit.model.eq("two_factor_all_items"), "latent_correlation"].iloc[0],
        "polychoric_HTMT": htmt_polychoric(full_cor, CSE_ITEMS, FLOW_ITEMS),
        "CSE_AVE": reliability.query("model == 'two_factor_all_items' and factor == 'CSE'").AVE.iloc[0],
        "FLOW_AVE": reliability.query("model == 'two_factor_all_items' and factor == 'FLOW'").AVE.iloc[0],
        "note": "Two-factor fit is compared descriptively; no unadjusted DWLS chi-square difference test.",
    }])
    distributions = []
    for item in CSE_ITEMS+FLOW_ITEMS:
        counts = item_data[item].value_counts().sort_index()
        for category, count in counts.items():
            distributions.append({"item": item, "label": meta.column_names_to_labels.get(item),
                                  "category": int(category), "count": int(count),
                                  "proportion": count/len(item_data)})
    distributions = pd.DataFrame(distributions)

    fit.to_csv(OUT/"ordinal_DWLS_fit.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(OUT/"ordinal_DWLS_loadings.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(OUT/"ordinal_reliability.csv", index=False, encoding="utf-8-sig")
    discriminant.to_csv(OUT/"ordinal_discriminant_validity.csv", index=False, encoding="utf-8-sig")
    full_cor.to_csv(OUT/"polychoric_correlation_matrix.csv", encoding="utf-8-sig")
    distributions.to_csv(OUT/"item_distributions.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT/"CSE_ordinal_measurement.xlsx") as writer:
        fit.to_excel(writer, sheet_name="fit", index=False)
        loadings.to_excel(writer, sheet_name="loadings", index=False)
        reliability.to_excel(writer, sheet_name="reliability", index=False)
        discriminant.to_excel(writer, sheet_name="discriminant", index=False)
        full_cor.to_excel(writer, sheet_name="polychoric_correlations")
        distributions.to_excel(writer, sheet_name="item_distributions", index=False)
    metadata = {
        "estimator": "semopy DWLS on polychoric correlations",
        "important_limitation": "Not mean/variance-adjusted WLSMV; chi-square and its p-value are not robust-corrected.",
        "missing_data": "complete item cases",
        "n": len(item_data),
        "semantic_overlap_items_removed": ["SMC1", "SMC6", "SMC9"],
    }
    (OUT/"measurement_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFIT\n", fit.to_string(index=False))
    print("\nDISCRIMINANT VALIDITY\n", discriminant.to_string(index=False))
    print("\nRELIABILITY\n", reliability.to_string(index=False))


if __name__ == "__main__":
    main()
