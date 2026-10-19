"""Phase-3 robustness, discriminant-validity, and conditional-association checks."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from construct_level_cv import (
    ROOT, prepare_data, make_model, BACKGROUND, DIGITAL, ATTENTION, FLOW,
    SELF_EVALUATION, FAMILY_ENVIRONMENT, THEORY_GROUPS, SEED,
)

warnings.filterwarnings("ignore", message="Found unknown categories")
OUT = ROOT / "results" / "phase3_robustness"
OUT.mkdir(exist_ok=True)


def repeated_oof(columns, d, y, splits, label):
    rows, score_rows = [], []
    for sid, (tr, te) in enumerate(splits, 1):
        model = make_model(columns).fit(d.iloc[tr][columns], y.iloc[tr])
        pred = model.predict(d.iloc[te][columns])
        score_rows.append({"model": label, "split_id": sid, "repeat": (sid-1)//5+1,
                           "fold": (sid-1)%5+1, "r2": r2_score(y.iloc[te], pred),
                           "mae": mean_absolute_error(y.iloc[te], pred),
                           "rmse": mean_squared_error(y.iloc[te], pred)**.5})
        rows.extend({"participant_id": int(i), "split_id": sid, "repeat": (sid-1)//5+1,
                     "model": label, "observed": float(y.iloc[i]), "predicted": float(p)}
                    for i, p in zip(te, pred))
    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def bootstrap_prediction_difference(oof_b, oof_c, reps=10000):
    b = oof_b.groupby("participant_id").agg(observed=("observed", "first"), pred_b=("predicted", "mean"))
    c = oof_c.groupby("participant_id").agg(pred_c=("predicted", "mean"))
    z = b.join(c).reset_index()
    y, pb, pc = z.observed.to_numpy(), z.pred_b.to_numpy(), z.pred_c.to_numpy()

    def metrics(idx):
        yy, bb, cc = y[idx], pb[idx], pc[idx]
        return np.array([r2_score(yy, cc)-r2_score(yy, bb),
                         mean_squared_error(yy, bb)-mean_squared_error(yy, cc),
                         mean_absolute_error(yy, bb)-mean_absolute_error(yy, cc)])

    observed = metrics(np.arange(len(z)))
    rng = np.random.default_rng(SEED)
    boots = np.empty((reps, 3))
    for k in range(reps):
        boots[k] = metrics(rng.integers(0, len(z), len(z)))
    names = ["delta_R2_C_minus_B", "MSE_reduction_B_minus_C", "MAE_reduction_B_minus_C"]
    summary = pd.DataFrame({
        "metric": names, "estimate": observed,
        "bootstrap_mean": boots.mean(axis=0),
        "percentile_ci95_low": np.quantile(boots, .025, axis=0),
        "percentile_ci95_high": np.quantile(boots, .975, axis=0),
        "bootstrap_positive_rate": (boots > 0).mean(axis=0),
        "bootstrap_reps": reps,
    })
    return z, summary, boots


def htmt(data, a_cols, b_cols):
    corr = data[a_cols+b_cols].corr().abs()
    hetero = corr.loc[a_cols, b_cols].to_numpy().mean()
    within_a = corr.loc[a_cols, a_cols].to_numpy()[np.triu_indices(len(a_cols), 1)].mean()
    within_b = corr.loc[b_cols, b_cols].to_numpy()[np.triu_indices(len(b_cols), 1)].mean()
    return hetero / np.sqrt(within_a * within_b)


def cfa_ml_fit(data, two_factor=True):
    """Normal-theory ML CFA on the item correlation matrix; latent variances fixed to 1."""
    x = (data - data.mean()) / data.std(ddof=1)
    s = x.cov().to_numpy()
    p, n = s.shape[0], len(x)
    sign_s, logdet_s = np.linalg.slogdet(s)
    assert sign_s > 0
    groups = np.array([0]*3 + [1]*9)

    def implied(theta):
        load = theta[:p]
        err = np.exp(theta[p:2*p])
        if two_factor:
            phi = np.array([[1., np.tanh(theta[-1])], [np.tanh(theta[-1]), 1.]])
            lam = np.zeros((p, 2)); lam[np.arange(p), groups] = load
        else:
            phi = np.ones((1, 1)); lam = load[:, None]
        return lam @ phi @ lam.T + np.diag(err)

    def objective(theta):
        sigma = implied(theta)
        sign, ld = np.linalg.slogdet(sigma)
        if sign <= 0:
            return 1e12
        try:
            return ld + np.trace(s @ np.linalg.inv(sigma)) - logdet_s - p
        except np.linalg.LinAlgError:
            return 1e12

    # Item-total correlations provide stable starting loadings.
    starts = []
    base_load = np.r_[np.repeat(.75, 3), np.repeat(.65, 9)]
    for jitter in [0, .05, -.05]:
        start = np.r_[base_load+jitter, np.log(np.maximum(.2, 1-(base_load+jitter)**2))]
        if two_factor:
            start = np.r_[start, np.arctanh(.5)]
        starts.append(start)
    fits = [minimize(objective, st, method="L-BFGS-B", options={"maxiter": 5000}) for st in starts]
    fit = min(fits, key=lambda r: r.fun)
    sigma = implied(fit.x)
    f = fit.fun
    q = 2*p + (1 if two_factor else 0)
    df = p*(p+1)//2 - q
    chisq = (n-1)*f
    # Independence/null model.
    sigma0 = np.diag(np.diag(s))
    f0 = np.linalg.slogdet(sigma0)[1] + np.trace(s @ np.linalg.inv(sigma0)) - logdet_s - p
    df0 = p*(p+1)//2 - p
    chi0 = (n-1)*f0
    cfi = 1 - max(chisq-df, 0)/max(chi0-df0, chisq-df, 1e-12)
    tli = (chi0/df0 - chisq/df)/(chi0/df0 - 1)
    rmsea = np.sqrt(max((chisq-df)/(df*(n-1)), 0))
    std_s = s / np.sqrt(np.outer(np.diag(s), np.diag(s)))
    std_sig = sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
    resid = std_s - std_sig
    srmr = np.sqrt(np.mean(resid[np.triu_indices(p, 1)]**2))
    rho = np.tanh(fit.x[-1]) if two_factor else 1.0
    return {"model": "correlated_two_factor" if two_factor else "one_factor",
            "converged": fit.success, "chi_square": chisq, "df": df,
            "p_value": chi2.sf(chisq, df), "CFI": cfi, "TLI": tli,
            "RMSEA": rmsea, "SRMR": srmr, "latent_correlation": rho,
            "AIC_fit_function": chisq + 2*q, "BIC_fit_function": chisq + q*np.log(n)}, fit.x[:p]


def hc3_analysis(d, columns, y):
    model = make_model(columns)
    pre = model.named_steps["pre"]
    X = pre.fit_transform(d[columns])
    if hasattr(X, "toarray"):
        X = X.toarray()
    names = pre.get_feature_names_out()
    Xdf = pd.DataFrame(X, columns=names)
    yz = (y-y.mean())/y.std(ddof=1)
    fit = sm.OLS(yz.to_numpy(), sm.add_constant(Xdf)).fit(cov_type="HC3")
    ci = fit.conf_int()
    table = pd.DataFrame({"feature": ["const"]+list(names), "coefficient": np.asarray(fit.params),
                          "hc3_se": np.asarray(fit.bse), "ci95_low": ci.iloc[:, 0].to_numpy(),
                          "ci95_high": ci.iloc[:, 1].to_numpy(),
                          "p_value": fit.pvalues})
    isb_idx = list(names).index("num__ISB")
    vif_isb = variance_inflation_factor(X, isb_idx)
    return table, float(vif_isb), fit.rsquared


def main():
    d, _ = prepare_data(); y = d.CPSA.astype(float)
    B = BACKGROUND + FLOW + ATTENTION + SELF_EVALUATION
    C = B + DIGITAL
    splits = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(d, y))
    oof_b, sc_b = repeated_oof(B, d, y, splits, "B_psychology")
    oof_c, sc_c = repeated_oof(C, d, y, splits, "C_plus_digital")
    participant_preds, boot_summary, boots = bootstrap_prediction_difference(oof_b, oof_c)

    # Order sensitivity: digital entered before psychological variables.
    A_digital = BACKGROUND + DIGITAL
    _, sc_ad = repeated_oof(A_digital, d, y, splits, "A_plus_digital_first")
    order_delta = pd.DataFrame({
        "comparison": ["digital after background", "psychology after background+digital"],
        "delta_r2_mean": [(sc_ad.r2-sc_b.assign(r2=0).r2).mean(), np.nan],
    })
    # Correct first comparison against a separately available background-only score from v2.
    bg_path = ROOT / "results" / "phase2_construct_cv" / "hierarchical_repeated_cv_scores.csv"
    old = pd.read_csv(bg_path)
    sc_a = old[old.model.eq("A_background")].sort_values("split_id")
    sc_ad = sc_ad.sort_values("split_id"); sc_c = sc_c.sort_values("split_id")
    order_delta.loc[0, "delta_r2_mean"] = np.mean(sc_ad.r2.to_numpy()-sc_a.r2.to_numpy())
    order_delta.loc[1, "delta_r2_mean"] = np.mean(sc_c.r2.to_numpy()-sc_ad.r2.to_numpy())
    order_delta["positive_splits"] = [int(np.sum(sc_ad.r2.to_numpy()>sc_a.r2.to_numpy())),
                                       int(np.sum(sc_c.r2.to_numpy()>sc_ad.r2.to_numpy()))]

    # Semantic-overlap sensitivity: omit skill-match, control, and accomplishment SMC items.
    reduced_items = ["SMC2", "SMC3", "SMC4", "SMC5", "SMC7", "SMC8"]
    d["SMC_reduced"] = d[reduced_items].mean(axis=1)
    B_red = ["SMC_reduced" if c == "SMC" else c for c in B]
    C_red = B_red + DIGITAL
    _, sc_br = repeated_oof(B_red, d, y, splits, "B_reduced_SMC")
    _, sc_cr = repeated_oof(C_red, d, y, splits, "C_reduced_SMC_plus_digital")
    semantic_sensitivity = pd.DataFrame([
        {"specification": "original_SMC_B", "r2_mean": sc_b.r2.mean()},
        {"specification": "reduced_SMC_B", "r2_mean": sc_br.r2.mean()},
        {"specification": "original_SMC_C", "r2_mean": sc_c.r2.mean()},
        {"specification": "reduced_SMC_C", "r2_mean": sc_cr.r2.mean()},
        {"specification": "digital_increment_original_SMC", "r2_mean": (sc_c.r2.to_numpy()-sc_b.r2.to_numpy()).mean()},
        {"specification": "digital_increment_reduced_SMC", "r2_mean": (sc_cr.r2.to_numpy()-sc_br.r2.to_numpy()).mean()},
    ])

    cse_items = ["CPSA1", "CPSA2", "CPSA3"]
    smc_items = [f"SMC{i}" for i in range(1, 10)]
    psychometric = d[cse_items+smc_items]
    fit1, load1 = cfa_ml_fit(psychometric, False)
    fit2, load2 = cfa_ml_fit(psychometric, True)
    cfa = pd.DataFrame([fit1, fit2])
    cfa_comparison = pd.DataFrame([{
        "comparison": "one_factor minus two_factor", "delta_chi_square": fit1["chi_square"]-fit2["chi_square"],
        "delta_df": fit1["df"]-fit2["df"],
        "p_value": chi2.sf(fit1["chi_square"]-fit2["chi_square"], fit1["df"]-fit2["df"]),
        "HTMT": htmt(d, cse_items, smc_items),
    }])
    loadings = pd.DataFrame({"item": cse_items+smc_items,
                             "one_factor_loading": load1, "two_factor_loading": load2})

    full = sum(THEORY_GROUPS.values(), [])
    hc3, isb_vif, ols_r2 = hc3_analysis(d, full, y)
    hc3_notes = pd.DataFrame({"statistic": ["ISB_VIF", "OLS_R2"], "value": [isb_vif, ols_r2]})

    representation = pd.DataFrame([
        {"feature_representation": "item+total original sensitivity", "cv_scheme": "single 5-fold", "r2": .417216},
        {"feature_representation": "construct-level model C", "cv_scheme": "5x5 repeated nested CV", "r2": sc_c.r2.mean()},
        {"feature_representation": "construct-level full model D", "cv_scheme": "5x5 repeated nested CV", "r2": old[old.model.eq('D_full_plus_family_environment')].r2.mean()},
    ])

    pd.concat([oof_b, oof_c]).to_csv(OUT/"participant_oof_predictions_long.csv", index=False, encoding="utf-8-sig")
    participant_preds.to_csv(OUT/"participant_mean_oof_predictions.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(OUT/"participant_bootstrap_increment.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boots, columns=["delta_R2", "MSE_reduction", "MAE_reduction"]).to_csv(OUT/"bootstrap_draws.csv", index=False)
    order_delta.to_csv(OUT/"entry_order_sensitivity.csv", index=False, encoding="utf-8-sig")
    semantic_sensitivity.to_csv(OUT/"SMC_semantic_overlap_sensitivity.csv", index=False, encoding="utf-8-sig")
    cfa.to_csv(OUT/"CPSA_SMC_CFA_fit.csv", index=False, encoding="utf-8-sig")
    cfa_comparison.to_csv(OUT/"CPSA_SMC_discriminant_validity.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(OUT/"CPSA_SMC_CFA_loadings.csv", index=False, encoding="utf-8-sig")
    hc3.to_csv(OUT/"construct_OLS_HC3.csv", index=False, encoding="utf-8-sig")
    hc3_notes.to_csv(OUT/"construct_OLS_diagnostics.csv", index=False, encoding="utf-8-sig")
    representation.to_csv(OUT/"feature_representation_sensitivity.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT/"CSE_phase3_robustness_results.xlsx") as w:
        boot_summary.to_excel(w, sheet_name="bootstrap_increment", index=False)
        participant_preds.to_excel(w, sheet_name="participant_predictions", index=False)
        order_delta.to_excel(w, sheet_name="entry_order", index=False)
        semantic_sensitivity.to_excel(w, sheet_name="SMC_sensitivity", index=False)
        cfa.to_excel(w, sheet_name="CFA_fit", index=False)
        cfa_comparison.to_excel(w, sheet_name="discriminant_validity", index=False)
        loadings.to_excel(w, sheet_name="CFA_loadings", index=False)
        hc3.to_excel(w, sheet_name="OLS_HC3", index=False)
        hc3_notes.to_excel(w, sheet_name="OLS_diagnostics", index=False)
        representation.to_excel(w, sheet_name="representation", index=False)

    print("BOOTSTRAP\n", boot_summary.to_string(index=False))
    print("\nORDER\n", order_delta.to_string(index=False))
    print("\nSMC SENSITIVITY\n", semantic_sensitivity.to_string(index=False))
    print("\nCFA\n", cfa.to_string(index=False))
    print("\nDISCRIMINANT\n", cfa_comparison.to_string(index=False))
    print("\nISB HC3\n", hc3[hc3.feature.eq("num__ISB")].to_string(index=False))
    print("ISB VIF", isb_vif)


if __name__ == "__main__":
    main()
