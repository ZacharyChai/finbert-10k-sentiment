"""
Does 10-K risk-factor tone predict forward returns and volatility?

Loads data/processed/filings.csv and runs:
  1. descriptives         -- tone over time, FinBERT vs Loughran-McDonald agreement
  2. construct validity   -- does tone line up with the firm's recent performance?
  3. predictive panels    -- forward abnormal return / forward vol ~ tone + controls,
                             firm + year fixed effects, two-way clustered SEs
  4. FinBERT vs LM horse race -- incremental predictive power over the dictionary
  5. tercile portfolio sort   -- the finance-standard high-minus-low presentation
  6. placebo              -- does tone "predict" PAST returns? (it partly does; that
                             is the main threat and it is reported, not hidden)
  7. multiple-testing ledger

Writes output/*.csv and figures/*.png. Run: python src/analysis.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from plots import (
    plot_coef_forest,
    plot_finbert_vs_lm,
    plot_tercile_bars,
    plot_tone_over_time,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

TONE_MEASURES = {
    "fb_net_sentiment": "FinBERT net sentiment (level)",
    "fb_pct_negative": "FinBERT % negative sentences (level)",
    "lm_neg_frac": "LM negative-word fraction (level)",
    "d_fb_net_sentiment": "FinBERT net sentiment (YoY change)",
    "d_lm_neg_frac": "LM negative fraction (YoY change)",
}
OUTCOMES = {
    "aret_1m": "abnormal return, +1 month",
    "aret_3m": "abnormal return, +3 months",
    "aret_6m": "abnormal return, +6 months",
    "fwd_vol": "forward realised volatility (annualised)",
    "vol_change": "change in realised vol (forward − trailing)",
}
CONTROLS = ["pre_vol", "pre_ret_6m", "ln_dollar_vol", "ln_section_chars"]


# --------------------------------------------------------------------------- #
# load & prep
# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    df = pd.read_csv(PROC / "filings.csv", parse_dates=["filing_date", "report_date"])
    from config import SECTORS

    df["sector"] = df["ticker"].map(SECTORS)
    df["ln_section_chars"] = np.log(df["section_chars"])
    df["firm_id"] = df["ticker"].astype("category").cat.codes
    # z-score every tone measure and control within the full sample so
    # coefficients read as "per 1 SD"
    for c in list(TONE_MEASURES) + ["pre_vol", "pre_ret_6m", "ln_dollar_vol", "ln_section_chars"]:
        if c in df:
            df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
    for oc in ["aret_1m", "aret_3m", "aret_6m", "fwd_vol", "vol_change"]:
        if oc in df:
            df[oc] = pd.to_numeric(df[oc], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# regression helper: firm + year FE, two-way clustered SE
# --------------------------------------------------------------------------- #
def panel_reg(df: pd.DataFrame, outcome: str, rhs: list[str],
              fe: tuple[str, ...] = ("firm_id", "fiscal_year")) -> dict:
    d = df.dropna(subset=[outcome, *rhs]).copy()
    if len(d) < 40:
        return {"n": len(d), "note": "too few obs"}
    terms = list(rhs) + [f"C({f})" for f in fe]
    formula = f"{outcome} ~ " + " + ".join(terms)
    model = smf.ols(formula, data=d).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[list(fe)].astype(str).agg("_".join, axis=1)}
        if len(fe) == 1 else {"groups": np.asarray(d[list(fe)])},
    )
    key = rhs[0]
    return {
        "n": int(model.nobs),
        "coef": float(model.params[key]),
        "se": float(model.bse[key]),
        "t": float(model.tvalues[key]),
        "p": float(model.pvalues[key]),
        "r2_within": float(model.rsquared),
    }


# --------------------------------------------------------------------------- #
# 1. descriptives
# --------------------------------------------------------------------------- #
def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    by_year = df.groupby("fiscal_year").agg(
        n=("ticker", "count"),
        fb_net=("fb_net_sentiment", "mean"),
        fb_pct_neg=("fb_pct_negative", "mean"),
        lm_neg=("lm_neg_frac", "mean"),
        section_chars=("section_chars", "median"),
    )
    corr = df[["fb_net_sentiment", "fb_pct_negative", "lm_neg_frac"]].corr()
    by_year.to_csv(OUT / "tone_by_year.csv")
    corr.to_csv(OUT / "tone_measure_correlations.csv")
    plot_tone_over_time(by_year, FIG / "tone_over_time.png")
    plot_finbert_vs_lm(df, FIG / "finbert_vs_lm.png")
    return by_year, corr


# --------------------------------------------------------------------------- #
# 2. construct validity: tone vs the firm's own recent performance
# --------------------------------------------------------------------------- #
def construct_validity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tone in ["fb_net_sentiment", "fb_pct_negative", "lm_neg_frac"]:
        for driver in ["pre_ret_6m", "pre_vol"]:
            r = panel_reg(df, tone, [f"z_{driver}"], fe=("fiscal_year",))
            rows.append({"tone": tone, "driver": driver, **r})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "construct_validity.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# 3 + 4. predictive panels and the FinBERT-vs-LM horse race
# --------------------------------------------------------------------------- #
def predictive_panels(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tone in TONE_MEASURES:
        z = f"z_{tone}"
        if z not in df:
            continue
        for oc in OUTCOMES:
            base = panel_reg(df, oc, [z] + [f"z_{c}" for c in CONTROLS])
            rows.append({"outcome": oc, "tone": tone, "spec": "tone + controls", **base})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "predictive_grid.csv", index=False)

    # horse race: FinBERT net sentiment AND LM negativity in the same model
    hr = []
    for oc in OUTCOMES:
        r = panel_reg(df, oc, ["z_fb_net_sentiment", "z_lm_neg_frac", *[f"z_{c}" for c in CONTROLS]])
        # panel_reg reports the first rhs; refit-read both explicitly
        d = df.dropna(subset=[oc, "z_fb_net_sentiment", "z_lm_neg_frac", *[f"z_{c}" for c in CONTROLS]])
        m = smf.ols(
            f"{oc} ~ z_fb_net_sentiment + z_lm_neg_frac + "
            + " + ".join(f"z_{c}" for c in CONTROLS)
            + " + C(firm_id) + C(fiscal_year)",
            data=d,
        ).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(d[["firm_id", "fiscal_year"]])})
        hr.append({
            "outcome": oc, "n": int(m.nobs),
            "fb_coef": float(m.params["z_fb_net_sentiment"]), "fb_p": float(m.pvalues["z_fb_net_sentiment"]),
            "lm_coef": float(m.params["z_lm_neg_frac"]), "lm_p": float(m.pvalues["z_lm_neg_frac"]),
        })
    horse = pd.DataFrame(hr)
    horse.to_csv(OUT / "horse_race.csv", index=False)
    return grid, horse


# --------------------------------------------------------------------------- #
# 5. tercile portfolio sort (within-year ranks)
# --------------------------------------------------------------------------- #
def tercile_sort(df: pd.DataFrame, tone: str) -> pd.DataFrame:
    d = df.dropna(subset=[tone]).copy()
    d["tercile"] = d.groupby("fiscal_year")[tone].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["T1_low", "T2", "T3_high"])
    )
    rows = []
    for oc in ["aret_1m", "aret_3m", "aret_6m", "fwd_vol"]:
        g = d.dropna(subset=[oc]).groupby("tercile")[oc].agg(["mean", "sem", "count"])
        spread = g.loc["T3_high", "mean"] - g.loc["T1_low", "mean"]
        se_sp = np.sqrt(g.loc["T3_high", "sem"] ** 2 + g.loc["T1_low", "sem"] ** 2)
        rows.append({
            "tone": tone, "outcome": oc,
            "T1_low": g.loc["T1_low", "mean"], "T2": g.loc["T2", "mean"],
            "T3_high": g.loc["T3_high", "mean"],
            "high_minus_low": spread, "hml_se": se_sp, "hml_t": spread / se_sp,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. placebo: does tone "predict" the PAST?
# --------------------------------------------------------------------------- #
def placebo_past(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tone in ["fb_net_sentiment", "fb_pct_negative", "lm_neg_frac",
                 "d_fb_net_sentiment", "d_lm_neg_frac"]:
        z = f"z_{tone}"
        if z not in df:
            continue
        r = panel_reg(df, "pre_ret_6m", [z] + [f"z_{c}" for c in ["pre_vol", "ln_dollar_vol"]])
        rows.append({"tone": tone, "target": "trailing 6m return (PAST)", **r})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "placebo_past.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    df = load()
    print(f"{len(df)} filing-year observations, {df['ticker'].nunique()} firms, "
          f"FY{df['fiscal_year'].min()}-{df['fiscal_year'].max()}")
    print(df.groupby("section_used")["ticker"].count().to_string())

    by_year, corr = descriptives(df)
    print("\n=== tone measure correlations ===")
    print(corr.round(3).to_string())

    cv = construct_validity(df)
    print("\n=== construct validity: tone ~ firm's own recent performance (year FE) ===")
    print(cv.round(4).to_string(index=False))

    grid, horse = predictive_panels(df)
    print("\n=== predictive grid: forward outcome ~ z(tone) + controls, firm+year FE ===")
    print(grid[["outcome", "tone", "n", "coef", "se", "p", "r2_within"]].round(4).to_string(index=False))
    print("\n=== FinBERT vs LM horse race (both in one model) ===")
    print(horse.round(4).to_string(index=False))

    sorts = pd.concat([tercile_sort(df, t) for t in
                       ["fb_net_sentiment", "fb_pct_negative", "lm_neg_frac"]], ignore_index=True)
    sorts.to_csv(OUT / "tercile_sorts.csv", index=False)
    print("\n=== tercile portfolio sort (within-year), high-minus-low ===")
    print(sorts.round(4).to_string(index=False))
    plot_tercile_bars(sorts, FIG / "tercile_sort.png")

    plac = placebo_past(df)
    print("\n=== PLACEBO: does tone 'predict' the trailing 6m return? ===")
    print(plac.round(4).to_string(index=False))

    plot_coef_forest(grid, FIG / "coef_forest.png")

    # multiple-testing ledger
    n_tests = len(grid)
    n_sig = int((grid["p"] < 0.05).sum())
    ledger = pd.DataFrame([{
        "n_predictive_tests": n_tests,
        "n_significant_p05": n_sig,
        "expected_false_positives_at_5pct": round(0.05 * n_tests, 1),
        "bonferroni_alpha": round(0.05 / n_tests, 4),
        "n_significant_after_bonferroni": int((grid["p"] < 0.05 / n_tests).sum()),
    }])
    ledger.to_csv(OUT / "multiple_testing_ledger.csv", index=False)
    print("\n=== multiple-testing ledger ===")
    print(ledger.to_string(index=False))

    df.to_csv(PROC / "analysis_frame.csv", index=False)
    print(f"\nfigures -> {FIG}\noutput  -> {OUT}")


if __name__ == "__main__":
    main()
