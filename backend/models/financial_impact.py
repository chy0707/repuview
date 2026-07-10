"""
Module 2: Reputation → Financial Impact Modeling
==================================================
Adapted from original macro-financial research:
- Granger Causality: test if reputation score Granger-causes abnormal returns / volatility
- Impulse Response Functions (IRF): trace the impact trajectory of a reputation shock
- Forecast Error Variance Decomposition (FEVD): quantify how much stock variance
  is explained by reputation vs market vs idiosyncratic factors

Research parallel:
- Original research tested: "Does US financial stress Granger-cause European sentiment?"
- This tests: "Does brand reputation score Granger-cause stock abnormal returns?"
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from scipy import stats
from typing import Dict, Tuple, Optional


def _ensure_stationary(series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
    """Apply differencing until ADF test rejects unit root at 5%."""
    d = 0
    s = series.dropna().copy()
    while d < max_diff:
        result = adfuller(s, autolag="AIC")
        if result[1] < 0.05:
            return s, d
        s = s.diff().dropna()
        d += 1
    return s, d


def granger_causality_test(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Test if reputation score Granger-causes:
    1. Daily abnormal returns
    2. Realized volatility changes

    Returns DataFrame with p-values for each ticker and lag.
    Follows the same pairwise Granger causality results.
    """
    results = []

    for ticker in df["ticker"].unique():
        # Merge market data with reputation scores
        market = df[df["ticker"] == ticker][["date", "daily_return", "realized_vol_20d"]].copy()
        market = market.set_index("date")

        rep = scores_df[scores_df["ticker"] == ticker][["date", "reputation_score"]].copy()
        rep = rep.set_index("date")

        merged = market.join(rep, how="inner").dropna()

        if len(merged) < max_lag * 3 + 20:
            continue

        # Test 1: reputation_score → daily_return
        try:
            test_data_ret = merged[["daily_return", "reputation_score"]].values
            gc_ret = grangercausalitytests(
                test_data_ret, maxlag=max_lag, verbose=False
            )
            for lag in range(1, max_lag + 1):
                f_pval = gc_ret[lag][0]["ssr_ftest"][1]
                chi2_pval = gc_ret[lag][0]["ssr_chi2test"][1]
                results.append({
                    "ticker": ticker,
                    "target": "daily_return",
                    "lag": lag,
                    "f_test_pvalue": round(f_pval, 4),
                    "chi2_test_pvalue": round(chi2_pval, 4),
                    "significant_5pct": f_pval < 0.05,
                })
        except Exception:
            pass

        # Test 2: reputation_score → realized_vol
        try:
            vol_data = merged[["realized_vol_20d", "reputation_score"]].dropna()
            if len(vol_data) > max_lag * 3 + 20:
                test_data_vol = vol_data.values
                gc_vol = grangercausalitytests(
                    test_data_vol, maxlag=max_lag, verbose=False
                )
                for lag in range(1, max_lag + 1):
                    f_pval = gc_vol[lag][0]["ssr_ftest"][1]
                    chi2_pval = gc_vol[lag][0]["ssr_chi2test"][1]
                    results.append({
                        "ticker": ticker,
                        "target": "realized_vol_20d",
                        "lag": lag,
                        "f_test_pvalue": round(f_pval, 4),
                        "chi2_test_pvalue": round(chi2_pval, 4),
                        "significant_5pct": f_pval < 0.05,
                    })
        except Exception:
            pass

    return pd.DataFrame(results)


def var_irf_fevd(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    ticker: str,
    var_lags: int = 2,
    irf_periods: int = 20,
    weekly: bool = True,
) -> Dict:
    """
    Fit VAR model and compute IRF + FEVD for a single company.

    VAR system: [reputation_score, daily_return, realized_vol_20d]

    This follows the same VAR(p) → IRF → FEVD pipeline,
    but at the company level instead of country level.

    Parameters:
    - weekly: if True, aggregate to weekly frequency (reduces noise)
    """
    market = df[df["ticker"] == ticker][["date", "daily_return", "realized_vol_20d"]].copy()
    market = market.set_index("date")

    rep = scores_df[scores_df["ticker"] == ticker][["date", "reputation_score"]].copy()
    rep = rep.set_index("date")

    merged = market.join(rep, how="inner").dropna()

    if weekly:
        merged = merged.resample("W").agg({
            "daily_return": "sum",
            "realized_vol_20d": "last",
            "reputation_score": "mean",
        }).dropna()

    # Ensure stationarity
    var_data = pd.DataFrame()
    diff_info = {}
    for col in ["reputation_score", "daily_return", "realized_vol_20d"]:
        s, d = _ensure_stationary(merged[col])
        var_data[col] = s
        diff_info[col] = d

    var_data = var_data.dropna()

    if len(var_data) < var_lags + 20:
        return {"error": f"Insufficient data for {ticker}"}

    # Fit VAR
    model = VAR(var_data)

    # Select optimal lag via AIC
    try:
        lag_selection = model.select_order(maxlags=min(8, len(var_data) // 5))
        optimal_lag = max(lag_selection.aic, 1)
        if optimal_lag > var_lags:
            optimal_lag = var_lags
    except Exception:
        optimal_lag = var_lags

    var_result = model.fit(optimal_lag)

    # IRF: Impulse Response Functions
    irf = var_result.irf(irf_periods)
    irf_data = {}
    var_names = var_data.columns.tolist()
    for i, impulse in enumerate(var_names):
        for j, response in enumerate(var_names):
            key = f"{impulse} → {response}"
            irf_data[key] = {
                "response": irf.irfs[:, j, i].tolist(),
                "lower": irf.ci[:, j, i, 0].tolist() if hasattr(irf, 'ci') else None,
                "upper": irf.ci[:, j, i, 1].tolist() if hasattr(irf, 'ci') else None,
            }

    # FEVD: Forecast Error Variance Decomposition
    fevd = var_result.fevd(irf_periods)
    fevd_data = {}
    for i, name in enumerate(var_names):
        fevd_data[name] = pd.DataFrame(
            fevd.decomp[i],
            columns=var_names,
            index=range(1, irf_periods + 1),
        )

    return {
        "ticker": ticker,
        "var_lags": optimal_lag,
        "diff_info": diff_info,
        "aic": var_result.aic,
        "bic": var_result.bic,
        "irf": irf_data,
        "irf_obj": irf,
        "fevd": fevd_data,
        "fevd_obj": fevd,
        "var_result": var_result,
        "n_obs": len(var_data),
    }


def compute_reputation_beta(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling 'reputation beta': sensitivity of stock returns
    to reputation score changes.

    Higher |beta| means the stock reacts more strongly to reputation shifts.
    Analogous to the original research's FEVD showing what % of variance is driven by external stress.
    """
    results = []

    for ticker in df["ticker"].unique():
        market = df[df["ticker"] == ticker][["date", "daily_return"]].set_index("date")
        rep = scores_df[scores_df["ticker"] == ticker][["date", "reputation_score"]].set_index("date")
        merged = market.join(rep, how="inner").dropna()

        rep_change = merged["reputation_score"].diff()
        ret = merged["daily_return"]

        for i in range(window, len(merged)):
            x = rep_change.iloc[i - window:i].values
            y = ret.iloc[i - window:i].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 20:
                continue
            slope, _, r_value, p_value, _ = stats.linregress(x[mask], y[mask])
            results.append({
                "date": merged.index[i],
                "ticker": ticker,
                "reputation_beta": round(slope, 6),
                "r_squared": round(r_value ** 2, 4),
                "p_value": round(p_value, 4),
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/reputation-capital-model/data")
    from sample_data import generate_dataset
    sys.path.insert(0, "/home/claude/reputation-capital-model/models")
    from pca_reputation_score import extract_reputation_scores

    df = generate_dataset()
    scores_df, _ = extract_reputation_scores(df)

    # Granger Causality
    print("=== Granger Causality Tests ===\n")
    gc_results = granger_causality_test(df, scores_df, max_lag=3)
    sig = gc_results[gc_results["significant_5pct"]]
    print(f"Total tests: {len(gc_results)}")
    print(f"Significant at 5%: {len(sig)}")
    print(f"\nSignificant results:\n{sig.to_string(index=False)}")

    # VAR / IRF / FEVD for one company
    print("\n\n=== VAR / IRF / FEVD (AAPL) ===\n")
    result = var_irf_fevd(df, scores_df, "AAPL")
    if "error" not in result:
        print(f"VAR lags: {result['var_lags']}")
        print(f"AIC: {result['aic']:.2f}")
        print(f"Observations: {result['n_obs']}")
        print(f"\nFEVD at horizon 10 (daily_return):")
        print(result['fevd']['daily_return'].iloc[9].round(3))
