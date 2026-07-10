"""
Reputation Capital Model — End-to-End Analysis Pipeline
========================================================
Run this to execute the full analysis and generate all outputs.

Usage:
    python analysis.py
"""

import sys
import os
import warnings
import json

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from sample_data import generate_dataset, get_reputation_events, COMPANIES
from pca_reputation_score import extract_reputation_scores, compute_reputation_regime
from financial_impact import granger_causality_test, var_irf_fevd, compute_reputation_beta
from spillover_analysis import compute_spillover_table, build_contagion_network, compute_rolling_spillover
from wavelet_analysis import reputation_stock_wavelet, pairwise_company_wavelet


def run_pipeline(save_dir: str = "outputs"):
    """Execute full analysis pipeline and save results."""
    os.makedirs(save_dir, exist_ok=True)

    # ── Step 1: Data ──
    print("=" * 60)
    print("REPUTATION CAPITAL MODEL — FULL ANALYSIS")
    print("=" * 60)

    print("\n[1/7] Generating dataset...")
    df = generate_dataset()
    events = get_reputation_events()
    print(f"  → {len(df):,} data points | {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  → {len(COMPANIES)} companies: {', '.join(COMPANIES.values())}")

    # ── Step 2: PCA Reputation Scores ──
    print("\n[2/7] Extracting reputation scores via PCA...")
    scores_df, pca_diag = extract_reputation_scores(df)
    scores_df = compute_reputation_regime(scores_df)
    for t, d in pca_diag.items():
        print(f"  → {t}: PC1 explains {d['explained_variance_ratio'][0]:.1%} of variance")

    # ── Step 3: Granger Causality ──
    print("\n[3/7] Running Granger causality tests...")
    gc_results = granger_causality_test(df, scores_df, max_lag=5)
    sig = gc_results[gc_results["significant_5pct"]]
    print(f"  → {len(gc_results)} tests total | {len(sig)} significant at 5%")
    if len(sig) > 0:
        for _, row in sig.iterrows():
            print(f"    ✓ {row['ticker']}: reputation → {row['target']} "
                  f"(lag={row['lag']}, p={row['f_test_pvalue']:.4f})")

    # ── Step 4: VAR / IRF / FEVD ──
    print("\n[4/7] Fitting VAR models & computing IRF/FEVD...")
    var_results = {}
    for ticker in COMPANIES:
        res = var_irf_fevd(df, scores_df, ticker)
        if "error" not in res:
            var_results[ticker] = res
            fevd_rep = res["fevd"]["daily_return"].iloc[-1]["reputation_score"]
            print(f"  → {ticker}: reputation explains {fevd_rep:.1%} of return variance")
        else:
            print(f"  → {ticker}: {res['error']}")

    # ── Step 5: Spillover Analysis ──
    print("\n[5/7] Computing Diebold-Yilmaz spillover index...")
    spillover = compute_spillover_table(scores_df)
    if "error" not in spillover:
        print(f"  → Total Spillover Index: {spillover['total_spillover_index']}%")
        print(f"  → Net spillovers:")
        for t, v in sorted(spillover["net"].items(), key=lambda x: -x[1]):
            role = "TRANSMITTER" if v > 0 else "RECEIVER"
            print(f"    {t}: {v:+.1f}% ({role})")

        network = build_contagion_network(spillover, threshold=3.0)
        print(f"  → Network: {network['n_edges']} contagion paths, density={network['density']}")
    else:
        network = None

    # ── Step 6: Rolling Spillover ──
    print("\n[6/7] Computing rolling spillover index...")
    rolling_spill = compute_rolling_spillover(scores_df, window=80, step=4)
    print(f"  → {len(rolling_spill)} rolling windows computed")
    if len(rolling_spill) > 0:
        print(f"  → Range: {rolling_spill['total_spillover_index'].min():.1f}% - "
              f"{rolling_spill['total_spillover_index'].max():.1f}%")

    # ── Step 7: Wavelet Analysis ──
    print("\n[7/7] Running wavelet analysis...")
    wavelet_results = {}
    for ticker in COMPANIES:
        wav = reputation_stock_wavelet(df, scores_df, ticker)
        wavelet_results[ticker] = wav
        bc = wav["band_coherence"]
        print(f"  → {ticker}: short={bc['short_term']}, med={bc['medium_term']}, long={bc['long_term']}")

    pair_coherence = pairwise_company_wavelet(scores_df)

    # ── Step 8: Reputation Beta ──
    print("\n[BONUS] Computing rolling reputation beta...")
    rep_beta = compute_reputation_beta(df, scores_df, window=60)
    print(f"  → {len(rep_beta):,} beta observations")

    # ── Save outputs ──
    print(f"\n{'=' * 60}")
    print("SAVING RESULTS...")

    df.to_csv(f"{save_dir}/market_data.csv", index=False)
    scores_df.to_csv(f"{save_dir}/reputation_scores.csv", index=False)
    gc_results.to_csv(f"{save_dir}/granger_causality.csv", index=False)
    if "error" not in spillover:
        spillover["spillover_table"].to_csv(f"{save_dir}/spillover_table.csv")
    rolling_spill.to_csv(f"{save_dir}/rolling_spillover.csv", index=False)
    pair_coherence.to_csv(f"{save_dir}/pairwise_coherence.csv", index=False)
    rep_beta.to_csv(f"{save_dir}/reputation_beta.csv", index=False)
    events.to_csv(f"{save_dir}/reputation_events.csv", index=False)

    print(f"  → All outputs saved to {save_dir}/")
    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")

    return {
        "df": df,
        "scores_df": scores_df,
        "gc_results": gc_results,
        "var_results": var_results,
        "spillover": spillover,
        "network": network,
        "rolling_spill": rolling_spill,
        "wavelet_results": wavelet_results,
        "pair_coherence": pair_coherence,
        "rep_beta": rep_beta,
        "events": events,
        "pca_diag": pca_diag,
    }


if __name__ == "__main__":
    run_pipeline()
