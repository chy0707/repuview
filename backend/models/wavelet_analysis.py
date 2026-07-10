"""
Module 4: Wavelet Time-Frequency Analysis
==========================================
Cross-wavelet transform between reputation scores and stock returns,
revealing co-movement across different time horizons.

Research parallel:
- Original research used wavelet scalograms showing stress-sentiment co-movement
  at different frequencies (short-term vs long-term cycles)
- This: shows reputation-stock co-movement at different horizons
  (day-to-day noise vs weekly/monthly systematic patterns)

Key insight: if co-movement is strong at medium frequencies (weekly/monthly)
but weak at high frequencies (daily), it suggests reputation effects are
structural, not just noise trading.
"""

import numpy as np
import pandas as pd
import pywt
from typing import Dict, Tuple


def compute_cwt(signal: np.ndarray, scales: np.ndarray = None,
                wavelet: str = "morl") -> Tuple[np.ndarray, np.ndarray]:
    """Compute Continuous Wavelet Transform."""
    if scales is None:
        scales = np.arange(2, min(len(signal) // 2, 128))
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    power = np.abs(coefficients) ** 2
    return power, scales


def cross_wavelet_power(
    signal_x: np.ndarray,
    signal_y: np.ndarray,
    scales: np.ndarray = None,
    wavelet: str = "morl",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-wavelet power spectrum between two signals.

    This uses the same cross-wavelet method from the original macro-financial research,
    showing how strongly two signals co-move at each time-frequency point.
    """
    if scales is None:
        n = min(len(signal_x), len(signal_y))
        scales = np.arange(2, min(n // 2, 128))

    cx, _ = pywt.cwt(signal_x, scales, wavelet)
    cy, _ = pywt.cwt(signal_y, scales, wavelet)

    # Cross-wavelet spectrum
    xwt = cx * np.conj(cy)
    power = np.abs(xwt)

    return power, scales


def wavelet_coherence_approx(
    signal_x: np.ndarray,
    signal_y: np.ndarray,
    scales: np.ndarray = None,
    smooth_window: int = 5,
    wavelet: str = "morl",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate wavelet coherence (squared) using smoothed cross-spectra.

    Coherence ranges from 0 (no relationship) to 1 (perfect co-movement)
    at each time-frequency point.
    """
    if scales is None:
        n = min(len(signal_x), len(signal_y))
        scales = np.arange(2, min(n // 2, 128))

    cx, _ = pywt.cwt(signal_x, scales, wavelet)
    cy, _ = pywt.cwt(signal_y, scales, wavelet)

    # Smooth spectra
    from scipy.ndimage import uniform_filter1d

    sxy = uniform_filter1d(cx * np.conj(cy), smooth_window, axis=1)
    sxx = uniform_filter1d(np.abs(cx) ** 2, smooth_window, axis=1)
    syy = uniform_filter1d(np.abs(cy) ** 2, smooth_window, axis=1)

    # Coherence
    denom = sxx * syy
    denom[denom == 0] = 1e-12
    coherence = np.abs(sxy) ** 2 / denom

    return coherence, scales


def reputation_stock_wavelet(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    ticker: str,
    freq: str = "W",
) -> Dict:
    """
    Full wavelet analysis for one company: reputation score vs stock returns.

    Returns cross-wavelet power, coherence, and scale-averaged metrics.
    """
    market = df[df["ticker"] == ticker][["date", "daily_return"]].set_index("date")
    rep = scores_df[scores_df["ticker"] == ticker][["date", "reputation_score"]].set_index("date")

    merged = market.join(rep, how="inner").dropna()

    if freq:
        merged = merged.resample(freq).agg({
            "daily_return": "sum",
            "reputation_score": "mean",
        }).dropna()

    x = merged["reputation_score"].values
    y = merged["daily_return"].values

    # Standardize
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)

    n = len(x)
    scales = np.arange(2, min(n // 2, 80))

    # Cross-wavelet power
    xwt_power, _ = cross_wavelet_power(x, y, scales)

    # Wavelet coherence
    coherence, _ = wavelet_coherence_approx(x, y, scales)

    # Scale-averaged coherence (by frequency band)
    # Short-term: scales 2-8 (~1-2 months at weekly freq)
    # Medium-term: scales 8-26 (~2-6 months)
    # Long-term: scales 26+ (~6+ months)
    bands = {
        "short_term": (2, 8),
        "medium_term": (8, 26),
        "long_term": (26, len(scales)),
    }

    band_coherence = {}
    for band_name, (s_low, s_high) in bands.items():
        mask = (scales >= s_low) & (scales < s_high)
        if mask.any():
            band_coherence[band_name] = round(
                coherence[mask, :].mean(), 3
            )
        else:
            band_coherence[band_name] = None

    return {
        "ticker": ticker,
        "dates": merged.index,
        "cross_wavelet_power": xwt_power,
        "coherence": coherence,
        "scales": scales,
        "band_coherence": band_coherence,
        "n_obs": n,
        "freq": freq,
    }


def pairwise_company_wavelet(
    scores_df: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """
    Compute pairwise wavelet coherence between all company pairs.

    This mirrors the country-pair wavelet analysis from the original research,
    revealing which companies have the strongest reputation co-movement.
    """
    pivot = scores_df.pivot_table(
        index="date", columns="ticker", values="reputation_score"
    )
    if freq:
        pivot = pivot.resample(freq).mean()
    pivot = pivot.dropna()

    tickers = pivot.columns.tolist()
    results = []

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i >= j:
                continue

            x = pivot[t1].values
            y = pivot[t2].values

            x = (x - x.mean()) / (x.std() + 1e-12)
            y = (y - y.mean()) / (y.std() + 1e-12)

            scales = np.arange(2, min(len(x) // 2, 80))
            coherence, _ = wavelet_coherence_approx(x, y, scales)

            results.append({
                "company_1": t1,
                "company_2": t2,
                "avg_coherence": round(coherence.mean(), 3),
                "max_coherence": round(coherence.max(), 3),
                "short_term_coh": round(
                    coherence[scales < 8, :].mean(), 3
                ) if (scales < 8).any() else None,
                "medium_term_coh": round(
                    coherence[(scales >= 8) & (scales < 26), :].mean(), 3
                ) if ((scales >= 8) & (scales < 26)).any() else None,
                "long_term_coh": round(
                    coherence[scales >= 26, :].mean(), 3
                ) if (scales >= 26).any() else None,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/reputation-capital-model/data")
    sys.path.insert(0, "/home/claude/reputation-capital-model/models")
    from sample_data import generate_dataset
    from pca_reputation_score import extract_reputation_scores

    df = generate_dataset()
    scores_df, _ = extract_reputation_scores(df)

    print("=== Wavelet Analysis ===\n")

    # Single company: reputation vs stock
    for ticker in ["AAPL", "META"]:
        result = reputation_stock_wavelet(df, scores_df, ticker)
        print(f"{ticker} - Reputation ↔ Stock Return Coherence:")
        for band, val in result["band_coherence"].items():
            print(f"  {band}: {val}")
        print()

    # Pairwise company coherence
    print("=== Pairwise Company Reputation Coherence ===\n")
    pairs = pairwise_company_wavelet(scores_df)
    print(pairs.sort_values("avg_coherence", ascending=False).to_string(index=False))
