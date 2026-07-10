"""
Real Data Pipeline — Orchestrator
===================================
Ties together market data + news sentiment, runs the full analysis,
and exports JSON for the frontend dashboard.

Usage:
    python pipeline.py                    # GDELT sentiment only (free)
    python pipeline.py --api-key sk-...   # + Anthropic AI sentiment
    python pipeline.py --days 180         # custom lookback

Output:
    outputs/dashboard_data.json   ← frontend reads this
    outputs/market_daily.csv      ← raw daily market data
    outputs/news_daily.csv        ← raw daily news sentiment
    outputs/reputation_scores.csv ← PCA reputation scores
    outputs/spillover.json        ← Diebold-Yilmaz results
"""

import sys
import os
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from data.market_data import fetch_stock_data, compute_market_features, resample_to_weekly
from data.news_data import build_daily_sentiment

import pandas as pd
import numpy as np


DEFAULT_TICKERS = {
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
}

PROFILES = {
    "AAPL": {"domain": "apple.com", "name": "Apple Inc.", "industry": "Consumer Electronics", "keywords": ["hardware", "iOS", "services", "privacy", "luxury brand"]},
    "GOOGL": {"domain": "google.com", "name": "Alphabet Inc.", "industry": "Digital Advertising / AI", "keywords": ["search", "cloud", "Android", "AI/ML", "antitrust"]},
    "META": {"domain": "meta.com", "name": "Meta Platforms Inc.", "industry": "Social Media", "keywords": ["social platforms", "advertising", "metaverse", "VR/AR", "content moderation"]},
    "MSFT": {"domain": "microsoft.com", "name": "Microsoft Corp.", "industry": "Enterprise Software / Cloud", "keywords": ["Azure", "Office 365", "OpenAI", "gaming", "enterprise"]},
    "AMZN": {"domain": "amazon.com", "name": "Amazon.com Inc.", "industry": "E-Commerce / Cloud", "keywords": ["AWS", "e-commerce", "logistics", "Prime", "Alexa"]},
}


def build_reputation_index(
    market_weekly: pd.DataFrame,
    news_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine market data + news sentiment into a unified weekly reputation index.

    The index blends:
    - Stock price momentum (normalized)
    - News sentiment (rolling average)
    - Volume anomaly (news spike detection)

    Weighted: 40% price, 40% sentiment, 20% volume signal
    Then normalized to oscillate around 100.
    """
    results = []

    for ticker in market_weekly["ticker"].unique():
        mw = market_weekly[market_weekly["ticker"] == ticker].copy()
        mw = mw.set_index("date")

        # Get news data for this ticker
        nd = news_daily[news_daily["ticker"] == ticker].copy() if not news_daily.empty else pd.DataFrame()

        if not nd.empty:
            nd = nd.set_index("date")
            # Resample news to weekly
            nw = nd.resample("W").agg({
                "sentiment": "mean",
                "news_count": "sum",
                "volume_zscore": "mean",
            })
            # Join
            combined = mw.join(nw, how="left")
        else:
            combined = mw.copy()
            combined["sentiment"] = 0
            combined["news_count"] = 0
            combined["volume_zscore"] = 0

        # Fill missing sentiment with 0 (neutral)
        combined["sentiment"] = combined["sentiment"].fillna(0)
        combined["volume_zscore"] = combined["volume_zscore"].fillna(0)

        # Normalize each component to z-scores
        def zscore(s):
            m, sd = s.rolling(52, min_periods=10).mean(), s.rolling(52, min_periods=10).std()
            return ((s - m) / sd.replace(0, np.nan)).fillna(0)

        z_price = zscore(combined["weekly_return"].rolling(4).mean().fillna(0))
        z_sent = zscore(combined["sentiment"])
        z_vol = -combined["volume_zscore"].clip(-3, 3) / 3  # High volume spikes = negative signal

        # Composite score: weighted average of z-scores
        composite = 0.4 * z_price + 0.4 * z_sent + 0.2 * z_vol

        # Convert to index around 100
        # Use cumulative sum approach: each week's score shifts the index
        index_vals = 100 + composite.cumsum() * 0.5  # damped

        # Clamp to reasonable range
        index_vals = index_vals.clip(85, 115)

        combined["reputation_index"] = index_vals.round(2)
        combined["ma30"] = combined["reputation_index"].rolling(30, min_periods=5).mean().round(2)
        combined["dev_from_ma"] = ((combined["reputation_index"] / combined["ma30"]) - 1).round(4) * 100
        combined["weekly_chg"] = combined["reputation_index"].pct_change().round(4) * 100
        combined["monthly_chg"] = combined["reputation_index"].pct_change(4).round(4) * 100
        combined["vol_20w"] = combined["weekly_chg"].rolling(20).std().round(3)
        combined["ticker"] = ticker

        results.append(combined.reset_index())

    return pd.concat(results, ignore_index=True)


def compute_spillover_from_real(rep_index: pd.DataFrame) -> dict:
    """
    Run Diebold-Yilmaz spillover on real reputation index data.
    """
    try:
        from models.spillover_analysis import compute_spillover_table

        # Build scores_df format expected by spillover module
        scores_df = rep_index[["date", "ticker", "reputation_index"]].rename(
            columns={"reputation_index": "reputation_score"}
        )

        result = compute_spillover_table(scores_df, forecast_horizon=10, var_lags=2, freq="W")

        if "error" not in result:
            return {
                "total_spillover_index": result["total_spillover_index"],
                "net": result["net"],
                "transmitted": result["transmitted"],
                "received": result["received"],
            }
    except Exception as e:
        print(f"  ⚠ Spillover computation failed: {e}")

    return {"total_spillover_index": 0, "net": {}, "transmitted": {}, "received": {}}


def export_frontend_json(
    rep_index: pd.DataFrame,
    spillover: dict,
    output_path: str = "outputs/dashboard_data.json",
):
    """
    Export all data as a single JSON file for the frontend to consume.
    """
    data = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "tickers": {},
        "spillover": spillover,
        "profiles": PROFILES,
    }

    for ticker in rep_index["ticker"].unique():
        td = rep_index[rep_index["ticker"] == ticker].sort_values("date")
        latest = td.iloc[-1] if len(td) > 0 else None

        data["tickers"][ticker] = {
            "name": DEFAULT_TICKERS.get(ticker, ticker),
            "latest": {
                "index": float(latest["reputation_index"]) if latest is not None else 100,
                "weekly_chg": float(latest["weekly_chg"]) if latest is not None and pd.notna(latest["weekly_chg"]) else 0,
                "monthly_chg": float(latest["monthly_chg"]) if latest is not None and pd.notna(latest["monthly_chg"]) else 0,
                "dev_from_ma": float(latest["dev_from_ma"]) if latest is not None and pd.notna(latest["dev_from_ma"]) else 0,
            },
            "history": [
                {
                    "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                    "index": float(row["reputation_index"]) if pd.notna(row["reputation_index"]) else 100,
                    "ma30": float(row["ma30"]) if pd.notna(row.get("ma30", None)) else None,
                    "dev_from_ma": float(row["dev_from_ma"]) if pd.notna(row.get("dev_from_ma", None)) else 0,
                    "weekly_chg": float(row["weekly_chg"]) if pd.notna(row.get("weekly_chg", None)) else 0,
                    "vol_20w": float(row["vol_20w"]) if pd.notna(row.get("vol_20w", None)) else None,
                }
                for _, row in td.iterrows()
            ],
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return data


def run(days_back: int = 365, api_key: str = None):
    """Full pipeline execution."""
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("REPUVIEW — REAL DATA PIPELINE")
    print("=" * 60)

    # ── Step 1: Market data ──
    print(f"\n[1/5] Fetching market data (yfinance)...")
    market_df = fetch_stock_data(start="2020-01-01")
    if market_df.empty:
        print("  ✗ No market data. Check internet connection.")
        return
    market_df = compute_market_features(market_df)
    market_weekly = resample_to_weekly(market_df)
    print(f"  ✓ {len(market_df):,} daily / {len(market_weekly):,} weekly rows")

    # ── Step 2: News sentiment ──
    print(f"\n[2/5] Fetching news sentiment (GDELT)...")
    all_news = []
    for ticker, company in DEFAULT_TICKERS.items():
        daily = build_daily_sentiment(ticker, company, days_back=min(days_back, 365), api_key=api_key)
        if not daily.empty:
            all_news.append(daily)

    news_df = pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()
    print(f"  ✓ {len(news_df):,} daily news rows across {len(all_news)} tickers")

    # ── Step 3: Build reputation index ──
    print(f"\n[3/5] Building reputation index...")
    rep_index = build_reputation_index(market_weekly, news_df)
    print(f"  ✓ {len(rep_index):,} weekly reputation index rows")

    for ticker in DEFAULT_TICKERS:
        latest = rep_index[rep_index["ticker"] == ticker].iloc[-1]
        chg = latest.get("weekly_chg", 0)
        chg = chg if pd.notna(chg) else 0
        print(f"    {ticker}: {latest['reputation_index']:.2f} ({chg:+.2f}% 1w)")

    # ── Step 4: Spillover analysis ──
    print(f"\n[4/5] Computing spillover index...")
    spillover = compute_spillover_from_real(rep_index)
    print(f"  ✓ Total Spillover Index: {spillover['total_spillover_index']}%")
    if spillover["net"]:
        for t, v in sorted(spillover["net"].items(), key=lambda x: -x[1]):
            print(f"    {t}: {v:+.1f}% ({'TRANSMITTER' if v > 0 else 'RECEIVER'})")

    # ── Step 5: Export ──
    print(f"\n[5/5] Exporting data...")
    market_df.to_csv("outputs/market_daily.csv", index=False)
    news_df.to_csv("outputs/news_daily.csv", index=False) if not news_df.empty else None
    rep_index.to_csv("outputs/reputation_index.csv", index=False)
    export_frontend_json(rep_index, spillover)

    with open("outputs/spillover.json", "w") as f:
        json.dump(spillover, f, indent=2)

    print(f"\n{'=' * 60}")
    print("✓ Pipeline complete. Outputs saved to outputs/")
    print("  → dashboard_data.json (frontend consumption)")
    print("  → market_daily.csv")
    print("  → news_daily.csv")
    print("  → reputation_index.csv")
    print("  → spillover.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RepuView — Real Data Pipeline")
    parser.add_argument("--days", type=int, default=365, help="Days of news to fetch (default: 365)")
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API key for AI sentiment (optional)")
    args = parser.parse_args()

    run(days_back=args.days, api_key=args.api_key)
