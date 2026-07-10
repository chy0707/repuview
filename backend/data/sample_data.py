"""
Sample Data Generator for Reputation Capital Model
===================================================
Generates realistic synthetic data mimicking:
- Daily news sentiment scores (from FinBERT-style NLP pipeline)
- Stock price / returns / volatility data (from Yahoo Finance)
- Social media buzz volume
- ESG controversy flags

In production, these would come from:
- GDELT / NewsAPI / MediaCloud → FinBERT sentiment
- Yahoo Finance / Alpha Vantage → market data
- Reddit API / Twitter API → social sentiment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Big Tech universe ────────────────────────────────────────────────────
COMPANIES = {
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
}

# ── Reputation event calendar (real events, synthetic magnitudes) ─────────
# These anchor points create realistic sentiment shocks
REPUTATION_EVENTS = [
    # (date, ticker, sentiment_shock, description)
    ("2020-03-15", None, -0.4, "COVID-19 market crash"),
    ("2020-07-29", "GOOGL", -0.3, "Antitrust hearing - Big Tech CEOs"),
    ("2020-07-29", "AAPL", -0.25, "Antitrust hearing - Big Tech CEOs"),
    ("2020-07-29", "META", -0.3, "Antitrust hearing - Big Tech CEOs"),
    ("2020-07-29", "AMZN", -0.3, "Antitrust hearing - Big Tech CEOs"),
    ("2021-02-03", "AMZN", -0.15, "Bezos steps down as CEO"),
    ("2021-10-04", "META", -0.5, "Facebook whistleblower / Frances Haugen"),
    ("2021-10-28", "META", -0.2, "Meta rebrand announcement"),
    ("2022-01-20", "MSFT", 0.3, "Activision acquisition announced"),
    ("2022-04-25", "META", -0.35, "Meta Q1 earnings miss / metaverse spend"),
    ("2022-11-09", "META", -0.4, "Meta mass layoffs 11,000"),
    ("2022-11-30", "MSFT", 0.35, "ChatGPT launch / Microsoft-OpenAI"),
    ("2023-01-20", "GOOGL", -0.3, "Google layoffs 12,000"),
    ("2023-01-20", "MSFT", -0.2, "Microsoft layoffs 10,000"),
    ("2023-02-07", "GOOGL", -0.4, "Bard demo error / stock drops 8%"),
    ("2023-05-10", "GOOGL", 0.25, "Google I/O AI announcements"),
    ("2023-06-05", "AAPL", 0.35, "Apple Vision Pro announcement"),
    ("2023-09-12", "AAPL", 0.15, "iPhone 15 launch"),
    ("2023-10-30", "META", 0.3, "Meta strong Q3 / 'Year of Efficiency'"),
    ("2024-01-25", "MSFT", 0.3, "Microsoft passes Apple as most valuable"),
    ("2024-04-09", "META", -0.2, "Meta AI content moderation controversy"),
    ("2024-06-10", "AAPL", 0.25, "Apple Intelligence announcement"),
    ("2024-09-09", "GOOGL", -0.3, "DOJ antitrust ruling against Google"),
    ("2024-10-29", "META", -0.15, "Meta AI spending concerns"),
    ("2025-01-15", None, -0.15, "DeepSeek disruption / AI competition fears"),
    ("2025-03-10", "AAPL", -0.2, "EU DMA compliance fines"),
]


def _generate_base_returns(n_days: int, seed: int = 42) -> pd.DataFrame:
    """Generate correlated daily returns for Big Tech stocks."""
    rng = np.random.default_rng(seed)

    # Correlation structure: Big Tech stocks are moderately correlated
    corr = np.array([
        [1.00, 0.65, 0.55, 0.70, 0.60],  # AAPL
        [0.65, 1.00, 0.60, 0.68, 0.58],  # GOOGL
        [0.55, 0.60, 1.00, 0.55, 0.50],  # META
        [0.70, 0.68, 0.55, 1.00, 0.62],  # MSFT
        [0.60, 0.58, 0.50, 0.62, 1.00],  # AMZN
    ])
    vols = np.array([0.015, 0.018, 0.022, 0.014, 0.020])  # daily vol
    cov = np.outer(vols, vols) * corr

    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_days, 5))
    returns = z @ L.T + np.array([0.0004, 0.0003, 0.0002, 0.0004, 0.0003])

    return pd.DataFrame(returns, columns=list(COMPANIES.keys()))


def _generate_sentiment(n_days: int, dates: pd.DatetimeIndex,
                        seed: int = 42) -> dict:
    """Generate daily news sentiment scores per company."""
    rng = np.random.default_rng(seed)
    sentiment = {}

    for ticker in COMPANIES:
        # Base sentiment: AR(1) process with mean ~0.05 (slightly positive)
        s = np.zeros(n_days)
        s[0] = 0.05
        for t in range(1, n_days):
            s[t] = 0.02 + 0.85 * s[t - 1] + rng.normal(0, 0.08)

        # Inject reputation events
        for evt_date, evt_ticker, shock, _ in REPUTATION_EVENTS:
            evt_dt = pd.Timestamp(evt_date)
            if evt_dt in dates:
                idx = dates.get_loc(evt_dt)
                if evt_ticker is None or evt_ticker == ticker:
                    # Shock decays over ~10 days
                    decay = np.exp(-np.arange(min(15, n_days - idx)) / 5)
                    end = min(idx + 15, n_days)
                    s[idx:end] += shock * decay[: end - idx]

                    # Cross-contagion: other Big Tech gets partial shock
                    if evt_ticker is not None and evt_ticker != ticker:
                        contagion = shock * 0.25
                        s[idx:end] += contagion * decay[: end - idx]

        # Clip to [-1, 1]
        s = np.clip(s, -1, 1)
        sentiment[ticker] = s

    return sentiment


def _generate_news_volume(n_days: int, dates: pd.DatetimeIndex,
                          seed: int = 42) -> dict:
    """Generate daily news article counts per company."""
    rng = np.random.default_rng(seed)
    volume = {}

    for ticker in COMPANIES:
        # Base volume: Poisson with slight uptrend
        base = rng.poisson(lam=25, size=n_days).astype(float)
        trend = np.linspace(0, 10, n_days)
        v = base + trend

        # Volume spikes during events
        for evt_date, evt_ticker, shock, _ in REPUTATION_EVENTS:
            evt_dt = pd.Timestamp(evt_date)
            if evt_dt in dates:
                idx = dates.get_loc(evt_dt)
                if evt_ticker is None or evt_ticker == ticker:
                    spike = abs(shock) * 150
                    decay = np.exp(-np.arange(min(7, n_days - idx)) / 2)
                    end = min(idx + 7, n_days)
                    v[idx:end] += spike * decay[: end - idx]

        volume[ticker] = v

    return volume


def generate_dataset(
    start_date: str = "2020-01-01",
    end_date: str = "2025-03-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate complete dataset for Reputation Capital analysis.

    Returns a MultiIndex DataFrame with columns:
    - close: simulated closing price
    - daily_return: daily log return
    - realized_vol_20d: 20-day rolling realized volatility
    - news_sentiment: daily aggregated news sentiment [-1, 1]
    - news_volume: daily news article count
    - sentiment_ma7: 7-day moving average sentiment
    - sentiment_spike: binary flag for sentiment > 2σ move
    """
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    returns_df = _generate_base_returns(n_days, seed)
    sentiment = _generate_sentiment(n_days, dates, seed)
    news_vol = _generate_news_volume(n_days, dates, seed)

    # Inject sentiment → return feedback (reputation affects stock price)
    rng = np.random.default_rng(seed + 1)
    for ticker in COMPANIES:
        sent = sentiment[ticker]
        # Lagged sentiment impact on returns (1-day lag)
        impact = np.roll(sent, 1) * 0.003 + np.roll(sent, 2) * 0.001
        impact[0:2] = 0
        returns_df[ticker] += impact

    # Build prices from returns
    all_records = []
    for ticker in COMPANIES:
        rets = returns_df[ticker].values
        prices = 100 * np.exp(np.cumsum(rets))  # start at 100

        # 20-day rolling realized volatility
        ret_series = pd.Series(rets)
        rvol = ret_series.rolling(20).std() * np.sqrt(252)

        # Sentiment features
        sent = sentiment[ticker]
        sent_series = pd.Series(sent)
        sent_ma7 = sent_series.rolling(7).mean()
        sent_std = sent_series.rolling(60).std()
        sent_spike = (
            (sent_series - sent_series.rolling(60).mean()).abs()
            > 2 * sent_std
        ).astype(int)

        for i in range(n_days):
            all_records.append({
                "date": dates[i],
                "ticker": ticker,
                "company": COMPANIES[ticker],
                "close": round(prices[i], 2),
                "daily_return": round(rets[i], 6),
                "realized_vol_20d": round(rvol.iloc[i], 4)
                if not np.isnan(rvol.iloc[i]) else None,
                "news_sentiment": round(sent[i], 4),
                "news_volume": int(news_vol[ticker][i]),
                "sentiment_ma7": round(sent_ma7.iloc[i], 4)
                if not np.isnan(sent_ma7.iloc[i]) else None,
                "sentiment_spike": int(sent_spike.iloc[i])
                if not np.isnan(sent_spike.iloc[i]) else 0,
            })

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])

    return df


def get_reputation_events() -> pd.DataFrame:
    """Return the event calendar as a DataFrame for overlay on charts."""
    records = []
    for evt_date, evt_ticker, shock, desc in REPUTATION_EVENTS:
        records.append({
            "date": pd.Timestamp(evt_date),
            "ticker": evt_ticker if evt_ticker else "ALL",
            "shock_magnitude": shock,
            "description": desc,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Generated {len(df):,} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Companies: {df['ticker'].unique().tolist()}")
    print(f"\nSample:\n{df.head(10)}")
    df.to_csv("/home/claude/reputation-capital-model/data/sample_dataset.csv",
              index=False)
    print("\nSaved to data/sample_dataset.csv")
