"""
Market Data Pipeline — Real Data via yfinance
==============================================
Pulls daily stock data, computes:
- Daily/weekly returns
- 20-day and 20-week realized volatility
- Rolling betas vs market (SPY)

No API key needed. Free and unlimited.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


DEFAULT_TICKERS = {
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
}


def fetch_stock_data(
    tickers: dict = None,
    start: str = "2020-01-01",
    end: str = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for all tickers.
    Returns a clean DataFrame with date, ticker, close, volume, daily_return.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    all_data = []

    for ticker, name in tickers.items():
        print(f"  Fetching {ticker} ({name})...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"    ⚠ No data for {ticker}")
                continue

            df = df.reset_index()
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

            df = df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            df["ticker"] = ticker
            df["company"] = name
            df["daily_return"] = df["close"].pct_change()

            all_data.append(df[["date", "ticker", "company", "close", "volume", "daily_return"]])
        except Exception as e:
            print(f"    ✗ Error fetching {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived market features per ticker:
    - realized_vol_20d: 20-day rolling realized volatility (annualized)
    - realized_vol_60d: 60-day rolling realized volatility
    - volume_zscore: z-score of volume vs 60-day rolling mean
    - drawdown: current drawdown from rolling 252-day high
    """
    result = []

    for ticker in df["ticker"].unique():
        td = df[df["ticker"] == ticker].copy().sort_values("date")

        # Realized volatility
        td["realized_vol_20d"] = td["daily_return"].rolling(20).std() * np.sqrt(252)
        td["realized_vol_60d"] = td["daily_return"].rolling(60).std() * np.sqrt(252)

        # Volume anomaly
        vol_mean = td["volume"].rolling(60).mean()
        vol_std = td["volume"].rolling(60).std()
        td["volume_zscore"] = (td["volume"] - vol_mean) / vol_std.replace(0, np.nan)

        # Drawdown from 252-day high
        rolling_max = td["close"].rolling(252, min_periods=1).max()
        td["drawdown"] = (td["close"] / rolling_max) - 1

        result.append(td)

    return pd.concat(result, ignore_index=True)


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily data to weekly for the frontend index display.
    Returns: date, ticker, close (last), weekly_return, realized_vol_20w
    """
    result = []

    for ticker in df["ticker"].unique():
        td = df[df["ticker"] == ticker].copy()
        td = td.set_index("date")

        weekly = td.resample("W").agg({
            "close": "last",
            "volume": "sum",
            "daily_return": lambda x: (1 + x).prod() - 1,  # compounded weekly return
            "company": "first",
        }).dropna(subset=["close"])

        weekly = weekly.rename(columns={"daily_return": "weekly_return"})
        weekly["ticker"] = ticker

        # 20-week rolling vol
        weekly["vol_20w"] = weekly["weekly_return"].rolling(20).std()

        # Normalize to index (100 = start)
        weekly["index"] = (weekly["close"] / weekly["close"].iloc[0]) * 100

        # 30-week moving average of index
        weekly["ma30"] = weekly["index"].rolling(30).mean()

        # % deviation from MA
        weekly["dev_from_ma"] = ((weekly["index"] / weekly["ma30"]) - 1) * 100

        weekly = weekly.reset_index()
        result.append(weekly)

    return pd.concat(result, ignore_index=True)


if __name__ == "__main__":
    print("=== Fetching Real Market Data ===\n")
    df = fetch_stock_data()
    print(f"\n✓ {len(df):,} daily rows fetched")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    df = compute_market_features(df)
    print(f"✓ Market features computed")

    weekly = resample_to_weekly(df)
    print(f"✓ {len(weekly):,} weekly rows generated")

    # Save
    df.to_csv("market_daily.csv", index=False)
    weekly.to_csv("market_weekly.csv", index=False)
    print(f"\n✓ Saved to market_daily.csv and market_weekly.csv")
