"""
News & Sentiment Pipeline — GDELT + Anthropic API
===================================================
1. GDELT DOC API: free news search, no API key, covers 100+ languages
2. Anthropic Claude: FinBERT-grade sentiment scoring per article

GDELT returns news article metadata (title, URL, source, date, tone).
Claude adds nuanced reputation-specific sentiment analysis.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt_news(
    query: str,
    days_back: int = 90,
    max_records: int = 250,
    source_lang: str = "english",
) -> pd.DataFrame:
    """
    Search GDELT for news articles matching a company query.

    GDELT DOC API is free, no authentication required.
    Returns: title, url, source, date, domain, tone (GDELT's built-in sentiment).
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    params = {
        "query": f'"{query}" sourcelang:{source_lang}',
        "mode": "artlist",
        "maxrecords": str(max_records),
        "format": "json",
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
        "sort": "datedesc",
    }

    try:
        resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ⚠ GDELT API error: {e}")
        return pd.DataFrame()

    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame()

    records = []
    for art in articles:
        records.append({
            "date": pd.to_datetime(art.get("seendate", ""), format="%Y%m%dT%H%M%SZ", errors="coerce"),
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "source": art.get("domain", ""),
            "source_country": art.get("sourcecountry", ""),
            "language": art.get("language", ""),
            "gdelt_tone": art.get("tone", 0),  # GDELT's built-in tone score
        })

    df = pd.DataFrame(records)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=False)
    return df


def aggregate_daily_news_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate news articles to daily counts and average tone.
    """
    if df.empty:
        return pd.DataFrame()

    df["date_day"] = df["date"].dt.date

    daily = df.groupby("date_day").agg(
        news_count=("title", "count"),
        avg_gdelt_tone=("gdelt_tone", "mean"),
        tone_std=("gdelt_tone", "std"),
    ).reset_index()

    daily = daily.rename(columns={"date_day": "date"})
    daily["date"] = pd.to_datetime(daily["date"])

    # Z-score of news volume (spike detection)
    vol_mean = daily["news_count"].rolling(30, min_periods=5).mean()
    vol_std = daily["news_count"].rolling(30, min_periods=5).std()
    daily["volume_zscore"] = (daily["news_count"] - vol_mean) / vol_std.replace(0, np.nan)

    # Normalize GDELT tone to [-1, 1] range
    # GDELT tone ranges roughly -10 to +10
    daily["sentiment_normalized"] = daily["avg_gdelt_tone"].clip(-10, 10) / 10

    return daily


def batch_sentiment_anthropic(
    articles: List[Dict],
    company: str,
    api_key: str,
    batch_size: int = 10,
) -> List[Dict]:
    """
    Score article headlines using Anthropic Claude for reputation-specific sentiment.
    More nuanced than GDELT's generic tone score.

    Falls back to GDELT tone if API call fails.
    """
    results = []
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        headlines = "\n".join(
            f"{j+1}. \"{a['title']}\"" for j, a in enumerate(batch)
        )

        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 500,
            "messages": [{
                "role": "user",
                "content": (
                    f"Score these {company} headlines for REPUTATION impact. "
                    f"Return ONLY a JSON array of numbers from -1.0 (very negative) to +1.0 (very positive). "
                    f"0 = neutral/irrelevant. Be conservative.\n\n{headlines}"
                ),
            }],
        }

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30,
            )
            data = resp.json()
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            scores = json.loads(text.strip().strip("`").replace("json", ""))
            if isinstance(scores, list) and len(scores) == len(batch):
                for j, a in enumerate(batch):
                    results.append({**a, "ai_sentiment": round(scores[j], 3)})
                continue
        except Exception:
            pass

        # Fallback: use GDELT tone
        for a in batch:
            results.append({**a, "ai_sentiment": round(a.get("gdelt_tone", 0) / 10, 3)})

        time.sleep(0.5)  # Rate limit courtesy

    return results


def build_daily_sentiment(
    ticker: str,
    company: str,
    days_back: int = 365,
    api_key: str = None,
) -> pd.DataFrame:
    """
    Full pipeline: GDELT search → daily aggregation → optional AI sentiment.

    Returns daily DataFrame with: date, news_count, sentiment, volume_zscore
    """
    print(f"  [{ticker}] Fetching GDELT news for '{company}'...")
    raw = fetch_gdelt_news(company, days_back=days_back)

    if raw.empty:
        print(f"  [{ticker}] No articles found")
        return pd.DataFrame()

    print(f"  [{ticker}] {len(raw)} articles found")

    # If API key provided, run AI sentiment on a sample
    if api_key and len(raw) > 0:
        print(f"  [{ticker}] Running AI sentiment scoring...")
        sample = raw.head(100).to_dict("records")
        scored = batch_sentiment_anthropic(sample, company, api_key)
        raw_scored = pd.DataFrame(scored)
        # Merge AI scores back by matching title
        raw = raw.merge(
            raw_scored[["title", "ai_sentiment"]].drop_duplicates("title"),
            on="title", how="left",
        )
    else:
        raw["ai_sentiment"] = raw["gdelt_tone"].clip(-10, 10) / 10

    # Aggregate to daily
    raw["date_day"] = raw["date"].dt.date
    daily = raw.groupby("date_day").agg(
        news_count=("title", "count"),
        sentiment=("ai_sentiment", "mean"),
        sentiment_std=("ai_sentiment", "std"),
        gdelt_tone=("gdelt_tone", "mean"),
    ).reset_index()

    daily = daily.rename(columns={"date_day": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ticker"] = ticker

    # Volume z-score
    vol_mean = daily["news_count"].rolling(30, min_periods=5).mean()
    vol_std = daily["news_count"].rolling(30, min_periods=5).std()
    daily["volume_zscore"] = (daily["news_count"] - vol_mean) / vol_std.replace(0, np.nan)

    return daily.sort_values("date")


if __name__ == "__main__":
    print("=== News & Sentiment Pipeline ===\n")

    # Test GDELT (no API key needed)
    for ticker, company in [("META", "Meta Platforms"), ("AAPL", "Apple")]:
        daily = build_daily_sentiment(ticker, company, days_back=90)
        if not daily.empty:
            print(f"  ✓ {ticker}: {len(daily)} days of news data")
            print(f"    Avg sentiment: {daily['sentiment'].mean():.3f}")
            print(f"    Avg daily articles: {daily['news_count'].mean():.1f}")
            print()

    print("Done. Pass api_key to build_daily_sentiment() for AI-enhanced scoring.")
