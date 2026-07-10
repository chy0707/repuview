"""
Module 1: Reputation Score Extraction via PCA
==============================================
Mirrors the PCA approach from original macro-financial research:
- Original research: dozens of macro indicators → 3 stress factors (financial, non-financial, consumer)
- This tool: news sentiment + volume + volatility + social signals → 1 composite reputation score

The PCA extracts the dominant latent factor driving reputation-related signals,
producing a single interpretable index per company per time period.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple


def prepare_pca_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Prepare multi-signal feature matrix for a single company.

    Features used:
    - news_sentiment: aggregated daily news sentiment from NLP
    - news_volume: daily article count (proxy for media attention)
    - realized_vol_20d: 20-day realized stock volatility
    - sentiment_ma7: smoothed sentiment trend
    - sentiment_return_interaction: sentiment × return cross-signal
    """
    comp = df[df["ticker"] == ticker].copy().sort_values("date")
    comp = comp.set_index("date")

    # Engineered features
    comp["sentiment_return_interaction"] = (
        comp["news_sentiment"] * comp["daily_return"]
    )
    comp["news_volume_zscore"] = (
        (comp["news_volume"] - comp["news_volume"].rolling(60).mean())
        / comp["news_volume"].rolling(60).std()
    )
    comp["sentiment_momentum"] = (
        comp["news_sentiment"] - comp["news_sentiment"].shift(5)
    )

    feature_cols = [
        "news_sentiment",
        "news_volume_zscore",
        "realized_vol_20d",
        "sentiment_ma7",
        "sentiment_return_interaction",
        "sentiment_momentum",
    ]

    features = comp[feature_cols].dropna()
    return features


def extract_reputation_scores(
    df: pd.DataFrame,
    n_components: int = 1,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply PCA to extract composite reputation scores for all companies.

    Returns:
    - scores_df: DataFrame with date, ticker, reputation_score
    - diagnostics: dict with explained variance, loadings, etc.
    """
    all_scores = []
    diagnostics = {}

    for ticker in df["ticker"].unique():
        features = prepare_pca_features(df, ticker)

        if len(features) < 30:
            continue

        # Z-score normalization (standard preprocessing)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # PCA
        pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
        scores = pca.fit_transform(X_scaled)

        # Flip sign if needed: higher sentiment should → higher reputation score
        # Check correlation with raw sentiment to decide sign
        sent_corr = np.corrcoef(
            scores[:, 0], features["news_sentiment"].values
        )[0, 1]
        if sent_corr < 0:
            scores[:, 0] *= -1
            pca.components_[0] *= -1

        # Store results
        score_df = pd.DataFrame({
            "date": features.index,
            "ticker": ticker,
            "reputation_score": scores[:, 0],
        })
        all_scores.append(score_df)

        # Diagnostics
        diagnostics[ticker] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(
                pca.explained_variance_ratio_
            ).tolist(),
            "loadings": pd.DataFrame(
                pca.components_.T,
                index=features.columns,
                columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            ),
            "feature_names": features.columns.tolist(),
            "n_observations": len(features),
        }

    scores_df = pd.concat(all_scores, ignore_index=True)

    # Normalize reputation scores to [0, 100] scale for interpretability
    for ticker in scores_df["ticker"].unique():
        mask = scores_df["ticker"] == ticker
        raw = scores_df.loc[mask, "reputation_score"]
        # Min-max to 0-100
        scores_df.loc[mask, "reputation_score_normalized"] = (
            (raw - raw.min()) / (raw.max() - raw.min()) * 100
        )

    return scores_df, diagnostics


def compute_reputation_regime(
    scores_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Classify reputation into regimes: Strong / Normal / At Risk / Crisis.
    Uses rolling z-score thresholds, similar to stress level classification in macro-financial research.
    """
    result = scores_df.copy()

    for ticker in result["ticker"].unique():
        mask = result["ticker"] == ticker
        score = result.loc[mask, "reputation_score"].copy()

        rolling_mean = score.rolling(window).mean()
        rolling_std = score.rolling(window).std()
        z = (score - rolling_mean) / rolling_std

        conditions = [
            z >= 1.0,
            (z >= -0.5) & (z < 1.0),
            (z >= -1.5) & (z < -0.5),
            z < -1.5,
        ]
        labels = ["Strong", "Normal", "At Risk", "Crisis"]
        result.loc[mask, "reputation_regime"] = np.select(
            conditions, labels, default="Normal"
        )
        result.loc[mask, "reputation_zscore"] = z.values

    return result


if __name__ == "__main__":
    from sample_data import generate_dataset

    df = generate_dataset()
    scores_df, diagnostics = extract_reputation_scores(df)
    scores_df = compute_reputation_regime(scores_df)

    print("=== Reputation Score Extraction ===\n")
    for ticker, diag in diagnostics.items():
        print(f"{ticker}:")
        print(f"  Variance explained by PC1: {diag['explained_variance_ratio'][0]:.1%}")
        print(f"  Observations: {diag['n_observations']}")
        print(f"  Top loadings:")
        loadings = diag['loadings']['PC1'].abs().sort_values(ascending=False)
        for feat, val in loadings.head(3).items():
            sign = "+" if diag['loadings'].loc[feat, 'PC1'] > 0 else "-"
            print(f"    {sign}{val:.3f}  {feat}")
        print()

    print("\nRegime distribution:")
    print(scores_df["reputation_regime"].value_counts())
    print(f"\nSample output:\n{scores_df.tail(10)}")
