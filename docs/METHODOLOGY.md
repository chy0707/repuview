# Methodology: RepuView

## Research Foundation

This tool builds on methods developed during my master's research, which studied the international transmission of economic stress across six economies (US, France, Germany, Italy, Spain, Belgium). The research proved that financial stress in one country Granger-causes sentiment shifts in others, with behavioral mechanisms — loss aversion, anchoring, herding — shaping the transmission dynamics.

RepuView applies this same analytical framework at the corporate level: **can brand reputation signals predict financial outcomes, and how does reputation risk spread across an industry?**

---

## Analytical Framework

### 1. Principal Component Analysis (PCA) — Reputation Score Extraction

**Original application**: Compressed 30+ macroeconomic indicators (money supply, debt-to-GDP, unemployment, interest rates) into three latent stress factors: Financial, Non-Financial, and Consumer.

**Reputation application**: Compresses multiple reputation-related signals into a single composite index per company.

Input features:
- `news_sentiment`: Daily aggregated sentiment from NLP analysis
- `news_volume_zscore`: Media attention anomaly (spike detection)
- `realized_vol_20d`: 20-day stock realized volatility
- `sentiment_ma7`: Smoothed sentiment trend (7-day MA)
- `sentiment_return_interaction`: Cross-signal between sentiment and returns
- `sentiment_momentum`: 5-day sentiment change

The covariance matrix Σ is computed, eigenvalues solved via `det(Σ − λI) = 0`, and the first principal component (explaining the most variance) becomes the reputation score. Sign is corrected so that positive sentiment correlates with a higher score.

The score is then normalized to a 100-baseline index, similar to financial indices (S&P 500, VIX), making cross-company comparison intuitive.

---

### 2. Granger Causality — Predictive Relationship Testing

**Original application**: Tested whether US financial stress Granger-causes European sentiment. Result: US stress significantly predicts sentiment in all five European countries (p < 0.01).

**Reputation application**: Tests whether the reputation score Granger-causes:
- Stock abnormal returns
- Realized volatility changes

The test compares a restricted model (returns predicted only by own history) against an unrestricted model (returns predicted by own history + lagged reputation score). If the reputation score coefficients are jointly significant (F-test), we conclude that reputation has predictive power for financial outcomes.

This is critical for the tool's credibility: it provides statistical evidence that reputation monitoring has financial value, not just PR value.

---

### 3. Vector Autoregression (VAR) → Impulse Response Functions (IRF)

**Original application**: Traced how a one-unit shock in US financial stress impacts European sentiment over a 10-month horizon. Found persistent effects lasting 6-10 months.

**Reputation application**: The VAR system models `[reputation_score, daily_return, realized_volatility]` jointly, capturing feedback loops. The IRF then shows:
- How a sudden reputation shock (e.g., data breach) propagates through stock returns over subsequent weeks
- Whether the effect is temporary (mean-reverting) or persistent
- How long until the financial impact decays to zero

This directly informs the **decay model** used in the dashboard's impact forecasting.

---

### 4. Forecast Error Variance Decomposition (FEVD)

**Original application**: Found that >50% of Belgium's sentiment variance was attributable to foreign stress (primarily US and French).

**Reputation application**: Quantifies what percentage of a stock's return variance is explained by:
- Reputation factors vs. market-wide factors vs. idiosyncratic noise

A higher FEVD for reputation → returns means the company's stock is more reputation-sensitive. This helps PR teams prioritize: companies with high reputation-FEVD should invest more in proactive reputation management.

---

### 5. Diebold-Yilmaz Spillover Index — Cross-Company Contagion

**Original application**: Mapped cross-country stress transmission networks. The US, France, and Germany emerged as net transmitters; Italy, Spain, and Belgium as net receivers.

**Reputation application**: Measures how a reputation crisis at one company spills over to others in the same sector.

The total spillover index S is computed from the FEVD matrix:

```
S = (Σ_{i≠j} θ_ij / Σ_{i,j} θ_ij) × 100
```

Directional spillovers decompose this into:
- **Net transmitters**: Companies whose reputation events affect others (e.g., Meta's content moderation issues affecting all social media stocks)
- **Net receivers**: Companies whose stock is more reactive to peers' reputation events

Key finding from Big Tech analysis: Meta is the largest net transmitter (+6.9%), Google the largest net receiver (-6.9%).

---

### 6. Wavelet Analysis — Multi-Horizon Co-Movement

**Original application**: Cross-wavelet scalograms revealed that stress-sentiment co-movement is strongest at medium-to-long frequencies (quarterly to annual cycles), not short-term noise.

**Reputation application**: The Continuous Wavelet Transform (CWT) decomposes reputation-stock co-movement across time horizons:

- **Short-term** (1-2 months): Day-trading noise, event reactions
- **Medium-term** (2-6 months): Business-cycle patterns, earnings effects
- **Long-term** (6+ months): Structural reputation shifts, brand equity changes

Key insight: If coherence increases from short to long term, the reputation-stock link is structural and persistent — not just noise traders reacting to headlines. This validates long-term reputation investment.

---

### 7. Half-Life Decay Model — Impact Forecasting

**Unique to this tool** (not in original research). Inspired by the IRF findings showing exponential decay of shock effects.

Each news event's reputation impact decays over time:

```
residual(t) = peak_impact × e^(−t × ln(2) / half_life)
```

Half-lives by event duration class:
| Class | Half-life | Examples |
|-------|-----------|----------|
| Short-term | 7 days | Product launches, minor PR, exec quotes |
| Medium-term | 28 days | Earnings misses, layoffs, lawsuits filed |
| Long-term | 84 days | Regulatory rulings, structural pivots, data breaches |

Impact scale (calibrated conservatively):
| Score | Meaning | Examples |
|-------|---------|----------|
| ±5.0 | Existential | Fraud discovery, massive recall, monopoly breakup |
| ±3.0 | Major | Mass layoffs, landmark regulation, blockbuster product |
| ±1.5 | Notable | Executive change, lawsuit, quarterly miss/beat |
| ±0.5 | Minor | Routine update, small partnership, minor controversy |
| ±0.2 | Noise | Industry trend mention, analyst opinion |

The **12-week projection chart** aggregates all active events' residuals forward in time, showing when net impact converges to zero.

---

## Data Architecture

### Real-Time Pipeline

```
yfinance (free)           GDELT DOC API (free)         Anthropic Claude API
    │                          │                              │
    ▼                          ▼                              ▼
Stock prices              News articles                AI sentiment scoring
Returns, volatility       Titles, sources, tone        Reputation-specific
Volume anomalies          Publication dates             impact analysis
    │                          │                              │
    └──────────┬───────────────┘                              │
               ▼                                              │
    PCA Reputation Index                                      │
    (40% price + 40% sentiment + 20% volume)                  │
               │                                              │
               ▼                                              ▼
    Spillover Analysis ◄──────────────── SWOT + Recovery Plan
    (Diebold-Yilmaz)                    (3-horizon strategy)
               │                              │
               └──────────┬───────────────────┘
                          ▼
              dashboard_data.json → Frontend
```

### Data Sources

| Source | What | Cost | Auth |
|--------|------|------|------|
| **yfinance** | Daily OHLCV, returns, volatility | Free | None |
| **GDELT** | News articles (100+ languages, 2015-present) | Free | None |
| **Anthropic API** | AI sentiment + SWOT + recovery plans | Pay-per-use | API key |
| **Clearbit** | Company logos | Free tier | None |

---

## References

- Diebold, F.X. & Yilmaz, K. (2012). Better to give than to receive: Predictive directional measurement of volatility spillovers. *International Journal of Forecasting*.
- Baker, S.R., Bloom, N. & Davis, S.J. (2016). Measuring economic policy uncertainty. *Quarterly Journal of Economics*.
- Kahneman, D. & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*.
- Shiller, R.J. (2017). Narrative economics. *AEA Presidential Address*.
- Oet, M.V. et al. (2011). The financial stress index. *Federal Reserve Bank of Cleveland Working Paper*.
- Bikhchandani, S., Hirshleifer, D. & Welch, I. (1992). A theory of fads, fashion, custom, and cultural change as informational cascades. *Journal of Political Economy*.
- Tsay, R.S. (2010). *Analysis of financial time series* (3rd ed.). Wiley.
