# RepuView — Product Requirements Document

**Version**: 1.0
**Author**: Haoyu Cheng
**Last Updated**: June 2026
**Status**: V1 Live

---

## 1. Problem Statement

Corporate reputation directly impacts market capitalization, but the tools available to measure it are either qualitative (PR team gut feel), lagging (post-crisis surveys), or siloed (social listening without financial linkage).

Three specific gaps exist in the market:

**No predictive quantification.** Reputation is discussed in boardrooms as "strong" or "at risk" — subjective assessments with no mathematical rigor. Financial risk has VaR, beta, Sharpe ratios. Reputation has nothing comparable.

**No cross-company contagion mapping.** When Meta faces a content moderation scandal, all social media stocks drop. When Boeing has a safety crisis, Airbus benefits. These reputation contagion dynamics are real and measurable, but no tool tracks them systematically.

**No decay-aware impact modeling.** A data breach 3 weeks ago has less reputational impact today than it did on day one, but existing sentiment tools treat every historical event equally. The half-life of reputation events varies dramatically by type, and this temporal dimension is ignored.

---

## 2. Target Users

### Primary: Corporate Communications & PR Teams
- Need: Real-time reputation monitoring with actionable recovery plans
- Pain: Cobble together Google Alerts, Brandwatch, and manual media analysis
- Value: One dashboard replacing 3-4 tools, with AI-generated strategy recommendations

### Secondary: Investor Relations & C-Suite
- Need: Quantified reputation risk exposure for board reporting and investor calls
- Pain: Can't answer "what's our reputation risk in basis points?"
- Value: Index-based scoring (100 = baseline) that speaks the language of finance

### Tertiary: PR Agency Strategists (e.g., Burson, Edelman, FTI)
- Need: Data-backed client deliverables for reputation advisory engagements
- Pain: Manual research for every pitch; insights are opinion-based, not evidence-based
- Value: Auto-generated SWOT + recovery plans as starting point for client work

---

## 3. Value Proposition

**For PR/Comms teams** who need to monitor and manage corporate reputation, **RepuView** is an AI-powered reputation intelligence platform **that** quantifies reputation dynamics with the same rigor as financial risk, **unlike** traditional media monitoring tools (Brandwatch, Meltwater, Cision) **because** it combines econometric methods (PCA, Granger causality, spillover analysis) with AI-generated strategic recommendations, providing not just "what happened" but "what it means" and "what to do about it."

---

## 4. Feature Matrix

### V1 — Current (Live)

| Feature | Description | Data Source |
|---------|-------------|-------------|
| Portfolio Dashboard | Monitor multiple companies with index-based reputation scores | yfinance + GDELT |
| Company Profile | Logo, market cap, industry, keywords (auto-fetched for any company) | Clearbit + Claude API |
| Historical Index | Weekly reputation index with 30-week MA, deviation, volatility | yfinance + GDELT |
| Live News Feed | Real-time news search with per-article AI impact scoring | Claude API + web search |
| Decay Model | Half-life projection (7d / 28d / 84d) with 12-week forecast chart | Client-side computation |
| SWOT Analysis | AI-generated strengths, weaknesses, opportunities, threats | Claude API |
| Recovery Plan | 3-horizon strategy (short / medium / long term) with audiences and messaging | Claude API |
| Spillover Network | Diebold-Yilmaz cross-company contagion mapping | Statsmodels (backend) |
| Portfolio Editing | Add/remove companies, custom search, +Add to Overview | Frontend state |

### V2 — Next

| Feature | Description | Why |
|---------|-------------|-----|
| FinBERT Local Sentiment | On-premise NLP for batch sentiment scoring | Reduce API costs, increase granularity |
| Event Database | Searchable historical reputation events with impact scores | Enable trend analysis, benchmarking |
| Sector Benchmarking | Compare a company's reputation index vs sector peers | Contextualize performance |
| PDF Report Export | One-click export of company analysis as branded PDF | Client deliverable for agencies |

### V3 — Planned

| Feature | Description | Why |
|---------|-------------|-----|
| FastAPI Backend | Hosted API serving precomputed data + real-time analysis | Production scalability |
| Multi-User Auth | User accounts with saved portfolios and alert preferences | SaaS readiness |
| Earnings Call NLP | Analyze earnings call transcripts for reputation signals | Leading indicator |
| ESG Overlay | Integrate MSCI/Sustainalytics ESG controversy data | Regulatory compliance angle |
| Slack/Email Alerts | Push notifications when reputation score breaches thresholds | Proactive monitoring |

---

## 5. User Stories

### Portfolio Management
- *As a PR director, I want to monitor my company and 4 key competitors in one view, so I can benchmark our reputation position.*
- *As an agency strategist, I want to add a prospective client's company and immediately see their reputation profile, so I can prepare for a pitch meeting.*

### Analysis
- *As an IR manager, I want to see how much of our stock volatility is explained by reputation factors vs market factors, so I can quantify reputation risk for the board.*
- *As a comms lead, I want to understand how a competitor's crisis affects our stock, so I can prepare proactive messaging.*

### Action
- *As a CMO, when a negative news event hits, I want to see its projected decay timeline and current residual impact, so I know whether to respond aggressively or wait it out.*
- *As a PR strategist, I want AI-generated audience targeting and messaging recommendations based on current reputation dynamics, so I can brief my team faster.*

---

## 6. Data Architecture

```
                    DATA LAYER                              ANALYSIS LAYER
              ┌─────────────────┐
              │    yfinance     │──── Stock prices, returns,
              │   (free, no key)│     volatility, drawdown
              └────────┬────────┘              │
                       │                       ▼
              ┌────────┴────────┐    ┌──────────────────┐
              │   GDELT DOC    │    │ Reputation Index  │
              │  (free, no key) │──── │  PCA composite:  │
              └────────┬────────┘    │  40% price +     │
                       │             │  40% sentiment + │
              ┌────────┴────────┐    │  20% volume      │
              │  Claude API    │    └────────┬─────────┘
              │  (pay-per-use)  │             │
              └────────┬────────┘             ▼
                       │             ┌──────────────────┐
                       │             │  Spillover Index  │
                       │             │  (Diebold-Yilmaz) │
                       │             └────────┬─────────┘
                       │                      │
                       ▼                      ▼
              ┌─────────────────────────────────────┐
              │     dashboard_data.json             │
              │  → React Frontend (LIVE/DEMO mode)  │
              └─────────────────────────────────────┘
```

---

## 7. Success Metrics

### Product KPIs

| Metric | Target (V1) | Measurement |
|--------|-------------|-------------|
| Reputation Index correlation with stock returns | R² > 0.15 | Weekly regression |
| Granger causality significance rate | > 60% of tested pairs at p < 0.05 | Statistical test |
| SWOT relevance (manual review) | > 80% of bullets rated "relevant" by domain expert | Qualitative review |
| News sentiment accuracy vs human baseline | > 75% agreement on positive/negative classification | Sample comparison |
| Dashboard load time (with real data) | < 3 seconds | Performance benchmark |
| Pipeline execution time (full run) | < 5 minutes for 5 companies, 1 year of data | Timing |

### Business KPIs (if productized)

| Metric | Target | Notes |
|--------|--------|-------|
| Time to generate client deliverable | < 15 min (from "add company" to PDF export) | Currently ~5 min for AI analysis |
| Cost per company analysis | < $0.50 in API costs | ~6 Claude API calls per company |
| User retention (weekly active) | > 40% month-over-month | For SaaS version |

---

## 8. Competitive Landscape

| Tool | Reputation Monitoring | Financial Linkage | AI Insights | Contagion Mapping | Decay Model |
|------|:---:|:---:|:---:|:---:|:---:|
| **RepuView** | ✅ | ✅ | ✅ | ✅ | ✅ |
| Brandwatch | ✅ | ❌ | ⚠️ (basic) | ❌ | ❌ |
| Meltwater | ✅ | ❌ | ⚠️ (basic) | ❌ | ❌ |
| Cision | ✅ | ❌ | ❌ | ❌ | ❌ |
| Burson Reputation Capital | ✅ | ✅ | ⚠️ (manual) | ❌ | ❌ |
| Bloomberg Terminal | ❌ (financial only) | ✅ | ❌ | ❌ | ❌ |

**Key differentiators:**
1. Only tool that applies econometric methods (Granger, Diebold-Yilmaz, wavelet) to reputation — bridging finance and PR
2. Only tool with explicit temporal decay modeling — answers "how much does this still matter?"
3. Only tool that maps cross-company reputation contagion — answers "who else is affected?"
4. AI-generated actionable strategy, not just monitoring dashboards

---

## 9. Go-to-Market

### Phase 1: Portfolio Piece (Current)
- Open-source on GitHub with demo mode
- Use in job applications and interviews to demonstrate product + technical thinking
- Collect feedback from PR/comms professionals via LinkedIn posts

### Phase 2: Agency Tool
- Package as an internal tool for PR agencies (Burson, Edelman, FTI)
- "Generate a client reputation brief in 15 minutes" value proposition
- Revenue: per-seat SaaS license ($99-299/mo) or per-report pricing

### Phase 3: Enterprise Platform
- Self-serve dashboard for corporate comms teams
- API access for integration with existing workflows (Salesforce, Slack)
- Revenue: enterprise contracts ($2K-10K/mo based on number of companies monitored)

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| AI hallucination in SWOT/recovery plans | Medium — could generate irrelevant advice | Ground all insights in specific news events; add "Based on: [headline]" citations |
| GDELT data gaps for non-English companies | Medium — reduced coverage for Asian/emerging markets | Add NewsAPI as backup source; support multi-language search |
| API cost scaling with user growth | High — Claude API costs grow linearly per analysis | Migrate to FinBERT for batch sentiment (V2); cache repeated analyses |
| Reputation index doesn't predict returns | High — undermines core value proposition | Validate with backtesting; be transparent about R² values; frame as "signal" not "prediction" |

---

## Appendix: Research Foundation

The analytical methods in RepuView were developed during master's research on cross-country financial stress transmission. The research studied how economic stress propagates across six economies (US, France, Germany, Italy, Spain, Belgium) using PCA, Granger causality, impulse response functions, FEVD, and wavelet analysis. Key finding: US financial stress Granger-causes sentiment shifts in all five European countries, with behavioral mechanisms (loss aversion, anchoring, herding) shaping the transmission dynamics.

RepuView translates this framework from country-level to company-level, asking: can brand reputation signals predict financial outcomes, and how does reputation risk spread across an industry?

See [METHODOLOGY.md](METHODOLOGY.md) for full technical details.
