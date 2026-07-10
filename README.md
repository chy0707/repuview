# RepuView

An AI-powered reputation risk intelligence platform that quantifies brand reputation dynamics, predicts financial impact, maps cross-company contagion, and generates strategic recovery plans.

Built on analytical methods developed during my master's research on cross-country financial stress transmission — applying macroeconomic stress modeling (PCA, Granger Causality, Diebold-Yilmaz Spillover, Wavelet Analysis) to corporate reputation. See the [methodology documentation](docs/METHODOLOGY.md) and [product requirements](docs/PRD.md).

**Live demo:** [repuview.vercel.app](https://repuview.vercel.app)

---

## What It Does

**Overview Dashboard** — Monitor a portfolio of companies with index-based reputation scores (100 = baseline), weekly % changes vs 30-week moving averages, and a Diebold-Yilmaz spillover network showing which companies transmit vs receive reputation risk.

**Company Deep Dive** — Full company profile, weekly historical reputation index with MA overlay, rolling volatility, deviation analysis, and AI-generated SWOT + multi-horizon recovery plans.

**Live News + AI** — Real-time news search with per-article reputation impact scoring using a half-life decay model, a 12-week decay projection, and net residual aggregation.

---

## Architecture

- **Frontend** (`/src`, `/index.html`) — React + Recharts single-page dashboard (Vite).
- **Serverless proxy** (`/api/anthropic.js`) — forwards AI requests to Anthropic server-side so the API key never reaches the browser; includes per-IP rate limiting.
- **Backend analysis engine** (`/backend`) — Python pipeline (yfinance + GDELT) and the econometric models (PCA, Granger, VAR/IRF/FEVD, Diebold-Yilmaz spillover, wavelet). Run locally to generate `dashboard_data.json` for LIVE mode.

Without live data the dashboard runs in **DEMO mode** using built-in synthetic data — all charts, the spillover network, and decay models render; only live AI news scoring needs the key.

---

## Deployment (Vercel)

Standard Vite project: the repo root is the project root. `/api` is auto-detected as serverless functions — no custom build config required.

To enable AI features, add `ANTHROPIC_API_KEY` in Vercel → Settings → Environment Variables, then redeploy.

---

## Local Development

```bash
# Frontend
npm install
npm run dev

# Backend analysis engine (optional — generates real data)
cd backend
pip install -r requirements.txt
python data/pipeline.py           # GDELT sentiment (free)
python data/pipeline.py --api-key sk-ant-...   # + AI sentiment
python analysis.py                # PCA / Granger / spillover / wavelet
```

Copy `backend/outputs/dashboard_data.json` to `public/` to switch the dashboard to LIVE mode.

---

## Tech Stack

**Frontend:** React · Recharts · Vite
**Proxy:** Vercel serverless function (Node)
**Backend:** Python · Pandas · NumPy · Scikit-learn · Statsmodels · SciPy · PyWavelets · NetworkX

---

## Author

**Haoyu Cheng** — MS Commerce & Economic Development, Northeastern University

## License

MIT
