"""
Reputation Capital Model — Interactive Dashboard
==================================================
Streamlit app for exploring reputation risk, financial impact,
and cross-company contagion dynamics.

Run: streamlit run app.py
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sample_data import generate_dataset, get_reputation_events, COMPANIES
from pca_reputation_score import extract_reputation_scores, compute_reputation_regime
from financial_impact import granger_causality_test, var_irf_fevd, compute_reputation_beta
from spillover_analysis import compute_spillover_table, build_contagion_network, compute_rolling_spillover
from wavelet_analysis import reputation_stock_wavelet, pairwise_company_wavelet


# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reputation Capital Model",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
    }
    .regime-strong { color: #10b981; font-weight: 700; }
    .regime-normal { color: #6b7280; font-weight: 600; }
    .regime-atrisk { color: #f59e0b; font-weight: 700; }
    .regime-crisis { color: #ef4444; font-weight: 700; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all_data():
    df = generate_dataset()
    events = get_reputation_events()
    scores_df, pca_diag = extract_reputation_scores(df)
    scores_df = compute_reputation_regime(scores_df)
    gc_results = granger_causality_test(df, scores_df, max_lag=5)

    var_results = {}
    for ticker in COMPANIES:
        res = var_irf_fevd(df, scores_df, ticker)
        if "error" not in res:
            var_results[ticker] = res

    spillover = compute_spillover_table(scores_df)
    network = build_contagion_network(spillover, threshold=3.0) if "error" not in spillover else None
    rolling_spill = compute_rolling_spillover(scores_df, window=80, step=4)
    rep_beta = compute_reputation_beta(df, scores_df, window=60)

    wavelet_results = {}
    for ticker in COMPANIES:
        wavelet_results[ticker] = reputation_stock_wavelet(df, scores_df, ticker)

    pair_coherence = pairwise_company_wavelet(scores_df)

    return {
        "df": df, "events": events, "scores_df": scores_df,
        "pca_diag": pca_diag, "gc_results": gc_results,
        "var_results": var_results, "spillover": spillover,
        "network": network, "rolling_spill": rolling_spill,
        "rep_beta": rep_beta, "wavelet_results": wavelet_results,
        "pair_coherence": pair_coherence,
    }


data = load_all_data()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Reputation Capital Model")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📊 Overview Dashboard",
            "🔬 PCA Reputation Scores",
            "📈 Financial Impact (Granger/IRF)",
            "🌐 Contagion Network (Spillover)",
            "🌊 Wavelet Analysis",
            "📋 Methodology",
        ],
    )

    st.markdown("---")
    st.markdown("**Built by Haoyu Cheng**")
    st.markdown(
        "Based on methods from master's research on "
        "Financial Stress & Systemic Risk"
    )
    st.markdown("---")
    st.caption(
        "This tool applies the same analytical framework "
        "(PCA, Granger Causality, Diebold-Yilmaz Spillover, Wavelet Analysis) "
        "used to study cross-country financial stress transmission "
        "to brand reputation dynamics."
    )


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Overview Dashboard
# ═════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.markdown('<p class="main-header">Reputation Capital Model</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        'AI-powered reputation risk monitoring for Big Tech — '
        'linking media sentiment to financial outcomes'
        '</p>',
        unsafe_allow_html=True,
    )

    scores = data["scores_df"]
    df = data["df"]

    # ── Current Status Cards ──
    cols = st.columns(5)
    for i, (ticker, name) in enumerate(COMPANIES.items()):
        latest = scores[scores["ticker"] == ticker].iloc[-1]
        prev = scores[scores["ticker"] == ticker].iloc[-5]
        delta = latest["reputation_score_normalized"] - prev["reputation_score_normalized"]
        regime = latest["reputation_regime"]

        regime_color = {
            "Strong": "🟢", "Normal": "🔵", "At Risk": "🟡", "Crisis": "🔴"
        }.get(regime, "⚪")

        with cols[i]:
            st.metric(
                label=f"{regime_color} {name}",
                value=f"{latest['reputation_score_normalized']:.0f}",
                delta=f"{delta:+.1f}",
            )
            st.caption(f"Regime: **{regime}**")

    st.markdown("---")

    # ── Reputation Score Time Series ──
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Reputation Score Trends")
        fig = go.Figure()
        colors = {"AAPL": "#555555", "GOOGL": "#4285F4", "META": "#0668E1",
                  "MSFT": "#00A4EF", "AMZN": "#FF9900"}
        for ticker in COMPANIES:
            ts = scores[scores["ticker"] == ticker]
            fig.add_trace(go.Scatter(
                x=ts["date"], y=ts["reputation_score_normalized"],
                name=ticker, line=dict(color=colors[ticker], width=2),
                hovertemplate=f"{ticker}<br>Score: %{{y:.1f}}<br>%{{x}}<extra></extra>",
            ))

        # Add event markers
        events = data["events"]
        for _, evt in events.iterrows():
            if evt["shock_magnitude"] < -0.25:
                fig.add_vline(
                    x=evt["date"], line_dash="dot",
                    line_color="rgba(239,68,68,0.3)",
                    annotation_text=evt["description"][:30],
                    annotation_position="top",
                    annotation_font_size=8,
                )

        fig.update_layout(
            height=400, template="plotly_white",
            legend=dict(orientation="h", y=-0.15),
            yaxis_title="Reputation Score (0-100)",
            margin=dict(t=20, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Spillover Network")
        spillover = data["spillover"]
        if "error" not in spillover:
            st.metric("Total Spillover Index",
                      f"{spillover['total_spillover_index']}%",
                      help="Higher = more interconnected reputation risk")

            net_df = pd.DataFrame({
                "Company": list(spillover["net"].keys()),
                "Net Spillover (%)": list(spillover["net"].values()),
            }).sort_values("Net Spillover (%)", ascending=True)

            fig2 = go.Figure(go.Bar(
                y=net_df["Company"],
                x=net_df["Net Spillover (%)"],
                orientation="h",
                marker_color=[
                    "#ef4444" if v < 0 else "#10b981"
                    for v in net_df["Net Spillover (%)"]
                ],
                text=[f"{v:+.1f}%" for v in net_df["Net Spillover (%)"]],
                textposition="outside",
            ))
            fig2.update_layout(
                height=300, template="plotly_white",
                xaxis_title="Net Spillover (%)",
                margin=dict(t=10, b=30, l=60),
                annotations=[
                    dict(x=0.02, y=1.05, xref="paper", yref="paper",
                         text="← Receiver | Transmitter →",
                         showarrow=False, font=dict(size=10, color="gray")),
                ],
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Rolling Spillover ──
    st.subheader("Reputation Contagion Over Time")
    rolling = data["rolling_spill"]
    if len(rolling) > 0:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=rolling["date"], y=rolling["total_spillover_index"],
            fill="tozeroy", fillcolor="rgba(102,126,234,0.15)",
            line=dict(color="#667eea", width=2),
            name="Total Spillover Index",
        ))
        fig3.update_layout(
            height=250, template="plotly_white",
            yaxis_title="Spillover Index (%)",
            margin=dict(t=10, b=30),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: PCA Reputation Scores
# ═════════════════════════════════════════════════════════════════════════
elif page == "🔬 PCA Reputation Scores":
    st.markdown("## PCA Reputation Score Extraction")
    st.markdown(
        "Composite reputation scores extracted from multi-source signals using "
        "Principal Component Analysis — the same dimensionality reduction technique "
        "used in the original macro-financial research to compress macro indicators into stress factors."
    )

    selected = st.selectbox("Select Company", list(COMPANIES.keys()),
                            format_func=lambda x: f"{x} — {COMPANIES[x]}")

    diag = data["pca_diag"][selected]

    col1, col2, col3 = st.columns(3)
    col1.metric("Variance Explained (PC1)",
                f"{diag['explained_variance_ratio'][0]:.1%}")
    col2.metric("Observations", f"{diag['n_observations']:,}")
    col3.metric("Features Used", f"{len(diag['feature_names'])}")

    # Loadings chart
    st.subheader("PC1 Factor Loadings")
    st.caption("Shows which signals contribute most to the reputation score")
    loadings = diag["loadings"]["PC1"].sort_values()

    fig = go.Figure(go.Bar(
        y=loadings.index,
        x=loadings.values,
        orientation="h",
        marker_color=["#ef4444" if v < 0 else "#10b981" for v in loadings.values],
    ))
    fig.update_layout(height=300, template="plotly_white",
                      xaxis_title="Loading Weight", margin=dict(t=10, l=200))
    st.plotly_chart(fig, use_container_width=True)

    # Regime timeline
    st.subheader("Reputation Regime Timeline")
    scores = data["scores_df"]
    ts = scores[scores["ticker"] == selected].copy()

    regime_colors = {"Strong": "#10b981", "Normal": "#6b7280",
                     "At Risk": "#f59e0b", "Crisis": "#ef4444"}

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ts["date"], y=ts["reputation_score_normalized"],
        line=dict(color="#1a1a2e", width=1.5),
        name="Score",
    ))

    for regime, color in regime_colors.items():
        mask = ts["reputation_regime"] == regime
        if mask.any():
            fig2.add_trace(go.Scatter(
                x=ts.loc[mask, "date"],
                y=ts.loc[mask, "reputation_score_normalized"],
                mode="markers", marker=dict(color=color, size=4),
                name=regime,
            ))

    fig2.update_layout(height=350, template="plotly_white",
                       yaxis_title="Score (0-100)", margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Financial Impact
# ═════════════════════════════════════════════════════════════════════════
elif page == "📈 Financial Impact (Granger/IRF)":
    st.markdown("## Reputation → Financial Impact")
    st.markdown(
        "Testing whether reputation signals **Granger-cause** stock returns and volatility, "
        "then tracing the impact trajectory via Impulse Response Functions (IRFs)."
    )

    # Granger Causality Table
    st.subheader("Granger Causality Results")
    gc = data["gc_results"]
    gc_display = gc.copy()
    gc_display["significant"] = gc_display["significant_5pct"].map(
        {True: "✅ Yes", False: "❌ No"}
    )

    selected_ticker = st.selectbox(
        "Filter by company", ["All"] + list(COMPANIES.keys())
    )
    if selected_ticker != "All":
        gc_display = gc_display[gc_display["ticker"] == selected_ticker]

    st.dataframe(
        gc_display[["ticker", "target", "lag", "f_test_pvalue", "significant"]],
        use_container_width=True, hide_index=True,
    )

    sig_count = gc[gc["significant_5pct"]].shape[0]
    st.info(
        f"**{sig_count}** significant Granger-causal relationships found. "
        f"This means reputation shifts have statistically significant "
        f"predictive power for stock movements."
    )

    st.markdown("---")

    # IRF
    st.subheader("Impulse Response Functions")
    irf_ticker = st.selectbox("Select company for IRF/FEVD",
                              list(data["var_results"].keys()))

    if irf_ticker in data["var_results"]:
        var_res = data["var_results"][irf_ticker]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**reputation_score → daily_return**")
            irf_key = "reputation_score → daily_return"
            if irf_key in var_res["irf"]:
                irf_vals = var_res["irf"][irf_key]["response"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=irf_vals, mode="lines+markers",
                    line=dict(color="#667eea", width=2),
                    marker=dict(size=5),
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    height=300, template="plotly_white",
                    xaxis_title="Weeks after shock",
                    yaxis_title="Response",
                    margin=dict(t=10),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**FEVD: daily_return decomposition**")
            if "daily_return" in var_res["fevd"]:
                fevd_df = var_res["fevd"]["daily_return"]
                fig2 = go.Figure()
                colors_fevd = {"reputation_score": "#667eea",
                               "daily_return": "#10b981",
                               "realized_vol_20d": "#f59e0b"}
                for col_name in fevd_df.columns:
                    fig2.add_trace(go.Scatter(
                        y=fevd_df[col_name] * 100,
                        name=col_name,
                        stackgroup="one",
                        line=dict(color=colors_fevd.get(col_name, "#999")),
                    ))
                fig2.update_layout(
                    height=300, template="plotly_white",
                    xaxis_title="Forecast Horizon (weeks)",
                    yaxis_title="Variance Share (%)",
                    margin=dict(t=10),
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig2, use_container_width=True)

    # Reputation Beta
    st.markdown("---")
    st.subheader("Rolling Reputation Beta")
    st.caption("Sensitivity of stock returns to reputation score changes (60-day rolling window)")
    beta_df = data["rep_beta"]
    beta_ticker = st.selectbox("Company", list(COMPANIES.keys()), key="beta_sel")
    bt = beta_df[beta_df["ticker"] == beta_ticker]
    if len(bt) > 0:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=bt["date"], y=bt["reputation_beta"],
            line=dict(color="#764ba2", width=1.5),
            fill="tozeroy", fillcolor="rgba(118,75,162,0.1)",
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        fig3.update_layout(height=250, template="plotly_white",
                           yaxis_title="Reputation β", margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Contagion Network
# ═════════════════════════════════════════════════════════════════════════
elif page == "🌐 Contagion Network (Spillover)":
    st.markdown("## Cross-Company Reputation Contagion")
    st.markdown(
        "Using the **Diebold-Yilmaz Spillover Index** to measure how "
        "reputation risk transmits between companies — the same methodology "
        "used in the original macro-financial research to track cross-country financial stress contagion."
    )

    spillover = data["spillover"]

    if "error" not in spillover:
        # Spillover table heatmap
        st.subheader("FEVD Spillover Table")
        st.caption("Each cell: % of company i's reputation variance explained by company j")
        table = spillover["spillover_table"]

        fig = px.imshow(
            table.values, x=table.columns, y=table.index,
            color_continuous_scale="RdBu_r", aspect="equal",
            text_auto=".1f",
        )
        fig.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        # Network visualization
        st.subheader("Contagion Network")
        network = data["network"]

        if network and network["n_edges"] > 0:
            # Simple force-directed layout using plotly
            G = network["graph"]
            pos = {
                "AAPL": (0, 1), "GOOGL": (1, 0.5), "META": (-1, 0.5),
                "MSFT": (0.5, -0.5), "AMZN": (-0.5, -0.5),
            }

            edge_x, edge_y = [], []
            for u, v, d in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(color="rgba(150,150,150,0.5)", width=2),
                hoverinfo="none",
            ))

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_colors = [
                "#10b981" if spillover["net"][n] > 0 else "#ef4444"
                for n in G.nodes()
            ]
            node_sizes = [
                15 + abs(spillover["net"][n]) * 3 for n in G.nodes()
            ]

            fig_net.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(color=node_colors, size=node_sizes,
                            line=dict(width=2, color="white")),
                text=list(G.nodes()),
                textposition="top center",
                textfont=dict(size=14, color="#1a1a2e"),
                hovertemplate="%{text}<br>Net: %{customdata:+.1f}%<extra></extra>",
                customdata=[spillover["net"][n] for n in G.nodes()],
            ))

            fig_net.update_layout(
                height=400, template="plotly_white",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(t=10),
                annotations=[
                    dict(x=0.01, y=0.99, xref="paper", yref="paper",
                         text="🟢 Transmitter  🔴 Receiver",
                         showarrow=False, font=dict(size=12)),
                ],
            )
            st.plotly_chart(fig_net, use_container_width=True)

        # Rolling spillover
        st.subheader("Time-Varying Contagion Intensity")
        rolling = data["rolling_spill"]
        if len(rolling) > 0:
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=rolling["date"], y=rolling["total_spillover_index"],
                fill="tozeroy", fillcolor="rgba(102,126,234,0.15)",
                line=dict(color="#667eea", width=2),
            ))
            fig_roll.update_layout(
                height=300, template="plotly_white",
                yaxis_title="Total Spillover Index (%)",
                margin=dict(t=10),
            )
            st.plotly_chart(fig_roll, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Wavelet Analysis
# ═════════════════════════════════════════════════════════════════════════
elif page == "🌊 Wavelet Analysis":
    st.markdown("## Wavelet Time-Frequency Analysis")
    st.markdown(
        "Cross-wavelet analysis reveals **at which time horizons** reputation "
        "and stock performance co-move — mirroring the wavelet approach from macro-financial research on "
        "scalograms (Figures 4.1–4.5)."
    )

    wav_ticker = st.selectbox("Select Company", list(COMPANIES.keys()))

    wav = data["wavelet_results"][wav_ticker]

    # Band coherence metrics
    col1, col2, col3 = st.columns(3)
    bc = wav["band_coherence"]
    col1.metric("Short-Term (1-2 months)", f"{bc['short_term']:.3f}",
                help="High-frequency co-movement")
    col2.metric("Medium-Term (2-6 months)", f"{bc['medium_term']:.3f}",
                help="Business-cycle co-movement")
    col3.metric("Long-Term (6+ months)", f"{bc['long_term']:.3f}",
                help="Structural co-movement")

    st.info(
        "📌 **Key finding**: Coherence **increases** from short to long term, "
        "suggesting reputation effects are structural and persistent — "
        "not just short-term noise trading. This mirrors findings from macro-financial stress research "
        "that cross-country stress transmission is strongest at medium-to-long "
        "frequency bands."
    )

    # Wavelet scalogram
    st.subheader(f"Cross-Wavelet Power: {wav_ticker} Reputation ↔ Stock Returns")

    fig = go.Figure(data=go.Heatmap(
        z=wav["cross_wavelet_power"],
        x=list(range(len(wav["dates"]))),
        y=wav["scales"],
        colorscale="Inferno",
        colorbar=dict(title="Power"),
    ))
    fig.update_layout(
        height=400, template="plotly_white",
        xaxis_title="Time Index (weeks)",
        yaxis_title="Scale (period in weeks)",
        margin=dict(t=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pairwise coherence
    st.markdown("---")
    st.subheader("Pairwise Reputation Coherence")
    st.caption("How strongly do companies' reputation scores co-move at different time horizons?")

    pairs = data["pair_coherence"]
    fig_pairs = px.bar(
        pairs.melt(
            id_vars=["company_1", "company_2"],
            value_vars=["short_term_coh", "medium_term_coh", "long_term_coh"],
            var_name="horizon", value_name="coherence",
        ),
        x="coherence", y=pairs.apply(
            lambda r: f"{r['company_1']}-{r['company_2']}", axis=1
        ).repeat(3).values,
        color="horizon",
        orientation="h",
        barmode="group",
        color_discrete_map={
            "short_term_coh": "#93c5fd",
            "medium_term_coh": "#667eea",
            "long_term_coh": "#1e40af",
        },
    )
    fig_pairs.update_layout(height=350, template="plotly_white",
                            margin=dict(t=10, l=100))
    st.plotly_chart(fig_pairs, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Methodology
# ═════════════════════════════════════════════════════════════════════════
elif page == "📋 Methodology":
    st.markdown("## Methodology")
    st.markdown(
        "This tool applies an analytical framework developed during master's research on macro-financial stress transmission "
        "to corporate reputation dynamics."
        "to brand reputation dynamics."
    )

    st.markdown("### Framework Translation")

    method_data = {
        "Method": [
            "PCA Dimensionality Reduction",
            "Granger Causality",
            "VAR → IRF",
            "VAR → FEVD",
            "Diebold-Yilmaz Spillover",
            "Wavelet Analysis",
        ],
        "Original Research (Country-Level)": [
            "Macro indicators → Financial / Non-Financial / Consumer stress factors",
            "Does US financial stress predict European sentiment?",
            "Trace 10-month impact of a US stress shock on France",
            "50%+ of Belgium's sentiment variance from foreign stress",
            "US, France, Germany = stress transmitters; Italy, Spain = receivers",
            "Medium-frequency co-movement strongest (Figure 4)",
        ],
        "This Tool (Company-Level)": [
            "News sentiment + volume + volatility → Reputation Score",
            "Does reputation score predict stock abnormal returns?",
            "Trace 20-week impact of a reputation shock on stock",
            "X% of return variance explained by reputation factor",
            "Meta = reputation transmitter; Google = receiver",
            "Long-term coherence > short-term → structural, not noise",
        ],
    }

    st.dataframe(pd.DataFrame(method_data), use_container_width=True,
                 hide_index=True)

    st.markdown("### Data Pipeline")
    st.markdown("""
    **Production Architecture** (what this would look like deployed):

    1. **Data Ingestion**: GDELT / NewsAPI → raw articles about target companies
    2. **NLP Sentiment**: FinBERT (financial domain BERT) → daily sentiment scores
    3. **Market Data**: Yahoo Finance API → prices, volumes, volatility
    4. **Feature Engineering**: PCA compression into composite reputation score
    5. **Causal Modeling**: VAR → Granger causality, IRF, FEVD
    6. **Network Analysis**: Diebold-Yilmaz spillover across companies
    7. **Time-Frequency**: Wavelet decomposition for multi-horizon analysis
    8. **Alert System**: Flag when reputation score breaches ±2σ threshold

    **Current Demo**: Uses synthetic data calibrated to real Big Tech events
    (2020-2025) to demonstrate the full analytical pipeline.
    """)

    st.markdown("### Key References")
    st.markdown("""
    - Diebold & Yilmaz (2012) — Spillover index framework
    - Baker, Bloom & Davis (2016) — Economic Policy Uncertainty
    - Kahneman & Tversky (1979) — Prospect Theory / loss aversion
    - Shiller (2017) — Narrative Economics
    - Oet et al. (2011, 2025) — Financial Stress Index, spillovers in financial intermediaries
    """)
