"""
Module 3: Cross-Company Reputation Contagion
=============================================
Implements the Diebold-Yilmaz Spillover Index at the company level.

Research parallel:
- Original research: measured how a stress shock in the US spills over to France, Germany, etc.
- This: measures how a reputation crisis at Meta spills over to Apple, Google, etc.

Key outputs:
- Total spillover index (system-wide connectedness)
- Directional spillovers (who transmits vs receives reputation risk)
- Net spillover positions (net transmitters vs net receivers)
- Network visualization of reputation contagion
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import networkx as nx
from typing import Dict, Tuple


def compute_spillover_table(
    scores_df: pd.DataFrame,
    forecast_horizon: int = 10,
    var_lags: int = 2,
    freq: str = "W",
) -> Dict:
    """
    Compute Diebold-Yilmaz spillover table from reputation scores.

    The spillover index S is defined as:
        S = (Σ_{i≠j} θ_ij / Σ_{i,j} θ_ij) × 100

    where θ_ij is the FEVD contribution of company j to company i.

    This follows the Diebold-Yilmaz spillover framework.
    """
    # Pivot to wide format: each column = one company's reputation score
    pivot = scores_df.pivot_table(
        index="date", columns="ticker", values="reputation_score"
    )

    # Resample to chosen frequency
    if freq:
        pivot = pivot.resample(freq).mean()

    pivot = pivot.dropna()

    tickers = pivot.columns.tolist()
    n = len(tickers)

    if len(pivot) < var_lags + forecast_horizon + 20:
        return {"error": "Insufficient data"}

    # Fit VAR
    model = VAR(pivot)
    try:
        result = model.fit(var_lags)
    except Exception as e:
        return {"error": str(e)}

    # FEVD
    fevd = result.fevd(forecast_horizon)

    # Build spillover table (at final horizon)
    spillover_matrix = np.zeros((n, n))
    for i in range(n):
        decomp = fevd.decomp[i][-1]  # last horizon
        total = decomp.sum()
        if total > 0:
            spillover_matrix[i, :] = decomp / total * 100

    spillover_df = pd.DataFrame(
        spillover_matrix,
        index=tickers,
        columns=tickers,
    ).round(2)

    # ── Directional spillovers ──
    # FROM others (received): row sum excluding diagonal
    received = {}
    for i, t in enumerate(tickers):
        received[t] = round(sum(
            spillover_matrix[i, j] for j in range(n) if j != i
        ), 2)

    # TO others (transmitted): column sum excluding diagonal
    transmitted = {}
    for j, t in enumerate(tickers):
        transmitted[t] = round(sum(
            spillover_matrix[i, j] for i in range(n) if i != j
        ), 2)

    # Net spillover
    net = {t: round(transmitted[t] - received[t], 2) for t in tickers}

    # Total spillover index
    off_diagonal = sum(
        spillover_matrix[i, j] for i in range(n) for j in range(n) if i != j
    )
    total_variance = spillover_matrix.sum()
    total_spillover_index = round(off_diagonal / total_variance * 100, 2) if total_variance > 0 else 0

    return {
        "spillover_table": spillover_df,
        "received": received,
        "transmitted": transmitted,
        "net": net,
        "total_spillover_index": total_spillover_index,
        "tickers": tickers,
        "n_obs": len(pivot),
        "var_lags": var_lags,
        "forecast_horizon": forecast_horizon,
    }


def compute_rolling_spillover(
    scores_df: pd.DataFrame,
    window: int = 100,
    step: int = 5,
    forecast_horizon: int = 10,
    var_lags: int = 2,
) -> pd.DataFrame:
    """
    Compute time-varying total spillover index using rolling windows.

    This captures how reputation contagion intensity changes over time,
    similar to time-varying connectedness analysis in macro-financial research.
    """
    pivot = scores_df.pivot_table(
        index="date", columns="ticker", values="reputation_score"
    ).resample("W").mean().dropna()

    results = []
    dates = pivot.index.tolist()

    for start_idx in range(0, len(dates) - window, step):
        end_idx = start_idx + window
        window_data = pivot.iloc[start_idx:end_idx]

        tickers = window_data.columns.tolist()
        n = len(tickers)

        try:
            model = VAR(window_data)
            result = model.fit(var_lags)
            fevd = result.fevd(forecast_horizon)

            spillover_matrix = np.zeros((n, n))
            for i in range(n):
                decomp = fevd.decomp[i][-1]
                total = decomp.sum()
                if total > 0:
                    spillover_matrix[i, :] = decomp / total * 100

            off_diag = sum(
                spillover_matrix[i, j]
                for i in range(n) for j in range(n) if i != j
            )
            total_var = spillover_matrix.sum()
            tsi = off_diag / total_var * 100 if total_var > 0 else 0

            results.append({
                "date": dates[end_idx - 1],
                "total_spillover_index": round(tsi, 2),
                "window_start": dates[start_idx],
                "window_end": dates[end_idx - 1],
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def build_contagion_network(spillover_result: Dict, threshold: float = 5.0) -> Dict:
    """
    Build a directed network graph from the spillover table.

    Edges represent reputation contagion paths where the FEVD contribution
    exceeds the threshold, following the connectedness network approach from macro-financial stress research.

    Returns network metrics and edge list for visualization.
    """
    table = spillover_result["spillover_table"]
    tickers = spillover_result["tickers"]
    net = spillover_result["net"]

    G = nx.DiGraph()

    for t in tickers:
        G.add_node(t, net_spillover=net[t])

    edges = []
    for i, source in enumerate(tickers):
        for j, target in enumerate(tickers):
            if i != j:
                weight = table.iloc[j, i]  # j receives from i
                if weight > threshold:
                    G.add_edge(source, target, weight=weight)
                    edges.append({
                        "source": source,
                        "target": target,
                        "weight": round(weight, 2),
                    })

    # Network metrics
    metrics = {}
    for t in tickers:
        metrics[t] = {
            "out_degree": G.out_degree(t),
            "in_degree": G.in_degree(t),
            "net_spillover": net[t],
            "role": "Transmitter" if net[t] > 0 else "Receiver",
        }
        if G.number_of_edges() > 0:
            try:
                metrics[t]["betweenness"] = round(
                    nx.betweenness_centrality(G).get(t, 0), 3
                )
            except Exception:
                metrics[t]["betweenness"] = 0

    return {
        "graph": G,
        "edges": pd.DataFrame(edges),
        "metrics": pd.DataFrame(metrics).T,
        "n_edges": G.number_of_edges(),
        "density": round(nx.density(G), 3) if G.number_of_nodes() > 1 else 0,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/reputation-capital-model/data")
    sys.path.insert(0, "/home/claude/reputation-capital-model/models")
    from sample_data import generate_dataset
    from pca_reputation_score import extract_reputation_scores

    df = generate_dataset()
    scores_df, _ = extract_reputation_scores(df)

    print("=== Diebold-Yilmaz Spillover Analysis ===\n")
    result = compute_spillover_table(scores_df)

    if "error" not in result:
        print(f"Total Spillover Index: {result['total_spillover_index']}%\n")
        print("Spillover Table (FEVD %):")
        print(result["spillover_table"].to_string())
        print(f"\nNet Spillovers (+ = transmitter, - = receiver):")
        for t, v in sorted(result["net"].items(), key=lambda x: -x[1]):
            role = "TRANSMITTER" if v > 0 else "RECEIVER"
            print(f"  {t}: {v:+.2f}%  ({role})")

        print("\n=== Contagion Network ===\n")
        network = build_contagion_network(result, threshold=3.0)
        print(f"Edges: {network['n_edges']}")
        print(f"Density: {network['density']}")
        print(f"\nNode metrics:\n{network['metrics'].to_string()}")
