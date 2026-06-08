"""
graph_sensitivity.py
====================
Sensitivity analysis: effect of distance threshold (d_max) on the
collaboration graph used in the GTVMin federated learning algorithm.

WHAT THIS SCRIPT DOES
----------------------
The production graph (System A in build_network.py) connects all station
pairs within d_max = 300 km using a Gaussian kernel with sigma = 150 km.
This script investigates how the graph structure changes when d_max is
tightened to 250, 200, and 150 km, keeping sigma = d_max / 2 in each
case so that the boundary weight stays constant at exp(-2) ≈ 0.14.

For each d_max value:
  1. Builds the adjacency matrix using build_graph(D, dmax, sigma).
  2. Reports total edges, per-station degree, isolation status, weight range.
  3. Flags the smallest d_max at which no station is isolated - i.e. the
     strictest viable threshold for running GTVMin (every station needs at
     least one neighbour for graph regularisation to have any effect).

OUTPUT
------
  results/graph_sensitivity.png   2×2 map figure, one panel per d_max value

This script does NOT modify build_network.py or any part of the FL pipeline.

Usage:
    python graph_sensitivity.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from build_network import (
    build_distance_matrix,
    STATION_COORDS,
    STATIONS,
    haversine_km,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# d_max values to sweep [km].  sigma = d_max / 2 in all cases so the
# boundary weight is exp(-d_max^2 / (2*(d_max/2)^2)) = exp(-2) ≈ 0.14.
DMAX_VALUES = [300, 250, 200, 150]

N           = len(STATIONS)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(D: np.ndarray,
                dmax: float,
                sigma: float) -> tuple:
    """
    Build a distance-based adjacency matrix with parameterised threshold.

    Applies the same Gaussian kernel as System A in build_network.py, but
    accepts dmax and sigma as explicit parameters instead of using the
    module-level constants.

        A[i,j] = exp(-d² / (2 * sigma²))   if d <= dmax
               = 0                          otherwise
        A[i,i] = 0  (no self-loops)

    Parameters
    ----------
    D     : pairwise distance matrix [km], shape (9, 9)
    dmax  : maximum distance for an edge to exist [km]
    sigma : Gaussian width parameter [km]

    Returns
    -------
    A     : np.ndarray, shape (9, 9), symmetric, values in [0, 1]
    edges : list of (i, j, weight, distance_km) for all pairs i < j
            where weight > 0
    """
    A = np.exp(-D**2 / (2 * sigma**2))
    A[D > dmax] = 0.0
    np.fill_diagonal(A, 0.0)

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                edges.append((i, j, float(A[i, j]), float(D[i, j])))

    return A, edges


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_graph_summary(dmax: float,
                        sigma: float,
                        A: np.ndarray,
                        edges: list) -> None:
    """
    Print total edges, per-station degree, isolation status, and weight range.

    Parameters
    ----------
    dmax  : distance threshold used [km]
    sigma : Gaussian width used [km]
    A     : adjacency matrix, shape (9, 9)
    edges : edge list from build_graph()
    """
    degrees   = (A > 0).sum(axis=1).astype(int)   # number of neighbours per station
    isolated  = [STATIONS[i] for i, d in enumerate(degrees) if d == 0]
    weights   = [w for _, _, w, _ in edges]
    w_min     = min(weights) if weights else 0.0
    w_max     = max(weights) if weights else 0.0

    print(f"\n  d_max = {dmax:.0f} km  |  sigma = {sigma:.0f} km")
    print(f"    Total edges  : {len(edges)} / {N*(N-1)//2} possible")
    print(f"    Weight range : {w_min:.4f} – {w_max:.4f}")
    print(f"    Degrees per station:")
    for i, station in enumerate(STATIONS):
        iso_flag = "  ← ISOLATED" if degrees[i] == 0 else ""
        print(f"      {station:<30s} {degrees[i]:2d}{iso_flag}")
    if isolated:
        print(f"    ⚠  Isolated stations: {', '.join(isolated)}")
    else:
        print(f"    ✓  No isolated stations")


# ---------------------------------------------------------------------------
# 2×2 map figure
# ---------------------------------------------------------------------------

def plot_sensitivity(results: list, save_path: str) -> None:
    """
    Plot a 2×2 grid of network maps, one panel per d_max value.

    Each panel shows:
      - Station nodes (red dots) at their geographic coordinates
      - Edges as lines with opacity and linewidth scaled by weight
        (same formula as plot_network() in build_network.py)
      - Panel title with d_max and edge count

    Parameters
    ----------
    results   : list of (dmax, sigma, A, edges) tuples in order
    save_path : absolute path for the output PNG
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 16))
    axes = axes.flatten()

    lats = [STATION_COORDS[s][0] for s in STATIONS]
    lons = [STATION_COORDS[s][1] for s in STATIONS]

    short_labels = {
        "Helsinki Kaisaniemi"       : "Helsinki",
        "Turku Artukainen"          : "Turku",
        "Oulu Vihreäsaari"         : "Oulu",
        "Tampere Härmälä"          : "Tampere",
        "Jyväskylä Airport"        : "Jyväskylä",
        "Kuopio Maaninka"           : "Kuopio",
        "Rovaniemi Apukka"          : "Rovaniemi",
        "Sodankylä"                 : "Sodankylä",
        "Inari Saariselkä"         : "Inari",
        "Hanko Tulliniemi"          : "Hanko",
        "Kajaani Airport"           : "Kajaani",
        "Kittilä Airport"           : "Kittilä",
        "Muonio Oustajärvi"        : "Muonio",
        "Pelkosenniemi Pyhätunturi" : "Pelkosenniemi",
        "Raahe Nahkiainen"          : "Raahe",
    }

    for ax, (dmax, sigma, A, edges) in zip(axes, results):

        w_max_plot = A.max() if A.max() > 0 else 1.0

        # Edges
        for i, j, w, _ in edges:
            alpha = 0.2 + 0.8 * (w / w_max_plot)
            lw    = 0.5 + 3.0 * (w / w_max_plot)
            ax.plot(
                [lons[i], lons[j]],
                [lats[i], lats[j]],
                color="steelblue", alpha=alpha, linewidth=lw, zorder=1,
            )

        # Nodes
        degrees  = (A > 0).sum(axis=1)
        colors   = ["tomato" if degrees[i] > 0 else "gold" for i in range(N)]
        ax.scatter(lons, lats, s=80, color=colors, zorder=3,
                   edgecolors="darkred", linewidths=0.8)

        # Labels
        for i, station in enumerate(STATIONS):
            ax.annotate(
                short_labels[station],
                xy=(lons[i], lats[i]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=8, zorder=4,
            )

        # Title
        n_isolated = int((degrees == 0).sum())
        iso_note   = f"  ! {n_isolated} isolated" if n_isolated else "  OK connected"
        ax.set_title(
            f"d_max = {dmax:.0f} km  |  σ = {sigma:.0f} km\n"
            f"{len(edges)} edges{iso_note}",
            fontsize=11, fontweight="bold",
        )

        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude",  fontsize=9)
        ax.set_xlim(19, 32)
        ax.set_ylim(59, 70.5)

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor="steelblue", alpha=1.0, label="High-weight edge"),
        mpatches.Patch(facecolor="steelblue", alpha=0.3, label="Low-weight edge"),
        mpatches.Patch(facecolor="tomato",              label="Station (connected)"),
        mpatches.Patch(facecolor="gold",                label="Station (isolated)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Graph Sensitivity Analysis - Effect of Distance Threshold d_max\n"
        "Gaussian kernel: A[i,j] = exp(−d² / 2σ²), σ = d_max / 2",
        fontsize=13, fontweight="bold", y=0.995,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.995])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  Graph Sensitivity Analysis")
    print(f"  d_max values : {DMAX_VALUES} km")
    print(f"  sigma        : d_max / 2  (boundary weight ≈ exp(-2) ≈ 0.14)")
    print(f"  Stations     : {N}")
    print("=" * 65)

    D = build_distance_matrix()

    results      = []
    viable_dmax  = None   # smallest d_max with no isolated station

    for dmax in DMAX_VALUES:
        sigma        = dmax / 2.0
        A, edges     = build_graph(D, dmax, sigma)
        results.append((dmax, sigma, A, edges))

        print_graph_summary(dmax, sigma, A, edges)

        degrees   = (A > 0).sum(axis=1)
        if (degrees > 0).all() and viable_dmax is None:
            # Record first (smallest in the sweep order) fully-connected dmax.
            # DMAX_VALUES is listed largest-first so we capture the last one
            # that keeps all stations connected - tracked below after the loop.
            pass

    # Find the smallest d_max (last in descending list) with no isolated nodes.
    for dmax, sigma, A, edges in reversed(results):
        degrees = (A > 0).sum(axis=1)
        if (degrees > 0).all():
            viable_dmax = dmax
            break

    print("\n" + "=" * 65)
    if viable_dmax is not None:
        print(f"  ✓ Strictest viable threshold: d_max = {viable_dmax:.0f} km")
        print(f"    (smallest d_max at which no station is isolated)")
    else:
        print("  ⚠  All tested d_max values produce at least one isolated station.")
    print("=" * 65)

    plot_sensitivity(
        results,
        save_path=os.path.join(RESULTS_DIR, "graph_sensitivity.png"),
    )

    print("\n✓ Graph sensitivity analysis complete.")


if __name__ == "__main__":
    main()
