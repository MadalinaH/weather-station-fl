"""
build_network.py
================
Phase 2

Constructs three weighted adjacency matrices over the 15 Finnish weather
stations that define the collaboration graph for the GTVMin FL algorithm.

WHY A GRAPH?
------------
In GTVMin each station's model is regularised toward its neighbours'
models.  The adjacency matrix A[i,i'] controls:
  - WHETHER two stations collaborate  (0 = no edge, >0 = edge)
  - HOW STRONGLY they are pulled together (larger weight = stronger pull)

TWO SYSTEMS ARE COMPARED
-------------------------
System A - Distance graph:
    Edge weight = Gaussian decay of geographic distance only.
    A[i,i'] = exp(-d^2 / (2 * sigma^2))  if d <= d_max, else 0
    Pure geography: close stations always collaborate regardless of
    whether their climates actually co-vary.

System B - Correlation graph:
    Edge weight = distance Gaussian * Pearson temperature correlation.
    A[i,i'] = max(0, corr(T_i, T_j)) * exp(-d^2 / (2 * sigma^2))
    Only stations that are BOTH close AND positively correlated collaborate.
    Pearson correlation is computed on training data only (no data leakage).

System D - Seasonal correlation graphs (winter + summer):
    Two adjacency matrices - one for winter (Oct-Mar), one for summer (Apr-Sep).
    Edge weight = seasonal_corr(T_i, T_j) * exp(-d^2 / (2 * sigma^2))  if d <= d_max.
    Correlations are computed separately on each season's training days only.
    Used with run_fl_seasonal() which blends the two graphs per round.

System C - k-nearest-neighbours (k-NN) graph:
    Each station connects to its k geographically closest neighbours.
    Edge weight = exp(-d^2 / (2 * sigma^2))  (same Gaussian as System A).
    Graph is made symmetric: edge (i,j) exists if j ∈ kNN(i) OR i ∈ kNN(j).
    Guarantees every station has at least k neighbours - no station can be
    isolated regardless of geographic spacing.

PARAMETERS
----------
    d_max = 200 km   - maximum distance for an edge to exist (Systems A & B)
                       (all 15 stations remain connected; 150 km is the
                        strictest viable threshold per sensitivity analysis)
    sigma = 100 km   - Gaussian width (weight = ~0.14 at d = d_max)
    k     = 3        - default nearest neighbours for System C

OUTPUT
------
    fl_project/data/adj_system_a.npy    15×15 adjacency matrix, System A
    fl_project/data/adj_system_b.npy    15×15 adjacency matrix, System B
    fl_project/data/adj_system_c.npy       15×15 adjacency matrix, System C (k=3)
    fl_project/data/adj_system_d_winter.npy  15×15 winter correlation matrix, System D
    fl_project/data/adj_system_d_summer.npy  15×15 summer correlation matrix, System D
    fl_project/results/network_system_a.png
    fl_project/results/network_system_b.png

Usage:
    python build_network.py

    The script loads training t2m data directly from the station CSVs in
    fl_project/data/ (uses 2022-01-01 to 2023-12-31 for correlations).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import radians, sin, cos, sqrt, atan2


# ---------------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------------

# Station name → (latitude, longitude) as specified in the project prompt.
STATION_COORDS = {
    "Helsinki Kaisaniemi"       : (60.1756, 24.9414),
    "Turku Artukainen"          : (60.5149, 22.2663),
    "Oulu Vihreäsaari"         : (65.0090, 25.3960),
    "Tampere Härmälä"          : (61.4940, 23.7700),
    "Jyväskylä Airport"        : (62.3996, 25.6787),
    "Kuopio Maaninka"           : (63.1484, 27.3084),
    "Rovaniemi Apukka"          : (66.5600, 26.0100),
    "Sodankylä"                 : (67.3668, 26.6500),
    "Inari Saariselkä"         : (68.4200, 27.4100),
    "Hanko Tulliniemi"          : (59.8171, 22.9083),
    "Kajaani Airport"           : (64.2853, 27.6924),
    "Kittilä Airport"           : (67.7014, 24.8467),
    "Muonio Oustajärvi"        : (67.9624, 23.6824),
    "Pelkosenniemi Pyhätunturi" : (67.0239, 27.2201),
    "Raahe Nahkiainen"          : (64.6736, 24.5603),
}

# Ordered list of station names - index in this list = node index in the matrix.
STATIONS = list(STATION_COORDS.keys())
N = len(STATIONS)   # 15

# Graph construction hyperparameters.
# Updated from d_max=300/sigma=150 to d_max=200/sigma=100 based on the
# graph sensitivity analysis (graph_sensitivity.py): 200 km is the
# strictest viable threshold that keeps all 15 stations connected.
D_MAX_KM = 200.0    # maximum distance for an edge to exist [km]
SIGMA_KM = 100.0    # Gaussian width parameter [km]

# Training period for Pearson correlation (System B).
# Correlations are computed on training data ONLY to avoid data leakage.
TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# CSV filename for each station (matches standardise_data.py output).
STATION_FILES = {
    # All 15 stations
    "Helsinki Kaisaniemi"       : "helsinki.csv",
    "Turku Artukainen"          : "turku.csv",
    "Oulu Vihreäsaari"         : "oulu.csv",
    "Tampere Härmälä"          : "tampere.csv",
    "Jyväskylä Airport"        : "jyvaskyla.csv",
    "Kuopio Maaninka"           : "kuopio.csv",
    "Rovaniemi Apukka"          : "rovaniemi.csv",
    "Sodankylä"                 : "sodankyla.csv",
    "Inari Saariselkä"         : "inari.csv",
    # 6 new stations
    "Hanko Tulliniemi"          : "hanko.csv",
    "Kajaani Airport"           : "kajaani.csv",
    "Kittilä Airport"           : "kittila.csv",
    "Muonio Oustajärvi"        : "muonio.csv",
    "Pelkosenniemi Pyhätunturi" : "pelkosenniemi.csv",
    "Raahe Nahkiainen"          : "raahe.csv",
}


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two points on Earth [km].

    Uses the Haversine formula, which is accurate for distances up to a
    few thousand kilometres.

    Parameters
    ----------
    lat1, lon1 : latitude and longitude of point 1 [decimal degrees]
    lat2, lon2 : latitude and longitude of point 2 [decimal degrees]

    Returns
    -------
    Distance in kilometres.
    """
    R = 6371.0   # Earth radius [km]
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# Pairwise distance matrix
# ---------------------------------------------------------------------------

def build_distance_matrix() -> np.ndarray:
    """
    Build the 15×15 pairwise Haversine distance matrix [km].

    D[i, j] = distance between station i and station j.
    The matrix is symmetric with zeros on the diagonal.

    Returns
    -------
    np.ndarray of shape (15, 15), dtype float64.
    """
    D = np.zeros((N, N))
    for i, s1 in enumerate(STATIONS):
        for j, s2 in enumerate(STATIONS):
            if i != j:
                lat1, lon1 = STATION_COORDS[s1]
                lat2, lon2 = STATION_COORDS[s2]
                D[i, j] = haversine_km(lat1, lon1, lat2, lon2)
    return D


# ---------------------------------------------------------------------------
# System A - Distance adjacency matrix
# ---------------------------------------------------------------------------

def build_adj_system_a(D: np.ndarray) -> np.ndarray:
    """
    Build the distance-based adjacency matrix (System A).

    A[i,i'] = exp(-d^2 / (2 * sigma^2))   if d <= d_max
            = 0                             otherwise
    Diagonal is always 0 (no self-loops).

    Parameters
    ----------
    D : pairwise distance matrix [km], shape (9, 9)

    Returns
    -------
    np.ndarray of shape (9, 9), symmetric, values in [0, 1].
    """
    A = np.exp(-D**2 / (2 * SIGMA_KM**2))  # Gaussian weights for all pairs
    A[D > D_MAX_KM] = 0.0                   # zero out pairs beyond d_max
    np.fill_diagonal(A, 0.0)                # no self-loops
    return A


# ---------------------------------------------------------------------------
# Load training t2m series for System B correlation
# ---------------------------------------------------------------------------

def load_training_t2m() -> dict:
    """
    Load daily mean temperature (t2m) for the training period from each
    station's CSV and return as a dict of pandas Series.

    Only the training period (TRAIN_START to TRAIN_END) is used so that
    the Pearson correlation does not see validation or test data.

    Returns
    -------
    dict mapping station_name → pd.Series of t2m values indexed by date.
    Missing days (NaN) are dropped before returning so that correlation is
    computed on the common set of available observations.
    """
    series = {}
    for station in STATIONS:
        csv_path = os.path.join(DATA_DIR, STATION_FILES[station])
        df = pd.read_csv(csv_path, parse_dates=["date"])
        # Filter to training period only.
        mask = (df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)
        s = df.loc[mask, ["date", "t2m"]].set_index("date")["t2m"].dropna()
        series[station] = s
    return series


# ---------------------------------------------------------------------------
# System B - Correlation adjacency matrix
# ---------------------------------------------------------------------------

def build_adj_system_b(D: np.ndarray, t2m_series: dict) -> np.ndarray:
    """
    Build the correlation-weighted adjacency matrix (System B).

    A[i,i'] = max(0, pearson_corr(T_i, T_j)) * exp(-d^2 / (2 * sigma^2))
                                               if d <= d_max
            = 0                                otherwise
    Diagonal is always 0 (no self-loops).

    Pearson correlation is computed on the intersection of days where both
    stations have non-NaN t2m observations.  If fewer than 2 overlapping
    days exist the correlation is set to 0.

    Parameters
    ----------
    D          : pairwise distance matrix [km], shape (9, 9)
    t2m_series : dict from load_training_t2m()

    Returns
    -------
    np.ndarray of shape (9, 9), symmetric, values in [0, 1].
    """
    # First compute the distance-based Gaussian weights (same as System A).
    gaussian = np.exp(-D**2 / (2 * SIGMA_KM**2))
    gaussian[D > D_MAX_KM] = 0.0

    A = np.zeros((N, N))
    for i, s1 in enumerate(STATIONS):
        for j, s2 in enumerate(STATIONS):
            if i == j or gaussian[i, j] == 0.0:
                continue  # skip diagonal and out-of-range pairs

            # Align the two series on their common dates.
            common = t2m_series[s1].index.intersection(t2m_series[s2].index)
            if len(common) < 2:
                continue   # not enough overlap to compute correlation

            t1 = t2m_series[s1].loc[common].values
            t2 = t2m_series[s2].loc[common].values

            # Pearson correlation using numpy (ddof=1 for sample correlation).
            corr = float(np.corrcoef(t1, t2)[0, 1])

            # Zero out negative or NaN correlations.
            if np.isnan(corr) or corr < 0:
                corr = 0.0

            A[i, j] = corr * gaussian[i, j]

    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------------------------
# System C - k-nearest-neighbours adjacency matrix
# ---------------------------------------------------------------------------

def build_adj_system_c(D: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """
    Build the k-nearest-neighbours (k-NN) adjacency matrix (System C).

    For each station i, identify its k closest stations by Haversine distance.
    The graph is made symmetric using the union rule: edge (i,j) exists if
    j is in i's kNN OR i is in j's kNN.  Edge weight uses the same Gaussian
    kernel as System A but without a hard d_max cutoff:

        A[i,j] = exp(-d² / (2 × sigma²))   if j ∈ kNN(i)  or  i ∈ kNN(j)
               = 0                           otherwise

    Unlike System A (d_max threshold), every station is guaranteed at least
    k neighbours regardless of geographic spacing - no station can be isolated.

    Parameters
    ----------
    D     : pairwise distance matrix [km], shape (N, N)
    k     : number of nearest neighbours per station
    sigma : Gaussian width [km]

    Returns
    -------
    np.ndarray of shape (N, N), symmetric, diagonal = 0.
    """
    local_N = D.shape[0]
    A = np.zeros((local_N, local_N))

    for i in range(local_N):
        distances = D[i].copy()
        distances[i] = np.inf          # exclude self
        nn_idx = np.argsort(distances)[:k]
        for j in nn_idx:
            w = np.exp(-D[i, j] ** 2 / (2.0 * sigma ** 2))
            A[i, j] = w
            A[j, i] = w                # symmetric (union)

    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------------------------
# System D - Seasonal correlation adjacency matrix
# ---------------------------------------------------------------------------

WINTER_MONTHS = [10, 11, 12, 1, 2, 3]   # Oct – Mar
SUMMER_MONTHS = [4,  5,  6,  7, 8, 9]   # Apr – Sep

def build_adj_seasonal(D: np.ndarray,
                       t2m_series: dict,
                       months: list,
                       sigma: float) -> np.ndarray:
    """
    Build a seasonal correlation-weighted adjacency matrix.

    Identical to build_adj_system_b() but Pearson correlations are computed
    only on days belonging to the specified calendar months.  This captures
    how stations co-vary during a specific meteorological season rather than
    across the whole year.

        A[i,j] = max(0, pearson_corr_season(T_i, T_j))
                 * exp(-d² / (2 * sigma²))   if d <= d_max
               = 0                            otherwise

    Parameters
    ----------
    D          : pairwise distance matrix [km], shape (N, N)
    t2m_series : dict station_name → pd.Series of t2m (training period only,
                 indexed by date - may include all months; filtering is done here)
    months     : list of integer month numbers to include (e.g. [10,11,12,1,2,3])
    sigma      : Gaussian width [km]

    Returns
    -------
    np.ndarray of shape (N, N), symmetric, values in [0, 1].
    """
    gaussian = np.exp(-D ** 2 / (2.0 * sigma ** 2))
    gaussian[D > D_MAX_KM] = 0.0

    A = np.zeros((N, N))
    for i, s1 in enumerate(STATIONS):
        for j, s2 in enumerate(STATIONS):
            if i == j or gaussian[i, j] == 0.0:
                continue

            # Filter each series to the target months only.
            ser1 = t2m_series[s1]
            ser2 = t2m_series[s2]

            # pd.Series indexed by date - filter by month.
            ser1_m = ser1[pd.to_datetime(ser1.index).month.isin(months)]
            ser2_m = ser2[pd.to_datetime(ser2.index).month.isin(months)]

            common = ser1_m.index.intersection(ser2_m.index)
            if len(common) < 2:
                continue

            t1   = ser1_m.loc[common].values
            t2   = ser2_m.loc[common].values
            corr = float(np.corrcoef(t1, t2)[0, 1])

            if np.isnan(corr) or corr < 0:
                corr = 0.0

            A[i, j] = corr * gaussian[i, j]

    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(name: str, A: np.ndarray) -> None:
    """
    Print a short summary of an adjacency matrix: edge count and weight range.

    An edge is counted once per unordered pair (i, j) with i < j.

    Parameters
    ----------
    name : system label for display (e.g. 'System A')
    A    : adjacency matrix, shape (9, 9)
    """
    # Count edges (upper triangle only to avoid double-counting).
    upper = A[np.triu_indices(N, k=1)]
    n_edges  = int(np.sum(upper > 0))
    weights  = upper[upper > 0]
    w_min    = float(weights.min()) if len(weights) else 0.0
    w_max    = float(weights.max()) if len(weights) else 0.0
    w_mean   = float(weights.mean()) if len(weights) else 0.0

    print(f"  {name}:")
    print(f"    Edges        : {n_edges} / {N*(N-1)//2} possible")
    print(f"    Weight range : {w_min:.4f} – {w_max:.4f}  (mean {w_mean:.4f})")


# ---------------------------------------------------------------------------
# Map visualisation
# ---------------------------------------------------------------------------

def plot_network(A: np.ndarray, title: str, save_path: str) -> None:
    """
    Plot the station graph on a schematic map of Finland and save to PNG.

    Stations are drawn as scatter points at their geographic coordinates.
    Edges are drawn as straight lines with opacity proportional to weight.
    The map background is a simple bounding-box representation of Finland
    (no external basemap required - only matplotlib is used).

    Parameters
    ----------
    A         : adjacency matrix, shape (9, 9)
    title     : plot title string
    save_path : absolute path for the output PNG
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 9))

    lats = [STATION_COORDS[s][0] for s in STATIONS]
    lons = [STATION_COORDS[s][1] for s in STATIONS]

    # Draw edges
    w_max = A.max() if A.max() > 0 else 1.0
    for i in range(N):
        for j in range(i + 1, N):
            w = A[i, j]
            if w <= 0:
                continue
            alpha = 0.2 + 0.8 * (w / w_max)   # scale opacity by weight
            lw    = 0.5 + 3.0 * (w / w_max)   # scale linewidth by weight
            ax.plot(
                [lons[i], lons[j]],
                [lats[i], lats[j]],
                color="steelblue", alpha=alpha, linewidth=lw, zorder=1,
            )

    # Draw station nodes
    ax.scatter(lons, lats, s=80, color="tomato", zorder=3,
               edgecolors="darkred", linewidths=0.8)

    # Station labels
    # Short labels to avoid overlap on the map.
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
    for i, station in enumerate(STATIONS):
        ax.annotate(
            short_labels[station],
            xy=(lons[i], lats[i]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, zorder=4,
        )

    # Edge weight legend
    legend_elements = [
        mpatches.Patch(facecolor="steelblue", alpha=1.0, label="High weight"),
        mpatches.Patch(facecolor="steelblue", alpha=0.3, label="Low weight"),
        mpatches.Patch(facecolor="tomato",    label="Station"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(19, 32)
    ax.set_ylim(59, 70.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {save_path}")


# ---------------------------------------------------------------------------
# Side-by-side network plot (for System D winter vs summer)
# ---------------------------------------------------------------------------

def _plot_network_pair(A_left: np.ndarray, A_right: np.ndarray,
                       title_left: str, title_right: str,
                       save_path: str) -> None:
    """
    Plot two adjacency matrices side by side on a schematic Finland map.

    Parameters
    ----------
    A_left      : adjacency matrix for left panel
    A_right     : adjacency matrix for right panel
    title_left  : title for left panel
    title_right : title for right panel
    save_path   : output PNG path
    """
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

    lats = [STATION_COORDS[s][0] for s in STATIONS]
    lons = [STATION_COORDS[s][1] for s in STATIONS]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))

    for ax, A, title in [(axes[0], A_left, title_left),
                          (axes[1], A_right, title_right)]:
        w_max = A.max() if A.max() > 0 else 1.0
        for i in range(N):
            for j in range(i + 1, N):
                w = A[i, j]
                if w <= 0:
                    continue
                alpha_val = 0.2 + 0.8 * (w / w_max)
                lw        = 0.5 + 3.0 * (w / w_max)
                ax.plot([lons[i], lons[j]], [lats[i], lats[j]],
                        color="steelblue", alpha=alpha_val, linewidth=lw, zorder=1)

        ax.scatter(lons, lats, s=80, color="tomato", zorder=3,
                   edgecolors="darkred", linewidths=0.8)

        for i, station in enumerate(STATIONS):
            ax.annotate(short_labels[station],
                        xy=(lons[i], lats[i]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, zorder=4)

        legend_elements = [
            mpatches.Patch(facecolor="steelblue", alpha=1.0, label="High weight"),
            mpatches.Patch(facecolor="steelblue", alpha=0.3, label="Low weight"),
            mpatches.Patch(facecolor="tomato",    label="Station"),
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=8)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(19, 32)
        ax.set_ylim(59, 70.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Build both adjacency matrices, save them, print summaries, and plot maps.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  FL Project - Phase 2: Building collaboration graphs")
    print(f"  Stations  : {N}")
    print(f"  d_max     : {D_MAX_KM} km")
    print(f"  sigma     : {SIGMA_KM} km")
    print("=" * 65 + "\n")

    # Pairwise distance matrix (shared by both systems)
    print("  Computing pairwise Haversine distances …")
    D = build_distance_matrix()

    # Print distance matrix for reference.
    print("\n  Distance matrix [km]:")
    header = "  " + "".join(f"{s[:6]:>10s}" for s in STATIONS)
    print(header)
    for i, s in enumerate(STATIONS):
        row = "  " + f"{s[:6]:>10s}" + "".join(f"{D[i,j]:>10.1f}" for j in range(N))
        print(row)

    # System A
    print("\n  Building System A (distance graph) …")
    A_system_a = build_adj_system_a(D)
    np.save(os.path.join(DATA_DIR, "adj_system_a.npy"), A_system_a)
    print("    Saved → adj_system_a.npy")
    print_summary("System A", A_system_a)

    # System B
    print("\n  Building System B (correlation graph) …")
    print(f"  Loading training t2m ({TRAIN_START} → {TRAIN_END}) …")
    t2m_series = load_training_t2m()
    A_system_b = build_adj_system_b(D, t2m_series)
    np.save(os.path.join(DATA_DIR, "adj_system_b.npy"), A_system_b)
    print("    Saved → adj_system_b.npy")
    print_summary("System B", A_system_b)

    # System C
    DEFAULT_K = 3
    print(f"\n  Building System C (k-NN graph, k={DEFAULT_K}) …")
    A_system_c = build_adj_system_c(D, k=DEFAULT_K, sigma=SIGMA_KM)
    np.save(os.path.join(DATA_DIR, "adj_system_c.npy"), A_system_c)
    print("    Saved → adj_system_c.npy")
    print_summary("System C", A_system_c)
    print(f"    k (default) : {DEFAULT_K}")
    degrees = [(A_system_c[i] > 0).sum() for i in range(N)]
    print(f"    Degree range: {min(degrees)} - {max(degrees)}  "
          f"(guaranteed min = {DEFAULT_K})")

    # Plots
    print("\n  Plotting networks …")
    plot_network(
        A_system_a,
        title=f"System A - Distance Graph\n(Gaussian decay, d_max={D_MAX_KM:.0f} km, σ={SIGMA_KM:.0f} km)",
        save_path=os.path.join(RESULTS_DIR, "network_system_a.png"),
    )
    plot_network(
        A_system_b,
        title="System B - Correlation Graph\n(Distance × Pearson correlation, training data only)",
        save_path=os.path.join(RESULTS_DIR, "network_system_b.png"),
    )
    plot_network(
        A_system_c,
        title=f"System C - k-NN Graph\n(k={DEFAULT_K} nearest neighbours, σ={SIGMA_KM:.0f} km)",
        save_path=os.path.join(RESULTS_DIR, "network_system_c.png"),
    )

    # System D - seasonal graphs
    print(f"\n  Building System D (seasonal graphs, σ={SIGMA_KM:.0f} km) …")
    A_winter = build_adj_seasonal(D, t2m_series, WINTER_MONTHS, SIGMA_KM)
    A_summer = build_adj_seasonal(D, t2m_series, SUMMER_MONTHS, SIGMA_KM)
    np.save(os.path.join(DATA_DIR, "adj_system_d_winter.npy"), A_winter)
    np.save(os.path.join(DATA_DIR, "adj_system_d_summer.npy"), A_summer)
    print("    Saved → adj_system_d_winter.npy")
    print("    Saved → adj_system_d_summer.npy")
    print_summary("System D (winter)", A_winter)
    print_summary("System D (summer)", A_summer)

    # Side-by-side winter / summer plot
    _plot_network_pair(
        A_winter, A_summer,
        title_left="System D - Winter Graph\n(Oct-Mar correlations)",
        title_right="System D - Summer Graph\n(Apr-Sep correlations)",
        save_path=os.path.join(RESULTS_DIR, "network_system_d.png"),
    )

    print("\n✓ Phase 2 complete.")


if __name__ == "__main__":
    main()
