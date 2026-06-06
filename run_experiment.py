"""
run_experiment.py
=================
Phase 6

Orchestrates the full FL experiment across THREE data splits to study
how training set size affects FL benefit.

SYSTEMS COMPARED
----------------
  Baseline  : local-only Ridge per station (alpha → 0, no graph).
  System A  : GTVMin FL with distance-based adjacency matrix.
  System B  : GTVMin FL with correlation-weighted adjacency matrix.
              Correlation weights are recomputed on each experiment's
              training data to avoid data leakage.

EXPERIMENTS
-----------
  Exp 1 - Full data    : Train 2022-2023 / Val Jan-Jun 2024 / Test Jul-Dec 2024
  Exp 2 - Reduced data : Train 2022      / Val Jan-Jun 2023 / Test Jul-Dec 2023
  Exp 3 - Minimal data : Train Jan-Jun 2022 / Val Jul-Sep 2022 / Test Oct-Dec 2022

HYPERPARAMETER SEARCH
---------------------
  Baseline  : alpha-only grid search (12 candidates).
  System A  : joint (alpha, sigma) grid search - 12 × 7 = 84 combinations.
  System B  : joint (alpha, sigma) grid search - 84 combinations.

  dmax is fixed at 200 km (strictest viable threshold).
  sigma controls Gaussian edge-weight decay within dmax.

  For System B, the Pearson correlation matrix is computed once per
  experiment on the training data, then combined with each sigma's
  Gaussian weights - avoiding redundant CSV reads.

  Criterion  : mean validation MSE (test set never seen during tuning)
  Best (alpha, sigma) selected independently per system per experiment.

GRAPH PARAMETERS
----------------
  dmax  = 200 km   (fixed - from graph sensitivity analysis)
  sigma = tuned    (candidates: 50, 75, 100, 125, 150, 175, 200 km)

OUTPUT FILES
------------
  results/experiment1_full_data.csv
  results/experiment2_reduced_data.csv
  results/experiment3_minimal_data.csv
  results/combined_summary.csv
  results/test_rmse_by_experiment.png
  results/test_rmse_by_station_exp{1,2,3}.png
  results/convergence_exp1.png
  results/sigma_heatmap_exp3.png      val RMSE heatmap over (alpha, sigma)
  data/adj_system_a.npy               updated with best sigma from Experiment 1
  data/adj_system_b.npy               updated with best sigma from Experiment 1

Usage:
    python run_experiment.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from fl_algorithm import run_fl
from evaluate import evaluate
import hdd_analysis

from prepare_data import load_station, build_xy, split_by_date, normalise
from build_network import (
    build_distance_matrix,
    STATIONS as STATION_NAMES,
    STATION_FILES,
    N as N_STATIONS,
)
from graph_sensitivity import build_graph


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DMAX_KM = 200.0   # fixed - from graph sensitivity analysis

ALPHA_CANDIDATES = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
                    10.0, 50.0, 100.0, 500.0]

SIGMA_CANDIDATES = [50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0]

T_ROUNDS    = 50
RANDOM_SEED = 42

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

np.random.seed(RANDOM_SEED)

EXPERIMENTS = [
    {
        "id"          : 1,
        "label"       : "Full data",
        "filename"    : "experiment1_full_data.csv",
        "train_start" : "2022-01-01", "train_end" : "2023-12-31",
        "val_start"   : "2024-01-01", "val_end"   : "2024-06-30",
        "test_start"  : "2024-07-01", "test_end"  : "2024-12-31",
    },
    {
        "id"          : 2,
        "label"       : "Reduced data",
        "filename"    : "experiment2_reduced_data.csv",
        "train_start" : "2022-01-01", "train_end" : "2022-12-31",
        "val_start"   : "2023-01-01", "val_end"   : "2023-06-30",
        "test_start"  : "2023-07-01", "test_end"  : "2023-12-31",
    },
    {
        "id"          : 3,
        "label"       : "Minimal data",
        "filename"    : "experiment3_minimal_data.csv",
        "train_start" : "2022-01-01", "train_end" : "2022-06-30",
        "val_start"   : "2022-07-01", "val_end"   : "2022-09-30",
        "test_start"  : "2022-10-01", "test_end"  : "2022-12-31",
    },
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_experiment_data(cfg: dict) -> dict:
    """
    Build train/val/test splits for all 9 stations using the date boundaries
    defined in experiment config cfg.
    """
    data = {}
    for station, filename in STATION_FILES.items():
        csv_path      = os.path.join(DATA_DIR, filename)
        df            = load_station(csv_path)
        X, y, dates   = build_xy(df)
        splits        = split_by_date(X, y, dates,
                                      train_start=cfg["train_start"],
                                      train_end=cfg["train_end"],
                                      val_start=cfg["val_start"],
                                      val_end=cfg["val_end"],
                                      test_start=cfg["test_start"],
                                      test_end=cfg["test_end"])
        splits        = normalise(splits)
        data[station] = splits
    return data


# ---------------------------------------------------------------------------
# Pearson correlation matrix (System B - computed once per experiment)
# ---------------------------------------------------------------------------

def compute_pearson_matrix(cfg: dict, D: np.ndarray) -> np.ndarray:
    """
    Compute the 9×9 Pearson correlation matrix for System B using only
    the training data from this experiment's training period.

    The correlations are clamped to [0, 1] (negative correlations → 0).
    This matrix is independent of sigma - it is computed once per experiment
    and then multiplied by the Gaussian weights for each sigma candidate.

    Parameters
    ----------
    cfg : experiment config dict (contains train_start / train_end)
    D   : pairwise distance matrix, shape (9, 9) - used only for ordering

    Returns
    -------
    np.ndarray of shape (9, 9) - clamped Pearson correlations, diagonal 0.
    """
    # Load t2m for the training period only.
    t2m_series = {}
    for station, filename in STATION_FILES.items():
        csv_path = os.path.join(DATA_DIR, filename)
        df   = pd.read_csv(csv_path, parse_dates=["date"])
        mask = (df["date"] >= cfg["train_start"]) & (df["date"] <= cfg["train_end"])
        s    = df.loc[mask, ["date", "t2m"]].set_index("date")["t2m"].dropna()
        t2m_series[station] = s

    stations   = list(STATION_FILES.keys())
    corr_matrix = np.zeros((N_STATIONS, N_STATIONS))

    for i, s1 in enumerate(stations):
        for j, s2 in enumerate(stations):
            if i == j:
                continue
            common = t2m_series[s1].index.intersection(t2m_series[s2].index)
            if len(common) < 2:
                continue
            t1   = t2m_series[s1].loc[common].values
            t2   = t2m_series[s2].loc[common].values
            corr = float(np.corrcoef(t1, t2)[0, 1])
            corr_matrix[i, j] = max(0.0, corr) if not np.isnan(corr) else 0.0

    np.fill_diagonal(corr_matrix, 0.0)
    return corr_matrix


def build_system_b_sigma(corr_matrix: np.ndarray,
                          D: np.ndarray,
                          sigma: float) -> np.ndarray:
    """
    Build System B adjacency matrix for a given sigma using a precomputed
    Pearson correlation matrix.

        A[i,j] = corr[i,j] * exp(-d² / (2*sigma²))   if d <= DMAX_KM
               = 0                                      otherwise

    Parameters
    ----------
    corr_matrix : clamped Pearson matrix from compute_pearson_matrix()
    D           : pairwise distance matrix [km]
    sigma       : Gaussian width [km]

    Returns
    -------
    np.ndarray of shape (9, 9)
    """
    A_dist, _ = build_graph(D, DMAX_KM, sigma)   # distance-only Gaussian
    A         = corr_matrix * A_dist
    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def run_baseline(data: dict) -> dict:
    """
    Fit an independent Ridge regression per station (alpha=1e-6 ≈ OLS).
    No graph, no collaboration.
    """
    weights = {}
    for station, splits in data.items():
        model = Ridge(alpha=1e-6, fit_intercept=False)
        model.fit(splits["X_train"], splits["y_train"])
        weights[station] = model.coef_
    return weights


# ---------------------------------------------------------------------------
# Hyperparameter search - alpha only (Baseline)
# ---------------------------------------------------------------------------

def hyperparameter_search_alpha(data: dict,
                                 adj_matrix: np.ndarray,
                                 system_name: str) -> tuple:
    """
    Grid search over ALPHA_CANDIDATES for a fixed adjacency matrix.
    Used by Baseline (no graph - adj_matrix ignored) via alpha ≈ 0.

    Returns best_alpha, list of (alpha, val_mse).
    """
    print(f"\n    Hyperparameter search — {system_name}")
    print(f"    {'Alpha':>10s}  {'Val MSE':>10s}  {'Val RMSE':>10s}")
    print("    " + "-" * 37)

    best_alpha = None
    best_mse   = np.inf
    results    = []

    for alpha in ALPHA_CANDIDATES:
        weights, _ = run_fl(data, adj_matrix, alpha=alpha, T=T_ROUNDS)
        _, mean_mse, mean_rmse = evaluate(data, weights, split="val")
        results.append((alpha, mean_mse))

        marker = "  ←" if mean_mse < best_mse else ""
        print(f"    {alpha:>10.4f}  {mean_mse:>10.4f}  {mean_rmse:>10.4f}{marker}")

        if mean_mse < best_mse:
            best_mse   = mean_mse
            best_alpha = alpha

    print(f"    Best alpha = {best_alpha}")
    return best_alpha, results


# ---------------------------------------------------------------------------
# Hyperparameter search - joint (alpha, sigma) for System A and B
# ---------------------------------------------------------------------------

def hyperparameter_search_joint(data: dict,
                                 D: np.ndarray,
                                 corr_matrix,   # np.ndarray or None
                                 system_name: str) -> tuple:
    """
    Joint grid search over ALPHA_CANDIDATES × SIGMA_CANDIDATES.

    For System A (corr_matrix=None): builds A = build_graph(D, DMAX_KM, sigma).
    For System B (corr_matrix given): builds A = corr_matrix * gaussian(sigma).

    Parameters
    ----------
    data        : prepared data dict
    D           : pairwise distance matrix [km]
    corr_matrix : Pearson matrix for System B; None for System A
    system_name : display label

    Returns
    -------
    best_alpha   : float
    best_sigma   : float
    grid_results : list of (alpha, sigma, val_mse) - all 84 evaluations
    """
    n_combos = len(ALPHA_CANDIDATES) * len(SIGMA_CANDIDATES)
    print(f"\n    Joint (alpha, sigma) search — {system_name}  [{n_combos} combinations]")
    print(f"    {'Alpha':>10s}  {'Sigma':>8s}  {'Val MSE':>10s}  {'Val RMSE':>10s}")
    print("    " + "-" * 47)

    best_alpha   = None
    best_sigma   = None
    best_mse     = np.inf
    grid_results = []

    for sigma in SIGMA_CANDIDATES:
        # Build adjacency matrix for this sigma.
        if corr_matrix is None:
            A, _ = build_graph(D, DMAX_KM, sigma)
        else:
            A = build_system_b_sigma(corr_matrix, D, sigma)

        for alpha in ALPHA_CANDIDATES:
            weights, _ = run_fl(data, A, alpha=alpha, T=T_ROUNDS)
            _, mean_mse, mean_rmse = evaluate(data, weights, split="val")
            grid_results.append((alpha, sigma, mean_mse))

            marker = "  ←" if mean_mse < best_mse else ""
            print(f"    {alpha:>10.4f}  {sigma:>8.1f}  "
                  f"{mean_mse:>10.4f}  {mean_rmse:>10.4f}{marker}")

            if mean_mse < best_mse:
                best_mse   = mean_mse
                best_alpha = alpha
                best_sigma = sigma

    print(f"    Best alpha = {best_alpha}  |  Best sigma = {best_sigma} km")
    return best_alpha, best_sigma, grid_results


# ---------------------------------------------------------------------------
# Run one complete experiment
# ---------------------------------------------------------------------------

def run_one_experiment(cfg: dict, D: np.ndarray) -> dict:
    """
    Run the full pipeline for a single experiment configuration:
      1. Prepare data with experiment-specific split dates.
      2. Compute Pearson correlation matrix for System B (training data only).
      3. Joint (alpha, sigma) search for System A and System B.
      4. Alpha-only search for Baseline.
      5. Final training with best parameters.
      6. Evaluate all three systems on train / val / test.

    Parameters
    ----------
    cfg : experiment config dict
    D   : pairwise distance matrix (precomputed, shared)

    Returns
    -------
    result dict with keys: data, weights_*, mse_history_*, metrics,
    per_station_*, best_alpha_a/b, best_sigma_a/b, A_a, A_b,
    grid_results_a/b.
    """
    print(f"\n  Preparing data ({cfg['train_start']} → {cfg['train_end']}) …")
    data    = prepare_experiment_data(cfg)
    station = list(data.keys())[0]
    print(f"  Split sizes (first station): "
          f"train={len(data[station]['y_train'])}  "
          f"val={len(data[station]['y_val'])}  "
          f"test={len(data[station]['y_test'])}")

    # Pearson correlation matrix for System B (computed once per experiment).
    print(f"  Computing Pearson correlation matrix "
          f"({cfg['train_start']} → {cfg['train_end']}) …")
    corr_matrix = compute_pearson_matrix(cfg, D)

    # Joint (alpha, sigma) search for System A.
    best_alpha_a, best_sigma_a, grid_a = hyperparameter_search_joint(
        data, D, corr_matrix=None, system_name="System A"
    )

    # Joint (alpha, sigma) search for System B.
    best_alpha_b, best_sigma_b, grid_b = hyperparameter_search_joint(
        data, D, corr_matrix=corr_matrix, system_name="System B"
    )

    # Alpha-only search for Baseline (sigma irrelevant - no graph).
    # Use a zero adjacency matrix so run_fl degenerates to local Ridge.
    A_zero = np.zeros((N_STATIONS, N_STATIONS))
    best_alpha_base, _ = hyperparameter_search_alpha(
        data, A_zero, system_name="Baseline"
    )

    # Build final adjacency matrices with best sigma.
    A_a, _ = build_graph(D, DMAX_KM, best_sigma_a)
    A_b     = build_system_b_sigma(corr_matrix, D, best_sigma_b)

    # Final training.
    print(f"\n    Final training — System A "
          f"(alpha={best_alpha_a}, sigma={best_sigma_a} km) …")
    weights_a, mse_history_a = run_fl(data, A_a, alpha=best_alpha_a, T=T_ROUNDS)

    print(f"    Final training — System B "
          f"(alpha={best_alpha_b}, sigma={best_sigma_b} km) …")
    weights_b, mse_history_b = run_fl(data, A_b, alpha=best_alpha_b, T=T_ROUNDS)

    print(f"    Fitting Baseline (alpha={best_alpha_base}) …")
    weights_base = run_baseline(data)

    # Evaluate all splits.
    metrics = {}
    for sys_name, weights in [("Baseline", weights_base),
                               ("System A", weights_a),
                               ("System B", weights_b)]:
        metrics[sys_name] = {}
        for split in ("train", "val", "test"):
            _, mean_mse, mean_rmse = evaluate(data, weights, split=split)
            metrics[sys_name][split] = (mean_mse, mean_rmse)

    per_station_base, _, _ = evaluate(data, weights_base, split="test")
    per_station_a,    _, _ = evaluate(data, weights_a,    split="test")
    per_station_b,    _, _ = evaluate(data, weights_b,    split="test")

    return {
        "data"             : data,
        "A_a"              : A_a,
        "A_b"              : A_b,
        "weights_base"     : weights_base,
        "weights_a"        : weights_a,
        "weights_b"        : weights_b,
        "mse_history_a"    : mse_history_a,
        "mse_history_b"    : mse_history_b,
        "metrics"          : metrics,
        "per_station_base" : per_station_base,
        "per_station_a"    : per_station_a,
        "per_station_b"    : per_station_b,
        "best_alpha_a"     : best_alpha_a,
        "best_alpha_b"     : best_alpha_b,
        "best_alpha_base"  : best_alpha_base,
        "best_sigma_a"     : best_sigma_a,
        "best_sigma_b"     : best_sigma_b,
        "grid_results_a"   : grid_a,
        "grid_results_b"   : grid_b,
    }


# ---------------------------------------------------------------------------
# Save per-experiment CSV
# ---------------------------------------------------------------------------

def save_experiment_csv(cfg: dict, result: dict) -> None:
    """
    Save a per-station results CSV for one experiment.

    Columns: station, system, train_mse, train_rmse, val_mse, val_rmse,
             test_mse, test_rmse, best_alpha, best_sigma.
    """
    rows     = []
    stations = list(result["data"].keys())

    for sys_name, weights, per_station in [
        ("Baseline", result["weights_base"], result["per_station_base"]),
        ("System A", result["weights_a"],    result["per_station_a"]),
        ("System B", result["weights_b"],    result["per_station_b"]),
    ]:
        alpha = (result["best_alpha_a"]    if sys_name == "System A"
                 else result["best_alpha_b"] if sys_name == "System B"
                 else result["best_alpha_base"])
        sigma = (result["best_sigma_a"]    if sys_name == "System A"
                 else result["best_sigma_b"] if sys_name == "System B"
                 else "-")

        for station in stations:
            tr_res,  _, _ = evaluate(result["data"], weights, split="train")
            val_res, _, _ = evaluate(result["data"], weights, split="val")

            rows.append({
                "station"    : station,
                "system"     : sys_name,
                "train_mse"  : round(tr_res[station]["mse"],   4),
                "train_rmse" : round(tr_res[station]["rmse"],  4),
                "val_mse"    : round(val_res[station]["mse"],  4),
                "val_rmse"   : round(val_res[station]["rmse"], 4),
                "test_mse"   : round(per_station[station]["mse"],  4),
                "test_rmse"  : round(per_station[station]["rmse"], 4),
                "best_alpha" : alpha,
                "best_sigma" : sigma,
            })

    df   = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, cfg["filename"])
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_test_rmse_by_station(results_base, results_a, results_b,
                               title, save_path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    stations  = list(results_base.keys())
    short     = [s.split()[0] for s in stations]
    rmse_base = [results_base[s]["rmse"] for s in stations]
    rmse_a    = [results_a[s]["rmse"]    for s in stations]
    rmse_b    = [results_b[s]["rmse"]    for s in stations]

    x     = np.arange(len(stations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, rmse_base, width, label="Baseline (local)",
           color="#d62728", alpha=0.85)
    ax.bar(x,          rmse_a,   width, label="System A (distance)",
           color="#1f77b4", alpha=0.85)
    ax.bar(x + width,  rmse_b,   width, label="System B (correlation)",
           color="#2ca02c", alpha=0.85)

    ax.set_xlabel("Station", fontsize=11)
    ax.set_ylabel("Test RMSE (°C)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(rmse_base), max(rmse_a), max(rmse_b)) * 1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_test_rmse_by_experiment(all_results, save_path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    exp_labels = [cfg["label"] for cfg, _ in all_results]
    rmse_base  = [r["metrics"]["Baseline"]["test"][1] for _, r in all_results]
    rmse_a     = [r["metrics"]["System A"]["test"][1] for _, r in all_results]
    rmse_b     = [r["metrics"]["System B"]["test"][1] for _, r in all_results]

    x = np.arange(len(exp_labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, rmse_base, "o-",  label="Baseline (local)",
            color="#d62728", linewidth=2, markersize=8)
    ax.plot(x, rmse_a,    "s-",  label="System A (distance)",
            color="#1f77b4", linewidth=2, markersize=8)
    ax.plot(x, rmse_b,    "^--", label="System B (correlation)",
            color="#2ca02c", linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=10)
    ax.set_xlabel("Experiment (decreasing training set size)", fontsize=11)
    ax.set_ylabel("Mean Test RMSE (°C)", fontsize=11)
    ax.set_title("Mean Test RMSE vs Training Set Size\n"
                 "Baseline vs System A vs System B",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_convergence(mse_history_a, mse_history_b, title, save_path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    rounds = np.arange(1, len(mse_history_a) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, mse_history_a, label="System A (distance)",
            color="#1f77b4", linewidth=2)
    ax.plot(rounds, mse_history_b, label="System B (correlation)",
            color="#2ca02c", linewidth=2, linestyle="--")
    ax.set_xlabel("FL Round", fontsize=11)
    ax.set_ylabel("Mean Training MSE (°C²)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_sigma_heatmap(grid_results: list, save_path: str) -> None:
    """
    Heatmap of mean validation RMSE over the (alpha, sigma) grid.

    alpha on the y-axis (log scale labels), sigma on the x-axis.
    Used for Experiment 3 System A to visualise joint sensitivity.

    Parameters
    ----------
    grid_results : list of (alpha, sigma, val_mse) from hyperparameter_search_joint
    save_path    : output PNG path
    """
    # Build 2-D grid: rows = alpha, cols = sigma.
    alphas = sorted(set(a for a, _, _ in grid_results))
    sigmas = sorted(set(s for _, s, _ in grid_results))

    grid = np.full((len(alphas), len(sigmas)), np.nan)
    for alpha, sigma, val_mse in grid_results:
        i = alphas.index(alpha)
        j = sigmas.index(sigma)
        grid[i, j] = np.sqrt(val_mse)   # convert MSE → RMSE

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd",
                   origin="lower",
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))

    # Annotate cells with RMSE value.
    for i in range(len(alphas)):
        for j in range(len(sigmas)):
            ax.text(j, i, f"{grid[i, j]:.3f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if grid[i, j] < np.nanmedian(grid) else "white")

    # Mark the best cell.
    best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
    ax.add_patch(plt.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        fill=False, edgecolor="blue", linewidth=2.5, label="Best"
    ))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Val RMSE (°C)", fontsize=10)

    # Axis labels - alpha shown with scientific notation for readability.
    alpha_labels = [f"{a:.4g}" for a in alphas]
    sigma_labels = [f"{int(s)}" for s in sigmas]

    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(sigma_labels, fontsize=9)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels(alpha_labels, fontsize=8)

    ax.set_xlabel("sigma (km)", fontsize=11)
    ax.set_ylabel("alpha", fontsize=11)
    ax.set_title(
        "Validation RMSE Heatmap — Experiment 3, System A\n"
        f"Joint (alpha, sigma) search  |  d_max = {DMAX_KM:.0f} km",
        fontsize=12, fontweight="bold",
    )
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fill=False,
                                      edgecolor="blue", linewidth=2)],
              labels=["Best (alpha, sigma)"], loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Print and save combined summary
# ---------------------------------------------------------------------------

def print_combined_summary(all_results: list) -> pd.DataFrame:
    """
    Print the combined summary table and return it as a DataFrame.

    Columns: Experiment, System, Train RMSE, Val RMSE, Test RMSE,
             Best Alpha, Best Sigma.
    """
    header = (f"\n  {'Experiment':<16s} {'System':<12s} "
              f"{'Train RMSE':>11s} {'Val RMSE':>10s} "
              f"{'Test RMSE':>10s} {'Best Alpha':>11s} {'Best Sigma':>11s}")
    print(header)
    print("  " + "-" * 85)

    rows = []
    for cfg, result in all_results:
        for sys_name in ["Baseline", "System A", "System B"]:
            tr_rmse  = result["metrics"][sys_name]["train"][1]
            val_rmse = result["metrics"][sys_name]["val"][1]
            te_rmse  = result["metrics"][sys_name]["test"][1]

            if sys_name == "System A":
                alpha = result["best_alpha_a"]
                sigma = result["best_sigma_a"]
            elif sys_name == "System B":
                alpha = result["best_alpha_b"]
                sigma = result["best_sigma_b"]
            else:
                alpha = result["best_alpha_base"]
                sigma = None

            alpha_str = f"{alpha}"
            sigma_str = f"{sigma} km" if sigma is not None else "-"

            print(f"  {cfg['label']:<16s} {sys_name:<12s} "
                  f"{tr_rmse:>11.4f} {val_rmse:>10.4f} "
                  f"{te_rmse:>10.4f} {alpha_str:>11s} {sigma_str:>11s}")

            rows.append({
                "experiment" : cfg["label"],
                "system"     : sys_name,
                "train_rmse" : round(tr_rmse,  4),
                "val_rmse"   : round(val_rmse, 4),
                "test_rmse"  : round(te_rmse,  4),
                "best_alpha" : alpha,
                "best_sigma" : sigma if sigma is not None else "-",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  FL Project — Phase 6: Running experiments")
    print(f"  Experiments      : {len(EXPERIMENTS)}")
    print(f"  d_max (fixed)    : {DMAX_KM} km")
    print(f"  Alpha candidates : {ALPHA_CANDIDATES}")
    print(f"  Sigma candidates : {SIGMA_CANDIDATES} km")
    print(f"  Grid size        : {len(ALPHA_CANDIDATES)} × {len(SIGMA_CANDIDATES)} "
          f"= {len(ALPHA_CANDIDATES)*len(SIGMA_CANDIDATES)} combinations")
    print(f"  FL rounds        : {T_ROUNDS}")
    print("=" * 65)

    D = build_distance_matrix()

    all_results = []

    for cfg in EXPERIMENTS:
        print("\n" + "=" * 65)
        print(f"  Running Experiment {cfg['id']} — {cfg['label']}")
        print(f"  Train : {cfg['train_start']} → {cfg['train_end']}")
        print(f"  Val   : {cfg['val_start']}   → {cfg['val_end']}")
        print(f"  Test  : {cfg['test_start']}  → {cfg['test_end']}")
        print("=" * 65)

        result = run_one_experiment(cfg, D)
        all_results.append((cfg, result))

        # Print per-experiment results table.
        print(f"\n  Results — Experiment {cfg['id']} ({cfg['label']})")
        metrics = result["metrics"]
        print(f"  {'System':<12s} {'Train MSE':>10s} {'Train RMSE':>11s} "
              f"{'Val MSE':>10s} {'Val RMSE':>10s} "
              f"{'Test MSE':>10s} {'Test RMSE':>10s}")
        print("  " + "-" * 76)
        for sys_name in ["Baseline", "System A", "System B"]:
            tr  = metrics[sys_name]["train"]
            val = metrics[sys_name]["val"]
            te  = metrics[sys_name]["test"]
            print(f"  {sys_name:<12s} {tr[0]:>10.4f} {tr[1]:>11.4f} "
                  f"{val[0]:>10.4f} {val[1]:>10.4f} "
                  f"{te[0]:>10.4f} {te[1]:>10.4f}")

        save_experiment_csv(cfg, result)

        # HDD analysis - Experiment 1 only.
        if cfg["id"] == 1:
            print("\n" + "=" * 65)
            print("  HDD Analysis — Experiment 1 (Test Set)")
            print("=" * 65)
            hdd_analysis.run(
                data         = result["data"],
                weights_base = result["weights_base"],
                weights_a    = result["weights_a"],
                weights_b    = result["weights_b"],
            )
            # Update .npy files with best-sigma adjacency matrices.
            np.save(os.path.join(DATA_DIR, "adj_system_a.npy"), result["A_a"])
            np.save(os.path.join(DATA_DIR, "adj_system_b.npy"), result["A_b"])
            print(f"  Updated adj_system_a.npy  (sigma={result['best_sigma_a']} km)")
            print(f"  Updated adj_system_b.npy  (sigma={result['best_sigma_b']} km)")

        # Sigma heatmap - Experiment 3, System A only.
        if cfg["id"] == 3:
            plot_sigma_heatmap(
                result["grid_results_a"],
                save_path=os.path.join(RESULTS_DIR, "sigma_heatmap_exp3.png"),
            )

        exp_id = cfg["id"]
        plot_test_rmse_by_station(
            result["per_station_base"],
            result["per_station_a"],
            result["per_station_b"],
            title=(f"Test RMSE per Station — Experiment {exp_id}: {cfg['label']}\n"
                   f"Train: {cfg['train_start']} → {cfg['train_end']}"),
            save_path=os.path.join(RESULTS_DIR,
                                   f"test_rmse_by_station_exp{exp_id}.png"),
        )

    # Convergence plot — Experiment 1 only.
    _, result_exp1 = all_results[0]
    cfg_exp1       = all_results[0][0]
    plot_convergence(
        result_exp1["mse_history_a"],
        result_exp1["mse_history_b"],
        title=(f"GTVMin Convergence — Experiment 1: {cfg_exp1['label']}\n"
               f"Mean Training MSE vs FL Round"),
        save_path=os.path.join(RESULTS_DIR, "convergence_exp1.png"),
    )

    plot_test_rmse_by_experiment(
        all_results,
        save_path=os.path.join(RESULTS_DIR, "test_rmse_by_experiment.png"),
    )

    print("\n" + "=" * 65)
    print("  Combined Summary — All Experiments")
    print("=" * 65)
    summary_df = print_combined_summary(all_results)

    summary_path = os.path.join(RESULTS_DIR, "combined_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved → {summary_path}")

    print("\n✓ Phase 6 complete.")


if __name__ == "__main__":
    main()
