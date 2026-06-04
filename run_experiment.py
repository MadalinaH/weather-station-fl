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
  Candidates : [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
  Criterion  : mean validation MSE (test set never seen during tuning)
  Best alpha selected independently per system per experiment.

OUTPUT FILES
------------
  results/experiment1_full_data.csv
  results/experiment2_reduced_data.csv
  results/experiment3_minimal_data.csv
  results/combined_summary.csv
  results/test_rmse_by_experiment.png
  results/test_rmse_by_station_exp1.png
  results/test_rmse_by_station_exp2.png
  results/test_rmse_by_station_exp3.png
  results/convergence_exp1.png

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

# Reuse data-preparation helpers from prepare_data.py (no modification needed).
from prepare_data import load_station, build_xy, split_by_date, normalise

# Reuse graph-building helpers from build_network.py.
from build_network import (
    build_distance_matrix,
    build_adj_system_a,
    build_adj_system_b,
    STATIONS as STATION_NAMES,
    STATION_FILES,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Extended alpha search range.
ALPHA_CANDIDATES = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
                    10.0, 50.0, 100.0, 500.0]

T_ROUNDS    = 50
RANDOM_SEED = 42

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

np.random.seed(RANDOM_SEED)

# Three experiment configurations.
# Each entry: (label, train_start, train_end, val_start, val_end, test_start, test_end)
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
# Data preparation for a specific experiment
# ---------------------------------------------------------------------------

def prepare_experiment_data(cfg: dict) -> dict:
    """
    Build train/val/test splits for all 9 stations using the date boundaries
    defined in experiment config cfg.

    Reuses load_station, build_xy, split_by_date, and normalise from
    prepare_data.py - no data is modified, only the split boundaries change.

    Parameters
    ----------
    cfg : one entry from EXPERIMENTS dict

    Returns
    -------
    Prepared data dict {station_name: {X_train, y_train, X_val, y_val,
                                       X_test, y_test}}
    """
    data = {}
    for station, filename in STATION_FILES.items():
        csv_path = os.path.join(DATA_DIR, filename)
        df                = load_station(csv_path)
        X, y, dates       = build_xy(df)
        splits            = split_by_date(X, y, dates,
                                          train_start=cfg["train_start"],
                                          train_end=cfg["train_end"],
                                          val_start=cfg["val_start"],
                                          val_end=cfg["val_end"],
                                          test_start=cfg["test_start"],
                                          test_end=cfg["test_end"])
        splits            = normalise(splits)
        data[station]     = splits
    return data


# ---------------------------------------------------------------------------
# Build System B adjacency matrix for a specific training period
# ---------------------------------------------------------------------------

def build_system_b_for_experiment(cfg: dict, D: np.ndarray) -> np.ndarray:
    """
    Compute System B adjacency matrix using only the training data from
    this experiment's training period.

    System B weights include Pearson correlation, which must be computed
    on training data only to avoid leaking future information into the
    graph structure.  Each experiment has a different training window,
    so we recompute the correlation - and therefore the full adj matrix -
    for each experiment separately.

    Parameters
    ----------
    cfg : experiment config dict (contains train_start / train_end)
    D   : pairwise distance matrix (shared across experiments)

    Returns
    -------
    np.ndarray of shape (9, 9) - System B adjacency matrix for this experiment
    """
    # Load t2m for the training period of this experiment only.
    t2m_series = {}
    for station, filename in STATION_FILES.items():
        csv_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(csv_path, parse_dates=["date"])
        mask = ((df["date"] >= cfg["train_start"]) &
                (df["date"] <= cfg["train_end"]))
        s = df.loc[mask, ["date", "t2m"]].set_index("date")["t2m"].dropna()
        t2m_series[station] = s

    return build_adj_system_b(D, t2m_series)


# ---------------------------------------------------------------------------
# Baseline: local Ridge per station, no collaboration
# ---------------------------------------------------------------------------

def run_baseline(data: dict) -> dict:
    """
    Fit an independent Ridge regression per station on its own training
    data only.  No graph, no collaboration.

    Uses alpha=1e-6 to approximate OLS while keeping Ridge numerically
    stable.

    Parameters
    ----------
    data : prepared data dict

    Returns
    -------
    dict {station_name: weight vector np.ndarray shape (3,)}
    """
    weights = {}
    for station, splits in data.items():
        model = Ridge(alpha=1e-6, fit_intercept=False)
        model.fit(splits["X_train"], splits["y_train"])
        weights[station] = model.coef_
    return weights


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def hyperparameter_search(data: dict,
                          adj_matrix: np.ndarray,
                          system_name: str) -> tuple:
    """
    Grid search over ALPHA_CANDIDATES, choosing the alpha with the lowest
    mean validation MSE.

    Parameters
    ----------
    data        : prepared data dict
    adj_matrix  : adjacency matrix for the system being tuned
    system_name : display label ('System A' or 'System B')

    Returns
    -------
    best_alpha   : float
    val_mse_list : list of (alpha, val_mse) for all candidates
    """
    print(f"\n    Hyperparameter search — {system_name}")
    print(f"    {'Alpha':>10s}  {'Val MSE':>10s}  {'Val RMSE':>10s}")
    print("    " + "-" * 37)

    val_mse_list = []
    best_alpha   = None
    best_mse     = np.inf

    for alpha in ALPHA_CANDIDATES:
        weights, _ = run_fl(data, adj_matrix, alpha=alpha, T=T_ROUNDS)
        _, mean_mse, mean_rmse = evaluate(data, weights, split="val")
        val_mse_list.append((alpha, mean_mse))

        marker = "  ←" if mean_mse < best_mse else ""
        print(f"    {alpha:>10.4f}  {mean_mse:>10.4f}  {mean_rmse:>10.4f}{marker}")

        if mean_mse < best_mse:
            best_mse   = mean_mse
            best_alpha = alpha

    print(f"    Best alpha = {best_alpha}")
    return best_alpha, val_mse_list


# ---------------------------------------------------------------------------
# Run one complete experiment
# ---------------------------------------------------------------------------

def run_one_experiment(cfg: dict, D: np.ndarray, A_a: np.ndarray) -> dict:
    """
    Run the full pipeline for a single experiment configuration:
      1. Prepare data with experiment-specific split dates.
      2. Build System B adj matrix on this experiment's training data.
      3. Hyperparameter search for System A and System B.
      4. Final training with best alpha.
      5. Baseline fitting.
      6. Evaluate all three systems on train / val / test.

    Parameters
    ----------
    cfg : experiment config dict
    D   : pairwise distance matrix (precomputed, shared)
    A_a : System A adjacency matrix (fixed, shared across experiments)

    Returns
    -------
    dict with keys:
      data          : prepared data for this experiment
      A_b           : System B adj matrix recomputed for this experiment
      weights_*     : trained weight dicts
      mse_history_* : training MSE histories
      metrics       : {system: {split: (mse, rmse)}}
      per_station_* : per-station evaluate() dicts for test split
      best_alpha_a/b: chosen alpha values
    """
    label = cfg["label"]
    print(f"\n  Preparing data ({cfg['train_start']} → {cfg['train_end']}) …")
    data = prepare_experiment_data(cfg)

    n_train = len(data[list(data.keys())[0]]["y_train"])
    n_val   = len(data[list(data.keys())[0]]["y_val"])
    n_test  = len(data[list(data.keys())[0]]["y_test"])
    print(f"  Split sizes (first station): "
          f"train={n_train}  val={n_val}  test={n_test}")

    # System B adj matrix recomputed on this experiment's training data.
    print(f"  Building System B graph (corr on {cfg['train_start']}–{cfg['train_end']}) …")
    A_b = build_system_b_for_experiment(cfg, D)
    n_edges_b = int(np.sum(np.triu(A_b, k=1) > 0))
    print(f"  System B edges: {n_edges_b}")

    # Hyperparameter search.
    best_alpha_a, _ = hyperparameter_search(data, A_a, "System A")
    best_alpha_b, _ = hyperparameter_search(data, A_b, "System B")

    # Final training.
    print(f"\n    Final training — System A (alpha={best_alpha_a}) …")
    weights_a, mse_history_a = run_fl(data, A_a, alpha=best_alpha_a, T=T_ROUNDS)

    print(f"    Final training — System B (alpha={best_alpha_b}) …")
    weights_b, mse_history_b = run_fl(data, A_b, alpha=best_alpha_b, T=T_ROUNDS)

    print(f"    Fitting Baseline …")
    weights_base = run_baseline(data)

    # Evaluate all three systems on all splits.
    metrics = {}
    for sys_name, weights in [("Baseline", weights_base),
                               ("System A", weights_a),
                               ("System B", weights_b)]:
        metrics[sys_name] = {}
        for split in ("train", "val", "test"):
            _, mean_mse, mean_rmse = evaluate(data, weights, split=split)
            metrics[sys_name][split] = (mean_mse, mean_rmse)

    # Per-station test results (for bar charts).
    per_station_base, _, _ = evaluate(data, weights_base, split="test")
    per_station_a,    _, _ = evaluate(data, weights_a,    split="test")
    per_station_b,    _, _ = evaluate(data, weights_b,    split="test")

    return {
        "data"             : data,
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
    }


# ---------------------------------------------------------------------------
# Save per-experiment CSV
# ---------------------------------------------------------------------------

def save_experiment_csv(cfg: dict, result: dict) -> None:
    """
    Save a per-station results CSV for one experiment.

    Columns: station, system, train_mse, train_rmse, val_mse, val_rmse,
             test_mse, test_rmse, best_alpha.

    Parameters
    ----------
    cfg    : experiment config dict
    result : output of run_one_experiment()
    """
    rows = []
    stations = list(result["data"].keys())

    for sys_name, weights, per_station in [
        ("Baseline", result["weights_base"], result["per_station_base"]),
        ("System A", result["weights_a"],    result["per_station_a"]),
        ("System B", result["weights_b"],    result["per_station_b"]),
    ]:
        alpha = (result["best_alpha_a"] if sys_name == "System A"
                 else result["best_alpha_b"] if sys_name == "System B"
                 else None)

        for station in stations:
            # Per-station train and val metrics.
            tr_res, _, _  = evaluate(result["data"], weights, split="train")
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
                "best_alpha" : alpha if alpha is not None else "-",
            })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, cfg["filename"])
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_test_rmse_by_station(results_base: dict,
                               results_a: dict,
                               results_b: dict,
                               title: str,
                               save_path: str) -> None:
    """
    Grouped bar chart: test RMSE per station, three bars per station
    (Baseline / System A / System B).

    Parameters
    ----------
    results_base : per-station test evaluate() dict for Baseline
    results_a    : per-station test evaluate() dict for System A
    results_b    : per-station test evaluate() dict for System B
    title        : plot title string
    save_path    : output PNG path
    """
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


def plot_test_rmse_by_experiment(all_results: list, save_path: str) -> None:
    """
    Line plot: mean test RMSE vs experiment (Full / Reduced / Minimal),
    one line per system.  Shows how FL benefit changes with training size.

    Parameters
    ----------
    all_results : list of (cfg, result) pairs in experiment order
    save_path   : output PNG path
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    exp_labels = [cfg["label"] for cfg, _ in all_results]
    rmse_base  = [r["metrics"]["Baseline"]["test"][1] for _, r in all_results]
    rmse_a     = [r["metrics"]["System A"]["test"][1] for _, r in all_results]
    rmse_b     = [r["metrics"]["System B"]["test"][1] for _, r in all_results]

    x = np.arange(len(exp_labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, rmse_base, "o-", label="Baseline (local)",
            color="#d62728", linewidth=2, markersize=8)
    ax.plot(x, rmse_a,    "s-", label="System A (distance)",
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


def plot_convergence(mse_history_a: list,
                     mse_history_b: list,
                     title: str,
                     save_path: str) -> None:
    """
    Line plot of mean training MSE vs FL round for System A and System B.

    Parameters
    ----------
    mse_history_a : per-round mean train MSE list for System A
    mse_history_b : per-round mean train MSE list for System B
    title         : plot title string
    save_path     : output PNG path
    """
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


# ---------------------------------------------------------------------------
# Print and save combined summary
# ---------------------------------------------------------------------------

def print_combined_summary(all_results: list) -> pd.DataFrame:
    """
    Print the combined summary table and return it as a DataFrame.

    Columns: Experiment, System, Train RMSE, Val RMSE, Test RMSE, Best Alpha.

    Parameters
    ----------
    all_results : list of (cfg, result) pairs

    Returns
    -------
    pd.DataFrame - the combined summary (also printed to console)
    """
    header = (f"\n  {'Experiment':<16s} {'System':<12s} "
              f"{'Train RMSE':>11s} {'Val RMSE':>10s} "
              f"{'Test RMSE':>10s} {'Best Alpha':>11s}")
    print(header)
    print("  " + "-" * 73)

    rows = []
    for cfg, result in all_results:
        for sys_name in ["Baseline", "System A", "System B"]:
            tr_rmse  = result["metrics"][sys_name]["train"][1]
            val_rmse = result["metrics"][sys_name]["val"][1]
            te_rmse  = result["metrics"][sys_name]["test"][1]
            alpha    = (result["best_alpha_a"] if sys_name == "System A"
                        else result["best_alpha_b"] if sys_name == "System B"
                        else None)
            alpha_str = f"{alpha}" if alpha is not None else "-"

            print(f"  {cfg['label']:<16s} {sys_name:<12s} "
                  f"{tr_rmse:>11.4f} {val_rmse:>10.4f} "
                  f"{te_rmse:>10.4f} {alpha_str:>11s}")

            rows.append({
                "experiment" : cfg["label"],
                "system"     : sys_name,
                "train_rmse" : round(tr_rmse,  4),
                "val_rmse"   : round(val_rmse, 4),
                "test_rmse"  : round(te_rmse,  4),
                "best_alpha" : alpha if alpha is not None else "-",
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
    print(f"  Alpha candidates : {ALPHA_CANDIDATES}")
    print(f"  FL rounds        : {T_ROUNDS}")
    print("=" * 65)

    # Precompute the distance matrix once (shared across all experiments).
    print("\n  Precomputing pairwise Haversine distances …")
    D   = build_distance_matrix()
    A_a = build_adj_system_a(D)
    n_edges_a = int(np.sum(np.triu(A_a, k=1) > 0))
    print(f"  System A edges: {n_edges_a}")

    # -----------------------------------------------------------------------
    # Run all three experiments
    # -----------------------------------------------------------------------
    all_results = []   # list of (cfg, result) for summary/plotting

    for cfg in EXPERIMENTS:
        print("\n" + "=" * 65)
        print(f"  Running Experiment {cfg['id']} — {cfg['label']}")
        print(f"  Train : {cfg['train_start']} → {cfg['train_end']}")
        print(f"  Val   : {cfg['val_start']}   → {cfg['val_end']}")
        print(f"  Test  : {cfg['test_start']}  → {cfg['test_end']}")
        print("=" * 65)

        result = run_one_experiment(cfg, D, A_a)
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

        # Save per-experiment CSV.
        save_experiment_csv(cfg, result)

        # Per-experiment bar chart.
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

    # -----------------------------------------------------------------------
    # Convergence plot for Experiment 1 only (as specified)
    # -----------------------------------------------------------------------
    _, result_exp1 = all_results[0]
    cfg_exp1       = all_results[0][0]
    plot_convergence(
        result_exp1["mse_history_a"],
        result_exp1["mse_history_b"],
        title=(f"GTVMin Convergence — Experiment 1: {cfg_exp1['label']}\n"
               f"Mean Training MSE vs FL Round"),
        save_path=os.path.join(RESULTS_DIR, "convergence_exp1.png"),
    )

    # -----------------------------------------------------------------------
    # Cross-experiment line plot
    # -----------------------------------------------------------------------
    plot_test_rmse_by_experiment(
        all_results,
        save_path=os.path.join(RESULTS_DIR, "test_rmse_by_experiment.png"),
    )

    # -----------------------------------------------------------------------
    # Combined summary table
    # -----------------------------------------------------------------------
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
