"""
hdd_analysis.py
===============
Post-processing step: convert predicted next-day temperatures into
Heating Degree Days (HDD) and compare against actual HDD.

WHAT IS HDD?
------------
Heating Degree Days (HDD) quantify the daily energy demand for space
heating.  A higher HDD means more heating is required that day.

Finnish Meteorological Institute standard formula:
    HDD = max(17 - T, 0)   if T < 12 °C
    HDD = 0                 otherwise

Where T is the daily mean temperature in °C.
  - Indoor baseline: 17 °C  (target indoor temperature)
  - Heating threshold: 12 °C (heating assumed to start below this)

WHY HDD MATTERS FOR FL?
------------------------
Temperature prediction errors translate directly into HDD estimation
errors, which affect energy planning and building management systems.
Computing HDD from FL predictions gives a practical, interpretable
measure of downstream utility - beyond raw RMSE.

WHAT THIS SCRIPT DOES
----------------------
Uses Experiment 1 (full data, 2022-2023 training) settings:
  - Baseline  : local Ridge per station (alpha=1e-6)
  - System A  : GTVMin with distance graph  (alpha=50, T=50)
  - System B  : GTVMin with correlation graph (alpha=50, T=50)

For each station and system, on the TEST set:
  1. Computes predicted temperatures: y_pred = X_test @ w[i]
  2. Converts predictions to daily HDD
  3. Converts true labels to actual daily HDD
  4. Reports mean daily HDD and HDD MAE per station per system

OUTPUT
------
  results/hdd_analysis.csv      per-station HDD metrics
  results/hdd_per_station.png   grouped bar chart

Usage:
    python hdd_analysis.py
    (also called from run_experiment.py at the end of Experiment 1)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from fl_algorithm import run_fl
from build_network import (
    build_distance_matrix,
    build_adj_system_a,
    build_adj_system_b,
    STATION_FILES,
)

# ---------------------------------------------------------------------------
# Configuration - mirrors Experiment 1 best-alpha values from run_experiment.py
# ---------------------------------------------------------------------------

# Best alpha found by hyperparameter search in Experiment 1.
ALPHA_SYSTEM_A = 50.0
ALPHA_SYSTEM_B = 50.0
T_ROUNDS       = 50

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# HDD computation
# ---------------------------------------------------------------------------

def compute_hdd(temperatures: np.ndarray) -> np.ndarray:
    """
    Compute daily Heating Degree Days from an array of mean temperatures.

    Finnish Meteorological Institute standard:
        HDD = max(17 - T, 0)   if T < 12 °C
        HDD = 0                 otherwise

    Parameters
    ----------
    temperatures : np.ndarray of daily mean temperatures [°C], shape (n,)

    Returns
    -------
    np.ndarray of daily HDD values [°C·days], shape (n,), all non-negative.
    """
    hdd = np.where(
        temperatures < 12.0,          # heating threshold
        np.maximum(17.0 - temperatures, 0.0),  # indoor baseline 17°C
        0.0,
    )
    return hdd


# ---------------------------------------------------------------------------
# Baseline weights
# ---------------------------------------------------------------------------

def fit_baseline(data: dict) -> dict:
    """
    Fit local Ridge regression per station (no collaboration).

    Uses alpha=1e-6 to approximate OLS while keeping Ridge numerically
    stable.  Matches exactly the Baseline in run_experiment.py.

    Parameters
    ----------
    data : prepared data dict from prepared_data.pkl

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
# HDD analysis core
# ---------------------------------------------------------------------------

def run_hdd_analysis(data: dict,
                     weights_base: dict,
                     weights_a: dict,
                     weights_b: dict) -> pd.DataFrame:
    """
    Compute HDD metrics for all stations and all three systems on the
    test split.

    For each station and system:
      - Predicts test temperatures: y_pred = X_test @ w[i]
      - Converts predictions → daily HDD
      - Converts true labels  → actual daily HDD
      - Computes mean daily HDD (predicted and actual)
      - Computes HDD MAE = mean(|HDD_pred - HDD_actual|)

    Parameters
    ----------
    data         : prepared data dict (contains X_test and y_test per station)
    weights_base : Baseline weight dict
    weights_a    : System A weight dict
    weights_b    : System B weight dict

    Returns
    -------
    pd.DataFrame with columns:
        station, system, mean_actual_hdd, mean_predicted_hdd, hdd_mae
    """
    rows = []

    for station in data:
        X_test = data[station]["X_test"]   # shape (n_test, 3)
        y_test = data[station]["y_test"]   # shape (n_test,)

        # Actual HDD from true labels - same for all systems.
        hdd_actual = compute_hdd(y_test)
        mean_actual_hdd = float(np.mean(hdd_actual))

        for sys_name, weights in [("Baseline", weights_base),
                                   ("System A", weights_a),
                                   ("System B", weights_b)]:
            y_pred   = X_test @ weights[station]     # predicted temperatures
            hdd_pred = compute_hdd(y_pred)           # convert to HDD

            mean_pred_hdd = float(np.mean(hdd_pred))
            hdd_mae       = float(np.mean(np.abs(hdd_pred - hdd_actual)))

            rows.append({
                "station"           : station,
                "system"            : sys_name,
                "mean_actual_hdd"   : round(mean_actual_hdd,  3),
                "mean_predicted_hdd": round(mean_pred_hdd,    3),
                "hdd_mae"           : round(hdd_mae,          3),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_hdd_table(df: pd.DataFrame) -> None:
    """
    Print a formatted HDD summary table to the console.

    One row per station, columns = Actual / Baseline / Sys A / Sys B HDD
    and MAE for System A.

    Parameters
    ----------
    df : output of run_hdd_analysis()
    """
    stations = df["station"].unique()

    header = (f"\n  {'Station':<25s} | {'Actual':>10s} | "
              f"{'Baseline':>12s} | {'Sys A':>9s} | {'Sys B':>9s} | "
              f"{'MAE (A)':>9s}")
    print(header)
    print("  " + "-" * 83)

    for station in stations:
        sub     = df[df["station"] == station].set_index("system")
        actual  = sub.loc["Baseline", "mean_actual_hdd"]   # same for all
        base_p  = sub.loc["Baseline", "mean_predicted_hdd"]
        a_p     = sub.loc["System A", "mean_predicted_hdd"]
        b_p     = sub.loc["System B", "mean_predicted_hdd"]
        mae_a   = sub.loc["System A", "hdd_mae"]

        print(f"  {station:<25s} | {actual:>10.3f} | "
              f"{base_p:>12.3f} | {a_p:>9.3f} | {b_p:>9.3f} | "
              f"{mae_a:>9.3f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_hdd_per_station(df: pd.DataFrame, save_path: str) -> None:
    """
    Grouped bar chart: mean daily HDD per station for Actual / Baseline /
    System A / System B.

    One group per station, four bars per group.
    y-axis: Mean Daily HDD (°C·days).

    Parameters
    ----------
    df        : output of run_hdd_analysis()
    save_path : absolute path for the output PNG
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    stations = list(df["station"].unique())
    short    = [s.split()[0] for s in stations]

    # Extract values in station order.
    actual_hdd  = [df[(df["station"] == s) & (df["system"] == "Baseline")]
                   ["mean_actual_hdd"].values[0] for s in stations]
    base_hdd    = [df[(df["station"] == s) & (df["system"] == "Baseline")]
                   ["mean_predicted_hdd"].values[0] for s in stations]
    a_hdd       = [df[(df["station"] == s) & (df["system"] == "System A")]
                   ["mean_predicted_hdd"].values[0] for s in stations]
    b_hdd       = [df[(df["station"] == s) & (df["system"] == "System B")]
                   ["mean_predicted_hdd"].values[0] for s in stations]

    x     = np.arange(len(stations))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.bar(x - 1.5*width, actual_hdd, width, label="Actual",
           color="#555555", alpha=0.85)
    ax.bar(x - 0.5*width, base_hdd,   width, label="Baseline (local)",
           color="#d62728", alpha=0.85)
    ax.bar(x + 0.5*width, a_hdd,      width, label="System A (distance)",
           color="#1f77b4", alpha=0.85)
    ax.bar(x + 1.5*width, b_hdd,      width, label="System B (correlation)",
           color="#2ca02c", alpha=0.85)

    ax.set_xlabel("Station", fontsize=11)
    ax.set_ylabel("Mean Daily HDD (°C·days)", fontsize=11)
    ax.set_title(
        "Mean Daily Heating Degree Days — Test Set (Experiment 1)\n"
        "FMI standard: HDD = max(17 - T, 0) if T < 12 °C, else 0",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(actual_hdd), max(base_hdd),
                       max(a_hdd), max(b_hdd)) * 1.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

    # Caption printed to console.
    print(
        "\n  Figure caption:\n"
        "  Mean daily Heating Degree Days (HDD) per station on the Experiment 1\n"
        "  test set (2024-07-01 → 2024-12-31). HDD is computed from predicted\n"
        "  next-day temperatures using the FMI standard (indoor baseline 17 °C,\n"
        "  heating threshold 12 °C). Bars show Actual HDD (from true labels)\n"
        "  alongside Baseline, System A, and System B predictions."
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def run(data: dict = None,
        weights_base: dict = None,
        weights_a: dict = None,
        weights_b: dict = None) -> pd.DataFrame:
    """
    Run the full HDD analysis pipeline.

    Can be called in two ways:
      1. With pre-computed weights (from run_experiment.py) - no re-training.
      2. Without arguments — loads data and re-trains from scratch using
         Experiment 1 best-alpha values (standalone mode).

    Parameters
    ----------
    data         : prepared data dict (optional - loaded from pkl if None)
    weights_base : Baseline weights (optional - recomputed if None)
    weights_a    : System A weights (optional - recomputed if None)
    weights_b    : System B weights (optional - recomputed if None)

    Returns
    -------
    pd.DataFrame - HDD analysis results (also saved to CSV and PNG)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data if not passed in
    if data is None:
        print("  Loading prepared_data.pkl …")
        with open(os.path.join(DATA_DIR, "prepared_data.pkl"), "rb") as f:
            data = pickle.load(f)

    # Train weights if not passed in
    if weights_base is None or weights_a is None or weights_b is None:
        print("  Building distance matrix and adjacency matrices …")
        D   = build_distance_matrix()
        A_a = build_adj_system_a(D)

        # System B: recompute correlation on Experiment 1 training data.
        print("  Computing System B correlation graph (2022-01-01 → 2023-12-31) …")
        t2m_series = {}
        for station, filename in STATION_FILES.items():
            csv_path = os.path.join(DATA_DIR, filename)
            df_raw = pd.read_csv(csv_path, parse_dates=["date"])
            mask = (df_raw["date"] >= "2022-01-01") & (df_raw["date"] <= "2023-12-31")
            s = df_raw.loc[mask, ["date", "t2m"]].set_index("date")["t2m"].dropna()
            t2m_series[station] = s
        A_b = build_adj_system_b(D, t2m_series)

        print(f"  Training Baseline …")
        weights_base = fit_baseline(data)

        print(f"  Training System A (alpha={ALPHA_SYSTEM_A}, T={T_ROUNDS}) …")
        weights_a, _ = run_fl(data, A_a, alpha=ALPHA_SYSTEM_A, T=T_ROUNDS)

        print(f"  Training System B (alpha={ALPHA_SYSTEM_B}, T={T_ROUNDS}) …")
        weights_b, _ = run_fl(data, A_b, alpha=ALPHA_SYSTEM_B, T=T_ROUNDS)

    # Compute HDD metrics
    print("  Computing HDD metrics …")
    hdd_df = run_hdd_analysis(data, weights_base, weights_a, weights_b)

    # Print table
    print("\n" + "=" * 65)
    print("  HDD Summary — Test Set (Experiment 1)")
    print("  Units: °C·days  |  FMI standard: max(17-T,0) if T<12°C")
    print("=" * 65)
    print_hdd_table(hdd_df)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "hdd_analysis.csv")
    hdd_df.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path}")

    # Save figure
    plot_hdd_per_station(
        hdd_df,
        save_path=os.path.join(RESULTS_DIR, "hdd_per_station.png"),
    )

    return hdd_df


def main():
    """Standalone entry point - runs HDD analysis from scratch."""
    print("=" * 65)
    print("  FL Project - HDD Analysis (Experiment 1, Test Set)")
    print("=" * 65 + "\n")
    run()
    print("\n✓ HDD analysis complete.")


if __name__ == "__main__":
    main()
