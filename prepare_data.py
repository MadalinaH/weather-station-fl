"""
prepare_data.py
===============
Phase 3

Reads the 9 clean daily CSVs produced by standardise_data.py and builds
the train / validation / test splits used by the FL algorithm.

WHAT THIS SCRIPT DOES
----------------------
For each station:

  1. Load the daily CSV and drop any rows with NaN values.

  2. Build feature matrix X and label vector y:
       X[t] = [t2m[t], tmin[t], tmax[t]]   today's temperatures  (3 features)
       y[t] = t2m[t+1]                      tomorrow's mean temp  (label)
     Note: ws_10min is excluded because two stations (Tampere Härmälä and
     Inari Saariselkä) have no wind speed measurements - dropping it keeps
     the feature dimension identical across all 9 FL clients.

  3. Split chronologically (no shuffling - time series order must be preserved):
       Train      : 2022-01-01 → 2023-12-31
       Validation : 2024-01-01 → 2024-06-30
       Test       : 2024-07-01 → 2024-12-31

  4. Normalise features using statistics from the training set only:
       X_scaled = (X - mean_train) / std_train
     The same mean and std are applied to validation and test sets.
     Labels (y) are NOT normalised so predictions stay in °C.

  5. Store all arrays in a nested dictionary and pickle it.

WHY PER-STATION NORMALISATION?
-------------------------------
In a real federated system each client never shares its raw data.
Fitting the scaler on each station's own training data mirrors this:
station i's normalisation parameters are derived exclusively from
station i's observations.

OUTPUT
------
  fl_project/data/prepared_data.pkl

  Structure:
    {
      station_name: {
        "X_train": np.ndarray  shape (n_train, 3)
        "y_train": np.ndarray  shape (n_train,)
        "X_val"  : np.ndarray  shape (n_val,   3)
        "y_val"  : np.ndarray  shape (n_val,)
        "X_test" : np.ndarray  shape (n_test,  3)
        "y_test" : np.ndarray  shape (n_test,)
      },
      ...
    }

Usage:
    python prepare_data.py
"""

import os
import pickle
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Features used in the model.  ws_10min is intentionally excluded - see
# module docstring for the reason.
FEATURE_COLS = ["t2m", "tmin", "tmax"]

# Label: next-day mean temperature.
LABEL_COL = "t2m"

# Chronological split boundaries.
TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"
VAL_START   = "2024-01-01"
VAL_END     = "2024-06-30"
TEST_START  = "2024-07-01"
TEST_END    = "2024-12-31"

# Station name → CSV filename mapping (must match standardise_data.py output).
STATIONS = {
    "Helsinki Kaisaniemi" : "helsinki_kaisaniemi.csv",
    "Turku Artukainen"    : "turku_artukainen.csv",
    "Oulu Vihreäsaari"   : "oulu_vihreasaari.csv",
    "Tampere Härmälä"    : "tampere_harmala.csv",
    "Jyväskylä Airport"  : "jyvaskyla_airport.csv",
    "Kuopio Maaninka"     : "kuopio_maaninka.csv",
    "Rovaniemi Apukka"    : "rovaniemi_apukka.csv",
    "Sodankylä"           : "sodankyla.csv",
    "Inari Saariselkä"   : "inari_saariselka.csv",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Step 1 - Load a station CSV and build (X, y, dates)
# ---------------------------------------------------------------------------

def load_station(csv_path: str) -> pd.DataFrame:
    """
    Load a daily station CSV, drop NaN rows, and return a clean DataFrame.

    Rows with any NaN in the feature or label columns are removed because
    the FL algorithm cannot handle missing values.  After dropping, the
    date index is contiguous but may have gaps.

    Parameters
    ----------
    csv_path : path to the station's daily CSV

    Returns
    -------
    pd.DataFrame with columns [date, t2m, tmin, tmax] sorted by date,
    with all NaN rows removed.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Keep only the columns we need.
    cols_needed = ["date"] + FEATURE_COLS
    df = df[cols_needed].dropna()

    return df


# ---------------------------------------------------------------------------
# Step 2 - Build feature matrix X and label vector y
# ---------------------------------------------------------------------------

def build_xy(df: pd.DataFrame):
    """
    Construct the feature matrix X and label vector y from a daily DataFrame.

    X[t] = [t2m[t], tmin[t], tmax[t]]   - today's temperature features
    y[t] = t2m[t+1]                      - tomorrow's mean temperature

    Because y[t] requires the next day's value, the last row of the
    DataFrame has no valid label and is dropped.  The date column is
    returned separately so we can apply the chronological split by date.

    Parameters
    ----------
    df : clean daily DataFrame from load_station()

    Returns
    -------
    X     : np.ndarray, shape (n-1, 3)
    y     : np.ndarray, shape (n-1,)
    dates : pd.Series of datetime, length n-1  (the feature-day dates)
    """
    X = df[FEATURE_COLS].values[:-1]       # all rows except the last
    y = df[LABEL_COL].values[1:]           # all rows except the first (= next day)
    dates = df["date"].iloc[:-1].reset_index(drop=True)
    return X, y, dates


# ---------------------------------------------------------------------------
# Step 3 - Chronological train / val / test split
# ---------------------------------------------------------------------------

def split_by_date(X: np.ndarray,
                  y: np.ndarray,
                  dates: pd.Series,
                  train_start: str = None,
                  train_end: str   = None,
                  val_start: str   = None,
                  val_end: str     = None,
                  test_start: str  = None,
                  test_end: str    = None) -> dict:
    """
    Split X and y into train, validation, and test sets by calendar date.

    Split boundaries are inclusive on both ends.  When the optional date
    parameters are omitted, the module-level constants (TRAIN_START etc.)
    are used - so calling this function without arguments reproduces the
    default Experiment 1 split.  Passing explicit dates allows
    run_experiment.py to reuse this function for all three experiments
    without modifying prepare_data.py.

    No shuffling is applied - preserving temporal order is essential for
    a time-series forecasting task.

    Parameters
    ----------
    X           : feature matrix, shape (n, 3)
    y           : label vector,   shape (n,)
    dates       : Series of feature-day dates, length n
    train_start : override for TRAIN_START (optional, ISO date string)
    train_end   : override for TRAIN_END
    val_start   : override for VAL_START
    val_end     : override for VAL_END
    test_start  : override for TEST_START
    test_end    : override for TEST_END

    Returns
    -------
    dict with keys X_train, y_train, X_val, y_val, X_test, y_test.
    """
    # Fall back to module-level constants if no override is provided.
    ts  = train_start or TRAIN_START
    te  = train_end   or TRAIN_END
    vs  = val_start   or VAL_START
    ve  = val_end     or VAL_END
    tss = test_start  or TEST_START
    tse = test_end    or TEST_END

    train_mask = (dates >= ts)  & (dates <= te)
    val_mask   = (dates >= vs)  & (dates <= ve)
    test_mask  = (dates >= tss) & (dates <= tse)

    return {
        "X_train" : X[train_mask],
        "y_train" : y[train_mask],
        "X_val"   : X[val_mask],
        "y_val"   : y[val_mask],
        "X_test"  : X[test_mask],
        "y_test"  : y[test_mask],
    }


# ---------------------------------------------------------------------------
# Step 4 - Feature normalisation
# ---------------------------------------------------------------------------

def normalise(splits: dict) -> dict:
    """
    Standardise features to zero mean and unit variance.

    The mean and standard deviation are computed ONLY on the training set
    and then applied to all three splits.  This prevents data leakage from
    the validation or test periods into the normalisation parameters.

    Labels (y_*) are deliberately left unnormalised so that predictions
    and evaluation metrics (MSE, RMSE) remain in degrees Celsius.

    If any feature has zero standard deviation on the training set (i.e.
    a constant column), std is set to 1 to avoid division by zero.

    Parameters
    ----------
    splits : dict from split_by_date()

    Returns
    -------
    The same dict with X_train, X_val, X_test replaced by normalised arrays.
    """
    mean = splits["X_train"].mean(axis=0)   # shape (3,)
    std  = splits["X_train"].std(axis=0)    # shape (3,)
    std[std == 0] = 1.0                     # guard against constant features

    splits["X_train"] = (splits["X_train"] - mean) / std
    splits["X_val"]   = (splits["X_val"]   - mean) / std
    splits["X_test"]  = (splits["X_test"]  - mean) / std

    return splits


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Run the full data preparation pipeline for all 9 stations and save
    the result as a pickle file.
    """
    print("=" * 65)
    print("  FL Project — Phase 3: Preparing data")
    print(f"  Features    : {FEATURE_COLS}")
    print(f"  Label       : next-day {LABEL_COL}")
    print(f"  Train       : {TRAIN_START} → {TRAIN_END}")
    print(f"  Validation  : {VAL_START}   → {VAL_END}")
    print(f"  Test        : {TEST_START}  → {TEST_END}")
    print("=" * 65 + "\n")

    prepared = {}

    for station_name, filename in STATIONS.items():
        csv_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(csv_path):
            print(f"  ✗ MISSING: {filename}  (skipping {station_name})")
            continue

        # Steps 1–4
        df             = load_station(csv_path)
        X, y, dates    = build_xy(df)
        splits         = split_by_date(X, y, dates)
        splits         = normalise(splits)

        prepared[station_name] = splits

        # Print a one-line summary per station.
        n_tr  = len(splits["y_train"])
        n_val = len(splits["y_val"])
        n_te  = len(splits["y_test"])
        print(f"  {station_name:30s}  "
              f"train={n_tr:4d}  val={n_val:3d}  test={n_te:3d}")

    # Save to pickle.
    out_path = os.path.join(DATA_DIR, "prepared_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(prepared, f)

    print(f"\n  Saved → {out_path}")

    # Sanity check: print shape and first feature stats for one station.
    sample = list(prepared.values())[0]
    print("\n  Sanity check (first station):")
    print(f"    X_train shape : {sample['X_train'].shape}")
    print(f"    X_train mean  : {sample['X_train'].mean(axis=0).round(4)}"
          f"  (should be ~[0,0,0])")
    print(f"    X_train std   : {sample['X_train'].std(axis=0).round(4)}"
          f"  (should be ~[1,1,1])")
    print(f"    y_train range : {sample['y_train'].min():.1f} °C"
          f" → {sample['y_train'].max():.1f} °C  (raw, not normalised)")

    print("\n✓ Phase 3 complete.")


if __name__ == "__main__":
    main()
