"""
fl_algorithm.py
===============
Phase 4

Implements the GTVMin (Graph Total Variation Minimisation) federated
learning algorithm using closed-form Ridge regression.

ALGORITHM OVERVIEW
------------------
Each station i maintains a local model weight vector w[i] of shape (3,)
corresponding to the three features [t2m, tmin, tmax].  Prediction for
station i on day t is:

    ŷ[t] = X[t] @ w[i]

In standard local Ridge regression each station would fit w[i] on its
own data independently.  GTVMin adds graph regularisation: each round,
station i's solution is pulled toward a weighted average of its
neighbours' current weights.  Stations that are geographically close
(and, in System B, temperature-correlated) are neighbours and
collaborate more strongly.

ONE ROUND OF GTVMin (for station i)
-------------------------------------
Let w_prev[i'] denote the weights of ALL stations at the START of round t
(synchronous update - no station sees another's updated weights mid-round).

  1. s_i     = sum_i'( A[i,i'] )                   neighbourhood strength
  2. theta_i = (1/s_i) * sum_i'( A[i,i'] * w_prev[i'] )
               weighted average of neighbours' weights
               (if s_i == 0, theta_i = zeros(3))

  3. y_tilde = y_train[i] - X_train[i] @ theta_i   modified labels

  4. Solve Ridge:
       model = Ridge(alpha = alpha * s_i, fit_intercept=False)
       model.fit(X_train[i], y_tilde)
       u_star = model.coef_

  5. w[i] = u_star + theta_i                        recover true weights

WHY RIDGE AND NOT GRADIENT DESCENT?
-------------------------------------
The GTVMin objective has a closed-form solution via Ridge regression.
This is more efficient and numerically stable than iterative gradient
descent - no learning rate to tune, guaranteed convergence in one solve
per round per station.

SYNCHRONOUS UPDATE
------------------
All w[i] are computed from w_prev (the snapshot at round start) before
any w[i] is updated.  This mirrors a real FL system where a central
coordinator collects weights, runs all updates, then broadcasts results.

Usage:
    Import and call run_fl() from run_experiment.py, or run this file
    directly for a quick self-test.
"""

import numpy as np
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Core FL algorithm
# ---------------------------------------------------------------------------

def run_fl(data: dict,
           adj_matrix: np.ndarray,
           alpha: float,
           T: int = 50) -> tuple:
    """
    Run GTVMin federated learning for T rounds using Ridge regression.

    Parameters
    ----------
    data : dict
        Prepared data dictionary from prepare_data.py.
        Keys are station names; values are dicts with keys
        X_train, y_train, X_val, y_val, X_test, y_test.

    adj_matrix : np.ndarray, shape (9, 9)
        Weighted adjacency matrix.  A[i,i'] > 0 means stations i and i'
        are neighbours with collaboration strength A[i,i'].
        Must be indexed in the same order as the stations list below.

    alpha : float
        Ridge regularisation base parameter.  The actual penalty applied
        to station i each round is alpha * s_i, where s_i is the sum of
        its edge weights.  Larger alpha = stronger pull toward neighbours.

    T : int, optional (default 50)
        Number of federated rounds.

    Returns
    -------
    weights : dict
        Final weight vectors {station_name: np.ndarray of shape (3,)}.

    mse_history : list of float
        Mean training MSE across all stations at the end of each round,
        length T.  Used to plot convergence curves.
    """
    # Station order must be consistent with the adjacency matrix row/column order.
    stations = list(data.keys())
    n = len(stations)
    n_features = data[stations[0]]["X_train"].shape[1]  # 3

    # Initialise all weights to zero
    # Shape: (n_stations, n_features)
    W = np.zeros((n, n_features))

    mse_history = []

    for round_idx in range(T):

        # Snapshot weights at the start of this round.
        # This is the synchronous update: every station sees the SAME
        # set of weights, not the partially-updated ones.
        W_prev = W.copy()

        # Update each station in turn (order doesn't matter - all use W_prev).
        for i, station in enumerate(stations):
            X_tr = data[station]["X_train"]   # shape (n_train, 3)
            y_tr = data[station]["y_train"]   # shape (n_train,)

            # Step 1: neighbourhood strength s_i
            # s_i is the sum of edge weights connecting station i to all
            # its neighbours.  It controls how strongly the graph pulls
            # station i toward the consensus.
            s_i = float(np.sum(adj_matrix[i]))   # scalar

            # Step 2: neighbourhood centroid theta_i
            if s_i > 0:
                # Weighted average of all neighbours' current weight vectors.
                # adj_matrix[i] has shape (n,); W_prev has shape (n, 3).
                # The matrix product gives a weighted sum over stations,
                # which we then normalise by s_i.
                theta_i = (adj_matrix[i] @ W_prev) / s_i   # shape (3,)
            else:
                # Isolated node - no neighbours, no graph regularisation.
                theta_i = np.zeros(n_features)

            # Step 3: modified labels y_tilde
            # Subtracting X @ theta_i from y reformulates the regularised
            # problem so it can be solved as a standard Ridge regression
            # on the residual u = w[i] - theta_i.
            y_tilde = y_tr - X_tr @ theta_i    # shape (n_train,)

            # Step 4: solve Ridge regression for u_star
            # The Ridge penalty is scaled by s_i: stations with stronger
            # neighbourhood connections are penalised more, pulling them
            # harder toward the neighbourhood consensus.
            # fit_intercept=False because our features are already centred
            # (standardised in prepare_data.py).
            ridge = Ridge(alpha=alpha * s_i if s_i > 0 else alpha,
                          fit_intercept=False)
            ridge.fit(X_tr, y_tilde)
            u_star = ridge.coef_   # shape (3,)

            # Step 5: recover true weight vector
            # w[i] = u_star + theta_i undoes the substitution from Step 3.
            W[i] = u_star + theta_i

        # Record mean training MSE across all stations this round
        round_mse = _compute_mean_mse(W, data, stations, split="train")
        mse_history.append(round_mse)

    # Pack weights back into a named dictionary for downstream use.
    weights = {station: W[i] for i, station in enumerate(stations)}

    return weights, mse_history


# ---------------------------------------------------------------------------
# Seasonal variant: GTVMin with winter / summer collaboration graphs
# ---------------------------------------------------------------------------

def run_fl_seasonal(data: dict,
                    A_winter: np.ndarray,
                    A_summer: np.ndarray,
                    alpha: float,
                    T: int = 50) -> tuple:
    """
    GTVMin federated learning with season-dependent collaboration graphs.

    Instead of a single adjacency matrix, two matrices capture how stations
    collaborate during winter (Oct-Mar) and summer (Apr-Sep) separately.
    Pearson correlations differ between seasons in Finland: in winter Arctic
    air masses dominate making northern stations highly correlated; in summer
    local topography and coastal effects create more diverse patterns.

    MATHEMATICAL DERIVATION
    -----------------------
    The standard GTVMin objective for station i (single graph):
        L = ||y - X w_i||² + α * s_i * ||w_i - θ_i||²

    With two seasonal graphs the objective becomes:
        L = ||y - X w_i||² + α * (s_w * ||w_i - θ_w||² + s_s * ||w_i - θ_s||²)

    where s_w / s_s are the winter / summer neighbourhood strengths and
    θ_w / θ_s are the winter / summer neighbour centroids.

    This is equivalent (up to a constant in w_i) to:
        L = ||y - X w_i||² + α * s_total * ||w_i - θ_blend||²

    with:
        s_total  = s_w + s_s
        θ_blend  = (s_w * θ_w + s_s * θ_s) / s_total

    The closed-form Ridge update then follows exactly as in the standard
    GTVMin algorithm, using θ_blend and s_total in place of θ_i and s_i.

    ONE ROUND (for station i)
    -------------------------
    1. s_w      = A_winter[i].sum()
       s_s      = A_summer[i].sum()
       s_total  = s_w + s_s
    2. θ_w      = (A_winter[i] @ W_prev) / s_w   (or zeros if s_w == 0)
       θ_s      = (A_summer[i] @ W_prev) / s_s   (or zeros if s_s == 0)
    3. θ_blend  = (s_w * θ_w + s_s * θ_s) / s_total
    4. ỹ        = y_train - X_train @ θ_blend
    5. Solve    Ridge(alpha = α * s_total).fit(X_train, ỹ) → u*
    6. w[i]     = u* + θ_blend

    Parameters
    ----------
    data     : prepared data dict from prepare_data.py
    A_winter : weighted adjacency matrix for winter months, shape (N, N)
    A_summer : weighted adjacency matrix for summer months, shape (N, N)
    alpha    : Ridge regularisation base parameter
    T        : number of federated rounds

    Returns
    -------
    weights     : dict {station_name: np.ndarray of shape (n_features,)}
    mse_history : list of float - mean training MSE per round, length T
    """
    stations   = list(data.keys())
    n          = len(stations)
    n_features = data[stations[0]]["X_train"].shape[1]

    W = np.zeros((n, n_features))
    mse_history = []

    for round_idx in range(T):
        W_prev = W.copy()

        for i, station in enumerate(stations):
            X_tr = data[station]["X_train"]
            y_tr = data[station]["y_train"]

            # Neighbourhood strengths for each season.
            s_w = float(np.sum(A_winter[i]))
            s_s = float(np.sum(A_summer[i]))
            s_total = s_w + s_s

            if s_total == 0:
                # Isolated in both graphs - degenerate to local Ridge.
                ridge = Ridge(alpha=alpha, fit_intercept=False)
                ridge.fit(X_tr, y_tr)
                W[i] = ridge.coef_
                continue

            # Seasonal neighbour centroids.
            theta_w = (A_winter[i] @ W_prev) / s_w if s_w > 0 else np.zeros(n_features)
            theta_s = (A_summer[i] @ W_prev) / s_s if s_s > 0 else np.zeros(n_features)

            # Strength-weighted blend - the effective centroid.
            theta_blend = (s_w * theta_w + s_s * theta_s) / s_total

            # Standard GTVMin update with the blended centroid.
            y_tilde = y_tr - X_tr @ theta_blend
            ridge   = Ridge(alpha=alpha * s_total, fit_intercept=False)
            ridge.fit(X_tr, y_tilde)
            W[i] = ridge.coef_ + theta_blend

        round_mse = _compute_mean_mse(W, data, stations, split="train")
        mse_history.append(round_mse)

    weights = {station: W[i] for i, station in enumerate(stations)}
    return weights, mse_history


# ---------------------------------------------------------------------------
# Helper: mean MSE across all stations for a given split
# ---------------------------------------------------------------------------

def _compute_mean_mse(W: np.ndarray,
                      data: dict,
                      stations: list,
                      split: str = "train") -> float:
    """
    Compute the mean MSE across all stations for a given data split.

    Parameters
    ----------
    W        : weight matrix, shape (n_stations, n_features)
    data     : prepared data dictionary
    stations : ordered list of station names (must match W row order)
    split    : one of 'train', 'val', 'test'

    Returns
    -------
    Mean MSE (float) across all stations.
    """
    mses = []
    for i, station in enumerate(stations):
        X = data[station][f"X_{split}"]
        y = data[station][f"y_{split}"]
        y_pred = X @ W[i]
        mse = float(np.mean((y_pred - y) ** 2))
        mses.append(mse)
    return float(np.mean(mses))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick sanity check: run FL on real prepared data with both adjacency
    matrices and print a summary of the results.
    """
    import os
    import pickle

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    print("Loading prepared data …")
    with open(os.path.join(DATA_DIR, "prepared_data.pkl"), "rb") as f:
        data = pickle.load(f)

    print("Loading adjacency matrices …")
    A_a = np.load(os.path.join(DATA_DIR, "adj_system_a.npy"))
    A_b = np.load(os.path.join(DATA_DIR, "adj_system_b.npy"))

    for system_name, adj in [("System A", A_a), ("System B", A_b)]:
        print(f"\nRunning GTVMin - {system_name}  (alpha=1.0, T=50) …")
        weights, mse_history = run_fl(data, adj, alpha=1.0, T=50)

        print(f"  Initial MSE  (round  1): {mse_history[0]:.4f}")
        print(f"  Final   MSE  (round 50): {mse_history[-1]:.4f}")
        print(f"  Converged: {mse_history[-1] < mse_history[0]}")
        print("  Final weights (first station):",
              list(data.keys())[0], "→",
              weights[list(data.keys())[0]].round(4))

    print("\n✓ fl_algorithm.py self-test complete.")
