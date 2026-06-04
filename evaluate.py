"""
evaluate.py
===========
Phase 5

Evaluation utility: given trained weight vectors and prepared data,
computes per-station MSE and RMSE on any split (train / val / test).

No training happens here - this module is purely for measuring prediction
quality after the FL algorithm (or baseline) has produced weights.

METRICS
-------
    MSE  = mean( (X_split @ w[i] - y_split)^2 )
    RMSE = sqrt(MSE)

RMSE is reported in the same units as the label (°C), making it directly
interpretable: an RMSE of 2.5 means predictions are on average 2.5°C off.

Usage:
    from evaluate import evaluate
    results, mean_mse, mean_rmse = evaluate(data, weights, split='test')
"""

import numpy as np


def evaluate(data: dict,
             weights: dict,
             split: str) -> tuple:
    """
    Evaluate model weights on a given data split for all stations.

    Parameters
    ----------
    data : dict
        Prepared data dictionary from prepare_data.py.
        Keys are station names; values contain X_train/val/test and
        y_train/val/test as numpy arrays.

    weights : dict
        Trained weight vectors {station_name: np.ndarray of shape (3,)}.
        Must contain an entry for every station in data.

    split : str
        Which split to evaluate on: 'train', 'val', or 'test'.

    Returns
    -------
    results : dict
        Per-station metrics: {station_name: {'mse': float, 'rmse': float}}.

    mean_mse : float
        Mean MSE averaged across all stations.

    mean_rmse : float
        Mean RMSE averaged across all stations.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test' — got '{split}'")

    results = {}

    for station, w in weights.items():
        X = data[station][f"X_{split}"]   # shape (n, 3)
        y = data[station][f"y_{split}"]   # shape (n,)

        y_pred = X @ w                     # linear prediction, no intercept

        mse  = float(np.mean((y_pred - y) ** 2))
        rmse = float(np.sqrt(mse))

        results[station] = {"mse": mse, "rmse": rmse}

    mean_mse  = float(np.mean([r["mse"]  for r in results.values()]))
    mean_rmse = float(np.mean([r["rmse"] for r in results.values()]))

    return results, mean_mse, mean_rmse
