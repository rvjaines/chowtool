"""
Synthetic Leakage Proof: Reproducing the Overlapping-Label Leakage Problem

This script reproduces the controlled experiment from:
https://github.com/eslazarev/purged-cross-validation/blob/main/examples/synthetic_leakage_proof.ipynb

The purpose is to demonstrate, experimentally, why ordinary shuffled sklearn cross-validation
is invalid for overlapping financial labels.

Key concept:
-----------
A 10-day forward label generated every day means neighbouring observations share most of their
outcome window. Randomly splitting observations therefore allows information from the same
future interval to appear in both train and test. This is label-interval overlap, not merely
temporal correlation.

Data construction:
-----------------
- e[t] ~ N(0, 1): i.i.d. Gaussian noise
- y[t] = mean(e[t : t + H]): forward-looking label depends ONLY on future noise
  * Neighbours share H-1 of H terms → strongly autocorrelated
  * True predictive power of any feature: exactly zero by construction
- x[t] = cumsum(Gamma(2, 1)): monotone increasing feature, unrelated to y
  * Serves as accidental clock when shuffled CV preserves temporal proximity

Prediction times: daily observations from 2020-01-01 onward
Evaluation times: prediction_time + H days (label resolves H days ahead)

Expected result:
---------------
Naive shuffled k-fold: ~0.8-0.9 R² (fabricated skill from label leakage)
PurgedKFold: ~0 or negative (correct "no skill" signal)
Train/test label overlap: 100% naive → 0% purged
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

try:
    from purgedcv import PurgedKFold
    from purgedcv.diagnostics import compute_overlap_fraction
except ImportError:
    raise ImportError(
        "purgedcv is required. Install with: pip install purgedcv"
    )


def build_synthetic_data(seed=0, n=1000, h=20):
    """
    Build synthetic data where true predictive power is zero.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    n : int
        Number of observations
    h : int
        Label horizon (days). y[t] = mean(e[t : t+h])

    Returns
    -------
    dict
        Keys: features, y, pred, evalu
        - features: (n, 1) feature matrix
        - y: (n,) target vector
        - pred: pd.Series of prediction times (dates)
        - evalu: pd.Series of evaluation times (dates)
    """
    rng = np.random.default_rng(seed)

    # Generate noise
    e = rng.standard_normal(n + h - 1)

    # Label: forward window over noise (depends only on future, unrelated to x)
    y = np.array([e[t : t + h].mean() for t in range(n)])

    # Feature: monotone increasing, unrelated to y
    # Acts as accidental clock when shuffled CV scatters observations
    x = np.cumsum(rng.gamma(shape=2.0, scale=1.0, size=n))
    features = x.reshape(-1, 1)

    # Timestamps
    pred = pd.Series(pd.date_range("2020-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=h)

    return {
        "features": features,
        "y": y,
        "pred": pred,
        "evalu": evalu,
        "spurious_corr": np.corrcoef(x, y)[0, 1],
    }


def run_cv_comparison(features, y, pred, evalu, n_splits=5, seed=0):
    """
    Compare naive shuffled KFold vs PurgedKFold on synthetic unpredictable target.

    Parameters
    ----------
    features : ndarray, shape (n, 1)
        Feature matrix
    y : ndarray, shape (n,)
        Target (built to be unpredictable from features)
    pred : pd.Series
        Prediction times (one per observation)
    evalu : pd.Series
        Evaluation times (label horizon end for each observation)
    n_splits : int
        Number of cross-validation folds
    seed : int
        Random seed for naive KFold shuffle

    Returns
    -------
    dict
        Results: model_name -> {naive_r2, purged_r2, naive_overlap, purged_overlap}
    """
    # Define CV splitters
    naive = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    purged = PurgedKFold(
        n_splits=n_splits,
        prediction_times=pred,
        evaluation_times=evalu,
    )

    # Define models
    models = {
        "predict-the-mean": DummyRegressor(strategy="mean"),
        "k-NN": KNeighborsRegressor(n_neighbors=10),
        "RandomForest": RandomForestRegressor(n_estimators=120, random_state=seed),
    }

    results = {}
    for name, model in models.items():
        naive_r2 = float(cross_val_score(model, features, y, cv=naive, scoring="r2").mean())
        purged_r2 = float(cross_val_score(model, features, y, cv=purged, scoring="r2").mean())

        # Measure label overlap
        naive_overlap_fractions = [
            compute_overlap_fraction(tr, te, pred, evalu)
            for tr, te in naive.split(features, y)
        ]
        purged_overlap_fractions = [
            compute_overlap_fraction(tr, te, pred, evalu)
            for tr, te in purged.split(features, y)
        ]

        results[name] = {
            "naive_r2": naive_r2,
            "purged_r2": purged_r2,
            "naive_overlap_mean": float(np.mean(naive_overlap_fractions)),
            "purged_overlap_mean": float(np.mean(purged_overlap_fractions)),
        }

    return results


def print_results(data, results):
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("SYNTHETIC LEAKAGE PROOF RESULTS")
    print("=" * 80)
    print(f"\nData: N={len(data['y'])}, H={len(data['evalu']) - len(data['y']) + 20}, seed=0")
    print(f"Spurious correlation corr(x, y) = {data['spurious_corr']:+.3f}")
    print("(Population relationship is 0 by construction; this finite-sample value is noise.)\n")

    print(f"{'Model':<20} {'Naive R²':>12} {'Purged R²':>12} {'Naive Overlap':>15} {'Purged Overlap':>15}")
    print("-" * 80)
    for name, r in results.items():
        print(
            f"{name:<20} {r['naive_r2']:>12.3f} {r['purged_r2']:>12.3f} "
            f"{r['naive_overlap_mean']:>14.1%} {r['purged_overlap_mean']:>14.1%}"
        )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
On a target nothing can predict, naive shuffled k-fold fabricates skill (R² ≈ 0.8–0.9).
With 100% of training rows overlapping test label horizons, that skill is copied, not learned.

PurgedKFold drops overlap to 0% and the fabricated skill collapses to ~0 or negative.
The exact negative number is not the point; **no positive skill** is the correct answer.

This demonstrates the core failure mode:
- Label intervals overlap: y[t] ∈ [pred[t], evalu[t]) and y[t+1] ∈ [pred[t+1], evalu[t+1])
  share most of their outcome window when H >> 1.
- Naive shuffled split allows train and test to hold observations from overlapping intervals.
- Flexible models (k-NN, RandomForest) exploit this leak by memorising near-neighbours.
- Purged split removes training observations whose label interval overlaps test, blocking leak.
""")


if __name__ == "__main__":
    # Build data
    data = build_synthetic_data(seed=0, n=1000, h=20)

    # Run comparison
    results = run_cv_comparison(
        data["features"],
        data["y"],
        data["pred"],
        data["evalu"],
        n_splits=5,
        seed=0,
    )

    # Print results
    print_results(data, results)
