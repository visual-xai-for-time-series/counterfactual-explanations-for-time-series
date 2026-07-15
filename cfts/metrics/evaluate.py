"""
Single-function interface for evaluating counterfactual explanations.

All base metrics (validity, proximity, sparsity, realism) are computed in one call.
Metric set matches what is used in cfts/cf_imfact/experiments/compare_ucr.py and
covers the three metrics from MASCOTS (ModelOriented/mascots/experiments/data/metrics.py):

    MASCOTS metric              Our key                 Notes
    ─────────────────────────── ─────────────────────── ──────────────────────────────────────
    validity                    validity                 Ours adds optional target_class check
    euclidean_distance(norm)    euclidean_dist_zscore   Z-score normalised before L2 (MASCOTS default)
    compactness (tol=1e-5)      compactness             We default to 0.01; pass 1e-5 to match MASCOTS
"""

from typing import Callable, Dict, Optional, Any
import numpy as np

from .proximity import (
    l2_distance,
    manhattan_distance,
    normalized_distance,
    dtw_distance,
    DTW_AVAILABLE,
)
from .realism import (
    autocorrelation_preservation,
    feature_range_validity,
    temporal_consistency,
)
from .sparsity import l0_norm, percentage_changed_points
from .validity import prediction_change


def evaluate_counterfactual(
    original_ts: np.ndarray,
    counterfactual_ts: np.ndarray,
    model: Optional[Callable] = None,
    target_class: Optional[int] = None,
    reference_data: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
    compactness_tolerance: float = 0.01,
    dtw_max_length: Optional[int] = 500,
) -> Dict[str, Any]:
    """
    Compute all base metrics for a single original / counterfactual pair.

    Args:
        original_ts:           Original time series.
        counterfactual_ts:     Generated counterfactual time series.
        model:                 Prediction model. If None, validity is skipped.
        target_class:          Desired target class for validity.
                               If None, any prediction change counts as valid.
        reference_data:        Reference dataset (n_samples, ...) used for
                               feature_range_validity. If None, that metric is skipped.
        tolerance:             Threshold for considering a value changed (sparsity).
        compactness_tolerance: Threshold for considering a value unchanged (compactness).
        dtw_max_length:        Skip DTW if series length exceeds this (default 500).
                               Set to None to always compute DTW regardless of length.

    Returns:
        Dictionary with metric names as keys and scalar values.

        Validity (requires model):
            - validity:                  1.0 if prediction changed to target, else 0.0
            - keane_validity:            Alias for validity (Keane et al. 2021 naming)
        Proximity:
            - l2_distance:               Euclidean (L2) distance
            - euclidean_dist_zscore:     L2 after z-score normalisation by original stats
                                         (matches MASCOTS euclidean_distance(normalize=True))
            - manhattan_distance:        L1 distance
            - normalized_distance:       L2 normalised by series range and length
            - dtw_distance:              DTW distance (skipped if dtaidistance not installed)
            - keane_proximity:           Alias for l2_distance (Keane et al. 2021 naming)
        Sparsity:
            - l0_norm:                   Number of changed time points
            - pct_changed:               Fraction of changed time points (0–1)
            - sparsity:                  Fraction of *unchanged* time points at tolerance (0–1)
            - compactness:               Fraction unchanged at compactness_tolerance (default 0.01).
                                         MASCOTS uses 1e-5; pass compactness_tolerance=1e-5 to match.
            - keane_compactness:         Alias for compactness (Keane et al. 2021 naming)
        Realism:
            - temporal_consistency:      Smoothness score of counterfactual (0–1, higher=smoother)
            - autocorr_preservation:     Autocorrelation similarity to original (0–1)
            - range_validity:            Fraction of CF values within reference range
                                         (requires reference_data)
    """
    results: Dict[str, Any] = {}

    # --- Validity ---
    if model is not None:
        results["validity"] = prediction_change(
            original_ts, counterfactual_ts, model, target_class=target_class
        )

    # --- Proximity ---
    results["l2_distance"] = l2_distance(original_ts, counterfactual_ts)

    # Z-score normalised L2 — matches MASCOTS euclidean_distance(normalize=True).
    # Normalise both series by the original's mean/std before computing L2.
    _mean, _std = float(original_ts.mean()), float(original_ts.std())
    if _std > 0:
        _orig_z = (original_ts - _mean) / _std
        _cf_z = (counterfactual_ts - _mean) / _std
    else:
        _orig_z, _cf_z = original_ts, counterfactual_ts
    _diff_z = _orig_z - _cf_z
    _diff_z = np.where(np.isnan(_diff_z), 0.0, _diff_z)
    results["euclidean_dist_zscore"] = float(np.linalg.norm(_diff_z))

    results["manhattan_distance"] = manhattan_distance(original_ts, counterfactual_ts)
    results["normalized_distance"] = normalized_distance(
        original_ts.reshape(-1), counterfactual_ts.reshape(-1)
    )

    _series_len = int(original_ts.size)
    if DTW_AVAILABLE and (dtw_max_length is None or _series_len <= dtw_max_length):
        results["dtw_distance"] = dtw_distance(original_ts, counterfactual_ts)

    # --- Sparsity ---
    results["l0_norm"] = l0_norm(original_ts, counterfactual_ts, tolerance=tolerance)
    results["pct_changed"] = percentage_changed_points(
        original_ts, counterfactual_ts, tolerance=tolerance
    )
    results["sparsity"] = 1.0 - results["pct_changed"]
    results["compactness"] = float(
        np.sum(np.abs(original_ts - counterfactual_ts) <= compactness_tolerance)
        / original_ts.size
    )

    # --- Realism ---
    results["temporal_consistency"] = temporal_consistency(counterfactual_ts)
    results["autocorr_preservation"] = autocorrelation_preservation(
        original_ts, counterfactual_ts
    )

    if reference_data is not None:
        results["range_validity"] = feature_range_validity(
            counterfactual_ts, reference_data
        )

    # --- Keane et al. (2021) aliases for single-pair convenience ---
    # For a single pair these are identical to the already-computed metrics.
    if "validity" in results:
        results["keane_validity"] = results["validity"]
    results["keane_proximity"] = results["l2_distance"]
    results["keane_compactness"] = results["compactness"]

    return results


__all__ = ["evaluate_counterfactual"]
