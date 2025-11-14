"""
Sparsity metrics for counterfactual explanations.

These metrics measure how sparse/minimal the changes are between 
the original and counterfactual time series.
"""

import numpy as np
from typing import List, Tuple


def l0_norm(original_ts: np.ndarray, counterfactual_ts: np.ndarray, 
           tolerance: float = 1e-6) -> int:
    """
    Calculates the L0 norm - number of changed time points.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
        tolerance: Threshold for considering a change significant
    
    Returns:
        Number of time points that were changed
    """
    differences = np.abs(original_ts - counterfactual_ts)
    return int(np.sum(differences > tolerance))


def percentage_changed_points(original_ts: np.ndarray, 
                            counterfactual_ts: np.ndarray,
                            tolerance: float = 1e-6) -> float:
    """
    Calculates the percentage of time points that were changed.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
        tolerance: Threshold for considering a change significant
    
    Returns:
        Percentage of changed points (0.0 to 1.0)
    """
    total_points = original_ts.size
    changed_points = l0_norm(original_ts, counterfactual_ts, tolerance)
    return float(changed_points / total_points)


def segment_based_sparsity(original_ts: np.ndarray, 
                          counterfactual_ts: np.ndarray,
                          tolerance: float = 1e-6) -> int:
    """
    Calculates the number of continuous segments that were modified.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
        tolerance: Threshold for considering a change significant
    
    Returns:
        Number of continuous segments that were changed
    """
    differences = np.abs(original_ts - counterfactual_ts)
    changed_mask = differences > tolerance
    
    if original_ts.ndim > 1:
        # For multivariate time series, consider any feature change
        changed_mask = np.any(changed_mask, axis=1)
    
    # Count segments by looking at transitions
    if len(changed_mask) == 0:
        return 0
    
    segments = 0
    in_segment = False
    
    for is_changed in changed_mask:
        if is_changed and not in_segment:
            segments += 1
            in_segment = True
        elif not is_changed:
            in_segment = False
    
    return segments


def feature_sparsity(original_ts: np.ndarray, 
                    counterfactual_ts: np.ndarray,
                    tolerance: float = 1e-6) -> float:
    """
    For multivariate time series, calculates sparsity at the feature level.
    
    Args:
        original_ts: Original time series data (should be 2D: time x features)
        counterfactual_ts: Generated counterfactual time series
        tolerance: Threshold for considering a change significant
    
    Returns:
        Percentage of features that were changed (0.0 to 1.0)
    """
    if original_ts.ndim == 1:
        # For univariate, fall back to point-wise sparsity
        return percentage_changed_points(original_ts, counterfactual_ts, tolerance)
    
    differences = np.abs(original_ts - counterfactual_ts)
    feature_changed = np.any(differences > tolerance, axis=0)
    
    return float(np.sum(feature_changed) / original_ts.shape[1])


def temporal_sparsity_profile(original_ts: np.ndarray, 
                             counterfactual_ts: np.ndarray,
                             tolerance: float = 1e-6) -> np.ndarray:
    """
    Creates a temporal profile showing where changes occur over time.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
        tolerance: Threshold for considering a change significant
    
    Returns:
        Binary array indicating which time points were changed
    """
    differences = np.abs(original_ts - counterfactual_ts)
    
    if original_ts.ndim > 1:
        # For multivariate, check if any feature changed at each time point
        changed_profile = np.any(differences > tolerance, axis=1)
    else:
        changed_profile = differences > tolerance
    
    return changed_profile.astype(int)


def gini_sparsity_coefficient(original_ts: np.ndarray, 
                             counterfactual_ts: np.ndarray) -> float:
    """
    Calculates Gini coefficient of the change magnitudes for sparsity assessment.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
    
    Returns:
        Gini coefficient (0 = perfectly equal changes, 1 = maximally sparse)
    """
    differences = np.abs(original_ts - counterfactual_ts).flatten()
    
    # Remove zeros for Gini calculation
    differences = differences[differences > 0]
    
    if len(differences) == 0:
        return 0.0
    
    # Sort the differences
    sorted_diffs = np.sort(differences)
    n = len(sorted_diffs)
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_diffs)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_diffs)) / (n * cumsum[-1]) - (n + 1) / n
    
    return float(gini)


__all__ = [
    'l0_norm',
    'percentage_changed_points', 
    'segment_based_sparsity',
    'feature_sparsity',
    'temporal_sparsity_profile',
    'gini_sparsity_coefficient'
]