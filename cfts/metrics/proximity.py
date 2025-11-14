"""
Proximity/Distance metrics for counterfactual explanations.

These metrics measure how close the counterfactual is to the original
time series using various distance measures suitable for time series data.
"""

import numpy as np
from typing import Optional
from scipy.spatial.distance import euclidean, mahalanobis as scipy_mahalanobis
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


def l2_distance(original_ts: np.ndarray, counterfactual_ts: np.ndarray) -> float:
    """
    Calculates the Euclidean (L2) distance between original and counterfactual time series.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
    
    Returns:
        L2 distance between the time series
    """
    return float(np.linalg.norm(original_ts - counterfactual_ts))


def manhattan_distance(original_ts: np.ndarray, counterfactual_ts: np.ndarray) -> float:
    """
    Calculates the Manhattan (L1) distance between original and counterfactual time series.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
    
    Returns:
        Manhattan distance between the time series
    """
    return float(np.sum(np.abs(original_ts - counterfactual_ts)))


def dtw_distance(original_ts: np.ndarray, counterfactual_ts: np.ndarray) -> float:
    """
    Calculates Dynamic Time Warping distance accounting for temporal alignment.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
    
    Returns:
        DTW distance between the time series
    """
    if not DTW_AVAILABLE:
        raise ImportError("dtaidistance package is required for DTW distance. Install with: pip install dtaidistance")
    
    # Flatten if multidimensional
    if original_ts.ndim > 1:
        original_flat = original_ts.flatten()
        cf_flat = counterfactual_ts.flatten()
    else:
        original_flat = original_ts
        cf_flat = counterfactual_ts
    
    return float(dtw.distance(original_flat, cf_flat))


def frechet_distance(original_ts: np.ndarray, counterfactual_ts: np.ndarray) -> float:
    """
    Calculates discrete Fréchet distance considering ordering and flow of time series.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
    
    Returns:
        Fréchet distance between the time series
    """
    def _discrete_frechet_distance(P: np.ndarray, Q: np.ndarray) -> float:
        """
        Compute discrete Fréchet distance between two curves.
        """
        n, m = len(P), len(Q)
        
        # Initialize memoization matrix
        memo = {}
        
        def _c(i: int, j: int) -> float:
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i == 0 and j == 0:
                result = np.linalg.norm(P[0] - Q[0])
            elif i > 0 and j == 0:
                result = max(_c(i-1, 0), np.linalg.norm(P[i] - Q[0]))
            elif i == 0 and j > 0:
                result = max(_c(0, j-1), np.linalg.norm(P[0] - Q[j]))
            elif i > 0 and j > 0:
                result = max(
                    min(_c(i-1, j), _c(i-1, j-1), _c(i, j-1)),
                    np.linalg.norm(P[i] - Q[j])
                )
            else:
                result = float('inf')
            
            memo[(i, j)] = result
            return result
        
        return _c(n-1, m-1)
    
    # Convert to 2D points if 1D
    if original_ts.ndim == 1:
        P = np.column_stack([np.arange(len(original_ts)), original_ts])
        Q = np.column_stack([np.arange(len(counterfactual_ts)), counterfactual_ts])
    else:
        P = original_ts
        Q = counterfactual_ts
    
    return float(_discrete_frechet_distance(P, Q))


def normalized_distance(original_ts: np.ndarray, 
                       counterfactual_ts: np.ndarray,
                       distance_func: callable = l2_distance) -> float:
    """
    Calculates normalized distance based on the range of the original time series.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
        distance_func: Distance function to use (default: l2_distance)
    
    Returns:
        Normalized distance between 0 and 1
    """
    raw_distance = distance_func(original_ts, counterfactual_ts)
    ts_range = np.max(original_ts) - np.min(original_ts)
    
    if ts_range == 0:
        return 0.0 if raw_distance == 0 else 1.0
    
    # Normalize by the maximum possible distance (range * sqrt(length))
    max_possible_distance = ts_range * np.sqrt(len(original_ts))
    return float(min(raw_distance / max_possible_distance, 1.0))


def mahalanobis_distance(original_ts: np.ndarray, 
                         counterfactual_ts: np.ndarray,
                         reference_data: Optional[np.ndarray] = None,
                         regularization: float = 1e-6) -> float:
    """
    Calculates the Mahalanobis distance between original and counterfactual time series.
    
    The Mahalanobis distance accounts for the covariance structure in the data, making it
    particularly useful when features/time points are correlated or have different scales.
    
    Args:
        original_ts: Original time series data (1D or flattened)
        counterfactual_ts: Generated counterfactual time series (1D or flattened)
        reference_data: Reference dataset to compute covariance matrix from.
                       Shape should be (n_samples, n_features) where n_features matches
                       the flattened length of the time series. If None, uses the
                       identity matrix (equivalent to Euclidean distance).
        regularization: Small value added to diagonal of covariance matrix for numerical
                       stability (default: 1e-6)
    
    Returns:
        Mahalanobis distance between the time series
    
    Raises:
        ValueError: If time series shapes don't match or reference data has wrong shape
    
    Example:
        >>> original = np.array([1.0, 2.0, 3.0])
        >>> counterfactual = np.array([1.5, 2.5, 3.5])
        >>> reference = np.random.randn(100, 3)
        >>> dist = mahalanobis_distance(original, counterfactual, reference)
    """
    # Flatten if multidimensional
    if original_ts.ndim > 1:
        original_flat = original_ts.flatten()
        cf_flat = counterfactual_ts.flatten()
    else:
        original_flat = original_ts
        cf_flat = counterfactual_ts
    
    if original_flat.shape != cf_flat.shape:
        raise ValueError(f"Time series shapes must match: {original_flat.shape} vs {cf_flat.shape}")
    
    # Compute difference vector
    diff = original_flat - cf_flat
    
    # Compute or use covariance matrix
    if reference_data is None:
        # Without reference data, use identity matrix (reduces to Euclidean distance)
        return float(np.sqrt(np.sum(diff ** 2)))
    
    # Ensure reference data is 2D
    if reference_data.ndim == 1:
        reference_data = reference_data.reshape(1, -1)
    
    # Flatten reference data if needed
    if reference_data.ndim > 2:
        n_samples = reference_data.shape[0]
        reference_data = reference_data.reshape(n_samples, -1)
    
    # Check dimensions
    if reference_data.shape[1] != len(original_flat):
        raise ValueError(
            f"Reference data feature dimension ({reference_data.shape[1]}) "
            f"must match time series length ({len(original_flat)})"
        )
    
    # Compute covariance matrix
    try:
        cov_matrix = np.cov(reference_data, rowvar=False)
        
        # Add regularization for numerical stability
        if cov_matrix.ndim == 0:
            # Single feature case
            cov_matrix = np.array([[cov_matrix + regularization]])
        else:
            cov_matrix += np.eye(cov_matrix.shape[0]) * regularization
        
        # Compute inverse covariance matrix
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        
        # Calculate Mahalanobis distance
        mahal_dist = np.sqrt(diff @ inv_cov_matrix @ diff.T)
        
        return float(mahal_dist)
        
    except np.linalg.LinAlgError:
        # If covariance matrix is singular even after regularization,
        # fall back to Euclidean distance
        import warnings
        warnings.warn(
            "Covariance matrix is singular. Falling back to Euclidean distance.",
            RuntimeWarning
        )
        return float(np.sqrt(np.sum(diff ** 2)))


__all__ = [
    'l2_distance', 
    'manhattan_distance', 
    'dtw_distance', 
    'frechet_distance',
    'normalized_distance',
    'mahalanobis_distance'
]