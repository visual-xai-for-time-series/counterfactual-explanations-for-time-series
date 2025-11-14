"""
Realism/Plausibility metrics for counterfactual explanations.

These metrics evaluate whether the generated counterfactuals are realistic
and plausible within the domain constraints and data distribution.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats
from scipy.stats import wasserstein_distance


def domain_constraint_violations(counterfactual_ts: np.ndarray,
                                constraints: Dict[str, Tuple[float, float]]) -> int:
    """
    Counts violations of domain-specific constraints.
    
    Args:
        counterfactual_ts: Generated counterfactual time series
        constraints: Dictionary with constraint names and (min, max) ranges
    
    Returns:
        Number of constraint violations
    """
    violations = 0
    
    if 'global_range' in constraints:
        min_val, max_val = constraints['global_range']
        violations += int(np.any(counterfactual_ts < min_val))
        violations += int(np.any(counterfactual_ts > max_val))
    
    if 'feature_ranges' in constraints and counterfactual_ts.ndim > 1:
        feature_ranges = constraints['feature_ranges']
        for i, (min_val, max_val) in enumerate(feature_ranges):
            if i < counterfactual_ts.shape[1]:
                violations += int(np.any(counterfactual_ts[:, i] < min_val))
                violations += int(np.any(counterfactual_ts[:, i] > max_val))
    
    return violations


def statistical_similarity(original_ts: np.ndarray, 
                          counterfactual_ts: np.ndarray,
                          reference_data: Optional[np.ndarray] = None,
                          method: str = 'ks_test') -> float:
    """
    Measures statistical similarity to original data distribution.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
        reference_data: Reference dataset for distribution comparison
        method: Statistical test method ('ks_test', 'wasserstein', 'kl_divergence')
    
    Returns:
        Statistical similarity score (interpretation depends on method)
    """
    if reference_data is None:
        reference_data = original_ts
    
    # Flatten for distribution comparison
    ref_flat = reference_data.flatten()
    cf_flat = counterfactual_ts.flatten()
    
    if method == 'ks_test':
        # Kolmogorov-Smirnov test (p-value)
        statistic, p_value = stats.ks_2samp(ref_flat, cf_flat)
        return float(p_value)
    
    elif method == 'wasserstein':
        # Wasserstein distance (lower is more similar)
        distance = wasserstein_distance(ref_flat, cf_flat)
        return float(distance)
    
    elif method == 'kl_divergence':
        # KL divergence using histograms
        # Create histograms with same bins
        bins = np.linspace(
            min(ref_flat.min(), cf_flat.min()),
            max(ref_flat.max(), cf_flat.max()),
            50
        )
        
        hist_ref, _ = np.histogram(ref_flat, bins=bins, density=True)
        hist_cf, _ = np.histogram(cf_flat, bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist_ref = hist_ref + eps
        hist_cf = hist_cf + eps
        
        # Normalize
        hist_ref = hist_ref / np.sum(hist_ref)
        hist_cf = hist_cf / np.sum(hist_cf)
        
        kl_div = np.sum(hist_cf * np.log(hist_cf / hist_ref))
        return float(kl_div)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def temporal_consistency(counterfactual_ts: np.ndarray, 
                        smoothness_threshold: float = 1.0) -> float:
    """
    Measures smooth transitions and realistic temporal patterns.
    
    Args:
        counterfactual_ts: Generated counterfactual time series
        smoothness_threshold: Threshold for acceptable temporal jumps
    
    Returns:
        Temporal consistency score (0 = inconsistent, 1 = perfectly smooth)
    """
    if len(counterfactual_ts) < 2:
        return 1.0
    
    # Calculate first-order differences
    diffs = np.diff(counterfactual_ts, axis=0)
    
    if counterfactual_ts.ndim > 1:
        # For multivariate, calculate norm of difference vectors
        diff_magnitudes = np.linalg.norm(diffs, axis=1)
    else:
        diff_magnitudes = np.abs(diffs)
    
    # Count violations of smoothness threshold
    violations = np.sum(diff_magnitudes > smoothness_threshold)
    consistency = 1.0 - (violations / len(diff_magnitudes))
    
    return float(max(0.0, consistency))


def feature_range_validity(counterfactual_ts: np.ndarray,
                          reference_data: np.ndarray,
                          percentile_threshold: float = 95.0) -> float:
    """
    Checks if counterfactual values stay within realistic feature ranges.
    
    Args:
        counterfactual_ts: Generated counterfactual time series
        reference_data: Reference dataset to determine valid ranges
        percentile_threshold: Percentile for determining valid range
    
    Returns:
        Percentage of values within valid ranges (0.0 to 1.0)
    """
    # Flatten both arrays for comparison
    cf_flat = counterfactual_ts.flatten()
    ref_flat = reference_data.flatten()
    
    # Calculate percentile ranges
    lower_bound = np.percentile(ref_flat, (100 - percentile_threshold) / 2)
    upper_bound = np.percentile(ref_flat, 100 - (100 - percentile_threshold) / 2)
    
    # Check validity
    valid_mask = (cf_flat >= lower_bound) & (cf_flat <= upper_bound)
    validity_ratio = np.mean(valid_mask)
    
    return float(validity_ratio)


def autocorrelation_preservation(original_ts: np.ndarray,
                                counterfactual_ts: np.ndarray,
                                max_lag: int = 10) -> float:
    """
    Measures how well the counterfactual preserves autocorrelation patterns.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
        max_lag: Maximum lag for autocorrelation comparison
    
    Returns:
        Autocorrelation similarity score (0 = completely different, 1 = identical)
    """
    def _autocorrelation(ts: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation for given lags."""
        if ts.ndim > 1:
            ts = ts.flatten()
        
        autocorrs = []
        for lag in range(1, max_lag + 1):
            if len(ts) > lag:
                corr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
                autocorrs.append(corr if not np.isnan(corr) else 0.0)
            else:
                autocorrs.append(0.0)
        
        return np.array(autocorrs)
    
    orig_autocorr = _autocorrelation(original_ts, max_lag)
    cf_autocorr = _autocorrelation(counterfactual_ts, max_lag)
    
    # Calculate similarity using correlation coefficient
    if np.std(orig_autocorr) == 0 or np.std(cf_autocorr) == 0:
        return 1.0 if np.allclose(orig_autocorr, cf_autocorr) else 0.0
    
    similarity = np.corrcoef(orig_autocorr, cf_autocorr)[0, 1]
    return float(max(0.0, similarity))


def spectral_similarity(original_ts: np.ndarray,
                       counterfactual_ts: np.ndarray) -> float:
    """
    Compares frequency domain characteristics of original and counterfactual.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
    
    Returns:
        Spectral similarity score (0 = completely different, 1 = identical)
    """
    if original_ts.ndim > 1:
        orig_flat = original_ts.flatten()
        cf_flat = counterfactual_ts.flatten()
    else:
        orig_flat = original_ts
        cf_flat = counterfactual_ts
    
    # Compute power spectral density
    orig_fft = np.abs(np.fft.fft(orig_flat))
    cf_fft = np.abs(np.fft.fft(cf_flat))
    
    # Normalize
    orig_fft = orig_fft / np.sum(orig_fft)
    cf_fft = cf_fft / np.sum(cf_fft)
    
    # Calculate similarity using correlation
    similarity = np.corrcoef(orig_fft, cf_fft)[0, 1]
    return float(max(0.0, similarity))


__all__ = [
    'domain_constraint_violations',
    'statistical_similarity',
    'temporal_consistency',
    'feature_range_validity',
    'autocorrelation_preservation',
    'spectral_similarity'
]