"""
Stability/Robustness metrics for counterfactual explanations.

These metrics evaluate the stability and robustness of counterfactual
generation algorithms across different conditions.
"""

import numpy as np
from typing import List, Callable, Dict, Any, Optional


def algorithmic_stability(algorithm: Callable,
                         original_ts: np.ndarray,
                         n_runs: int = 10,
                         distance_metric: str = 'euclidean',
                         **algorithm_kwargs) -> float:
    """
    Measures consistency of results across multiple runs of the algorithm.
    
    Args:
        algorithm: Counterfactual generation algorithm function
        original_ts: Original time series data
        n_runs: Number of algorithm runs for stability assessment
        distance_metric: Distance metric for comparing results
        **algorithm_kwargs: Additional arguments for the algorithm
    
    Returns:
        Stability score (0 = unstable, 1 = perfectly stable)
    """
    counterfactuals = []
    
    # Run algorithm multiple times
    for _ in range(n_runs):
        try:
            cf = algorithm(original_ts, **algorithm_kwargs)
            counterfactuals.append(cf)
        except Exception:
            # If algorithm fails, consider it unstable
            return 0.0
    
    if len(counterfactuals) < 2:
        return 0.0
    
    # Calculate pairwise distances between results
    cf_matrix = np.array([cf.flatten() for cf in counterfactuals])
    
    total_distance = 0.0
    num_pairs = 0
    
    for i in range(len(counterfactuals)):
        for j in range(i + 1, len(counterfactuals)):
            if distance_metric == 'euclidean':
                dist = np.linalg.norm(cf_matrix[i] - cf_matrix[j])
            elif distance_metric == 'manhattan':
                dist = np.sum(np.abs(cf_matrix[i] - cf_matrix[j]))
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            
            total_distance += dist
            num_pairs += 1
    
    avg_distance = total_distance / num_pairs if num_pairs > 0 else 0.0
    
    # Normalize by the range of the original time series
    ts_range = np.max(original_ts) - np.min(original_ts)
    max_possible_distance = ts_range * np.sqrt(original_ts.size)
    
    if max_possible_distance == 0:
        return 1.0 if avg_distance == 0 else 0.0
    
    # Convert distance to stability (lower distance = higher stability)
    stability = 1.0 - min(avg_distance / max_possible_distance, 1.0)
    
    return float(stability)


def input_perturbation_robustness(algorithm: Callable,
                                 original_ts: np.ndarray,
                                 perturbation_strength: float = 0.01,
                                 n_perturbations: int = 10,
                                 **algorithm_kwargs) -> float:
    """
    Measures sensitivity to small changes in input time series.
    
    Args:
        algorithm: Counterfactual generation algorithm function
        original_ts: Original time series data
        perturbation_strength: Magnitude of input perturbations
        n_perturbations: Number of perturbations to test
        **algorithm_kwargs: Additional arguments for the algorithm
    
    Returns:
        Robustness score (0 = sensitive, 1 = robust)
    """
    try:
        # Generate baseline counterfactual
        baseline_cf = algorithm(original_ts, **algorithm_kwargs)
    except Exception:
        return 0.0
    
    perturbation_distances = []
    
    for _ in range(n_perturbations):
        # Add random perturbation to input
        noise = np.random.normal(0, perturbation_strength, original_ts.shape)
        perturbed_ts = original_ts + noise
        
        try:
            # Generate counterfactual for perturbed input
            perturbed_cf = algorithm(perturbed_ts, **algorithm_kwargs)
            
            # Calculate distance between baseline and perturbed counterfactuals
            distance = np.linalg.norm(baseline_cf.flatten() - perturbed_cf.flatten())
            perturbation_distances.append(distance)
        
        except Exception:
            # If algorithm fails on perturbed input, consider it non-robust
            perturbation_distances.append(float('inf'))
    
    if len(perturbation_distances) == 0:
        return 0.0
    
    # Calculate average perturbation sensitivity
    avg_perturbation_distance = np.mean(perturbation_distances)
    
    # Normalize by the magnitude of the baseline counterfactual
    baseline_magnitude = np.linalg.norm(baseline_cf.flatten())
    
    if baseline_magnitude == 0:
        return 1.0 if avg_perturbation_distance == 0 else 0.0
    
    # Convert to robustness score (lower sensitivity = higher robustness)
    if np.isinf(avg_perturbation_distance):
        return 0.0
    
    sensitivity = avg_perturbation_distance / baseline_magnitude
    robustness = 1.0 / (1.0 + sensitivity)  # Sigmoid-like transformation
    
    return float(robustness)


def model_robustness(algorithm: Callable,
                    original_ts: np.ndarray,
                    models: List[Callable],
                    **algorithm_kwargs) -> float:
    """
    Evaluates performance consistency across different model architectures.
    
    Args:
        algorithm: Counterfactual generation algorithm function
        original_ts: Original time series data
        models: List of different trained models
        **algorithm_kwargs: Additional arguments for the algorithm
    
    Returns:
        Model robustness score (0 = model-specific, 1 = model-agnostic)
    """
    if len(models) < 2:
        return 1.0
    
    counterfactuals = []
    
    for model in models:
        try:
            # Update algorithm kwargs with current model
            kwargs_with_model = algorithm_kwargs.copy()
            kwargs_with_model['model'] = model
            
            cf = algorithm(original_ts, **kwargs_with_model)
            counterfactuals.append(cf)
        except Exception:
            # If algorithm fails with this model, consider it non-robust
            return 0.0
    
    if len(counterfactuals) < 2:
        return 0.0
    
    # Calculate pairwise distances between counterfactuals from different models
    cf_matrix = np.array([cf.flatten() for cf in counterfactuals])
    
    total_distance = 0.0
    num_pairs = 0
    
    for i in range(len(counterfactuals)):
        for j in range(i + 1, len(counterfactuals)):
            distance = np.linalg.norm(cf_matrix[i] - cf_matrix[j])
            total_distance += distance
            num_pairs += 1
    
    avg_distance = total_distance / num_pairs if num_pairs > 0 else 0.0
    
    # Normalize by the range of the original time series
    ts_range = np.max(original_ts) - np.min(original_ts)
    max_possible_distance = ts_range * np.sqrt(original_ts.size)
    
    if max_possible_distance == 0:
        return 1.0 if avg_distance == 0 else 0.0
    
    # Convert distance to robustness (lower distance = higher robustness)
    robustness = 1.0 - min(avg_distance / max_possible_distance, 1.0)
    
    return float(robustness)


def hyperparameter_sensitivity(algorithm: Callable,
                              original_ts: np.ndarray,
                              hyperparameter_configs: List[Dict[str, Any]]) -> float:
    """
    Measures sensitivity to hyperparameter changes.
    
    Args:
        algorithm: Counterfactual generation algorithm function
        original_ts: Original time series data
        hyperparameter_configs: List of hyperparameter configurations to test
    
    Returns:
        Hyperparameter stability score (0 = sensitive, 1 = stable)
    """
    if len(hyperparameter_configs) < 2:
        return 1.0
    
    counterfactuals = []
    
    for config in hyperparameter_configs:
        try:
            cf = algorithm(original_ts, **config)
            counterfactuals.append(cf)
        except Exception:
            # If algorithm fails with this configuration, it's sensitive
            return 0.0
    
    if len(counterfactuals) < 2:
        return 0.0
    
    # Calculate coefficient of variation for each time point
    cf_matrix = np.array([cf.flatten() for cf in counterfactuals])
    
    # Calculate mean and std for each feature across configurations
    means = np.mean(cf_matrix, axis=0)
    stds = np.std(cf_matrix, axis=0)
    
    # Calculate coefficient of variation (avoid division by zero)
    cv_values = np.where(means != 0, stds / np.abs(means), 0)
    
    # Average coefficient of variation as sensitivity measure
    avg_cv = np.mean(cv_values)
    
    # Convert to stability score (lower CV = higher stability)
    stability = 1.0 / (1.0 + avg_cv)
    
    return float(stability)


def convergence_stability(optimization_histories: List[List[float]]) -> float:
    """
    Analyzes convergence behavior across multiple optimization runs.
    
    Args:
        optimization_histories: List of optimization objective histories
    
    Returns:
        Convergence stability score (0 = unstable, 1 = stable)
    """
    if len(optimization_histories) < 2:
        return 1.0
    
    # Find minimum length to compare same number of iterations
    min_length = min(len(history) for history in optimization_histories)
    
    if min_length == 0:
        return 0.0
    
    # Truncate all histories to same length
    truncated_histories = [history[:min_length] for history in optimization_histories]
    
    # Calculate coefficient of variation at each iteration
    cv_per_iteration = []
    
    for i in range(min_length):
        values_at_iteration = [history[i] for history in truncated_histories]
        mean_val = np.mean(values_at_iteration)
        std_val = np.std(values_at_iteration)
        
        if mean_val != 0:
            cv = std_val / abs(mean_val)
        else:
            cv = 0.0 if std_val == 0 else float('inf')
        
        cv_per_iteration.append(cv)
    
    # Calculate stability as inverse of average CV
    avg_cv = np.mean(cv_per_iteration)
    
    if np.isinf(avg_cv):
        return 0.0
    
    stability = 1.0 / (1.0 + avg_cv)
    
    return float(stability)


__all__ = [
    'algorithmic_stability',
    'input_perturbation_robustness',
    'model_robustness',
    'hyperparameter_sensitivity',
    'convergence_stability'
]