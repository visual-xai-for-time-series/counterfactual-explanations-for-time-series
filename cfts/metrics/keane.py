"""
Evaluation metrics as defined by Keane et al. (2021).

This module implements the three main evaluation metrics for counterfactual
explanations: validity, proximity, and compactness, as motivated by:
Keane, M. T., Kenny, E. M., Delaney, E., & Smyth, B. (2021). 
If only we had better counterfactual explanations: Five key deficits to 
rectify in the evaluation of counterfactual XAI techniques. 
In IJCAI (Vol. 21, pp. 4466-4474).

References:
- Verma et al. (2020)
- Mothilal et al. (2020)
- Pawelczyk et al. (2020)
- Delaney et al. (2021)
- Karlsson et al. (2020)
"""

import numpy as np
from typing import Callable, List, Union


def validity(original_ts_list: Union[np.ndarray, List[np.ndarray]],
            counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]],
            model: Callable,
            target_classes: Union[int, List[int]] = None) -> float:
    """
    Measures whether the generated counterfactuals lead to valid transformations 
    to the desired target class.
    
    Validity reports the fraction of counterfactuals predicted as the opposite 
    class (i.e., have crossed the decision boundary).
    
    Formula:
        Validity = (1/n) * Σ I(f(x'_i) = y_target)
    
    where:
        - f is the model prediction
        - x'_i is one counterfactual sample
        - y_target is the target class
        - n is the count of samples in the dataset
        - I is the indicator function (1 if true, 0 if false)
    
    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
        model: Trained model for prediction
        target_classes: Target class(es) for the counterfactuals. 
                       If int, same target for all samples.
                       If List, different target per sample.
                       If None, validity checks if prediction changed from original.
    
    Returns:
        Validity score: fraction of valid counterfactuals (0.0 to 1.0).
        Higher is better.
    
    Examples:
        >>> # Single target class for all samples
        >>> validity_score = validity(originals, counterfactuals, model, target_classes=1)
        
        >>> # Different target class per sample
        >>> validity_score = validity(originals, counterfactuals, model, 
        ...                          target_classes=[1, 0, 1, 0])
        
        >>> # Just check if prediction changed (any class)
        >>> validity_score = validity(originals, counterfactuals, model)
    """
    # Convert to list if needed
    if isinstance(original_ts_list, np.ndarray):
        if original_ts_list.ndim == 2:
            original_ts_list = [original_ts_list]
        elif original_ts_list.ndim == 3:
            original_ts_list = [original_ts_list[i] for i in range(len(original_ts_list))]
    
    if isinstance(counterfactual_ts_list, np.ndarray):
        if counterfactual_ts_list.ndim == 2:
            counterfactual_ts_list = [counterfactual_ts_list]
        elif counterfactual_ts_list.ndim == 3:
            counterfactual_ts_list = [counterfactual_ts_list[i] for i in range(len(counterfactual_ts_list))]
    
    n = len(counterfactual_ts_list)
    
    if n == 0:
        return 0.0
    
    # Handle target classes
    if target_classes is None:
        # Check if prediction changed from original
        target_list = [None] * n
    elif isinstance(target_classes, int):
        # Same target for all samples
        target_list = [target_classes] * n
    else:
        # Different target per sample
        target_list = target_classes
    
    if len(target_list) != n:
        raise ValueError(f"Number of target classes ({len(target_list)}) must match "
                        f"number of counterfactuals ({n})")
    
    valid_count = 0
    
    for i, (cf, target) in enumerate(zip(counterfactual_ts_list, target_list)):
        # Get model prediction for counterfactual
        cf_pred = model(cf)
        
        # Convert to numpy if needed
        if hasattr(cf_pred, 'numpy'):
            cf_pred = cf_pred.detach().numpy() if hasattr(cf_pred, 'detach') else cf_pred.numpy()
        
        # Get predicted class
        if isinstance(cf_pred, np.ndarray) and cf_pred.ndim > 0 and cf_pred.size > 1:
            cf_class = np.argmax(cf_pred)
        else:
            cf_class = int(cf_pred)
        
        # Check validity
        if target is not None:
            # Check if prediction matches target class
            if cf_class == target:
                valid_count += 1
        else:
            # Check if prediction changed from original
            if i < len(original_ts_list):
                orig_pred = model(original_ts_list[i])
                if hasattr(orig_pred, 'numpy'):
                    orig_pred = orig_pred.detach().numpy() if hasattr(orig_pred, 'detach') else orig_pred.numpy()
                
                if isinstance(orig_pred, np.ndarray) and orig_pred.ndim > 0 and orig_pred.size > 1:
                    orig_class = np.argmax(orig_pred)
                else:
                    orig_class = int(orig_pred)
                
                if cf_class != orig_class:
                    valid_count += 1
    
    return float(valid_count / n)


def proximity(original_ts_list: Union[np.ndarray, List[np.ndarray]],
             counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]]) -> float:
    """
    Measures the feature-wise distance between the generated counterfactuals 
    and the corresponding original samples.
    
    Proximity is defined as the average Euclidean distance between the 
    transformed and the original time series.
    
    Formula:
        Proximity = (1/n) * Σ ||x_i - x'_i||_2
    
    where:
        - x_i is the original time series
        - x'_i is the generated counterfactual
        - n is the count of samples
        - ||·||_2 is the Euclidean (L2) norm
    
    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
    
    Returns:
        Proximity score: average Euclidean distance.
        Lower is better.
    
    Examples:
        >>> proximity_score = proximity(originals, counterfactuals)
        >>> print(f"Average distance: {proximity_score:.4f}")
    """
    # Convert to list if needed
    if isinstance(original_ts_list, np.ndarray):
        if original_ts_list.ndim == 2:
            original_ts_list = [original_ts_list]
        elif original_ts_list.ndim == 3:
            original_ts_list = [original_ts_list[i] for i in range(len(original_ts_list))]
    
    if isinstance(counterfactual_ts_list, np.ndarray):
        if counterfactual_ts_list.ndim == 2:
            counterfactual_ts_list = [counterfactual_ts_list]
        elif counterfactual_ts_list.ndim == 3:
            counterfactual_ts_list = [counterfactual_ts_list[i] for i in range(len(counterfactual_ts_list))]
    
    n = len(counterfactual_ts_list)
    
    if n == 0:
        return 0.0
    
    if len(original_ts_list) != n:
        raise ValueError(f"Number of originals ({len(original_ts_list)}) must match "
                        f"number of counterfactuals ({n})")
    
    total_distance = 0.0
    
    for orig, cf in zip(original_ts_list, counterfactual_ts_list):
        # Convert to numpy arrays if needed
        if hasattr(orig, 'numpy'):
            orig = orig.detach().numpy() if hasattr(orig, 'detach') else orig.numpy()
        if hasattr(cf, 'numpy'):
            cf = cf.detach().numpy() if hasattr(cf, 'detach') else cf.numpy()
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(orig - cf)
        total_distance += distance
    
    return float(total_distance / n)


def compactness(original_ts_list: Union[np.ndarray, List[np.ndarray]],
               counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]],
               tolerance: float = 0.01) -> float:
    """
    Measures the fraction of time series steps that remain unchanged in the 
    generated counterfactuals compared to the original samples.
    
    Compactness (also reported as sparsity in literature) captures the amount 
    of information that remains unchanged from the original time series.
    
    Formula:
        Compactness = (1/n) * Σ (Σ_t I(|x_i,t - x'_i,t| ≤ tol)) / T
    
    where:
        - x_i,t is the value at time step t in original time series i
        - x'_i,t is the value at time step t in counterfactual i
        - tol is the tolerance parameter for considering values unchanged
        - T is the total number of time steps
        - n is the count of samples
        - I is the indicator function (1 if true, 0 if false)
    
    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
        tolerance: Tolerance parameter for considering a value unchanged (default: 0.01)
    
    Returns:
        Compactness score: fraction of unchanged values (0.0 to 1.0).
        Higher is better.
    
    Examples:
        >>> compactness_score = compactness(originals, counterfactuals, tolerance=0.01)
        >>> print(f"Fraction unchanged: {compactness_score:.2%}")
    """
    # Convert to list if needed
    if isinstance(original_ts_list, np.ndarray):
        if original_ts_list.ndim == 2:
            original_ts_list = [original_ts_list]
        elif original_ts_list.ndim == 3:
            original_ts_list = [original_ts_list[i] for i in range(len(original_ts_list))]
    
    if isinstance(counterfactual_ts_list, np.ndarray):
        if counterfactual_ts_list.ndim == 2:
            counterfactual_ts_list = [counterfactual_ts_list]
        elif counterfactual_ts_list.ndim == 3:
            counterfactual_ts_list = [counterfactual_ts_list[i] for i in range(len(counterfactual_ts_list))]
    
    n = len(counterfactual_ts_list)
    
    if n == 0:
        return 0.0
    
    if len(original_ts_list) != n:
        raise ValueError(f"Number of originals ({len(original_ts_list)}) must match "
                        f"number of counterfactuals ({n})")
    
    total_unchanged_fraction = 0.0
    
    for orig, cf in zip(original_ts_list, counterfactual_ts_list):
        # Convert to numpy arrays if needed
        if hasattr(orig, 'numpy'):
            orig = orig.detach().numpy() if hasattr(orig, 'detach') else orig.numpy()
        if hasattr(cf, 'numpy'):
            cf = cf.detach().numpy() if hasattr(cf, 'detach') else cf.numpy()
        
        # Calculate absolute differences
        differences = np.abs(orig - cf)
        
        # Count unchanged values (within tolerance)
        unchanged_mask = differences <= tolerance
        unchanged_count = np.sum(unchanged_mask)
        total_count = orig.size
        
        # Calculate fraction for this sample
        fraction_unchanged = unchanged_count / total_count
        total_unchanged_fraction += fraction_unchanged
    
    return float(total_unchanged_fraction / n)


def evaluate_keane_metrics(original_ts_list: Union[np.ndarray, List[np.ndarray]],
                          counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]],
                          model: Callable,
                          target_classes: Union[int, List[int]] = None,
                          tolerance: float = 0.01) -> dict:
    """
    Evaluate all three Keane et al. (2021) metrics at once.
    
    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
        model: Trained model for prediction
        target_classes: Target class(es) for validity metric
        tolerance: Tolerance parameter for compactness metric (default: 0.01)
    
    Returns:
        Dictionary containing all three metrics:
            - 'validity': fraction of valid counterfactuals (higher is better)
            - 'proximity': average Euclidean distance (lower is better)
            - 'compactness': fraction of unchanged values (higher is better)
    
    Examples:
        >>> results = evaluate_keane_metrics(originals, counterfactuals, model, 
        ...                                  target_classes=1, tolerance=0.01)
        >>> print(f"Validity: {results['validity']:.2%}")
        >>> print(f"Proximity: {results['proximity']:.4f}")
        >>> print(f"Compactness: {results['compactness']:.2%}")
    """
    return {
        'validity': validity(original_ts_list, counterfactual_ts_list, model, target_classes),
        'proximity': proximity(original_ts_list, counterfactual_ts_list),
        'compactness': compactness(original_ts_list, counterfactual_ts_list, tolerance)
    }


__all__ = [
    'validity',
    'proximity',
    'compactness',
    'evaluate_keane_metrics'
]
