"""
Validity metrics for counterfactual explanations.

These metrics evaluate whether the generated counterfactuals achieve
the desired prediction changes and cross decision boundaries effectively.
"""

import numpy as np
import torch
from typing import Union, Callable, Any


def prediction_change(original_ts: np.ndarray, 
                     counterfactual_ts: np.ndarray,
                     model: Callable,
                     target_class: int = None) -> float:
    """
    Measures whether the counterfactual changes the model's prediction.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
        model: Trained model for prediction
        target_class: Desired target class (if None, just check if prediction changed)
    
    Returns:
        1.0 if prediction changed to target class (or any change if target_class=None), 0.0 otherwise
    """
    original_pred = model(original_ts)
    cf_pred = model(counterfactual_ts)
    
    if hasattr(original_pred, 'numpy'):
        original_pred = original_pred.numpy()
    if hasattr(cf_pred, 'numpy'):
        cf_pred = cf_pred.numpy()
    
    # Get predicted classes
    if isinstance(original_pred, np.ndarray) and original_pred.ndim > 0:
        orig_class = np.argmax(original_pred) if original_pred.size > 1 else int(original_pred.item())
    else:
        orig_class = int(original_pred)
        
    if isinstance(cf_pred, np.ndarray) and cf_pred.ndim > 0:
        cf_class = np.argmax(cf_pred) if cf_pred.size > 1 else int(cf_pred.item())
    else:
        cf_class = int(cf_pred)
    
    if target_class is not None:
        return float(cf_class == target_class)
    else:
        return float(orig_class != cf_class)


def class_probability_confidence(counterfactual_ts: np.ndarray,
                                model: Callable,
                                target_class: int) -> float:
    """
    Evaluates the confidence of the model's prediction on the counterfactual.
    
    Args:
        counterfactual_ts: Generated counterfactual time series
        model: Trained model for prediction
        target_class: Target class for the counterfactual
    
    Returns:
        Probability/confidence score for the target class
    """
    cf_pred = model(counterfactual_ts)
    
    if hasattr(cf_pred, 'numpy'):
        cf_pred = cf_pred.numpy()
    
    # Apply softmax if raw logits
    if cf_pred.max() > 1.0 or cf_pred.min() < 0.0:
        cf_pred = np.exp(cf_pred) / np.sum(np.exp(cf_pred), axis=-1, keepdims=True)
    
    # Handle both 1D and 2D prediction arrays
    if isinstance(cf_pred, np.ndarray) and cf_pred.size > 1:
        return float(cf_pred[target_class])
    else:
        return float(cf_pred)


def decision_boundary_distance(original_ts: np.ndarray,
                              counterfactual_ts: np.ndarray, 
                              model: Callable,
                              num_steps: int = 100) -> float:
    """
    Measures how far the counterfactual moves across the decision boundary.
    
    Args:
        original_ts: Original time series data
        counterfactual_ts: Generated counterfactual time series
        model: Trained model for prediction
        num_steps: Number of interpolation steps to find boundary
    
    Returns:
        Distance from original point to decision boundary
    """
    original_pred = model(original_ts)
    cf_pred = model(counterfactual_ts)
    
    if hasattr(original_pred, 'numpy'):
        original_pred = original_pred.numpy()
    if hasattr(cf_pred, 'numpy'):
        cf_pred = cf_pred.numpy()
    
    # Get predicted classes
    if isinstance(original_pred, np.ndarray) and original_pred.size > 1:
        orig_class = np.argmax(original_pred)
    else:
        orig_class = int(original_pred)
        
    if isinstance(cf_pred, np.ndarray) and cf_pred.size > 1:
        cf_class = np.argmax(cf_pred)
    else:
        cf_class = int(cf_pred)
    
    # If no class change, return 0
    if orig_class == cf_class:
        return 0.0
    
    # Binary search for decision boundary
    low, high = 0.0, 1.0
    boundary_point = None
    
    for _ in range(num_steps):
        mid = (low + high) / 2.0
        interpolated = (1 - mid) * original_ts + mid * counterfactual_ts
        pred = model(interpolated)
        
        if hasattr(pred, 'numpy'):
            pred = pred.numpy()
        
        if isinstance(pred, np.ndarray) and pred.size > 1:
            pred_class = np.argmax(pred)
        else:
            pred_class = int(pred)
        
        if pred_class == orig_class:
            low = mid
        else:
            high = mid
            boundary_point = interpolated
    
    if boundary_point is not None:
        return float(np.linalg.norm(boundary_point - original_ts))
    else:
        return float(np.linalg.norm(counterfactual_ts - original_ts))


__all__ = ['prediction_change', 'class_probability_confidence', 'decision_boundary_distance']