"""
Wrapper for the original TSInterpret LEFTIST explanation method.

This wrapper provides a simple interface to the original LEFTIST feature attribution
method from Guillemé et al. (2019), allowing it to be used alongside counterfactual
generation methods in this repository.

Original LEFTIST: "Agnostic Local Explanation for Time Series Classification"
Reference: https://github.com/fzi-forschungszentrum-informatik/TSInterpret
"""

import numpy as np
from typing import Optional, Tuple, Union

# Check if TSInterpret is available
try:
    from TSInterpret.InterpretabilityModels.leftist.leftist import LEFTIST
    TSINTERPRET_AVAILABLE = True
except ImportError:
    TSINTERPRET_AVAILABLE = False


def _check_tsinterpret():
    """Check if TSInterpret is installed and raise helpful error if not."""
    if not TSINTERPRET_AVAILABLE:
        raise ImportError(
            "TSInterpret is required to use the original LEFTIST explanation method.\n"
            "Install it with: pip install tsinterpret\n"
            "For more information: https://github.com/fzi-forschungszentrum-informatik/TSInterpret"
        )


def leftist_explain(
    sample: np.ndarray,
    dataset: tuple,
    model,
    mode: str = 'time',
    backend: str = 'PYT',
    transform_name: str = 'straight_line',
    learning_process_name: str = 'Lime',
    nb_interpretable_feature: int = 10,
    nb_neighbors: int = 1000,
    explanation_size: int = 5,
    idx_label: Optional[int] = None,
    random_state: int = 0
) -> np.ndarray:
    """
    Generate explanation using the original LEFTIST method from TSInterpret.
    
    This is the ORIGINAL LEFTIST explanation method that returns attribution weights
    showing which segments are important for classification (not counterfactuals).
    
    Args:
        sample: Time series sample to explain. Shape depends on mode:
                - If mode='time': (1, time, features) or (time, features)
                - If mode='feat': (1, features, time) or (features, time)
        dataset: Tuple of (X, y) reference dataset
        model: Trained classification model (PyTorch, TensorFlow, or sklearn)
        mode: 'time' or 'feat' - indicates which dimension is the time axis
        backend: 'PYT' (PyTorch), 'TF' (TensorFlow), or 'SK' (sklearn)
        transform_name: Type of transformation for neighbor generation:
                       - 'uniform' or 'mean': Mean transform
                       - 'straight_line': Linear interpolation
                       - 'background': Random background replacement
        learning_process_name: 'Lime' or 'Shap' - explanation algorithm
        nb_interpretable_feature: Number of segments to divide the time series into
        nb_neighbors: Number of neighbors to generate for explanation
        explanation_size: Number of most important features to highlight
        idx_label: Index of label to explain (None = explain all labels)
        random_state: Random seed for reproducibility
        
    Returns:
        Attribution weights showing segment importance.
        Shape: (n_labels, time_length) if idx_label is None, else (time_length,)
        
    Example:
        >>> from cfts.cf_leftist_original import leftist_explain
        >>> import torch
        >>> 
        >>> # Prepare data
        >>> X_train = ...  # (N, time, features)
        >>> y_train = ...  # (N,)
        >>> model = ...    # PyTorch model
        >>> sample = X_train[0]
        >>> 
        >>> # Get explanation
        >>> explanation = leftist_explain(
        ...     sample=sample,
        ...     dataset=(X_train, y_train),
        ...     model=model,
        ...     mode='time',
        ...     backend='PYT',
        ...     nb_interpretable_feature=10,
        ...     explanation_size=5
        ... )
        >>> 
        >>> # explanation now contains attribution weights for each segment
        >>> print(f"Explanation shape: {explanation.shape}")
    
    Reference:
        Guillemé, M., Masson, V., Rozé, L., & Termier, A. (2019).
        "Agnostic Local Explanation for Time Series Classification."
        2019 IEEE 31st International Conference on Tools with Artificial
        Intelligence (ICTAI). IEEE, 2019.
    """
    _check_tsinterpret()
    
    # Ensure sample has batch dimension
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    if sample.ndim == 2:
        if mode == 'time':
            sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        else:
            sample = sample.reshape(1, sample.shape[0], sample.shape[1])
    
    # Initialize LEFTIST explainer
    explainer = LEFTIST(
        model=model,
        data=dataset,
        mode=mode,
        backend=backend,
        transform_name=transform_name,
        segmentator_name='uniform',
        learning_process_name=learning_process_name,
        nb_interpretable_feature=nb_interpretable_feature,
        nb_neighbors=nb_neighbors,
        explanation_size=explanation_size
    )
    
    # Generate explanation
    explanation = explainer.explain(
        instance=sample,
        idx_label=idx_label,
        random_state=random_state
    )
    
    return np.array(explanation)


def leftist_explain_batch(
    samples: np.ndarray,
    dataset: tuple,
    model,
    mode: str = 'time',
    backend: str = 'PYT',
    transform_name: str = 'straight_line',
    learning_process_name: str = 'Lime',
    nb_interpretable_feature: int = 10,
    nb_neighbors: int = 1000,
    explanation_size: int = 5,
    idx_label: Optional[int] = None,
    random_state: int = 0,
    verbose: bool = False
) -> np.ndarray:
    """
    Generate explanations for multiple samples using the original LEFTIST method.
    
    This function processes multiple samples and returns their explanations.
    
    Args:
        samples: Multiple time series samples. Shape:
                - If mode='time': (batch, time, features)
                - If mode='feat': (batch, features, time)
        dataset: Tuple of (X, y) reference dataset
        model: Trained classification model
        mode: 'time' or 'feat'
        backend: 'PYT', 'TF', or 'SK'
        transform_name: Type of transformation ('uniform', 'straight_line', 'background')
        learning_process_name: 'Lime' or 'Shap'
        nb_interpretable_feature: Number of segments
        nb_neighbors: Number of neighbors to generate
        explanation_size: Number of important features to highlight
        idx_label: Index of label to explain (None = explain all labels)
        random_state: Random seed
        verbose: If True, print progress
        
    Returns:
        Batch of attribution weights.
        Shape: (batch, n_labels, time_length) if idx_label is None,
               else (batch, time_length)
        
    Example:
        >>> explanations = leftist_explain_batch(
        ...     samples=X_test[:10],
        ...     dataset=(X_train, y_train),
        ...     model=model,
        ...     backend='PYT',
        ...     verbose=True
        ... )
    """
    _check_tsinterpret()
    
    explanations = []
    for i, sample in enumerate(samples):
        if verbose and i % 10 == 0:
            print(f"Processing sample {i+1}/{len(samples)}")
        
        explanation = leftist_explain(
            sample=sample,
            dataset=dataset,
            model=model,
            mode=mode,
            backend=backend,
            transform_name=transform_name,
            learning_process_name=learning_process_name,
            nb_interpretable_feature=nb_interpretable_feature,
            nb_neighbors=nb_neighbors,
            explanation_size=explanation_size,
            idx_label=idx_label,
            random_state=random_state
        )
        explanations.append(explanation)
    
    return np.array(explanations)


# Convenience function for quick explanations
def quick_explain(
    sample: np.ndarray,
    dataset: tuple,
    model,
    backend: str = 'PYT'
) -> np.ndarray:
    """
    Quick explanation with default parameters.
    
    Args:
        sample: Time series sample to explain
        dataset: Tuple of (X, y) reference dataset
        model: Trained classification model
        backend: 'PYT', 'TF', or 'SK'
        
    Returns:
        Attribution weights
    """
    return leftist_explain(
        sample=sample,
        dataset=dataset,
        model=model,
        mode='time',
        backend=backend,
        transform_name='straight_line',
        learning_process_name='Lime',
        nb_interpretable_feature=10,
        nb_neighbors=1000,
        explanation_size=5
    )
