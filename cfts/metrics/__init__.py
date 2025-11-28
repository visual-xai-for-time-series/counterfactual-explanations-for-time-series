"""
Metrics module for evaluating counterfactual explanations for time series.

This module provides comprehensive metrics for benchmarking counterfactual 
generation algorithms across multiple evaluation dimensions.
"""

from .validity import *
from .proximity import *
from .sparsity import *
from .realism import *
from .diversity import *
from .stability import *
from .keane import *
from .utils import *

__all__ = [
    # Validity metrics
    'prediction_change',
    'class_probability_confidence',
    'decision_boundary_distance',
    
    # Proximity metrics
    'l2_distance',
    'dtw_distance', 
    'frechet_distance',
    'manhattan_distance',
    'normalized_distance',
    'mahalanobis_distance',
    
    # Sparsity metrics
    'l0_norm',
    'percentage_changed_points',
    'segment_based_sparsity',
    'feature_sparsity',
    'temporal_sparsity_profile',
    'gini_sparsity_coefficient',
    
    # Realism metrics
    'domain_constraint_violations',
    'statistical_similarity',
    'temporal_consistency',
    'feature_range_validity',
    'autocorrelation_preservation',
    'spectral_similarity',
    
    # Diversity metrics
    'pairwise_distance',
    'coverage_metric',
    'novelty_metric',
    'diversity_index',
    'intra_cluster_distance',
    'feature_diversity',
    
    # Stability metrics
    'algorithmic_stability',
    'input_perturbation_robustness',
    'model_robustness',
    'hyperparameter_sensitivity',
    'convergence_stability',
    
    # Keane et al. (2021) metrics
    'validity',
    'proximity',
    'compactness',
    'evaluate_keane_metrics',
    
    # Utility classes and functions
    'CounterfactualEvaluator',
    'create_metric_suite',
    'benchmark_algorithms'
]