"""
Utility functions and metric suites for comprehensive counterfactual evaluation.

This module provides high-level interfaces for evaluating counterfactual
generation algorithms using multiple metrics simultaneously.
"""

import numpy as np
from typing import Dict, List, Callable, Any, Optional, Tuple
import warnings

from .validity import prediction_change, class_probability_confidence, decision_boundary_distance
from .proximity import l2_distance, manhattan_distance, normalized_distance
from .sparsity import l0_norm, percentage_changed_points, segment_based_sparsity
from .realism import (statistical_similarity, temporal_consistency, 
                     feature_range_validity, autocorrelation_preservation)
from .diversity import pairwise_distance, coverage_metric, novelty_metric, diversity_index
from .stability import algorithmic_stability, input_perturbation_robustness


class CounterfactualEvaluator:
    """
    Comprehensive evaluator for counterfactual generation algorithms.
    """
    
    def __init__(self, reference_data: Optional[np.ndarray] = None):
        """
        Initialize the evaluator.
        
        Args:
            reference_data: Reference dataset for distribution-based metrics
        """
        self.reference_data = reference_data
        
    def evaluate_single(self, 
                       original_ts: np.ndarray,
                       counterfactual_ts: np.ndarray,
                       model: Callable,
                       target_class: int,
                       constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Evaluate a single counterfactual using all applicable metrics.
        
        Args:
            original_ts: Original time series
            counterfactual_ts: Generated counterfactual
            model: Trained model for predictions
            target_class: Target class for the counterfactual
            constraints: Domain constraints for realism evaluation
        
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # Validity metrics
        try:
            results['prediction_change'] = prediction_change(
                original_ts, counterfactual_ts, model, target_class
            )
            results['class_confidence'] = class_probability_confidence(
                counterfactual_ts, model, target_class
            )
            results['boundary_distance'] = decision_boundary_distance(
                original_ts, counterfactual_ts, model
            )
        except Exception as e:
            warnings.warn(f"Error computing validity metrics: {e}")
        
        # Proximity metrics
        try:
            results['l2_distance'] = l2_distance(original_ts, counterfactual_ts)
            results['manhattan_distance'] = manhattan_distance(original_ts, counterfactual_ts)
            results['normalized_distance'] = normalized_distance(original_ts, counterfactual_ts)
        except Exception as e:
            warnings.warn(f"Error computing proximity metrics: {e}")
        
        # Sparsity metrics
        try:
            results['l0_norm'] = l0_norm(original_ts, counterfactual_ts)
            results['percentage_changed'] = percentage_changed_points(original_ts, counterfactual_ts)
            results['segment_sparsity'] = segment_based_sparsity(original_ts, counterfactual_ts)
        except Exception as e:
            warnings.warn(f"Error computing sparsity metrics: {e}")
        
        # Realism metrics
        try:
            if self.reference_data is not None:
                results['statistical_similarity'] = statistical_similarity(
                    original_ts, counterfactual_ts, self.reference_data
                )
                results['range_validity'] = feature_range_validity(
                    counterfactual_ts, self.reference_data
                )
            
            results['temporal_consistency'] = temporal_consistency(counterfactual_ts)
            results['autocorr_preservation'] = autocorrelation_preservation(
                original_ts, counterfactual_ts
            )
        except Exception as e:
            warnings.warn(f"Error computing realism metrics: {e}")
        
        return results
    
    def evaluate_multiple(self,
                         original_ts: np.ndarray,
                         counterfactuals: List[np.ndarray],
                         model: Callable,
                         target_class: int) -> Dict[str, float]:
        """
        Evaluate multiple counterfactuals for diversity and consistency.
        
        Args:
            original_ts: Original time series
            counterfactuals: List of generated counterfactuals
            model: Trained model for predictions
            target_class: Target class for counterfactuals
        
        Returns:
            Dictionary of diversity and consistency metrics
        """
        results = {}
        
        if len(counterfactuals) < 2:
            warnings.warn("Need at least 2 counterfactuals for diversity metrics")
            return results
        
        # Diversity metrics
        try:
            results['pairwise_distance'] = pairwise_distance(counterfactuals)
            results['diversity_index'] = diversity_index(counterfactuals)
            
            if self.reference_data is not None:
                results['coverage'] = coverage_metric(counterfactuals, self.reference_data)
                results['novelty'] = novelty_metric(counterfactuals, self.reference_data)
        except Exception as e:
            warnings.warn(f"Error computing diversity metrics: {e}")
        
        # Validity consistency across counterfactuals
        try:
            validity_scores = []
            for cf in counterfactuals:
                validity = prediction_change(original_ts, cf, model, target_class)
                validity_scores.append(validity)
            
            results['validity_consistency'] = np.std(validity_scores)
            results['validity_success_rate'] = np.mean(validity_scores)
        except Exception as e:
            warnings.warn(f"Error computing validity consistency: {e}")
        
        return results
    
    def evaluate_algorithm(self,
                          algorithm: Callable,
                          test_cases: List[Tuple[np.ndarray, int]],
                          model: Callable,
                          n_runs: int = 5,
                          **algorithm_kwargs) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of a counterfactual generation algorithm.
        
        Args:
            algorithm: Counterfactual generation function
            test_cases: List of (original_ts, target_class) tuples
            model: Trained model for predictions
            n_runs: Number of runs for stability assessment
            **algorithm_kwargs: Additional arguments for the algorithm
        
        Returns:
            Dictionary with aggregated results across test cases
        """
        all_results = {
            'single_metrics': [],
            'stability_metrics': [],
            'multiple_metrics': []
        }
        
        for original_ts, target_class in test_cases:
            # Single counterfactual evaluation
            try:
                cf = algorithm(original_ts, target_class=target_class, **algorithm_kwargs)
                single_results = self.evaluate_single(original_ts, cf, model, target_class)
                all_results['single_metrics'].append(single_results)
            except Exception as e:
                warnings.warn(f"Failed to generate single counterfactual: {e}")
                continue
            
            # Stability evaluation
            try:
                stability = algorithmic_stability(
                    algorithm, original_ts, n_runs=n_runs, 
                    target_class=target_class, **algorithm_kwargs
                )
                robustness = input_perturbation_robustness(
                    algorithm, original_ts, target_class=target_class, **algorithm_kwargs
                )
                
                all_results['stability_metrics'].append({
                    'algorithmic_stability': stability,
                    'input_robustness': robustness
                })
            except Exception as e:
                warnings.warn(f"Failed stability evaluation: {e}")
            
            # Multiple counterfactuals evaluation
            try:
                counterfactuals = []
                for _ in range(min(5, n_runs)):
                    cf = algorithm(original_ts, target_class=target_class, **algorithm_kwargs)
                    counterfactuals.append(cf)
                
                if len(counterfactuals) >= 2:
                    multiple_results = self.evaluate_multiple(
                        original_ts, counterfactuals, model, target_class
                    )
                    all_results['multiple_metrics'].append(multiple_results)
            except Exception as e:
                warnings.warn(f"Failed multiple counterfactuals evaluation: {e}")
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        return aggregated
    
    def _aggregate_results(self, all_results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """Aggregate results across test cases."""
        aggregated = {}
        
        for category, results_list in all_results.items():
            if not results_list:
                continue
            
            aggregated[category] = {}
            
            # Get all metric names
            all_metrics = set()
            for result in results_list:
                all_metrics.update(result.keys())
            
            # Calculate mean and std for each metric
            for metric in all_metrics:
                values = [result.get(metric, np.nan) for result in results_list]
                values = [v for v in values if not np.isnan(v)]
                
                if values:
                    aggregated[category][f'{metric}_mean'] = np.mean(values)
                    aggregated[category][f'{metric}_std'] = np.std(values)
                    aggregated[category][f'{metric}_median'] = np.median(values)
        
        return aggregated


def create_metric_suite(suite_type: str = 'comprehensive') -> List[str]:
    """
    Create predefined metric suites for different evaluation scenarios.
    
    Args:
        suite_type: Type of suite ('basic', 'comprehensive', 'research', 'production')
    
    Returns:
        List of metric names to include
    """
    suites = {
        'basic': [
            'prediction_change',
            'l2_distance', 
            'percentage_changed',
            'temporal_consistency'
        ],
        
        'comprehensive': [
            'prediction_change', 'class_confidence', 'boundary_distance',
            'l2_distance', 'manhattan_distance', 'normalized_distance',
            'l0_norm', 'percentage_changed', 'segment_sparsity',
            'statistical_similarity', 'temporal_consistency', 'range_validity',
            'pairwise_distance', 'diversity_index',
            'algorithmic_stability', 'input_robustness'
        ],
        
        'research': [
            'prediction_change', 'class_confidence', 'boundary_distance',
            'l2_distance', 'normalized_distance',
            'percentage_changed', 'segment_sparsity',
            'statistical_similarity', 'temporal_consistency', 
            'autocorr_preservation', 'range_validity',
            'pairwise_distance', 'coverage', 'novelty', 'diversity_index',
            'algorithmic_stability', 'input_robustness'
        ],
        
        'production': [
            'prediction_change',
            'normalized_distance',
            'percentage_changed',
            'temporal_consistency',
            'range_validity'
        ]
    }
    
    return suites.get(suite_type, suites['comprehensive'])


def benchmark_algorithms(algorithms: Dict[str, Callable],
                        test_cases: List[Tuple[np.ndarray, int]],
                        model: Callable,
                        reference_data: Optional[np.ndarray] = None,
                        suite_type: str = 'comprehensive') -> Dict[str, Dict]:
    """
    Benchmark multiple counterfactual generation algorithms.
    
    Args:
        algorithms: Dictionary of {name: algorithm_function}
        test_cases: List of (original_ts, target_class) tuples
        model: Trained model for predictions
        reference_data: Reference dataset for metrics
        suite_type: Metric suite to use
    
    Returns:
        Dictionary with results for each algorithm
    """
    evaluator = CounterfactualEvaluator(reference_data)
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"Evaluating algorithm: {name}")
        try:
            algo_results = evaluator.evaluate_algorithm(
                algorithm, test_cases, model
            )
            results[name] = algo_results
        except Exception as e:
            warnings.warn(f"Failed to evaluate {name}: {e}")
            results[name] = {}
    
    return results


__all__ = [
    'CounterfactualEvaluator',
    'create_metric_suite', 
    'benchmark_algorithms'
]