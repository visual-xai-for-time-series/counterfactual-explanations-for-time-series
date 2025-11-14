"""
Diversity metrics for counterfactual explanations.

These metrics evaluate the diversity of generated counterfactuals when
multiple counterfactuals are generated for the same input.
"""

import numpy as np
from typing import List, Optional
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans


def pairwise_distance(counterfactuals: List[np.ndarray], 
                     distance_metric: str = 'euclidean') -> float:
    """
    Calculates average pairwise distance between generated counterfactuals.
    
    Args:
        counterfactuals: List of counterfactual time series
        distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine')
    
    Returns:
        Average pairwise distance
    """
    if len(counterfactuals) < 2:
        return 0.0
    
    # Flatten each counterfactual for distance calculation
    cf_matrix = np.array([cf.flatten() for cf in counterfactuals])
    
    # Calculate pairwise distances
    distances = pairwise_distances(cf_matrix, metric=distance_metric)
    
    # Get upper triangular part (excluding diagonal)
    upper_tri_indices = np.triu_indices(len(counterfactuals), k=1)
    pairwise_dists = distances[upper_tri_indices]
    
    return float(np.mean(pairwise_dists))


def coverage_metric(counterfactuals: List[np.ndarray],
                   reference_data: np.ndarray,
                   k_clusters: int = 10) -> float:
    """
    Measures how well counterfactuals span the feature space using clustering.
    
    Args:
        counterfactuals: List of counterfactual time series
        reference_data: Reference dataset for feature space definition
        k_clusters: Number of clusters for coverage assessment
    
    Returns:
        Coverage score (0 = poor coverage, 1 = excellent coverage)
    """
    if len(counterfactuals) == 0:
        return 0.0
    
    # Flatten all data
    cf_matrix = np.array([cf.flatten() for cf in counterfactuals])
    ref_matrix = reference_data.reshape(reference_data.shape[0], -1)
    
    # Fit clusters on reference data
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    kmeans.fit(ref_matrix)
    
    # Predict clusters for counterfactuals
    cf_clusters = kmeans.predict(cf_matrix)
    
    # Calculate coverage as the fraction of clusters represented
    unique_clusters = len(np.unique(cf_clusters))
    coverage = unique_clusters / k_clusters
    
    return float(coverage)


def novelty_metric(counterfactuals: List[np.ndarray],
                  training_data: np.ndarray,
                  k_neighbors: int = 5) -> float:
    """
    Measures uniqueness compared to training data using k-NN distance.
    
    Args:
        counterfactuals: List of counterfactual time series
        training_data: Training dataset for novelty comparison
        k_neighbors: Number of neighbors for distance calculation
    
    Returns:
        Average novelty score (higher = more novel)
    """
    if len(counterfactuals) == 0:
        return 0.0
    
    # Flatten all data
    cf_matrix = np.array([cf.flatten() for cf in counterfactuals])
    train_matrix = training_data.reshape(training_data.shape[0], -1)
    
    novelty_scores = []
    
    for cf in cf_matrix:
        # Calculate distances to all training samples
        distances = np.linalg.norm(train_matrix - cf, axis=1)
        
        # Get k nearest neighbors
        k_nearest_distances = np.partition(distances, k_neighbors)[:k_neighbors]
        
        # Average distance to k nearest neighbors as novelty score
        novelty = np.mean(k_nearest_distances)
        novelty_scores.append(novelty)
    
    return float(np.mean(novelty_scores))


def diversity_index(counterfactuals: List[np.ndarray]) -> float:
    """
    Calculates Shannon diversity index for counterfactuals.
    
    Args:
        counterfactuals: List of counterfactual time series
    
    Returns:
        Shannon diversity index
    """
    if len(counterfactuals) < 2:
        return 0.0
    
    # Flatten and discretize counterfactuals for diversity calculation
    cf_matrix = np.array([cf.flatten() for cf in counterfactuals])
    
    # Quantize values into bins for discrete counting
    n_bins = min(50, len(counterfactuals) * 2)
    quantized = []
    
    for i in range(cf_matrix.shape[1]):
        feature_values = cf_matrix[:, i]
        bins = np.linspace(feature_values.min(), feature_values.max(), n_bins)
        quantized_feature = np.digitize(feature_values, bins)
        quantized.append(quantized_feature)
    
    quantized_matrix = np.array(quantized).T
    
    # Convert each row to a tuple for hashing
    unique_patterns = []
    for row in quantized_matrix:
        unique_patterns.append(tuple(row))
    
    # Count unique patterns
    unique_counts = {}
    for pattern in unique_patterns:
        unique_counts[pattern] = unique_counts.get(pattern, 0) + 1
    
    # Calculate Shannon diversity
    total = len(unique_patterns)
    diversity = 0.0
    
    for count in unique_counts.values():
        proportion = count / total
        if proportion > 0:
            diversity -= proportion * np.log(proportion)
    
    return float(diversity)


def intra_cluster_distance(counterfactuals: List[np.ndarray],
                          original_ts: np.ndarray) -> float:
    """
    Measures diversity within counterfactuals relative to original instance.
    
    Args:
        counterfactuals: List of counterfactual time series
        original_ts: Original time series for reference
    
    Returns:
        Average intra-cluster distance normalized by distance to original
    """
    if len(counterfactuals) < 2:
        return 0.0
    
    # Calculate pairwise distance among counterfactuals
    avg_pairwise = pairwise_distance(counterfactuals)
    
    # Calculate average distance from counterfactuals to original
    orig_flat = original_ts.flatten()
    cf_matrix = np.array([cf.flatten() for cf in counterfactuals])
    
    orig_distances = [np.linalg.norm(cf - orig_flat) for cf in cf_matrix]
    avg_orig_distance = np.mean(orig_distances)
    
    # Normalize intra-cluster distance by distance to original
    if avg_orig_distance == 0:
        return 0.0
    
    normalized_diversity = avg_pairwise / avg_orig_distance
    return float(normalized_diversity)


def feature_diversity(counterfactuals: List[np.ndarray]) -> np.ndarray:
    """
    Calculates diversity for each feature/time point separately.
    
    Args:
        counterfactuals: List of counterfactual time series
    
    Returns:
        Array of diversity scores for each feature/time point
    """
    if len(counterfactuals) < 2:
        return np.zeros(counterfactuals[0].size)
    
    cf_matrix = np.array([cf.flatten() for cf in counterfactuals])
    
    # Calculate standard deviation for each feature as diversity measure
    feature_diversities = np.std(cf_matrix, axis=0)
    
    return feature_diversities


__all__ = [
    'pairwise_distance',
    'coverage_metric',
    'novelty_metric',
    'diversity_index',
    'intra_cluster_distance',
    'feature_diversity'
]