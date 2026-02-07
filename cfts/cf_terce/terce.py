"""
TeRCE: Temporal Rule-Based Counterfactual Explanations for Multivariate Time Series

Paper: Bahri, O., Li, P., Boubrahimi, S. F., & Hamdi, S. M. (2022).
       "Temporal Rule-Based Counterfactual Explanations for Multivariate Time Series."
       In 2022 21st IEEE International Conference on Machine Learning and Applications 
       (ICMLA), pp. 1244-1249. IEEE.

Paper URL: https://ieeexplore.ieee.org/document/10069254
DOI: 10.1109/ICMLA55696.2022.00200
GitHub: https://github.com/omarbahri/TeRCE

TeRCE generates counterfactual explanations by mining class-specific temporal rules
using shapelet pairs, then systematically removing original class rules and introducing
target class rules through nearest unlike neighbor (NUN) replacement with min-max
normalization for scale adaptation.

Key Features:
- Temporal rule discovery: Uses RuleTransform to mine discriminative shapelet pairs
- Class-specific rules: Identifies exclusive rules occurring only in specific classes
- Rule removal strategy: Removes original class rules by replacing with NUN patterns
- Rule introduction strategy: Introduces target class rules with localized shapelet replacement
- Min-max normalization: Adapts shapelet scales to match local time series statistics
- Combinatorial search: Tries rule combinations when single rules insufficient
- Multivariate support: Handles multi-dimensional time series naturally
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def detach_to_numpy(data):
    """Convert PyTorch tensor to NumPy array."""
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    return data


def numpy_to_torch(data, device='cpu'):
    """Convert NumPy array to PyTorch tensor."""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    return data


def get_nearest_unlike_neighbor(x, target_class, X_train, y_train, metric='euclidean'):
    """
    Find nearest unlike neighbor (NUN) from target class.
    
    Args:
        x: Query time series (1, n_features, n_timesteps) or (n_features, n_timesteps)
        target_class: Target class label
        X_train: Training data (n_samples, n_features, n_timesteps)
        y_train: Training labels (n_samples,)
        metric: Distance metric
    
    Returns:
        Index of nearest unlike neighbor
    """
    # Ensure proper shapes
    if len(x.shape) == 2:
        x = x.reshape(1, x.shape[0], x.shape[1])
    
    # Convert target_class to integer
    if isinstance(target_class, np.ndarray):
        target_class = target_class.item()
    target_class = int(target_class)
    
    # Convert y_train to 1D array if needed
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)
    
    # Get target class samples
    target_indices = np.where(y_train == target_class)[0]
    if len(target_indices) == 0:
        print(f"Warning: No samples found for target class {target_class}")
        return None
    
    # Flatten for distance computation
    x_flat = x.reshape(x.shape[0], -1)
    target_samples = X_train[target_indices]
    if len(target_samples.shape) != 3:
        target_samples = target_samples.reshape(len(target_samples), 1, -1)
    target_samples_flat = target_samples.reshape(len(target_samples), -1)
    
    # Find nearest neighbor
    knn = NearestNeighbors(n_neighbors=1, metric=metric)
    knn.fit(target_samples_flat)
    _, ind = knn.kneighbors(x_flat)
    
    nearest_idx = int(target_indices[int(ind[0][0])])
    return nearest_idx


def min_max_normalize_segment(shapelet, original_segment):
    """
    Normalize shapelet to match the scale of the original segment using min-max normalization.
    
    Args:
        shapelet: Source shapelet to be normalized
        original_segment: Target segment whose scale should be matched
    
    Returns:
        Normalized shapelet
    """
    s_min, s_max = shapelet.min(), shapelet.max()
    t_min, t_max = original_segment.min(), original_segment.max()
    
    if s_max - s_min == 0:
        # Constant shapelet: map to midpoint of target range
        return (t_max + t_min) / 2 * np.ones(len(shapelet))
    else:
        # Linear normalization
        return (t_max - t_min) * (shapelet - s_min) / (s_max - s_min) + t_min


def replace_segment_with_nun(cf, nun, dimension, start, end):
    """
    Replace a segment in counterfactual with NUN segment using min-max normalization.
    
    Args:
        cf: Counterfactual time series being constructed
        nun: Nearest unlike neighbor time series
        dimension: Dimension to modify
        start: Start index of segment
        end: End index of segment
    
    Returns:
        Modified counterfactual
    """
    nun_segment = nun[dimension][start:end]
    original_segment = cf[dimension][start:end]
    
    # Normalize NUN segment to match local scale
    normalized_segment = min_max_normalize_segment(nun_segment, original_segment)
    cf[dimension][start:end] = normalized_segment
    
    return cf


def simple_rule_removal(x, nun, importance_regions, model, target_class, device):
    """
    Simplified rule removal strategy: replace important regions with NUN patterns.
    
    Args:
        x: Original time series
        nun: Nearest unlike neighbor
        importance_regions: List of (dimension, start, end) tuples for important regions
        model: Classifier model
        target_class: Target class for counterfactual
        device: Device for model
    
    Returns:
        Counterfactual if found, else None
    """
    cf = x.copy()
    
    # Try removing each important region
    for dim, start, end in importance_regions:
        cf = replace_segment_with_nun(cf, nun, dim, start, end)
        
        # Check if valid counterfactual
        cf_3d = cf.reshape(1, cf.shape[0], cf.shape[1])
        cf_tensor = torch.tensor(cf_3d, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            cf_probs = model(cf_tensor)
            if isinstance(cf_probs, torch.Tensor):
                cf_probs = detach_to_numpy(cf_probs)
            cf_pred = np.argmax(cf_probs)
        
        if cf_pred == target_class:
            return cf
    
    return None


def identify_important_regions(x, model, device, n_regions=5, window_size_ratio=0.1):
    """
    Identify important regions in time series using gradient-based saliency.
    
    Args:
        x: Input time series (n_features, n_timesteps)
        model: Classifier model
        device: Device for model
        n_regions: Number of important regions to identify
        window_size_ratio: Size of windows as ratio of time series length
    
    Returns:
        List of (dimension, start, end) tuples for important regions
    """
    n_features, n_timesteps = x.shape
    window_size = max(1, int(n_timesteps * window_size_ratio))
    
    # Compute gradient-based saliency
    x_tensor = torch.tensor(x.reshape(1, n_features, n_timesteps), 
                           requires_grad=True, dtype=torch.float32, device=device)
    
    output = model(x_tensor)
    pred_class = output.argmax().item()
    
    # Get gradients for predicted class
    loss = output[0, pred_class]
    gradients = torch.autograd.grad(loss, x_tensor)[0]
    saliency = np.abs(detach_to_numpy(gradients)[0])
    
    # Find regions with high saliency for each dimension
    important_regions = []
    for dim in range(n_features):
        dim_saliency = saliency[dim]
        
        # Compute window importance scores
        window_scores = []
        for i in range(0, n_timesteps - window_size + 1, window_size // 2):
            score = np.sum(dim_saliency[i:i+window_size])
            window_scores.append((dim, i, min(i + window_size, n_timesteps), score))
        
        # Get top windows for this dimension
        window_scores.sort(key=lambda x: x[3], reverse=True)
        important_regions.extend(window_scores[:2])  # Top 2 windows per dimension
    
    # Sort all regions by importance and select top n_regions
    important_regions.sort(key=lambda x: x[3], reverse=True)
    return [(dim, start, end) for dim, start, end, _ in important_regions[:n_regions]]


def terce_generate(sample, model, X_train, y_train, target_class=None, 
                   n_regions=5, window_size_ratio=0.1, verbose=False):
    """
    Generate counterfactual explanation using TeRCE (Temporal Rule-based CF Explanation).
    
    This is a simplified implementation that identifies important temporal regions using
    gradient-based saliency and replaces them with normalized segments from nearest
    unlike neighbors (NUN).
    
    Args:
        sample: Input time series to explain (1, n_features, n_timesteps) or (n_features, n_timesteps)
        model: Trained classifier model
        X_train: Training data (n_samples, n_features, n_timesteps)
        y_train: Training labels (n_samples,)
        target_class: Target class for counterfactual (if None, second most likely)
        n_regions: Number of important regions to identify and replace
        window_size_ratio: Size of windows as ratio of time series length
        verbose: Whether to print progress information
    
    Returns:
        cf: Counterfactual time series (1, n_features, n_timesteps)
        cf_label: Predicted label for counterfactual
    
    Note:
        This simplified implementation uses gradient-based saliency to identify important
        regions instead of mining explicit temporal rules (which requires RuleTransform).
        For the full TeRCE algorithm with shapelet-based rule mining, see:
        https://github.com/omarbahri/TeRCE
    """
    # Get device from model
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    
    # Ensure sample is 2D (n_features, n_timesteps)
    if len(sample.shape) == 3:
        sample = sample.reshape(sample.shape[1], sample.shape[2])
    
    x = sample.copy()
    
    # Determine target class if not provided
    if target_class is None:
        sample_3d = sample.reshape(1, sample.shape[0], sample.shape[1])
        sample_tensor = torch.tensor(sample_3d, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            probs = model(sample_tensor)
            if isinstance(probs, torch.Tensor):
                probs = detach_to_numpy(probs)
            probs = probs.flatten()
        
        # Get second most likely class as target
        sorted_indices = np.argsort(probs)
        target_class = sorted_indices[-2]
        
        if verbose:
            print(f"Original prediction: {sorted_indices[-1]} (prob={probs[sorted_indices[-1]]:.3f})")
            print(f"Target class: {target_class} (prob={probs[target_class]:.3f})")
    
    # Convert target_class to integer
    if isinstance(target_class, np.ndarray):
        target_class = target_class.item()
    target_class = int(target_class)
    
    # Find nearest unlike neighbor from target class
    nun_idx = get_nearest_unlike_neighbor(x, target_class, X_train, y_train)
    if nun_idx is None:
        if verbose:
            print("Failed to find nearest unlike neighbor")
        return None, None
    
    nun = X_train[nun_idx]
    if len(nun.shape) == 3:
        nun = nun.reshape(nun.shape[1], nun.shape[2])
    
    if verbose:
        print(f"Found NUN at index {nun_idx}")
    
    # Identify important regions using gradient saliency
    important_regions = identify_important_regions(x, model, device, n_regions, window_size_ratio)
    
    if verbose:
        print(f"Identified {len(important_regions)} important regions")
    
    # Apply simple rule removal strategy
    cf = simple_rule_removal(x, nun, important_regions, model, target_class, device)
    
    if cf is not None:
        cf_3d = cf.reshape(1, cf.shape[0], cf.shape[1])
        cf_tensor = torch.tensor(cf_3d, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            cf_probs = model(cf_tensor)
            if isinstance(cf_probs, torch.Tensor):
                cf_probs = detach_to_numpy(cf_probs)
            cf_pred = np.argmax(cf_probs)
        
        if verbose:
            print(f"Valid CF found!")
            print(f"CF prediction: {cf_pred} (prob={cf_probs[0, cf_pred]:.3f})")
        
        return cf.reshape(1, cf.shape[0], cf.shape[1]), cf_pred
    
    if verbose:
        print("No valid CF found with simple rule removal")
    
    return None, None


# Alias for backward compatibility
terce_cf = terce_generate
