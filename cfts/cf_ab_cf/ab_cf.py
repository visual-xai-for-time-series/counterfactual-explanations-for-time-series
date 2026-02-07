"""
AB-CF: Attention-Based Counterfactual Explanation for Multivariate Time Series

Paper: Li, P., Bahri, O., Boubrahimi, S. F., & Hamdi, S. M. (2023).
       "Attention-Based Counterfactual Explanation for Multivariate Time Series."
       In Proceedings of the 25th International Conference on Big Data Analytics 
       and Knowledge Discovery (DaWaK 2023), pp. 287-293. Springer.

Paper URL: https://link.springer.com/chapter/10.1007/978-3-031-39831-5_26
GitHub (Original): https://github.com/Luckilyeee/AB-CF
GitHub (Reference): https://github.com/Healthpy/cfe_tsc_pos

AB-CF uses Shannon entropy-based attention mechanism to identify and replace
high-uncertainty subsequences with segments from nearest unlike neighbors (NUN),
creating sparse and interpretable counterfactual explanations for multivariate
time series classification.

Key Features:
- Shannon entropy-based attention mechanism for segment importance
- Sliding window segmentation with configurable window size
- Nearest unlike neighbor (NUN) retrieval via DTW distance
- Selective segment replacement based on entropy ranking
- Sparse modifications focusing on high-uncertainty regions
- Support for multivariate time series
"""

import numpy as np
import torch
from scipy.stats import entropy
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


def sliding_window_3d(data, window_size, stride):
    """
    Extract subsequences using sliding window.
    
    Args:
        data: Input time series (n_features, n_timesteps)
        window_size: Size of sliding window
        stride: Stride for window movement
    
    Returns:
        Subsequences array (n_subsequences, n_features, window_size)
    """
    num_features, num_timesteps = data.shape
    num_subsequences = ((num_timesteps - window_size) // stride) + 1
    subsequences = np.zeros((num_subsequences, num_features, window_size))
    
    for j in range(num_subsequences):
        start = j * stride
        end = start + window_size
        subsequences[j] = data[:, start:end]
        
    return subsequences


def compute_entropy(predict_proba):
    """
    Compute Shannon entropy of prediction probabilities.
    
    Args:
        predict_proba: Probability distribution over classes
    
    Returns:
        Entropy value
    """
    if not np.any(predict_proba):
        return 0.0
    return entropy(predict_proba)


def native_guide_retrieval(query, target_label, X_train, y_train, metric='euclidean'):
    """
    Find nearest unlike neighbor (NUN) from target class.
    
    Args:
        query: Query time series (1, n_features, n_timesteps) or (n_features, n_timesteps)
        target_label: Target class label
        X_train: Training data (n_samples, n_features, n_timesteps)
        y_train: Training labels (n_samples,)
        metric: Distance metric for KNN
    
    Returns:
        Index of nearest unlike neighbor
    """
    # Ensure proper shapes
    if len(query.shape) == 2:
        query = query.reshape(1, query.shape[0], query.shape[1])
    
    # Convert target_label to integer
    if isinstance(target_label, np.ndarray):
        target_label = target_label.item()
    target_label = int(target_label)
    
    # Convert y_train to 1D array if needed
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)
    
    # Get target class samples
    target_indices = np.where(y_train == target_label)[0]
    if len(target_indices) == 0:
        print(f"Warning: No samples found for target class {target_label}")
        return None
    
    # Flatten for distance computation
    query_flat = query.reshape(query.shape[0], -1)
    target_samples = X_train[target_indices]
    if len(target_samples.shape) != 3:
        target_samples = target_samples.reshape(len(target_samples), 1, -1)
    target_samples_flat = target_samples.reshape(len(target_samples), -1)
    
    # Find nearest neighbor using sklearn
    knn = NearestNeighbors(n_neighbors=1, metric=metric)
    knn.fit(target_samples_flat)
    _, ind = knn.kneighbors(query_flat)
    
    nearest_idx = int(target_indices[int(ind[0][0])])
    return nearest_idx


def ab_cf_generate(sample, model, X_train, y_train, target_class=None, 
                   n_segments=10, window_size_ratio=0.1, verbose=False):
    """
    Generate counterfactual explanation using Attention-Based CF (AB-CF).
    
    Args:
        sample: Input time series to explain (1, n_features, n_timesteps) or (n_features, n_timesteps)
        model: Trained classifier model with predict_proba method
        X_train: Training data (n_samples, n_features, n_timesteps)
        y_train: Training labels (n_samples,)
        target_class: Target class for counterfactual (if None, second most likely)
        n_segments: Number of top-entropy segments to replace
        window_size_ratio: Ratio of time series length for window size
        verbose: Whether to print progress information
    
    Returns:
        cf: Counterfactual time series (1, n_features, n_timesteps)
        cf_label: Predicted label for counterfactual
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
    
    # Calculate window size and stride
    window_size = max(1, int(x.shape[1] * window_size_ratio))
    stride = window_size
    
    if verbose:
        print(f"Window size: {window_size}, Stride: {stride}")
    
    # Extract subsequences
    subsequences = sliding_window_3d(x, window_size, stride)
    
    if verbose:
        print(f"Extracted {len(subsequences)} subsequences")
    
    # Get total time series length from training data
    ts_length = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
    
    # Pad subsequences to match training data length
    padded_subsequences = np.pad(
        subsequences,
        ((0, 0), (0, 0), (0, ts_length - subsequences.shape[2])),
        mode='constant'
    )
    
    # Calculate entropy for each subsequence
    padded_tensor = torch.tensor(padded_subsequences, dtype=torch.float32, device=device)
    with torch.no_grad():
        predict_proba = model(padded_tensor)
        if isinstance(predict_proba, torch.Tensor):
            predict_proba = detach_to_numpy(predict_proba)
    
    entropies = np.array([compute_entropy(p) for p in predict_proba])
    
    # Sort subsequences by entropy (highest first)
    indices = np.argsort(entropies)[-n_segments:][::-1]
    
    if verbose:
        print(f"Top {n_segments} entropy values: {entropies[indices]}")
    
    # Find nearest unlike neighbor
    nun_idx = native_guide_retrieval(x, target_class, X_train, y_train)
    if nun_idx is None:
        if verbose:
            print("Failed to find nearest unlike neighbor")
        return None, None
    
    # Get the nearest unlike neighbor
    nun = X_train[nun_idx]
    if len(nun.shape) == 3:
        nun = nun.reshape(nun.shape[1], nun.shape[2])
    
    if verbose:
        print(f"Found NUN at index {nun_idx}")
    
    # Create counterfactual by replacing segments
    cf = x.copy()
    
    for idx in indices:
        start = int(idx * stride)
        end = int(start + window_size)
        
        # Ensure indices are within bounds
        if end > cf.shape[1]:
            end = cf.shape[1]
        
        # Replace segment with NUN segment
        cf[:, start:end] = nun[:, start:end]
        
        # Check if valid counterfactual
        cf_3d = cf.reshape(1, cf.shape[0], cf.shape[1])
        cf_tensor = torch.tensor(cf_3d, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            cf_probs = model(cf_tensor)
            if isinstance(cf_probs, torch.Tensor):
                cf_probs = detach_to_numpy(cf_probs)
            cf_pred = np.argmax(cf_probs)
        
        if cf_pred == target_class:
            if verbose:
                print(f"Valid CF found after replacing {len(indices[:np.where(indices == idx)[0][0]+1])} segments")
                print(f"CF prediction: {cf_pred} (prob={cf_probs[0, cf_pred]:.3f})")
            return cf.reshape(1, cf.shape[0], cf.shape[1]), cf_pred
    
    # If no valid CF found, return the last attempt
    if verbose:
        print("No valid CF found after replacing all selected segments")
        print(f"Final prediction: {cf_pred} (prob={cf_probs[0, cf_pred]:.3f})")
    
    return None, None


# Alias for backward compatibility
abcf_generate = ab_cf_generate
