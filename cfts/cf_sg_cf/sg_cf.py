"""
SG-CF: Shapelet-Guided Counterfactual Explanation for Time Series Classification

Implementation based on Li et al. (2022):
"SG-CF: Shapelet-Guided Counterfactual Explanation for Time Series Classification"
2022 IEEE International Conference on Big Data (Big Data)

SG-CF extends the Wachter counterfactual framework with shapelet-based guidance to:
1. Extract discriminative shapelets from training data
2. Use shapelets to identify critical regions for modification
3. Guide gradient-based optimization to focus on shapelet locations
4. Balance validity, proximity, sparsity, and contiguity

The method uses shapelet distance in the loss function and gradient masking to
concentrate modifications within shapelet regions, generating more interpretable
and focused counterfactuals.

Reference:
@inproceedings{li2022sg,
  title={SG-CF: Shapelet-Guided Counterfactual Explanation for Time Series Classification},
  author={Li, Peiyu and Bahri, Omar and Boubrahimi, Souka{\"\i}na Filali and Hamdi, Shah Muhammad},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={1564--1569},
  year={2022},
  organization={IEEE},
  doi={10.1109/bigdata55660.2022.10020866}
}

Links:
- Paper: https://ieeexplore.ieee.org/document/10020866
- GitHub: https://github.com/Luckilyeee/SG-CF
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional, Tuple, List, Dict, Union
from sklearn.cluster import KMeans
import warnings


def extract_shapelets_simple(X_train, y_train, n_shapelets_per_class=10,
                             shapelet_lengths=None, random_state=42):
    """
    Extract discriminative shapelets from training data using k-means clustering.
    
    Args:
        X_train: Training time series (N, C, L) or (N, L)
        y_train: Training labels
        n_shapelets_per_class: Number of shapelets to extract per class
        shapelet_lengths: List of shapelet lengths (default: [0.2, 0.4, 0.6] * series length)
        random_state: Random seed
        
    Returns:
        Dictionary mapping class labels to list of (shapelet, start_idx, length) tuples
    """
    np.random.seed(random_state)
    
    # Ensure proper shape (N, C, L)
    if X_train.ndim == 2:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    n_samples, n_channels, n_timesteps = X_train.shape
    classes = np.unique(y_train)
    
    if shapelet_lengths is None:
        shapelet_lengths = [
            int(n_timesteps * 0.2),
            int(n_timesteps * 0.4),
            int(n_timesteps * 0.6)
        ]
    
    shapelets_by_class = {c: [] for c in classes}
    
    # Extract shapelets for each class
    for c in classes:
        X_class = X_train[y_train == c]
        
        # Flatten to univariate if multivariate
        if n_channels > 1:
            X_class_flat = X_class.reshape(X_class.shape[0], -1)
        else:
            X_class_flat = X_class[:, 0, :]
        
        for length in shapelet_lengths:
            if length >= X_class_flat.shape[1]:
                continue
            
            # Extract candidate subsequences
            candidates = []
            candidate_indices = []
            for i, ts in enumerate(X_class_flat):
                for start in range(0, X_class_flat.shape[1] - length + 1, max(1, length // 4)):
                    shapelet = ts[start:start + length]
                    # Normalize
                    if shapelet.std() > 0:
                        shapelet = (shapelet - shapelet.mean()) / shapelet.std()
                    candidates.append(shapelet)
                    candidate_indices.append((i, start, length))
            
            if len(candidates) == 0:
                continue
            
            candidates = np.array(candidates)
            
            # Use k-means to find representative shapelets
            n_clusters = min(n_shapelets_per_class, len(candidates))
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                kmeans.fit(candidates)
                
                # For each cluster center, find the closest actual candidate
                for center in kmeans.cluster_centers_:
                    distances = np.linalg.norm(candidates - center, axis=1)
                    closest_idx = np.argmin(distances)
                    orig_idx, start_pos, shap_len = candidate_indices[closest_idx]
                    
                    # Store original shapelet (not normalized)
                    original_shapelet = X_class_flat[orig_idx, start_pos:start_pos + shap_len]
                    shapelets_by_class[c].append((original_shapelet, start_pos, shap_len))
    
    return shapelets_by_class


def compute_shapelet_distance(ts, shapelet):
    """
    Compute minimum distance between time series and shapelet using sliding window.
    
    Args:
        ts: Time series tensor (can be multi-dimensional)
        shapelet: Shapelet array/tensor
        
    Returns:
        Minimum distance across all positions
    """
    if isinstance(shapelet, np.ndarray):
        shapelet = torch.tensor(shapelet, dtype=torch.float32, device=ts.device)
    
    if ts.dim() == 1:
        ts_1d = ts
    else:
        ts_1d = ts.flatten()
    
    shapelet_len = len(shapelet)
    
    if len(ts_1d) < shapelet_len:
        return torch.norm(ts_1d - shapelet[:len(ts_1d)])
    
    # Sliding window distance computation
    min_dist = float('inf')
    for i in range(len(ts_1d) - shapelet_len + 1):
        window = ts_1d[i:i + shapelet_len]
        
        # Normalize window
        if window.std() > 0:
            window_norm = (window - window.mean()) / window.std()
        else:
            window_norm = window - window.mean()
        
        # Normalize shapelet
        if shapelet.std() > 0:
            shapelet_norm = (shapelet - shapelet.mean()) / shapelet.std()
        else:
            shapelet_norm = shapelet - shapelet.mean()
        
        dist = torch.norm(window_norm - shapelet_norm)
        if dist < min_dist:
            min_dist = dist
    
    return min_dist


def find_prominent_segment(gradient, seg_len):
    """
    Find the segment with maximum gradient magnitude (most important for change).
    
    Args:
        gradient: Gradient tensor
        seg_len: Length of segment to find
        
    Returns:
        (start_idx, end_idx) tuple
    """
    if isinstance(gradient, torch.Tensor):
        gradient = gradient.detach().cpu().numpy()
    
    gradient = gradient.flatten()
    
    max_gradient = 0
    idx_start, idx_end = 0, seg_len
    
    for i in range(len(gradient) - seg_len + 1):
        seg_sum = np.sum(np.abs(gradient[i:i + seg_len]))
        if seg_sum > max_gradient:
            max_gradient = seg_sum
            idx_start, idx_end = i, i + seg_len
    
    return idx_start, idx_end


def sg_cf(sample, dataset, model, target_class=None,
          prototype=None, shapelets=None,
          max_iter=1000, max_lambda_steps=10,
          lambda_init=0.1, learning_rate=0.1,
          segment_rate_init=0.05, segment_rate_max=0.7, segment_rate_step=0.01,
          target_proba=0.95, distance='l1', early_stop=50,
          device=None, verbose=False):
    """
    Generate counterfactual using SG-CF (Shapelet-Guided Counterfactual) method.
    
    This implements the SG-CF algorithm from Li et al. (2022) which combines:
    - Wachter-style optimization with distance and prediction loss
    - Shapelet-based guidance to focus modifications
    - Gradient masking within shapelet regions
    - Progressive segment expansion strategy
    
    Args:
        sample: Time series instance to explain (numpy array)
        dataset: Training dataset for shapelet extraction
        model: Trained classifier model
        target_class: Target class for counterfactual (optional)
        prototype: Prototype time series from target class (optional)
        shapelets: Pre-extracted shapelets (optional)
        max_iter: Maximum iterations per lambda step
        max_lambda_steps: Maximum lambda adjustment steps
        lambda_init: Initial lambda value for distance-prediction trade-off
        learning_rate: Learning rate for Adam optimizer
        segment_rate_init: Initial segment rate (fraction of time series)
        segment_rate_max: Maximum segment rate
        segment_rate_step: Increment for segment rate
        target_proba: Target probability threshold
        distance: Distance metric ('l1' or 'l2')
        early_stop: Early stopping threshold
        device: Device to run on
        verbose: Print progress information
        
    Returns:
        Tuple of (counterfactual, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    sample_orig = sample.copy()
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(sample_tensor.shape) == 2:
        sample_tensor = sample_tensor.unsqueeze(0)
    
    # Get original prediction
    with torch.no_grad():
        original_pred = model(sample_tensor)
        original_class = torch.argmax(original_pred).item()
        original_proba = torch.softmax(original_pred, dim=-1).squeeze().cpu().numpy()
    
    # Determine target class
    if target_class is None:
        sorted_classes = torch.argsort(original_pred, descending=True).squeeze()
        target_class = sorted_classes[1].item()
    
    if original_class == target_class:
        if verbose:
            print("SG-CF: Sample already in target class")
        return None, None
    
    if verbose:
        print(f"SG-CF: Original class={original_class} (p={original_proba[original_class]:.3f}), "
              f"Target class={target_class} (p={original_proba[target_class]:.3f})")
    
    # Extract shapelets if not provided
    if shapelets is None:
        if verbose:
            print("SG-CF: Extracting shapelets from training data...")
        
        X_train, y_train = [], []
        for i in range(min(len(dataset), 500)):
            item = dataset[i]
            ts, label = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, 0)
            ts_np = np.array(ts)
            if ts_np.ndim == 1:
                ts_np = ts_np.reshape(1, -1)
            elif ts_np.ndim == 3:
                ts_np = ts_np.squeeze(0)
            X_train.append(ts_np)
            
            # Convert label to scalar
            if hasattr(label, 'shape') and len(label.shape) > 0:
                label_scalar = int(np.argmax(label))
            else:
                label_scalar = int(label)
            y_train.append(label_scalar)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        shapelets = extract_shapelets_simple(X_train, y_train, n_shapelets_per_class=10)
    
    # Get target class shapelets
    target_shapelets = shapelets.get(target_class, [])
    if verbose and len(target_shapelets) > 0:
        print(f"SG-CF: Using {len(target_shapelets)} shapelets from target class")
    
    # Get prototype if not provided
    if prototype is None:
        # Find nearest neighbor from target class in training data
        target_samples = X_train[y_train == target_class]
        if len(target_samples) > 0:
            distances = [np.linalg.norm(sample - ts) for ts in target_samples]
            prototype = target_samples[np.argmin(distances)]
        else:
            prototype = sample.copy()
    
    proto_tensor = torch.tensor(prototype, dtype=torch.float32, device=device)
    if len(proto_tensor.shape) == 2:
        proto_tensor = proto_tensor.unsqueeze(0)
    
    # Distance function
    if distance == 'l1':
        dist_fn = lambda x, y: torch.sum(torch.abs(x - y))
    else:
        dist_fn = lambda x, y: torch.sqrt(torch.sum((x - y) ** 2))
    
    # Initial lambda sweep to find bounds
    n_orders = 10
    n_steps = max_iter // n_orders
    lambdas = np.array([lambda_init / (10 ** i) for i in range(n_orders)])
    cf_count = np.zeros_like(lambdas)
    
    if verbose:
        print(f"SG-CF: Initial lambda sweep with {n_orders} orders")
    
    # Quick lambda sweep
    for ix, lam_val in enumerate(lambdas):
        cf_tensor = sample_tensor.clone().detach().requires_grad_(True)
        optimizer = Adam([cf_tensor], lr=learning_rate)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Compute losses
            loss_dist_orig = dist_fn(cf_tensor, sample_tensor)
            loss_dist_proto = dist_fn(cf_tensor, proto_tensor)
            
            # Shapelet distance
            loss_shapelet = torch.tensor(0.0, device=device)
            if len(target_shapelets) > 0:
                shapelet_dists = []
                for shap, _, _ in target_shapelets[:3]:  # Use top 3 shapelets
                    shapelet_dists.append(compute_shapelet_distance(cf_tensor.squeeze(), shap))
                loss_shapelet = torch.mean(torch.stack(shapelet_dists))
            
            # Prediction loss
            pred = model(cf_tensor)
            target_tensor = torch.tensor([target_class], dtype=torch.long, device=device)
            loss_pred = nn.CrossEntropyLoss()(pred, target_tensor)
            
            # Combined loss (following SG-CF paper)
            loss_total = lam_val * (loss_dist_orig + loss_dist_proto + 0.5 * loss_shapelet) + 2 * loss_pred ** 2
            
            loss_total.backward()
            optimizer.step()
        
        # Check if counterfactual achieved
        with torch.no_grad():
            cf_pred = model(cf_tensor)
            if torch.argmax(cf_pred).item() == target_class:
                cf_count[ix] += 1
    
    # Find lambda bounds
    lb_ix = np.where(cf_count > 0)[0][0] if np.any(cf_count > 0) else 0
    ub_ix = np.where(cf_count == 0)[0][-1] if np.any(cf_count == 0) else 0
    
    lam_lb = lambdas[lb_ix]
    lam_ub = lambdas[ub_ix] if ub_ix > 0 else lambda_init
    lam = (lam_lb + lam_ub) / 2
    
    if verbose:
        print(f"SG-CF: Lambda bounds: [{lam_lb:.6f}, {lam_ub:.6f}], starting at {lam:.6f}")
        print(f"SG-CF: CF count: {cf_count}")
    
    # Main optimization with shapelet-guided segment focusing
    best_cf = None
    best_pred = None
    best_validity = 0.0
    best_distance = float('inf')
    
    segment_rate = segment_rate_init
    
    while segment_rate < segment_rate_max:
        cf_tensor = sample_tensor.clone().detach().requires_grad_(True)
        
        seg_len = int(np.round(segment_rate * cf_tensor.shape[-1]))
        seg_len = max(5, seg_len)  # Minimum segment length
        
        if verbose:
            print(f"\nSG-CF: Segment rate={segment_rate:.2f}, seg_len={seg_len}")
        
        for l_step in range(max_lambda_steps):
            optimizer = Adam([cf_tensor], lr=learning_rate)
            
            found = 0
            not_found = 0
            shapelet_region_start = None
            shapelet_region_end = None
            
            for i in range(max_iter):
                optimizer.zero_grad()
                
                # Compute losses
                loss_dist_orig = dist_fn(cf_tensor, sample_tensor)
                loss_dist_proto = dist_fn(cf_tensor, proto_tensor)
                
                # Shapelet distance
                loss_shapelet = torch.tensor(0.0, device=device)
                if len(target_shapelets) > 0:
                    shapelet_dists = []
                    for shap, _, _ in target_shapelets[:5]:
                        shapelet_dists.append(compute_shapelet_distance(cf_tensor.squeeze(), shap))
                    loss_shapelet = torch.mean(torch.stack(shapelet_dists))
                
                # Prediction loss
                pred = model(cf_tensor)
                target_tensor = torch.tensor([target_class], dtype=torch.long, device=device)
                loss_pred = nn.CrossEntropyLoss()(pred, target_tensor)
                
                # Combined loss
                loss_total = lam * (loss_dist_orig + loss_dist_proto + 0.5 * loss_shapelet) + 2 * loss_pred ** 2
                
                loss_total.backward()
                
                # Apply shapelet-guided gradient masking
                if i == 0 and l_step == 0:
                    # Determine prominent segment based on gradient
                    gradient = cf_tensor.grad.squeeze().cpu().detach().numpy()
                    shapelet_region_start, shapelet_region_end = find_prominent_segment(
                        gradient, seg_len
                    )
                    
                    if verbose and l_step == 0:
                        print(f"  Shapelet region: [{shapelet_region_start}, {shapelet_region_end}]")
                
                # Mask gradients to focus on shapelet region
                if shapelet_region_start is not None and shapelet_region_end is not None:
                    grad_mask = torch.zeros_like(cf_tensor.grad)
                    grad_mask[..., shapelet_region_start:shapelet_region_end] = \
                        cf_tensor.grad[..., shapelet_region_start:shapelet_region_end]
                    cf_tensor.grad = grad_mask
                
                optimizer.step()
                
                # Check counterfactual condition
                with torch.no_grad():
                    cf_pred = model(cf_tensor)
                    cf_proba = torch.softmax(cf_pred, dim=-1).squeeze().cpu().numpy()
                    current_class = np.argmax(cf_proba)
                    current_validity = cf_proba[target_class]
                    current_distance = dist_fn(cf_tensor, sample_tensor).item()
                    
                    if current_class == target_class:
                        found += 1
                        not_found = 0
                        
                        # Update best if better validity or lower distance
                        if current_validity > best_validity or \
                           (current_validity >= target_proba and current_distance < best_distance):
                            best_validity = current_validity
                            best_distance = current_distance
                            best_cf = cf_tensor.clone().detach()
                            best_pred = cf_proba
                            
                            if verbose and current_validity >= target_proba:
                                print(f"  Iter {i}: Found CF with validity={current_validity:.4f}, "
                                      f"distance={current_distance:.4f}")
                    else:
                        found = 0
                        not_found += 1
                    
                    # Early stopping
                    if found >= early_stop or not_found >= early_stop:
                        break
            
            # Check if target probability reached
            if best_validity >= target_proba:
                if verbose:
                    print(f"SG-CF: Success! Validity={best_validity:.4f}, Distance={best_distance:.4f}")
                break
            
            # Adjust lambda via bisection
            if found >= 5:
                lam_lb = max(lam, lam_lb)
                if lam_ub < 1e9:
                    lam = (lam_lb + lam_ub) / 2
                else:
                    lam *= 10
            elif found < 5:
                lam_ub = min(lam_ub, lam)
                if lam_lb > 0:
                    lam = (lam_lb + lam_ub) / 2
                else:
                    lam /= 10
            
            if verbose and l_step < max_lambda_steps - 1:
                print(f"  Lambda step {l_step}: found={found}, adjusting lambda to {lam:.6f}")
        
        # Check if we reached target probability
        if best_validity >= target_proba:
            break
        
        # Increase segment rate
        segment_rate += segment_rate_step
    
    if best_cf is None:
        if verbose:
            print(f"SG-CF: Failed to generate valid counterfactual. Best validity: {best_validity:.4f}")
        return None, None
    
    # Convert back to original format
    cf_result = best_cf.squeeze().cpu().numpy()
    if sample_orig.ndim == 1:
        cf_result = cf_result.squeeze()
    
    if verbose:
        print(f"SG-CF: Final - Validity={best_validity:.4f}, Distance={best_distance:.4f}, "
              f"Target prob={best_pred[target_class]:.4f}")
    
    return cf_result, best_pred


def sg_cf_explain(sample, dataset, model, target_class=None,
                 max_iter=1000, max_lambda_steps=10,
                 lambda_init=0.1, learning_rate=0.1,
                 device=None, verbose=False):
    """
    Generate SG-CF explanation with detailed information.
    
    Returns both counterfactual and explanation details including:
    - Shapelets used for guidance
    - Shapelet regions modified
    - Distance and validity metrics
    
    Args:
        sample: Time series to explain
        dataset: Training dataset
        model: Classifier model
        target_class: Target class
        max_iter: Maximum iterations
        max_lambda_steps: Maximum lambda steps
        lambda_init: Initial lambda
        learning_rate: Learning rate
        device: Device to use
        verbose: Print details
        
    Returns:
        Dictionary with counterfactual, prediction, and explanation details
    """
    # Extract shapelets
    X_train, y_train = [], []
    for i in range(min(len(dataset), 500)):
        item = dataset[i]
        ts, label = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, 0)
        ts_np = np.array(ts).reshape(1, -1) if np.array(ts).ndim == 1 else np.array(ts)
        X_train.append(ts_np)
        y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    shapelets = extract_shapelets_simple(X_train, y_train)
    
    # Get original prediction
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        original_pred = model(sample_tensor)
        original_class = torch.argmax(original_pred).item()
    
    if target_class is None:
        sorted_classes = torch.argsort(original_pred, descending=True).squeeze()
        target_class = sorted_classes[1].item()
    
    # Generate counterfactual
    cf, cf_pred = sg_cf(
        sample, dataset, model, target_class,
        shapelets=shapelets,
        max_iter=max_iter, max_lambda_steps=max_lambda_steps,
        lambda_init=lambda_init, learning_rate=learning_rate,
        device=device, verbose=False
    )
    
    # Prepare explanation
    explanation = {
        'counterfactual': cf,
        'prediction': cf_pred,
        'original_class': original_class,
        'target_class': target_class,
        'shapelets_available': {c: len(s) for c, s in shapelets.items()},
        'n_target_shapelets': len(shapelets.get(target_class, [])),
        'distance': np.linalg.norm(cf - sample) if cf is not None else None,
        'success': cf is not None
    }
    
    return explanation


# Alias for compatibility
sg_cf_generate = sg_cf
