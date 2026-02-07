"""
DisCOX: Discord-based Counterfactual Explanations for Time Series Classification

Implementation based on Bahri et al. (2024):
"Discord-based counterfactual explanations for time series classification"
Data Mining and Knowledge Discovery, Springer (2024)

DisCOX identifies and modifies discordant subsequences (the most anomalous patterns)
in time series to generate counterfactual explanations. The method:
1. Uses matrix profile to identify discord subsequences
2. Applies targeted modifications to discord regions
3. Leverages prototype-based replacement from target class
4. Optimizes for validity, proximity, and interpretability

Discords are subsequences that are most dissimilar to all other subsequences,
representing unusual or anomalous patterns that are often critical for classification.

Reference:
@article{bahri2024discox,
  title={Discord-based counterfactual explanations for time series classification},
  author={Bahri, Omar and Li, Peiyu and Boubrahimi, Soukaina Filali and Hamdi, Shah Muhammad},
  journal={Data Mining and Knowledge Discovery},
  year={2024},
  publisher={Springer},
  doi={10.1007/s10618-024-01028-9}
}

Links:
- Paper: https://link.springer.com/article/10.1007/s10618-024-01028-9
- DOI: 10.1007/s10618-024-01028-9
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Union
import warnings


def compute_matrix_profile(ts, window_size, exclude_radius=None):
    """
    Compute matrix profile for a time series.
    
    The matrix profile stores the minimum distance from each subsequence
    to its nearest non-trivial neighbor.
    
    Args:
        ts: Time series (1D array)
        window_size: Length of subsequences to compare
        exclude_radius: Radius to exclude trivial matches (default: window_size // 2)
        
    Returns:
        matrix_profile: Array of minimum distances for each subsequence
        profile_index: Index of nearest neighbor for each subsequence
    """
    if exclude_radius is None:
        exclude_radius = window_size // 2
    
    n = len(ts)
    n_subs = n - window_size + 1
    
    matrix_profile = np.full(n_subs, np.inf)
    profile_index = np.full(n_subs, -1, dtype=int)
    
    # Compute pairwise distances
    for i in range(n_subs):
        sub_i = ts[i:i + window_size]
        
        # Normalize subsequence
        if sub_i.std() > 0:
            sub_i_norm = (sub_i - sub_i.mean()) / sub_i.std()
        else:
            sub_i_norm = sub_i - sub_i.mean()
        
        for j in range(n_subs):
            # Skip trivial matches (overlapping or nearby)
            if abs(i - j) < exclude_radius:
                continue
            
            sub_j = ts[j:j + window_size]
            
            # Normalize subsequence
            if sub_j.std() > 0:
                sub_j_norm = (sub_j - sub_j.mean()) / sub_j.std()
            else:
                sub_j_norm = sub_j - sub_j.mean()
            
            # Compute Euclidean distance
            dist = np.sqrt(np.sum((sub_i_norm - sub_j_norm) ** 2))
            
            if dist < matrix_profile[i]:
                matrix_profile[i] = dist
                profile_index[i] = j
    
    return matrix_profile, profile_index


def find_top_k_discords(ts, window_size, k=3, exclude_radius=None):
    """
    Find the top-k most discordant subsequences in a time series.
    
    Discords are subsequences with the largest minimum distance to all
    other non-overlapping subsequences.
    
    Args:
        ts: Time series (1D array)
        window_size: Length of discord subsequences
        k: Number of top discords to return
        exclude_radius: Radius to exclude trivial matches
        
    Returns:
        List of (discord_index, discord_score) tuples
    """
    matrix_profile, _ = compute_matrix_profile(ts, window_size, exclude_radius)
    
    # Find top-k discord locations
    discord_indices = np.argsort(matrix_profile)[-k:][::-1]
    discord_scores = matrix_profile[discord_indices]
    
    return [(int(idx), float(score)) for idx, score in zip(discord_indices, discord_scores)]


def replace_discord_with_prototype(ts, discord_idx, window_size, prototype_ts, 
                                   prototype_class_samples, blend_factor=0.5):
    """
    Replace discord region with a corresponding region from a prototype.
    
    Args:
        ts: Original time series
        discord_idx: Start index of discord
        window_size: Length of discord
        prototype_ts: Prototype time series from target class
        prototype_class_samples: Additional samples from target class for averaging
        blend_factor: Blending factor (0=full prototype, 1=keep original)
        
    Returns:
        Modified time series with discord replaced
    """
    ts_modified = ts.copy()
    
    # Extract discord region
    discord_region = ts[discord_idx:discord_idx + window_size]
    
    # Get prototype region at same location
    if len(prototype_ts) >= discord_idx + window_size:
        proto_region = prototype_ts[discord_idx:discord_idx + window_size]
    else:
        # If prototype is shorter, use a centered region
        proto_start = max(0, len(prototype_ts) // 2 - window_size // 2)
        proto_region = prototype_ts[proto_start:proto_start + window_size]
    
    # Average with multiple prototype samples if available
    if len(prototype_class_samples) > 1:
        proto_regions = []
        for proto_sample in prototype_class_samples[:5]:  # Use up to 5 samples
            if len(proto_sample) >= discord_idx + window_size:
                proto_regions.append(proto_sample[discord_idx:discord_idx + window_size])
        
        if proto_regions:
            proto_region = np.mean(proto_regions, axis=0)
    
    # Scale prototype region to match local statistics
    if discord_region.std() > 0 and proto_region.std() > 0:
        scaled_proto = (proto_region - proto_region.mean()) / proto_region.std()
        scaled_proto = scaled_proto * discord_region.std() + discord_region.mean()
    else:
        scaled_proto = proto_region
    
    # Blend original and prototype
    replacement = blend_factor * discord_region + (1 - blend_factor) * scaled_proto
    
    ts_modified[discord_idx:discord_idx + window_size] = replacement
    
    return ts_modified


def discox_cf(sample, dataset, model, target_class=None,
              window_size=None, k_discords=3,
              modification_strategy='prototype',
              blend_factor=0.3, max_attempts=20,
              device=None, verbose=False):
    """
    Generate counterfactual using DisCOX (Discord-based Counterfactual) method.
    
    DisCOX identifies discordant (anomalous) subsequences and replaces them with
    patterns from the target class to generate interpretable counterfactuals.
    
    Args:
        sample: Time series instance to explain (numpy array)
        dataset: Training dataset for prototype extraction
        model: Trained classifier model
        target_class: Target class for counterfactual (optional)
        window_size: Length of discord subsequences (default: 10% of series length)
        k_discords: Number of top discords to consider
        modification_strategy: Strategy for modification ('prototype', 'amplify', 'attenuate', 'invert')
        blend_factor: Blending between original and prototype (0=full prototype, 1=keep original)
        max_attempts: Maximum modification attempts
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
    
    n_channels, n_timesteps = sample.shape if sample.ndim == 2 else (1, sample.shape[0])
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    # Determine window size
    if window_size is None:
        window_size = max(5, n_timesteps // 10)
    
    # Get original prediction
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
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
            print("DisCOX: Sample already in target class")
        return None, None
    
    if verbose:
        print(f"DisCOX: Original class={original_class} (p={original_proba[original_class]:.3f}), "
              f"Target class={target_class} (p={original_proba[target_class]:.3f})")
        print(f"DisCOX: Window size={window_size}, Strategy={modification_strategy}")
    
    # Extract training data from dataset
    X_train, y_train = [], []
    for i in range(min(len(dataset), 500)):
        item = dataset[i]
        ts, label = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, 0)
        ts_np = np.array(ts).reshape(1, -1) if np.array(ts).ndim == 1 else np.array(ts)
        X_train.append(ts_np)
        y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Get target class samples for prototype
    target_samples = X_train[y_train == target_class]
    if len(target_samples) == 0:
        if verbose:
            print("DisCOX: No target class samples found in dataset")
        return None, None
    
    # Find nearest neighbor from target class as prototype
    distances = [np.linalg.norm(sample - ts) for ts in target_samples]
    prototype_idx = np.argmin(distances)
    prototype = target_samples[prototype_idx]
    
    if verbose:
        print(f"DisCOX: Using {len(target_samples)} target class samples for prototype")
    
    # Process each channel independently
    best_cf = None
    best_pred = None
    best_validity = 0.0
    
    for channel in range(n_channels):
        ts_channel = sample[channel] if n_channels > 1 else sample.flatten()
        
        # Find top-k discords in this channel
        discords = find_top_k_discords(ts_channel, window_size, k=k_discords)
        
        if verbose:
            print(f"  Channel {channel}: Found {len(discords)} discords")
            for idx, (discord_idx, score) in enumerate(discords):
                print(f"    Discord {idx+1}: index={discord_idx}, score={score:.4f}")
        
        # Try modifying each discord
        for discord_rank, (discord_idx, discord_score) in enumerate(discords):
            if modification_strategy == 'prototype':
                # Replace with prototype region
                proto_channel = prototype[channel] if n_channels > 1 else prototype.flatten()
                target_samples_channel = [ts[channel] if n_channels > 1 else ts.flatten() 
                                         for ts in target_samples]
                
                # Try different blend factors
                blend_factors = [blend_factor, blend_factor * 0.5, blend_factor * 1.5, 0.0, 0.5]
                
                for bf in blend_factors:
                    cf = sample.copy()
                    cf_channel = ts_channel.copy()
                    
                    cf_channel = replace_discord_with_prototype(
                        cf_channel, discord_idx, window_size,
                        proto_channel, target_samples_channel, bf
                    )
                    
                    if n_channels > 1:
                        cf[channel] = cf_channel
                    else:
                        cf = cf_channel.reshape(1, -1)
                    
                    # Test counterfactual
                    cf_tensor = torch.tensor(cf, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        cf_pred = model(cf_tensor)
                        cf_proba = torch.softmax(cf_pred, dim=-1).squeeze().cpu().numpy()
                        cf_class = np.argmax(cf_proba)
                    
                    if cf_class == target_class:
                        if verbose:
                            print(f"  Success! Discord {discord_rank+1}, blend={bf:.2f}, "
                                  f"target_prob={cf_proba[target_class]:.4f}")
                        
                        if cf_proba[target_class] > best_validity:
                            best_validity = cf_proba[target_class]
                            best_cf = cf.copy()
                            best_pred = cf_proba
            
            elif modification_strategy == 'amplify':
                # Amplify discord region
                factors = [1.5, 2.0, 3.0, 0.5, 0.25]
                for factor in factors:
                    cf = sample.copy()
                    cf_channel = ts_channel.copy()
                    cf_channel[discord_idx:discord_idx + window_size] *= factor
                    
                    if n_channels > 1:
                        cf[channel] = cf_channel
                    else:
                        cf = cf_channel.reshape(1, -1)
                    
                    # Test
                    cf_tensor = torch.tensor(cf, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        cf_pred = model(cf_tensor)
                        cf_proba = torch.softmax(cf_pred, dim=-1).squeeze().cpu().numpy()
                        cf_class = np.argmax(cf_proba)
                    
                    if cf_class == target_class and cf_proba[target_class] > best_validity:
                        best_validity = cf_proba[target_class]
                        best_cf = cf.copy()
                        best_pred = cf_proba
            
            elif modification_strategy == 'invert':
                # Invert discord region
                cf = sample.copy()
                cf_channel = ts_channel.copy()
                discord_region = cf_channel[discord_idx:discord_idx + window_size]
                cf_channel[discord_idx:discord_idx + window_size] = -discord_region
                
                if n_channels > 1:
                    cf[channel] = cf_channel
                else:
                    cf = cf_channel.reshape(1, -1)
                
                # Test
                cf_tensor = torch.tensor(cf, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    cf_pred = model(cf_tensor)
                    cf_proba = torch.softmax(cf_pred, dim=-1).squeeze().cpu().numpy()
                    cf_class = np.argmax(cf_proba)
                
                if cf_class == target_class and cf_proba[target_class] > best_validity:
                    best_validity = cf_proba[target_class]
                    best_cf = cf.copy()
                    best_pred = cf_proba
            
            # Early exit if found good counterfactual
            if best_validity > 0.9:
                break
        
        # Early exit if found good counterfactual
        if best_validity > 0.9:
            break
    
    if best_cf is None:
        if verbose:
            print(f"DisCOX: Failed to generate valid counterfactual. Best validity: {best_validity:.4f}")
        return None, None
    
    # Convert back to original format
    cf_result = best_cf.squeeze() if sample_orig.ndim == 1 else best_cf
    
    if verbose:
        print(f"DisCOX: Final validity={best_validity:.4f}, "
              f"target_prob={best_pred[target_class]:.4f}")
    
    return cf_result, best_pred


def discox_explain(sample, dataset, model, target_class=None,
                  window_size=None, k_discords=3,
                  device=None, verbose=False):
    """
    Generate DisCOX explanation with detailed discord information.
    
    Returns both counterfactual and explanation details including:
    - Discord locations and scores
    - Modified regions
    - Prototype information
    
    Args:
        sample: Time series to explain
        dataset: Training dataset
        model: Classifier model
        target_class: Target class
        window_size: Discord window size
        k_discords: Number of discords to identify
        device: Device to use
        verbose: Print details
        
    Returns:
        Dictionary with counterfactual, prediction, and explanation details
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    sample_orig = sample.copy()
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    n_channels, n_timesteps = sample.shape
    
    if window_size is None:
        window_size = max(5, n_timesteps // 10)
    
    # Get original prediction
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        original_pred = model(sample_tensor)
        original_class = torch.argmax(original_pred).item()
    
    if target_class is None:
        sorted_classes = torch.argsort(original_pred, descending=True).squeeze()
        target_class = sorted_classes[1].item()
    
    # Find discords
    discord_info = {}
    for channel in range(n_channels):
        ts_channel = sample[channel] if n_channels > 1 else sample.flatten()
        discords = find_top_k_discords(ts_channel, window_size, k=k_discords)
        discord_info[f'channel_{channel}'] = discords
    
    # Generate counterfactual
    cf, cf_pred = discox_cf(
        sample_orig, dataset, model, target_class,
        window_size=window_size, k_discords=k_discords,
        device=device, verbose=False
    )
    
    # Prepare explanation
    explanation = {
        'counterfactual': cf,
        'prediction': cf_pred,
        'original_class': original_class,
        'target_class': target_class,
        'window_size': window_size,
        'discord_info': discord_info,
        'n_discords_found': sum(len(discords) for discords in discord_info.values()),
        'success': cf is not None
    }
    
    return explanation


# Aliases for compatibility
discox_generate = discox_cf
discox_generate_cf = discox_cf
