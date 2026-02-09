"""
DisCOX: Discord-based Counterfactual Explanations for Time Series Classification

Implementation based on Bahri et al. (2024):
"Discord-based counterfactual explanations for time series classification"
Data Mining and Knowledge Discovery, Springer (2024)

DisCOX iteratively discovers discord subsequences using matrix profile and maps them
to corresponding regions from target class prototypes. The method:
1. Uses STUMPY for efficient matrix profile computation and discord discovery
2. Iteratively maps discordant regions to target class patterns
3. Handles overlapping mappings with complex precedence rules
4. Fills short unmapped intervals by extending adjacent mappings
5. Blends original and mapped time series with variable weights

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

Dependencies:
- stumpy: For efficient matrix profile computation and discord discovery
  Install with: pip install stumpy
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Union
import warnings

# Try to import stumpy for matrix profile computation
try:
    import stumpy
    HAS_STUMPY = True
except ImportError:
    HAS_STUMPY = False
    warnings.warn("STUMPY not available. Install with 'pip install stumpy' for full DisCOX functionality.")


def ind_list_to_intervals(lst):
    """
    Split list of indices into separate contiguous intervals.
    
    Args:
        lst: List of indices
        
    Returns:
        intervals: List of interval index lists
        intervals_inds: List of positions in original list
    """
    if len(lst) == 0:
        return [], []
    
    intervals = []
    intervals_inds = []
    curr_interval = []
    curr_interval_inds = []
    
    prev_i = lst[0]
    curr_interval.append(prev_i)
    curr_interval_inds.append(0)
    
    for ind, i in enumerate(lst[1:]):
        if i == prev_i + 1:
            curr_interval.append(i)
            curr_interval_inds.append(ind + 1)
        else:
            intervals.append(curr_interval)
            intervals_inds.append(curr_interval_inds)
            curr_interval = [i]
            curr_interval_inds = [ind + 1]
        prev_i = i
    
    intervals.append(curr_interval)
    intervals_inds.append(curr_interval_inds)
    
    return intervals, intervals_inds


def fill_short_intervals(mapping, target_ts, X_target_c_sep, thres=3):
    """
    Fill short non-mapped subsequences by extending their longest adjacent mapped subsequence.
    
    Args:
        mapping: Array mapping each timestep to target class prototype index
        target_ts: Current target time series being constructed
        X_target_c_sep: Concatenated target class samples (separated by NaN)
        thres: Threshold for short intervals
        
    Returns:
        Updated mapping and target_ts
    """
    # Non-mapped indices
    non_map = np.where(np.isnan(mapping))[0]
    if len(non_map) == 0:
        return mapping, target_ts
    
    non_map_intervals, _ = ind_list_to_intervals(list(non_map))
    
    # Process each non-mapped interval
    for interval in non_map_intervals:
        mapped_intervals, mapped_intervals_inds = ind_list_to_intervals(
            list(mapping[~np.isnan(mapping)].astype(int))
        )
        
        if len(interval) < thres:
            # Find position of current non-mapped subsequence
            for mii_i, mii in enumerate(mapped_intervals_inds):
                if len(mii) == 0:
                    continue
                    
                # Check if non-mapped interval is adjacent
                if interval[0] == mii[-1] + 1:
                    if mii_i == 0:
                        # At beginning - fill from next mapped interval
                        fill_from = mii_i + 1 if mii_i + 1 < len(mapped_intervals_inds) else 0
                        plus = 1
                    elif mii_i == len(mapped_intervals_inds) - 1:
                        # At end - fill from previous
                        fill_from = mii_i
                        plus = -1
                    else:
                        # In middle - fill from longer adjacent
                        if len(mapped_intervals_inds[mii_i]) > len(mapped_intervals_inds[mii_i + 1]):
                            fill_from = mii_i
                            plus = -1
                        else:
                            fill_from = mii_i + 1
                            plus = 1
                    
                    # Fill the interval
                    try:
                        if fill_from < len(mapped_intervals) and len(mapped_intervals[fill_from]) > 0:
                            if plus == 1:
                                temp_mapped = np.array(mapped_intervals[fill_from])[-len(interval):]
                            else:
                                temp_mapped = np.array(mapped_intervals[fill_from])[:len(interval)]
                            
                            # Check bounds
                            valid_inds = (temp_mapped >= 0) & (temp_mapped < len(X_target_c_sep))
                            if np.any(valid_inds):
                                temp_target_ts = X_target_c_sep[temp_mapped[valid_inds].astype(int)]
                                mapping[interval][:len(temp_target_ts)] = temp_mapped[valid_inds]
                                target_ts[interval][:len(temp_target_ts)] = temp_target_ts
                    except:
                        pass
                    break
    
    return mapping, target_ts


def interpolate_short(X_cf, mapping, thres=3):
    """
    Interpolate short unmapped intervals in the counterfactual.
    
    Args:
        X_cf: Counterfactual time series
        mapping: Mapping array
        thres: Threshold for short intervals
        
    Returns:
        Interpolated counterfactual
    """
    import pandas as pd
    
    # Mark short unmapped intervals with NaN
    mapping_copy = mapping.copy()
    mapping_copy[mapping_copy == np.inf] = np.nan
    non_map = np.where(np.isnan(mapping_copy))[0]
    
    if len(non_map) == 0:
        return X_cf
    
    non_map_intervals, _ = ind_list_to_intervals(list(non_map))
    
    for interval in non_map_intervals:
        if len(interval) < thres:
            X_cf[interval] = np.nan
    
    # Interpolate using pandas
    X_cf_df = pd.DataFrame(X_cf)
    mask = X_cf_df.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    
    for i in range(X_cf_df.shape[1]):
        mask[i] = (grp.groupby(i)['ones'].transform('count') < thres) | X_cf_df[i].notnull()
    
    X_cf_interpolated = X_cf_df.interpolate().bfill()[mask]
    
    return X_cf_interpolated.to_numpy().reshape(-1)


def discox_cf(sample, dataset, model, target_class=None,
              window_size=None, max_iterations=100,
              weight_steps=None, device=None, verbose=False):
    """
    Generate counterfactual using DisCOX (Discord-based Counterfactual) method.
    
    This implements the full DisCOX algorithm with iterative discord discovery,
    mapping to target class prototypes, and weighted blending.
    
    Args:
        sample: Time series instance to explain (numpy array)
        dataset: Training dataset for prototype extraction
        model: Trained classifier model
        target_class: Target class for counterfactual (optional)
        window_size: Discord window size (default: 10% of series length)
        max_iterations: Maximum discord discovery iterations
        weight_steps: Array of blend weights to try (default: 0.01 to 1.0)
        device: Device to run on
        verbose: Print progress information
        
    Returns:
        Tuple of (counterfactual, prediction) or (None, None) if failed
    """
    if not HAS_STUMPY:
        raise ImportError("DisCOX requires STUMPY library. Install with: pip install stumpy")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample - ensure it's 1D for stumpy
    sample_orig = sample.copy()
    sample = np.array(sample).flatten()  # Force 1D
    
    n_timesteps = len(sample)
    
    # Determine window size
    if window_size is None:
        window_size = max(5, int(0.1 * n_timesteps))
    
    # Weight steps for blending
    if weight_steps is None:
        weight_steps = np.concatenate([
            np.arange(0.01, 0.5, 0.01),
            np.arange(0.5, 1.01, 0.02)
        ])
    
    # Get original prediction
    def predict_sample(ts):
        ts_tensor = torch.tensor(ts, dtype=torch.float32, device=device)
        # Ensure correct shape: (batch=1, channels=1, timesteps)
        if len(ts_tensor.shape) == 1:
            ts_tensor = ts_tensor.reshape(1, 1, -1)
        elif len(ts_tensor.shape) == 2:
            # Could be (1, timesteps) or (timesteps, 1)
            if ts_tensor.shape[0] == 1:
                ts_tensor = ts_tensor.reshape(1, 1, -1)
            else:
                ts_tensor = ts_tensor.reshape(1, 1, -1)
        with torch.no_grad():
            pred = model(ts_tensor)
            # Model already applies softmax, so don't apply it again
            proba = pred.squeeze().cpu().numpy()
        return np.argmax(proba), proba
    
    original_class, original_proba = predict_sample(sample)
    
    # Determine target class
    if target_class is None:
        sorted_indices = np.argsort(original_proba)[::-1]
        target_class = int(sorted_indices[1])
    
    if original_class == target_class:
        if verbose:
            print("DisCOX: Sample already in target class")
        return None, None
    
    if verbose:
        print(f"DisCOX: Original class={original_class} (p={original_proba[original_class]:.3f}), "
              f"Target class={target_class} (p={original_proba[target_class]:.3f})")
        print(f"DisCOX: Window size={window_size}")
    
    # Extract training data from dataset
    X_train, y_train = [], []
    for i in range(min(len(dataset), 1000)):
        item = dataset[i]
        ts, label = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, 0)
        ts_np = np.array(ts).flatten()
        X_train.append(ts_np)
        
        # Convert label to scalar
        if hasattr(label, 'shape') and len(label.shape) > 0:
            label_scalar = int(np.argmax(label))
        else:
            label_scalar = int(label)
        y_train.append(label_scalar)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Get target class samples and create concatenated prototype
    target_samples = X_train[y_train == target_class]
    if len(target_samples) == 0:
        if verbose:
            print("DisCOX: No target class samples found")
        return None, None
    
    # Create concatenated target class prototype (separated by NaN)
    X_target_c_sep = []
    for ts in target_samples:
        X_target_c_sep.append(list(ts))
        X_target_c_sep.append([np.nan])
    X_target_c_sep = np.array([item for sublist in X_target_c_sep for item in sublist][:-1])
    
    if verbose:
        print(f"DisCOX: Using {len(target_samples)} target class samples, "
              f"concatenated length={len(X_target_c_sep)}")
    
    # Compute matrix profile using STUMPY
    mp = stumpy.stump(sample, window_size, X_target_c_sep, 
                     ignore_trivial=False, normalize=False)
    mp_values = mp[:, 0].astype(float)
    mp_indices = mp[:, 1].astype(int)
    
    # Initialize mapping and target time series
    mapping = np.full(n_timesteps, fill_value=np.nan)
    when_mapped = np.full(n_timesteps, fill_value=np.nan)
    target_ts = sample.copy()
    when = 1
    
    # Iterative discord discovery and mapping
    n_non_mapped = np.inf
    max_len_non_mapped = np.inf
    iteration = 0
    cf_found = False
    
    while n_non_mapped and max_len_non_mapped >= 3 and iteration < max_iterations and not cf_found:
        # Find discord (maximum matrix profile value)
        unmapped_indices = np.where(np.isnan(mapping))[0]
        if len(unmapped_indices) == 0:
            break
        
        valid_mp = mp_values.copy()
        mapped_indices = np.where(~np.isnan(mapping))[0]
        valid_mp[mapped_indices] = -np.inf
        
        if np.all(valid_mp == -np.inf):
            break
        
        discord_idx = np.argmax(valid_mp)
        discord_start = discord_idx
        discord_end = min(discord_idx + window_size, n_timesteps)
        
        # Find corresponding region in target prototype
        target_idx = mp_indices[discord_idx]
        # Bounds check: ensure target_idx is valid
        # Also ensure window doesn't exceed array bounds
        if target_idx < 0 or target_idx >= len(X_target_c_sep):
            mp_values[discord_idx] = -np.inf
            continue
        
        target_start = target_idx
        # Ensure target_end doesn't exceed array length
        target_end = min(target_idx + window_size, len(X_target_c_sep))
        
        # Map the discord region
        already_mapped = ~np.isnan(mapping[discord_start:discord_end])
        
        if not np.all(already_mapped):
            # Map unmapped indices
            map_len = min(discord_end - discord_start, target_end - target_start)
            
            # Ensure we don't exceed bounds
            if target_start + map_len > len(X_target_c_sep):
                map_len = len(X_target_c_sep) - target_start
            if discord_start + map_len > len(target_ts):
                map_len = len(target_ts) - discord_start
            
            if map_len <= 0:
                mp_values[discord_idx] = -np.inf
                continue
            
            # Ensure target_start + map_len is within bounds
            actual_map_len = min(map_len, len(X_target_c_sep) - target_start)
            if actual_map_len <= 0:
                mp_values[discord_idx] = -np.inf
                continue
            
            # Extract target subsequence with bounds checking
            target_subseq = X_target_c_sep[target_start:target_start + actual_map_len]
            
            # Filter out NaN values from target
            valid_target = ~np.isnan(target_subseq)
            if np.any(valid_target):
                # Ensure destination indices are within bounds
                dest_len = min(actual_map_len, len(target_ts) - discord_start)
                if dest_len <= 0:
                    mp_values[discord_idx] = -np.inf
                    continue
                
                # Adjust valid_target to match destination length
                valid_target_adjusted = valid_target[:dest_len]
                target_subseq_adjusted = target_subseq[:dest_len]
                
                if np.any(valid_target_adjusted):
                    target_ts[discord_start:discord_start + dest_len][valid_target_adjusted] = \
                        target_subseq_adjusted[valid_target_adjusted]
                    mapping[discord_start:discord_start + dest_len][valid_target_adjusted] = \
                        np.arange(target_start, target_start + dest_len)[valid_target_adjusted]
                    when_mapped[discord_start:discord_start + dest_len][valid_target_adjusted] = when
                    when += 1
        
        # Mark discord as processed
        mp_values[discord_idx] = -np.inf
        
        # Try to fill short intervals
        try:
            mapping, target_ts = fill_short_intervals(mapping, target_ts, X_target_c_sep, thres=3)
        except:
            pass
        
        # Try different blend weights
        for w in weight_steps:
            X_cf = (1 - w) * sample + w * target_ts
            X_cf = interpolate_short(X_cf, mapping, thres=3)
            
            cf_class, cf_proba = predict_sample(X_cf)
            
            if cf_class == target_class:
                cf_found = True
                if verbose:
                    print(f"DisCOX: CF found at iteration {iteration}, weight={w:.3f}, "
                          f"target_prob={cf_proba[target_class]:.4f}")
                return X_cf, cf_proba
        
        # Update non-mapped statistics
        nan_map = np.where(np.isnan(mapping))[0]
        if len(nan_map) > 0:
            nan_map_intervals, _ = ind_list_to_intervals(list(nan_map))
            max_len_non_mapped = max([len(interval) for interval in nan_map_intervals])
            n_non_mapped = len(nan_map_intervals)
        else:
            n_non_mapped = 0
        
        iteration += 1
        
        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}: {len(nan_map)} unmapped, max_len={max_len_non_mapped}")
    
    # Final attempt with full mapping
    if not cf_found:
        try:
            mapping, target_ts = fill_short_intervals(mapping, target_ts, X_target_c_sep, thres=3)
        except:
            pass
        
        for w in weight_steps:
            X_cf = (1 - w) * sample + w * target_ts
            X_cf = interpolate_short(X_cf, mapping, thres=3)
            
            cf_class, cf_proba = predict_sample(X_cf)
            
            if cf_class == target_class:
                if verbose:
                    print(f"DisCOX: CF found (final), weight={w:.3f}, "
                          f"target_prob={cf_proba[target_class]:.4f}")
                return X_cf, cf_proba
    
    if verbose:
        print("DisCOX: Failed to generate valid counterfactual")
    
    return None, None


def discox_explain(sample, dataset, model, target_class=None,
                  window_size=None, device=None, verbose=False):
    """
    Generate DisCOX explanation with detailed discord and mapping information.
    
    Args:
        sample: Time series to explain
        dataset: Training dataset
        model: Classifier model
        target_class: Target class
        window_size: Discord window size
        device: Device to use
        verbose: Print details
        
    Returns:
        Dictionary with counterfactual, prediction, and explanation details
    """
    cf, cf_pred = discox_cf(
        sample, dataset, model, target_class,
        window_size=window_size, device=device, verbose=verbose
    )
    
    # Get original prediction
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    # Ensure correct shape: (batch=1, channels=1, timesteps)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.reshape(1, 1, -1)
    elif len(sample_tensor.shape) == 2:
        if sample_tensor.shape[0] == 1 or sample_tensor.shape[1] == 1:
            sample_tensor = sample_tensor.reshape(1, 1, -1)
    
    with torch.no_grad():
        original_pred = model(sample_tensor)
        original_class = torch.argmax(original_pred).item()
    
    if target_class is None:
        sorted_classes = torch.argsort(original_pred, descending=True).squeeze()
        target_class = sorted_classes[1].item()
    
    explanation = {
        'counterfactual': cf,
        'prediction': cf_pred,
        'original_class': original_class,
        'target_class': target_class,
        'window_size': window_size if window_size else int(0.1 * len(sample)),
        'success': cf is not None,
        'distance': np.linalg.norm(cf - sample) if cf is not None else None
    }
    
    return explanation


# Aliases for compatibility
discox_generate = discox_cf
discox_generate_cf = discox_cf
