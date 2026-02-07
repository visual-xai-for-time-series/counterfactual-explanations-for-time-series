"""
MG-CF: Motif-Guided Counterfactual Explanations for Time Series Classification

Implementation based on Li et al. (2022):
"Motif-Guided Time Series Counterfactual Explanations"

MG-CF uses shapelet transform to extract discriminative motifs (subsequences) from
training data and generates counterfactuals by replacing the corresponding motif 
region in the original instance with the motif from the target class.

Algorithm:
1. Extract discriminative motifs using Shapelet Transform for each class
2. Sort motifs by information gain and select best motif per class
3. For a query instance, identify the motif region that matches the target class
4. Replace that region with the target class motif to create counterfactual

Reference:
@inproceedings{li2022motif,
  title={Motif-guided time series counterfactual explanations},
  author={Li, Peiyu and Boubrahimi, Souka{\"\i}na Filali and Hamdi, Shah Muhammad},
  booktitle={International Conference on Pattern Recognition},
  pages={203--215},
  year={2022},
  organization={Springer}
}

Links:
- Paper: https://arxiv.org/abs/2211.04411
- arXiv: 2211.04411v3
- GitHub: https://github.com/Luckilyeee/motif_guided_cf
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from scipy.stats import entropy
import warnings

# Try to import pyts for shapelet transform
try:
    from pyts.transformation import ShapeletTransform
    HAS_PYTS = True
except ImportError:
    HAS_PYTS = False
    warnings.warn("pyts not available. Install with 'pip install pyts' for full MG-CF functionality.")


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data"""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device"""
    return torch.from_numpy(data).float().to(device)


def extract_shapelets(X_train, y_train, n_shapelets=100, lengths_ratio=[0.3, 0.5, 0.7], 
                     random_state=42, verbose=False):
    """
    Extract discriminative shapelets (motifs) using Shapelet Transform.
    
    This implements Algorithm 1 (Motif Mining) from the MG-CF paper.
    
    Args:
        X_train: Training time series data (n_samples, length) or (n_samples, channels, length)
        y_train: Training labels (binary classification)
        n_shapelets: Number of shapelet candidates to extract
        lengths_ratio: Ratios of time series length for shapelet lengths (0.3, 0.5, 0.7 in paper)
        random_state: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary mapping class labels to best motif:
            {class: {'index': sample_idx, 'start': start_idx, 'end': end_idx, 'shapelet': array}}
    """
    if not HAS_PYTS:
        raise ImportError("MG-CF requires pyts library. Install with: pip install pyts")
    
    if verbose:
        print(f"MG-CF: Extracting shapelets from {len(X_train)} training samples...")
    
    # Ensure proper shape for pyts
    if X_train.ndim == 3:
        # (n_samples, channels, length) -> (n_samples, length) for univariate
        # pyts expects (n_samples, length)
        if X_train.shape[1] == 1:
            X_train = X_train.squeeze(1)
        else:
            # For multivariate, use first channel or flatten
            X_train = X_train[:, 0, :]
    
    # Determine shapelet lengths based on time series length
    ts_length = X_train.shape[1]
    window_sizes = [int(ts_length * ratio) for ratio in lengths_ratio]
    
    if verbose:
        print(f"MG-CF: Using window sizes: {window_sizes}")
    
    # Apply Shapelet Transform to extract and rank shapelets by information gain
    st = ShapeletTransform(
        n_shapelets=n_shapelets,
        window_sizes=window_sizes,
        random_state=random_state,
        sort=True  # Sort by information gain
    )
    
    st.fit_transform(X_train, y_train)
    
    # Extract shapelet indices (sample_idx, start_idx, end_idx)
    indices = st.indices_
    
    if verbose:
        print(f"MG-CF: Extracted {len(indices)} shapelets")
    
    # Organize shapelets by class label
    # For each class, keep the best shapelet (highest information gain)
    best_motifs = {}
    
    for i in range(len(indices)):
        sample_idx, start_idx, end_idx = indices[i]
        
        # Get the class label of this sample
        label = y_train[sample_idx]
        
        # Extract the actual shapelet
        shapelet = X_train[sample_idx, start_idx:end_idx]
        
        # If this is the first motif for this class, or it's ranked higher (lower index = better)
        if label not in best_motifs:
            best_motifs[label] = {
                'index': int(sample_idx),
                'start': int(start_idx),
                'end': int(end_idx),
                'shapelet': shapelet,
                'rank': i
            }
            
            if verbose:
                print(f"  Class {label}: Motif from sample {sample_idx}[{start_idx}:{end_idx}], "
                      f"length={end_idx-start_idx}, rank={i}")
    
    return best_motifs


def mg_cf_generate(sample, dataset, model, target_class=None,
                  motifs=None, n_shapelets=100, lengths_ratio=[0.3, 0.5, 0.7],
                  device=None, verbose=False):
    """
    Generate counterfactual using MG-CF (Motif-Guided Counterfactual) method.
    
    This implements Algorithm 2 (CF Generation) from the MG-CF paper.
    The algorithm:
    1. Extracts discriminative motifs from training data (if not provided)
    2. Identifies the target class motif
    3. Replaces the corresponding region in the query instance with target class motif
    4. Verifies if the counterfactual flips the prediction
    
    Args:
        sample: Time series instance to explain (length,) or (channels, length)
        dataset: Training dataset for motif extraction (list of tuples (x, y))
        model: Trained classifier model
        target_class: Target class for counterfactual (optional)
        motifs: Pre-extracted motifs (optional, will extract if None)
        n_shapelets: Number of shapelet candidates to extract
        lengths_ratio: Ratios for shapelet lengths
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
    
    # Get original prediction
    def predict_sample(ts):
        if ts.ndim == 1:
            ts = ts.reshape(1, -1)
        ts_tensor = torch.tensor(ts, dtype=torch.float32, device=device)
        if len(ts_tensor.shape) == 2:
            ts_tensor = ts_tensor.unsqueeze(0)
        with torch.no_grad():
            pred = model(ts_tensor)
            proba = torch.softmax(pred, dim=-1).squeeze().cpu().numpy()
        return np.argmax(proba), proba
    
    original_class, original_proba = predict_sample(sample)
    
    # Determine target class
    if target_class is None:
        sorted_indices = np.argsort(original_proba)[::-1]
        target_class = int(sorted_indices[1])
    
    if original_class == target_class:
        if verbose:
            print("MG-CF: Sample already in target class")
        return None, None
    
    if verbose:
        print(f"MG-CF: Original class={original_class} (p={original_proba[original_class]:.3f}), "
              f"Target class={target_class} (p={original_proba[target_class]:.3f})")
    
    # Extract training data from dataset
    X_train, y_train = [], []
    for i in range(min(len(dataset), 1000)):
        try:
            item = dataset[i]
            ts, label = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, 0)
            ts_np = np.array(ts)
            if ts_np.ndim == 1:
                ts_np = ts_np.reshape(1, -1)
            elif ts_np.ndim == 3:
                ts_np = ts_np.squeeze(0)
            
            # Flatten for motif extraction
            if ts_np.ndim == 2 and ts_np.shape[0] > 1:
                ts_np = ts_np[0, :]  # Use first channel
            elif ts_np.ndim == 2:
                ts_np = ts_np.flatten()
            
            X_train.append(ts_np)
            y_train.append(label)
        except Exception as e:
            if verbose:
                print(f"  Warning: Skipping sample {i}: {e}")
            continue
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    if verbose:
        print(f"MG-CF: Loaded {len(X_train)} training samples, shape={X_train.shape}")
    
    # Extract motifs if not provided
    if motifs is None:
        if verbose:
            print("MG-CF: Extracting motifs from training data...")
        
        motifs = extract_shapelets(
            X_train, y_train,
            n_shapelets=n_shapelets,
            lengths_ratio=lengths_ratio,
            verbose=verbose
        )
    
    # Check if target class motif exists
    if target_class not in motifs:
        if verbose:
            print(f"MG-CF: No motif found for target class {target_class}")
        return None, None
    
    # Get target class motif information
    motif_info = motifs[target_class]
    motif_sample_idx = motif_info['index']
    motif_start = motif_info['start']
    motif_end = motif_info['end']
    
    if verbose:
        print(f"MG-CF: Using target class motif from sample {motif_sample_idx}[{motif_start}:{motif_end}]")
    
    # Create counterfactual by replacing motif region
    cf_sample = sample.copy()
    
    # Get the motif from training sample
    target_motif = X_train[motif_sample_idx, motif_start:motif_end]
    
    # Replace the corresponding region in the query sample
    if cf_sample.shape[-1] >= motif_end:
        # Ensure we're replacing the right region
        if cf_sample.ndim == 1:
            cf_sample[motif_start:motif_end] = target_motif
        else:
            # For multi-channel, replace on first channel or all channels
            if cf_sample.shape[0] == 1:
                cf_sample[0, motif_start:motif_end] = target_motif
            else:
                cf_sample[:, motif_start:motif_end] = target_motif
    else:
        if verbose:
            print(f"MG-CF: Warning - Motif region exceeds sample length")
        return None, None
    
    # Verify counterfactual
    cf_class, cf_proba = predict_sample(cf_sample)
    
    if verbose:
        print(f"MG-CF: Counterfactual class={cf_class} (p={cf_proba[cf_class]:.3f})")
        if cf_class == target_class:
            print("MG-CF: Successfully generated valid counterfactual!")
        else:
            print("MG-CF: Warning - Counterfactual did not flip to target class")
    
    # Return in original shape
    if sample_orig.ndim == 1:
        cf_sample = cf_sample.squeeze()
    
    return cf_sample, cf_proba


def mg_cf_explain(sample, dataset, model, target_class=None,
                 motifs=None, device=None, verbose=False, **kwargs):
    """
    Generate MG-CF explanation with detailed information.
    
    Args:
        sample: Time series to explain
        dataset: Training dataset
        model: Classifier model
        target_class: Target class
        motifs: Pre-extracted motifs
        device: Device to use
        verbose: Print details
        **kwargs: Additional arguments for mg_cf_generate
        
    Returns:
        Dictionary with counterfactual, prediction, and explanation details
    """
    cf, cf_pred = mg_cf_generate(
        sample, dataset, model, target_class,
        motifs=motifs, device=device, verbose=verbose, **kwargs
    )
    
    # Get original prediction
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)
    elif len(sample_tensor.shape) == 2:
        sample_tensor = sample_tensor.unsqueeze(0)
    
    with torch.no_grad():
        original_pred = model(sample_tensor)
        original_proba = torch.softmax(original_pred, dim=-1).squeeze().cpu().numpy()
        original_class = np.argmax(original_proba)
    
    if target_class is None:
        sorted_classes = np.argsort(original_proba)[::-1]
        target_class = int(sorted_classes[1])
    
    explanation = {
        'counterfactual': cf,
        'prediction': cf_pred,
        'original_class': original_class,
        'target_class': target_class,
        'success': cf is not None and np.argmax(cf_pred) == target_class if cf is not None else False,
        'distance': np.linalg.norm(cf - sample) if cf is not None else None,
        'sparsity': 1.0 - (np.sum(np.abs(cf - sample) > 1e-6) / len(sample)) if cf is not None else None
    }
    
    return explanation


def mg_cf_batch(samples, dataset, model, target_class=None,
               n_shapelets=100, lengths_ratio=[0.3, 0.5, 0.7],
               device=None, verbose=False):
    """
    Generate counterfactuals for a batch of samples using MG-CF.
    
    Extracts motifs once and reuses for all samples for efficiency.
    
    Args:
        samples: List or array of time series samples
        dataset: Training dataset
        model: Classifier model
        target_class: Target class (optional, determined per sample if None)
        n_shapelets: Number of shapelet candidates
        lengths_ratio: Ratios for shapelet lengths
        device: Device to use
        verbose: Print progress
        
    Returns:
        List of (counterfactual, prediction) tuples
    """
    if verbose:
        print("MG-CF Batch: Extracting motifs once for all samples...")
    
    # Extract motifs once
    X_train, y_train = [], []
    for item in dataset:
        ts, label = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, 0)
        ts_np = np.array(ts)
        if ts_np.ndim >= 2:
            ts_np = ts_np.flatten() if ts_np.shape[0] == 1 else ts_np[0, :]
        X_train.append(ts_np)
        y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    motifs = extract_shapelets(
        X_train, y_train,
        n_shapelets=n_shapelets,
        lengths_ratio=lengths_ratio,
        verbose=verbose
    )
    
    # Generate counterfactuals
    results = []
    for i, sample in enumerate(samples):
        if verbose and i % 10 == 0:
            print(f"MG-CF Batch: Processing sample {i}/{len(samples)}")
        
        cf, pred = mg_cf_generate(
            sample, dataset, model, target_class,
            motifs=motifs, device=device, verbose=False
        )
        
        results.append((cf, pred))
    
    if verbose:
        valid = sum(1 for cf, _ in results if cf is not None)
        print(f"MG-CF Batch: Generated {valid}/{len(samples)} valid counterfactuals")
    
    return results


# Aliases for compatibility
mgcf_generate = mg_cf_generate
mgcf_explain = mg_cf_explain
