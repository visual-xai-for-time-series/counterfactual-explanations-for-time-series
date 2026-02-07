import numpy as np
import torch
from scipy.stats import pearsonr


def detach_to_numpy(data):
    # move pytorch data to cpu and detach it to numpy data
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    # convert numpy array to pytorch and move it to the device
    return torch.from_numpy(data).float().to(device)


####
# MG-CF: Motif-Guided Counterfactual Explanations for Time Series
#
# Paper: https://arxiv.org/abs/2211.04411
#
# This motif-based counterfactual explanation method uses shapelet-like motifs
# and matrix profile analysis to identify important subsequences and guide the
# generation of counterfactual explanations for time series classification.
#
# Related work on Matrix Profile:
# Yeh, C. M., et al. (2016). "Matrix profile I: All pairs similarity joins for
# time series." 2016 IEEE 16th International Conference on Data Mining (ICDM)
####


def _generate_candidates(time_series, lengths):
    """Generate candidate motif subsequences of specified lengths.
    
    Args:
        time_series: Time series of shape (length,) or (channels, length)
        lengths: List of subsequence lengths to extract
        
    Returns:
        List of tuples (start_idx, end_idx, subsequence)
    """
    candidates = []
    
    # Handle different input shapes
    if len(time_series.shape) == 1:
        ts = time_series
        for l in lengths:
            for start in range(len(ts) - l + 1):
                end = start + l
                subseq = ts[start:end]
                candidates.append((start, end, subseq))
    else:
        # For multivariate, extract candidates from each channel
        for l in lengths:
            for start in range(time_series.shape[-1] - l + 1):
                end = start + l
                subseq = time_series[:, start:end] if len(time_series.shape) == 2 else time_series[start:end]
                candidates.append((start, end, subseq))
    
    return candidates


def _subsequence_distance(subseq, time_series):
    """Compute minimum distance between a subsequence and all possible positions in time series.
    
    Args:
        subseq: Subsequence array
        time_series: Full time series array
        
    Returns:
        Minimum Euclidean distance
    """
    l = len(subseq) if subseq.ndim == 1 else subseq.shape[-1]
    ts_len = len(time_series) if time_series.ndim == 1 else time_series.shape[-1]
    
    min_dist = float('inf')
    
    for start in range(ts_len - l + 1):
        if time_series.ndim == 1:
            candidate = time_series[start:start + l]
        else:
            candidate = time_series[:, start:start + l]
        
        dist = np.sqrt(np.sum((subseq - candidate) ** 2))
        min_dist = min(min_dist, dist)
    
    return min_dist


def _find_distances(subseq, dataset):
    """Find minimum distances between a subsequence and all time series in dataset.
    
    Args:
        subseq: Candidate motif subsequence
        dataset: List or array of time series
        
    Returns:
        Array of distances
    """
    distances = []
    
    for i in range(len(dataset)):
        # Extract time series (handle dataset format with (x, y) tuples)
        if isinstance(dataset[i], tuple):
            ts = dataset[i][0]
        else:
            ts = dataset[i]
        
        # Convert to numpy if needed
        if isinstance(ts, torch.Tensor):
            ts = detach_to_numpy(ts)
        
        dist = _subsequence_distance(subseq, ts)
        distances.append(dist)
    
    return np.array(distances)


def _assess_candidate_quality(distances, labels):
    """Assess the discriminatory power of a motif using information gain.
    
    Args:
        distances: Array of distances to each time series
        labels: Array of class labels
        
    Returns:
        Quality score (information gain)
    """
    # Sort distances
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Calculate information gain for each split point
    n = len(labels)
    n_pos = np.sum(labels)
    n_neg = n - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    # Initial entropy
    p_pos = n_pos / n
    p_neg = n_neg / n
    initial_entropy = -p_pos * np.log2(p_pos + 1e-10) - p_neg * np.log2(p_neg + 1e-10)
    
    best_gain = 0.0
    
    # Try each split point
    for i in range(1, n):
        # Left split
        left_labels = sorted_labels[:i]
        left_pos = np.sum(left_labels)
        left_neg = len(left_labels) - left_pos
        
        # Right split
        right_labels = sorted_labels[i:]
        right_pos = np.sum(right_labels)
        right_neg = len(right_labels) - right_pos
        
        # Calculate entropy for left and right
        left_entropy = 0.0
        if left_pos > 0 and left_neg > 0:
            p_left_pos = left_pos / len(left_labels)
            p_left_neg = left_neg / len(left_labels)
            left_entropy = -p_left_pos * np.log2(p_left_pos + 1e-10) - p_left_neg * np.log2(p_left_neg + 1e-10)
        
        right_entropy = 0.0
        if right_pos > 0 and right_neg > 0:
            p_right_pos = right_pos / len(right_labels)
            p_right_neg = right_neg / len(right_labels)
            right_entropy = -p_right_pos * np.log2(p_right_pos + 1e-10) - p_right_neg * np.log2(p_right_neg + 1e-10)
        
        # Weighted entropy
        weighted_entropy = (len(left_labels) / n) * left_entropy + (len(right_labels) / n) * right_entropy
        
        # Information gain
        gain = initial_entropy - weighted_entropy
        best_gain = max(best_gain, gain)
    
    return best_gain


def mine_motifs(dataset, lengths_ratio=[0.3, 0.5, 0.7], verbose=False):
    """Mine discriminative motifs from a time series dataset using shapelet transform.
    
    This implements Algorithm 1 from the MG-CF paper.
    
    Args:
        dataset: List/array of (x, y) tuples where x is time series and y is binary label
        lengths_ratio: Ratios of time series length to use for motif lengths
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping class labels to best motif info:
            {class: (ts_idx, start_idx, end_idx, motif, quality)}
    """
    if verbose:
        print("MG-CF: Mining discriminative motifs...")
    
    # Extract data and labels
    X = []
    y = []
    for item in dataset:
        if isinstance(item, tuple):
            X.append(item[0])
            y.append(item[1])
        else:
            raise ValueError("Dataset must contain (x, y) tuples")
    
    y = np.array(y)
    
    # Determine motif lengths based on time series length
    sample_ts = X[0]
    if isinstance(sample_ts, torch.Tensor):
        sample_ts = detach_to_numpy(sample_ts)
    
    ts_length = len(sample_ts) if sample_ts.ndim == 1 else sample_ts.shape[-1]
    lengths = [int(ts_length * ratio) for ratio in lengths_ratio]
    
    if verbose:
        print(f"MG-CF: Extracting motifs of lengths {lengths} from {len(X)} time series")
    
    # Store best motifs for each class
    best_motifs = {}
    
    # Process each time series
    for ts_idx in range(len(X)):
        ts = X[ts_idx]
        if isinstance(ts, torch.Tensor):
            ts = detach_to_numpy(ts)
        
        # Generate candidate subsequences
        candidates = _generate_candidates(ts, lengths)
        
        if verbose and ts_idx % 20 == 0:
            print(f"MG-CF: Processing time series {ts_idx}/{len(X)}, {len(candidates)} candidates")
        
        # Evaluate each candidate
        for start_idx, end_idx, subseq in candidates:
            # Find distances to all time series
            distances = _find_distances(subseq, dataset)
            
            # Assess quality using information gain
            quality = _assess_candidate_quality(distances, y)
            
            # Get class of this time series
            ts_class = y[ts_idx]
            # Convert to scalar if it's an array
            if hasattr(ts_class, 'shape') and len(ts_class.shape) > 0:
                ts_class = int(np.argmax(ts_class)) if ts_class.shape[0] > 1 else int(ts_class[0])
            else:
                ts_class = int(ts_class)
            
            # Update best motif for this class if better quality found
            if ts_class not in best_motifs or quality > best_motifs[ts_class][4]:
                best_motifs[ts_class] = (ts_idx, start_idx, end_idx, subseq, quality)
    
    if verbose:
        for cls, (ts_idx, start, end, motif, quality) in best_motifs.items():
            print(f"MG-CF: Class {cls} - Best motif from TS {ts_idx} [{start}:{end}], quality={quality:.4f}")
    
    return best_motifs


def mg_cf_generate(sample, dataset, model, motifs=None, target=None, 
                   lengths_ratio=[0.3, 0.5, 0.7], verbose=False):
    """Generate counterfactual explanation using Motif-Guided approach.
    
    This implements Algorithm 2 from the MG-CF paper. It generates a counterfactual
    by replacing the discriminative motif region of the input sample with the 
    corresponding motif from the target class.
    
    Args:
        sample: Input time series to explain (numpy array or torch tensor)
        dataset: Training dataset used for motif mining (list of (x, y) tuples)
        model: Trained classifier model
        motifs: Pre-mined motifs (if None, will mine from dataset)
        target: Target class for counterfactual (if None, will use opposite of prediction)
        lengths_ratio: Ratios for motif lengths if mining is needed
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (counterfactual_sample, prediction_scores) or (None, None) if failed
    """
    device = next(model.parameters()).device
    
    def model_predict(data):
        # Ensure proper input format for model
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        else:
            data_tensor = data
            
        # Handle different input shapes for model
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.reshape(1, 1, -1)
        elif len(data_tensor.shape) == 2:
            if data_tensor.shape[0] > data_tensor.shape[1]:
                data_tensor = data_tensor.T
            data_tensor = data_tensor.unsqueeze(0)
            
        return detach_to_numpy(model(data_tensor))
    
    # Convert sample to numpy
    if isinstance(sample, torch.Tensor):
        sample_np = detach_to_numpy(sample)
    else:
        sample_np = np.array(sample)
    
    original_shape = sample_np.shape
    
    # Get original prediction
    y_orig = model_predict(sample_np)[0]
    orig_class = int(np.argmax(y_orig))
    
    # Determine target class
    if target is None:
        # Find the class with second highest probability
        sorted_indices = np.argsort(y_orig)[::-1]
        target = int(sorted_indices[1])
    
    if verbose:
        print(f"MG-CF Generate: Original class {orig_class}, Target class {target}")
    
    # Mine motifs if not provided
    if motifs is None:
        if verbose:
            print("MG-CF Generate: Mining motifs from dataset...")
        motifs = mine_motifs(dataset, lengths_ratio=lengths_ratio, verbose=verbose)
    
    # Check if we have motifs for the target class
    if target not in motifs:
        if verbose:
            print(f"MG-CF Generate: No motif found for target class {target}")
        return None, None
    
    # Extract motif information for target class
    ts_idx, start_idx, end_idx, motif, quality = motifs[target]
    
    # Get the source time series from dataset
    if isinstance(dataset[ts_idx], tuple):
        source_ts = dataset[ts_idx][0]
    else:
        source_ts = dataset[ts_idx]
    
    if isinstance(source_ts, torch.Tensor):
        source_ts = detach_to_numpy(source_ts)
    
    # Create counterfactual by replacing motif region
    cf_sample = sample_np.copy()
    
    if verbose:
        print(f"MG-CF Generate: Replacing region [{start_idx}:{end_idx}] with target class motif")
    
    # Replace the motif region
    if cf_sample.ndim == 1:
        # Univariate case
        motif_segment = source_ts[start_idx:end_idx] if source_ts.ndim == 1 else source_ts[:, start_idx:end_idx].flatten()
        cf_sample[start_idx:end_idx] = motif_segment
    else:
        # Multivariate case
        if source_ts.ndim == 1:
            # Source is univariate, broadcast to all channels
            motif_segment = source_ts[start_idx:end_idx]
            cf_sample[:, start_idx:end_idx] = motif_segment
        else:
            # Both are multivariate
            cf_sample[:, start_idx:end_idx] = source_ts[:, start_idx:end_idx]
    
    # Get prediction for counterfactual
    y_cf = model_predict(cf_sample)[0]
    cf_class = int(np.argmax(y_cf))
    
    if verbose:
        print(f"MG-CF Generate: Counterfactual class {cf_class}, target {target}")
        if cf_class == target:
            print(f"MG-CF Generate: Successfully generated valid counterfactual!")
        else:
            print(f"MG-CF Generate: Counterfactual did not flip to target class")
    
    return cf_sample, y_cf


def mg_cf_batch(samples, dataset, model, motifs=None, target=None,
                lengths_ratio=[0.3, 0.5, 0.7], verbose=False):
    """Generate counterfactual explanations for a batch of samples.
    
    Args:
        samples: List or array of time series samples
        dataset: Training dataset for motif mining
        model: Trained classifier model
        motifs: Pre-mined motifs (if None, will mine once and reuse)
        target: Target class (if None, will determine per sample)
        lengths_ratio: Ratios for motif lengths
        verbose: Whether to print progress
        
    Returns:
        List of (counterfactual, prediction) tuples
    """
    # Mine motifs once if not provided
    if motifs is None:
        if verbose:
            print("MG-CF Batch: Mining motifs from dataset (one-time operation)...")
        motifs = mine_motifs(dataset, lengths_ratio=lengths_ratio, verbose=verbose)
    
    results = []
    
    for i, sample in enumerate(samples):
        if verbose and i % 10 == 0:
            print(f"MG-CF Batch: Processing sample {i}/{len(samples)}")
        
        cf, pred = mg_cf_generate(
            sample, 
            dataset, 
            model, 
            motifs=motifs, 
            target=target,
            verbose=False
        )
        
        results.append((cf, pred))
    
    if verbose:
        valid_count = sum(1 for cf, pred in results if cf is not None)
        print(f"MG-CF Batch: Generated {valid_count}/{len(samples)} valid counterfactuals")
    
    return results
