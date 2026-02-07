import numpy as np
import torch
try:
    import stumpy
except ImportError:
    raise ImportError("stumpy is required for this implementation. Install with: pip install stumpy")


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


####
# Motif-guided Time Series Counterfactual Explanations - STUMPY version
#
# This is an optimized implementation using STUMPY (matrix profile) for efficient
# motif discovery instead of brute-force subsequence search.
#
# STUMPY provides:
# - Fast matrix profile computation using STOMP algorithm
# - Efficient motif discovery using matrix profile indices
# - GPU acceleration support
# - Handles both univariate and multivariate time series
####


def _extract_subsequence_with_mp(time_series, m, class_label, dataset_labels):
    """Extract discriminative subsequences using matrix profile.
    
    Args:
        time_series: Time series of shape (length,) or (channels, length)
        m: Motif/subsequence length
        class_label: Label of this time series
        dataset_labels: Labels for all time series in dataset
        
    Returns:
        List of (start_idx, quality_score, subsequence, matrix_profile_distance)
    """
    # Handle univariate vs multivariate
    if time_series.ndim == 1:
        # Compute matrix profile for univariate
        mp = stumpy.stump(time_series, m)
        # mp is (n_subsequences, 4) with columns: [distance, index, left_index, right_index]
        
        # Extract subsequences with their distances
        candidates = []
        for i in range(len(mp)):
            start = i
            end = start + m
            subseq = time_series[start:end]
            mp_distance = mp[i, 0]  # Matrix profile distance
            candidates.append((start, mp_distance, subseq))
    else:
        # For multivariate, use stumpy's multivariate support
        # stumpy expects shape (n_channels, n_timepoints)
        mp = stumpy.mstump(time_series, m)
        # mstump returns similar structure but for multivariate
        
        candidates = []
        for i in range(len(mp)):
            start = i
            end = start + m
            subseq = time_series[:, start:end]
            mp_distance = mp[i, 0]
            candidates.append((start, mp_distance, subseq))
    
    return candidates


def _compute_discriminatory_score(subseq, dataset, labels, current_label):
    """Compute how well a subsequence discriminates between classes using matrix profile.
    
    Args:
        subseq: Subsequence to evaluate
        dataset: List of time series
        labels: Array of class labels
        current_label: Label of the class we're finding motifs for
        
    Returns:
        Discriminatory score (higher is better)
    """
    m = len(subseq) if subseq.ndim == 1 else subseq.shape[-1]
    distances = []
    
    for i, ts in enumerate(dataset):
        if isinstance(ts, tuple):
            ts = ts[0]
        if isinstance(ts, torch.Tensor):
            ts = detach_to_numpy(ts)
        
        # Use stumpy for efficient distance computation
        if subseq.ndim == 1 and ts.ndim == 1:
            # Univariate case - use mass (Mueen's Algorithm for Similarity Search)
            dist_profile = stumpy.mass(subseq, ts)
            min_dist = np.min(dist_profile)
        elif subseq.ndim > 1 and ts.ndim > 1:
            # Multivariate case
            # Compute z-normalized Euclidean distance for each position
            min_dist = float('inf')
            for start in range(ts.shape[-1] - m + 1):
                candidate = ts[:, start:start + m]
                dist = np.sqrt(np.sum((subseq - candidate) ** 2))
                min_dist = min(min_dist, dist)
        else:
            # Mixed dimensions - fall back to simple distance
            min_dist = float('inf')
            for start in range((len(ts) if ts.ndim == 1 else ts.shape[-1]) - m + 1):
                if ts.ndim == 1:
                    candidate = ts[start:start + m]
                else:
                    candidate = ts[:, start:start + m]
                dist = np.sqrt(np.sum((subseq - candidate) ** 2))
                min_dist = min(min_dist, dist)
        
        distances.append(min_dist)
    
    distances = np.array(distances)
    labels_array = np.array(labels)
    
    # Calculate discriminatory power using information gain
    quality = _assess_candidate_quality(distances, labels_array)
    
    return quality


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
    
    # Handle different label formats
    if len(sorted_labels.shape) > 1 and sorted_labels.shape[1] > 1:
        # One-hot encoded
        label_classes = np.argmax(sorted_labels, axis=1)
    else:
        label_classes = sorted_labels.flatten()
    
    # Count unique classes
    unique_classes = np.unique(label_classes)
    if len(unique_classes) < 2:
        return 0.0
    
    # For binary classification
    target_class = unique_classes[0]
    n_pos = np.sum(label_classes == target_class)
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
        left_labels = label_classes[:i]
        left_pos = np.sum(left_labels == target_class)
        left_neg = len(left_labels) - left_pos
        
        # Right split
        right_labels = label_classes[i:]
        right_pos = np.sum(right_labels == target_class)
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


def mine_motifs_stumpy(dataset, lengths_ratio=[0.3, 0.5, 0.7], top_k=10, verbose=False):
    """Mine discriminative motifs using STUMPY's matrix profile for efficient discovery.
    
    This is an optimized version of Algorithm 1 from the MG-CF paper using matrix profiles.
    
    Args:
        dataset: List/array of (x, y) tuples where x is time series and y is binary label
        lengths_ratio: Ratios of time series length to use for motif lengths
        top_k: Number of top candidate subsequences to evaluate per length (default 10)
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping class labels to best motif info:
            {class: (ts_idx, start_idx, end_idx, motif, quality)}
    """
    if verbose:
        print("MG-CF STUMPY: Mining discriminative motifs using matrix profiles...")
    
    # Extract data and labels
    X = []
    y = []
    for item in dataset:
        if isinstance(item, tuple):
            X.append(item[0])
            y.append(item[1])
        else:
            raise ValueError("Dataset must contain (x, y) tuples")
    
    y_array = np.array(y)
    
    # Determine motif lengths based on time series length
    sample_ts = X[0]
    if isinstance(sample_ts, torch.Tensor):
        sample_ts = detach_to_numpy(sample_ts)
    
    ts_length = len(sample_ts) if sample_ts.ndim == 1 else sample_ts.shape[-1]
    lengths = [int(ts_length * ratio) for ratio in lengths_ratio]
    
    if verbose:
        print(f"MG-CF STUMPY: Extracting motifs of lengths {lengths} from {len(X)} time series")
    
    # Store best motifs for each class
    best_motifs = {}
    
    # Process each time series
    for ts_idx in range(len(X)):
        ts = X[ts_idx]
        if isinstance(ts, torch.Tensor):
            ts = detach_to_numpy(ts)
        
        # Get class label
        ts_class = y_array[ts_idx]
        if hasattr(ts_class, 'shape') and len(ts_class.shape) > 0:
            ts_class = int(np.argmax(ts_class)) if ts_class.shape[0] > 1 else int(ts_class[0])
        else:
            ts_class = int(ts_class)
        
        if verbose and ts_idx % 20 == 0:
            print(f"MG-CF STUMPY: Processing time series {ts_idx}/{len(X)}")
        
        # For each motif length
        for m in lengths:
            # Compute matrix profile using STUMPY
            try:
                # Ensure ts is float64 (STUMPY requirement)
                ts_float64 = ts.astype(np.float64) if ts.dtype != np.float64 else ts
                
                # Check if univariate or multivariate
                # If 2D with first dim = 1, it's univariate data with channel dim
                is_univariate = ts.ndim == 1 or (ts.ndim == 2 and ts.shape[0] == 1)
                
                if is_univariate:
                    # Univariate: use STUMP
                    # Flatten if needed
                    ts_1d = ts_float64.flatten() if ts_float64.ndim > 1 else ts_float64
                    mp = stumpy.stump(ts_1d, m)
                    # mp is shape (n_subsequences, 4) with columns: [distance, index, left, right]
                    
                    # Get top-k most distinctive subsequences (highest matrix profile values)
                    # High matrix profile distance means the subsequence is unique/discriminative
                    top_indices = np.argsort(mp[:, 0])[-top_k:][::-1]
                    
                    candidates = []
                    for idx in top_indices:
                        start = idx
                        end = start + m
                        # Reconstruct in original shape
                        if ts.ndim == 1:
                            subseq = ts[start:end]
                        else:
                            subseq = ts[:, start:end]
                        candidates.append((start, end, subseq))
                        
                else:
                    # Multivariate: use MSTUMP
                    mp = stumpy.mstump(ts_float64, m)
                    # mstump returns different structure, need to check
                    if isinstance(mp, tuple):
                        mp = mp[0]  # Get the matrix profile part
                    top_indices = np.argsort(mp[:, 0])[-top_k:][::-1]
                    
                    candidates = []
                    for idx in top_indices:
                        start = idx
                        end = start + m
                        subseq = ts[:, start:end]
                        candidates.append((start, end, subseq))
                
                # Evaluate discriminatory power of top candidates
                for start_idx, end_idx, subseq in candidates:
                    quality = _compute_discriminatory_score(subseq, X, y_array, ts_class)
                    
                    # Update best motif for this class if better quality found
                    if ts_class not in best_motifs or quality > best_motifs[ts_class][4]:
                        best_motifs[ts_class] = (ts_idx, start_idx, end_idx, subseq, quality)
                        
            except Exception as e:
                if verbose:
                    print(f"MG-CF STUMPY: Error processing TS {ts_idx} with length {m}: {e}")
                continue
    
    if verbose:
        for cls, (ts_idx, start, end, motif, quality) in best_motifs.items():
            print(f"MG-CF STUMPY: Class {cls} - Best motif from TS {ts_idx} [{start}:{end}], quality={quality:.4f}")
    
    return best_motifs


def mg_cf_generate_stumpy(sample, dataset, model, motifs=None, target=None, 
                          lengths_ratio=[0.3, 0.5, 0.7], top_k=10, verbose=False):
    """Generate counterfactual explanation using STUMPY-based Motif-Guided approach.
    
    This implements Algorithm 2 from the MG-CF paper using matrix profiles for
    efficient motif discovery.
    
    Args:
        sample: Input time series to explain (numpy array or torch tensor)
        dataset: Training dataset used for motif mining (list of (x, y) tuples)
        model: Trained classifier model
        motifs: Pre-mined motifs (if None, will mine from dataset using STUMPY)
        target: Target class for counterfactual (if None, will use opposite of prediction)
        lengths_ratio: Ratios for motif lengths if mining is needed
        top_k: Number of top candidates to evaluate per motif length
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
        print(f"MG-CF STUMPY Generate: Original class {orig_class}, Target class {target}")
    
    # Mine motifs if not provided
    if motifs is None:
        if verbose:
            print("MG-CF STUMPY Generate: Mining motifs from dataset using matrix profiles...")
        motifs = mine_motifs_stumpy(dataset, lengths_ratio=lengths_ratio, top_k=top_k, verbose=verbose)
    
    # Check if we have motifs for the target class
    if target not in motifs:
        if verbose:
            print(f"MG-CF STUMPY Generate: No motif found for target class {target}")
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
        print(f"MG-CF STUMPY Generate: Replacing region [{start_idx}:{end_idx}] with target class motif (quality={quality:.4f})")
    
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
        print(f"MG-CF STUMPY Generate: Counterfactual class {cf_class}, target {target}")
        if cf_class == target:
            print(f"MG-CF STUMPY Generate: Successfully generated valid counterfactual!")
        else:
            print(f"MG-CF STUMPY Generate: Counterfactual did not flip to target class")
    
    return cf_sample, y_cf


def mg_cf_batch_stumpy(samples, dataset, model, motifs=None, target=None,
                       lengths_ratio=[0.3, 0.5, 0.7], top_k=10, verbose=False):
    """Generate counterfactual explanations for a batch using STUMPY-based approach.
    
    Args:
        samples: List or array of time series samples
        dataset: Training dataset for motif mining
        model: Trained classifier model
        motifs: Pre-mined motifs (if None, will mine once and reuse)
        target: Target class (if None, will determine per sample)
        lengths_ratio: Ratios for motif lengths
        top_k: Number of top candidates per length
        verbose: Whether to print progress
        
    Returns:
        List of (counterfactual, prediction) tuples
    """
    # Mine motifs once if not provided
    if motifs is None:
        if verbose:
            print("MG-CF STUMPY Batch: Mining motifs using matrix profiles (one-time operation)...")
        motifs = mine_motifs_stumpy(dataset, lengths_ratio=lengths_ratio, top_k=top_k, verbose=verbose)
    
    results = []
    
    for i, sample in enumerate(samples):
        if verbose and i % 10 == 0:
            print(f"MG-CF STUMPY Batch: Processing sample {i}/{len(samples)}")
        
        cf, pred = mg_cf_generate_stumpy(
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
        print(f"MG-CF STUMPY Batch: Generated {valid_count}/{len(samples)} valid counterfactuals")
    
    return results
