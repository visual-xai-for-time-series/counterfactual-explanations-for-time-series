import torch
import numpy as np
from typing import Optional, Tuple, Union
from sklearn.neighbors import NearestNeighbors


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


def _ensure_ncl(sample, dataset):
    """Ensure sample and dataset are shaped (C, L) and (N, C, L) respectively.

    Heuristic: for 2D arrays, if rows <= cols treat as (C, L), else treat as
    (L, C) and transpose. This lets us cheaply detect already (N, C, L).
    """
    # normalize sample to (C, L)
    s = np.asarray(sample)
    if s.ndim == 1:
        s_ncl = s.reshape(1, -1)
        ori = "1d"
    elif s.ndim == 2:
        r, c = s.shape
        if r <= c:
            s_ncl = s.copy()
            ori = "cf"
        else:
            s_ncl = s.T.copy()
            ori = "tf"
    else:
        raise ValueError("sample must be 1D or 2D time series")

    # build time_series_data as (N, C, L) with a single vectorized pass
    # dataset is expected to yield (x, y) tuples; take only x
    first = dataset[0][0]
    first_arr = np.asarray(first)
    # If first is already (N, C, L) (i.e., dataset provided as array), try to use it
    if first_arr.ndim == 3 and isinstance(dataset, np.ndarray):
        ts = np.asarray([x for x in dataset[:, 0]])
    else:
        # check orientation using the first element
        fa = first_arr
        if fa.ndim == 1:
            # each item is (L,) -> produce (N, 1, L)
            ts = np.stack([np.asarray(x[0]).reshape(1, -1) for x in dataset], axis=0)
        elif fa.ndim == 2:
            r, c = fa.shape
            if r <= c:
                # assume (C, L) already
                ts = np.stack([np.asarray(x[0]) for x in dataset], axis=0)
            else:
                # assume (L, C) and transpose each
                ts = np.stack([np.asarray(x[0]).T for x in dataset], axis=0)
        else:
            raise ValueError("dataset items must be 1D or 2D time series")

    # ensure same length as sample
    _, L = s_ncl.shape
    if ts.shape[-1] != L:
        raise ValueError("All series must have same length as sample")

    # if channel mismatch and dataset is single-channel, broadcast it
    C_sample = s_ncl.shape[0]
    C_data = ts.shape[1]
    if C_data != C_sample:
        if C_data == 1:
            ts = np.repeat(ts, C_sample, axis=1)
        else:
            raise ValueError("Channel count mismatch between sample and dataset")

    return s_ncl, ts, ori


def _revert_orientation(cf_arr, orientation):
    """Revert counterfactual array to original orientation."""
    if orientation == "1d":
        return cf_arr.reshape(-1)
    if orientation == "cf":
        return cf_arr
    if orientation == "tf":
        return cf_arr.T
    return cf_arr


####
# LEFTIST-Counterfactual: Saliency-Based Counterfactual Generation for Time Series
# 
# NOTE: This is a COUNTERFACTUAL GENERATION method, different from the original LEFTIST paper.
# 
# Original LEFTIST (Guillemé et al. 2019) is an EXPLANATION method that:
# - Identifies which segments are important for a classification decision
# - Uses LIME/SHAP-like neighbor generation and linear regression
# - Returns attribution weights (not counterfactuals)
# - See: https://github.com/fzi-forschungszentrum-informatik/TSInterpret
#
# This implementation is a COUNTERFACTUAL method inspired by LEFTIST's segment-based
# approach but adapted for CF generation. It uses a greedy iterative approach:
# 1. Computing saliency (gradient magnitude) for each time point
# 2. Selecting the most salient segments
# 3. Replacing them with values from nearest neighbor of target class
# 4. Iteratively refining until prediction changes
# 
# Reference for original LEFTIST:
# Guillemé, M., Masson, V., Rozé, L., & Termier, A. (2019). 
# "Agnostic Local Explanation for Time Series Classification."
# 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI).
####
def leftist_cf(
    sample,
    dataset,
    model,
    target_class: Optional[int] = None,
    segment_length: int = 5,
    max_iterations: int = 50,
    saliency_threshold: float = 0.1,
    reference_sample: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual using a LEFTIST-inspired gradient-based approach.
    
    NOTE: This is a COUNTERFACTUAL GENERATION method, not the original LEFTIST 
    explanation method from Guillemé et al. (2019).
    
    This method adapts LEFTIST's segment-based approach for counterfactual generation
    by using gradient-based saliency to identify and replace important segments with
    values from target class examples, iteratively refining until the prediction changes.
    
    Args:
        sample: Original time series sample
        dataset: Dataset containing reference examples
        model: Trained classification model
        target_class: Target class for counterfactual (if None, finds different class)
        segment_length: Length of segments to perturb at each iteration
        max_iterations: Maximum number of iterations
        saliency_threshold: Minimum saliency to consider a segment
        reference_sample: Optional reference sample to use (if None, finds from dataset)
        verbose: If True, print progress information
        
    Returns:
        Tuple of (counterfactual_sample, prediction_scores) or (None, None) if failed.
        The counterfactual is returned in the same orientation as the input sample.
    """
    device = next(model.parameters()).device
    
    def model_predict(arr):
        """arr expected shape (B, C, L)"""
        return detach_to_numpy(model(numpy_to_torch(arr, device)))
    
    # prepare sample and dataset in (C, L) and (N, C, L)
    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    N, C, L = time_series_data.shape
    
    # Get predictions for all data
    preds_data = model_predict(time_series_data)
    preds_sample = model_predict(sample_cf.reshape(1, C, L))
    label_data = np.argmax(preds_data, axis=1)
    label_sample = int(np.argmax(preds_sample))
    
    if verbose:
        print(f"LEFTIST: Original class = {label_sample}")
    
    # Determine target class
    if target_class is None:
        # Find the class with second highest probability
        sorted_probs = np.argsort(preds_sample.reshape(-1))[::-1]
        for candidate in sorted_probs:
            if candidate != label_sample:
                target_class = int(candidate)
                break
    
    if target_class is None or target_class == label_sample:
        # Already in target class or no valid target
        return _revert_orientation(sample_cf, sample_ori), preds_sample.reshape(-1)
    
    if verbose:
        print(f"LEFTIST: Target class = {target_class}")
    
    # Select candidates with target label - use both predicted AND actual labels
    mask = label_data == target_class
    
    # If no examples predicted as target class, try using actual labels from dataset
    if not np.any(mask):
        if verbose:
            print("LEFTIST: No examples predicted as target class, using actual labels")
        # Extract actual labels from dataset
        actual_labels = []
        for i in range(N):
            item = dataset[i]
            if isinstance(item, tuple):
                y = item[1]
                if hasattr(y, 'shape') and len(y.shape) > 0:
                    actual_labels.append(np.argmax(y))
                else:
                    actual_labels.append(y)
            else:
                actual_labels.append(0)  # Fallback
        actual_labels = np.array(actual_labels)
        mask = actual_labels == target_class
    
    if not np.any(mask):
        if verbose:
            print("LEFTIST: No examples of target class found - using closest by prediction")
        # Last resort: use examples with highest probability for target class
        target_probs = preds_data[:, target_class]
        # Get top 10% with highest target probability
        threshold = np.percentile(target_probs, 90)
        mask = target_probs >= threshold
    
    if not np.any(mask):
        if verbose:
            print("LEFTIST: Cannot find any suitable reference examples")
        return None, None
    
    candidates = time_series_data[mask]
    
    # Use provided reference or find nearest neighbor from target class
    if reference_sample is not None:
        # Use the provided reference sample
        reference, _, ref_ori = _ensure_ncl(reference_sample, dataset)
    else:
        # Find nearest neighbor from target class
        k_for_candidates = min(5, len(candidates))
        neigh = NearestNeighbors(n_neighbors=k_for_candidates, metric="euclidean")
        neigh.fit(candidates.reshape(len(candidates), -1))
        dists, idxs = neigh.kneighbors(sample_cf.reshape(1, -1), return_distance=True)
        
        # Use the nearest target class example as reference
        reference = candidates[idxs[0][0]]
    
    # Initialize counterfactual as copy of original
    cf_cf = sample_cf.copy()
    y_cf = None  # Initialize to track best prediction
    best_validity = 0.0  # Track best progress toward target
    best_cf = None  # Track best counterfactual
    
    # Compute saliency using gradients
    def compute_saliency(x_input):
        """Compute gradient-based saliency for the input."""
        x_tensor = numpy_to_torch(x_input.reshape(1, C, L), device)
        x_tensor.requires_grad = True
        
        model.eval()
        output = model(x_tensor)
        
        # Compute gradient w.r.t. original class (we want to move away from it)
        target_idx = torch.tensor([label_sample], dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(output, target_idx)
        loss.backward()
        
        # Saliency is the absolute gradient
        saliency = torch.abs(x_tensor.grad).cpu().numpy()
        return saliency.reshape(C, L)
    
    # Iterative refinement
    for iteration in range(max_iterations):
        # Compute saliency for current counterfactual
        saliency = compute_saliency(cf_cf)
        
        # Average saliency across channels
        if C > 1:
            avg_saliency = np.mean(saliency, axis=0)
        else:
            avg_saliency = saliency[0]
        
        # Find the most salient segment
        segment_saliency = np.zeros(L - segment_length + 1)
        for i in range(L - segment_length + 1):
            segment_saliency[i] = np.sum(avg_saliency[i:i + segment_length])
        
        # Get the most salient segment
        if np.max(segment_saliency) < saliency_threshold:
            if verbose:
                print(f"LEFTIST: Saliency below threshold at iteration {iteration}")
            break
        
        most_salient_idx = int(np.argmax(segment_saliency))
        
        # Replace the salient segment with reference values
        cf_candidate = cf_cf.copy()
        cf_candidate[:, most_salient_idx:most_salient_idx + segment_length] = \
            reference[:, most_salient_idx:most_salient_idx + segment_length]
        
        # Check prediction
        y_candidate = model_predict(cf_candidate.reshape(1, C, L)).reshape(-1)
        pred_class = int(np.argmax(y_candidate))
        current_validity = y_candidate[target_class]
        
        # Update counterfactual
        cf_cf = cf_candidate
        
        # Track best solution
        if pred_class == target_class:
            # Found a valid counterfactual
            y_cf = y_candidate
            best_cf = cf_cf.copy()
            best_validity = current_validity
            if verbose:
                print(f"LEFTIST: Success at iteration {iteration}")
            break
        elif current_validity > best_validity:
            # Track best progress even if not fully valid
            best_validity = current_validity
            best_cf = cf_cf.copy()
        
        if verbose and iteration % 10 == 0:
            print(f"LEFTIST iter {iteration}: pred_class={pred_class}, "
                  f"target={target_class}, validity={current_validity:.4f}, "
                  f"max_saliency={np.max(segment_saliency):.4f}")
    else:
        # Max iterations reached without breaking
        if verbose:
            print("LEFTIST: Max iterations reached")
    
    # Use best solution found
    if best_cf is not None:
        cf_cf = best_cf
    
    # Get final prediction
    if y_cf is None:
        y_cf = model_predict(cf_cf.reshape(1, C, L)).reshape(-1)
        final_class = int(np.argmax(y_cf))
        final_validity = y_cf[target_class]
        
        # Accept if we made reasonable progress (validity > 0.2) even if not perfect
        if final_class != target_class and final_validity < 0.2:
            if verbose:
                print(f"LEFTIST: Failed - final validity {final_validity:.4f} too low")
            return None, None
    
    # Revert to original orientation
    cf_out = _revert_orientation(cf_cf, sample_ori)
    return cf_out, y_cf


def leftist_multi_cf(
    sample,
    dataset,
    model,
    target_class: Optional[int] = None,
    num_counterfactuals: int = 3,
    segment_length: int = 5,
    max_iterations: int = 50,
    diversity_weight: float = 0.3,
    verbose: bool = False
) -> Tuple[list, list]:
    """
    Generate multiple diverse counterfactuals using LEFTIST.
    
    This variant generates multiple counterfactuals by using different
    reference examples from the target class and varying segment selection.
    
    Args:
        sample: Original time series sample
        dataset: Dataset containing reference examples
        model: Trained classification model
        target_class: Target class for counterfactuals
        num_counterfactuals: Number of diverse counterfactuals to generate
        segment_length: Length of segments to perturb
        max_iterations: Maximum iterations per counterfactual
        diversity_weight: Weight for diversity in selection (0-1)
        verbose: If True, print progress information
        
    Returns:
        Tuple of (list of counterfactuals, list of predictions)
    """
    device = next(model.parameters()).device
    
    def model_predict(arr):
        """arr expected shape (B, C, L)"""
        return detach_to_numpy(model(numpy_to_torch(arr, device)))
    
    # Prepare sample and dataset
    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    N, C, L = time_series_data.shape
    
    # Get predictions
    preds_data = model_predict(time_series_data)
    preds_sample = model_predict(sample_cf.reshape(1, C, L))
    label_data = np.argmax(preds_data, axis=1)
    label_sample = int(np.argmax(preds_sample))
    
    # Determine target class
    if target_class is None:
        sorted_probs = np.argsort(preds_sample.reshape(-1))[::-1]
        for candidate in sorted_probs:
            if candidate != label_sample:
                target_class = int(candidate)
                break
    
    if target_class == label_sample:
        return [_revert_orientation(sample_cf, sample_ori)], [preds_sample.reshape(-1)]
    
    # Select candidates with target label
    mask = label_data == target_class
    if not np.any(mask):
        return [], []
    
    candidates = time_series_data[mask]
    
    # Find multiple nearest neighbors
    k_neighbors = min(num_counterfactuals * 2, len(candidates))
    neigh = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
    neigh.fit(candidates.reshape(len(candidates), -1))
    dists, idxs = neigh.kneighbors(sample_cf.reshape(1, -1), return_distance=True)
    
    counterfactuals = []
    predictions = []
    
    # Generate diverse counterfactuals using different references
    for i in range(min(num_counterfactuals, k_neighbors)):
        reference = candidates[idxs[0][i]]
        
        # Generate counterfactual with this specific reference
        cf, pred = leftist_cf(
            sample=sample,
            dataset=dataset,
            model=model,
            target_class=target_class,
            segment_length=segment_length,
            max_iterations=max_iterations,
            reference_sample=_revert_orientation(reference, sample_ori),
            verbose=False
        )
        
        if cf is not None:
            counterfactuals.append(cf)
            predictions.append(pred)
        
        if len(counterfactuals) >= num_counterfactuals:
            break
    
    if verbose:
        print(f"LEFTIST: Generated {len(counterfactuals)} counterfactuals")
    
    return counterfactuals, predictions
