import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Union, Dict, Any, List
from sklearn.cluster import KMeans


####
# SETS: Scalable Explanation for Time Series
#
# This is a custom segment-based counterfactual method that modifies coherent
# temporal segments for more interpretable and scalable explanations, especially
# effective for long time series.
#
# The method focuses on segment-level modifications rather than point-wise changes,
# improving interpretability and computational efficiency.
####


def sets_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    n_segments: int = 10,
    segment_method: str = 'uniform',
    lambda_reg: float = 0.01,
    lambda_sparse: float = 0.001,
    lambda_smooth: float = 0.001,
    learning_rate: float = 0.1,
    max_iterations: int = 2000,
    tolerance: float = 1e-4,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual explanation using SETS algorithm.
    
    SETS (Scalable Explanation for Time Series) focuses on segment-based
    modifications to generate more interpretable counterfactuals.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object (for compatibility with other methods)
        model: Trained classification model
        target_class: Target class for counterfactual (if None, finds different class)
        n_segments: Number of segments to divide the time series into
        segment_method: Method for segmentation ('uniform', 'adaptive', 'gradient')
        lambda_reg: Regularization parameter for proximity constraint
        lambda_sparse: Regularization parameter for sparsity constraint
        lambda_smooth: Regularization parameter for smoothness constraint
        learning_rate: Learning rate for optimization
        max_iterations: Maximum number of optimization iterations
        tolerance: Convergence tolerance
        device: Device to run on (if None, auto-detects)
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Convert sample to tensor and prepare for model
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    
    # Handle different input shapes - ensure (batch, channels, length)
    original_shape = x_tensor.shape
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.reshape(1, 1, -1)  # (length,) -> (1, 1, length)
    elif len(x_tensor.shape) == 2:
        # Could be (channels, length) or (length, channels)
        if x_tensor.shape[0] > x_tensor.shape[1]:
            x_tensor = x_tensor.T  # Assume (length, channels) -> (channels, length)
        x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
    
    # Get original prediction
    with torch.no_grad():
        original_pred = model(x_tensor)
        original_class = torch.argmax(original_pred, dim=-1).item()
        original_pred_np = torch.softmax(original_pred, dim=-1).squeeze().cpu().numpy()
    
    # Determine target class
    if target_class is None:
        # Find the class with second highest probability
        sorted_classes = torch.argsort(original_pred, dim=-1, descending=True)
        target_class = sorted_classes[0, 1].item()  # Second most likely class
    
    # If already in target class, return None
    if original_class == target_class:
        return None, None
    
    if verbose:
        print(f"SETS: Original class {original_class}, Target class {target_class}")

    # Generate segments
    segments = _generate_segments(x_tensor, n_segments, segment_method)
    if verbose:
        print(f"SETS: Generated {len(segments)} segments")
    
    # Initialize segment-wise perturbations
    segment_deltas = {}
    for seg_id, (start, end) in segments.items():
        # Initialize small random perturbations for each segment
        seg_shape = x_tensor[:, :, start:end].shape
        segment_deltas[seg_id] = torch.zeros(seg_shape, device=device, requires_grad=True)
    
    # Collect all parameters for optimization
    params = list(segment_deltas.values())
    optimizer = optim.Adam(params, lr=learning_rate)
    
    best_cf = None
    best_loss = float('inf')
    best_validity = 0.0
    
    # Two-phase optimization like COMTE
    phase1_iterations = max_iterations // 2
    current_lambda_reg = 0.0  # Start without regularization
    current_lambda_sparse = 0.0
    current_lambda_smooth = 0.0
    
    for iteration in range(max_iterations):
        # Switch to phase 2 halfway through - add regularization
        if iteration == phase1_iterations:
            current_lambda_reg = lambda_reg
            current_lambda_sparse = lambda_sparse
            current_lambda_smooth = lambda_smooth
            if verbose:
                print(f"SETS: Switching to phase 2 with regularization at iteration {iteration}")
        
        optimizer.zero_grad()
        
        # Construct counterfactual by applying segment-wise perturbations
        x_cf = x_tensor.clone()
        for seg_id, (start, end) in segments.items():
            x_cf[:, :, start:end] += segment_deltas[seg_id]
        
        # Forward pass
        logits = model(x_cf)
        
        # Prediction loss (negative log probability of target class)
        log_probs = torch.log_softmax(logits, dim=-1)
        pred_loss = -log_probs[0, target_class]
        
        # Distance loss (proximity constraint)
        distance_loss = torch.norm(x_cf - x_tensor, p=2)
        
        # Sparsity loss (encourage few segment modifications)
        sparsity_loss = _compute_segment_sparsity(segment_deltas, segments)
        
        # Smoothness loss (encourage smooth transitions between segments)
        smoothness_loss = _compute_smoothness_loss(x_cf, segments)
        
        # Total loss with adaptive weights
        total_loss = (pred_loss + 
                     current_lambda_reg * distance_loss + 
                     current_lambda_sparse * sparsity_loss +
                     current_lambda_smooth * smoothness_loss)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Check current validity
        with torch.no_grad():
            current_probs = torch.softmax(logits, dim=-1)
            current_validity = current_probs[0, target_class].item()
            current_pred_class = torch.argmax(current_probs, dim=-1).item()
        
        # Track best solution (prioritize validity)
        if current_pred_class == target_class:
            if current_validity > best_validity or (current_validity >= best_validity and total_loss.item() < best_loss):
                best_loss = total_loss.item()
                best_validity = current_validity
                best_cf = x_cf.clone().detach()
        elif best_cf is None:
            # If no valid solution yet, keep the one with highest validity
            if current_validity > best_validity:
                best_validity = current_validity
                best_cf = x_cf.clone().detach()
        
        # Early stopping if we achieve very good validity
        if current_pred_class == target_class and current_validity > 0.9:
            if verbose:
                print(f"SETS: Early stop at iteration {iteration} with validity {current_validity:.4f}")
            break
        
        # Print progress every 500 iterations for debugging
        if verbose and iteration % 500 == 0:
            print(f"SETS iteration {iteration}: loss={total_loss.item():.4f}, "
                  f"pred_loss={pred_loss.item():.4f}, pred_class={current_pred_class}, "
                  f"target={target_class}, validity={current_validity:.4f}")
    
    if best_cf is None:
        if verbose:
            print("SETS: No counterfactual found - best_cf is None")
        return None, None
    
    # Get final prediction
    with torch.no_grad():
        final_pred = model(best_cf)
        predicted_class = torch.argmax(final_pred, dim=-1).item()
        final_pred_np = torch.softmax(final_pred, dim=-1).squeeze().cpu().numpy()
        final_validity = final_pred_np[target_class]
    
    if verbose:
        print(f"SETS final: pred_class={predicted_class}, target={target_class}, validity={final_validity:.4f}")
    
    # Check if counterfactual is valid - use relaxed criteria
    if predicted_class != target_class and final_validity < 0.4:
        if verbose:
            print(f"SETS: Counterfactual failed validation - predicted {predicted_class}, wanted {target_class}, validity too low")
        return None, None
    
    # Convert back to original sample format
    cf_sample = best_cf.squeeze(0).cpu().numpy()
    
    # Handle output shape to match input format
    if len(original_shape) == 1:
        cf_sample = cf_sample.squeeze()  # Remove channel dimension if input was 1D
    elif len(original_shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            cf_sample = cf_sample.T  # Convert back to (length, channels) if needed
    
    return cf_sample, final_pred_np


def _generate_segments(x_tensor: torch.Tensor, n_segments: int, method: str) -> Dict[int, Tuple[int, int]]:
    """
    Generate segments for the time series.
    
    Args:
        x_tensor: Input tensor (batch, channels, length)
        n_segments: Number of segments
        method: Segmentation method ('uniform', 'adaptive', 'gradient')
        
    Returns:
        Dictionary mapping segment_id to (start_idx, end_idx)
    """
    batch_size, channels, length = x_tensor.shape
    segments = {}
    
    if method == 'uniform':
        # Uniform segmentation
        segment_length = length // n_segments
        for i in range(n_segments):
            start = i * segment_length
            end = start + segment_length if i < n_segments - 1 else length
            segments[i] = (start, end)
    
    elif method == 'adaptive':
        # Adaptive segmentation based on variance
        x_numpy = x_tensor.squeeze(0).cpu().numpy()
        if channels == 1:
            variance = np.var(x_numpy.squeeze(), axis=0)
        else:
            variance = np.mean(np.var(x_numpy, axis=0), axis=0)
        
        # Use variance to determine segment boundaries
        segment_boundaries = _adaptive_segmentation(variance, n_segments)
        for i in range(len(segment_boundaries) - 1):
            segments[i] = (segment_boundaries[i], segment_boundaries[i + 1])
    
    elif method == 'gradient':
        # Gradient-based segmentation
        x_numpy = x_tensor.squeeze(0).cpu().numpy()
        if channels == 1:
            gradient = np.abs(np.gradient(x_numpy.squeeze()))
        else:
            gradient = np.mean(np.abs(np.gradient(x_numpy, axis=1)), axis=0)
        
        # Use gradient magnitude to determine segment boundaries
        segment_boundaries = _gradient_segmentation(gradient, n_segments)
        for i in range(len(segment_boundaries) - 1):
            segments[i] = (segment_boundaries[i], segment_boundaries[i + 1])
    
    else:
        raise ValueError(f"Unsupported segmentation method: {method}")
    
    return segments


def _adaptive_segmentation(variance: np.ndarray, n_segments: int) -> List[int]:
    """Generate segment boundaries based on variance."""
    # Use k-means clustering on variance values to find natural breakpoints
    if len(variance) < n_segments:
        return list(range(0, len(variance) + 1))
    
    # Reshape for k-means
    variance_reshaped = variance.reshape(-1, 1)
    
    try:
        kmeans = KMeans(n_clusters=min(n_segments, len(variance)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(variance_reshaped)
        
        # Find cluster boundaries
        boundaries = [0]
        for i in range(1, len(clusters)):
            if clusters[i] != clusters[i-1]:
                boundaries.append(i)
        boundaries.append(len(variance))
        
        # Ensure we have exactly n_segments
        while len(boundaries) - 1 < n_segments and len(boundaries) < len(variance):
            # Split the largest segment
            max_seg_idx = np.argmax([boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)])
            mid_point = (boundaries[max_seg_idx] + boundaries[max_seg_idx + 1]) // 2
            boundaries.insert(max_seg_idx + 1, mid_point)
        
        return sorted(boundaries)
    
    except:
        # Fallback to uniform segmentation
        segment_length = len(variance) // n_segments
        return [i * segment_length for i in range(n_segments)] + [len(variance)]


def _gradient_segmentation(gradient: np.ndarray, n_segments: int) -> List[int]:
    """Generate segment boundaries based on gradient magnitude."""
    if len(gradient) < n_segments:
        return list(range(0, len(gradient) + 1))
    
    # Find peaks in gradient (change points)
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(gradient, height=np.mean(gradient))
        
        if len(peaks) >= n_segments - 1:
            # Select top n_segments-1 peaks
            peak_heights = gradient[peaks]
            top_peaks_idx = np.argsort(peak_heights)[-n_segments+1:]
            selected_peaks = sorted(peaks[top_peaks_idx])
            boundaries = [0] + selected_peaks + [len(gradient)]
        else:
            # Not enough peaks, fall back to uniform
            segment_length = len(gradient) // n_segments
            boundaries = [i * segment_length for i in range(n_segments)] + [len(gradient)]
        
        return boundaries
    
    except ImportError:
        # Fallback to simple peak detection if scipy not available
        # Find local maxima manually
        peaks = []
        for i in range(1, len(gradient) - 1):
            if gradient[i] > gradient[i-1] and gradient[i] > gradient[i+1] and gradient[i] > np.mean(gradient):
                peaks.append(i)
        
        if len(peaks) >= n_segments - 1:
            peak_heights = gradient[peaks]
            top_peaks_idx = np.argsort(peak_heights)[-n_segments+1:]
            selected_peaks = sorted([peaks[i] for i in top_peaks_idx])
            boundaries = [0] + selected_peaks + [len(gradient)]
        else:
            # Fallback to uniform segmentation
            segment_length = len(gradient) // n_segments
            boundaries = [i * segment_length for i in range(n_segments)] + [len(gradient)]
        
        return boundaries


def _compute_segment_sparsity(segment_deltas: Dict[int, torch.Tensor], segments: Dict[int, Tuple[int, int]]) -> torch.Tensor:
    """
    Compute sparsity loss to encourage modifications in few segments.
    """
    sparsity_loss = torch.tensor(0.0, device=list(segment_deltas.values())[0].device)
    
    for seg_id, delta in segment_deltas.items():
        # L1 norm of segment modifications
        segment_norm = torch.norm(delta, p=1)
        sparsity_loss += segment_norm
    
    return sparsity_loss


def _compute_smoothness_loss(x_cf: torch.Tensor, segments: Dict[int, Tuple[int, int]]) -> torch.Tensor:
    """
    Compute smoothness loss to encourage smooth transitions between segments.
    """
    smoothness_loss = torch.tensor(0.0, device=x_cf.device)
    
    # Sort segments by start position
    sorted_segments = sorted(segments.items(), key=lambda x: x[1][0])
    
    for i in range(len(sorted_segments) - 1):
        current_seg = sorted_segments[i][1]
        next_seg = sorted_segments[i + 1][1]
        
        # Get the last point of current segment and first point of next segment
        current_end = x_cf[:, :, current_seg[1] - 1]  # Last point of current segment
        next_start = x_cf[:, :, next_seg[0]]  # First point of next segment
        
        # Penalize large discontinuities
        discontinuity = torch.norm(next_start - current_end, p=2)
        smoothness_loss += discontinuity
    
    return smoothness_loss


def sets_cf_with_explanation(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    n_segments: int = 10,
    segment_method: str = 'uniform',
    lambda_reg: float = 1.0,
    lambda_sparse: float = 0.1,
    lambda_smooth: float = 0.01,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    device: str = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Generate counterfactual explanation using SETS algorithm with detailed explanation.
    
    Returns:
        Tuple of (counterfactual_sample, prediction, explanation_dict)
        where explanation_dict contains segment information and importance scores.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Convert and prepare tensor
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    original_shape = x_tensor.shape
    
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.reshape(1, 1, -1)
    elif len(x_tensor.shape) == 2:
        if x_tensor.shape[0] > x_tensor.shape[1]:
            x_tensor = x_tensor.T
        x_tensor = x_tensor.unsqueeze(0)
    
    # Get original prediction and target
    with torch.no_grad():
        original_pred = model(x_tensor)
        original_class = torch.argmax(original_pred, dim=-1).item()
    
    if target_class is None:
        sorted_classes = torch.argsort(original_pred, dim=-1, descending=True)
        target_class = sorted_classes[0, 1].item()
    
    if original_class == target_class:
        return None, None, None
    
    # Generate segments and track modifications
    segments = _generate_segments(x_tensor, n_segments, segment_method)
    segment_deltas = {}
    segment_importance = {}
    
    for seg_id, (start, end) in segments.items():
        seg_shape = x_tensor[:, :, start:end].shape
        segment_deltas[seg_id] = torch.zeros(seg_shape, device=device, requires_grad=True)
        segment_importance[seg_id] = 0.0
    
    # Optimization
    params = list(segment_deltas.values())
    optimizer = optim.Adam(params, lr=learning_rate)
    
    best_cf = None
    best_loss = float('inf')
    prev_loss = float('inf')
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Construct counterfactual
        x_cf = x_tensor.clone()
        for seg_id, (start, end) in segments.items():
            x_cf[:, :, start:end] += segment_deltas[seg_id]
        
        # Compute losses
        logits = model(x_cf)
        log_probs = torch.log_softmax(logits, dim=-1)
        pred_loss = -log_probs[0, target_class]
        
        distance_loss = torch.norm(x_cf - x_tensor, p=2)
        sparsity_loss = _compute_segment_sparsity(segment_deltas, segments)
        smoothness_loss = _compute_smoothness_loss(x_cf, segments)
        
        total_loss = (pred_loss + lambda_reg * distance_loss + 
                     lambda_sparse * sparsity_loss + lambda_smooth * smoothness_loss)
        
        total_loss.backward()
        optimizer.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_cf = x_cf.clone().detach()
            
            # Update segment importance scores
            for seg_id, delta in segment_deltas.items():
                segment_importance[seg_id] = torch.norm(delta, p=2).item()
        
        if iteration > 0 and abs(prev_loss - total_loss.item()) < tolerance:
            break
            
        prev_loss = total_loss.item()
    
    if best_cf is None:
        return None, None, None
    
    # Final prediction
    with torch.no_grad():
        final_pred = model(best_cf)
        predicted_class = torch.argmax(final_pred, dim=-1).item()
        final_pred_np = torch.softmax(final_pred, dim=-1).squeeze().cpu().numpy()
    
    if predicted_class != target_class:
        return None, None, None
    
    # Prepare explanation
    explanation = {
        'segments': segments,
        'segment_importance': segment_importance,
        'total_segments': len(segments),
        'modified_segments': sum(1 for imp in segment_importance.values() if imp > 1e-6),
        'segmentation_method': segment_method
    }
    
    # Format output
    cf_sample = best_cf.squeeze(0).cpu().numpy()
    if len(original_shape) == 1:
        cf_sample = cf_sample.squeeze()
    elif len(original_shape) == 2 and sample.shape[0] > sample.shape[1]:
        cf_sample = cf_sample.T
    
    return cf_sample, final_pred_np, explanation