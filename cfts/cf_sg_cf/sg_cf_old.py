import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


####
# SG-CF: Shapelet-Guided Counterfactual Explanations
#
# This is a custom implementation that uses shapelet-based representations to
# identify and modify discriminative subsequences for counterfactual generation.
#
# Inspired by shapelet-based time series analysis methods adapted for
# counterfactual explanations, focusing on modifying the most discriminative
# temporal patterns.
####


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


def manhattan_dist(x, y):
    """Calculate L1/Manhattan distance between tensors."""
    return torch.sum(torch.abs(x - y))


def euclidean_dist(x, y):
    """Calculate L2/Euclidean distance between tensors."""
    return torch.sqrt(torch.sum(torch.abs(x - y) ** 2))


def sliding_window(x, window_size):
    """Create sliding windows over the sequence dimension."""
    # x shape: (batch, channels, length)
    batch, channels, length = x.shape
    if length < window_size:
        return x.unsqueeze(2)  # Return as is with extra dimension
    
    # Use unfold to create sliding windows
    # unfold(dimension, size, step)
    windows = x.unfold(2, window_size, 1)  # (batch, channels, num_windows, window_size)
    return windows


def shapelet_distance(x, shapelet):
    """
    Calculate minimum distance between time series and shapelet using sliding window.
    
    Args:
        x: Input time series tensor (batch, channels, length)
        shapelet: Shapelet tensor (1, channels, shapelet_length) or (shapelet_length,)
    
    Returns:
        Minimum distance across all sliding window positions
    """
    if len(shapelet.shape) == 1:
        shapelet = shapelet.reshape(1, 1, -1)
    elif len(shapelet.shape) == 2:
        shapelet = shapelet.unsqueeze(0)
    
    shapelet_len = shapelet.shape[2]
    
    # Create sliding windows
    windows = sliding_window(x, shapelet_len)  # (batch, channels, num_windows, window_size)
    
    if windows.shape[2] == 1:
        # If only one window (shapelet longer than series), just compute distance
        dist = torch.sum(torch.abs(windows.squeeze(2) - shapelet.squeeze(0)))
        return dist
    
    # Reshape for distance computation
    # windows: (batch, channels, num_windows, window_size)
    # shapelet: (1, channels, shapelet_length)
    
    # Compute distances for all windows
    num_windows = windows.shape[2]
    distances = []
    for i in range(num_windows):
        window = windows[:, :, i, :]  # (batch, channels, window_size)
        dist = torch.sum(torch.abs(window - shapelet.squeeze(0)))
        distances.append(dist)
    
    # Return minimum distance
    min_dist = torch.stack(distances).min()
    return min_dist


def getprominentsegment(gradient, seglen):
    """
    Find the segment with maximum gradient magnitude.
    
    Args:
        gradient: 1D array of gradients
        seglen: Length of segment to find
    
    Returns:
        Start and end indices of the prominent segment
    """
    gradient = gradient.cpu().detach().numpy() if torch.is_tensor(gradient) else gradient
    maxgradient = 0
    idx_start, idx_end = 0, seglen
    
    for i in range(len(gradient) - seglen):
        seg_sum = np.sum(np.abs(gradient[i:i + seglen]))
        if seg_sum > maxgradient:
            maxgradient = seg_sum
            idx_start, idx_end = i, i + seglen
    
    return idx_start, idx_end


####
# SG-CF: Shapelet-Guided Counterfactual Explanation for Time Series Classification
#
# Paper: https://ieeexplore.ieee.org/abstract/document/10020866/
# GitHub: https://github.com/Luckilyeee/SG-CF
#
# This is an optimization-based approach that uses shapelets (discriminative subsequences)
# to guide counterfactual generation for time series data.
####
def sg_cf(sample,
          model,
          shapelet=None,
          prototype=None,
          target=None,
          shapelet_start_idx=0,
          shapelet_end_idx=None,
          max_iter=1000,
          max_lambda_steps=10,
          lambda_init=0.1,
          learning_rate=0.1,
          segment_rate=0.05,
          target_proba=0.95,
          distance='l1',
          verbose=False):
    """
    Generate counterfactual explanation using Shapelet-Guided approach.
    
    Args:
        sample: Original time series sample to explain (numpy array)
        model: PyTorch model for classification
        shapelet: Discriminative shapelet from target class (numpy array)
        prototype: Prototype time series from target class (numpy array)
        target: Target class for counterfactual (int), if None uses second most likely
        shapelet_start_idx: Start index of shapelet in prototype (int)
        shapelet_end_idx: End index of shapelet in prototype (int)
        max_iter: Maximum iterations for optimization (int)
        max_lambda_steps: Maximum steps for lambda adjustment (int)
        lambda_init: Initial lambda value for loss weighting (float)
        learning_rate: Learning rate for Adam optimizer (float)
        segment_rate: Rate to determine CF segment length (float)
        target_proba: Target probability threshold (float)
        distance: Distance metric ('l1' or 'l2')
        verbose: Print debug information (bool)
    
    Returns:
        Counterfactual sample (numpy array) and prediction (numpy array)
    """
    device = next(model.parameters()).device
    
    def model_predict(data):
        """Ensure proper input format for model prediction."""
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
    
    # Convert inputs to tensors
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.reshape(1, 1, -1)
    elif len(sample_tensor.shape) == 2:
        if sample_tensor.shape[0] > sample_tensor.shape[1]:
            sample_tensor = sample_tensor.T
        sample_tensor = sample_tensor.unsqueeze(0)
    
    # Get original prediction
    y_orig = model_predict(sample)[0]
    label_orig = int(np.argmax(y_orig))
    
    # Determine target class
    if target is None:
        sorted_indices = np.argsort(y_orig)[::-1]
        target = int(sorted_indices[1])  # Second most likely class
    
    if verbose:
        print(f"SG-CF: Original class {label_orig}, Target class {target}")
    
    # Setup prototype
    if prototype is not None:
        proto_tensor = torch.tensor(prototype, dtype=torch.float32, device=device)
        if len(proto_tensor.shape) == 1:
            proto_tensor = proto_tensor.reshape(1, 1, -1)
        elif len(proto_tensor.shape) == 2:
            if proto_tensor.shape[0] > proto_tensor.shape[1]:
                proto_tensor = proto_tensor.T
            proto_tensor = proto_tensor.unsqueeze(0)
    else:
        proto_tensor = sample_tensor.clone()
    
    # Setup shapelet
    if shapelet is not None:
        shapelet_tensor = torch.tensor(shapelet, dtype=torch.float32, device=device)
        if len(shapelet_tensor.shape) == 1:
            shapelet_tensor = shapelet_tensor.reshape(1, 1, -1)
        elif len(shapelet_tensor.shape) == 2:
            shapelet_tensor = shapelet_tensor.unsqueeze(0)
        shapelet_len = shapelet_tensor.shape[2]
    else:
        shapelet_tensor = None
        shapelet_len = 0
    
    # Setup shapelet indices
    if shapelet_end_idx is None:
        shapelet_end_idx = shapelet_start_idx + shapelet_len if shapelet_len > 0 else sample_tensor.shape[2]
    
    # Distance function
    def dist_fn(x, y):
        if distance == 'l1':
            return manhattan_dist(x, y)
        else:
            return euclidean_dist(x, y)
    
    # Lambda sweep to find bounds
    n_orders = 10
    n_steps = max_iter // n_orders
    lambdas = np.array([lambda_init / (10 ** i) for i in range(n_orders)])
    
    if verbose:
        print(f"Lambda sweep: {lambdas}")
        print(f"Number of steps per lambda: {n_steps}")
    
    cf_count = np.zeros_like(lambdas)
    
    # Initialize counterfactual
    cf_tensor = sample_tensor.clone().detach()
    cf_tensor.requires_grad_(True)
    
    # Initial lambda sweep to find bounds
    for ix, lam_val in enumerate(lambdas):
        optimizer = Adam([cf_tensor], lr=learning_rate)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Compute loss components
            loss_dist_orig = dist_fn(cf_tensor, sample_tensor)
            loss_dist_proto = dist_fn(cf_tensor, proto_tensor)
            
            # Shapelet distance if provided
            if shapelet_tensor is not None:
                loss_shapelet = shapelet_distance(cf_tensor, shapelet_tensor)
            else:
                loss_shapelet = torch.tensor(0.0, device=device)
            
            # Prediction loss
            pred = model(cf_tensor)
            target_tensor = torch.tensor([target], dtype=torch.long, device=device)
            loss_pred = nn.CrossEntropyLoss()(pred, target_tensor)
            
            # Combined loss
            loss_total = lam_val * (loss_dist_orig + loss_dist_proto + loss_shapelet) + 2 * loss_pred ** 2
            
            loss_total.backward()
            optimizer.step()
        
        # Check if counterfactual condition is met
        y_cf = model_predict(cf_tensor.detach())[0]
        if np.argmax(y_cf) == target:
            cf_count[ix] += 1
    
    if verbose:
        print(f"CF count: {cf_count}")
    
    # Find lambda bounds
    try:
        lb_ix = np.where(cf_count > 0)[0][0]
    except IndexError:
        lb_ix = 0
    
    try:
        ub_ix = np.where(cf_count == 0)[0][-1]
    except IndexError:
        ub_ix = 0
    
    lam_lb = lambdas[lb_ix]
    lam_ub = lambdas[ub_ix] if ub_ix > 0 else lambda_init
    lam = (lam_lb + lam_ub) / 2
    
    if verbose:
        print(f"Lambda bounds: [{lam_lb}, {lam_ub}], starting at {lam}")
    
    # Main optimization loop with segment guidance
    rate = segment_rate
    best_cf = None
    best_pred = None
    best_validity = 0.0
    
    while rate < 0.7:
        cf_tensor = sample_tensor.clone().detach()
        cf_tensor.requires_grad_(True)
        
        seg_len = int(np.round(rate * cf_tensor.shape[2], decimals=0))
        
        if verbose:
            print(f"Segment rate: {rate:.2f}, Segment length: {seg_len}")
        
        for l_step in range(max_lambda_steps):
            optimizer = Adam([cf_tensor], lr=learning_rate)
            
            found = 0
            not_found = 0
            
            for i in range(max_iter):
                optimizer.zero_grad()
                
                # Compute gradients
                pred = model(cf_tensor)
                target_tensor = torch.tensor([target], dtype=torch.long, device=device)
                
                loss_dist_orig = dist_fn(cf_tensor, sample_tensor)
                loss_dist_proto = dist_fn(cf_tensor, proto_tensor)
                
                if shapelet_tensor is not None:
                    loss_shapelet = shapelet_distance(cf_tensor, shapelet_tensor)
                else:
                    loss_shapelet = torch.tensor(0.0, device=device)
                
                loss_pred = nn.CrossEntropyLoss()(pred, target_tensor)
                loss_total = lam * (loss_dist_orig + loss_dist_proto + loss_shapelet) + 2 * loss_pred ** 2
                
                loss_total.backward()
                
                # Apply shapelet-guided gradient masking
                if i == 0 and l_step == 0 and shapelet_tensor is not None:
                    # Determine gradient-based prominent segment on first iteration
                    gradient = cf_tensor.grad.squeeze().cpu().detach().numpy()
                    if len(gradient.shape) > 1:
                        gradient = gradient.flatten()
                    
                    # Use provided shapelet indices or find prominent segment
                    idx_start = shapelet_start_idx
                    idx_end = shapelet_end_idx
                
                # Mask gradients to focus on shapelet region
                if shapelet_tensor is not None:
                    grad_mask = torch.zeros_like(cf_tensor.grad)
                    grad_mask[:, :, idx_start:idx_end] = cf_tensor.grad[:, :, idx_start:idx_end]
                    cf_tensor.grad = grad_mask
                
                optimizer.step()
                
                # Check counterfactual condition
                with torch.no_grad():
                    y_cf = model_predict(cf_tensor)[0]
                    current_class = np.argmax(y_cf)
                    current_validity = y_cf[target]
                    
                    if current_class == target:
                        found += 1
                        not_found = 0
                        
                        if current_validity > best_validity:
                            best_validity = current_validity
                            best_cf = cf_tensor.clone().detach()
                            best_pred = y_cf
                    else:
                        found = 0
                        not_found += 1
                    
                    # Early stopping
                    if found >= 50 or not_found >= 50:
                        break
            
            # Check if target probability reached
            if best_validity >= target_proba:
                if verbose:
                    print(f"SG-CF: Found counterfactual with validity {best_validity:.4f}")
                break
            
            # Adjust lambda (bisection)
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
        
        # Check if we reached target probability
        if best_validity >= target_proba:
            break
        
        # Increase segment rate
        rate += 0.01
    
    if best_cf is None:
        if verbose:
            print(f"SG-CF: Could not find valid counterfactual. Best validity: {best_validity:.4f}")
        return None, None
    
    # Convert back to original sample format
    cf_result = detach_to_numpy(best_cf.squeeze(0))
    if len(sample.shape) == 1:
        cf_result = cf_result.squeeze()
    elif len(sample.shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            cf_result = cf_result.T
    
    if verbose:
        print(f"SG-CF: Final validity {best_validity:.4f}, Target class probability {best_pred[target]:.4f}")
    
    return cf_result, best_pred
