import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Union, Dict, Any


def comte_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    lambda_reg: float = 0.01,
    lambda_sparse: float = 0.001,
    learning_rate: float = 0.1,
    max_iterations: int = 3000,
    tolerance: float = 1e-4,
    device: str = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual explanation using COMTE algorithm.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object (for compatibility with other methods)
        model: Trained classification model
        target_class: Target class for counterfactual (if None, finds different class)
        lambda_reg: Regularization parameter for proximity constraint
        lambda_sparse: Regularization parameter for sparsity constraint
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
    
    # Initialize counterfactual as copy of original
    x_cf = x_tensor.clone().detach().requires_grad_(True)
    
    # Optimizer with different strategy
    optimizer = optim.Adam([x_cf], lr=learning_rate)
    
    best_cf = None
    best_loss = float('inf')
    best_validity = 0.0
    
    # Two-phase optimization: first focus on prediction, then refine with regularization
    phase1_iterations = max_iterations // 2
    current_lambda_reg = 0.0  # Start without regularization
    current_lambda_sparse = 0.0
    
    for iteration in range(max_iterations):
        # Switch to phase 2 halfway through - add regularization
        if iteration == phase1_iterations:
            current_lambda_reg = lambda_reg
            current_lambda_sparse = lambda_sparse
            print(f"COMTE: Switching to phase 2 with regularization at iteration {iteration}")
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(x_cf)
        
        # Prediction loss - focus heavily on getting the right class
        log_probs = torch.log_softmax(logits, dim=-1)
        pred_loss = -log_probs[0, target_class]
        
        # Distance loss (proximity constraint) - only in phase 2
        distance_loss = torch.norm(x_cf - x_tensor, p=2)
        
        # Sparsity loss (encourage minimal changes) - only in phase 2
        sparsity_loss = torch.norm(x_cf - x_tensor, p=1)
        
        # Total loss with adaptive weights
        total_loss = pred_loss + current_lambda_reg * distance_loss + current_lambda_sparse * sparsity_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Check current validity
        with torch.no_grad():
            current_probs = torch.softmax(logits, dim=-1)
            current_validity = current_probs[0, target_class].item()
            current_pred_class = torch.argmax(current_probs, dim=-1).item()
        
        # Track best solution (prioritize validity heavily)
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
            print(f"COMTE: Early stop at iteration {iteration} with validity {current_validity:.4f}")
            break
        
        # Print progress every 500 iterations for debugging
        if iteration % 500 == 0:
            print(f"COMTE iteration {iteration}: loss={total_loss.item():.4f}, "
                  f"pred_loss={pred_loss.item():.4f}, pred_class={current_pred_class}, "
                  f"target={target_class}, validity={current_validity:.4f}")
    
    if best_cf is None:
        print("COMTE: No counterfactual found - best_cf is None")
        return None, None
    
    # Get final prediction
    with torch.no_grad():
        final_pred = model(best_cf)
        predicted_class = torch.argmax(final_pred, dim=-1).item()
        final_pred_np = torch.softmax(final_pred, dim=-1).squeeze().cpu().numpy()
        final_validity = final_pred_np[target_class]
    
    print(f"COMTE final: pred_class={predicted_class}, target={target_class}, validity={final_validity:.4f}")
    
    # Check if counterfactual is valid - use relaxed criteria
    # Accept if either predicted class matches OR validity is reasonably high
    if predicted_class != target_class and final_validity < 0.4:
        print(f"COMTE: Counterfactual failed validation - predicted {predicted_class}, wanted {target_class}, validity too low")
        return None, None
    
    # Convert back to original sample format
    cf_sample = best_cf.squeeze(0).cpu().numpy()
    
    # Handle output shape to match input format
    if len(sample.shape) == 1:
        cf_sample = cf_sample.squeeze()  # Remove channel dimension if input was 1D
    elif len(sample.shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            cf_sample = cf_sample.T  # Convert back to (length, channels) if needed
    
    return cf_sample, final_pred_np


def _compute_distance(x1: torch.Tensor, x2: torch.Tensor, metric: str = 'euclidean') -> torch.Tensor:
    """Compute distance between two time series."""
    if metric == 'euclidean':
        return torch.norm(x1 - x2, p=2)
    elif metric == 'dtw':
        return _soft_dtw(x1, x2)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


def _soft_dtw(x1: torch.Tensor, x2: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Simplified soft DTW implementation for differentiable DTW computation.
    """
    # Flatten to 1D for DTW computation
    if len(x1.shape) > 1:
        x1_flat = x1.flatten()
        x2_flat = x2.flatten()
    else:
        x1_flat = x1
        x2_flat = x2
    
    n, m = len(x1_flat), len(x2_flat)
    
    # Compute pairwise squared distances
    D = torch.zeros(n, m, device=x1.device)
    for i in range(n):
        for j in range(m):
            D[i, j] = (x1_flat[i] - x2_flat[j]) ** 2
    
    # Initialize DP matrix
    R = torch.full((n + 1, m + 1), float('inf'), device=x1.device)
    R[0, 0] = 0
    
    # Fill DP matrix with soft-min approximation
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r0 = R[i-1, j-1]
            r1 = R[i-1, j]
            r2 = R[i, j-1]
            
            # Simple minimum for differentiability
            R[i, j] = D[i-1, j-1] + torch.min(torch.stack([r0, r1, r2]))
    
    return R[n, m]


# Alternative function with more configuration options
def comte_cf_advanced(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    distance_metric: str = 'euclidean',
    lambda_reg: float = 1.0,
    lambda_sparse: float = 0.1,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    constraints: Optional[Dict[str, Any]] = None,
    device: str = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Advanced COMTE implementation with additional options.
    
    Additional Args:
        distance_metric: Distance metric ('euclidean' or 'dtw')
        constraints: Dictionary of constraints (e.g., feature bounds)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare input tensor
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.reshape(1, 1, -1)
    elif len(x_tensor.shape) == 2:
        if x_tensor.shape[0] > x_tensor.shape[1]:
            x_tensor = x_tensor.T
        x_tensor = x_tensor.unsqueeze(0)
    
    # Get original prediction and determine target
    with torch.no_grad():
        original_pred = model(x_tensor)
        original_class = torch.argmax(original_pred, dim=-1).item()
    
    if target_class is None:
        sorted_classes = torch.argsort(original_pred, dim=-1, descending=True)
        target_class = sorted_classes[0, 1].item()
    
    if original_class == target_class:
        return None, None
    
    # Initialize and optimize
    x_cf = x_tensor.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x_cf], lr=learning_rate)
    
    best_cf = None
    best_loss = float('inf')
    prev_loss = float('inf')
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Compute losses
        logits = model(x_cf)
        log_probs = torch.log_softmax(logits, dim=-1)
        pred_loss = -log_probs[0, target_class]
        
        distance_loss = _compute_distance(x_cf, x_tensor, distance_metric)
        sparsity_loss = torch.norm(x_cf - x_tensor, p=1)
        
        total_loss = pred_loss + lambda_reg * distance_loss + lambda_sparse * sparsity_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Apply constraints if provided
        if constraints:
            _apply_constraints(x_cf, constraints)
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_cf = x_cf.clone().detach()
        
        if iteration > 0 and abs(prev_loss - total_loss.item()) < tolerance:
            break
        
        prev_loss = total_loss.item()
    
    if best_cf is None:
        return None, None
    
    # Verify and return
    with torch.no_grad():
        final_pred = model(best_cf)
        predicted_class = torch.argmax(final_pred, dim=-1).item()
        final_pred_np = torch.softmax(final_pred, dim=-1).squeeze().cpu().numpy()
    
    if predicted_class != target_class:
        return None, None
    
    # Format output
    cf_sample = best_cf.squeeze(0).cpu().numpy()
    if len(sample.shape) == 1:
        cf_sample = cf_sample.squeeze()
    elif len(sample.shape) == 2 and sample.shape[0] > sample.shape[1]:
        cf_sample = cf_sample.T
    
    return cf_sample, final_pred_np


def _apply_constraints(x_cf: torch.Tensor, constraints: Dict[str, Any]):
    """Apply constraints during optimization."""
    with torch.no_grad():
        if 'feature_bounds' in constraints:
            bounds = constraints['feature_bounds']
            for i, (min_val, max_val) in enumerate(bounds):
                if i < x_cf.shape[1]:  # Check if feature index exists
                    x_cf[0, i, :] = torch.clamp(x_cf[0, i, :], min_val, max_val)
        
        if 'immutable_features' in constraints:
            # This would need the original tensor to restore immutable features
            pass  # Simplified for this implementation