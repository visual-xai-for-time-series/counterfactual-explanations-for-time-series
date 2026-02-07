import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Union, Dict, Any


####
# TSCF: Time Series CounterFactuals
#
# This is a custom gradient-based optimization method with temporal smoothness
# constraints designed to generate realistic counterfactual explanations for
# time series classification.
#
# The method combines standard counterfactual generation techniques with
# time series-specific regularization (smoothness, sparsity) to maintain
# temporal coherence in generated counterfactuals.
####


def detach_to_numpy(data):
    """Move PyTorch data to CPU and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to PyTorch tensor and move it to device."""
    return torch.from_numpy(data).float().to(device)


def tscf_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    lambda_l1: float = 0.01,
    lambda_l2: float = 0.01,
    lambda_smooth: float = 0.001,
    learning_rate: float = 0.1,
    max_iterations: int = 2000,
    tolerance: float = 1e-5,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual explanation using TSCF (Time Series CounterFactuals) algorithm.
    
    TSCF uses gradient-based optimization with temporal smoothness constraints
    to generate realistic counterfactual explanations for time series classification.
    
    Args:
        sample: Original time series sample (can be 1D, 2D, or 3D)
        dataset: Dataset object (for compatibility with other methods)
        model: Trained classification model
        target_class: Target class for counterfactual (if None, finds different class)
        lambda_l1: L1 regularization weight for sparsity
        lambda_l2: L2 regularization weight for proximity
        lambda_smooth: Temporal smoothness regularization weight
        learning_rate: Learning rate for optimization
        max_iterations: Maximum number of optimization iterations
        tolerance: Convergence tolerance for early stopping
        device: Device to run on (if None, auto-detects)
        verbose: Print debug information during optimization
        
    Returns:
        Tuple of (counterfactual_sample, prediction_scores) or (None, None) if failed
        - counterfactual_sample: Shape matches input sample
        - prediction_scores: Class probabilities for the counterfactual
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Store original sample shape for later restoration
    original_shape = sample.shape
    original_sample = np.asarray(sample)
    
    # Convert sample to tensor and ensure proper shape (batch, channels, length)
    x_tensor = torch.tensor(original_sample, dtype=torch.float32, device=device)
    
    # Normalize shape to (1, C, L)
    if len(x_tensor.shape) == 1:
        # (L,) -> (1, 1, L)
        x_tensor = x_tensor.reshape(1, 1, -1)
    elif len(x_tensor.shape) == 2:
        # Could be (C, L) or (L, C)
        if x_tensor.shape[0] > x_tensor.shape[1]:
            # Likely (L, C), transpose to (C, L)
            x_tensor = x_tensor.T
        # Add batch dimension: (C, L) -> (1, C, L)
        x_tensor = x_tensor.unsqueeze(0)
    elif len(x_tensor.shape) == 3:
        # Already has batch dimension
        pass
    else:
        raise ValueError(f"Unsupported sample shape: {original_shape}")
    
    # Get original prediction
    with torch.no_grad():
        original_logits = model(x_tensor)
        original_probs = torch.softmax(original_logits, dim=-1)
        original_class = torch.argmax(original_probs, dim=-1).item()
        original_probs_np = original_probs.squeeze().cpu().numpy()
    
    if verbose:
        print(f"TSCF: Original class = {original_class}, probabilities = {original_probs_np}")
    
    # Determine target class
    if target_class is None:
        # Find the class with second highest probability
        sorted_classes = torch.argsort(original_probs, dim=-1, descending=True)
        target_class = sorted_classes[0, 1].item()
    
    # If already in target class, return None
    if original_class == target_class:
        if verbose:
            print(f"TSCF: Sample already in target class {target_class}")
        return None, None
    
    if verbose:
        print(f"TSCF: Target class = {target_class}")
    
    # Initialize counterfactual as copy of original with gradient tracking
    x_cf = x_tensor.clone().detach().requires_grad_(True)
    
    # Optimizer
    optimizer = optim.Adam([x_cf], lr=learning_rate)
    
    # Track best solution
    best_cf = None
    best_loss = float('inf')
    best_validity = 0.0
    prev_loss = float('inf')
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(x_cf)
        probs = torch.softmax(logits, dim=-1)
        
        # Classification loss - cross entropy to target class
        target_tensor = torch.tensor([target_class], dtype=torch.long, device=device)
        cls_loss = nn.functional.cross_entropy(logits, target_tensor)
        
        # Proximity losses
        l1_loss = torch.norm(x_cf - x_tensor, p=1)
        l2_loss = torch.norm(x_cf - x_tensor, p=2)
        
        # Temporal smoothness loss - encourage smooth changes over time
        # Compute differences between consecutive time points
        diff = x_cf[:, :, 1:] - x_cf[:, :, :-1]
        smoothness_loss = torch.norm(diff, p=2)
        
        # Total loss
        total_loss = (
            cls_loss + 
            lambda_l1 * l1_loss + 
            lambda_l2 * l2_loss + 
            lambda_smooth * smoothness_loss
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Evaluate current solution
        with torch.no_grad():
            current_probs = torch.softmax(model(x_cf), dim=-1)
            current_class = torch.argmax(current_probs, dim=-1).item()
            current_validity = current_probs[0, target_class].item()
        
        # Update best solution
        if current_class == target_class:
            # Valid counterfactual found - prefer lower loss
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_validity = current_validity
                best_cf = x_cf.clone().detach()
                if verbose and iteration % 100 == 0:
                    print(f"TSCF: New best at iter {iteration}: "
                          f"validity={current_validity:.4f}, loss={total_loss.item():.4f}")
        elif best_cf is None and current_validity > best_validity:
            # No valid solution yet - track highest validity
            best_validity = current_validity
            best_cf = x_cf.clone().detach()
        
        # Verbose output
        if verbose and iteration % 500 == 0:
            print(f"TSCF iter {iteration}: loss={total_loss.item():.4f}, "
                  f"cls={cls_loss.item():.4f}, l2={l2_loss.item():.4f}, "
                  f"validity={current_validity:.4f}, pred_class={current_class}")
        
        # Early stopping - achieved high validity
        if current_validity > 0.95 and current_class == target_class:
            if verbose:
                print(f"TSCF: Early stop at iteration {iteration} with validity {current_validity:.4f}")
            break
        
        # Convergence check
        if abs(prev_loss - total_loss.item()) < tolerance:
            if verbose:
                print(f"TSCF: Converged at iteration {iteration}")
            break
        
        prev_loss = total_loss.item()
    
    # If no valid counterfactual found
    if best_cf is None:
        if verbose:
            print("TSCF: Failed to find valid counterfactual")
        return None, None
    
    # Get final prediction for best counterfactual
    with torch.no_grad():
        final_logits = model(best_cf)
        final_probs = torch.softmax(final_logits, dim=-1)
        final_probs_np = final_probs.squeeze().cpu().numpy()
        final_class = torch.argmax(final_probs, dim=-1).item()
    
    if verbose:
        print(f"TSCF: Final class = {final_class}, probabilities = {final_probs_np}")
    
    # Convert back to numpy and restore original shape
    cf_np = detach_to_numpy(best_cf)
    
    # Restore original shape
    if len(original_shape) == 1:
        # Was 1D (L,)
        cf_np = cf_np.squeeze()
    elif len(original_shape) == 2:
        # Was 2D - need to determine if it was (C, L) or (L, C)
        cf_np = cf_np.squeeze(0)  # Remove batch dimension -> (C, L)
        if original_shape[0] > original_shape[1]:
            # Original was (L, C), so transpose back
            cf_np = cf_np.T
    elif len(original_shape) == 3:
        # Was already 3D, keep batch dimension if original had it
        pass
    
    return cf_np, final_probs_np


def tscf_batch_cf(
    samples: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    lambda_l1: float = 0.01,
    lambda_l2: float = 0.01,
    lambda_smooth: float = 0.001,
    learning_rate: float = 0.1,
    max_iterations: int = 2000,
    tolerance: float = 1e-5,
    device: str = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate counterfactual explanations for multiple samples.
    
    Args:
        samples: Multiple time series samples, shape (N, ...) where ... is the time series shape
        dataset: Dataset object
        model: Trained classification model
        target_class: Target class for all counterfactuals
        lambda_l1: L1 regularization weight
        lambda_l2: L2 regularization weight
        lambda_smooth: Temporal smoothness weight
        learning_rate: Learning rate for optimization
        max_iterations: Maximum iterations per sample
        tolerance: Convergence tolerance
        device: Device to run on
        verbose: Print debug information
        
    Returns:
        Tuple of (counterfactuals, predictions) for all samples
    """
    counterfactuals = []
    predictions = []
    
    for i, sample in enumerate(samples):
        if verbose:
            print(f"\n--- Processing sample {i+1}/{len(samples)} ---")
        
        cf, pred = tscf_cf(
            sample=sample,
            dataset=dataset,
            model=model,
            target_class=target_class,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_smooth=lambda_smooth,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            device=device,
            verbose=verbose
        )
        
        if cf is not None:
            counterfactuals.append(cf)
            predictions.append(pred)
        else:
            # Use original sample if no counterfactual found
            counterfactuals.append(sample)
            predictions.append(None)
    
    return np.array(counterfactuals), predictions
