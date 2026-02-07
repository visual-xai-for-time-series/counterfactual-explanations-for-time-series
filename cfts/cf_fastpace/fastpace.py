import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional, Tuple, List, Dict
from scipy.spatial.distance import cdist


####
# FastPACE: Fast Planning-based Counterfactual Explanations
#
# This is a custom planning-based counterfactual generation method that uses
# feature importance to identify key time steps and plans gradual changes
# towards target class examples with plausibility constraints.
#
# The method combines gradient-based feature importance with instance-based
# planning to generate realistic and interpretable counterfactuals.
####


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


def euclidean_dist(x, y):
    """Compute Euclidean distance between two tensors."""
    return torch.sqrt(torch.sum((x - y) ** 2))


def compute_feature_importance(sample_t, model, target_class, device):
    """
    Compute feature importance using gradient-based saliency.
    This helps identify which time steps are most important for planning changes.
    """
    sample_t.requires_grad_(True)
    pred = model(sample_t)
    
    # Get gradient w.r.t. target class
    pred[0, target_class].backward()
    
    # Use absolute gradient as importance score
    importance = torch.abs(sample_t.grad).detach()
    sample_t.grad = None
    sample_t.requires_grad_(False)
    
    return importance


def plan_interventions(original, target_examples, importance, n_steps=10, step_size=0.5):
    """
    Plan a sequence of interventions (changes) to transform the original time series.
    
    This is the core "planning" component of FastPACE:
    - Identifies key time steps to modify based on feature importance
    - Plans gradual changes towards target class examples
    - Ensures changes are feasible and smooth
    
    Args:
        original: Original time series tensor
        target_examples: Examples from target class for guidance
        importance: Feature importance scores
        n_steps: Number of planning steps
        step_size: Size of each intervention step
        
    Returns:
        List of planned interventions (time_step, change_direction, magnitude)
    """
    # Flatten importance for easier processing
    importance_flat = importance.flatten()
    original_flat = original.flatten()
    
    # Find most important time steps
    _, top_indices = torch.topk(importance_flat, min(len(importance_flat), n_steps * 2))
    
    # Compute target direction from target class examples
    if len(target_examples) > 0:
        target_mean = torch.mean(target_examples, dim=0).flatten()
        direction = target_mean - original_flat
    else:
        # Fallback: use gradient direction
        direction = importance_flat * torch.sign(torch.randn_like(importance_flat))
    
    # Plan interventions at most important time steps
    interventions = []
    for idx in top_indices[:n_steps]:
        idx_int = int(idx.item())
        change_direction = float(torch.sign(direction[idx_int]).item())
        magnitude = float(min(abs(direction[idx_int].item()), step_size))
        interventions.append((idx_int, change_direction, magnitude))
    
    return interventions


def apply_interventions(sample_t, interventions, alpha=1.0):
    """
    Apply planned interventions to the time series.
    
    Args:
        sample_t: Time series tensor
        interventions: List of (time_step, direction, magnitude) tuples
        alpha: Scaling factor for intervention strength
        
    Returns:
        Modified time series with interventions applied
    """
    modified = sample_t.clone()
    modified_flat = modified.flatten()
    
    for time_step, direction, magnitude in interventions:
        modified_flat[time_step] += alpha * direction * magnitude
    
    return modified_flat.reshape(sample_t.shape)


def check_plausibility(cf_sample, reference_samples, threshold=2.0):
    """
    Check if counterfactual is plausible based on reference samples.
    Uses statistical distance to determine if CF is within reasonable bounds.
    
    Args:
        cf_sample: Counterfactual sample (numpy array)
        reference_samples: Reference samples from target class (numpy array)
        threshold: Standard deviation threshold for plausibility
        
    Returns:
        plausibility_score: Score between 0 and 1 (higher is more plausible)
    """
    if len(reference_samples) == 0:
        return 1.0  # No reference, assume plausible
    
    # Compute mean and std of reference samples
    ref_mean = np.mean(reference_samples, axis=0)
    ref_std = np.std(reference_samples, axis=0) + 1e-8
    
    # Check how many standard deviations away CF is
    z_scores = np.abs((cf_sample.flatten() - ref_mean.flatten()) / ref_std.flatten())
    mean_z = np.mean(z_scores)
    
    # Convert to plausibility score (higher is better)
    plausibility = 1.0 / (1.0 + mean_z / threshold)
    
    return float(plausibility)


####
# FastPACE: Fast PlAnning of Counterfactual Explanations
#
# A planning-based approach for generating counterfactual explanations:
# 1. Identifies important features using gradient-based saliency
# 2. Plans a sequence of feasible interventions
# 3. Applies interventions iteratively with plausibility checking
# 4. Refines counterfactual through gradient-based optimization
#
# Key differences from pure gradient methods:
# - Explicit planning of actionable changes
# - Plausibility constraints based on target class distribution
# - Sequential application of interventions
# - Combines planning with optimization for efficiency
#
# This implementation follows the style of wachter.py while adding
# planning-based intervention strategies.
####
def fastpace_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target: Optional[int] = None,
    n_planning_steps: int = 10,
    intervention_step_size: float = 0.3,
    lambda_proximity: float = 1.0,
    lambda_plausibility: float = 0.5,
    max_refinement_iterations: int = 500,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual explanation using FastPACE algorithm.
    
    FastPACE (Fast PlAnning of Counterfactual Explanations) uses a planning-based
    approach to generate counterfactuals through feasible interventions:
    
    1. Identify important features via gradient-based saliency
    2. Plan a sequence of interventions towards target class
    3. Apply interventions with plausibility checking
    4. Refine with gradient-based optimization
    
    Args:
        sample: Original time series sample (can be 1D, 2D, or 3D)
            - 1D: (length,)
            - 2D: (channels, length) or (length, channels)
            - 3D: (batch, channels, length)
        dataset: Dataset object containing training samples for planning
        model: Trained PyTorch classification model
        target: Target class for counterfactual (if None, chooses second most likely class)
        n_planning_steps: Number of interventions to plan (default: 10)
        intervention_step_size: Size of each planned intervention (default: 0.3)
        lambda_proximity: Weight for proximity to original (default: 1.0)
        lambda_plausibility: Weight for plausibility constraint (default: 0.5)
        max_refinement_iterations: Max iterations for refinement phase (default: 500)
        learning_rate: Learning rate for refinement optimizer (default: 0.01)
        tolerance: Convergence tolerance for refinement (default: 1e-6)
        verbose: Whether to print progress information
        
    Returns:
        counterfactual: Generated counterfactual in same shape as input
        cf_prediction: Model prediction probabilities for the counterfactual
        
    Example:
        >>> cf, pred = fastpace_cf(sample, dataset, model, target=1, verbose=True)
        >>> print(f"Counterfactual class: {np.argmax(pred)}")
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Store original shape for output conversion
    original_shape = sample.shape
    
    # Convert sample to proper tensor format: (batch, channels, length)
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.reshape(1, 1, -1)  # (length,) -> (1, 1, length)
    elif len(sample_tensor.shape) == 2:
        # Assume (channels, length) if first dim is smaller, else (length, channels)
        if sample_tensor.shape[0] > sample_tensor.shape[1]:
            sample_tensor = sample_tensor.T
        sample_tensor = sample_tensor.unsqueeze(0)  # Add batch dimension
    
    sample_t = sample_tensor.clone()
    
    # Get initial prediction
    with torch.no_grad():
        y_orig = model(sample_t)
        y_orig_np = detach_to_numpy(y_orig)[0]
        original_class = int(np.argmax(y_orig_np))
    
    # Determine target class
    if target is None:
        # Find the class with second highest probability
        sorted_indices = np.argsort(y_orig_np)[::-1]
        target = int(sorted_indices[1])
    
    if verbose:
        print(f"FastPACE: Original class {original_class}, Target class {target}")
        print(f"FastPACE: Planning {n_planning_steps} interventions")
    
    # Phase 1: PLANNING - Identify important features and plan interventions
    if verbose:
        print("FastPACE: Phase 1 - Planning interventions...")
    
    importance = compute_feature_importance(sample_t.clone(), model, target, device)
    
    # Gather target class examples for planning and plausibility
    target_examples = []
    target_examples_np = []
    for i in range(min(len(dataset), 200)):  # Limit search
        x, y = dataset[i]
        y_class = np.argmax(y) if hasattr(y, 'shape') and len(y.shape) > 0 and y.shape[0] > 1 else (int(y) if not hasattr(y, 'shape') else int(y.item()) if y.shape == () else int(y[0]))
        if y_class == target:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
            if len(x_tensor.shape) == 1:
                x_tensor = x_tensor.reshape(1, 1, -1)
            elif len(x_tensor.shape) == 2:
                if x_tensor.shape[0] > x_tensor.shape[1]:
                    x_tensor = x_tensor.T
                x_tensor = x_tensor.unsqueeze(0)
            target_examples.append(x_tensor)
            target_examples_np.append(x_tensor.cpu().numpy())
        if len(target_examples) >= 30:  # Collect up to 30 examples
            break
    
    if len(target_examples) > 0:
        target_examples_tensor = torch.cat(target_examples, dim=0)
    else:
        target_examples_tensor = torch.empty(0)
    
    target_examples_np = np.array(target_examples_np).squeeze() if target_examples_np else np.array([])
    
    # Plan interventions
    interventions = plan_interventions(
        sample_t, 
        target_examples_tensor, 
        importance, 
        n_steps=n_planning_steps,
        step_size=intervention_step_size
    )
    
    if verbose:
        print(f"FastPACE: Planned {len(interventions)} interventions at key time steps")
    
    # Phase 2: INTERVENTION - Apply planned changes iteratively
    if verbose:
        print("FastPACE: Phase 2 - Applying planned interventions...")
    
    cf_candidate = sample_t.clone()
    best_cf = None
    best_validity = 0.0
    
    # Apply interventions gradually with plausibility checking
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        cf_test = apply_interventions(sample_t, interventions, alpha=alpha)
        
        with torch.no_grad():
            pred = model(cf_test)
            pred_np = detach_to_numpy(pred)[0]
            pred_class = int(np.argmax(pred_np))
            validity = pred_np[target]
        
        # Check plausibility
        cf_test_np = detach_to_numpy(cf_test.squeeze(0))
        plausibility = check_plausibility(cf_test_np, target_examples_np)
        
        if verbose:
            print(f"  α={alpha:.2f}: class={pred_class}, validity={validity:.4f}, plausibility={plausibility:.4f}")
        
        # Update best candidate
        if validity > best_validity and plausibility > 0.3:  # Require minimum plausibility
            best_validity = validity
            best_cf = cf_test.clone()
        
        # Early exit if target achieved
        if pred_class == target and plausibility > 0.5:
            if verbose:
                print(f"FastPACE: Target achieved at α={alpha:.2f}")
            cf_candidate = cf_test
            break
    else:
        # Use best candidate if target not achieved
        if best_cf is not None:
            cf_candidate = best_cf
    
    # Phase 3: REFINEMENT - Gradient-based optimization for fine-tuning
    if verbose:
        print("FastPACE: Phase 3 - Refining counterfactual...")
    
    cf_candidate.requires_grad_(True)
    optimizer = Adam([cf_candidate], lr=learning_rate)
    
    ce_loss = nn.CrossEntropyLoss()
    target_t = torch.tensor([target], dtype=torch.long, device=device)
    
    prev_loss = float('inf')
    refinement_improved = False
    
    for iteration in range(max_refinement_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(cf_candidate)
        
        # Combined loss: classification + proximity + plausibility
        cls_loss = ce_loss(pred, target_t)
        prox_loss = euclidean_dist(sample_t, cf_candidate)
        
        # Plausibility loss: distance to nearest target class example
        if len(target_examples_tensor) > 0:
            distances = torch.stack([
                euclidean_dist(cf_candidate, ex) 
                for ex in target_examples_tensor
            ])
            plaus_loss = torch.min(distances)
        else:
            plaus_loss = torch.tensor(0.0, device=device)
        
        total_loss = cls_loss + lambda_proximity * prox_loss + lambda_plausibility * plaus_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Check for improvement
        with torch.no_grad():
            y_cf = detach_to_numpy(model(cf_candidate))[0]
            current_class = int(np.argmax(y_cf))
            current_validity = y_cf[target]
        
        if current_validity > best_validity:
            best_validity = current_validity
            refinement_improved = True
        
        # Verbose output
        if verbose and iteration % 100 == 0:
            print(f"  iter {iteration}: class={current_class}, validity={current_validity:.4f}, loss={total_loss.item():.4f}")
        
        # Early stopping
        if current_class == target:
            if verbose:
                print(f"FastPACE: Refinement achieved target at iteration {iteration}")
            break
        
        # Convergence check
        if abs(prev_loss - total_loss.item()) < tolerance:
            break
        
        prev_loss = total_loss.item()
    
    # Final evaluation
    with torch.no_grad():
        final_pred = model(cf_candidate)
        final_pred_np = detach_to_numpy(final_pred)[0]
        final_class = int(np.argmax(final_pred_np))
    
    if verbose:
        print(f"FastPACE: Final class={final_class}, validity={best_validity:.4f}, refinement_improved={refinement_improved}")
    
    # Convert back to original shape
    best_cf_np = detach_to_numpy(cf_candidate.squeeze(0))  # Remove batch dimension
    
    if len(original_shape) == 1:
        # Convert back to 1D
        best_cf_np = best_cf_np.squeeze()
    elif len(original_shape) == 2:
        # Convert back to original 2D orientation
        if original_shape[0] > original_shape[1]:
            # Was (length, channels), need to transpose back
            best_cf_np = best_cf_np.T
    
    return best_cf_np, final_pred_np


def fastpace_batch_cf(
    samples: np.ndarray,
    dataset,
    model: nn.Module,
    target: Optional[int] = None,
    n_planning_steps: int = 10,
    intervention_step_size: float = 0.3,
    lambda_proximity: float = 1.0,
    lambda_plausibility: float = 0.5,
    max_refinement_iterations: int = 500,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate counterfactual explanations for multiple samples using FastPACE.
    
    Args:
        samples: Multiple time series samples, shape (N, ...) where ... is the time series shape
        dataset: Dataset object containing training samples
        model: Trained PyTorch classification model
        target: Target class for all counterfactuals (if None, computed per sample)
        n_planning_steps: Number of interventions to plan
        intervention_step_size: Size of each planned intervention
        lambda_proximity: Weight for proximity to original
        lambda_plausibility: Weight for plausibility constraint
        max_refinement_iterations: Max iterations for refinement phase
        learning_rate: Learning rate for refinement optimizer
        tolerance: Convergence tolerance
        verbose: Whether to print progress information
        
    Returns:
        counterfactuals: Generated counterfactuals with same shape as input
        predictions: Model predictions for all counterfactuals
    """
    n_samples = len(samples)
    counterfactuals = []
    predictions = []
    
    if verbose:
        print(f"FastPACE Batch: Generating counterfactuals for {n_samples} samples")
    
    for i, sample in enumerate(samples):
        if verbose:
            print(f"\n--- Sample {i+1}/{n_samples} ---")
        
        cf, pred = fastpace_cf(
            sample=sample,
            dataset=dataset,
            model=model,
            target=target,
            n_planning_steps=n_planning_steps,
            intervention_step_size=intervention_step_size,
            lambda_proximity=lambda_proximity,
            lambda_plausibility=lambda_plausibility,
            max_refinement_iterations=max_refinement_iterations,
            learning_rate=learning_rate,
            tolerance=tolerance,
            verbose=verbose
        )
        
        if cf is not None:
            counterfactuals.append(cf)
            predictions.append(pred)
        else:
            # If counterfactual generation failed, return original
            counterfactuals.append(sample)
            device = next(model.parameters()).device
            sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
            if len(sample_tensor.shape) == 1:
                sample_tensor = sample_tensor.reshape(1, 1, -1)
            elif len(sample_tensor.shape) == 2:
                if sample_tensor.shape[0] > sample_tensor.shape[1]:
                    sample_tensor = sample_tensor.T
                sample_tensor = sample_tensor.unsqueeze(0)
            with torch.no_grad():
                pred = detach_to_numpy(model(sample_tensor))[0]
            predictions.append(pred)
    
    return np.array(counterfactuals), np.array(predictions)
