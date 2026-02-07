import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


def detach_to_numpy(data):
    """Move PyTorch data to CPU and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to PyTorch tensor and move it to device."""
    return torch.from_numpy(data).float().to(device)


####
# TimeX: Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency
# 
# Paper: https://arxiv.org/abs/2306.02109
# Repository: https://github.com/mims-harvard/TimeX
#
# TimeX is a time series explainer that learns interpretable surrogate models
# through self-supervised model behavior consistency. It generates saliency-based
# explanations by learning which time points are most important for the model's
# predictions via a mask generator and transformer-based architecture.
#
# This implementation provides a simplified wrapper for generating explanations
# from pre-trained TimeX models in the same style as other counterfactual methods.
####


def timex_explanation(
    sample: np.ndarray,
    model: nn.Module,
    timex_model: Optional[nn.Module] = None,
    return_saliency: bool = True,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate TimeX saliency-based explanation for a time series sample.
    
    TimeX learns to explain time series classifiers by training an interpretable
    surrogate model that maintains consistency with the original model's behavior.
    It generates temporal saliency maps showing which time points are important
    for the model's predictions.
    
    Note: This function requires a pre-trained TimeX model. TimeX models must be
    trained using the model behavior consistency objective on your specific
    classification model and dataset before generating explanations.
    
    Args:
        sample: Original time series sample (can be 1D, 2D, or 3D)
        model: The original trained classification model to explain
        timex_model: Pre-trained TimeX explanation model (required)
        return_saliency: If True, returns saliency map; if False, returns masked input
        device: Device to run on (if None, auto-detects)
        verbose: Print debug information
        
    Returns:
        Tuple of (explanation, prediction_scores)
        - explanation: Saliency map or masked input with same shape as sample
        - prediction_scores: Class probabilities from the original model
        
    Raises:
        ValueError: If timex_model is not provided (TimeX requires pre-training)
    """
    if timex_model is None:
        raise ValueError(
            "TimeX requires a pre-trained explanation model. "
            "Please provide a TimeX model trained with model behavior consistency loss. "
            "See https://github.com/mims-harvard/TimeX for training details."
        )
    
    # Set device
    if device is None:
        device = next(model.parameters()).device
    
    # Move models to device and set to eval mode
    model.to(device)
    model.eval()
    timex_model.to(device)
    timex_model.eval()
    
    # Store original sample shape for later restoration
    original_shape = sample.shape
    original_sample = np.asarray(sample)
    
    # Convert sample to tensor and ensure proper shape (T, B, d) for TimeX
    # TimeX expects: (time_steps, batch_size, features)
    x_tensor = torch.tensor(original_sample, dtype=torch.float32, device=device)
    
    # Normalize shape to (T, 1, d)
    if len(x_tensor.shape) == 1:
        # (L,) -> (L, 1, 1) - univariate series
        x_tensor = x_tensor.reshape(-1, 1, 1)
    elif len(x_tensor.shape) == 2:
        # Could be (L, C) or (C, L)
        if x_tensor.shape[0] < x_tensor.shape[1]:
            # Likely (C, L), transpose to (L, C)
            x_tensor = x_tensor.T
        # (L, C) -> (L, 1, C) - add batch dimension
        x_tensor = x_tensor.unsqueeze(1)
    elif len(x_tensor.shape) == 3:
        # Assume (B, C, L) format, convert to (L, B, C)
        if x_tensor.shape[0] == 1:  # batch first
            x_tensor = x_tensor.permute(2, 0, 1)  # (1, C, L) -> (L, 1, C)
        # else assume already in (T, B, d) format
    else:
        raise ValueError(f"Unsupported sample shape: {original_shape}")
    
    # Get time series length
    T = x_tensor.shape[0]
    
    # Create times tensor - TimeX uses time indices
    # Times should be (T, B) format
    times = torch.arange(1, T + 1, dtype=torch.float32, device=device).unsqueeze(1)
    
    if verbose:
        print(f"TimeX: Input shape (T, B, d): {x_tensor.shape}")
        print(f"TimeX: Times shape (T, B): {times.shape}")
    
    # Get original model prediction
    with torch.no_grad():
        # Convert to model's expected input format (batch first: B, C, L)
        model_input = x_tensor.permute(1, 2, 0)  # (T, B, d) -> (B, d, T)
        original_logits = model(model_input)
        original_probs = torch.softmax(original_logits, dim=-1)
        original_class = torch.argmax(original_probs, dim=-1).item()
        original_probs_np = original_probs.squeeze().cpu().numpy()
    
    if verbose:
        print(f"TimeX: Original class = {original_class}")
        print(f"TimeX: Class probabilities = {original_probs_np}")
    
    # Generate TimeX explanation
    with torch.no_grad():
        try:
            # TimeX's get_saliency_explanation returns dict with:
            # - 'mask_in': mask logits before reparameterization (T, B, d)
            # - 'ste_mask': straight-through estimator mask (binary-like)
            # - 'smooth_src': potentially smoothed input
            explanation_dict = timex_model.get_saliency_explanation(
                x_tensor, 
                times, 
                captum_input=False
            )
            
            if return_saliency:
                # Use mask_in as saliency - higher values = more important
                # mask_in shape: (B, T, d) after transpose in get_saliency_explanation
                saliency = explanation_dict['mask_in']  # (B, T, d)
                explanation_tensor = saliency.squeeze(0)  # Remove batch dim -> (T, d)
                
                if verbose:
                    print(f"TimeX: Saliency shape: {saliency.shape}")
                    print(f"TimeX: Saliency stats - min: {saliency.min():.4f}, "
                          f"max: {saliency.max():.4f}, mean: {saliency.mean():.4f}")
            else:
                # Use ste_mask to create masked input
                ste_mask = explanation_dict['ste_mask']  # Binary-like mask
                # Apply mask to original input
                if len(ste_mask.shape) == 2:
                    ste_mask = ste_mask.unsqueeze(-1)  # (B, T) -> (B, T, 1)
                
                # Transpose to match input: (B, T, d) -> (T, d)
                masked_input = x_tensor.squeeze(1) * ste_mask.transpose(0, 1).squeeze(1)
                explanation_tensor = masked_input
                
                if verbose:
                    print(f"TimeX: Masked input shape: {masked_input.shape}")
                    
        except Exception as e:
            if verbose:
                print(f"TimeX: Error generating explanation: {str(e)}")
            # If TimeX model doesn't have get_saliency_explanation, try forward pass
            try:
                output_dict = timex_model(x_tensor, times, captum_input=False)
                if return_saliency:
                    explanation_tensor = output_dict['mask_logits'].squeeze(1)
                else:
                    explanation_tensor = output_dict['smooth_src'].squeeze(1)
            except Exception as e2:
                if verbose:
                    print(f"TimeX: Forward pass also failed: {str(e2)}")
                return None, None
    
    # Convert explanation to numpy
    explanation_np = detach_to_numpy(explanation_tensor)
    
    # Reshape explanation to match original input shape
    if len(original_shape) == 1:
        # Original was (L,), explanation is (L, 1) -> flatten
        explanation_np = explanation_np.squeeze()
    elif len(original_shape) == 2:
        # Original was (L, C) or (C, L)
        if original_shape[0] < original_shape[1]:
            # Original was (C, L), transpose back
            explanation_np = explanation_np.T
    elif len(original_shape) == 3:
        # Original was (B, C, L), reshape appropriately
        if original_shape[0] == 1:
            explanation_np = explanation_np.T.reshape(original_shape)
    
    if verbose:
        print(f"TimeX: Final explanation shape: {explanation_np.shape}")
    
    return explanation_np, original_probs_np


def timex_generate_from_mask(
    sample: np.ndarray,
    mask: np.ndarray,
    baseline: str = 'zero',
    device: str = None
) -> np.ndarray:
    """
    Apply a TimeX-generated mask to create a modified sample.
    
    This utility function applies a temporal saliency mask to the input,
    zeroing out or replacing less important time steps.
    
    Args:
        sample: Original time series sample
        mask: Binary or continuous mask (same shape as sample)
        baseline: Baseline replacement value ('zero', 'mean', or 'noise')
        device: Device to run on
        
    Returns:
        Modified sample with mask applied
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    sample_tensor = numpy_to_torch(sample, device)
    mask_tensor = numpy_to_torch(mask, device)
    
    # Ensure mask is in [0, 1] range
    if mask_tensor.max() > 1.0:
        # Assume logits, apply sigmoid
        mask_tensor = torch.sigmoid(mask_tensor)
    
    # Create baseline
    if baseline == 'zero':
        baseline_tensor = torch.zeros_like(sample_tensor)
    elif baseline == 'mean':
        baseline_tensor = torch.ones_like(sample_tensor) * sample_tensor.mean()
    elif baseline == 'noise':
        baseline_tensor = torch.randn_like(sample_tensor) * sample_tensor.std() + sample_tensor.mean()
    else:
        baseline_tensor = torch.zeros_like(sample_tensor)
    
    # Apply mask: keep important parts (high mask values), replace rest with baseline
    modified = sample_tensor * mask_tensor + baseline_tensor * (1 - mask_tensor)
    
    return detach_to_numpy(modified)


def timex_top_k_mask(
    saliency: np.ndarray,
    k: int = None,
    threshold: float = None
) -> np.ndarray:
    """
    Create a binary mask from TimeX saliency keeping only top-k or threshold values.
    
    Args:
        saliency: Continuous saliency map from TimeX
        k: Keep top k time points (if None, use threshold)
        threshold: Keep values above threshold (if k is None)
        
    Returns:
        Binary mask with same shape as saliency
    """
    if k is not None:
        # Flatten, get top k indices, create mask
        flat_saliency = saliency.flatten()
        threshold_value = np.partition(flat_saliency, -k)[-k]
        mask = (saliency >= threshold_value).astype(np.float32)
    elif threshold is not None:
        mask = (saliency >= threshold).astype(np.float32)
    else:
        # Default: keep top 50% of values
        median_val = np.median(saliency)
        mask = (saliency >= median_val).astype(np.float32)
    
    return mask
