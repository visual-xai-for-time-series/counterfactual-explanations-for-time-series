import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.neighbors import NearestNeighbors


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


####
# CELS: Counterfactual Explanations via Learned Saliency
#
# Univariate CELS Paper: Li, P., Bahri, O., Filali, S., & Hamdi, S. M. (2023).
#        "CELS: Counterfactual Explanations for Time Series Data via Learned Saliency Maps"
#        IEEE International Conference on Big Data (BigData), 2023, pp. 718-727
#
# Multivariate M-CELS Paper: Li, P., Bahri, O., Boubrahimi, S. F., & Hamdi, S. M. (2024).
#        "M-CELS: Counterfactual Explanation for Multivariate Time Series Data 
#        Guided by Learned Saliency Maps"
#        arXiv preprint arXiv:2411.02649
#
# Paper URLs:
# - CELS: https://ieeexplore.ieee.org/document/10386229
# - M-CELS: https://arxiv.org/abs/2411.02649
# GitHub: https://github.com/Healthpy/cfe_tsc_pos
#
# This implementation provides both univariate (CELS) and multivariate (M-CELS) 
# counterfactual generation methods that use learned saliency maps to identify
# important time steps/features and generate counterfactuals through nearest unlike
# neighbor replacement guided by the learned saliency.
#
# Key Features:
# - Learns saliency maps to identify important time steps
# - Nearest unlike neighbor replacement strategy
# - Optimizes validity, sparsity, and temporal coherence
# - Supports both univariate and multivariate time series
# - AutoCELS wrapper for automatic method selection
####
def cels_generate(sample,
                  model,
                  X_train,
                  y_train,
                  target=None,
                  learning_rate=0.01,
                  max_iter=100,
                  lambda_valid=1.0,
                  lambda_budget=0.1,
                  lambda_tv=0.1,
                  tv_beta=2,
                  enable_lr_decay=True,
                  lr_decay=0.9991,
                  threshold=0.5,
                  verbose=False):
    """Generate counterfactual for univariate time series using CELS.
    
    Args:
        sample: Original time series sample
        model: Trained classifier model
        X_train: Training data for finding nearest unlike neighbor
        y_train: Training labels
        target: Target class (if None, uses second most probable class)
        learning_rate: Learning rate for optimization
        max_iter: Maximum optimization iterations
        lambda_valid: Weight for validity loss
        lambda_budget: Weight for budget (sparsity) loss
        lambda_tv: Weight for total variation loss
        tv_beta: Beta parameter for TV norm
        enable_lr_decay: Whether to use learning rate decay
        lr_decay: Learning rate decay factor
        threshold: Threshold for applying saliency mask
        verbose: Whether to print progress information
        
    Returns:
        Counterfactual sample and prediction, or (None, None) if failed
    """
    device = next(model.parameters()).device

    def model_predict(data):
        """Get model prediction."""
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

    # Validate input dimensions - CELS only supports univariate
    if (len(sample.shape) == 3 and sample.shape[1] > 1) or \
       (len(sample.shape) == 2 and sample.shape[0] > 1):
        raise ValueError("CELS only supports univariate time series. Use M-CELS for multivariate data.")
    
    # Reshape if needed
    if len(sample.shape) == 3:
        sample = sample.reshape(sample.shape[1], sample.shape[2])
    
    # Get initial prediction
    y_original = model_predict(sample)[0]
    label_original = int(np.argmax(y_original))
    
    if target is None:
        # Find the class with second highest probability
        sorted_indices = np.argsort(y_original)[::-1]
        target = int(sorted_indices[1])
    
    if verbose:
        print(f"CELS: Original class {label_original}, Target class {target}")
    
    # Find nearest unlike neighbor
    # Handle both one-hot encoded and scalar labels
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        # One-hot encoded labels
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        # Scalar labels
        y_train_labels = y_train.flatten()
    
    target_mask = y_train_labels == target
    target_instances = X_train[target_mask]
    if len(target_instances) == 0:
        if verbose:
            print("CELS: No training instances found for target class")
        return None, None
    
    # Reshape for distance computation
    sample_flat = sample.reshape(1, -1)
    target_instances_flat = target_instances.reshape(len(target_instances), -1)
    
    nbrs = NearestNeighbors(n_neighbors=1).fit(target_instances_flat)
    _, indices = nbrs.kneighbors(sample_flat)
    nun = target_instances[indices[0][0]]
    
    if verbose:
        print(f"CELS: Found nearest unlike neighbor")
    
    # Learn saliency map
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device, requires_grad=True)
    nun_tensor = torch.tensor(nun, dtype=torch.float32, device=device, requires_grad=False)
    
    # Initialize mask
    mask_init = np.random.uniform(size=[1, sample.shape[-1]], low=0, high=1)
    mask = Variable(torch.from_numpy(mask_init).float().to(device), requires_grad=True)
    
    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    if enable_lr_decay:
        scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    
    softmax = nn.Softmax(dim=-1)
    best_loss = float('inf')
    counter = 0
    max_no_improve = 30
    imp_threshold = 0.001
    
    for i in range(max_iter):
        # Generate counterfactual with current mask
        cf_tensor = x_tensor * (1 - mask) + nun_tensor * mask
        
        # Ensure proper shape for model
        cf_input = cf_tensor.reshape(1, 1, -1).float()
        output = softmax(model(cf_input))
        
        # Compute losses
        valid_loss = 1 - output[0, target]
        budget_loss = torch.mean(torch.abs(mask))
        
        # TV norm for smoothness
        diffs = torch.abs(mask[..., 1:] - mask[..., :-1])
        tv_loss = torch.mean(torch.pow(diffs, tv_beta))
        
        total_loss = (lambda_valid * valid_loss + 
                     lambda_budget * budget_loss +
                     lambda_tv * tv_loss)
        
        # Early stopping check
        if best_loss - total_loss < imp_threshold:
            counter += 1
        else:
            counter = 0
            best_loss = total_loss
        
        if counter >= max_no_improve:
            if verbose:
                print(f"CELS: Early stopping at iteration {i}")
            break
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if enable_lr_decay:
            scheduler.step()
        
        # Clamp mask to [0, 1]
        mask.data.clamp_(0, 1)
        
        # Debug output
        if verbose and i % 20 == 0:
            current_pred = int(output.argmax().item())
            target_prob = float(output[0, target].item())
            print(f"CELS iter {i}: pred_class={current_pred}, target={target}, "
                  f"target_prob={target_prob:.4f}, loss={total_loss.item():.4f}")
    
    # Apply threshold to get final saliency mask
    saliency = torch.sigmoid(mask).detach().cpu().numpy()
    binary_mask = np.where(saliency > threshold, 1, 0)
    
    # Generate final counterfactual
    sample_np = sample
    nun_np = nun
    cf_final = sample_np * (1 - binary_mask) + nun_np * binary_mask
    
    # Get final prediction
    y_cf = model_predict(cf_final)[0]
    cf_class = int(np.argmax(y_cf))
    
    if verbose:
        print(f"CELS: Final class {cf_class}, Target {target}, Success: {cf_class == target}")
    
    # Return in expected shape
    return cf_final.reshape(1, cf_final.shape[0], cf_final.shape[1]), y_cf


####
# M-CELS: Multivariate Counterfactual Explanations via Learned Saliency
#
# Extension of CELS for multivariate time series
#
####
def m_cels_generate(sample,
                    model,
                    X_train,
                    y_train,
                    target=None,
                    learning_rate=0.01,
                    max_iter=100,
                    lambda_valid=1.0,
                    lambda_sparsity=0.1,
                    lambda_smoothness=0.1,
                    enable_lr_decay=True,
                    lr_decay=0.9991,
                    verbose=False):
    """Generate counterfactual for multivariate time series using M-CELS.
    
    Args:
        sample: Original time series sample (shape: [n_features, seq_length])
        model: Trained classifier model
        X_train: Training data for finding nearest unlike neighbor
        y_train: Training labels
        target: Target class (if None, uses second most probable class)
        learning_rate: Learning rate for optimization
        max_iter: Maximum optimization iterations
        lambda_valid: Weight for validity loss
        lambda_sparsity: Weight for sparsity loss
        lambda_smoothness: Weight for smoothness loss
        enable_lr_decay: Whether to use learning rate decay
        lr_decay: Learning rate decay factor
        verbose: Whether to print progress information
        
    Returns:
        Counterfactual sample and prediction, or (None, None) if failed
    """
    device = next(model.parameters()).device

    def model_predict(data):
        """Get model prediction."""
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        else:
            data_tensor = data
            
        # Handle different input shapes for model
        if len(data_tensor.shape) == 2:
            data_tensor = data_tensor.unsqueeze(0)  # Add batch dimension
            
        return detach_to_numpy(model(data_tensor))

    # Validate input dimensions - M-CELS only supports multivariate
    if (len(sample.shape) == 2 and sample.shape[0] == 1) or \
       (len(sample.shape) == 3 and sample.shape[1] == 1):
        raise ValueError("M-CELS is designed for multivariate time series. Use CELS for univariate data.")
    
    # Ensure proper 3D shape (samples, variables, timesteps)
    if len(sample.shape) == 2:
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])
    
    # Get initial prediction
    y_original = model_predict(sample)[0]
    label_original = int(np.argmax(y_original))
    
    if target is None:
        # Find the class with second highest probability
        sorted_indices = np.argsort(y_original)[::-1]
        target = int(sorted_indices[1])
    
    if verbose:
        print(f"M-CELS: Original class {label_original}, Target class {target}")
    
    # Find nearest unlike neighbor
    # Handle both one-hot encoded and scalar labels
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        # One-hot encoded labels
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        # Scalar labels
        y_train_labels = y_train.flatten()
    
    target_mask = y_train_labels == target
    target_samples = X_train[target_mask]
    
    if len(target_samples) == 0:
        if verbose:
            print("M-CELS: No training instances found for target class")
        return None, None
    
    # Reshape for proper distance computation
    sample_flat = sample.reshape(1, -1)
    samples_flat = target_samples.reshape(len(target_samples), -1)
    
    nbrs = NearestNeighbors(n_neighbors=1).fit(samples_flat)
    _, indices = nbrs.kneighbors(sample_flat)
    nun = target_samples[indices[0][0]]
    
    if verbose:
        print(f"M-CELS: Found nearest unlike neighbor")
    
    # Compute initial saliency based on perturbation sensitivity
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device, requires_grad=True)
    output = model(x_tensor)
    softmax = nn.Softmax(dim=-1)
    output_probs = softmax(output)
    target_prob = output_probs[0, target]
    target_prob.backward()
    
    # Use gradient magnitude as initial saliency
    saliency_init = torch.abs(x_tensor.grad).detach().cpu().numpy()
    # Normalize
    saliency_init = (saliency_init - saliency_init.min()) / (saliency_init.max() - saliency_init.min() + 1e-8)
    
    # Initialize mask from saliency
    mask = Variable(torch.from_numpy(saliency_init).float().to(device), requires_grad=True)
    
    nun_tensor = torch.tensor(nun, dtype=torch.float32, device=device, requires_grad=False)
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device, requires_grad=False)
    
    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    if enable_lr_decay:
        scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    
    best_cf = None
    best_loss = float('inf')
    
    D = sample.shape[1]  # Number of features
    T = sample.shape[2]  # Sequence length
    
    for i in range(max_iter):
        # Generate counterfactual
        cf_tensor = x_tensor * (1 - mask) + nun_tensor * mask
        output = softmax(model(cf_tensor))
        
        # Compute loss components
        valid_loss = 1 - output[0, target]
        sparsity_loss = torch.mean(torch.abs(mask))
        
        # Smoothness loss (temporal and feature-wise)
        smoothness_temporal = torch.mean(torch.abs(mask[:, :, 1:] - mask[:, :, :-1]))
        smoothness_features = torch.mean(torch.abs(mask[:, 1:, :] - mask[:, :-1, :]))
        smoothness_loss = (smoothness_temporal + smoothness_features) / 2
        
        total_loss = (lambda_valid * valid_loss +
                     lambda_sparsity * sparsity_loss +
                     lambda_smoothness * smoothness_loss)
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_cf = cf_tensor.detach().cpu().numpy()
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if enable_lr_decay:
            scheduler.step()
        
        # Clamp mask values
        mask.data.clamp_(0, 1)
        
        # Check validity
        current_class = int(output.argmax().item())
        if current_class == target:
            if verbose:
                print(f"M-CELS: Found valid counterfactual at iteration {i}")
            best_cf = cf_tensor.detach().cpu().numpy()
            break
        
        # Debug output
        if verbose and i % 20 == 0:
            target_prob = float(output[0, target].item())
            print(f"M-CELS iter {i}: pred_class={current_class}, target={target}, "
                  f"target_prob={target_prob:.4f}, loss={total_loss.item():.4f}")
    
    if best_cf is None:
        if verbose:
            print("M-CELS: Failed to generate counterfactual")
        return None, None
    
    # Get final prediction
    y_cf = model_predict(best_cf)[0]
    
    if verbose:
        cf_class = int(np.argmax(y_cf))
        print(f"M-CELS: Final class {cf_class}, Target {target}, Success: {cf_class == target}")
    
    return best_cf, y_cf


def cels_auto(sample,
              model,
              X_train,
              y_train,
              target=None,
              verbose=False,
              **kwargs):
    """Automatically select between CELS and M-CELS based on input dimensionality.
    
    Args:
        sample: Original time series sample
        model: Trained classifier model
        X_train: Training data
        y_train: Training labels
        target: Target class (if None, uses second most probable class)
        verbose: Whether to print progress information
        **kwargs: Additional arguments passed to CELS or M-CELS
        
    Returns:
        Counterfactual sample and prediction, or (None, None) if failed
    """
    # Determine if multivariate
    is_multivariate = (len(sample.shape) == 3 and sample.shape[1] > 1) or \
                      (len(sample.shape) == 2 and sample.shape[0] > 1)
    
    if is_multivariate:
        if verbose:
            print("Auto-CELS: Using M-CELS for multivariate data")
        return m_cels_generate(sample, model, X_train, y_train, target, verbose=verbose, **kwargs)
    else:
        if verbose:
            print("Auto-CELS: Using CELS for univariate data")
        return cels_generate(sample, model, X_train, y_train, target, verbose=verbose, **kwargs)
