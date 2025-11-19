import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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


def _simple_revert(cf_arr, orientation):
    """Revert counterfactual to original orientation."""
    if orientation == "1d":
        return cf_arr.reshape(-1)
    if orientation == "cf":
        return cf_arr
    if orientation == "tf":
        return cf_arr.T
    return cf_arr


def _revert_orientation(cf_arr, orientation):
    return _simple_revert(cf_arr, orientation)


def glacier_cf(sample, dataset, model, target_class=None, lambda_sparse=0.1, lambda_proximity=1.0,
               lambda_diversity=0.1, max_iterations=1000, learning_rate=0.01, tolerance=1e-6,
               initialization_method='closest_different', verbose=False):
    """
    GLACIER: Gradient-based Learning of Approximate Counterfactual Explanations for Recurrent neural networks
    
    Generates counterfactual explanations by optimizing a multi-objective loss function using gradients.
    The method aims to find minimal perturbations to the input time series that change the model's prediction.
    
    Args:
        sample: Input time series sample to explain
        dataset: Training dataset for finding initialization and reference points
        model: Trained PyTorch model
        target_class: Desired target class for counterfactual (if None, finds closest different class)
        lambda_sparse: Weight for sparsity regularization (promotes minimal changes)
        lambda_proximity: Weight for proximity constraint (keeps CF close to original)
        lambda_diversity: Weight for diversity constraint (ensures CF is different from original)
        max_iterations: Maximum number of optimization iterations
        learning_rate: Learning rate for gradient descent
        tolerance: Convergence tolerance
        initialization_method: Method for initializing the counterfactual ('closest_different', 'random', 'original')
    
    Returns:
        counterfactual: Generated counterfactual in same orientation as input
        cf_prediction: Model prediction for the counterfactual
    """
    device = next(model.parameters()).device
    
    def model_predict(arr):
        """Get model predictions for input array."""
        return detach_to_numpy(model(numpy_to_torch(arr, device)))
    
    def model_predict_torch(arr_torch):
        """Get model predictions for torch tensor."""
        return model(arr_torch)
    
    # Prepare sample and dataset in (C, L) and (N, C, L)
    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    N, C, L = time_series_data.shape
    
    # Get original prediction
    sample_torch = numpy_to_torch(sample_cf.reshape(1, C, L), device)
    original_pred = model_predict_torch(sample_torch)
    original_class = torch.argmax(original_pred, dim=1).item()
    
    # Determine target class
    if target_class is None:
        # Find predictions for all dataset samples
        preds_data = model_predict(time_series_data)
        labels_data = np.argmax(preds_data, axis=1)
        
        # Find most common class different from original
        unique_classes, counts = np.unique(labels_data[labels_data != original_class], return_counts=True)
        if len(unique_classes) == 0:
            # If no different classes found, use the second most probable class from original prediction
            target_class = torch.argsort(original_pred, dim=1, descending=True)[0, 1].item()
        else:
            target_class = unique_classes[np.argmax(counts)]
    
    # Initialize counterfactual
    if initialization_method == 'closest_different':
        # Find closest sample with target class
        preds_data = model_predict(time_series_data)
        labels_data = np.argmax(preds_data, axis=1)
        target_mask = labels_data == target_class
        
        if np.any(target_mask):
            target_samples = time_series_data[target_mask]
            # Calculate distances to find closest
            sample_flat = sample_cf.reshape(1, -1)
            target_flat = target_samples.reshape(len(target_samples), -1)
            distances = np.sum((target_flat - sample_flat) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            cf_init = target_samples[closest_idx].copy()
        else:
            cf_init = sample_cf.copy()
    elif initialization_method == 'random':
        # Random sample from dataset
        random_idx = np.random.randint(0, N)
        cf_init = time_series_data[random_idx].copy()
    else:  # 'original'
        cf_init = sample_cf.copy()
    
    # Convert to torch tensor and enable gradients
    cf_tensor = numpy_to_torch(cf_init.reshape(1, C, L), device)
    cf_tensor.requires_grad_(True)
    
    # Optimizer
    optimizer = optim.Adam([cf_tensor], lr=learning_rate)
    
    # Target tensor
    target_tensor = torch.tensor([target_class], dtype=torch.long, device=device)
    
    # Loss function components
    cross_entropy = nn.CrossEntropyLoss()
    
    prev_loss = float('inf')
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        cf_pred = model_predict_torch(cf_tensor)
        
        # Classification loss (minimize prediction for target class)
        classification_loss = cross_entropy(cf_pred, target_tensor)
        
        # Proximity loss (L2 distance from original)
        proximity_loss = torch.mean((cf_tensor - sample_torch) ** 2)
        
        # Sparsity loss (L1 norm of difference)
        sparsity_loss = torch.mean(torch.abs(cf_tensor - sample_torch))
        
        # Diversity loss (ensure CF is sufficiently different from original)
        diversity_loss = 1.0 / (1.0 + torch.mean((cf_tensor - sample_torch) ** 2))
        
        # Combined loss
        total_loss = (classification_loss + 
                     lambda_proximity * proximity_loss + 
                     lambda_sparse * sparsity_loss + 
                     lambda_diversity * diversity_loss)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Check convergence
        if abs(prev_loss - total_loss.item()) < tolerance:
            break
        prev_loss = total_loss.item()
        
        # Check if we've achieved the target class
        cf_class = torch.argmax(cf_pred, dim=1).item()
        if cf_class == target_class and iteration > 100:  # Allow some iterations for stabilization
            break
    
    # Extract final counterfactual
    cf_final = detach_to_numpy(cf_tensor)[0]  # Remove batch dimension
    cf_prediction = detach_to_numpy(model_predict_torch(cf_tensor))[0]
    
    # Revert to original orientation
    cf_out = _revert_orientation(cf_final, sample_ori)
    
    return cf_out, cf_prediction


def glacier_cf_multi_objective(sample, dataset, model, target_class=None, lambda_sparse=0.1, 
                              lambda_proximity=1.0, lambda_plausibility=0.5, max_iterations=1000,
                              learning_rate=0.01, tolerance=1e-6):
    """
    Enhanced GLACIER with plausibility constraint.
    
    This version includes an additional plausibility constraint that encourages the counterfactual
    to be similar to real samples from the target class distribution.
    
    Args:
        sample: Input time series sample to explain
        dataset: Training dataset for finding reference points
        model: Trained PyTorch model
        target_class: Desired target class for counterfactual
        lambda_sparse: Weight for sparsity regularization
        lambda_proximity: Weight for proximity constraint
        lambda_plausibility: Weight for plausibility constraint
        max_iterations: Maximum number of optimization iterations
        learning_rate: Learning rate for gradient descent
        tolerance: Convergence tolerance
    
    Returns:
        counterfactual: Generated counterfactual in same orientation as input
        cf_prediction: Model prediction for the counterfactual
    """
    device = next(model.parameters()).device
    
    def model_predict(arr):
        return detach_to_numpy(model(numpy_to_torch(arr, device)))
    
    def model_predict_torch(arr_torch):
        return model(arr_torch)
    
    # Prepare sample and dataset
    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    N, C, L = time_series_data.shape
    
    # Get original prediction
    sample_torch = numpy_to_torch(sample_cf.reshape(1, C, L), device)
    original_pred = model_predict_torch(sample_torch)
    original_class = torch.argmax(original_pred, dim=1).item()
    
    # Determine target class and get target samples
    if target_class is None:
        preds_data = model_predict(time_series_data)
        labels_data = np.argmax(preds_data, axis=1)
        unique_classes = np.unique(labels_data[labels_data != original_class])
        target_class = unique_classes[0] if len(unique_classes) > 0 else (original_class + 1) % model.fc2[0].out_features
    
    # Get target class samples for plausibility constraint
    preds_data = model_predict(time_series_data)
    labels_data = np.argmax(preds_data, axis=1)
    target_mask = labels_data == target_class
    target_samples = time_series_data[target_mask] if np.any(target_mask) else time_series_data
    target_samples_torch = numpy_to_torch(target_samples, device)
    
    # Initialize counterfactual
    if np.any(target_mask):
        sample_flat = sample_cf.reshape(1, -1)
        target_flat = target_samples.reshape(len(target_samples), -1)
        distances = np.sum((target_flat - sample_flat) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        cf_init = target_samples[closest_idx].copy()
    else:
        cf_init = sample_cf.copy()
    
    cf_tensor = numpy_to_torch(cf_init.reshape(1, C, L), device)
    cf_tensor.requires_grad_(True)
    
    optimizer = optim.Adam([cf_tensor], lr=learning_rate)
    target_tensor = torch.tensor([target_class], dtype=torch.long, device=device)
    cross_entropy = nn.CrossEntropyLoss()
    
    prev_loss = float('inf')
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        cf_pred = model_predict_torch(cf_tensor)
        
        # Classification loss
        classification_loss = cross_entropy(cf_pred, target_tensor)
        
        # Proximity loss
        proximity_loss = torch.mean((cf_tensor - sample_torch) ** 2)
        
        # Sparsity loss
        sparsity_loss = torch.mean(torch.abs(cf_tensor - sample_torch))
        
        # Plausibility loss (distance to target class distribution)
        cf_expanded = cf_tensor.expand(len(target_samples_torch), -1, -1)
        plausibility_loss = torch.mean(torch.min(
            torch.sum((cf_expanded - target_samples_torch) ** 2, dim=(1, 2))
        ))
        
        # Combined loss
        total_loss = (classification_loss + 
                     lambda_proximity * proximity_loss + 
                     lambda_sparse * sparsity_loss + 
                     lambda_plausibility * plausibility_loss)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Check convergence
        if abs(prev_loss - total_loss.item()) < tolerance:
            break
        prev_loss = total_loss.item()
        
        # Check if target achieved
        cf_class = torch.argmax(cf_pred, dim=1).item()
        if cf_class == target_class and iteration > 100:
            break
    
    # Extract final result
    cf_final = detach_to_numpy(cf_tensor)[0]
    cf_prediction = detach_to_numpy(model_predict_torch(cf_tensor))[0]
    
    # Revert to original orientation
    cf_out = _revert_orientation(cf_final, sample_ori)
    
    return cf_out, cf_prediction
