import numpy as np
import torch


def detach_to_numpy(data):
    """Convert PyTorch tensor to numpy array."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to PyTorch tensor and move to device."""
    return torch.from_numpy(data).float().to(device)


####
# DisCOX: Discord-based Counterfactual Explanations
#
# This method finds and modifies discordant subsequences (most unusual patterns)
# in time series to generate counterfactual explanations using matrix profile.
#
# Related concept: Keogh, E., Lin, J., & Fu, A. (2005).
#                  "Hot sax: Efficiently finding the most unusual time series subsequence."
#                  Fifth IEEE International Conference on Data Mining (ICDM'05)
#
# The discord is identified using a matrix profile approach, which finds the
# subsequence that has the largest minimum distance to all other non-overlapping
# subsequences in the time series.
####


def find_discord(x, window_size=10):
    """Find the most discordant subsequence using matrix profile.
    
    Args:
        x: Time series data of shape (channels, timesteps)
        window_size: Size of the sliding window for discord detection
        
    Returns:
        Index of the start of the most discordant subsequence
    """
    n_features, n_timesteps = x.shape
    profile = np.zeros(n_timesteps - window_size + 1)
    
    for i in range(len(profile)):
        current = x[:, i:i+window_size]
        # Calculate distance to all other windows
        distances = []
        for j in range(len(profile)):
            if abs(i-j) >= window_size:  # Non-overlapping windows
                compare = x[:, j:j+window_size]
                dist = np.sqrt(np.sum((current - compare) ** 2))
                distances.append(dist)
        profile[i] = np.min(distances) if distances else np.inf
    
    return np.argmax(profile)


def discox_generate_cf(sample, model, target=None, window_size=10, 
                       modification_factor=1.1, max_attempts=20, verbose=False):
    """Generate counterfactual explanation using DisCOX method.
    
    Args:
        sample: Original time series sample (shape can be 1D, 2D, or 3D)
        model: PyTorch model for classification
        target: Target class for the counterfactual (if None, uses second most likely)
        window_size: Size of the sliding window for discord detection
        modification_factor: Factor to multiply/divide the discord region
        max_attempts: Maximum number of modification attempts
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (counterfactual, prediction) or (None, None) if failed
    """
    device = next(model.parameters()).device
    
    def model_predict(data):
        """Helper function to get model predictions."""
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
    
    # Convert sample to proper format for processing (channels, timesteps)
    original_shape = sample.shape
    if len(sample.shape) == 1:
        # (timesteps,) -> (1, timesteps)
        x = sample.reshape(1, -1)
    elif len(sample.shape) == 3:
        # (batch, channels, timesteps) -> (channels, timesteps)
        x = sample.reshape(sample.shape[1], sample.shape[2])
    else:
        # (channels, timesteps) or (timesteps, channels)
        if sample.shape[0] > sample.shape[1]:
            x = sample.T
        else:
            x = sample.copy()
    
    # Get initial prediction
    y_orig = model_predict(x)[0]
    label_orig = np.argmax(y_orig)
    
    if target is None:
        # Find the class with second highest probability
        sorted_indices = np.argsort(y_orig)[::-1]
        target = int(sorted_indices[1])
    
    if verbose:
        print(f"DisCOX: Original class {label_orig}, Target class {target}")
        print(f"DisCOX: Input shape: {original_shape}, Processing shape: {x.shape}")
    
    # Check if window size is valid
    if window_size > x.shape[1]:
        if verbose:
            print(f"DisCOX: Window size {window_size} exceeds time series length {x.shape[1]}, adjusting to {x.shape[1] // 2}")
        window_size = max(1, x.shape[1] // 2)
    
    # Find the discord (most unusual subsequence)
    discord_idx = find_discord(x, window_size)
    
    if verbose:
        print(f"DisCOX: Discord found at index {discord_idx}")
    
    # Try different modifications until valid counterfactual is found
    modifications = [
        modification_factor,           # Amplify
        1 / modification_factor,       # Attenuate
        -modification_factor,          # Invert and amplify
        -1 / modification_factor,      # Invert and attenuate
        modification_factor ** 2,      # Strong amplify
        1 / (modification_factor ** 2), # Strong attenuate
    ]
    
    # Try modifications on the discord region
    for attempt, mod in enumerate(modifications):
        if attempt >= max_attempts:
            break
            
        cf = x.copy()
        cf[:, discord_idx:discord_idx+window_size] *= mod
        
        # Get prediction for this counterfactual
        y_cf = model_predict(cf)[0]
        label_cf = np.argmax(y_cf)
        
        if verbose and attempt % 2 == 0:
            print(f"DisCOX attempt {attempt}: modification={mod:.3f}, pred_class={label_cf}, target={target}, target_prob={y_cf[target]:.4f}")
        
        # Check if we found a valid counterfactual
        if label_cf == target:
            if verbose:
                print(f"DisCOX: Found counterfactual at attempt {attempt} with modification factor {mod}")
            
            # Convert back to original shape
            if len(original_shape) == 1:
                cf = cf.flatten()
            elif len(original_shape) == 3:
                cf = cf.reshape(1, cf.shape[0], cf.shape[1])
            elif original_shape[0] > original_shape[1]:
                cf = cf.T
            
            return cf, y_cf
    
    # If basic modifications don't work, try multiple discords
    if verbose:
        print("DisCOX: Trying multiple discord modifications...")
    
    # Find top-k most discordant regions
    n_features, n_timesteps = x.shape
    profile = np.zeros(n_timesteps - window_size + 1)
    
    for i in range(len(profile)):
        current = x[:, i:i+window_size]
        distances = []
        for j in range(len(profile)):
            if abs(i-j) >= window_size:
                compare = x[:, j:j+window_size]
                dist = np.sqrt(np.sum((current - compare) ** 2))
                distances.append(dist)
        profile[i] = np.min(distances) if distances else np.inf
    
    # Get top 3 discord locations
    top_discords = np.argsort(profile)[-3:][::-1]
    
    for attempt, discord_idx in enumerate(top_discords):
        for mod in modifications[:4]:  # Try basic modifications
            cf = x.copy()
            cf[:, discord_idx:discord_idx+window_size] *= mod
            
            y_cf = model_predict(cf)[0]
            label_cf = np.argmax(y_cf)
            
            if label_cf == target:
                if verbose:
                    print(f"DisCOX: Found counterfactual using discord #{attempt} with modification {mod}")
                
                # Convert back to original shape
                if len(original_shape) == 1:
                    cf = cf.flatten()
                elif len(original_shape) == 3:
                    cf = cf.reshape(1, cf.shape[0], cf.shape[1])
                elif original_shape[0] > original_shape[1]:
                    cf = cf.T
                
                return cf, y_cf
    
    if verbose:
        print("DisCOX: Failed to find valid counterfactual")
    
    return None, None


def discox_batch_generate(dataset, model, indices=None, target=None, 
                         window_size=10, modification_factor=1.1, 
                         max_attempts=20, verbose=False):
    """Generate counterfactuals for a batch of samples using DisCOX.
    
    Args:
        dataset: Dataset containing samples (can be list, array, or torch dataset)
        model: PyTorch model for classification
        indices: Indices of samples to process (if None, processes all)
        target: Target class for counterfactuals (if None, uses second most likely for each)
        window_size: Size of the sliding window for discord detection
        modification_factor: Factor to multiply/divide the discord region
        max_attempts: Maximum number of modification attempts per sample
        verbose: Whether to print debug information
        
    Returns:
        List of tuples (original_sample, counterfactual, prediction) for successful cases
    """
    if indices is None:
        indices = range(len(dataset))
    
    results = []
    success_count = 0
    
    for idx in indices:
        if hasattr(dataset, '__getitem__'):
            sample = dataset[idx]
            if isinstance(sample, tuple):
                sample = sample[0]  # Extract data from (data, label) tuple
        else:
            sample = dataset[idx]
        
        if isinstance(sample, torch.Tensor):
            sample = detach_to_numpy(sample)
        
        cf, pred = discox_generate_cf(
            sample, model, target=target, window_size=window_size,
            modification_factor=modification_factor, max_attempts=max_attempts,
            verbose=verbose
        )
        
        if cf is not None:
            results.append((sample, cf, pred))
            success_count += 1
        
        if verbose and (idx + 1) % 10 == 0:
            print(f"DisCOX batch: Processed {idx + 1}/{len(indices)}, Success rate: {success_count}/{idx + 1}")
    
    if verbose:
        print(f"DisCOX batch complete: {success_count}/{len(indices)} successful")
    
    return results
