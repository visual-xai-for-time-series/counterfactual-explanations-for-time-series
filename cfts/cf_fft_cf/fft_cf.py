import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


####
# FFT-CF: Frequency-based Counterfactual Explanations for Time Series
#
# Reference: Delaney, E., Greene, D., & Keane, M. T. (2021).
#            "Instance-Based Counterfactual Explanations for Time Series Classification."
#            In International Conference on Case-Based Reasoning (ICCBR 2021).
#            Springer, Cham. pp. 32-47.
#            https://doi.org/10.1007/978-3-030-86957-1_3
#
# Additional reference: Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2022).
#                       "Interpreting Deep Learning Models for Multivariate Time Series Forecasting."
#                       Joint European Conference on Machine Learning and Knowledge Discovery in Databases.
#                       Springer, Cham.
#
# This method uses Fast Fourier Transform (FFT) to decompose time series into frequency components,
# then iteratively modifies frequency coefficients (amplitude and/or phase) to find counterfactual
# explanations that change the model's prediction while maintaining temporal structure and realism.
#
# Key advantages:
# - Preserves overall temporal patterns by manipulating frequency domain
# - Can focus on specific frequency bands (low/high frequencies)
# - Maintains smoothness and temporal coherence
# - Efficient for long time series due to FFT's O(n log n) complexity
####


def fft_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    max_iterations: int = 1000,
    frequency_bands: str = "all",  # "all", "low", "high", "mid"
    modification_strategy: str = "amplitude",  # "amplitude", "phase", "both"
    step_size: float = 0.05,
    lambda_proximity: float = 0.1,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual explanations by manipulating frequency components using FFT.
    
    Args:
        sample: Original time series sample (shape: (length,), (channels, length), or (1, channels, length))
        dataset: Dataset object (for compatibility)
        model: Trained classification model
        target_class: Target class for counterfactual (if None, finds different class)
        max_iterations: Maximum number of optimization iterations
        frequency_bands: Which frequency bands to modify ("all", "low", "high", "mid")
        modification_strategy: What to modify ("amplitude", "phase", "both")
        step_size: Step size for frequency coefficient modifications
        lambda_proximity: Weight for proximity constraint
        device: Device to run on (if None, auto-detects)
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Prepare sample
    original_shape = sample.shape
    sample_array = np.copy(sample)
    
    # Ensure shape is (channels, length) or (length,)
    if len(sample_array.shape) == 3:  # (batch, channels, length)
        sample_array = sample_array.squeeze(0)
    
    # Determine if univariate or multivariate
    if len(sample_array.shape) == 1:
        is_univariate = True
        n_channels = 1
        length = sample_array.shape[0]
        sample_array = sample_array.reshape(1, -1)  # (1, length)
    else:
        is_univariate = False
        if sample_array.shape[0] > sample_array.shape[1]:
            sample_array = sample_array.T  # Assume (length, channels) -> (channels, length)
        n_channels, length = sample_array.shape
    
    def model_predict(data: np.ndarray) -> np.ndarray:
        """Helper function to get model predictions."""
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if len(data_tensor.shape) == 2:
                data_tensor = data_tensor.unsqueeze(0)  # Add batch dimension
            output = model(data_tensor)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()
        return probs
    
    # Get original prediction
    original_pred = model_predict(sample_array)
    original_class = np.argmax(original_pred)
    
    # Determine target class
    if target_class is None:
        sorted_classes = np.argsort(original_pred)[::-1]
        target_class = sorted_classes[1]  # Second most likely class
    
    if original_class == target_class:
        if verbose:
            print("Sample already in target class.")
        return None, None
    
    if verbose:
        print(f"FFT-CF: Original class {original_class}, Target class {target_class}")
        print(f"Strategy: {modification_strategy}, Frequency bands: {frequency_bands}")
    
    # Perform FFT on each channel
    fft_coeffs = []
    for ch in range(n_channels):
        fft_result = np.fft.fft(sample_array[ch])
        fft_coeffs.append(fft_result)
    fft_coeffs = np.array(fft_coeffs)
    
    # Determine which frequencies to modify based on frequency_bands
    freq_mask = _get_frequency_mask(length, frequency_bands)
    
    # Initialize counterfactual FFT coefficients
    cf_fft_coeffs = np.copy(fft_coeffs)
    
    best_cf = None
    best_pred = None
    best_target_prob = 0.0
    
    # Iterative modification of frequency components
    for iteration in range(max_iterations):
        # Try different modifications to frequency coefficients
        improvements = []
        
        for ch in range(n_channels):
            for freq_idx in range(length):
                if not freq_mask[freq_idx]:
                    continue  # Skip frequencies not in selected band
                
                # Store original coefficient
                original_coeff = cf_fft_coeffs[ch, freq_idx]
                
                # Try different modifications based on strategy
                if modification_strategy in ["amplitude", "both"]:
                    # Modify amplitude
                    amplitude = np.abs(original_coeff)
                    phase = np.angle(original_coeff)
                    
                    # Try increasing and decreasing amplitude
                    for amp_delta in [step_size, -step_size]:
                        new_amplitude = amplitude * (1 + amp_delta)
                        new_coeff = new_amplitude * np.exp(1j * phase)
                        
                        # Apply modification
                        cf_fft_coeffs[ch, freq_idx] = new_coeff
                        
                        # Ensure Hermitian symmetry for real signal
                        if freq_idx > 0 and freq_idx < length:
                            mirror_idx = length - freq_idx
                            cf_fft_coeffs[ch, mirror_idx] = np.conj(new_coeff)
                        
                        # Convert back to time domain
                        cf_sample = _fft_to_time_domain(cf_fft_coeffs)
                        
                        # Get prediction
                        cf_pred = model_predict(cf_sample)
                        cf_class = np.argmax(cf_pred)
                        target_prob = cf_pred[target_class]
                        
                        # Calculate improvement score
                        proximity_penalty = np.sum((cf_sample - sample_array) ** 2)
                        score = target_prob - lambda_proximity * proximity_penalty
                        
                        improvements.append({
                            'score': score,
                            'target_prob': target_prob,
                            'channel': ch,
                            'freq_idx': freq_idx,
                            'new_coeff': new_coeff,
                            'cf_sample': cf_sample,
                            'cf_pred': cf_pred,
                            'cf_class': cf_class
                        })
                        
                        # Restore original coefficient
                        cf_fft_coeffs[ch, freq_idx] = original_coeff
                        if freq_idx > 0 and freq_idx < length:
                            mirror_idx = length - freq_idx
                            cf_fft_coeffs[ch, mirror_idx] = np.conj(original_coeff)
                
                if modification_strategy in ["phase", "both"]:
                    # Modify phase
                    amplitude = np.abs(original_coeff)
                    phase = np.angle(original_coeff)
                    
                    # Try phase shifts
                    for phase_delta in [step_size * np.pi, -step_size * np.pi]:
                        new_phase = phase + phase_delta
                        new_coeff = amplitude * np.exp(1j * new_phase)
                        
                        # Apply modification
                        cf_fft_coeffs[ch, freq_idx] = new_coeff
                        
                        # Ensure Hermitian symmetry for real signal
                        if freq_idx > 0 and freq_idx < length:
                            mirror_idx = length - freq_idx
                            cf_fft_coeffs[ch, mirror_idx] = np.conj(new_coeff)
                        
                        # Convert back to time domain
                        cf_sample = _fft_to_time_domain(cf_fft_coeffs)
                        
                        # Get prediction
                        cf_pred = model_predict(cf_sample)
                        cf_class = np.argmax(cf_pred)
                        target_prob = cf_pred[target_class]
                        
                        # Calculate improvement score
                        proximity_penalty = np.sum((cf_sample - sample_array) ** 2)
                        score = target_prob - lambda_proximity * proximity_penalty
                        
                        improvements.append({
                            'score': score,
                            'target_prob': target_prob,
                            'channel': ch,
                            'freq_idx': freq_idx,
                            'new_coeff': new_coeff,
                            'cf_sample': cf_sample,
                            'cf_pred': cf_pred,
                            'cf_class': cf_class
                        })
                        
                        # Restore original coefficient
                        cf_fft_coeffs[ch, freq_idx] = original_coeff
                        if freq_idx > 0 and freq_idx < length:
                            mirror_idx = length - freq_idx
                            cf_fft_coeffs[ch, mirror_idx] = np.conj(original_coeff)
        
        if not improvements:
            if verbose:
                print("No valid modifications found.")
            break
        
        # Select best improvement
        best_improvement = max(improvements, key=lambda x: x['score'])
        
        # Apply best modification
        ch = best_improvement['channel']
        freq_idx = best_improvement['freq_idx']
        new_coeff = best_improvement['new_coeff']
        
        cf_fft_coeffs[ch, freq_idx] = new_coeff
        if freq_idx > 0 and freq_idx < length:
            mirror_idx = length - freq_idx
            cf_fft_coeffs[ch, mirror_idx] = np.conj(new_coeff)
        
        # Update best counterfactual if target achieved or better probability
        if best_improvement['target_prob'] > best_target_prob:
            best_target_prob = best_improvement['target_prob']
            best_cf = best_improvement['cf_sample']
            best_pred = best_improvement['cf_pred']
        
        # Check if target class achieved
        if best_improvement['cf_class'] == target_class:
            if verbose:
                print(f"FFT-CF: Target class achieved at iteration {iteration}")
                print(f"Target probability: {best_improvement['target_prob']:.4f}")
            
            # Reshape to original shape
            final_cf = _reshape_to_original(best_improvement['cf_sample'], original_shape)
            return final_cf, best_improvement['cf_pred']
        
        # Progress reporting
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: Best target prob = {best_target_prob:.4f}, "
                  f"Current class = {best_improvement['cf_class']}")
    
    # If no exact counterfactual found, return best attempt
    if best_cf is not None and best_target_prob > 0.3:  # Some reasonable threshold
        if verbose:
            print(f"FFT-CF: Max iterations reached. Best target probability: {best_target_prob:.4f}")
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    
    if verbose:
        print("FFT-CF: Failed to generate valid counterfactual")
    return None, None


def _get_frequency_mask(length: int, frequency_bands: str) -> np.ndarray:
    """
    Create a mask indicating which frequency bins to modify.
    
    Args:
        length: Length of the time series
        frequency_bands: Which bands to modify ("all", "low", "high", "mid")
        
    Returns:
        Boolean mask array
    """
    mask = np.zeros(length, dtype=bool)
    
    if frequency_bands == "all":
        mask[:] = True
    elif frequency_bands == "low":
        # Low frequencies (first 25%)
        cutoff = max(1, length // 4)
        mask[:cutoff] = True
    elif frequency_bands == "high":
        # High frequencies (last 25%)
        cutoff = max(1, length // 4)
        mask[-cutoff:] = True
    elif frequency_bands == "mid":
        # Mid frequencies (middle 50%)
        low_cutoff = length // 4
        high_cutoff = 3 * length // 4
        mask[low_cutoff:high_cutoff] = True
    else:
        mask[:] = True  # Default to all
    
    return mask


def _fft_to_time_domain(fft_coeffs: np.ndarray) -> np.ndarray:
    """
    Convert FFT coefficients back to time domain.
    
    Args:
        fft_coeffs: FFT coefficients array (channels, length)
        
    Returns:
        Time domain signal (channels, length)
    """
    n_channels = fft_coeffs.shape[0]
    time_series = []
    
    for ch in range(n_channels):
        time_signal = np.fft.ifft(fft_coeffs[ch]).real
        time_series.append(time_signal)
    
    return np.array(time_series)


def _reshape_to_original(sample: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Reshape sample back to its original shape.
    
    Args:
        sample: Processed sample (channels, length)
        original_shape: Original shape of input
        
    Returns:
        Reshaped sample
    """
    if len(original_shape) == 1:
        return sample.flatten()
    elif len(original_shape) == 2:
        if original_shape[0] < original_shape[1]:
            return sample  # (channels, length)
        else:
            return sample.T  # (length, channels)
    else:  # len == 3
        return sample.reshape(original_shape)


def fft_gradient_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    max_iterations: int = 500,
    learning_rate: float = 0.01,
    lambda_proximity: float = 0.1,
    lambda_smoothness: float = 0.05,
    frequency_regularization: bool = True,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual using gradient-based optimization in frequency domain.
    
    This variant uses gradient descent to optimize frequency coefficients directly,
    potentially more efficient than the greedy search approach.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object (for compatibility)
        model: Trained classification model
        target_class: Target class for counterfactual
        max_iterations: Maximum optimization iterations
        learning_rate: Learning rate for gradient descent
        lambda_proximity: Weight for proximity constraint
        lambda_smoothness: Weight for smoothness constraint in frequency domain
        frequency_regularization: Whether to regularize high frequencies
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    original_shape = sample.shape
    sample_array = np.copy(sample)
    
    if len(sample_array.shape) == 3:
        sample_array = sample_array.squeeze(0)
    
    if len(sample_array.shape) == 1:
        sample_array = sample_array.reshape(1, -1)
    elif sample_array.shape[0] > sample_array.shape[1]:
        sample_array = sample_array.T
    
    n_channels, length = sample_array.shape
    
    # Convert to tensor
    x_tensor = torch.tensor(sample_array, dtype=torch.float32, device=device)
    
    # Get original prediction
    with torch.no_grad():
        x_model_input = x_tensor.unsqueeze(0)
        original_pred = model(x_model_input)
        original_class = torch.argmax(original_pred, dim=-1).item()
        original_pred_np = torch.softmax(original_pred, dim=-1).squeeze().cpu().numpy()
    
    # Determine target
    if target_class is None:
        sorted_classes = torch.argsort(original_pred, descending=True)[0]
        target_class = sorted_classes[1].item()
    
    if original_class == target_class:
        return None, None
    
    if verbose:
        print(f"FFT-Gradient-CF: Original class {original_class}, Target class {target_class}")
    
    # Perform FFT and extract magnitude and phase
    fft_result = torch.fft.rfft(x_tensor, dim=-1)
    
    # Initialize learnable parameters (real and imaginary parts)
    fft_real = fft_result.real.clone().detach().requires_grad_(True)
    fft_imag = fft_result.imag.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([fft_real, fft_imag], lr=learning_rate)
    
    best_cf = None
    best_pred = None
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Reconstruct complex FFT from real and imaginary parts
        fft_complex = torch.complex(fft_real, fft_imag)
        
        # Convert back to time domain
        x_cf = torch.fft.irfft(fft_complex, n=length, dim=-1)
        
        # Get prediction
        x_cf_input = x_cf.unsqueeze(0)
        pred = model(x_cf_input)
        pred_probs = torch.softmax(pred, dim=-1)
        
        # Loss components
        # 1. Classification loss (maximize target class probability)
        classification_loss = -torch.log(pred_probs[0, target_class] + 1e-10)
        
        # 2. Proximity loss (minimize distance from original)
        proximity_loss = torch.sum((x_cf - x_tensor) ** 2)
        
        # 3. Smoothness in frequency domain (penalize high-frequency noise)
        smoothness_loss = 0
        if lambda_smoothness > 0:
            # Penalize large high-frequency components
            freq_indices = torch.arange(fft_complex.shape[-1], device=device)
            freq_weights = freq_indices.float() / fft_complex.shape[-1]
            magnitude = torch.abs(fft_complex)
            smoothness_loss = torch.sum(magnitude * freq_weights.unsqueeze(0))
        
        # Total loss
        loss = classification_loss + lambda_proximity * proximity_loss + lambda_smoothness * smoothness_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Check if target achieved
        with torch.no_grad():
            pred_class = torch.argmax(pred_probs, dim=-1).item()
            target_prob = pred_probs[0, target_class].item()
            
            if pred_class == target_class:
                best_cf = x_cf.cpu().numpy()
                best_pred = pred_probs.squeeze().cpu().numpy()
                
                if verbose:
                    print(f"FFT-Gradient-CF: Target achieved at iteration {iteration}")
                    print(f"Target probability: {target_prob:.4f}")
                
                final_cf = _reshape_to_original(best_cf, original_shape)
                return final_cf, best_pred
            
            # Update best
            if best_cf is None or target_prob > best_pred[target_class]:
                best_cf = x_cf.cpu().numpy()
                best_pred = pred_probs.squeeze().cpu().numpy()
            
            if verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: Loss = {loss.item():.4f}, "
                      f"Target prob = {target_prob:.4f}, Class = {pred_class}")
    
    if best_cf is not None and best_pred[target_class] > 0.3:
        if verbose:
            print(f"FFT-Gradient-CF: Max iterations reached. Best target prob: {best_pred[target_class]:.4f}")
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    
    if verbose:
        print("FFT-Gradient-CF: Failed to generate valid counterfactual")
    return None, None


def fft_nn_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    k: int = 5,
    blend_ratio: float = 0.5,
    frequency_bands: str = "all",
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual using nearest neighbor and FFT-based blending.
    
    This approach:
    1. Finds k nearest neighbors from the target class
    2. Performs FFT on both the original and nearest neighbor
    3. Blends frequency components to create a counterfactual
    4. Converts back to time domain
    
    This is more instance-based and often produces more realistic counterfactuals.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object containing training/test samples
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors to consider
        blend_ratio: Ratio for blending (0=all original, 1=all neighbor)
        frequency_bands: Which frequency bands to modify
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    original_shape = sample.shape
    sample_array = np.copy(sample)
    
    if len(sample_array.shape) == 3:
        sample_array = sample_array.squeeze(0)
    
    if len(sample_array.shape) == 1:
        sample_array = sample_array.reshape(1, -1)
    elif sample_array.shape[0] > sample_array.shape[1]:
        sample_array = sample_array.T
    
    n_channels, length = sample_array.shape
    
    def model_predict(data: np.ndarray) -> np.ndarray:
        """Helper function to get model predictions."""
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if len(data_tensor.shape) == 2:
                data_tensor = data_tensor.unsqueeze(0)
            output = model(data_tensor)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()
        return probs
    
    # Get original prediction
    original_pred = model_predict(sample_array)
    original_class = np.argmax(original_pred)
    
    # Determine target class
    if target_class is None:
        sorted_classes = np.argsort(original_pred)[::-1]
        target_class = sorted_classes[1]
    
    if original_class == target_class:
        if verbose:
            print("Sample already in target class.")
        return None, None
    
    if verbose:
        print(f"FFT-NN-CF: Finding nearest neighbors from target class {target_class}")
    
    # Find nearest neighbors from target class
    target_samples = []
    distances = []
    
    for i in range(len(dataset)):
        data_item = dataset[i]
        if isinstance(data_item, tuple):
            candidate, label = data_item
        else:
            candidate = data_item
            label = None
        
        # Get class label
        if label is not None:
            if hasattr(label, 'shape') and len(label.shape) > 0:
                candidate_class = np.argmax(label)
            else:
                candidate_class = int(label)
        else:
            continue
        
        # Only consider samples from target class
        if candidate_class == target_class:
            # Prepare candidate
            candidate_array = np.array(candidate)
            if len(candidate_array.shape) == 1:
                candidate_array = candidate_array.reshape(1, -1)
            elif candidate_array.shape[0] > candidate_array.shape[1]:
                candidate_array = candidate_array.T
            
            # Calculate distance
            dist = np.linalg.norm(sample_array - candidate_array)
            target_samples.append(candidate_array)
            distances.append(dist)
    
    if len(target_samples) == 0:
        if verbose:
            print("No samples found in target class")
        return None, None
    
    # Select k nearest neighbors
    k = min(k, len(target_samples))
    nearest_indices = np.argsort(distances)[:k]
    
    if verbose:
        print(f"Found {len(target_samples)} target class samples, using {k} nearest")
    
    # Try blending with each nearest neighbor
    best_cf = None
    best_pred = None
    best_target_prob = 0.0
    
    freq_mask = _get_frequency_mask(length, frequency_bands)
    
    for idx in nearest_indices:
        neighbor = target_samples[idx]
        
        # Perform FFT on both
        fft_sample = []
        fft_neighbor = []
        
        for ch in range(n_channels):
            fft_sample.append(np.fft.fft(sample_array[ch]))
            fft_neighbor.append(np.fft.fft(neighbor[ch]))
        
        fft_sample = np.array(fft_sample)
        fft_neighbor = np.array(fft_neighbor)
        
        # Try different blend ratios
        for ratio in [0.3, 0.5, 0.7, 0.9]:
            # Blend frequency components
            fft_blended = np.copy(fft_sample)
            
            for ch in range(n_channels):
                for freq_idx in range(length):
                    if freq_mask[freq_idx]:
                        # Blend this frequency component
                        fft_blended[ch, freq_idx] = (
                            (1 - ratio) * fft_sample[ch, freq_idx] +
                            ratio * fft_neighbor[ch, freq_idx]
                        )
            
            # Convert back to time domain
            cf_candidate = _fft_to_time_domain(fft_blended)
            
            # Get prediction
            cf_pred = model_predict(cf_candidate)
            cf_class = np.argmax(cf_pred)
            target_prob = cf_pred[target_class]
            
            # Check if target achieved
            if cf_class == target_class:
                if verbose:
                    print(f"FFT-NN-CF: Target achieved with blend ratio {ratio:.2f}")
                    print(f"Target probability: {target_prob:.4f}")
                
                final_cf = _reshape_to_original(cf_candidate, original_shape)
                return final_cf, cf_pred
            
            # Update best
            if target_prob > best_target_prob:
                best_target_prob = target_prob
                best_cf = cf_candidate
                best_pred = cf_pred
    
    # Return best attempt
    if best_cf is not None and best_target_prob > 0.3:
        if verbose:
            print(f"FFT-NN-CF: Best target probability: {best_target_prob:.4f}")
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    
    if verbose:
        print("FFT-NN-CF: Failed to generate valid counterfactual")
    return None, None


def fft_adaptive_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    k: int = 5,
    use_saliency: bool = True,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Improvement #1: Adaptive Frequency Band Selection using gradient-based saliency.
    
    Identifies which frequency components are most important for the model's decision
    and focuses modifications on those frequencies.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors to consider
        use_saliency: Whether to use gradient saliency for frequency selection
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    original_shape = sample.shape
    sample_array = np.copy(sample)
    
    if len(sample_array.shape) == 3:
        sample_array = sample_array.squeeze(0)
    if len(sample_array.shape) == 1:
        sample_array = sample_array.reshape(1, -1)
    elif sample_array.shape[0] > sample_array.shape[1]:
        sample_array = sample_array.T
    
    n_channels, length = sample_array.shape
    
    def model_predict(data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if len(data_tensor.shape) == 2:
                data_tensor = data_tensor.unsqueeze(0)
            output = model(data_tensor)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()
        return probs
    
    # Get original prediction
    original_pred = model_predict(sample_array)
    original_class = np.argmax(original_pred)
    
    if target_class is None:
        sorted_classes = np.argsort(original_pred)[::-1]
        target_class = sorted_classes[1]
    
    if original_class == target_class:
        return None, None
    
    # Compute frequency saliency if enabled
    freq_importance = None
    if use_saliency:
        try:
            # Compute gradient-based saliency in frequency domain
            x_tensor = torch.tensor(sample_array, dtype=torch.float32, device=device).unsqueeze(0)
            x_tensor.requires_grad = True
            
            # Forward pass
            output = model(x_tensor)
            
            # Backward pass for target class
            model.zero_grad()
            output[0, original_class].backward()
            
            # Get gradients
            gradients = x_tensor.grad.squeeze().cpu().numpy()
            
            # Compute FFT of gradients to get frequency importance
            freq_importance = np.zeros(length)
            for ch in range(n_channels):
                grad_fft = np.abs(np.fft.fft(gradients[ch]))
                freq_importance += grad_fft[:length]
            
            # Normalize
            freq_importance = freq_importance / (np.max(freq_importance) + 1e-10)
            
            if verbose:
                print(f"FFT-Adaptive: Computed frequency saliency (top freq: {np.argmax(freq_importance)})")
        except:
            freq_importance = None
            if verbose:
                print("FFT-Adaptive: Saliency computation failed, using uniform importance")
    
    # Find nearest neighbors from target class
    target_samples = []
    distances = []
    
    for i in range(len(dataset)):
        data_item = dataset[i]
        if isinstance(data_item, tuple):
            candidate, label = data_item
        else:
            continue
        
        if label is not None:
            if hasattr(label, 'shape') and len(label.shape) > 0:
                candidate_class = np.argmax(label)
            else:
                candidate_class = int(label)
        else:
            continue
        
        if candidate_class == target_class:
            candidate_array = np.array(candidate)
            if len(candidate_array.shape) == 1:
                candidate_array = candidate_array.reshape(1, -1)
            elif candidate_array.shape[0] > candidate_array.shape[1]:
                candidate_array = candidate_array.T
            
            dist = np.linalg.norm(sample_array - candidate_array)
            target_samples.append(candidate_array)
            distances.append(dist)
    
    if len(target_samples) == 0:
        return None, None
    
    k = min(k, len(target_samples))
    nearest_indices = np.argsort(distances)[:k]
    
    if verbose:
        print(f"FFT-Adaptive: Using {k} nearest neighbors with adaptive frequency selection")
    
    best_cf = None
    best_pred = None
    best_target_prob = 0.0
    
    # Blend with adaptive frequency weighting
    for idx in nearest_indices:
        neighbor = target_samples[idx]
        
        # FFT of both
        fft_sample = np.array([np.fft.fft(sample_array[ch]) for ch in range(n_channels)])
        fft_neighbor = np.array([np.fft.fft(neighbor[ch]) for ch in range(n_channels)])
        
        # Try different blending strategies
        for ratio in [0.3, 0.5, 0.7, 0.9]:
            fft_blended = np.copy(fft_sample)
            
            for ch in range(n_channels):
                for freq_idx in range(length):
                    if freq_importance is not None:
                        # Blend more aggressively for important frequencies
                        adaptive_ratio = ratio * (0.5 + 0.5 * freq_importance[freq_idx])
                    else:
                        adaptive_ratio = ratio
                    
                    fft_blended[ch, freq_idx] = (
                        (1 - adaptive_ratio) * fft_sample[ch, freq_idx] +
                        adaptive_ratio * fft_neighbor[ch, freq_idx]
                    )
            
            # Convert back
            cf_candidate = _fft_to_time_domain(fft_blended)
            cf_pred = model_predict(cf_candidate)
            cf_class = np.argmax(cf_pred)
            target_prob = cf_pred[target_class]
            
            if cf_class == target_class:
                if verbose:
                    print(f"FFT-Adaptive: Target achieved (ratio={ratio:.2f}, prob={target_prob:.4f})")
                final_cf = _reshape_to_original(cf_candidate, original_shape)
                return final_cf, cf_pred
            
            if target_prob > best_target_prob:
                best_target_prob = target_prob
                best_cf = cf_candidate
                best_pred = cf_pred
    
    if best_cf is not None and best_target_prob > 0.3:
        if verbose:
            print(f"FFT-Adaptive: Best target prob: {best_target_prob:.4f}")
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    
    return None, None


def fft_iterative_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    k: int = 5,
    refine_iterations: int = 50,
    refine_lr: float = 0.01,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Improvement #2: Iterative Refinement with Local Optimization.
    
    Starts with NN blending (like fft_nn_cf), then refines the result using
    gradient descent in frequency domain to minimize distance while maintaining validity.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors for initial blend
        refine_iterations: Number of refinement iterations
        refine_lr: Learning rate for refinement
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    # First, get initial counterfactual using NN approach
    initial_cf, initial_pred = fft_nn_cf(sample, dataset, model, target_class, k, 
                                          device=device, verbose=False)
    
    if initial_cf is None:
        if verbose:
            print("FFT-Iterative: Initial generation failed")
        return None, None
    
    if verbose:
        print(f"FFT-Iterative: Initial CF generated, starting refinement...")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare samples
    original_shape = sample.shape
    sample_array = np.copy(sample)
    cf_array = np.copy(initial_cf)
    
    if len(sample_array.shape) == 3:
        sample_array = sample_array.squeeze(0)
    if len(sample_array.shape) == 1:
        sample_array = sample_array.reshape(1, -1)
    elif sample_array.shape[0] > sample_array.shape[1]:
        sample_array = sample_array.T
    
    if len(cf_array.shape) == 3:
        cf_array = cf_array.squeeze(0)
    if len(cf_array.shape) == 1:
        cf_array = cf_array.reshape(1, -1)
    elif cf_array.shape[0] > cf_array.shape[1]:
        cf_array = cf_array.T
    
    n_channels, length = sample_array.shape
    
    # Transform to frequency domain
    fft_original = np.array([np.fft.fft(sample_array[ch]) for ch in range(n_channels)])
    fft_cf = np.array([np.fft.fft(cf_array[ch]) for ch in range(n_channels)])
    
    # Convert to learnable parameters (real and imaginary parts)
    fft_cf_real = torch.tensor(fft_cf.real, dtype=torch.float32, device=device, requires_grad=True)
    fft_cf_imag = torch.tensor(fft_cf.imag, dtype=torch.float32, device=device, requires_grad=True)
    fft_original_tensor = torch.tensor(fft_original, dtype=torch.complex64, device=device)
    
    optimizer = torch.optim.Adam([fft_cf_real, fft_cf_imag], lr=refine_lr)
    
    best_cf = cf_array
    best_pred = initial_pred
    best_distance = np.linalg.norm(cf_array - sample_array)
    
    for iteration in range(refine_iterations):
        optimizer.zero_grad()
        
        # Reconstruct complex FFT
        fft_complex = torch.complex(fft_cf_real, fft_cf_imag)
        
        # Convert to time domain
        time_series = torch.fft.ifft(fft_complex, dim=-1).real
        
        # Get prediction
        if time_series.dim() == 2:
            time_series_input = time_series.unsqueeze(0)
        else:
            time_series_input = time_series
        
        pred = model(time_series_input)
        pred_probs = torch.softmax(pred, dim=-1)
        
        # Loss: maximize target class probability + minimize distance in frequency domain
        target_loss = -torch.log(pred_probs[0, target_class] + 1e-10)
        freq_distance = torch.sum(torch.abs(fft_complex - fft_original_tensor))
        
        loss = target_loss + 0.001 * freq_distance
        
        loss.backward()
        optimizer.step()
        
        # Check if still valid
        with torch.no_grad():
            pred_class = torch.argmax(pred_probs).item()
            target_prob = pred_probs[0, target_class].item()
            
            if pred_class == target_class:
                # Update if distance improved
                current_cf = time_series.cpu().numpy()
                current_distance = np.linalg.norm(current_cf - sample_array)
                
                if current_distance < best_distance:
                    best_distance = current_distance
                    best_cf = current_cf
                    best_pred = pred_probs.squeeze().cpu().numpy()
                    
                    if verbose and iteration % 10 == 0:
                        print(f"  Iter {iteration}: distance={current_distance:.4f}, prob={target_prob:.4f}")
    
    if verbose:
        print(f"FFT-Iterative: Refinement complete, final distance={best_distance:.4f}")
    
    final_cf = _reshape_to_original(best_cf, original_shape)
    return final_cf, best_pred


def fft_smart_blend_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    k: int = 5,
    search_method: str = "binary",  # "binary" or "golden"
    tolerance: float = 0.01,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Improvement #3: Smart Blend Ratio Selection.
    
    Uses binary search or golden ratio search to find optimal blend ratio,
    instead of trying fixed ratios.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors
        search_method: "binary" or "golden" search
        tolerance: Convergence tolerance for search
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    original_shape = sample.shape
    sample_array = np.copy(sample)
    
    if len(sample_array.shape) == 3:
        sample_array = sample_array.squeeze(0)
    if len(sample_array.shape) == 1:
        sample_array = sample_array.reshape(1, -1)
    elif sample_array.shape[0] > sample_array.shape[1]:
        sample_array = sample_array.T
    
    n_channels, length = sample_array.shape
    
    def model_predict(data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if len(data_tensor.shape) == 2:
                data_tensor = data_tensor.unsqueeze(0)
            output = model(data_tensor)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()
        return probs
    
    original_pred = model_predict(sample_array)
    original_class = np.argmax(original_pred)
    
    if target_class is None:
        sorted_classes = np.argsort(original_pred)[::-1]
        target_class = sorted_classes[1]
    
    if original_class == target_class:
        return None, None
    
    # Find nearest neighbors
    target_samples = []
    distances = []
    
    for i in range(len(dataset)):
        data_item = dataset[i]
        if isinstance(data_item, tuple):
            candidate, label = data_item
        else:
            continue
        
        if label is not None:
            if hasattr(label, 'shape') and len(label.shape) > 0:
                candidate_class = np.argmax(label)
            else:
                candidate_class = int(label)
        else:
            continue
        
        if candidate_class == target_class:
            candidate_array = np.array(candidate)
            if len(candidate_array.shape) == 1:
                candidate_array = candidate_array.reshape(1, -1)
            elif candidate_array.shape[0] > candidate_array.shape[1]:
                candidate_array = candidate_array.T
            
            dist = np.linalg.norm(sample_array - candidate_array)
            target_samples.append(candidate_array)
            distances.append(dist)
    
    if len(target_samples) == 0:
        return None, None
    
    k = min(k, len(target_samples))
    nearest_indices = np.argsort(distances)[:k]
    
    def evaluate_blend_ratio(ratio: float, fft_sample, fft_neighbor) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate a specific blend ratio."""
        fft_blended = np.copy(fft_sample)
        
        for ch in range(n_channels):
            for freq_idx in range(length):
                fft_blended[ch, freq_idx] = (
                    (1 - ratio) * fft_sample[ch, freq_idx] +
                    ratio * fft_neighbor[ch, freq_idx]
                )
        
        cf_candidate = _fft_to_time_domain(fft_blended)
        cf_pred = model_predict(cf_candidate)
        cf_class = np.argmax(cf_pred)
        target_prob = cf_pred[target_class]
        
        # Score: high if target achieved, based on probability otherwise
        score = 1.0 if cf_class == target_class else target_prob
        
        return score, cf_candidate, cf_pred
    
    def binary_search(fft_sample, fft_neighbor):
        """Binary search for optimal blend ratio."""
        left, right = 0.0, 1.0
        best_score = 0.0
        best_ratio = 0.5
        best_cf = None
        best_pred = None
        
        while right - left > tolerance:
            mid = (left + right) / 2.0
            
            score, cf_candidate, cf_pred = evaluate_blend_ratio(mid, fft_sample, fft_neighbor)
            
            if score > best_score:
                best_score = score
                best_ratio = mid
                best_cf = cf_candidate
                best_pred = cf_pred
            
            # If target achieved, try to get closer to original (lower ratio)
            if score >= 1.0:
                right = mid
            else:
                # Otherwise, increase ratio to get closer to neighbor
                left = mid
        
        return best_score, best_cf, best_pred, best_ratio
    
    def golden_ratio_search(fft_sample, fft_neighbor):
        """Golden ratio search for optimal blend ratio."""
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi
        
        left, right = 0.0, 1.0
        tol = tolerance
        
        # Initial points
        x1 = left + resphi * (right - left)
        x2 = right - resphi * (right - left)
        
        score1, cf1, pred1 = evaluate_blend_ratio(x1, fft_sample, fft_neighbor)
        score2, cf2, pred2 = evaluate_blend_ratio(x2, fft_sample, fft_neighbor)
        
        best_score = max(score1, score2)
        best_cf = cf1 if score1 > score2 else cf2
        best_pred = pred1 if score1 > score2 else pred2
        best_ratio = x1 if score1 > score2 else x2
        
        while abs(right - left) > tol:
            if score1 > score2:
                right = x2
                x2 = x1
                score2 = score1
                cf2, pred2 = cf1, pred1
                
                x1 = left + resphi * (right - left)
                score1, cf1, pred1 = evaluate_blend_ratio(x1, fft_sample, fft_neighbor)
            else:
                left = x1
                x1 = x2
                score1 = score2
                cf1, pred1 = cf2, pred2
                
                x2 = right - resphi * (right - left)
                score2, cf2, pred2 = evaluate_blend_ratio(x2, fft_sample, fft_neighbor)
            
            if max(score1, score2) > best_score:
                best_score = max(score1, score2)
                best_cf = cf1 if score1 > score2 else cf2
                best_pred = pred1 if score1 > score2 else pred2
                best_ratio = x1 if score1 > score2 else x2
        
        return best_score, best_cf, best_pred, best_ratio
    
    if verbose:
        print(f"FFT-SmartBlend: Using {search_method} search with {k} neighbors")
    
    best_overall_score = 0.0
    best_overall_cf = None
    best_overall_pred = None
    best_overall_ratio = 0.0
    
    for idx in nearest_indices:
        neighbor = target_samples[idx]
        
        # FFT
        fft_sample = np.array([np.fft.fft(sample_array[ch]) for ch in range(n_channels)])
        fft_neighbor = np.array([np.fft.fft(neighbor[ch]) for ch in range(n_channels)])
        
        # Search for optimal ratio
        if search_method == "binary":
            score, cf_candidate, cf_pred, ratio = binary_search(fft_sample, fft_neighbor)
        else:  # golden
            score, cf_candidate, cf_pred, ratio = golden_ratio_search(fft_sample, fft_neighbor)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_cf = cf_candidate
            best_overall_pred = cf_pred
            best_overall_ratio = ratio
        
        # If target achieved with high confidence, stop
        if score >= 1.0 and np.argmax(cf_pred) == target_class:
            if verbose:
                print(f"FFT-SmartBlend: Target achieved with ratio={ratio:.4f}")
            final_cf = _reshape_to_original(cf_candidate, original_shape)
            return final_cf, cf_pred
    
    if best_overall_cf is not None and best_overall_score > 0.3:
        if verbose:
            print(f"FFT-SmartBlend: Best ratio={best_overall_ratio:.4f}, score={best_overall_score:.4f}")
        final_cf = _reshape_to_original(best_overall_cf, original_shape)
        return final_cf, best_overall_pred
    
    return None, None


def fft_freq_distance_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    k: int = 5,
    freq_weight_strategy: str = "importance",  # "uniform", "importance", "decay"
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Improvement #4: Frequency-Domain Distance Metric for Neighbor Selection.
    
    Selects nearest neighbors based on frequency domain similarity rather than
    time domain distance.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors
        freq_weight_strategy: How to weight frequencies ("uniform", "importance", "decay")
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    original_shape = sample.shape
    sample_array = np.copy(sample)
    
    if len(sample_array.shape) == 3:
        sample_array = sample_array.squeeze(0)
    if len(sample_array.shape) == 1:
        sample_array = sample_array.reshape(1, -1)
    elif sample_array.shape[0] > sample_array.shape[1]:
        sample_array = sample_array.T
    
    n_channels, length = sample_array.shape
    
    def model_predict(data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if len(data_tensor.shape) == 2:
                data_tensor = data_tensor.unsqueeze(0)
            output = model(data_tensor)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()
        return probs
    
    original_pred = model_predict(sample_array)
    original_class = np.argmax(original_pred)
    
    if target_class is None:
        sorted_classes = np.argsort(original_pred)[::-1]
        target_class = sorted_classes[1]
    
    if original_class == target_class:
        return None, None
    
    # Compute FFT of original
    fft_sample = np.array([np.fft.fft(sample_array[ch]) for ch in range(n_channels)])
    
    # Compute frequency weights
    if freq_weight_strategy == "importance":
        # Weight by magnitude (more important frequencies have higher magnitude)
        freq_weights = np.mean(np.abs(fft_sample), axis=0)
        freq_weights = freq_weights / (np.max(freq_weights) + 1e-10)
    elif freq_weight_strategy == "decay":
        # Higher frequencies decay (assume low frequencies more important)
        freq_weights = np.exp(-np.arange(length) / (length / 5))
    else:  # uniform
        freq_weights = np.ones(length)
    
    # Find nearest neighbors using frequency domain distance
    target_samples = []
    freq_distances = []
    
    for i in range(len(dataset)):
        data_item = dataset[i]
        if isinstance(data_item, tuple):
            candidate, label = data_item
        else:
            continue
        
        if label is not None:
            if hasattr(label, 'shape') and len(label.shape) > 0:
                candidate_class = np.argmax(label)
            else:
                candidate_class = int(label)
        else:
            continue
        
        if candidate_class == target_class:
            candidate_array = np.array(candidate)
            if len(candidate_array.shape) == 1:
                candidate_array = candidate_array.reshape(1, -1)
            elif candidate_array.shape[0] > candidate_array.shape[1]:
                candidate_array = candidate_array.T
            
            # Compute FFT of candidate
            fft_candidate = np.array([np.fft.fft(candidate_array[ch]) for ch in range(n_channels)])
            
            # Weighted frequency domain distance
            freq_diff = np.abs(fft_sample - fft_candidate)
            weighted_dist = np.sum(freq_diff * freq_weights)
            
            target_samples.append(candidate_array)
            freq_distances.append(weighted_dist)
    
    if len(target_samples) == 0:
        return None, None
    
    k = min(k, len(target_samples))
    nearest_indices = np.argsort(freq_distances)[:k]
    
    if verbose:
        print(f"FFT-FreqDist: Selected {k} neighbors using {freq_weight_strategy} freq distance")
    
    best_cf = None
    best_pred = None
    best_target_prob = 0.0
    
    # Blend with selected neighbors
    for idx in nearest_indices:
        neighbor = target_samples[idx]
        
        fft_neighbor = np.array([np.fft.fft(neighbor[ch]) for ch in range(n_channels)])
        
        for ratio in [0.3, 0.5, 0.7, 0.9]:
            fft_blended = (1 - ratio) * fft_sample + ratio * fft_neighbor
            
            cf_candidate = _fft_to_time_domain(fft_blended)
            cf_pred = model_predict(cf_candidate)
            cf_class = np.argmax(cf_pred)
            target_prob = cf_pred[target_class]
            
            if cf_class == target_class:
                if verbose:
                    print(f"FFT-FreqDist: Target achieved (ratio={ratio:.2f})")
                final_cf = _reshape_to_original(cf_candidate, original_shape)
                return final_cf, cf_pred
            
            if target_prob > best_target_prob:
                best_target_prob = target_prob
                best_cf = cf_candidate
                best_pred = cf_pred
    
    if best_cf is not None and best_target_prob > 0.3:
        if verbose:
            print(f"FFT-FreqDist: Best target prob: {best_target_prob:.4f}")
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    
    return None, None


def fft_wavelet_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    k: int = 5,
    wavelet: str = "db4",
    level: int = 3,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Improvement #5: Multi-Resolution Approach using Wavelet Transform.
    
    Uses Discrete Wavelet Transform (DWT) instead of FFT for better handling
    of non-stationary signals and local temporal features.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors
        wavelet: Wavelet type (e.g., "db4", "haar", "sym4")
        level: Decomposition level
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    try:
        import pywt
    except ImportError:
        if verbose:
            print("FFT-Wavelet: PyWavelets not installed, falling back to FFT")
        return fft_nn_cf(sample, dataset, model, target_class, k, device=device, verbose=verbose)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    original_shape = sample.shape
    sample_array = np.copy(sample)
    
    if len(sample_array.shape) == 3:
        sample_array = sample_array.squeeze(0)
    if len(sample_array.shape) == 1:
        sample_array = sample_array.reshape(1, -1)
    elif sample_array.shape[0] > sample_array.shape[1]:
        sample_array = sample_array.T
    
    n_channels, length = sample_array.shape
    
    def model_predict(data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if len(data_tensor.shape) == 2:
                data_tensor = data_tensor.unsqueeze(0)
            output = model(data_tensor)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()
        return probs
    
    original_pred = model_predict(sample_array)
    original_class = np.argmax(original_pred)
    
    if target_class is None:
        sorted_classes = np.argsort(original_pred)[::-1]
        target_class = sorted_classes[1]
    
    if original_class == target_class:
        return None, None
    
    # Compute wavelet decomposition
    def wavelet_decompose(signal):
        """Decompose signal using DWT."""
        coeffs_list = []
        for ch in range(signal.shape[0]):
            coeffs = pywt.wavedec(signal[ch], wavelet, level=level)
            coeffs_list.append(coeffs)
        return coeffs_list
    
    def wavelet_reconstruct(coeffs_list):
        """Reconstruct signal from DWT coefficients."""
        reconstructed = []
        for coeffs in coeffs_list:
            signal = pywt.waverec(coeffs, wavelet)
            # Ensure same length as original
            if len(signal) > length:
                signal = signal[:length]
            elif len(signal) < length:
                signal = np.pad(signal, (0, length - len(signal)))
            reconstructed.append(signal)
        return np.array(reconstructed)
    
    # Decompose original
    sample_coeffs = wavelet_decompose(sample_array)
    
    # Find nearest neighbors
    target_samples = []
    distances = []
    
    for i in range(len(dataset)):
        data_item = dataset[i]
        if isinstance(data_item, tuple):
            candidate, label = data_item
        else:
            continue
        
        if label is not None:
            if hasattr(label, 'shape') and len(label.shape) > 0:
                candidate_class = np.argmax(label)
            else:
                candidate_class = int(label)
        else:
            continue
        
        if candidate_class == target_class:
            candidate_array = np.array(candidate)
            if len(candidate_array.shape) == 1:
                candidate_array = candidate_array.reshape(1, -1)
            elif candidate_array.shape[0] > candidate_array.shape[1]:
                candidate_array = candidate_array.T
            
            dist = np.linalg.norm(sample_array - candidate_array)
            target_samples.append(candidate_array)
            distances.append(dist)
    
    if len(target_samples) == 0:
        return None, None
    
    k = min(k, len(target_samples))
    nearest_indices = np.argsort(distances)[:k]
    
    if verbose:
        print(f"FFT-Wavelet: Using {wavelet} wavelet (level={level}) with {k} neighbors")
    
    best_cf = None
    best_pred = None
    best_target_prob = 0.0
    
    # Blend in wavelet domain
    for idx in nearest_indices:
        neighbor = target_samples[idx]
        neighbor_coeffs = wavelet_decompose(neighbor)
        
        for ratio in [0.3, 0.5, 0.7, 0.9]:
            # Blend wavelet coefficients at each level
            blended_coeffs = []
            for ch in range(n_channels):
                ch_coeffs = []
                for level_idx in range(len(sample_coeffs[ch])):
                    blended = (1 - ratio) * sample_coeffs[ch][level_idx] + ratio * neighbor_coeffs[ch][level_idx]
                    ch_coeffs.append(blended)
                blended_coeffs.append(ch_coeffs)
            
            # Reconstruct
            cf_candidate = wavelet_reconstruct(blended_coeffs)
            cf_pred = model_predict(cf_candidate)
            cf_class = np.argmax(cf_pred)
            target_prob = cf_pred[target_class]
            
            if cf_class == target_class:
                if verbose:
                    print(f"FFT-Wavelet: Target achieved (ratio={ratio:.2f})")
                final_cf = _reshape_to_original(cf_candidate, original_shape)
                return final_cf, cf_pred
            
            if target_prob > best_target_prob:
                best_target_prob = target_prob
                best_cf = cf_candidate
                best_pred = cf_pred
    
    if best_cf is not None and best_target_prob > 0.3:
        if verbose:
            print(f"FFT-Wavelet: Best target prob: {best_target_prob:.4f}")
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    
    return None, None


def fft_hybrid_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    k: int = 5,
    analyze_importance: bool = True,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Improvement #6: Hybrid Amplitude-Phase Strategy.
    
    Adaptively modifies amplitude for some frequencies and phase for others,
    based on which contributes more to the classification decision.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors
        analyze_importance: Whether to analyze amplitude vs phase importance
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    original_shape = sample.shape
    sample_array = np.copy(sample)
    
    if len(sample_array.shape) == 3:
        sample_array = sample_array.squeeze(0)
    if len(sample_array.shape) == 1:
        sample_array = sample_array.reshape(1, -1)
    elif sample_array.shape[0] > sample_array.shape[1]:
        sample_array = sample_array.T
    
    n_channels, length = sample_array.shape
    
    def model_predict(data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if len(data_tensor.shape) == 2:
                data_tensor = data_tensor.unsqueeze(0)
            output = model(data_tensor)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()
        return probs
    
    original_pred = model_predict(sample_array)
    original_class = np.argmax(original_pred)
    
    if target_class is None:
        sorted_classes = np.argsort(original_pred)[::-1]
        target_class = sorted_classes[1]
    
    if original_class == target_class:
        return None, None
    
    # Analyze amplitude vs phase importance
    amplitude_importance = None
    phase_importance = None
    
    if analyze_importance:
        try:
            fft_sample = np.array([np.fft.fft(sample_array[ch]) for ch in range(n_channels)])
            
            # Test impact of amplitude changes
            amp_scores = []
            for freq_idx in range(min(20, length)):  # Test first 20 frequencies
                test_fft = fft_sample.copy()
                for ch in range(n_channels):
                    # Perturb amplitude
                    magnitude = np.abs(test_fft[ch, freq_idx])
                    phase = np.angle(test_fft[ch, freq_idx])
                    test_fft[ch, freq_idx] = (magnitude * 1.1) * np.exp(1j * phase)
                
                test_signal = _fft_to_time_domain(test_fft)
                test_pred = model_predict(test_signal)
                amp_score = np.abs(test_pred[original_class] - original_pred[original_class])
                amp_scores.append(amp_score)
            
            # Test impact of phase changes
            phase_scores = []
            for freq_idx in range(min(20, length)):
                test_fft = fft_sample.copy()
                for ch in range(n_channels):
                    # Perturb phase
                    magnitude = np.abs(test_fft[ch, freq_idx])
                    phase = np.angle(test_fft[ch, freq_idx])
                    test_fft[ch, freq_idx] = magnitude * np.exp(1j * (phase + 0.1))
                
                test_signal = _fft_to_time_domain(test_fft)
                test_pred = model_predict(test_signal)
                phase_score = np.abs(test_pred[original_class] - original_pred[original_class])
                phase_scores.append(phase_score)
            
            amplitude_importance = np.array(amp_scores)
            phase_importance = np.array(phase_scores)
            
            if verbose:
                amp_dom = np.sum(amplitude_importance > phase_importance)
                print(f"FFT-Hybrid: {amp_dom}/{len(amp_scores)} frequencies amplitude-dominated")
        except:
            if verbose:
                print("FFT-Hybrid: Importance analysis failed, using balanced strategy")
    
    # Find nearest neighbors
    target_samples = []
    distances = []
    
    for i in range(len(dataset)):
        data_item = dataset[i]
        if isinstance(data_item, tuple):
            candidate, label = data_item
        else:
            continue
        
        if label is not None:
            if hasattr(label, 'shape') and len(label.shape) > 0:
                candidate_class = np.argmax(label)
            else:
                candidate_class = int(label)
        else:
            continue
        
        if candidate_class == target_class:
            candidate_array = np.array(candidate)
            if len(candidate_array.shape) == 1:
                candidate_array = candidate_array.reshape(1, -1)
            elif candidate_array.shape[0] > candidate_array.shape[1]:
                candidate_array = candidate_array.T
            
            dist = np.linalg.norm(sample_array - candidate_array)
            target_samples.append(candidate_array)
            distances.append(dist)
    
    if len(target_samples) == 0:
        return None, None
    
    k = min(k, len(target_samples))
    nearest_indices = np.argsort(distances)[:k]
    
    best_cf = None
    best_pred = None
    best_target_prob = 0.0
    
    # Hybrid blending strategy
    for idx in nearest_indices:
        neighbor = target_samples[idx]
        
        fft_sample = np.array([np.fft.fft(sample_array[ch]) for ch in range(n_channels)])
        fft_neighbor = np.array([np.fft.fft(neighbor[ch]) for ch in range(n_channels)])
        
        for ratio in [0.3, 0.5, 0.7, 0.9]:
            fft_blended = np.copy(fft_sample)
            
            for ch in range(n_channels):
                for freq_idx in range(length):
                    amp_sample = np.abs(fft_sample[ch, freq_idx])
                    phase_sample = np.angle(fft_sample[ch, freq_idx])
                    amp_neighbor = np.abs(fft_neighbor[ch, freq_idx])
                    phase_neighbor = np.angle(fft_neighbor[ch, freq_idx])
                    
                    # Decide whether to modify amplitude or phase
                    if amplitude_importance is not None and freq_idx < len(amplitude_importance):
                        modify_amplitude = amplitude_importance[freq_idx] > phase_importance[freq_idx]
                    else:
                        # Default: modify amplitude for low freq, phase for high freq
                        modify_amplitude = freq_idx < length // 3
                    
                    if modify_amplitude:
                        # Blend amplitude, keep original phase
                        new_amp = (1 - ratio) * amp_sample + ratio * amp_neighbor
                        fft_blended[ch, freq_idx] = new_amp * np.exp(1j * phase_sample)
                    else:
                        # Blend phase, keep original amplitude
                        # Use circular interpolation for phase
                        phase_diff = phase_neighbor - phase_sample
                        # Wrap to [-π, π]
                        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
                        new_phase = phase_sample + ratio * phase_diff
                        fft_blended[ch, freq_idx] = amp_sample * np.exp(1j * new_phase)
            
            cf_candidate = _fft_to_time_domain(fft_blended)
            cf_pred = model_predict(cf_candidate)
            cf_class = np.argmax(cf_pred)
            target_prob = cf_pred[target_class]
            
            if cf_class == target_class:
                if verbose:
                    print(f"FFT-Hybrid: Target achieved (ratio={ratio:.2f})")
                final_cf = _reshape_to_original(cf_candidate, original_shape)
                return final_cf, cf_pred
            
            if target_prob > best_target_prob:
                best_target_prob = target_prob
                best_cf = cf_candidate
                best_pred = cf_pred
    
    if best_cf is not None and best_target_prob > 0.3:
        if verbose:
            print(f"FFT-Hybrid: Best target prob: {best_target_prob:.4f}")
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    
    return None, None


def fft_progressive_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    k: int = 5,
    steps_per_neighbor: int = 5,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Progressive Neighbor Switching Strategy.
    
    Progressively switches between k nearest neighbors during optimization,
    gradually moving from the query sample toward the counterfactual by
    optimizing toward each neighbor in sequence.
    
    Algorithm:
    1. Find k nearest neighbors in target class
    2. Sort by distance (closest first)
    3. For each neighbor in sequence:
       - Perform gradient steps toward that neighbor
       - Use result as starting point for next neighbor
    4. Return best counterfactual found
    
    This creates a "path" through the feature space, switching between
    different target neighbors to find an optimal counterfactual.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors to switch between
        steps_per_neighbor: Optimization steps per neighbor (default=5)
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    original_shape = sample.shape
    sample_array = np.copy(sample)
    
    if len(sample_array.shape) == 3:
        sample_array = sample_array.squeeze(0)
    if len(sample_array.shape) == 1:
        sample_array = sample_array.reshape(1, -1)
    
    n_channels, length = sample_array.shape
    
    # Model prediction wrapper
    def model_predict(arr):
        with torch.no_grad():
            tensor = torch.from_numpy(arr).float().to(device)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            pred = model(tensor)
            return pred.cpu().numpy().reshape(-1)
    
    # Compute FFT of original sample
    fft_sample = np.array([np.fft.fft(sample_array[ch]) for ch in range(n_channels)])
    
    # Find k nearest neighbors in target class
    target_samples = []
    distances = []
    
    for i in range(len(dataset)):
        data_item = dataset[i]
        if isinstance(data_item, tuple):
            candidate, label = data_item
        else:
            continue
        
        if label is not None:
            if hasattr(label, 'shape') and len(label.shape) > 0:
                candidate_class = np.argmax(label)
            else:
                candidate_class = int(label)
        else:
            continue
        
        if candidate_class == target_class:
            candidate_array = np.array(candidate)
            if len(candidate_array.shape) == 1:
                candidate_array = candidate_array.reshape(1, -1)
            elif candidate_array.shape[0] > candidate_array.shape[1]:
                candidate_array = candidate_array.T
            
            dist = np.linalg.norm(sample_array - candidate_array)
            target_samples.append(candidate_array)
            distances.append(dist)
    
    if len(target_samples) == 0:
        return None, None
    
    k = min(k, len(target_samples))
    
    # Sort by distance and get k nearest
    nearest_indices = np.argsort(distances)[:k]
    neighbors = [target_samples[idx] for idx in nearest_indices]
    neighbor_dists = [distances[idx] for idx in nearest_indices]
    
    if verbose:
        print(f"FFT-Progressive: Found {k} neighbors, distances: {[f'{d:.2f}' for d in neighbor_dists[:3]]}")
    
    # Progressive optimization: switch between neighbors
    current_fft = fft_sample.copy()
    best_cf = None
    best_pred = None
    best_target_prob = 0.0
    
    # Track progress through neighbors
    for neighbor_idx, neighbor in enumerate(neighbors):
        if verbose:
            print(f"FFT-Progressive: Optimizing toward neighbor {neighbor_idx + 1}/{k}")
        
        # Compute FFT of current neighbor
        fft_neighbor = np.array([np.fft.fft(neighbor[ch]) for ch in range(n_channels)])
        
        # Perform gradient steps toward this neighbor
        for step in range(steps_per_neighbor):
            # Blend ratios: gradually increase influence of neighbor
            ratio = (step + 1) / steps_per_neighbor
            
            # Blend current state with neighbor
            fft_blended = (1 - ratio) * current_fft + ratio * fft_neighbor
            
            # Convert to time domain
            cf_candidate = _fft_to_time_domain(fft_blended)
            cf_pred = model_predict(cf_candidate)
            cf_class = np.argmax(cf_pred)
            target_prob = cf_pred[target_class] if target_class < len(cf_pred) else 0.0
            
            # Check if this is successful
            if cf_class == target_class and target_prob > best_target_prob:
                best_target_prob = target_prob
                best_cf = cf_candidate.copy()
                best_pred = cf_pred.copy()
                
                if verbose:
                    print(f"  Step {step+1}/{steps_per_neighbor}: Found CF with prob {target_prob:.4f}")
                
                # Early stopping if we have high confidence
                if target_prob > 0.9:
                    if verbose:
                        print(f"FFT-Progressive: High confidence reached, stopping early")
                    final_cf = _reshape_to_original(best_cf, original_shape)
                    return final_cf, best_pred
            
            # Update current state for next step
            current_fft = fft_blended.copy()
        
        # After completing steps toward this neighbor, check if we found a good CF
        if best_cf is not None and best_target_prob > 0.7:
            # Good enough, but continue to see if we can improve
            if verbose:
                print(f"FFT-Progressive: Good CF found (prob={best_target_prob:.4f}), continuing...")
    
    # Return best counterfactual found
    if best_cf is not None and best_target_prob > 0.3:
        if verbose:
            print(f"FFT-Progressive: Best target prob: {best_target_prob:.4f}")
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    
    return None, None


def fft_confidence_threshold_cf(
    sample,
    dataset,
    model,
    target_class,
    k=5,
    confidence_threshold=0.85,
    max_steps=50,
    refine_iterations=30,
    verbose=False
):
    """
    FFT-based CF with confidence-threshold early stopping for better sparsity.
    
    Combines Iterative Refinement's high success rate with early stopping
    to minimize modifications - stops as soon as target confidence is reached.
    
    Args:
        sample: Original time series to explain
        dataset: Dataset containing neighbor samples
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors to consider
        confidence_threshold: Stop when this confidence is reached (default: 0.85)
        max_steps: Maximum optimization steps
        refine_iterations: Maximum refinement iterations per step
        verbose: Print progress information
    
    Returns:
        counterfactual: Generated counterfactual time series
        prediction: Model prediction for the counterfactual
    """
    device = next(model.parameters()).device
    original_shape = sample.shape
    sample = sample.reshape(-1)
    
    # Get initial prediction
    sample_tensor = torch.FloatTensor(sample).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        original_pred = model(sample_tensor).cpu().numpy()[0]
    original_class = np.argmax(original_pred)
    
    if verbose:
        print(f"FFT-Confidence: Original class {original_class}, target {target_class}")
        print(f"FFT-Confidence: Confidence threshold = {confidence_threshold}")
    
    # Find k nearest neighbors from target class
    target_samples = dataset[dataset[:, -1] == target_class][:, :-1]
    if len(target_samples) == 0:
        return _reshape_to_original(sample, original_shape), original_pred
    
    distances = np.linalg.norm(target_samples - sample.reshape(1, -1), axis=1)
    nearest_indices = np.argsort(distances)[:k]
    nearest_neighbors = target_samples[nearest_indices]
    
    # Start with nearest neighbor baseline
    sample_fft = np.fft.fft(sample)
    nn_fft = np.fft.fft(nearest_neighbors[0])
    
    best_cf = None
    best_pred = original_pred
    best_confidence = 0.0
    
    # Iterative refinement with early stopping
    current_fft = nn_fft.copy()
    
    for step in range(max_steps):
        # Try current FFT
        cf_time = np.fft.ifft(current_fft).real
        cf_tensor = torch.FloatTensor(cf_time).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(cf_tensor).cpu().numpy()[0]
        
        pred_class = np.argmax(pred)
        target_confidence = pred[target_class]
        
        # Update best if this is a valid CF
        if pred_class == target_class:
            if target_confidence > best_confidence:
                best_cf = cf_time.copy()
                best_pred = pred
                best_confidence = target_confidence
                
                if verbose:
                    print(f"  Step {step}: Found CF with confidence {target_confidence:.4f}")
                
                # Early stopping if confidence threshold reached
                if target_confidence >= confidence_threshold:
                    if verbose:
                        print(f"FFT-Confidence: Threshold reached, stopping early")
                    final_cf = _reshape_to_original(best_cf, original_shape)
                    return final_cf, best_pred
        
        # Refinement: blend with other neighbors
        for i in range(min(refine_iterations, max_steps - step)):
            # Select a neighbor to blend with
            neighbor_idx = (step + i) % k
            neighbor_fft = np.fft.fft(nearest_neighbors[neighbor_idx])
            
            # Compute gradient-based saliency
            cf_tensor_grad = torch.FloatTensor(cf_time).unsqueeze(0).unsqueeze(0).to(device).requires_grad_(True)
            output = model(cf_tensor_grad)
            target_output = output[0, target_class]
            target_output.backward()
            
            gradient = cf_tensor_grad.grad.cpu().numpy().flatten()
            gradient_fft = np.fft.fft(gradient)
            saliency = np.abs(gradient_fft)
            
            # Focus on high-saliency frequencies
            top_freqs = np.argsort(saliency)[-len(saliency)//4:]
            
            # Blend only important frequencies
            alpha = 0.3 / (1 + i * 0.1)  # Decreasing blend
            refined_fft = current_fft.copy()
            refined_fft[top_freqs] = (1 - alpha) * current_fft[top_freqs] + alpha * neighbor_fft[top_freqs]
            
            # Test refined version
            cf_time = np.fft.ifft(refined_fft).real
            cf_tensor = torch.FloatTensor(cf_time).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(cf_tensor).cpu().numpy()[0]
            
            pred_class = np.argmax(pred)
            target_confidence = pred[target_class]
            
            if pred_class == target_class and target_confidence > best_confidence:
                best_cf = cf_time.copy()
                best_pred = pred
                best_confidence = target_confidence
                current_fft = refined_fft.copy()
                
                if verbose:
                    print(f"  Refine {i}: Improved to confidence {target_confidence:.4f}")
                
                # Early stopping check
                if target_confidence >= confidence_threshold:
                    if verbose:
                        print(f"FFT-Confidence: Threshold reached during refinement")
                    final_cf = _reshape_to_original(best_cf, original_shape)
                    return final_cf, best_pred
    
    # Return best found
    if best_cf is not None:
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    else:
        final_cf = _reshape_to_original(sample, original_shape)
        return final_cf, original_pred


def fft_hybrid_enhanced_cf(
    sample,
    dataset,
    model,
    target_class,
    k=5,
    analyze_importance=True,
    fallback_on_failure=True,
    verbose=False
):
    """
    Enhanced hybrid amplitude-phase CF with fallback mechanism.
    
    Improves on fft_hybrid_cf by adding fallback to NN-based initialization
    when phase-only modification fails, boosting success rate while maintaining
    the excellent proximity/compactness characteristics.
    
    Args:
        sample: Original time series to explain
        dataset: Dataset containing neighbor samples
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors to consider
        analyze_importance: Whether to analyze amplitude vs phase importance
        fallback_on_failure: Use NN fallback if hybrid fails
        verbose: Print progress information
    
    Returns:
        counterfactual: Generated counterfactual time series
        prediction: Model prediction for the counterfactual
    """
    device = next(model.parameters()).device
    original_shape = sample.shape
    sample = sample.reshape(-1)
    
    # Get initial prediction
    sample_tensor = torch.FloatTensor(sample).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        original_pred = model(sample_tensor).cpu().numpy()[0]
    original_class = np.argmax(original_pred)
    
    if verbose:
        print(f"FFT-Hybrid-Enhanced: Original class {original_class}, target {target_class}")
    
    # Find k nearest neighbors from target class
    target_samples = dataset[dataset[:, -1] == target_class][:, :-1]
    if len(target_samples) == 0:
        return _reshape_to_original(sample, original_shape), original_pred
    
    distances = np.linalg.norm(target_samples - sample.reshape(1, -1), axis=1)
    nearest_indices = np.argsort(distances)[:k]
    nearest_neighbors = target_samples[nearest_indices]
    
    # FFT of sample and nearest neighbor
    sample_fft = np.fft.fft(sample)
    sample_amp = np.abs(sample_fft)
    sample_phase = np.angle(sample_fft)
    
    best_cf = None
    best_pred = original_pred
    best_score = -np.inf
    
    # Strategy 1: Try phase-only modification first (most effective)
    if verbose:
        print("  Strategy 1: Phase-only modification...")
    
    for i, neighbor in enumerate(nearest_neighbors[:3]):  # Try top 3
        neighbor_fft = np.fft.fft(neighbor)
        neighbor_phase = np.angle(neighbor_fft)
        
        # Replace phase, keep amplitude
        hybrid_fft = sample_amp * np.exp(1j * neighbor_phase)
        cf_time = np.fft.ifft(hybrid_fft).real
        
        cf_tensor = torch.FloatTensor(cf_time).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(cf_tensor).cpu().numpy()[0]
        
        pred_class = np.argmax(pred)
        if pred_class == target_class:
            score = pred[target_class]
            if score > best_score:
                best_cf = cf_time
                best_pred = pred
                best_score = score
                if verbose:
                    print(f"    Phase-only CF found with neighbor {i}, confidence: {score:.4f}")
    
    # Strategy 2: Try amplitude-only modification
    if best_cf is None and verbose:
        print("  Strategy 2: Amplitude-only modification...")
    
    if best_cf is None:
        for i, neighbor in enumerate(nearest_neighbors[:3]):
            neighbor_fft = np.fft.fft(neighbor)
            neighbor_amp = np.abs(neighbor_fft)
            
            # Replace amplitude, keep phase
            hybrid_fft = neighbor_amp * np.exp(1j * sample_phase)
            cf_time = np.fft.ifft(hybrid_fft).real
            
            cf_tensor = torch.FloatTensor(cf_time).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(cf_tensor).cpu().numpy()[0]
            
            pred_class = np.argmax(pred)
            if pred_class == target_class:
                score = pred[target_class]
                if score > best_score:
                    best_cf = cf_time
                    best_pred = pred
                    best_score = score
                    if verbose:
                        print(f"    Amplitude-only CF found with neighbor {i}, confidence: {score:.4f}")
    
    # Strategy 3: Adaptive blending with importance analysis
    if best_cf is None and analyze_importance:
        if verbose:
            print("  Strategy 3: Adaptive importance-based blending...")
        
        # Analyze which component (amp/phase) is more important
        neighbor_fft = np.fft.fft(nearest_neighbors[0])
        neighbor_amp = np.abs(neighbor_fft)
        neighbor_phase = np.angle(neighbor_fft)
        
        # Test phase importance
        test_fft_phase = sample_amp * np.exp(1j * neighbor_phase)
        cf_phase = np.fft.ifft(test_fft_phase).real
        cf_tensor = torch.FloatTensor(cf_phase).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_phase = model(cf_tensor).cpu().numpy()[0]
        phase_score = pred_phase[target_class]
        
        # Test amplitude importance
        test_fft_amp = neighbor_amp * np.exp(1j * sample_phase)
        cf_amp = np.fft.ifft(test_fft_amp).real
        cf_tensor = torch.FloatTensor(cf_amp).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_amp = model(cf_tensor).cpu().numpy()[0]
        amp_score = pred_amp[target_class]
        
        # Adaptive blending based on importance
        if phase_score > amp_score:
            # Phase is more important - blend more phase
            alpha_phase = 0.8
            alpha_amp = 0.2
        else:
            # Amplitude is more important - blend more amplitude
            alpha_phase = 0.2
            alpha_amp = 0.8
        
        blended_amp = (1 - alpha_amp) * sample_amp + alpha_amp * neighbor_amp
        blended_phase = (1 - alpha_phase) * sample_phase + alpha_phase * neighbor_phase
        hybrid_fft = blended_amp * np.exp(1j * blended_phase)
        
        cf_time = np.fft.ifft(hybrid_fft).real
        cf_tensor = torch.FloatTensor(cf_time).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(cf_tensor).cpu().numpy()[0]
        
        pred_class = np.argmax(pred)
        if pred_class == target_class:
            best_cf = cf_time
            best_pred = pred
            best_score = pred[target_class]
            if verbose:
                print(f"    Adaptive blend CF found, confidence: {best_score:.4f}")
    
    # Fallback Strategy: Use NN-based approach if hybrid failed
    if best_cf is None and fallback_on_failure:
        if verbose:
            print("  Fallback: Using NN-based approach...")
        
        # Simple NN blend as fallback
        nn_fft = np.fft.fft(nearest_neighbors[0])
        alpha = 0.7
        fallback_fft = (1 - alpha) * sample_fft + alpha * nn_fft
        
        cf_time = np.fft.ifft(fallback_fft).real
        cf_tensor = torch.FloatTensor(cf_time).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(cf_tensor).cpu().numpy()[0]
        
        pred_class = np.argmax(pred)
        if pred_class == target_class:
            best_cf = cf_time
            best_pred = pred
            if verbose:
                print(f"    Fallback CF found, confidence: {pred[target_class]:.4f}")
    
    # Return result
    if best_cf is not None:
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    else:
        final_cf = _reshape_to_original(sample, original_shape)
        return final_cf, original_pred


def fft_band_optimizer_cf(
    sample,
    dataset,
    model,
    target_class,
    k=5,
    num_bands=3,
    use_saliency=True,
    verbose=False
):
    """
    Multi-band frequency optimization with adaptive band-wise NN selection.
    
    Divides frequency spectrum into bands (low/mid/high), analyzes importance
    of each band using gradient saliency, and applies targeted NN blending
    per band for fine-grained control.
    
    Args:
        sample: Original time series to explain
        dataset: Dataset containing neighbor samples
        model: Trained classification model
        target_class: Target class for counterfactual
        k: Number of nearest neighbors to consider
        num_bands: Number of frequency bands (default: 3 for low/mid/high)
        use_saliency: Use gradient saliency to weight bands
        verbose: Print progress information
    
    Returns:
        counterfactual: Generated counterfactual time series
        prediction: Model prediction for the counterfactual
    """
    device = next(model.parameters()).device
    original_shape = sample.shape
    sample = sample.reshape(-1)
    n_samples = len(sample)
    
    # Get initial prediction
    sample_tensor = torch.FloatTensor(sample).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        original_pred = model(sample_tensor).cpu().numpy()[0]
    original_class = np.argmax(original_pred)
    
    if verbose:
        print(f"FFT-Band: Original class {original_class}, target {target_class}")
        print(f"FFT-Band: Using {num_bands} frequency bands")
    
    # Find k nearest neighbors from target class
    target_samples = dataset[dataset[:, -1] == target_class][:, :-1]
    if len(target_samples) == 0:
        return _reshape_to_original(sample, original_shape), original_pred
    
    distances = np.linalg.norm(target_samples - sample.reshape(1, -1), axis=1)
    nearest_indices = np.argsort(distances)[:k]
    nearest_neighbors = target_samples[nearest_indices]
    
    # FFT of sample
    sample_fft = np.fft.fft(sample)
    
    # Define frequency bands
    freq_len = len(sample_fft)
    band_size = freq_len // num_bands
    bands = []
    for i in range(num_bands):
        start = i * band_size
        end = start + band_size if i < num_bands - 1 else freq_len
        bands.append((start, end))
    
    if verbose:
        print(f"  Band ranges: {bands}")
    
    # Compute band importance using gradient saliency
    band_importance = np.ones(num_bands)
    
    if use_saliency:
        sample_tensor_grad = torch.FloatTensor(sample).unsqueeze(0).unsqueeze(0).to(device).requires_grad_(True)
        output = model(sample_tensor_grad)
        target_output = output[0, target_class]
        target_output.backward()
        
        gradient = sample_tensor_grad.grad.cpu().numpy().flatten()
        gradient_fft = np.fft.fft(gradient)
        saliency = np.abs(gradient_fft)
        
        # Compute average saliency per band
        for i, (start, end) in enumerate(bands):
            band_importance[i] = np.mean(saliency[start:end])
        
        # Normalize importance
        band_importance = band_importance / (np.sum(band_importance) + 1e-8)
        
        if verbose:
            print(f"  Band importance: {band_importance}")
    
    # Find best neighbor per band based on frequency-domain distance
    best_neighbors_per_band = []
    
    for band_idx, (start, end) in enumerate(bands):
        # Compute distance in this frequency band only
        band_distances = []
        for neighbor in nearest_neighbors:
            neighbor_fft = np.fft.fft(neighbor)
            band_dist = np.linalg.norm(sample_fft[start:end] - neighbor_fft[start:end])
            band_distances.append(band_dist)
        
        # Select neighbor with minimum distance in this band
        best_idx = np.argmin(band_distances)
        best_neighbors_per_band.append(nearest_neighbors[best_idx])
        
        if verbose:
            print(f"  Band {band_idx}: Using neighbor {best_idx}, distance: {band_distances[best_idx]:.4f}")
    
    # Construct hybrid FFT by blending per band
    hybrid_fft = sample_fft.copy()
    
    for band_idx, (start, end) in enumerate(bands):
        neighbor = best_neighbors_per_band[band_idx]
        neighbor_fft = np.fft.fft(neighbor)
        
        # Blend strength based on band importance
        alpha = 0.3 + 0.6 * band_importance[band_idx]  # Range: 0.3 to 0.9
        
        # Apply band-specific blending
        hybrid_fft[start:end] = (1 - alpha) * sample_fft[start:end] + alpha * neighbor_fft[start:end]
        
        if verbose:
            print(f"  Band {band_idx}: Blend alpha = {alpha:.3f}")
    
    # Convert back to time domain
    cf_time = np.fft.ifft(hybrid_fft).real
    
    # Test if this is a valid counterfactual
    cf_tensor = torch.FloatTensor(cf_time).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(cf_tensor).cpu().numpy()[0]
    
    pred_class = np.argmax(pred)
    
    if pred_class == target_class:
        if verbose:
            print(f"FFT-Band: Success! Confidence: {pred[target_class]:.4f}")
        final_cf = _reshape_to_original(cf_time, original_shape)
        return final_cf, pred
    
    # Refinement: Try adjusting blend ratios
    if verbose:
        print("  Refinement: Adjusting band blend ratios...")
    
    best_cf = None
    best_pred = original_pred
    best_score = 0.0
    
    for iteration in range(10):
        # Try different blend strategies
        hybrid_fft = sample_fft.copy()
        
        for band_idx, (start, end) in enumerate(bands):
            neighbor = best_neighbors_per_band[band_idx]
            neighbor_fft = np.fft.fft(neighbor)
            
            # Increase alpha gradually, weighted by importance
            alpha = min(0.95, 0.4 + 0.1 * iteration + 0.4 * band_importance[band_idx])
            hybrid_fft[start:end] = (1 - alpha) * sample_fft[start:end] + alpha * neighbor_fft[start:end]
        
        cf_time = np.fft.ifft(hybrid_fft).real
        cf_tensor = torch.FloatTensor(cf_time).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(cf_tensor).cpu().numpy()[0]
        
        pred_class = np.argmax(pred)
        if pred_class == target_class:
            score = pred[target_class]
            if score > best_score:
                best_cf = cf_time
                best_pred = pred
                best_score = score
                if verbose:
                    print(f"    Iteration {iteration}: Found CF with confidence {score:.4f}")
    
    # Return best result
    if best_cf is not None:
        final_cf = _reshape_to_original(best_cf, original_shape)
        return final_cf, best_pred
    else:
        final_cf = _reshape_to_original(sample, original_shape)
        return final_cf, original_pred
