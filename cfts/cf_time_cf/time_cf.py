"""
Time-CF: Shapelet-based Model-agnostic Counterfactual Local Explanations for Time Series

Implementation based on Huang et al. (2024):
"Shapelet-based Model-agnostic Counterfactual Local Explanations for Time Series Classification"

Time-CF leverages shapelets and TimeGAN to provide counterfactual explanations for
arbitrary time series classifiers. The method:
1. Extracts shapelet candidates using Random Shapelet Transform (RST)
2. Sorts shapelets by information gain and selects top N discriminative shapelets
3. Trains TimeGAN on instances from OTHER classes (not the to-be-explained class)
4. Generates M synthetic instances using TimeGAN
5. For each shapelet candidate, crops the same time interval from generated instances
6. Replaces shapelet regions in the original instance with synthetic shapelets
7. Tests if replacement creates valid counterfactual (flips prediction)
8. Returns counterfactual with minimum Hamming distance

Reference:
@article{huang2024timecf,
  title={Shapelet-based Model-agnostic Counterfactual Local Explanations for Time Series Classification},
  author={Huang, Qi and Chen, Wei and B{\"a}ck, Thomas and van Stein, Niki},
  journal={arXiv preprint arXiv:2402.01343},
  year={2024}
}

Links:
- Paper: https://arxiv.org/abs/2402.01343
- arXiv: 2402.01343
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy
from typing import List, Tuple, Optional, Dict
import warnings


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data"""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device"""
    return torch.from_numpy(data).float().to(device)


# ============================================================================
# TimeGAN Implementation
# ============================================================================

class TimeGANModule(nn.Module):
    """
    RNN module for TimeGAN architecture
    
    Based on: Yoon, J., Jarrett, D., & Van der Schaar, M. (2019).
    "Time-series generative adversarial networks."
    """
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation=torch.sigmoid):
        super(TimeGANModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        
        out, hidden = self.gru(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        if self.activation is not None:
            out = self.activation(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device).float()


def train_timegan(data, hidden_dim=24, n_layers=3, n_epochs=100, batch_size=128, device='cpu', verbose=False):
    """
    Train TimeGAN model on the given data
    
    Args:
        data: Training data as numpy array (n_samples, seq_len, n_features)
        hidden_dim: Hidden dimension size
        n_layers: Number of RNN layers
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
        verbose: Print training progress
        
    Returns:
        Trained networks and data shape info
    """
    # Prepare data
    if isinstance(data, list):
        data = np.array(data)
    
    if data.ndim == 2:
        data = data.reshape(data.shape[0], data.shape[1], 1)
    elif data.ndim == 3 and data.shape[1] < data.shape[2]:
        # Reshape (samples, channels, length) -> (samples, length, channels)
        data = np.transpose(data, (0, 2, 1))
    
    n_samples, seq_len, n_features = data.shape
    
    # Create data loader
    data_tensor = torch.FloatTensor(data).to(device)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, n_samples), 
                          shuffle=True, drop_last=True)
    
    actual_batch_size = min(batch_size, n_samples)
    
    # Initialize networks
    embedder = TimeGANModule(n_features, hidden_dim, hidden_dim, n_layers).to(device)
    recovery = TimeGANModule(hidden_dim, n_features, hidden_dim, n_layers).to(device)
    generator = TimeGANModule(n_features, hidden_dim, hidden_dim, n_layers).to(device)
    supervisor = TimeGANModule(hidden_dim, hidden_dim, hidden_dim, n_layers).to(device)
    discriminator = TimeGANModule(hidden_dim, 1, hidden_dim, n_layers, activation=None).to(device)
    
    # Optimizers
    embedder_optimizer = optim.Adam(embedder.parameters(), lr=0.001)
    recovery_optimizer = optim.Adam(recovery.parameters(), lr=0.001)
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    supervisor_optimizer = optim.Adam(supervisor.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    if verbose:
        print("Training TimeGAN...")
    
    # Stage 1: Autoencoder training (embedder + recovery)
    for epoch in range(n_epochs // 3):
        for batch in dataloader:
            X = batch[0]
            
            embedder_optimizer.zero_grad()
            recovery_optimizer.zero_grad()
            
            H, _ = embedder(X)
            H = H.reshape(actual_batch_size, seq_len, hidden_dim)
            X_tilde, _ = recovery(H)
            X_tilde = X_tilde.reshape(actual_batch_size, seq_len, n_features)
            
            E_loss = 10 * torch.sqrt(mse_loss(X, X_tilde) + 1e-8)
            E_loss.backward()
            
            embedder_optimizer.step()
            recovery_optimizer.step()
    
    # Stage 2: Supervised training (supervisor)
    for epoch in range(n_epochs // 3):
        for batch in dataloader:
            X = batch[0]
            
            embedder_optimizer.zero_grad()
            supervisor_optimizer.zero_grad()
            
            H, _ = embedder(X)
            H = H.reshape(actual_batch_size, seq_len, hidden_dim)
            H_hat_supervised, _ = supervisor(H)
            H_hat_supervised = H_hat_supervised.reshape(actual_batch_size, seq_len, hidden_dim)
            
            S_loss = mse_loss(H[:, 1:, :], H_hat_supervised[:, :-1, :])
            S_loss.backward()
            
            embedder_optimizer.step()
            supervisor_optimizer.step()
    
    # Stage 3: Joint training (generator + discriminator)
    for epoch in range(n_epochs // 3):
        for batch in dataloader:
            X = batch[0]
            
            # Random latent codes
            Z = torch.rand(actual_batch_size, seq_len, n_features).to(device)
            
            # Generator forward
            E_hat, _ = generator(Z)
            E_hat = E_hat.reshape(actual_batch_size, seq_len, hidden_dim)
            H_hat, _ = supervisor(E_hat)
            H_hat = H_hat.reshape(actual_batch_size, seq_len, hidden_dim)
            X_hat, _ = recovery(H_hat)
            X_hat = X_hat.reshape(actual_batch_size, seq_len, n_features)
            
            # Discriminator forward
            with torch.no_grad():
                H, _ = embedder(X)
                H = H.reshape(actual_batch_size, seq_len, hidden_dim)
            
            Y_real, _ = discriminator(H)
            Y_fake, _ = discriminator(H_hat.detach())
            Y_fake_e, _ = discriminator(E_hat.detach())
            
            # Discriminator loss and update
            discriminator_optimizer.zero_grad()
            D_loss_real = bce_loss(Y_real.reshape(-1, 1), torch.ones_like(Y_real.reshape(-1, 1)))
            D_loss_fake = bce_loss(Y_fake.reshape(-1, 1), torch.zeros_like(Y_fake.reshape(-1, 1)))
            D_loss_fake_e = bce_loss(Y_fake_e.reshape(-1, 1), torch.zeros_like(Y_fake_e.reshape(-1, 1)))
            D_loss = D_loss_real + D_loss_fake + D_loss_fake_e
            
            if D_loss.item() > 0.15:
                D_loss.backward()
                discriminator_optimizer.step()
            
            # Generator forward again for generator update
            E_hat, _ = generator(Z)
            E_hat = E_hat.reshape(actual_batch_size, seq_len, hidden_dim)
            H_hat, _ = supervisor(E_hat)
            H_hat = H_hat.reshape(actual_batch_size, seq_len, hidden_dim)
            
            Y_fake_g, _ = discriminator(H_hat)
            
            generator_optimizer.zero_grad()
            supervisor_optimizer.zero_grad()
            
            H_for_super, _ = embedder(X)
            H_for_super = H_for_super.reshape(actual_batch_size, seq_len, hidden_dim)
            
            G_loss_U = bce_loss(Y_fake_g.reshape(-1, 1), torch.ones_like(Y_fake_g.reshape(-1, 1)))
            G_loss_S = mse_loss(H_for_super[:, 1:, :], H_hat[:, :-1, :])
            
            X_hat_for_V, _ = recovery(H_hat)
            X_hat_for_V = X_hat_for_V.reshape(actual_batch_size, seq_len, n_features)
            G_loss_V = torch.mean(torch.abs(torch.std(X_hat_for_V, dim=0) - torch.std(X, dim=0) + 1e-8))
            
            G_loss = G_loss_U + 100 * torch.sqrt(G_loss_S + 1e-8) + 100 * G_loss_V
            G_loss.backward()
            
            generator_optimizer.step()
            supervisor_optimizer.step()
    
    if verbose:
        print("TimeGAN training complete")
    
    return embedder, recovery, generator, supervisor, discriminator, (n_features, seq_len)


def generate_synthetic_samples(generator, supervisor, recovery, n_features, seq_len, 
                               hidden_dim, n_samples=32, device='cpu'):
    """Generate synthetic samples using trained TimeGAN"""
    generator.eval()
    supervisor.eval()
    recovery.eval()
    
    with torch.no_grad():
        Z = torch.rand(n_samples, seq_len, n_features).to(device)
        
        E_hat, _ = generator(Z)
        E_hat = E_hat.reshape(n_samples, seq_len, hidden_dim)
        H_hat, _ = supervisor(E_hat)
        H_hat = H_hat.reshape(n_samples, seq_len, hidden_dim)
        X_hat, _ = recovery(H_hat)
        X_hat = X_hat.reshape(n_samples, seq_len, n_features)
        
        return detach_to_numpy(X_hat)


# ============================================================================
# Random Shapelet Transform
# ============================================================================

def zscore_normalize(ts):
    """Z-score normalization of time series"""
    mean = np.mean(ts)
    std = np.std(ts)
    if std == 0:
        return np.zeros_like(ts)
    return (ts - mean) / std


def subsequence_distance(subseq, ts, start, length):
    """Compute normalized Euclidean distance between subsequence and time series segment"""
    if start + length > len(ts):
        return float('inf')
    
    ts_subseq = ts[start:start + length]
    subseq_norm = zscore_normalize(subseq)
    ts_subseq_norm = zscore_normalize(ts_subseq)
    
    return np.sqrt(np.sum((subseq_norm - ts_subseq_norm) ** 2))


def shapelet_transform_distance(shapelet_data, ts):
    """
    Compute minimum distance between shapelet and all subsequences in time series
    
    Args:
        shapelet_data: Dictionary with 'shapelet', 'channel', 'length'
        ts: Time series (channels, length) or (length,)
        
    Returns:
        Minimum distance
    """
    shapelet = shapelet_data['shapelet']
    channel = shapelet_data.get('channel', 0)
    length = shapelet_data['length']
    
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    
    if channel >= ts.shape[0]:
        return float('inf')
    
    ts_channel = ts[channel]
    min_dist = float('inf')
    
    for start in range(len(ts_channel) - length + 1):
        dist = subsequence_distance(shapelet, ts_channel, start, length)
        min_dist = min(min_dist, dist)
    
    return min_dist


def calculate_information_gain(shapelet_data, X_train, y_train):
    """
    Calculate information gain of a shapelet candidate
    
    Args:
        shapelet_data: Dictionary with shapelet info
        X_train: Training data (n_samples, channels, length) or (n_samples, length)
        y_train: Training labels
        
    Returns:
        Information gain value
    """
    # Compute distances for all training samples
    distances = np.array([shapelet_transform_distance(shapelet_data, x) for x in X_train])
    
    # Sort by distance and compute information gain
    sorted_indices = np.argsort(distances)
    
    # Original entropy
    unique_labels, counts = np.unique(y_train, return_counts=True)
    original_entropy = entropy(counts / len(y_train), base=2)
    
    best_gain = 0
    
    # Try different split points
    for split_idx in range(1, len(sorted_indices)):
        left_indices = sorted_indices[:split_idx]
        right_indices = sorted_indices[split_idx:]
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            continue
        
        # Compute entropy for left and right splits
        left_labels = y_train[left_indices]
        right_labels = y_train[right_indices]
        
        left_unique, left_counts = np.unique(left_labels, return_counts=True)
        right_unique, right_counts = np.unique(right_labels, return_counts=True)
        
        left_entropy = entropy(left_counts / len(left_labels), base=2) if len(left_labels) > 0 else 0
        right_entropy = entropy(right_counts / len(right_labels), base=2) if len(right_labels) > 0 else 0
        
        # Weighted entropy
        weighted_entropy = (len(left_labels) / len(y_train)) * left_entropy + \
                          (len(right_labels) / len(y_train)) * right_entropy
        
        # Information gain
        gain = original_entropy - weighted_entropy
        best_gain = max(best_gain, gain)
    
    return best_gain


def extract_random_shapelets(X_train, y_train, n_shapelets=10, min_length=5, 
                            max_length=None, n_candidates=100):
    """
    Extract shapelet candidates using Random Shapelet Transform
    
    Args:
        X_train: Training data (n_samples, channels, length) or (n_samples, length)
        y_train: Training labels
        n_shapelets: Number of top shapelets to return
        min_length: Minimum shapelet length
        max_length: Maximum shapelet length (default: half of series length)
        n_candidates: Number of random candidates to sample
        
    Returns:
        List of top shapelet dictionaries sorted by information gain
    """
    if X_train.ndim == 2:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    n_samples, n_channels, seq_len = X_train.shape
    
    if max_length is None:
        max_length = seq_len // 2
    
    # Randomly sample shapelet candidates
    candidates = []
    
    for _ in range(n_candidates):
        # Random sample, channel, start position, and length
        sample_idx = np.random.randint(0, n_samples)
        channel = np.random.randint(0, n_channels)
        length = np.random.randint(min_length, min(max_length + 1, seq_len))
        start = np.random.randint(0, seq_len - length + 1)
        
        shapelet = X_train[sample_idx, channel, start:start + length]
        
        shapelet_data = {
            'shapelet': shapelet,
            'channel': channel,
            'start': start,
            'length': length,
            'sample_idx': sample_idx
        }
        
        # Calculate information gain
        info_gain = calculate_information_gain(shapelet_data, X_train, y_train)
        shapelet_data['information_gain'] = info_gain
        
        candidates.append(shapelet_data)
    
    # Sort by information gain and return top N
    candidates.sort(key=lambda x: x['information_gain'], reverse=True)
    
    return candidates[:n_shapelets]


def find_shapelet_position(shapelet_data, ts):
    """
    Find the best matching position of shapelet in time series
    
    Args:
        shapelet_data: Dictionary with 'shapelet', 'channel', 'length'
        ts: Time series (channels, length) or (length,)
        
    Returns:
        (start_position, min_distance)
    """
    shapelet = shapelet_data['shapelet']
    channel = shapelet_data.get('channel', 0)
    length = shapelet_data['length']
    
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    
    if channel >= ts.shape[0]:
        return (0, float('inf'))
    
    ts_channel = ts[channel]
    min_dist = float('inf')
    best_start = 0
    
    for start in range(len(ts_channel) - length + 1):
        dist = subsequence_distance(shapelet, ts_channel, start, length)
        if dist < min_dist:
            min_dist = dist
            best_start = start
    
    return (best_start, min_dist)


def crop_synthetic_shapelet(synthetic_sample, start, length, channel=0):
    """
    Crop a shapelet from synthetic sample at the same position
    
    Args:
        synthetic_sample: Synthetic time series (length, features) or (channels, length)
        start: Start position
        length: Shapelet length
        channel: Channel index
        
    Returns:
        Cropped shapelet
    """
    if synthetic_sample.ndim == 2 and synthetic_sample.shape[0] > synthetic_sample.shape[1]:
        # (length, features) format - transpose
        synthetic_sample = synthetic_sample.T
    
    if synthetic_sample.ndim == 1:
        synthetic_sample = synthetic_sample.reshape(1, -1)
    
    if channel >= synthetic_sample.shape[0]:
        channel = 0
    
    if start + length > synthetic_sample.shape[1]:
        return synthetic_sample[channel, -length:]
    
    return synthetic_sample[channel, start:start + length]


# ============================================================================
# Time-CF Main Algorithm
# ============================================================================

def time_cf_generate(sample, dataset, model, target_class=None,
                    n_shapelets=10, M=32, min_shapelet_length=5,
                    max_shapelet_length=None, n_shapelet_candidates=100,
                    timegan_epochs=100, timegan_batch_size=128,
                    timegan_hidden_dim=24, timegan_n_layers=3,
                    device=None, verbose=False):
    """
    Generate counterfactual using Time-CF algorithm.
    
    This implements the full Time-CF algorithm from the paper:
    1. Extract shapelet candidates using Random Shapelet Transform
    2. Sort by information gain, keep top N
    3. Train TimeGAN on instances from other classes (not to-be-explained class)
    4. Generate M fake instances using TimeGAN
    5. For each shapelet candidate, find its position in the original instance
    6. Crop the same interval from each generated fake instance
    7. Replace the shapelet region in original with fake shapelets
    8. Test if replacement creates valid counterfactual
    9. Return counterfactual with minimum Hamming distance
    
    Args:
        sample: Time series instance to explain (channels, length) or (length,)
        dataset: Training dataset for shapelet extraction and TimeGAN training
        model: Trained classifier model
        target_class: Target class (optional)
        n_shapelets: Number of top shapelets to use (N in paper)
        M: Number of synthetic instances to generate
        min_shapelet_length: Minimum shapelet length
        max_shapelet_length: Maximum shapelet length
        n_shapelet_candidates: Number of random candidates for RST
        timegan_epochs: Number of TimeGAN training epochs
        timegan_batch_size: Batch size for TimeGAN
        timegan_hidden_dim: Hidden dimension for TimeGAN
        timegan_n_layers: Number of RNN layers for TimeGAN
        device: Device to run on
        verbose: Print progress information
        
    Returns:
        Tuple of (counterfactual, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare sample
    sample_orig = sample.copy()
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    n_channels, seq_len = sample.shape
    
    if max_shapelet_length is None:
        max_shapelet_length = seq_len // 2
    
    # Get original prediction
    def predict_sample(ts):
        if ts.ndim == 1:
            ts = ts.reshape(1, -1)
        ts_tensor = torch.tensor(ts, dtype=torch.float32, device=device)
        if len(ts_tensor.shape) == 2:
            ts_tensor = ts_tensor.unsqueeze(0)
        with torch.no_grad():
            pred = model(ts_tensor)
            proba = torch.softmax(pred, dim=-1).squeeze().cpu().numpy()
        return np.argmax(proba), proba
    
    original_class, original_proba = predict_sample(sample)
    
    # Determine target class
    if target_class is None:
        sorted_indices = np.argsort(original_proba)[::-1]
        target_class = int(sorted_indices[1])
    
    if original_class == target_class:
        if verbose:
            print("Time-CF: Sample already in target class")
        return None, None
    
    if verbose:
        print(f"Time-CF: Original class={original_class} (p={original_proba[original_class]:.3f}), "
              f"Target class={target_class} (p={original_proba[target_class]:.3f})")
    
    # Extract training data from dataset
    X_train, y_train = [], []
    for i in range(min(len(dataset), 1000)):
        try:
            item = dataset[i]
            ts, label = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, 0)
            ts_np = np.array(ts)
            if ts_np.ndim == 1:
                ts_np = ts_np.reshape(1, -1)
            elif ts_np.ndim == 3:
                ts_np = ts_np.squeeze(0)
            X_train.append(ts_np)
            
            # Convert label to scalar
            if hasattr(label, 'shape') and len(label.shape) > 0:
                label_scalar = int(np.argmax(label))
            else:
                label_scalar = int(label)
            y_train.append(label_scalar)
        except:
            continue
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    if X_train.ndim == 2:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    if verbose:
        print(f"Time-CF: Loaded {len(X_train)} training samples")
    
    # Step 1: Extract shapelet candidates using Random Shapelet Transform
    if verbose:
        print(f"Time-CF: Extracting {n_shapelets} shapelets from {n_shapelet_candidates} candidates...")
    
    shapelets = extract_random_shapelets(
        X_train, y_train, n_shapelets=n_shapelets,
        min_length=min_shapelet_length, max_length=max_shapelet_length,
        n_candidates=n_shapelet_candidates
    )
    
    if verbose:
        top_gains = [f"{s['information_gain']:.4f}" for s in shapelets[:3]]
        print(f"Time-CF: Top shapelet information gains: {top_gains}")
    
    # Step 2: Train TimeGAN on instances from OTHER classes
    # Filter data: keep only instances NOT from original class
    other_class_mask = y_train != original_class
    X_other = X_train[other_class_mask]
    
    if len(X_other) < 10:
        if verbose:
            print("Time-CF: Warning - Not enough samples from other classes, using all data")
        X_other = X_train
    
    if verbose:
        print(f"Time-CF: Training TimeGAN on {len(X_other)} instances from other classes...")
    
    timegan_result = train_timegan(
        X_other, hidden_dim=timegan_hidden_dim, n_layers=timegan_n_layers,
        n_epochs=timegan_epochs, batch_size=timegan_batch_size,
        device=device, verbose=verbose
    )
    
    embedder, recovery, generator, supervisor, discriminator, (n_features, seq_len_gan) = timegan_result
    
    # Step 3: Generate M fake instances using TimeGAN
    if verbose:
        print(f"Time-CF: Generating {M} synthetic instances...")
    
    synthetic_samples = generate_synthetic_samples(
        generator, supervisor, recovery, n_features, seq_len_gan,
        timegan_hidden_dim, n_samples=M, device=device
    )
    
    # Step 4-8: For each shapelet, crop from fake instances and replace in original
    if verbose:
        print("Time-CF: Testing counterfactual candidates...")
    
    counterfactual_candidates = []
    
    for shapelet_info in shapelets:
        # Find position of shapelet in original instance
        start, min_dist = find_shapelet_position(shapelet_info, sample)
        length = shapelet_info['length']
        channel = shapelet_info.get('channel', 0)
        
        # For each synthetic sample
        for synth_sample in synthetic_samples:
            # Crop synthetic shapelet from same interval
            fake_shapelet = crop_synthetic_shapelet(synth_sample, start, length, channel)
            
            # Replace shapelet in original instance
            cf_candidate = sample.copy()
            if start + length <= cf_candidate.shape[1]:
                cf_candidate[channel, start:start + length] = fake_shapelet
            
            # Test if valid counterfactual
            cf_class, cf_proba = predict_sample(cf_candidate)
            
            if cf_class == target_class:
                # Compute Hamming distance (number of different time steps)
                hamming_dist = np.sum(np.abs(cf_candidate - sample) > 1e-6)
                
                counterfactual_candidates.append({
                    'cf': cf_candidate,
                    'proba': cf_proba,
                    'hamming': hamming_dist,
                    'shapelet_info': shapelet_info
                })
                
                if verbose:
                    print(f"  Found CF: Hamming distance={hamming_dist}, "
                          f"target_prob={cf_proba[target_class]:.4f}")
    
    if len(counterfactual_candidates) == 0:
        if verbose:
            print("Time-CF: No valid counterfactual found")
        return None, None
    
    # Step 9: Return counterfactual with minimum Hamming distance
    counterfactual_candidates.sort(key=lambda x: x['hamming'])
    best_cf = counterfactual_candidates[0]
    
    if verbose:
        print(f"Time-CF: Best counterfactual - Hamming distance: {best_cf['hamming']}, "
              f"Target prob: {best_cf['proba'][target_class]:.4f}")
        print(f"  Using shapelet with info gain: {best_cf['shapelet_info']['information_gain']:.4f}")
    
    # Return in original shape
    cf_result = best_cf['cf']
    if sample_orig.ndim == 1:
        cf_result = cf_result.squeeze()
    
    return cf_result, best_cf['proba']


def time_cf_explain(sample, dataset, model, target_class=None,
                   device=None, verbose=False, **kwargs):
    """
    Generate Time-CF explanation with detailed information
    
    Args:
        sample: Time series to explain
        dataset: Training dataset
        model: Classifier model
        target_class: Target class
        device: Device to use
        verbose: Print details
        **kwargs: Additional arguments for time_cf_generate
        
    Returns:
        Dictionary with counterfactual, prediction, and explanation details
    """
    cf, cf_pred = time_cf_generate(
        sample, dataset, model, target_class,
        device=device, verbose=verbose, **kwargs
    )
    
    # Get original prediction
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)
    elif len(sample_tensor.shape) == 2:
        sample_tensor = sample_tensor.unsqueeze(0)
    
    with torch.no_grad():
        original_pred = model(sample_tensor)
        original_proba = torch.softmax(original_pred, dim=-1).squeeze().cpu().numpy()
        original_class = np.argmax(original_proba)
    
    if target_class is None:
        sorted_classes = np.argsort(original_proba)[::-1]
        target_class = int(sorted_classes[1])
    
    explanation = {
        'counterfactual': cf,
        'prediction': cf_pred,
        'original_class': original_class,
        'target_class': target_class,
        'success': cf is not None,
        'distance': np.linalg.norm(cf - sample) if cf is not None else None,
        'hamming_distance': np.sum(np.abs(cf - sample) > 1e-6) if cf is not None else None
    }
    
    return explanation


# Aliases for compatibility
timecf_generate = time_cf_generate
timecf_explain = time_cf_explain
