"""
Time-CF: TimeGAN and Shapelet-based Counterfactual Explanations for Time Series

This implementation uses TimeGAN to generate synthetic time series samples and
shapelet transformations to identify and replace discriminative segments for
generating counterfactuals.

Related work on TimeGAN:
Yoon, J., Jarrett, D., & Van der Schaar, M. (2019).
"Time-series generative adversarial networks."
Advances in Neural Information Processing Systems, 32

This is a custom implementation combining TimeGAN generative modeling with
shapelet-based feature extraction for counterfactual generation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data"""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device"""
    return torch.from_numpy(data).float().to(device)


class TimeGANModule(nn.Module):
    """
    RNN module for TimeGAN architecture (Embedder, Recovery, Generator, Supervisor, Discriminator)
    """
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation=torch.sigmoid):
        super(TimeGANModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        
        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden=None):
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        
        # GRU forward pass
        out, hidden = self.gru(x, hidden)
        
        # Reshape for FC layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        # Apply activation
        if self.activation is not None:
            out = self.activation(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device).float()


def extract_time(data):
    """Extract time information from data"""
    time_steps = []
    max_seq_len = 0
    
    for sample in data:
        seq_len = sample.shape[0] if sample.ndim == 2 else sample.shape[1]
        time_steps.append(seq_len)
        max_seq_len = max(max_seq_len, seq_len)
    
    return time_steps, max_seq_len


def random_generator(batch_size, z_dim, time_steps, max_seq_len):
    """Generate random noise for TimeGAN generator"""
    Z = []
    for i in range(batch_size):
        t = time_steps[i] if isinstance(time_steps, list) else time_steps
        temp_Z = np.random.uniform(0, 1, [t, z_dim])
        Z.append(temp_Z)
    return Z


def train_timegan(data, hidden_dim=24, n_layers=3, n_epochs=100, batch_size=128, device='cpu'):
    """
    Train TimeGAN model on the given data
    
    Args:
        data: Training data as numpy array or list of arrays
        hidden_dim: Hidden dimension size
        n_layers: Number of RNN layers
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
        
    Returns:
        Trained networks: embedder, recovery, generator, supervisor, discriminator
        And data shape info: (n_features, seq_len)
    """
    # Convert data to tensor
    if isinstance(data, list):
        data = np.array(data)
    
    if data.ndim == 2:
        # Reshape (samples, length) -> (samples, length, 1)
        data = data.reshape(data.shape[0], data.shape[1], 1)
    elif data.ndim == 3 and data.shape[1] > data.shape[2]:
        # Reshape (samples, channels, length) -> (samples, length, channels)
        data = np.transpose(data, (0, 2, 1))
    
    n_samples, seq_len, n_features = data.shape
    
    # Create data loader
    data_tensor = torch.FloatTensor(data).to(device)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
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
    
    # Training loop - simplified version
    print("Training TimeGAN...")
    
    # Stage 1: Autoencoder training
    for epoch in range(n_epochs // 3):
        for batch in dataloader:
            X = batch[0]
            
            # Embedder + Recovery
            embedder_optimizer.zero_grad()
            recovery_optimizer.zero_grad()
            
            H, _ = embedder(X)
            H = H.reshape(batch_size, seq_len, hidden_dim)
            X_tilde, _ = recovery(H)
            X_tilde = X_tilde.reshape(batch_size, seq_len, n_features)
            
            E_loss = 10 * torch.sqrt(mse_loss(X, X_tilde))
            E_loss.backward()
            
            embedder_optimizer.step()
            recovery_optimizer.step()
    
    # Stage 2: Supervised training
    for epoch in range(n_epochs // 3):
        for batch in dataloader:
            X = batch[0]
            
            embedder_optimizer.zero_grad()
            supervisor_optimizer.zero_grad()
            
            H, _ = embedder(X)
            H = H.reshape(batch_size, seq_len, hidden_dim)
            H_hat_supervised, _ = supervisor(H)
            H_hat_supervised = H_hat_supervised.reshape(batch_size, seq_len, hidden_dim)
            
            S_loss = mse_loss(H[:, 1:, :], H_hat_supervised[:, :-1, :])
            S_loss.backward()
            
            embedder_optimizer.step()
            supervisor_optimizer.step()
    
    # Stage 3: Joint training
    for epoch in range(n_epochs // 3):
        for batch in dataloader:
            X = batch[0]
            
            # Random latent codes
            Z = torch.rand(batch_size, seq_len, n_features).to(device)
            
            # Generator forward
            E_hat, _ = generator(Z)
            E_hat = E_hat.reshape(batch_size, seq_len, hidden_dim)
            H_hat, _ = supervisor(E_hat)
            H_hat = H_hat.reshape(batch_size, seq_len, hidden_dim)
            X_hat, _ = recovery(H_hat)
            X_hat = X_hat.reshape(batch_size, seq_len, n_features)
            
            # Discriminator forward - compute embeddings
            with torch.no_grad():
                H, _ = embedder(X)
                H = H.reshape(batch_size, seq_len, hidden_dim)
            
            Y_real, _ = discriminator(H)
            Y_fake, _ = discriminator(H_hat.detach())
            Y_fake_e, _ = discriminator(E_hat.detach())
            
            # Discriminator loss
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
            E_hat = E_hat.reshape(batch_size, seq_len, hidden_dim)
            H_hat, _ = supervisor(E_hat)
            H_hat = H_hat.reshape(batch_size, seq_len, hidden_dim)
            
            # Discriminator forward for generator loss
            Y_fake_g, _ = discriminator(H_hat)
            
            generator_optimizer.zero_grad()
            supervisor_optimizer.zero_grad()
            
            # Compute H with gradients for supervised loss
            H_for_super, _ = embedder(X)
            H_for_super = H_for_super.reshape(batch_size, seq_len, hidden_dim)
            
            G_loss_U = bce_loss(Y_fake_g.reshape(-1, 1), torch.ones_like(Y_fake_g.reshape(-1, 1)))
            G_loss_S = mse_loss(H_for_super[:, 1:, :], H_hat[:, :-1, :])
            
            # Recovery loss
            X_hat_for_V, _ = recovery(H_hat)
            X_hat_for_V = X_hat_for_V.reshape(batch_size, seq_len, n_features)
            G_loss_V = torch.mean(torch.abs(torch.std(X_hat_for_V, dim=0) - torch.std(X, dim=0)))
            
            G_loss = G_loss_U + 100 * torch.sqrt(G_loss_S + 1e-8) + 100 * G_loss_V
            G_loss.backward()
            
            generator_optimizer.step()
            supervisor_optimizer.step()
    
    print("TimeGAN training complete")
    
    return embedder, recovery, generator, supervisor, discriminator, (n_features, seq_len)


def generate_synthetic_samples(generator, supervisor, recovery, n_features, seq_len, 
                               hidden_dim, n_samples=32, device='cpu'):
    """Generate synthetic samples using trained TimeGAN"""
    generator.eval()
    supervisor.eval()
    recovery.eval()
    
    with torch.no_grad():
        # Generate random noise with correct dimensions
        Z = torch.rand(n_samples, seq_len, n_features).to(device)
        
        # Pass through generator pipeline
        E_hat, _ = generator(Z)
        E_hat = E_hat.reshape(n_samples, seq_len, hidden_dim)
        H_hat, _ = supervisor(E_hat)
        H_hat = H_hat.reshape(n_samples, seq_len, hidden_dim)
        X_hat, _ = recovery(H_hat)
        X_hat = X_hat.reshape(n_samples, seq_len, n_features)
        
        # Transpose back to (n_samples, n_features, seq_len)
        X_hat = X_hat.transpose(1, 2)
        
        return detach_to_numpy(X_hat)


def extract_shapelets_gradient(model, x, device, n_shapelets=10, min_length=5, max_length=None):
    """
    Extract discriminative shapelets using gradient-based importance
    
    Args:
        model: Classifier model
        x: Input time series (channels, length)
        device: Device to run on
        n_shapelets: Number of shapelets to extract
        min_length: Minimum shapelet length
        max_length: Maximum shapelet length
        
    Returns:
        List of shapelet info dictionaries
    """
    n_channels, seq_len = x.shape
    if max_length is None:
        max_length = seq_len // 2
    
    # Compute gradient-based importance
    x_input = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    x_tensor = torch.autograd.Variable(x_input, requires_grad=True)
    
    # Handle different model input formats
    if x_tensor.dim() == 2:
        x_tensor = x_tensor.unsqueeze(0)
    
    model.eval()
    output = model(x_tensor)
    pred_class = output.argmax(dim=1).item()
    
    # Compute gradients
    model.zero_grad()
    loss = output[0, pred_class]
    loss.backward()
    
    # Get gradients - handle case where grad might be None
    if x_tensor.grad is not None:
        importance_map = torch.abs(x_tensor.grad).squeeze().cpu().numpy()
    else:
        # Fallback to uniform importance if gradients not available
        importance_map = np.ones((n_channels, seq_len))
    
    # Ensure importance_map has correct shape
    if importance_map.ndim == 1:
        importance_map = importance_map.reshape(1, -1)
    
    # Extract shapelets
    shapelets = []
    for length in range(min_length, min(max_length + 1, seq_len)):
        for start in range(seq_len - length + 1):
            if n_channels == 1:
                shapelet = x[0, start:start+length]
                importance = np.mean(importance_map[0, start:start+length])
                shapelets.append({
                    'channel': 0,
                    'start': start,
                    'length': length,
                    'importance': importance,
                    'shapelet': shapelet
                })
            else:
                for ch in range(n_channels):
                    shapelet = x[ch, start:start+length]
                    importance = np.mean(importance_map[ch, start:start+length])
                    shapelets.append({
                        'channel': ch,
                        'start': start,
                        'length': length,
                        'importance': importance,
                        'shapelet': shapelet
                    })
    
    # Sort by importance and return top shapelets
    shapelets.sort(key=lambda s: s['importance'], reverse=True)
    return shapelets[:n_shapelets]


def time_cf_generate(sample,
                     model,
                     X_train=None,
                     y_train=None,
                     target=None,
                     hidden_dim=24,
                     n_layers=3,
                     n_epochs=50,
                     batch_size=128,
                     n_shapelets=10,
                     n_synthetic=32,
                     verbose=False):
    """
    Generate counterfactual using TimeGAN and shapelet-based approach
    
    Args:
        sample: Original time series to explain (channels, length) or (length,)
        model: Trained classifier model
        X_train: Training data for TimeGAN (optional)
        y_train: Training labels (optional)
        target: Target class (optional, will use second most likely if not provided)
        hidden_dim: Hidden dimension for TimeGAN
        n_layers: Number of RNN layers for TimeGAN
        n_epochs: Number of training epochs for TimeGAN
        batch_size: Batch size for TimeGAN training
        n_shapelets: Number of shapelets to extract
        n_synthetic: Number of synthetic samples to generate
        verbose: Print progress information
        
    Returns:
        counterfactual: Generated counterfactual (same shape as sample)
        prediction: Model prediction on counterfactual
    """
    device = next(model.parameters()).device
    
    # Ensure proper shape
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    n_channels, seq_len = sample.shape
    
    # Get initial prediction
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
    y_pred = detach_to_numpy(model(sample_tensor))[0]
    original_class = np.argmax(y_pred)
    
    # Determine target class
    if target is None:
        sorted_indices = np.argsort(y_pred)[::-1]
        target = int(sorted_indices[1])
    
    if verbose:
        print(f"Time-CF: Original class {original_class}, Target class {target}")
    
    # Train TimeGAN if training data provided
    if X_train is not None and y_train is not None:
        if verbose:
            print("Training TimeGAN on provided data...")
        
        # Filter data by target class
        target_mask = y_train == target
        X_target = X_train[target_mask] if target_mask.sum() > 0 else X_train
        
        result = train_timegan(
            X_target, hidden_dim, n_layers, n_epochs, 
            min(batch_size, len(X_target)), device
        )
        embedder, recovery, generator, supervisor, discriminator, (n_features, seq_len_gan) = result
        
        # Generate synthetic samples
        if verbose:
            print("Generating synthetic samples...")
        synthetic_samples = generate_synthetic_samples(
            generator, supervisor, recovery, n_features, seq_len_gan, 
            hidden_dim, n_synthetic, device
        )
    else:
        # Generate random perturbations if no training data
        if verbose:
            print("No training data provided, using random perturbations...")
        synthetic_samples = sample + np.random.normal(0, 0.1, (n_synthetic, n_channels, seq_len))
    
    # Extract important shapelets
    if verbose:
        print("Extracting shapelets...")
    shapelets = extract_shapelets_gradient(model, sample, device, n_shapelets)
    
    # Generate counterfactuals by replacing shapelets
    if verbose:
        print("Generating counterfactuals...")
    
    candidates = []
    
    for shapelet_info in shapelets:
        start = shapelet_info['start']
        length = shapelet_info['length']
        channel = shapelet_info.get('channel', 0)
        
        for synth_sample in synthetic_samples:
            cf_candidate = sample.copy()
            
            # Replace shapelet segment
            if channel < synth_sample.shape[0] and start + length <= synth_sample.shape[1]:
                cf_candidate[channel, start:start+length] = synth_sample[channel, start:start+length]
            
            # Check if valid counterfactual
            cf_tensor = torch.tensor(cf_candidate, dtype=torch.float32, device=device).unsqueeze(0)
            cf_pred = detach_to_numpy(model(cf_tensor))[0]
            cf_class = np.argmax(cf_pred)
            
            if cf_class == target:
                distance = np.linalg.norm(cf_candidate - sample)
                candidates.append((cf_candidate, cf_pred, distance))
                
                if verbose:
                    print(f"  Found valid CF with distance {distance:.4f}")
    
    if not candidates:
        if verbose:
            print("Time-CF: No valid counterfactual found")
        return None, None
    
    # Return the closest valid counterfactual
    candidates.sort(key=lambda x: x[2])
    best_cf, best_pred, best_dist = candidates[0]
    
    if verbose:
        print(f"Time-CF: Best counterfactual distance: {best_dist:.4f}")
    
    return best_cf, best_pred


def time_cf_batch(samples, model, X_train=None, y_train=None, target=None, **kwargs):
    """
    Generate counterfactuals for a batch of samples
    
    Args:
        samples: Batch of time series (batch_size, channels, length)
        model: Trained classifier model
        X_train: Training data for TimeGAN
        y_train: Training labels
        target: Target class (optional)
        **kwargs: Additional arguments for time_cf_generate
        
    Returns:
        counterfactuals: List of generated counterfactuals
        predictions: List of predictions for counterfactuals
    """
    counterfactuals = []
    predictions = []
    
    for i, sample in enumerate(samples):
        cf, pred = time_cf_generate(
            sample, model, X_train, y_train, target, 
            verbose=kwargs.get('verbose', False)
        )
        counterfactuals.append(cf)
        predictions.append(pred)
    
    return counterfactuals, predictions
