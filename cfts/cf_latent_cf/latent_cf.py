import numpy as np
import torch
import torch.nn as nn


####
# Latent CF: Latent Space Counterfactual Explanations
#
# This is a simple autoencoder-based approach that projects time series into
# latent space, optimizes in the latent space for improved efficiency, then
# projects back to the original space for interpretable counterfactuals.
#
# The method uses standard latent space optimization techniques adapted for
# time series counterfactual generation.
####


def detach_to_numpy(data):
    # move pytorch data to cpu and detach it to numpy data
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    # convert numpy array to pytorch and move it to the device
    return torch.from_numpy(data).float().to(device)


class AutoEncoder(nn.Module):
    """Simple autoencoder for latent space representation."""
    
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)


####
# LatentCF++: Latent Space Counterfactual Generation
#
# Generates counterfactuals by optimizing in the latent space of an autoencoder.
# This approach ensures more realistic counterfactuals by constraining the search
# to the learned manifold of the data distribution.
#
####
def latent_cf_generate(sample,
                       dataset,
                       model,
                       target=None,
                       latent_dim=8,
                       max_iter=100,
                       lr=0.01,
                       lambda_dist=0.1,
                       autoencoder=None,
                       verbose=False):
    """Generate counterfactual using latent space optimization.
    
    Args:
        sample: Input time series to generate counterfactual for
        dataset: Training dataset to train autoencoder (if not provided)
        model: Classifier model
        target: Target class for counterfactual
        latent_dim: Dimensionality of latent space
        max_iter: Maximum optimization iterations
        lr: Learning rate for optimization
        lambda_dist: Weight for distance regularization in latent space
        autoencoder: Pre-trained autoencoder (optional)
        verbose: Print progress information
        
    Returns:
        cf: Generated counterfactual time series
        y_cf: Prediction for counterfactual
    """
    device = next(model.parameters()).device
    
    def model_predict(data):
        # Ensure proper input format for model
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
    
    # Convert sample to proper format
    sample_flat = sample.reshape(-1)
    
    # Get initial prediction
    y_original = model_predict(sample.reshape(sample.shape))[0]
    label_original = np.argmax(y_original)
    
    if target is None:
        # Find the class with second highest probability
        sorted_indices = np.argsort(y_original)[::-1]
        target = int(sorted_indices[1])
    
    if verbose:
        print(f"LatentCF: Original class {label_original}, Target class {target}")
    
    # Train autoencoder if not provided
    if autoencoder is None:
        if verbose:
            print("LatentCF: Training autoencoder...")
        
        # Prepare training data
        X_train = []
        for i in range(len(dataset)):
            x_i = dataset[i][0]
            X_train.append(x_i.reshape(-1))
        X_train = np.array(X_train)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        
        input_dim = sample_flat.shape[0]
        autoencoder = AutoEncoder(input_dim, latent_dim).to(device)
        ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        
        # Train autoencoder
        for epoch in range(100):
            ae_optimizer.zero_grad()
            recon = autoencoder(X_train_tensor)
            loss = nn.MSELoss()(recon, X_train_tensor)
            loss.backward()
            ae_optimizer.step()
            
            if verbose and epoch % 20 == 0:
                print(f"LatentCF: Autoencoder epoch {epoch}, loss={loss.item():.4f}")
    
    autoencoder.eval()
    
    # Convert sample to tensor and encode to latent space
    x_flat = torch.tensor(sample_flat, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        z_original = autoencoder.encode(x_flat)
    
    # Initialize latent variable for optimization
    z = z_original.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    
    best_cf = None
    best_validity = 0.0
    
    if verbose:
        print(f"LatentCF: Starting optimization in latent space...")
    
    # Optimization loop
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Decode latent representation
        cf_flat = autoencoder.decode(z)
        
        # Reshape for model prediction
        cf = cf_flat.reshape(sample.shape)
        
        # Get model prediction
        if len(cf.shape) == 1:
            cf_input = cf.reshape(1, 1, -1)
        elif len(cf.shape) == 2:
            if cf.shape[0] > cf.shape[1]:
                cf_input = cf.T.unsqueeze(0)
            else:
                cf_input = cf.unsqueeze(0)
        else:
            cf_input = cf
        
        pred = model(cf_input)
        
        # Classification loss - maximize target class probability
        target_prob = pred[0, target]
        
        # Distance in latent space
        latent_dist = torch.norm(z - z_original)
        
        # Combined loss
        loss = -target_prob + lambda_dist * latent_dist
        
        loss.backward()
        optimizer.step()
        
        # Check validity
        y_cf = detach_to_numpy(pred)[0]
        current_class = int(np.argmax(y_cf))
        current_validity = y_cf[target]
        
        # Track best validity
        if current_validity > best_validity:
            best_validity = current_validity
            best_cf = detach_to_numpy(cf_flat).reshape(sample.shape)
        
        # Debug output
        if verbose and iteration % 20 == 0:
            print(f"LatentCF iter {iteration}: pred_class={current_class}, target={target}, "
                  f"validity={current_validity:.4f}, loss={loss.item():.4f}, latent_dist={latent_dist.item():.4f}")
        
        # Stop if target class achieved
        if current_class == target:
            if verbose:
                print(f"LatentCF: Found counterfactual at iteration {iteration}")
            best_cf = detach_to_numpy(cf_flat).reshape(sample.shape)
            break
    
    if verbose:
        print(f"LatentCF: Best validity achieved: {best_validity:.4f}")
    
    if best_cf is None:
        if verbose:
            print("LatentCF: No counterfactual found")
        return None, None
    
    # Get final prediction
    y_cf = model_predict(best_cf)
    
    # Convert back to original sample format
    if len(sample.shape) == 1:
        best_cf = best_cf.squeeze()
    elif len(sample.shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            best_cf = best_cf.T if len(best_cf.shape) == 2 else best_cf
    
    return best_cf, y_cf


####
# Helper function to pre-train autoencoder on dataset
####
def train_autoencoder(dataset, latent_dim=8, epochs=100, lr=0.001, device='cpu', verbose=False):
    """Pre-train an autoencoder on the dataset.
    
    Args:
        dataset: Training dataset
        latent_dim: Dimensionality of latent space
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        verbose: Print progress information
        
    Returns:
        autoencoder: Trained autoencoder model
    """
    # Prepare training data
    X_train = []
    for i in range(len(dataset)):
        x_i = dataset[i][0]
        X_train.append(x_i.reshape(-1))
    X_train = np.array(X_train)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    
    input_dim = X_train.shape[1]
    autoencoder = AutoEncoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    
    if verbose:
        print(f"Training autoencoder with input_dim={input_dim}, latent_dim={latent_dim}")
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = autoencoder(X_train_tensor)
        loss = nn.MSELoss()(recon, X_train_tensor)
        loss.backward()
        optimizer.step()
        
        if verbose and epoch % 20 == 0:
            print(f"Autoencoder epoch {epoch}/{epochs}, loss={loss.item():.4f}")
    
    autoencoder.eval()
    
    if verbose:
        print("Autoencoder training complete")
    
    return autoencoder
 