import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


####
# Conditional Generative Models for Counterfactual Explanations (CGM)
#
# Paper: Van Looveren, A., Klaise, J., Vacanti, G., & Cobb, O. (2021).
#        "Conditional Generative Models for Counterfactual Explanations"
#        arXiv:2101.10123
#
# Paper URL: https://arxiv.org/abs/2101.10123
#
# This method uses conditional generative models (e.g., conditional VAE/GAN)
# to generate sparse, in-distribution counterfactual explanations. The approach
# generates counterfactuals by conditioning a generative model on the desired
# target prediction, allowing batches of counterfactuals to be generated with
# a single forward pass.
#
# Key features:
# - Uses conditional generative models trained on the dataset
# - Generates in-distribution counterfactuals
# - Optimizes in latent space for efficiency
# - Supports batch generation
# - Maintains sparsity through regularization
####


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


class ConditionalEncoder(nn.Module):
    """Encoder that takes input x and condition c to produce latent z."""
    
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim + condition_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x, c):
        """Encode x conditioned on c."""
        xc = torch.cat([x, c], dim=-1)
        h = self.encoder(xc)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConditionalDecoder(nn.Module):
    """Decoder that takes latent z and condition c to reconstruct x."""
    
    def __init__(self, latent_dim, condition_dim, output_dim, hidden_dims=[64, 128]):
        super().__init__()
        layers = []
        prev_dim = latent_dim + condition_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z, c):
        """Decode z conditioned on c."""
        zc = torch.cat([z, c], dim=-1)
        return self.decoder(zc)


class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder for counterfactual generation."""
    
    def __init__(self, input_dim, num_classes, latent_dim=16, hidden_dims=[128, 64]):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.encoder = ConditionalEncoder(input_dim, num_classes, latent_dim, hidden_dims)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, input_dim, 
                                         hidden_dims=list(reversed(hidden_dims)))
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, c):
        """Forward pass: encode, sample, decode."""
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar
        
    def encode(self, x, c):
        """Encode input with condition."""
        mu, logvar = self.encoder(x, c)
        return mu, logvar
        
    def decode(self, z, c):
        """Decode latent with condition."""
        return self.decoder(z, c)
        
    def generate(self, c, z=None):
        """Generate sample conditioned on c."""
        if z is None:
            z = torch.randn(c.shape[0], self.latent_dim, device=c.device)
        return self.decoder(z, c)


def train_conditional_vae(cvae, dataset, num_classes, num_epochs=50, batch_size=32, 
                         lr=1e-3, device='cpu', verbose=False):
    """Train the conditional VAE on the dataset.
    
    Args:
        cvae: ConditionalVAE model
        dataset: Training dataset (list of (x, y) tuples)
        num_classes: Number of classes
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on
        verbose: Print training progress
        
    Returns:
        cvae: Trained ConditionalVAE model
    """
    cvae.to(device)
    optimizer = Adam(cvae.parameters(), lr=lr)
    
    # Prepare data
    X = []
    Y = []
    for x, y in dataset:
        X.append(x.flatten())
        Y.append(y if isinstance(y, int) else np.argmax(y))
    
    X = np.array(X)
    Y = np.array(Y)
    
    num_samples = len(X)
    
    for epoch in range(num_epochs):
        cvae.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            x_batch = torch.FloatTensor(X[batch_indices]).to(device)
            y_batch = Y[batch_indices]
            
            # One-hot encode labels
            c_batch = torch.zeros(len(y_batch), num_classes, device=device)
            c_batch[range(len(y_batch)), y_batch] = 1.0
            
            # Forward pass
            x_recon, mu, logvar = cvae(x_batch, c_batch)
            
            # Loss: reconstruction + KL divergence
            recon_loss = nn.functional.mse_loss(x_recon, x_batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        if verbose and (epoch + 1) % 10 == 0:
            print(f"CGM Training Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    cvae.eval()
    return cvae


####
# CGM Counterfactual Generation
#
# Generates counterfactuals by:
# 1. Training a conditional VAE on the dataset (if not provided)
# 2. Encoding the input sample
# 3. Optimizing in latent space conditioned on target class
# 4. Decoding to get counterfactual in original space
####
def cgm_generate(sample,
                dataset,
                model,
                target=None,
                latent_dim=16,
                max_iter=200,
                lr=0.01,
                lambda_validity=1.0,
                lambda_proximity=0.5,
                lambda_sparsity=0.01,
                cvae=None,
                train_vae=True,
                verbose=False):
    """Generate counterfactual using Conditional Generative Model.
    
    Args:
        sample: Input time series to generate counterfactual for
        dataset: Training dataset to train conditional VAE
        model: Classifier model
        target: Target class for counterfactual
        latent_dim: Dimensionality of latent space
        max_iter: Maximum optimization iterations
        lr: Learning rate for optimization
        lambda_validity: Weight for validity loss (prediction matching target)
        lambda_proximity: Weight for proximity loss (distance to original)
        lambda_sparsity: Weight for sparsity loss (L1 regularization)
        cvae: Pre-trained conditional VAE (optional)
        train_vae: Whether to train VAE if not provided
        verbose: Print progress information
        
    Returns:
        cf: Generated counterfactual time series
        y_cf: Prediction for counterfactual
    """
    device = next(model.parameters()).device
    
    # Convert sample to proper format
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    original_shape = sample.shape
    
    # Flatten for VAE processing
    if len(sample_tensor.shape) == 1:
        sample_flat = sample_tensor
        input_dim = len(sample_flat)
    elif len(sample_tensor.shape) == 2:
        sample_flat = sample_tensor.flatten()
        input_dim = len(sample_flat)
    else:
        sample_flat = sample_tensor.flatten()
        input_dim = len(sample_flat)
    
    # Get initial prediction
    if len(sample.shape) == 1:
        model_input = sample_tensor.reshape(1, 1, -1)
    elif len(sample.shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            model_input = sample_tensor.T.unsqueeze(0)
        else:
            model_input = sample_tensor.unsqueeze(0)
    else:
        model_input = sample_tensor.unsqueeze(0)
    
    y_orig = detach_to_numpy(model(model_input))[0]
    label_orig = int(np.argmax(y_orig))
    num_classes = len(y_orig)
    
    # Determine target class
    if target is None:
        sorted_indices = np.argsort(y_orig)[::-1]
        target = int(sorted_indices[1])  # Second most likely class
    
    if verbose:
        print(f"CGM: Original class {label_orig}, Target class {target}")
    
    # Initialize or train conditional VAE
    if cvae is None and train_vae:
        if verbose:
            print("CGM: Training conditional VAE...")
        cvae = ConditionalVAE(input_dim, num_classes, latent_dim=latent_dim)
        cvae = train_conditional_vae(cvae, dataset, num_classes, 
                                    num_epochs=50, device=device, verbose=verbose)
    elif cvae is None:
        if verbose:
            print("CGM: No VAE provided and train_vae=False, using simple optimization")
        # Fallback to direct optimization without VAE
        return cgm_generate_simple(sample, dataset, model, target, max_iter, 
                                  lr, lambda_proximity, verbose)
    
    cvae.to(device).eval()
    
    # Encode original sample with its true class
    sample_flat = sample_flat.unsqueeze(0)
    c_orig = torch.zeros(1, num_classes, device=device)
    c_orig[0, label_orig] = 1.0
    
    with torch.no_grad():
        mu_orig, logvar_orig = cvae.encode(sample_flat, c_orig)
    
    # Initialize latent variable for optimization (start from encoded original)
    z_cf = mu_orig.clone().detach().requires_grad_(True)
    
    # Target condition (one-hot encoded target class)
    c_target = torch.zeros(1, num_classes, device=device)
    c_target[0, target] = 1.0
    
    optimizer = Adam([z_cf], lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    best_cf = None
    best_validity = 0.0
    best_loss = float('inf')
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Decode latent with target condition
        x_cf_flat = cvae.decode(z_cf, c_target)
        
        # Reshape for model prediction
        if len(original_shape) == 1:
            x_cf_model = x_cf_flat.reshape(1, 1, -1)
        elif len(original_shape) == 2:
            x_cf_reshaped = x_cf_flat.reshape(original_shape)
            if original_shape[0] > original_shape[1]:
                x_cf_model = x_cf_reshaped.T.unsqueeze(0)
            else:
                x_cf_model = x_cf_reshaped.unsqueeze(0)
        else:
            x_cf_model = x_cf_flat.reshape(1, *original_shape)
        
        # Get prediction
        pred_cf = model(x_cf_model)
        
        # Validity loss: Cross-entropy with target class
        target_tensor = torch.tensor([target], dtype=torch.long, device=device)
        validity_loss = ce_loss_fn(pred_cf, target_tensor)
        
        # Proximity loss: Distance in latent space
        proximity_loss = torch.norm(z_cf - mu_orig, p=2)
        
        # Sparsity loss: L1 norm in latent space
        sparsity_loss = torch.norm(z_cf - mu_orig, p=1)
        
        # Total loss
        loss = (lambda_validity * validity_loss + 
                lambda_proximity * proximity_loss +
                lambda_sparsity * sparsity_loss)
        
        loss.backward()
        optimizer.step()
        
        # Check validity
        with torch.no_grad():
            y_cf = detach_to_numpy(pred_cf)[0]
            pred_class = int(np.argmax(y_cf))
            validity_score = y_cf[target]
            
            # Store best candidate
            if validity_score > best_validity or (validity_score == best_validity and loss.item() < best_loss):
                best_validity = validity_score
                best_loss = loss.item()
                x_cf_best = cvae.decode(z_cf, c_target)
                best_cf = detach_to_numpy(x_cf_best.squeeze(0))
            
            # Debug output
            if verbose and iteration % 50 == 0:
                print(f"CGM iter {iteration}: pred_class={pred_class}, target={target}, "
                      f"validity={validity_score:.4f}, loss={loss.item():.4f}")
            
            # Early stopping if target achieved
            if pred_class == target:
                if verbose:
                    print(f"CGM: Found counterfactual at iteration {iteration}")
                break
    
    if best_cf is None:
        if verbose:
            print("CGM: No valid counterfactual found")
        return None, None
    
    # Reshape back to original format
    best_cf = best_cf.reshape(original_shape)
    
    # Get final prediction
    if len(original_shape) == 1:
        cf_model_input = torch.FloatTensor(best_cf).reshape(1, 1, -1).to(device)
    elif len(original_shape) == 2:
        cf_tensor = torch.FloatTensor(best_cf).to(device)
        if original_shape[0] > original_shape[1]:
            cf_model_input = cf_tensor.T.unsqueeze(0)
        else:
            cf_model_input = cf_tensor.unsqueeze(0)
    else:
        cf_model_input = torch.FloatTensor(best_cf).unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_cf_final = detach_to_numpy(model(cf_model_input))[0]
    
    if verbose:
        final_class = int(np.argmax(y_cf_final))
        print(f"CGM: Final prediction class={final_class}, target={target}, "
              f"confidence={y_cf_final[target]:.4f}")
    
    return best_cf, y_cf_final


####
# Simple CGM without VAE (fallback)
#
# Direct optimization in input space with proximity constraint
####
def cgm_generate_simple(sample,
                       dataset,
                       model,
                       target=None,
                       max_iter=200,
                       lr=0.01,
                       lambda_proximity=0.5,
                       verbose=False):
    """Simple CGM variant without VAE, using direct optimization."""
    device = next(model.parameters()).device
    
    # Convert sample to tensor
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    original_shape = sample.shape
    
    # Get initial prediction
    if len(sample.shape) == 1:
        model_input = sample_tensor.reshape(1, 1, -1)
    elif len(sample.shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            model_input = sample_tensor.T.unsqueeze(0)
        else:
            model_input = sample_tensor.unsqueeze(0)
    else:
        model_input = sample_tensor.unsqueeze(0)
    
    y_orig = detach_to_numpy(model(model_input))[0]
    label_orig = int(np.argmax(y_orig))
    
    if target is None:
        sorted_indices = np.argsort(y_orig)[::-1]
        target = int(sorted_indices[1])
    
    if verbose:
        print(f"CGM Simple: Original class {label_orig}, Target class {target}")
    
    # Initialize counterfactual from a random sample in dataset
    dataset_len = len(dataset)
    ridx = np.random.randint(0, dataset_len)
    x_cf = torch.tensor(dataset[ridx][0], dtype=torch.float32, device=device)
    x_cf = x_cf.flatten().reshape(sample_tensor.flatten().shape)
    x_cf.requires_grad_(True)
    
    optimizer = Adam([x_cf], lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    best_cf = None
    best_validity = 0.0
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Reshape for model
        if len(original_shape) == 1:
            x_cf_model = x_cf.reshape(1, 1, -1)
        elif len(original_shape) == 2:
            x_cf_reshaped = x_cf.reshape(original_shape)
            if original_shape[0] > original_shape[1]:
                x_cf_model = x_cf_reshaped.T.unsqueeze(0)
            else:
                x_cf_model = x_cf_reshaped.unsqueeze(0)
        else:
            x_cf_model = x_cf.reshape(1, *original_shape)
        
        pred_cf = model(x_cf_model)
        
        # Loss
        target_tensor = torch.tensor([target], dtype=torch.long, device=device)
        validity_loss = ce_loss_fn(pred_cf, target_tensor)
        proximity_loss = torch.norm(x_cf - sample_tensor.flatten(), p=2)
        
        loss = validity_loss + lambda_proximity * proximity_loss
        
        loss.backward()
        optimizer.step()
        
        # Check progress
        with torch.no_grad():
            y_cf = detach_to_numpy(pred_cf)[0]
            pred_class = int(np.argmax(y_cf))
            validity_score = y_cf[target]
            
            if validity_score > best_validity:
                best_validity = validity_score
                best_cf = detach_to_numpy(x_cf)
            
            if verbose and iteration % 50 == 0:
                print(f"CGM Simple iter {iteration}: pred_class={pred_class}, "
                      f"validity={validity_score:.4f}")
            
            if pred_class == target:
                if verbose:
                    print(f"CGM Simple: Found counterfactual at iteration {iteration}")
                break
    
    if best_cf is None:
        return None, None
    
    best_cf = best_cf.reshape(original_shape)
    
    # Final prediction
    if len(original_shape) == 1:
        cf_model_input = torch.FloatTensor(best_cf).reshape(1, 1, -1).to(device)
    elif len(original_shape) == 2:
        cf_tensor = torch.FloatTensor(best_cf).to(device)
        if original_shape[0] > original_shape[1]:
            cf_model_input = cf_tensor.T.unsqueeze(0)
        else:
            cf_model_input = cf_tensor.unsqueeze(0)
    else:
        cf_model_input = torch.FloatTensor(best_cf).unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_cf_final = detach_to_numpy(model(cf_model_input))[0]
    
    return best_cf, y_cf_final
