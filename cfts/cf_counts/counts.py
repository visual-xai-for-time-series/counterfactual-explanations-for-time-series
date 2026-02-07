import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


####
# CounTS: Counterfactual Time Series
#
# Paper: "Self-Interpretable Time Series Prediction with Counterfactual Explanations"
#        arXiv:2306.06024
#
# A variational Bayesian deep learning model that generates actionable and causally
# plausible explanations for time series predictions. Unlike post-hoc methods, CounTS
# is a self-interpretable model built on a structural causal model (SCM).
#
# The model performs counterfactual reasoning through three steps:
# 1. Abduction: Estimating posterior distribution of latent factors given observation
# 2. Action: Applying do-intervention to the time series or underlying factors
# 3. Prediction: Generating counterfactual outcome based on modified factors
####


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


class CounTSEncoder(nn.Module):
    """
    Encoder network q_phi(z | x, y) for CounTS.
    Maps input time series x and outcome y to latent space z.
    Uses LSTM to handle temporal dependencies.
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # LSTM for processing time series
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Outcome embedding
        self.y_embedding = nn.Linear(num_classes, hidden_dim)
        
        # Map to latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim * 2 + hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 + hidden_dim, latent_dim)
        
    def forward(self, x, y):
        """
        Args:
            x: Input time series (batch, seq_len, input_dim) or (batch, input_dim, seq_len)
            y: Outcome as one-hot or probabilities (batch, num_classes)
        Returns:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)
        """
        # Handle different input formats
        # Expected: (batch, seq_len, input_dim)
        if len(x.shape) == 2:
            # (batch, seq_len) -> (batch, seq_len, 1) for univariate
            x = x.unsqueeze(-1)
        elif len(x.shape) == 3:
            # Ensure x is (batch, seq_len, input_dim) for LSTM
            if x.shape[1] < x.shape[2]:
                x = x.transpose(1, 2)
        
        # Process time series through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state (concatenate forward and backward)
        h_combined = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, hidden_dim * 2)
        
        # Embed outcome
        y_emb = self.y_embedding(y)  # (batch, hidden_dim)
        
        # Combine time series representation and outcome
        combined = torch.cat([h_combined, y_emb], dim=1)
        
        # Compute latent distribution parameters
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar


class CounTSDecoder(nn.Module):
    """
    Decoder network p_theta(x | z) for CounTS.
    Reconstructs time series from latent representation z.
    """
    
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Map latent to initial hidden state
        self.fc_z = nn.Linear(latent_dim, hidden_dim)
        
        # LSTM for generating time series
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        """
        Args:
            z: Latent representation (batch, latent_dim)
        Returns:
            x_recon: Reconstructed time series (batch, seq_len, output_dim)
        """
        batch_size = z.shape[0]
        
        # Initialize hidden state from latent
        h_0 = torch.tanh(self.fc_z(z)).unsqueeze(0)  # (1, batch, hidden_dim)
        c_0 = torch.zeros_like(h_0)
        
        # Repeat latent as input for each time step
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, latent_dim)
        
        # Generate sequence
        lstm_out, _ = self.lstm(z_repeated, (h_0, c_0))
        
        # Map to output dimension
        x_recon = self.fc_out(lstm_out)  # (batch, seq_len, output_dim)
        
        return x_recon


class CounTSPredictor(nn.Module):
    """
    Predictor network h_omega(y | z) for CounTS.
    Predicts outcome from latent representation.
    """
    
    def __init__(self, latent_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, z):
        """
        Args:
            z: Latent representation (batch, latent_dim)
        Returns:
            y_pred: Predicted outcome logits (batch, num_classes)
        """
        return self.fc(z)


class CounTSModel(nn.Module):
    """
    Complete CounTS model combining encoder, decoder, and predictor.
    Implements VAE with causal structure for counterfactual generation.
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, seq_len):
        super().__init__()
        self.encoder = CounTSEncoder(input_dim, hidden_dim, latent_dim, num_classes)
        self.decoder = CounTSDecoder(latent_dim, hidden_dim, input_dim, seq_len)
        self.predictor = CounTSPredictor(latent_dim, hidden_dim, num_classes)
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        z = mu + eps * sigma where eps ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, y):
        """
        Forward pass through complete model.
        
        Args:
            x: Input time series
            y: Outcome (one-hot or probabilities)
        Returns:
            x_recon: Reconstructed time series
            y_pred: Predicted outcome
            mu: Latent mean
            logvar: Latent log variance
            z: Sampled latent representation
        """
        # Encode
        mu, logvar = self.encoder(x, y)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode and predict
        x_recon = self.decoder(z)
        y_pred = self.predictor(z)
        
        return x_recon, y_pred, mu, logvar, z
        
    def encode(self, x, y):
        """Encode input to latent distribution parameters."""
        return self.encoder(x, y)
        
    def decode(self, z):
        """Decode latent to time series."""
        return self.decoder(z)
        
    def predict(self, z):
        """Predict outcome from latent."""
        return self.predictor(z)


def counts_vae_loss(x, x_recon, y, y_pred, mu, logvar, 
                    beta_recon=1.0, beta_pred=1.0, beta_kl=1.0):
    """
    Combined loss function for training CounTS model.
    
    Loss = Reconstruction Loss + Prediction Loss + KL Divergence
    
    Args:
        x: Original input
        x_recon: Reconstructed input
        y: True outcome
        y_pred: Predicted outcome
        mu: Latent mean
        logvar: Latent log variance
        beta_recon: Weight for reconstruction loss
        beta_pred: Weight for prediction loss
        beta_kl: Weight for KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # Prediction loss (Cross Entropy)
    if len(y.shape) == 1:
        pred_loss = F.cross_entropy(y_pred, y, reduction='sum')
    else:
        # If y is one-hot or probabilities, convert to class indices
        y_labels = torch.argmax(y, dim=1)
        pred_loss = F.cross_entropy(y_pred, y_labels, reduction='sum')
    
    # KL divergence: D_KL(q(z|x,y) || p(z)) where p(z) = N(0, I)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = beta_recon * recon_loss + beta_pred * pred_loss + beta_kl * kl_loss
    
    return total_loss, recon_loss, pred_loss, kl_loss


def train_counts_model(model, dataset, num_epochs=100, batch_size=32, 
                       lr=0.001, beta_recon=1.0, beta_pred=1.0, beta_kl=0.1,
                       verbose=False):
    """
    Train the CounTS model on the given dataset.
    
    Args:
        model: CounTS model instance
        dataset: Training dataset (list of (x, y) tuples)
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        beta_recon: Weight for reconstruction loss
        beta_pred: Weight for prediction loss
        beta_kl: Weight for KL divergence
        verbose: Print training progress
    """
    device = next(model.parameters()).device
    optimizer = Adam(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon = 0
        total_pred = 0
        total_kl = 0
        
        # Shuffle dataset
        indices = torch.randperm(len(dataset))
        
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Prepare batch
            x_batch = []
            y_batch = []
            for idx in batch_indices:
                x, y = dataset[idx]
                # Ensure x is numpy array
                x_np = np.asarray(x, dtype=np.float32)
                if x_np.ndim == 1:
                    x_np = x_np.reshape(-1, 1)  # (seq_len, 1) for univariate
                elif x_np.ndim == 2:
                    # Ensure shape is (seq_len, input_dim)
                    if x_np.shape[0] < x_np.shape[1]:
                        x_np = x_np.T
                x_batch.append(x_np)
                
                # Convert y to class index if it's one-hot encoded
                if hasattr(y, 'shape') and len(y.shape) > 0:
                    y_class = np.argmax(y) if len(y.shape) > 0 else int(y)
                else:
                    y_class = int(y)
                y_batch.append(y_class)
            
            x_batch = torch.stack([torch.from_numpy(x) for x in x_batch]).to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)
            
            # Convert y to one-hot for encoder
            y_onehot = F.one_hot(y_batch, num_classes=model.num_classes).float()
            
            # Forward pass
            x_recon, y_pred, mu, logvar, z = model(x_batch, y_onehot)
            
            # Compute loss
            loss, recon_loss, pred_loss, kl_loss = counts_vae_loss(
                x_batch, x_recon, y_batch, y_pred, mu, logvar,
                beta_recon, beta_pred, beta_kl
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_pred += pred_loss.item()
            total_kl += kl_loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataset)
            avg_recon = total_recon / len(dataset)
            avg_pred = total_pred / len(dataset)
            avg_kl = total_kl / len(dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}, "
                  f"Recon={avg_recon:.4f}, Pred={avg_pred:.4f}, KL={avg_kl:.4f}")


def counts_generate_counterfactual(sample, 
                                   counts_model,
                                   target_class,
                                   original_class=None,
                                   max_iter=500,
                                   lr=0.01,
                                   lambda_validity=1.0,
                                   lambda_proximity=0.5,
                                   lambda_actionability=0.1,
                                   feasibility_bounds=None,
                                   verbose=False):
    """
    Generate counterfactual explanation using trained CounTS model.
    
    Implements the three-step counterfactual generation process:
    1. Abduction: Encode input to latent space
    2. Action: Optimize latent perturbation
    3. Prediction: Decode modified latent to counterfactual
    
    Args:
        sample: Original time series (numpy array)
        counts_model: Trained CounTS model
        target_class: Desired target class
        original_class: Original predicted class (computed if None)
        max_iter: Maximum optimization iterations
        lr: Learning rate for latent optimization
        lambda_validity: Weight for validity (prediction) loss
        lambda_proximity: Weight for proximity loss (distance in latent space)
        lambda_actionability: Weight for actionability constraint
        feasibility_bounds: Tuple (min, max) for constraining counterfactual values
        verbose: Print progress information
        
    Returns:
        x_cf: Counterfactual time series
        y_cf: Prediction for counterfactual
    """
    device = next(counts_model.parameters()).device
    counts_model.eval()
    
    # Prepare sample
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif len(x_tensor.shape) == 2:
        x_tensor = x_tensor.unsqueeze(0)  # Add batch dim
    
    # Step 1: Abduction - Get latent representation
    with torch.no_grad():
        # Get original prediction
        temp_mu, temp_logvar = counts_model.encode(x_tensor, 
                                                     torch.zeros(1, counts_model.num_classes).to(device))
        temp_z = counts_model.reparameterize(temp_mu, temp_logvar)
        y_orig_logits = counts_model.predict(temp_z)
        y_orig_probs = F.softmax(y_orig_logits, dim=1)
        
        if original_class is None:
            original_class = torch.argmax(y_orig_probs, dim=1).item()
        
        if verbose:
            print(f"CounTS: Original class {original_class}, Target class {target_class}")
        
        # Encode with actual outcome for better latent representation
        y_orig_onehot = F.one_hot(torch.tensor([original_class]), 
                                   num_classes=counts_model.num_classes).float().to(device)
        mu, logvar = counts_model.encode(x_tensor, y_orig_onehot)
        z_orig = counts_model.reparameterize(mu, logvar)
    
    # Step 2: Action - Optimize perturbation in latent space
    # Initialize delta (perturbation) as learnable parameter
    delta = torch.zeros_like(z_orig, requires_grad=True, device=device)
    optimizer = Adam([delta], lr=lr)
    
    target_tensor = torch.tensor([target_class], dtype=torch.long, device=device)
    
    best_cf = None
    best_validity = 0.0
    best_loss = float('inf')
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Modified latent representation
        z_cf = z_orig + delta
        
        # Decode to counterfactual
        x_cf = counts_model.decode(z_cf)
        
        # Predict outcome
        y_cf_logits = counts_model.predict(z_cf)
        y_cf_probs = F.softmax(y_cf_logits, dim=1)
        
        # Loss components
        
        # 1. Validity loss: Encourage prediction towards target class
        validity_loss = F.cross_entropy(y_cf_logits, target_tensor)
        
        # 2. Proximity loss: Minimize perturbation in latent space
        proximity_loss = torch.norm(delta, p=2)
        
        # 3. Actionability loss: Keep counterfactual realistic
        # Penalize large changes in the time series space
        actionability_loss = torch.norm(x_cf - x_tensor, p=2)
        
        # Total counterfactual loss
        cf_loss = (lambda_validity * validity_loss + 
                   lambda_proximity * proximity_loss + 
                   lambda_actionability * actionability_loss)
        
        # Backward pass
        cf_loss.backward()
        optimizer.step()
        
        # Apply feasibility bounds if specified
        with torch.no_grad():
            if feasibility_bounds is not None:
                z_cf_bounded = z_orig + delta
                x_cf_test = counts_model.decode(z_cf_bounded)
                x_cf_clamped = torch.clamp(x_cf_test, feasibility_bounds[0], feasibility_bounds[1])
                # If bounds are violated, adjust delta
                if not torch.allclose(x_cf_test, x_cf_clamped):
                    # Project back to feasible region
                    delta.data = delta.data * 0.95  # Reduce perturbation slightly
        
        # Track best counterfactual
        current_validity = y_cf_probs[0, target_class].item()
        current_pred_class = torch.argmax(y_cf_probs, dim=1).item()
        
        if current_pred_class == target_class:
            if cf_loss.item() < best_loss:
                best_loss = cf_loss.item()
                best_cf = x_cf.detach().clone()
                best_validity = current_validity
        elif current_validity > best_validity:
            best_validity = current_validity
            best_cf = x_cf.detach().clone()
        
        # Debug output
        if verbose and iteration % 100 == 0:
            print(f"CounTS iter {iteration}: pred_class={current_pred_class}, "
                  f"target={target_class}, validity={current_validity:.4f}, "
                  f"loss={cf_loss.item():.4f}")
        
        # Early stopping if target is reached with high confidence
        if current_pred_class == target_class and current_validity > 0.9:
            if verbose:
                print(f"CounTS: Found high-confidence counterfactual at iteration {iteration}")
            best_cf = x_cf.detach().clone()
            break
    
    # Step 3: Return best counterfactual
    if best_cf is None:
        if verbose:
            print(f"CounTS: No valid counterfactual found. Best validity: {best_validity:.4f}")
        # Return the last attempt even if not perfect
        best_cf = x_cf.detach()
    
    # Get final prediction
    with torch.no_grad():
        z_final = z_orig + delta
        y_final_logits = counts_model.predict(z_final)
        y_final = F.softmax(y_final_logits, dim=1)
    
    # Convert to numpy and match original shape
    x_cf_np = detach_to_numpy(best_cf.squeeze(0))  # Remove batch dimension
    y_cf_np = detach_to_numpy(y_final.squeeze(0))
    
    # Match original sample shape
    if len(sample.shape) == 1:
        x_cf_np = x_cf_np.squeeze()
    elif len(sample.shape) == 2 and sample.shape[0] > sample.shape[1]:
        x_cf_np = x_cf_np.T
    
    return x_cf_np, y_cf_np


def counts_cf_with_pretrained_model(sample,
                                   dataset,
                                   classifier_model,
                                   target=None,
                                   counts_model=None,
                                   latent_dim=16,
                                   hidden_dim=64,
                                   train_epochs=50,
                                   max_iter=500,
                                   lr_cf=0.01,
                                   lambda_validity=1.0,
                                   lambda_proximity=0.5,
                                   lambda_actionability=0.1,
                                   feasibility_bounds=None,
                                   verbose=False):
    """
    Generate counterfactual using CounTS with automatic model training.
    
    This is a convenience function that handles model creation and training
    if a pre-trained CounTS model is not provided.
    
    Args:
        sample: Original time series
        dataset: Training dataset for CounTS model
        classifier_model: Pre-trained classifier (for reference, not used in CounTS)
        target: Target class (computed if None)
        counts_model: Pre-trained CounTS model (trained if None)
        latent_dim: Latent space dimensionality
        hidden_dim: Hidden layer dimensionality
        train_epochs: Number of training epochs for CounTS
        max_iter: Maximum counterfactual optimization iterations
        lr_cf: Learning rate for counterfactual generation
        lambda_validity: Weight for validity loss
        lambda_proximity: Weight for proximity loss
        lambda_actionability: Weight for actionability constraint
        feasibility_bounds: Tuple (min, max) for constraining values
        verbose: Print progress information
        
    Returns:
        x_cf: Counterfactual time series
        y_cf: Prediction for counterfactual
    """
    device = next(classifier_model.parameters()).device
    
    # Prepare sample
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)
    elif len(sample_tensor.shape) == 2:
        sample_tensor = sample_tensor.unsqueeze(0)
    
    # Get original prediction from classifier
    with torch.no_grad():
        y_orig = classifier_model(sample_tensor)
        y_orig_np = detach_to_numpy(y_orig)[0]
        original_class = int(np.argmax(y_orig_np))
    
    # Determine target class
    if target is None:
        sorted_indices = np.argsort(y_orig_np)[::-1]
        target = int(sorted_indices[1])
    
    # Determine input dimensions from dataset
    x_sample, y_sample = dataset[0]
    x_sample_tensor = torch.tensor(x_sample, dtype=torch.float32)
    
    if len(x_sample_tensor.shape) == 1:
        seq_len = x_sample_tensor.shape[0]
        input_dim = 1
    elif len(x_sample_tensor.shape) == 2:
        if x_sample_tensor.shape[0] > x_sample_tensor.shape[1]:
            seq_len = x_sample_tensor.shape[0]
            input_dim = x_sample_tensor.shape[1]
        else:
            seq_len = x_sample_tensor.shape[1]
            input_dim = x_sample_tensor.shape[0]
    else:
        seq_len = x_sample_tensor.shape[-1]
        input_dim = x_sample_tensor.shape[-2]
    
    # Determine number of classes
    num_classes = len(np.unique([y for _, y in dataset]))
    
    # Create and train CounTS model if not provided
    if counts_model is None:
        if verbose:
            print(f"CounTS: Training VAE model with {latent_dim}D latent space...")
        
        counts_model = CounTSModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_classes=num_classes,
            seq_len=seq_len
        ).to(device)
        
        # Train the model
        train_counts_model(
            counts_model, 
            dataset,
            num_epochs=train_epochs,
            batch_size=32,
            lr=0.001,
            beta_recon=1.0,
            beta_pred=1.0,
            beta_kl=0.1,
            verbose=verbose
        )
        
        if verbose:
            print("CounTS: Model training complete.")
    
    # Generate counterfactual
    x_cf, y_cf = counts_generate_counterfactual(
        sample,
        counts_model,
        target_class=target,
        original_class=original_class,
        max_iter=max_iter,
        lr=lr_cf,
        lambda_validity=lambda_validity,
        lambda_proximity=lambda_proximity,
        lambda_actionability=lambda_actionability,
        feasibility_bounds=feasibility_bounds,
        verbose=verbose
    )
    
    return x_cf, y_cf
