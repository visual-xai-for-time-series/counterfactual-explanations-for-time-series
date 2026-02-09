import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


####
# SPARCE: Generating SPARse Counterfactual Explanations for Multivariate Time Series
#
# Paper: Lang, J., Giese, M., Ilg, W., & Otte, S. (2022).
#        "Generating Sparse Counterfactual Explanations For Multivariate Time Series"
#        arXiv preprint arXiv:2206.00931
#
# Paper URL: https://arxiv.org/abs/2206.00931
# GitHub: https://github.com/janalang/SPARCE
#
# SPARCE uses a GAN-based architecture to generate sparse counterfactual explanations
# for time series data. The generator creates residuals (modifications) that are added
# to the input query to produce counterfactuals. The approach regularizes the loss
# function with:
# - Adversarial loss (discriminator-based)
# - Classification loss (target class prediction)
# - Similarity loss (L1 norm between query and counterfactual)
# - Sparsity loss (L0 norm encouraging sparse modifications)
# - Jerk loss (smoothness of trajectory changes)
####


class ResidualGeneratorLSTM(nn.Module):
    """LSTM-based generator that produces counterfactual modifications (residuals).
    
    The generator outputs residuals that are added to the input query to create
    counterfactuals, rather than generating entire sequences.
    """
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Only apply dropout if num_layers > 1
        dropout = 0.4 if layer_dim > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_dim, output_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, output_dim)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        x = x.float()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        fc1_out = self.fc1(lstm_out)
        fc2_out = self.fc2(lstm_out)
        
        act1_out = self.act1(fc1_out)
        act2_out = self.act2(fc2_out)
        act_out = act1_out - act2_out  # Residuals can be positive or negative
        
        return act_out  # Keep as float32
    
    def init_hidden(self, x):
        device = x.device
        h0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim).to(device)
        return h0, c0


class DiscriminatorLSTM(nn.Module):
    """Bidirectional LSTM discriminator for distinguishing real from generated sequences."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # Only apply dropout if num_layers > 1
        dropout = 0.4 if layer_dim > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        x = x.float()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        # Take only the last timestep output for classification
        lstm_out = lstm_out[:, -1, :]
        fc_out = self.fc(lstm_out)
        act_out = self.act(fc_out)
        return act_out
    
    def init_hidden(self, x):
        device = x.device
        h0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim).to(device)
        return h0, c0


def compute_similarity_loss(deltas):
    """Penalizes large differences between query and counterfactual (L1 loss).
    
    Args:
        deltas: Residuals/modifications applied to the query
        
    Returns:
        L1 distance normalized by dimensions
    """
    deltas_flat = torch.reshape(deltas, (deltas.shape[0], deltas.shape[1] * deltas.shape[2]))
    dist = torch.linalg.norm(deltas_flat, ord=1, dim=1)
    dist = dist / (deltas.shape[1] * deltas.shape[2])  # Normalize without in-place
    return dist


def compute_sparsity_loss(deltas, threshold=1e-3):
    """Penalizes high numbers of modified time steps and features (L0-like loss).
    
    Args:
        deltas: Residuals/modifications applied to the query
        threshold: Threshold below which modifications are considered zero
        
    Returns:
        Approximate L0 norm (number of non-zero elements)
    """
    deltas_flat = torch.reshape(deltas, (deltas.shape[0], deltas.shape[1] * deltas.shape[2]))
    # Approximate L0 norm by counting elements above threshold
    non_zero = (torch.abs(deltas_flat) > threshold).float()
    sparsity = torch.sum(non_zero, dim=1)
    sparsity = sparsity / (deltas.shape[1] * deltas.shape[2])  # Normalize without in-place
    return sparsity


def compute_jerk_loss(deltas):
    """Penalizes large differences between modifications in consecutive time steps.
    
    This encourages smooth trajectories in the counterfactual modifications.
    
    Args:
        deltas: Residuals/modifications applied to the query
        
    Returns:
        Jerk loss measuring smoothness
    """
    device = deltas.device
    # Add zero padding at the beginning to compute differences
    deltas_extended = torch.zeros((deltas.shape[0], 1, deltas.shape[2])).to(device)
    deltas_extended = torch.cat((deltas_extended, deltas), dim=1)
    deltas_extended = deltas_extended[:, :-1, :]
    
    # Compute differences between consecutive timesteps
    jerk_loss = torch.linalg.norm(deltas - deltas_extended, dim=2).sum(dim=1)
    jerk_loss = jerk_loss / (deltas.shape[1] * deltas.shape[2])  # Normalize without in-place
    return jerk_loss


def sparce_gan_cf(sample,
                  dataset,
                  model,
                  target=None,
                  lambda_adv=1.0,
                  lambda_cls=1.0,
                  lambda_sim=1.0,
                  lambda_sparse=1.0,
                  lambda_jerk=1.0,
                  num_epochs=50,
                  batch_size=32,
                  lr=0.0002,
                  hidden_dim_gen=256,
                  hidden_dim_disc=16,
                  verbose=False):
    """Generate sparse counterfactual explanations using SPARCE GAN architecture.
    
    This implementation uses a GAN with a residual generator and discriminator to
    produce sparse counterfactual explanations. The generator learns to create
    minimal modifications (residuals) that achieve the target class prediction.
    
    Args:
        sample: Input time series sample to explain (shape: [channels, length] or [length,])
        dataset: Dataset for sampling target examples (not used in current implementation)
        model: Pretrained classifier model
        target: Target class for counterfactual (if None, uses second most likely class)
        lambda_adv: Weight for adversarial loss
        lambda_cls: Weight for classification loss
        lambda_sim: Weight for similarity loss (L1)
        lambda_sparse: Weight for sparsity loss (L0)
        lambda_jerk: Weight for jerk loss (smoothness)
        num_epochs: Number of training epochs
        batch_size: Batch size (not used in single sample mode)
        lr: Learning rate
        hidden_dim_gen: Hidden dimension for generator
        hidden_dim_disc: Hidden dimension for discriminator
        verbose: Print training progress
        
    Returns:
        Tuple of (counterfactual, prediction) or (None, None) if unsuccessful
    """
    device = next(model.parameters()).device
    
    # Prepare input sample - ensure float32 dtype
    if isinstance(sample, torch.Tensor):
        sample_tensor = sample.float().to(device)
    else:
        sample_np = np.asarray(sample, dtype=np.float32)
        sample_tensor = torch.from_numpy(sample_np).to(device)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.reshape(1, 1, -1)  # (length,) -> (1, 1, length)
    elif len(sample_tensor.shape) == 2:
        if sample_tensor.shape[0] > sample_tensor.shape[1]:
            sample_tensor = sample_tensor.T  # (length, channels) -> (channels, length)
        sample_tensor = sample_tensor.unsqueeze(0)  # Add batch dimension
    
    batch_size_actual, num_timesteps, num_features = sample_tensor.shape
    
    # Get initial prediction and determine target
    with torch.no_grad():
        y_orig = model(sample_tensor)
        y_orig_np = detach_to_numpy(y_orig)[0]
        label_orig = int(np.argmax(y_orig_np))
    
    if target is None:
        # Find second most likely class
        sorted_indices = np.argsort(y_orig_np)[::-1]
        target = int(sorted_indices[1])
    
    target_tensor = torch.tensor([target], dtype=torch.long, device=device)
    
    if verbose:
        print(f"SPARCE: Original class {label_orig}, Target class {target}")
        print(f"Sample shape: {sample_tensor.shape}")
    
    # Initialize generator and discriminator
    generator = ResidualGeneratorLSTM(num_features, hidden_dim_gen, 2, num_features).to(device)
    discriminator = DiscriminatorLSTM(num_features, hidden_dim_disc, 1).to(device)
    
    # Optimizers
    gen_optimizer = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss functions
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    best_cf = None
    best_pred = None
    best_validity = 0.0
    
    for epoch in range(num_epochs):
        # ============ Train Generator ============
        generator.train()
        discriminator.eval()
        
        gen_optimizer.zero_grad()
        
        # Generate residuals and create counterfactual
        deltas = generator(sample_tensor)
        cf = sample_tensor.clone() + deltas
        
        # Get classifier prediction on counterfactual
        cf_pred = model(cf)
        cf_class = torch.argmax(cf_pred, dim=1)
        
        # Compute losses
        # 1. Classification loss - encourage target class
        cls_loss = ce_loss(cf_pred, target_tensor)
        
        # 2. Adversarial loss - fool discriminator
        fake_out = discriminator(cf)
        adv_loss = bce_loss(fake_out, torch.ones_like(fake_out))
        
        # 3. Similarity loss - stay close to original
        sim_loss = compute_similarity_loss(deltas).mean()
        
        # 4. Sparsity loss - minimize modifications
        sparse_loss = compute_sparsity_loss(deltas).mean()
        
        # 5. Jerk loss - smooth modifications
        jerk_loss_val = compute_jerk_loss(deltas).mean()
        
        # Combined generator loss
        gen_loss = (lambda_adv * adv_loss + 
                   lambda_cls * cls_loss + 
                   lambda_sim * sim_loss + 
                   lambda_sparse * sparse_loss + 
                   lambda_jerk * jerk_loss_val)
        
        gen_loss.backward()
        gen_optimizer.step()
        
        # Track best counterfactual (detach before storing)
        with torch.no_grad():
            cf_pred_np = detach_to_numpy(cf_pred)[0]
            current_validity = cf_pred_np[target]
            
            if current_validity > best_validity:
                best_validity = current_validity
                best_cf = detach_to_numpy(cf.squeeze(0))
                best_pred = cf_pred_np
        
        # ============ Train Discriminator ============
        generator.eval()
        discriminator.train()
        
        disc_optimizer.zero_grad()
        
        # Real samples (original input) should be classified as real (1)
        real_out = discriminator(sample_tensor)
        real_loss = bce_loss(real_out, torch.ones_like(real_out))
        
        # Fake samples (counterfactuals) should be classified as fake (0)
        with torch.no_grad():
            cf_detached = sample_tensor.clone() + generator(sample_tensor)
        fake_out = discriminator(cf_detached.detach())
        fake_loss = bce_loss(fake_out, torch.zeros_like(fake_out))
        
        # Combined discriminator loss
        disc_loss = (real_loss + fake_loss) / 2
        
        disc_loss.backward()
        disc_optimizer.step()
        
        # Verbose output
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            cf_class_np = int(cf_class.item())
            print(f"Epoch {epoch}: pred_class={cf_class_np}, target={target}, "
                  f"validity={current_validity:.4f}, "
                  f"gen_loss={gen_loss.item():.4f}, disc_loss={disc_loss.item():.4f}")
            print(f"  cls={cls_loss.item():.4f}, adv={adv_loss.item():.4f}, "
                  f"sim={sim_loss.item():.4f}, sparse={sparse_loss.item():.4f}, "
                  f"jerk={jerk_loss_val.item():.4f}")
        
        # Early stopping if target class achieved
        if int(cf_class.item()) == target:
            if verbose:
                print(f"SPARCE: Found counterfactual at epoch {epoch}")
            break
    
    if verbose:
        print(f"SPARCE: Best validity achieved: {best_validity:.4f}")
    
    if best_cf is None:
        if verbose:
            print("SPARCE: No counterfactual found")
        return None, None
    
    # Convert back to original sample format
    if len(sample.shape) == 1:
        best_cf = best_cf.squeeze()
    elif len(sample.shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            best_cf = best_cf.T
    
    return best_cf, best_pred


def sparce_gradient_cf(sample,
                       model,
                       target=None,
                       lambda_cls=1.0,
                       lambda_sim=1.0,
                       lambda_sparse=1.0,
                       lambda_jerk=1.0,
                       max_iter=500,
                       lr=0.01,
                       verbose=False):
    """Simplified gradient-based SPARCE without GAN components.
    
    This version directly optimizes the residuals without a generator network,
    using only the classifier and the regularization losses.
    
    Args:
        sample: Input time series sample
        model: Pretrained classifier
        target: Target class
        lambda_cls: Weight for classification loss
        lambda_sim: Weight for similarity loss
        lambda_sparse: Weight for sparsity loss
        lambda_jerk: Weight for jerk loss
        max_iter: Maximum optimization iterations
        lr: Learning rate
        verbose: Print progress
        
    Returns:
        Tuple of (counterfactual, prediction) or (None, None)
    """
    device = next(model.parameters()).device
    
    # Prepare input
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.reshape(1, 1, -1)
    elif len(sample_tensor.shape) == 2:
        if sample_tensor.shape[0] > sample_tensor.shape[1]:
            sample_tensor = sample_tensor.T
        sample_tensor = sample_tensor.unsqueeze(0)
    
    # Get initial prediction
    with torch.no_grad():
        y_orig = model(sample_tensor)
        y_orig_np = detach_to_numpy(y_orig)[0]
        label_orig = int(np.argmax(y_orig_np))
    
    if target is None:
        sorted_indices = np.argsort(y_orig_np)[::-1]
        target = int(sorted_indices[1])
    
    target_tensor = torch.tensor([target], dtype=torch.long, device=device)
    
    if verbose:
        print(f"SPARCE Gradient: Original class {label_orig}, Target class {target}")
    
    # Initialize residuals (modifications to apply)
    deltas = torch.zeros_like(sample_tensor, requires_grad=True)
    optimizer = Adam([deltas], lr=lr)
    
    ce_loss = nn.CrossEntropyLoss()
    
    best_cf = None
    best_pred = None
    best_validity = 0.0
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Create counterfactual
        cf = sample_tensor + deltas
        
        # Get prediction
        cf_pred = model(cf)
        
        # Compute losses
        cls_loss = ce_loss(cf_pred, target_tensor)
        sim_loss = compute_similarity_loss(deltas).mean()
        sparse_loss = compute_sparsity_loss(deltas).mean()
        jerk_loss_val = compute_jerk_loss(deltas).mean()
        
        # Combined loss
        total_loss = (lambda_cls * cls_loss + 
                     lambda_sim * sim_loss + 
                     lambda_sparse * sparse_loss + 
                     lambda_jerk * jerk_loss_val)
        
        total_loss.backward()
        optimizer.step()
        
        # Track progress
        with torch.no_grad():
            cf_pred_np = detach_to_numpy(cf_pred)[0]
            cf_class = int(np.argmax(cf_pred_np))
            current_validity = cf_pred_np[target]
            
            if current_validity > best_validity:
                best_validity = current_validity
                best_cf = detach_to_numpy(cf.squeeze(0))
                best_pred = cf_pred_np
        
        if verbose and iteration % 100 == 0:
            print(f"Iter {iteration}: pred_class={cf_class}, target={target}, "
                  f"validity={current_validity:.4f}, loss={total_loss.item():.4f}")
        
        # Early stopping
        if cf_class == target:
            if verbose:
                print(f"SPARCE Gradient: Found counterfactual at iteration {iteration}")
            break
    
    if verbose:
        print(f"SPARCE Gradient: Best validity: {best_validity:.4f}")
    
    if best_cf is None:
        return None, None
    
    # Convert back to original format
    if len(sample.shape) == 1:
        best_cf = best_cf.squeeze()
    elif len(sample.shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            best_cf = best_cf.T
    
    return best_cf, best_pred
