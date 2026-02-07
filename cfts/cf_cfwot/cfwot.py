import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical, Normal


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


####
# CFWoT: Counterfactual Explanations WithOut Training datasets
#
# Paper: Sun, X., Aoki, R., & Wilson, K. H. (2024).
#        "Counterfactual Explanations for Multivariate Time-Series 
#        without Training Datasets"
#        arXiv:2405.18563
#
# Paper URL: https://arxiv.org/abs/2405.18563
#
# A reinforcement learning-based counterfactual explanation method for both
# static and multivariate time-series data. CFWoT operates without requiring
# training datasets and is model-agnostic, supporting both differentiable and
# non-differentiable models.
#
# Key Features:
# - No training dataset required
# - Model-agnostic (works with any predictive model)
# - Supports multivariate time-series and static data
# - Handles continuous and discrete features
# - Allows user preferences (feature feasibility weights)
# - Supports causal constraints and actionable features
####


class PolicyNetwork(nn.Module):
    """
    Policy network for CFWoT that outputs distributions for action selection.
    
    The action consists of three components:
    - a_time: which time step to intervene on
    - a_feat: which feature to modify
    - a_stre: the strength/value of the intervention
    """
    
    def __init__(self, K, D, D_C, D_D, N_dis_list, hidden_size=1000, hidden_size2=100):
        """
        Args:
            K: Number of time steps
            D: Total number of features
            D_C: Number of continuous features
            D_D: Number of discrete features
            N_dis_list: List of number of possible values for each discrete feature
            hidden_size: Size of first hidden layer
            hidden_size2: Size of second hidden layer
        """
        super(PolicyNetwork, self).__init__()
        
        self.K = K
        self.D = D
        self.D_C = D_C
        self.D_D = D_D
        self.N_dis_list = N_dis_list if N_dis_list else []
        
        # Input: flattened time series
        input_size = K * D
        
        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        
        # Time step selection head
        self.time_head = nn.Linear(hidden_size2, K)
        
        # Feature selection head
        self.feat_head = nn.Linear(hidden_size2, D)
        
        # Continuous feature intervention strength heads (mean and std)
        if D_C > 0:
            self.cont_mu_head = nn.Linear(hidden_size2, D_C)
            self.cont_sigma_head = nn.Linear(hidden_size2, D_C)
        
        # Discrete feature intervention value heads
        if D_D > 0 and len(N_dis_list) > 0:
            total_discrete_values = sum(N_dis_list)
            self.disc_head = nn.Linear(hidden_size2, total_discrete_values)
    
    def forward(self, x):
        """
        Forward pass to compute action distribution parameters.
        
        Args:
            x: Input tensor of shape (batch, K, D) or flattened (batch, K*D)
        
        Returns:
            Dictionary with distribution parameters
        """
        # Flatten input if needed
        if len(x.shape) == 3:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        
        # Shared layers
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        
        # Time step distribution (categorical)
        p_time = torch.softmax(self.time_head(h), dim=-1)
        
        # Feature distribution (categorical)
        p_feat = torch.softmax(self.feat_head(h), dim=-1)
        
        result = {
            'p_time': p_time,
            'p_feat': p_feat
        }
        
        # Continuous feature strength (Gaussian)
        if self.D_C > 0:
            mu = self.cont_mu_head(h)
            sigma = torch.exp(self.cont_sigma_head(h))  # Ensure positive
            result['mu_cont'] = mu
            result['sigma_cont'] = sigma
        
        # Discrete feature values (categorical for each feature)
        if self.D_D > 0 and len(self.N_dis_list) > 0:
            disc_logits = self.disc_head(h)
            # Split logits for each discrete feature
            p_disc_list = []
            start_idx = 0
            for n_vals in self.N_dis_list:
                end_idx = start_idx + n_vals
                p_disc = torch.softmax(disc_logits[:, start_idx:end_idx], dim=-1)
                p_disc_list.append(p_disc)
                start_idx = end_idx
            result['p_disc'] = p_disc_list
        
        return result


def cfwot(sample,
          model,
          target=None,
          D_act=None,
          D_non_act=None,
          D_immu=None,
          D_dis=None,
          N_dis=None,
          W_fsib=None,
          C_SCM=None,
          C_range=None,
          lambda_pxmt=0.001,
          M_E=100,
          M_T=100,
          gamma=0.99,
          lr=0.0001,
          lambda_WD=0.0,
          verbose=False,
          device=None):
    """
    CFWoT: Generate counterfactual explanations without training datasets.
    
    Args:
        sample: Original input sample, shape (K, D) for time series or (D,) for static
        model: Predictive model (callable that takes input and returns predictions)
        target: Target class for counterfactual (int)
        D_act: List of actionable feature indices (default: all features)
        D_non_act: List of non-actionable feature indices
        D_immu: List of immutable feature indices
        D_dis: List of discrete feature indices
        N_dis: Dictionary mapping discrete feature index to number of values
        W_fsib: Feature feasibility weights (dict or array), default 1.0 for all
        C_SCM: Causal constraints (structural causal model rules)
        C_range: Feature range constraints (dict mapping feature idx to (min, max))
        lambda_pxmt: Proximity weight in reward function
        M_E: Maximum number of episodes
        M_T: Maximum number of interventions per episode
        gamma: Discount factor for RL
        lr: Learning rate for policy network
        lambda_WD: Weight decay for regularization
        verbose: Print progress information
        device: Torch device (auto-detected if None)
    
    Returns:
        best_cf: Best counterfactual found (numpy array)
        best_pred: Prediction for best counterfactual
    """
    
    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert sample to numpy if needed
    if isinstance(sample, torch.Tensor):
        sample = detach_to_numpy(sample)
    
    # Ensure sample is at least 2D (K, D)
    # For univariate time series: (length,) -> (length, 1)
    if len(sample.shape) == 1:
        sample = sample.reshape(-1, 1)  # (L,) -> (L, 1) for time series
    
    K, D = sample.shape  # K: time steps, D: features
    
    # Setup actionable/non-actionable/immutable features
    if D_act is None:
        D_act = list(range(D))
    if D_non_act is None:
        D_non_act = []
    if D_immu is None:
        D_immu = []
    
    # Setup discrete features
    if D_dis is None:
        D_dis = []
    if N_dis is None:
        N_dis = {}
    
    D_C = len([d for d in D_act if d not in D_dis])  # Continuous actionable features
    D_D = len([d for d in D_act if d in D_dis])      # Discrete actionable features
    
    # Create list of N values for discrete features in order
    N_dis_list = [N_dis.get(d, 2) for d in D_act if d in D_dis]
    
    # Setup feature feasibility weights
    if W_fsib is None:
        W_fsib = {d: 1.0 for d in range(D)}
    elif isinstance(W_fsib, (list, np.ndarray)):
        W_fsib = {i: float(w) for i, w in enumerate(W_fsib)}
    
    # Get initial prediction
    def model_predict(x):
        """Helper to get model prediction."""
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x_tensor = x
        
        # Ensure proper shape for model: (batch, channels, length)
        if len(x_tensor.shape) == 2:
            # (length, features) -> (1, features, length) or (1, 1, length) for univariate
            if x_tensor.shape[1] == 1:
                x_tensor = x_tensor.T.unsqueeze(0)  # (L, 1) -> (1, 1, L)
            else:
                x_tensor = x_tensor.T.unsqueeze(0)  # (L, C) -> (1, C, L)
        elif len(x_tensor.shape) == 1:
            x_tensor = x_tensor.reshape(1, 1, -1)  # (L,) -> (1, 1, L)
        
        with torch.no_grad():
            pred = model(x_tensor)
        return detach_to_numpy(pred)[0]
    
    # Get initial prediction and determine target if not specified
    y_orig = model_predict(sample)
    label_orig = int(np.argmax(y_orig))
    
    if target is None:
        # Choose second most likely class as target
        sorted_indices = np.argsort(y_orig)[::-1]
        target = int(sorted_indices[1])
    
    if verbose:
        print(f"CFWoT: Original class {label_orig}, Target class {target}")
        print(f"CFWoT: K={K}, D={D}, D_C={D_C}, D_D={D_D}")
    
    # Initialize policy network
    policy_net = PolicyNetwork(K, len(D_act), D_C, D_D, N_dis_list).to(device)
    optimizer = Adam(policy_net.parameters(), lr=lr, weight_decay=lambda_WD)
    
    # Storage for found counterfactuals
    O = []  # Set of valid counterfactuals found
    
    # Proximity function (L1 for continuous, L0 for discrete)
    def compute_proximity(x1, x2):
        """Compute weighted proximity between two samples."""
        total_dist = 0.0
        for d in range(D):
            w = W_fsib.get(d, 1.0)
            if d in D_dis:
                # L0 norm for discrete features
                total_dist += w * np.sum(x1[:, d] != x2[:, d])
            else:
                # L1 norm for continuous features
                total_dist += w * np.sum(np.abs(x1[:, d] - x2[:, d]))
        return total_dist
    
    # State transition function
    def state_transition(x_t, action):
        """
        Apply action to current state.
        
        action = {a_time, a_feat, a_stre}
        """
        x_next = x_t.copy()
        a_time = action['a_time']
        a_feat = action['a_feat']
        a_stre = action['a_stre']
        
        # Get actual feature index from actionable features
        feat_idx = D_act[a_feat]
        
        # Apply intervention from time step a_time onwards
        if feat_idx in D_dis:
            # Discrete feature: set to new value
            x_next[a_time:, feat_idx] = a_stre
        else:
            # Continuous feature: add strength
            x_next[a_time:, feat_idx] += a_stre
        
        return x_next
    
    # Constraint checking
    def check_constraints(x):
        """Check if state satisfies constraints."""
        # Check range constraints
        if C_range is not None:
            for feat_idx, (min_val, max_val) in C_range.items():
                if np.any(x[:, feat_idx] < min_val) or np.any(x[:, feat_idx] > max_val):
                    # Clip to range
                    x[:, feat_idx] = np.clip(x[:, feat_idx], min_val, max_val)
        
        # Check causal constraints (simplified - apply rules if provided)
        if C_SCM is not None:
            # C_SCM would be a function or set of rules
            # For now, just pass through
            pass
        
        return x
    
    # Reward function
    def compute_reward(x, y_pred):
        """
        Reward function combining prediction and proximity.
        
        r = 1 - lambda_pxmt * D_pxmt  if f(x) == target
        r = 0                          otherwise
        """
        pred_class = int(np.argmax(y_pred))
        
        if pred_class == target:
            prox = compute_proximity(sample, x)
            reward = 1.0 - lambda_pxmt * prox
            return reward, True  # Valid CF
        else:
            return 0.0, False  # Not valid
    
    # Training loop (episodes)
    for episode in range(M_E):
        # Reset to original sample
        x_t = sample.copy()
        
        # Storage for episode
        states = []
        actions = []
        log_probs = []
        rewards = []
        
        # Episode loop (interventions)
        for t in range(M_T):
            # Convert state to tensor
            x_t_tensor = torch.tensor(x_t, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get action distributions from policy (keep gradients for log_prob)
            policy_net.train()
            action_dist = policy_net(x_t_tensor)
            
            # Sample action
            # 1. Sample time step
            p_time = action_dist['p_time']
            dist_time = Categorical(p_time)
            a_time_idx = dist_time.sample()
            log_prob = dist_time.log_prob(a_time_idx)
            
            # 2. Sample feature
            p_feat = action_dist['p_feat']
            dist_feat = Categorical(p_feat)
            a_feat_idx = dist_feat.sample()
            log_prob += dist_feat.log_prob(a_feat_idx)
            
            a_time = int(a_time_idx.item())
            a_feat = int(a_feat_idx.item())
            feat_idx = D_act[a_feat]
            
            # 3. Sample strength/value
            if feat_idx in D_dis:
                # Discrete feature - sample categorical value
                disc_idx = [d for d in D_act if d in D_dis].index(feat_idx)
                p_disc = action_dist['p_disc'][disc_idx]
                dist_disc = Categorical(p_disc)
                a_stre_idx = dist_disc.sample()
                log_prob += dist_disc.log_prob(a_stre_idx)
                a_stre = int(a_stre_idx.item())
            else:
                # Continuous feature - sample from Gaussian
                cont_idx = [d for d in D_act if d not in D_dis].index(feat_idx)
                mu = action_dist['mu_cont'][0, cont_idx]
                sigma = action_dist['sigma_cont'][0, cont_idx]
                dist_cont = Normal(mu, sigma)
                a_stre_sample = dist_cont.sample()
                log_prob += dist_cont.log_prob(a_stre_sample)
                a_stre = float(a_stre_sample.item())
            
            # Store state and action
            states.append(x_t.copy())
            action = {'a_time': a_time, 'a_feat': a_feat, 'a_stre': a_stre}
            actions.append(action)
            log_probs.append(log_prob)
            
            # Apply action (state transition)
            x_t_next = state_transition(x_t, action)
            
            # Check constraints
            x_t_next = check_constraints(x_t_next)
            
            # Get prediction
            y_t_next = model_predict(x_t_next)
            
            # Compute reward
            reward, is_valid = compute_reward(x_t_next, y_t_next)
            rewards.append(reward)
            
            # Update state
            x_t = x_t_next
            
            # If valid CF found, add to set O
            if is_valid and not any(np.allclose(x_t, cf[0]) for cf in O):
                proximity = compute_proximity(sample, x_t)
                O.append((x_t.copy(), y_t_next, proximity))
                if verbose:
                    print(f"CFWoT: Episode {episode}, Step {t}: Found valid CF with proximity {proximity:.4f}")
            
            # Early stopping if valid CF found
            if is_valid:
                break
        
        # Compute returns (discounted cumulative rewards)
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        if len(returns) > 1:
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Policy gradient update
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        if len(policy_loss) > 0:
            optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            optimizer.step()
        
        # Progress reporting
        if verbose and episode % 10 == 0:
            avg_reward = np.mean(rewards) if rewards else 0.0
            print(f"CFWoT: Episode {episode}/{M_E}, Avg Reward: {avg_reward:.4f}, Valid CFs: {len(O)}")
    
    # Return best CF (lowest proximity)
    if len(O) == 0:
        if verbose:
            print("CFWoT: No valid counterfactual found")
        return None, None
    
    # Sort by proximity and return best
    O_sorted = sorted(O, key=lambda x: x[2])
    best_cf = O_sorted[0][0]
    best_pred = O_sorted[0][1]
    
    if verbose:
        print(f"CFWoT: Best CF proximity: {O_sorted[0][2]:.4f}")
    
    # Return in original shape
    if len(sample.shape) == 1 or (len(sample.shape) == 2 and sample.shape[0] == 1):
        best_cf = best_cf.squeeze()
    
    return best_cf, best_pred
