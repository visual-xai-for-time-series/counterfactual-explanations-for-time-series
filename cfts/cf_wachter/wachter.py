import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam


def detach_to_numpy(data):
    # move pytorch data to cpu and detach it to numpy data
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    # convert numpy array to pytorch and move it to the device
    return torch.from_numpy(data).float().to(device)


def manhattan_dist(x, y):
    return torch.sum(torch.abs(x - y))


def euclidean_dist(x, y):
    return torch.sqrt(torch.sum(torch.abs(x - y) ** 2))


####
# Wachter et al.: Counterfactual Explanations without Opening the Black Box
#
# Paper: Wachter, S., Mittelstadt, B., & Russell, C. (2017).
#        "Counterfactual explanations without opening the black box:
#        Automated decisions and the GDPR."
#        Harvard Journal of Law & Technology, 31, 841-887
#
# Paper URL: https://arxiv.org/abs/1711.00399
#
# Classic counterfactual explanation method using gradient-based optimization
# or genetic algorithms to find minimal perturbations that change the model's
# prediction. Focuses on proximity while achieving target prediction.
#
# This is a genetic model-agnostic variant using the sensitivity of the model.
####
def wachter_genetic_cf(sample, model, target=None, max_steps=1000, step_size=0.1, verbose=False):

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

    # Convert sample to proper format for processing
    sample_flat = sample.reshape(-1)
    sample_cf = np.copy(sample_flat)
    
    # Get initial prediction
    y_cf = model_predict(sample_cf.reshape(sample.shape))[0]
    label_cf = np.argmax(y_cf)
    
    if not target:
        # Find the class with second highest probability (not just binary 0/1)
        sorted_indices = np.argsort(y_cf)[::-1]  # Sort in descending order
        target = int(sorted_indices[1])  # Second most likely class
    
    if verbose:
        print(f"Wachter Genetic: Original class {label_cf}, Target class {target}, step_size={step_size}")

    # Iterate until the counterfactual prediction is different from the original prediction or the maximum number of steps is reached
    for step in range(max_steps):
        # Compute the model prediction for each possible change to the input features
        y_preds = []
        feature_changes = []
        
        for i in range(len(sample_flat)):
            # plus variant
            sample_plus = sample_cf.copy()
            sample_plus[i] += step_size
            y_pred_plus = model_predict(sample_plus.reshape(sample.shape))[0]
            y_preds.append(y_pred_plus)
            feature_changes.append((i, +step_size))

            # minus variant
            sample_minus = sample_cf.copy()
            sample_minus[i] -= step_size
            y_pred_minus = model_predict(sample_minus.reshape(sample.shape))[0]
            y_preds.append(y_pred_minus)
            feature_changes.append((i, -step_size))

        # Find the change that results in the greatest increase in target class probability
        y_preds = np.array(y_preds)
        target_improvements = y_preds[:, target] - y_cf[target]  # How much target class probability improves
        
        best_change_idx = np.argmax(target_improvements)
        best_feature_idx, best_step = feature_changes[best_change_idx]
        best_improvement = target_improvements[best_change_idx]
        
        # Apply the best change
        sample_cf[best_feature_idx] += best_step
        
        # Get new prediction
        y_cf = model_predict(sample_cf.reshape(sample.shape))[0]
        current_class = np.argmax(y_cf)
        current_target_prob = y_cf[target]
        
        # Debug output every 200 steps
        if verbose and step % 200 == 0:
            print(f"Wachter Genetic step {step}: pred_class={current_class}, target={target}, "
                  f"target_prob={current_target_prob:.4f}, improvement={best_improvement:.4f}")
        
        # If the counterfactual prediction matches target, return it
        if current_class == target:
            if verbose:
                print(f"Wachter Genetic: Found counterfactual at step {step}")
            return sample_cf.reshape(sample.shape), y_cf

    if verbose:
        print(f"Wachter Genetic: Max steps reached. Final target probability: {y_cf[target]:.4f}")
    # If the maximum number of steps is reached without finding a counterfactual, return None
    return None, None


####
# Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR
#
# https://arxiv.org/abs/1711.00399
#
#
# This is a optimization variant using the gradients of the model.
#
####
def wachter_gradient_cf(sample,
                        dataset,
                        model,
                        target=None,
                        lb=None,
                        lb_step=None,
                        max_cfs=1000,
                        full_random=False,
                        distance='euclidean',
                        verbose=False):
    """Gradient-based Wachter counterfactual implemented with pure PyTorch tensors.

    Fixes errors caused by mixing torch tensors with numpy/scipy indexing (e.g.
    passing a torch index into a numpy array). Operates on tensors on the same
    device as the model and keeps conversions to/from numpy only for I/O.
    """
    device = next(model.parameters()).device

    # prepare input shapes: ensure consistent with model expectations
    # Convert sample to the format expected by the model: (batch, channels, length)
    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(sample_tensor.shape) == 1:
        sample_tensor = sample_tensor.reshape(1, 1, -1)  # (length,) -> (1, 1, length)
    elif len(sample_tensor.shape) == 2:
        if sample_tensor.shape[0] > sample_tensor.shape[1]:
            sample_tensor = sample_tensor.T  # Assume (length, channels) -> (channels, length)
        sample_tensor = sample_tensor.unsqueeze(0)  # Add batch dimension
    
    sample_t = sample_tensor.clone()

    # initial prediction and label
    y_cf = detach_to_numpy(model(sample_t))[0]
    label_cf = int(np.argmax(y_cf))
    if target is None:
        # Find the class with second highest probability (not just binary 0/1)
        sorted_indices = np.argsort(y_cf)[::-1]  # Sort in descending order
        target = int(sorted_indices[1])  # Second most likely class
    target_t = torch.tensor([target], dtype=torch.long, device=device)
    
    if verbose:
        print(f"Wachter Gradient: Original class {label_cf}, Target class {target}")

    # distance functions using torch tensors
    def dist(x, y):
        if distance == 'euclidean':
            return euclidean_dist(x, y)
        return manhattan_dist(x, y)

    # classification loss
    ce_loss = nn.CrossEntropyLoss()

    # loss function combining classification loss and distance
    def loss_fn(pred, cf):
        cls_term = ce_loss(pred, target_t)  # pred shape (1, num_classes)
        return lb * (cls_term ** 2) + dist(sample_t, cf)

    # initialize candidate counterfactual - start from original sample with noise
    sample_cf = sample_t.clone().detach()
    # Add small random noise to start optimization
    noise = torch.randn_like(sample_cf) * 0.1
    sample_cf = sample_cf + noise

    # if not fully random, seed candidate from a random dataset element
    if not full_random:
        dataset_len = len(dataset)
        ridx = int(torch.randint(0, dataset_len, (1,)).item())
        x0 = dataset[ridx][0]
        x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
        
        # Handle different input shapes for dataset sample
        if len(x0_tensor.shape) == 1:
            x0_tensor = x0_tensor.reshape(1, 1, -1)
        elif len(x0_tensor.shape) == 2:
            if x0_tensor.shape[0] > x0_tensor.shape[1]:
                x0_tensor = x0_tensor.T
            x0_tensor = x0_tensor.unsqueeze(0)
        
        sample_cf = x0_tensor.clone()

    # default lb and lb_step computed from current distance if not provided
    cur_dist = float(dist(sample_t, sample_cf).item())
    if lb is None:
        lb = max(0.1, cur_dist / 10.0)  # Start with smaller regularization
    if lb_step is None:
        lb_step = max(0.01, cur_dist / 100.0)  # Smaller step increases

    # ensure lb values are floats (used in loss arithmetic)
    lb = float(lb)
    lb_step = float(lb_step)

    sample_cf.requires_grad_(True)
    optimizer = Adam([sample_cf], lr=1e-2)  # Increased learning rate

    cfs = []
    best_validity = 0.0
    
    for iteration in range(max_cfs):
        optimizer.zero_grad()
        pred = model(sample_cf)
        loss = loss_fn(pred, sample_cf)
        loss.backward()
        optimizer.step()

        # gradually increase regularization weight (but more slowly)
        if iteration % 10 == 0:  # Only increase every 10 iterations
            lb += lb_step

        # record candidate
        y_cf = detach_to_numpy(model(sample_cf))[0]
        sample_cf_np = detach_to_numpy(sample_cf.squeeze(0))  # Remove batch dimension
        current_validity = y_cf[target]
        
        cfs.append([sample_cf_np, float(loss.item()), y_cf])
        
        # Track best validity
        if current_validity > best_validity:
            best_validity = current_validity

        # Debug output every 200 iterations
        if verbose and iteration % 200 == 0:
            pred_class = int(np.argmax(y_cf))
            print(f"Wachter Gradient iter {iteration}: pred_class={pred_class}, target={target}, "
                  f"validity={current_validity:.4f}, loss={loss.item():.4f}")

        # stop if prediction matches target
        if int(np.argmax(y_cf)) == target:
            if verbose:
                print(f"Wachter Gradient: Found counterfactual at iteration {iteration}")
            break
    
    if verbose:
        print(f"Wachter Gradient: Best validity achieved: {best_validity:.4f}")

    if not cfs:
        if verbose:
            print("Wachter Gradient: No counterfactual candidates found")
        return None, None

    # return best candidate by loss
    cfs = sorted(cfs, key=lambda x: x[1])
    best_cf = cfs[0][0]
    best_pred = cfs[0][2]
    
    # Convert back to original sample format
    if len(sample.shape) == 1:
        best_cf = best_cf.squeeze()  # Remove extra dimensions for 1D input
    elif len(sample.shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            best_cf = best_cf.T  # Convert back to (length, channels) if needed
    
    return best_cf, best_pred
