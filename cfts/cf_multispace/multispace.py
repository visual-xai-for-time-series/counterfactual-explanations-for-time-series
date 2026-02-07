import torch
import numpy as np
import copy
from captum.attr import GradientShap


####
# Multi-SpaCE: Multi-Objective Subsequence-based Sparse Counterfactual Explanations
#
# Paper: Refoyo, M., & Luengo, D. (2024).
#        "Multi-SpaCE: Multi-Objective Subsequence-based Sparse Counterfactual Explanations
#        for Multivariate Time Series Classification."
#        arXiv preprint arXiv:2501.04009
#
# Repository: https://github.com/MarioRefoyo/Multi-SpaCE
#
# Multi-SpaCE uses feature importance for guided initialization, subsequence
# optimization, and evolutionary search to generate diverse, sparse counterfactuals
# for multivariate time series.
####


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


def _ensure_ncl(sample, dataset):
    """Ensure sample and dataset are shaped (C, L) and (N, C, L) respectively.
    
    Heuristic: for 2D arrays, if rows <= cols treat as (C, L), else treat as
    (L, C) and transpose. This lets us cheaply detect already (N, C, L).
    """
    # normalize sample to (C, L)
    s = np.asarray(sample)
    if s.ndim == 1:
        s_ncl = s.reshape(1, -1)
        ori = "1d"
    elif s.ndim == 2:
        r, c = s.shape
        if r <= c:
            s_ncl = s.copy()
            ori = "cf"
        else:
            s_ncl = s.T.copy()
            ori = "tf"
    else:
        raise ValueError("sample must be 1D or 2D time series")

    # build time_series_data as (N, C, L) with a single vectorized pass
    first = dataset[0][0]
    first_arr = np.asarray(first)
    if first_arr.ndim == 3 and isinstance(dataset, np.ndarray):
        ts = np.asarray([x for x in dataset[:, 0]])
    else:
        fa = first_arr
        if fa.ndim == 1:
            ts = np.stack([np.asarray(x[0]).reshape(1, -1) for x in dataset], axis=0)
        elif fa.ndim == 2:
            r, c = fa.shape
            if r <= c:
                ts = np.stack([np.asarray(x[0]) for x in dataset], axis=0)
            else:
                ts = np.stack([np.asarray(x[0]).T for x in dataset], axis=0)
        else:
            raise ValueError("dataset items must be 1D or 2D time series")

    _, L = s_ncl.shape
    if ts.shape[-1] != L:
        raise ValueError("All series must have same length as sample")

    C_sample = s_ncl.shape[0]
    C_data = ts.shape[1]
    if C_data != C_sample:
        if C_data == 1:
            ts = np.repeat(ts, C_sample, axis=1)
        else:
            raise ValueError("Channel count mismatch between sample and dataset")

    return s_ncl, ts, ori


def _revert_orientation(cf_arr, orientation):
    """Revert counterfactual to original orientation."""
    if orientation == "1d":
        return cf_arr.reshape(-1)
    if orientation == "cf":
        return cf_arr
    if orientation == "tf":
        return cf_arr.T
    return cf_arr


def calculate_feature_importance(model, sample, dataset, device):
    """Calculate feature importance using GradientShap.
    
    Args:
        model: PyTorch model
        sample: Sample to explain (C, L)
        dataset: Training dataset for baselines
        device: Device to run on
        
    Returns:
        importance: Feature importance scores (C, L)
    """
    weights = GradientShap(model)
    sample_tensor = numpy_to_torch(sample.reshape(1, sample.shape[0], sample.shape[1]), device)
    
    # Use subset of training data as baselines
    n_baselines = min(50, len(dataset))
    baseline_indices = np.random.choice(len(dataset), n_baselines, replace=False)
    baselines_list = []
    for idx in baseline_indices:
        item = dataset[idx][0]
        if isinstance(item, np.ndarray):
            baselines_list.append(item)
        else:
            baselines_list.append(np.asarray(item))
    
    baselines = np.stack(baselines_list, axis=0)
    if baselines.ndim == 2:
        baselines = baselines.reshape(n_baselines, 1, -1)
    elif baselines.ndim == 3 and baselines.shape[1] > baselines.shape[2]:
        baselines = np.transpose(baselines, (0, 2, 1))
    
    baselines_tensor = numpy_to_torch(baselines, device)
    
    # Get prediction for target class
    pred = model(sample_tensor)
    target_class = int(torch.argmax(pred, dim=1)[0])
    
    attributions = weights.attribute(sample_tensor, baselines=baselines_tensor, target=target_class)
    attr_np = detach_to_numpy(attributions)
    
    if attr_np.ndim == 3:
        importance = np.abs(attr_np[0])  # (C, L)
    else:
        importance = np.abs(attr_np)
    
    return importance


def multi_space_cf(sample, dataset, model, weight_function=GradientShap, 
                   iterate=None, sub_len=1, population_size=50, max_iterations=100,
                   sparsity_weight=0.3, validity_weight=0.7, verbose=False):
    """Multi-SpaCE counterfactual generation with multi-objective optimization.
    
    This is a simplified implementation of Multi-SpaCE that uses:
    - Feature importance for initialization
    - Subsequence-based optimization
    - Multi-objective fitness (validity, sparsity, plausibility)
    
    Args:
        sample: Input time series
        dataset: Training dataset
        model: Classifier model
        weight_function: Attribution method (default GradientShap)
        iterate: Number of iterations (default: sequence length)
        sub_len: Starting subsequence length
        population_size: Number of candidates to maintain
        max_iterations: Maximum optimization iterations
        sparsity_weight: Weight for sparsity objective
        validity_weight: Weight for validity objective
        
    Returns:
        cf: Counterfactual explanation
        y_cf: Prediction probabilities
    """
    device = next(model.parameters()).device

    def model_predict(arr):
        # arr expected shape (B, C, L)
        return detach_to_numpy(model(numpy_to_torch(arr, device)))

    # Prepare sample and dataset in (C, L) and (N, C, L)
    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    N, C, L = time_series_data.shape

    if iterate is None:
        iterate = L

    # Get predictions
    preds_data = model_predict(time_series_data)
    preds_sample = model_predict(sample_cf.reshape(1, C, L))
    label_data = np.argmax(preds_data, axis=1)
    label_sample = int(np.argmax(preds_sample))

    # Select candidates with different label
    mask = label_data != label_sample
    if not np.any(mask):
        return _revert_orientation(sample_cf, sample_ori), preds_sample.reshape(-1)

    candidates = time_series_data[mask]
    candidates_labels = label_data[mask]

    # Find native guide (nearest unlike neighbor - NUN)
    distances = np.sum((candidates.reshape(len(candidates), -1) - sample_cf.reshape(1, -1))**2, axis=1)
    nun_idx = np.argmin(distances)
    native_guide = candidates[nun_idx]
    cf_label = int(candidates_labels[nun_idx])

    # Calculate feature importance for both sample and NUN
    importance_sample = calculate_feature_importance(model, sample_cf, dataset, device)
    importance_nun = calculate_feature_importance(model, native_guide.reshape(C, L), dataset, device)
    
    # Combined importance heatmap
    combined_importance = (importance_sample + importance_nun) / 2
    
    # Sum over channels to get time importance
    time_importance = np.sum(combined_importance, axis=0)  # (L,)
    
    # Initialize population of masks (which subsequences to replace)
    # Use feature importance to guide initialization
    population = []
    
    for _ in range(population_size):
        # Create binary mask indicating which positions to replace
        mask_cf = np.zeros((C, L), dtype=bool)
        
        # Start with most important regions
        n_points = np.random.randint(1, max(2, L // 4))
        
        # Select positions based on importance
        probs = time_importance / (time_importance.sum() + 1e-10)
        selected_positions = np.random.choice(L, size=min(n_points, L), 
                                             replace=False, p=probs)
        
        # Create subsequences around selected positions
        for pos in selected_positions:
            subseq_len = np.random.randint(1, min(sub_len + 5, L // 2))
            start = max(0, pos - subseq_len // 2)
            end = min(L, start + subseq_len)
            mask_cf[:, start:end] = True
        
        population.append(mask_cf)
    
    population = np.array(population)
    
    # Evolutionary optimization
    best_cf = sample_cf.copy()
    best_fitness = -np.inf
    best_probs = preds_sample.reshape(-1)
    
    for iteration in range(max_iterations):
        # Generate counterfactuals from population
        cfs = np.zeros((population_size, C, L))
        for i, mask in enumerate(population):
            cf_candidate = sample_cf.copy()
            cf_candidate[mask] = native_guide[mask]
            cfs[i] = cf_candidate
        
        # Evaluate fitness
        preds = model_predict(cfs)
        
        # Multi-objective fitness
        validity = preds[:, cf_label]  # Probability of target class
        sparsity = 1.0 - (population.sum(axis=(1, 2)) / (C * L))  # Fewer changes is better
        
        # Count subsequences (fewer is better)
        n_subsequences = np.zeros(population_size)
        for i, mask in enumerate(population):
            # Count transitions in each channel
            for c in range(C):
                transitions = np.diff(mask[c].astype(int), prepend=0)
                n_subsequences[i] += np.sum(transitions == 1)
        
        # Normalize subsequences
        max_subseq = max(n_subsequences.max(), 1)
        subsequence_score = 1.0 - (n_subsequences / max_subseq)
        
        # Combined fitness
        fitness = (validity_weight * validity + 
                  sparsity_weight * sparsity * 0.5 + 
                  sparsity_weight * subsequence_score * 0.5)
        
        # Check for valid counterfactuals
        valid_indices = np.where(np.argmax(preds, axis=1) == cf_label)[0]
        
        if len(valid_indices) > 0:
            # Among valid ones, select sparsest
            valid_fitness = sparsity[valid_indices] + subsequence_score[valid_indices]
            best_valid_idx = valid_indices[np.argmax(valid_fitness)]
            
            if fitness[best_valid_idx] > best_fitness:
                best_fitness = fitness[best_valid_idx]
                best_cf = cfs[best_valid_idx]
                best_probs = preds[best_valid_idx]
                
            # If we found a good valid CF, we can stop early
            if sparsity[best_valid_idx] > 0.7:  # 70% unchanged
                break
        
        # Update best overall if better
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fitness:
            best_fitness = fitness[best_idx]
            best_cf = cfs[best_idx]
            best_probs = preds[best_idx]
        
        # Selection and mutation for next generation
        if iteration < max_iterations - 1:
            # Select elite individuals
            elite_size = max(2, population_size // 10)
            elite_indices = np.argsort(fitness)[-elite_size:]
            new_population = [population[i].copy() for i in elite_indices]
            
            # Generate offspring through mutation
            while len(new_population) < population_size:
                # Select parent
                parent_idx = np.random.choice(elite_indices)
                child = population[parent_idx].copy()
                
                # Mutation: flip some positions
                mutation_rate = 0.05
                for c in range(C):
                    for t in range(L):
                        if np.random.random() < mutation_rate:
                            child[c, t] = not child[c, t]
                
                # Mutation: add/remove subsequence
                if np.random.random() < 0.3:
                    subseq_len = np.random.randint(1, L // 4)
                    start = np.random.randint(0, L - subseq_len)
                    if np.random.random() < 0.5:
                        # Add subsequence
                        child[:, start:start + subseq_len] = True
                    else:
                        # Remove subsequence
                        child[:, start:start + subseq_len] = False
                
                new_population.append(child)
            
            population = np.array(new_population[:population_size])
    
    # Revert to original orientation
    cf_out = _revert_orientation(best_cf, sample_ori)
    return cf_out, best_probs
