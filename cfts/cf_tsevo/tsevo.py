"""
TSEvo: Evolutionary Counterfactual Explanations for Time Series Classification

Paper: Höllig, J., Kulbach, C., & Thoma, S. (2022).
       "TSEvo: Evolutionary counterfactual explanations for time series classification."
       2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA), IEEE

Repository: https://github.com/fzi-forschungszentrum-informatik/TSInterpret

This implementation uses NSGA-II multi-objective evolutionary optimization to find
counterfactuals by optimizing three objectives:
1. Validity (target class probability)
2. Proximity (distance to original instance)
3. Sparsity (number of changed features)

The method uses reference set mutation, crossover, Gaussian mutation, and
segment-based swapping for diverse and realistic counterfactuals.
"""

import torch
import numpy as np
from typing import Optional, Tuple
import random


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data"""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device"""
    return torch.from_numpy(data).float().to(device)


class Individual:
    """Individual solution in the evolutionary algorithm"""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.fitness = [float('inf'), float('inf'), float('inf')]  # [validity, proximity, sparsity]
        
    def __repr__(self):
        return f"Individual(fitness={self.fitness})"


def tournament_selection(population, k=3):
    """Tournament selection for NSGA-II"""
    tournament = random.sample(population, k)
    # Select based on Pareto dominance and crowding distance
    return min(tournament, key=lambda x: (sum(x.fitness), random.random()))


def crossover(ind1, ind2):
    """Uniform crossover"""
    child1_data = ind1.data.copy()
    child2_data = ind2.data.copy()
    
    shape = child1_data.shape
    mask = np.random.rand(*shape) < 0.5
    
    temp = child1_data.copy()
    child1_data[mask] = child2_data[mask]
    child2_data[mask] = temp[mask]
    
    return Individual(child1_data), Individual(child2_data)


def mutate_gaussian(individual, mutation_rate=0.1, sigma=0.1):
    """Gaussian mutation"""
    mutated = individual.data.copy()
    mask = np.random.rand(*mutated.shape) < mutation_rate
    mutated[mask] += np.random.normal(0, sigma, size=mutated.shape)[mask]
    return Individual(mutated)


def mutate_swap_with_reference(individual, reference_set, window_size=None):
    """Swap segments with reference set (authentic opposing information)"""
    if len(reference_set) == 0:
        return individual
        
    mutated = individual.data.copy()
    sample_ref = random.choice(reference_set)
    
    # Ensure same shape
    if sample_ref.shape != mutated.shape:
        return individual
    
    if window_size is None:
        window_size = max(1, mutated.shape[-1] // 10)
    
    # Random window to swap
    if mutated.ndim == 1:
        length = mutated.shape[0]
        if length > window_size:
            start = np.random.randint(0, length - window_size)
            mutated[start:start+window_size] = sample_ref[start:start+window_size]
    elif mutated.ndim == 2:
        # For multivariate, swap random channel segments
        C, L = mutated.shape
        channel = np.random.randint(0, C)
        if L > window_size:
            start = np.random.randint(0, L - window_size)
            mutated[channel, start:start+window_size] = sample_ref[channel, start:start+window_size]
    
    return Individual(mutated)


def pareto_dominates(fitness1, fitness2):
    """Check if fitness1 dominates fitness2 (all <= and at least one <)"""
    return all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2)) and \
           any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))


def fast_non_dominated_sort(population):
    """NSGA-II fast non-dominated sorting"""
    fronts = [[]]
    domination_counts = [0] * len(population)
    dominated_solutions = [[] for _ in range(len(population))]
    
    for i, ind_i in enumerate(population):
        for j, ind_j in enumerate(population):
            if i == j:
                continue
            if pareto_dominates(ind_i.fitness, ind_j.fitness):
                dominated_solutions[i].append(j)
            elif pareto_dominates(ind_j.fitness, ind_i.fitness):
                domination_counts[i] += 1
        
        if domination_counts[i] == 0:
            fronts[0].append(i)
    
    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)
    
    return fronts[:-1]  # Remove last empty front


def nsga2_select(population, offspring, pop_size):
    """NSGA-II selection"""
    combined = population + offspring
    fronts = fast_non_dominated_sort(combined)
    
    new_population = []
    for front_indices in fronts:
        if len(new_population) + len(front_indices) <= pop_size:
            new_population.extend([combined[i] for i in front_indices])
        else:
            # Fill remaining slots from this front
            remaining = pop_size - len(new_population)
            # Simple selection: take first 'remaining' individuals
            new_population.extend([combined[i] for i in front_indices[:remaining]])
            break
    
    return new_population[:pop_size]


def tsevo_cf(
    sample: np.ndarray,
    dataset,
    model: torch.nn.Module,
    target_class: Optional[int] = None,
    population_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual explanation using TSEvo algorithm.
    
    TSEvo uses NSGA-II multi-objective evolutionary optimization with three objectives:
    - Validity: maximize probability of target class
    - Proximity: minimize distance to original instance
    - Sparsity: minimize number of changed features
    
    Args:
        sample: Original time series sample
        dataset: Dataset object (for compatibility with other methods)
        model: Trained classification model
        target_class: Target class for counterfactual (if None, finds different class)
        population_size: Size of evolutionary population
        generations: Number of evolutionary generations
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
        device: Device to run model on
        verbose: If True, print progress information
        
    Returns:
        Tuple of (counterfactual, prediction) where:
        - counterfactual: Generated counterfactual time series (same shape as input)
        - prediction: Model's prediction probabilities for the counterfactual
        Returns (None, None) if no valid counterfactual is found
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    model.eval()
    
    # Normalize sample to (C, L) format
    sample_original = np.array(sample, dtype=np.float64)
    original_shape = sample_original.shape
    
    if sample_original.ndim == 1:
        sample_cf = sample_original.reshape(1, -1)
    elif sample_original.ndim == 2:
        sample_cf = sample_original.copy()
    else:
        if verbose:
            print(f"Unsupported sample shape: {sample_original.shape}")
        return None, None
    
    C, L = sample_cf.shape
    
    # Get original prediction
    with torch.no_grad():
        sample_tensor = numpy_to_torch(sample_cf.reshape(1, C, L), device)
        orig_pred = model(sample_tensor)
        orig_probs = torch.softmax(orig_pred, dim=-1).squeeze()
        orig_class = torch.argmax(orig_probs).item()
        orig_probs_np = detach_to_numpy(orig_probs)
    
    # Determine target class
    if target_class is None:
        # Choose class with second highest probability
        sorted_classes = np.argsort(orig_probs_np)[::-1]
        for cls in sorted_classes:
            if cls != orig_class:
                target_class = cls
                break
        if target_class is None:
            target_class = (orig_class + 1) % len(orig_probs_np)
    
    if verbose:
        print(f"Original class: {orig_class}, Target class: {target_class}")
        print(f"Original prediction: {orig_probs_np}")
    
    # Build reference set (samples of target class)
    reference_set = []
    try:
        for i in range(len(dataset)):
            x, y = dataset[i]
            x_arr = np.array(x, dtype=np.float64)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(1, -1)
            
            # Get prediction for this sample
            with torch.no_grad():
                x_tensor = numpy_to_torch(x_arr.reshape(1, C, L), device)
                pred = model(x_tensor)
                pred_class = torch.argmax(pred, dim=-1).item()
            
            if pred_class == target_class:
                reference_set.append(x_arr)
            
            if len(reference_set) >= 50:  # Limit reference set size
                break
    except:
        reference_set = []
    
    if verbose:
        print(f"Reference set size: {len(reference_set)}")
    
    # Fitness evaluation function
    def evaluate_fitness(individual):
        """Evaluate three objectives: validity, proximity, sparsity"""
        ind_data = individual.data
        
        # Ensure correct shape
        if ind_data.shape != sample_cf.shape:
            ind_data = ind_data.reshape(C, L)
        
        with torch.no_grad():
            ind_tensor = numpy_to_torch(ind_data.reshape(1, C, L), device)
            pred = model(ind_tensor)
            probs = torch.softmax(pred, dim=-1).squeeze()
            probs_np = detach_to_numpy(probs)
        
        # Objective 1: Validity (minimize distance from target class probability to 1.0)
        validity = 1.0 - probs_np[target_class]
        
        # Objective 2: Proximity (L2 distance to original)
        proximity = np.sqrt(np.sum((ind_data - sample_cf) ** 2)) / (C * L)
        
        # Objective 3: Sparsity (ratio of changed values)
        sparsity = np.count_nonzero(ind_data - sample_cf) / (C * L)
        
        individual.fitness = [validity, proximity, sparsity]
        return individual
    
    # Initialize population
    population = []
    for _ in range(population_size):
        if len(reference_set) > 0 and random.random() < 0.5:
            # Initialize from reference set
            init_data = random.choice(reference_set).copy()
        else:
            # Initialize with small perturbation of original
            init_data = sample_cf.copy() + np.random.normal(0, 0.1, sample_cf.shape)
        
        ind = Individual(init_data)
        evaluate_fitness(ind)
        population.append(ind)
    
    # Evolution loop
    best_individual = None
    best_validity = float('inf')
    
    for gen in range(generations):
        offspring = []
        
        while len(offspring) < population_size:
            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = Individual(parent1.data.copy()), Individual(parent2.data.copy())
            
            # Mutation
            if random.random() < mutation_rate:
                if len(reference_set) > 0 and random.random() < 0.5:
                    child1 = mutate_swap_with_reference(child1, reference_set)
                else:
                    child1 = mutate_gaussian(child1, mutation_rate=0.2)
            
            if random.random() < mutation_rate:
                if len(reference_set) > 0 and random.random() < 0.5:
                    child2 = mutate_swap_with_reference(child2, reference_set)
                else:
                    child2 = mutate_gaussian(child2, mutation_rate=0.2)
            
            # Evaluate offspring
            evaluate_fitness(child1)
            evaluate_fitness(child2)
            
            offspring.extend([child1, child2])
        
        # Selection for next generation
        population = nsga2_select(population, offspring[:population_size], population_size)
        
        # Track best individual (lowest validity score and valid class)
        for ind in population:
            if ind.fitness[0] < best_validity:
                # Check if it actually predicts target class
                with torch.no_grad():
                    ind_tensor = numpy_to_torch(ind.data.reshape(1, C, L), device)
                    pred = model(ind_tensor)
                    pred_class = torch.argmax(pred, dim=-1).item()
                
                if pred_class == target_class:
                    best_validity = ind.fitness[0]
                    best_individual = ind
        
        if verbose and gen % 20 == 0:
            avg_validity = np.mean([ind.fitness[0] for ind in population])
            avg_proximity = np.mean([ind.fitness[1] for ind in population])
            avg_sparsity = np.mean([ind.fitness[2] for ind in population])
            print(f"Gen {gen}: Avg Validity={avg_validity:.4f}, Proximity={avg_proximity:.4f}, Sparsity={avg_sparsity:.4f}")
            if best_individual:
                print(f"  Best: {best_individual.fitness}")
    
    # Return best individual if found
    if best_individual is not None:
        cf_data = best_individual.data
        
        # Reshape to original format
        if original_shape != cf_data.shape:
            if len(original_shape) == 1:
                cf_data = cf_data.reshape(-1)
            else:
                cf_data = cf_data.reshape(original_shape)
        
        # Get final prediction
        with torch.no_grad():
            if cf_data.ndim == 1:
                cf_tensor = numpy_to_torch(cf_data.reshape(1, 1, -1), device)
            elif cf_data.ndim == 2:
                cf_tensor = numpy_to_torch(cf_data.reshape(1, C, L), device)
            else:
                cf_tensor = numpy_to_torch(cf_data, device)
            
            final_pred = model(cf_tensor)
            final_probs = torch.softmax(final_pred, dim=-1).squeeze()
            final_probs_np = detach_to_numpy(final_probs)
        
        if verbose:
            print(f"\nFinal counterfactual prediction: {final_probs_np}")
            print(f"Predicted class: {np.argmax(final_probs_np)}")
        
        return cf_data, final_probs_np
    
    # If no valid counterfactual found, return best from first Pareto front
    fronts = fast_non_dominated_sort(population)
    if len(fronts) > 0 and len(fronts[0]) > 0:
        best_front_ind = population[fronts[0][0]]
        cf_data = best_front_ind.data
        
        if original_shape != cf_data.shape:
            if len(original_shape) == 1:
                cf_data = cf_data.reshape(-1)
            else:
                cf_data = cf_data.reshape(original_shape)
        
        with torch.no_grad():
            if cf_data.ndim == 1:
                cf_tensor = numpy_to_torch(cf_data.reshape(1, 1, -1), device)
            elif cf_data.ndim == 2:
                cf_tensor = numpy_to_torch(cf_data.reshape(1, C, L), device)
            else:
                cf_tensor = numpy_to_torch(cf_data, device)
            
            final_pred = model(cf_tensor)
            final_probs = torch.softmax(final_pred, dim=-1).squeeze()
            final_probs_np = detach_to_numpy(final_probs)
        
        if verbose:
            print("\nNo perfect counterfactual found, returning best Pareto solution")
            print(f"Final prediction: {final_probs_np}")
        
        return cf_data, final_probs_np
    
    if verbose:
        print("Failed to generate counterfactual")
    
    return None, None
