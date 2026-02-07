import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from sklearn.metrics import pairwise_distances
import random


####
# MOC/DANDL: Multi-Objective Counterfactuals
#
# Paper: Dandl, S., Molnar, C., Binder, M., & Bischl, B. (2020).
#        "Multi-objective counterfactual explanations."
#        arXiv preprint arXiv:2004.11165
#
# Repository: https://github.com/susanne-207/moc
#
# MOC uses multi-objective evolutionary optimization to find Pareto-optimal
# counterfactuals that balance validity, proximity, sparsity, and plausibility.
# Returns multiple diverse counterfactuals on the Pareto frontier.
####


def moc_cf(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    population_size: int = 100,
    generations: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    tournament_size: int = 3,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual explanation using MOC (Multi-Objective Counterfactuals) algorithm.
    
    MOC uses multi-objective evolutionary optimization to find counterfactuals on the Pareto frontier,
    optimizing for validity, proximity, sparsity, and plausibility simultaneously.
    
    Args:
        sample: Original time series sample
        dataset: Dataset object (for compatibility with other methods)
        model: Trained classification model
        target_class: Target class for counterfactual (if None, finds different class)
        population_size: Size of the evolutionary population
        generations: Number of evolutionary generations
        mutation_rate: Probability of mutation for each gene
        crossover_rate: Probability of crossover between parents
        tournament_size: Size of tournament for parent selection
        device: Device to run on (if None, auto-detects)
        
    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Convert sample to tensor for model prediction
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    
    # Handle different input shapes
    original_shape = sample.shape
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.reshape(1, 1, -1)
    elif len(x_tensor.shape) == 2:
        if x_tensor.shape[0] > x_tensor.shape[1]:
            x_tensor = x_tensor.T
        x_tensor = x_tensor.unsqueeze(0)
    
    # Get original prediction
    with torch.no_grad():
        original_pred = model(x_tensor)
        original_class = torch.argmax(original_pred, dim=-1).item()
        original_pred_np = torch.softmax(original_pred, dim=-1).squeeze().cpu().numpy()
    
    # Determine target class
    if target_class is None:
        sorted_classes = torch.argsort(original_pred, dim=-1, descending=True)
        target_class = sorted_classes[0, 1].item()
    
    if original_class == target_class:
        return None, None
    
    if verbose:
        print(f"MOC: Original class {original_class}, Target class {target_class}")

    # Initialize MOC optimizer
    moc = MOCOptimizer(
        original_sample=sample,
        model=model,
        target_class=target_class,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        device=device,
        verbose=verbose
    )
    
    # Run multi-objective optimization
    pareto_front = moc.optimize(generations)
    
    if verbose:
        print(f"MOC: Found {len(pareto_front) if pareto_front else 0} solutions in Pareto front")
    
    if not pareto_front:
        if verbose:
            print("MOC: No counterfactual found - empty Pareto front")
        return None, None    # Select best counterfactual from Pareto front (highest validity, then lowest distance)
    best_cf = None
    best_score = -1
    
    for individual in pareto_front:
        cf_sample, objectives = individual
        validity, proximity, sparsity, plausibility = objectives
        
        # Prioritize validity heavily
        score = validity * 10 - proximity * 0.01  # Much stronger emphasis on validity
        if score > best_score:
            best_score = score
            best_cf = cf_sample
    
    if best_cf is None:
        if verbose:
            print("MOC: No valid solution found in Pareto front")
        return None, None
    
    # Get final prediction
    cf_tensor = torch.tensor(best_cf, dtype=torch.float32, device=device)
    if len(cf_tensor.shape) == 1:
        cf_tensor = cf_tensor.reshape(1, 1, -1)
    elif len(cf_tensor.shape) == 2:
        if cf_tensor.shape[0] > cf_tensor.shape[1]:
            cf_tensor = cf_tensor.T
        cf_tensor = cf_tensor.unsqueeze(0)
    
    with torch.no_grad():
        final_pred = model(cf_tensor)
        final_pred_np = torch.softmax(final_pred, dim=-1).squeeze().cpu().numpy()
        predicted_class = torch.argmax(final_pred, dim=-1).item()
        final_validity = final_pred_np[target_class]
    
    if verbose:
        print(f"MOC final: pred_class={predicted_class}, target={target_class}, validity={final_validity:.4f}")
    
    # Use relaxed validation like other methods
    if predicted_class != target_class and final_validity < 0.4:
        if verbose:
            print(f"MOC: Counterfactual failed validation - predicted {predicted_class}, wanted {target_class}, validity too low")
        return None, None
    
    return best_cf, final_pred_np


class MOCOptimizer:
    """Multi-Objective Counterfactual Optimizer using NSGA-II inspired approach."""
    
    def __init__(
        self,
        original_sample: np.ndarray,
        model: nn.Module,
        target_class: int,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        device: str = 'cpu',
        verbose: bool = False
    ):
        self.original_sample = original_sample
        self.model = model
        self.target_class = target_class
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.device = device
        self.verbose = verbose
        
        # Calculate statistics for realistic bounds
        if hasattr(self.original_sample, 'flatten'):
            flat_sample = self.original_sample.flatten()
            self.sample_mean = np.mean(flat_sample)
            self.sample_std = np.std(flat_sample)
            self.sample_min = np.min(flat_sample)
            self.sample_max = np.max(flat_sample)
        else:
            self.sample_mean = 0.0
            self.sample_std = 1.0
            self.sample_min = -3.0
            self.sample_max = 3.0
    
    def initialize_population(self) -> List[np.ndarray]:
        """Initialize random population around the original sample."""
        population = []
        
        for _ in range(self.population_size):
            # Start with original sample and add noise
            noise_scale = self.sample_std * 0.1
            noise = np.random.normal(0, noise_scale, self.original_sample.shape)
            individual = self.original_sample + noise
            
            # Clip to reasonable bounds
            individual = np.clip(individual, 
                               self.sample_min - 2 * self.sample_std,
                               self.sample_max + 2 * self.sample_std)
            
            population.append(individual)
        
        return population
    
    def evaluate_objectives(self, individual: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Evaluate the four objectives for MOC:
        1. Validity: How well does it achieve the target class?
        2. Proximity: How close is it to the original?
        3. Sparsity: How many features were changed?
        4. Plausibility: How realistic is the counterfactual?
        """
        # Convert to tensor for model prediction
        x_tensor = torch.tensor(individual, dtype=torch.float32, device=self.device)
        
        if len(x_tensor.shape) == 1:
            x_tensor = x_tensor.reshape(1, 1, -1)
        elif len(x_tensor.shape) == 2:
            if x_tensor.shape[0] > x_tensor.shape[1]:
                x_tensor = x_tensor.T
            x_tensor = x_tensor.unsqueeze(0)
        
        # 1. Validity: Probability of target class
        with torch.no_grad():
            pred = self.model(x_tensor)
            probs = torch.softmax(pred, dim=-1)
            validity = probs[0, self.target_class].item()
        
        # 2. Proximity: L2 distance (to minimize)
        proximity = np.linalg.norm(individual - self.original_sample)
        
        # 3. Sparsity: Number of changed features (to minimize)
        diff = np.abs(individual - self.original_sample)
        threshold = self.sample_std * 0.01  # 1% of std as threshold for "changed"
        sparsity = np.sum(diff > threshold) / diff.size
        
        # 4. Plausibility: How realistic the values are (to maximize)
        # Simple measure: inverse of how far values are from typical range
        z_scores = np.abs((individual - self.sample_mean) / (self.sample_std + 1e-8))
        plausibility = 1.0 / (1.0 + np.mean(z_scores))
        
        return validity, proximity, sparsity, plausibility
    
    def is_dominated(self, obj1: Tuple[float, float, float, float], 
                    obj2: Tuple[float, float, float, float]) -> bool:
        """Check if obj1 is dominated by obj2 (obj2 is better in all objectives)."""
        # For MOC: maximize validity and plausibility, minimize proximity and sparsity
        v1, p1, s1, pl1 = obj1
        v2, p2, s2, pl2 = obj2
        
        # obj2 dominates obj1 if obj2 is better or equal in all objectives
        # and strictly better in at least one
        better_or_equal = (v2 >= v1) and (p2 <= p1) and (s2 <= s1) and (pl2 >= pl1)
        strictly_better = (v2 > v1) or (p2 < p1) or (s2 < s1) or (pl2 > pl1)
        
        return better_or_equal and strictly_better
    
    def fast_non_dominated_sort(self, population: List[np.ndarray], 
                               objectives: List[Tuple[float, float, float, float]]) -> List[List[int]]:
        """NSGA-II fast non-dominated sorting."""
        n = len(population)
        domination_counts = [0] * n  # Number of solutions that dominate solution i
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by solution i
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.is_dominated(objectives[i], objectives[j]):
                        domination_counts[i] += 1
                    elif self.is_dominated(objectives[j], objectives[i]):
                        dominated_solutions[i].append(j)
            
            if domination_counts[i] == 0:
                fronts[0].append(i)
        
        front_idx = 0
        while front_idx < len(fronts) and len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            
            if len(next_front) > 0:
                fronts.append(next_front)
            front_idx += 1
        
        # Remove empty fronts
        return [front for front in fronts if len(front) > 0]

    def crowding_distance(self, front: List[int], 
                         objectives: List[Tuple[float, float, float, float]]) -> List[float]:
        """Calculate crowding distance for diversity preservation."""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        distances = [0.0] * len(front)
        n_objectives = 4
        
        for obj_idx in range(n_objectives):
            # Sort by objective value and keep track of original indices
            front_with_values = [(idx, objectives[idx][obj_idx]) for idx in front]
            front_sorted = sorted(front_with_values, key=lambda x: x[1])
            
            # Boundary points get infinite distance
            front_indices = [x[0] for x in front_sorted]
            distances[front.index(front_indices[0])] = float('inf')
            distances[front.index(front_indices[-1])] = float('inf')
            
            # Calculate range
            obj_values = [x[1] for x in front_sorted]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range == 0:
                continue
            
            # Calculate distances for intermediate points
            for i in range(1, len(front_sorted) - 1):
                idx_in_front = front.index(front_indices[i])
                if distances[idx_in_front] != float('inf'):
                    distances[idx_in_front] += (obj_values[i+1] - obj_values[i-1]) / obj_range
        
        return distances
    
    def tournament_selection(self, population: List[np.ndarray], 
                           objectives: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Tournament selection for parent selection."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        
        # Select best from tournament (first front, then crowding distance)
        fronts = self.fast_non_dominated_sort([population[i] for i in tournament_indices],
                                            [objectives[i] for i in tournament_indices])
        
        # Choose from first front
        if len(fronts[0]) == 1:
            return population[tournament_indices[fronts[0][0]]]
        
        # Use crowding distance for tie-breaking
        distances = self.crowding_distance(fronts[0], [objectives[tournament_indices[i]] for i in fronts[0]])
        best_idx = fronts[0][np.argmax(distances)]
        
        return population[tournament_indices[best_idx]]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX) for real-valued optimization."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        eta = 20  # Distribution index
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Flatten for easier processing
        flat1 = offspring1.flatten()
        flat2 = offspring2.flatten()
        
        for i in range(len(flat1)):
            if random.random() <= 0.5:  # 50% chance for each gene
                y1, y2 = flat1[i], flat2[i]
                
                if abs(y1 - y2) > 1e-14:
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    # Calculate beta
                    rand = random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
                    
                    # Create offspring
                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    flat1[i] = c1
                    flat2[i] = c2
        
        # Reshape back
        offspring1 = flat1.reshape(parent1.shape)
        offspring2 = flat2.reshape(parent2.shape)
        
        return offspring1, offspring2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation for real-valued optimization."""
        eta_m = 20  # Distribution index for mutation
        mutated = individual.copy()
        flat = mutated.flatten()
        
        for i in range(len(flat)):
            if random.random() < self.mutation_rate:
                # Current value
                y = flat[i]
                
                # Calculate bounds (simplified)
                lower = self.sample_min - 2 * self.sample_std
                upper = self.sample_max + 2 * self.sample_std
                
                # Polynomial mutation
                delta1 = (y - lower) / (upper - lower)
                delta2 = (upper - y) / (upper - lower)
                
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                # Apply mutation
                flat[i] = y + delta_q * (upper - lower)
                
                # Ensure bounds
                flat[i] = max(lower, min(upper, flat[i]))
        
        return flat.reshape(individual.shape)
    
    def optimize(self, generations: int) -> List[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        """Run the multi-objective optimization."""
        if self.verbose:
            print(f"MOC: Starting optimization with {self.population_size} individuals for {generations} generations")
        
        # Initialize population
        population = self.initialize_population()
        if self.verbose:
            print(f"MOC: Initialized population of {len(population)} individuals")
        
        best_validity = 0.0
        valid_solutions = 0
        
        for generation in range(generations):
            # Evaluate objectives
            objectives = [self.evaluate_objectives(ind) for ind in population]
            
            # Track statistics
            validities = [obj[0] for obj in objectives]
            current_best_validity = max(validities)
            current_valid_solutions = sum(1 for v in validities if v > 0.5)
            
            if current_best_validity > best_validity:
                best_validity = current_best_validity
            if current_valid_solutions > valid_solutions:
                valid_solutions = current_valid_solutions
            
            # Print progress every 10 generations
            if generation % 10 == 0:
                if self.verbose:
                    print(f"MOC generation {generation}: best_validity={current_best_validity:.4f}, "
                          f"valid_solutions={current_valid_solutions}/{len(population)}")
            
            # Generate offspring
            offspring = []
            for _ in range(self.population_size // 2):
                # Select parents
                try:
                    parent1 = self.tournament_selection(population, objectives)
                    parent2 = self.tournament_selection(population, objectives)
                    
                    # Crossover
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    # Mutation
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    offspring.extend([child1, child2])
                except Exception as e:
                    if self.verbose:
                        print(f"MOC: Error in generation {generation}: {e}")
                    # Add random individuals if selection fails
                    noise1 = np.random.normal(0, self.sample_std * 0.2, self.original_sample.shape)
                    noise2 = np.random.normal(0, self.sample_std * 0.2, self.original_sample.shape)
                    offspring.extend([self.original_sample + noise1, self.original_sample + noise2])
            
            # Combine population and offspring
            combined_pop = population + offspring[:self.population_size]
            combined_obj = objectives + [self.evaluate_objectives(ind) for ind in offspring[:self.population_size]]
            
            # Select next generation using NSGA-II
            try:
                fronts = self.fast_non_dominated_sort(combined_pop, combined_obj)
            except Exception as e:
                if self.verbose:
                    print(f"MOC: Error in non-dominated sorting: {e}")
                # Fallback: select by validity only
                combined_with_scores = list(zip(combined_pop, combined_obj))
                combined_with_scores.sort(key=lambda x: x[1][0], reverse=True)  # Sort by validity
                population = [x[0] for x in combined_with_scores[:self.population_size]]
                continue
            
            new_population = []
            new_objectives = []
            
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    # Add entire front
                    for idx in front:
                        new_population.append(combined_pop[idx])
                        new_objectives.append(combined_obj[idx])
                else:
                    # Add part of front based on crowding distance
                    try:
                        distances = self.crowding_distance(front, combined_obj)
                        sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                        
                        remaining = self.population_size - len(new_population)
                        for idx, _ in sorted_front[:remaining]:
                            new_population.append(combined_pop[idx])
                            new_objectives.append(combined_obj[idx])
                    except Exception as e:
                        if self.verbose:
                            print(f"MOC: Error in crowding distance: {e}")
                        # Fallback: add first few from front
                        remaining = self.population_size - len(new_population)
                        for idx in front[:remaining]:
                            new_population.append(combined_pop[idx])
                            new_objectives.append(combined_obj[idx])
                    break
            
            population = new_population
            objectives = new_objectives
        
        if self.verbose:
            print(f"MOC: Optimization complete. Best validity achieved: {best_validity:.4f}")
        
        # Return Pareto front
        final_objectives = [self.evaluate_objectives(ind) for ind in population]
        fronts = self.fast_non_dominated_sort(population, final_objectives)
        
        pareto_front = []
        if fronts and len(fronts) > 0:
            if self.verbose:
                print(f"MOC: First front has {len(fronts[0])} solutions")
            for idx in fronts[0]:  # First front is Pareto optimal
                pareto_front.append((population[idx], final_objectives[idx]))
        else:
            if self.verbose:
                print("MOC: No fronts found, returning best by validity")
            # Fallback: return best individuals by validity
            pop_with_obj = list(zip(population, final_objectives))
            pop_with_obj.sort(key=lambda x: x[1][0], reverse=True)  # Sort by validity
            pareto_front = pop_with_obj[:min(10, len(pop_with_obj))]  # Return top 10
        
        return pareto_front


def moc_cf_diverse(
    sample: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    n_counterfactuals: int = 5,
    population_size: int = 100,
    generations: int = 50,
    device: str = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, float]]]:
    """
    Generate multiple diverse counterfactuals using MOC.
    
    Returns:
        Tuple of (list_of_counterfactuals, list_of_predictions, list_of_metrics)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Initialize MOC optimizer
    moc = MOCOptimizer(
        original_sample=sample,
        model=model,
        target_class=target_class,
        population_size=population_size,
        device=device
    )
    
    # Run optimization
    pareto_front = moc.optimize(generations)
    
    if not pareto_front or len(pareto_front) == 0:
        return [], [], []
    
    # Select diverse counterfactuals from Pareto front
    counterfactuals = []
    predictions = []
    metrics_list = []
    
    # Sort by different criteria to get diversity
    pareto_front_sorted = sorted(pareto_front, key=lambda x: x[1][0], reverse=True)  # Sort by validity
    
    selected_indices = []
    for i in range(min(n_counterfactuals, len(pareto_front_sorted))):
        if i == 0:
            # First one: highest validity
            selected_indices.append(0)
        else:
            # Select most diverse from already selected
            best_idx = 0
            best_diversity = -1
            
            for j in range(len(pareto_front_sorted)):
                if j in selected_indices:
                    continue
                
                # Calculate diversity as minimum distance to selected counterfactuals
                min_dist = float('inf')
                for selected_idx in selected_indices:
                    dist = np.linalg.norm(pareto_front_sorted[j][0] - pareto_front_sorted[selected_idx][0])
                    min_dist = min(min_dist, dist)
                
                if min_dist > best_diversity:
                    best_diversity = min_dist
                    best_idx = j
            
            selected_indices.append(best_idx)
    
    # Extract selected counterfactuals
    for idx in selected_indices:
        cf_sample, objectives = pareto_front_sorted[idx]
        
        # Get prediction
        cf_tensor = torch.tensor(cf_sample, dtype=torch.float32, device=device)
        if len(cf_tensor.shape) == 1:
            cf_tensor = cf_tensor.reshape(1, 1, -1)
        elif len(cf_tensor.shape) == 2:
            if cf_tensor.shape[0] > cf_tensor.shape[1]:
                cf_tensor = cf_tensor.T
            cf_tensor = cf_tensor.unsqueeze(0)
        
        with torch.no_grad():
            pred = model(cf_tensor)
            pred_np = torch.softmax(pred, dim=-1).squeeze().cpu().numpy()
        
        # Create metrics
        validity, proximity, sparsity, plausibility = objectives
        metrics = {
            'validity': validity,
            'proximity': proximity,
            'sparsity': sparsity,
            'plausibility': plausibility,
            'predicted_class': int(np.argmax(pred_np))
        }
        
        counterfactuals.append(cf_sample)
        predictions.append(pred_np)
        metrics_list.append(metrics)
    
    return counterfactuals, predictions, metrics_list