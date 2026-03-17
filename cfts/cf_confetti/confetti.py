import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import warnings


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


####
# CONFETTI: COuNterfactual Explanations For Time Series
#
#
# Paper URL: https://arxiv.org/html/2511.13237v2
# Code: https://github.com/serval-uni-lu/confetti
#
# CONFETTI generates counterfactual explanations for multivariate time series
# by combining:
# 1. Nearest Unlike Neighbour (NUN) search
# 2. Naive subsequence replacement stage
# 3. Multi-objective optimization (NSGA-III)
#
# Supports confidence-based constraints, sparsity control, and proximity minimization.
####


def confetti_genetic_cf(
    sample,
    model,
    reference_data,
    reference_labels=None,
    target=None,
    theta=0.51,
    max_iterations=100,
    population_size=50,
    mutation_rate=0.1,
    subsequence_length=None,
    verbose=False
):
    """
    Simplified genetic-based CONFETTI counterfactual generation.
    
    This is a lightweight implementation that uses a genetic algorithm approach
    to find counterfactuals by selectively replacing subsequences from a nearest
    unlike neighbour (NUN).
    
    Parameters
    ----------
    sample : array-like
        The input time series to explain. Shape: (length,) or (channels, length)
    model : torch.nn.Module
        The trained PyTorch model
    reference_data : array-like
        Reference dataset for finding nearest unlike neighbours.
        Shape: (n_samples, channels, length) or (n_samples, length)
    reference_labels : array-like, optional
        Labels for reference data. If None, will be predicted by model.
    target : int, optional
        Target class for counterfactual. If None, uses second most likely class.
    theta : float, default=0.51
        Minimum confidence threshold for valid counterfactual
    max_iterations : int, default=100
        Maximum number of genetic algorithm iterations
    population_size : int, default=50
        Size of the genetic algorithm population
    mutation_rate : float, default=0.1
        Probability of mutation for each gene
    subsequence_length : int, optional
        Length of subsequence to replace. If None, automatically determined.
    verbose : bool, default=False
        Print progress information
        
    Returns
    -------
    counterfactual : array-like or None
        The generated counterfactual, or None if unsuccessful
    prediction : array-like or None
        Model prediction for the counterfactual, or None if unsuccessful
    """
    device = next(model.parameters()).device
    
    def model_predict(data):
        """Helper to predict with proper shape handling."""
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        else:
            data_tensor = data
            
        # Handle different input shapes
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.reshape(1, 1, -1)
        elif len(data_tensor.shape) == 2:
            if data_tensor.shape[0] > data_tensor.shape[1]:
                data_tensor = data_tensor.T
            data_tensor = data_tensor.unsqueeze(0)
            
        return detach_to_numpy(model(data_tensor))
    
    # Prepare sample
    sample_flat = sample.reshape(-1)
    
    # Get initial prediction
    y_orig = model_predict(sample.reshape(sample.shape))[0]
    label_orig = np.argmax(y_orig)
    
    # Determine target class
    if target is None:
        sorted_indices = np.argsort(y_orig)[::-1]
        target = int(sorted_indices[1])
    
    # Prepare reference data
    if isinstance(reference_data, np.ndarray):
        if len(reference_data.shape) == 2:
            # Assume (n_samples, length) for univariate
            ref_data = reference_data.reshape(reference_data.shape[0], 1, -1)
        else:
            ref_data = reference_data
    else:
        ref_data = np.array(reference_data)
    
    # Get or predict reference labels
    if reference_labels is None:
        reference_labels = []
        for ref_sample in ref_data:
            pred = model_predict(ref_sample)
            reference_labels.append(np.argmax(pred))
        reference_labels = np.array(reference_labels)
    
    # Find Nearest Unlike Neighbour (NUN)
    unlike_indices = np.where(reference_labels != label_orig)[0]
    
    if len(unlike_indices) == 0:
        if verbose:
            print("CONFETTI: No unlike neighbours found in reference data")
        return None, None
    
    # Compute distances to find nearest unlike neighbour
    distances = []
    sample_for_dist = sample.reshape(-1)
    for idx in unlike_indices:
        ref_flat = ref_data[idx].reshape(-1)
        dist = np.linalg.norm(sample_for_dist - ref_flat)
        distances.append(dist)
    
    nun_idx = unlike_indices[np.argmin(distances)]
    nun = ref_data[nun_idx]
    nun_label = reference_labels[nun_idx]
    
    if verbose:
        print(f"CONFETTI: Original class {label_orig}, Target class {target}")
        print(f"CONFETTI: Found NUN with label {nun_label} at index {nun_idx}")
    
    # Determine subsequence length
    total_length = len(sample_flat)
    if subsequence_length is None:
        subsequence_length = max(1, total_length // 10)  # Start with 10% of length
    
    # Genetic Algorithm for subsequence selection
    # Binary mask: 1 = use NUN value, 0 = keep original value
    def create_individual():
        """Create random binary mask."""
        return np.random.randint(0, 2, size=total_length)
    
    def evaluate_individual(individual):
        """
        Evaluate fitness of an individual.
        Returns: (confidence in target class, sparsity)
        Higher confidence is better, lower sparsity is better
        """
        # Create counterfactual by applying mask
        cf = sample_flat.copy()
        nun_flat = nun.reshape(-1)
        for i in range(total_length):
            if individual[i] == 1:
                cf[i] = nun_flat[i]
        
        # Get prediction
        pred = model_predict(cf.reshape(sample.shape))[0]
        confidence = pred[target]
        sparsity = np.sum(individual) / total_length  # Fraction of changed values
        
        return confidence, sparsity
    
    def crossover(parent1, parent2):
        """Two-point crossover."""
        child = parent1.copy()
        point1 = np.random.randint(0, total_length)
        point2 = np.random.randint(0, total_length)
        if point1 > point2:
            point1, point2 = point2, point1
        child[point1:point2] = parent2[point1:point2]
        return child
    
    def mutate(individual):
        """Bit-flip mutation."""
        mutated = individual.copy()
        for i in range(total_length):
            if np.random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    best_individual = None
    best_confidence = 0.0
    best_pred = None
    
    # Evolution loop
    for iteration in range(max_iterations):
        # Evaluate population
        fitness_scores = []
        for individual in population:
            confidence, sparsity = evaluate_individual(individual)
            # Multi-objective: maximize confidence, minimize sparsity
            # Use weighted sum for simplicity
            fitness = confidence - 0.3 * sparsity  # Confidence is more important
            fitness_scores.append((fitness, confidence, sparsity))
        
        # Find best individual
        best_idx = np.argmax([f[0] for f in fitness_scores])
        current_best_confidence = fitness_scores[best_idx][1]
        
        if current_best_confidence > best_confidence:
            best_confidence = current_best_confidence
            best_individual = population[best_idx].copy()
            
            # Create counterfactual
            cf = sample_flat.copy()
            nun_flat = nun.reshape(-1)
            for i in range(total_length):
                if best_individual[i] == 1:
                    cf[i] = nun_flat[i]
            best_pred = model_predict(cf.reshape(sample.shape))[0]
        
        if verbose and iteration % 20 == 0:
            print(f"CONFETTI iter {iteration}: best_confidence={best_confidence:.4f}, "
                  f"sparsity={fitness_scores[best_idx][2]:.4f}")
        
        # Check if we found a valid counterfactual
        if best_confidence >= theta and np.argmax(best_pred) == target:
            if verbose:
                print(f"CONFETTI: Found valid counterfactual at iteration {iteration}")
            break
        
        # Selection: tournament selection
        new_population = []
        for _ in range(population_size):
            # Tournament
            tournament_idx = np.random.choice(population_size, size=3, replace=False)
            tournament_fitness = [fitness_scores[i][0] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            new_population.append(population[winner_idx].copy())
        
        # Crossover and mutation
        offspring = []
        for i in range(0, population_size, 2):
            parent1 = new_population[i]
            parent2 = new_population[min(i + 1, population_size - 1)]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            offspring.append(mutate(child1))
            offspring.append(mutate(child2))
        
        population = offspring[:population_size]
    
    if best_individual is None:
        if verbose:
            print("CONFETTI: Failed to find counterfactual")
        return None, None
    
    # Generate final counterfactual
    cf = sample_flat.copy()
    nun_flat = nun.reshape(-1)
    for i in range(total_length):
        if best_individual[i] == 1:
            cf[i] = nun_flat[i]
    
    cf_shaped = cf.reshape(sample.shape)
    
    if verbose:
        print(f"CONFETTI: Final confidence in target class: {best_confidence:.4f}")
        print(f"CONFETTI: Predicted class: {np.argmax(best_pred)}")
    
    return cf_shaped, best_pred


def confetti_package_cf(
    sample,
    model,
    reference_data,
    reference_labels=None,
    reference_weights=None,
    target=None,
    theta=0.51,
    alpha=0.5,
    n_partitions=3,
    population_size=100,
    maximum_number_of_generations=100,
    crossover_probability=1.0,
    mutation_probability=0.9,
    optimize_confidence=True,
    optimize_sparsity=True,
    optimize_proximity=True,
    proximity_distance="euclidean",
    dtw_window=None,
    verbose=False
):
    """
    Generate counterfactual using the official CONFETTI package.
    
    This function wraps the official CONFETTI implementation from the
    'confetti' Python package for comparison purposes.
    
    Parameters
    ----------
    sample : array-like
        The input time series to explain
    model : torch.nn.Module or str
        The trained model or path to model file
    reference_data : array-like
        Reference dataset for finding nearest unlike neighbours
    reference_labels : array-like, optional
        Labels for reference data
    reference_weights : array-like, optional
        Feature importance weights
    target : int, optional
        Target class for counterfactual
    theta : float, default=0.51
        Minimum confidence threshold
    alpha : float, default=0.5
        Trade-off between confidence and sparsity
    n_partitions : int, default=3
        Number of partitions for NSGA-III reference directions
    population_size : int, default=100
        Size of evolutionary population
    maximum_number_of_generations : int, default=100
        Maximum number of generations
    crossover_probability : float, default=1.0
        Crossover probability
    mutation_probability : float, default=0.9
        Mutation probability
    optimize_confidence : bool, default=True
        Whether to optimize confidence
    optimize_sparsity : bool, default=True
        Whether to optimize sparsity
    optimize_proximity : bool, default=True
        Whether to optimize proximity
    proximity_distance : str, default="euclidean"
        Distance metric for proximity
    dtw_window : int, optional
        DTW window size if using DTW distance
    verbose : bool, default=False
        Print progress information
        
    Returns
    -------
    counterfactual : array-like or None
        The generated counterfactual, or None if unsuccessful
    prediction : array-like or None
        Model prediction for the counterfactual, or None if unsuccessful
    info : dict
        Additional information about the generation process
        
    Raises
    ------
    ImportError
        If the official 'confetti' package is not installed
    """
    try:
        from confetti.explainer import CONFETTI
    except ImportError:
        raise ImportError(
            "The official 'confetti' package is not installed. "
            "Install it with: pip install confetti-ts\n"
            "Or use the simplified confetti_genetic_cf implementation instead."
        )
    
    # Convert PyTorch model to format expected by CONFETTI package
    # CONFETTI package expects Keras models, so we need to save/convert if needed
    if isinstance(model, nn.Module):
        warnings.warn(
            "CONFETTI package expects Keras models. "
            "For PyTorch models, use confetti_genetic_cf instead.",
            UserWarning
        )
        return None, None, {"error": "PyTorch model not supported by package"}
    
    # Initialize CONFETTI explainer
    explainer = CONFETTI(model_path=model)
    
    # Prepare data shapes
    if isinstance(sample, np.ndarray):
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1, 1)  # (timesteps, channels)
        elif len(sample.shape) == 2:
            sample = sample.reshape(1, sample.shape[0], sample.shape[1])
    
    if isinstance(reference_data, np.ndarray):
        if len(reference_data.shape) == 2:
            reference_data = reference_data.reshape(
                reference_data.shape[0], reference_data.shape[1], 1
            )
    
    try:
        # Generate counterfactuals
        results = explainer.generate(
            instances_to_explain=sample,
            reference_data=reference_data,
            reference_weights=reference_weights,
            alpha=alpha,
            theta=theta,
            n_partitions=n_partitions,
            population_size=population_size,
            maximum_number_of_generations=maximum_number_of_generations,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            optimize_confidence=optimize_confidence,
            optimize_sparsity=optimize_sparsity,
            optimize_proximity=optimize_proximity,
            proximity_distance=proximity_distance,
            dtw_window=dtw_window,
            verbose=verbose,
        )
        
        if results is None or len(results.counterfactual_sets) == 0:
            return None, None, {"error": "No counterfactual found"}
        
        # Extract best counterfactual
        cf_set = results.counterfactual_sets[0]
        best_cf = cf_set.best_solution
        
        if best_cf is None:
            return None, None, {"error": "No valid counterfactual found"}
        
        # Get prediction for counterfactual
        cf_array = best_cf.counterfactual
        pred_label = best_cf.label
        
        info = {
            "nun": cf_set.nearest_unlike_neighbour,
            "original_label": cf_set.original_label,
            "cf_label": pred_label,
            "num_cfs_found": len(cf_set.all_counterfactuals),
        }
        
        # Return in simplified format
        return cf_array, pred_label, info
        
    except Exception as e:
        if verbose:
            print(f"CONFETTI package error: {str(e)}")
        return None, None, {"error": str(e)}


def compare_confetti_implementations(
    sample,
    model,
    reference_data,
    reference_labels=None,
    target=None,
    theta=0.51,
    max_iterations=100,
    verbose=True
):
    """
    Compare the simplified genetic implementation with the official package.
    
    This function runs both the simplified confetti_genetic_cf and the
    official CONFETTI package (if available) and compares their results.
    
    Parameters
    ----------
    sample : array-like
        The input time series to explain
    model : torch.nn.Module
        The trained PyTorch model
    reference_data : array-like
        Reference dataset
    reference_labels : array-like, optional
        Labels for reference data
    target : int, optional
        Target class
    theta : float, default=0.51
        Minimum confidence threshold
    max_iterations : int, default=100
        Maximum iterations for genetic algorithm
    verbose : bool, default=True
        Print comparison information
        
    Returns
    -------
    results : dict
        Dictionary containing results from both implementations
    """
    results = {
        "simplified": None,
        "package": None,
        "comparison": {}
    }
    
    # Run simplified implementation
    if verbose:
        print("=" * 60)
        print("Running simplified CONFETTI implementation...")
        print("=" * 60)
    
    cf_simple, pred_simple = confetti_genetic_cf(
        sample=sample,
        model=model,
        reference_data=reference_data,
        reference_labels=reference_labels,
        target=target,
        theta=theta,
        max_iterations=max_iterations,
        verbose=verbose
    )
    
    results["simplified"] = {
        "counterfactual": cf_simple,
        "prediction": pred_simple,
        "success": cf_simple is not None
    }
    
    # Try to run package implementation
    if verbose:
        print("\n" + "=" * 60)
        print("Attempting to run official CONFETTI package...")
        print("=" * 60)
    
    try:
        cf_pkg, pred_pkg, info_pkg = confetti_package_cf(
            sample=sample,
            model=model,
            reference_data=reference_data,
            reference_labels=reference_labels,
            target=target,
            theta=theta,
            verbose=verbose
        )
        
        results["package"] = {
            "counterfactual": cf_pkg,
            "prediction": pred_pkg,
            "info": info_pkg,
            "success": cf_pkg is not None
        }
    except ImportError as e:
        if verbose:
            print(f"\nOfficial package not available: {str(e)}")
        results["package"] = {
            "error": str(e),
            "success": False
        }
    
    # Compare results
    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
    
    if results["simplified"]["success"]:
        if verbose:
            print("✓ Simplified implementation: SUCCESS")
            if pred_simple is not None:
                print(f"  - Predicted class: {np.argmax(pred_simple)}")
                if target is not None:
                    print(f"  - Target class: {target}")
                print(f"  - Confidence in target: {pred_simple[target] if target is not None else 'N/A':.4f}")
    else:
        if verbose:
            print("✗ Simplified implementation: FAILED")
    
    if results["package"]["success"]:
        if verbose:
            print("✓ Official package: SUCCESS")
            if "cf_label" in results["package"]["info"]:
                print(f"  - Predicted class: {results['package']['info']['cf_label']}")
    else:
        if verbose:
            print(f"✗ Official package: {results['package'].get('error', 'FAILED')}")
    
    # Calculate metrics if both succeeded
    if results["simplified"]["success"] and results["package"]["success"]:
        cf_simple = results["simplified"]["counterfactual"]
        cf_pkg = results["package"]["counterfactual"]
        
        # Calculate distance between counterfactuals
        if cf_simple is not None and cf_pkg is not None:
            try:
                cf_simple_flat = cf_simple.reshape(-1)
                cf_pkg_flat = cf_pkg.reshape(-1)
                
                if len(cf_simple_flat) == len(cf_pkg_flat):
                    l2_diff = np.linalg.norm(cf_simple_flat - cf_pkg_flat)
                    results["comparison"]["l2_distance"] = l2_diff
                    
                    if verbose:
                        print(f"\nL2 distance between counterfactuals: {l2_diff:.4f}")
            except Exception as e:
                if verbose:
                    print(f"\nCould not compare counterfactuals: {str(e)}")
    
    return results
