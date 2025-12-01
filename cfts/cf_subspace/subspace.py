"""
Sub-SpaCE: Subsequence-based Sparse Counterfactual Explanations for Time Series Classification

Based on the implementation from:
https://github.com/MarioRefoyo/Sub-SpaCE

This module implements the Sub-SpaCE counterfactual explanation method for time series classification.
It uses an evolutionary algorithm with subsequence-based representations to generate sparse counterfactuals.
"""

import torch
import numpy as np
import copy
from abc import ABC, abstractmethod
import random


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(data).float().to(device)


class OutlierCalculator:
    """
    Simple outlier calculator based on reconstruction error.
    For a full implementation, use an autoencoder-based approach.
    """
    def __init__(self, calibration_data):
        self.calibration_data = calibration_data
        # Simple std-based outlier detection
        self.mean = np.mean(calibration_data, axis=0, keepdims=True)
        self.std = np.std(calibration_data, axis=0, keepdims=True) + 1e-8
        
    def get_outlier_scores(self, data):
        """Calculate outlier scores based on deviation from mean.
        
        Args:
            data: Input data - either (L, C) for single sample or (N, L, C) for batch
            
        Returns:
            scores: Outlier scores of shape (N,) or scalar for single sample
        """
        original_was_2d = data.ndim == 2
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)  # (L, C) -> (1, L, C)
        
        # Ensure mean and std have the right shape for broadcasting
        # data is (N, L, C), mean and std should be (1, L, C)
        if self.mean.shape[1:] != data.shape[1:]:
            # Shape mismatch - need to handle
            # This might happen if calibration was done with different shape
            print(f"Warning: shape mismatch - data: {data.shape}, mean: {self.mean.shape}")
        
        # Calculate normalized deviation
        normalized_diff = np.abs(data - self.mean) / self.std  # (N, L, C)
        scores = np.mean(normalized_diff, axis=(1, 2))  # Should give (N,)
        
        # Ensure scores is 1D
        scores = np.atleast_1d(scores).flatten()
        
        # Normalize to [0, 1]
        max_score = np.max(scores) if np.max(scores) > 0 else 1.0
        normalized_scores = scores / max_score
        
        # Return scalar if input was single sample (2D input)
        if original_was_2d:
            return float(normalized_scores.flat[0])
        return normalized_scores


def fitness_function_final(ms, predicted_probs, desired_class, outlier_scores,
                           invalid_penalization, alpha, beta, eta, gamma, sparsity_balancer):
    """
    Fitness function for evaluating counterfactual candidates.
    
    Args:
        ms: Binary masks indicating which subsequences to replace
        predicted_probs: Predicted class probabilities
        desired_class: Target class for counterfactuals
        outlier_scores: Outlier scores for each candidate
        invalid_penalization: Penalty for invalid counterfactuals
        alpha: Weight for classification probability
        beta: Weight for sparsity
        eta: Weight for outlier score
        gamma: Exponent for subsequence percentage
        sparsity_balancer: Balance between point sparsity and subsequence sparsity
    """
    # Sparsity calculator - ms is (pop_size, L, C)
    total_elements = ms.shape[1] * ms.shape[2]  # L * C
    ones_pct = ms.sum(axis=(1, 2)) / total_elements  # (pop_size,)
    
    # Count subsequences - transitions from 0 to 1 along time axis
    subsequences = np.count_nonzero(np.diff(ms, prepend=0, axis=1) == 1, axis=(1, 2))  # (pop_size,)
    max_subsequences = (ms.shape[1] // 2) * ms.shape[2]  # (L // 2) * C
    subsequences_pct = subsequences / max_subsequences  # (pop_size,)
    
    sparsity_term = sparsity_balancer * ones_pct + (1 - sparsity_balancer) * subsequences_pct ** gamma  # (pop_size,)

    # Penalization for not prob satisfied
    desired_class_probs = predicted_probs[:, desired_class]
    predicted_classes = np.argmax(predicted_probs, axis=1)
    penalization = (predicted_classes != desired_class).astype(int)

    # Clip outlier scores
    if outlier_scores is not None:
        outlier_scores = outlier_scores.copy()
        outlier_scores[outlier_scores < 0] = 0
    else:
        outlier_scores = np.zeros(len(predicted_probs))

    # Calculate fitness
    fit = alpha * desired_class_probs - beta * sparsity_term - eta * outlier_scores - penalization * invalid_penalization

    return fit, desired_class_probs


class EvolutionaryOptimizer(ABC):
    """Base class for evolutionary optimizers."""
    
    def __init__(self, fitness_func, prediction_func, population_size, elite_number, offsprings_number, max_iter,
                 init_pct, reinit,
                 invalid_penalization, alpha, beta, eta, gamma, sparsity_balancer,
                 feature_axis, verbose=False):
        # Assert elite numbers and replacement count do not surpass population size
        if elite_number + offsprings_number > population_size:
            raise ValueError('Elites and offsprings counts must not be greater than population size')
        # Assert valid offspring number
        if offsprings_number % 2 != 0:
            raise ValueError('Offspring number must be even')

        self.population_size = population_size
        self.elite_number = elite_number
        self.offsprings_number = offsprings_number
        self.rest_number = population_size - elite_number - offsprings_number

        self.fitness_func = fitness_func
        self.invalid_penalization = invalid_penalization
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.sparsity_balancer = sparsity_balancer

        self.prediction_func = prediction_func
        self.max_iter = max_iter
        self.original_init_pct = init_pct

        self.feature_axis = feature_axis
        self.reinit = reinit
        self.verbose = verbose

    @abstractmethod
    def init_population(self, importance_heatmap=None):
        pass

    def init(self, x_orig, nun_example, desired_class, outlier_calculator=None, importance_heatmap=None):
        self.x_orig = x_orig
        self.nun_example = nun_example
        self.desired_class = desired_class
        self.outlier_calculator = outlier_calculator
        self.importance_heatmap = importance_heatmap
        self.init_pct = copy.deepcopy(self.original_init_pct)
        self.init_population(self.importance_heatmap)

        # Get dimensionality attributes
        if self.feature_axis == 2:
            self.n_features = x_orig.shape[1]
            self.ts_length = x_orig.shape[0]
        else:
            self.n_features = x_orig.shape[0]
            self.ts_length = x_orig.shape[1]

        # Compute initial outlier scores
        if self.outlier_calculator is not None:
            self.outlier_scores_orig = self.outlier_calculator.get_outlier_scores(self.x_orig)
            self.outlier_score_nun = self.outlier_calculator.get_outlier_scores(self.nun_example)
        else:
            self.outlier_scores_orig = 0
            self.outlier_score_nun = 0

    @abstractmethod
    def mutate(self, sub_population):
        pass

    def get_single_crossover_mask(self, subpopulation):
        split_points = np.random.randint(0, subpopulation.shape[1], size=subpopulation.shape[0] // 2)
        mask = np.arange(subpopulation.shape[1]) < split_points[:, np.newaxis]
        return mask

    def produce_offsprings(self, subpopulation, number):
        # Put features as individual examples
        if self.feature_axis == 2:
            adapted_subpopulation = np.swapaxes(subpopulation, 2, 1)
        else:
            adapted_subpopulation = subpopulation
        adapted_number = number * self.n_features
        adapted_subpopulation = adapted_subpopulation.reshape(adapted_number, -1)

        # Generate random split points and create mask
        mask = self.get_single_crossover_mask(adapted_subpopulation)

        # Generate random matches
        matches = np.random.choice(np.arange(adapted_subpopulation.shape[0]), 
                                  size=(adapted_subpopulation.shape[0] // 2, 2), replace=False)

        # Create the two partial offsprings
        offsprings1 = np.empty((adapted_number//2, adapted_subpopulation.shape[1]))
        offsprings1[mask] = adapted_subpopulation[matches[:, 0]][mask]
        offsprings1[~mask] = adapted_subpopulation[matches[:, 1]][~mask]
        offsprings2 = np.zeros((adapted_number//2, adapted_subpopulation.shape[1]))
        offsprings2[mask] = adapted_subpopulation[matches[:, 1]][mask]
        offsprings2[~mask] = adapted_subpopulation[matches[:, 0]][~mask]
        
        # Calculate adapted offspring
        adapted_offsprings = np.concatenate([offsprings1, offsprings2])

        # Mutate offsprings
        adapted_offsprings = self.mutate(adapted_offsprings)

        # Get final offsprings (matching original dimensionality)
        adapted_offsprings = adapted_offsprings.reshape(number, self.n_features, -1)
        if self.feature_axis == 2:
            offsprings = np.swapaxes(adapted_offsprings, 2, 1)
        else:
            offsprings = adapted_offsprings

        return offsprings

    @staticmethod
    def get_counterfactuals(x_orig, nun_example, population):
        """
        Generate counterfactuals by mixing x_orig and nun_example based on population masks.
        
        Args:
            x_orig: Original sample (L, C)
            nun_example: Native guide sample (L, C)
            population: Binary masks (pop_size, L, C)
        
        Returns:
            Counterfactuals in (pop_size, C, L) format for model prediction
        """
        population_mask = population.astype(bool)
        population_size = population.shape[0]
        # Replicate x_orig and nun_example in array
        x_orig_ext = np.tile(x_orig, (population_size, 1, 1))
        nun_ext = np.tile(nun_example, (population_size, 1, 1))
        # Generate counterfactuals in (pop_size, L, C)
        counterfactuals = np.zeros(population_mask.shape)
        counterfactuals[~population_mask] = x_orig_ext[~population_mask]
        counterfactuals[population_mask] = nun_ext[population_mask]
        # Convert to (pop_size, C, L) for model
        counterfactuals = np.transpose(counterfactuals, (0, 2, 1))
        return counterfactuals

    def compute_fitness(self):
        # Get counterfactuals
        population_cfs = self.get_counterfactuals(self.x_orig, self.nun_example, self.population)

        # Get desired class probs
        predicted_probs = self.prediction_func(population_cfs)

        # Get outlier scores - convert from (pop_size, C, L) to (pop_size, L, C)
        if self.outlier_calculator is not None:
            population_cfs_outlier = np.transpose(population_cfs, (0, 2, 1))  # (pop_size, C, L) -> (pop_size, L, C)
            outlier_scores = self.outlier_calculator.get_outlier_scores(population_cfs_outlier)
            increase_outlier_score = outlier_scores - (self.outlier_scores_orig + self.outlier_score_nun) / 2
        else:
            outlier_scores = None
            increase_outlier_score = None

        # Get fitness function
        fitness, desired_class_probs = self.fitness_func(
            self.population, predicted_probs, self.desired_class, increase_outlier_score,
            self.invalid_penalization, self.alpha, self.beta, self.eta,
            self.gamma, self.sparsity_balancer)
        return fitness, desired_class_probs

    @staticmethod
    def roulette(fitness, number):
        scaled_fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-5)
        selection_probs = scaled_fitness / scaled_fitness.sum()
        if np.isnan(selection_probs).any():
            print('NaN found in candidate probabilities')
            print(f'Fitness {fitness}')
            print(f'Selection probs: {selection_probs}')
        selected_indexes = np.random.choice(scaled_fitness.shape[0], number, p=selection_probs)
        return selected_indexes

    def select_candidates(self, population, fitness, number):
        selected_indexes = self.roulette(fitness, number)
        return population[selected_indexes]

    def optimize(self):
        # Keep track of the best solution
        best_score = -100
        best_sample = None
        best_classification_prob = 0
        fitness_evolution = []
        cf_evolution = []

        # Compute initial fitness
        fitness, _ = self.compute_fitness()
        i = np.argsort(fitness)[-1]
        fitness_evolution.append(fitness[i])
        best_cf = self.get_counterfactuals(self.x_orig, self.nun_example, np.expand_dims(self.population[i], axis=0))
        cf_evolution.append(np.squeeze(best_cf, axis=0))

        # Run evolution
        iteration = 0
        while iteration < self.max_iter:
            # Init new population
            new_population = np.empty(self.population.shape)

            # Elites: Select elites and add to new population
            elites_idx = np.argsort(fitness)[-self.elite_number:]
            new_population[:self.elite_number, :] = self.population[elites_idx]

            # Cross-over and mutation
            # Select parents
            candidate_population = self.select_candidates(self.population, fitness, self.offsprings_number)
            # Produce offsprings
            offsprings = self.produce_offsprings(candidate_population, self.offsprings_number)
            # Add to the population
            new_population[self.elite_number:self.offsprings_number + self.elite_number] = offsprings

            # The rest of the population is random selected
            random_indexes = np.random.randint(self.population_size, size=self.rest_number)
            if self.rest_number > 0:
                new_population[-self.rest_number:] = self.population[random_indexes]

            # Change population
            self.population = new_population.astype(int)
            
            # Keep track of the best solution
            fitness, class_probs = self.compute_fitness()
            i = np.argsort(fitness)[-1]
            fitness_evolution.append(fitness[i])
            best_cf = self.get_counterfactuals(self.x_orig, self.nun_example, np.expand_dims(self.population[i], axis=0))
            cf_evolution.append(np.squeeze(best_cf, axis=0))
            
            if fitness[i] > best_score:
                best_score = fitness[i]
                best_sample = self.population[i]
                best_classification_prob = class_probs[i]

            # Handle while loop updates
            if self.reinit and (iteration == 50) and (self.init_pct < 0.9) and (fitness[i] < -self.invalid_penalization+1):
                if self.verbose:
                    print('Failed to find a valid counterfactual in 50 iterations. '
                          'Restarting process with more activations in init')
                iteration = 0
                self.init_pct = min(0.9, self.init_pct + 0.2)  # Cap at 0.9
                self.init_population(self.importance_heatmap)
                fitness, class_probs = self.compute_fitness()
            else:
                iteration += 1

            # Reinit if all fitness are equal
            if np.all(fitness == fitness[0]):
                if self.verbose:
                    print(f'Found convergence of solutions in {iteration} iteration. Final prob {best_classification_prob}')
                if best_classification_prob > 0.5:
                    break
                elif self.reinit and self.init_pct < 0.9:
                    iteration = 0
                    self.init_pct = min(0.9, self.init_pct + 0.2)
                    self.init_population(self.importance_heatmap)
                    fitness, class_probs = self.compute_fitness()
                else:
                    # Can't reinit anymore, exit with best found
                    if self.verbose:
                        print(f'Cannot improve further. Returning best solution with prob {best_classification_prob}')
                    break
            
            # Early stopping if we found a good counterfactual
            if best_classification_prob > 0.5:
                if self.verbose:
                    print(f'Found valid counterfactual at iteration {iteration} with prob {best_classification_prob}')
                break

        return best_sample, best_classification_prob, fitness_evolution, cf_evolution


class NSubsequenceEvolutionaryOptimizer(EvolutionaryOptimizer):
    """Evolutionary optimizer for subsequence-based counterfactuals."""
    
    def __init__(self, fitness_func, prediction_func,
                 population_size=100, elite_number=4, offsprings_number=96, max_iter=100,
                 change_subseq_mutation_prob=0.05, add_subseq_mutation_prob=0,
                 init_pct=0.4, reinit=True,
                 invalid_penalization=100, alpha=0.2, beta=0.6, eta=0.2, gamma=0.25, sparsity_balancer=0.4,
                 feature_axis=2, verbose=False):
        super().__init__(fitness_func, prediction_func, population_size, elite_number, offsprings_number, max_iter,
                         init_pct, reinit,
                         invalid_penalization, alpha, beta, eta, gamma, sparsity_balancer, feature_axis, verbose)
        self.change_subseq_mutation_prob = change_subseq_mutation_prob
        self.add_subseq_mutation_prob = add_subseq_mutation_prob

    def init_population(self, importance_heatmap=None):
        # Init population
        random_data = np.random.uniform(0, 1, (self.population_size,) + self.x_orig.shape)
        if importance_heatmap is not None:
            mix_ratio = 0.6
            inducted_data = (mix_ratio*random_data + (1-mix_ratio)*importance_heatmap) / 2
        else:
            inducted_data = random_data
        # Calculate quantile and population - ensure init_pct is valid
        init_pct_safe = np.clip(self.init_pct, 0.0, 1.0)
        quantile_80 = np.quantile(inducted_data.flatten(), 1-init_pct_safe)
        self.population = (inducted_data > quantile_80).astype(int)

    @staticmethod
    def shrink_mutation(population, mutation_prob):
        # Get mask of the subsequence beginnings and endings
        mask_beginnings = np.diff(population, 1, prepend=0)
        mask_beginnings = np.in1d(mask_beginnings, 1).reshape(mask_beginnings.shape)
        mask_endings = np.flip(np.diff(np.flip(population, axis=1), 1, prepend=0), axis=1)
        mask_endings = np.in1d(mask_endings, 1).reshape(mask_endings.shape)
        # Generate complete mask
        beginnings_endings_mask = mask_beginnings + mask_endings

        # Generate mutation
        random_mutations = (np.random.uniform(0, 1, population.shape) < mutation_prob).astype(int)
        # Get mutated population
        valid_mutations = np.zeros(population.shape).astype(int)
        valid_mutations[beginnings_endings_mask] = random_mutations[beginnings_endings_mask]
        mutated_population = (population + valid_mutations) % 2
        return mutated_population

    @staticmethod
    def extend_mutation(population, mutation_prob):
        # Get potential extension locations
        ones_mask = np.in1d(population, 1).reshape(population.shape)
        # Get before and after ones masks
        before_ones_mask = np.roll(ones_mask, -1, axis=1)
        before_ones_mask[:, ones_mask.shape[1] - 1] = False
        after_ones_mask = np.roll(ones_mask, 1, axis=1)
        after_ones_mask[:, 0] = False
        # Generate complete mask
        before_after_ones_mask = before_ones_mask + after_ones_mask
        before_after_ones_mask[ones_mask] = False

        # Generate mutation
        random_mutations = (np.random.uniform(0, 1, population.shape) < mutation_prob).astype(int)
        # Get mutated population
        valid_mutations = np.zeros(population.shape).astype(int)
        valid_mutations[before_after_ones_mask] = random_mutations[before_after_ones_mask]
        mutated_population = np.clip(population + valid_mutations, 0, 1)
        return mutated_population

    @staticmethod
    def add_subsequence_mutation(population, mutation_prob):
        # Get potential extension locations
        ones_mask = np.in1d(population, 1).reshape(population.shape)
        # Get before and after ones masks
        before_ones_mask = np.roll(ones_mask, -1, axis=1)
        before_ones_mask[:, ones_mask.shape[1] - 1] = False
        after_ones_mask = np.roll(ones_mask, 1, axis=1)
        after_ones_mask[:, 0] = False
        # Generate complete mask
        before_after_ones_mask = before_ones_mask + after_ones_mask
        before_after_ones_mask[ones_mask] = False

        # Get potential positions mask
        possibilities_mask = ~(before_after_ones_mask + ones_mask)

        # Get new subsequences
        new_subsequences = np.zeros(population.shape).astype(int)
        for i, row in enumerate(possibilities_mask):
            # Flip a coin to mutate or not
            if np.random.random() < mutation_prob:
                valid_idx = np.where(row == True)[0]
                # Get random index and length to add subsequence
                if len(valid_idx) > 0:
                    chosen_idx = np.random.choice(valid_idx)
                    subseq_len = min(population.shape[1] - chosen_idx, np.random.randint(2, 6))
                    new_subsequences[i, chosen_idx:chosen_idx + subseq_len] = 1

        # Get mutated population
        mutated_population = np.clip(population + new_subsequences, 0, 1)
        return mutated_population

    def mutate(self, sub_population):
        # Compute mutation values
        mutated_sub_population = self.shrink_mutation(sub_population, self.change_subseq_mutation_prob)
        mutated_sub_population = self.extend_mutation(mutated_sub_population, self.change_subseq_mutation_prob)
        if self.add_subseq_mutation_prob > 0:
            mutated_sub_population = self.add_subsequence_mutation(mutated_sub_population, self.add_subseq_mutation_prob)
        return mutated_sub_population


def calculate_heatmap_torch(model, x_orig, device):
    """
    Calculate importance heatmap using gradients (simplified Grad-CAM approach).
    
    Args:
        model: PyTorch model (expects input as B, C, L)
        x_orig: Input time series in (L, C) format
        device: PyTorch device
    
    Returns:
        Importance heatmap of same shape as input (L, C)
    """
    model.eval()
    L, C = x_orig.shape
    # Convert to (B, C, L) format for model
    x_tensor = numpy_to_torch(x_orig.T.reshape(1, C, L), device)
    x_tensor.requires_grad = True
    
    # Forward pass
    output = model(x_tensor)
    pred_class = torch.argmax(output, dim=1)
    
    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()
    
    # Get gradients and convert back to (L, C)
    gradients = x_tensor.grad.data
    heatmap = torch.abs(gradients).squeeze(0)  # (C, L)
    heatmap = detach_to_numpy(heatmap).T  # Convert to (L, C)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def subspace_cf(sample, dataset, model, nun_example=None, desired_class=None,
                            population_size=100, elite_number=4, offsprings_number=96, 
                            max_iter=100, change_subseq_mutation_prob=0.05, 
                            add_subseq_mutation_prob=0, init_pct=0.2, reinit=True,
                            invalid_penalization=100, alpha=0.2, beta=0.6, 
                            eta=0.2, gamma=0.25, sparsity_balancer=0.4,
                            verbose=False):
    """
    Generate counterfactual explanation using Sub-SpaCE method.
    
    Args:
        sample: Input time series to explain (L, C) or (C, L) or 1D
        dataset: Training dataset for finding native guide
        model: PyTorch classification model
        nun_example: Native Unexplained Neighbor (if None, will be found)
        desired_class: Target class for counterfactual (if None, will be inferred)
        population_size: Size of evolutionary population
        elite_number: Number of elite individuals to preserve
        offsprings_number: Number of offspring to generate
        max_iter: Maximum iterations for evolution
        change_subseq_mutation_prob: Probability of changing subsequence boundaries
        add_subseq_mutation_prob: Probability of adding new subsequences
        init_pct: Initial percentage of activated positions
        reinit: Whether to reinitialize on failure
        invalid_penalization: Penalty for invalid counterfactuals
        alpha: Weight for classification probability
        beta: Weight for sparsity
        eta: Weight for outlier score
        gamma: Exponent for subsequence percentage
        sparsity_balancer: Balance between point and subsequence sparsity
        verbose: Whether to print progress
        
    Returns:
        Tuple of (counterfactual, prediction_scores)
    """
    device = next(model.parameters()).device
    
    def model_predict(arr):
        # arr expected shape (B, C, L) - PyTorch convention
        return detach_to_numpy(model(numpy_to_torch(arr, device)))
    
    # Normalize sample to (L, C) format for internal processing
    sample = np.asarray(sample)
    if sample.ndim == 1:
        sample = sample.reshape(-1, 1)  # (L, 1)
    elif sample.ndim == 2:
        # Assume (L, C) if L > C, else (C, L)
        if sample.shape[0] < sample.shape[1]:
            sample = sample.T
    
    L, C = sample.shape
    
    # Get predictions for sample (convert to B, C, L for model)
    preds_sample = model_predict(sample.T.reshape(1, C, L))
    label_sample = int(np.argmax(preds_sample))
    
    # Find or use provided NUN
    if nun_example is None:
        # Find native guide: nearest neighbor from target class
        if verbose:
            print("Finding native guide (NUN) from dataset...")
        
        # Get target class
        if desired_class is None:
            # Find most common other class in dataset
            dataset_labels = []
            for item in dataset:
                data = item[0] if isinstance(item, tuple) else item
                data_arr = np.asarray(data)
                if data_arr.ndim == 1:
                    data_arr = data_arr.reshape(-1, 1)
                elif data_arr.ndim == 2 and data_arr.shape[0] < data_arr.shape[1]:
                    data_arr = data_arr.T
                pred = model_predict(data_arr.T.reshape(1, C, L))
                dataset_labels.append(int(np.argmax(pred)))
            
            # Find most common class that's different from sample
            from collections import Counter
            class_counts = Counter(dataset_labels)
            if label_sample in class_counts:
                del class_counts[label_sample]
            target_class_temp = class_counts.most_common(1)[0][0] if class_counts else 1 - label_sample
        else:
            target_class_temp = desired_class
        
        # Find nearest neighbor from target class
        best_distance = float('inf')
        best_nun = None
        for item in dataset:
            data = item[0] if isinstance(item, tuple) else item
            data_arr = np.asarray(data)
            if data_arr.ndim == 1:
                data_arr = data_arr.reshape(-1, 1)
            elif data_arr.ndim == 2 and data_arr.shape[0] < data_arr.shape[1]:
                data_arr = data_arr.T
            
            # Check if it's target class
            pred = model_predict(data_arr.T.reshape(1, C, L))
            if int(np.argmax(pred)) == target_class_temp:
                # Calculate distance
                distance = np.linalg.norm(data_arr - sample)
                if distance < best_distance:
                    best_distance = distance
                    best_nun = data_arr
        
        if best_nun is not None:
            nun_example = best_nun
            if verbose:
                print(f"Found NUN with distance {best_distance:.3f}")
        else:
            if verbose:
                print("Warning: Could not find NUN from target class, using random perturbation")
            nun_example = sample + np.random.randn(*sample.shape) * 0.1
    else:
        nun_example = np.asarray(nun_example)
        if nun_example.ndim == 1:
            nun_example = nun_example.reshape(-1, 1)
        elif nun_example.ndim == 2 and nun_example.shape[0] < nun_example.shape[1]:
            nun_example = nun_example.T
    
    # Determine desired class
    if desired_class is None:
        preds_nun = model_predict(nun_example.T.reshape(1, C, L))
        desired_class = int(np.argmax(preds_nun))
    
    # Calculate importance heatmaps (using L, C format internally)
    heatmap_sample = calculate_heatmap_torch(model, sample, device)
    heatmap_nun = calculate_heatmap_torch(model, nun_example, device)
    combined_heatmap = (heatmap_sample + heatmap_nun) / 2
    
    # Create simple outlier calculator
    if dataset is not None and len(dataset) > 0:
        # Extract training data - follow multispace.py pattern
        train_samples = []
        
        for x in dataset[:min(100, len(dataset))]:
            item = x[0] if isinstance(x, tuple) else x
            item_arr = np.asarray(item) if not isinstance(item, np.ndarray) else item
            
            # Flatten any extra dimensions and reshape to (L,) first
            if item_arr.ndim > 2:
                item_arr = item_arr.reshape(-1)
            
            # Normalize to 1D array of length L
            if item_arr.ndim == 2:
                # If 2D, flatten to 1D
                if item_arr.shape[0] == 1:
                    item_arr = item_arr.flatten()  # (1, L) -> (L,)
                elif item_arr.shape[1] == 1:
                    item_arr = item_arr.flatten()  # (L, 1) -> (L,)
                else:
                    # Multi-channel, take first channel or flatten
                    item_arr = item_arr.flatten()
            
            # Now item_arr should be 1D (L,)
            if item_arr.ndim == 1 and len(item_arr) == L:
                train_samples.append(item_arr)
        
        if len(train_samples) >= 10:  # Need at least 10 samples
            # Stack into (N, L)
            train_data = np.stack(train_samples, axis=0)
            # Reshape to (N, L, 1) for univariate
            train_data = train_data.reshape(len(train_samples), L, 1)
            outlier_calc = OutlierCalculator(train_data)
        else:
            outlier_calc = OutlierCalculator(sample.reshape(1, L, C))
    else:
        outlier_calc = OutlierCalculator(sample.reshape(1, L, C))
    
    # Initialize optimizer
    optimizer = NSubsequenceEvolutionaryOptimizer(
        fitness_function_final, model_predict,
        population_size, elite_number, offsprings_number, max_iter,
        change_subseq_mutation_prob, add_subseq_mutation_prob,
        init_pct, reinit,
        invalid_penalization, alpha, beta, eta, gamma, sparsity_balancer,
        feature_axis=2,
        verbose=verbose
    )
    
    # Initialize and run optimization
    optimizer.init(sample, nun_example, desired_class, outlier_calc, combined_heatmap)
    found_mask, prob, fitness_evol, cf_evol = optimizer.optimize()
    
    if found_mask is None:
        if verbose:
            print('Failed to converge')
        return sample, preds_sample.reshape(-1)
    
    # Generate final counterfactual (returns in B, C, L format)
    cf = optimizer.get_counterfactuals(sample, nun_example, np.expand_dims(found_mask, axis=0))
    cf = cf.squeeze(0)  # Remove batch dimension -> (C, L)
    
    # Get final predictions
    y_cf = model_predict(cf.reshape(1, C, L)).reshape(-1)
    
    # Convert back to (L, C) format for consistency with other methods
    cf = cf.T
    
    return cf, y_cf
