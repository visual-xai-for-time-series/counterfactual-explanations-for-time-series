import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import warnings

from cfts.cf__abstract.abstract import (
    batched_predict,
    ensure_ncl,
    revert_orientation,
    subsample_dataset,
)


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
    target_class=None,
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
    target_class : int, optional
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
    
    # Determine target_class class
    if target_class is None:
        sorted_indices = np.argsort(y_orig)[::-1]
        target_class = int(sorted_indices[1])
    
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
        print(f"CONFETTI: Original class {label_orig}, Target class {target_class}")
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
        Returns: (confidence in target_class class, sparsity)
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
        confidence = pred[target_class]
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
        if best_confidence >= theta and np.argmax(best_pred) == target_class:
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
        print(f"CONFETTI: Final confidence in target_class class: {best_confidence:.4f}")
        print(f"CONFETTI: Predicted class: {np.argmax(best_pred)}")
    
    return cf_shaped, best_pred


####
# confetti_nsga_cf — a faithful(er) reimplementation of the real algorithm
#
# `confetti_genetic_cf` above shares CONFETTI's starting idea (NUN + evolutionary
# search over a replace-with-NUN mask) but differs from the published algorithm
# in two structural ways: it searches a *full-length* mask instead of a
# contiguous window, and it optimizes a single weighted-sum fitness instead of a
# genuine multi-objective Pareto search. `confetti_nsga_cf` closes both gaps:
#
#   - NUN search restricted to reference candidates the model classifies with
#     confidence >= theta (mirrors `CONFETTI._nearest_unlike_neighbour`), falling
#     back to an unrestricted search only if nothing clears that bar.
#   - Binary search over contiguous window length, starting from the full
#     series and shrinking whenever a window yields a valid counterfactual
#     (mirrors `CONFETTI._optimization`'s `low`/`high` search). As in the
#     official-package baseline used in cf_confetti_forda_comparison.ipynb, the
#     window always starts at position 0 — the real package's alternative,
#     CAM-guided starting point, is skipped here for the same architecture
#     reason documented in that notebook (this repo's SimpleCNN has no
#     GAP-before-classifier and downsamples the time axis, so CAM's temporal
#     resolution wouldn't line up with the full-length window).
#   - A real multi-objective evolutionary search (NSGA-II — non-dominated
#     sorting + crowding distance) over three objectives per window: target_class-class
#     confidence, sparsity (fraction of the window perturbed), and proximity
#     (L2 distance to the original) — instead of a single weighted-sum fitness.
#     Binary tournament selection, two-point crossover and bit-flip mutation
#     mirror the official package's own operators (`TwoPointCrossover`,
#     `BitflipMutation`, `BinaryRandomSampling`): Bernoulli(0.5) initial
#     population, per-bit mutation probability `min(0.5, 1/window)` gated by a
#     per-individual `mutation_probability`, `crossover_probability` gating
#     whether two-point crossover is applied to a mating pair at all.
#   - Final selection among all windows' successful candidates via the same
#     alpha-weighted (confidence, sparsity) rule as
#     `CONFETTI._select_best_solution` (proximity is optimized for during the
#     search but, matching the original, not part of final selection).
#
# What is intentionally NOT reimplemented: the official package's NSGA-**III**
# reference-direction machinery (`das_dennis`, `NSGA3`) and its compiled Rust
# core. This uses NSGA-**II** instead — a simpler, well-known multi-objective
# algorithm sharing the same non-dominated-sorting/crowding-distance backbone,
# implemented here in plain NumPy. For three objectives on a population this
# size, NSGA-II and NSGA-III behave similarly; the difference matters far more
# at higher objective counts, which this problem doesn't have.
####


def _confetti_label(y) -> int:
    """Collapse a one-hot vector or scalar label to an int class index."""
    arr = np.asarray(y)
    return int(np.argmax(arr)) if arr.ndim > 0 and arr.size > 1 else int(arr)


def _confetti_softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax; applied unconditionally so confidence/theta comparisons
    are always valid probabilities regardless of whether the caller's model
    already ends in its own softmax (same rationale as cf_comte/comte.py's
    `_predict_probs`)."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _fast_non_dominated_sort(objectives: np.ndarray) -> List[np.ndarray]:
    """Standard NSGA-II non-dominated sort, vectorized.

    Parameters
    ----------
    objectives : (N, M) array, all objectives minimized.

    Returns
    -------
    list of index arrays, one per front, best (rank-0) front first.
    """
    n = objectives.shape[0]
    if n == 0:
        return []

    # dom[p, q] == True iff individual p Pareto-dominates individual q.
    less_eq = np.all(objectives[:, None, :] <= objectives[None, :, :], axis=2)
    less = np.any(objectives[:, None, :] < objectives[None, :, :], axis=2)
    dom = less_eq & less
    np.fill_diagonal(dom, False)

    counts = dom.sum(axis=0)  # counts[q] = number of individuals dominating q
    assigned = np.zeros(n, dtype=bool)

    fronts: List[np.ndarray] = []
    current = np.where(counts == 0)[0]
    while len(current) > 0:
        fronts.append(current)
        assigned[current] = True
        counts = counts - dom[current, :].sum(axis=0)
        current = np.where((counts == 0) & ~assigned)[0]

    return fronts


def _crowding_distance(front_objectives: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance for a single front. Boundary points get inf."""
    F, M = front_objectives.shape
    distance = np.zeros(F)
    if F <= 2:
        return np.full(F, np.inf)

    for m in range(M):
        order = np.argsort(front_objectives[:, m])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        m_min, m_max = front_objectives[order[0], m], front_objectives[order[-1], m]
        if m_max == m_min:
            continue
        for k in range(1, F - 1):
            distance[order[k]] += (
                front_objectives[order[k + 1], m] - front_objectives[order[k - 1], m]
            ) / (m_max - m_min)
    return distance


def _nsga2_rank_and_crowd(objectives: np.ndarray):
    """Return (rank, crowding_distance) arrays aligned to `objectives` rows."""
    fronts = _fast_non_dominated_sort(objectives)
    rank = np.empty(len(objectives), dtype=int)
    crowd = np.empty(len(objectives), dtype=float)
    for r, front in enumerate(fronts):
        rank[front] = r
        crowd[front] = _crowding_distance(objectives[front])
    return rank, crowd, fronts


def _nsga2_tournament_select(pop_size: int, rank: np.ndarray, crowd: np.ndarray, rng) -> int:
    """Binary tournament: lower rank wins; ties broken by larger crowding distance."""
    a, b = rng.integers(0, pop_size, size=2)
    if rank[a] < rank[b]:
        return int(a)
    if rank[b] < rank[a]:
        return int(b)
    return int(a) if crowd[a] >= crowd[b] else int(b)


def _nsga2_two_point_crossover(p1: np.ndarray, p2: np.ndarray, rng) -> Tuple[np.ndarray, np.ndarray]:
    n = len(p1)
    if n < 2:
        return p1.copy(), p2.copy()
    i, j = sorted(rng.integers(0, n, size=2))
    c1, c2 = p1.copy(), p2.copy()
    c1[i:j], c2[i:j] = p2[i:j], p1[i:j]
    return c1, c2


def _nsga2_bitflip_mutate(individual: np.ndarray, prob_var: float, rng) -> np.ndarray:
    flips = rng.random(len(individual)) < prob_var
    out = individual.copy()
    out[flips] = 1 - out[flips]
    return out


def _nsga2_window_search(
    sample_cl: np.ndarray,
    nun_cl: np.ndarray,
    window: int,
    model: nn.Module,
    device: torch.device,
    target_class: int,
    population_size: int,
    max_generations: int,
    crossover_probability: float,
    mutation_probability: float,
    rng,
):
    """NSGA-II search over which positions in `sample_cl[:, :window]` to replace
    with `nun_cl`'s values at those same positions, minimizing
    (1 - target_class confidence, sparsity, proximity).

    Returns
    -------
    cand : (P, C, L) final-generation candidate counterfactuals
    scores : (P, num_classes) raw model outputs for `cand`
    objs : (P, 3) objective values for `cand`
    success_mask : (P,) bool — cand[i]'s predicted class == target_class
    """
    C, L = sample_cl.shape
    prob_var = min(0.5, 1.0 / max(window, 1))
    nun_region = nun_cl[:, :window]
    orig_region = sample_cl[:, :window]

    def evaluate(pop_mask: np.ndarray):
        P = len(pop_mask)
        cand = np.repeat(sample_cl[None, :, :], P, axis=0)
        mask_b = pop_mask.astype(bool)[:, None, :]  # (P, 1, window)
        cand[:, :, :window] = np.where(
            mask_b, np.broadcast_to(nun_region, (P, C, window)), np.broadcast_to(orig_region, (P, C, window))
        )
        scores = batched_predict(model, cand, device, batch_size=max(P, 1))
        probs = _confetti_softmax(scores)
        conf_target = probs[:, target_class]
        sparsity = pop_mask.sum(axis=1) / window
        proximity = np.linalg.norm((cand - sample_cl[None]).reshape(P, -1), axis=1)
        objs = np.stack([1.0 - conf_target, sparsity, proximity], axis=1)
        preds = np.argmax(scores, axis=1)
        return cand, scores, objs, preds

    population = (rng.random((population_size, window)) < 0.5).astype(np.int64)  # BinaryRandomSampling-style
    cand, scores, objs, preds = evaluate(population)

    for _ in range(max_generations):
        rank, crowd, _ = _nsga2_rank_and_crowd(objs)

        offspring = []
        while len(offspring) < population_size:
            i1 = _nsga2_tournament_select(population_size, rank, crowd, rng)
            i2 = _nsga2_tournament_select(population_size, rank, crowd, rng)
            p1, p2 = population[i1], population[i2]
            if rng.random() < crossover_probability:
                c1, c2 = _nsga2_two_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if rng.random() < mutation_probability:
                c1 = _nsga2_bitflip_mutate(c1, prob_var, rng)
            if rng.random() < mutation_probability:
                c2 = _nsga2_bitflip_mutate(c2, prob_var, rng)
            offspring.append(c1)
            if len(offspring) < population_size:
                offspring.append(c2)
        offspring = np.array(offspring[:population_size])
        off_cand, off_scores, off_objs, off_preds = evaluate(offspring)

        combined_pop = np.concatenate([population, offspring], axis=0)
        combined_objs = np.concatenate([objs, off_objs], axis=0)
        combined_cand = np.concatenate([cand, off_cand], axis=0)
        combined_scores = np.concatenate([scores, off_scores], axis=0)
        combined_preds = np.concatenate([preds, off_preds], axis=0)

        fronts = _fast_non_dominated_sort(combined_objs)
        new_indices: list[int] = []
        for front in fronts:
            if len(new_indices) + len(front) <= population_size:
                new_indices.extend(front.tolist())
            else:
                crowd_f = _crowding_distance(combined_objs[front])
                order = np.argsort(-crowd_f)
                remaining = population_size - len(new_indices)
                new_indices.extend(front[order[:remaining]].tolist())
                break
        new_indices_arr = np.array(new_indices)

        population = combined_pop[new_indices_arr]
        objs = combined_objs[new_indices_arr]
        cand = combined_cand[new_indices_arr]
        scores = combined_scores[new_indices_arr]
        preds = combined_preds[new_indices_arr]

    success_mask = preds == target_class
    return cand, scores, objs, success_mask


def confetti_nsga_cf(
    sample: np.ndarray | list,
    model: nn.Module,
    target_class: Optional[int] = None,
    dataset: list | np.ndarray = None,
    theta: float = 0.51,
    alpha: float = 0.5,
    population_size: int = 60,
    max_generations: int = 40,
    crossover_probability: float = 1.0,
    mutation_probability: float = 0.9,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """CONFETTI reimplementation closer to the official algorithm than
    `confetti_genetic_cf`: confidence-gated NUN search + binary search over a
    contiguous replacement window + genuine multi-objective (NSGA-II) search
    within that window. See the module comment above for exactly what mirrors
    the official package vs. what's simplified (NSGA-II instead of NSGA-III, no
    CAM-guided naive stage).

    Follows the same signature pattern as every other CF method in this
    repository (`comte_cf`, `native_guide_uni_cf`, …).

    Parameters
    ----------
    sample:
        Query time series. Accepts 1-D `(L,)`, `(C, L)` or `(L, C)` NumPy
        arrays.
    model:
        Trained PyTorch classifier accepting `(B, C, L)` and returning
        `(B, num_classes)` logits or probabilities.
    target_class:
        Class index to flip toward. When `None`, mirrors the official
        package's own behaviour: any reference sample with a different
        predicted label is an eligible NUN, and the target_class becomes whichever
        label the chosen NUN carries (for binary datasets this is always "the
        other class").
    dataset:
        Sequence of `(x, y)` pairs used as the NUN candidate pool. Required.
    theta:
        Minimum predicted confidence a NUN candidate must have in its own
        predicted class to be eligible (mirrors `CONFETTI`'s `theta`). Relaxed
        automatically if no candidate clears it.
    alpha:
        Trade-off weight between confidence and sparsity when selecting the
        final counterfactual among all windows' successes (mirrors
        `CONFETTI._select_best_solution`'s `alpha`).
    population_size, max_generations, crossover_probability, mutation_probability:
        NSGA-II parameters for the per-window search; same names/defaults as
        `CONFETTI.generate_counterfactuals`'s own GA parameters (population
        size and generation count are kept smaller by default here since this
        is a pure-NumPy search, not the compiled Rust core).
    max_samples:
        If set, subsample `dataset` to at most this many items first.
    seed:
        Seeds the internal NSGA-II random generator.
    verbose:
        Print per-window progress when `True`.

    Returns
    -------
    counterfactual : np.ndarray, same shape/orientation as `sample`.
    scores : np.ndarray, shape (num_classes,) — raw model output for it.
    """
    if dataset is None:
        raise ValueError("confetti_nsga_cf requires a dataset to search for a nearest unlike neighbour.")

    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)

    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)

    sample_cl, ts, ori = ensure_ncl(np.asarray(sample, dtype=np.float32), dataset)
    C, L = sample_cl.shape
    labels = np.array([_confetti_label(dataset[i][1]) for i in range(len(dataset))])

    scores_orig = batched_predict(model, sample_cl.reshape(1, C, L), device)[0]
    probs_orig = _confetti_softmax(scores_orig[None])[0]
    label_orig = int(np.argmax(probs_orig))

    ref_scores = batched_predict(model, ts, device)
    ref_probs = _confetti_softmax(ref_scores)
    pred_labels = np.argmax(ref_scores, axis=1)
    pred_confidence = ref_probs[np.arange(len(ts)), pred_labels]

    unlike_mask = (pred_labels == target_class) if target_class is not None else (pred_labels != label_orig)
    confident_mask = unlike_mask & (pred_confidence >= theta)

    if not np.any(confident_mask):
        if verbose:
            print(f"[confetti_nsga_cf] no NUN candidate clears theta={theta}; relaxing confidence gate.")
        confident_mask = unlike_mask

    if not np.any(confident_mask):
        if verbose:
            print("[confetti_nsga_cf] no unlike neighbour found at all.")
        return revert_orientation(sample_cl, ori), scores_orig

    candidate_idx = np.where(confident_mask)[0]
    dists = np.linalg.norm((ts[candidate_idx] - sample_cl[None]).reshape(len(candidate_idx), -1), axis=1)
    nun_idx = candidate_idx[np.argmin(dists)]
    nun_cl = ts[nun_idx]

    if target_class is None:
        target_class = int(pred_labels[nun_idx])

    if label_orig == target_class:
        if verbose:
            print(f"[confetti_nsga_cf] sample already predicted as target_class={target_class}.")
        return revert_orientation(sample_cl, ori), scores_orig

    successes = []  # list of {"cf", "scores", "objs"}
    best_effort = {"cf": sample_cl, "scores": scores_orig, "conf": float(probs_orig[target_class])}

    low, high = 1, L
    while low <= high:
        window = (low + high) // 2
        cand, scores, objs, success_mask = _nsga2_window_search(
            sample_cl, nun_cl, window, model, device, target_class,
            population_size, max_generations, crossover_probability, mutation_probability, rng,
        )

        gen_best = int(np.argmax(1.0 - objs[:, 0]))
        if (1.0 - objs[gen_best, 0]) > best_effort["conf"]:
            best_effort = {"cf": cand[gen_best], "scores": scores[gen_best], "conf": float(1.0 - objs[gen_best, 0])}

        succeeded = bool(np.any(success_mask))
        if verbose:
            print(f"[confetti_nsga_cf] window={window} succeeded={succeeded} n_success={int(success_mask.sum())}")

        if succeeded:
            for i in np.where(success_mask)[0]:
                successes.append({"cf": cand[i], "scores": scores[i], "objs": objs[i]})
            high = window - 1
        else:
            low = window + 1

    if not successes:
        if verbose:
            print("[confetti_nsga_cf] no window reached target_class; returning best-effort candidate.")
        return revert_orientation(best_effort["cf"], ori), best_effort["scores"]

    confidences = np.array([1.0 - s["objs"][0] for s in successes])
    sparsities = np.array([s["objs"][1] for s in successes])
    # Selection mirrors CONFETTI._select_best_solution: alpha-weighted
    # confidence vs. sparsity only (proximity guided the search but, matching
    # the original, isn't part of final selection).
    selection_score = alpha * confidences - (1 - alpha) * sparsities
    best = successes[int(np.argmax(selection_score))]

    if verbose:
        print(
            f"[confetti_nsga_cf] done — {len(successes)} successful candidate(s) across all windows, "
            f"picked confidence={float(1.0 - best['objs'][0]):.4f} sparsity={float(best['objs'][1]):.4f}"
        )

    return revert_orientation(best["cf"], ori), best["scores"]


# Default CONFETTI entry point for this repo, matching the `<method>_cf` naming
# convention every other method uses (comte_cf, native_guide_uni_cf, …).
# confetti_nsga_cf is the variant structurally closest to the published
# algorithm (confidence-gated NUN search, contiguous-window binary search,
# multi-objective NSGA-II) — see confetti_forda_comparison.ipynb for the
# comparison against confetti_genetic_cf and the official package that
# motivated picking it as the default over confetti_genetic_cf.
confetti_cf = confetti_nsga_cf


def confetti_package_cf(
    sample,
    model,
    reference_data,
    reference_labels=None,
    reference_weights=None,
    target_class=None,
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
    target_class : int, optional
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
    target_class=None,
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
    target_class : int, optional
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
        target_class=target_class,
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
            target_class=target_class,
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
                if target_class is not None:
                    print(f"  - Target class: {target_class}")
                print(f"  - Confidence in target_class: {pred_simple[target_class] if target_class is not None else 'N/A':.4f}")
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
