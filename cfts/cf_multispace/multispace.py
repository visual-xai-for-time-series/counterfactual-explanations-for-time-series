import torch
import torch.nn as nn
import numpy as np
import copy
from abc import ABC, abstractmethod
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


def multispace_fast(sample, model, dataset=None, weight_function=GradientShap,
                   iterate=None, sub_len=1, population_size=50, max_iterations=100,
                   sparsity_weight=0.3, validity_weight=0.7, verbose=False):
    """Multi-SpaCE counterfactual generation, adapted for speed.

    Same subsequence-based, mask-swap-with-NUN idea as multispace_cf, but with several
    adaptations that trade fidelity to the official repository's NSGA-II strategy for a much
    simpler, single-objective search:
    - A single scalar fitness (weighted sum of validity and sparsity) instead of a genuine
      multi-objective Pareto front -- see multispace_cf's AutoencoderOutlierCalculator and
      MOEvolutionaryOptimizer for the real 4-objective, non-dominated-sorted version.
    - No staged grouped/pruning search, no plausibility term, and independent per-element
      mutation instead of the boundary-aware shrink/extend/remove operators multispace_cf uses.
    - Validity is only a fitness weight (`validity_weight`), not a hard constraint, so the
      returned candidate is not guaranteed to actually reach the target class.

    This generally makes multispace_fast faster but less sparse, less contiguous, and less
    plausible than multispace_cf -- see cfts/cf_multispace/multispace_comparison.ipynb for a
    measured comparison against both multispace_cf and the vendored official code.

    Args:
        sample: Input time series
        model: Classifier model
        dataset: Training dataset (required)
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
    if dataset is None:
        raise ValueError("multispace_fast requires a dataset to find the NUN.")
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


####
# multispace_cf: closer to the official MarioRefoyo/Multi-SpaCE algorithm
#
# multispace_fast above is a simplified, single-objective genetic algorithm. The published
# method is a genuine multi-objective genetic algorithm (NSGA-II) with a staged
# grouped -> individual -> pruning search and a real autoencoder-based plausibility objective.
# multispace_cf below ports that algorithm from the official repository
# (https://github.com/MarioRefoyo/Multi-SpaCE, methods/MultiSubSpaCE/*), using the paper's own
# recommended "final" hyperparameters (experiments/params_cf/multisubspace_final.json):
# grouped_iter=75, individual stage skipped (iter=0 in that config), pruning_iter=25,
# population_size=100, change_subseq_mutation_prob=0.75, final_pruning_mutation_prob=0.75,
# init_pct=0.2, plausibility_objective="ios", init_fi="none" (no feature-importance-guided
# init). The optional windowed-mask optimization (mask_window_pct) is dropped -- it is unused
# by the paper's own recommended config and adds substantial complexity aimed at very long
# series.
####


class _ConvAutoencoder1D(nn.Module):
    """Small conv1d encoder/decoder used by AutoencoderOutlierCalculator to score plausibility
    via reconstruction error -- the autoencoder-based plausibility objective multispace_fast
    above has no equivalent of."""

    def __init__(self, channels, latent_channels=16):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(channels, 16, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(16, latent_channels, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(inplace=True),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(latent_channels, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(16, channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        if out.shape[-1] != x.shape[-1]:
            out = torch.nn.functional.interpolate(out, size=x.shape[-1], mode="linear", align_corners=False)
        return out


def train_outlier_autoencoder(calibration_data, device, epochs=40, batch_size=64, lr=1e-3):
    """Train a small conv1d autoencoder for AutoencoderOutlierCalculator.

    Args:
        calibration_data: In-distribution reference samples, shape (N, L, C).
        device: Torch device to train on.

    Returns:
        Trained autoencoder (nn.Module).
    """
    calibration_data = np.asarray(calibration_data, dtype=np.float32)
    n_samples = calibration_data.shape[0]
    channels = calibration_data.shape[2]
    autoencoder = _ConvAutoencoder1D(channels).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    data_cl = numpy_to_torch(np.transpose(calibration_data, (0, 2, 1)), device)  # (N, L, C) -> (N, C, L)

    autoencoder.train()
    for _ in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, batch_size):
            batch = data_cl[perm[start:start + batch_size]]
            optimizer.zero_grad()
            loss = torch.mean((autoencoder(batch) - batch) ** 2)
            loss.backward()
            optimizer.step()
    autoencoder.eval()
    return autoencoder


class AutoencoderOutlierCalculator:
    """Reconstruction-error-based outlier calculator matching the official Multi-SpaCE
    repository's AEOutlierCalculator (methods/outlier_calculators.py): mean absolute
    reconstruction error from a real trained autoencoder, scaled to [~0, 1] against
    in-distribution calibration data.
    """

    def __init__(self, autoencoder, calibration_data, device):
        """
        Args:
            autoencoder: Trained autoencoder (e.g. from train_outlier_autoencoder), operating
                on (N, C, L) channel-first input/output.
            calibration_data: In-distribution reference samples, shape (N, L, C).
            device: Torch device the autoencoder lives on.
        """
        self.autoencoder = autoencoder
        self.device = device
        self.length = calibration_data.shape[1]
        self.n_channels = calibration_data.shape[2]

        calibration_scores = self._get_raw_outlier_scores(calibration_data)
        self.min_score = min(0, calibration_scores.min())
        self.max_score = calibration_scores.max()

    def _get_raw_outlier_scores(self, data):
        data = np.asarray(data, dtype=np.float32).reshape(-1, self.length, self.n_channels)
        x_lc = numpy_to_torch(data, self.device)
        x_cl = x_lc.permute(0, 2, 1)
        with torch.no_grad():
            recon_cl = self.autoencoder(x_cl)
        recon_lc = recon_cl.permute(0, 2, 1)
        errors = torch.mean(torch.abs(x_lc - recon_lc), dim=(1, 2))
        return detach_to_numpy(errors)

    def get_outlier_scores(self, data):
        raw = self._get_raw_outlier_scores(data)
        scaled = (raw - self.min_score) / (self.max_score - self.min_score)
        return scaled.flatten()


def fitness_function_mo(ms, predicted_probs, target_class, outlier_scores, original_outlier_score,
                        invalid_penalization_scalar):
    """4-objective fitness: [validity, -sparsity, -contiguity, -plausibility]. Ports
    methods/MultiSubSpaCE/FitnessFunctions.py::fitness_function_mo exactly."""
    objectives_fitness = np.empty((ms.shape[0], 4))
    objectives_fitness[:] = np.nan

    # Predicted probs
    objectives_fitness[:, 0] = predicted_probs[:, target_class]

    # Sparsity
    ones_pct = ms.sum(axis=(1, 2)) / (ms.shape[1] * ms.shape[2])
    objectives_fitness[:, 1] = -ones_pct

    # Subsequences (contiguity)
    subsequences = np.count_nonzero(np.diff(ms, prepend=0, axis=1) == 1, axis=(1, 2))
    feature_avg_subsequences = subsequences / ms.shape[2]
    subsequences_pct = feature_avg_subsequences / (ms.shape[1] // 2)
    objectives_fitness[:, 2] = -1 * (subsequences_pct ** 0.25)

    # Outlier scores (increase in outlier score)
    increase_outlier_scores = outlier_scores - original_outlier_score
    increase_outlier_scores[increase_outlier_scores < 0] = 0
    objectives_fitness[:, 3] = -increase_outlier_scores

    # Apply penalization to all objectives, so invalid candidates fall behind the pareto front
    predicted_classes = np.argmax(predicted_probs, axis=1)
    penalization_vector = (predicted_classes != target_class).astype(int)
    penalization_matrix = np.repeat(penalization_vector.reshape(-1, 1), repeats=4, axis=1)
    objectives_fitness = objectives_fitness - invalid_penalization_scalar * penalization_matrix

    return objectives_fitness


class MOEvolutionaryOptimizer(ABC):
    """NSGA-II (non-dominated sorting genetic algorithm II) base class. Ports
    methods/MultiSubSpaCE/MOEvolutionaryOptimizers.py::MOEvolutionaryOptimizer, minus the
    optional windowed-mask optimization (mask_window_pct), which the paper's own recommended
    config does not use, and the vestigial unused `model_wrapper` argument to init()."""

    def __init__(self, fitness_func, prediction_func, population_size, max_iter,
                 init_pct, reinit, init_random_mix_ratio,
                 invalid_penalization,
                 individual_channel_search):
        self.population_size = population_size
        self.fitness_func = fitness_func
        self.invalid_penalization = invalid_penalization
        self.prediction_func = prediction_func
        self.max_iter = max_iter
        self.original_init_pct = init_pct
        self.individual_channel_search = individual_channel_search
        self.reinit = reinit
        self.init_random_mix_ratio = init_random_mix_ratio

    def increase_init_pct(self):
        if self.init_pct >= 1:
            self.init_pct = 1
            return
        next_init_pct = self.init_pct + 0.2
        if next_init_pct >= 1 and not self.tried_almost_full_init:
            self.init_pct = 0.95
            self.tried_almost_full_init = True
        else:
            self.init_pct = min(1, next_init_pct)

    def get_mask_shape(self):
        if self.individual_channel_search:
            return self.population_size, self.ts_length, self.n_features
        return self.population_size, self.ts_length, 1

    def init_population(self, importance_heatmap=None):
        if self.individual_channel_search:
            random_data = np.random.uniform(0, 1, self.get_mask_shape())
            if importance_heatmap is not None:
                inducted_data = (self.init_random_mix_ratio * random_data
                                  + (1 - self.init_random_mix_ratio) * importance_heatmap) / 2
            else:
                inducted_data = random_data
        else:
            random_data = np.random.uniform(0, 1, (self.population_size, self.ts_length, 1))
            if importance_heatmap is not None:
                importance_heatmap_mean = importance_heatmap.mean(axis=1).reshape(self.ts_length, 1)
                inducted_data = (self.init_random_mix_ratio * random_data
                                  + (1 - self.init_random_mix_ratio) * importance_heatmap_mean) / 2
            else:
                inducted_data = random_data

        quantile = np.quantile(inducted_data.flatten(), 1 - self.init_pct)
        return (inducted_data > quantile).astype(int)

    def init(self, x_orig, nun_example, target_class, init_mask=None,
             outlier_calculator=None, importance_heatmap=None):
        self.x_orig = x_orig
        self.nun_example = nun_example
        self.target_class = target_class
        self.outlier_calculator = outlier_calculator
        self.importance_heatmap = importance_heatmap
        self.init_pct = copy.deepcopy(self.original_init_pct)
        self.tried_almost_full_init = False

        if self.outlier_calculator is not None:
            self.outlier_scores_orig = self.outlier_calculator.get_outlier_scores(self.x_orig)
        else:
            self.outlier_scores_orig = np.zeros((1,))

        self.n_features = x_orig.shape[1]
        self.ts_length = x_orig.shape[0]

        if init_mask is not None:
            expected_channels = self.n_features if self.individual_channel_search else 1
            if init_mask.shape[2] != expected_channels:
                raise ValueError(
                    f"Init mask must have {expected_channels} channel(s) for the current optimizer mode. "
                    f"Got {init_mask.shape[2]}."
                )
            random_idx = np.random.randint(len(init_mask), size=self.population_size)
            past_population = init_mask[random_idx].astype(int)

            offsprings_population = self.produce_offsprings(past_population, self.population_size)
            complete_population = np.vstack((past_population, offsprings_population))
            objectives_fitness = self.compute_fitness(complete_population)
            fronts = self.fast_non_dominated_sorting(objectives_fitness)
            _, sorted_population, _, sorted_ranks, _ = self.crowing_distance_sorting(
                fronts, complete_population, objectives_fitness, self.population_size
            )
            best_front_individuals = np.where(np.array(sorted_ranks) == 0)[0]
            random_choices = np.random.choice(len(best_front_individuals), self.population_size)
            population = sorted_population[best_front_individuals[random_choices]]
        else:
            population = self.init_population(self.importance_heatmap)

        self.population = population

    @abstractmethod
    def mutate(self, sub_population):
        pass

    @staticmethod
    def get_single_crossover_mask(subpopulation):
        split_points = np.random.randint(0, subpopulation.shape[1], size=subpopulation.shape[0] // 2)
        mask = np.arange(subpopulation.shape[1]) < split_points[:, np.newaxis]
        return mask

    def produce_offsprings(self, subpopulation, number):
        # Put channels as individual examples
        adapted_subpopulation = np.swapaxes(subpopulation, 2, 1)
        subpopulation_n_features = subpopulation.shape[2]
        adapted_number = number * subpopulation_n_features
        adapted_subpopulation = adapted_subpopulation.reshape(adapted_number, -1)

        mask = self.get_single_crossover_mask(adapted_subpopulation)
        matches = np.random.choice(np.arange(adapted_subpopulation.shape[0]),
                                   size=(adapted_subpopulation.shape[0] // 2, 2), replace=False)

        offsprings1 = np.empty((adapted_number // 2, adapted_subpopulation.shape[1]))
        offsprings1[mask] = adapted_subpopulation[matches[:, 0]][mask]
        offsprings1[~mask] = adapted_subpopulation[matches[:, 1]][~mask]
        offsprings2 = np.zeros((adapted_number // 2, adapted_subpopulation.shape[1]))
        offsprings2[mask] = adapted_subpopulation[matches[:, 1]][mask]
        offsprings2[~mask] = adapted_subpopulation[matches[:, 0]][~mask]
        adapted_offsprings = np.concatenate([offsprings1, offsprings2])

        adapted_offsprings = self.mutate(adapted_offsprings)

        adapted_offsprings = adapted_offsprings.reshape(number, subpopulation_n_features, -1)
        return np.swapaxes(adapted_offsprings, 2, 1)

    def get_counterfactuals(self, x_orig, nun_example, population):
        population_size = population.shape[0]
        if self.individual_channel_search:
            population_mask = population.astype(bool)
        else:
            population = np.repeat(population, self.n_features, axis=2)
            population_mask = population.astype(bool)

        x_orig_ext = np.tile(x_orig, (population_size, 1, 1))
        nun_ext = np.tile(nun_example, (population_size, 1, 1))

        counterfactuals = np.zeros(population_mask.shape)
        counterfactuals[~population_mask] = x_orig_ext[~population_mask]
        counterfactuals[population_mask] = nun_ext[population_mask]
        return counterfactuals

    def compute_fitness(self, population):
        population_cfs = self.get_counterfactuals(self.x_orig, self.nun_example, population)
        predicted_probs = self.prediction_func(population_cfs)

        if self.outlier_calculator is not None:
            outlier_scores = self.outlier_calculator.get_outlier_scores(population_cfs)
        else:
            outlier_scores = np.zeros((predicted_probs.shape[0], 1))

        return self.fitness_func(population, predicted_probs, self.target_class, outlier_scores,
                                 self.outlier_scores_orig, self.invalid_penalization)

    @staticmethod
    def tournament_mo(ranks, crowding_distances, number):
        population_size = len(ranks)
        random_pairs = np.random.randint(population_size, size=(number, 2))

        random_pairs_ranks = np.ones(random_pairs.shape) * -1
        random_pairs_ranks[:, 0] = np.array(ranks)[random_pairs[:, 0]]
        random_pairs_ranks[:, 1] = np.array(ranks)[random_pairs[:, 1]]
        random_pairs_ranks_winners = np.argmin(random_pairs_ranks, axis=1)
        pairs_ranks_winners = random_pairs[range(population_size), random_pairs_ranks_winners]

        random_pairs_cdist = np.ones(random_pairs.shape) * -1
        random_pairs_cdist[:, 0] = np.array(crowding_distances)[random_pairs[:, 0]]
        random_pairs_cdist[:, 1] = np.array(crowding_distances)[random_pairs[:, 1]]
        random_pairs_cdist_winners = np.argmax(random_pairs_cdist, axis=1)
        pairs_cdist_winners = random_pairs[range(population_size), random_pairs_cdist_winners]

        aux = np.diff(random_pairs_ranks, axis=1)
        equal_ranks_indexes = np.where(aux == 0)[0]
        selected_indexes = pairs_ranks_winners.copy()
        if len(equal_ranks_indexes) > 0:
            selected_indexes[equal_ranks_indexes] = pairs_cdist_winners[equal_ranks_indexes]

        return selected_indexes

    def select_candidates_mo(self, population, ranks, crowding_distances, number):
        selected_indexes = self.tournament_mo(ranks, crowding_distances, number)
        return population[selected_indexes]

    @staticmethod
    def fast_non_dominated_sorting(objectives_fitness):
        population_size = objectives_fitness.shape[0]
        repeated_objectives = np.repeat(objectives_fitness, repeats=population_size, axis=0)
        stacked_objectives = np.tile(objectives_fitness, (population_size, 1))

        l_eq_matrix = repeated_objectives <= stacked_objectives
        l_matrix = repeated_objectives < stacked_objectives
        and_cond_vector = l_eq_matrix.all(axis=1)
        or_cond_vector = l_matrix.any(axis=1)
        dominated_vector = and_cond_vector * or_cond_vector
        dominated_by_matrix = dominated_vector.reshape(population_size, population_size, order='C')

        fronts = []
        current_excluded_individuals = []
        dominated_by_matrix_copy = dominated_by_matrix.copy()
        while dominated_by_matrix_copy.sum() > 0:
            domination_count = dominated_by_matrix_copy.sum(axis=1)
            original_front_individuals = np.where(domination_count == 0)[0].tolist()
            front_individuals = list(set(original_front_individuals) - set(current_excluded_individuals))
            fronts.append(front_individuals)
            current_excluded_individuals = current_excluded_individuals + front_individuals
            dominated_by_matrix_copy[:, front_individuals] = False

        final_front_individuals = list(set(range(len(objectives_fitness))) - set(current_excluded_individuals))
        fronts.append(final_front_individuals)
        return fronts

    @staticmethod
    def calculate_front_crowding_distance(objectives_fitness, front_indexes):
        front_len = len(front_indexes)
        n_objectives = objectives_fitness.shape[1]
        front_all_objectives_fitness = objectives_fitness[front_indexes, :]
        front_distance = np.ones((front_len, n_objectives)) * np.inf
        for o in range(n_objectives):
            front_objective_fitness = front_all_objectives_fitness[:, o]
            order_by_objective_fitness = np.argsort(front_objective_fitness)
            o_min = front_objective_fitness[order_by_objective_fitness[0]]
            o_max = front_objective_fitness[order_by_objective_fitness[-1]]
            norm_o = o_max - o_min

            if norm_o == 0:
                front_distance[:, o] = 0
                continue

            for n in range(1, front_len - 1):
                idx_minus = order_by_objective_fitness[n - 1]
                idx = order_by_objective_fitness[n]
                idx_plus = order_by_objective_fitness[n + 1]
                front_distance[idx, o] = front_objective_fitness[idx_plus] - front_objective_fitness[idx_minus]

            front_distance[:, o] = front_distance[:, o] / norm_o

        return front_distance.sum(axis=1)

    @staticmethod
    def crowing_distance_sorting(fronts, population, population_objective_fitness, desired_size):
        ranks, cdists, sorted_idx = [], [], []
        current_population_count = 0
        for i_front, front in enumerate(fronts):
            front_len = len(front)
            front_crowing_distance = MOEvolutionaryOptimizer.calculate_front_crowding_distance(
                population_objective_fitness, front
            )
            sort_idx_crowing_distance = np.argsort(front_crowing_distance)[::-1]
            sorted_front = np.array(front)[sort_idx_crowing_distance]

            if (current_population_count + front_len) > desired_size:
                front_needed_len = desired_size - current_population_count
                sorted_idx = sorted_idx + sorted_front[:front_needed_len].tolist()
                ranks = ranks + [i_front] * front_needed_len
                cdists = cdists + front_crowing_distance[sort_idx_crowing_distance[:front_needed_len]].tolist()
                break
            else:
                sorted_idx = sorted_idx + sorted_front.tolist()
                ranks = ranks + [i_front] * front_len
                cdists = cdists + front_crowing_distance[sort_idx_crowing_distance].tolist()
            current_population_count += front_len

        sorted_population = population[sorted_idx]
        sorted_objectives = population_objective_fitness[sorted_idx]
        return sorted_idx, sorted_population, sorted_objectives, ranks, cdists

    @staticmethod
    def calculate_front_avg_fitness(front, objective_fitness):
        return objective_fitness[front, :].mean()

    def optimize(self):
        best_score = -100
        best_individuals = None
        best_avg_fitness_evolution = []

        objectives_fitness = self.compute_fitness(self.population)
        fronts = self.fast_non_dominated_sorting(objectives_fitness)
        _, sorted_population, _, sorted_ranks, sorted_cdists = self.crowing_distance_sorting(
            fronts, self.population, objectives_fitness, self.population_size
        )

        best_avg_fitness = self.calculate_front_avg_fitness(fronts[0], objectives_fitness)
        best_avg_fitness_evolution.append(best_avg_fitness)

        iteration = 0
        while iteration < self.max_iter:
            selected_candidates = self.select_candidates_mo(sorted_population, sorted_ranks, sorted_cdists, self.population_size)
            offsprings_population = self.produce_offsprings(selected_candidates, self.population_size)
            complete_population = np.vstack((sorted_population, offsprings_population))

            objectives_fitness = self.compute_fitness(complete_population)
            fronts = self.fast_non_dominated_sorting(objectives_fitness)
            _, sorted_population, _, sorted_ranks, sorted_cdists = self.crowing_distance_sorting(
                fronts, complete_population, objectives_fitness, self.population_size
            )
            self.population = sorted_population

            best_avg_fitness = self.calculate_front_avg_fitness(fronts[0], objectives_fitness)
            best_avg_fitness_evolution.append(best_avg_fitness)
            if best_avg_fitness > best_score:
                best_score = best_avg_fitness
                best_front_individuals = np.where(np.array(sorted_ranks) == 0)[0]
                best_individuals = sorted_population[best_front_individuals]

            if self.reinit and (iteration == 50) and (self.init_pct < 1) and (best_avg_fitness < -self.invalid_penalization + 1):
                iteration = 0
                self.increase_init_pct()
                self.population = self.init_population(self.importance_heatmap)
                objectives_fitness = self.compute_fitness(self.population)
                fronts = self.fast_non_dominated_sorting(objectives_fitness)
                _, sorted_population, _, sorted_ranks, sorted_cdists = self.crowing_distance_sorting(
                    fronts, self.population, objectives_fitness, self.population_size
                )
            else:
                iteration += 1

            if np.all(self.population == self.population[0]):
                population_cfs = self.get_counterfactuals(self.x_orig, self.nun_example, self.population)
                predicted_class = np.argmax(self.prediction_func(population_cfs), axis=1)[0]
                if predicted_class == self.target_class:
                    break
                else:
                    iteration = 0
                    self.increase_init_pct()
                    self.population = self.init_population(self.importance_heatmap)
                    objectives_fitness = self.compute_fitness(self.population)
                    fronts = self.fast_non_dominated_sorting(objectives_fitness)
                    _, sorted_population, _, sorted_ranks, sorted_cdists = self.crowing_distance_sorting(
                        fronts, self.population, objectives_fitness, self.population_size
                    )

        return best_individuals, best_avg_fitness_evolution


class IntegratedPruningNSubsequenceEvolutionaryOptimizer(MOEvolutionaryOptimizer):
    """Mutation operators that only ever act at existing subsequence boundaries (shrink/extend),
    optionally create new subsequences, and optionally remove whole subsequences (pruning).
    Ports methods/MultiSubSpaCE/MOEvolutionaryOptimizers.py::
    IntegratedPruningNSubsequenceEvolutionaryOptimizer."""

    def __init__(self, fitness_func, prediction_func,
                 population_size=100, max_iter=100,
                 change_subseq_mutation_prob=0.05, add_subseq_mutation_prob=0, remove_subseq_mutation_prob=0.05,
                 init_pct=0.4, reinit=True, init_random_mix_ratio=0.5,
                 invalid_penalization=100,
                 individual_channel_search=False):
        super().__init__(fitness_func, prediction_func, population_size, max_iter,
                         init_pct, reinit, init_random_mix_ratio,
                         invalid_penalization, individual_channel_search)
        self.change_subseq_mutation_prob = change_subseq_mutation_prob
        self.add_subseq_mutation_prob = add_subseq_mutation_prob
        self.remove_subseq_mutation_prob = remove_subseq_mutation_prob

    @staticmethod
    def add_subsequence_mutation(population, mutation_prob):
        ones_mask = np.in1d(population, 1).reshape(population.shape)
        before_ones_mask = np.roll(ones_mask, -1, axis=1)
        before_ones_mask[:, ones_mask.shape[1] - 1] = False
        after_ones_mask = np.roll(ones_mask, 1, axis=1)
        after_ones_mask[:, 0] = False
        before_after_ones_mask = before_ones_mask + after_ones_mask
        before_after_ones_mask[ones_mask] = False
        possibilities_mask = ~(before_after_ones_mask + ones_mask)

        new_subsequences = np.zeros(population.shape).astype(int)
        for i, row in enumerate(possibilities_mask):
            if np.random.random() < mutation_prob:
                valid_idx = np.where(row == True)[0]
                if len(valid_idx) > 0:
                    chosen_idx = np.random.choice(valid_idx)
                    subseq_len = min(population.shape[1] - chosen_idx, np.random.randint(2, 6))
                    new_subsequences[i, chosen_idx:chosen_idx + subseq_len] = 1

        return np.clip(population + new_subsequences, 0, 1)

    @staticmethod
    def extend_mutation(population, mutation_prob):
        ones_mask = np.in1d(population, 1).reshape(population.shape)
        before_ones_mask = np.roll(ones_mask, -1, axis=1)
        before_ones_mask[:, ones_mask.shape[1] - 1] = False
        after_ones_mask = np.roll(ones_mask, 1, axis=1)
        after_ones_mask[:, 0] = False
        before_after_ones_mask = before_ones_mask + after_ones_mask
        before_after_ones_mask[ones_mask] = False

        random_mutations = (np.random.uniform(0, 1, population.shape) < mutation_prob).astype(int)
        valid_mutations = np.zeros(population.shape).astype(int)
        valid_mutations[before_after_ones_mask] = random_mutations[before_after_ones_mask]
        return (population + valid_mutations) % 2

    @staticmethod
    def shrink_mutation(population, mutation_prob):
        mask_beginnings = np.diff(population, 1, prepend=0)
        mask_beginnings = np.in1d(mask_beginnings, 1).reshape(mask_beginnings.shape)
        mask_endings = np.flip(np.diff(np.flip(population, axis=1), 1, prepend=0), axis=1)
        mask_endings = np.in1d(mask_endings, 1).reshape(mask_endings.shape)
        beginnings_endings_mask = mask_beginnings + mask_endings

        random_mutations = (np.random.uniform(0, 1, population.shape) < mutation_prob).astype(int)
        valid_mutations = np.zeros(population.shape).astype(int)
        valid_mutations[beginnings_endings_mask] = random_mutations[beginnings_endings_mask]
        return (population + valid_mutations) % 2

    @staticmethod
    def remove_subsequence_mutation(population, mutation_prob):
        subseq_diff = np.diff(population, 1, prepend=0, append=0)
        mask_beginnings = np.in1d(subseq_diff, 1).reshape(subseq_diff.shape)
        end_point_mask = np.in1d(subseq_diff, -1).reshape(subseq_diff.shape)

        random_mutations = (np.random.uniform(0, 1, (population.shape[0], population.shape[1] + 1)) < mutation_prob).astype(int)
        valid_mutations = np.zeros((population.shape[0], population.shape[1] + 1)).astype(int)
        valid_mutations[mask_beginnings] = random_mutations[mask_beginnings]
        valid_mutations[end_point_mask] = -random_mutations[mask_beginnings]

        subseq_diff_mutated = subseq_diff - valid_mutations
        mutated_population = np.cumsum(subseq_diff_mutated, axis=1)
        return mutated_population[:, :-1]

    def mutate(self, sub_population):
        mutated_sub_population = sub_population.copy()
        if self.change_subseq_mutation_prob > 0:
            mutated_sub_population = self.shrink_mutation(mutated_sub_population, self.change_subseq_mutation_prob)
            mutated_sub_population = self.extend_mutation(mutated_sub_population, self.change_subseq_mutation_prob)
        if self.add_subseq_mutation_prob > 0:
            mutated_sub_population = self.add_subsequence_mutation(mutated_sub_population, self.add_subseq_mutation_prob)
        if self.remove_subseq_mutation_prob > 0:
            mutated_sub_population = self.remove_subsequence_mutation(mutated_sub_population, self.remove_subseq_mutation_prob)
        return mutated_sub_population


def multispace_cf(sample, model, target_class=None, dataset=None, nun_example=None,
                    population_size=100, grouped_iter=75, pruning_iter=25,
                    change_subseq_mutation_prob=0.75, add_subseq_mutation_prob=0,
                    final_pruning_mutation_prob=0.75,
                    init_pct=0.2, reinit=True, init_random_mix_ratio=0.5,
                    invalid_penalization=100,
                    autoencoder=None, ae_epochs=40,
                    return_front=False,
                    verbose=False):
    """
    Generate counterfactual explanation using the Multi-SpaCE method.

    This repo's canonical Multi-SpaCE implementation: reproduces the official
    MarioRefoyo/Multi-SpaCE repository's algorithm rather than a speed-adapted approximation
    of it -- a genuine multi-objective genetic algorithm (NSGA-II, non-dominated sorting +
    crowding distance over 4 objectives: validity, sparsity, contiguity, and an
    autoencoder-based plausibility term), run in two stages matching the paper's own
    recommended "final" configuration (experiments/params_cf/multisubspace_final.json):

    - Stage 1 ("grouped"): a single temporal mask shared across all channels is optimized for
      `grouped_iter` generations -- cheap because it searches one mask regardless of channel
      count.
    - Stage 2 ("pruning"): the best front from stage 1 is expanded to a per-channel mask and
      refined for `pruning_iter` generations with a mutation that can remove whole subsequences,
      trimming away changes that turned out to be unnecessary.

    (The official repo also supports a middle "individual channels" stage; the paper's own
    "final" config sets its iteration count to 0, so it is skipped here too.) Validity is
    enforced as a hard constraint via `invalid_penalization`, not a soft objective weight:
    candidates that don't hit `target_class` are penalized enough to always rank behind valid
    ones in the non-dominated sort.

    Because NSGA-II searches a whole Pareto front rather than a single point, this returns the
    *sparsest valid solution* on the final front by default. Pass `return_front=True` to get the
    whole front instead.

    Args:
        sample: Input time series to explain (L, C) or (C, L) or 1D
        model: PyTorch classification model
        target_class: Target class for counterfactual (if None, will be inferred the same way
            multispace_fast does: the label of the nearest globally different-predicted-class
            neighbor in `dataset`)
        dataset: Training dataset for finding the NUN and calibrating the autoencoder (required)
        nun_example: Native Unexplained Neighbor (if None, will be found)
        population_size: Size of the NSGA-II population
        grouped_iter: Generations for the shared-mask stage
        pruning_iter: Generations for the per-channel pruning stage
        change_subseq_mutation_prob: Shrink/extend mutation probability (grouped stage)
        add_subseq_mutation_prob: Add-subsequence mutation probability
        final_pruning_mutation_prob: Remove-subsequence mutation probability (pruning stage)
        init_pct: Initial percentage of activated positions
        reinit: Whether to reinitialize on failure to find a valid solution
        init_random_mix_ratio: Unused (kept for signature parity -- multispace_cf does not use
            feature-importance-guided initialization, matching the paper's own "final" config's
            `init_fi: "none"`)
        invalid_penalization: Penalty pushing invalid candidates off the Pareto front
        autoencoder: Pre-trained plausibility autoencoder (e.g. from a previous multispace_cf
            call's outlier calculator, or train_outlier_autoencoder directly) to avoid retraining
            one from `dataset` on every call. If None, one is trained on-the-fly.
        ae_epochs: Training epochs for the on-the-fly autoencoder (ignored if `autoencoder` given)
        return_front: If True, return (cfs_front, y_cfs_front) for the whole Pareto front instead
            of a single representative counterfactual.
        verbose: Whether to print progress

    Returns:
        Tuple of (counterfactual, prediction_scores), or (cfs_front, y_cfs_front) if
        return_front=True.
    """
    if dataset is None:
        raise ValueError("multispace_cf requires a dataset to find the NUN and calibrate the autoencoder.")
    device = next(model.parameters()).device

    def model_predict(arr):
        # arr expected shape (B, C, L) - PyTorch convention. Some models in this repo apply
        # softmax internally (e.g. SimpleCNN) and some don't (e.g. SimpleCNNMulti, used by the
        # multivariate examples this function targets) -- the fitness function needs genuine
        # class probabilities (see fitness_function_mo's use against invalid_penalization), so
        # softmax is applied here unless the raw output already looks like one.
        raw = detach_to_numpy(model(numpy_to_torch(arr, device)))
        row_sums = raw.sum(axis=1)
        looks_like_probs = np.allclose(row_sums, 1.0, atol=1e-3) and (raw >= -1e-6).all()
        if looks_like_probs:
            return raw
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    # Normalize sample and dataset to (C, L) / (N, C, L), same helper multispace_fast uses
    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    C, L = sample_cf.shape

    preds_data = model_predict(time_series_data)
    preds_sample = model_predict(sample_cf.reshape(1, C, L))
    label_data = np.argmax(preds_data, axis=1)
    label_sample = int(np.argmax(preds_sample))

    # Find (or use provided) NUN / target_class -- same "global nearest unlike neighbor" rule
    # multispace_fast uses, with an explicit target_class override multispace_fast doesn't have.
    if nun_example is None:
        candidate_mask = (label_data == target_class) if target_class is not None else (label_data != label_sample)
        if not np.any(candidate_mask):
            if verbose:
                print(f"No candidates found for target_class={target_class}; returning original sample")
            cf_out = _revert_orientation(sample_cf, sample_ori)
            return cf_out, preds_sample.reshape(-1)

        candidates, candidates_labels = time_series_data[candidate_mask], label_data[candidate_mask]
        distances = np.sum((candidates.reshape(len(candidates), -1) - sample_cf.reshape(1, -1)) ** 2, axis=1)
        nun_idx = np.argmin(distances)
        nun_example = candidates[nun_idx]
        if target_class is None:
            target_class = int(candidates_labels[nun_idx])
    else:
        nun_arr = np.asarray(nun_example)
        if nun_arr.ndim == 1:
            nun_arr = nun_arr.reshape(1, -1)
        elif nun_arr.ndim == 2 and nun_arr.shape[0] > nun_arr.shape[1]:
            nun_arr = nun_arr.T
        nun_example = nun_arr
        if target_class is None:
            preds_nun = model_predict(nun_example.reshape(1, C, L))
            target_class = int(np.argmax(preds_nun))

    # Work in channel-last (L, C) internally, matching the official repo's mask convention
    x_orig_lc = sample_cf.T
    nun_lc = nun_example.T

    def predict_lc(batch_lc):
        # (B, L, C) -> (B, C, L) for the model
        return model_predict(np.transpose(batch_lc, (0, 2, 1)))

    # Autoencoder-based outlier calculator (matches the official AEOutlierCalculator)
    calibration_data = np.transpose(time_series_data[:min(300, len(time_series_data))], (0, 2, 1))  # (N,C,L)->(N,L,C)
    if autoencoder is None:
        if verbose:
            print(f"Training plausibility autoencoder on {len(calibration_data)} reference samples...")
        autoencoder = train_outlier_autoencoder(calibration_data, device, epochs=ae_epochs)
    outlier_calculator = AutoencoderOutlierCalculator(autoencoder, calibration_data, device)

    # Stage 1: grouped (shared mask across channels)
    grouped_optimizer = IntegratedPruningNSubsequenceEvolutionaryOptimizer(
        fitness_function_mo, predict_lc,
        population_size, grouped_iter,
        change_subseq_mutation_prob, add_subseq_mutation_prob, 0,
        init_pct, reinit, init_random_mix_ratio,
        invalid_penalization,
        individual_channel_search=False,
    )
    grouped_optimizer.init(x_orig_lc, nun_lc, target_class, outlier_calculator=outlier_calculator)
    grouped_front, fitness_evolution = grouped_optimizer.optimize()

    if grouped_front is None:
        if verbose:
            print("Failed to converge in grouped stage")
        cf_out = _revert_orientation(sample_cf, sample_ori)
        if return_front:
            return cf_out.reshape(1, *cf_out.shape), preds_sample.reshape(1, -1)
        return cf_out, preds_sample.reshape(-1)

    # Hand off to stage 2: expand shared mask to all channels
    init_mask = np.repeat(grouped_front, C, axis=2)

    # Stage 2: pruning (per-channel mask, can remove whole subsequences)
    pruning_optimizer = IntegratedPruningNSubsequenceEvolutionaryOptimizer(
        fitness_function_mo, predict_lc,
        population_size, pruning_iter,
        0, add_subseq_mutation_prob, final_pruning_mutation_prob,
        init_pct, reinit, init_random_mix_ratio,
        invalid_penalization,
        individual_channel_search=True,
    )
    pruning_optimizer.init(x_orig_lc, nun_lc, target_class, init_mask=init_mask, outlier_calculator=outlier_calculator)
    pruning_front, fitness_evolution_pruning = pruning_optimizer.optimize()
    fitness_evolution = fitness_evolution + fitness_evolution_pruning

    if pruning_front is None:
        if verbose:
            print("Failed to converge in pruning stage; falling back to grouped-stage front")
        cfs_lc = grouped_optimizer.get_counterfactuals(x_orig_lc, nun_lc, init_mask)
    else:
        cfs_lc = pruning_optimizer.get_counterfactuals(x_orig_lc, nun_lc, pruning_front)

    y_cfs = predict_lc(cfs_lc)

    if return_front:
        cfs_out = np.stack([_revert_orientation(cf_lc.T, sample_ori) for cf_lc in cfs_lc], axis=0)
        return cfs_out, y_cfs

    diffs = np.abs(cfs_lc - x_orig_lc[np.newaxis]) > 1e-6
    sparsity = 1.0 - diffs.reshape(len(cfs_lc), -1).mean(axis=1)
    valid_mask = np.argmax(y_cfs, axis=1) == target_class
    if not valid_mask.any():
        if verbose:
            print("No valid counterfactual found in final Pareto front; returning original sample")
        cf_out = _revert_orientation(sample_cf, sample_ori)
        return cf_out, preds_sample.reshape(-1)

    candidate_sparsity = np.where(valid_mask, sparsity, -1)
    best_idx = int(np.argmax(candidate_sparsity))

    cf_out = _revert_orientation(cfs_lc[best_idx].T, sample_ori)
    return cf_out, y_cfs[best_idx]
