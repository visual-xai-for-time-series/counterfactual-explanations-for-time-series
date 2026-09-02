"""
Evaluation metrics inspired by Keane et al. (2021).

Keane, M. T., Kenny, E. M., Delaney, E., & Smyth, B. (2021).
If only we had better counterfactual explanations: Five key deficits to
rectify in the evaluation of counterfactual XAI techniques.
In IJCAI (Vol. 21, pp. 4466-4474).

That paper is a critique/roadmap, not a metrics paper: it does not itself
give equations, and its own §8.2 benchmarking proposal lists four metrics --
Proximity (report both L1- and L2-norm), Sparsity (a frequency profile over
1-5 feature-differences, not one averaged score), Coverage (Eq. 1-2,
XP_Coverage), and Relative Distance (mean CF distance / mean NUN distance).
This module covers all four, plus a `validity` metric common throughout the
wider CF-XAI literature that the paper itself does not formalize:

    keane_validity           - fraction of CFs reaching the target class
    keane_proximity          - §8.2 Proximity, L2 side
    keane_proximity_l1       - §8.2 Proximity, L1 side
    keane_compactness        - single-score fraction-unchanged (this repo's
                                own averaged complement to sparsity; the
                                paper's own "Sparsity" is keane_sparsity_profile)
    keane_sparsity_profile   - §8.2 Sparsity, 1-5 feature-difference frequency table
    keane_coverage           - §6/§8.2 Coverage (Eq. 1-2), operationalized
                                per (original, counterfactual) pair
    keane_relative_distance  - §8.2 Relative Distance, vs. a NUN
    keane_evaluate_metrics   - convenience wrapper computing all of the above

References:
- Verma, S., Dickerson, J., & Hines, K. (2020). Counterfactual explanations
  for machine learning: A review. arXiv preprint arXiv:2010.10596.
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning
  classifiers through diverse counterfactual explanations. In Proceedings of
  the 2020 Conference on Fairness, Accountability, and Transparency
  (FAT* '20) (pp. 607-617).
- Pawelczyk, M., Broelemann, K., & Kasneci, G. (2020). Learning
  model-agnostic counterfactual explanations for tabular data. In
  Proceedings of The Web Conference 2020 (WWW '20) (pp. 3126-3132).
- Delaney, E., Greene, D., & Keane, M. T. (2021). Instance-based
  counterfactual explanations for time series classification. In
  International Conference on Case-Based Reasoning (ICCBR 2021),
  LNCS vol. 12877 (pp. 32-47). Springer.
- Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020). Locally and
  globally explainable time series tweaking. Knowledge and Information
  Systems, 62(5), 1671-1700.
"""

import numpy as np
from typing import Callable, List, Union


def _to_sample_list(ts_list: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
    """
    Normalize a batch of time series into a list of per-sample arrays.

    Mirrors the shape convention used throughout this module: a 3D array of
    shape (n, time_steps, features) is split into n per-sample arrays along
    axis 0; a 2D array is treated as a single sample (as are Python lists,
    returned unchanged).
    """
    if isinstance(ts_list, np.ndarray):
        if ts_list.ndim == 2:
            return [ts_list]
        elif ts_list.ndim == 3:
            return [ts_list[i] for i in range(len(ts_list))]
    return list(ts_list)


def _to_numpy(x) -> np.ndarray:
    """Convert a torch tensor (or other array-like) to a plain numpy array."""
    if hasattr(x, 'numpy'):
        return x.detach().numpy() if hasattr(x, 'detach') else x.numpy()
    return np.asarray(x)


def _count_differences(orig: np.ndarray, cf: np.ndarray, tolerance: float) -> int:
    """Count entries where |orig - cf| exceeds `tolerance` -- the complement of keane_compactness's per-sample count."""
    orig = _to_numpy(orig)
    cf = _to_numpy(cf)
    return int(np.sum(np.abs(orig - cf) > tolerance))


def keane_validity(original_ts_list: Union[np.ndarray, List[np.ndarray]],
            counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]],
            model: Callable,
            target_classes: Union[int, List[int]] = None) -> float:
    """
    Measures whether the generated counterfactuals lead to valid transformations 
    to the desired target class.
    
    Validity reports the fraction of counterfactuals predicted as the opposite 
    class (i.e., have crossed the decision boundary).
    
    Formula:
        Validity = (1/n) * Σ I(f(x'_i) = y_target)
    
    where:
        - f is the model prediction
        - x'_i is one counterfactual sample
        - y_target is the target class
        - n is the count of samples in the dataset
        - I is the indicator function (1 if true, 0 if false)
    
    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
        model: Trained model for prediction
        target_classes: Target class(es) for the counterfactuals. 
                       If int, same target for all samples.
                       If List, different target per sample.
                       If None, validity checks if prediction changed from original.
    
    Returns:
        Validity score: fraction of valid counterfactuals (0.0 to 1.0).
        Higher is better.
    
    Examples:
        >>> # Single target class for all samples
        >>> validity_score = keane_validity(originals, counterfactuals, model, target_classes=1)

        >>> # Different target class per sample
        >>> validity_score = keane_validity(originals, counterfactuals, model,
        ...                          target_classes=[1, 0, 1, 0])

        >>> # Just check if prediction changed (any class)
        >>> validity_score = keane_validity(originals, counterfactuals, model)
    """
    # Convert to list if needed
    if isinstance(original_ts_list, np.ndarray):
        if original_ts_list.ndim == 2:
            original_ts_list = [original_ts_list]
        elif original_ts_list.ndim == 3:
            original_ts_list = [original_ts_list[i] for i in range(len(original_ts_list))]
    
    if isinstance(counterfactual_ts_list, np.ndarray):
        if counterfactual_ts_list.ndim == 2:
            counterfactual_ts_list = [counterfactual_ts_list]
        elif counterfactual_ts_list.ndim == 3:
            counterfactual_ts_list = [counterfactual_ts_list[i] for i in range(len(counterfactual_ts_list))]
    
    n = len(counterfactual_ts_list)
    
    if n == 0:
        return 0.0
    
    # Handle target classes
    if target_classes is None:
        # Check if prediction changed from original
        target_list = [None] * n
    elif isinstance(target_classes, int):
        # Same target for all samples
        target_list = [target_classes] * n
    else:
        # Different target per sample
        target_list = target_classes
    
    if len(target_list) != n:
        raise ValueError(f"Number of target classes ({len(target_list)}) must match "
                        f"number of counterfactuals ({n})")
    
    valid_count = 0
    
    for i, (cf, target) in enumerate(zip(counterfactual_ts_list, target_list)):
        # Get model prediction for counterfactual
        cf_pred = model(cf)
        
        # Convert to numpy if needed
        if hasattr(cf_pred, 'numpy'):
            cf_pred = cf_pred.detach().numpy() if hasattr(cf_pred, 'detach') else cf_pred.numpy()
        
        # Get predicted class
        if isinstance(cf_pred, np.ndarray) and cf_pred.ndim > 0 and cf_pred.size > 1:
            cf_class = np.argmax(cf_pred)
        else:
            cf_class = int(cf_pred)
        
        # Check validity
        if target is not None:
            # Check if prediction matches target class
            if cf_class == target:
                valid_count += 1
        else:
            # Check if prediction changed from original
            if i < len(original_ts_list):
                orig_pred = model(original_ts_list[i])
                if hasattr(orig_pred, 'numpy'):
                    orig_pred = orig_pred.detach().numpy() if hasattr(orig_pred, 'detach') else orig_pred.numpy()
                
                if isinstance(orig_pred, np.ndarray) and orig_pred.ndim > 0 and orig_pred.size > 1:
                    orig_class = np.argmax(orig_pred)
                else:
                    orig_class = int(orig_pred)
                
                if cf_class != orig_class:
                    valid_count += 1
    
    return float(valid_count / n)


def keane_proximity(original_ts_list: Union[np.ndarray, List[np.ndarray]],
                    counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]]) -> float:
    """
    Measures the feature-wise distance between the generated counterfactuals 
    and the corresponding original samples.
    
    Proximity is defined as the average Euclidean distance between the 
    transformed and the original time series.
    
    Formula:
        Proximity = (1/n) * Σ ||x_i - x'_i||_2
    
    where:
        - x_i is the original time series
        - x'_i is the generated counterfactual
        - n is the count of samples
        - ||·||_2 is the Euclidean (L2) norm
    
    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
    
    Returns:
        Proximity score: average Euclidean distance.
        Lower is better.
    
    Examples:
        >>> proximity_score = keane_proximity(originals, counterfactuals)
        >>> print(f"Average distance: {proximity_score:.4f}")
    """
    # Convert to list if needed
    if isinstance(original_ts_list, np.ndarray):
        if original_ts_list.ndim == 2:
            original_ts_list = [original_ts_list]
        elif original_ts_list.ndim == 3:
            original_ts_list = [original_ts_list[i] for i in range(len(original_ts_list))]
    
    if isinstance(counterfactual_ts_list, np.ndarray):
        if counterfactual_ts_list.ndim == 2:
            counterfactual_ts_list = [counterfactual_ts_list]
        elif counterfactual_ts_list.ndim == 3:
            counterfactual_ts_list = [counterfactual_ts_list[i] for i in range(len(counterfactual_ts_list))]
    
    n = len(counterfactual_ts_list)
    
    if n == 0:
        return 0.0
    
    if len(original_ts_list) != n:
        raise ValueError(f"Number of originals ({len(original_ts_list)}) must match "
                        f"number of counterfactuals ({n})")
    
    total_distance = 0.0
    
    for orig, cf in zip(original_ts_list, counterfactual_ts_list):
        # Convert to numpy arrays if needed
        if hasattr(orig, 'numpy'):
            orig = orig.detach().numpy() if hasattr(orig, 'detach') else orig.numpy()
        if hasattr(cf, 'numpy'):
            cf = cf.detach().numpy() if hasattr(cf, 'detach') else cf.numpy()
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(orig - cf)
        total_distance += distance

    return float(total_distance / n)


def keane_proximity_l1(original_ts_list: Union[np.ndarray, List[np.ndarray]],
                       counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]]) -> float:
    """
    Manhattan (L1) counterpart to keane_proximity.

    Keane et al. (2021, §8.2 "Proximity") explicitly recommend reporting
    both the L1-norm and the L2-norm side by side -- "the L1-norm probably
    has to be used, alongside the L2-norm, until it is clear which is the
    more psychologically-valid measure" -- rather than committing to one.
    keane_proximity() covers the L2 side; this covers the L1 side.

    Formula:
        Proximity_L1 = (1/n) * Σ ||x_i - x'_i||_1

    where:
        - x_i is the original time series
        - x'_i is the generated counterfactual
        - n is the count of samples
        - ||·||_1 is the Manhattan (L1) norm

    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)

    Returns:
        Proximity score: average Manhattan distance.
        Lower is better.

    Examples:
        >>> l1_score = keane_proximity_l1(originals, counterfactuals)
        >>> l2_score = keane_proximity(originals, counterfactuals)
        >>> print(f"L1: {l1_score:.4f}, L2: {l2_score:.4f}")
    """
    original_ts_list = _to_sample_list(original_ts_list)
    counterfactual_ts_list = _to_sample_list(counterfactual_ts_list)

    n = len(counterfactual_ts_list)

    if n == 0:
        return 0.0

    if len(original_ts_list) != n:
        raise ValueError(f"Number of originals ({len(original_ts_list)}) must match "
                        f"number of counterfactuals ({n})")

    total_distance = 0.0

    for orig, cf in zip(original_ts_list, counterfactual_ts_list):
        orig = _to_numpy(orig)
        cf = _to_numpy(cf)
        total_distance += np.sum(np.abs(orig - cf))

    return float(total_distance / n)


def keane_compactness(original_ts_list: Union[np.ndarray, List[np.ndarray]],
               counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]],
               tolerance: float = 0.01) -> float:
    """
    Measures the fraction of time series steps that remain unchanged in the 
    generated counterfactuals compared to the original samples.
    
    Compactness (also reported as sparsity in literature) captures the amount 
    of information that remains unchanged from the original time series.
    
    Formula:
        Compactness = (1/n) * Σ (Σ_t I(|x_i,t - x'_i,t| ≤ tol)) / T
    
    where:
        - x_i,t is the value at time step t in original time series i
        - x'_i,t is the value at time step t in counterfactual i
        - tol is the tolerance parameter for considering values unchanged
        - T is the total number of time steps
        - n is the count of samples
        - I is the indicator function (1 if true, 0 if false)
    
    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
        tolerance: Tolerance parameter for considering a value unchanged (default: 0.01)
    
    Returns:
        Compactness score: fraction of unchanged values (0.0 to 1.0).
        Higher is better.
    
    Examples:
        >>> compactness_score = keane_compactness(originals, counterfactuals, tolerance=0.01)
        >>> print(f"Fraction unchanged: {compactness_score:.2%}")
    """
    # Convert to list if needed
    if isinstance(original_ts_list, np.ndarray):
        if original_ts_list.ndim == 2:
            original_ts_list = [original_ts_list]
        elif original_ts_list.ndim == 3:
            original_ts_list = [original_ts_list[i] for i in range(len(original_ts_list))]
    
    if isinstance(counterfactual_ts_list, np.ndarray):
        if counterfactual_ts_list.ndim == 2:
            counterfactual_ts_list = [counterfactual_ts_list]
        elif counterfactual_ts_list.ndim == 3:
            counterfactual_ts_list = [counterfactual_ts_list[i] for i in range(len(counterfactual_ts_list))]
    
    n = len(counterfactual_ts_list)
    
    if n == 0:
        return 0.0
    
    if len(original_ts_list) != n:
        raise ValueError(f"Number of originals ({len(original_ts_list)}) must match "
                        f"number of counterfactuals ({n})")
    
    total_unchanged_fraction = 0.0
    
    for orig, cf in zip(original_ts_list, counterfactual_ts_list):
        # Convert to numpy arrays if needed
        if hasattr(orig, 'numpy'):
            orig = orig.detach().numpy() if hasattr(orig, 'detach') else orig.numpy()
        if hasattr(cf, 'numpy'):
            cf = cf.detach().numpy() if hasattr(cf, 'detach') else cf.numpy()
        
        # Calculate absolute differences
        differences = np.abs(orig - cf)
        
        # Count unchanged values (within tolerance)
        unchanged_mask = differences <= tolerance
        unchanged_count = np.sum(unchanged_mask)
        total_count = orig.size
        
        # Calculate fraction for this sample
        fraction_unchanged = unchanged_count / total_count
        total_unchanged_fraction += fraction_unchanged

    return float(total_unchanged_fraction / n)


def keane_sparsity_profile(original_ts_list: Union[np.ndarray, List[np.ndarray]],
                           counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]],
                           tolerance: float = 0.01, max_bin: int = 5) -> dict:
    """
    Breaks counterfactuals out by their number of feature/time-point
    differences, rather than collapsing sparsity into one averaged score.

    Keane et al. (2021, §5 "Deficit #3: The Shape of Sparsity" and §8.2
    "Sparcity") argue that a single mean hides too much: "distance metrics
    report averages, tests need to be broken out by sparcity levels showing
    the frequency of CFs generated; the range from 1-5 feature-differences
    seems critical, as it may well discriminate models (e.g., a model
    producing most of its CFs with >4 differences may be questionable)."
    This reports exactly that frequency table, using the same tolerance-based
    "changed" test as keane_compactness()'s "unchanged" test.

    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
        tolerance: Tolerance parameter for considering a value changed (default: 0.01),
                   the same convention as keane_compactness()'s `tolerance`.
        max_bin: Highest individual bin reported (default: 5, matching the
                 paper's "1-5 feature-differences" range); difference counts
                 above this are pooled into a single f">{max_bin}" bin.

    Returns:
        Dictionary with:
            - 'n_differences': list of per-sample difference counts (int)
            - 'counts': {0: n, 1: n, ..., max_bin: n, f'>{max_bin}': n} --
                        raw counterfactual counts per sparsity level
            - 'fractions': same keys as 'counts', normalized by n (0.0 to 1.0)
            - 'mean_differences': average number of differences per counterfactual
            - 'median_differences': median number of differences per counterfactual

    Examples:
        >>> profile = keane_sparsity_profile(originals, counterfactuals, max_bin=5)
        >>> print(profile['fractions'])
        >>> print(f"Mean differences: {profile['mean_differences']:.2f}")
    """
    original_ts_list = _to_sample_list(original_ts_list)
    counterfactual_ts_list = _to_sample_list(counterfactual_ts_list)

    n = len(counterfactual_ts_list)
    bin_labels = list(range(0, max_bin + 1)) + [f'>{max_bin}']

    if n == 0:
        return {
            'n_differences': [],
            'counts': {label: 0 for label in bin_labels},
            'fractions': {label: 0.0 for label in bin_labels},
            'mean_differences': 0.0,
            'median_differences': 0.0,
        }

    if len(original_ts_list) != n:
        raise ValueError(f"Number of originals ({len(original_ts_list)}) must match "
                        f"number of counterfactuals ({n})")

    n_differences = [
        _count_differences(orig, cf, tolerance)
        for orig, cf in zip(original_ts_list, counterfactual_ts_list)
    ]

    counts = {label: 0 for label in bin_labels}
    for d in n_differences:
        label = d if d <= max_bin else f'>{max_bin}'
        counts[label] += 1

    fractions = {label: count / n for label, count in counts.items()}

    return {
        'n_differences': n_differences,
        'counts': counts,
        'fractions': fractions,
        'mean_differences': float(np.mean(n_differences)),
        'median_differences': float(np.median(n_differences)),
    }


def keane_coverage(original_ts_list: Union[np.ndarray, List[np.ndarray]],
                   counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]],
                   tolerance: float = 0.01, max_differences: int = 3,
                   explains_fn: Callable = None) -> float:
    """
    Fraction of counterfactuals that are "explanatorily competent" -- i.e.
    good enough to plausibly explain their original instance.

    Keane et al. (2021, §6 "Deficit #4: Covering Coverage" and §8.2
    "Coverage") define this via a dataset-internal coverage set:

        XP_Coverage_Set(C) = {c' ∈ C | ∃c ∈ C-{c'} & explains(c, c')}   (Eq. 1)
        XP_Coverage(C) = |XP_Coverage_Set(C)| / |C|                     (Eq. 2)

    where `explains(c, c')` is a psychologically-acceptable-CF predicate the
    paper deliberately leaves open, noting that Keane and Smyth [2020]
    "adopted the simple expedient of defining it as any CF with ≤2
    feature-differences" and that §8.2 suggests "even a rough one any CF
    with ≤3 feature-differences" as a usable default.

    This operationalizes that for the (original, generated counterfactual)
    *pairs* this repository evaluates -- rather than Eq. 1's search over an
    existing dataset of native/NUN-based counterfactuals -- as the fraction
    of pairs satisfying `explains(original_i, cf_i)`, using the paper's own
    default acceptability rule (number of tolerance-based differences ≤
    max_differences) unless a custom `explains_fn` is supplied.

    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
        tolerance: Tolerance parameter for considering a value changed (default: 0.01),
                   used by the default acceptability rule (ignored if `explains_fn` is given).
        max_differences: Default acceptability threshold -- a counterfactual
                        "explains" its original if it has at most this many
                        tolerance-based differences (default: 3, per §8.2;
                        pass 2 to match Keane & Smyth [2020]'s own choice).
        explains_fn: Optional custom explains(original, counterfactual) -> bool
                     predicate overriding the default ≤ max_differences rule.

    Returns:
        Coverage score: fraction of counterfactuals judged "explanatory" (0.0 to 1.0).
        Higher is better.

    Examples:
        >>> coverage_score = keane_coverage(originals, counterfactuals, max_differences=3)
        >>> print(f"Coverage: {coverage_score:.2%}")
    """
    original_ts_list = _to_sample_list(original_ts_list)
    counterfactual_ts_list = _to_sample_list(counterfactual_ts_list)

    n = len(counterfactual_ts_list)

    if n == 0:
        return 0.0

    if len(original_ts_list) != n:
        raise ValueError(f"Number of originals ({len(original_ts_list)}) must match "
                        f"number of counterfactuals ({n})")

    if explains_fn is None:
        explains_fn = lambda orig, cf: _count_differences(orig, cf, tolerance) <= max_differences

    explained_count = sum(
        1 for orig, cf in zip(original_ts_list, counterfactual_ts_list)
        if explains_fn(orig, cf)
    )

    return float(explained_count / n)


def keane_relative_distance(original_ts_list: Union[np.ndarray, List[np.ndarray]],
                            counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]],
                            native_cf_ts_list: Union[np.ndarray, List[np.ndarray]] = None,
                            reference_ts_list: Union[np.ndarray, List[np.ndarray]] = None,
                            reference_labels: Union[np.ndarray, List] = None,
                            target_classes: Union[int, List[int]] = None,
                            distance_fn: Callable = None) -> float:
    """
    Ratio of the mean generated-counterfactual distance to the mean "native
    counterfactual" (Nearest Unlike Neighbour, NUN) distance.

    Keane et al. (2021, §8.2 "Relative Distance") propose "a relative-
    distance measure ... comparing the mean distance of CF-pairs (between
    the test and CF instance) over the mean distance of 'native
    counterfactuals' (NUNs)", to test the instance-guided insight that a
    method's generated CFs should be closer to the original than a 'natural'
    same-dataset counterfactual already is.

    Formula:
        RelativeDistance = mean_i d(x_i, x'_i) / mean_i d(x_i, NUN_i)

    A score < 1 means the generated counterfactuals are, on average, closer
    to their originals than the nearest same-dataset instance of the target
    class already is; a score >= 1 means the method does no better than
    simply picking the NUN itself.

    NUNs can be supplied directly (`native_cf_ts_list`, one per original), or
    computed by nearest-neighbour search against a reference dataset
    (`reference_ts_list` + `reference_labels`, matched against
    `target_classes` per original).

    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
        native_cf_ts_list: Optional precomputed NUN per original, same
            length/shape convention as counterfactual_ts_list. Takes
            precedence over reference_ts_list if both are given.
        reference_ts_list: Optional reference dataset (e.g. training data) to
            search for NUNs in, used only if native_cf_ts_list is None.
        reference_labels: Class label for each entry of reference_ts_list.
            Required together with reference_ts_list.
        target_classes: Target class(es) each NUN must match against
            reference_labels. If int, same target for every sample; if a
            list, one target per sample; if None, the nearest reference
            sample of any class is used. Required together with reference_ts_list.
        distance_fn: Optional distance(a, b) -> float callable (default: L2 norm).

    Returns:
        Relative distance ratio (>= 0). Lower is better (closer to the
        instance-guided ideal); NaN if the mean NUN distance is 0.

    Examples:
        >>> # With precomputed NUNs
        >>> rel = keane_relative_distance(originals, counterfactuals, native_cf_ts_list=nuns)

        >>> # Computed against a reference/training dataset
        >>> rel = keane_relative_distance(originals, counterfactuals,
        ...                                reference_ts_list=X_train, reference_labels=y_train,
        ...                                target_classes=1)
    """
    original_ts_list = _to_sample_list(original_ts_list)
    counterfactual_ts_list = _to_sample_list(counterfactual_ts_list)

    n = len(counterfactual_ts_list)

    if n == 0:
        return 0.0

    if len(original_ts_list) != n:
        raise ValueError(f"Number of originals ({len(original_ts_list)}) must match "
                        f"number of counterfactuals ({n})")

    if distance_fn is None:
        distance_fn = lambda a, b: float(np.linalg.norm(_to_numpy(a) - _to_numpy(b)))

    cf_distances = [distance_fn(orig, cf) for orig, cf in zip(original_ts_list, counterfactual_ts_list)]

    if native_cf_ts_list is not None:
        native_cf_ts_list = _to_sample_list(native_cf_ts_list)
        if len(native_cf_ts_list) != n:
            raise ValueError(f"Number of native counterfactuals ({len(native_cf_ts_list)}) must "
                            f"match number of counterfactuals ({n})")
        nun_distances = [distance_fn(orig, nun) for orig, nun in zip(original_ts_list, native_cf_ts_list)]
    elif reference_ts_list is not None:
        if reference_labels is None:
            raise ValueError("reference_labels is required together with reference_ts_list")

        reference_ts_list = _to_sample_list(reference_ts_list)
        reference_labels = np.asarray(reference_labels)

        if isinstance(target_classes, int) or target_classes is None:
            target_list = [target_classes] * n
        else:
            target_list = list(target_classes)

        nun_distances = []
        for orig, target in zip(original_ts_list, target_list):
            candidate_idx = (np.where(reference_labels == target)[0]
                            if target is not None else np.arange(len(reference_ts_list)))
            if len(candidate_idx) == 0:
                raise ValueError(f"No reference samples found for target class {target}")

            candidate_distances = [distance_fn(orig, reference_ts_list[i]) for i in candidate_idx]
            nun_distances.append(min(candidate_distances))
    else:
        raise ValueError("Either native_cf_ts_list or (reference_ts_list + reference_labels) "
                        "must be provided to determine the NUNs")

    mean_cf_distance = float(np.mean(cf_distances))
    mean_nun_distance = float(np.mean(nun_distances))

    if mean_nun_distance == 0:
        return float('nan')

    return mean_cf_distance / mean_nun_distance


def keane_evaluate_metrics(original_ts_list: Union[np.ndarray, List[np.ndarray]],
                          counterfactual_ts_list: Union[np.ndarray, List[np.ndarray]],
                          model: Callable,
                          target_classes: Union[int, List[int]] = None,
                          tolerance: float = 0.01,
                          sparsity_max_bin: int = 5,
                          coverage_max_differences: int = 3,
                          native_cf_ts_list: Union[np.ndarray, List[np.ndarray]] = None,
                          reference_ts_list: Union[np.ndarray, List[np.ndarray]] = None,
                          reference_labels: Union[np.ndarray, List] = None) -> dict:
    """
    Evaluate all Keane et al. (2021)-inspired metrics at once: the three
    original per-pair scores (validity, proximity, compactness) plus the
    paper's full §8.2 benchmark suite (L1 proximity, a sparsity frequency
    profile, coverage and -- when NUN data is supplied -- relative distance).

    Args:
        original_ts_list: List of original time series or array of shape (n, time_steps, features)
        counterfactual_ts_list: List of generated counterfactuals or array of shape (n, time_steps, features)
        model: Trained model for prediction
        target_classes: Target class(es) for the validity metric (and, when
            reference_ts_list is used, for NUN lookup)
        tolerance: Tolerance parameter shared by compactness, the sparsity
            profile, and the default coverage rule (default: 0.01)
        sparsity_max_bin: Highest bin reported by the sparsity profile (default: 5)
        coverage_max_differences: Acceptability threshold for coverage (default: 3)
        native_cf_ts_list: Optional precomputed NUNs; enables 'relative_distance'
        reference_ts_list: Optional reference dataset; enables 'relative_distance'
            (NUNs are searched for in it) if native_cf_ts_list is not given
        reference_labels: Class labels for reference_ts_list, required alongside it

    Returns:
        Dictionary containing:
            - 'validity': fraction of valid counterfactuals (higher is better)
            - 'proximity': average Euclidean (L2) distance (lower is better)
            - 'proximity_l1': average Manhattan (L1) distance (lower is better)
            - 'compactness': fraction of unchanged values (higher is better)
            - 'sparsity_profile': per-sparsity-level counts/fractions (see keane_sparsity_profile)
            - 'coverage': fraction of "explanatorily competent" counterfactuals (higher is better)
            - 'relative_distance': mean CF distance / mean NUN distance (lower is better) --
                                   only present if native_cf_ts_list or
                                   reference_ts_list+reference_labels is given

    Examples:
        >>> results = keane_evaluate_metrics(originals, counterfactuals, model,
        ...                                  target_classes=1, tolerance=0.01)
        >>> print(f"Validity: {results['validity']:.2%}")
        >>> print(f"Proximity (L2/L1): {results['proximity']:.4f} / {results['proximity_l1']:.4f}")
        >>> print(f"Compactness: {results['compactness']:.2%}")
        >>> print(f"Coverage: {results['coverage']:.2%}")
    """
    results = {
        'validity': keane_validity(original_ts_list, counterfactual_ts_list, model, target_classes),
        'proximity': keane_proximity(original_ts_list, counterfactual_ts_list),
        'proximity_l1': keane_proximity_l1(original_ts_list, counterfactual_ts_list),
        'compactness': keane_compactness(original_ts_list, counterfactual_ts_list, tolerance),
        'sparsity_profile': keane_sparsity_profile(original_ts_list, counterfactual_ts_list,
                                                   tolerance=tolerance, max_bin=sparsity_max_bin),
        'coverage': keane_coverage(original_ts_list, counterfactual_ts_list,
                                   tolerance=tolerance, max_differences=coverage_max_differences),
    }

    if native_cf_ts_list is not None or reference_ts_list is not None:
        results['relative_distance'] = keane_relative_distance(
            original_ts_list, counterfactual_ts_list,
            native_cf_ts_list=native_cf_ts_list,
            reference_ts_list=reference_ts_list,
            reference_labels=reference_labels,
            target_classes=target_classes,
        )

    return results


__all__ = [
    'keane_validity',
    'keane_proximity',
    'keane_proximity_l1',
    'keane_compactness',
    'keane_sparsity_profile',
    'keane_coverage',
    'keane_relative_distance',
    'keane_evaluate_metrics',
]
