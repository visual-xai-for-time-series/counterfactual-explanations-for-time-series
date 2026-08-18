from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors

from cfts.cf__abstract.abstract import (
    batched_predict,
    detach_to_numpy,
    numpy_to_torch,
    ensure_ncl,
    revert_orientation,
    subsample_dataset,
)


####
# CoMTE: Counterfactual Explanations for Multivariate Time Series
#
# Paper: Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2021).
#        "Counterfactual Explanations for Multivariate Time Series."
#        2021 International Conference on Applied Artificial Intelligence (ICAPAI)
#        arXiv:2008.10781
#
# Original repo: https://github.com/peaclab/CoMTE  (explainers.py)
#
# `comte_cf` (below) is a faithful reimplementation of the algorithm the paper
# and official repo actually describe: swap whole channels from a real,
# correctly-classified "distractor" example into the query, searching
# greedily over which channels to swap. It does NOT touch gradients anywhere.
#
# `comte_cf_gradient` / `comte_ts_cf_gradient` / `comte_cf_advanced_gradient`
# (further below) run Adam gradient descent directly on the continuous
# waveform against a cross-entropy + L2/L1 objective. They used to be named
# `comte_cf` / `comte_ts_cf` / `comte_cf_advanced` in this module, but that
# does not match the paper's mechanism (no real distractor, no discrete
# channel search) — they are kept under the `_gradient` suffix for backward
# compatibility with existing example scripts, not because they implement
# CoMTE. See cfts/cf_comte/comte_forda_comparison.ipynb for a worked
# comparison of all of these against the vendored official code.
#
# What `comte_cf` reimplements, mapped to explainers.py's own classes:
#   - BaseExplanation.construct_per_class_trees / _get_distractors:
#     restrict candidates to dataset samples whose *true* label and *predicted*
#     label both equal the target class ("true positives"), then k-NN search
#     (flattened Euclidean distance) for the `num_distractors` closest ones.
#   - BruteForceSearch.explain / _find_best: greedy search that, at each step,
#     tries swapping every remaining differing unit singly and commits whichever
#     one most increases the target-class probability, stopping as soon as the
#     prediction flips (this mirrors the library's own default `dont_stop=False`
#     behaviour — there is no equivalent parameter here since the loop always
#     needs it).
#
# What is intentionally NOT reimplemented:
#   - OptimizedSearch's discrete optimization over the swap mask (mlrose's
#     random-hill-climb, 5 restarts / 1000 attempts by default). That requires
#     an extra heavyweight dependency, and in the FordA comparison notebook it
#     produced counterfactuals of essentially the same quality as the plain
#     greedy BruteForceSearch at ~80x the runtime — BruteForceSearch is also
#     already OptimizedSearch's own fallback in the official code whenever the
#     discrete optimizer finds nothing. `comte_cf` sticks to the cheap,
#     dependency-light greedy variant.
#
# Univariate adaptation: the official swap unit is a whole *channel*, which
# only makes sense for genuinely multivariate data (its own examples are
# NATOPS — 24 channels — and an HPC telemetry dataset). For univariate input
# (C == 1) this reimplementation instead splits the series into `n_segments`
# equal contiguous chunks and swaps whole chunks — the identical adaptation
# used in cfts/cf_comte/comte_forda_comparison.ipynb — since a single channel
# gives the channel-swap mechanism nothing to search over otherwise.
####


def _label(y) -> int:
    """Collapse a one-hot vector or scalar label to an int class index."""
    arr = np.asarray(y)
    return int(np.argmax(arr)) if arr.ndim > 0 and arr.size > 1 else int(arr)


def _predict_raw(model: torch.nn.Module, x_cl: np.ndarray, device: torch.device) -> np.ndarray:
    """(C, L) -> raw model output (num_classes,), whatever activation the model itself uses."""
    with torch.no_grad():
        out = model(numpy_to_torch(x_cl.reshape(1, *x_cl.shape), device))
    return detach_to_numpy(out).reshape(-1)


def _predict_probs(model: torch.nn.Module, x_cl: np.ndarray, device: torch.device) -> np.ndarray:
    """(C, L) -> softmax probabilities (num_classes,), used only for internal comparisons

    (the official repo always compares `predict_proba` outputs when deciding which
    channel to swap next; softmaxing here guarantees valid probabilities to compare
    regardless of whether the caller's model already ends in its own softmax)."""
    with torch.no_grad():
        out = model(numpy_to_torch(x_cl.reshape(1, *x_cl.shape), device))
        probs = torch.softmax(out, dim=-1)
    return detach_to_numpy(probs).reshape(-1)


def _build_unit_masks(C: int, L: int, n_segments: int) -> List[np.ndarray]:
    """Return a list of boolean (C, L) masks, one per swap "unit".

    C > 1 (genuinely multivariate): one unit per channel — the official
    algorithm's actual swap granularity.
    C == 1 (univariate): split into `n_segments` equal contiguous chunks and
    swap whole chunks (see module docstring).
    """
    if C > 1:
        masks = []
        for c in range(C):
            m = np.zeros((C, L), dtype=bool)
            m[c, :] = True
            masks.append(m)
        return masks

    n_segments = max(1, min(n_segments, L))
    edges = np.linspace(0, L, n_segments + 1).astype(int)
    masks = []
    for i in range(n_segments):
        m = np.zeros((C, L), dtype=bool)
        m[:, edges[i]:edges[i + 1]] = True
        masks.append(m)
    return masks


def _greedy_swap_search(
    sample_cl: np.ndarray,
    distractor_cl: np.ndarray,
    unit_masks: List[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    target_class: int,
    max_features: Optional[int],
) -> Tuple[List[int], np.ndarray, np.ndarray, float, bool]:
    """Greedy channel/segment-swap search for one distractor (BruteForceSearch._find_best/.explain).

    At each step, tries swapping every not-yet-swapped unit that still differs
    from the distractor, keeps whichever single swap gives the highest
    target-class probability, and commits it only if that is an improvement
    over the current probability. Stops as soon as the prediction flips to
    `target_class`, or when no remaining swap helps.

    Returns
    -------
    explanation : list[int]
        Indices of `unit_masks` that were swapped in the best attempt found.
    cf : np.ndarray, shape (C, L)
    scores : np.ndarray, shape (num_classes,)
        Raw model output for `cf`.
    prob : float
        Softmax probability assigned to `target_class` for `cf`.
    success : bool
        Whether `cf`'s predicted class equals `target_class`.
    """
    n_units = len(unit_masks)
    if max_features is None:
        max_features = n_units

    remaining = [
        i for i in range(n_units)
        if np.any(sample_cl[unit_masks[i]] != distractor_cl[unit_masks[i]])
    ]

    modified = sample_cl.copy()
    explanation: List[int] = []
    best_explanation: List[int] = []
    best_prob = -1.0

    while True:
        probs = _predict_probs(model, modified, device)
        if int(np.argmax(probs)) == target_class:
            current_prob = float(probs[target_class])
            if current_prob > best_prob:
                best_prob = current_prob
                best_explanation = list(explanation)
            break  # mirrors the library's default dont_stop=False: stop on first success

        if len(explanation) >= max_features:
            break

        best_unit = None
        best_val = float(probs[target_class])  # only accept swaps that improve on this
        for u in remaining:
            if u in explanation:
                continue
            candidate = modified.copy()
            candidate[unit_masks[u]] = distractor_cl[unit_masks[u]]
            cand_probs = _predict_probs(model, candidate, device)
            val = float(cand_probs[target_class])
            if val > best_val:
                best_val = val
                best_unit = u

        if best_unit is None:
            break

        modified[unit_masks[best_unit]] = distractor_cl[unit_masks[best_unit]]
        explanation.append(best_unit)

    if best_explanation:
        cf = sample_cl.copy()
        for u in best_explanation:
            cf[unit_masks[u]] = distractor_cl[unit_masks[u]]
        final_prob = best_prob
        success = True
    else:
        # Never reached the target class with this distractor — return the
        # partial best-effort attempt (all committed swaps) rather than None,
        # so the function always returns a usable counterfactual candidate.
        cf = modified
        final_prob = float(_predict_probs(model, cf, device)[target_class])
        success = False

    scores = _predict_raw(model, cf, device)
    return (best_explanation or explanation), cf, scores, final_prob, success


def comte_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    n_segments: int = 10,
    num_distractors: int = 2,
    max_features: int | None = None,
    max_samples: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a counterfactual using the real CoMTE distractor-swap algorithm.

    Faithfully reimplements the mechanism in `explainers.py` from the official
    repo (https://github.com/peaclab/CoMTE): find real, correctly-classified
    examples of the target class ("distractors") nearest to the query, then
    greedily swap in whichever channels (or, for univariate data, contiguous
    time segments — see module docstring) most increase the target-class
    probability, stopping as soon as the prediction flips. See the module
    docstring for exactly what is and isn't reimplemented relative to the
    original.

    Follows the same signature pattern as every other CF method in this
    repository (`abstract_cf`, `native_guide_uni_cf`, `mascots_cf`, …) so it
    plugs straight into the existing evaluation and example scripts.

    Parameters
    ----------
    sample:
        Query time series. Accepts 1-D `(L,)`, `(C, L)` or `(L, C)` NumPy
        arrays (or anything that converts with `np.asarray`).
    model:
        Trained PyTorch classifier. Must accept input of shape `(B, C, L)`
        and return logits or probabilities of shape `(B, num_classes)`.
    target_class:
        Class index to flip toward. When `None`, uses the least-likely class
        under the original prediction (`argmin` of the model output) — this
        matches the official repo's own default, not the "second-most-likely"
        convention used by some other methods in this repository.
    dataset:
        Sequence of `(x, y)` pairs used as the distractor pool. Required —
        unlike some other methods here, CoMTE's mechanism depends on knowing
        each candidate's *true* label to build "true positive" distractor
        pools (`x` is a time series, `y` a one-hot vector or scalar label).
    n_segments:
        Number of equal contiguous chunks to split univariate (`C == 1`)
        series into for the swap search. Ignored when `C > 1` (real channels
        are used as swap units instead).
    num_distractors:
        Number of nearest same-target-class distractors to try; the best
        result (by achieved target-class probability, successful flips first)
        across all of them is returned.
    max_features:
        Optional cap on how many units (channels/segments) may be swapped in
        a single distractor's search. `None` means no cap (bounded naturally
        by the number of units anyway). Note: the official repo accepts a
        same-named `num_features` parameter on `BruteForceSearch.explain` but
        never actually uses it internally — here it is a real, enforced cap.
    max_samples:
        If set, subsample `dataset` to at most this many items before
        building the distractor pool (for runtime on large datasets).
    seed:
        Present for interface consistency; unused (the search here is fully
        deterministic given the model, sample, and dataset).
    verbose:
        Print progress when `True`.

    Returns
    -------
    counterfactual : np.ndarray
        Counterfactual time series in the **same shape / orientation** as
        `sample`.
    scores : np.ndarray, shape (num_classes,)
        Raw model output for the counterfactual.
    """
    if dataset is None:
        raise ValueError("comte_cf requires a dataset to search for distractors.")

    device = next(model.parameters()).device

    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)

    sample_cl, ts, ori = ensure_ncl(np.asarray(sample, dtype=np.float32), dataset)
    C, L = sample_cl.shape
    labels = np.array([_label(dataset[i][1]) for i in range(len(dataset))])

    scores_orig = _predict_raw(model, sample_cl, device)
    probs_orig = _predict_probs(model, sample_cl, device)
    label_orig = int(np.argmax(probs_orig))

    if target_class is None:
        target_class = int(np.argmin(probs_orig))

    if label_orig == target_class:
        if verbose:
            print(f"[comte_cf] sample already predicted as target_class={target_class}.")
        return revert_orientation(sample_cl, ori), scores_orig

    # --- distractor pool: dataset samples whose true label AND predicted
    #     label both equal target_class ("true positives", matching
    #     construct_per_class_trees in the official repo) ---
    pred_labels = np.argmax(batched_predict(model, ts, device), axis=1)
    true_positive_mask = (labels == target_class) & (pred_labels == target_class)

    if not np.any(true_positive_mask):
        if verbose:
            print(
                f"[comte_cf] no true-positive distractors for target_class={target_class}; "
                "relaxing to predicted-label-only match."
            )
        true_positive_mask = pred_labels == target_class

    if not np.any(true_positive_mask):
        if verbose:
            print(f"[comte_cf] no distractors found at all for target_class={target_class}.")
        return revert_orientation(sample_cl, ori), scores_orig

    candidates = ts[true_positive_mask]  # (M, C, L)

    # --- k-NN over flattened series (mirrors _get_distractors' KDTree query) ---
    k = min(num_distractors, len(candidates))
    neigh = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neigh.fit(candidates.reshape(len(candidates), -1))
    _, idxs = neigh.kneighbors(sample_cl.reshape(1, -1))
    distractors = [candidates[i] for i in idxs[0]]

    unit_masks = _build_unit_masks(C, L, n_segments)

    best_cf = sample_cl.copy()
    best_scores = scores_orig.copy()
    best_prob = float(probs_orig[target_class])
    best_success = False

    for d_idx, distractor_cl in enumerate(distractors):
        explanation, cf, scores, prob, success = _greedy_swap_search(
            sample_cl, distractor_cl, unit_masks, model, device, target_class, max_features
        )

        if verbose:
            print(
                f"[comte_cf] distractor {d_idx + 1}/{len(distractors)}: "
                f"swapped {len(explanation)} unit(s), success={success}, "
                f"p(target)={prob:.4f}"
            )

        if (success and not best_success) or (success == best_success and prob > best_prob):
            best_cf = cf
            best_scores = scores
            best_prob = prob
            best_success = success

    if verbose:
        print(
            f"[comte_cf] done — original={label_orig} target={target_class} "
            f"final={int(np.argmax(best_scores))} success={best_success} p(target)={best_prob:.4f}"
        )

    return revert_orientation(best_cf, ori), best_scores


####
# Legacy gradient-based variants (originally named comte_cf / comte_ts_cf /
# comte_cf_advanced in this module).
#
# These optimize the continuous waveform directly against a differentiable
# objective via Adam — they never touch a real distractor or discrete channel
# search, so they do NOT implement the paper's mechanism (see module
# docstring above). Kept under the `_gradient` suffix for backward
# compatibility with existing example scripts.
####


def comte_cf_gradient(
    sample: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
    dataset=None,
    lambda_reg: float = 0.01,
    lambda_sparse: float = 0.001,
    learning_rate: float = 0.1,
    max_iterations: int = 3000,
    tolerance: float = 1e-4,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate counterfactual explanation using gradient-based adversarial perturbation.

    Args:
        sample: Original time series sample
        dataset: Dataset object (for compatibility with other methods)
        model: Trained classification model
        target_class: Target class for counterfactual (if None, finds different class)
        lambda_reg: Regularization parameter for proximity constraint
        lambda_sparse: Regularization parameter for sparsity constraint
        learning_rate: Learning rate for optimization
        max_iterations: Maximum number of optimization iterations
        tolerance: Convergence tolerance
        device: Device to run on (if None, auto-detects)

    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model to device and set to eval mode
    model.to(device)
    model.eval()

    # Convert sample to tensor and prepare for model
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device)

    # Handle different input shapes - ensure (batch, channels, length)
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.reshape(1, 1, -1)  # (length,) -> (1, 1, length)
    elif len(x_tensor.shape) == 2:
        # Could be (channels, length) or (length, channels)
        if x_tensor.shape[0] > x_tensor.shape[1]:
            x_tensor = x_tensor.T  # Assume (length, channels) -> (channels, length)
        x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension

    # Get original prediction
    with torch.no_grad():
        original_pred = model(x_tensor)
        original_class = torch.argmax(original_pred, dim=-1).item()
        original_pred_np = torch.softmax(original_pred, dim=-1).squeeze().cpu().numpy()

    # Determine target class
    if target_class is None:
        # Find the class with second highest probability
        sorted_classes = torch.argsort(original_pred, dim=-1, descending=True)
        target_class = sorted_classes[0, 1].item()  # Second most likely class

    # If already in target class, return None
    if original_class == target_class:
        return None, None

    # Initialize counterfactual as copy of original
    x_cf = x_tensor.clone().detach().requires_grad_(True)

    # Optimizer with different strategy
    optimizer = optim.Adam([x_cf], lr=learning_rate)

    best_cf = None
    best_loss = float('inf')
    best_validity = 0.0

    # Two-phase optimization: first focus on prediction, then refine with regularization
    phase1_iterations = max_iterations // 2
    current_lambda_reg = 0.0  # Start without regularization
    current_lambda_sparse = 0.0

    for iteration in range(max_iterations):
        # Switch to phase 2 halfway through - add regularization
        if iteration == phase1_iterations:
            current_lambda_reg = lambda_reg
            current_lambda_sparse = lambda_sparse
            if verbose:
                print(f"COMTE: Switching to phase 2 with regularization at iteration {iteration}")

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_cf)

        # Prediction loss - focus heavily on getting the right class
        log_probs = torch.log_softmax(logits, dim=-1)
        pred_loss = -log_probs[0, target_class]

        # Distance loss (proximity constraint) - only in phase 2
        distance_loss = torch.norm(x_cf - x_tensor, p=2)

        # Sparsity loss (encourage minimal changes) - only in phase 2
        sparsity_loss = torch.norm(x_cf - x_tensor, p=1)

        # Total loss with adaptive weights
        total_loss = pred_loss + current_lambda_reg * distance_loss + current_lambda_sparse * sparsity_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Check current validity
        with torch.no_grad():
            current_probs = torch.softmax(logits, dim=-1)
            current_validity = current_probs[0, target_class].item()
            current_pred_class = torch.argmax(current_probs, dim=-1).item()

        # Track best solution (prioritize validity heavily)
        if current_pred_class == target_class:
            if current_validity > best_validity or (current_validity >= best_validity and total_loss.item() < best_loss):
                best_loss = total_loss.item()
                best_validity = current_validity
                best_cf = x_cf.clone().detach()
        elif best_cf is None:
            # If no valid solution yet, keep the one with highest validity
            if current_validity > best_validity:
                best_validity = current_validity
                best_cf = x_cf.clone().detach()

                # Early stopping if we've achieved good validity
        if current_validity > 0.99:
            if verbose:
                print(f"COMTE: Early stop at iteration {iteration} with validity {current_validity:.4f}")
            break

        # Debug output every 500 iterations
        if verbose and iteration % 500 == 0:
            print(f"COMTE iteration {iteration}: loss={total_loss.item():.4f}, "
                  f"validity={current_validity:.4f}, pred_class={current_pred_class}")

    if best_cf is None:
        if verbose:
            print("COMTE: No counterfactual found - best_cf is None")
        return None, None

    # Get final prediction
    with torch.no_grad():
        final_pred = model(best_cf)
        predicted_class = torch.argmax(final_pred, dim=-1).item()
        final_pred_np = torch.softmax(final_pred, dim=-1).squeeze().cpu().numpy()
        final_validity = final_pred_np[target_class]

    if verbose:
        print(f"COMTE final: pred_class={predicted_class}, target={target_class}, validity={final_validity:.4f}")

    # Check if counterfactual is valid - use relaxed criteria
    # Accept if either predicted class matches OR validity is reasonably high
    if predicted_class != target_class and final_validity < 0.4:
        if verbose:
            print(f"COMTE: Counterfactual failed validation - predicted {predicted_class}, wanted {target_class}, validity too low")
        return None, None

    # Convert back to original sample format
    cf_sample = best_cf.squeeze(0).cpu().numpy()

    # Handle output shape to match input format
    if len(sample.shape) == 1:
        cf_sample = cf_sample.squeeze()  # Remove channel dimension if input was 1D
    elif len(sample.shape) == 2:
        if sample.shape[0] > sample.shape[1]:
            cf_sample = cf_sample.T  # Convert back to (length, channels) if needed

    return cf_sample, final_pred_np


def _compute_distance(x1: torch.Tensor, x2: torch.Tensor, metric: str = 'euclidean') -> torch.Tensor:
    """Compute distance between two time series."""
    if metric == 'euclidean':
        return torch.norm(x1 - x2, p=2)
    elif metric == 'dtw':
        return _soft_dtw(x1, x2)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


def _soft_dtw(x1: torch.Tensor, x2: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Simplified soft DTW implementation for differentiable DTW computation.
    """
    # Flatten to 1D for DTW computation
    if len(x1.shape) > 1:
        x1_flat = x1.flatten()
        x2_flat = x2.flatten()
    else:
        x1_flat = x1
        x2_flat = x2

    n, m = len(x1_flat), len(x2_flat)

    # Compute pairwise squared distances
    D = torch.zeros(n, m, device=x1.device)
    for i in range(n):
        for j in range(m):
            D[i, j] = (x1_flat[i] - x2_flat[j]) ** 2

    # Initialize DP matrix
    R = torch.full((n + 1, m + 1), float('inf'), device=x1.device)
    R[0, 0] = 0

    # Fill DP matrix with soft-min approximation
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r0 = R[i-1, j-1]
            r1 = R[i-1, j]
            r2 = R[i, j-1]

            # Simple minimum for differentiability
            R[i, j] = D[i-1, j-1] + torch.min(torch.stack([r0, r1, r2]))

    return R[n, m]


# Alternative function with more configuration options
def comte_cf_advanced_gradient(
    sample: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
    dataset=None,
    distance_metric: str = 'euclidean',
    lambda_reg: float = 1.0,
    lambda_sparse: float = 0.1,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    constraints: Optional[Dict[str, Any]] = None,
    device: str = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Advanced gradient-based variant with additional configuration options.

    Additional Args:
        distance_metric: Distance metric ('euclidean' or 'dtw')
        constraints: Dictionary of constraints (e.g., feature bounds)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    # Prepare input tensor
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.reshape(1, 1, -1)
    elif len(x_tensor.shape) == 2:
        if x_tensor.shape[0] > x_tensor.shape[1]:
            x_tensor = x_tensor.T
        x_tensor = x_tensor.unsqueeze(0)

    # Get original prediction and determine target
    with torch.no_grad():
        original_pred = model(x_tensor)
        original_class = torch.argmax(original_pred, dim=-1).item()

    if target_class is None:
        sorted_classes = torch.argsort(original_pred, dim=-1, descending=True)
        target_class = sorted_classes[0, 1].item()

    if original_class == target_class:
        return None, None

    # Initialize and optimize
    x_cf = x_tensor.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x_cf], lr=learning_rate)

    best_cf = None
    best_loss = float('inf')
    prev_loss = float('inf')

    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Compute losses
        logits = model(x_cf)
        log_probs = torch.log_softmax(logits, dim=-1)
        pred_loss = -log_probs[0, target_class]

        distance_loss = _compute_distance(x_cf, x_tensor, distance_metric)
        sparsity_loss = torch.norm(x_cf - x_tensor, p=1)

        total_loss = pred_loss + lambda_reg * distance_loss + lambda_sparse * sparsity_loss

        total_loss.backward()
        optimizer.step()

        # Apply constraints if provided
        if constraints:
            _apply_constraints(x_cf, constraints)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_cf = x_cf.clone().detach()

        if iteration > 0 and abs(prev_loss - total_loss.item()) < tolerance:
            break

        prev_loss = total_loss.item()

    if best_cf is None:
        return None, None

    # Verify and return
    with torch.no_grad():
        final_pred = model(best_cf)
        predicted_class = torch.argmax(final_pred, dim=-1).item()
        final_pred_np = torch.softmax(final_pred, dim=-1).squeeze().cpu().numpy()

    if predicted_class != target_class:
        return None, None

    # Format output
    cf_sample = best_cf.squeeze(0).cpu().numpy()
    if len(sample.shape) == 1:
        cf_sample = cf_sample.squeeze()
    elif len(sample.shape) == 2 and sample.shape[0] > sample.shape[1]:
        cf_sample = cf_sample.T

    return cf_sample, final_pred_np


def _apply_constraints(x_cf: torch.Tensor, constraints: Dict[str, Any]):
    """Apply constraints during optimization."""
    with torch.no_grad():
        if 'feature_bounds' in constraints:
            bounds = constraints['feature_bounds']
            for i, (min_val, max_val) in enumerate(bounds):
                if i < x_cf.shape[1]:  # Check if feature index exists
                    x_cf[0, i, :] = torch.clamp(x_cf[0, i, :], min_val, max_val)

        if 'immutable_features' in constraints:
            # This would need the original tensor to restore immutable features
            pass  # Simplified for this implementation


def comte_ts_cf_gradient(
    sample: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
    dataset=None,
    lambda_reg: float = 0.01,
    lambda_sparse: float = 0.001,
    lambda_smooth: float = 0.01,
    lambda_temporal: float = 0.005,
    learning_rate: float = 0.1,
    max_iterations: int = 3000,
    tolerance: float = 1e-4,
    device: str = None,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Gradient-based variant with added temporal consistency constraints.

    Extends the gradient-based adversarial perturbation with additional
    regularization terms specifically designed for time series data:
    - Temporal smoothness: Encourages gradual changes over time
    - Trend preservation: Maintains local temporal trends
    - Pattern consistency: Preserves important temporal patterns

    Args:
        sample: Original time series sample
        dataset: Dataset object (for compatibility)
        model: Trained classification model
        target_class: Target class for counterfactual
        lambda_reg: Proximity constraint weight
        lambda_sparse: Sparsity constraint weight
        lambda_smooth: Temporal smoothness weight (penalizes rapid changes)
        lambda_temporal: Temporal consistency weight (preserves trends)
        learning_rate: Learning rate for optimization
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance
        device: Device to run on
        verbose: Print debug information

    Returns:
        Tuple of (counterfactual_sample, prediction) or (None, None) if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model to device and set to eval mode
    model.to(device)
    model.eval()

    # Convert sample to tensor and prepare for model
    x_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    original_shape = sample.shape

    # Handle different input shapes - ensure (batch, channels, length)
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.reshape(1, 1, -1)
    elif len(x_tensor.shape) == 2:
        if x_tensor.shape[0] > x_tensor.shape[1]:
            x_tensor = x_tensor.T
        x_tensor = x_tensor.unsqueeze(0)

    B, C, L = x_tensor.shape

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

    # Compute original temporal properties for preservation
    with torch.no_grad():
        # First-order differences (velocity)
        original_diff1 = x_tensor[:, :, 1:] - x_tensor[:, :, :-1]
        # Second-order differences (acceleration)
        original_diff2 = original_diff1[:, :, 1:] - original_diff1[:, :, :-1]

    # Initialize counterfactual
    x_cf = x_tensor.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x_cf], lr=learning_rate)

    best_cf = None
    best_loss = float('inf')
    best_validity = 0.0

    # Adaptive regularization: start with prediction focus, add constraints later
    phase1_iterations = max_iterations // 3
    phase2_iterations = 2 * max_iterations // 3

    for iteration in range(max_iterations):
        # Adjust regularization weights progressively
        if iteration < phase1_iterations:
            # Phase 1: Focus on prediction
            curr_lambda_reg = 0.0
            curr_lambda_sparse = 0.0
            curr_lambda_smooth = 0.0
            curr_lambda_temporal = 0.0
        elif iteration < phase2_iterations:
            # Phase 2: Add proximity and smoothness
            curr_lambda_reg = lambda_reg * 0.5
            curr_lambda_sparse = lambda_sparse * 0.5
            curr_lambda_smooth = lambda_smooth
            curr_lambda_temporal = lambda_temporal * 0.5
        else:
            # Phase 3: Full regularization
            curr_lambda_reg = lambda_reg
            curr_lambda_sparse = lambda_sparse
            curr_lambda_smooth = lambda_smooth
            curr_lambda_temporal = lambda_temporal

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_cf)
        log_probs = torch.log_softmax(logits, dim=-1)
        pred_loss = -log_probs[0, target_class]

        # Proximity loss (L2 distance)
        distance_loss = torch.norm(x_cf - x_tensor, p=2)

        # Sparsity loss (L1 distance)
        sparsity_loss = torch.norm(x_cf - x_tensor, p=1)

        # Temporal smoothness loss: penalize large changes between consecutive time points
        cf_diff1 = x_cf[:, :, 1:] - x_cf[:, :, :-1]
        smoothness_loss = torch.norm(cf_diff1, p=2)

        # Temporal consistency: preserve local trends (second-order smoothness)
        cf_diff2 = cf_diff1[:, :, 1:] - cf_diff1[:, :, :-1]
        temporal_loss = torch.norm(cf_diff2 - original_diff2, p=2)

        # Total loss
        total_loss = (pred_loss +
                     curr_lambda_reg * distance_loss +
                     curr_lambda_sparse * sparsity_loss +
                     curr_lambda_smooth * smoothness_loss +
                     curr_lambda_temporal * temporal_loss)

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Evaluate current solution
        with torch.no_grad():
            current_probs = torch.softmax(logits, dim=-1)
            current_validity = current_probs[0, target_class].item()
            current_pred_class = torch.argmax(current_probs, dim=-1).item()

        # Track best solution
        if current_pred_class == target_class:
            if current_validity > best_validity or \
               (current_validity >= best_validity and total_loss.item() < best_loss):
                best_loss = total_loss.item()
                best_validity = current_validity
                best_cf = x_cf.clone().detach()
        elif best_cf is None or current_validity > best_validity:
            best_validity = current_validity
            best_cf = x_cf.clone().detach()

        # Early stopping
        if current_validity > 0.99 and current_pred_class == target_class:
            if verbose:
                print(f"CoMTE-TS: Early stop at iteration {iteration} with validity {current_validity:.4f}")
            break

        # Debug output
        if verbose and iteration % 500 == 0:
            print(f"CoMTE-TS iter {iteration}: loss={total_loss.item():.4f}, "
                  f"pred={pred_loss.item():.4f}, smooth={smoothness_loss.item():.4f}, "
                  f"validity={current_validity:.4f}, pred_class={current_pred_class}")

    if best_cf is None:
        if verbose:
            print("CoMTE-TS: No counterfactual found")
        return None, None

    # Get final prediction
    with torch.no_grad():
        final_pred = model(best_cf)
        predicted_class = torch.argmax(final_pred, dim=-1).item()
        final_pred_np = torch.softmax(final_pred, dim=-1).squeeze().cpu().numpy()
        final_validity = final_pred_np[target_class]

    if verbose:
        print(f"CoMTE-TS final: pred_class={predicted_class}, target={target_class}, "
              f"validity={final_validity:.4f}")

    # Relaxed validation
    if predicted_class != target_class and final_validity < 0.3:
        if verbose:
            print("CoMTE-TS: Counterfactual failed validation")
        return None, None

    # Convert back to original format
    cf_sample = best_cf.squeeze(0).cpu().numpy()

    if len(original_shape) == 1:
        cf_sample = cf_sample.squeeze()
    elif len(original_shape) == 2:
        if original_shape[0] > original_shape[1]:
            cf_sample = cf_sample.T

    return cf_sample, final_pred_np
