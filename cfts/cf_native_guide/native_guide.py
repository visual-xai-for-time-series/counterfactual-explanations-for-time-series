from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from captum.attr import GradientShap
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
# Native Guide: Instance-based Counterfactual Explanations for Time Series
#
# Paper: Delaney, E., Greene, D., & Keane, M. T. (2021).
#        "Instance-based counterfactual explanations for time series classification."
#        International Conference on Case-Based Reasoning, Springer
#
# Repository: https://github.com/e-delaney/Instance-Based_CFE_TSC
#
# Algorithm outline:
#   1. Predict the class of the query sample.
#   2. Find the Nearest Unlike Neighbor (NUN): the closest training example
#      that belongs to a *different* class (or a specific target class when
#      `target_class` is supplied).
#   3. Use gradient attribution (default: GradientShap) on the NUN to rank
#      time-step importance — high-attribution windows are replaced first.
#   4. Iteratively copy growing windows from the NUN into the query until the
#      model flips its prediction to the desired counterfactual class.
####


def native_guide_uni_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    weight_function=GradientShap,
    max_iter: int | None = None,
    sub_len: int = 1,
    max_samples: int | None = None,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Native Guide counterfactual for a single time-series sample.

    The method finds the Nearest Unlike Neighbor (NUN) in ``dataset`` and
    iteratively transplants its most attribution-important windows into
    ``sample`` until the classifier changes its prediction.  When
    ``target_class`` is given, only NUN candidates from that class are
    considered, allowing the caller to steer the counterfactual toward a
    specific desired outcome.

    Follows the same signature pattern as every other CF method in this
    repository (``abstract_cf``, ``glacier_cf``, ``cem_cf``, …) so it plugs
    straight into the existing evaluation and example scripts.

    Parameters
    ----------
    sample:
        The query time series whose counterfactual is sought.  Accepts 1-D
        ``(L,)``, ``(C, L)`` or ``(L, C)`` NumPy arrays (or anything that
        converts with ``np.asarray``).
    model:
        Trained PyTorch classifier.  Must accept input of shape ``(B, C, L)``
        and return logits or probabilities of shape ``(B, num_classes)``.
    target_class:
        When set, restrict the NUN search to candidates that the model
        classifies as ``target_class``.  This directs the counterfactual toward
        a specific class rather than the nearest class that differs from the
        query's predicted label.  When ``None``, any class differing from the
        query's predicted label is eligible (original Native Guide behaviour).
    dataset:
        Training data used to find the NUN.  A sequence of ``(x, y)`` pairs
        where each ``x`` is a time series, or a NumPy array of shape
        ``(N, C, L)``.  Labels from ``y`` are not used; the model predicts them
        afresh to avoid label noise.
    weight_function:
        Any Captum attribution class (default: ``GradientShap``) that accepts
        ``(input, baselines, target)``.  Controls which time steps are
        considered most influential for the NUN.
    max_iter:
        Maximum number of window-growth iterations.  Defaults to the series
        length ``L``, which guarantees convergence at the cost of a full copy.
    sub_len:
        Initial window size in time steps.  The window grows by ``sub_len``
        each iteration until the model flips or ``max_iter`` is exhausted.
    verbose:
        Print per-iteration diagnostics when ``True``.

    Returns
    -------
    counterfactual : np.ndarray
        Counterfactual time series in the **same shape / orientation** as
        ``sample``.
    scores : np.ndarray, shape (num_classes,)
        Model output (logits or softmax scores) for the counterfactual.
    """
    device = next(model.parameters()).device

    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)

    # --- 1. Normalise shapes ---
    sample_cl, ts, ori = ensure_ncl(np.asarray(sample, dtype=np.float32), dataset)
    N, C, L = ts.shape

    if max_iter is None:
        max_iter = L  # worst-case: grow window across the full series

    # --- 2. Predict labels for the dataset and the query sample ---
    preds_data = batched_predict(model, ts, device)           # (N, num_classes)
    with torch.no_grad():
        preds_sample = detach_to_numpy(
            model(numpy_to_torch(sample_cl.reshape(1, C, L), device))
        )  # (1, num_classes)
    label_data = np.argmax(preds_data, axis=1)                # (N,)
    label_sample = int(np.argmax(preds_sample))

    # --- 3. Select NUN candidates ---
    # If target_class is specified, restrict candidates to that class only.
    # Otherwise fall back to any sample the model classifies differently from
    # the query — the original Native Guide behaviour.
    if target_class is not None:
        if target_class == label_sample:
            raise ValueError(
                f"target_class ({target_class}) is the same as the query's predicted "
                f"class ({label_sample}). Choose a different target class."
            )
        mask = label_data == target_class
        if not np.any(mask):
            # No training sample was classified as the requested target class;
            # return the unmodified query so the caller can detect the failure.
            if verbose:
                print(
                    f"[NativeGuide] No candidate found for target_class={target_class}. "
                    "Returning original sample unchanged."
                )
            return revert_orientation(sample_cl, ori), preds_sample.reshape(-1)
    else:
        # Any class that differs from the query class is eligible.
        mask = label_data != label_sample
        if not np.any(mask):
            if verbose:
                print("[NativeGuide] All dataset samples share the query class. "
                      "Returning original sample unchanged.")
            return revert_orientation(sample_cl, ori), preds_sample.reshape(-1)

    candidates = ts[mask]                    # (M, C, L)
    candidates_labels = label_data[mask]     # (M,)

    # --- 4. Find the NUN via nearest-neighbour search ---
    # We search over a pool of k neighbours to skip any exact duplicates of the
    # query that happen to be misclassified, but in practice the first hit is
    # almost always the true NUN.
    k_pool = max(1, min(int(L * 0.25), len(candidates)))
    neigh = NearestNeighbors(
        n_neighbors=min(k_pool + 1, len(candidates)), metric="euclidean"
    )
    neigh.fit(candidates.reshape(len(candidates), -1))
    _, idxs = neigh.kneighbors(sample_cl.reshape(1, -1), return_distance=True)

    # Walk neighbours closest-first; pick the first one with the desired label.
    # (When target_class is set every candidate already has the right label, so
    # the loop always exits on the first iteration.)
    native_guide = None
    cf_label = None
    desired_label = target_class  # None means "any label != label_sample"
    for idx in idxs[0]:
        cand_label = int(candidates_labels[idx])
        if desired_label is None or cand_label == desired_label:
            native_guide = candidates[idx]
            cf_label = cand_label
            break

    if native_guide is None:
        # Fallback: use the single closest candidate regardless of label
        native_guide = candidates[0]
        cf_label = int(candidates_labels[0])

    if verbose:
        print(f"[NativeGuide] Query class: {label_sample} | NUN class: {cf_label}")

    # --- 5. Compute time-step importance via gradient attribution on the NUN ---
    # The attribution highlights which time steps in the NUN are most
    # responsible for the model predicting `cf_label`.  We copy the highest-
    # attribution windows first so that early iterations have the greatest
    # chance of flipping the prediction.
    attributor = weight_function(model)
    baselines = numpy_to_torch(ts, device)  # full dataset as baselines
    attributions = attributor.attribute(
        numpy_to_torch(native_guide.reshape(1, C, L), device),
        baselines=baselines,
        target=cf_label,
    )
    attr_np = detach_to_numpy(attributions)  # (1, C, L) or (C, L)

    # Collapse channel dimension so importance is a 1-D vector over time steps.
    if attr_np.ndim == 3:
        importance = np.sum(np.abs(attr_np[0]), axis=0)  # (L,)
    else:
        importance = np.sum(np.abs(attr_np), axis=0)     # (L,)

    def find_most_influential_window(length: int) -> int:
        """Return the start index of the length-`length` window with the highest
        summed attribution score (sliding-window convolution trick)."""
        if length >= len(importance):
            return 0
        conv = np.convolve(importance, np.ones(length, dtype=importance.dtype), mode="valid")
        return int(np.argmax(conv))

    # --- 6. Iteratively transplant windows from the NUN into the query ---
    # Each iteration grows the window by `sub_len`.  We stop as soon as the
    # model's argmax matches `cf_label`, meaning the counterfactual is valid.
    cf = sample_cl.copy()
    scores_cf = preds_sample.reshape(-1)

    for i in range(max_iter):
        length = i + sub_len
        if length > L:
            break

        start = find_most_influential_window(length)
        end = start + length

        cf_candidate = cf.copy()
        cf_candidate[:, start:end] = native_guide[:, start:end]
        with torch.no_grad():
            scores_candidate = detach_to_numpy(
                model(numpy_to_torch(cf_candidate.reshape(1, C, L), device))
            ).reshape(-1)

        cf = cf_candidate
        scores_cf = scores_candidate

        if cf_label == int(np.argmax(scores_cf)):
            # Model now predicts the desired counterfactual class — done.
            break

    if verbose and cf_label != int(np.argmax(scores_cf)):
        print(
            f"[NativeGuide] Warning: counterfactual did not flip to class {cf_label} "
            f"after {max_iter} iterations. Returning best attempt."
        )

    # --- 7. Restore the original input orientation before returning ---
    return revert_orientation(cf, ori), scores_cf
