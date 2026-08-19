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
    use_abs_importance: bool = False,
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
    use_abs_importance:
        When ``False`` (default), rank candidate windows by the *signed* sum
        of the NUN's attribution — matching the original paper/repo, where
        the (unrectified) CAM weight vector is summed directly, so the
        window most positively supporting ``cf_label`` is copied first. When
        ``True``, rank by attribution *magnitude* instead (``abs`` before
        summing), which can select windows that argue against ``cf_label``
        as readily as windows that argue for it.
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
    # `candidates` is already restricted to the desired label(s) (see the
    # masking above), so the single nearest neighbour is the NUN.
    neigh = NearestNeighbors(n_neighbors=1, metric="euclidean")
    neigh.fit(candidates.reshape(len(candidates), -1))
    _, idxs = neigh.kneighbors(sample_cl.reshape(1, -1), return_distance=True)

    nun_idx = int(idxs[0, 0])
    native_guide = candidates[nun_idx]
    cf_label = int(candidates_labels[nun_idx])

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
    if attr_np.ndim == 3:
        attr_np = attr_np[0]                              # (C, L)

    # Collapse channel dimension so importance is a 1-D vector over time steps.
    # By default we sum the *signed* attribution (matching the original CAM
    # weight vector, which is not rectified) so that the window search below
    # favours regions that actually support `cf_label`. Setting
    # `use_abs_importance=True` instead ranks windows by attribution
    # *magnitude*, regardless of whether they support or oppose `cf_label`.
    if use_abs_importance:
        importance = np.sum(np.abs(attr_np), axis=0)     # (L,)
    else:
        importance = np.sum(attr_np, axis=0)              # (L,)

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


####
# Native Guide — DBA variant (NG-DBA)
#
# Paper: Delaney, E., Greene, D., & Keane, M. T. (2021).
#        "Instance-based counterfactual explanations for time series classification."
#        International Conference on Case-Based Reasoning, Springer
#
# Repository: https://github.com/e-delaney/Instance-Based_CFE_TSC (dba.py)
#
# This is the sibling variant to `native_guide_uni_cf` (which follows the
# repo's NG-CAM script): instead of transplanting a growing window from the
# NUN, NG-DBA blends the query and its NUN directly in *value* space via
# DTW Barycenter Averaging (DBA), walking the blend weight from "all query"
# towards "all NUN" until the target class is confidently reached.
#
# Algorithm outline:
#   1. Predict the class of the query sample.
#   2. Find the Nearest Unlike Neighbor (NUN) via a *DTW* k-NN search
#      (tslearn's KNeighborsTimeSeries) — not euclidean — matching the
#      original NG-DBA script, which is the whole reason DBA (a DTW-aware
#      barycenter) rather than a linear interpolation is used for the blend.
#   3. Blend query and NUN with `dtw_barycenter_averaging([query, nun],
#      weights=[1-beta, beta])`, starting at beta=0 and stepping by
#      `beta_step` until the target class's probability exceeds
#      `prob_threshold`.
#   4. If beta reaches 1 without crossing the threshold, default to the raw
#      NUN itself (matching the original script's "defaulting" behaviour).
####


def native_guide_dba_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    max_samples: int | None = None,
    beta_step: float = 0.05,
    prob_threshold: float = 0.5,
    dtw_max_iter: int = 10,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Native Guide / NG-DBA counterfactual via DTW barycenter blending.

    Univariate only (DBA barycenter averaging here is computed over a single
    channel per the original script). See the module-level notes above for
    the algorithm and its relationship to `native_guide_uni_cf` (NG-CAM
    style).

    Parameters
    ----------
    sample, model, target_class, dataset:
        Same semantics as `native_guide_uni_cf`.
    max_samples:
        Optional cap on how much of `dataset` is used for NUN search
        (stratified subsample) — DTW k-NN is O(N) DTW alignments per query.
    beta_step:
        Increment added to the DBA blend weight each iteration (bake-off
        default: 0.05).
    prob_threshold:
        Target-class probability the blend must exceed to be accepted
        (bake-off default: 0.5).
    dtw_max_iter:
        Max iterations for the inner DBA optimisation itself (tslearn's
        `dtw_barycenter_averaging(..., max_iter=...)`), not to be confused
        with the beta-stepping loop.
    verbose:
        Print per-iteration diagnostics when True.

    Returns
    -------
    counterfactual : np.ndarray, same shape/orientation as `sample`.
    scores : np.ndarray, shape (num_classes,) — model output for the CF.
    """
    try:
        from tslearn.neighbors import KNeighborsTimeSeries
        from tslearn.barycenters import dtw_barycenter_averaging
    except ImportError as e:
        raise ImportError(
            "native_guide_dba_cf requires tslearn (pip install tslearn>=0.6.3)."
        ) from e

    device = next(model.parameters()).device

    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)

    sample_cl, ts, ori = ensure_ncl(np.asarray(sample, dtype=np.float32), dataset)
    N, C, L = ts.shape
    if C != 1:
        raise ValueError(
            "native_guide_dba_cf currently supports univariate series only "
            f"(got {C} channels); use native_guide_uni_cf for the window-"
            "transplant (NG-CAM style) variant instead."
        )

    preds_data = batched_predict(model, ts, device)
    with torch.no_grad():
        preds_sample = detach_to_numpy(
            model(numpy_to_torch(sample_cl.reshape(1, C, L), device))
        )
    label_data = np.argmax(preds_data, axis=1)
    label_sample = int(np.argmax(preds_sample))

    if target_class is None:
        # Bake-off's NG-DBA always targets the 2nd most probable class.
        sorted_idx = np.argsort(preds_sample.reshape(-1))[::-1]
        target_class = int(sorted_idx[1])

    if target_class == label_sample:
        raise ValueError(
            f"target_class ({target_class}) is the same as the query's predicted "
            f"class ({label_sample}). Choose a different target class."
        )

    mask = label_data == target_class
    if not np.any(mask):
        if verbose:
            print(
                f"[NG-DBA] No candidate found for target_class={target_class}. "
                "Returning original sample unchanged."
            )
        return revert_orientation(sample_cl, ori), preds_sample.reshape(-1)

    candidates = ts[mask]  # (M, C, L)

    # --- DTW k-NN NUN search (tslearn wants (n_series, sz, d)) ---
    candidates_lc = candidates.transpose(0, 2, 1)  # (M, L, C)
    query_lc = sample_cl.T.reshape(1, L, C)        # (1, L, C)

    knn = KNeighborsTimeSeries(n_neighbors=1, metric="dtw")
    knn.fit(candidates_lc)
    _, idxs = knn.kneighbors(query_lc, return_distance=True)
    nun_idx = int(idxs[0, 0])
    nun_cl = candidates[nun_idx]  # (C, L)

    if verbose:
        print(f"[NG-DBA] Query class: {label_sample} | NUN class (target): {target_class}")

    # --- Grow the DBA blend weight from "all query" to "all NUN" ---
    query_lc1 = sample_cl.T  # (L, C)
    nun_lc1 = nun_cl.T       # (L, C)

    beta = 0.0
    defaulted = True
    cf_cl = nun_cl  # fallback if beta reaches 1 without crossing the threshold

    while beta < 1.0:
        blend_lc = dtw_barycenter_averaging(
            [query_lc1, nun_lc1],
            barycenter_size=L,
            max_iter=dtw_max_iter,
            weights=np.array([1.0 - beta, beta]),
        )
        blend_cl = blend_lc.T.reshape(C, L).astype(np.float32)

        with torch.no_grad():
            pred = detach_to_numpy(
                model(numpy_to_torch(blend_cl.reshape(1, C, L), device))
            ).reshape(-1)
        prob_target = float(pred[target_class])

        if verbose:
            print(f"[NG-DBA] beta={beta:.2f} target_prob={prob_target:.3f}")

        if prob_target > prob_threshold:
            cf_cl = blend_cl
            defaulted = False
            break

        beta += beta_step

    if verbose and defaulted:
        print("[NG-DBA] beta reached 1.0 without crossing prob_threshold; defaulting to raw NUN.")

    with torch.no_grad():
        scores_cf = detach_to_numpy(
            model(numpy_to_torch(cf_cl.reshape(1, C, L), device))
        ).reshape(-1)

    return revert_orientation(cf_cl, ori), scores_cf
