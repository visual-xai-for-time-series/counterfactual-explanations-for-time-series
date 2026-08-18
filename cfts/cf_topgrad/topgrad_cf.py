# https://link.springer.com/chapter/10.1007/978-3-032-31933-3_35

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from sklearn.neighbors import NearestNeighbors

from cfts.cf__abstract.abstract import (
    batched_predict,
    detach_to_numpy,
    ensure_ncl,
    numpy_to_torch,
    revert_orientation,
    subsample_dataset,
)


####
# TopGrad-CF: Gradient-Guided Counterfactual Explanations for Time Series
# Classification
#
# Paper: Hosseinzadeh, P. et al. (2026).
#        "TopGrad-CF: Gradient-Guided Counterfactual Explanations for Time
#        Series Classification." ICPR 2026.
#
# Paper URL: https://link.springer.com/chapter/10.1007/978-3-032-31933-3_35
# Repository: https://github.com/pouyahosseinzadeh/TopGrad-CF
#
# Algorithm outline (ported from the authors' TensorFlow-1 / Alibi-based
# reference implementation to plain PyTorch and this project's CFMethod
# contract; see "Deviations" below for where this port intentionally departs
# from that reference):
#   1. Find a "prototype": the nearest training sample already predicted as
#      the (derived) target class — a Nearest-Unlike-Neighbour, exactly as in
#      Native Guide (`cf_native_guide`).
#   2. Optimise the counterfactual with Adam to minimise a loss combining
#      classification pressure toward the target class, L1/L2 proximity to
#      both the original sample and the prototype, and a smoothness penalty
#      on the candidate itself.
#   3. At every step, keep only the top `top_k_frac` fraction (by magnitude)
#      of the loss gradient and zero out the rest before the optimiser step —
#      the "TopGrad" masking that gives the method its name, restricting each
#      update to the handful of time steps the model is currently most
#      sensitive to, rather than moving every point at once.
#   4. Run this in two stages: (a) a coarse sweep across orders of magnitude
#      of the proximity weight `lam` to bracket a workable value, then (b) a
#      refinement stage that additionally restricts every update to a single
#      contiguous "prominent segment" — the window with the largest
#      |sample - prototype| gap — and grows that window across outer steps
#      until the counterfactual is confidently valid or the window covers
#      most of the series. `lam` is nudged after each outer step depending on
#      how often a valid counterfactual was found, echoing the reference's
#      lambda-bisection search.
#
# Deviations from the reference implementation (confirmed by cloning and actually
# running the reference repo end-to-end against a real classifier — not just reading
# it — in cfts/cf_topgrad/topgrad_coffee_comparison.ipynb):
#   - The classifier loss (`classification_weight`) is always part of the
#     optimised objective. In the reference code it is computed but never
#     added to the final loss for autograd/Keras models — confirmed by
#     running `Main.py`'s own experiment, where the reference's counterfactual
#     barely moves the target-class probability at all — leaving prototype
#     proximity as the only force pulling the candidate toward the target
#     class; adding the classifier loss back makes the masked gradient
#     actually driven by the classifier (matching the "gradient-guided"
#     framing of the paper title) and makes convergence far more reliable
#     across arbitrary datasets/models.
#   - The proximity/sparsity/smoothness terms use the *mean* absolute/squared
#     difference rather than the reference's `tf.reduce_sum`. The reference
#     never needs this because it has no classifier term to balance against;
#     once one is added, an unnormalised sum grows with the series length `L`
#     while the classifier loss stays O(1), so on long series (e.g. length
#     286) the classifier term becomes negligible and the flip above stops
#     mattering in practice — confirmed empirically while building the
#     comparison notebook. Averaging keeps the balance between
#     `classification_weight` and `lam`/`sparsity_weight`/`smoothness_weight`
#     roughly independent of `L`.
#   - `max_iter`, `n_lambda_orders`, `max_lam_steps` and `max_outer_steps`
#     default much lower than the reference (1000 / 10 / 10 / ~65 outer
#     steps) to keep a single call tractable when benchmarked alongside every
#     other method in this repository. Raise them for a closer reproduction.
#   - Lambda adjustment between outer steps is a simplified heuristic
#     (multiply/divide by a constant factor) rather than the reference's
#     bisection with tracked upper/lower bounds.
#   - Pure PyTorch autograd is used throughout; the reference's numerical
#     gradient fallback for non-differentiable black-box models is dropped
#     since every model in this repository is a PyTorch module.
####


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _distance(x: torch.Tensor, y: torch.Tensor, kind: str) -> torch.Tensor:
    """Mean (not summed) L1/L2 distance, so its magnitude stays O(1) regardless of
    series length — see "Deviations" in the module docstring for why this matters
    once a length-independent classifier loss is mixed into the same objective."""
    if kind == "l2":
        return torch.sqrt(torch.mean((x - y) ** 2) + 1e-12)
    return torch.mean(torch.abs(x - y))


def _prominent_segment(diff: np.ndarray, seg_len: int) -> Tuple[int, int]:
    """Return the (start, end) window of length ``seg_len`` over the 1-D
    array ``diff`` with the largest summed magnitude (sliding-window search,
    ``getprominentsegment`` in the reference implementation)."""
    seg_len = max(1, min(seg_len, len(diff)))
    if seg_len >= len(diff):
        return 0, len(diff)
    conv = np.convolve(np.abs(diff), np.ones(seg_len), mode="valid")
    start = int(np.argmax(conv))
    return start, start + seg_len


def topgrad_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    max_samples: int | None = None,
    max_iter: int = 200,
    early_stop: int = 20,
    lam_init: float = 0.1,
    n_lambda_orders: int = 5,
    max_lam_steps: int = 5,
    tol: float = 0.05,
    target_proba: float = 1.0,
    success_proba: float = 0.95,
    learning_rate_init: float = 0.1,
    decay: bool = True,
    top_k_frac: float = 0.03,
    seg_rate_init: float = 0.05,
    seg_rate_step: float = 0.01,
    seg_rate_max: float = 0.7,
    max_outer_steps: int | None = None,
    sparsity_weight: float = 1.0,
    smoothness_weight: float = 1.0,
    classification_weight: float = 1.0,
    distance: str = "l1",
    seed: int | None = None,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a TopGrad-CF counterfactual for a single time-series sample.

    Follows the same signature pattern as every other CF method in this
    repository (``abstract_cf``, ``native_guide_uni_cf``, ``glacier_cf``, …)
    so it plugs straight into the existing evaluation and example scripts.
    See the module docstring above for the algorithm outline and a list of
    deliberate deviations from the authors' reference implementation.

    Parameters
    ----------
    sample:
        The query time series whose counterfactual is sought. Accepts 1-D
        ``(L,)``, ``(C, L)`` or ``(L, C)`` NumPy arrays (or anything that
        converts with ``np.asarray``).
    model:
        Trained PyTorch classifier. Must accept input of shape ``(B, C, L)``
        and return logits or probabilities of shape ``(B, num_classes)``.
    target_class:
        Desired class for the counterfactual. When ``None``, the highest-
        scoring class other than the query's predicted class is used
        (matching the reference's ``target_class='other'`` default).
    dataset:
        Training data used to find the prototype (nearest unlike neighbour).
        A sequence of ``(x, y)`` pairs, or a NumPy array of shape
        ``(N, C, L)``. Labels from ``y`` are not used; the model predicts
        them afresh to avoid label noise. Required — TopGrad-CF has no
        prototype-free mode.
    max_samples:
        Subsample ``dataset`` to at most this many items (stratified by
        predicted label) before the nearest-neighbour search, for speed on
        large training sets. ``None`` uses the full dataset.
    max_iter:
        Maximum number of masked-gradient steps per (lambda, segment)
        combination.
    early_stop:
        Stop a gradient-descent run early once the validity condition
        (``|predicted target-class probability - target_proba| <= tol``) has
        held, or failed to hold, for this many consecutive steps.
    lam_init:
        Initial proximity weight used to seed the lambda bracket sweep.
    n_lambda_orders:
        Number of orders of magnitude to sweep (``lam_init / 10**i``) when
        bracketing a workable lambda before the refinement stage.
    max_lam_steps:
        Number of lambda-adjustment rounds per segment size in the
        refinement stage.
    tol:
        Tolerance used by the validity condition above.
    target_proba:
        Target softmax probability for the counterfactual's target class
        that the validity condition checks against.
    success_proba:
        Once the target-class probability reaches this value the outer
        segment-growth loop stops early — the counterfactual is considered
        confidently valid.
    learning_rate_init:
        Initial Adam learning rate for each gradient-descent run.
    decay:
        Linearly decay the learning rate to zero over each run when
        ``True``.
    top_k_frac:
        Fraction (by magnitude) of eligible gradient positions kept at each
        step; the rest are zeroed before the optimiser step. This is the
        "TopGrad" mechanism the method is named for.
    seg_rate_init, seg_rate_step, seg_rate_max:
        Schedule for the prominent segment's length as a fraction of the
        series length ``L``: starts at ``seg_rate_init``, grows by
        ``seg_rate_step`` every outer step, and the search stops once it
        would exceed ``seg_rate_max``.
    max_outer_steps:
        Safety cap on the number of segment-growth steps. ``None`` derives
        it from the rate schedule, capped at 20 for tractability.
    sparsity_weight:
        Weight of an additional L1 term pulling the counterfactual toward
        the original sample (kept separate from the lambda-scaled proximity
        term, as in the reference's ``sparsity_loss``).
    smoothness_weight:
        Weight of a total-variation-style penalty on the counterfactual's
        first differences (the reference's ``smoothness_loss``).
    classification_weight:
        Weight of the classifier loss pulling the counterfactual toward
        ``target_class``. See "Deviations" in the module docstring — this is
        always active here.
    distance:
        ``'l1'`` (default, matches the reference) or ``'l2'`` distance used
        for both the proximity-to-original and proximity-to-prototype terms.
    seed:
        Seed for PyTorch's RNG, for reproducibility.
    verbose:
        Print per-step diagnostics when ``True``.

    Returns
    -------
    counterfactual : np.ndarray
        Counterfactual time series in the **same shape / orientation** as
        ``sample``.
    scores : np.ndarray, shape (num_classes,)
        Model output (logits / softmax scores) for the counterfactual.
    """
    if dataset is None:
        raise ValueError(
            "topgrad_cf requires a dataset to select the target-class "
            "prototype (nearest unlike neighbour)."
        )

    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)

    sample_cl, ts, ori = ensure_ncl(np.asarray(sample, dtype=np.float32), dataset)
    N, C, L = ts.shape

    # --- 1. Original prediction & derived target class ----------------------
    with torch.no_grad():
        scores_orig = detach_to_numpy(
            model(numpy_to_torch(sample_cl.reshape(1, C, L), device))
        ).reshape(-1)
    label_orig = int(np.argmax(scores_orig))

    if target_class is None:
        ranked = np.argsort(-scores_orig)
        target_class = int(ranked[0] if ranked[0] != label_orig else ranked[1])
    if target_class == label_orig:
        raise ValueError(
            f"target_class ({target_class}) equals the query's predicted "
            f"class ({label_orig}). Choose a different target class."
        )

    # --- 2. Prototype: nearest training sample already predicted target_class
    preds_data = batched_predict(model, ts, device)
    label_data = np.argmax(preds_data, axis=1)
    mask = label_data == target_class
    if not np.any(mask):
        if verbose:
            print(
                f"[TopGrad-CF] No dataset sample classified as "
                f"target_class={target_class}. Returning original sample unchanged."
            )
        return revert_orientation(sample_cl, ori), scores_orig

    candidates = ts[mask]
    neigh = NearestNeighbors(n_neighbors=1, metric="euclidean")
    neigh.fit(candidates.reshape(len(candidates), -1))
    _, idxs = neigh.kneighbors(sample_cl.reshape(1, -1))
    prototype = candidates[int(idxs[0, 0])]  # (C, L)

    if verbose:
        print(f"[TopGrad-CF] Query class: {label_orig} | target class: {target_class}")

    # --- 3. Optimisation setup -----------------------------------------------
    orig_t = numpy_to_torch(sample_cl.reshape(1, C, L), device)
    proto_t = numpy_to_torch(prototype.reshape(1, C, L), device)
    target_t = torch.tensor([target_class], dtype=torch.long, device=device)
    ce_loss = nn.CrossEntropyLoss()

    def run(cf_np: np.ndarray, lam: float, n_steps: int, region: Tuple[int, int] | None,
            patience: int | None) -> dict:
        """Run up to ``n_steps`` masked-gradient Adam updates starting from
        ``cf_np``. Returns a dict with the final candidate, its scores, the
        number of steps the validity condition held, and the best valid
        candidate seen (by proximity to the original sample)."""
        cf_t = numpy_to_torch(cf_np.reshape(1, C, L), device).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([cf_t], lr=learning_rate_init)

        region_flat = None
        if region is not None:
            start, end = region
            region_mask = np.zeros((C, L), dtype=bool)
            region_mask[:, start:end] = True
            region_flat = region_mask.reshape(-1)

        n_hits = 0
        found_streak = 0
        not_found_streak = 0
        best_cf_np, best_scores, best_dist = None, None, np.inf
        cf_np_now, scores_now = cf_np.copy(), scores_orig.copy()

        for step in range(max(1, n_steps)):
            optimizer.zero_grad()
            pred = model(cf_t)
            probs = torch.softmax(pred, dim=1)
            cls_term = ce_loss(pred, target_t)
            proba_term = (probs[0, target_class] - target_proba) ** 2
            dist_o = _distance(cf_t, orig_t, distance)
            dist_p = _distance(cf_t, proto_t, distance)
            smoothness = torch.mean((cf_t[..., 1:] - cf_t[..., :-1]) ** 2)
            loss = (
                classification_weight * (cls_term + proba_term)
                + lam * (dist_o + dist_p)
                + sparsity_weight * dist_o
                + smoothness_weight * smoothness
            )
            loss.backward()

            # --- TopGrad masking: keep only the top `top_k_frac` fraction of
            # eligible gradient positions (by magnitude), zero out the rest.
            with torch.no_grad():
                grad_flat = detach_to_numpy(cf_t.grad).reshape(-1)
                eligible = region_flat if region_flat is not None else np.ones_like(grad_flat, dtype=bool)
                idx = np.where(eligible)[0]
                keep_mask = np.zeros_like(grad_flat)
                if len(idx) > 0:
                    k = max(1, int(round(top_k_frac * len(idx))))
                    top = idx[np.argsort(-np.abs(grad_flat[idx]))[:k]]
                    keep_mask[top] = 1.0
                cf_t.grad.mul_(numpy_to_torch(keep_mask.reshape(1, C, L), device))

            if decay:
                for g in optimizer.param_groups:
                    g["lr"] = learning_rate_init * max(0.0, 1.0 - step / max(n_steps, 1))
            optimizer.step()

            with torch.no_grad():
                cf_np_now = detach_to_numpy(cf_t).reshape(C, L)
                scores_now = detach_to_numpy(model(cf_t)).reshape(-1)
            proba_now = float(_softmax_np(scores_now)[target_class])
            is_valid = int(np.argmax(scores_now)) == target_class

            if is_valid:
                dist_now = float(np.abs(cf_np_now - sample_cl).sum())
                if dist_now < best_dist:
                    best_dist, best_cf_np, best_scores = dist_now, cf_np_now.copy(), scores_now.copy()

            is_hit = abs(proba_now - target_proba) <= tol
            if is_hit:
                n_hits += 1
                found_streak += 1
                not_found_streak = 0
            else:
                found_streak = 0
                not_found_streak += 1

            if verbose and n_steps > 0 and step % max(1, n_steps // 5) == 0:
                print(
                    f"[TopGrad-CF] lam={lam:.4g} step={step:4d} "
                    f"valid={is_valid} target_proba={proba_now:.3f}"
                )

            if patience is not None and (found_streak >= patience or not_found_streak >= patience):
                break

        return {
            "cf": cf_np_now,
            "scores": scores_now,
            "n_hits": n_hits,
            "best_cf": best_cf_np,
            "best_scores": best_scores,
            "best_dist": best_dist,
        }

    # --- 4. Phase A: bracket a workable lambda via an exponential sweep -----
    n_orders = max(1, n_lambda_orders)
    steps_per_order = max(1, max_iter // n_orders)
    lams_sweep = [lam_init / (10.0 ** i) for i in range(n_orders)]

    best_cf = None
    best_scores = None
    best_dist = np.inf

    cf_np = sample_cl.copy()
    last_scores = scores_orig
    success_per_order = np.zeros(n_orders, dtype=bool)
    for ix, lam_ix in enumerate(lams_sweep):
        result = run(cf_np, lam_ix, steps_per_order, region=None, patience=None)
        cf_np = result["cf"]
        last_scores = result["scores"]
        success_per_order[ix] = result["n_hits"] > 0
        if result["best_cf"] is not None and result["best_dist"] < best_dist:
            best_dist = result["best_dist"]
            best_cf = result["best_cf"]
            best_scores = result["best_scores"]

    hits = np.where(success_per_order)[0]
    misses = np.where(~success_per_order)[0]
    lb_ix = int(hits[1]) if len(hits) > 1 else (int(hits[0]) if len(hits) else 0)
    ub_ix = int(misses[-1]) if len(misses) else 0
    lam = (lams_sweep[lb_ix] + lams_sweep[ub_ix]) / 2.0

    if verbose:
        print(f"[TopGrad-CF] Lambda bracket: [{lams_sweep[lb_ix]:.4g}, {lams_sweep[ub_ix]:.4g}] -> lam={lam:.4g}")

    # --- 5. Phase B: growing prominent segment + lambda refinement ----------
    if max_outer_steps is None:
        max_outer_steps = max(1, min(20, int(np.ceil((seg_rate_max - seg_rate_init) / max(seg_rate_step, 1e-6))) + 1))

    diff = np.abs(prototype - sample_cl).sum(axis=0)  # (L,) channel-summed |sample - prototype|

    current_proba = float(_softmax_np(last_scores)[target_class])

    rate = seg_rate_init
    outer_step = 0
    while outer_step < max_outer_steps and rate <= seg_rate_max and current_proba < success_proba:
        seg_len = max(1, int(round(rate * L)))
        region = _prominent_segment(diff, seg_len)

        cf_np = sample_cl.copy()
        last_scores = scores_orig
        for lam_step in range(max_lam_steps):
            result = run(cf_np, lam, max_iter, region=region, patience=early_stop)
            cf_np = result["cf"]
            last_scores = result["scores"]

            if result["best_cf"] is not None and result["best_dist"] < best_dist:
                best_dist = result["best_dist"]
                best_cf = result["best_cf"]
                best_scores = result["best_scores"]

            # simplified lambda bisection: push toward proximity when we found
            # several valid steps, relax it (favour validity) otherwise
            if result["n_hits"] >= 5:
                lam *= 1.5
            else:
                lam /= 2.0

            current_proba = float(_softmax_np(last_scores)[target_class])
            if current_proba >= success_proba:
                break

        if verbose:
            print(
                f"[TopGrad-CF] outer_step={outer_step} rate={rate:.2f} "
                f"seg={region} target_proba={current_proba:.3f}"
            )

        rate += seg_rate_step
        outer_step += 1

    # --- 6. Pick the best counterfactual found ------------------------------
    if best_cf is not None:
        cf, scores_cf = best_cf, best_scores
    elif int(np.argmax(last_scores)) == target_class:
        cf, scores_cf = cf_np, last_scores
    else:
        if verbose:
            print(
                f"[TopGrad-CF] Warning: counterfactual did not reach class "
                f"{target_class} within the search budget. Returning best attempt."
            )
        cf, scores_cf = cf_np, last_scores

    return revert_orientation(cf, ori), scores_cf
