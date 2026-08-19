"""
TimeX-CF: Wachter-style counterfactual with a DTW class-prototype term.

NOTE ON NAMING — read this before using either "TimeX" in this repo:

This is **not** the same algorithm as `cfts/cf_timex/timex.py`'s
``timex_explanation``, which wraps Harvard's "TimeX: Encoding Time-Series
Explanations through Self-Supervised Model Behavior Consistency"
(arXiv:2306.02109, github.com/mims-harvard/TimeX) — a saliency/attribution
method that requires a separately pretrained surrogate model and produces a
saliency map, not a counterfactual.

Both are called "TimeX" by their respective authors, but they are unrelated
papers. The collision is inherited from the benchmark this module was
written to compare against — TS-Counterfactual-Explanation-Bake-off
(https://github.com/Luckilyeee/TS-Counterfactual-Explanation-Bake-off,
companion code to "Counterfactual Explanation Bake-off: A Review and
Experimental Evaluation for Time Series Classification", Machine Learning
Journal 2026) — whose own "TimeX" method
(``Wachter_TimeX_SG/mainTimeX.py``, citing
https://sites.google.com/view/timex-cf) is the one implemented here, as
``timex_cf``.

Reading bake-off's vendored ``alibi.explainers.tfcounterfactual_timex
.TFCounterFactual`` source (and the plain Wachter
``alibi.explainers.counterfactual.Counterfactual`` it's structurally
derived from) gives the *exact* loss composition, which is important to get
right because it differs from what a "Wachter loss" is often assumed to
look like:

    loss_pred = (softmax(cf)[target_class] - target_proba) ** 2
    loss_dist = lam * (L1(x, cf) + L1(prototype, cf))
    loss      = loss_pred + loss_dist

i.e. **lam scales the distance terms, not the prediction term.** This is
the correct direction for what bake-off's outer lambda-bisection search
actually does: *decrease* lam when no valid CF is found (cheapen movement
so the optimiser is freer to chase validity) and *increase* it when one is
found (penalise movement more, pulling back toward the original for a
sparser/more-proximal solution) — bisecting toward the smallest lam that
still finds a valid CF. `dist_proto` (the addition specific to TimeX) is a
second, unweighted-relative-to-`dist`, DTW-barycenter-of-the-target-class
proximity term folded into that same `lam`-scaled distance sum.

Two more fidelity notes from reading the actual run configuration
(``mainTimeX.py``): at the settings bake-off's own script uses,
**``max_lam_steps=1``** — i.e. no bisection actually runs; a single fixed
``lam_init=0.1`` is used — and the optimiser is Adam with
``learning_rate_init=0.1`` decayed linearly to ~0 over ``max_iter=500``
steps (``tf.train.polynomial_decay``, default power 1). This module matches
both: `timex_cf` uses one fixed `lam` (no growth) and a linearly-decayed
learning rate, rather than porting the (at these settings, unused)
bisection loop.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from cfts.cf__abstract.abstract import (
    batched_predict,
    detach_to_numpy,
    ensure_ncl,
    numpy_to_torch,
    revert_orientation,
    subsample_dataset,
)


def _manhattan(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(x - y))


def _euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((x - y) ** 2))


def compute_dtw_prototype(
    ts: np.ndarray,
    label_data: np.ndarray,
    target_class: int,
    proto_max_samples: int = 30,
    dtw_max_iter: int = 10,
) -> np.ndarray:
    """DTW-barycenter of up to `proto_max_samples` target-class instances.

    Parameters
    ----------
    ts: (N, C, L) array of reference series.
    label_data: (N,) predicted/true class index for each series in `ts`.
    target_class: class whose instances the barycenter is computed over.
    proto_max_samples: cap on how many target-class series feed the DBA
        (DBA cost grows with both sample count and series length).
    dtw_max_iter: iterations for tslearn's internal DBA refinement.

    Returns
    -------
    prototype : np.ndarray, shape (C, L), float32.
    """
    from tslearn.barycenters import dtw_barycenter_averaging

    mask = label_data == target_class
    target_series = ts[mask][:proto_max_samples]  # (M, C, L)
    if len(target_series) == 0:
        raise ValueError(
            f"No dataset samples were classified as target_class={target_class}; "
            "cannot build a DTW prototype for it."
        )

    C, L = ts.shape[1], ts.shape[2]
    # tslearn wants a list/array of (sz, d) series, i.e. (M, L, C).
    target_series_lc = target_series.transpose(0, 2, 1)
    proto_lc = dtw_barycenter_averaging(
        list(target_series_lc), barycenter_size=L, max_iter=dtw_max_iter
    )
    return proto_lc.T.reshape(C, L).astype(np.float32)


def timex_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    lam: float = 0.1,
    target_proba: float = 1.0,
    lambda_proto: float = 1.0,
    max_iter: int = 500,
    learning_rate_init: float = 0.1,
    max_samples: int | None = None,
    proto_max_samples: int = 30,
    dtw_max_iter: int = 10,
    distance: str = "manhattan",
    prototype: np.ndarray | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bake-off-style TimeX counterfactual: Wachter loss + DTW-prototype term.

    Loss minimised per step (see module docstring for why lam lands here
    and not on the prediction term)::

        loss_pred = (softmax(cf)[target_class] - target_proba) ** 2
        loss_dist = lam * (dist(x, cf) + lambda_proto * dist(prototype, cf))
        loss      = loss_pred + loss_dist

    with a single fixed `lam` (no bisection — matches bake-off's own
    `mainTimeX.py` run configuration, which sets `max_lam_steps=1`) and an
    Adam learning rate linearly decayed from `learning_rate_init` to ~0 over
    `max_iter` steps (matches `tf.train.polynomial_decay`'s default power=1
    behaviour, which is what `TFCounterFactual` uses).

    Parameters
    ----------
    sample, model, target_class:
        Same semantics as every other method in this repository — see
        `cfts/cf__abstract/abstract.py`'s `CFMethod` contract.
    dataset:
        Training data used both for the DTW prototype and (indirectly) for
        determining `target_class` when it isn't given. Required.
    lam:
        Fixed weight on the distance terms (bake-off default: `0.1`).
        Larger values keep the counterfactual closer to `x` / the
        prototype at the cost of making validity harder to reach; smaller
        values do the opposite.
    target_proba:
        Desired softmax probability for `target_class` (bake-off default:
        `1.0`, i.e. push for maximum confidence, not just a bare flip).
    lambda_proto:
        Relative weight of the prototype term versus the query-proximity
        term inside `loss_dist`. `0.0` reduces this exactly to plain
        Wachter (query-proximity only, no prototype pull).
    max_iter:
        Number of gradient steps (bake-off default: `500`).
    learning_rate_init:
        Initial Adam learning rate, linearly decayed to ~0 over `max_iter`
        steps (bake-off default: `0.1`).
    max_samples:
        Optional cap on how much of `dataset` is used (stratified
        subsample) before prototype computation and target-class inference.
    proto_max_samples:
        Cap on how many target-class series feed the DTW barycenter.
    dtw_max_iter:
        Iterations for tslearn's internal DBA refinement when building the
        prototype.
    distance:
        `"manhattan"` (L1, bake-off's default) or `"euclidean"` (L2).
    prototype:
        Optional pre-computed prototype, shape (C, L), to use instead of
        recomputing one from `dataset`. The prototype only depends on
        `target_class` and the training set, not on the specific query, so
        callers running many queries that share a target class can compute
        it once with :func:`compute_dtw_prototype` and pass it here to skip
        the (relatively expensive) per-call DBA — same caching argument as
        this repo's `sg_cf_fast`/`time_cf_generate_fast` make for their own
        per-call-but-query-independent setup work.
    verbose:
        Print per-iteration diagnostics when True.

    Returns
    -------
    counterfactual : np.ndarray, same shape/orientation as `sample`, or None
        if optimisation never produced a candidate (should not normally
        happen — the best-loss candidate is always kept, valid or not).
    scores : np.ndarray, shape (num_classes,), or None.
    """
    if dataset is None:
        raise ValueError("timex_cf requires a dataset to compute the target-class DTW prototype.")

    device = next(model.parameters()).device

    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)

    sample_cl, ts, ori = ensure_ncl(np.asarray(sample, dtype=np.float32), dataset)
    C, L = sample_cl.shape

    sample_tensor = numpy_to_torch(sample_cl.reshape(1, C, L), device)

    with torch.no_grad():
        y_orig = detach_to_numpy(model(sample_tensor)).reshape(-1)
    label_orig = int(np.argmax(y_orig))

    if target_class is None:
        sorted_idx = np.argsort(y_orig)[::-1]
        target_class = int(sorted_idx[1])

    if target_class == label_orig:
        raise ValueError(
            f"target_class ({target_class}) is the same as the query's predicted "
            f"class ({label_orig}). Choose a different target class."
        )

    if verbose:
        print(f"TimeX-CF: Original class {label_orig}, Target class {target_class}")

    # --- Target-class DTW prototype (bake-off computes this once, up front,
    # via tslearn.barycenters.dtw_barycenter_averaging) ---
    if prototype is not None:
        prototype_cl = np.asarray(prototype, dtype=np.float32).reshape(C, L)
        if verbose:
            print("TimeX-CF: Using caller-supplied prototype")
    else:
        preds_data = batched_predict(model, ts, device)
        label_data = np.argmax(preds_data, axis=1)
        prototype_cl = compute_dtw_prototype(
            ts, label_data, target_class, proto_max_samples=proto_max_samples, dtw_max_iter=dtw_max_iter
        )
        if verbose:
            print(f"TimeX-CF: Built DTW prototype from up to {proto_max_samples} target-class samples")
    prototype_tensor = numpy_to_torch(prototype_cl.reshape(1, C, L), device)

    dist_fn = _manhattan if distance == "manhattan" else _euclidean
    softmax = nn.Softmax(dim=-1)

    # Bake-off's TFCounterFactual initialises its search at the query `X`
    # itself (`_initialize` -> `init='identity'` -> `X_init = X`).
    cf_tensor = sample_tensor.clone().detach()
    cf_tensor.requires_grad_(True)

    optimizer = Adam([cf_tensor], lr=learning_rate_init)
    target_proba_tensor = torch.tensor(float(target_proba), device=device)

    best_cf = None
    best_pred = None
    best_loss = None

    for iteration in range(max_iter):
        # Linear LR decay to ~0 over max_iter (tf.train.polynomial_decay, power=1).
        current_lr = learning_rate_init * max(0.0, 1.0 - iteration / max_iter)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.zero_grad()
        pred = model(cf_tensor)
        prob_target = softmax(pred)[0, target_class]

        loss_pred = (prob_target - target_proba_tensor) ** 2
        loss_dist = lam * (
            dist_fn(sample_tensor, cf_tensor) + lambda_proto * dist_fn(prototype_tensor, cf_tensor)
        )
        loss = loss_pred + loss_dist
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_cf = detach_to_numpy(model(cf_tensor)).reshape(-1)
        cf_np = detach_to_numpy(cf_tensor.squeeze(0))
        current_loss = float(loss.item())

        if best_loss is None or current_loss < best_loss:
            best_loss = current_loss
            best_cf = cf_np
            best_pred = y_cf

        if verbose and iteration % 100 == 0:
            print(
                f"TimeX-CF iter {iteration}: pred_class={int(np.argmax(y_cf))}, "
                f"target_class={target_class}, loss={current_loss:.4f} "
                f"(pred={float(loss_pred):.4f}, dist={float(loss_dist):.4f}), lr={current_lr:.4f}"
            )

        if int(np.argmax(y_cf)) == target_class and float(prob_target.item()) >= target_proba - 1e-3:
            if verbose:
                print(f"TimeX-CF: Found counterfactual at iteration {iteration}")
            break

    if best_cf is None:
        return None, None

    return revert_orientation(best_cf.reshape(C, L), ori), best_pred
