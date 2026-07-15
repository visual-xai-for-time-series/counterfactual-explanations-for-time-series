"""
MASCOTS: Model-Agnostic Symbolic COunterfactual explanations for Time Series.

Uses BorfExplainer (vendored from DawidPludowski/borf) as its engine:

    explainer = BorfExplainer(prediction_fn, prediction_fn_proba)
    explainer.build(X_train)
    cfs, meta = explainer.counterfactual(X_obs, target_cls)

Optional extras (fall back gracefully when absent):
  - ``gpytorch``  – enables ``swap_method="gaussian"``
  - ``shap``      – enables ``attribution_name="shap"``

Paper : Płudowski, D., Spinnato, F., Wilczyński, P., Kotowski, K., Ntagiou, E. V.,
        Guidotti, R., & Biecek, P. (2025). MASCOTS: Model-Agnostic Symbolic
        COunterfactual explanations for Time Series. arXiv:2503.22389.
GitHub: https://github.com/DawidPludowski/borf
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch

from cfts.cf__abstract.abstract import (
    detach_to_numpy,
    numpy_to_torch,
    ensure_ncl,
    revert_orientation,
    subsample_dataset,
)
from ._borf_explainer import BorfExplainer


def mascots_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    max_iter: int = 100,
    swap_method: Literal["scalar", "gaussian"] = "scalar",
    n_restarts: int = 3,
    C: float = 0.1,
    select_top_k: int = 5,
    attribution_name: str = "coef",
    seed: int | None = None,
    max_samples: int | None = None,
    verbose: bool = False,
    borf_config: list | None = None,
    prebuilt_explainer=None,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a counterfactual explanation using the MASCOTS algorithm.

    Parameters
    ----------
    sample:
        Query time series.  Accepts 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``
        NumPy arrays (or anything that converts with ``np.asarray``).
    model:
        PyTorch classifier whose forward pass accepts ``(B, C, L)`` tensors
        and returns ``(B, num_classes)`` logits / probabilities.
    target_class:
        Class index to flip toward.  When ``None`` the second-most-probable
        class under the original prediction is used.
    dataset:
        Sequence of ``(x, y)`` pairs used to build the BoRF surrogate.
        May also be a NumPy array of shape ``(N, C, L)``.  Required.
    max_iter:
        Maximum number of word-swap attempts per restart.
    swap_method:
        ``"scalar"`` (default) or ``"gaussian"`` (requires gpytorch).
    n_restarts:
        Number of independent restarts introduced for diversity.
    C:
        Penalty weight for the word-difference measure.
    select_top_k:
        Top-k swap candidates from which one is chosen at random each step.
    attribution_name:
        ``"coef"`` (default – logistic-regression coefficients, no extra deps)
        or ``"shap"`` (requires the shap package).
    seed:
        Integer seed for reproducibility, or ``None``.
    verbose:
        Print progress when ``True``.

    Returns
    -------
    counterfactual : np.ndarray
        Counterfactual time series in the **same shape / orientation** as
        *sample*.
    scores : np.ndarray, shape (num_classes,)
        Model output for the best counterfactual found.
    """
    if dataset is None:
        raise ValueError("mascots_cf requires a dataset to build the BoRF surrogate.")

    device = next(model.parameters()).device

    # ── normalise inputs ──────────────────────────────────────────────────────
    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)
    sample_cl, ts, ori = ensure_ncl(np.asarray(sample, dtype=np.float32), dataset)
    C_dim, L = sample_cl.shape
    N = ts.shape[0]

    # ── original prediction ───────────────────────────────────────────────────
    with torch.no_grad():
        scores_orig = detach_to_numpy(
            model(numpy_to_torch(sample_cl.reshape(1, C_dim, L), device))
        ).reshape(-1)
    label_orig = int(np.argmax(scores_orig))

    if target_class is None:
        sorted_cls = np.argsort(scores_orig)[::-1]
        target_class = int(sorted_cls[1]) if len(sorted_cls) > 1 else (1 - label_orig)

    if label_orig == target_class:
        return revert_orientation(sample_cl, ori), scores_orig

    # ── build model adapter functions ─────────────────────────────────────────
    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def prediction_fn(X: np.ndarray) -> np.ndarray:
        """(N, C, L) float32 → (N,) int labels."""
        X_t = numpy_to_torch(np.asarray(X, dtype=np.float32), device)
        with torch.no_grad():
            logits = detach_to_numpy(model(X_t))
        return np.argmax(logits, axis=1).astype(int)

    def prediction_fn_proba(X: np.ndarray) -> np.ndarray:
        """(N, C, L) float32 → (N, n_classes) float probabilities."""
        X_t = numpy_to_torch(np.asarray(X, dtype=np.float32), device)
        with torch.no_grad():
            logits = detach_to_numpy(model(X_t))
        return _softmax(logits)

    # ── build BorfExplainer ───────────────────────────────────────────────────
    if prebuilt_explainer is not None:
        explainer = prebuilt_explainer
    else:
        if verbose:
            print(f"[mascots_cf] building BorfExplainer on N={N} samples …")

        kwargs_borf = {} if borf_config is None else {"borf_config": borf_config}
        explainer = BorfExplainer(prediction_fn, prediction_fn_proba, **kwargs_borf)

        build_metrics = explainer.build(
            ts,
            attribution_name=attribution_name,
            seed=seed if seed is not None else 42,
        )

        if verbose:
            print(f"[mascots_cf] surrogate metrics: {build_metrics}")

    # ── generate counterfactuals ──────────────────────────────────────────────
    X_obs = sample_cl.reshape(1, C_dim, L).astype(np.float64)

    cfs, _ = explainer.counterfactual(
        X_obs,
        target_cls=target_class,
        swap_method=swap_method,
        max_borf_changes=max_iter,
        C=C,
        select_top_k=select_top_k,
        n_restarts=n_restarts,
        returns_meta=True,
        seed=seed,
    )

    # ── pick best result ──────────────────────────────────────────────────────
    best_cf = sample_cl.copy()
    best_scores = scores_orig.copy()
    best_prob = float(scores_orig[target_class])

    for cf_i in cfs:
        cf_cl = cf_i.astype(np.float32)
        with torch.no_grad():
            s = detach_to_numpy(
                model(numpy_to_torch(cf_cl.reshape(1, C_dim, L), device))
            ).reshape(-1)
        prob = _softmax(s.reshape(1, -1))[0, target_class]
        if prob > best_prob:
            best_prob = prob
            best_cf = cf_cl.copy()
            best_scores = s.copy()

    if verbose:
        print(
            f"[mascots_cf] done – original={label_orig}  "
            f"cf={int(np.argmax(best_scores))}  "
            f"p_target={best_prob:.3f}"
        )

    return revert_orientation(best_cf, ori), best_scores
