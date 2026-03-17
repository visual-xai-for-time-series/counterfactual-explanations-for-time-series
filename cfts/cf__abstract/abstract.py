"""
Reference / abstract implementation – template for new CF methods.

This module provides:
  1. Shared utility helpers (tensor conversion, shape normalisation) that every
     concrete method in this library uses and can import from here.
  2. ``abstract_cf`` – a minimal but fully functional counterfactual function
     that follows the exact same signature pattern as every other method in the
     repository (native_guide_uni_cf, glacier_cf, cem_cf, …).  It is
     deliberately trivial (iterative Gaussian noise) so contributors can copy
     it as a starting point.

Implementing a new method
--------------------------
Copy this file, rename the module and the top-level ``<name>_cf`` function,
then replace the body of the core loop with the real algorithm.  Keep the same
return signature: ``(counterfactual, scores)`` where both are NumPy arrays and
``counterfactual`` has the same shape / orientation as the input ``sample``.

Minimal skeleton
----------------
    import numpy as np
    import torch
    from cfts.cf__abstract.abstract import (
        detach_to_numpy, numpy_to_torch, ensure_cl, revert_orientation,
    )

    def my_cf(sample, model, max_iter=200, verbose=False):
        device = next(model.parameters()).device
        sample_cl, ori = ensure_cl(np.asarray(sample, dtype=np.float32))
        C, L = sample_cl.shape

        # --- original prediction ---
        with torch.no_grad():
            scores_orig = detach_to_numpy(
                model(numpy_to_torch(sample_cl.reshape(1, C, L), device))
            ).reshape(-1)
        label_orig = int(np.argmax(scores_orig))

        cf = sample_cl.copy()
        scores_cf = scores_orig.copy()

        for i in range(max_iter):
            # ... your algorithm here ...
            pass

        return revert_orientation(cf, ori), scores_cf
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shared tensor / array helpers
#   These are duplicated in several concrete modules; they live here so that
#   new implementations can import from one place.
# ---------------------------------------------------------------------------

def detach_to_numpy(data: torch.Tensor) -> np.ndarray:
    """Move a PyTorch tensor to CPU, detach from the computation graph, and
    return it as a NumPy array."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a NumPy array to a float32 PyTorch tensor on *device*."""
    return torch.from_numpy(np.asarray(data, dtype=np.float32)).to(device)


# ---------------------------------------------------------------------------
# Shape normalisation helpers
#   Every concrete method receives a sample whose orientation may vary (1-D
#   array, (C, L), (L, C)).  The helpers below canonicalise to (C, L) for
#   internal processing and revert at the end.
# ---------------------------------------------------------------------------

def ensure_cl(sample: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return *sample* normalised to shape (C, L) and an orientation tag.

    Orientation tags
    ----------------
    ``"1d"``  – original was a 1-D array  → reverted with  ``.reshape(-1)``
    ``"cl"``  – original was (C, L)        → returned as-is
    ``"lc"``  – original was (L, C)        → reverted with  ``.T``

    Heuristic for 2-D arrays: if  rows ≤ cols  assume (C, L), otherwise (L, C).
    """
    s = np.asarray(sample, dtype=np.float32)
    if s.ndim == 1:
        return s.reshape(1, -1), "1d"
    if s.ndim == 2:
        r, c = s.shape
        if r <= c:
            return s.copy(), "cl"
        return s.T.copy(), "lc"
    raise ValueError(
        f"sample must be a 1-D or 2-D array, got shape {s.shape}"
    )


def revert_orientation(arr_cl: np.ndarray, ori: str) -> np.ndarray:
    """Undo the normalisation applied by :func:`ensure_cl`.

    Parameters
    ----------
    arr_cl:
        Array in (C, L) layout.
    ori:
        Orientation tag returned by :func:`ensure_cl`.
    """
    if ori == "1d":
        return arr_cl.reshape(-1)
    if ori == "lc":
        return arr_cl.T.copy()
    return arr_cl  # "cl" – already correct


def ensure_ncl(
    sample: np.ndarray,
    dataset,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Normalise *sample* to (C, L) and *dataset* items to (N, C, L).

    Parameters
    ----------
    sample:
        The query time series; 1-D or 2-D.
    dataset:
        A sequence of (x, y) pairs where each *x* is a time series.  May also
        be a NumPy array of shape (N, C, L).

    Returns
    -------
    sample_cl : np.ndarray, shape (C, L)
    ts        : np.ndarray, shape (N, C, L)
    ori       : str  – orientation tag (see :func:`ensure_cl`)
    """
    sample_cl, ori = ensure_cl(sample)
    C, L = sample_cl.shape

    first_x = np.asarray(dataset[0][0])
    if first_x.ndim == 1:
        ts = np.stack(
            [np.asarray(item[0], dtype=np.float32).reshape(1, -1) for item in dataset],
            axis=0,
        )
    elif first_x.ndim == 2:
        r, c = first_x.shape
        if r <= c:
            ts = np.stack(
                [np.asarray(item[0], dtype=np.float32) for item in dataset], axis=0
            )
        else:
            ts = np.stack(
                [np.asarray(item[0], dtype=np.float32).T for item in dataset], axis=0
            )
    else:
        raise ValueError("Dataset items must be 1-D or 2-D time series.")

    if ts.shape[-1] != L:
        raise ValueError(
            f"Dataset series length {ts.shape[-1]} != sample length {L}."
        )

    C_data = ts.shape[1]
    if C_data != C:
        if C_data == 1:
            ts = np.repeat(ts, C, axis=1)
        else:
            raise ValueError(
                f"Channel mismatch: sample has {C} channels, dataset has {C_data}."
            )

    return sample_cl, ts, ori


####
# abstract_cf – Reference / template implementation (not a research method)
#
# Paper: N/A – this is a template, not a published algorithm.
#
# Strategy:
#   Repeatedly add small Gaussian noise to the input until the predicted class
#   changes.  Each unsuccessful attempt doubles the noise scale after every
#   ``escalate_every`` iterations.  This is the simplest possible
#   "search in input space" strategy and serves only as a copy-paste template
#   – replace the core loop with the real algorithm.
####

def abstract_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    max_iter: int = 200,
    noise_scale: float = 0.05,
    escalate_every: int = 10,
    seed: int | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference counterfactual using iterative random Gaussian perturbation.

    Follows the exact same signature pattern as every other CF method in this
    repository (native_guide_uni_cf, glacier_cf, cem_cf, …) so it plugs
    straight into the existing evaluation and example scripts.

    Parameters
    ----------
    sample:
        Query time series.  Accepts 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``
        NumPy arrays (or anything that converts with ``np.asarray``).
    model:
        PyTorch classifier whose forward pass accepts ``(B, C, L)`` tensors
        and returns ``(B, num_classes)`` logits / probabilities.
    max_iter:
        Maximum number of perturbation attempts before giving up.
    noise_scale:
        Initial standard deviation of the Gaussian noise added per attempt.
    escalate_every:
        Double the noise scale after this many failed attempts.
    seed:
        Integer seed for reproducibility, or ``None`` for random behaviour.
    verbose:
        Print per-iteration diagnostics when ``True``.

    Returns
    -------
    counterfactual : np.ndarray
        Counterfactual time series in the **same shape / orientation** as
        *sample*.
    scores : np.ndarray, shape (num_classes,)
        Model output (logits or softmax scores) for the counterfactual.

    Example
    -------
    >>> cf, scores = abstract_cf(sample_np, model, max_iter=200, verbose=True)
    >>> label_orig = int(np.argmax(scores_orig))
    >>> label_cf   = int(np.argmax(scores))
    >>> print(f"class flip: {label_orig} -> {label_cf}")
    """
    device = next(model.parameters()).device

    # --- normalise input to (C, L) -----------------------------------------
    sample_cl, ori = ensure_cl(np.asarray(sample, dtype=np.float32))
    C, L = sample_cl.shape

    # --- original prediction ------------------------------------------------
    with torch.no_grad():
        scores_orig = detach_to_numpy(
            model(numpy_to_torch(sample_cl.reshape(1, C, L), device))
        ).reshape(-1)
    label_orig = int(np.argmax(scores_orig))

    rng = np.random.default_rng(seed)
    cf = sample_cl.copy()
    scores_cf = scores_orig.copy()
    scale = noise_scale

    # --- core loop ----------------------------------------------------------
    # Replace this loop with the actual algorithm when implementing a new method.
    for i in range(max_iter):
        candidate = sample_cl + rng.normal(0.0, scale, size=(C, L)).astype(np.float32)

        with torch.no_grad():
            scores_candidate = detach_to_numpy(
                model(numpy_to_torch(candidate.reshape(1, C, L), device))
            ).reshape(-1)
        label_candidate = int(np.argmax(scores_candidate))

        if verbose:
            print(
                f"[abstract_cf] iter {i:3d}  "
                f"noise_scale={scale:.4f}  "
                f"predicted={label_candidate}  "
                f"original={label_orig}"
            )

        if label_candidate != label_orig:
            cf = candidate
            scores_cf = scores_candidate
            break

        # escalate noise magnitude when stuck
        if escalate_every > 0 and (i + 1) % escalate_every == 0:
            scale *= 2.0

    # --- revert to original orientation and return --------------------------
    return revert_orientation(cf, ori), scores_cf
