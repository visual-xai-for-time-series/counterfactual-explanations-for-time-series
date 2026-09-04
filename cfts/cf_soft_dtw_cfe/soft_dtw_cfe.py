# https://github.com/genwro-ai/soft-dtw-counterfactual-explanations
# https://arxiv.org/abs/2603.08349
# https://genwro-ai.github.io/soft-dtw-counterfactual-explanations/

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit

from cfts.cf__abstract.abstract import (
    detach_to_numpy,
    ensure_ncl,
    numpy_to_torch,
    revert_orientation,
    subsample_dataset,
)


####
# Soft-DTW-CFE: Towards Plausibility in Time Series Counterfactual
# Explanations
#
# Paper: Kostrzewa, M., Galus, K., Zięba, M. (2026).
#        "Towards Plausibility in Time Series Counterfactual Explanations."
#        ACIIDS 2026.
#
# Paper URL: https://arxiv.org/abs/2603.08349
# Repository: https://github.com/genwro-ai/soft-dtw-counterfactual-explanations
# Docs: https://genwro-ai.github.io/soft-dtw-counterfactual-explanations/
#
# Algorithm outline (ported from the authors' `soft_dtw_cfe.method` package
# -- `dtw.py` / `soft_dtw_loss.py` / `solver.py` -- to this project's
# CFMethod contract; see "Deviations" below for where this port intentionally
# departs from that reference):
#   1. Build a bank of training series for each class, then pick the
#      `k_neighbors` series closest to the query *from the target class*,
#      using Soft-DTW as the distance. Soft-DTW is Dynamic Time Warping with
#      the path-selecting `min` in its recursion replaced by a soft-min,
#      `-gamma * log(sum(exp(-r / gamma)))`, which makes the whole alignment
#      cost differentiable in its inputs.
#   2. Starting from the query itself, optimise a counterfactual `x_cf` with
#      Adam to minimise a four-term objective every step:
#        - proximity    -- mean squared distance to the original series,
#        - sparsity     -- mean absolute distance to the original series,
#        - validity     -- hinge (or cross-entropy) pressure toward
#                           `target_class`,
#        - plausibility -- mean Soft-DTW distance to the `k_neighbors`
#                           target-class exemplars found in step 1.
#      The plausibility term is the paper's central contribution: it pulls
#      the counterfactual toward the *temporal shape* of real target-class
#      series (via DTW's elastic alignment) rather than only matching their
#      values timestep-by-timestep, as an L1/L2 proximity term alone would.
#      Because Soft-DTW is implemented here as a custom autograd Function
#      (forward/backward recursions below), its gradient participates
#      directly in every Adam step alongside the other three terms.
#   3. Return the series produced after `steps` Adam updates.
#
# Deviations from the reference implementation:
#   - The reference builds its per-class neighbor bank once, offline, from
#     the entire training set (`CounterfactualSolver.compute_class_samples`)
#     and reuses it across many `solve()` calls. This module instead
#     receives a fresh `dataset` on every call (the `<name>_cf` contract
#     used throughout this repository), so the bank is rebuilt each time
#     from a stratified subsample capped at `max_samples` (via
#     `subsample_dataset`, the same "cap dataset size for one call" pattern
#     used by `cf_diffcf`/`cf_topgrad`) rather than the full training set.
#   - `target_class` defaults to the highest-scoring class other than the
#     query's predicted class when not given, matching every other method in
#     this repository -- the reference's `solve()` always requires the
#     caller to pass `y_target` explicitly.
#   - The reference's `select_knn_dtw_batch` groups a *batch* of queries by
#     their (possibly differing) target classes before running k-NN search.
#     This contract only ever solves for one query against one target class
#     at a time, so `_select_knn_dtw` below drops that grouping and searches
#     a single candidate pool directly -- behaviourally identical for a
#     batch of one.
#   - The reference's Isolation-Forest plausibility scoring
#     (`compute_isolation_forest_scores`) and `tqdm` progress bar are
#     dropped: both are evaluation-time instrumentation, not part of the
#     counterfactual-generation objective, and this repository has its own
#     `cfts/metrics` module for post-hoc evaluation.
#   - The Numba-jitted Soft-DTW kernels are compiled with `cache=True` (the
#     reference omits it) so the compilation cost is paid once per machine
#     instead of once per process, matching the convention already used by
#     `cfts/cf_mascots/_fast_borf.py`.
#   - Published results use `lambda_validity=1.0, k_neighbors=10` (the
#     values the reference's own experiments pass to `evaluate_solver.py`);
#     the defaults below instead match its `SolverConfig` dataclass
#     (`lambda_validity=10.0, k_neighbors=5`). Pass the paper's values
#     explicitly to reproduce its reported numbers.
####


# ---------------------------------------------------------------------------
# Soft-DTW: a differentiable relaxation of Dynamic Time Warping.
# Ported from https://github.com/Sleepwalking/pytorch-softdtw, as vendored by
# the reference repo's `soft_dtw_cfe/method/dtw.py`.
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _compute_softdtw(D, gamma):
    """Soft-DTW forward recursion: fill the (N+2, M+2) accumulated-cost
    table `R` for every item in the batch, using a soft-min in place of the
    hard `min` of classical DTW: -gamma * log(sum(exp(-r / gamma)))."""
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for k in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                r0 = -R[k, i - 1, j - 1] / gamma
                r1 = -R[k, i - 1, j] / gamma
                r2 = -R[k, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[k, i, j] = D[k, i - 1, j - 1] + softmin
    return R


@jit(nopython=True, cache=True)
def _compute_softdtw_backward(D_, R, gamma):
    """Soft-DTW backward recursion: propagate d(loss)/d(R[N, M]) back
    through the accumulated-cost table to give d(loss)/d(D[i, j]) for every
    cell of the pairwise distance matrix `D_`."""
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1 : N + 1, 1 : M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = (
                    E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
                )
    return E[:, 1 : N + 1, 1 : M + 1]


class _SoftDTW(torch.autograd.Function):
    """Custom autograd Function wrapping the Numba forward/backward passes
    above: differentiates the Soft-DTW alignment cost w.r.t. the pairwise
    distance matrix `D`, not the raw series. `SoftDTW.forward` below chains
    this with a plain-PyTorch distance matrix so the full path back to the
    input series stays differentiable."""

    @staticmethod
    def forward(ctx, D, gamma):
        dev = D.device
        dtype = D.dtype
        gamma_t = torch.Tensor([gamma]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma_t.item()
        R = torch.Tensor(_compute_softdtw(D_, g_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma_t)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma_t = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma_t.item()
        E = torch.Tensor(_compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None


class SoftDTW(nn.Module):
    """Differentiable Soft-DTW distance between two batches of series.

    Parameters
    ----------
    gamma:
        Soft-min smoothing temperature. ``gamma -> 0`` recovers hard DTW;
        larger values make the alignment (and its gradient) smoother.
    normalize:
        When ``True``, returns the normalized "Soft-DTW divergence"
        ``DTW(x, y) - 1/2 * (DTW(x, x) + DTW(y, y))``, which is non-negative
        and zero iff ``x == y``, rather than the raw (biased) alignment cost.
    """

    def __init__(self, gamma: float = 1.0, normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.func_dtw = _SoftDTW.apply

    def calc_distance_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pairwise squared-Euclidean distance matrix between timesteps of
        `x` (B, N, D) and `y` (B, M, D), via (a-b)^2 = a^2 + b^2 - 2ab --
        memory-efficient for long series compared to broadcasting the full
        (B, N, M, D) difference tensor."""
        x_sq = torch.sum(x**2, dim=-1, keepdim=True)
        y_sq = torch.sum(y**2, dim=-1, keepdim=True)
        dist = (
            x_sq.expand(-1, -1, y.size(1))
            + y_sq.transpose(1, 2).expand(-1, x.size(1), -1)
            - 2 * torch.bmm(x, y.transpose(1, 2).contiguous())
        )
        return torch.clamp(dist, min=0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """`x`, `y`: (B, T, D) or (T, D). Returns (B,) or scalar Soft-DTW
        distance(s)."""
        assert len(x.shape) == len(y.shape)
        squeeze = False
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            squeeze = True
        if self.normalize:
            D_xy = self.calc_distance_matrix(x, y)
            out_xy = self.func_dtw(D_xy, self.gamma)
            D_xx = self.calc_distance_matrix(x, x)
            out_xx = self.func_dtw(D_xx, self.gamma)
            D_yy = self.calc_distance_matrix(y, y)
            out_yy = self.func_dtw(D_yy, self.gamma)
            result = out_xy - 0.5 * (out_xx + out_yy)
        else:
            D_xy = self.calc_distance_matrix(x, y)
            result = self.func_dtw(D_xy, self.gamma)
        return result.squeeze(0) if squeeze else result


# ---------------------------------------------------------------------------
# Per-class neighbor bank and Soft-DTW-based k-NN selection.
# Ported from the reference repo's `soft_dtw_cfe/method/soft_dtw_loss.py`.
# ---------------------------------------------------------------------------

def _build_class_sample_bank(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> dict:
    """Group training series `X` (N, C, L) by their integer label `y` (N,)
    into ``{class_idx: Tensor of shape (N_c, C, L)}``."""
    X = X.to(device)
    y = y.to(device).view(-1)
    bank: dict = {}
    with torch.no_grad():
        for c in range(num_classes):
            mask = y == c
            bank[c] = X[mask].detach() if mask.any() else torch.empty(0, *X.shape[1:], device=device)
    return bank


def _select_knn_dtw(
    x: torch.Tensor,
    candidates: torch.Tensor,
    k: int,
    gamma: float,
    normalize: bool,
) -> torch.Tensor:
    """`k` nearest series to `x` (B, C, L) among `candidates` (N_c, C, L) by
    Soft-DTW distance, computed one candidate at a time to bound memory.

    Returns (B, k, C, L); pads by repeating the farthest kept neighbor when
    fewer than `k` candidates are available.
    """
    B, C, L = x.shape
    device = x.device
    dtw = SoftDTW(gamma=gamma, normalize=normalize)
    x_dtw = x.transpose(1, 2)  # (B, L, C)
    candidates_dtw = candidates.transpose(1, 2)  # (N_c, L, C)

    num_candidates = candidates.size(0)
    distances = torch.zeros(B, num_candidates, device=device)

    with torch.no_grad():
        for i in range(num_candidates):
            cand = candidates_dtw[i : i + 1].expand(B, -1, -1)
            distances[:, i] = dtw(x_dtw, cand)

        k_eff = min(k, num_candidates)
        _, indices = torch.topk(distances, k_eff, largest=False, dim=1)
        neighbors = candidates[indices]  # (B, k_eff, C, L)

        if k_eff < k:
            pad = neighbors[:, -1:, :, :].expand(B, k - k_eff, C, L)
            neighbors = torch.cat([neighbors, pad], dim=1)

    return neighbors


def _mean_soft_dtw_to_neighbors(
    x: torch.Tensor,
    neighbors: torch.Tensor,
    gamma: float,
    normalize: bool,
) -> torch.Tensor:
    """Mean Soft-DTW distance from each series in `x` (B, C, L) to its
    `neighbors` (B, K, C, L) -- the plausibility loss `L_DTW`."""
    B, K, C, L = neighbors.shape
    x_dtw = x.transpose(1, 2)  # (B, L, C)
    neighbors_dtw = neighbors.transpose(2, 3)  # (B, K, L, C)

    dtw = SoftDTW(gamma=gamma, normalize=normalize)
    x_rep = x_dtw.unsqueeze(1).expand(B, K, L, C).reshape(B * K, L, C)
    y_rep = neighbors_dtw.reshape(B * K, L, C)
    dists = dtw(x_rep, y_rep).view(B, K)
    return dists.mean()


def _label(y) -> int:
    """Collapse a one-hot vector or scalar label to an int class index."""
    arr = np.asarray(y)
    return int(np.argmax(arr)) if arr.ndim > 0 and arr.size > 1 else int(arr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def soft_dtw_cfe_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    max_samples: int | None = 200,
    steps: int = 500,
    lr: float = 0.01,
    lambda_proximity: float = 1.0,
    lambda_sparsity: float = 1.0,
    lambda_validity: float = 10.0,
    k_neighbors: int = 5,
    dtw_gamma: float = 1.0,
    dtw_normalize: bool = True,
    p_target_min: float = 0.5,
    valid_objective: str = "hinge",
    weight_decay: float = 0.0,
    seed: int | None = None,
    verbose: bool = False,
    print_every: int = 50,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Soft-DTW-CFE counterfactual for a single time-series sample.

    Implements the :class:`cfts.cf__abstract.abstract.CFMethod` contract; see
    its docstring for the shared parameter/return semantics. Parameters below
    are specific to this implementation -- see the module docstring above for
    the algorithm outline and a list of deliberate deviations from the
    authors' reference implementation.

    Parameters
    ----------
    dataset:
        Training data used to build the per-class Soft-DTW neighbor bank, as
        ``(x, y)`` pairs. Required -- Soft-DTW-CFE has no plausibility-free
        mode.
    max_samples:
        Stratified subsample cap (across all classes) applied to `dataset`
        before building the neighbor bank, for tractability on large
        training sets. ``None`` uses the full dataset.
    steps:
        Number of Adam updates applied to the counterfactual.
    lr:
        Adam learning rate.
    lambda_proximity, lambda_sparsity:
        Weights on the mean-squared (`L_prox`) and mean-absolute (`L_sparse`)
        distance to the original sample.
    lambda_validity:
        Weight on the combined validity + plausibility term,
        ``lambda_validity * (L_valid + L_DTW)``, matching the paper's `L_CF
        = L_prox + L_sparse + lambda * (L_valid + L_DTW)`.
    k_neighbors:
        Number of target-class training series the plausibility term aligns
        the counterfactual to.
    dtw_gamma:
        Soft-DTW smoothing temperature, used both to rank neighbors and in
        the plausibility loss.
    dtw_normalize:
        Use the normalized Soft-DTW divergence (see :class:`SoftDTW`) rather
        than the raw alignment cost, for both neighbor ranking and the loss.
    p_target_min:
        Target-class probability threshold `tau` used by the ``"hinge"``
        validity loss, ``max(0, tau - p(target_class | x_cf))``.
    valid_objective:
        ``"hinge"`` (default) or ``"ce"`` (cross-entropy toward
        `target_class`) validity loss.
    weight_decay:
        Adam weight decay applied to the counterfactual tensor itself.
    seed:
        Seed for PyTorch's RNG, for reproducibility.
    verbose:
        Print per-``print_every``-step diagnostics when ``True``.
    print_every:
        Verbose print interval, in optimisation steps.

    Returns
    -------
    counterfactual : np.ndarray
        Counterfactual time series in the **same shape / orientation** as
        ``sample``.
    scores : np.ndarray, shape (num_classes,)
        Model output (logits / softmax scores) for the counterfactual.

    Example
    -------
    >>> cf, scores = soft_dtw_cfe_cf(sample_np, model, dataset=train_dataset, verbose=True)
    >>> label_cf = int(np.argmax(scores))
    """
    if dataset is None:
        raise ValueError(
            "soft_dtw_cfe_cf requires a dataset to build the target-class "
            "neighbor bank used by the plausibility (Soft-DTW) term."
        )

    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    dataset_sub = dataset
    if max_samples is not None and len(dataset) > max_samples:
        dataset_sub = subsample_dataset(dataset, max_samples)

    sample_cl, ts, ori = ensure_ncl(np.asarray(sample, dtype=np.float32), dataset_sub)
    C, L = sample_cl.shape
    labels = np.array(
        [_label(dataset_sub[i][1]) for i in range(len(dataset_sub))], dtype=np.int64
    )

    # --- original prediction & derived target class --------------------------
    x_orig = numpy_to_torch(sample_cl.reshape(1, C, L), device)
    with torch.no_grad():
        scores_orig = detach_to_numpy(model(x_orig)).reshape(-1)
    num_classes = scores_orig.shape[-1]
    label_orig = int(np.argmax(scores_orig))

    if target_class is None:
        if num_classes == 2:
            target_class = 1 - label_orig
        else:
            ranked = np.argsort(-scores_orig)
            target_class = int(ranked[0] if ranked[0] != label_orig else ranked[1])
    target_class = int(target_class)
    if target_class == label_orig:
        raise ValueError(
            f"target_class ({target_class}) equals the query's predicted "
            f"class ({label_orig}). Choose a different target class."
        )

    # --- per-class neighbor bank & Soft-DTW k-NN ------------------------------
    ts_t = numpy_to_torch(ts, device)
    y_t = torch.from_numpy(labels).to(device)
    bank = _build_class_sample_bank(ts_t, y_t, num_classes, device)

    candidates = bank[target_class]
    if candidates.numel() == 0:
        if verbose:
            print(
                f"[Soft-DTW-CFE] No dataset sample labeled target_class="
                f"{target_class}. Returning original sample unchanged."
            )
        return revert_orientation(sample_cl, ori), scores_orig

    if verbose:
        print(
            f"[Soft-DTW-CFE] original class {label_orig}, target class "
            f"{target_class}, neighbor pool size {candidates.shape[0]}"
        )

    # neighbors are selected once from the original query and stay fixed for
    # the whole optimisation below (matching the reference's `solve()`)
    neighbors = _select_knn_dtw(
        x_orig, candidates, k=k_neighbors, gamma=dtw_gamma, normalize=dtw_normalize
    )  # (1, k, C, L)

    # --- gradient-based optimisation in input space ---------------------------
    x_cf = x_orig.clone().detach().requires_grad_(True)
    target_t = torch.tensor([target_class], device=device, dtype=torch.long)
    optimizer = torch.optim.Adam([x_cf], lr=lr, weight_decay=weight_decay)

    for it in range(steps):
        optimizer.zero_grad(set_to_none=True)

        logits = model(x_cf)
        probs_target = F.softmax(logits, dim=1)[:, target_class]
        if valid_objective == "ce":
            loss_valid = F.cross_entropy(logits, target_t)
        else:
            loss_valid = F.relu(p_target_min - probs_target).mean()

        diff = x_cf - x_orig
        loss_prox = (diff**2).mean()
        loss_sparse = diff.abs().mean()

        loss_plaus = _mean_soft_dtw_to_neighbors(
            x_cf, neighbors, gamma=dtw_gamma, normalize=dtw_normalize
        )

        loss = (
            lambda_proximity * loss_prox
            + lambda_sparsity * loss_sparse
            + lambda_validity * (loss_plaus + loss_valid)
        )
        loss.backward()
        optimizer.step()

        if verbose and (it % print_every == 0 or it == steps - 1):
            print(
                f"[Soft-DTW-CFE] it={it:4d} "
                f"loss_prox={loss_prox.item():.4f} "
                f"loss_sparse={loss_sparse.item():.4f} "
                f"loss_valid={loss_valid.item():.4f} "
                f"loss_plaus={loss_plaus.item():.4f} "
                f"p_target={probs_target.item():.3f}"
            )

    # --- final prediction & return ---------------------------------------------
    with torch.no_grad():
        scores_cf = detach_to_numpy(model(x_cf)).reshape(-1)
    cf_cl = detach_to_numpy(x_cf).reshape(C, L)

    return revert_orientation(cf_cl, ori), scores_cf
