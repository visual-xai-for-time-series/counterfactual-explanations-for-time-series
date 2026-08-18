from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

from cfts.cf__abstract.abstract import (
    batched_predict,
    detach_to_numpy,
    ensure_ncl,
    numpy_to_torch,
    revert_orientation,
)


####
# FastPACE: Fast PlAnning of Counterfactual Explanations for Time Series Classification
#
# Paper: Refoyo, M., Boleas, Y., & Luengo, D. (2026).
#        "FastPACE: Fast PlAnning of Counterfactual Explanations for Time Series
#        Classification." Data Mining and Knowledge Discovery.
#        https://doi.org/10.1007/s10618-026-01242-7
#        (preprint: Research Square, https://doi.org/10.21203/rs.3.rs-8611408/v1)
#
# Repository: https://github.com/MarioRefoyo/FastPACE
#
# Algorithm outline (Sections 3.1-3.5 of the paper):
#   1. Find the Nearest Unlike Neighbor (NUN): the closest training example, in
#      Euclidean distance, that the black-box classifier predicts as a different
#      class (Eq. 1). A counterfactual x' is built by replacing a subset of
#      (time step, channel) entries of x with the NUN's values at those entries,
#      encoded by a binary mask M.
#   2. Cast counterfactual generation as an episodic MDP: the state is the current
#      mask M, actions flip entries of M (XOR), and the initial mask is all-ones
#      (x' == NUN), which is valid *by construction* since the NUN is, by
#      definition, predicted as a different class (Section 3.2).
#   3. Solve the MDP with Model Predictive Control: at every step, plan a
#      finite-horizon action sequence with the Cross-Entropy Method (CEM,
#      Algorithm 1), execute only the first action, and replan (Section 3.3-3.4).
#   4. To keep the action space tractable, actions operate on *blocks* — contiguous
#      groups of time steps combined with clusters of similarly-behaving channels
#      — rather than individual (time step, channel) entries. FastPACE runs this
#      block-CEM planner across a coarse-to-fine sequence of granularity levels,
#      each level starting from the previous level's best mask (Section 3.5,
#      Algorithm 2).
#   5. The planning objective (Eq. 2) is the same weighted combination of terms
#      used in Sub-SpaCE/Multi-SpaCE: adversarial probability, sparsity,
#      contiguity, and a plausibility term based on the Increase in Outlier Score
#      (IOS) from a reconstruction autoencoder, plus a large penalty when the
#      candidate is not (yet) predicted as the target class.
#   6. Validity by design: because the trajectory always starts at the NUN (valid)
#      and each candidate mask along the way is checked against the classifier,
#      the last mask found to be valid is kept as a fallback, so the returned
#      counterfactual always flips the prediction to the target class.
####


# ---------------------------------------------------------------------------
# Plausibility autoencoder (Section 3.1, "Plausibility Loss")
# ---------------------------------------------------------------------------

class _ConvAutoencoder1D(nn.Module):
    """Small conv1d encoder/decoder trained per-dataset to score plausibility.

    This is the ``f_AE`` block of Figure 1 in the paper: not part of the
    black-box classifier, just a reconstruction model trained once on
    in-distribution samples so that the Increase-in-Outlier-Score (IOS)
    plausibility term can penalize counterfactuals that reconstruct worse
    than the original instance.
    """

    def __init__(self, channels: int, latent_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(self.encoder(x))
        if out.shape[-1] != x.shape[-1]:
            # Stride-2 convs round the length down/up depending on parity; snap
            # back to the exact input length rather than constraining callers
            # to lengths that round-trip cleanly through two stride-2 layers.
            out = F.interpolate(out, size=x.shape[-1], mode="linear", align_corners=False)
        return out


def train_plausibility_autoencoder(
    ts: np.ndarray,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = False,
) -> Tuple[nn.Module, float]:
    """Train the conv autoencoder used for the IOS plausibility term.

    Parameters
    ----------
    ts:
        Reference (in-distribution) time series, shape ``(N, C, L)``. Typically
        the training split of the dataset being explained.
    device:
        Device to train on.

    Returns
    -------
    autoencoder:
        Trained ``nn.Module``.
    e_max:
        Maximum per-sample reconstruction error observed on ``ts`` — "the
        maximum reconstruction error on the training set" (Section 3.1), used
        to normalize the plausibility term into a comparable range.
    """
    N, C, L = ts.shape
    ae = _ConvAutoencoder1D(C).to(device)
    optimizer = Adam(ae.parameters(), lr=lr)
    ts_t = numpy_to_torch(ts, device)

    ae.train()
    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            batch = ts_t[idx]
            optimizer.zero_grad()
            loss = torch.mean((ae(batch) - batch) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose and epochs > 0 and epoch % max(1, epochs // 5) == 0:
            print(f"[FastPACE AE] epoch {epoch}/{epochs}: loss={epoch_loss:.6f}")

    ae.eval()
    with torch.no_grad():
        errors = detach_to_numpy(torch.mean((ae(ts_t) - ts_t) ** 2, dim=(1, 2)))
    e_max = float(errors.max()) + 1e-8
    return ae, e_max


def _reconstruction_error(ae: nn.Module, x_t: torch.Tensor) -> torch.Tensor:
    """Per-sample reconstruction MSE. ``x_t``: (B, C, L) -> (B,)."""
    with torch.no_grad():
        return torch.mean((ae(x_t) - x_t) ** 2, dim=(1, 2))


# ---------------------------------------------------------------------------
# Nearest Unlike Neighbor (Section 3.1, Eq. 1)
# ---------------------------------------------------------------------------

def _find_nun(
    sample_cl: np.ndarray,
    ts: np.ndarray,
    labels_pool: np.ndarray,
    label_sample: int,
    target_class: Optional[int],
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Nearest instance (Euclidean, over the *predicted* labels) with a different
    class than ``label_sample`` (or, if given, the closest instance predicted as
    exactly ``target_class``)."""
    if target_class is not None:
        pool_mask = labels_pool == target_class
    else:
        pool_mask = labels_pool != label_sample
    if not np.any(pool_mask):
        return None, None

    pool = ts[pool_mask]
    pool_labels = labels_pool[pool_mask]
    neigh = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(pool.reshape(len(pool), -1))
    _, idx = neigh.kneighbors(sample_cl.reshape(1, -1))
    nun_idx = int(idx[0, 0])
    return pool[nun_idx], int(pool_labels[nun_idx])


# ---------------------------------------------------------------------------
# Hierarchical block-based action space (Section 3.5)
# ---------------------------------------------------------------------------

def _contiguous_blocks(size: int, n_blocks: int) -> np.ndarray:
    """Assign each of `size` positions to one of (up to) `n_blocks` contiguous
    groups. Used for the temporal partition G_L."""
    n_blocks = max(1, min(n_blocks, size))
    block_id = np.empty(size, dtype=int)
    for b, idxs in enumerate(np.array_split(np.arange(size), n_blocks)):
        block_id[idxs] = b
    return block_id


def _cluster_channels(ts: np.ndarray, n_clusters: int) -> np.ndarray:
    """Group channels that evolve similarly via agglomerative clustering on
    concatenated per-channel training series (Section 3.5, G_C)."""
    N, C, L = ts.shape
    if C == 1:
        return np.zeros(1, dtype=int)
    if n_clusters >= C:
        return np.arange(C)
    flat = np.transpose(ts, (1, 0, 2)).reshape(C, -1)  # (C, N*L)
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", metric="euclidean").fit_predict(flat)
    return labels


def _granularity_to_blocks(frac_ts: float, frac_ch: float, L: int, C: int) -> Tuple[int, int]:
    """Convert a [TS_frac, CH_frac] granularity level (Section 4.1) to block counts."""
    n_time = max(1, min(L, round(1.0 / frac_ts))) if frac_ts > 0 else 1
    n_channel = max(1, min(C, round(1.0 / frac_ch))) if frac_ch > 0 else 1
    return n_time, n_channel


# ---------------------------------------------------------------------------
# Objective function O(x, M, xnun, ynun) (Eq. 2)
# ---------------------------------------------------------------------------

def _compute_objective(
    masks: np.ndarray,  # (B, C, L) bool
    x: np.ndarray,  # (C, L)
    xnun: np.ndarray,  # (C, L)
    model: nn.Module,
    device: torch.device,
    target_class: int,
    ae: Optional[nn.Module],
    e_max: float,
    err_x: float,
    alpha: float,
    beta: float,
    eta: float,
    lam: float,
    delta: float,
    invalid_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched evaluation of Eq. (2) for a batch of candidate masks.

    Returns ``(objective, predicted_class)``, each shape ``(B,)``.
    """
    B, C, L = masks.shape
    x_b = np.broadcast_to(x, (B, C, L))
    xnun_b = np.broadcast_to(xnun, (B, C, L))
    cfs = np.where(masks, xnun_b, x_b).astype(np.float32)  # Eq. (1)

    cfs_t = numpy_to_torch(cfs, device)
    with torch.no_grad():
        probs = detach_to_numpy(model(cfs_t))
    pred_class = np.argmax(probs, axis=1)
    l_adv = probs[:, target_class]

    l_spa = masks.reshape(B, -1).sum(axis=1) / (C * L)

    # New-subsequence beginnings: M_{i-1,j}=0, M_{i,j}=1 (i from the 2nd time step
    # onward, i.e. i>=2 in the paper's 1-indexed notation -> index 1..L-1 here).
    starts = masks[:, :, 1:] & (~masks[:, :, :-1])
    n_subseq = starts.reshape(B, -1).sum(axis=1).astype(np.float64)
    l_sub = np.power(n_subseq / ((C * L) / 2.0), delta)

    if ae is not None:
        err_cf = detach_to_numpy(_reconstruction_error(ae, cfs_t))
        # "Increase" in outlier score: zero when the CF reconstructs at least as
        # well as the original, positive when it reconstructs worse. (The
        # extracted preprint text renders this as min(0, ...), which would
        # instead reward better reconstruction and never penalize worse
        # reconstruction — inconsistent with the paper's own description,
        # "penalizes counterfactuals whose reconstruction error increases", and
        # with plausibility/outlier scores being non-negative in Section 4.4.)
        l_ios = np.maximum(0.0, err_cf - err_x) / e_max
    else:
        l_ios = np.zeros(B, dtype=np.float64)

    invalid = (pred_class != target_class).astype(np.float64)
    objective = alpha * l_adv - beta * l_spa - eta * l_sub - lam * l_ios - invalid_penalty * invalid
    return objective, pred_class


def _mask_is_valid(
    mask: np.ndarray, x: np.ndarray, xnun: np.ndarray, model: nn.Module, device: torch.device, target_class: int
) -> bool:
    cf = np.where(mask, xnun, x).astype(np.float32)
    with torch.no_grad():
        probs = detach_to_numpy(model(numpy_to_torch(cf.reshape(1, *cf.shape), device)))
    return int(np.argmax(probs[0])) == target_class


# ---------------------------------------------------------------------------
# Cross-Entropy Method planning (Section 3.4, Algorithm 1)
# ---------------------------------------------------------------------------

def _plan_cem(
    mask: np.ndarray,
    x: np.ndarray,
    xnun: np.ndarray,
    model: nn.Module,
    device: torch.device,
    target_class: int,
    ae: Optional[nn.Module],
    e_max: float,
    err_x: float,
    time_block_id: np.ndarray,
    channel_block_id: np.ndarray,
    n_time_blocks: int,
    n_channel_blocks: int,
    horizon: int,
    n_samples: int,
    cem_iterations: int,
    elite_fraction: float,
    smoothing: float,
    alpha: float,
    beta: float,
    eta: float,
    lam: float,
    delta: float,
    invalid_penalty: float,
    rng: np.random.Generator,
) -> int:
    """One MPC planning call: return the first action of the best CEM trajectory.

    The last action index (``n_time_blocks * n_channel_blocks``) is STOP.
    """
    n_actions = n_time_blocks * n_channel_blocks + 1
    stop_idx = n_actions - 1
    horizon = min(horizon, n_actions)
    pi = np.full(n_actions, 1.0 / n_actions)

    def action_to_mask(a: int) -> np.ndarray:
        gl, gc = divmod(a, n_channel_blocks)
        return (time_block_id[None, :] == gl) & (channel_block_id[:, None] == gc)

    best_traj: Optional[np.ndarray] = None
    best_score = -np.inf

    for _ in range(cem_iterations):
        trajectories = np.stack(
            [rng.choice(n_actions, size=horizon, replace=False, p=pi) for _ in range(n_samples)]
        )  # (n_samples, horizon)

        # Joint action mask per trajectory = XOR of its per-step block masks
        # (Eq. 7); STOP is a no-op that never touches the mask.
        joint = np.zeros((n_samples, *mask.shape), dtype=bool)
        for a in range(n_actions - 1):
            hits = (trajectories == a).any(axis=1)
            if hits.any():
                joint[hits] ^= action_to_mask(a)

        terminal_masks = mask[None] ^ joint
        scores, _ = _compute_objective(
            terminal_masks, x, xnun, model, device, target_class, ae, e_max, err_x, alpha, beta, eta, lam, delta,
            invalid_penalty,
        )

        n_elite = max(1, int(np.ceil(elite_fraction * n_samples)))
        elite_idx = np.argsort(scores)[-n_elite:]
        elite_scores = scores[elite_idx]

        iter_best = elite_idx[-1]
        if scores[iter_best] > best_score:
            best_score = scores[iter_best]
            best_traj = trajectories[iter_best]

        # Fitness-weighted update (Eq. 9-10 / Algorithm 1 lines 13-17). The
        # objective can be negative (invalid-class penalty, sparsity/contiguity
        # terms), so elite scores are shifted to be non-negative before
        # normalizing into weights — the paper's w_i = G_i / sum(G_elite) is only
        # well-behaved when scores are already non-negative.
        shifted = elite_scores - elite_scores.min() + 1e-8
        weights = shifted / shifted.sum()
        pi_hat = np.zeros(n_actions)
        for w, traj in zip(weights, trajectories[elite_idx]):
            pi_hat[traj] += w
        pi = smoothing * pi_hat + (1 - smoothing) * pi
        pi = (pi + 1e-8) / (pi + 1e-8).sum()

    return int(best_traj[0]) if best_traj is not None else stop_idx


def _run_level(
    mask: np.ndarray,
    x: np.ndarray,
    xnun: np.ndarray,
    model: nn.Module,
    device: torch.device,
    target_class: int,
    ae: Optional[nn.Module],
    e_max: float,
    err_x: float,
    time_block_id: np.ndarray,
    channel_block_id: np.ndarray,
    n_time_blocks: int,
    n_channel_blocks: int,
    horizon: int,
    n_samples_multiplier: int,
    cem_iterations: int,
    elite_fraction: float,
    smoothing: float,
    alpha: float,
    beta: float,
    eta: float,
    lam: float,
    delta: float,
    invalid_penalty: float,
    max_steps: int,
    rng: np.random.Generator,
    last_valid_mask: np.ndarray,
    verbose: bool,
    level_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve one granularity level's MDP (Algorithm 2, lines 6-14): repeatedly plan
    with CEM and apply the first action until STOP or the step budget is exhausted."""
    n_actions = n_time_blocks * n_channel_blocks + 1
    n_samples = n_samples_multiplier * n_actions
    stop_idx = n_actions - 1

    def action_to_mask(a: int) -> np.ndarray:
        gl, gc = divmod(a, n_channel_blocks)
        return (time_block_id[None, :] == gl) & (channel_block_id[:, None] == gc)

    steps_taken = 0
    while steps_taken < max_steps:
        a_star = _plan_cem(
            mask, x, xnun, model, device, target_class, ae, e_max, err_x, time_block_id, channel_block_id,
            n_time_blocks, n_channel_blocks, horizon, n_samples, cem_iterations, elite_fraction, smoothing, alpha,
            beta, eta, lam, delta, invalid_penalty, rng,
        )
        if a_star == stop_idx:
            if verbose:
                print(f"[FastPACE] level {level_idx}: STOP after {steps_taken} step(s)")
            break

        mask = mask ^ action_to_mask(a_star)
        if _mask_is_valid(mask, x, xnun, model, device, target_class):
            last_valid_mask = mask.copy()
        steps_taken += 1
        if verbose:
            print(f"[FastPACE] level {level_idx} step {steps_taken}: block action {a_star}")

    return mask, last_valid_mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fastpace_cf(
    sample: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
    dataset=None,
    granularity_levels: Optional[Sequence[Tuple[float, float]]] = None,
    horizon: int = 3,
    n_samples_multiplier: int = 2,
    cem_iterations: int = 3,
    elite_fraction: float = 0.1,
    smoothing: float = 0.75,
    alpha: float = 0.1,
    beta: float = 0.3,
    eta: float = 0.4,
    lam: float = 0.2,
    delta: float = 0.25,
    invalid_penalty: float = 10.0,
    max_steps_per_level: Optional[int] = None,
    autoencoder: Optional[nn.Module] = None,
    ae_max_error: Optional[float] = None,
    ae_epochs: int = 20,
    max_reference_samples: Optional[int] = 300,
    seed: Optional[int] = None,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a FastPACE counterfactual explanation for a single sample.

    Follows the same signature pattern as every other CF method in this
    repository (``native_guide_uni_cf``, ``abstract_cf``, …): accepts a query
    sample, a dataset of ``(x, y)`` pairs used to find the Nearest Unlike
    Neighbor (NUN) and to train the plausibility autoencoder, and a trained
    classifier. Returns ``(counterfactual, scores)`` in the original
    orientation of ``sample``.

    Parameters
    ----------
    sample:
        Query time series. Accepts 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``.
    model:
        Trained PyTorch classifier, ``(B, C, L) -> (B, num_classes)``.
    target_class:
        If given, restrict the NUN search to instances the classifier predicts
        as this class. If ``None`` (default, matching the paper), the NUN is
        the closest instance predicted as *any* different class, and its
        predicted class becomes the target.
    dataset:
        Sequence of ``(x, y)`` pairs (or an ``(N, C, L)`` array) used both as
        the NUN search pool and as reference data for the plausibility
        autoencoder. Labels ``y`` are not used — the classifier's own
        predictions define classes, matching the paper's NUN definition.
        Required (raises ``ValueError`` if omitted).
    granularity_levels:
        Sequence of ``(temporal_fraction, channel_fraction)`` pairs defining
        the coarse-to-fine hierarchy (Section 3.5). Defaults to the paper's
        settings: ``[(0.1, 1.0), (0.05, 1.0), (0.025, 1.0)]`` for univariate
        data (``C == 1``) and ``[(0.2, 0.2), (0.2, 0.05), (0.05, 0.05)]`` for
        multivariate data.
    horizon, n_samples_multiplier, cem_iterations, elite_fraction, smoothing:
        CEM planning hyperparameters (Section 3.4 / Algorithm 1). Defaults
        (``H=3``, ``N=2|A|``, ``K=3``, ``ε=0.1``, ``μ=0.75``) match Section 4.1.
    alpha, beta, eta, lam, delta:
        Objective weights for adversarial / sparsity / contiguity / plausibility
        (Eq. 2). Defaults (``α=0.1, β=0.3, η=0.4, λ=0.2``) match Sub-SpaCE's
        weighting, reused by FastPACE per Section 4.1; ``δ`` (the contiguity
        exponent) is left unspecified in the paper's text, so it defaults to
        Sub-SpaCE's own convention of 0.25.
    invalid_penalty:
        Large constant (``ν``) subtracted whenever a candidate is not
        predicted as ``target_class`` (Eq. 2), making the validity constraint
        act as an approximate hard constraint during planning.
    max_steps_per_level:
        Cap on planning steps (mask updates) per granularity level. Defaults
        to the number of blocks at that level.
    autoencoder, ae_max_error:
        Pre-trained plausibility autoencoder and its ``e_max`` (see
        :func:`train_plausibility_autoencoder`). If omitted, one is trained
        on-the-fly from ``dataset`` — pass a pre-trained pair (e.g. from
        :func:`fastpace_batch_cf`) to avoid retraining for every sample.
    max_reference_samples:
        Cap on the number of dataset instances used for the NUN search,
        channel clustering, and autoencoder training (for speed on large
        datasets). ``None`` uses the full dataset.
    verbose:
        Print per-step planning diagnostics.

    Returns
    -------
    counterfactual : np.ndarray
        Counterfactual time series in the same shape/orientation as ``sample``.
    scores : np.ndarray, shape (num_classes,)
        Model output for the counterfactual.
    """
    if dataset is None:
        raise ValueError("fastpace_cf requires a dataset for the NUN search and plausibility autoencoder.")
    device = next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(seed)

    sample_cl, ts, ori = ensure_ncl(np.asarray(sample, dtype=np.float32), dataset)
    C, L = sample_cl.shape

    if max_reference_samples is not None and len(ts) > max_reference_samples:
        keep = rng.choice(len(ts), size=max_reference_samples, replace=False)
        ts = ts[keep]

    # --- NUN search over the classifier's own predictions (Eq. 1) ---
    labels_pool = np.argmax(batched_predict(model, ts, device), axis=1)
    with torch.no_grad():
        scores_sample = detach_to_numpy(
            model(numpy_to_torch(sample_cl.reshape(1, C, L), device))
        ).reshape(-1)
    label_sample = int(np.argmax(scores_sample))

    if target_class is not None and target_class == label_sample:
        raise ValueError(
            f"target_class ({target_class}) is the same as the query's predicted class ({label_sample})."
        )

    xnun, ynun = _find_nun(sample_cl, ts, labels_pool, label_sample, target_class)
    if xnun is None:
        if verbose:
            print(f"[FastPACE] No NUN candidate found for target_class={target_class}. Returning sample unchanged.")
        return revert_orientation(sample_cl, ori), scores_sample
    target_class = ynun

    # --- Plausibility autoencoder (Section 3.1) ---
    if autoencoder is None:
        if verbose:
            print(f"[FastPACE] Training plausibility autoencoder on {len(ts)} reference samples...")
        autoencoder, ae_max_error = train_plausibility_autoencoder(ts, device, epochs=ae_epochs, verbose=verbose)
    e_max = ae_max_error if ae_max_error is not None else 1.0
    err_x = float(
        detach_to_numpy(_reconstruction_error(autoencoder, numpy_to_torch(sample_cl.reshape(1, C, L), device)))[0]
    )

    # --- Granularity hierarchy (Section 4.1) ---
    if granularity_levels is None:
        granularity_levels = (
            [(0.1, 1.0), (0.05, 1.0), (0.025, 1.0)] if C == 1 else [(0.2, 0.2), (0.2, 0.05), (0.05, 0.05)]
        )

    # M0 = all-ones -> x' == NUN, valid by construction (Section 3.2).
    mask = np.ones((C, L), dtype=bool)
    last_valid_mask = mask.copy()

    for level_idx, (frac_ts, frac_ch) in enumerate(granularity_levels):
        n_time_blocks, n_channel_blocks = _granularity_to_blocks(frac_ts, frac_ch, L, C)
        time_block_id = _contiguous_blocks(L, n_time_blocks)
        channel_block_id = _cluster_channels(ts, n_channel_blocks)
        n_time_blocks = int(time_block_id.max()) + 1
        n_channel_blocks = int(channel_block_id.max()) + 1

        steps = max_steps_per_level if max_steps_per_level is not None else n_time_blocks * n_channel_blocks

        mask, last_valid_mask = _run_level(
            mask, sample_cl, xnun, model, device, target_class, autoencoder, e_max, err_x, time_block_id,
            channel_block_id, n_time_blocks, n_channel_blocks, horizon, n_samples_multiplier, cem_iterations,
            elite_fraction, smoothing, alpha, beta, eta, lam, delta, invalid_penalty, steps, rng, last_valid_mask,
            verbose, level_idx,
        )

    # --- Validity by design: fall back to the last mask known to be valid ---
    final_mask = mask if _mask_is_valid(mask, sample_cl, xnun, model, device, target_class) else last_valid_mask

    cf = np.where(final_mask, xnun, sample_cl).astype(np.float32)
    with torch.no_grad():
        scores_cf = detach_to_numpy(model(numpy_to_torch(cf.reshape(1, C, L), device))).reshape(-1)

    if verbose:
        print(f"[FastPACE] Final class: {int(np.argmax(scores_cf))} (target {target_class})")

    return revert_orientation(cf, ori), scores_cf


def fastpace_batch_cf(
    samples: np.ndarray,
    dataset,
    model: nn.Module,
    target_class: Optional[int] = None,
    ae_epochs: int = 20,
    max_reference_samples: Optional[int] = 300,
    seed: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate FastPACE counterfactuals for multiple samples.

    Trains the plausibility autoencoder once on ``dataset`` and reuses it for
    every sample (the NUN, being query-specific, is still searched per
    sample). All other keyword arguments are forwarded to :func:`fastpace_cf`.

    Returns
    -------
    counterfactuals : np.ndarray, shape (N, ...)
    predictions : np.ndarray, shape (N, num_classes)
    """
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)

    first_sample = np.asarray(samples[0], dtype=np.float32)
    _, ts, _ = ensure_ncl(first_sample, dataset)
    if max_reference_samples is not None and len(ts) > max_reference_samples:
        keep = rng.choice(len(ts), size=max_reference_samples, replace=False)
        ts = ts[keep]

    if verbose:
        print(f"[FastPACE Batch] Training shared plausibility autoencoder on {len(ts)} reference samples...")
    autoencoder, e_max = train_plausibility_autoencoder(ts, device, epochs=ae_epochs, verbose=verbose)

    counterfactuals, predictions = [], []
    for i, sample in enumerate(samples):
        if verbose:
            print(f"--- Sample {i + 1}/{len(samples)} ---")
        cf, pred = fastpace_cf(
            sample,
            model,
            target_class=target_class,
            dataset=dataset,
            autoencoder=autoencoder,
            ae_max_error=e_max,
            max_reference_samples=max_reference_samples,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )
        counterfactuals.append(cf)
        predictions.append(pred)

    return np.array(counterfactuals), np.array(predictions)
