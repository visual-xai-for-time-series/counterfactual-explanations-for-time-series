"""
TimeXcf++: information-bottleneck counterfactual generator, built on the
TimeX++ explanation-bottleneck architecture.

Paper (defines the counterfactual path implemented here, "TimeXcf++"):
       Zheng, X., Liu, Z., Chen, Z., Akewar, M., Bhimani, J., Liu, J., Sha,
       M., Ni, J., Cheng, W., & Luo, D. (2026). "Towards A Unified
       Information Bottleneck Framework for Time Series Explanations."
       arXiv:2608.25897.

Paper (defines the shared bottleneck-extractor architecture this module's
       extractor is scaled down from -- "TimeX++"):
       Liu, Z., Wang, T., Shi, J., Zheng, X., Chen, Z., Song, L., Dong, W.,
       Obeysekera, J., Shirani, F., & Luo, D. (2024). "TimeX++: Learning
       Time-Series Explanations with Information Bottleneck." Proceedings
       of the 41st International Conference on Machine Learning (ICML 2024).

Paper URLs: https://arxiv.org/abs/2608.25897 (unified IB framework --
            Sections on "Counterfactual Explanations via Controlled
            Information Removal" and Algorithm 1 are what this module
            follows)
            https://arxiv.org/abs/2405.09308 (original TimeX++, defines the
            transformer bottleneck extractor g_phi both papers share)
GitHub (authors' code -- the TimeX++ ICML2024 release; the unified-framework
        paper's own counterfactual path has no separate public release at
        the time of writing, so this module follows the paper text rather
        than a reference implementation for that part):
        https://github.com/zichuan-liu/TimeXplusplus

NAMING / SCOPE NOTE -- read this before using this module, mirroring the note
at the top of ``cfts/cf_timex/timex_cf.py`` for the sibling "TimeX" package:

Earlier versions of this module targeted only the ICML2024 TimeX++ paper,
which is an *explainer*, not a counterfactual generator -- its label-
consistency term is trained to *preserve* the query's own predicted class,
not flip it, so producing a counterfactual meant retargeting that objective
by hand. arXiv:2608.25897 changes that: it unifies the same information-
bottleneck architecture into two explicit paths sharing one bottleneck
extractor g_phi -- an attribution path (TimeXa++, label-consistency against
the *original* class) and a counterfactual path (TimeXcf++, label-
consistency against a *target* class Y'', with a generator/reference
structure of its own). This module now implements TimeXcf++'s counterfactual
path directly, as published, rather than adapting TimeXa++'s attribution
machinery:

  1. Bottleneck extractor: a transformer encoder g_phi maps the input X to a
     stochastic per-(timestep, channel) selection probability pi.
  2. Mask sampling: M = STE(Bern(pi)) -- a straight-through-estimator
     Bernoulli sample. Forward pass draws a hard {0,1} mask; backward pass
     lets gradients bypass the discrete sampling op and flow to pi as if it
     were the identity.
  3. Perturbation generator psi_cf(X, M) -> E: an MLP predicting the
     counterfactual edit to apply within the masked region.
  4. Noise generator psi_n(X, M, X_ref) -> epsilon: an MLP folding in a
     training-set reference instance X_ref whose label is the target class
     Y'', used only during training so psi_cf sees target-class-informed
     variation; epsilon -> 0 at inference.
  5. Explanation-embedded counterfactual instance:
       X_tilde_cf = X + M*E + epsilon   (training)
       X_tilde_cf = X + M*E             (inference, epsilon dropped)
  6. Structural ("bound") loss anchors the *complement* of M to the
     original query X itself (X_tilde_cf^r = X) -- a hard causal constraint
     that keeps every edit confined to the sparse region M selects, unlike
     the attribution path's Gaussian-padded reference (irrelevant here and
     not built by this module at all).
  7. Label-consistency loss is computed directly against the target class:
     Y^expl = target_class, via the same JS-divergence term the paper uses
     for both paths -- this is the paper's own counterfactual objective, not
     a hand-retargeted attribution objective.
  8. Total: L = L_LC + alpha * L_M + beta * (L_KL + L_bound).

Further simplifications made to keep training tractable within a single
``<name>_cf`` call (same "train inline, budget far below the paper's offline
configuration" tradeoff ``cf_diffcf/diffcf.py`` and ``cf_cfe4mts/cfe4mts.py``
make for their own trained components):

  - A fresh extractor/generator set is trained per call and discarded after
    producing one counterfactual (``timexplusplus_cf``), matching this
    repository's single-call signature. Every training instance's
    label-consistency loss targets the *query's* fixed ``target_class``
    (rather than a per-instance independently sampled class) since the
    trained set is thrown away afterwards anyway. ``timexplusplus_fit`` /
    ``timexplusplus_generate`` are exposed separately for callers who want
    to reuse one trained set across several queries that share a target
    class. The noise generator psi_n is trained (it shapes psi_cf and the
    extractor's gradients through epsilon during training) but, like the
    paper's own inference rule, is never called by ``timexplusplus_generate``
    and so isn't kept in ``FittedTimeXPlusPlus``.
  - The extractor is a single Transformer encoder + linear head rather than
    the paper's full encoder-decoder, to fit this repository's per-call
    training budget (the same scale-down ``cf_cfe4mts/cfe4mts.py``'s
    single-layer-LSTM discriminator makes versus its own paper's
    architecture). Mask *sampling while training* (the straight-through
    Bernoulli estimator) matches the paper exactly -- see
    ``_ste_bernoulli_sample``. The paper specifies no inference-time rule for
    turning `pi` into a mask for a single query; this module picks the
    highest-`pi` r-fraction of entries deterministically (see ``_topk_mask``)
    rather than thresholding at 0.5, which would silently return an empty
    mask whenever `r` (the paper's own default is 0.1 here) pulls every
    entry's probability below that threshold.
  - The paper states psi_cf and psi_n "translate the structural bottleneck M
    into the generated explanation-embedded instance" but does not publish
    their architecture (layer count, width, per-timestep vs. sequence-level).
    This module uses small per-timestep MLPs with shared weights (agnostic
    to series length), predicting a perturbation/noise residual rather than
    an unconstrained absolute output -- the same choice most gradient-based
    methods in this repository make for their own trained components (e.g.
    CFE4MTS's central noiser, GLACIER, SPARCE).
  - L_KL (distribution-shift) is approximated by first-and-second-moment
    matching between X_tilde and the training batch's per-channel
    mean/std, rather than a parametric KL between full distributions -- the
    paper itself states this term regularises distribution shift without
    publishing a closed form to implement.
  - The reference instance X_ref the noise generator conditions on ("a
    training-set instance whose label is Y''") is sampled from `dataset`
    using the frozen classifier `model`'s *predicted* label, not `dataset`'s
    ground-truth label -- consistent with how every other trained method in
    this repository derives labels for a query-supplied `model` (e.g.
    CFE4MTS's noiser uses `argmax(model(x))`, not `y`). When `model` predicts
    no training instance as `target_class` (e.g. a rare or, for this model,
    unreachable class), falls back to the training instances `model` scores
    highest for `target_class`, so the noise generator still receives a
    sensibly target-class-informed reference rather than raising.
  - Model sizes (encoder depth/width) and epoch counts default far below
    what an offline-trained TimeXcf++ would use; pass a larger
    ``d_model``/``num_encoder_layers``/``epochs`` for higher fidelity at the
    cost of runtime.
  - ``r`` (target mask sparsity) defaults to ``0.1``, the paper's own stated
    default for the counterfactual path (TimeXcf++), stricter than the
    attribution path's ``0.5`` -- an earlier version of this module used the
    attribution default, which is a materially looser edit than the paper
    intends here: under this module's mask semantics (M marks the region
    *edited* by the counterfactual, not the region *kept* truthful, since
    the roles invert between the two paths) a smaller r keeps edits sparse,
    which is what a minimal counterfactual should look like.
  - The paper's Implementation Details do not publish numeric alpha/beta/
    lambda_con weights for either path, so this module's ``alpha=0.5``,
    ``beta=0.25`` defaults follow this repository's own convention instead:
    keep the validity term (L_LC) dominant over the compactness/distribution
    terms, the same principle CFE4MTS applies with
    ``lambda_clas=10.0`` versus ``lambda_gen=1.0``/``lambda_dist=0.01``.
    ``alpha``/``beta``/``r``/``gamma`` (this module's name for the paper's
    ``lambda_con`` continuity weight) remain free to set to any value for a
    more explanation-like (lower-validity, higher-compactness) counterfactual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from cfts.cf__abstract.abstract import (
    batched_predict,
    detach_to_numpy,
    ensure_cl,
    ensure_ncl,
    numpy_to_torch,
    revert_orientation,
    subsample_dataset,
)


# ---------------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------------

def _outputs_are_probabilities(scores: np.ndarray) -> bool:
    """Detect whether `model`'s raw output is already a softmax distribution.

    Same check as ``cf_cfe4mts/cfe4mts.py``'s helper of the same name: most
    classifiers here return raw logits, for which softmax must be applied
    before computing the JS-divergence label-consistency loss below; a few
    pretrained models (e.g. `SimpleCNN`) already end in `nn.Softmax`, so
    applying softmax again would silently flatten the validity gradient.
    """
    return bool(
        np.all(scores >= -1e-6) and np.allclose(scores.sum(axis=1), 1.0, atol=1e-3)
    )


def _to_probs(raw: torch.Tensor, outputs_are_probs: bool) -> torch.Tensor:
    return raw if outputs_are_probs else torch.softmax(raw, dim=-1)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence between two batches of categorical distributions.

    Paper's label-consistency loss L_LC; here Y^expl = target_class (the
    counterfactual path -- see module docstring), computed between the
    model's prediction on X_tilde_cf and a one-hot ``target_class``.
    """
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p.clamp_min(eps)) - torch.log(m.clamp_min(eps)))).sum(dim=-1)
    kl_qm = (q * (torch.log(q.clamp_min(eps)) - torch.log(m.clamp_min(eps)))).sum(dim=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def _bernoulli_kl(pi: torch.Tensor, r: float, eps: float = 1e-6) -> torch.Tensor:
    """Elementwise KL(Bernoulli(pi) || Bernoulli(r)), averaged over all entries.

    Paper Eq. for the compactness quantifier's KL term:
    sum_{t,d} [pi_td * log(pi_td/r) + (1-pi_td) * log((1-pi_td)/(1-r))].
    Computed here per (timestep, channel) entry of `pi` and averaged rather
    than summed, so the loss magnitude doesn't scale with series
    length/channel count (consistent with every other loss term in this
    module being a mean, not a raw sum).
    """
    pi = pi.clamp(eps, 1 - eps)
    r_t = torch.as_tensor(r, dtype=pi.dtype, device=pi.device).clamp(eps, 1 - eps)
    kl = pi * (torch.log(pi) - torch.log(r_t)) + (1 - pi) * (torch.log(1 - pi) - torch.log(1 - r_t))
    return kl.mean()


def _ste_bernoulli_sample(probs: torch.Tensor) -> torch.Tensor:
    """Straight-through Bernoulli mask sample, M = STE(Bern(pi)) -- training only.

    Paper's mask-sampling description: the forward pass draws a deterministic
    binary mask M = STE(Bern(pi)); the backward pass lets gradients bypass
    the discrete sampling operator, so pi's parameters are optimised as if M
    were the identity. Draws one Bernoulli(pi) sample per entry with that
    straight-through gradient. Used only while training -- see `_topk_mask`
    for the deterministic mask `timexplusplus_generate` applies at inference.
    """
    hard = (torch.rand_like(probs) < probs).float()
    return probs + (hard - probs).detach()


def _topk_mask(probs: torch.Tensor, r: float) -> torch.Tensor:
    """Deterministic inference-time mask: the highest r-fraction of `probs` entries.

    The compactness loss (`_bernoulli_kl`) only pulls the *mean* of `pi`
    toward `r` during training; with r < 0.5 (the paper's own counterfactual-
    path default, 0.1) that routinely leaves every individual entry below
    0.5, so a fixed `pi > 0.5` threshold at inference would silently return
    an empty mask (and hence a no-op counterfactual) even when training
    converged. Selecting the top round(r * T * D) entries instead
    deterministically reproduces the trained sparsity level -- the same
    top-k-by-score convention used to binarise a continuous relevance map in
    the wider IB/L0 explanation literature (e.g. L2X, INVASE); the paper
    itself specifies no inference-time rule (see `_ste_bernoulli_sample`).
    """
    flat = probs.reshape(probs.shape[0], -1)
    k = max(1, min(flat.shape[1], int(round(r * flat.shape[1]))))
    cutoff = flat.topk(k, dim=1).values[:, -1:]
    return (flat >= cutoff).float().reshape(probs.shape)


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    """Yield shuffled index batches, dropping a trailing batch of size <= 1."""
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if len(idx) > 1:
            yield idx


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ExplanationExtractor(nn.Module):
    """Transformer-encoder bottleneck extractor g_phi producing per-(t,d) mask logits.

    Paper: "transformer encoder-decoder architecture producing stochastic
    masks," shared by both the attribution and counterfactual paths. Scaled
    down to a single encoder + linear head by default to fit this
    repository's per-call training budget (the same scale-down
    ``cf_cfe4mts/cfe4mts.py``'s single-layer-LSTM discriminator makes versus
    the paper's own architecture); mask *sampling* from this extractor's
    output while training (see ``_ste_bernoulli_sample``) matches the paper
    exactly.
    """

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mask_head = nn.Linear(d_model, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> mask logits (B, L, C)."""
        h = self.input_proj(x) + self.pos_embedding
        h = self.encoder(h)
        return self.mask_head(h)


class PerturbationGenerator(nn.Module):
    """MLP psi_cf mapping [X, M] to the counterfactual edit E.

    Paper: X_tilde_cf = X + M*E + epsilon, with E = psi_cf(X, M) predicting
    the edit to apply inside the masked region. Applied per-timestep with
    shared weights (agnostic to series length); the paper does not publish
    psi_cf's architecture (see module docstring), so this follows the same
    "predict a perturbation, not an unconstrained absolute output" stability
    choice most gradient-based methods in this repository make for their own
    trained components (e.g. CFE4MTS's central noiser, GLACIER, SPARCE).
    """

    def __init__(self, num_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x, mask: (B, L, C) -> perturbation E, (B, L, C)."""
        return self.net(torch.cat([x, mask], dim=-1))


class NoiseGenerator(nn.Module):
    """MLP psi_n mapping [X, M, X_ref] to training-time exploration noise epsilon.

    Paper: epsilon = psi_n(X, M, X_ref), where X_ref is a training instance
    labelled as the target class Y'' -- injected only while training so
    psi_cf sees target-class-informed variation; the paper sets epsilon -> 0
    at inference, so this network is trained (it shapes psi_cf's and the
    extractor's gradients) but discarded once `timexplusplus_fit` returns --
    `timexplusplus_generate` never calls it. Architecture choices mirror
    `PerturbationGenerator` (see its docstring); the paper doesn't publish
    psi_n's architecture either.
    """

    def __init__(self, num_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_channels * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """x, mask, x_ref: (B, L, C) -> epsilon, (B, L, C)."""
        return self.net(torch.cat([x, mask, x_ref], dim=-1))


# ---------------------------------------------------------------------------
# Fit / generate / cf
# ---------------------------------------------------------------------------

@dataclass
class FittedTimeXPlusPlus:
    """A trained TimeXcf++ extractor/generator set, targeting one fixed class.

    Reusable across queries that (a) share this ``target_class`` and (b)
    have the same (num_channels, seq_len) shape the set was trained on --
    see ``timexplusplus_generate``. Does not hold the noise generator psi_n
    (trained but discarded, see module docstring): only ``extractor`` and
    ``perturbation_net`` are needed to reproduce the paper's inference rule
    X_tilde_cf = X + M*E.
    """

    extractor: "ExplanationExtractor"
    perturbation_net: "PerturbationGenerator"
    num_channels: int
    seq_len: int
    num_classes: int
    target_class: int
    device: torch.device
    data_mean: torch.Tensor  # (C,)
    data_std: torch.Tensor  # (C,)
    outputs_are_probs: bool
    r: float  # target mask sparsity trained with -- see `_topk_mask`
    history: dict | None = None


def timexplusplus_fit(
    dataset: list | np.ndarray,
    model: torch.nn.Module,
    target_class: int,
    alpha: float = 0.5,
    beta: float = 0.25,
    r: float = 0.1,
    gamma: float = 0.1,
    d_model: int = 32,
    nhead: int = 2,
    num_encoder_layers: int = 1,
    hidden_dim_generator: int = 32,
    lr: float = 3e-3,
    epochs: int = 80,
    batch_size: int = 16,
    max_train_samples: int = 150,
    seed: int | None = None,
    verbose: bool = False,
) -> FittedTimeXPlusPlus:
    """Train the TimeXcf++ bottleneck extractor + perturbation/noise generators.

    Minimises, per batch::

        L_LC    = mean JS( softmax(model(X_tilde_cf)), one_hot(target_class) )
        L_M     = mean KL(Bernoulli(pi) || Bernoulli(r))
                  + gamma * mean |pi[t] - pi[t-1]|             (continuity term)
        L_bound = mean || (1 - M) * (X_tilde_cf - X) ||^2      (structural anchor)
        L_KL    = || mean(X_tilde_cf) - data_mean ||^2 + || std(X_tilde_cf) - data_std ||^2
                  (moment-matching approximation of the paper's distribution-shift KL)
        L       = L_LC + alpha * L_M + beta * (L_KL + L_bound)

    where M = STE(Bern(pi)) is the sampled bottleneck mask, E = psi_cf(X, M)
    the learned perturbation, X_ref a training instance `model` predicts as
    `target_class`, epsilon = psi_n(X, M, X_ref) training-time noise, and
    X_tilde_cf = X + M*E + epsilon (see module docstring for the full
    pipeline and why L_bound anchors against X itself rather than a
    Gaussian-padded reference).

    Parameters
    ----------
    dataset:
        Sequence of (x, y) pairs (or an (N, C, L) array) used to train the
        extractor/generators. Required.
    model:
        Frozen PyTorch classifier being explained; ``(B, C, L) -> (B, num_classes)``.
    target_class:
        Class every training instance's label-consistency loss targets (see
        module docstring for why this is fixed rather than per-instance).
    alpha, beta:
        Weights on the compactness (L_M) and distribution/structural
        (L_KL + L_bound) terms, matching the paper's objective
        L = L_LC + alpha*L_M + beta*(L_KL+L_bound).
    r:
        Target mask sparsity (mean fraction of the series treated as the
        "edited" region under M). Defaults to the paper's stated
        counterfactual-path value (0.1); the paper's attribution path uses
        0.5.
    gamma:
        Weight of the continuity (temporal-smoothness) penalty inside L_M
        (the paper's lambda_con).
    d_model, nhead, num_encoder_layers:
        Size of the transformer bottleneck extractor.
    hidden_dim_generator:
        Hidden width of both MLP generators (psi_cf and psi_n).
    lr:
        Adam learning rate (extractor and generators share one optimiser).
    epochs, batch_size:
        Training length / mini-batch size.
    max_train_samples:
        Class-balanced cap on the number of training instances (see
        `subsample_dataset`), to keep training tractable.
    seed:
        Integer seed for reproducibility, or ``None`` for random behaviour.
    verbose:
        Print per-epoch diagnostics when ``True``.

    Returns
    -------
    FittedTimeXPlusPlus
    """
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_data = subsample_dataset(dataset, max_train_samples)
    first_x = np.asarray(train_data[0][0], dtype=np.float32)
    sample_cl, _ = ensure_cl(first_x)
    C, L = sample_cl.shape
    _, ts, _ = ensure_ncl(first_x, train_data)

    with torch.no_grad():
        train_scores = batched_predict(model, ts, device, batch_size=max(batch_size, 64))
    num_classes = train_scores.shape[1]
    outputs_are_probs = _outputs_are_probabilities(train_scores)
    if verbose:
        print(
            f"[timexplusplus_fit] model output detected as "
            f"{'softmax probabilities' if outputs_are_probs else 'raw logits'}"
        )

    # Reference pool for the noise generator's X_ref (paper: "a training
    # instance whose label is Y''"), keyed on `model`'s *predicted* label --
    # see module docstring for why. Falls back to the instances `model`
    # scores highest for `target_class` when it predicts that class for
    # nothing in the (sub-sampled) training set.
    train_labels_np = np.argmax(train_scores, axis=1)
    ref_pool_idx = np.flatnonzero(train_labels_np == target_class)
    if ref_pool_idx.size == 0:
        k = max(1, min(len(train_scores), max(5, len(train_scores) // 10)))
        ref_pool_idx = np.argsort(train_scores[:, target_class])[::-1][:k]

    orig_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    ts_lc = numpy_to_torch(ts, device).transpose(1, 2)  # (N, L, C)
    data_mean = ts_lc.mean(dim=(0, 1))  # (C,)
    data_std = ts_lc.std(dim=(0, 1)).clamp_min(1e-6)  # (C,)

    target_onehot_row = F.one_hot(
        torch.tensor(target_class, device=device), num_classes
    ).float()

    extractor = ExplanationExtractor(C, L, d_model, nhead, num_encoder_layers).to(device)
    perturbation_net = PerturbationGenerator(C, hidden_dim_generator).to(device)
    noise_net = NoiseGenerator(C, hidden_dim_generator).to(device)
    optimizer = Adam(
        list(extractor.parameters()) + list(perturbation_net.parameters()) + list(noise_net.parameters()),
        lr=lr,
    )

    n = ts_lc.shape[0]
    history: dict[str, list[float]] = {"loss": [], "lc": [], "m": [], "kl": [], "bound": []}
    try:
        for epoch in range(epochs):
            extractor.train()
            perturbation_net.train()
            noise_net.train()
            epoch_losses = {"loss": [], "lc": [], "m": [], "kl": [], "bound": []}

            for idx in _batches(n, batch_size, rng):
                idx_t = torch.as_tensor(idx, device=device)
                x = ts_lc[idx_t]  # (b, L, C)
                b = x.shape[0]

                optimizer.zero_grad()

                mask_logits = extractor(x)
                mask_prob = torch.sigmoid(mask_logits)
                mask_sample = _ste_bernoulli_sample(mask_prob)

                ref_choice = ref_pool_idx[rng.integers(0, len(ref_pool_idx), size=b)]
                x_ref = ts_lc[torch.as_tensor(ref_choice, device=device)]

                e = perturbation_net(x, mask_sample)
                eps = noise_net(x, mask_sample, x_ref)
                x_tilde = x + mask_sample * e + eps

                raw_scores = model(x_tilde.transpose(1, 2))
                probs = _to_probs(raw_scores, outputs_are_probs)
                target_onehot = target_onehot_row.unsqueeze(0).expand(b, -1)
                l_lc = _js_divergence(probs, target_onehot).mean()

                l_m = _bernoulli_kl(mask_prob, r)
                connective = (mask_prob[:, 1:, :] - mask_prob[:, :-1, :]).abs().mean()
                l_m = l_m + gamma * connective

                l_bound = (((1 - mask_sample) * (x_tilde - x)) ** 2).mean()
                l_kl = ((x_tilde.mean(dim=(0, 1)) - data_mean) ** 2).mean() + \
                    ((x_tilde.std(dim=(0, 1)) - data_std) ** 2).mean()

                loss = l_lc + alpha * l_m + beta * (l_kl + l_bound)
                loss.backward()
                optimizer.step()

                epoch_losses["loss"].append(loss.item())
                epoch_losses["lc"].append(l_lc.item())
                epoch_losses["m"].append(l_m.item())
                epoch_losses["kl"].append(l_kl.item())
                epoch_losses["bound"].append(l_bound.item())

            for k in history:
                history[k].append(float(np.mean(epoch_losses[k])) if epoch_losses[k] else float("nan"))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(
                    f"[timexplusplus_fit] epoch {epoch:4d}  "
                    f"L={history['loss'][-1]:.4f}  L_LC={history['lc'][-1]:.4f}  "
                    f"L_M={history['m'][-1]:.4f}  L_KL={history['kl'][-1]:.4f}  "
                    f"L_bound={history['bound'][-1]:.4f}"
                )
    finally:
        for p, rg in zip(model.parameters(), orig_requires_grad):
            p.requires_grad_(rg)

    extractor.eval()
    perturbation_net.eval()
    return FittedTimeXPlusPlus(
        extractor=extractor, perturbation_net=perturbation_net, num_channels=C, seq_len=L,
        num_classes=num_classes, target_class=target_class, device=device,
        data_mean=data_mean, data_std=data_std, outputs_are_probs=outputs_are_probs,
        r=r, history=history,
    )


def timexplusplus_generate(
    fitted: FittedTimeXPlusPlus,
    sample: np.ndarray | list,
    model: torch.nn.Module,
    return_mask: bool = False,
    verbose: bool = False,
):
    """Generate a counterfactual with an already-trained extractor/generator set.

    A single forward pass -- no training -- implementing the paper's
    inference rule X_tilde_cf = X + M*E (epsilon dropped, see module
    docstring). `fitted` must have been trained with the same `target_class`
    this query should move to (retrain with `timexplusplus_fit` for a
    different target class).

    Parameters
    ----------
    fitted:
        Output of `timexplusplus_fit`, trained on the same `model` and on
        data with the same (C, L) shape as `sample`.
    sample:
        Query time series. Accepts 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``.
    model:
        The same frozen classifier `fitted` was trained against.
    return_mask:
        If True, also return the deterministic bottleneck mask M (same
        shape as `sample`) the counterfactual edit was confined to -- the
        paper's counterfactual-path explanation output (distinct from the
        attribution path's importance mask). Off by default so the return
        arity matches this repository's ``(counterfactual, scores)``
        ``<name>_cf`` contract.
    verbose:
        Print the target/counterfactual class when ``True``.

    Returns
    -------
    counterfactual : np.ndarray, same shape/orientation as `sample`.
    scores : np.ndarray, shape (num_classes,).
    mask : np.ndarray, same shape/orientation as `sample` -- only when
        `return_mask=True`.
    """
    device = fitted.device
    sample_cl, ori = ensure_cl(np.asarray(sample, dtype=np.float32))
    C, L = sample_cl.shape
    if (C, L) != (fitted.num_channels, fitted.seq_len):
        raise ValueError(
            f"sample shape (C={C}, L={L}) does not match the shape "
            f"(C={fitted.num_channels}, L={fitted.seq_len}) `fitted` was trained on."
        )

    sample_lc = numpy_to_torch(sample_cl, device).T.unsqueeze(0)  # (1, L, C)

    fitted.extractor.eval()
    fitted.perturbation_net.eval()
    with torch.no_grad():
        mask_logits = fitted.extractor(sample_lc)
        mask_prob = torch.sigmoid(mask_logits)
        mask = _topk_mask(mask_prob, fitted.r)
        e = fitted.perturbation_net(sample_lc, mask)
        x_tilde = sample_lc + mask * e
        cf_cl_t = x_tilde.squeeze(0).T  # (C, L)
        scores_cf = detach_to_numpy(model(cf_cl_t.unsqueeze(0))).reshape(-1)

    cf = detach_to_numpy(cf_cl_t).reshape(C, L)

    if verbose:
        label_cf = int(np.argmax(scores_cf))
        print(
            f"[timexplusplus_generate] target={fitted.target_class}  "
            f"counterfactual={label_cf}  mean_mask={float(mask.mean()):.3f}"
        )

    cf_out = revert_orientation(cf, ori)
    if not return_mask:
        return cf_out, scores_cf
    mask_np = detach_to_numpy(mask.squeeze(0).T).reshape(C, L)
    return cf_out, scores_cf, revert_orientation(mask_np, ori)


def timexplusplus_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    alpha: float = 0.5,
    beta: float = 0.25,
    r: float = 0.1,
    gamma: float = 0.1,
    d_model: int = 32,
    nhead: int = 2,
    num_encoder_layers: int = 1,
    hidden_dim_generator: int = 32,
    lr: float = 3e-3,
    epochs: int = 80,
    batch_size: int = 16,
    max_train_samples: int = 150,
    seed: int | None = None,
    return_mask: bool = False,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """TimeXcf++: information-bottleneck counterfactual generator (see module
    docstring for the full pipeline and the simplifications made versus the
    published method).

    Trains a fresh :func:`timexplusplus_fit` extractor/generator set on
    `dataset` towards `target_class`, then generates a counterfactual for
    `sample` with a single forward pass via :func:`timexplusplus_generate`.
    Follows the same signature pattern as every other CF method in this
    repository (see `cf__abstract.abstract.abstract_cf`).

    This is a thin `timexplusplus_fit` + `timexplusplus_generate`
    composition kept for the single-call signature; it retrains from scratch
    on every call -- the same tradeoff `cf_cfe4mts/cfe4mts.py::cfe4mts_cf`
    documents for its own noiser/discriminator. Call `timexplusplus_fit`
    once and reuse it with `timexplusplus_generate` when explaining several
    queries that share a target class.

    Parameters
    ----------
    sample:
        Query time series. Accepts 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``.
    model:
        Frozen PyTorch classifier being explained; ``(B, C, L) -> (B, num_classes)``.
    target_class:
        Desired class of the counterfactual. Defaults to the second most
        likely class for `sample` under `model` when not given.
    dataset:
        Sequence of (x, y) pairs (or an (N, C, L) array) used to train the
        extractor/generators. Required.
    alpha, beta, r, gamma, d_model, nhead, num_encoder_layers,
    hidden_dim_generator, lr, epochs, batch_size, max_train_samples,
    seed, verbose:
        See :func:`timexplusplus_fit`.
    return_mask:
        See :func:`timexplusplus_generate`. When True, returns a 3-tuple
        `(counterfactual, scores, mask)` instead of the usual 2-tuple.

    Returns
    -------
    counterfactual : np.ndarray, same shape/orientation as `sample`.
    scores : np.ndarray, shape (num_classes,).

    Example
    -------
    >>> cf, scores = timexplusplus_cf(sample_np, model, target_class=1, dataset=dataset_train)
    """
    if dataset is None:
        raise ValueError(
            "timexplusplus_cf requires a training dataset (it trains a "
            "bottleneck extractor + generators, it does not run a "
            "per-sample optimisation)."
        )

    device = next(model.parameters()).device
    sample_cl, _ = ensure_cl(np.asarray(sample, dtype=np.float32))
    C, L = sample_cl.shape
    with torch.no_grad():
        scores_orig = detach_to_numpy(
            model(numpy_to_torch(sample_cl.reshape(1, C, L), device))
        ).reshape(-1)
    label_orig = int(np.argmax(scores_orig))

    if target_class is None:
        target_class = int(np.argsort(scores_orig)[::-1][1])
    if target_class == label_orig:
        raise ValueError(
            f"target_class ({target_class}) is the same as the query's predicted "
            f"class ({label_orig}). Choose a different target class."
        )

    fitted = timexplusplus_fit(
        dataset, model, target_class,
        alpha=alpha, beta=beta, r=r, gamma=gamma,
        d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
        hidden_dim_generator=hidden_dim_generator, lr=lr,
        epochs=epochs, batch_size=batch_size, max_train_samples=max_train_samples,
        seed=seed, verbose=verbose,
    )
    return timexplusplus_generate(fitted, sample, model, return_mask=return_mask, verbose=verbose)
