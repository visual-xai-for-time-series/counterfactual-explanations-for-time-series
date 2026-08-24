"""
TimeX++-CF: information-bottleneck explanation extractor + conditioner,
retargeted to produce counterfactuals.

Paper: Liu, Z., Wang, T., Shi, J., Xu, L., Zhou, Q., Tripathi, P., Cai, X., &
       Hu, X. (2024). "TimeX++: Learning Time-Series Explanations with
       Information Bottleneck." Proceedings of the 41st International
       Conference on Machine Learning (ICML 2024).

Paper URL: https://arxiv.org/abs/2405.09308
GitHub (authors' code): https://github.com/zichuan-liu/TimeXplusplus

NAMING / SCOPE NOTE — read this before using this module, mirroring the note
at the top of ``cfts/cf_timex/timex_cf.py`` for the sibling "TimeX" package:

TimeX++ as published is an *explainer*, not a counterfactual generator. Its
pipeline (paper Section 3) is:

  1. Explanation extraction: a transformer encoder-decoder maps the input X
     to a stochastic per-timestep mask distribution (a "sample" of which
     timesteps are relevant).
  2. Compactness quantifier: a KL term pulls the mask's marginal probability
     toward a target sparsity, plus a "connective" penalty that favours
     mask segments that are temporally contiguous over ones scattered
     across the series.
  3. Reference instance: a baseline X_r is built by keeping the masked-in
     (important) part of X and replacing the masked-out part with Gaussian
     noise matched to the data distribution ("Gaussian padding").
  4. Conditioner: an MLP Psi_theta maps [mask, X] to an "explanation-embedded
     instance" X_tilde -- an in-distribution instance that isolates what the
     mask says is relevant.
  5. Informativeness objective: X_tilde is pushed to (a) keep the *original*
     model prediction (label-consistency, via a JS-divergence term) while
     (b) staying close in distribution to the reference X_r and away from
     out-of-distribution artefacts.
  6. Total: L = L_LC + alpha * L_M + beta * (L_KL + L_dr).

Step 5(a) is exactly what makes this an *explanation* method rather than a
counterfactual one: X_tilde is trained to keep the model's prediction
unchanged, not to flip it. To fit this repository's ``<name>_cf`` contract
(a counterfactual towards ``target_class``), this module keeps the rest of
the architecture and objective intact but retargets step 5(a): the
label-consistency term is computed against ``target_class`` instead of the
query's own predicted class, so the conditioner learns to produce an
in-distribution instance the *black-box classifier flips to*, using the
mask/compactness/reference machinery to keep that edit sparse and plausible.
This is the same "keep the mechanism, retarget the objective toward
``target_class``" move ``cf_timex/timex_cf.py`` documents for its own
(differently-shaped) Wachter-style TimeX-CF -- see that module's docstring
for the fuller discussion of why two unrelated papers share the "TimeX" name
in this codebase.

Further simplifications made to keep training tractable within a single
``<name>_cf`` call (same "train inline, budget far below the paper's offline
configuration" tradeoff ``cf_diffcf/diffcf.py`` and ``cf_cfe4mts/cfe4mts.py``
make for their own trained components):

  - A fresh extractor/conditioner pair is trained per call and discarded
    after producing one counterfactual (``timexplusplus_cf``), matching this
    repository's single-call signature. Every training instance's
    label-consistency loss targets the *query's* fixed ``target_class``
    (rather than a per-instance randomly sampled class, as CFE4MTS's
    amortised noiser does) since the trained pair is thrown away afterwards
    anyway. ``timexplusplus_fit`` / ``timexplusplus_generate`` are exposed
    separately for callers who want to reuse one trained pair across several
    queries that share a target class.
  - The stochastic mask uses the binary/relaxed-Bernoulli ("Gumbel-Sigmoid")
    concrete relaxation of Maddison et al. (2017), not the specific
    transformer-decoder parameterisation of the paper's mask distribution.
  - L_KL (distribution-shift) is approximated by first-and-second-moment
    matching between X_tilde and the training batch's per-channel
    mean/std, rather than a parametric KL between full distributions.
  - Model sizes (encoder depth/width) and epoch counts default far below
    what an offline-trained TimeX++ would use; pass a larger
    ``d_model``/``num_encoder_layers``/``epochs`` for higher fidelity at the
    cost of runtime.
  - Default ``alpha``/``beta`` are tuned lower than a literal reading of the
    paper's own weighting (``alpha=2.0``, ``beta=1.0``) would suggest:
    at those values the compactness/distribution terms routinely outweighed
    the (retargeted) validity term during testing and the search never
    reached ``target_class``. This repository's other trained methods keep
    validity dominant over their other loss terms (e.g. CFE4MTS's
    ``lambda_clas=10.0`` versus ``lambda_gen=1.0``/``lambda_dist=0.01``);
    the defaults here (``alpha=0.5``, ``beta=0.25``) follow the same
    principle. ``alpha``/``beta`` remain free to set to any value, including
    the paper's own, for a more explanation-like (lower-validity,
    higher-compactness) counterfactual.
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

    Paper's label-consistency loss L_LC (Section 3.3); here computed between
    the model's prediction on X_tilde and a one-hot ``target_class`` instead
    of the query's own predicted class -- see module docstring.
    """
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p.clamp_min(eps)) - torch.log(m.clamp_min(eps)))).sum(dim=-1)
    kl_qm = (q * (torch.log(q.clamp_min(eps)) - torch.log(m.clamp_min(eps)))).sum(dim=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def _bernoulli_kl(p: torch.Tensor, r: float, eps: float = 1e-6) -> torch.Tensor:
    """KL(Bernoulli(p) || Bernoulli(r)), p a scalar mean mask probability.

    Part of the compactness quantifier L_M: pulls the mask's average
    "selected" probability toward the target sparsity `r`.
    """
    p = p.clamp(eps, 1 - eps)
    r_t = torch.as_tensor(r, dtype=p.dtype, device=p.device).clamp(eps, 1 - eps)
    return p * (torch.log(p) - torch.log(r_t)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - r_t))


def _binary_concrete_sample(logits: torch.Tensor, tau: float, training: bool) -> torch.Tensor:
    """Relaxed-Bernoulli ("Gumbel-Sigmoid") sample from mask logits.

    Maddison et al. (2017), "The Concrete Distribution", binary case.
    Stochastic and differentiable during training; deterministic
    (plain sigmoid) at inference.
    """
    if not training:
        return torch.sigmoid(logits)
    u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    noise = torch.log(u) - torch.log1p(-u)
    return torch.sigmoid((logits + noise) / tau)


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
    """Transformer-encoder explanation extractor producing per-timestep mask logits.

    Paper: "transformer encoder-decoder architecture producing stochastic
    masks." Scaled down to a single encoder + linear head by default to fit
    this repository's per-call training budget (the same scale-down
    ``cf_cfe4mts/cfe4mts.py``'s single-layer-LSTM discriminator makes versus
    the paper's own architecture).
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


class ExplanationConditioner(nn.Module):
    """MLP conditioner Psi_theta mapping [X, mask] to an explanation-embedded instance.

    Paper: "MLP-based network generating in-distribution instances." Applied
    per-timestep with shared weights (so it is agnostic to series length),
    predicting a residual on top of the query -- the same "predict a
    perturbation, not an unconstrained absolute output" stability choice
    most gradient-based methods in this repository make (e.g. CFE4MTS's
    central noiser, GLACIER, SPARCE).
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
        """x, mask: (B, L, C) -> explanation-embedded instance X_tilde, (B, L, C)."""
        return x + self.net(torch.cat([x, mask], dim=-1))


# ---------------------------------------------------------------------------
# Fit / generate / cf
# ---------------------------------------------------------------------------

@dataclass
class FittedTimeXPlusPlus:
    """A trained TimeX++-CF extractor/conditioner pair, targeting one fixed class.

    Reusable across queries that (a) share this ``target_class`` and (b)
    have the same (num_channels, seq_len) shape the pair was trained on --
    see ``timexplusplus_generate``.
    """

    extractor: "ExplanationExtractor"
    conditioner: "ExplanationConditioner"
    num_channels: int
    seq_len: int
    num_classes: int
    target_class: int
    device: torch.device
    data_mean: torch.Tensor  # (C,)
    data_std: torch.Tensor  # (C,)
    outputs_are_probs: bool
    history: dict | None = None


def timexplusplus_fit(
    dataset: list | np.ndarray,
    model: torch.nn.Module,
    target_class: int,
    alpha: float = 0.5,
    beta: float = 0.25,
    r: float = 0.5,
    gamma: float = 0.1,
    d_model: int = 32,
    nhead: int = 2,
    num_encoder_layers: int = 1,
    hidden_dim_conditioner: int = 32,
    tau: float = 1.0,
    lr: float = 3e-3,
    epochs: int = 80,
    batch_size: int = 16,
    max_train_samples: int = 150,
    seed: int | None = None,
    verbose: bool = False,
) -> FittedTimeXPlusPlus:
    """Train the TimeX++-CF explanation extractor + conditioner (see module docstring).

    Minimises, per batch::

        L_LC = mean JS( softmax(model(X_tilde)), one_hot(target_class) )
        L_M  = KL(Bernoulli(mean mask prob) || Bernoulli(r))
               + gamma * mean |mask[t] - mask[t-1]|          (connective term)
        L_dr = MSE(X_tilde, X_r)                             (reference-distance term)
        L_KL = || mean(X_tilde) - data_mean ||^2 + || std(X_tilde) - data_std ||^2
               (moment-matching approximation of the paper's distribution-shift KL)
        L    = L_LC + alpha * L_M + beta * (L_KL + L_dr)

    where X_r is the "Gaussian padding" reference instance: the masked-in
    part of X kept as-is, the masked-out part replaced by noise sampled from
    the training batch's per-channel mean/std.

    Parameters
    ----------
    dataset:
        Sequence of (x, y) pairs (or an (N, C, L) array) used to train the
        extractor/conditioner. Required.
    model:
        Frozen PyTorch classifier being explained; ``(B, C, L) -> (B, num_classes)``.
    target_class:
        Class every training instance's label-consistency loss targets (see
        module docstring for why this is fixed rather than per-instance).
    alpha, beta:
        Weights on the compactness (L_M) and distribution (L_KL + L_dr) terms,
        matching the paper's objective L = L_LC + alpha*L_M + beta*(L_KL+L_dr).
    r:
        Target mask sparsity (mean fraction of the series treated as
        "selected"/kept rather than replaced).
    gamma:
        Weight of the connective (temporal-smoothness) penalty inside L_M.
    d_model, nhead, num_encoder_layers:
        Size of the transformer explanation extractor.
    hidden_dim_conditioner:
        Hidden width of the MLP conditioner.
    tau:
        Temperature of the binary-concrete mask relaxation.
    lr:
        Adam learning rate (extractor and conditioner share one optimiser).
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
    conditioner = ExplanationConditioner(C, hidden_dim_conditioner).to(device)
    optimizer = Adam(list(extractor.parameters()) + list(conditioner.parameters()), lr=lr)

    n = ts_lc.shape[0]
    history: dict[str, list[float]] = {"loss": [], "lc": [], "m": [], "kl": [], "dr": []}
    try:
        for epoch in range(epochs):
            extractor.train()
            conditioner.train()
            epoch_losses = {"loss": [], "lc": [], "m": [], "kl": [], "dr": []}

            for idx in _batches(n, batch_size, rng):
                idx_t = torch.as_tensor(idx, device=device)
                x = ts_lc[idx_t]  # (b, L, C)
                b = x.shape[0]

                optimizer.zero_grad()

                mask_logits = extractor(x)
                mask_prob = torch.sigmoid(mask_logits)
                mask_sample = _binary_concrete_sample(mask_logits, tau, training=True)

                noise = torch.randn_like(x) * data_std + data_mean
                x_r = mask_sample * x + (1 - mask_sample) * noise

                x_tilde = conditioner(x, mask_sample)

                raw_scores = model(x_tilde.transpose(1, 2))
                probs = _to_probs(raw_scores, outputs_are_probs)
                target_onehot = target_onehot_row.unsqueeze(0).expand(b, -1)
                l_lc = _js_divergence(probs, target_onehot).mean()

                kl_term = _bernoulli_kl(mask_prob.mean(), r)
                connective = (mask_prob[:, 1:, :] - mask_prob[:, :-1, :]).abs().mean()
                l_m = kl_term + gamma * connective

                l_dr = F.mse_loss(x_tilde, x_r)
                l_kl = ((x_tilde.mean(dim=(0, 1)) - data_mean) ** 2).mean() + \
                    ((x_tilde.std(dim=(0, 1)) - data_std) ** 2).mean()

                loss = l_lc + alpha * l_m + beta * (l_kl + l_dr)
                loss.backward()
                optimizer.step()

                epoch_losses["loss"].append(loss.item())
                epoch_losses["lc"].append(l_lc.item())
                epoch_losses["m"].append(l_m.item())
                epoch_losses["kl"].append(l_kl.item())
                epoch_losses["dr"].append(l_dr.item())

            for k in history:
                history[k].append(float(np.mean(epoch_losses[k])) if epoch_losses[k] else float("nan"))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(
                    f"[timexplusplus_fit] epoch {epoch:4d}  "
                    f"L={history['loss'][-1]:.4f}  L_LC={history['lc'][-1]:.4f}  "
                    f"L_M={history['m'][-1]:.4f}  L_KL={history['kl'][-1]:.4f}  "
                    f"L_dr={history['dr'][-1]:.4f}"
                )
    finally:
        for p, rg in zip(model.parameters(), orig_requires_grad):
            p.requires_grad_(rg)

    extractor.eval()
    conditioner.eval()
    return FittedTimeXPlusPlus(
        extractor=extractor, conditioner=conditioner, num_channels=C, seq_len=L,
        num_classes=num_classes, target_class=target_class, device=device,
        data_mean=data_mean, data_std=data_std, outputs_are_probs=outputs_are_probs,
        history=history,
    )


def timexplusplus_generate(
    fitted: FittedTimeXPlusPlus,
    sample: np.ndarray | list,
    model: torch.nn.Module,
    return_mask: bool = False,
    verbose: bool = False,
):
    """Generate a counterfactual with an already-trained extractor/conditioner pair.

    A single forward pass -- no training. `fitted` must have been trained
    with the same `target_class` this query should move to (retrain with
    `timexplusplus_fit` for a different target class).

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
        If True, also return the deterministic explanation mask (same shape
        as `sample`), the closest thing to TimeX++'s original saliency
        output. Off by default so the return arity matches this
        repository's ``(counterfactual, scores)`` ``<name>_cf`` contract.
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
    fitted.conditioner.eval()
    with torch.no_grad():
        mask_logits = fitted.extractor(sample_lc)
        mask_prob = torch.sigmoid(mask_logits)
        x_tilde = fitted.conditioner(sample_lc, mask_prob)
        cf_cl_t = x_tilde.squeeze(0).T  # (C, L)
        scores_cf = detach_to_numpy(model(cf_cl_t.unsqueeze(0))).reshape(-1)

    cf = detach_to_numpy(cf_cl_t).reshape(C, L)

    if verbose:
        label_cf = int(np.argmax(scores_cf))
        print(
            f"[timexplusplus_generate] target={fitted.target_class}  "
            f"counterfactual={label_cf}  mean_mask={float(mask_prob.mean()):.3f}"
        )

    cf_out = revert_orientation(cf, ori)
    if not return_mask:
        return cf_out, scores_cf
    mask_np = detach_to_numpy(mask_prob.squeeze(0).T).reshape(C, L)
    return cf_out, scores_cf, revert_orientation(mask_np, ori)


def timexplusplus_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    alpha: float = 0.5,
    beta: float = 0.25,
    r: float = 0.5,
    gamma: float = 0.1,
    d_model: int = 32,
    nhead: int = 2,
    num_encoder_layers: int = 1,
    hidden_dim_conditioner: int = 32,
    tau: float = 1.0,
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
    """TimeX++-CF: information-bottleneck extractor + conditioner, retargeted
    to produce a counterfactual towards `target_class` (see module docstring
    for the explainer -> counterfactual reinterpretation and the
    simplifications made versus the published method).

    Trains a fresh :func:`timexplusplus_fit` extractor/conditioner pair on
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
        extractor/conditioner. Required.
    alpha, beta, r, gamma, d_model, nhead, num_encoder_layers,
    hidden_dim_conditioner, tau, lr, epochs, batch_size, max_train_samples,
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
            "timexplusplus_cf requires a training dataset (it trains an "
            "explanation extractor + conditioner, it does not run a "
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
        hidden_dim_conditioner=hidden_dim_conditioner, tau=tau, lr=lr,
        epochs=epochs, batch_size=batch_size, max_train_samples=max_train_samples,
        seed=seed, verbose=verbose,
    )
    return timexplusplus_generate(fitted, sample, model, return_mask=return_mask, verbose=verbose)
