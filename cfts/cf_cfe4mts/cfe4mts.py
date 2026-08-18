# https://hal.science/hal-04928456v2/file/m1254.pdf

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
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


####
# CFE4MTS: Plausible Conditional Generation-based Counterfactual Explanations
#          for Multivariate Time Series Classification
#
# Paper: Sevellec, P., Fromont, E., Gaudel, R., Roze, L., & Sammarco, M. (2025).
#        "Plausible Conditional Generation-based Counterfactual Explanations
#        for Multivariate Times Series Classification."
#        ECAI 2025 - European Conference on Artificial Intelligence.
#
# Paper URL: https://hal.science/hal-04928456v2/file/m1254.pdf
# GitHub (authors' code): https://github.com/PaulSevellec/CFE4MTS
#
# CFE4MTS is a conditional, generation-based counterfactual method for MTS
# classification. It is a multivariate, class-conditional extension of
# CFE4SITS [Dantas et al., 2023]. A GAN-like architecture is trained once on
# the dataset; at inference, a counterfactual for any query/target-class pair
# is produced by a single forward pass (no per-sample optimisation).
#
# Architecture (Figure 1 of the paper, central-noiser / central-discriminator
# variant, which the paper's ablations show is the best performing one):
#   - Noiser N(X, y_target) -> delta: a "generator" that, given the query MTS
#     and a one-hot target class, predicts an additive perturbation delta
#     (not the counterfactual itself, cf. CounterGAN). The counterfactual is
#     X_CF = X + delta. The central noiser concatenates all channels into one
#     vector and feeds it through a 3-layer perceptron (BatchNorm + ReLU +
#     Dropout after both hidden layers).
#   - Discriminator D(X, y) -> [0, 1]: a single-direction LSTM (only the last
#     time step's output is used) conditioned on the class, judging whether
#     the (MTS, class) pair is plausible/real.
#   - Classifier (the frozen black-box `model` being explained): used only to
#     evaluate/backpropagate through the predicted class of X_CF, never
#     updated.
#
# Loss functions minimised by the noiser (Section 3):
#   L_cla  = CrossEntropy(classifier(X_CF), y_target)             (validity)
#   L_dist = mean_i sum_k sum_t d(t, t~_k^i)^2 * |delta_{k,t}^i|  (sparsity)
#            with t~_k = argmax_t |delta_{k,t}| and the *circular* distance
#            d(t, t~_k) = min((t - t~_k) % T, (t~_k - t) % T)
#   L_gen  = -mean_i log D(X_CF^i ; y_target^i)                    (fool D)
#   L_noiser = lambda_gen * L_gen + lambda_clas * L_cla + lambda_dist * L_dist
#
# Loss minimised by the discriminator:
#   L_disc = -mean_i [log D(X^i ; y_hat^i) + log(1 - D(X_CF^i ; y_target^i))]
#
# Training target classes are sampled uniformly at random (different from the
# classifier's predicted label) for every training instance, so that a single
# trained noiser generalises to any (query, target class) pair at inference
# ("conditional" setting of Table 4, the best-performing one in the paper).
####


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode integer class labels."""
    return torch.nn.functional.one_hot(labels, num_classes).float()


def _sample_random_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Uniformly sample, for each label, a different class in [0, num_classes)."""
    offset = torch.randint(1, num_classes, labels.shape, device=labels.device)
    return (labels + offset) % num_classes


def _outputs_are_probabilities(scores: np.ndarray) -> bool:
    """Detect whether `model`'s raw output is already a softmax distribution.

    Most classifiers in this repository return unnormalised logits, for
    which `nn.CrossEntropyLoss` (softmax + NLL) is the correct match to the
    paper's L_cla = -log p(y_target). A few pretrained models (e.g.
    `SimpleCNN`, used for FordA) end in their own `nn.Softmax`, so applying
    `CrossEntropyLoss` on top would silently double-softmax the output and
    cripple the validity gradient. Checked once (from a batch of real model
    outputs) rather than per-sample, since a given model is consistently one
    or the other.
    """
    return bool(
        np.all(scores >= -1e-6) and np.allclose(scores.sum(axis=1), 1.0, atol=1e-3)
    )


def _cla_loss(scores: torch.Tensor, target: torch.Tensor, outputs_are_probs: bool) -> torch.Tensor:
    """Paper's L_cla = -mean log p(y_target), from whatever `model` returns."""
    if outputs_are_probs:
        p_target = scores.gather(1, target.unsqueeze(1)).squeeze(1)
        return -torch.log(p_target.clamp_min(1e-12)).mean()
    return nn.functional.cross_entropy(scores, target)


def _dist_loss(delta: torch.Tensor) -> torch.Tensor:
    """CFE4MTS sparsity/proximity loss L_dist (paper Eq. below Eq. 1).

    Penalises perturbation mass proportionally to its *circular* distance
    (mod T) from the time step of peak absolute perturbation, per channel,
    so that changes concentrate around a single contiguous window instead of
    being scattered across the sequence.
    """
    B, C, T = delta.shape
    abs_delta = delta.abs()
    t_tilde = abs_delta.argmax(dim=2).float().unsqueeze(-1)  # (B, C, 1)
    t_idx = torch.arange(T, device=delta.device, dtype=delta.dtype).view(1, 1, T)
    diff = t_idx - t_tilde
    d = torch.minimum(torch.remainder(diff, T), torch.remainder(-diff, T))
    return (d.pow(2) * abs_delta).sum(dim=(1, 2)).mean()


class CentralNoiser(nn.Module):
    """Central noiser (CN): 3-layer perceptron predicting a whole-MTS perturbation.

    Concatenates the flattened MTS with the one-hot target class and outputs
    a perturbation of the same (C, L) shape as the input, per Section 3 /
    Section 4.3 of the paper ("central noiser processes a concatenation of
    the channels of the MTS ... 3-layer perceptron with batch normalisation,
    ReLU ... and drop-out ... after both hidden layers").
    """

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        flat_dim = num_channels * seq_len
        self.net = nn.Sequential(
            nn.Linear(flat_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, x: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        inp = torch.cat([x.reshape(b, -1), target_onehot], dim=1)
        return self.net(inp).reshape(b, self.num_channels, self.seq_len)


class CentralDiscriminator(nn.Module):
    """Central discriminator (CD): single-direction LSTM + FC on the last step.

    The class conditioning is broadcast to every time step and concatenated
    to the per-step channel values before the LSTM, so the discriminator
    scores the plausibility of the (MTS, class) pair.
    """

    def __init__(self, num_channels: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(num_channels + num_classes, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, class_onehot: torch.Tensor) -> torch.Tensor:
        b, c, length = x.shape
        x_t = x.transpose(1, 2)  # (B, L, C)
        cond = class_onehot.unsqueeze(1).expand(b, length, class_onehot.shape[1])
        out, _ = self.lstm(torch.cat([x_t, cond], dim=2))
        return torch.sigmoid(self.fc(out[:, -1, :]))


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    """Yield shuffled index batches, dropping a trailing batch of size 1
    (BatchNorm1d requires batch size > 1 in train mode)."""
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if len(idx) > 1:
            yield idx


@dataclass
class FittedCFE4MTS:
    """A trained CFE4MTS noiser, reusable across many query samples.

    CFE4MTS's central selling point (Section 4.4) is that training happens
    once on the dataset and inference is then a single forward pass per
    query -- unlike per-sample optimisation methods. `cfe4mts_fit` performs
    the (comparatively expensive) training step; `cfe4mts_generate` applies
    the result to as many samples/target classes as needed without retraining.
    """

    noiser: "CentralNoiser"
    num_channels: int
    seq_len: int
    num_classes: int
    device: torch.device
    history: dict[str, list[float]] | None = None


def cfe4mts_fit(
    dataset: list | np.ndarray,
    model: torch.nn.Module,
    hidden_dim_noiser: int = 128,
    hidden_dim_disc: int = 64,
    dropout: float = 0.3,
    lambda_gen: float = 1.0,
    lambda_clas: float = 10.0,
    lambda_dist: float = 0.01,
    lr_noiser: float = 1e-3,
    lr_disc: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 32,
    max_train_samples: int = 500,
    seed: int | None = None,
    verbose: bool = False,
) -> FittedCFE4MTS:
    """Train the CFE4MTS central-noiser / central-discriminator (Section 3).

    See `cfe4mts_cf` for the loss functions and architecture. Call this once
    per (dataset, model) pair, then reuse the returned `FittedCFE4MTS` with
    `cfe4mts_generate` for every query sample -- retraining per sample (as
    `cfe4mts_cf` does, to match this repository's single-call CF signature)
    defeats the point of the method.

    Parameters
    ----------
    dataset:
        Sequence of (x, y) pairs (or an (N, C, L) array) used to train the
        noiser/discriminator.
    model:
        Frozen PyTorch classifier being explained; ``(B, C, L) -> (B, num_classes)``.
    hidden_dim_noiser, hidden_dim_disc:
        Hidden sizes of the central noiser (MLP) and central discriminator (LSTM).
    dropout:
        Dropout probability applied after both hidden layers of the noiser.
    lambda_gen, lambda_clas, lambda_dist:
        Weights of the noiser's adversarial, classification (validity) and
        distance (sparsity) losses, matching the paper's L_noiser.
    lr_noiser, lr_disc:
        Adam learning rates for the noiser and discriminator.
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
    FittedCFE4MTS
        Trained noiser plus the shape/device metadata `cfe4mts_generate` needs.
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

    # --- frozen classifier: predicted labels for the training set, and no grads ---
    with torch.no_grad():
        train_scores = batched_predict(model, ts, device, batch_size=max(batch_size, 64))
    train_labels_np = np.argmax(train_scores, axis=1)
    num_classes = train_scores.shape[1]
    outputs_are_probs = _outputs_are_probabilities(train_scores)
    if verbose:
        print(
            f"[cfe4mts_fit] model output detected as "
            f"{'softmax probabilities' if outputs_are_probs else 'raw logits'}"
        )

    orig_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    ts_t = numpy_to_torch(ts, device)
    labels_t = torch.tensor(train_labels_np, dtype=torch.long, device=device)

    noiser = CentralNoiser(C, L, num_classes, hidden_dim_noiser, dropout).to(device)
    discriminator = CentralDiscriminator(C, num_classes, hidden_dim_disc).to(device)
    noiser_opt = Adam(noiser.parameters(), lr=lr_noiser)
    disc_opt = Adam(discriminator.parameters(), lr=lr_disc)
    bce_loss = nn.BCELoss()

    n = ts_t.shape[0]
    history: dict[str, list[float]] = {"noiser": [], "disc": []}
    try:
        for epoch in range(epochs):
            noiser.train()
            discriminator.train()
            gen_losses, disc_losses = [], []

            for idx in _batches(n, batch_size, rng):
                idx_t = torch.as_tensor(idx, device=device)
                x_batch = ts_t[idx_t]
                yhat_batch = labels_t[idx_t]
                ytgt_batch = _sample_random_targets(yhat_batch, num_classes)
                yhat_oh = _one_hot(yhat_batch, num_classes)
                ytgt_oh = _one_hot(ytgt_batch, num_classes)

                # --- noiser step ---
                noiser_opt.zero_grad()
                delta = noiser(x_batch, ytgt_oh)
                x_cf = x_batch + delta

                l_cla = _cla_loss(model(x_cf), ytgt_batch, outputs_are_probs)
                l_dist = _dist_loss(delta)
                l_gen = bce_loss(discriminator(x_cf, ytgt_oh), torch.ones(len(idx), 1, device=device))
                l_noiser = lambda_gen * l_gen + lambda_clas * l_cla + lambda_dist * l_dist
                l_noiser.backward()
                noiser_opt.step()
                gen_losses.append(l_noiser.item())

                # --- discriminator step (fresh, detached counterfactuals) ---
                disc_opt.zero_grad()
                with torch.no_grad():
                    x_cf_fresh = x_batch + noiser(x_batch, ytgt_oh)
                d_real = discriminator(x_batch, yhat_oh)
                d_fake = discriminator(x_cf_fresh, ytgt_oh)
                l_disc = bce_loss(d_real, torch.ones(len(idx), 1, device=device)) + \
                    bce_loss(d_fake, torch.zeros(len(idx), 1, device=device))
                l_disc.backward()
                disc_opt.step()
                disc_losses.append(l_disc.item())

            history["noiser"].append(float(np.mean(gen_losses)))
            history["disc"].append(float(np.mean(disc_losses)))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(
                    f"[cfe4mts_fit] epoch {epoch:4d}  "
                    f"L_noiser={history['noiser'][-1]:.4f}  "
                    f"L_disc={history['disc'][-1]:.4f}"
                )
    finally:
        for p, rg in zip(model.parameters(), orig_requires_grad):
            p.requires_grad_(rg)

    noiser.eval()
    return FittedCFE4MTS(
        noiser=noiser, num_channels=C, seq_len=L, num_classes=num_classes,
        device=device, history=history,
    )


def cfe4mts_generate(
    fitted: FittedCFE4MTS,
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a counterfactual with an already-trained noiser (see `cfe4mts_fit`).

    A single forward pass -- no training -- which is CFE4MTS's main practical
    advantage over per-sample optimisation methods (paper Section 4.4).

    Parameters
    ----------
    fitted:
        Output of `cfe4mts_fit`, trained on the same `model` and on data with
        the same (C, L) shape as `sample`.
    sample:
        Query time series. Accepts 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``.
    model:
        The same frozen classifier `fitted` was trained against.
    target_class:
        Desired class of the counterfactual. Defaults to the second most
        likely class for `sample` under `model` when not given.
    verbose:
        Print the original/target/counterfactual class when ``True``.

    Returns
    -------
    counterfactual : np.ndarray
        Counterfactual time series in the **same shape / orientation** as
        `sample`.
    scores : np.ndarray, shape (num_classes,)
        Model output for the counterfactual.
    """
    device = fitted.device
    sample_cl, ori = ensure_cl(np.asarray(sample, dtype=np.float32))
    C, L = sample_cl.shape
    if (C, L) != (fitted.num_channels, fitted.seq_len):
        raise ValueError(
            f"sample shape (C={C}, L={L}) does not match the shape "
            f"(C={fitted.num_channels}, L={fitted.seq_len}) `fitted` was trained on."
        )

    with torch.no_grad():
        scores_orig = detach_to_numpy(
            model(numpy_to_torch(sample_cl.reshape(1, C, L), device))
        ).reshape(-1)
    label_orig = int(np.argmax(scores_orig))

    if target_class is None:
        target_class = int(np.argsort(scores_orig)[::-1][1])

    sample_t = numpy_to_torch(sample_cl.reshape(1, C, L), device)
    target_oh = _one_hot(torch.tensor([target_class], device=device), fitted.num_classes)
    with torch.no_grad():
        cf_t = sample_t + fitted.noiser(sample_t, target_oh)
        scores_cf = detach_to_numpy(model(cf_t)).reshape(-1)
    cf = detach_to_numpy(cf_t).reshape(C, L)

    if verbose:
        label_cf = int(np.argmax(scores_cf))
        print(
            f"[cfe4mts_generate] original={label_orig}  target={target_class}  "
            f"counterfactual={label_cf}"
        )

    return revert_orientation(cf, ori), scores_cf


def cfe4mts_cf(
    sample: np.ndarray | list,
    dataset: list | np.ndarray,
    model: torch.nn.Module,
    target_class: int | None = None,
    hidden_dim_noiser: int = 128,
    hidden_dim_disc: int = 64,
    dropout: float = 0.3,
    lambda_gen: float = 1.0,
    lambda_clas: float = 10.0,
    lambda_dist: float = 0.01,
    lr_noiser: float = 1e-3,
    lr_disc: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 32,
    max_train_samples: int = 500,
    seed: int | None = None,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """CFE4MTS: conditional generation-based counterfactuals for MTS.

    Trains the central-noiser / central-discriminator variant of CFE4MTS
    (Section 3 of the paper) on `dataset`, then generates a counterfactual
    for `sample` towards `target_class` with a single forward pass of the
    trained noiser. Follows the same signature pattern as every other CF
    method in this repository (see `cf__abstract.abstract.abstract_cf`).

    This is a thin `cfe4mts_fit` + `cfe4mts_generate` composition kept for
    that single-call signature; it retrains a fresh noiser on every call.
    When explaining more than one sample against the same `dataset`/`model`,
    call `cfe4mts_fit` once and reuse it with `cfe4mts_generate` instead --
    that is the method's actual, near-instantaneous inference regime.

    Parameters
    ----------
    sample:
        Query time series. Accepts 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``.
    dataset:
        Sequence of (x, y) pairs (or an (N, C, L) array) used to train the
        noiser/discriminator. Required: CFE4MTS is a trained generative
        method, not a per-sample optimiser.
    model:
        Frozen PyTorch classifier being explained; ``(B, C, L) -> (B, num_classes)``.
    target_class:
        Desired class of the counterfactual. Defaults to the second most
        likely class for `sample` under `model` when not given.
    hidden_dim_noiser, hidden_dim_disc:
        Hidden sizes of the central noiser (MLP) and central discriminator (LSTM).
    dropout:
        Dropout probability applied after both hidden layers of the noiser.
    lambda_gen, lambda_clas, lambda_dist:
        Weights of the noiser's adversarial, classification (validity) and
        distance (sparsity) losses, matching the paper's L_noiser.
    lr_noiser, lr_disc:
        Adam learning rates for the noiser and discriminator.
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
    counterfactual : np.ndarray
        Counterfactual time series in the **same shape / orientation** as
        `sample`.
    scores : np.ndarray, shape (num_classes,)
        Model output for the counterfactual.

    Example
    -------
    >>> cf, scores = cfe4mts_cf(sample_np, dataset_train, model, target_class=2)
    """
    if dataset is None:
        raise ValueError("cfe4mts_cf requires a training dataset (it trains a "
                          "conditional noiser/discriminator, it does not run a "
                          "per-sample optimisation).")

    fitted = cfe4mts_fit(
        dataset,
        model,
        hidden_dim_noiser=hidden_dim_noiser,
        hidden_dim_disc=hidden_dim_disc,
        dropout=dropout,
        lambda_gen=lambda_gen,
        lambda_clas=lambda_clas,
        lambda_dist=lambda_dist,
        lr_noiser=lr_noiser,
        lr_disc=lr_disc,
        epochs=epochs,
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        seed=seed,
        verbose=verbose,
    )
    return cfe4mts_generate(fitted, sample, model, target_class=target_class, verbose=verbose)
