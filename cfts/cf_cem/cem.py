import numpy as np
import torch
import torch.nn as nn


####
# Contrastive Explanations for a Deep Learning Model on Time-Series Data
#
# Paper: Labaien, J., Zugasti, E., De Carlos, X. (2020).
#        "Contrastive Explanations for a Deep Learning Model on Time-Series Data."
#        DaWaK 2020. LNCS vol 12393.
#        https://doi.org/10.1007/978-3-030-59065-9_19
#
# Reference implementation: IBM Contrastive Explanation Method (aen_CEM.py)
#        https://github.com/IBM/Contrastive-Explanation-Method/blob/master/aen_CEM.py
#
# Method:
#   CEM finds either Pertinent Negatives (PN) or Pertinent Positives (PP) by
#   solving a regularised optimisation problem with FISTA (Beck & Teboulle, 2009).
#
#   Pertinent Negative (PN):
#     Find minimal delta such that f(x0 + delta) != f(x0).
#     Optimisation objective:
#       min  c * f_kappa_neg(x0, delta)
#            + beta * |delta|_1
#            + |delta|_2^2
#            + gamma * |x0 + delta - AE(x0 + delta)|_2^2
#     where f_kappa_neg = max(score_y0 - max_{i!=y0} score_i, -kappa)
#     Projection constraint: delta >= 0  (only additions are allowed).
#
#   Pertinent Positive (PP):
#     Find minimal sub-set of x0 that still causes f(delta) == f(x0).
#     Optimisation objective:
#       min  c * f_kappa_pos(x0, delta)
#            + beta * |delta|_1
#            + |delta|_2^2
#            + gamma * |delta - AE(delta)|_2^2
#     where f_kappa_pos = max(max_{i!=y0} score_i - score_y0, -kappa)
#     Projection constraint: 0 <= delta <= x0  (only existing parts kept).
#
#   Binary search is used to tune the constant c.
####


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(arr, device):
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)


def _to_numpy(t):
    return t.detach().cpu().numpy()


def _ensure_ncl(sample):
    """Return sample as (C, L) numpy array and a string encoding the original shape."""
    s = np.asarray(sample, dtype=np.float32)
    if s.ndim == 1:
        return s.reshape(1, -1), "1d"
    if s.ndim == 2:
        r, c = s.shape
        if r <= c:
            return s.copy(), "cl"
        return s.T.copy(), "lc"
    raise ValueError("sample must be 1-D or 2-D (C,L) / (L,C)")


def _revert(arr_cl, ori):
    if ori == "1d":
        return arr_cl.reshape(-1)
    if ori == "lc":
        return arr_cl.T
    return arr_cl


# ---------------------------------------------------------------------------
# Optional lightweight LSTM autoencoder (used when autoencoder=None)
# ---------------------------------------------------------------------------

class _LSTMAutoencoder(nn.Module):
    """Simple LSTM-based autoencoder for time-series, as described in Section 3.2."""

    def __init__(self, n_channels, seq_len, hidden=16, latent=4):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len

        # Encoder
        self.enc_lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.enc_fc = nn.Linear(hidden, latent)

        # Decoder
        self.dec_lstm = nn.LSTM(latent, hidden, batch_first=True)
        self.dec_fc = nn.Linear(hidden, n_channels)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C) for LSTM
        B, C, L = x.shape
        x_t = x.permute(0, 2, 1)                              # (B, L, C)
        _, (h_n, _) = self.enc_lstm(x_t)                     # h_n: (1, B, hidden)
        latent = self.enc_fc(h_n.squeeze(0))                  # (B, latent)
        dec_in = latent.unsqueeze(1).expand(-1, L, -1)        # (B, L, latent)
        out, _ = self.dec_lstm(dec_in)                        # (B, L, hidden)
        recon = torch.sigmoid(self.dec_fc(out))               # (B, L, C)
        return recon.permute(0, 2, 1)                         # (B, C, L)


def train_autoencoder(
    dataset,
    n_channels,
    seq_len,
    hidden=16,
    latent=4,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    device=None,
    verbose=False,
):
    """Train a lightweight LSTM autoencoder on *dataset*.

    Parameters
    ----------
    dataset : array-like, shape (N, C, L) or list of (x, y) tuples
        Training data used to fit the autoencoder.
    n_channels : int
        Number of channels C.
    seq_len : int
        Sequence length L.
    hidden : int
        LSTM hidden size (default 16).
    latent : int
        Latent vector size (default 4).
    epochs : int
        Training epochs (default 100).
    batch_size : int
        Mini-batch size (default 32).
    lr : float
        Adam learning rate (default 1e-3).
    device : torch.device or None
        Target device (defaults to CPU).
    verbose : bool
        Print loss every 10 epochs.

    Returns
    -------
    _LSTMAutoencoder
        Trained autoencoder module (eval mode).
    """
    if device is None:
        device = torch.device("cpu")

    # Build (N, C, L) array
    if isinstance(dataset, np.ndarray) and dataset.ndim == 3:
        data = dataset.astype(np.float32)
    else:
        samples = []
        for item in dataset:
            x = item[0] if isinstance(item, (tuple, list)) else item
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            elif x.ndim == 2 and x.shape[0] > x.shape[1]:
                x = x.T
            samples.append(x)
        data = np.stack(samples, axis=0)

    ae = _LSTMAutoencoder(n_channels, seq_len, hidden=hidden, latent=latent).to(device)
    optim = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    N = len(data)

    ae.train()
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        epoch_loss = 0.0
        for start in range(0, N, batch_size):
            batch = data[perm[start:start + batch_size]]
            xb = _to_tensor(batch, device)
            optim.zero_grad()
            recon = ae(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(batch)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  AE epoch {epoch+1}/{epochs}  loss={epoch_loss/N:.6f}")

    ae.eval()
    return ae


# ---------------------------------------------------------------------------
# Core FISTA-based CEM optimiser
# ---------------------------------------------------------------------------

def _fista_step(adv_s, adv, grad, lr, beta, zt, x0, mode):
    """One FISTA iteration (projection + soft-threshold + momentum).

    Follows the IBM reference closely:
      1. Gradient descent on adv_s.
      2. Soft-threshold (elastic-net proximal) to get new adv.
      3. PN / PP projection.
      4. Momentum update to get new adv_s.
      5. PN / PP projection on adv_s.
    """
    # 1. Gradient step
    adv_s_new = adv_s - lr * grad

    # 2. Soft-threshold (proximal L1)
    diff = adv_s_new - x0
    pos = torch.clamp(diff - beta, min=0.0)
    neg = torch.clamp(diff + beta, max=0.0)
    thrsh = x0 + torch.where(diff > beta, pos, torch.where(diff < -beta, neg, torch.zeros_like(diff)))

    # 3. PN / PP projection
    if mode == "PN":
        # only keep values that are >= x0 (additions only)
        new_adv = torch.where(thrsh > x0, thrsh, x0)
    else:  # PP
        # only keep values that are <= x0 (keep existing parts)
        new_adv = torch.clamp(thrsh, min=torch.zeros_like(x0), max=x0)

    # 4. Momentum
    new_adv_s = new_adv + zt * (new_adv - adv)

    # 5. Project adv_s as well
    if mode == "PN":
        new_adv_s = torch.where(new_adv_s > x0, new_adv_s, x0)
    else:
        new_adv_s = torch.clamp(new_adv_s, min=torch.zeros_like(x0), max=x0)

    return new_adv, new_adv_s


def _attack_loss(model, adv, adv_s, x0, y0, autoencoder, c, kappa, beta, gamma, mode, device):
    """Compute overall loss and the optimisation target (uses adv_s for FISTA)."""

    def _classification_input(tensor):
        # PP uses delta = x0 - adv as the classification input
        return (x0 - tensor) if mode == "PP" else tensor

    def _ae_input(tensor):
        # AE regularises delta for PP, and adv for PN
        return (x0 - tensor) if mode == "PP" else tensor

    def _scores(inp):
        out = model(inp.unsqueeze(0))
        if out.dim() == 2:
            out = out.squeeze(0)
        return out  # (num_classes,)

    scores = _scores(_classification_input(adv))
    scores_s = _scores(_classification_input(adv_s))

    y0_score = scores[y0]
    y0_score_s = scores_s[y0]
    other_mask = torch.ones(scores.shape[0], device=device, dtype=torch.bool)
    other_mask[y0] = False
    max_other = scores[other_mask].max()
    max_other_s = scores_s[other_mask].max()

    if mode == "PN":
        # Encourage adv to be classified differently from y0
        f_attack = torch.clamp(y0_score - max_other + kappa, min=0.0)
        f_attack_s = torch.clamp(y0_score_s - max_other_s + kappa, min=0.0)
    else:  # PP
        # Encourage delta to still be classified as y0
        f_attack = torch.clamp(max_other - y0_score + kappa, min=0.0)
        f_attack_s = torch.clamp(max_other_s - y0_score_s + kappa, min=0.0)

    # Elastic-net distances (on delta, not adv)
    delta = x0 - adv if mode == "PP" else adv - x0
    delta_s = x0 - adv_s if mode == "PP" else adv_s - x0
    l2 = (delta ** 2).sum()
    l2_s = (delta_s ** 2).sum()
    l1 = delta.abs().sum()

    # AE reconstruction loss
    ae_loss = torch.tensor(0.0, device=device)
    ae_loss_s = torch.tensor(0.0, device=device)
    if autoencoder is not None:
        ae_in = _ae_input(adv).unsqueeze(0)
        ae_in_s = _ae_input(adv_s).unsqueeze(0)
        ae_loss = gamma * ((ae_in - autoencoder(ae_in)) ** 2).sum()
        ae_loss_s = gamma * ((ae_in_s - autoencoder(ae_in_s)) ** 2).sum()

    # Loss used for optimisation (L1 handled by soft-threshold, so omitted here)
    loss_opt = c * f_attack_s + l2_s + ae_loss_s

    # Full loss for bookkeeping
    loss_full = c * f_attack + l2 + ae_loss + beta * l1

    return loss_opt, loss_full, scores, f_attack


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cem_cf(
    sample,
    model,
    mode="PN",
    autoencoder=None,
    kappa=0.5,
    beta=1e-1,
    gamma=0.2,
    c_init=10.0,
    c_steps=5,
    max_iterations=1000,
    learning_rate=1e-2,
    verbose=False,
):
    """CEM counterfactual for time-series classification.

    Implements the Contrastive Explanation Method (CEM) of Dhurandhar et al. (2018)
    adapted for time-series data as described in Labaien et al. (2020), using
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for optimisation.

    Parameters
    ----------
    sample : array-like, shape (L,) | (C, L) | (L, C)
        Input time series to explain.  A single sample (no batch dimension).
    model : torch.nn.Module
        Trained classifier that accepts a batch tensor of shape (1, C, L) and
        returns raw scores of shape (1, num_classes) or (num_classes,).
    mode : {"PN", "PP"}
        ``"PN"`` finds a Pertinent Negative – the minimal change to *x0* that
        flips the prediction (counterfactual).  ``"PP"`` finds a Pertinent
        Positive – the minimal sub-sequence of *x0* that is sufficient to keep
        the original prediction.  (default ``"PN"``)
    autoencoder : torch.nn.Module or None
        Autoencoder that maps ``(1, C, L)`` tensors to ``(1, C, L)`` tensors.
        When provided, an AE reconstruction term keeps the solution on the data
        manifold.  Pass a ``_LSTMAutoencoder`` (from :func:`train_autoencoder`)
        or any compatible model.  ``None`` disables this regularisation term.
    kappa : float
        Confidence margin that controls the separation between the target class
        score and the best competing class score.  Must be in ``[0, 1]`` for
        softmax outputs.  (default ``0.5``)
    beta : float
        Weight of the L1 (sparsity) regularisation term of the elastic net.
        (default ``0.1``)
    gamma : float
        Weight of the AE reconstruction loss term.  Ignored when
        ``autoencoder=None``.  (default ``0.2``)
    c_init : float
        Initial value of the constant *c* used in binary search.  (default
        ``10.0``)
    c_steps : int
        Number of binary-search steps used to tune *c*.  (default ``5``)
    max_iterations : int
        Number of FISTA gradient steps per binary-search stage.  (default
        ``1000``)
    learning_rate : float
        Step size for the gradient-descent component of FISTA.  (default
        ``1e-2``)
    verbose : bool
        Print optimisation progress every ``max_iterations // 10`` steps.

    Returns
    -------
    cf : numpy.ndarray
        Counterfactual series in the same shape / orientation as *sample*.
        For PN mode this is ``x0 + delta``; for PP mode this is ``delta``.
        Returns *sample* unchanged if no valid counterfactual is found.
    scores_cf : numpy.ndarray, shape (num_classes,)
        Model output scores for the returned counterfactual.

    Notes
    -----
    The implementation follows the FISTA update from the IBM reference
    (aen_CEM.py) but uses PyTorch auto-differentiation instead of TensorFlow
    variable assignment, making it framework-agnostic for the model and
    autoencoder.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Prepare sample: normalise to (C, L)
    s, ori = _ensure_ncl(sample)               # (C, L) float32
    x0 = _to_tensor(s, device)                 # (C, L)

    # Initial prediction
    with torch.no_grad():
        init_scores = model(x0.unsqueeze(0))
        if init_scores.dim() == 2:
            init_scores = init_scores.squeeze(0)
        y0 = int(init_scores.argmax())

    model.eval()
    if autoencoder is not None:
        autoencoder.eval()

    # Binary search bounds
    c_lb = 0.0
    c_ub = 1e10
    c = float(c_init)

    best_dist = float("inf")
    best_cf = x0.clone()
    best_scores = _to_numpy(init_scores)

    def _success(scores_np):
        """True if the counterfactual prediction is valid for the given mode."""
        pred = int(np.argmax(scores_np))
        return (pred != y0) if mode == "PN" else (pred == y0)

    for step in range(c_steps):
        # Reset FISTA variables to x0 at the start of each binary-search step
        adv = x0.clone().detach()
        adv_s = x0.clone().detach()
        adv_s.requires_grad_(True)

        current_best_dist = float("inf")
        current_best_score = None

        for it in range(max_iterations):
            zt = float(it) / (float(it) + 3.0)

            # Forward + backward through adv_s only
            if adv_s.grad is not None:
                adv_s.grad.zero_()

            loss_opt, loss_full, scores, f_atk = _attack_loss(
                model, adv.detach(), adv_s,
                x0, y0, autoencoder,
                c, kappa, beta, gamma, mode, device,
            )
            loss_opt.backward()

            with torch.no_grad():
                grad = adv_s.grad.clone()

                # FISTA: gradient step + soft-threshold + projection + momentum
                new_adv, new_adv_s = _fista_step(
                    adv_s.detach(), adv.detach(), grad,
                    learning_rate, beta, zt, x0, mode,
                )

                adv.copy_(new_adv)
                adv_s.copy_(new_adv_s)

            adv_s.grad = None

            # Track best solution
            scores_np = _to_numpy(scores.detach())
            delta_np = (adv - x0) if mode == "PN" else (x0 - adv)
            dist = float((delta_np ** 2).sum() + beta * delta_np.abs().sum())

            if dist < current_best_dist and _success(scores_np):
                current_best_dist = dist
                current_best_score = scores_np.copy()

            if dist < best_dist and _success(scores_np):
                best_dist = dist
                best_scores = scores_np.copy()
                best_cf = adv.clone().detach()

            if verbose and (it + 1) % max(1, max_iterations // 10) == 0:
                pred = int(np.argmax(scores_np))
                print(
                    f"  [step {step+1}/{c_steps}  iter {it+1}/{max_iterations}]"
                    f"  c={c:.4f}  loss={loss_full.item():.4f}"
                    f"  f_atk={f_atk.item():.4f}  dist={dist:.4f}"
                    f"  pred={pred}  y0={y0}"
                )

        # Binary search update for c
        if current_best_score is not None and _success(current_best_score):
            c_ub = min(c_ub, c)
            if c_ub < 1e9:
                c = (c_lb + c_ub) / 2.0
        else:
            c_lb = max(c_lb, c)
            if c_ub < 1e9:
                c = (c_lb + c_ub) / 2.0
            else:
                c *= 10.0

    cf_np = _to_numpy(best_cf)      # (C, L)
    cf_out = _revert(cf_np, ori)    # restore original orientation
    return cf_out, best_scores
