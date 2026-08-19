"""
DiffCF: Generating Realistic Time-Series Counterfactuals via Diffusion-Guided Sampling

Paper: "Generating Realistic Time-Series Counterfactuals via Diffusion-Guided
        Sampling", accepted at ECML PKDD 2026.

GitHub: https://github.com/Luckilyeee/DiffCF/tree/main

DiffCF trains an unconditional denoising diffusion model (a 1-D UNet epsilon
predictor) on the training distribution, then turns a query series into a
counterfactual with an SDEdit-style guided reverse process:

  1. The query is partially noised with the forward process up to a timestep
     ``t_start = round(timesteps * start_ratio)`` instead of starting from
     pure noise, so the sampler only has to "repair" the series rather than
     invent one from scratch.
  2. The noised series is denoised back to t=0 with DDIM steps. During the
     first part of that trajectory (``guidance_start_ratio``), each step's
     denoised estimate x0_hat is nudged with the gradient of a combined
     objective: increase log p(target_class | x0_hat) under the classifier
     being explained, while decreasing its L1 distance to the original
     series and a second-derivative smoothness penalty. Each gradient term
     is normalised to unit norm before being weighted (w_cls / w_dist /
     w_smooth) so no single term can dominate purely from having a larger
     scale.
  3. If the resulting counterfactual does not flip the classifier, the
     process retries with a later ``start_ratio`` (more noise, more freedom
     to change) and a stronger classification weight, up to ``max_retries``
     times.

This module re-implements that pipeline (diffusion model, DDIM sampler,
classifier guidance, retry loop) against the ``<name>_cf`` contract used
throughout this repository. The diffusion model is untrained by default, so
``diffcf_cf`` trains a lightweight one on ``dataset`` before sampling unless
a pre-trained ``diffusion_model``/``diffusion`` pair (see
:func:`train_diffcf_diffusion`) is supplied — the same "train inline or pass
a pre-trained generator" pattern used by ``cf_latent_cf``. Model sizes and
epoch counts default much lower than the paper's offline configs (e.g.
``model_channels=64``, ``epochs=5000``) so a call finishes in reasonable time;
pass a pre-trained diffusion model for higher-fidelity counterfactuals.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def detach_to_numpy(data):
    """Move pytorch data to cpu and detach it to numpy data."""
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    """Convert numpy array to pytorch and move it to the device."""
    return torch.from_numpy(np.asarray(data, dtype=np.float32)).to(device)


# ---------------------------------------------------------------------------
# Shape helpers — normalise a raw sample to (C, L) and back, matching the
# orientation convention (1-D, (C, L), or (L, C)) used across this repo.
# ---------------------------------------------------------------------------

def ensure_cl(sample):
    arr = np.asarray(sample, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1), "1d"
    if arr.ndim == 2:
        r, c = arr.shape
        if r <= c:
            return arr.copy(), "cl"
        return arr.T.copy(), "lc"
    raise ValueError(f"sample must be 1-D or 2-D, got shape {arr.shape}")


def revert_orientation(arr_cl, ori):
    if ori == "1d":
        return arr_cl.reshape(-1)
    if ori == "lc":
        return arr_cl.T.copy()
    return arr_cl


def _extract_series(dataset, max_samples=300, seed=None):
    """Pull up to `max_samples` raw series out of a dataset of (x, y) pairs
    (or an (N, C, L) array), normalised to shape (N, C, L)."""
    n = len(dataset)
    idx = np.arange(n)
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
    series = []
    for i in idx:
        item = dataset[int(i)]
        x = item[0] if isinstance(item, (tuple, list)) else item
        x_cl, _ = ensure_cl(np.asarray(x))
        series.append(x_cl)
    return np.stack(series, axis=0)


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


# ---------------------------------------------------------------------------
# Epsilon-predictor network: a small 1-D UNet with sinusoidal time embeddings
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _group_norm_groups(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class ResBlock(nn.Module):
    """Conv1d residual block with an additive time-embedding bias."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate")
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class UNet1D(nn.Module):
    """Epsilon predictor eps_theta(x_t, t) for the forward diffusion noise.

    A compact version of DiffCF's UNet1D: `depth` down/up stages of
    stride-2 ResBlocks around a bottleneck ResBlock, with skip connections
    and a shared sinusoidal time embedding. Series shorter than
    ``2 ** depth`` samples, or lengths not divisible by it, are handled by
    replicate-padding on the way in and cropping back on the way out.
    """

    def __init__(self, in_channels, base_channels=32, depth=2, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        ch = base_channels
        for _ in range(depth):
            self.downs.append(nn.ModuleList([
                ResBlock(ch, ch * 2, time_dim),
                nn.Conv1d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            ]))
            ch *= 2

        self.mid = ResBlock(ch, ch, time_dim)

        for _ in range(depth):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
                ResBlock(ch + (ch // 2), ch // 2, time_dim),
            ]))
            ch //= 2

        self.out_norm = nn.GroupNorm(_group_norm_groups(ch), ch)
        self.out_conv = nn.Conv1d(ch, in_channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.act = nn.SiLU()

    def forward(self, x, t):
        assert x.ndim == 3, "Expected [B, C, T]"
        orig_len = x.shape[-1]

        divisor = 2 ** len(self.downs)
        if orig_len % divisor != 0:
            pad_len = divisor - (orig_len % divisor)
            x = F.pad(x, (0, pad_len), mode="replicate")

        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        h = self.in_conv(x)
        skips = []
        for block, down in self.downs:
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)
        h = self.mid(h, t_emb)
        for up, block in self.ups:
            h = up(h)
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                min_len = min(h.shape[-1], skip.shape[-1])
                h = h[..., :min_len]
                skip = skip[..., :min_len]
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
        h = self.act(self.out_norm(h))
        h = self.out_conv(h)

        return h[..., :orig_len]


# ---------------------------------------------------------------------------
# Gaussian diffusion process
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """Forward (noising) process plus the training loss for `model`."""

    def __init__(self, timesteps=1000, schedule="cosine"):
        self.timesteps = timesteps
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        """Sample x_t ~ q(x_t | x0) for each timestep in `t`."""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bar.to(x0.device)
        sqrt_alpha_bar = alpha_bar[t].view(-1, 1, 1).sqrt()
        sqrt_one_minus = (1 - alpha_bar[t]).view(-1, 1, 1).sqrt()
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def training_losses(self, model, x0, t, loss_type="mse_tv", lambda_tv=0.01, lambda_smooth=0.0):
        """Denoising loss: MSE on the predicted noise, plus optional total-
        variation / second-derivative smoothness penalties on the implied
        x0 estimate (encourages the model to denoise towards smooth series)."""
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        pred_noise = model(x_t, t)
        loss_mse = F.mse_loss(pred_noise, noise)
        if loss_type == "mse":
            return loss_mse

        alpha_bar_t = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1)
        x0_hat = (x_t - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()

        loss = loss_mse
        if lambda_tv > 0:
            loss_tv = torch.mean(torch.abs(x0_hat[:, :, 1:] - x0_hat[:, :, :-1]))
            loss = loss + lambda_tv * loss_tv
        if lambda_smooth > 0 and x0_hat.shape[-1] > 2:
            d2 = x0_hat[:, :, 2:] - 2 * x0_hat[:, :, 1:-1] + x0_hat[:, :, :-2]
            loss = loss + lambda_smooth * torch.mean(d2 ** 2)

        return loss


# ---------------------------------------------------------------------------
# DDIM sampler
# ---------------------------------------------------------------------------

def get_ddim_timesteps(ddim_steps, ddpm_steps, start_ratio=1.0):
    """Descending timestep sequence of length `ddim_steps`, starting from
    `round(ddpm_steps * start_ratio)` and ending at 0."""
    start_step = int(ddpm_steps * start_ratio)
    start_step = min(max(start_step, 1), ddpm_steps)
    return torch.linspace(start_step - 1, 0, ddim_steps, dtype=torch.long)


def predict_x0_from_eps(x_t, eps, alpha_bar_t):
    return (x_t - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()


def ddim_step(x_t, eps, t, t_prev, alpha_bar, eta=0.0):
    """One deterministic (eta=0) or stochastic DDIM update; also returns the
    x0 estimate implied by `eps` at this step."""
    alpha_bar_t = alpha_bar[t]
    alpha_bar_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
    x0 = predict_x0_from_eps(x_t, eps, alpha_bar_t)
    sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
    noise = torch.randn_like(x_t)
    dir_term = (1 - alpha_bar_prev - sigma ** 2).sqrt() * eps
    x_prev = alpha_bar_prev.sqrt() * x0 + dir_term + sigma * noise
    return x_prev, x0


# ---------------------------------------------------------------------------
# Classifier guidance applied to the denoised x0 estimate at each DDIM step
# ---------------------------------------------------------------------------

def _normalize_grad(g, eps=1e-6):
    norm = torch.sqrt(torch.sum(g ** 2, dim=(1, 2), keepdim=True))
    return g / (norm + eps)


def _clip_grad(g, max_norm):
    if not max_norm or max_norm <= 0:
        return g
    norm = torch.sqrt(torch.sum(g ** 2, dim=(1, 2), keepdim=True))
    scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
    return g * scale


def _gaussian_kernel1d(sigma, device):
    radius = max(int(3 * sigma), 1)
    xs = torch.arange(-radius, radius + 1, device=device).float()
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    return (kernel / kernel.sum()).view(1, 1, -1)


def _smooth_grad(g, sigma):
    if sigma <= 0:
        return g
    kernel = _gaussian_kernel1d(sigma, g.device)
    g_padded = F.pad(g, (kernel.shape[-1] // 2,) * 2, mode="reflect")
    return F.conv1d(g_padded, kernel.expand(g.shape[1], 1, -1), groups=g.shape[1])


def _augment(x, rng, max_shift=3, scale_std=0.01, noise_std=0.01):
    shift = int(rng.integers(-max_shift, max_shift + 1))
    scale = 1.0 + float(rng.standard_normal()) * scale_std
    noise = torch.randn_like(x) * noise_std
    return torch.roll(x * scale + noise, shifts=shift, dims=-1)


def _step_size_for_t(step_size, t, timesteps, schedule="linear", min_ratio=0.2):
    if schedule == "linear":
        ratio = max(min_ratio, 1.0 - float(t) / float(max(timesteps - 1, 1)))
        return step_size * ratio
    return step_size


def _should_apply_guidance(t, timesteps, start_ratio):
    if start_ratio is None or start_ratio >= 1.0:
        return True
    return int(t) <= int(timesteps * start_ratio)


def compute_guidance(x_t, t, eps_pred, diffusion, classifier, x_orig, target,
                      w_cls=1.0, w_dist=1.0, w_smooth=1.0,
                      stabilization="grad_smooth", aug_k=4, grad_smooth_sigma=1.5,
                      rng=None):
    """Combine a classification, a proximity and a smoothness gradient into a
    single guidance direction on the x0 estimate implied by `eps_pred`.

    Returns
    -------
    x0_hat : the (detached) denoised estimate at this step.
    g_total : the combined, weighted, unnormalised-scale gradient to add to
        x_t (each term is normalised to unit norm before weighting).
    """
    alpha_bar = diffusion.alpha_bar.to(x_t.device)
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1)
    x0_hat = (x_t - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
    x0_hat = x0_hat.clamp(-3, 3).detach().requires_grad_(True)

    if stabilization == "aug_avg":
        rng = rng or np.random.default_rng(0)
        logps = []
        for _ in range(aug_k):
            logits = classifier(_augment(x0_hat, rng))
            logps.append(F.log_softmax(logits, dim=-1)[:, target])
        logp = torch.stack(logps, dim=0).mean(dim=0)
    else:
        logits = classifier(x0_hat)
        logp = F.log_softmax(logits, dim=-1)[:, target]

    g_cls = torch.autograd.grad(logp.sum(), x0_hat, retain_graph=True)[0]
    if stabilization == "grad_smooth":
        g_cls = _smooth_grad(g_cls, grad_smooth_sigma)

    dist = torch.abs(x0_hat - x_orig).mean(dim=(1, 2))
    g_dist = torch.autograd.grad(dist.sum(), x0_hat, retain_graph=True)[0]

    g_total = w_cls * _normalize_grad(g_cls) - w_dist * _normalize_grad(g_dist)

    if w_smooth > 0 and x0_hat.shape[-1] > 2:
        d2 = x0_hat[:, :, 2:] - 2 * x0_hat[:, :, 1:-1] + x0_hat[:, :, :-2]
        smooth = (d2 ** 2).mean(dim=(1, 2))
        g_smooth = torch.autograd.grad(smooth.sum(), x0_hat, retain_graph=True)[0]
        g_total = g_total - w_smooth * _normalize_grad(g_smooth)

    return x0_hat.detach(), g_total.detach()


# ---------------------------------------------------------------------------
# Diffusion model training (used inline by diffcf_cf, or standalone to
# pre-train a model once and reuse it across calls)
# ---------------------------------------------------------------------------

def train_diffcf_diffusion(dataset, device="cpu", max_train_samples=300,
                            timesteps=1000, schedule="cosine",
                            unet_base_channels=32, unet_depth=2, unet_time_dim=64,
                            epochs=300, lr=2e-4, batch_size=32,
                            loss_type="mse_tv", lambda_tv=0.01, lambda_smooth=0.0,
                            seed=None, verbose=False):
    """Train a :class:`UNet1D` epsilon predictor + :class:`GaussianDiffusion`
    process on `dataset`, plus the per-channel z-score stats used to
    normalise series before feeding them to the model.

    Returns
    -------
    unet : trained UNet1D (in eval mode)
    diffusion : GaussianDiffusion process holding the noise schedule
    norm_stats : (mean, std) arrays of shape (C,) computed over `dataset`
    """
    train_np = _extract_series(dataset, max_samples=max_train_samples, seed=seed)
    mean = train_np.mean(axis=(0, 2), keepdims=True)
    std = train_np.std(axis=(0, 2), keepdims=True) + 1e-8
    train_norm = (train_np - mean) / std

    C = train_norm.shape[1]
    train_t = numpy_to_torch(train_norm, device)

    diffusion = GaussianDiffusion(timesteps=timesteps, schedule=schedule)
    unet = UNet1D(C, base_channels=unet_base_channels, depth=unet_depth, time_dim=unet_time_dim).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    n = train_t.shape[0]
    unet.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            x0 = train_t[batch_idx]
            t = torch.randint(0, timesteps, (x0.shape[0],), device=device, dtype=torch.long)

            optimizer.zero_grad()
            loss = diffusion.training_losses(unet, x0, t, loss_type=loss_type,
                                              lambda_tv=lambda_tv, lambda_smooth=lambda_smooth)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x0.shape[0]

        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"[DiffCF] diffusion epoch {epoch}/{epochs}  loss={epoch_loss / n:.4f}")

    unet.eval()
    return unet, diffusion, (mean.reshape(-1), std.reshape(-1))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def diffcf_cf(sample,
              model,
              target_class=None,
              dataset=None,
              diffusion_model=None,
              diffusion=None,
              norm_stats=None,
              max_train_samples=300,
              timesteps=1000,
              schedule="cosine",
              unet_base_channels=32,
              unet_depth=2,
              unet_time_dim=64,
              diffusion_epochs=300,
              diffusion_lr=2e-4,
              diffusion_batch_size=32,
              loss_type="mse_tv",
              lambda_tv=0.01,
              lambda_smooth=0.0,
              ddim_steps=100,
              start_ratio=0.2,
              max_retries=3,
              retry_start_ratio_inc=0.1,
              w_cls=1.0,
              w_dist=1.0,
              w_smooth=1.0,
              retry_w_cls_mult=1.5,
              step_size=0.08,
              step_size_schedule="linear",
              step_size_min_ratio=0.2,
              eta=0.0,
              guidance_start_ratio=0.5,
              grad_clip_norm=1.0,
              stabilization="grad_smooth",
              grad_smooth_sigma=1.5,
              aug_k=4,
              seed=None,
              verbose=False):
    """Generate a counterfactual with DiffCF's diffusion-guided sampler.

    Implements the :class:`cfts.cf__abstract.abstract.CFMethod` contract; see
    its docstring for the shared parameter/return semantics. Parameters below
    are specific to this implementation.

    Parameters
    ----------
    dataset:
        Training series used to fit the diffusion model, as (x, y) pairs or
        an (N, C, L) array. Required unless a pre-trained `diffusion_model` /
        `diffusion` / `norm_stats` triple (e.g. from
        :func:`train_diffcf_diffusion`) is supplied instead.
    diffusion_model, diffusion, norm_stats:
        A pre-trained :class:`UNet1D`, its matching :class:`GaussianDiffusion`
        process, and the (mean, std) per-channel normalisation stats it was
        trained with. When given, `dataset` is not needed and no training
        happens. All three must be supplied together.
    max_train_samples:
        Cap on how many series are drawn from `dataset` to train the
        diffusion model inline.
    timesteps, schedule:
        Number of forward-process steps and beta schedule ("cosine" or
        "linear") for the diffusion model.
    unet_base_channels, unet_depth, unet_time_dim:
        Size of the epsilon-predictor UNet. Kept small by default so inline
        training is fast; increase for higher-fidelity counterfactuals
        (matching the paper's defaults needs `unet_base_channels=64`,
        `unet_depth=3`, and thousands of training epochs).
    diffusion_epochs, diffusion_lr, diffusion_batch_size:
        Inline training schedule for the diffusion model. Ignored when a
        pre-trained model is supplied.
    loss_type, lambda_tv, lambda_smooth:
        Diffusion training loss: plain "mse", or "mse_tv" which adds a
        total-variation (and optionally second-derivative smoothness) term
        on the implied x0 estimate.
    ddim_steps:
        Number of reverse-process steps used per sampling attempt.
    start_ratio:
        Fraction of `timesteps` used to noise the query series before
        denoising (SDEdit-style partial noising) on the first attempt.
    max_retries, retry_start_ratio_inc, retry_w_cls_mult:
        On failure to flip the class, retry with `start_ratio` increased by
        `retry_start_ratio_inc` and the classification weight multiplied by
        `retry_w_cls_mult`, up to `max_retries` attempts.
    w_cls, w_dist, w_smooth:
        Weights on the (unit-normalised) classification, proximity and
        smoothness guidance gradients.
    step_size, step_size_schedule, step_size_min_ratio:
        Guidance gradient-ascent step size, and whether it stays constant or
        decays linearly (down to `step_size_min_ratio` of `step_size`) as
        the reverse process approaches t=0.
    eta:
        DDIM stochasticity (0 = deterministic DDIM, 1 = DDPM-like ancestral
        sampling).
    guidance_start_ratio:
        Guidance is only applied for timesteps <= `guidance_start_ratio *
        timesteps`, leaving the last stretch of denoising unguided so fine
        detail is filled in without further pushing the classifier score.
    grad_clip_norm:
        Per-sample L2 clip applied to the combined guidance gradient before
        the update.
    stabilization, grad_smooth_sigma, aug_k:
        How the classification gradient is stabilised before use: "none",
        "grad_smooth" (1-D Gaussian blur of the gradient, sigma=`grad_smooth_sigma`),
        or "aug_avg" (average the gradient over `aug_k` random shift/scale/noise
        augmentations of x0_hat).
    seed:
        Seed for the inline dataset subsampling and augmentation RNG.

    Example
    -------
    >>> cf, scores = diffcf_cf(sample_np, model, dataset=train_dataset, verbose=True)
    >>> label_cf = int(np.argmax(scores))
    """
    if diffusion_model is None and (dataset is None or diffusion is None or norm_stats is None):
        if dataset is None:
            raise ValueError(
                "diffcf_cf requires a dataset (to train the diffusion model) "
                "unless a pre-trained diffusion_model/diffusion/norm_stats "
                "triple is supplied."
            )

    device = next(model.parameters()).device
    rng_np = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(seed)

    # --- normalise input to (C, L) -----------------------------------------
    sample_cl, ori = ensure_cl(sample)
    C, L = sample_cl.shape

    # --- original prediction & target class ---------------------------------
    with torch.no_grad():
        scores_orig = detach_to_numpy(
            model(numpy_to_torch(sample_cl.reshape(1, C, L), device))
        ).reshape(-1)
    num_classes = scores_orig.shape[-1]
    label_orig = int(np.argmax(scores_orig))

    if target_class is None:
        if num_classes == 2:
            target_class = 1 - label_orig
        else:
            order = np.argsort(scores_orig)[::-1]
            target_class = int(order[1] if order[0] == label_orig else order[0])
    target_class = int(target_class)

    if verbose:
        print(f"[DiffCF] original class {label_orig}, target class {target_class}")

    # --- diffusion model: use pre-trained, or train inline on `dataset` -----
    if diffusion_model is not None:
        unet, mean, std = diffusion_model, norm_stats[0], norm_stats[1]
        if diffusion is None:
            diffusion = GaussianDiffusion(timesteps=timesteps, schedule=schedule)
    else:
        if verbose:
            print("[DiffCF] training diffusion model inline...")
        unet, diffusion, (mean, std) = train_diffcf_diffusion(
            dataset, device=device, max_train_samples=max_train_samples,
            timesteps=timesteps, schedule=schedule,
            unet_base_channels=unet_base_channels, unet_depth=unet_depth, unet_time_dim=unet_time_dim,
            epochs=diffusion_epochs, lr=diffusion_lr, batch_size=diffusion_batch_size,
            loss_type=loss_type, lambda_tv=lambda_tv, lambda_smooth=lambda_smooth,
            seed=seed, verbose=verbose,
        )
    unet.eval()

    mean_b = torch.as_tensor(np.asarray(mean, dtype=np.float32), device=device).view(1, C, 1)
    std_b = torch.as_tensor(np.asarray(std, dtype=np.float32), device=device).view(1, C, 1)

    x_orig = (numpy_to_torch(sample_cl.reshape(1, C, L), device) - mean_b) / std_b
    target_t = torch.tensor([target_class], device=device, dtype=torch.long)

    # --- classifier wrapper: undo z-score normalisation before calling the
    # model under explanation, which was trained on the original scale -------
    def classifier(x_norm):
        return model(x_norm * std_b + mean_b)

    # --- guided DDIM sampling with retries -----------------------------------
    best_cf_norm = None
    for retry in range(max_retries):
        start_ratio_retry = min(1.0, start_ratio + retry * retry_start_ratio_inc)
        w_cls_retry = w_cls * (retry_w_cls_mult ** retry)

        step_ts = get_ddim_timesteps(ddim_steps, diffusion.timesteps, start_ratio=start_ratio_retry).to(device)
        t_start = int(step_ts[0].item())
        t_batch = torch.full((1,), t_start, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x_orig, t_batch)

        x0_hat = x_orig
        for i, t in enumerate(step_ts):
            t_batch = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
            apply_guidance = _should_apply_guidance(t.item(), diffusion.timesteps, guidance_start_ratio)

            if apply_guidance:
                with torch.enable_grad():
                    x_t = x_t.detach().requires_grad_(True)
                    eps_pred = unet(x_t, t_batch)
                    x0_hat, g_total = compute_guidance(
                        x_t, t_batch, eps_pred, diffusion, classifier, x_orig, target_t,
                        w_cls=w_cls_retry, w_dist=w_dist, w_smooth=w_smooth,
                        stabilization=stabilization, aug_k=aug_k, grad_smooth_sigma=grad_smooth_sigma,
                        rng=rng_np,
                    )
                    g_total = _clip_grad(g_total, grad_clip_norm)
                    step_size_t = _step_size_for_t(step_size, t.item(), diffusion.timesteps,
                                                    step_size_schedule, step_size_min_ratio)
                    x_t = (x_t + step_size_t * g_total).detach()
            else:
                x_t = x_t.detach()

            t_prev = int(step_ts[i + 1].item()) if i + 1 < len(step_ts) else -1
            with torch.no_grad():
                eps_pred = unet(x_t, t_batch)
                x_t, x0_hat = ddim_step(x_t, eps_pred, t.item(), t_prev, diffusion.alpha_bar.to(device), eta=eta)

        best_cf_norm = x0_hat.detach()
        with torch.no_grad():
            cf_pred = int(torch.argmax(classifier(best_cf_norm), dim=-1).item())

        if verbose:
            print(f"[DiffCF] retry {retry}: start_ratio={start_ratio_retry:.2f}, "
                  f"w_cls={w_cls_retry:.2f}, predicted={cf_pred}, target={target_class}")

        if cf_pred == target_class:
            break

    # --- de-normalise, revert orientation, and score the final candidate ----
    cf_cl = detach_to_numpy((best_cf_norm * std_b + mean_b).reshape(C, L))
    with torch.no_grad():
        scores_cf = detach_to_numpy(
            model(numpy_to_torch(cf_cl.reshape(1, C, L), device))
        ).reshape(-1)

    return revert_orientation(cf_cl, ori), scores_cf
