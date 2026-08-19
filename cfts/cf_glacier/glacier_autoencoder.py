"""
Autoencoder for the "AE" family of Glacier variants.

Bake-off's Glacier benchmarks 8 paper-aligned variants
(`Glacier-{AE,NoAE}-{Unc,Loc,Glob,Unif}`, see `RUN_GUIDE.md`): every
combination of searching in the *original* space ("NoAE") vs. the
*latent* space of a trained autoencoder ("AE"), crossed with 4 step-weight
modes. `glacier_reimp.glacier_reimp` already supports an `autoencoder=
(encoder_fn, decoder_fn)` override for the AE/NoAE axis and
`step_weights="uniform"/"unconstrained"/"local"` for three of the four
weight modes; this module supplies the fourth ingredient — a trained
autoencoder — and `glacier_reimp.compute_global_step_weights` (added
alongside this file) supplies the fourth weight mode. See
`glacier_reimp.GLACIER_VARIANTS` / `glacier_variant()` for the convenience
wrapper that ties all 8 combinations together under bake-off's own names.

Architecture ported from bake-off's `Glacier/src/keras_models.py::Autoencoder`
(Conv1D+MaxPool encoder, Conv1D+UpSampling decoder) — same layer sizes and
downsampling factor (4x), translated from Keras to PyTorch:

    Encoder: Conv1d(C,64,k3) -> ReLU -> MaxPool(2)
             -> Conv1d(64,32,k3) -> ReLU -> MaxPool(2)
    Decoder: Conv1d(32,32,k3) -> ReLU -> Upsample(2)
             -> Conv1d(32,64,k3) -> ReLU -> Upsample(2)
             -> Conv1d(64,C,k3)               (linear output, matches Keras)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlacierAutoencoder(nn.Module):
    """Conv1D autoencoder matching bake-off Glacier's architecture.

    `encode` downsamples the time axis by 4x; `decode` upsamples it back.
    Both operate on `(B, C, L)` tensors and are used directly as the
    `encoder_fn`/`decoder_fn` pair `glacier_reimp` expects.
    """

    def __init__(self, n_features: int = 1):
        super().__init__()
        self.enc_conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.dec_conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv1d(64, n_features, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) -> (B, 32, L // 4)."""
        x = F.relu(self.enc_conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.enc_conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        return x

    def decode(self, z: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        """(B, 32, L // 4) -> (B, C, L)."""
        x = F.relu(self.dec_conv1(z))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.dec_conv2(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.dec_conv3(x)  # linear output, matches Keras Autoencoder's last layer
        if target_length is not None and x.shape[-1] != target_length:
            # Guard against off-by-one rounding on odd lengths (bake-off pads
            # to a multiple of 4 with `conditional_pad`; we crop/pad instead
            # to avoid requiring that preprocessing step).
            if x.shape[-1] > target_length:
                x = x[..., :target_length]
            else:
                x = F.pad(x, (0, target_length - x.shape[-1]))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), target_length=x.shape[-1])


def train_glacier_autoencoder(
    X_train: np.ndarray,
    n_features: int = 1,
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    seed: int | None = 42,
    verbose: bool = False,
) -> GlacierAutoencoder:
    """Train a `GlacierAutoencoder` with plain MSE reconstruction loss.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, C, L) or (N, L)
        Training series. Reshaped to (N, C, L) internally.
    n_features : number of channels (C).
    n_epochs, batch_size, learning_rate : standard training knobs.
    device : torch device; defaults to CUDA if available, else CPU.
    seed : torch seed for reproducible initialisation/shuffling, or None.
    verbose : print per-epoch loss when True.

    Returns
    -------
    model : GlacierAutoencoder, in eval() mode, on `device`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    X = np.asarray(X_train, dtype=np.float32)
    if X.ndim == 2:
        X = X.reshape(X.shape[0], n_features, X.shape[1])
    X_t = torch.from_numpy(X).to(device)

    model = GlacierAutoencoder(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    n = X_t.shape[0]
    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch = X_t[perm[start : start + batch_size]]
            optimizer.zero_grad()
            recon = model(batch)
            loss = mse(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch.shape[0]
        epoch_loss /= n
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"GlacierAutoencoder epoch {epoch}: recon_mse={epoch_loss:.6f}")

    model.eval()
    return model


def make_autoencoder_fns(model: GlacierAutoencoder):
    """Return (encoder_fn, decoder_fn) callables for `glacier_reimp`'s
    `autoencoder=` parameter, closing over a trained `GlacierAutoencoder`.
    """

    def encoder_fn(x: torch.Tensor) -> torch.Tensor:
        return model.encode(x)

    def decoder_fn(z: torch.Tensor) -> torch.Tensor:
        # `glacier_reimp` always decodes back to the original query length;
        # infer it from what the encoder was fed isn't available here, so
        # decode at the model's natural 4x upsampling and let the caller's
        # shapes line up (FordA's L=500 is exactly divisible by 4).
        return model.decode(z)

    return encoder_fn, decoder_fn
