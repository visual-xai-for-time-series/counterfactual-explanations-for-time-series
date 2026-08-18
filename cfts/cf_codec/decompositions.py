"""
cf_codec.decompositions - pluggable series -> components transforms.

This is CoDec's "Component Decomposition" stage (see ``CoDec_presentation.pdf``,
slide 5/6): it generalizes IMFACT's EMD-only decomposition into a small,
swappable registry (workplan Phase 1). Every :class:`Decomposer` splits a 1-D
channel into an ordered, *stable* list of components that sum back to the
original signal, so components are comparable across different series of the
same dataset (required by :mod:`cfts.cf_codec.matching`).

Implemented here - one class per row of the "Choosing a Decomposition"
heuristic table (``CoDec_presentation.pdf`` slide 9, ``CoDec_workplan.md``
Phase 1), all registered in :data:`DECOMPOSERS`:

- :class:`EMDDecomposer` (``"emd"``) - the IMFACT baseline, wraps
  ``emd.sift.sift`` (same call ``cfts/cf_imfact/imfact.py``'s
  ``_decompose(..., decomposer="emd")`` path makes, so results are directly
  comparable to IMFACT). *Multi-scale, non-stationary oscillations.*
- :class:`WaveletDecomposer` (``"wavelet"``) - multi-level discrete wavelet
  transform (PyWavelets), one component per resolution level via
  multiresolution analysis. *Multi-scale, non-stationary oscillations.*
- :class:`FourierBandDecomposer` (``"fourier"``) - trend + frequency-band
  split via ``scipy.signal.stft``/``istft``. No extra dependency; also used
  as :class:`STLDecomposer`'s fallback.
- :class:`STLDecomposer` (``"stl"``) - trend/seasonal/residual split via
  statsmodels' STL when an ACF-estimated period is found; falls back to
  :class:`FourierBandDecomposer` otherwise. *Strong trend + seasonal
  structure.*
- :class:`EigenDecomposer` (``"eigen"``) - Singular Spectrum Analysis:
  trajectory-matrix SVD + diagonal averaging, applied per channel (see
  ``codec.py``'s per-channel loop) so it covers the "eigen/PCA" family
  without needing a dedicated cross-channel interface. *Correlated
  multivariate signals.*
- :class:`ShapeletDecomposer` (``"shapelet"``) - localized windows picked by
  local-variance saliency, masked to zero elsewhere so reconstruction stays
  exact. *Discriminative local shape (e.g. ECG, gesture).*
- :class:`ChangepointDecomposer` (``"changepoint"``) - piecewise-constant
  regime segmentation via ``ruptures`` (PELT/L2). *Abrupt regime shifts or
  level changes.*
- :class:`QuantileDecomposer` (``"quantile"``) - rolling-median trend +
  quantile-thresholded spike/noise split, robust to outliers. *Heavy noise,
  few clean repeating patterns.*

``WaveletDecomposer``, ``STLDecomposer`` (in the periodic branch), and
``ChangepointDecomposer`` need ``PyWavelets``, ``statsmodels``, and
``ruptures`` respectively - see ``cfts/cf_codec/requirements.txt``. Every
decomposer still satisfies the same contract: ``decompose(x)`` returns
components summing back to ``x`` (exactly, by construction, for every
strategy here - see each class's docstring for how).
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import signal


class Decomposer(ABC):
    """Splits a 1-D series into components that sum back to the original."""

    @abstractmethod
    def decompose(self, x: np.ndarray) -> np.ndarray:
        """Return components as an array of shape ``(n_components, L)`` with
        ``components.sum(axis=0)`` approximately equal to ``x``. Order must be
        stable across calls so component ``i`` is comparable across series."""
        raise NotImplementedError

    def reconstruct(self, components: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`decompose`: sum components back into a series."""
        return np.asarray(components).sum(axis=0)


class EMDDecomposer(Decomposer):
    """Empirical Mode Decomposition into Intrinsic Mode Functions (IMFs) -
    the IMFACT baseline decomposer. Wraps ``emd.sift.sift``; the last IMF it
    returns is already the residual/trend, so ``decompose`` output sums
    exactly back to the (float64-cast) input.
    """

    def __init__(self, max_imfs: int = 6):
        self.max_imfs = max_imfs

    def decompose(self, x: np.ndarray) -> np.ndarray:
        import emd as _emd

        x64 = np.asarray(x, dtype=np.float64)
        try:
            imfs = _emd.sift.sift(x64, max_imfs=self.max_imfs)  # (L, n_imfs)
            comps = imfs.T.astype(np.float32)  # (n_imfs, L)
        except Exception:
            # Degenerate signals (constant, too short, no interior extrema)
            # can make the sifter fail to find an envelope. Fall back to a
            # trivial single-component "decomposition" rather than crashing
            # the whole search - matching/perturbation still work on it.
            comps = x.reshape(1, -1).astype(np.float32)
        if comps.shape[0] < 1:
            comps = x.reshape(1, -1).astype(np.float32)
        return comps


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(x)))
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, (pad, window - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def _match_length(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) == length:
        return x
    if len(x) > length:
        return x[:length]
    return np.pad(x, (0, length - len(x)), mode="edge")


class FourierBandDecomposer(Decomposer):
    """Trend + frequency-band decomposition via ``scipy.signal.stft``.

    Splits the detrended series into ``n_bands`` contiguous frequency bands
    (each reconstructed with an inverse STFT of a hard band mask) plus one
    trend component from a moving average. Used for the "strong trend +
    seasonal structure" row of the decomposition heuristic (workplan §3,
    ``CoDec_presentation.pdf`` slide 9) in place of STL, which needs
    ``statsmodels``.
    """

    def __init__(self, n_bands: int = 3, nperseg: int | None = None, trend_window: int | None = None):
        self.n_bands = n_bands
        self.nperseg = nperseg
        self.trend_window = trend_window

    def decompose(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        L = x.shape[0]
        trend = _moving_average(x, self.trend_window or max(3, L // 10))
        residual = x - trend
        nperseg = max(4, min(self.nperseg or max(8, L // 4), L))
        _, _, Zxx = signal.stft(residual, nperseg=nperseg, boundary=None, padded=False)
        n_bins = Zxx.shape[0]
        n_bands = max(1, min(self.n_bands, n_bins))
        edges = np.linspace(0, n_bins, n_bands + 1, dtype=int)
        comps = []
        assigned = np.zeros(L)
        # A hard rectangular mask on frequency bins isn't guaranteed to
        # satisfy the NOLA (nonzero overlap-add) condition scipy checks for
        # perfect time-domain invertibility - expected here, not a bug, and
        # corrected for below by folding the reconstruction slack into the
        # trend component so components still sum exactly back to x.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="NOLA condition failed")
            for b in range(n_bands):
                mask = np.zeros(n_bins, dtype=bool)
                mask[edges[b] : edges[b + 1]] = True
                _, band = signal.istft(Zxx * mask[:, None], nperseg=nperseg, boundary=None)
                band = _match_length(band, L)
                comps.append(band)
                assigned += band
        trend = trend + (residual - assigned)
        comps.append(trend)
        return np.stack(comps, axis=0).astype(np.float32)


class WaveletDecomposer(Decomposer):
    """Multi-level discrete wavelet transform (DWT) via PyWavelets.

    Each component is one level's contribution to the multiresolution
    reconstruction: zero every level's coefficients except one and
    inverse-transform, for every level in turn (the standard "MRA" view of a
    DWT). ``mode="periodization"`` keeps every level's inverse transform
    exactly length ``L`` with no boundary padding artifacts, so components
    sum back to ``x`` to numerical precision.
    """

    def __init__(self, wavelet: str = "db4", level: int | None = None):
        self.wavelet = wavelet
        self.level = level

    def decompose(self, x: np.ndarray) -> np.ndarray:
        import pywt

        x = np.asarray(x, dtype=np.float64)
        L = x.shape[0]
        max_level = pywt.dwt_max_level(L, pywt.Wavelet(self.wavelet).dec_len)
        level = max(1, min(self.level or max_level, max_level)) if max_level > 0 else 0
        if level < 1:
            return x.reshape(1, -1).astype(np.float32)

        coeffs = pywt.wavedec(x, self.wavelet, level=level, mode="periodization")
        comps = []
        for i in range(len(coeffs)):
            zeroed = [c if j == i else np.zeros_like(c) for j, c in enumerate(coeffs)]
            recon = pywt.waverec(zeroed, self.wavelet, mode="periodization")
            comps.append(_match_length(recon, L))
        return np.stack(comps, axis=0).astype(np.float32)


class STLDecomposer(Decomposer):
    """Trend/seasonal/residual split via statsmodels' ``STL``.

    STL needs a periodicity to split on. This estimates one from the
    autocorrelation function (first lag with ACF above ``acf_threshold``,
    searched from ``min_period``); when no confident period is found (short,
    noisy, or non-periodic series) or STL itself fails, decomposition falls
    back to :class:`FourierBandDecomposer` - matching the "Choosing a
    Decomposition" slide's "STL / Fourier (STFT)" pairing for the same row.
    """

    def __init__(self, period: int | None = None, min_period: int = 4, acf_threshold: float = 0.3):
        self.period = period
        self.min_period = min_period
        self.acf_threshold = acf_threshold
        self._fallback = FourierBandDecomposer()

    def _estimate_period(self, x: np.ndarray) -> int | None:
        L = len(x)
        max_lag = min(L // 2, 200)
        if max_lag <= self.min_period:
            return None
        xc = x - x.mean()
        denom = float(np.dot(xc, xc))
        if denom < 1e-12:
            return None
        lags = range(self.min_period, max_lag)
        acf = np.array([np.dot(xc[:-lag], xc[lag:]) / denom for lag in lags])
        if acf.size == 0:
            return None
        peak = int(np.argmax(acf))
        if acf[peak] < self.acf_threshold:
            return None
        return peak + self.min_period

    def decompose(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        L = x.shape[0]
        period = self.period or self._estimate_period(x)
        if not period or period < 2 or 2 * period >= L:
            return self._fallback.decompose(x)
        try:
            from statsmodels.tsa.seasonal import STL

            res = STL(x, period=period, robust=True).fit()
            comps = np.stack([res.trend, res.seasonal, res.resid], axis=0)
        except Exception:
            return self._fallback.decompose(x)
        return comps.astype(np.float32)


class EigenDecomposer(Decomposer):
    """Singular Spectrum Analysis (SSA): embeds the series into a trajectory
    (Hankel) matrix, takes its SVD, and reconstructs each leading singular
    triple back into a 1-D "elementary component" via diagonal averaging
    (Golyandina et al., 2001) - the eigen/PCA-family decomposer from the
    heuristic table. Applied per channel (see ``codec.py``'s per-channel
    loop), so it also covers correlated multivariate series without a
    separate cross-channel interface.

    The residual (input minus the leading ``n_components`` elementary
    reconstructions) is appended as the last component, so reconstruction is
    exact regardless of how much variance the leading components capture.
    """

    def __init__(self, window: int | None = None, n_components: int = 4):
        self.window = window
        self.n_components = n_components

    @staticmethod
    def _hankelize(mat: np.ndarray, length: int) -> np.ndarray:
        """Average an elementary ``w x k`` matrix along its anti-diagonals
        back into a length-``length`` series (the standard SSA diagonal-
        averaging reconstruction step)."""
        w, k = mat.shape
        out = np.empty(length)
        for d in range(length):
            i0, i1 = max(0, d - k + 1), min(w - 1, d)
            out[d] = np.mean([mat[i, d - i] for i in range(i0, i1 + 1)])
        return out

    def decompose(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        L = x.shape[0]
        w = self.window or max(2, min(L // 4, 40))
        w = max(2, min(w, L - 1)) if L > 2 else 1
        if w < 2:
            return x.reshape(1, -1).astype(np.float32)
        k = L - w + 1
        traj = np.array([x[i : i + w] for i in range(k)]).T  # (w, k)

        U, S, Vt = np.linalg.svd(traj, full_matrices=False)
        n_comp = max(1, min(self.n_components, S.shape[0]))

        comps = []
        for i in range(n_comp):
            elem = S[i] * np.outer(U[:, i], Vt[i, :])
            comps.append(self._hankelize(elem, L))
        residual = x - np.sum(comps, axis=0)
        comps.append(residual)
        return np.stack(comps, axis=0).astype(np.float32)


class ShapeletDecomposer(Decomposer):
    """Localized "shapelet-like" components: slides a window of length
    ``shapelet_length`` across the series, scores each position by local
    variance (a saliency proxy for "discriminative local shape" - true
    class-contrastive shapelet mining needs a labeled dataset, which
    ``decompose(x)`` doesn't receive), and greedily keeps the
    ``n_shapelets`` highest-scoring non-overlapping windows.

    Each component is ``x`` masked to zero outside its window, and the
    residual covers everything else - unlike a textbook shapelet transform's
    non-additive distance features, this keeps reconstruction exact (every
    point of ``x`` belongs to exactly one component), which is the
    reconstruction approximation the workplan flags as needing to be
    documented for this decomposer.
    """

    def __init__(self, shapelet_length: int | None = None, n_shapelets: int = 3):
        self.shapelet_length = shapelet_length
        self.n_shapelets = n_shapelets

    def decompose(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        L = x.shape[0]
        w = self.shapelet_length or max(4, L // 8)
        w = max(2, min(w, L))
        if w >= L:
            return x.reshape(1, -1).astype(np.float32)

        scores = np.array([np.var(x[i : i + w]) for i in range(L - w + 1)])
        n_shapelets = max(1, min(self.n_shapelets, L // w))

        chosen: list[int] = []
        for start in np.argsort(scores)[::-1]:
            start = int(start)
            if any(start < s + w and s < start + w for s in chosen):
                continue  # overlaps an already-chosen window
            chosen.append(start)
            if len(chosen) >= n_shapelets:
                break
        chosen.sort()

        comps = []
        covered = np.zeros(L, dtype=bool)
        for start in chosen:
            comp = np.zeros(L)
            comp[start : start + w] = x[start : start + w]
            comps.append(comp)
            covered[start : start + w] = True
        comps.append(np.where(covered, 0.0, x))
        return np.stack(comps, axis=0).astype(np.float32)


class ChangepointDecomposer(Decomposer):
    """Piecewise-constant regime segmentation via ``ruptures`` (PELT, L2
    cost). Each component isolates one detected segment's values (zero
    elsewhere) - the "abrupt regime shifts / level changes" row of the
    heuristic table, where smooth bases (EMD/wavelets/Fourier) blur the
    shift points that changepoint segmentation localizes directly.
    Reconstruction is exact: every point belongs to exactly one segment.
    """

    def __init__(self, penalty: float = 3.0, max_segments: int = 6, min_size: int = 2):
        self.penalty = penalty
        self.max_segments = max_segments
        self.min_size = min_size

    def decompose(self, x: np.ndarray) -> np.ndarray:
        import ruptures as rpt

        x = np.asarray(x, dtype=np.float64)
        L = x.shape[0]
        try:
            bkps = rpt.Pelt(model="l2", min_size=self.min_size, jump=1).fit(x).predict(pen=self.penalty)
        except Exception:
            bkps = [L]

        bounds = sorted(set([0] + [b for b in bkps if 0 < b < L] + [L]))
        if len(bounds) > self.max_segments + 1:
            idx = np.linspace(0, len(bounds) - 1, self.max_segments + 1).astype(int)
            bounds = sorted(set(bounds[i] for i in idx))

        comps = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            seg = np.zeros(L)
            seg[start:end] = x[start:end]
            comps.append(seg)
        if not comps:
            comps = [x.copy()]
        return np.stack(comps, axis=0).astype(np.float32)


def _rolling_quantile(x: np.ndarray, window: int, q: float) -> np.ndarray:
    window = max(1, min(window, len(x)))
    pad = window // 2
    xp = np.pad(x, (pad, window - 1 - pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(xp, window)
    return np.quantile(windows, q, axis=-1)[: len(x)]


class QuantileDecomposer(Decomposer):
    """Robust, quantile-based split for heavy-noise series: a rolling-median
    trend (robust to outliers, unlike :class:`FourierBandDecomposer`'s
    moving-average trend), a "spike" component isolating residual points
    outside the rolling ``[low_q, high_q]`` quantile band, and a residual
    noise component for everything else. "Heavy noise, few clean repeating
    patterns" row of the heuristic table.
    """

    def __init__(self, window: int | None = None, low_q: float = 0.1, high_q: float = 0.9):
        self.window = window
        self.low_q = low_q
        self.high_q = high_q

    def decompose(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        L = x.shape[0]
        w = self.window or max(3, L // 10)

        trend = _rolling_quantile(x, w, 0.5)
        residual = x - trend
        lo = _rolling_quantile(residual, w, self.low_q)
        hi = _rolling_quantile(residual, w, self.high_q)
        is_spike = (residual < lo) | (residual > hi)

        spike = np.where(is_spike, residual, 0.0)
        noise = np.where(is_spike, 0.0, residual)
        return np.stack([trend, spike, noise], axis=0).astype(np.float32)


DECOMPOSERS: dict[str, type[Decomposer]] = {
    "emd": EMDDecomposer,
    "wavelet": WaveletDecomposer,
    "fourier": FourierBandDecomposer,
    "stl": STLDecomposer,
    "eigen": EigenDecomposer,
    "shapelet": ShapeletDecomposer,
    "changepoint": ChangepointDecomposer,
    "quantile": QuantileDecomposer,
}


def make_decomposer(name: str, **kwargs) -> Decomposer:
    """Instantiate a registered :class:`Decomposer` by name (see :data:`DECOMPOSERS`)."""
    if name not in DECOMPOSERS:
        raise ValueError(f"Unknown decomposition '{name}'. Available: {sorted(DECOMPOSERS)}")
    return DECOMPOSERS[name](**kwargs)
