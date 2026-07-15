"""
Self-contained implementation of the fast_borf BoRF pipeline.

Vendored from https://github.com/DawidPludowski/borf (MIT licence).
Flattened into a single file to remove the external fast_borf dependency.

Paper: Płudowski, D., Spinnato, F., Wilczyński, P., Kotowski, K., Ntagiou, E. V.,
       Guidotti, R., & Biecek, P. (2025). MASCOTS: Model-Agnostic Symbolic
       COunterfactual explanations for Time Series. arXiv:2503.22389.

Covers:
  - SAX transform + heuristic configs
  - BorfSaxSingleTransformer, BorfPipelineBuilder (sklearn FeatureUnion pipeline)
  - ReshapeTo2D, ZeroColumnsRemover, ToScipySparse (pipeline helpers)
  - BagOfReceptiveFields + ReceptiveField (XAI alignment)
"""

from __future__ import annotations

import math
import itertools
from typing import Any, Literal, Optional, Sequence, Tuple

import numba as nb
import numpy as np
import sparse
from numpy.typing import NDArray
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, make_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (from fast_borf/constants.py)
# ─────────────────────────────────────────────────────────────────────────────

FASTMATH = True

HASHMAP_2_SYMBOLS = np.array([
    [ 0,  4,  3,  2,  1,  5,  6,  7,  8],
    [ 1,  5,  7,  8, -1, -1, -1, -1, -1],
    [ 2,  6,  8, -1, -1, -1, -1, -1, -1],
    [ 3,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 4, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 5, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 6, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 7, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 8, -1, -1, -1, -1, -1, -1, -1, -1],
])

_NORM_BINS_LISTS = [
    [0.0],
    [-0.43072735, 0.43072735],
    [-0.67448977, 0.0, 0.67448977],
    [-0.84162118, -0.25334711, 0.25334711, 0.84162118],
    [-0.9674215, -0.43072735, 0.0, 0.43072735, 0.9674215],
    [-1.06757049, -0.56594888, -0.18001237, 0.18001237, 0.56594888, 1.06757049],
    [-1.1503494, -0.67448977, -0.31863939, 0.0, 0.31863939, 0.67448977, 1.1503494],
    [-1.22064043, -0.76470966, -0.43072735, -0.1397103, 0.1397103, 0.43072735, 0.76470966, 1.22064043],
]


# ─────────────────────────────────────────────────────────────────────────────
# Numba math utilities  (from fast_borf/utils.py)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(fastmath=True, cache=True)
def _erfinv(x: float) -> float:
    w = -math.log((1 - x) * (1 + x))
    if w < 5:
        w = w - 2.5
        p = 2.81022636e-08
        p = 3.43273939e-07 + p * w
        p = -3.5233877e-06 + p * w
        p = -4.39150654e-06 + p * w
        p = 0.00021858087 + p * w
        p = -0.00125372503 + p * w
        p = -0.00417768164 + p * w
        p = 0.246640727 + p * w
        p = 1.50140941 + p * w
    else:
        w = math.sqrt(w) - 3
        p = -0.000200214257
        p = 0.000100950558 + p * w
        p = 0.00134934322 + p * w
        p = -0.00367342844 + p * w
        p = 0.00573950773 + p * w
        p = -0.0076224613 + p * w
        p = 0.00943887047 + p * w
        p = 1.00167406 + p * w
        p = 2.83297682 + p * w
    return p * x


@nb.vectorize(['float64(float64, float64, float64)'], cache=True)
def _ppf(x, mu, std):
    return mu + math.sqrt(2) * _erfinv(2 * x - 1) * std


@nb.njit(cache=True)
def get_norm_bins(alphabet_size: int) -> NDArray[np.float64]:
    return _ppf(np.linspace(0.0, 1.0, alphabet_size + 1)[1:-1], 0.0, 1.0)


@nb.njit(fastmath=True, cache=True)
def are_window_size_and_dilation_compatible_with_signal_length(
    window_size: int, dilation: int, signal_length: int
) -> bool:
    return window_size + (window_size - 1) * (dilation - 1) <= signal_length


@nb.njit(fastmath=True, cache=True)
def get_n_windows(
    sequence_size: int, window_size: int, dilation: int = 1, stride: int = 1, padding: int = 0
) -> int:
    return 1 + math.floor(
        (sequence_size + 2 * padding - window_size - (dilation - 1) * (window_size - 1)) / stride
    )


@nb.njit(fastmath=True, cache=True)
def convert_to_base_10(number: int, base: int) -> int:
    result = 0
    multiplier = 1
    while number > 0:
        digit = number % 10
        result += digit * multiplier
        multiplier *= base
        number //= 10
    return result


@nb.njit(cache=True)
def is_empty(a: np.ndarray) -> bool:
    return a.size == 0


@nb.njit(cache=True)
def is_valid_windowing(sequence_size: int, window_size: int, dilation: int) -> bool:
    return sequence_size >= window_size * dilation


# ─────────────────────────────────────────────────────────────────────────────
# Z-score  (from fast_borf/zscore.py)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(fastmath=FASTMATH, cache=True)
def _zscore_threshold(
    a: float, mu: float, sigma: float, sigma_global: float, sigma_threshold: float
) -> float:
    if sigma_global == 0:
        return 0.0
    if sigma / sigma_global < sigma_threshold:
        return 0.0
    if sigma == 0:
        return 0.0
    return (a - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# Moving statistics  (from fast_borf/moving.py)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(fastmath=FASTMATH, cache=True)
def _move_mean(a: np.ndarray, window_width: int) -> np.ndarray:
    out = np.empty_like(a)
    asum = 0.0
    for i in range(window_width):
        asum += a[i]
        out[i] = asum / (i + 1)
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / window_width
    return out


@nb.njit(fastmath=FASTMATH, cache=True)
def _move_std(a: np.ndarray, window_width: int, ddof: int = 0) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float64)
    mean = 0.0
    M2 = 0.0
    for i in range(window_width):
        delta = a[i] - mean
        mean += delta / (i + 1)
        M2 += delta * (a[i] - mean)
    variance = M2 / (window_width - ddof) if (window_width - ddof) > 0 else 0.0
    out[window_width - 1] = math.sqrt(max(variance, 0.0))
    for i in range(window_width, len(a)):
        x0 = a[i - window_width]
        xn = a[i]
        new_avg = mean + (xn - x0) / window_width
        new_var = (
            variance + (xn - new_avg + x0 - mean) * (xn - x0) / (window_width - ddof)
            if (window_width - ddof) > 0
            else 0.0
        )
        out[i] = math.sqrt(max(new_var, 0.0))
        mean = new_avg
        variance = new_var
    for i in range(window_width - 1):
        out[i] = np.nan
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SAX  (from fast_borf/symbolic_aggregate_approximation/symbolic_aggregate_approximation_clean.py)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def sax(
    a: np.ndarray,
    window_size: int,
    word_length: int,
    bins: np.ndarray,
    stride: int = 1,
    dilation: int = 1,
    min_window_to_signal_std_ratio: float = 0.0,
) -> np.ndarray:
    """Sliding-window SAX transform. Returns shape (n_windows, word_length) uint8."""
    n_windows = get_n_windows(a.size, window_size, dilation, stride)
    n_windows_moving = get_n_windows(a.size, window_size, dilation)
    global_std = np.std(a)
    if global_std == 0:
        return np.zeros((n_windows, word_length), dtype=np.uint8)
    seg_size = window_size // word_length
    n_segments = get_n_windows(a.size, seg_size, dilation)
    segment_means = np.full(n_segments, np.nan)
    window_means = np.full(n_windows_moving, np.nan)
    window_stds = np.full(n_windows_moving, np.nan)
    for d in range(dilation):
        window_means[d::dilation] = _move_mean(a[d::dilation], window_size)[window_size - 1:]
        window_stds[d::dilation] = _move_std(a[d::dilation], window_size)[window_size - 1:]
        segment_means[d::dilation] = _move_mean(a[d::dilation], seg_size)[seg_size - 1:]
    out = np.zeros((n_windows, word_length))
    for i in range(n_windows):
        for j in range(word_length):
            out[i, j] = _zscore_threshold(
                segment_means[(i * stride) + (j * seg_size * dilation)],
                window_means[i * stride],
                window_stds[i * stride],
                global_std,
                min_window_to_signal_std_ratio,
            )
    return np.digitize(out, bins).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Hash-based unique  (from fast_borf/hash_unique.py)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _hash_fn(v: int) -> np.uint64:
    byte_mask = np.uint64(255)
    bs = np.uint64(v)
    FNV_primer = np.uint64(1099511628211)
    FNV_bias = np.uint64(14695981039346656037)
    h = FNV_bias
    for shift in (0, 8, 16, 24, 32, 40, 48, 56):
        h = h * FNV_primer
        h = h ^ ((bs >> np.uint64(shift)) & byte_mask)
    return h


@nb.njit(cache=True)
def unique(ar: np.ndarray):
    """Return (unique_values, counts) using open-address hash table."""
    l_raw = int(math.ceil(math.log2(max(len(ar), 2))))
    l = 2 << l_raw
    mask = l - 1
    uniques = np.empty(l, dtype=ar.dtype)
    uniques_cnt = np.zeros(l, dtype=np.int64)
    total = 0
    for v in ar:
        h = _hash_fn(v)
        index = int(h) & mask
        while True:
            if uniques_cnt[index] == 0:
                uniques_cnt[index] += 1
                uniques[index] = v
                total += 1
                break
            elif uniques[index] == v:
                uniques_cnt[index] += 1
                break
            else:
                index = (index + 1) & mask
    out_vals = np.empty(total, dtype=ar.dtype)
    out_cnts = np.empty(total, dtype=np.int64)
    t = 0
    for i in range(l):
        if uniques_cnt[i] > 0:
            out_vals[t] = uniques[i]
            out_cnts[t] = uniques_cnt[i]
            t += 1
    return out_vals, out_cnts


# ─────────────────────────────────────────────────────────────────────────────
# Integer encoding helpers  (from fast_borf/bag_of_patterns/utils.py)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def array_to_int(arr: np.ndarray) -> int:
    """Encode a symbol array as a decimal integer: [1,2,3] → 123."""
    result = 0
    for i in range(len(arr)):
        result = result * 10 + arr[i]
    return result


@nb.njit(cache=True)
def ndindex_2d_array(idx: int, dim2_shape: int):
    return idx // dim2_shape, idx % dim2_shape


# ─────────────────────────────────────────────────────────────────────────────
# XAI integer helpers  (from fast_borf/xai/utils.py)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(fastmath=True, cache=True)
def int_to_array_new_base(number: int, base: int, word_length: int) -> np.ndarray:
    """Decode integer back to symbol array in the given base."""
    array = np.zeros(word_length, dtype=np.int32)
    for i in range(word_length):
        power = word_length - i - 1
        array[i] = number // (base ** power)
        number %= base ** power
    return array


@nb.njit(fastmath=True, cache=True)
def _array_to_int_new_base(array: np.ndarray, base: int) -> int:
    """Encode a symbol array to an integer in the given base."""
    word_length = array.shape[0]
    result = 0
    for i in range(word_length):
        result += int(array[i]) * (base ** (word_length - i - 1))
    return result


def _words_to_int_np(sax_words: np.ndarray, base: int) -> np.ndarray:
    """Vectorised: (n_windows, word_length) uint8 → (n_windows,) int64."""
    n, wl = sax_words.shape
    pows = np.power(base, np.arange(wl - 1, -1, -1), dtype=np.int64)
    return (sax_words.astype(np.int64) @ pows)


# ─────────────────────────────────────────────────────────────────────────────
# BoP core transform  (from fast_borf/bag_of_patterns/borf_new_sax.py)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _bop_transform_single(
    a: np.ndarray,
    window_size: int,
    word_length: int,
    alphabet_size: int,
    bins: np.ndarray,
    dilation: int,
    stride: int = 1,
    min_window_to_signal_std_ratio: float = 0.0,
):
    """SAX-encode a signal and return (unique_word_ints, counts)."""
    sax_words = sax(a, window_size, word_length, bins, stride, dilation, min_window_to_signal_std_ratio)
    word_ints = np.empty(len(sax_words), dtype=np.int64)
    for i in range(len(sax_words)):
        word_ints[i] = convert_to_base_10(array_to_int(sax_words[i]), alphabet_size)
    return unique(word_ints)


@nb.njit(cache=True)
def _bop_transform_single_conf(
    a: np.ndarray,
    ts_idx: int,
    signal_idx: int,
    window_size: int,
    word_length: int,
    alphabet_size: int,
    bins: np.ndarray,
    dilation: int,
    stride: int = 1,
    min_window_to_signal_std_ratio: float = 0.0,
) -> np.ndarray:
    """Return (ts_idx, signal_idx, word_int, count) rows."""
    words, counts = _bop_transform_single(
        a, window_size, word_length, alphabet_size, bins, dilation, stride,
        min_window_to_signal_std_ratio,
    )
    ts_idxs = np.full(len(words), ts_idx, dtype=np.int64)
    sig_idxs = np.full(len(words), signal_idx, dtype=np.int64)
    return np.column_stack((ts_idxs, sig_idxs, words, counts))


@nb.njit(parallel=True, nogil=True, cache=True)
def transform_sax_patterns(
    panel: np.ndarray,
    window_size: int,
    word_length: int,
    alphabet_size: int,
    stride: int,
    dilation: int,
    min_window_to_signal_std_ratio: float = 0.0,
) -> np.ndarray:
    """
    Transform a panel (N, C, L) → COO triplets (ts_idx, signal_idx, word_int, count).
    """
    bins = get_norm_bins(alphabet_size)
    n_signals = len(panel[0])
    n_ts = len(panel)
    iterations = n_ts * n_signals
    counts = np.zeros(iterations + 1, dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal = np.asarray(panel[ts_idx][signal_idx]).astype(np.float64)
        # strip NaNs
        valid = signal[~np.isnan(signal)]
        if not are_window_size_and_dilation_compatible_with_signal_length(window_size, dilation, valid.size):
            continue
        counts[i + 1] = len(
            _bop_transform_single_conf(
                valid, ts_idx, signal_idx, window_size, word_length, alphabet_size,
                bins, dilation, stride, min_window_to_signal_std_ratio,
            )
        )
    cum = np.cumsum(counts)
    out = np.empty((int(cum[-1]), 4), dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal = np.asarray(panel[ts_idx][signal_idx]).astype(np.float64)
        valid = signal[~np.isnan(signal)]
        if not are_window_size_and_dilation_compatible_with_signal_length(window_size, dilation, valid.size):
            continue
        rows = _bop_transform_single_conf(
            valid, ts_idx, signal_idx, window_size, word_length, alphabet_size,
            bins, dilation, stride, min_window_to_signal_std_ratio,
        )
        out[cum[i]: cum[i + 1]] = rows
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic config generation  (from fast_borf/heuristic.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_window_sizes(
    m_min: int,
    m_max: int,
    min_window_size: int = 4,
    max_window_size: Optional[int] = None,
    power: float = 2.0,
) -> np.ndarray:
    if max_window_size is None:
        max_window_size = m_max
    m, windows, windows_min = 2, [], []
    while m <= max_window_size:
        (windows_min if m < min_window_size else windows).append(m)
        m = int(m * power)
    windows = np.array(windows, dtype=int)
    windows_min = np.array(windows_min[1:], dtype=int)
    if not is_empty(windows_min) and m_min <= windows_min.max() * power:
        windows = np.concatenate([windows_min, windows])
    return windows


def _get_word_lengths(n_word_lengths: int = 4, start: int = 0) -> np.ndarray:
    return np.array([2 ** i for i in range(start, n_word_lengths + start)])


def _get_dilations(
    max_length: int, min_dilation: int = 1, max_dilation: Optional[int] = None
) -> np.ndarray:
    if max_dilation is None:
        max_dilation = np.log2(max_length)
    dilations, s = [], min_dilation
    while s <= max_dilation:
        dilations.append(s)
        s *= 2
    return np.array(dilations)


def heuristic_function_sax(
    time_series_min_length: int,
    time_series_max_length: int,
    window_size_min_window_size: int = 4,
    window_size_max_window_size: Optional[int] = None,
    word_lengths_n_word_lengths: int = 4,
    alphabets_min_symbols: int = 2,
    alphabets_max_symbols: int = 3,
    alphabets_step: int = 1,
    dilations_min_dilation: int = 1,
    dilations_max_dilation: Optional[int] = None,
    complexity: Literal["quadratic", "linear"] = "quadratic",
) -> list[dict]:
    """Generate SAX configuration grid."""
    window_sizes = _get_window_sizes(
        time_series_min_length, time_series_max_length,
        window_size_min_window_size, window_size_max_window_size,
    ).tolist()
    word_lengths = _get_word_lengths(word_lengths_n_word_lengths).tolist()
    dilations = _get_dilations(
        time_series_max_length, dilations_min_dilation, dilations_max_dilation
    ).tolist()
    alphabet_sizes = np.arange(alphabets_min_symbols, alphabets_max_symbols, alphabets_step).tolist()

    configs = []
    for ws, wl, d, alph in itertools.product(window_sizes, word_lengths, dilations, alphabet_sizes):
        if wl > ws:
            continue
        if not is_valid_windowing(time_series_max_length, ws, int(d)):
            continue
        stride = wl if complexity == "linear" else 1
        configs.append(
            dict(window_size=int(ws), word_length=int(wl), dilation=int(d),
                 stride=int(stride), alphabet_size=int(alph))
        )
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# sklearn pipeline helpers  (from fast_borf/pipeline/)
# ─────────────────────────────────────────────────────────────────────────────

class ReshapeTo2D(BaseEstimator, TransformerMixin):
    """Flatten (N, C, W) sparse COO → (N, C*W) sparse COO. Stores unraveled index."""

    def __init__(self, keep_unraveled_index: bool = False):
        self.keep_unraveled_index = keep_unraveled_index
        self.unraveled_index_: Optional[np.ndarray] = None
        self.original_shape_: Optional[tuple] = None

    def fit(self, X, y=None):
        self.original_shape_ = X.shape
        if self.keep_unraveled_index:
            self.unraveled_index_ = np.hstack(
                [np.unravel_index(np.arange(np.prod(X.shape[1:])), X.shape[1:])]
            ).T
        return self

    def transform(self, X):
        return X.reshape((X.shape[0], -1))


class ZeroColumnsRemover(BaseEstimator, TransformerMixin):
    """Remove columns that are all-zero across the training set."""

    def __init__(self, axis: int = 0, map_features: bool = False):
        self.axis = axis
        self.map_features = map_features
        self.n_original_columns_: Optional[int] = None
        self.columns_to_keep_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        self.n_original_columns_ = X.shape[1]
        self.columns_to_keep_ = np.argwhere(X.any(axis=self.axis)).ravel()
        return self

    def transform(self, X):
        return X[..., self.columns_to_keep_]


class ToScipySparse(BaseEstimator, TransformerMixin):
    """Convert sparse.COO → scipy csr_matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.to_scipy_sparse()

    def inverse_transform(self, X):
        return sparse.COO.from_scipy_sparse(X)


# ─────────────────────────────────────────────────────────────────────────────
# BorfSaxSingleTransformer  (from fast_borf/classes/bag_of_receptive_fields_sax/borf_single.py)
# ─────────────────────────────────────────────────────────────────────────────

class BorfSaxSingleTransformer(BaseEstimator, TransformerMixin):
    """Single-config SAX BoRF transformer. Output: sparse.COO (N, C, n_words)."""

    def __init__(
        self,
        window_size: int = 4,
        dilation: int = 1,
        alphabet_size: int = 3,
        word_length: int = 2,
        stride: int = 1,
        min_window_to_signal_std_ratio: float = 0.0,
        n_jobs: int = 1,
        prefix: str = "",
        **kwargs,  # absorb unused heuristic keys
    ):
        self.window_size = window_size
        self.dilation = dilation
        self.word_length = word_length
        self.stride = stride
        self.alphabet_size = alphabet_size
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.n_jobs = n_jobs
        self.prefix = prefix
        self.n_words = alphabet_size ** word_length
        if n_jobs != 1:
            try:
                from numba import set_num_threads
                import psutil
                n = psutil.cpu_count(logical=False) if n_jobs == -1 else n_jobs
                set_num_threads(max(1, n))
            except Exception:
                pass

    def get_params(self, deep: bool = True) -> dict:
        return dict(
            window_size=self.window_size,
            dilation=self.dilation,
            alphabet_size=self.alphabet_size,
            word_length=self.word_length,
            stride=self.stride,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> sparse.COO:
        X_arr = np.asarray(X, dtype=np.float64)
        shape = (len(X_arr), len(X_arr[0]), self.n_words)
        out = transform_sax_patterns(
            panel=X_arr,
            window_size=self.window_size,
            dilation=self.dilation,
            alphabet_size=self.alphabet_size,
            word_length=self.word_length,
            stride=self.stride,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
        )
        if len(out) == 0:
            return sparse.COO(coords=np.empty((3, 0), dtype=np.int64),
                              data=np.empty(0, dtype=np.int64), shape=shape)
        return sparse.COO(coords=out[:, :3].T, data=out[:, 3], shape=shape)


# ─────────────────────────────────────────────────────────────────────────────
# BorfPipelineBuilder  (adapted from fast_borf/classes/bag_of_receptive_fields_sax/borf_multi.py)
# Note: accepts `configs` directly to avoid the `awkward` dependency.
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipeline(
    configs: list[dict],
    min_window_to_signal_std_ratio: float = 0.0,
    n_jobs_numba: int = 1,
    n_jobs: int = 1,
    transformer_weights=None,
    pipeline_objects: Optional[Sequence[Tuple]] = None,
) -> FeatureUnion:
    """Build a sklearn FeatureUnion from a list of SAX configs."""
    if pipeline_objects is None:
        pipeline_objects = []
    transformers = []
    for config in configs:
        borf = BorfSaxSingleTransformer(
            **config,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
            n_jobs=n_jobs_numba,
        )
        pipe = make_pipeline(borf, *[obj(**kwargs) for obj, kwargs in pipeline_objects])
        transformers.append(pipe)
    return FeatureUnion(
        transformer_list=[(str(i), transformers[i]) for i in range(len(transformers))],
        n_jobs=n_jobs,
        transformer_weights=transformer_weights,
    )


class BorfPipelineBuilder:
    """
    Build a BoRF FeatureUnion from pre-computed configs.

    Unlike the original implementation, this accepts `configs` directly in
    ``__init__`` (no ``awkward`` dependency for length auto-detection).

    Usage::

        pipe = BorfPipelineBuilder(configs=[...], pipeline_objects=[...]).build(X)
        X_transformed = pipe.fit_transform(X)
    """

    def __init__(
        self,
        configs: list[dict],
        min_window_to_signal_std_ratio: float = 0.0,
        n_jobs: int = 1,
        n_jobs_numba: int = 1,
        transformer_weights=None,
        pipeline_objects: Optional[Sequence[Tuple]] = None,
        complexity: Literal["quadratic", "linear"] = "quadratic",
        **kwargs,
    ):
        self.configs = configs
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.n_jobs = n_jobs
        self.n_jobs_numba = n_jobs_numba
        self.transformer_weights = transformer_weights
        self.pipeline_objects = pipeline_objects
        self.complexity = complexity

    def build(self, X=None) -> FeatureUnion:
        """Return an unfitted FeatureUnion. X is accepted but not needed."""
        return _build_pipeline(
            configs=self.configs,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
            n_jobs_numba=self.n_jobs_numba,
            n_jobs=self.n_jobs,
            transformer_weights=self.transformer_weights,
            pipeline_objects=self.pipeline_objects,
        )


# ─────────────────────────────────────────────────────────────────────────────
# XAI – ReceptiveField  (from fast_borf/xai/receptive_field.py)
# ─────────────────────────────────────────────────────────────────────────────

class ReceptiveField:
    """Metadata and alignment info for one BoRF feature (a SAX word in a config)."""

    def __init__(
        self,
        compressed_word_int: int,
        signal_idx: int,
        word_length: int,
        window_size: int,
        dilation: int,
        stride: int,
        alphabet_size: int,
        min_window_to_signal_std_ratio: float,
        conf_idx: Optional[int] = None,
        feature_idx: Optional[int] = None,
        feature_values=None,
        alignments=None,
        mappings=None,
        feature_importance=None,
        class_labels=None,
        signal_labels=None,
        **kwargs,
    ):
        self.compressed_word_int = compressed_word_int
        self.signal_idx = signal_idx
        self.word_length = word_length
        self.window_size = window_size
        self.dilation = dilation
        self.stride = stride
        self.alphabet_size = alphabet_size
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.conf_idx = conf_idx
        self.feature_idx = feature_idx
        self.feature_values = feature_values
        self.feature_importance = feature_importance
        self.feature_importance_norm = None
        self.alignments = alignments
        self.mappings = mappings
        self.class_labels = class_labels
        self.signal_labels = signal_labels

        self.plot_idx = np.arange(self.window_size * self.dilation, step=self.dilation)
        self.word_array = int_to_array_new_base(
            self.compressed_word_int, self.alphabet_size, self.word_length
        )


# ─────────────────────────────────────────────────────────────────────────────
# XAI – SAX alignment mapping  (from fast_borf/xai/sax_mapping.py)
# ─────────────────────────────────────────────────────────────────────────────

def _wsax_matrix_row_position_to_indices(
    i: int, dilation: int, stride: int, word_length: int, segment_size: int
) -> np.ndarray:
    """Compute time-step indices for window at position i. Returns (word_length, segment_size)."""
    out = np.empty((word_length, segment_size), dtype=np.int64)
    for j in range(word_length):
        for k in range(segment_size):
            out[j, k] = (i * stride) + (j * dilation * segment_size) + (k * dilation)
    return out


def _wsax_signal_alignment_conversion(
    a: np.ndarray,
    window_size: int,
    word_length: int,
    alphabet_size: int,
    bins: np.ndarray,
    dilation: int = 1,
    stride: int = 1,
    min_window_to_signal_std_ratio: float = 0.0,
) -> dict:
    """
    Map each SAX word in the signal to its time-step alignment.
    Returns dict: word_int → np.ndarray (n_occurrences, word_length, segment_size).
    """
    sax_words = sax(a, window_size, word_length, bins, stride, dilation, min_window_to_signal_std_ratio)
    sax_words_np = np.asarray(sax_words, dtype=np.int64)  # (n_windows, word_length)
    word_ints = _words_to_int_np(sax_words_np, alphabet_size)  # (n_windows,)
    segment_size = window_size // word_length
    conversion: dict[int, np.ndarray] = {}
    for i, word_int in enumerate(word_ints):
        key = int(word_int)
        positions = _wsax_matrix_row_position_to_indices(
            i, dilation, stride, word_length, segment_size
        )[np.newaxis, :, :]  # (1, word_length, segment_size)
        if key not in conversion:
            conversion[key] = positions
        else:
            conversion[key] = np.vstack([conversion[key], positions])
    return conversion


def _wsax_panel_alignment_conversion(
    panel: np.ndarray,
    window_size: int,
    word_length: int,
    alphabet_size: int,
    dilation: int = 1,
    stride: int = 1,
    min_window_to_signal_std_ratio: float = 0.0,
    **kwargs,
) -> list:
    """
    For each time series and each signal, build the alignment dict.
    Returns panel_conversion[ts_idx][signal_idx] = dict.
    """
    bins = get_norm_bins(alphabet_size)
    panel_conversion = []
    for j in range(len(panel)):  # ts_idx
        sax_conversion = []
        for i in range(len(panel[j])):  # signal_idx
            signal = np.asarray(panel[j][i], dtype=np.float64)
            signal = signal[~np.isnan(signal)]
            if not are_window_size_and_dilation_compatible_with_signal_length(
                window_size, dilation, signal.size
            ):
                sax_conversion.append({})
                continue
            sax_conversion.append(
                _wsax_signal_alignment_conversion(
                    signal, window_size, word_length, alphabet_size, bins,
                    dilation, stride, min_window_to_signal_std_ratio,
                )
            )
        panel_conversion.append(sax_conversion)
    return panel_conversion


def wsax_configurations_alignment_conversion(
    panel: np.ndarray, configurations: list[dict]
) -> list:
    """
    Build alignment dicts for all configs.
    Returns configurations_conversion[conf_idx][ts_idx][signal_idx] = dict.
    """
    return [_wsax_panel_alignment_conversion(panel=panel, **cfg) for cfg in configurations]


# ─────────────────────────────────────────────────────────────────────────────
# XAI – Pipeline mapping  (from fast_borf/xai/pipeline_mapping.py)
# ─────────────────────────────────────────────────────────────────────────────

def _map_single_conf_features_to_words(
    unraveled_index: np.ndarray, columns_kept: np.ndarray
) -> np.ndarray:
    return unraveled_index[columns_kept]


def map_borf_to_conf(
    borf: FeatureUnion,
    reshaper_position: int = 1,
    zero_columns_remover_position: int = 2,
) -> np.ndarray:
    """
    Map post-compression feature indices → (conf_idx, signal_idx, word_idx).
    Returns shape (n_features, 3).
    """
    mapping = np.empty((0, 3), dtype=np.int64)
    for conf_idx, (_, pipeline) in enumerate(borf.transformer_list):
        reshaper: ReshapeTo2D = pipeline[reshaper_position]
        remover: ZeroColumnsRemover = pipeline[zero_columns_remover_position]
        words_map = _map_single_conf_features_to_words(
            reshaper.unraveled_index_, remover.columns_to_keep_
        )  # (n_kept, 2): (signal_idx, word_idx)
        conf_col = np.full((len(words_map), 1), conf_idx, dtype=np.int64)
        mapping = np.vstack([mapping, np.hstack([conf_col, words_map])])
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# XAI – BagOfReceptiveFields  (from fast_borf/xai/mapping.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_receptive_fields(
    X: np.ndarray,
    X_transformed,
    features: np.ndarray,
    configs: list[dict],
    mapping: np.ndarray,
) -> tuple[dict, list]:
    """
    Build a ReceptiveField per feature, computing time-step alignments for X.

    Parameters
    ----------
    X : (N, C, L) float array
    X_transformed : scipy sparse (N, n_features), already fit-transformed
    features : 1-D array of feature indices to process
    configs : list of SAX config dicts
    mapping : (n_all_features, 3) array of (conf_idx, signal_idx, word_idx)

    Returns
    -------
    receptive_fields : dict  feature_idx → ReceptiveField
    sax_converted_X  : nested list  [conf_idx][ts_idx][signal_idx] = dict
    """
    sax_converted_X = wsax_configurations_alignment_conversion(X, configs)
    receptive_fields: dict[int, ReceptiveField] = {}

    for feature in features:
        conf_idx, signal_idx, word_idx = int(mapping[feature, 0]), int(mapping[feature, 1]), int(mapping[feature, 2])
        config = configs[conf_idx]
        word_length = config["word_length"]
        window_size = config["window_size"]
        segment_size = window_size // word_length

        alignments_per_ts = []
        mappings_per_ts = []
        for i in range(len(X)):
            sig_dict = sax_converted_X[conf_idx][i][signal_idx]
            if word_idx in sig_dict:
                align = sig_dict[word_idx]  # (n_occ, word_length, segment_size)
                alignments_per_ts.append(align)
                mappings_per_ts.append(np.array(X[i, signal_idx])[align] if align.size > 0 else
                                       np.empty((0, word_length, segment_size)))
            else:
                alignments_per_ts.append(np.empty((0, word_length, segment_size), dtype=np.int64))
                mappings_per_ts.append(np.empty((0, word_length, segment_size)))

        if issparse(X_transformed):
            feat_values = np.asarray(X_transformed[:, [feature]].todense()).ravel()
        else:
            feat_values = np.asarray(X_transformed[:, feature]).ravel()

        receptive_fields[feature] = ReceptiveField(
            compressed_word_int=word_idx,
            signal_idx=signal_idx,
            conf_idx=conf_idx,
            feature_idx=feature,
            feature_values=feat_values,
            alignments=alignments_per_ts,
            mappings=mappings_per_ts,
            **config,
        )

    return receptive_fields, sax_converted_X


class BagOfReceptiveFields:
    """
    Map BoRF features back to time-series receptive fields for XAI.

    Parameters
    ----------
    borf : fitted sklearn FeatureUnion from BorfPipelineBuilder
    borf_position : index of BorfSaxSingleTransformer in each sub-pipeline (default 0)
    reshaper_position : index of ReshapeTo2D in each sub-pipeline (default 1)
    zero_columns_remover_position : index of ZeroColumnsRemover (default 2)
    """

    def __init__(
        self,
        borf: FeatureUnion,
        borf_position: int = 0,
        reshaper_position: int = 1,
        zero_columns_remover_position: int = 2,
    ):
        self.borf = borf
        self.borf_position = borf_position
        self.reshaper_position = reshaper_position
        self.zero_columns_remover_position = zero_columns_remover_position

        self.mapping = map_borf_to_conf(borf, reshaper_position, zero_columns_remover_position)
        self.configs = [
            pipeline[borf_position].get_params()
            for _, pipeline in borf.transformer_list
        ]

        self.X_: Optional[np.ndarray] = None
        self.y_true_ = None
        self.y_pred_ = None
        self.X_transformed_ = None
        self.receptive_fields_: Optional[dict] = None
        self.X_sax_: Optional[list] = None
        self.task_: Optional[str] = None

    def build(self, X: np.ndarray, y_true=None, y_pred=None, task: str = "classification"):
        """
        Build receptive fields for data X.

        Parameters
        ----------
        X : (N, C, L) array — may be a single sample (N=1)
        y_true, y_pred : optional labels (stored but not required for alignment)
        task : "classification" or "regression"
        """
        self.task_ = task
        self.X_ = np.asarray(X)
        self.y_true_ = y_true
        self.y_pred_ = y_pred
        self.X_transformed_ = self.borf.transform(self.X_)
        features = np.arange(len(self.mapping))
        self.receptive_fields_, self.X_sax_ = _build_receptive_fields(
            self.X_, self.X_transformed_, features, self.configs, self.mapping
        )
        return self
