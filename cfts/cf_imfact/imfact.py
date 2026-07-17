from __future__ import annotations

from typing import Tuple

import numpy as np

import torch

from scipy import signal
from scipy.interpolate import splrep, splev
from scipy.signal import argrelextrema
from scipy.spatial.distance import jensenshannon

from cfts.cf__abstract.abstract import (
    detach_to_numpy,
    numpy_to_torch,
    ensure_cl,
    ensure_ncl,
    revert_orientation,
    subsample_dataset,
)


# ---------------------------------------------------------------------------
# IMFACT helpers
# ---------------------------------------------------------------------------

def _rilling_pad(indmin: np.ndarray, indmax: np.ndarray, X: np.ndarray, pad_width: int = 2):
    """Reflect boundary extrema using the Rilling (2003) method to reduce end effects.

    Mirrors the logic of emd._sift_core._pad_extrema_rilling so that _sift_imfs
    produces envelopes consistent with emd.sift.sift.

    Returns (tmin, xmin, tmax, xmax) with padded extrema locations and values.
    """
    t = np.arange(len(X))

    # --- LEFT boundary ---
    if indmax[0] < indmin[0]:
        if X[0] > X[indmin[0]]:
            lmax = np.flipud(indmax[1:pad_width + 1])
            lmin = np.flipud(indmin[:pad_width])
            lsym = indmax[0]
        else:
            lmax = np.flipud(indmax[:pad_width])
            lmin = np.r_[np.flipud(indmin[:pad_width - 1]), 0]
            lsym = 0
    else:
        if X[0] > X[indmax[0]]:
            lmax = np.flipud(indmax[:pad_width])
            lmin = np.flipud(indmin[1:pad_width + 1])
            lsym = indmin[0]
        else:
            lmin = np.flipud(indmin[:pad_width])
            lmax = np.r_[np.flipud(indmax[:pad_width - 1]), 0]
            lsym = 0

    tlmin = 2 * lsym - lmin
    tlmax = 2 * lsym - lmax

    if tlmin[0] >= t[0] or tlmax[0] >= t[0]:
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[:pad_width])
        else:
            lmin = np.flipud(indmin[:pad_width])
        lsym = 0
        tlmin = 2 * lsym - lmin
        tlmax = 2 * lsym - lmax

    # --- RIGHT boundary ---
    if indmax[-1] < indmin[-1]:
        if X[-1] < X[indmax[-1]]:
            rmax = np.flipud(indmax[-pad_width:])
            rmin = np.flipud(indmin[-pad_width - 1:-1])
            rsym = indmin[-1]
        else:
            rmax = np.r_[X.shape[0] - 1, np.flipud(indmax[-(pad_width - 2):])]
            rmin = np.flipud(indmin[-(pad_width - 1):])
            rsym = X.shape[0] - 1
    else:
        if X[-1] > X[indmin[-1]]:
            rmax = np.flipud(indmax[-pad_width - 1:-1])
            rmin = np.flipud(indmin[-pad_width:])
            rsym = indmax[-1]
        else:
            rmax = np.flipud(indmax[-(pad_width - 1):])
            rmin = np.r_[X.shape[0] - 1, np.flipud(indmin[-(pad_width - 2):])]
            rsym = X.shape[0] - 1

    trmin = 2 * rsym - rmin
    trmax = 2 * rsym - rmax

    if trmin[-1] <= t[-1] or trmax[-1] <= t[-1]:
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[-pad_width - 1:-1])
        else:
            rmin = np.flipud(indmin[-pad_width - 1:-1])
        rsym = len(X)
        trmin = 2 * rsym - rmin
        trmax = 2 * rsym - rmax

    tmin = np.r_[tlmin, t[indmin], trmin]
    tmax = np.r_[tlmax, t[indmax], trmax]
    xmin = np.r_[X[lmin], X[indmin], X[rmin]]
    xmax = np.r_[X[lmax], X[indmax], X[rmax]]
    return tmin, xmin, tmax, xmax


def _splrep_envelope(locs: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """Fit a B-spline through padded extrema and evaluate over [0, n)."""
    t_range = np.arange(locs[0], locs[-1])
    env = splev(t_range, splrep(locs, vals))
    mask = (t_range >= 0) & (t_range < n)
    return env[mask]


def _sift_imfs(data: np.ndarray, max_imfs: int = 10, max_sift: int = 1000,
               sd_thresh: float = 0.1, pad_width: int = 2,
               sift_thresh: float = 1e-8) -> np.ndarray:
    """Extract Intrinsic Mode Functions from a 1-D signal via EMD sifting.

    Matches the behaviour of ``emd.sift.sift``:
    - Rilling (2003) boundary padding to reduce end effects.
    - B-spline envelope interpolation (``scipy.interpolate.splrep``).
    - SD-based inner stopping criterion (default threshold 0.1).
    - Energy-based outer stopping via ``sift_thresh``.

    Returns
    -------
    np.ndarray, shape (n_imfs, L)
    """
    imfs = []
    residual = data.astype(np.float64).copy()
    n = len(data)

    for _ in range(max_imfs):
        h = residual.copy()

        for _ in range(max_sift):
            max_idx = argrelextrema(h, np.greater)[0]
            min_idx = argrelextrema(h, np.less)[0]

            if len(max_idx) < pad_width + 1 or len(min_idx) < pad_width + 1:
                break

            try:
                tmax, xmax, tmin, xmin = _rilling_pad(min_idx, max_idx, h, pad_width)
                upper = _splrep_envelope(tmax, xmax, n)
                lower = _splrep_envelope(tmin, xmin, n)
            except (ValueError, TypeError):
                break

            mean_env = (upper + lower) / 2.0
            prev_h = h.copy()
            h = h - mean_env

            sd = float(np.sum((prev_h - h) ** 2) / (np.sum(prev_h ** 2) + 1e-10))
            if sd < sd_thresh:
                break

        imfs.append(h)
        residual = residual - h

        if np.sum(np.abs(residual)) < sift_thresh:
            break

        n_ext = len(argrelextrema(residual, np.greater)[0]) + len(argrelextrema(residual, np.less)[0])
        if n_ext < 2 * pad_width:
            break

    imfs.append(residual)
    return np.array(imfs, dtype=np.float32)


def _decompose(channel_data: np.ndarray, decomposer: str, max_imfs: int) -> np.ndarray:
    """Decompose a 1-D channel into IMFs, returning shape ``(n_imfs, L)``."""
    if decomposer == "sift_imfs":
        return _sift_imfs(channel_data, max_imfs=max_imfs)
    elif decomposer == "emd":
        import emd as _emd
        imfs = _emd.sift.sift(channel_data.astype(np.float64), max_imfs=max_imfs)
        return imfs.T.astype(np.float32)  # (samples, n_imfs) → (n_imfs, samples)
    else:
        raise ValueError(f"decomposer must be 'sift_imfs' or 'emd', got '{decomposer}'")


def _psd(data: np.ndarray) -> np.ndarray:
    """Normalised Welch PSD suitable for Jensen-Shannon distance."""
    _, pxx = signal.welch(data.astype(np.float64), scaling='spectrum')
    pxx = pxx + 1e-12
    return pxx / pxx.sum()


def _fingerprint_histogram(data: np.ndarray) -> np.ndarray:
    """Welch fingerprint histogram used in the legacy IMFACT implementation."""
    _, pxx = signal.welch(np.asarray(data, dtype=np.float64), scaling='spectrum')
    pxx = pxx + 1e-12
    return pxx / pxx.sum()


def _class_variance(series_list: list) -> float:
    """Mean squared deviation of per-series PSD from the class mean PSD."""
    if not series_list:
        return 0.0
    psds = np.stack([_psd(s) for s in series_list], axis=0)
    mean_psd = psds.mean(axis=0)
    return float(np.mean((psds - mean_psd) ** 2))


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    """Safe min-max normalisation for 1-D arrays."""
    x = np.asarray(x, dtype=np.float64)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x_min) / (x_max - x_min)


def _select_native_guide(
    sample_cl: np.ndarray,
    ts: np.ndarray,
    labels: np.ndarray,
    label_orig: int,
    model: torch.nn.Module,
    device: torch.device,
    target_class: int | None,
) -> tuple[np.ndarray, int, float, dict]:
    """Select a single native guide with a hybrid NUN score.

    Thin wrapper around :func:`_select_native_guides` with ``n_guides=1``.
    """
    guides, guide_labels, guide_js, guide_diags = _select_native_guides(
        sample_cl=sample_cl,
        ts=ts,
        labels=labels,
        label_orig=label_orig,
        model=model,
        device=device,
        target_class=target_class,
        n_guides=1,
    )
    return guides[0], int(guide_labels[0]), float(guide_js[0]), guide_diags[0]


def _select_native_guides(
    sample_cl: np.ndarray,
    ts: np.ndarray,
    labels: np.ndarray,
    label_orig: int,
    model: torch.nn.Module,
    device: torch.device,
    target_class: int | None,
    n_guides: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Select top-k native guides ranked by the same hybrid NUN score."""
    if target_class is None:
        mask = labels != label_orig
    else:
        target_class = int(target_class)
        if target_class == label_orig:
            raise ValueError("target_class must differ from the query sample class")
        mask = labels == target_class

    if not np.any(mask):
        if target_class is None:
            raise ValueError("No unlike-class candidate found for native guide selection")
        raise ValueError(f"No native guide found for target_class={target_class}")

    cand_ts = ts[mask]
    cand_labels = labels[mask]
    n_cands = cand_ts.shape[0]
    C, L = sample_cl.shape

    src_psd_per_c = [_psd(sample_cl[c]) for c in range(C)]
    js_dists = np.zeros(n_cands, dtype=np.float64)
    for i in range(n_cands):
        js_dists[i] = float(np.mean([
            jensenshannon(src_psd_per_c[c], _psd(cand_ts[i, c])) for c in range(C)
        ]))

    denom = np.sqrt(float(C * L)) + 1e-12
    l2_dists = np.linalg.norm(cand_ts.reshape(n_cands, -1) - sample_cl.reshape(1, -1), axis=1) / denom

    with torch.no_grad():
        cand_scores = detach_to_numpy(model(numpy_to_torch(cand_ts, device)))

    n_classes = int(cand_scores.shape[1])
    if target_class is not None:
        flip_class = int(target_class)
        margin = cand_scores[:, flip_class] - cand_scores[:, label_orig]
        pred_penalty = (np.argmax(cand_scores, axis=1) != flip_class).astype(np.float64)
    else:
        class_mask = np.ones(n_classes, dtype=bool)
        class_mask[label_orig] = False
        best_other = np.max(cand_scores[:, class_mask], axis=1)
        margin = best_other - cand_scores[:, label_orig]
        pred_penalty = np.zeros_like(margin)

    js_norm = _minmax_norm(js_dists)
    l2_norm = _minmax_norm(l2_dists)
    margin_cost = 1.0 - _minmax_norm(margin)
    score = 0.55 * js_norm + 0.25 * l2_norm + 0.20 * margin_cost + 0.10 * pred_penalty

    order = np.argsort(score)
    k = int(max(1, min(n_guides, len(order))))
    top_idx = order[:k]

    diagnostics = []
    for i in top_idx:
        diagnostics.append(
            {
                "hybrid_score": float(score[i]),
                "js_dist": float(js_dists[i]),
                "l2_dist": float(l2_dists[i]),
                "margin": float(margin[i]),
                "label": int(cand_labels[i]),
            }
        )

    return cand_ts[top_idx], cand_labels[top_idx], js_dists[top_idx], diagnostics


# ---------------------------------------------------------------------------
# Top-level CF function
# ---------------------------------------------------------------------------

####
# IMFACT-CF: Counterfactual Explanations via Empirical Mode Decomposition
#
# Reference: imfact_old.py (internal)
#
# Strategy:
#   1. Find a native guide in the dataset: the nearest sample (by Jensen-Shannon
#      distance on Welch PSD) that belongs to a different class, or to a
#      specific target_class when one is provided.
#   2. Decompose source and native guide into Intrinsic Mode Functions (IMFs)
#      using basic EMD sifting.
#   3. Initialise per-IMF interpolation weights (w_source=1, w_target=0).
#   4. Iteratively step weights towards the native guide, scaling each IMF's
#      step by its PSD distance (method="distance"), class-level PSD variance
#      difference (method="variance"), the strongest/weakest IMF PSD distances
#      (method="extremes"), IMF amplitude range ordering (method="maxmin"),
#      or a coarse-to-fine schedule that
#      unlocks one more IMF every few iterations.
#   5. Reconstruct the signal and query the model; stop on a class flip.
####
def imfact_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    method: str = "distance",
    step: float = 0.05,
    max_iter: int = 200,
    max_imfs: int = 10,
    coarse_stage_iters: int = 10,
    n_nuns: int = 1,
    nun_switch: str = "cycle",
    decomposer: str = "sift_imfs",
    max_samples: int | None = None,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """IMFACT-based counterfactual explanation for time series classification.

    Follows the same signature pattern as every other CF method in this
    repository (native_guide_uni_cf, glacier_cf, cem_cf, …) so it plugs
    straight into the existing evaluation and example scripts.

    Parameters
    ----------
    sample:
        Query time series; 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``.
    model:
        PyTorch classifier with signature ``forward(B, C, L) -> (B, n_classes)``.
    target_class:
        Optional class label to restrict the native guide search to, and the
        stopping criterion: iteration stops as soon as the model predicts
        ``target_class``.  When ``None``, the guide is taken from any class
        different from the query's predicted class, and iteration stops on
        any class flip.
    dataset:
        Training set as a sequence of ``(x, y)`` pairs.
    method:
        IMF weighting strategy: ``"distance"`` (JSD between interpolated and
        target IMF PSDs), ``"fingerprint"`` (legacy Welch-fingerprint IMF
        distances as in ``imfact_old.py``), ``"variance"`` (class-level PSD
        variance difference), ``"extremes"`` (step only the most and least
        distant IMFs between query and native guide in PSD space), ``"maxmin"``
        (prioritise IMFs with larger amplitude range, i.e. max-min), or
        ``"coarse_to_fine"`` (iteratively unlock IMFs from coarse to fine).
    step:
        Base interpolation step per iteration.
    max_iter:
        Maximum interpolation steps before returning best candidate.
    max_imfs:
        Maximum number of IMFs to extract per channel.
    coarse_stage_iters:
        Number of iterations spent on each coarse-to-fine stage.  Only used
        when ``method`` is ``"coarse_to_fine"``.
    n_nuns:
        Number of top-ranked native guides to use. ``1`` keeps the original
        behavior. Values ``>1`` enable multi-guide updates.
    nun_switch:
        Switching rule when ``n_nuns > 1``. ``"cycle"`` rotates guides each
        iteration; ``"closest_psd"`` picks the currently closest guide in PSD
        space to the latest candidate signal.
    decomposer:
        EMD back-end to use for IMF extraction. ``"sift_imfs"`` uses the
        built-in Rilling-padded sifter; ``"emd"`` delegates to
        ``emd.sift.sift`` from the *emd-signal* package (must be installed).
    verbose:
        Print per-iteration diagnostics when ``True``.

    Returns
    -------
    counterfactual : np.ndarray
        In the same shape / orientation as *sample*.
    scores : np.ndarray, shape (n_classes,)
        Model output scores for the counterfactual.
    """
    if method not in ("distance", "fingerprint", "variance", "extremes", "maxmin", "coarse_to_fine"):
        raise ValueError("method must be 'distance', 'fingerprint', 'variance', 'extremes', 'maxmin', or 'coarse_to_fine'")
    if decomposer not in ("sift_imfs", "emd"):
        raise ValueError("decomposer must be 'sift_imfs' or 'emd'")
    if nun_switch not in ("cycle", "closest_psd"):
        raise ValueError("nun_switch must be 'cycle' or 'closest_psd'")
    if n_nuns < 1:
        raise ValueError("n_nuns must be >= 1")

    device = next(model.parameters()).device

    # --- normalise input to (C, L) and dataset to (N, C, L) ----------------
    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)
    sample_cl, ts, ori = ensure_ncl(sample, dataset)
    C, L = sample_cl.shape
    N = ts.shape[0]
    raw_labels = np.array([item[1] for item in dataset])
    # Support both integer labels and one-hot encoded labels
    labels = np.argmax(raw_labels, axis=1).astype(int) if raw_labels.ndim > 1 else raw_labels.astype(int)

    def _predict(arr_ncl: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return detach_to_numpy(model(numpy_to_torch(arr_ncl, device)))

    # --- original prediction ------------------------------------------------
    scores_orig = _predict(sample_cl.reshape(1, C, L)).reshape(-1)
    label_orig = int(np.argmax(scores_orig))

    # --- find native guide / NUN(s) with a hybrid score --------------------
    try:
        guides, guide_labels, guide_js, guide_diags = _select_native_guides(
            sample_cl=sample_cl,
            ts=ts,
            labels=labels,
            label_orig=label_orig,
            model=model,
            device=device,
            target_class=target_class,
            n_guides=n_nuns,
        )
    except ValueError:
        # Conservative fallback: if no candidate exists, return original sample.
        return revert_orientation(sample_cl, ori), scores_orig

    # Keep first guide as canonical for backward-compatible bookkeeping.
    native_guide = guides[0]
    ng_label = int(guide_labels[0])
    ng_js = float(guide_js[0])
    ng_diag = guide_diags[0]

    if verbose:
        print(f"[imfact_cf] using {len(guides)} guide(s), switch={nun_switch}")
        for idx, diag in enumerate(guide_diags):
            print(
                f"[imfact_cf] guide#{idx} label={diag['label']} JS-dist={diag['js_dist']:.4f} "
                f"L2-dist={diag['l2_dist']:.4f} margin={diag['margin']:.4f} hybrid={diag['hybrid_score']:.4f}"
            )

    # --- IMF decomposition per channel --------------------------------------
    src_imfs = [_decompose(sample_cl[c], decomposer, max_imfs) for c in range(C)]
    guide_imfs = []
    n_imfs_per_c = [len(src_imfs[c]) for c in range(C)]
    for g in guides:
        g_imfs = [_decompose(g[c], decomposer, max_imfs) for c in range(C)]
        guide_imfs.append(g_imfs)
        for c in range(C):
            n_imfs_per_c[c] = max(n_imfs_per_c[c], len(g_imfs[c]))

    # Align number of IMFs per channel with zero-padding for source and all guides.
    for c in range(C):
        while len(src_imfs[c]) < n_imfs_per_c[c]:
            src_imfs[c] = np.vstack([src_imfs[c], np.zeros((1, L), dtype=np.float32)])
    for g_idx in range(len(guide_imfs)):
        for c in range(C):
            while len(guide_imfs[g_idx][c]) < n_imfs_per_c[c]:
                guide_imfs[g_idx][c] = np.vstack([guide_imfs[g_idx][c], np.zeros((1, L), dtype=np.float32)])

    # --- initialise per-channel per-IMF interpolation weights ---------------
    weights = [
        [{"w_source": 1.0, "w_target": 0.0} for _ in range(n_imfs_per_c[c])]
        for c in range(C)
    ]

    active_guide_idx = 0

    def _set_active_guide(iter_idx: int, current_signal: np.ndarray):
        nonlocal active_guide_idx
        if len(guides) == 1:
            active_guide_idx = 0
            return
        if nun_switch == "cycle":
            active_guide_idx = int(iter_idx % len(guides))
            return

        # closest_psd: choose guide closest to current signal in mean channel PSD distance.
        sig_psd_per_c = [_psd(current_signal[c]) for c in range(C)]
        dists = np.zeros(len(guides), dtype=np.float64)
        for g_idx in range(len(guides)):
            dists[g_idx] = float(np.mean([
                jensenshannon(sig_psd_per_c[c], _psd(guides[g_idx][c])) for c in range(C)
            ]))
        active_guide_idx = int(np.argmin(dists))

    def _active_ng_imfs() -> list[np.ndarray]:
        return guide_imfs[active_guide_idx]

    def _reconstruct() -> np.ndarray:
        result = np.zeros((C, L), dtype=np.float32)
        ng_imfs = _active_ng_imfs()
        for c in range(C):
            for k, wk in enumerate(weights[c]):
                # Preserves the original formula from imfact_old (w_target applied twice)
                result[c] += wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
        return result

    def _distance_step_weights() -> list[np.ndarray]:
        """Normalised JSD between current interpolated IMF PSDs and native guide IMF PSDs."""
        step_w = []
        ng_imfs = _active_ng_imfs()
        for c in range(C):
            d = np.zeros(n_imfs_per_c[c])
            for k, wk in enumerate(weights[c]):
                interp_imf = wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
                d[k] = jensenshannon(_psd(interp_imf), _psd(ng_imfs[c][k]))
            mx = np.max(d)
            step_w.append(d / (mx + 1e-12))
        return step_w

    def _fingerprint_step_weights() -> list[np.ndarray]:
        """Legacy IMF fingerprint distance using Welch histograms (imfact_old style)."""
        step_w = []
        ng_imfs = _active_ng_imfs()
        for c in range(C):
            d = np.zeros(n_imfs_per_c[c])
            for k, wk in enumerate(weights[c]):
                interp_imf = wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
                hist_interp = _fingerprint_histogram(interp_imf)
                hist_target = _fingerprint_histogram(ng_imfs[c][k])
                d[k] = jensenshannon(hist_interp, hist_target)
            mx = np.max(d)
            step_w.append(d / (mx + 1e-12))
        return step_w

    def _variance_step_weights() -> list[np.ndarray]:
        """Normalised per-IMF class variance difference (source vs target class)."""
        src_series = [ts[i, 0] for i in range(N) if labels[i] == label_orig]
        if target_class is None:
            target_mask = labels != label_orig
        else:
            target_mask = labels == int(target_class)
        tgt_series = [ts[i, 0] for i in range(N) if target_mask[i]]
        step_w = []
        for c in range(C):
            v = np.zeros(n_imfs_per_c[c])
            for k in range(n_imfs_per_c[c]):
                var_src = _class_variance(src_series)
                var_tgt = _class_variance(tgt_series)
                v[k] = abs(var_src - var_tgt)
            mx = np.max(v)
            step_w.append(v / (mx + 1e-12))
        return step_w

    def _extreme_step_weights() -> list[np.ndarray]:
        """Keep only the strongest and weakest IMF distances for each channel."""
        step_w = []
        ng_imfs = _active_ng_imfs()
        for c in range(C):
            d = np.zeros(n_imfs_per_c[c])
            for k, wk in enumerate(weights[c]):
                interp_imf = wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
                d[k] = jensenshannon(_psd(interp_imf), _psd(ng_imfs[c][k]))

            if d.size == 0:
                step_w.append(d)
                continue

            max_idx = int(np.argmax(d))
            min_idx = int(np.argmin(d))
            w = np.zeros_like(d)
            w[max_idx] = 1.0
            w[min_idx] = float(d[min_idx] / (d[max_idx] + 1e-12))
            step_w.append(w)

        return step_w

    def _maxmin_step_weights() -> list[np.ndarray]:
        """Prioritise IMFs with larger amplitude range (max-min)."""
        step_w = []
        ng_imfs = _active_ng_imfs()
        for c in range(C):
            amp = np.zeros(n_imfs_per_c[c], dtype=np.float64)
            for k in range(n_imfs_per_c[c]):
                # Use both source and active guide IMF ranges to avoid overfitting
                # the ranking to one side of the interpolation.
                src_range = float(np.ptp(src_imfs[c][k]))
                ng_range = float(np.ptp(ng_imfs[c][k]))
                amp[k] = 0.5 * (src_range + ng_range)

            mx = float(np.max(amp)) if amp.size else 0.0
            if mx > 0.0:
                step_w.append((amp / (mx + 1e-12)).astype(np.float64))
            else:
                step_w.append(np.zeros_like(amp, dtype=np.float64))

        return step_w

    def _coarse_to_fine_step_weights(iter_idx: int) -> list[np.ndarray]:
        """Unlock one more IMF every ``coarse_stage_iters`` iterations.

        The coarsest IMF is the last one returned by the decomposition.
        """
        if coarse_stage_iters <= 0:
            raise ValueError("coarse_stage_iters must be > 0 for coarse_to_fine mode")

        stage = iter_idx // coarse_stage_iters
        step_w = []
        ng_imfs = _active_ng_imfs()
        for c in range(C):
            d = np.zeros(n_imfs_per_c[c])
            active_count = min(n_imfs_per_c[c], 1 + stage)
            active_start = max(0, n_imfs_per_c[c] - active_count)

            for k in range(active_start, n_imfs_per_c[c]):
                wk = weights[c][k]
                interp_imf = wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
                d[k] = jensenshannon(_psd(interp_imf), _psd(ng_imfs[c][k]))

            active_slice = d[active_start:]
            mx = np.max(active_slice) if active_slice.size else 0.0
            if mx > 0:
                d[active_start:] = d[active_start:] / (mx + 1e-12)
            step_w.append(d)

        return step_w

    # Pre-compute variance weights once (independent of current weights)
    precomputed_var_weights = _variance_step_weights() if method == "variance" else None

    def _advance_weights(iter_idx: int):
        _set_active_guide(iter_idx, cf)
        if method == "distance":
            sw = _distance_step_weights()
        elif method == "fingerprint":
            sw = _fingerprint_step_weights()
        elif method == "variance":
            sw = precomputed_var_weights
        elif method == "extremes":
            sw = _extreme_step_weights()
        elif method == "maxmin":
            sw = _maxmin_step_weights()
        elif method == "coarse_to_fine":
            sw = _coarse_to_fine_step_weights(iter_idx)
        else:
            raise RuntimeError(f"Unhandled method: {method}")
        for c in range(C):
            for k, wk in enumerate(weights[c]):
                delta = step * float(sw[c][k])
                wk["w_target"] = min(1.0, wk["w_target"] + delta)
                wk["w_source"] = max(0.0, wk["w_source"] - delta)

    # --- main interpolation loop --------------------------------------------
    cf = sample_cl.copy()
    scores_cf = scores_orig.copy()

    target_stop_class = None if target_class is None else int(target_class)

    for i in range(max_iter):
        _advance_weights(i)
        candidate = _reconstruct()

        scores_cand = _predict(candidate.reshape(1, C, L)).reshape(-1)
        label_cand = int(np.argmax(scores_cand))

        if verbose:
            print(
                f"[imfact_cf] iter={i:3d}  guide={active_guide_idx} "
                f"predicted={label_cand}  original={label_orig}"
            )

        cf = candidate
        scores_cf = scores_cand

        # Untargeted mode: stop on first class flip.
        # Targeted mode: stop only when the selected target class is reached.
        if target_stop_class is None:
            if label_cand != label_orig:
                break
        else:
            if label_cand == target_stop_class:
                break

    return revert_orientation(cf, ori), scores_cf


####
# trace_imfact_variant_path – per-iteration tracing wrapper around imfact_cf
#
# Records the full interpolation history (signal, prediction, confidence,
# target probability, L2 distance to the native guide) at every iteration
# so that callers can inspect or visualise how the counterfactual evolves.
# The algorithm is identical to imfact_cf with n_nuns=1; only the return
# value changes from a (cf, scores) tuple to a rich history dict.
####
def trace_imfact_variant_path(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    method: str = "distance",
    step: float = 0.05,
    max_iter: int = 25,
    max_imfs: int = 10,
    coarse_stage_iters: int = 10,
    decomposer: str = "sift_imfs",
    max_samples: int | None = None,
    verbose: bool = False,
    *args,
    **kwargs,
) -> dict:
    """Trace the per-iteration path of a single IMFACT variant.

    Runs the same interpolation loop as :func:`imfact_cf` (with ``n_nuns=1``)
    and records a snapshot at every iteration so the counterfactual trajectory
    can be inspected, plotted, or projected into a latent space.

    Follows the same signature pattern as every other CF method in this
    repository so it plugs straight into existing evaluation scripts.

    Parameters
    ----------
    sample:
        Query time series; 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``.
    model:
        PyTorch classifier with signature ``forward(B, C, L) -> (B, n_classes)``.
    target_class:
        Class index whose probability is tracked in the history ``target_prob``
        field.  Also restricts the native guide search to this class and acts
        as the stopping criterion: iteration stops as soon as the model
        predicts ``target_class``.  When ``None``, any unlike-class candidate
        is eligible as a guide, iteration stops on any class flip, and
        ``target_prob`` records the max non-original class score.
    dataset:
        Training set as a sequence of ``(x, y)`` pairs used to find the NUN.
    method:
        IMF weighting strategy — same options as :func:`imfact_cf`:
        ``"distance"``, ``"fingerprint"``, ``"variance"``, ``"extremes"``,
        ``"maxmin"``, or ``"coarse_to_fine"``.
    step:
        Base interpolation step per iteration.
    max_iter:
        Maximum number of iterations before returning the best candidate.
    max_imfs:
        Maximum number of IMFs to extract per channel.
    coarse_stage_iters:
        Iterations per coarse-to-fine stage (only used for ``"coarse_to_fine"``).
    decomposer:
        ``"sift_imfs"`` (built-in) or ``"emd"`` (requires *emd-signal* package).
    verbose:
        Print per-iteration diagnostics when ``True``.

    Returns
    -------
    dict with keys:

    ``method`` : str
        The weighting strategy used.
    ``history`` : list[dict]
        One entry per iteration (plus iteration ``-1`` for the original sample).
        Each entry has: ``iteration``, ``signal`` (C, L), ``pred_class``,
        ``confidence``, ``target_prob``, ``l2_to_guide``.
    ``original_scores`` : np.ndarray, shape (n_classes,)
    ``final_cf`` : np.ndarray, shape (C, L)
        Counterfactual in (C, L) layout (use :func:`revert_orientation` if the
        original orientation is needed).
    ``final_scores`` : np.ndarray, shape (n_classes,)
    ``native_guide`` : np.ndarray, shape (C, L)
    ``native_guide_label`` : int
    ``original_class`` : int
    ``target_class`` : int | None
    """
    if method not in ("distance", "fingerprint", "variance", "extremes", "maxmin", "coarse_to_fine"):
        raise ValueError(
            "method must be 'distance', 'fingerprint', 'variance', 'extremes', 'maxmin', or 'coarse_to_fine'"
        )
    if decomposer not in ("sift_imfs", "emd"):
        raise ValueError("decomposer must be 'sift_imfs' or 'emd'")

    device = next(model.parameters()).device

    # --- 1. Normalise input shapes ---
    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)
    sample_cl, ts, ori = ensure_ncl(sample, dataset)
    C, L = sample_cl.shape
    N = ts.shape[0]
    raw_labels = np.array([item[1] for item in dataset])
    labels = np.argmax(raw_labels, axis=1).astype(int) if raw_labels.ndim > 1 else raw_labels.astype(int)

    def _predict(arr_ncl: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return detach_to_numpy(model(numpy_to_torch(arr_ncl, device)))

    # --- 2. Original prediction ---
    scores_orig = _predict(sample_cl.reshape(1, C, L)).reshape(-1)
    label_orig = int(np.argmax(scores_orig))

    # --- 3. Select a single native guide (NUN) with the hybrid score ---
    try:
        native_guide, ng_label, _, ng_diag = _select_native_guide(
            sample_cl=sample_cl,
            ts=ts,
            labels=labels,
            label_orig=label_orig,
            model=model,
            device=device,
            target_class=target_class,
        )
    except ValueError:
        # No candidate found — return a trivial history with the original sample.
        trivial_entry = {
            "iteration": -1,
            "signal": sample_cl.copy(),
            "pred_class": label_orig,
            "confidence": float(np.max(scores_orig)),
            "target_prob": float(scores_orig[target_class]) if target_class is not None else float(np.max(scores_orig)),
            "l2_to_guide": 0.0,
        }
        return {
            "method": method,
            "history": [trivial_entry],
            "original_scores": scores_orig,
            "final_cf": sample_cl.copy(),
            "final_scores": scores_orig,
            "native_guide": sample_cl.copy(),
            "native_guide_label": label_orig,
            "original_class": label_orig,
            "target_class": target_class,
        }

    if verbose:
        print(
            f"[trace] guide label={ng_label} JS-dist={ng_diag['js_dist']:.4f} "
            f"L2-dist={ng_diag['l2_dist']:.4f} hybrid={ng_diag['hybrid_score']:.4f}"
        )

    # --- 4. IMF decomposition ---
    src_imfs = [_decompose(sample_cl[c], decomposer, max_imfs) for c in range(C)]
    ng_imfs = [_decompose(native_guide[c], decomposer, max_imfs) for c in range(C)]

    n_imfs_per_c = [max(len(src_imfs[c]), len(ng_imfs[c])) for c in range(C)]
    for c in range(C):
        while len(src_imfs[c]) < n_imfs_per_c[c]:
            src_imfs[c] = np.vstack([src_imfs[c], np.zeros((1, L), dtype=np.float32)])
        while len(ng_imfs[c]) < n_imfs_per_c[c]:
            ng_imfs[c] = np.vstack([ng_imfs[c], np.zeros((1, L), dtype=np.float32)])

    # --- 5. Initialise per-IMF interpolation weights ---
    weights = [
        [{"w_source": 1.0, "w_target": 0.0} for _ in range(n_imfs_per_c[c])]
        for c in range(C)
    ]

    def _reconstruct() -> np.ndarray:
        result = np.zeros((C, L), dtype=np.float32)
        for c in range(C):
            for k, wk in enumerate(weights[c]):
                result[c] += wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
        return result

    def _distance_step_weights() -> list[np.ndarray]:
        step_w = []
        for c in range(C):
            d = np.zeros(n_imfs_per_c[c])
            for k, wk in enumerate(weights[c]):
                interp = wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
                d[k] = jensenshannon(_psd(interp), _psd(ng_imfs[c][k]))
            step_w.append(d / (np.max(d) + 1e-12))
        return step_w

    def _fingerprint_step_weights() -> list[np.ndarray]:
        step_w = []
        for c in range(C):
            d = np.zeros(n_imfs_per_c[c])
            for k, wk in enumerate(weights[c]):
                interp = wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
                d[k] = jensenshannon(_fingerprint_histogram(interp), _fingerprint_histogram(ng_imfs[c][k]))
            step_w.append(d / (np.max(d) + 1e-12))
        return step_w

    def _variance_step_weights() -> list[np.ndarray]:
        src_series = [ts[i, 0] for i in range(N) if labels[i] == label_orig]
        if target_class is None:
            tgt_series = [ts[i, 0] for i in range(N) if labels[i] != label_orig]
        else:
            tgt_series = [ts[i, 0] for i in range(N) if labels[i] == int(target_class)]
        base = abs(_class_variance(src_series) - _class_variance(tgt_series))
        step_w = []
        for c in range(C):
            w = np.full(n_imfs_per_c[c], base)
            mx = np.max(w)
            step_w.append(w / (mx + 1e-12) if mx > 0 else w)
        return step_w

    def _extreme_step_weights() -> list[np.ndarray]:
        step_w = []
        for c in range(C):
            d = np.zeros(n_imfs_per_c[c])
            for k, wk in enumerate(weights[c]):
                interp = wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
                d[k] = jensenshannon(_psd(interp), _psd(ng_imfs[c][k]))
            if d.size == 0:
                step_w.append(d)
                continue
            w = np.zeros_like(d)
            max_idx, min_idx = int(np.argmax(d)), int(np.argmin(d))
            w[max_idx] = 1.0
            w[min_idx] = float(d[min_idx] / (d[max_idx] + 1e-12))
            step_w.append(w)
        return step_w

    def _maxmin_step_weights() -> list[np.ndarray]:
        step_w = []
        for c in range(C):
            amp = np.zeros(n_imfs_per_c[c], dtype=np.float64)
            for k in range(n_imfs_per_c[c]):
                amp[k] = 0.5 * (float(np.ptp(src_imfs[c][k])) + float(np.ptp(ng_imfs[c][k])))
            mx = float(np.max(amp)) if amp.size else 0.0
            step_w.append((amp / (mx + 1e-12)) if mx > 0.0 else np.zeros_like(amp))
        return step_w

    def _coarse_to_fine_step_weights(iter_idx: int) -> list[np.ndarray]:
        if coarse_stage_iters <= 0:
            raise ValueError("coarse_stage_iters must be > 0 for coarse_to_fine mode")
        stage = iter_idx // coarse_stage_iters
        step_w = []
        for c in range(C):
            d = np.zeros(n_imfs_per_c[c])
            active_count = min(n_imfs_per_c[c], 1 + stage)
            active_start = max(0, n_imfs_per_c[c] - active_count)
            for k in range(active_start, n_imfs_per_c[c]):
                wk = weights[c][k]
                interp = wk["w_source"] * src_imfs[c][k] + wk["w_target"] * ng_imfs[c][k] * wk["w_target"]
                d[k] = jensenshannon(_psd(interp), _psd(ng_imfs[c][k]))
            active_slice = d[active_start:]
            mx = np.max(active_slice) if active_slice.size else 0.0
            if mx > 0:
                d[active_start:] = d[active_start:] / (mx + 1e-12)
            step_w.append(d)
        return step_w

    precomputed_var_weights = _variance_step_weights() if method == "variance" else None

    def _advance_weights(iter_idx: int):
        if method == "distance":
            sw = _distance_step_weights()
        elif method == "fingerprint":
            sw = _fingerprint_step_weights()
        elif method == "variance":
            sw = precomputed_var_weights
        elif method == "extremes":
            sw = _extreme_step_weights()
        elif method == "maxmin":
            sw = _maxmin_step_weights()
        elif method == "coarse_to_fine":
            sw = _coarse_to_fine_step_weights(iter_idx)
        else:
            raise RuntimeError(f"Unhandled method: {method}")
        for c in range(C):
            for k, wk in enumerate(weights[c]):
                delta = step * float(sw[c][k])
                wk["w_target"] = min(1.0, wk["w_target"] + delta)
                wk["w_source"] = max(0.0, wk["w_source"] - delta)

    # --- 6. Record initial state, then iterate ---
    guide_flat = native_guide.reshape(-1)

    def _make_entry(iteration: int, sig: np.ndarray, scores: np.ndarray) -> dict:
        n_classes = len(scores)
        if target_class is not None:
            t_prob = float(scores[target_class])
        else:
            non_orig = [scores[c] for c in range(n_classes) if c != label_orig]
            t_prob = float(max(non_orig)) if non_orig else 0.0
        return {
            "iteration": iteration,
            "signal": sig.copy(),
            "pred_class": int(np.argmax(scores)),
            "confidence": float(np.max(scores)),
            "target_prob": t_prob,
            "l2_to_guide": float(np.linalg.norm(sig.reshape(-1) - guide_flat)),
        }

    history = [_make_entry(-1, sample_cl, scores_orig)]

    cf = sample_cl.copy()
    scores_cf = scores_orig.copy()
    target_stop = target_class  # None → any flip; int → specific class

    for i in range(max_iter):
        _advance_weights(i)
        candidate = _reconstruct()
        scores_cand = _predict(candidate.reshape(1, C, L)).reshape(-1)
        label_cand = int(np.argmax(scores_cand))

        history.append(_make_entry(i, candidate, scores_cand))
        cf = candidate
        scores_cf = scores_cand

        if verbose:
            print(
                f"[trace] iter={i:3d}  method={method}  "
                f"pred={label_cand}  orig={label_orig}  "
                f"target_prob={scores_cand[target_class]:.4f}" if target_class is not None
                else f"[trace] iter={i:3d}  method={method}  pred={label_cand}  orig={label_orig}"
            )

        stopped = (target_stop is None and label_cand != label_orig) or (
            target_stop is not None and label_cand == target_stop
        )
        if stopped:
            break

    return {
        "method": method,
        "history": history,
        "original_scores": scores_orig,
        "final_cf": cf,
        "final_scores": scores_cf,
        "native_guide": native_guide,
        "native_guide_label": ng_label,
        "original_class": label_orig,
        "target_class": target_class,
    }
