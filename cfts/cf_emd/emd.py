from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema
from scipy.spatial.distance import jensenshannon

from cfts.cf__abstract.abstract import (
    detach_to_numpy,
    numpy_to_torch,
    ensure_cl,
    ensure_ncl,
    revert_orientation,
)


# ---------------------------------------------------------------------------
# EMD helpers
# ---------------------------------------------------------------------------

def _sift_imfs(data: np.ndarray, max_imfs: int = 10, max_sift: int = 100, sd_thresh: float = 0.2) -> np.ndarray:
    """Extract Intrinsic Mode Functions from a 1-D signal via basic EMD sifting.

    Returns
    -------
    np.ndarray, shape (n_imfs, L)
    """
    imfs = []
    residual = data.astype(np.float64).copy()
    t = np.arange(len(data))

    for _ in range(max_imfs):
        h = residual.copy()

        for _ in range(max_sift):
            max_idx = argrelextrema(h, np.greater)[0]
            min_idx = argrelextrema(h, np.less)[0]

            if len(max_idx) < 2 or len(min_idx) < 2:
                break

            max_t = np.r_[0, max_idx, len(h) - 1]
            max_v = np.r_[h[max_idx[0]], h[max_idx], h[max_idx[-1]]]
            min_t = np.r_[0, min_idx, len(h) - 1]
            min_v = np.r_[h[min_idx[0]], h[min_idx], h[min_idx[-1]]]

            upper = CubicSpline(max_t, max_v)(t)
            lower = CubicSpline(min_t, min_v)(t)
            mean_env = (upper + lower) / 2.0

            prev_h = h.copy()
            h = h - mean_env

            sd = float(np.sum((prev_h - h) ** 2) / (np.sum(prev_h ** 2) + 1e-10))
            if sd < sd_thresh:
                break

        imfs.append(h)
        residual = residual - h

        n_ext = len(argrelextrema(residual, np.greater)[0]) + len(argrelextrema(residual, np.less)[0])
        if n_ext < 2:
            break

    imfs.append(residual)
    return np.array(imfs, dtype=np.float32)


def _psd(data: np.ndarray) -> np.ndarray:
    """Normalised Welch PSD suitable for Jensen-Shannon distance."""
    _, pxx = signal.welch(data.astype(np.float64), scaling='spectrum')
    pxx = pxx + 1e-12
    return pxx / pxx.sum()


def _fingerprint_histogram(data: np.ndarray) -> np.ndarray:
    """Welch fingerprint histogram used in the legacy EMD implementation."""
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
    scores_orig: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    guide_class: int | None,
) -> tuple[np.ndarray, int, float, dict]:
    """Select a native guide with a hybrid NUN score.

    The score blends:
    - PSD Jensen-Shannon distance (spectral similarity)
    - L2 distance in raw signal space (shape proximity)
    - Model margin towards a target flip class (flip-likelihood)
    """
    if guide_class is None:
        mask = labels != label_orig
    else:
        guide_class = int(guide_class)
        if guide_class == label_orig:
            raise ValueError("guide_class must differ from the query sample class")
        mask = labels == guide_class

    if not np.any(mask):
        if guide_class is None:
            raise ValueError("No unlike-class candidate found for native guide selection")
        raise ValueError(f"No native guide found for guide_class={guide_class}")

    cand_ts = ts[mask]
    cand_labels = labels[mask]
    n_cands = cand_ts.shape[0]
    C, L = sample_cl.shape

    # Mean PSD distance across channels.
    src_psd_per_c = [_psd(sample_cl[c]) for c in range(C)]
    js_dists = np.zeros(n_cands, dtype=np.float64)
    for i in range(n_cands):
        js_dists[i] = float(np.mean([
            jensenshannon(src_psd_per_c[c], _psd(cand_ts[i, c])) for c in range(C)
        ]))

    # Raw-space proximity (normalised by signal length).
    denom = np.sqrt(float(C * L)) + 1e-12
    l2_dists = np.linalg.norm(cand_ts.reshape(n_cands, -1) - sample_cl.reshape(1, -1), axis=1) / denom

    # Model-side flip likelihood for candidates.
    with torch.no_grad():
        cand_scores = detach_to_numpy(model(numpy_to_torch(cand_ts, device)))

    n_classes = int(cand_scores.shape[1])
    if guide_class is not None:
        target_class = int(guide_class)
        margin = cand_scores[:, target_class] - cand_scores[:, label_orig]
        pred_penalty = (np.argmax(cand_scores, axis=1) != target_class).astype(np.float64)
    else:
        class_mask = np.ones(n_classes, dtype=bool)
        class_mask[label_orig] = False
        best_other = np.max(cand_scores[:, class_mask], axis=1)
        margin = best_other - cand_scores[:, label_orig]
        pred_penalty = np.zeros_like(margin)

    js_norm = _minmax_norm(js_dists)
    l2_norm = _minmax_norm(l2_dists)
    margin_cost = 1.0 - _minmax_norm(margin)

    # Weighting tuned to keep NUN close while preferring flip-ready guides.
    score = 0.55 * js_norm + 0.25 * l2_norm + 0.20 * margin_cost + 0.10 * pred_penalty

    best_idx = int(np.argmin(score))
    diagnostics = {
        "hybrid_score": float(score[best_idx]),
        "js_dist": float(js_dists[best_idx]),
        "l2_dist": float(l2_dists[best_idx]),
        "margin": float(margin[best_idx]),
    }
    return cand_ts[best_idx], int(cand_labels[best_idx]), float(js_dists[best_idx]), diagnostics


def _select_native_guides(
    sample_cl: np.ndarray,
    ts: np.ndarray,
    labels: np.ndarray,
    label_orig: int,
    scores_orig: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    guide_class: int | None,
    n_guides: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Select top-k native guides ranked by the same hybrid NUN score."""
    if guide_class is None:
        mask = labels != label_orig
    else:
        guide_class = int(guide_class)
        if guide_class == label_orig:
            raise ValueError("guide_class must differ from the query sample class")
        mask = labels == guide_class

    if not np.any(mask):
        if guide_class is None:
            raise ValueError("No unlike-class candidate found for native guide selection")
        raise ValueError(f"No native guide found for guide_class={guide_class}")

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
    if guide_class is not None:
        target_class = int(guide_class)
        margin = cand_scores[:, target_class] - cand_scores[:, label_orig]
        pred_penalty = (np.argmax(cand_scores, axis=1) != target_class).astype(np.float64)
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
# EMD-CF: Counterfactual Explanations via Empirical Mode Decomposition
#
# Reference: emd_old.py (internal)
#
# Strategy:
#   1. Find a native guide in the dataset: the nearest sample (by Jensen-Shannon
#      distance on Welch PSD) that belongs to a different class, or to a
#      specific guide_class when one is provided.
#   2. Decompose source and native guide into Intrinsic Mode Functions (IMFs)
#      using basic EMD sifting.
#   3. Initialise per-IMF interpolation weights (w_source=1, w_target=0).
#   4. Iteratively step weights towards the native guide, scaling each IMF's
#      step by its PSD distance (method="distance"), class-level PSD variance
#      difference (method="variance"), or the strongest/weakest IMF distances
#      (method="extremes" / "maxmin"), or a coarse-to-fine schedule that
#      unlocks one more IMF every few iterations.
#   5. Reconstruct the signal and query the model; stop on a class flip.
####
def emd_cf(
    sample: np.ndarray,
    dataset,
    model: torch.nn.Module,
    method: str = "distance",
    guide_class: int | None = None,
    step: float = 0.05,
    max_iter: int = 200,
    max_imfs: int = 10,
    coarse_stage_iters: int = 10,
    n_nuns: int = 1,
    nun_switch: str = "cycle",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """EMD-based counterfactual explanation for time series classification.

    Parameters
    ----------
    sample:
        Query time series; 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``.
    dataset:
        Training set as a sequence of ``(x, y)`` pairs.
    model:
        PyTorch classifier with signature ``forward(B, C, L) -> (B, n_classes)``.
    method:
        IMF weighting strategy: ``"distance"`` (JSD between interpolated and
        target IMF PSDs), ``"fingerprint"`` (legacy Welch-fingerprint IMF
        distances as in ``emd_old.py``), ``"variance"`` (class-level PSD
        variance difference), or ``"extremes"``/``"maxmin"`` (step only the
        most and least distant IMFs between the query sample and native guide),
        or ``"coarse_to_fine"`` (iteratively unlock IMFs from coarse to fine).
    guide_class:
        Optional class label to restrict the native guide search to.  When
        ``None``, the guide is taken from any class different from the query's
        predicted class.
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
    if nun_switch not in ("cycle", "closest_psd"):
        raise ValueError("nun_switch must be 'cycle' or 'closest_psd'")
    if n_nuns < 1:
        raise ValueError("n_nuns must be >= 1")

    device = next(model.parameters()).device

    # --- normalise input to (C, L) and dataset to (N, C, L) ----------------
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
            scores_orig=scores_orig,
            model=model,
            device=device,
            guide_class=guide_class,
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
        print(f"[emd_cf] using {len(guides)} guide(s), switch={nun_switch}")
        for idx, diag in enumerate(guide_diags):
            print(
                f"[emd_cf] guide#{idx} label={diag['label']} JS-dist={diag['js_dist']:.4f} "
                f"L2-dist={diag['l2_dist']:.4f} margin={diag['margin']:.4f} hybrid={diag['hybrid_score']:.4f}"
            )

    # --- EMD decomposition per channel --------------------------------------
    src_imfs = [_sift_imfs(sample_cl[c], max_imfs=max_imfs) for c in range(C)]
    guide_imfs = []
    n_imfs_per_c = [len(src_imfs[c]) for c in range(C)]
    for g in guides:
        g_imfs = [_sift_imfs(g[c], max_imfs=max_imfs) for c in range(C)]
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
                # Preserves the original formula from emd_old (w_target applied twice)
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
        """Legacy IMF fingerprint distance using Welch histograms (emd_old style)."""
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
        if guide_class is None:
            target_mask = labels != label_orig
        else:
            target_mask = labels == int(guide_class)
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
        elif method == "coarse_to_fine":
            sw = _coarse_to_fine_step_weights(iter_idx)
        else:
            sw = _extreme_step_weights()
        for c in range(C):
            for k, wk in enumerate(weights[c]):
                delta = step * float(sw[c][k])
                wk["w_target"] = min(1.0, wk["w_target"] + delta)
                wk["w_source"] = max(0.0, wk["w_source"] - delta)

    # --- main interpolation loop --------------------------------------------
    cf = sample_cl.copy()
    scores_cf = scores_orig.copy()

    target_stop_class = None if guide_class is None else int(guide_class)

    for i in range(max_iter):
        _advance_weights(i)
        candidate = _reconstruct()

        scores_cand = _predict(candidate.reshape(1, C, L)).reshape(-1)
        label_cand = int(np.argmax(scores_cand))

        if verbose:
            print(
                f"[emd_cf] iter={i:3d}  guide={active_guide_idx} "
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
