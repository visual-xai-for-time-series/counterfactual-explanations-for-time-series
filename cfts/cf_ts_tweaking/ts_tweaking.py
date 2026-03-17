import torch

import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def detach_to_numpy(data):
    # move pytorch data to cpu and detach it to numpy data
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    # convert numpy array to pytorch and move it to the device
    return torch.from_numpy(data).float().to(device)


####
# Time Series Tweaking: Locally and Globally Explainable Time Series Tweaking
#
# Paper: Karlsson, I., Rebane, J., Papapetrou, P. & Gionis, A. (2020).
#        "Locally and globally explainable time series tweaking."
#        Knowledge and Information Systems, 62, 1671-1700
#
# Paper URL: https://link.springer.com/article/10.1007/s10115-019-01389-4
# ArXiv (earlier version): https://arxiv.org/abs/1809.05183
#
# This module implements three counterfactual explanation algorithms for
# time series classification:
#
#   1. Global Tweaking (τ_NN) - Algorithm 1
#      Uses k-NN classifier with k-means clustering to find the minimal
#      transformation that changes the classifier's prediction globally.
#
#   2. Local Irreversible Tweaking (τ_SF) - Algorithm 2
#      Uses a shapelet-based approach with a fitted shapelet forest to find
#      local subsequence modifications that flip the prediction by pushing
#      subsequences past the shapelet distance threshold.
#
#   3. Local Reversible Tweaking (τ_SF-R) - Algorithm 3
#      Similar to irreversible, but constrains the modification to stay
#      within the distance threshold sphere, producing more conservative
#      (reversible) transformations.
####


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_ncl(sample, dataset):
    """Ensure sample and dataset are shaped (C, L) and (N, C, L) respectively.

    Heuristic: for 2D arrays, if rows <= cols treat as (C, L), else treat as
    (L, C) and transpose.
    """
    s = np.asarray(sample)
    if s.ndim == 1:
        s_ncl = s.reshape(1, -1)
        ori = "1d"
    elif s.ndim == 2:
        r, c = s.shape
        if r <= c:
            s_ncl = s.copy()
            ori = "cf"
        else:
            s_ncl = s.T.copy()
            ori = "tf"
    else:
        raise ValueError("sample must be 1D or 2D time series")

    first = dataset[0][0]
    first_arr = np.asarray(first)

    if first_arr.ndim == 3 and isinstance(dataset, np.ndarray):
        ts = np.asarray([x for x in dataset[:, 0]])
    else:
        fa = first_arr
        if fa.ndim == 1:
            ts = np.stack([np.asarray(x[0]).reshape(1, -1) for x in dataset], axis=0)
        elif fa.ndim == 2:
            r, c = fa.shape
            if r <= c:
                ts = np.stack([np.asarray(x[0]) for x in dataset], axis=0)
            else:
                ts = np.stack([np.asarray(x[0]).T for x in dataset], axis=0)
        else:
            raise ValueError("dataset items must be 1D or 2D time series")

    _, L = s_ncl.shape
    if ts.shape[-1] != L:
        raise ValueError("All series must have same length as sample")

    C_sample = s_ncl.shape[0]
    C_data = ts.shape[1]
    if C_data != C_sample:
        if C_data == 1:
            ts = np.repeat(ts, C_sample, axis=1)
        else:
            raise ValueError("Channel count mismatch between sample and dataset")

    return s_ncl, ts, ori


def _simple_revert(cf_arr, orientation):
    """Revert counterfactual array back to the original sample orientation."""
    if orientation == "1d":
        return cf_arr.reshape(-1)
    if orientation == "cf":
        return cf_arr
    if orientation == "tf":
        return cf_arr.T
    return cf_arr


def _euclidean_distance(a, b):
    """Euclidean distance between two arrays (flattened)."""
    return np.sqrt(np.sum((a.ravel() - b.ravel()) ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm 1: Global Time Series Tweaking (τ_NN)
#
# Given a time series T classified as class c by a k-NN classifier, find the
# minimal transformation so that the k-NN classifier reclassifies T as a
# desired target class y.
#
# Approach:
#   1. Cluster the training data per class using k-means.
#   2. For the target class, select centroids whose nearest training series
#      are predominantly labeled as the target class.
#   3. Transform T by interpolating it toward the selected centroids
#      (weighted by inverse distance) until the k-NN prediction flips.
# ──────────────────────────────────────────────────────────────────────────────

def ts_tweaking_knn_cf(sample, dataset, model, target=None, k=5, n_clusters=5,
                       alpha_steps=20, verbose=False):
    """Global time series tweaking (Algorithm 1, τ_NN) from Karlsson et al.

    Generates a counterfactual by interpolating the query time series toward
    cluster centroids of the target class in the training set, then verifying
    the prediction flip via the provided black-box model.

    Parameters
    ----------
    sample : array-like, shape (L,) or (C, L)
        The query time series to explain.
    dataset : list of (x, y) tuples
        Training data where x is a time series and y is its label.
    model : torch.nn.Module
        A trained PyTorch classifier. Must accept input of shape (B, C, L).
    target : int or None
        Desired target class. If None, uses the nearest unlike neighbor's class.
    k : int
        Number of nearest neighbors for the k-NN logic (default 5).
    n_clusters : int
        Number of k-means clusters per class (default 5).
    alpha_steps : int
        Number of interpolation steps to try between 0 and 1 (default 20).
    verbose : bool
        If True, print progress information.

    Returns
    -------
    cf : np.ndarray
        The counterfactual time series (same shape as input sample).
    scores : np.ndarray
        The model's output scores/probabilities for the counterfactual.
    """
    device = next(model.parameters()).device

    def model_predict(arr):
        with torch.no_grad():
            return detach_to_numpy(model(numpy_to_torch(arr, device)))

    # --- Prepare data in (C, L) and (N, C, L) format ---
    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    N, C, L = time_series_data.shape

    labels = np.array([int(np.argmax(d[1])) if hasattr(d[1], '__len__') else int(d[1]) for d in dataset])
    unique_classes = np.unique(labels)

    # Predict the query's current class
    preds_sample = model_predict(sample_cf.reshape(1, C, L))
    label_sample = int(np.argmax(preds_sample))

    # Determine target class
    if target is None:
        # Use the nearest unlike neighbor (NUN) class
        mask_diff = labels != label_sample
        if not np.any(mask_diff):
            return _simple_revert(sample_cf, sample_ori), preds_sample.reshape(-1)
        candidates = time_series_data[mask_diff]
        candidate_labels = labels[mask_diff]
        neigh = NearestNeighbors(n_neighbors=1, metric="euclidean")
        neigh.fit(candidates.reshape(len(candidates), -1))
        _, idxs = neigh.kneighbors(sample_cf.reshape(1, -1))
        target = int(candidate_labels[idxs[0, 0]])

    if label_sample == target:
        if verbose:
            print("Sample already classified as target class.")
        return _simple_revert(sample_cf, sample_ori), preds_sample.reshape(-1)

    # --- Step 1: Cluster training data of the target class ---
    mask_target = labels == target
    target_data = time_series_data[mask_target]  # (N_target, C, L)
    n_target = len(target_data)

    if n_target == 0:
        if verbose:
            print(f"No training samples for target class {target}.")
        return _simple_revert(sample_cf, sample_ori), preds_sample.reshape(-1)

    actual_n_clusters = min(n_clusters, n_target)
    target_flat = target_data.reshape(n_target, -1)  # (N_target, C*L)

    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
    kmeans.fit(target_flat)
    centroids = kmeans.cluster_centers_.reshape(actual_n_clusters, C, L)

    # --- Step 2: Select centroids where majority of nearest training
    #             series have the target label ---
    # For each centroid, find its k nearest training series (from ALL data)
    # and check if the majority are labeled as the target class.
    all_flat = time_series_data.reshape(N, -1)
    k_check = min(k, N)
    neigh_all = NearestNeighbors(n_neighbors=k_check, metric="euclidean")
    neigh_all.fit(all_flat)

    valid_centroids = []
    for ci in range(actual_n_clusters):
        centroid_flat = centroids[ci].reshape(1, -1)
        _, nn_idxs = neigh_all.kneighbors(centroid_flat)
        nn_labels = labels[nn_idxs[0]]
        majority_target = np.sum(nn_labels == target) > k_check / 2
        if majority_target:
            valid_centroids.append(centroids[ci])

    if len(valid_centroids) == 0:
        # Fallback: use all centroids
        valid_centroids = [centroids[ci] for ci in range(actual_n_clusters)]

    valid_centroids = np.array(valid_centroids)  # (M, C, L)

    # --- Step 3: Transform T toward valid centroids ---
    # Compute the weighted average centroid (weighted by inverse distance)
    distances = np.array([
        _euclidean_distance(sample_cf, vc) for vc in valid_centroids
    ])
    # Avoid division by zero
    distances = np.maximum(distances, 1e-10)
    weights = 1.0 / distances
    weights = weights / weights.sum()

    # Weighted target direction
    target_direction = np.zeros_like(sample_cf)
    for w, vc in zip(weights, valid_centroids):
        target_direction += w * vc
    # target_direction is now the weighted centroid we're moving toward

    # Interpolate from sample toward target_direction with increasing alpha
    best_cf = sample_cf.copy()
    best_scores = preds_sample.reshape(-1)
    best_cost = float("inf")

    for step in range(1, alpha_steps + 1):
        alpha = step / alpha_steps
        cf_candidate = (1 - alpha) * sample_cf + alpha * target_direction
        y_candidate = model_predict(cf_candidate.reshape(1, C, L)).reshape(-1)
        pred_class = int(np.argmax(y_candidate))

        if pred_class == target:
            cost = _euclidean_distance(sample_cf, cf_candidate)
            if cost < best_cost:
                best_cf = cf_candidate.copy()
                best_scores = y_candidate.copy()
                best_cost = cost
            # Found a successful tweak — keep looking for lower cost,
            # but the first success at smallest alpha is typically optimal
            break

    if verbose:
        pred_final = int(np.argmax(best_scores))
        status = "SUCCESS" if pred_final == target else "FAILED"
        print(f"[τ_NN] {status}: original={label_sample}, target={target}, "
              f"predicted={pred_final}, cost={best_cost:.4f}")

    return _simple_revert(best_cf, sample_ori), best_scores


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm 2: Local Irreversible Time Series Tweaking (τ_SF)
#
# Given a shapelet forest classifier, find the minimal subsequence
# modifications that flip the prediction. For each discriminative shapelet,
# modify the corresponding subsequence of T to push it past (or pull it
# within) the distance threshold, so the tree traversal reaches a leaf
# with the target class.
#
# Since we use a black-box PyTorch model rather than a shapelet forest
# directly, we approximate this by:
#   1. Extracting discriminative subsequences (shapelets) from the training
#      data of the target class.
#   2. Finding the best-matching location in the query for each shapelet.
#   3. Replacing the query's subsequence with the target shapelet
#      (irreversible: full replacement past the threshold).
#   4. Greedily applying the replacement that flips the prediction with
#      minimal cost.
# ──────────────────────────────────────────────────────────────────────────────

def _extract_shapelets(time_series_data, labels, target_class, n_shapelets=10,
                       min_len=3, max_len_ratio=0.5, random_state=42):
    """Extract candidate shapelets from target class training data.

    Uses random sampling of subsequences (as in Random Shapelet Forests).
    Returns a list of (shapelet, channel_idx) tuples.
    """
    rng = np.random.RandomState(random_state)
    mask = labels == target_class
    target_data = time_series_data[mask]  # (N_t, C, L)

    if len(target_data) == 0:
        return []

    N_t, C, L = target_data.shape
    max_len = max(min_len + 1, int(L * max_len_ratio))

    shapelets = []
    for _ in range(n_shapelets):
        # Random sample, channel, length, start
        idx = rng.randint(N_t)
        ch = rng.randint(C)
        s_len = rng.randint(min_len, max_len + 1)
        s_len = min(s_len, L)
        start = rng.randint(0, L - s_len + 1)
        shapelet = target_data[idx, ch, start:start + s_len].copy()
        shapelets.append((shapelet, ch))

    return shapelets


def _find_best_match(series_channel, shapelet):
    """Find the starting index in series_channel that best matches the shapelet.

    Returns (best_start_idx, best_distance).
    """
    s_len = len(shapelet)
    L = len(series_channel)
    if s_len > L:
        return 0, float("inf")

    best_dist = float("inf")
    best_idx = 0
    for i in range(L - s_len + 1):
        subseq = series_channel[i:i + s_len]
        dist = np.sqrt(np.sum((subseq - shapelet) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx, best_dist


def ts_tweaking_irreversible_cf(sample, dataset, model, target=None,
                                n_shapelets=20, min_shapelet_len=3,
                                max_shapelet_ratio=0.5, random_state=42,
                                verbose=False):
    """Local irreversible time series tweaking (Algorithm 2, τ_SF).

    Generates a counterfactual by replacing subsequences in the query with
    discriminative shapelets from the target class. The replacement fully
    substitutes the original subsequence (irreversible tweak).

    Parameters
    ----------
    sample : array-like, shape (L,) or (C, L)
        The query time series to explain.
    dataset : list of (x, y) tuples
        Training data where x is a time series and y is its label.
    model : torch.nn.Module
        A trained PyTorch classifier.
    target : int or None
        Desired target class. If None, uses the nearest unlike neighbor's class.
    n_shapelets : int
        Number of candidate shapelets to extract (default 20).
    min_shapelet_len : int
        Minimum shapelet length (default 3).
    max_shapelet_ratio : float
        Maximum shapelet length as fraction of series length (default 0.5).
    random_state : int
        Random seed for shapelet extraction.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    cf : np.ndarray
        The counterfactual time series (same shape as input sample).
    scores : np.ndarray
        The model's output scores/probabilities for the counterfactual.
    """
    device = next(model.parameters()).device

    def model_predict(arr):
        with torch.no_grad():
            return detach_to_numpy(model(numpy_to_torch(arr, device)))

    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    N, C, L = time_series_data.shape
    labels = np.array([int(np.argmax(d[1])) if hasattr(d[1], '__len__') else int(d[1]) for d in dataset])

    preds_sample = model_predict(sample_cf.reshape(1, C, L))
    label_sample = int(np.argmax(preds_sample))

    # Determine target class
    if target is None:
        mask_diff = labels != label_sample
        if not np.any(mask_diff):
            return _simple_revert(sample_cf, sample_ori), preds_sample.reshape(-1)
        candidates = time_series_data[mask_diff]
        candidate_labels = labels[mask_diff]
        neigh = NearestNeighbors(n_neighbors=1, metric="euclidean")
        neigh.fit(candidates.reshape(len(candidates), -1))
        _, idxs = neigh.kneighbors(sample_cf.reshape(1, -1))
        target = int(candidate_labels[idxs[0, 0]])

    if label_sample == target:
        if verbose:
            print("Sample already classified as target class.")
        return _simple_revert(sample_cf, sample_ori), preds_sample.reshape(-1)

    # --- Extract candidate shapelets from target class ---
    shapelets = _extract_shapelets(
        time_series_data, labels, target,
        n_shapelets=n_shapelets,
        min_len=min_shapelet_len,
        max_len_ratio=max_shapelet_ratio,
        random_state=random_state,
    )

    if len(shapelets) == 0:
        if verbose:
            print(f"No shapelets found for target class {target}.")
        return _simple_revert(sample_cf, sample_ori), preds_sample.reshape(-1)

    # --- Rank shapelets by how well they match (lowest distance = most
    #     discriminative location to modify) ---
    shapelet_matches = []
    for shapelet, ch in shapelets:
        best_idx, best_dist = _find_best_match(sample_cf[ch], shapelet)
        shapelet_matches.append((shapelet, ch, best_idx, best_dist))

    # Sort by distance (smallest first — closest match locations are where
    # the time series is most similar to the shapelet, so replacing there
    # produces the smallest perturbation)
    shapelet_matches.sort(key=lambda x: x[3])

    # --- Greedily apply shapelet replacements ---
    cf_current = sample_cf.copy()
    best_cf = sample_cf.copy()
    best_scores = preds_sample.reshape(-1)
    best_cost = float("inf")
    found = False

    for shapelet, ch, start_idx, _ in shapelet_matches:
        s_len = len(shapelet)
        end_idx = start_idx + s_len

        # Irreversible: full replacement of the subsequence
        cf_candidate = cf_current.copy()
        cf_candidate[ch, start_idx:end_idx] = shapelet

        y_candidate = model_predict(cf_candidate.reshape(1, C, L)).reshape(-1)
        pred_class = int(np.argmax(y_candidate))

        cost = _euclidean_distance(sample_cf, cf_candidate)

        if pred_class == target:
            if cost < best_cost:
                best_cf = cf_candidate.copy()
                best_scores = y_candidate.copy()
                best_cost = cost
                found = True
            break  # First successful greedy replacement
        else:
            # Accumulate: keep this replacement and try adding more
            cf_current = cf_candidate.copy()

    # If greedy accumulation didn't work, try each shapelet independently
    if not found:
        for shapelet, ch, start_idx, _ in shapelet_matches:
            s_len = len(shapelet)
            end_idx = start_idx + s_len

            cf_candidate = sample_cf.copy()
            cf_candidate[ch, start_idx:end_idx] = shapelet

            y_candidate = model_predict(cf_candidate.reshape(1, C, L)).reshape(-1)
            pred_class = int(np.argmax(y_candidate))

            if pred_class == target:
                cost = _euclidean_distance(sample_cf, cf_candidate)
                if cost < best_cost:
                    best_cf = cf_candidate.copy()
                    best_scores = y_candidate.copy()
                    best_cost = cost
                    found = True

    # Last resort: use the accumulated CF even if it didn't fully flip
    if not found:
        y_current = model_predict(cf_current.reshape(1, C, L)).reshape(-1)
        best_cf = cf_current.copy()
        best_scores = y_current.copy()

    if verbose:
        pred_final = int(np.argmax(best_scores))
        status = "SUCCESS" if pred_final == target else "FAILED"
        print(f"[τ_SF] {status}: original={label_sample}, target={target}, "
              f"predicted={pred_final}, cost={best_cost:.4f}")

    return _simple_revert(best_cf, sample_ori), best_scores


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm 3: Local Reversible Time Series Tweaking (τ_SF-R)
#
# Same as irreversible, but instead of fully replacing the subsequence
# with the shapelet, we interpolate between the original subsequence and
# the shapelet. We find the minimal interpolation weight (alpha) that
# flips the prediction, producing a more conservative counterfactual.
#
# Geometrically, this keeps the modified subsequence within the
# m-sphere defined by the shapelet and the distance threshold,
# rather than pushing it all the way to the shapelet center.
# ──────────────────────────────────────────────────────────────────────────────

def ts_tweaking_reversible_cf(sample, dataset, model, target=None,
                              n_shapelets=20, min_shapelet_len=3,
                              max_shapelet_ratio=0.5, alpha_steps=20,
                              random_state=42, verbose=False):
    """Local reversible time series tweaking (Algorithm 3, τ_SF-R).

    Generates a counterfactual by interpolating subsequences in the query
    toward discriminative shapelets from the target class. Uses the minimal
    interpolation weight (alpha) that flips the prediction (reversible tweak).

    Parameters
    ----------
    sample : array-like, shape (L,) or (C, L)
        The query time series to explain.
    dataset : list of (x, y) tuples
        Training data where x is a time series and y is its label.
    model : torch.nn.Module
        A trained PyTorch classifier.
    target : int or None
        Desired target class. If None, uses the nearest unlike neighbor's class.
    n_shapelets : int
        Number of candidate shapelets to extract (default 20).
    min_shapelet_len : int
        Minimum shapelet length (default 3).
    max_shapelet_ratio : float
        Maximum shapelet length as fraction of series length (default 0.5).
    alpha_steps : int
        Number of interpolation steps between 0 and 1 (default 20).
    random_state : int
        Random seed for shapelet extraction.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    cf : np.ndarray
        The counterfactual time series (same shape as input sample).
    scores : np.ndarray
        The model's output scores/probabilities for the counterfactual.
    """
    device = next(model.parameters()).device

    def model_predict(arr):
        with torch.no_grad():
            return detach_to_numpy(model(numpy_to_torch(arr, device)))

    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    N, C, L = time_series_data.shape
    labels = np.array([int(np.argmax(d[1])) if hasattr(d[1], '__len__') else int(d[1]) for d in dataset])

    preds_sample = model_predict(sample_cf.reshape(1, C, L))
    label_sample = int(np.argmax(preds_sample))

    # Determine target class
    if target is None:
        mask_diff = labels != label_sample
        if not np.any(mask_diff):
            return _simple_revert(sample_cf, sample_ori), preds_sample.reshape(-1)
        candidates = time_series_data[mask_diff]
        candidate_labels = labels[mask_diff]
        neigh = NearestNeighbors(n_neighbors=1, metric="euclidean")
        neigh.fit(candidates.reshape(len(candidates), -1))
        _, idxs = neigh.kneighbors(sample_cf.reshape(1, -1))
        target = int(candidate_labels[idxs[0, 0]])

    if label_sample == target:
        if verbose:
            print("Sample already classified as target class.")
        return _simple_revert(sample_cf, sample_ori), preds_sample.reshape(-1)

    # --- Extract candidate shapelets from target class ---
    shapelets = _extract_shapelets(
        time_series_data, labels, target,
        n_shapelets=n_shapelets,
        min_len=min_shapelet_len,
        max_len_ratio=max_shapelet_ratio,
        random_state=random_state,
    )

    if len(shapelets) == 0:
        if verbose:
            print(f"No shapelets found for target class {target}.")
        return _simple_revert(sample_cf, sample_ori), preds_sample.reshape(-1)

    # --- Rank shapelets by match distance ---
    shapelet_matches = []
    for shapelet, ch in shapelets:
        best_idx, best_dist = _find_best_match(sample_cf[ch], shapelet)
        shapelet_matches.append((shapelet, ch, best_idx, best_dist))

    shapelet_matches.sort(key=lambda x: x[3])

    # --- For each shapelet, try increasing alpha until prediction flips ---
    best_cf = sample_cf.copy()
    best_scores = preds_sample.reshape(-1)
    best_cost = float("inf")
    found = False

    for shapelet, ch, start_idx, _ in shapelet_matches:
        s_len = len(shapelet)
        end_idx = start_idx + s_len
        original_subseq = sample_cf[ch, start_idx:end_idx].copy()

        for step in range(1, alpha_steps + 1):
            alpha = step / alpha_steps

            # Reversible: interpolate between original and shapelet
            cf_candidate = sample_cf.copy()
            cf_candidate[ch, start_idx:end_idx] = (
                (1 - alpha) * original_subseq + alpha * shapelet
            )

            y_candidate = model_predict(cf_candidate.reshape(1, C, L)).reshape(-1)
            pred_class = int(np.argmax(y_candidate))

            if pred_class == target:
                cost = _euclidean_distance(sample_cf, cf_candidate)
                if cost < best_cost:
                    best_cf = cf_candidate.copy()
                    best_scores = y_candidate.copy()
                    best_cost = cost
                    found = True
                break  # Found minimal alpha for this shapelet

    # If single-shapelet interpolation didn't work, try combining shapelets
    if not found:
        cf_current = sample_cf.copy()
        for shapelet, ch, start_idx, _ in shapelet_matches:
            s_len = len(shapelet)
            end_idx = start_idx + s_len
            original_subseq = cf_current[ch, start_idx:end_idx].copy()

            for step in range(1, alpha_steps + 1):
                alpha = step / alpha_steps
                cf_candidate = cf_current.copy()
                cf_candidate[ch, start_idx:end_idx] = (
                    (1 - alpha) * original_subseq + alpha * shapelet
                )
                y_candidate = model_predict(cf_candidate.reshape(1, C, L)).reshape(-1)
                pred_class = int(np.argmax(y_candidate))

                if pred_class == target:
                    cost = _euclidean_distance(sample_cf, cf_candidate)
                    if cost < best_cost:
                        best_cf = cf_candidate.copy()
                        best_scores = y_candidate.copy()
                        best_cost = cost
                        found = True
                    break

            if found:
                break

            # Apply full interpolation for this shapelet and move on
            cf_current[ch, start_idx:end_idx] = (
                0.5 * original_subseq + 0.5 * shapelet
            )

    # Last resort: use accumulated result
    if not found:
        y_current = model_predict(cf_current.reshape(1, C, L)).reshape(-1)
        best_cf = cf_current.copy()
        best_scores = y_current.copy()

    if verbose:
        pred_final = int(np.argmax(best_scores))
        status = "SUCCESS" if pred_final == target else "FAILED"
        print(f"[τ_SF-R] {status}: original={label_sample}, target={target}, "
              f"predicted={pred_final}, cost={best_cost:.4f}")

    return _simple_revert(best_cf, sample_ori), best_scores
