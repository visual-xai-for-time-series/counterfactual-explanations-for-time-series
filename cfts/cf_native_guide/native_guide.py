import torch

import numpy as np

from captum.attr import GradientShap

from sklearn.neighbors import NearestNeighbors


def detach_to_numpy(data):
    # move pytorch data to cpu and detach it to numpy data
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    # convert numpy array to pytorch and move it to the device
    return torch.from_numpy(data).float().to(device)


####
# Instance-based Counterfactual Explanations for Time Series Classification
# (paper reference omitted for brevity)
####
def _ensure_ncl(sample, dataset):
    """Ensure sample and dataset are shaped (C, L) and (N, C, L) respectively.

    Heuristic: for 2D arrays, if rows <= cols treat as (C, L), else treat as
    (L, C) and transpose. This lets us cheaply detect already (N, C, L).
    """
    # normalize sample to (C, L)
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

    # build time_series_data as (N, C, L) with a single vectorized pass
    # dataset is expected to yield (x, y) tuples; take only x
    first = dataset[0][0]
    first_arr = np.asarray(first)
    # If first is already (N, C, L) (i.e., dataset provided as array), try to use it
    if first_arr.ndim == 3 and isinstance(dataset, np.ndarray):
        ts = np.asarray([x for x in dataset[:, 0]])
    else:
        # check orientation using the first element
        fa = first_arr
        if fa.ndim == 1:
            # each item is (L,) -> produce (N, 1, L)
            ts = np.stack([np.asarray(x[0]).reshape(1, -1) for x in dataset], axis=0)
        elif fa.ndim == 2:
            r, c = fa.shape
            if r <= c:
                # assume (C, L) already
                ts = np.stack([np.asarray(x[0]) for x in dataset], axis=0)
            else:
                # assume (L, C) and transpose each
                ts = np.stack([np.asarray(x[0]).T for x in dataset], axis=0)
        else:
            raise ValueError("dataset items must be 1D or 2D time series")

    # ensure same length as sample
    _, L = s_ncl.shape
    if ts.shape[-1] != L:
        raise ValueError("All series must have same length as sample")

    # if channel mismatch and dataset is single-channel, broadcast it
    C_sample = s_ncl.shape[0]
    C_data = ts.shape[1]
    if C_data != C_sample:
        if C_data == 1:
            ts = np.repeat(ts, C_sample, axis=1)
        else:
            raise ValueError("Channel count mismatch between sample and dataset")

    return s_ncl, ts, ori


def native_guide_uni_cf(sample, dataset, model, weight_function=GradientShap, iterate=None, sub_len=1, verbose=False):
    """Native Guide counterfactual supporting multivariate inputs.

    Returns counterfactual in same orientation as input sample, plus model scores.
    """
    device = next(model.parameters()).device

    def model_predict(arr):
        # arr expected shape (B, C, L)
        return detach_to_numpy(model(numpy_to_torch(arr, device)))

    # prepare sample and dataset in (C, L) and (N, C, L)
    sample_cf, time_series_data, sample_ori = _ensure_ncl(sample, dataset)
    N, C, L = time_series_data.shape

    if iterate is None:
        iterate = L

    # get predictions
    preds_data = model_predict(time_series_data)
    preds_sample = model_predict(sample_cf.reshape(1, C, L))
    label_data = np.argmax(preds_data, axis=1)
    label_sample = int(np.argmax(preds_sample))

    # select candidates with different label
    mask = label_data != label_sample
    if not np.any(mask):
        return (_revert_orientation if ' _revert_orientation' in globals() else _simple_revert)(sample_cf, sample_ori), preds_sample.reshape(-1)

    candidates = time_series_data[mask]
    candidates_labels = label_data[mask]

    # choose k neighbors (at least 1)
    k_for_candidates = max(1, min(int(L * 0.25), len(candidates)))
    neigh = NearestNeighbors(n_neighbors=min(k_for_candidates + 1, len(candidates)), metric="euclidean")
    neigh.fit(candidates.reshape(len(candidates), -1))
    dists, idxs = neigh.kneighbors(sample_cf.reshape(1, -1), return_distance=True)

    native_guide = None
    cf_label = None
    # find first neighbor with different class (skip exact matches)
    for idx in idxs[0]:
        if candidates_labels[idx] != label_sample:
            native_guide = candidates[idx]
            cf_label = int(candidates_labels[idx])
            break
    if native_guide is None:
        native_guide = candidates[0]
        cf_label = int(candidates_labels[0])

    # compute attributions using provided weight_function
    weights = weight_function(model)
    baselines = numpy_to_torch(time_series_data, device)
    attributions = weights.attribute(numpy_to_torch(native_guide.reshape(1, C, L), device),
                                     baselines=baselines,
                                     target=int(cf_label))
    attr_np = detach_to_numpy(attributions)
    # attr_np shape (1, C, L) typically
    if attr_np.ndim == 3:
        importance = np.sum(np.abs(attr_np[0]), axis=0)  # (L,)
    else:
        importance = np.sum(np.abs(attr_np), axis=0)

    # sliding-window sum to get most influential window start
    def find_most_influential_array(length):
        if length >= len(importance):
            return 0
        conv = np.convolve(importance, np.ones(length, dtype=importance.dtype), mode="valid")
        return int(np.argmax(conv))

    # iterative replacement
    cf_cf = sample_cf.copy()
    y_cf = preds_sample.reshape(-1)
    for i in range(iterate):
        length = i + sub_len
        if length > L:
            break
        start = find_most_influential_array(length)
        end = start + length
        cf_candidate = cf_cf.copy()
        cf_candidate[:, start:end] = native_guide[:, start:end]
        y_candidate = model_predict(cf_candidate.reshape(1, C, L)).reshape(-1)
        cf_cf = cf_candidate
        y_cf = y_candidate
        if cf_label == int(np.argmax(y_cf)):
            break

    # revert to original orientation
    cf_out = _revert_orientation(cf_cf, sample_ori) if ' _revert_orientation' in globals() else _simple_revert(cf_cf, sample_ori)
    return cf_out, y_cf


# simple revert used if helper not present in older file versions
def _simple_revert(cf_arr, orientation):
    if orientation == "1d":
        return cf_arr.reshape(-1)
    if orientation == "cf":
        return cf_arr
    if orientation == "tf":
        return cf_arr.T
    return cf_arr


def _revert_orientation(cf_arr, orientation):
    return _simple_revert(cf_arr, orientation)

