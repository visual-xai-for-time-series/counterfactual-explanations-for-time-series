import numpy as np
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

try:
    from tslearn.neighbors import KNeighborsTimeSeries
    _HAS_TSLEARN = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    _HAS_TSLEARN = False


####
# M-CELS: Counterfactual Explanation for Multivariate Time Series Guided by Learned Saliency Maps
#
# Paper: Li, P., Bahri, O., Boubrahimi, S. F., & Hamdi, S. M. (2024).
#        "M-CELS: Counterfactual Explanation for Multivariate Time Series Data
#         Guided by Learned Saliency Maps"
#        arXiv preprint arXiv:2411.02649
#
# Original repo: https://github.com/ModelOriented/mascots/tree/main/experiments/competitors/mcels
# Core implementation: nte/models/saliency_model/counterfactual_multi_ori.py
#
# Key algorithmic differences vs. the local cels.py M-CELS:
#
#  1. NUN retrieval uses KNeighborsTimeSeries (tslearn) on (C, L) shaped data,
#     not sklearn NearestNeighbors on flattened arrays.
#
#  2. Mask shape is (C, L) — same as the data — not a batched (1, C, L) tensor.
#     This means the mask covers both feature channels and time simultaneously.
#
#  3. Mask initialised from random uniform [0, 1], not gradient-based saliency.
#
#  4. TV norm is computed on the *flattened* mask (all elements concatenated),
#     measuring variation across both dimensions at once:
#         tv_norm = mean(|mask_flat[:-1] - mask_flat[1:]|^tv_beta)
#     The local version incorrectly splits temporal and feature smoothness.
#
#  5. Loss = l_max_coeff  * (1 - p_cf)
#           + l_budget_coeff * mean(|mask|)  [if enable_budget]
#           + l_tv_coeff    * tv_norm(mask)  [if enable_tvnorm]
#
#  6. Early stopping fires after 100 consecutive iterations without improvement
#     of at least 0.001 in total loss (local uses 30).
#
#  7. cf_label is *always* the second-most-probable class from the model —
#     it is not a user parameter (the local version exposes `target`).
#
#  8. Post-processing: values in the optimised mask below 0.5 are zeroed before
#     producing the final counterfactual.
####


def _tv_norm(mask_tensor, tv_beta):
    """Total variation norm on the flattened mask (matches original implementation)."""
    flat = mask_tensor.reshape(-1)
    return torch.mean(torch.pow(torch.abs(flat[:-1] - flat[1:]), tv_beta))


def _nun_retrieval_tslearn(background_data, background_labels, query, cf_label):
    """Retrieve the nearest unlike neighbor using KNeighborsTimeSeries.

    background_data : (N, C, L)
    query           : (C, L)
    cf_label        : int
    """
    target_idx = [i for i, lbl in enumerate(background_labels) if lbl == cf_label]
    if not target_idx:
        raise ValueError(f"No background samples with label {cf_label}")
    candidates = background_data[target_idx]  # (K, C, L)

    knn = KNeighborsTimeSeries(n_neighbors=1, metric="euclidean")
    knn.fit(candidates)
    _, ind = knn.kneighbors(query.reshape(1, *query.shape), return_distance=True)
    return candidates[ind[0][0]]


def _nun_retrieval_sklearn(background_data, background_labels, query, cf_label):
    """Fallback NUN retrieval using sklearn (flattened arrays)."""
    target_idx = [i for i, lbl in enumerate(background_labels) if lbl == cf_label]
    if not target_idx:
        raise ValueError(f"No background samples with label {cf_label}")
    candidates = background_data[target_idx]
    candidates_flat = candidates.reshape(len(candidates), -1)
    query_flat = query.reshape(1, -1)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(candidates_flat)
    _, ind = nn.kneighbors(query_flat)
    return candidates[ind[0][0]]


def _retrieve_nun(background_data, background_labels, query, cf_label):
    if _HAS_TSLEARN:
        return _nun_retrieval_tslearn(background_data, background_labels, query, cf_label)
    return _nun_retrieval_sklearn(background_data, background_labels, query, cf_label)


def mcels_generate(
    sample,
    model,
    background_data,
    background_labels,
    lr=1e-3,
    max_itr=1000,
    enable_lr_decay=True,
    lr_decay=0.9991,
    enable_budget=True,
    l_budget_coeff=1.0,
    enable_tvnorm=True,
    l_tv_norm_coeff=1.0,
    l_max_coeff=1.0,
    tv_beta=2,
    random_state=None,
    verbose=False,
):
    """Generate a counterfactual explanation for a multivariate time series.

    Faithfully reimplements the original M-CELS algorithm from
    nte/models/saliency_model/counterfactual_multi_ori.py.

    Parameters
    ----------
    sample : np.ndarray, shape (C, L)
        Input multivariate time series (channels × timesteps).
    model : callable
        PyTorch model accepting (1, C, L) tensors, returning (1, n_classes) logits.
    background_data : np.ndarray, shape (N, C, L)
        Dataset used for NUN retrieval.
    background_labels : array-like, shape (N,)
        Integer class labels for `background_data`.
    lr : float
        Adam learning rate.
    max_itr : int
        Maximum number of gradient steps (loop runs while i <= max_itr).
    enable_lr_decay : bool
        Apply ExponentialLR decay to the Adam optimizer.
    lr_decay : float
        Multiplicative factor per step for LR decay.
    enable_budget : bool
        Include the L1 budget (sparsity) term in the loss.
    l_budget_coeff : float
        Coefficient for the budget loss term.
    enable_tvnorm : bool
        Include the TV-norm (smoothness) term in the loss.
    l_tv_norm_coeff : float
        Coefficient for the TV-norm loss term.
    l_max_coeff : float
        Coefficient for the prediction margin loss term.
    tv_beta : float
        Exponent for the TV norm (2 = squared differences).
    random_state : int, optional
        Random seed.
    verbose : bool
        Print progress information.

    Returns
    -------
    mask : np.ndarray, shape (C, L)
        Final thresholded saliency mask (values below 0.5 set to 0).
    cf : np.ndarray, shape (C, L)
        Counterfactual time series.
    target_prob : float
        Model probability assigned to the target class for the CF.
    cf_label : int
        The target class (second-most probable class of the original sample).
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)  # treat univariate as (1, L)

    C, L = sample.shape
    background_labels = np.asarray(background_labels).flatten().astype(int)

    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")

    softmax = torch.nn.Softmax(dim=-1)

    def predict(arr_cl):
        t = torch.tensor(arr_cl, dtype=torch.float32, device=device).reshape(1, C, L)
        with torch.no_grad():
            return softmax(model(t)).cpu().numpy()[0]

    # Determine cf_label: second-most probable class (mirrors original)
    orig_probs = predict(sample)
    cf_label = int(np.argsort(orig_probs)[::-1][1])
    if verbose:
        print(f"M-CELS: original class={np.argmax(orig_probs)}, cf_label={cf_label}")

    # Retrieve Nearest Unlike Neighbor
    nun = _retrieve_nun(background_data, background_labels, sample, cf_label)
    nun = np.asarray(nun, dtype=np.float32).reshape(C, L)
    if verbose:
        print("M-CELS: NUN retrieved")

    # Tensors (no batch dimension — matches original)
    data_t = torch.tensor(sample, dtype=torch.float32, device=device)   # (C, L)
    nun_t  = torch.tensor(nun,    dtype=torch.float32, device=device)   # (C, L)

    # Mask: shape (C, L), random uniform init
    mask_init = np.random.uniform(size=(C, L), low=0.0, high=1.0).astype(np.float32)
    mask = Variable(torch.from_numpy(mask_init).to(device), requires_grad=True)

    optimizer = torch.optim.Adam([mask], lr=lr)
    if enable_lr_decay:
        scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    best_loss = float("inf")
    counter = 0
    max_no_improve = 100  # original uses 100, not 30
    imp_threshold = 0.001

    i = 0
    while i <= max_itr:
        # Perturbation: cf = data * (1 - mask) + NUN * mask
        cf_t = data_t * (1 - mask) + nun_t * mask  # (C, L)

        pred = softmax(model(cf_t.reshape(1, C, L)))  # (1, n_classes)

        l_maximize   = 1.0 - pred[0, cf_label]
        l_budget     = torch.mean(torch.abs(mask)) * float(enable_budget)
        l_tv         = _tv_norm(mask, tv_beta)     * float(enable_tvnorm)

        loss = (l_max_coeff * l_maximize
                + l_budget_coeff * l_budget
                + l_tv_norm_coeff * l_tv)

        if best_loss - float(loss.item()) < imp_threshold:
            counter += 1
        else:
            counter = 0
            best_loss = float(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if enable_lr_decay:
            scheduler.step()

        mask.data.clamp_(0.0, 1.0)

        if verbose and i % 100 == 0:
            print(f"M-CELS iter {i}: cf_prob={float(pred[0, cf_label].item()):.4f}, "
                  f"loss={float(loss.item()):.4f}")

        if counter >= max_no_improve:
            if verbose:
                print(f"M-CELS: early stopping at iter {i}")
            break

        i += 1

    # Post-processing: threshold mask at 0.5, zero below threshold
    mask_np = mask.cpu().detach().numpy()          # (C, L)
    thresholded = np.where(mask_np >= 0.5, mask_np, 0.0)

    thresholded_t = torch.tensor(thresholded, dtype=torch.float32, device=device)
    cf_final_t = data_t * (1 - thresholded_t) + nun_t * thresholded_t

    with torch.no_grad():
        pred_final = softmax(model(cf_final_t.reshape(1, C, L)))
    target_prob = float(pred_final[0, cf_label].item())

    cf_final = cf_final_t.cpu().detach().numpy()   # (C, L)

    if verbose:
        final_class = int(np.argmax(pred_final.cpu().numpy()[0]))
        print(f"M-CELS: final class={final_class}, target={cf_label}, "
              f"success={final_class == cf_label}, target_prob={target_prob:.4f}")

    return thresholded, cf_final, target_prob, cf_label
