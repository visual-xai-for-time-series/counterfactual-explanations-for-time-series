import numpy as np
import torch
import torch.optim as optim


####
# GLACIER: Guided Locally Constrained Counterfactual Explanations
#
# This is a faithful PyTorch reimplementation of the original TensorFlow/Keras
# implementation from:
#   Wang, Z., Samsten, I., Miliou, I., Mochaourab, R., & Papapetrou, P. (2024).
#   "Glacier: Guided locally constrained counterfactual explanations for time series classification."
#   Machine Learning, Springer
#
# Original repo: https://github.com/ModelOriented/mascots/tree/main/experiments/competitors/glacier/src
#
# Key algorithmic differences vs the local glacier.py:
#   - Loss = pred_margin_weight * MSE(pred_prob, target_prob)
#           + (1 - pred_margin_weight) * weighted_MAE(cf, original, step_weights)
#     (original uses MSE-to-probability + weighted MAE, not cross-entropy + L2 + L1)
#   - step_weights: "local" uses LIMESegment to zero out unimportant timesteps,
#     "uniform" uses ones, "unconstrained" uses zeros (no proximity term)
#   - Stopping condition: predicted probability >= probability threshold (not argmax)
#   - Binary classification only (target label = 1 - pred_label)
#   - Optional autoencoder: search is performed in latent space when provided
####


def _to_torch(arr, device):
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)


def _to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def _pred_margin_mse(pred_prob, target_prob_tensor):
    """MSE between predicted probability and desired probability threshold."""
    return torch.mean((pred_prob - target_prob_tensor) ** 2)


def _weighted_mae(original, cf, step_weights):
    """Weighted MAE between original and counterfactual.

    step_weights: tensor of shape matching (original - cf), values in {0, 1}.
    Zero weights mask out timesteps that are locally unimportant (from LIMESegment).
    """
    return torch.mean(step_weights * torch.abs(original - cf))


def _get_lime_step_weights(sample_np, model_fn, n_timesteps, n_segments=10):
    """Approximate local step weights using segment occlusion (LIMESegment proxy).

    Occludes each of `n_segments` equal-length segments with the segment mean,
    measures prediction drop, and zeros out the least-informative 25% of timesteps.
    This mirrors the original's LIMESegment approach without requiring the
    LIMESegment package.

    sample_np : (L,) or (1, L) numpy array
    model_fn  : callable (1, L) -> scalar probability for target class
    """
    sample_1d = sample_np.reshape(-1)
    L = len(sample_1d)
    seg_len = max(1, L // n_segments)
    seg_boundaries = list(range(0, L, seg_len)) + [L]

    base_prob = float(model_fn(sample_1d.reshape(1, -1)))
    importances = np.zeros(n_segments)

    for i in range(n_segments):
        start, end = seg_boundaries[i], seg_boundaries[i + 1]
        masked = sample_1d.copy()
        masked[start:end] = masked[start:end].mean()
        importances[i] = base_prob - float(model_fn(masked.reshape(1, -1)))

    threshold = np.percentile(importances, 25)
    weights = np.ones(L, dtype=np.float32)
    for i in range(n_segments):
        if importances[i] <= threshold:
            start, end = seg_boundaries[i], seg_boundaries[i + 1]
            weights[start:end] = 0.0

    return weights.reshape(1, -1)


def compute_global_step_weights(X_train, y_train, predict_label_fn, n_timesteps, n_segments=10, max_samples=200):
    """Approximate bake-off's "global" step weights (occlusion proxy for
    `wildboar.explain.IntervalImportance`, avoiding the extra dependency).

    Bake-off's `get_global_weights` fits `IntervalImportance` **once, on the
    whole training set**, unlike `_get_lime_step_weights`'s per-sample
    "local" importance — and, notably, frees the *most* important 25% of
    segments (`masking_idx = where(importance >= 75th percentile)`) rather
    than the least important 25% that "local" frees. The intuition: the
    globally most class-discriminative regions are exactly where a change
    is needed to flip the prediction, so those are the ones opened up for
    perturbation; everything else stays constrained (weight 1) to preserve
    the rest of the signal.

    This proxy measures the same thing "local" does (occlude a segment with
    its own mean, look at the resulting prediction change) but aggregated
    as a classification-**accuracy** drop across `X_train`/`y_train` instead
    of a probability drop for one sample, matching what interval importance
    over a whole training set actually measures.

    Because this only depends on the training set (not the query), it is
    query-independent — compute it once and pass the resulting array
    directly as `step_weights=` to avoid recomputing it per call (the same
    caching argument this repo's `sg_cf_fast`/`timex_cf`'s `prototype=` make
    for their own query-independent setup work).

    Parameters
    ----------
    X_train : (N, L) array of training series.
    y_train : (N,) array of integer class labels.
    predict_label_fn : callable, (M, L) array -> (M,) predicted class labels.
    n_timesteps : L.
    n_segments : number of equal-length segments to score (bake-off: 10).
    max_samples : cap on how many training samples are used (occlusion
        scoring is O(n_segments * max_samples) predict_label_fn calls).

    Returns
    -------
    weights : np.ndarray, shape (1, L), values in {0, 1}.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)
    if len(X_train) > max_samples:
        idx = np.random.RandomState(0).choice(len(X_train), max_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    seg_len = max(1, n_timesteps // n_segments)
    seg_boundaries = list(range(0, n_timesteps, seg_len)) + [n_timesteps]
    n_segs = len(seg_boundaries) - 1

    base_preds = np.array([predict_label_fn(x) for x in X_train])
    base_acc = float(np.mean(base_preds == y_train))

    importances = np.zeros(n_segs)
    for i in range(n_segs):
        start, end = seg_boundaries[i], seg_boundaries[i + 1]
        X_occ = X_train.copy()
        X_occ[:, start:end] = X_occ[:, start:end].mean(axis=1, keepdims=True)
        occ_preds = np.array([predict_label_fn(x) for x in X_occ])
        occ_acc = float(np.mean(occ_preds == y_train))
        importances[i] = base_acc - occ_acc

    threshold = np.percentile(importances, 75)
    weights = np.ones(n_timesteps, dtype=np.float32)
    for i in range(n_segs):
        if importances[i] >= threshold:
            start, end = seg_boundaries[i], seg_boundaries[i + 1]
            weights[start:end] = 0.0

    return weights.reshape(1, -1)


def glacier_reimp(
    sample,
    model,
    target_label=None,
    dataset=None,
    autoencoder=None,
    probability=0.5,
    pred_margin_weight=0.5,
    step_weights="uniform",
    init="identity",
    max_iter=100,
    learning_rate=1e-4,
    tolerance=1e-6,
    random_state=None,
):
    """Reimplementation of GLACIER faithful to the original TF/Keras paper code.

    Loss = pred_margin_weight * MSE(pred_prob, probability)
         + (1 - pred_margin_weight) * weighted_MAE(cf, original, step_weights)

    Parameters
    ----------
    sample : array-like, shape (L,) or (1, L)
        Input time series to explain (univariate).
    model : callable
        PyTorch model that outputs a 2D tensor of shape (batch, 2) for binary
        classification, or any callable (np array -> np array) that returns
        class probabilities.
    target_label : int, optional
        Target class (0 or 1). If None, uses 1 - predicted_label (binary flip).
    dataset : array-like, optional
        Training dataset. Required for `init="nun"` and for
        `step_weights="local"`/`"global"`. Shape (N, L) or list of (x, y) tuples.
    autoencoder : tuple or None
        If provided, a (encoder_fn, decoder_fn) pair of callables operating on
        torch tensors of shape (1, 1, L). The search is performed in latent space.
    probability : float
        Desired probability threshold for the target class (default 0.5).
    pred_margin_weight : float
        Weight for prediction margin loss in [0, 1]. The proximity loss weight
        is (1 - pred_margin_weight). Use 1.0 for unconstrained search.
    step_weights : str or array-like
        "uniform"       — all timesteps equally penalized (ones)
        "unconstrained" — no proximity penalty (zeros)
        "local"         — LIMESegment proxy: frees the *least* class-important
                          segments *for this sample* (per-query)
        "global"        — IntervalImportance proxy: frees the *most*
                          class-important segments *across the training set*
                          (query-independent — requires `dataset`; see
                          `compute_global_step_weights` to precompute once)
        array-like      — custom weight vector of shape (L,) or (1, L)
    init : str
        "identity" (default) — start the search at the query `x` itself,
        matching bake-off's own `ModifiedLatentCF._initialize`
        (`z = tf.Variable(x, ...)` — no nearest-neighbour lookup at all).
        "nun"       — start from the closest target-class training sample
        instead (requires `dataset`). Not what the paper/bake-off's code
        does, and can make the optimisation loop exit almost immediately
        if that neighbour already satisfies the validity threshold — at
        which point different `step_weights` settings become
        indistinguishable, since barely any gradient descent happens.
        Kept as an opt-in for callers who want faster/easier convergence
        over paper-fidelity.
    max_iter : int
        Maximum gradient descent iterations.
    learning_rate : float
        Adam learning rate.
    tolerance : float
        Convergence tolerance on pred_margin_loss.
    random_state : int, optional
        Random seed.

    Returns
    -------
    cf : np.ndarray, shape (L,)
        Counterfactual time series.
    cf_prob : float
        Predicted probability for the target class.
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    sample_np = np.asarray(sample, dtype=np.float32).reshape(-1)
    L = len(sample_np)

    # Determine device from model parameters (if PyTorch model)
    try:
        device = next(model.parameters()).device
        is_torch_model = True
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
        is_torch_model = False

    def predict_prob(x_np_1l):
        """Return probability for class 1 (shape: scalar)."""
        x_t = _to_torch(x_np_1l.reshape(1, 1, L), device)
        if is_torch_model:
            with torch.no_grad():
                out = model(x_t)
            probs = torch.softmax(out, dim=1) if out.shape[-1] > 1 else torch.sigmoid(out)
        else:
            out = torch.tensor(model(x_np_1l.reshape(1, -1)), dtype=torch.float32)
            probs = out
        return _to_numpy(probs)[0]

    # Predicted class for original sample
    orig_probs = predict_prob(sample_np)
    pred_label = int(np.argmax(orig_probs))

    if target_label is None:
        target_label = 1 - pred_label  # binary flip (mirrors original)

    # `xs` is needed below both for `init="nun"` and for `step_weights="global"`,
    # so it's parsed once here regardless of which of those is actually used.
    xs = None
    if dataset is not None:
        try:
            xs = [np.asarray(x[0], dtype=np.float32).reshape(-1) for x in dataset]
        except (TypeError, IndexError):
            xs = [np.asarray(x, dtype=np.float32).reshape(-1) for x in dataset]

    # --- Initialize the search candidate ---
    if init == "identity":
        # Faithful to bake-off's own `_initialize`: start at the query itself.
        cf_init = sample_np.copy()
    elif init == "nun":
        if xs is None:
            raise ValueError("init='nun' requires a dataset.")
        target_xs = [x for x in xs if np.argmax(predict_prob(x)) == target_label]
        if target_xs:
            dists = [np.sum((x - sample_np) ** 2) for x in target_xs]
            cf_init = target_xs[int(np.argmin(dists))].copy()
        else:
            cf_init = sample_np.copy()
    else:
        raise ValueError(f"Unknown init: {init!r}. Choose 'identity' or 'nun'.")

    # --- Step weights ---
    proximity_weight = 1.0 - pred_margin_weight

    if isinstance(step_weights, str):
        if step_weights == "uniform":
            sw = np.ones((1, L), dtype=np.float32)
        elif step_weights == "unconstrained":
            sw = np.zeros((1, L), dtype=np.float32)
        elif step_weights == "local":

            def model_fn_for_lime(x_2d):
                p = predict_prob(x_2d.reshape(-1))
                return p[target_label]

            sw = _get_lime_step_weights(sample_np, model_fn_for_lime, L)
        elif step_weights == "global":
            if dataset is None:
                raise ValueError("step_weights='global' requires a dataset (see compute_global_step_weights).")

            def predict_label_fn(x_1d):
                return int(np.argmax(predict_prob(x_1d)))

            X_g = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in xs], axis=0)
            y_g = np.array([predict_label_fn(x) for x in X_g])
            sw = compute_global_step_weights(X_g, y_g, predict_label_fn, L)
        else:
            raise ValueError(f"Unknown step_weights: {step_weights!r}. "
                             "Choose 'uniform', 'unconstrained', 'local', or 'global'.")
    else:
        sw = np.asarray(step_weights, dtype=np.float32).reshape(1, -1)
        if sw.shape[-1] != L:
            raise ValueError("step_weights length must match sample length")

    sw_tensor = _to_torch(sw, device)  # (1, L)

    # --- Autoencoder setup ---
    encoder_fn, decoder_fn = None, None
    if autoencoder is not None:
        encoder_fn, decoder_fn = autoencoder

    # --- Optimization ---
    original_tensor = _to_torch(sample_np.reshape(1, 1, L), device)  # (1, 1, L)
    target_prob_tensor = torch.tensor([[probability]], dtype=torch.float32, device=device)

    if encoder_fn is not None:
        with torch.no_grad():
            z_init = encoder_fn(_to_torch(cf_init.reshape(1, 1, L), device))
        z = torch.nn.Parameter(z_init.clone())
    else:
        z = torch.nn.Parameter(_to_torch(cf_init.reshape(1, 1, L), device))

    optimizer = optim.Adam([z], lr=learning_rate)

    prev_pred_margin_loss = float("inf")

    for _ in range(max_iter):
        optimizer.zero_grad()

        decoded = decoder_fn(z) if decoder_fn is not None else z  # (1, 1, L)

        if is_torch_model:
            out = model(decoded)
            probs = torch.softmax(out, dim=1) if out.shape[-1] > 1 else torch.sigmoid(out)
        else:
            decoded_np = _to_numpy(decoded)
            out = torch.tensor(model(decoded_np.reshape(1, -1)), dtype=torch.float32, device=device)
            probs = out

        pred_prob_target = probs[:, target_label : target_label + 1]  # (1, 1)

        pred_margin_loss = _pred_margin_mse(pred_prob_target, target_prob_tensor)

        # sw_tensor is (1, L), decoded is (1, 1, L) -> broadcast over channel dim
        weighted_steps_loss = _weighted_mae(
            original_tensor.squeeze(1),  # (1, L)
            decoded.squeeze(1),          # (1, L)
            sw_tensor,                   # (1, L)
        )

        loss = pred_margin_weight * pred_margin_loss + proximity_weight * weighted_steps_loss
        loss.backward()
        optimizer.step()

        # Stopping: pred_margin_loss converged and probability threshold reached
        prob_val = float(_to_numpy(pred_prob_target)[0, 0])
        pml_val = float(_to_numpy(pred_margin_loss))

        if prob_val >= probability and abs(prev_pred_margin_loss - pml_val) < tolerance:
            break
        prev_pred_margin_loss = pml_val

    # --- Extract result ---
    with torch.no_grad():
        decoded_final = decoder_fn(z) if decoder_fn is not None else z
        if is_torch_model:
            out_final = model(decoded_final)
            probs_final = torch.softmax(out_final, dim=1) if out_final.shape[-1] > 1 else torch.sigmoid(out_final)
        else:
            out_final = torch.tensor(
                model(_to_numpy(decoded_final).reshape(1, -1)), dtype=torch.float32, device=device
            )
            probs_final = out_final

    cf = _to_numpy(decoded_final).reshape(-1)
    cf_prob = float(_to_numpy(probs_final)[0, target_label])

    return cf, cf_prob


####
# The 8 paper-aligned Glacier variants (bake-off's RUN_GUIDE.md, section 2):
# every combination of AE / NoAE crossed with the 4 step_weights modes.
# Bake-off's CSV records these under two columns, `method`
# ("Autoencoder" / "No autoencoder") and `step_weight_type` ("unconstrained"
# / "local" / "global" / "uniform"); `glacier_variant()` below is a thin
# dispatcher over `glacier_reimp` using bake-off's own variant names so a
# caller doesn't have to remember which combination of `autoencoder=`/
# `step_weights=` each one maps to.
####

GLACIER_VARIANTS = {
    "Glacier-AE-Unc": dict(use_ae=True, step_weights="unconstrained"),
    "Glacier-AE-Loc": dict(use_ae=True, step_weights="local"),
    "Glacier-AE-Glob": dict(use_ae=True, step_weights="global"),
    "Glacier-AE-Unif": dict(use_ae=True, step_weights="uniform"),
    "Glacier-NoAE-Unc": dict(use_ae=False, step_weights="unconstrained"),
    "Glacier-NoAE-Loc": dict(use_ae=False, step_weights="local"),
    "Glacier-NoAE-Glob": dict(use_ae=False, step_weights="global"),
    "Glacier-NoAE-Unif": dict(use_ae=False, step_weights="uniform"),
}


def glacier_variant(variant_name, sample, model, autoencoder=None, precomputed_global_weights=None, **kwargs):
    """Run one of bake-off's 8 paper-aligned Glacier variants by name.

    See `GLACIER_VARIANTS` for the full list. Each name is
    "Glacier-{AE,NoAE}-{Unc,Loc,Glob,Unif}", e.g. `"Glacier-AE-Loc"`.

    Parameters
    ----------
    variant_name : str
        One of the 8 keys in `GLACIER_VARIANTS`.
    sample, model :
        Same as `glacier_reimp`.
    autoencoder : (encoder_fn, decoder_fn) tuple, required for "-AE-" variants
        (e.g. from `glacier_autoencoder.make_autoencoder_fns`). Ignored for
        "-NoAE-" variants — pass `None` there.
    precomputed_global_weights : np.ndarray, optional
        For "-Glob-" variants only: a weight array already computed by
        `compute_global_step_weights`, used in place of recomputing it from
        `dataset` on every call. Global weights don't depend on the query —
        callers running many queries should compute this once and pass it
        here (same caching argument as `sg_cf_fast`/`timex_cf`'s own
        query-independent setup work). Ignored for non-"-Glob-" variants.
    **kwargs :
        Forwarded to `glacier_reimp` (`dataset`, `probability`,
        `pred_margin_weight`, `init`, `max_iter`, `learning_rate`,
        `tolerance`, `random_state`). `dataset` is required whenever the
        variant's `step_weights` is `"local"`, or `"global"` without
        `precomputed_global_weights`.

    Returns
    -------
    Same as `glacier_reimp`: `(cf, cf_prob)`.
    """
    if variant_name not in GLACIER_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant_name!r}. Choose one of: {sorted(GLACIER_VARIANTS)}"
        )
    spec = GLACIER_VARIANTS[variant_name]

    if spec["use_ae"]:
        if autoencoder is None:
            raise ValueError(f"{variant_name} requires autoencoder=(encoder_fn, decoder_fn).")
        ae_arg = autoencoder
    else:
        ae_arg = None

    step_weights = spec["step_weights"]
    if step_weights == "global" and precomputed_global_weights is not None:
        step_weights = precomputed_global_weights

    return glacier_reimp(sample, model, autoencoder=ae_arg, step_weights=step_weights, **kwargs)
