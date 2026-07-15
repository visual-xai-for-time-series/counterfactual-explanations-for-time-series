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


def glacier_reimp(
    sample,
    model,
    target_label=None,
    dataset=None,
    autoencoder=None,
    probability=0.5,
    pred_margin_weight=0.5,
    step_weights="uniform",
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
        Training dataset used to initialize the CF from the nearest target-class
        sample. Shape (N, L) or list of (x, y) tuples.
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
        "local"         — LIMESegment proxy: mask unimportant segments
        array-like      — custom weight vector of shape (L,) or (1, L)
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

    # --- Initialize CF from closest dataset sample of target class ---
    if dataset is not None:
        try:
            xs = [np.asarray(x[0], dtype=np.float32).reshape(-1) for x in dataset]
        except (TypeError, IndexError):
            xs = [np.asarray(x, dtype=np.float32).reshape(-1) for x in dataset]

        target_xs = []
        for x in xs:
            p = predict_prob(x)
            if np.argmax(p) == target_label:
                target_xs.append(x)

        if target_xs:
            dists = [np.sum((x - sample_np) ** 2) for x in target_xs]
            cf_init = target_xs[int(np.argmin(dists))].copy()
        else:
            cf_init = sample_np.copy()
    else:
        cf_init = sample_np.copy()

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
        else:
            raise ValueError(f"Unknown step_weights: {step_weights!r}. "
                             "Choose 'uniform', 'unconstrained', or 'local'.")
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
