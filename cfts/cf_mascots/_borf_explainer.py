"""
Self-contained BorfExplainer engine for the MASCOTS counterfactual method.

Vendored and adapted from DawidPludowski/borf (MIT licence):
  borf/mascots/explainer/borf.py      → BorfExplainer
  borf/mascots/explainer/swapping.py  → scalar_swap / gaussian_swap
  borf/mascots/explainer/gp.py        → ExactGPModel
  borf/mascots/attributions/          → attribution helpers

Key adaptations vs. the original:
  - ``bin.utils.models.MLP`` (pytorch-lightning) replaced by ``_MLP``
    (plain PyTorch, no extra dependency).
  - ``gpytorch`` is optional: gaussian_swap falls back to scalar_swap.
  - ``shap`` is optional: ShapAttribution falls back to _CoefficientAttribution.
  - BoRF pipeline provided by ``_fast_borf`` (no external fast_borf package needed).
  - Attribution uses a LogisticRegression surrogate for coefficient extraction.

Paper: Płudowski, D., Spinnato, F., Wilczyński, P., Kotowski, K., Ntagiou, E. V.,
       Guidotti, R., & Biecek, P. (2025). MASCOTS: Model-Agnostic Symbolic
       COunterfactual explanations for Time Series. arXiv:2503.22389.
GitHub: https://github.com/DawidPludowski/borf
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy import stats
from scipy.sparse import csr_matrix, issparse
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate MLP  (replaces bin.utils.models.MLP from the original repo)
# ─────────────────────────────────────────────────────────────────────────────

def _to_dense(X) -> np.ndarray:
    return X.toarray() if issparse(X) else np.asarray(X, dtype=np.float32)


class _MLP:
    """Feedforward surrogate trained on BoRF features.

    Matches the calling convention BorfExplainer._train_surrogate expects:
        mlp.fit(X_train, y_train, X_val, y_val)
        mlp.predict(X) -> (N, n_classes) softmax probabilities
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        lr: float = 1e-4,
        max_epochs: int = 200,
        batch_size: int = 64,
        patience: int = 20,
    ) -> None:
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sizes = [input_size] + [hidden_size] * n_layers + [output_size]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers).to(self.device)
        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _t(self, X) -> torch.Tensor:
        return torch.from_numpy(_to_dense(X)).float().to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        Xt, yt = self._t(X_train), self._t(_to_dense(y_train))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        best_val, patience_left, best_state = float("inf"), self.patience, None

        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(Xt.shape[0])
            for s in range(0, Xt.shape[0], self.batch_size):
                idx = perm[s: s + self.batch_size]
                tgt = yt[idx].argmax(dim=1) if yt.ndim == 2 else yt[idx].long()
                opt.zero_grad()
                loss_fn(self.net(Xt[idx]), tgt).backward()
                opt.step()

            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    Xv, yv = self._t(X_val), self._t(_to_dense(y_val))
                    yv = yv.argmax(dim=1) if yv.ndim == 2 else yv.long()
                    val_loss = loss_fn(self.net(Xv), yv).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, X) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(self._t(X)), dim=1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# SAX word swapping  (mascots/explainer/swapping.py + gp.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_quantiles(n_bins: int, bound_scalar: float | None = None) -> NDArray[np.float64]:
    qs = stats.norm.ppf(np.linspace(0, 1, num=n_bins + 1))
    if bound_scalar is not None:
        qs = np.clip(qs, a_min=qs[1] * bound_scalar, a_max=qs[-2] * bound_scalar)
    return qs


def retrieve_word(X_sub: NDArray, alphabet_size: int, symbol_size: int) -> NDArray[np.int64]:
    """Map a time-series segment to a SAX word (array of integer symbols)."""
    n_symbols = X_sub.shape[0] // symbol_size
    qs = _get_quantiles(alphabet_size)
    X = X_sub.copy()
    std = X.std()
    if std < 1e-8:
        return np.zeros(n_symbols, dtype=np.int64)
    X = (X - X.mean()) / std
    word = [
        np.digitize(X[i * symbol_size: (i + 1) * symbol_size].mean(), qs, right=False)
        for i in range(n_symbols)
    ]
    return np.array(word, dtype=np.int64) - 1


def scalar_swap(
    X_sub: NDArray,
    word: NDArray[np.int64],
    alphabet_size: int,
    symbol_size: int,
    quantile_bound: float = 2.0,
) -> NDArray:
    """Shift each symbol of *X_sub* to match *word* via mean-level correction."""
    current_word = retrieve_word(X_sub, alphabet_size, symbol_size)
    out = X_sub.copy()
    mean, std = out.mean(), out.std()
    if std < 1e-8:
        return out
    out_norm = (out - mean) / std
    qs = _get_quantiles(alphabet_size, quantile_bound)
    requested = np.vstack([qs[word], qs[word + 1]]).mean(axis=0)
    diffs = requested - out_norm.reshape(-1, symbol_size).mean(axis=1)
    out += (np.repeat(diffs, symbol_size) * std + mean) * np.repeat(
        current_word != word, symbol_size
    ).astype(int)
    return out


def gaussian_swap(
    X_sub: NDArray,
    word: NDArray[np.int64],
    alphabet_size: int,
    symbol_size: int,
    quantile_bound: float = 2.0,
    train: bool = True,
) -> NDArray:
    """GP-smoothed SAX swap. Falls back to scalar_swap when gpytorch is absent."""
    try:
        import gpytorch
    except ImportError:
        return scalar_swap(X_sub, word, alphabet_size, symbol_size, quantile_bound)

    out = X_sub.copy()
    mean, std = out.mean(), out.std()
    if std < 1e-8:
        return out
    out_norm = (out - mean) / std

    qs = _get_quantiles(alphabet_size, quantile_bound)
    word_org = retrieve_word(X_sub, alphabet_size, symbol_size)
    means_org = np.array([
        out_norm[symbol_size * i: symbol_size * (i + 1)].mean()
        for i in range(len(word))
    ])
    shifts = np.array([
        (qs[word[i]] + qs[word[i] + 1]) / 2 - means_org[i] if word[i] != word_org[i] else 0.0
        for i in range(len(word))
    ])
    for i in range(len(word)):
        out_norm[symbol_size * i: symbol_size * (i + 1)] += shifts[i]

    # GP smoothing for changed symbols
    class _GP(gpytorch.models.ExactGP):
        def __init__(self, tx, ty):
            super().__init__(tx, ty, gpytorch.likelihoods.GaussianLikelihood())
            self.mean = gpytorch.means.ZeroMean()
            self.covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean(x), self.covar(x))

    tx = torch.arange(len(word) * symbol_size, dtype=torch.float)
    ty = torch.tensor(out_norm, dtype=torch.float)
    gp = _GP(tx, ty)
    if train:
        gp.train()
        gp.likelihood.train()
        opt = torch.optim.Adam(gp.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
        for _ in range(100):
            opt.zero_grad()
            (-mll(gp(tx), ty)).backward()
            opt.step()
    gp.eval()
    gp.likelihood.eval()
    with torch.no_grad():
        for i in range(len(word)):
            if word[i] != word_org[i]:
                sl = slice(symbol_size * i, symbol_size * (i + 1))
                out_norm[sl] = gp(tx[sl]).mean.numpy()
    return out_norm * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Attribution  (mascots/attributions/attributions.py + utils.py)
# ─────────────────────────────────────────────────────────────────────────────

class _AttributionBase(ABC):
    def __init__(self, surrogate, X_train: NDArray) -> None:
        self.surrogate = surrogate
        self.X_train = X_train

    def build(self) -> None:
        self._build()

    @abstractmethod
    def _build(self) -> None: ...

    def explain(self, X: NDArray) -> NDArray:
        return self._explain(X)

    @abstractmethod
    def _explain(self, X: NDArray) -> NDArray: ...


class _CoefficientAttribution(_AttributionBase):
    """Logistic-regression coefficients as a proxy for SHAP values.

    Returns F of shape ``(n_classes, n_samples, n_features)`` where every
    sample row is identical (global linear weights).
    """

    def _build(self) -> None:
        coef = self.surrogate.coef_
        self._coef = np.vstack([-coef, coef]) if coef.shape[0] == 1 else coef

    def _explain(self, X: NDArray) -> NDArray:
        n = X.shape[0] if hasattr(X, "shape") else 1
        return np.tile(self._coef[:, np.newaxis, :], (1, n, 1))


class _ShapAttribution(_AttributionBase):
    """SHAP-based attribution (requires the ``shap`` package)."""

    def __init__(self, surrogate, X_train: NDArray, scope: str = "local", mode: str = "normal") -> None:
        super().__init__(surrogate, X_train)
        self.scope = scope
        self.mode = mode

    def _build(self) -> None:
        import shap
        _mapping = {
            "normal": shap.Explainer,
            "linear": shap.LinearExplainer,
            "tree": shap.TreeExplainer,
            "kernel": shap.KernelExplainer,
        }
        X = self.X_train.toarray() if issparse(self.X_train) else self.X_train
        self._attr = _mapping.get(self.mode, shap.Explainer)(self.surrogate, X)
        if self.scope == "global":
            imp = np.array(self._attr.shap_values(X[0])).swapaxes(1, 2).swapaxes(0, 1)
            self._F = np.abs(imp).mean(axis=1)

    def _explain(self, X: NDArray) -> NDArray:
        Xd = X.toarray() if issparse(X) else X
        if self.scope == "global":
            return np.tile(self._F[:, np.newaxis, :], (1, Xd.shape[0], 1))
        return np.array(self._attr.shap_values(Xd)).swapaxes(1, 2).swapaxes(0, 1)


def _build_attribution(
    surrogate, X_train: NDArray, attribution_name: str = "coef", **kwargs
) -> _AttributionBase:
    if attribution_name == "shap":
        try:
            import shap  # noqa: F401
            attr = _ShapAttribution(surrogate, X_train, **kwargs)
        except ImportError:
            attr = _CoefficientAttribution(surrogate, X_train)
    else:
        attr = _CoefficientAttribution(surrogate, X_train)
    attr.build()
    return attr


# ─────────────────────────────────────────────────────────────────────────────
# BoRF pipeline helpers  (mascots/explainer/pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def _require_fast_borf():
    from ._fast_borf import BorfPipelineBuilder, BagOfReceptiveFields
    return BorfPipelineBuilder, BagOfReceptiveFields


def _default_borf_args() -> dict[str, Any]:
    from ._fast_borf import ReshapeTo2D, ToScipySparse, ZeroColumnsRemover
    return {
        "min_window_to_signal_std_ratio": 0,
        "pipeline_objects": [
            (ReshapeTo2D, dict(keep_unraveled_index=True)),
            (ZeroColumnsRemover, dict(axis=0)),
            (ToScipySparse, dict()),
        ],
        "n_jobs": 1,
        "n_jobs_numba": 1,
    }


def _auto_borf_config(n_timestamps: int) -> list[dict[str, Any]]:
    configs = []
    for i in range(3, n_timestamps):
        ws = 2 ** i
        if ws > n_timestamps:
            break
        configs.append({"window_size": ws, "stride": 1, "dilation": 1, "word_length": 4, "alphabet_size": 3})
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# BorfExplainer
# ─────────────────────────────────────────────────────────────────────────────

class BorfExplainer:
    """Adapted MASCOTS BorfExplainer.

    Usage (mirrors the original mascots API)::

        explainer = BorfExplainer(prediction_fn, prediction_fn_proba)
        metrics   = explainer.build(X_train)
        cfs, meta = explainer.counterfactual(X_obs, target_cls=1)

    Parameters
    ----------
    prediction_fn:
        ``(X: ndarray (N, C, L)) -> ndarray (N,)`` integer labels.
    prediction_fn_proba:
        ``(X: ndarray (N, C, L)) -> ndarray (N, n_classes)`` probabilities.
    borf_config:
        List of window-config dicts for ``fast_borf``, or ``"auto"`` to derive
        them from the training-data shape.
    borf_args:
        Extra keyword arguments forwarded to ``BorfPipelineBuilder``.
    """

    def __init__(
        self,
        prediction_fn,
        prediction_fn_proba,
        borf_config: list[dict] | Literal["auto"] = "auto",
        borf_args: dict[str, Any] | None = None,
    ) -> None:
        self.prediction_fn = prediction_fn
        self.prediction_fn_proba = prediction_fn_proba
        self._borf_config_arg = borf_config
        self._borf_args = borf_args or {}

    # ── public ────────────────────────────────────────────────────────────────

    def build(
        self,
        X: NDArray[np.float64],
        attribution_name: str = "coef",
        attribution_args: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Fit the BoRF extractor and MLP surrogate.

        Parameters
        ----------
        X:
            Training data, shape ``(N, C, L)``.
        attribution_name:
            ``"coef"`` (default, no extra deps) or ``"shap"`` (requires shap).
        attribution_args:
            Passed to the attribution constructor.
        seed:
            Random seed for the train/val split.

        Returns
        -------
        dict with surrogate evaluation metrics.
        """
        BorfPipelineBuilder, BagOfReceptiveFields = _require_fast_borf()
        from sklearn.pipeline import make_pipeline

        borf_config = (
            _auto_borf_config(X.shape[2])
            if self._borf_config_arg == "auto"
            else self._borf_config_arg
        )
        borf_args = {**_default_borf_args(), **self._borf_args}

        self.borf = BorfPipelineBuilder(configs=borf_config, **borf_args).build(X)
        X_transformed: csr_matrix = self.borf.fit_transform(X)
        y = self.prediction_fn_proba(X)

        mlp = _MLP(input_size=X_transformed.shape[1], output_size=y.shape[1])
        self.borf_pipeline = make_pipeline(self.borf, mlp)
        metrics = self._train_surrogate(X_transformed, y, seed)

        # Use LogisticRegression as the attribution surrogate (original mascots approach).
        # _MLP lacks .coef_, so we train a lightweight LR for coefficient-based attribution.
        from sklearn.linear_model import LogisticRegression
        y_cls = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)
        X_dense = X_transformed.toarray()
        lr = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
        lr.fit(X_dense, y_cls)

        self.attribution = _build_attribution(
            lr, X_dense, attribution_name, **(attribution_args or {})
        )

        self.mapper = BagOfReceptiveFields(self.borf)
        self.mapper.build(np.array(X))
        self.mapper.task_ = "classification"
        self.mapper_info = self._build_mapper_info()

        return metrics

    def counterfactual(
        self,
        X_obs: NDArray[np.float64],
        target_cls: int,
        swap_method: Literal["scalar", "gaussian"] = "scalar",
        max_borf_changes: int = 100,
        min_word_len_change: int = -1,
        max_word_len_change: int = -1,
        min_symbol_size: int = -1,
        max_symbol_size: int = -1,
        allow_only_shapes: tuple = (),
        allow_only_idx: tuple = (),
        select_top_k: int = 5,
        C: float = 0.1,
        n_restarts: int = 5,
        returns_meta: bool = True,
        seed: int | None = None,
    ) -> NDArray | tuple[NDArray, dict]:
        """Generate counterfactuals for *X_obs*.

        Parameters
        ----------
        X_obs:
            Single observation, shape ``(1, C, L)``.
        target_cls:
            Target class index.
        swap_method:
            ``"scalar"`` or ``"gaussian"`` (requires gpytorch).
        max_borf_changes:
            Maximum swap iterations per restart.
        n_restarts:
            Number of independent restarts.
        returns_meta:
            When ``True`` also return per-restart metadata dict.

        Returns
        -------
        counterfactuals : ndarray  shape ``(n_restarts, C, L)``
        meta            : dict  (only when *returns_meta* is ``True``)
        """
        np.random.seed(seed)
        assert X_obs.shape[0] == 1

        self._constraints = dict(
            min_word_len=min_word_len_change,
            max_word_len=max_word_len_change,
            min_sym=min_symbol_size,
            max_sym=max_symbol_size,
            only_shapes=allow_only_shapes,
            only_idx=allow_only_idx,
        )

        y = self.prediction_fn(X_obs)
        counterfactuals, meta_all = [], {}

        for r in range(n_restarts):
            new_X = X_obs.copy()
            new_X_t = self.borf.transform(new_X)
            meta_all[r] = {}
            failed = 0

            for step in range(1, max_borf_changes + 1):
                proposal, record = self._swap_step(
                    new_X, new_X_t, y, target_cls, C, select_top_k, swap_method
                )
                if proposal is None:
                    break
                assert y != target_cls
                if (
                    self.prediction_fn_proba(proposal)[0, target_cls]
                    <= self.prediction_fn_proba(new_X)[0, target_cls]
                ):
                    failed += 1
                    if select_top_k == 1:
                        break
                else:
                    new_X = proposal
                    new_X_t = self.borf.transform(new_X)
                    if returns_meta:
                        meta_all[r][step] = record
                if self.prediction_fn(new_X) != y:
                    break

            counterfactuals.append(new_X)
            meta_all[r]["effective_iter"] = step - failed

        result = np.vstack(counterfactuals)
        return (result, meta_all) if returns_meta else result

    # ── internals ─────────────────────────────────────────────────────────────

    def _train_surrogate(self, X_t: csr_matrix, y: NDArray, seed: int) -> dict[str, Any]:
        X_tr, X_val, y_tr, y_val = train_test_split(X_t, y, random_state=seed)
        mlp: _MLP = self.borf_pipeline[-1]
        mlp.fit(X_tr, y_tr, X_val, y_val)
        y_pred = mlp.predict(X_t)
        try:
            y_cls = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)
            return {"accuracy": (y_cls == y_pred.argmax(axis=1)).mean()}
        except Exception:
            return {}

    def _build_mapper_info(self) -> dict[int, dict[str, int]]:
        info: dict[int, dict] = {}
        for idx, rf in self.mapper.receptive_fields_.items():
            n_sym = rf.word_length
            sym_size = rf.window_size // rf.word_length
            info[idx] = {
                "word_length": rf.window_size,
                "alphabet_size": rf.alphabet_size,
                "n_symbols": n_sym,
                "symbol_size": sym_size,
                "signal_idx": rf.signal_idx,
                "dilation": rf.dilation,
                "stride": rf.stride,
            }
        return info

    def _swap_step(
        self,
        X_org: NDArray,
        X_t: csr_matrix,
        y: NDArray,
        target_cls: int,
        C: float,
        select_top_k: int,
        swap_method: str,
    ) -> tuple[NDArray | None, dict]:
        self.mapper.build(X_org, y, self.prediction_fn(X_org))
        self.mapper.task_ = "classification"

        F = self.attribution.explain(X_t)
        importance = np.ravel(F[target_cls, :, :] - F[y[0], :, :])
        n_feat = importance.shape[0]

        imp_pw = (importance.reshape(-1, 1) - importance.reshape(1, -1)).reshape(-1)
        imp_pw[imp_pw < 0] = 0
        mx = imp_pw.max()
        if mx > 0:
            imp_pw /= mx

        penalty = C * np.array(
            [[self._word_diff(i, j) for i in range(n_feat)] for j in range(n_feat)]
        ).reshape(-1)

        order = np.argsort(-imp_pw + penalty)
        order = order[order % n_feat != 0]
        pairs = [(el // n_feat, el % n_feat) for el in order]

        existing = np.arange(X_t.shape[1])[np.ravel(np.asarray(X_t[[0], :].todense()) > 0)]

        top_k: list[NDArray] = []
        meta: dict[str, Any] = {
            "removed_shapes_candidates": [],
            "added_shapes_candidates": [],
            "added_channel_candidates": [],
            "expected_gains": [],
            "windows_size": [],
            "indicies": [],
        }

        for new_idx, old_idx in pairs:
            if len(top_k) == select_top_k:
                break
            if not self._word_ok(new_idx):
                continue
            if not self._words_compatible(new_idx, old_idx):
                continue
            if old_idx not in existing:
                continue

            alph, sym_size, target_word, sig = self._word_info(new_idx)
            n_seq = self.mapper.receptive_fields_[old_idx].alignments[0].shape[0]
            indices = (
                self.mapper.receptive_fields_[old_idx]
                .alignments[0][np.random.choice(n_seq)]
                .reshape(-1)
            )

            swap_fn = gaussian_swap if swap_method == "gaussian" else scalar_swap
            proposal_seg = swap_fn(X_org[0, sig, indices], target_word, alph, sym_size)

            new_obs = X_org.copy()
            new_obs[0, sig, indices] = proposal_seg
            top_k.append(new_obs)

            meta["added_shapes_candidates"].append(target_word)
            meta["removed_shapes_candidates"].append(
                self.mapper.receptive_fields_[old_idx].word_array
            )
            meta["added_channel_candidates"].append(sig)
            meta["expected_gains"].append(importance[new_idx] - importance[old_idx])
            meta["windows_size"].append(self.mapper_info[new_idx]["word_length"])
            meta["indicies"].append(indices)

        if not top_k:
            return None, {}

        chosen = np.random.choice(len(top_k))
        meta["selected_swap"] = [chosen]
        meta["new_obs"] = top_k[chosen]
        return top_k[chosen], meta

    def _word_info(self, idx: int) -> tuple:
        m = self.mapper_info[idx]
        return m["alphabet_size"], m["symbol_size"], self.mapper.receptive_fields_[idx].word_array, m["signal_idx"]

    def _word_diff(self, idx1: int, idx2: int) -> float:
        w1 = self.mapper.receptive_fields_[idx1].word_array
        w2 = self.mapper.receptive_fields_[idx2].word_array
        return float("inf") if w1.shape != w2.shape else float(np.abs(w1 - w2).sum())

    def _words_compatible(self, idx1: int, idx2: int) -> bool:
        return all(
            self.mapper_info[idx1][k] == self.mapper_info[idx2][k]
            for k in ("word_length", "alphabet_size", "n_symbols", "symbol_size", "signal_idx", "dilation")
        )

    def _word_ok(self, idx: int) -> bool:
        m = self.mapper_info[idx]
        c = self._constraints
        if m["dilation"] != 1:
            return False
        if c["min_word_len"] > 0 and m["word_length"] < c["min_word_len"]:
            return False
        if c["max_word_len"] > 0 and m["word_length"] > c["max_word_len"]:
            return False
        if c["min_sym"] > 0 and m["symbol_size"] < c["min_sym"]:
            return False
        if c["max_sym"] > 0 and m["symbol_size"] > c["max_sym"]:
            return False
        if c["only_shapes"] and not any(
            (np.stack(c["only_shapes"]) - self.mapper.receptive_fields_[idx].word_array).sum(axis=0) == 0
        ):
            return False
        if c["only_idx"] and m["signal_idx"] not in c["only_idx"]:
            return False
        return True
