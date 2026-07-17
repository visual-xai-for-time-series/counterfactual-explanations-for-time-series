"""Compare IMFACT vs Native Guide vs Wachter on FruitFlies.

Evaluates counterfactual methods on correctly-classified FruitFlies test
samples and saves metric summary, bar charts, a UMAP projection, and
signal-space waveform plots to an output directory.

FruitFlies sequences are ~5000 points long. Wachter's gradient descent on
series of this length can require significant memory in constrained
environments. Pass --exclude-wachter to drop it if that's an issue.

Usage:
    python compare_imfact_guide_wachter.py [--n-samples 50] [--out-dir ./output]
    python compare_imfact_guide_wachter.py --exclude-wachter
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import warnings

import matplotlib.pyplot as plt
import numpy as np
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")  # suppress OMP nested-parallelism info before umap/numba load
import pandas as pd
import torch
import umap
from aeon.datasets import load_classification
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _find_repo_root(start: str) -> str:
    current = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(current, "cfts")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError("Could not locate repository root containing cfts/")


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")

for _p in (REPO_ROOT, EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base.data import TimeSeriesDataset, collate_sparse
from base.model import SimpleCNN
from cfts.cf_imfact.imfact import imfact_cf
from cfts.cf_native_guide.native_guide import native_guide_uni_cf
from cfts.cf_wachter.wachter import wachter_gradient_cf
from cfts.metrics import evaluate_counterfactual

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({"font.size": 14})

N_BACKGROUND = 512
DOWNSAMPLE = 1

METHOD_COLORS = {
    "imfact_default": "#e63946",
    "native_guide": "#1d3557",
    "wachter": "#ff7f11",
    "glacier": "#2a9d8f",
    "mascots": "#8338ec",
}
BAR_COLORS = ["#2a9d8f", "#457b9d", "#e76f51", "#06d6a0", "#8338ec"]
_METHOD_ORDER_DEFAULT = ["imfact_default", "native_guide", "wachter", "glacier", "mascots"]
METHOD_ORDER: list[str] = _METHOD_ORDER_DEFAULT  # overridden in main() if --exclude-* flags are passed


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_fruitflies(split: str):
    data_root = os.path.join(REPO_ROOT, "data", "UCR")
    dataset_dir = os.path.join(data_root, "FruitFlies")
    train_ts = os.path.join(dataset_dir, "FruitFlies_TRAIN.ts")
    test_ts = os.path.join(dataset_dir, "FruitFlies_TEST.ts")

    if not (os.path.exists(train_ts) and os.path.exists(test_ts)):
        archive = os.path.join(data_root, "FruitFlies.zip")
        if not os.path.exists(archive):
            url = "https://timeseriesclassification.com/aeon-toolkit/FruitFlies.zip"
            print(f"Downloading FruitFlies archive from {url} …")
            urllib.request.urlretrieve(url, archive)
        print("Extracting FruitFlies .ts files …")
        os.makedirs(dataset_dir, exist_ok=True)
        subprocess.run(
            ["unzip", "-j", "-o", archive, "FruitFlies_TRAIN.ts", "FruitFlies_TEST.ts", "-d", dataset_dir],
            check=True,
        )

    X, y = load_classification(name="FruitFlies", split=split, extract_path=data_root)
    encoder = OneHotEncoder(categories="auto", sparse_output=False)
    y_enc = encoder.fit_transform(np.expand_dims(y, axis=-1))
    dataset = TimeSeriesDataset(X=X, y=y_enc, name="FruitFlies", mapping=encoder.categories_)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_sparse)
    return dataloader, dataset


class _DownsampledView:
    def __init__(self, dataset, factor: int):
        self._dataset = dataset
        self._factor = factor

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        x, y = self._dataset[idx]
        arr = np.asarray(x, dtype=np.float32)
        if self._factor > 1:
            arr = arr[..., :: self._factor]
        return arr, y

    def __getattr__(self, name):
        return getattr(self._dataset, name)


class _SubsetView:
    """Stratified cap: picks up to max_size samples balanced across all classes to avoid OOM.

    Uses dataset.y (numpy array) for fast label access without iterating over time series.
    Guarantees every class is represented — required for NUN / native-guide search.
    """
    def __init__(self, dataset, max_size: int):
        raw_y = np.asarray(dataset.y)
        labels = raw_y.argmax(axis=-1) if raw_y.ndim == 2 else raw_y.astype(int)
        labels = labels[: len(dataset)]  # trim to actual length (proxy may be shorter)

        classes = np.unique(labels)
        per_class = max(1, max_size // len(classes))
        indices: list[int] = []
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            indices.extend(cls_idx[:per_class].tolist())

        self._indices = sorted(indices[:max_size])
        self._dataset = dataset

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        if idx >= len(self._indices):
            raise IndexError(idx)
        return self._dataset[self._indices[idx]]

    def __getattr__(self, name):
        return getattr(self._dataset, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_channel_first(sample: np.ndarray) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr if arr.shape[0] <= arr.shape[1] else arr.T
    raise ValueError(f"Unsupported sample shape: {arr.shape}")


def to_class_index(label) -> int:
    label_arr = np.asarray(label)
    if label_arr.ndim == 0:
        return int(label_arr)
    return int(np.argmax(label_arr))


def predict_scores(model: torch.nn.Module, sample: np.ndarray, device: torch.device) -> np.ndarray:
    ts = torch.tensor(sample, dtype=torch.float32, device=device)
    if ts.ndim == 1:
        ts = ts.reshape(1, 1, -1)
    elif ts.ndim == 2:
        if ts.shape[0] > ts.shape[1]:
            ts = ts.T
        ts = ts.unsqueeze(0)
    with torch.no_grad():
        return model(ts).detach().cpu().numpy().reshape(-1)


def model_wrapper_factory(model: torch.nn.Module, device: torch.device):
    def wrapped(ts: np.ndarray) -> np.ndarray:
        return predict_scores(model, np.asarray(ts, dtype=np.float32), device)
    return wrapped


def select_correct_indices(model, dataset, max_count: int, device) -> list[int]:
    selected = []
    for idx in range(len(dataset)):
        sample, label = dataset[idx]
        scores = predict_scores(model, np.asarray(sample, dtype=np.float32), device)
        if int(np.argmax(scores)) == to_class_index(label):
            selected.append(idx)
        if len(selected) >= max_count:
            break
    return selected


def infer_target_class(scores: np.ndarray) -> int:
    return int(np.argsort(scores)[::-1][1])


# ---------------------------------------------------------------------------
# CF runners
# ---------------------------------------------------------------------------

def run_imfact_default(sample, dataset, model, target_class):
    return imfact_cf(
        sample=sample,
        dataset=dataset,
        model=model,
        method="distance",
        target_class=target_class,
        step=0.05,
        max_iter=80,
        max_imfs=6,
        n_nuns=1,
        nun_switch="cycle",
        verbose=False,
    )


def run_native_guide(sample, dataset, model, target_class):
    return native_guide_uni_cf(sample=sample, model=model, target_class=target_class, dataset=dataset)


def run_wachter(sample, dataset, model, target_class):
    return wachter_gradient_cf(
        sample=sample,
        dataset=dataset,
        model=model,
        target=target_class,
        max_cfs=3,
        distance="euclidean",
        verbose=False,
    )


def run_glacier(sample, dataset, model, target_class):
    from cfts.cf_glacier.glacier import glacier_cf
    return glacier_cf(
        sample=sample, dataset=dataset, model=model,
        target_class=target_class, max_iterations=500, verbose=False,
    )


def run_mascots(sample, dataset, model, target_class):
    from cfts.cf_mascots.mascots import mascots_cf
    return mascots_cf(
        sample=sample, model=model,
        target_class=target_class, dataset=dataset,
        max_iter=100, verbose=False,
    )


RUNNERS = {
    "imfact_default": run_imfact_default,
    "native_guide": run_native_guide,
    "wachter": run_wachter,
    "glacier": run_glacier,
    "mascots": run_mascots,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(selected_indices, dataset_test, model, device, reference_data, search_dataset=None) -> tuple[pd.DataFrame, dict]:
    # search_dataset is passed to native_guide, wachter, glacier, and mascots to avoid OOM
    # on the full 17k-sample FruitFlies set. IMFACT always uses dataset_test for NUN search.
    model_for_metrics = model_wrapper_factory(model, device)
    records = []
    all_cf_outputs: dict = {}

    for i, idx in enumerate(selected_indices):
        print(f"  [{i + 1}/{len(selected_indices)}] sample {idx}", flush=True)
        sample, label = dataset_test[idx]
        sample = np.asarray(sample, dtype=np.float32)

        scores_orig = predict_scores(model, sample, device)
        pred_orig = int(np.argmax(scores_orig))
        true_label = to_class_index(label)
        target_class = infer_target_class(scores_orig)

        all_cf_outputs[idx] = {"sample": sample, "true_label": true_label, "target_class": target_class}

        for method_name in METHOD_ORDER:
            base = {
                "sample_idx": idx,
                "method": method_name,
                "true_label": true_label,
                "pred_orig": pred_orig,
                "target_class": target_class,
                "pred_cf": None,
                "l2_norm": np.nan,
                "normalized_distance": np.nan,
                "sparsity": np.nan,
                "range_validity": np.nan,
                "autocorr": np.nan,
                "keane_validity": np.nan,
                "keane_proximity": np.nan,
                "keane_compactness": np.nan,
                "validity": 0.0,
                "temporal_consistency": np.nan,
                "confidence": np.nan,
                "elapsed": np.nan,
                "error": None,
            }

            start_time = time.time()
            try:
                # All methods use search_dataset (capped) to avoid OOM on FruitFlies 17k-sample test set.
                ds = search_dataset or dataset_test
                cf, pred_cf_scores = RUNNERS[method_name](sample, ds, model, target_class)
            except Exception as exc:
                elapsed = time.time() - start_time
                records.append({**base, "elapsed": elapsed, "error": f"{type(exc).__name__}: {exc}"})
                continue
            elapsed = time.time() - start_time

            if cf is None or pred_cf_scores is None:
                records.append({**base, "elapsed": elapsed, "error": "Method returned None"})
                continue

            cf = np.asarray(cf, dtype=np.float32)
            _scores = np.asarray(pred_cf_scores).reshape(-1)
            pred_cf = int(np.argmax(_scores))
            _exp = np.exp(_scores - _scores.max())
            cf_confidence = float(_exp[pred_cf] / _exp.sum())

            s_cf = to_channel_first(sample)
            c_cf = to_channel_first(cf)

            _m = evaluate_counterfactual(
                s_cf, c_cf,
                model=model_for_metrics, target_class=int(target_class),
                reference_data=reference_data,
            )

            records.append({
                **base,
                "elapsed": elapsed,
                "pred_cf": pred_cf,
                "l2_norm": _m["l2_distance"],
                "normalized_distance": _m["normalized_distance"],
                "sparsity": _m["sparsity"],
                "range_validity": _m.get("range_validity", np.nan),
                "autocorr": _m["autocorr_preservation"],
                "keane_validity": _m["keane_validity"],
                "keane_proximity": _m["keane_proximity"],
                "keane_compactness": _m["keane_compactness"],
                "validity": _m["validity"],
                "temporal_consistency": _m["temporal_consistency"],
                "confidence": cf_confidence,
            })

            all_cf_outputs[idx][method_name] = cf

    return pd.DataFrame(records), all_cf_outputs


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    successful_df = results_df[results_df["validity"] == 1.0].copy()
    agg_all = (
        results_df.groupby("method", dropna=False)
        .agg(
            n_total=("sample_idx", "count"),
            validity_rate=("validity", "mean"),
            elapsed_mean=("elapsed", "mean"),
        )
        .reset_index()
    )
    agg_suc = (
        successful_df.groupby("method", dropna=False)
        .agg(
            l2_norm_mean=("l2_norm", "mean"),
            normalized_distance_mean=("normalized_distance", "mean"),
            sparsity_mean=("sparsity", "mean"),
            range_validity_mean=("range_validity", "mean"),
            autocorr_mean=("autocorr", "mean"),
            keane_validity_mean=("keane_validity", "mean"),
            keane_proximity_mean=("keane_proximity", "mean"),
            keane_compactness_mean=("keane_compactness", "mean"),
            temporal_consistency_mean=("temporal_consistency", "mean"),
            confidence_mean=("confidence", "mean"),
        )
        .reset_index()
    )
    summary = agg_all.merge(agg_suc, on="method", how="left")
    summary["validity_rate"] = summary["validity_rate"].fillna(0.0)
    return summary.sort_values("method").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bar_metrics(summary_df: pd.DataFrame, out_path: str) -> None:
    plot_df = summary_df.sort_values("method").copy()
    methods = plot_df["method"].tolist()
    colors = BAR_COLORS[: len(methods)]

    def _bar(ax, vals, title, ylim=(0, 1.05)):
        ax.bar(methods, vals, color=colors)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    l2_score = 1.0 / (1.0 + plot_df["l2_norm_mean"])
    norm_dist_score = 1.0 / (1.0 + plot_df["normalized_distance_mean"])

    time_max = max(float(plot_df["elapsed_mean"].fillna(0).max()) * 1.1, 0.01)

    fig, axes = plt.subplots(5, 3, figsize=(16, 20))
    _bar(axes[0, 0], plot_df["validity_rate"], "Validity (higher better)")
    _bar(axes[0, 1], plot_df["sparsity_mean"], "Sparsity (higher better)")
    _bar(axes[0, 2], plot_df["range_validity_mean"], "Range Validity (higher better)")
    _bar(axes[1, 0], plot_df["autocorr_mean"], "Autocorr Preservation (higher better)")
    _bar(axes[1, 1], plot_df["keane_validity_mean"], "Keane Validity (higher better)")
    _bar(axes[1, 2], plot_df["keane_compactness_mean"], "Keane Compactness (higher better)")
    _bar(axes[2, 0], l2_score, "L2 Proximity Score (higher better)")
    _bar(axes[2, 1], norm_dist_score, "Normalised Distance Score (higher better)")
    axes[2, 2].axis("off")
    _bar(axes[3, 0], plot_df["temporal_consistency_mean"], "Temporal Consistency (higher better)")
    _bar(axes[3, 1], plot_df["confidence_mean"], "Confidence (higher better)")
    _bar(axes[3, 2], plot_df["keane_proximity_mean"], "Keane Proximity (lower better)", ylim=(0, max(plot_df["keane_proximity_mean"].fillna(0).max() * 1.1, 0.01)))
    _bar(axes[4, 0], plot_df["elapsed_mean"], "Average Execution Time (s, lower better)", ylim=(0, time_max))
    axes[4, 1].axis("off")
    axes[4, 2].axis("off")

    fig.suptitle("FruitFlies — IMFACT vs Native Guide vs Wachter", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart: {out_path}")

    # Canonical CF-evaluation dimensions: validity (higher better), proximity
    # (lower better), sparsity (higher better), plausibility (higher better).
    canonical_path = f"{out_path[:-4]}_canonical.png" if out_path.endswith(".png") else f"{out_path}_canonical.png"
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 4.5))
    fig2.suptitle("FruitFlies — Validity / Proximity / Sparsity / Plausibility", fontsize=13, y=1.05)

    axes2[0].bar(methods, plot_df["validity_rate"], color=colors)
    axes2[0].set_title("Validity ↑")
    axes2[0].set_ylim(0, 1.05)
    axes2[0].tick_params(axis="x", rotation=20)
    axes2[0].grid(True, axis="y", alpha=0.3)

    axes2[1].bar(methods, plot_df["l2_norm_mean"], color=colors)
    axes2[1].set_title("Proximity ↓ (L2)")
    axes2[1].tick_params(axis="x", rotation=20)
    axes2[1].grid(True, axis="y", alpha=0.3)

    axes2[2].bar(methods, plot_df["sparsity_mean"], color=colors)
    axes2[2].set_title("Sparsity ↑")
    axes2[2].set_ylim(0, 1.05)
    axes2[2].tick_params(axis="x", rotation=20)
    axes2[2].grid(True, axis="y", alpha=0.3)

    axes2[3].bar(methods, plot_df["range_validity_mean"], color=colors)
    axes2[3].set_title("Plausibility ↑")
    axes2[3].set_ylim(0, 1.05)
    axes2[3].tick_params(axis="x", rotation=20)
    axes2[3].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(canonical_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved canonical summary plot: {canonical_path}")


def plot_umap(results_df: pd.DataFrame, all_cf_outputs: dict, selected_indices: list,
              dataset_test_raw, out_path: str) -> None:
    rng = np.random.default_rng(42)
    bg_idx = sorted(
        rng.choice(len(dataset_test_raw), size=min(N_BACKGROUND, len(dataset_test_raw)), replace=False).tolist()
    )

    def _ds(arr):
        arr = np.asarray(arr, dtype=np.float32)
        return arr[..., ::DOWNSAMPLE] if DOWNSAMPLE > 1 else arr

    background_data = np.stack(
        [_ds(to_channel_first(dataset_test_raw[i][0])).reshape(-1) for i in bg_idx], axis=0
    )
    background_labels = np.array([to_class_index(dataset_test_raw[i][1]) for i in bg_idx])

    sample_validity_counts = (
        results_df[results_df["sample_idx"].isin(selected_indices)]
        .groupby("sample_idx")["validity"]
        .sum()
        .sort_values(ascending=False)
    )
    full_success = [int(i) for i, c in sample_validity_counts.items() if float(c) == float(len(METHOD_ORDER))]
    rep_idx = full_success[0] if full_success else int(sample_validity_counts.index[0])

    rep_payload = all_cf_outputs[rep_idx]
    rep_sample = to_channel_first(rep_payload["sample"]).reshape(1, -1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(background_data) - 1)),
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    )
    background_emb = reducer.fit_transform(background_data)
    sample_emb = reducer.transform(rep_sample)[0]

    sample_results = results_df[results_df["sample_idx"] == rep_idx]
    metric_lookup = {
        row["method"]: {
            "l2": float(row["l2_norm"]),
            "validity": float(row["validity"]) if not pd.isna(row["validity"]) else 0.0,
            "pred": int(row["pred_cf"]) if not pd.isna(row["pred_cf"]) else None,
        }
        for _, row in sample_results.iterrows()
    }
    orig_row = sample_results.iloc[0]
    true_label = int(orig_row["true_label"])
    initial_pred = int(orig_row["pred_orig"])

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in np.unique(background_labels):
        mask = background_labels == cls
        ax.scatter(background_emb[mask, 0], background_emb[mask, 1], s=10, alpha=0.25, label=f"class {cls}")

    ax.scatter(sample_emb[0], sample_emb[1], s=140, c="black", marker="o", edgecolors="white", label="original")

    annotation_lines = [f"sample {rep_idx}", f"true={true_label}", f"pred={initial_pred}"]
    for method_name in METHOD_ORDER:
        cf = rep_payload.get(method_name)
        if cf is None:
            continue
        cf_emb = reducer.transform(to_channel_first(cf).reshape(1, -1))[0]
        m = metric_lookup.get(method_name, {})
        l2_val = m.get("l2", np.nan)
        worked = m.get("validity", 0.0) == 1.0
        pred_cf = m.get("pred", None)

        annotation_lines.append(
            f"{method_name}: pred={pred_cf}, L2={l2_val:.3f}, {'OK' if worked else 'FAIL'}"
        )
        ax.scatter(
            cf_emb[0], cf_emb[1],
            s=150, c=METHOD_COLORS[method_name], marker="X",
            edgecolors="white", linewidths=1.0,
            label=f"{method_name} (pred={pred_cf}, {'OK' if worked else 'FAIL'})",
        )
        ax.plot([sample_emb[0], cf_emb[0]], [sample_emb[1], cf_emb[1]],
                color=METHOD_COLORS[method_name], alpha=0.5)

    ax.text(0.02, 0.98, "\n".join(annotation_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    ax.set_title(f"UMAP projection — FruitFlies sample {rep_idx}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP plot: {out_path}")


def plot_waveforms(results_df: pd.DataFrame, all_cf_outputs: dict, selected_indices: list, out_path: str) -> None:
    sample_validity_counts = (
        results_df[results_df["sample_idx"].isin(selected_indices)]
        .groupby("sample_idx")["validity"]
        .sum()
        .sort_values(ascending=False)
    )
    full_success = [int(i) for i, c in sample_validity_counts.items() if float(c) == float(len(METHOD_ORDER))]
    rep_idx = full_success[0] if full_success else int(sample_validity_counts.index[0])

    rep_payload = all_cf_outputs[rep_idx]
    x = to_channel_first(rep_payload["sample"])[0]
    x_axis = np.arange(len(x))

    sample_results = results_df[results_df["sample_idx"] == rep_idx]
    info_lookup = {
        row["method"]: {"validity": float(row["validity"]) if not pd.isna(row["validity"]) else 0.0, "pred": int(row["pred_cf"]) if not pd.isna(row["pred_cf"]) else None}
        for _, row in sample_results.iterrows()
    }
    orig_row = sample_results.iloc[0]
    true_label = int(orig_row["true_label"])
    initial_pred = int(orig_row["pred_orig"])

    n_rows = 1 + len(METHOD_ORDER)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.8 * n_rows), sharex=True)

    status_parts = [
        f"{m}: {'OK' if info_lookup.get(m, {}).get('validity', 0.0) == 1.0 else 'FAIL'}" for m in METHOD_ORDER
    ]
    fig.suptitle(
        f"Waveforms for FruitFlies sample {rep_idx} | true={true_label} | pred={initial_pred} | " + " | ".join(status_parts),
        fontsize=11, y=0.995,
    )

    axes[0].plot(x, label="original", color="black", linewidth=1.8, alpha=0.75)
    axes[0].set_title(f"Original (true={true_label}, pred={initial_pred})")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    for i, method_name in enumerate(METHOD_ORDER, start=1):
        ax = axes[i]
        cf = rep_payload.get(method_name)
        info = info_lookup.get(method_name, {})
        worked = info.get("validity", 0.0) == 1.0
        pred_cf = info.get("pred", None)

        ax.plot(x, label="original", color="black", linewidth=1.0, linestyle="--", alpha=0.25)
        if cf is None:
            ax.text(0.5, 0.5, f"{method_name}: no counterfactual", ha="center", va="center", transform=ax.transAxes)
        else:
            cf_line = to_channel_first(cf)[0]
            ax.plot(cf_line, label=method_name, linewidth=1.6, alpha=0.95, color="#1f77b4")

        ax.set_title(f"{method_name} (pred={pred_cf}, true={true_label}, {'OK' if worked else 'FAIL'})")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time step")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved waveform plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Compare IMFACT vs Native Guide vs Wachter on FruitFlies")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of correctly classified test samples to evaluate (default: 50)")
    parser.add_argument("--out-dir", type=str, default=os.path.join(SCRIPT_DIR, "results"),
                        help="Directory for saved outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-wachter", action="store_true",
                        help="Exclude Wachter from the comparison (useful in memory-constrained environments).")
    parser.add_argument("--exclude-glacier", action="store_true",
                        help="Exclude Glacier from the comparison (useful for very long series).")
    parser.add_argument("--exclude-mascots", action="store_true",
                        help="Exclude MASCOTS from the comparison (useful for very long series).")
    parser.add_argument("--max-search-samples", type=int, default=500,
                        help="Max dataset size passed to native_guide/wachter to avoid OOM (default: 500).")
    return parser.parse_args()


def main():
    global METHOD_ORDER
    args = parse_args()

    METHOD_ORDER = list(_METHOD_ORDER_DEFAULT)
    if args.exclude_wachter:
        METHOD_ORDER = [m for m in METHOD_ORDER if m != "wachter"]
    if args.exclude_glacier:
        METHOD_ORDER = [m for m in METHOD_ORDER if m != "glacier"]
    if args.exclude_mascots:
        METHOD_ORDER = [m for m in METHOD_ORDER if m != "mascots"]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Repo root: {REPO_ROOT}")

    _, dataset_train_raw = _load_fruitflies("train")
    _, dataset_test_raw = _load_fruitflies("test")

    output_classes = dataset_train_raw.y_shape[1]
    raw_length = int(dataset_train_raw.X_shape[2])
    dataset_length = raw_length // DOWNSAMPLE if DOWNSAMPLE > 1 else raw_length

    dataset_train = _DownsampledView(dataset_train_raw, DOWNSAMPLE)
    dataset_test = _DownsampledView(dataset_test_raw, DOWNSAMPLE)

    model_path = os.path.join(REPO_ROOT, "models", f"simple_cnn_fruitflies_{output_classes}_len{dataset_length}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = SimpleCNN(output_channels=output_classes, input_length=dataset_length).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {model_path}")

    y_true, y_pred = [], []
    with torch.no_grad():
        for sample, label in dataset_test:
            scores = predict_scores(model, np.asarray(sample, dtype=np.float32), device)
            y_true.append(to_class_index(label))
            y_pred.append(int(np.argmax(scores)))
    test_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Train: {len(dataset_train)} | Test: {len(dataset_test)} | Macro F1: {test_f1:.4f}")

    reference_count = min(256, len(dataset_train))
    reference_data = np.stack(
        [np.asarray(dataset_train[i][0], dtype=np.float32) for i in range(reference_count)], axis=0
    )

    selected_indices = select_correct_indices(model, dataset_test, max_count=args.n_samples, device=device)
    print(f"Evaluating {len(selected_indices)} correctly classified samples …")

    search_dataset = _SubsetView(dataset_test, args.max_search_samples)
    print(f"Search dataset capped at {len(search_dataset)} samples for all methods (full test: {len(dataset_test)}).")

    results_df, all_cf_outputs = evaluate(selected_indices, dataset_test, model, device, reference_data, search_dataset=search_dataset)

    summary_df = build_summary(results_df)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    csv_path = os.path.join(args.out_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV: {csv_path}")

    summary_df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    plot_bar_metrics(summary_df, os.path.join(args.out_dir, "bar_metrics.png"))
    plot_umap(results_df, all_cf_outputs, selected_indices, dataset_test_raw,
              os.path.join(args.out_dir, "umap_projection.png"))
    plot_waveforms(results_df, all_cf_outputs, selected_indices, os.path.join(args.out_dir, "waveforms.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
