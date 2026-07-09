"""Compare IMFACT vs Native Guide vs Wachter on any UCR/UEA time-series dataset.

Evaluates counterfactual methods on correctly-classified test samples and saves
metric summary CSVs, bar charts, a UMAP projection, and waveform plots.

Usage:
    python compare_ucr.py --dataset GunPoint
    python compare_ucr.py --dataset FaultDetectionA --n-samples 50 --out-dir ./results
    python compare_ucr.py --dataset FruitFlies --exclude-wachter --max-search-samples 300
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
import warnings

import matplotlib.pyplot as plt
import numpy as np
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
from cfts.metrics import (
    autocorrelation_preservation,
    dtw_distance,
    evaluate_keane_metrics,
    feature_range_validity,
    l2_distance,
    normalized_distance,
    percentage_changed_points,
    prediction_change,
    temporal_consistency,
)

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({"font.size": 14})

METHOD_COLORS = {
    "imfact_variance_nun3": "#e63946",
    "native_guide": "#1d3557",
    "wachter": "#ff7f11",
}
BAR_COLORS = ["#2a9d8f", "#457b9d", "#e76f51"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _ensure_dataset_files(dataset_name: str, data_root: str) -> str:
    dataset_dir = os.path.join(data_root, dataset_name)
    train_file = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(dataset_dir, f"{dataset_name}_TEST.ts")

    if os.path.exists(train_file) and os.path.exists(test_file):
        return data_root

    os.makedirs(dataset_dir, exist_ok=True)
    archive_path = os.path.join(data_root, f"{dataset_name}.zip")
    if not os.path.exists(archive_path):
        url = f"https://timeseriesclassification.com/aeon-toolkit/{dataset_name}.zip"
        print(f"Downloading {dataset_name} from {url} ...")
        urllib.request.urlretrieve(url, archive_path)

    print(f"Extracting {dataset_name} ...")
    subprocess.run(
        ["unzip", "-j", "-o", archive_path,
         f"{dataset_name}_TRAIN.ts", f"{dataset_name}_TEST.ts",
         "-d", dataset_dir],
        check=True,
    )
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        raise FileNotFoundError(f"Failed to prepare {dataset_name} .ts files in {dataset_dir}")
    return data_root


def _load_dataset(dataset_name: str, split: str, batch_size: int = 256, shuffle: bool = False):
    data_root = os.path.join(REPO_ROOT, "data", "UCR")
    try:
        X, y = load_classification(name=dataset_name, split=split, extract_path=data_root)
    except NotImplementedError as exc:
        print(f"Zip extraction fallback for {dataset_name}: {exc}")
        data_root = _ensure_dataset_files(dataset_name, data_root)
        X, y = load_classification(name=dataset_name, split=split, extract_path=data_root)

    encoder = OneHotEncoder(categories="auto", sparse_output=False)
    y_enc = encoder.fit_transform(np.expand_dims(y, axis=-1))
    dataset = TimeSeriesDataset(X=X, y=y_enc, name=dataset_name, mapping=encoder.categories_)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_sparse)
    return dataloader, dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.reshape(inputs.shape[0], 1, -1).float().to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, labels.argmax(dim=-1))
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
    return running_loss / max(1, len(dataloader))


def _validate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels, running_loss = [], [], 0.0
    for inputs, labels in dataloader:
        inputs = inputs.reshape(inputs.shape[0], 1, -1).float().to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            preds = model(inputs)
            running_loss += float(criterion(preds, labels.argmax(dim=-1)).item())
        all_preds.extend(preds.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.argmax(dim=-1).cpu().numpy())
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0.0
    return running_loss / max(1, len(dataloader)), acc


def _load_or_train(model_path: str, model, dataloader_train, dataloader_test, device, epochs: int):
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    print(f"No checkpoint at {model_path}; training for {epochs} epochs ...")
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        train_loss = _train_epoch(model, dataloader_train, criterion, optimizer, device)
        val_loss, val_acc = _validate(model, dataloader_test, criterion, device)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_channel_first(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr if arr.shape[0] <= arr.shape[1] else arr.T
    raise ValueError(f"Unsupported shape: {arr.shape}")


def _to_class_index(label) -> int:
    arr = np.asarray(label)
    return int(arr) if arr.ndim == 0 else int(np.argmax(arr))


def _predict(model, sample: np.ndarray, device) -> np.ndarray:
    ts = torch.tensor(np.asarray(sample, dtype=np.float32), device=device)
    if ts.ndim == 1:
        ts = ts.reshape(1, 1, -1)
    elif ts.ndim == 2:
        if ts.shape[0] > ts.shape[1]:
            ts = ts.T
        ts = ts.unsqueeze(0)
    with torch.no_grad():
        return model(ts).detach().cpu().numpy().reshape(-1)


def _model_wrapper(model, device):
    def wrapped(ts: np.ndarray) -> np.ndarray:
        return _predict(model, np.asarray(ts, dtype=np.float32), device)
    return wrapped


def _select_correct_indices(model, dataset, max_count: int, device) -> list[int]:
    selected = []
    for idx in range(len(dataset)):
        sample, label = dataset[idx]
        scores = _predict(model, np.asarray(sample, dtype=np.float32), device)
        if int(np.argmax(scores)) == _to_class_index(label):
            selected.append(idx)
        if len(selected) >= max_count:
            break
    return selected


def _infer_target_class(scores: np.ndarray) -> int:
    return int(np.argsort(scores)[::-1][1])


class _SubsetView:
    """Limits a dataset to its first max_size items to avoid OOM on large datasets."""

    def __init__(self, dataset, max_size: int):
        self._dataset = dataset
        self._max_size = min(max_size, len(dataset))

    def __len__(self):
        return self._max_size

    def __getitem__(self, idx):
        if idx >= self._max_size:
            raise IndexError(idx)
        return self._dataset[idx]

    def __getattr__(self, name):
        return getattr(self._dataset, name)


# ---------------------------------------------------------------------------
# CF runners
# ---------------------------------------------------------------------------

def _run_imfact(sample, dataset, model, target_class):
    return imfact_cf(
        sample=sample, dataset=dataset, model=model,
        method="variance", guide_class=target_class,
        step=0.05, max_iter=200, max_imfs=10,
        n_nuns=3, nun_switch="cycle", verbose=False,
    )


def _run_native_guide(sample, dataset, model, target_class):
    return native_guide_uni_cf(sample=sample, dataset=dataset, model=model)


def _run_wachter(sample, dataset, model, target_class):
    return wachter_gradient_cf(
        sample=sample, dataset=dataset, model=model,
        target=target_class, max_cfs=400, distance="euclidean", verbose=False,
    )


RUNNERS = {
    "imfact_variance_nun3": _run_imfact,
    "native_guide": _run_native_guide,
    "wachter": _run_wachter,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    selected_indices: list[int],
    dataset_test,
    model,
    device,
    reference_data: np.ndarray,
    method_order: list[str],
    search_dataset=None,
) -> tuple[pd.DataFrame, dict]:
    model_fn = _model_wrapper(model, device)
    records = []
    all_cf_outputs: dict = {}

    for idx in selected_indices:
        sample, label = dataset_test[idx]
        sample = np.asarray(sample, dtype=np.float32)
        scores_orig = _predict(model, sample, device)
        pred_orig = int(np.argmax(scores_orig))
        true_label = _to_class_index(label)
        target_class = _infer_target_class(scores_orig)

        all_cf_outputs[idx] = {"sample": sample, "true_label": true_label, "target_class": target_class}

        for method_name in method_order:
            base = {
                "sample_idx": idx, "method": method_name,
                "true_label": true_label, "pred_orig": pred_orig, "target_class": target_class,
                "pred_cf": None,
                "l2_norm": np.nan, "dtw_proximity": np.nan, "normalized_distance": np.nan,
                "sparsity": np.nan, "range_validity": np.nan, "autocorr": np.nan,
                "temporal_consistency": np.nan, "confidence": np.nan, "validity": 0.0,
                "keane_validity": np.nan, "keane_proximity": np.nan, "keane_compactness": np.nan,
                "error": None,
            }

            try:
                ds = dataset_test if method_name == "imfact_variance_nun3" else (search_dataset or dataset_test)
                cf, pred_cf_scores = RUNNERS[method_name](sample, ds, model, target_class)
            except Exception as exc:
                records.append({**base, "error": f"{type(exc).__name__}: {exc}"})
                continue

            if cf is None or pred_cf_scores is None:
                records.append({**base, "error": "Method returned None"})
                continue

            cf = np.asarray(cf, dtype=np.float32)
            _scores = np.asarray(pred_cf_scores).reshape(-1)
            pred_cf = int(np.argmax(_scores))
            _exp = np.exp(_scores - _scores.max())
            cf_confidence = float(_exp[pred_cf] / _exp.sum())

            s_cf = _to_channel_first(sample)
            c_cf = _to_channel_first(cf)

            keane = evaluate_keane_metrics(
                original_ts_list=s_cf, counterfactual_ts_list=c_cf,
                model=model_fn, target_classes=int(target_class),
            )

            records.append({
                **base,
                "pred_cf": pred_cf,
                "l2_norm": float(l2_distance(s_cf, c_cf)),
                "dtw_proximity": float(dtw_distance(s_cf, c_cf)),
                "normalized_distance": float(normalized_distance(s_cf.reshape(-1), c_cf.reshape(-1))),
                "sparsity": float(1.0 - percentage_changed_points(s_cf, c_cf)),
                "range_validity": float(feature_range_validity(c_cf, reference_data)),
                "autocorr": float(autocorrelation_preservation(s_cf, c_cf)),
                "temporal_consistency": float(temporal_consistency(c_cf)),
                "confidence": cf_confidence,
                "validity": float(prediction_change(s_cf, c_cf, model_fn, target_class=target_class)),
                "keane_validity": float(keane["validity"]),
                "keane_proximity": float(keane["proximity"]),
                "keane_compactness": float(keane["compactness"]),
            })
            all_cf_outputs[idx][method_name] = cf

    return pd.DataFrame(records), all_cf_outputs


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    successful_df = results_df[results_df["validity"] == 1.0].copy()
    agg_all = (
        results_df.groupby("method", dropna=False)
        .agg(n_total=("sample_idx", "count"),
             validity_rate=("validity", "mean"))
        .reset_index()
    )
    agg_suc = (
        successful_df.groupby("method", dropna=False)
        .agg(
            l2_norm_mean=("l2_norm", "mean"),
            dtw_proximity_mean=("dtw_proximity", "mean"),
            normalized_distance_mean=("normalized_distance", "mean"),
            sparsity_mean=("sparsity", "mean"),
            range_validity_mean=("range_validity", "mean"),
            autocorr_mean=("autocorr", "mean"),
            temporal_consistency_mean=("temporal_consistency", "mean"),
            confidence_mean=("confidence", "mean"),
            keane_validity_mean=("keane_validity", "mean"),
            keane_proximity_mean=("keane_proximity", "mean"),
            keane_compactness_mean=("keane_compactness", "mean"),
        )
        .reset_index()
    )
    summary = agg_all.merge(agg_suc, on="method", how="left")
    summary["validity_rate"] = summary["validity_rate"].fillna(0.0)
    return summary.sort_values("method").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bar_metrics(summary_df: pd.DataFrame, dataset_name: str, out_path: str) -> None:
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
    dtw_score = 1.0 / (1.0 + plot_df["dtw_proximity_mean"])
    norm_dist_score = 1.0 / (1.0 + plot_df["normalized_distance_mean"])
    kp_max = max(float(plot_df["keane_proximity_mean"].fillna(0).max()) * 1.1, 0.01)

    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    _bar(axes[0, 0], plot_df["validity_rate"], "Validity (higher better)")
    _bar(axes[0, 1], plot_df["sparsity_mean"], "Sparsity (higher better)")
    _bar(axes[0, 2], plot_df["range_validity_mean"], "Range Validity (higher better)")
    _bar(axes[1, 0], plot_df["autocorr_mean"], "Autocorr Preservation (higher better)")
    _bar(axes[1, 1], plot_df["keane_validity_mean"], "Keane Validity (higher better)")
    _bar(axes[1, 2], plot_df["keane_compactness_mean"], "Keane Compactness (higher better)")
    _bar(axes[2, 0], l2_score, "L2 Proximity Score (higher better)")
    _bar(axes[2, 1], dtw_score, "DTW Proximity Score (higher better)")
    _bar(axes[2, 2], norm_dist_score, "Normalised Distance Score (higher better)")
    _bar(axes[3, 0], plot_df["temporal_consistency_mean"], "Temporal Consistency (higher better)")
    _bar(axes[3, 1], plot_df["confidence_mean"], "Confidence (higher better)")
    _bar(axes[3, 2], plot_df["keane_proximity_mean"], "Keane Proximity (lower better)", ylim=(0, kp_max))

    fig.suptitle(f"{dataset_name} — IMFACT vs Native Guide vs Wachter", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart: {out_path}")


def plot_umap(
    results_df: pd.DataFrame,
    all_cf_outputs: dict,
    selected_indices: list[int],
    dataset_test,
    dataset_name: str,
    method_order: list[str],
    out_path: str,
    n_background: int = 512,
) -> None:
    rng = np.random.default_rng(42)
    bg_idx = sorted(
        rng.choice(len(dataset_test), size=min(n_background, len(dataset_test)), replace=False).tolist()
    )
    background_data = np.stack(
        [_to_channel_first(dataset_test[i][0]).reshape(-1) for i in bg_idx], axis=0
    )
    background_labels = np.array([_to_class_index(dataset_test[i][1]) for i in bg_idx])

    sample_validity_counts = (
        results_df[results_df["sample_idx"].isin(selected_indices)]
        .groupby("sample_idx")["validity"].sum()
        .sort_values(ascending=False)
    )
    full_success = [int(i) for i, c in sample_validity_counts.items() if float(c) == float(len(method_order))]
    rep_idx = full_success[0] if full_success else int(sample_validity_counts.index[0])

    rep_payload = all_cf_outputs[rep_idx]
    rep_sample = _to_channel_first(rep_payload["sample"]).reshape(1, -1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(background_data) - 1)),
        min_dist=0.15, metric="euclidean", random_state=42,
    )
    background_emb = reducer.fit_transform(background_data)
    sample_emb = reducer.transform(rep_sample)[0]

    sample_results = results_df[results_df["sample_idx"] == rep_idx]
    metric_lookup = {
        row["method"]: {
            "l2": float(row["l2_norm"]),
            "dtw": float(row["dtw_proximity"]),
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
    for method_name in method_order:
        cf = rep_payload.get(method_name)
        if cf is None:
            continue
        cf_emb = reducer.transform(_to_channel_first(cf).reshape(1, -1))[0]
        m = metric_lookup.get(method_name, {})
        l2_val = m.get("l2", np.nan)
        dtw_val = m.get("dtw", np.nan)
        worked = m.get("validity", 0.0) == 1.0
        pred_cf = m.get("pred", None)
        color = METHOD_COLORS.get(method_name, "#888888")

        annotation_lines.append(
            f"{method_name}: pred={pred_cf}, L2={l2_val:.3f}, DTW={dtw_val:.3f}, {'OK' if worked else 'FAIL'}"
        )
        ax.scatter(cf_emb[0], cf_emb[1], s=150, c=color, marker="X",
                   edgecolors="white", linewidths=1.0,
                   label=f"{method_name} (pred={pred_cf}, {'OK' if worked else 'FAIL'})")
        ax.plot([sample_emb[0], cf_emb[0]], [sample_emb[1], cf_emb[1]], color=color, alpha=0.5)

    ax.text(0.02, 0.98, "\n".join(annotation_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    ax.set_title(f"UMAP projection — {dataset_name} sample {rep_idx}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP plot: {out_path}")


def plot_waveforms(
    results_df: pd.DataFrame,
    all_cf_outputs: dict,
    selected_indices: list[int],
    dataset_name: str,
    method_order: list[str],
    out_path: str,
) -> None:
    sample_validity_counts = (
        results_df[results_df["sample_idx"].isin(selected_indices)]
        .groupby("sample_idx")["validity"].sum()
        .sort_values(ascending=False)
    )
    full_success = [int(i) for i, c in sample_validity_counts.items() if float(c) == float(len(method_order))]
    rep_idx = full_success[0] if full_success else int(sample_validity_counts.index[0])

    rep_payload = all_cf_outputs[rep_idx]
    x = _to_channel_first(rep_payload["sample"])[0]
    x_axis = np.arange(len(x))

    sample_results = results_df[results_df["sample_idx"] == rep_idx]
    info_lookup = {
        row["method"]: {
            "validity": float(row["validity"]) if not pd.isna(row["validity"]) else 0.0,
            "pred": int(row["pred_cf"]) if not pd.isna(row["pred_cf"]) else None,
        }
        for _, row in sample_results.iterrows()
    }
    orig_row = sample_results.iloc[0]
    true_label = int(orig_row["true_label"])
    initial_pred = int(orig_row["pred_orig"])

    n_rows = 1 + len(method_order)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.8 * n_rows), sharex=True)

    status_parts = [
        f"{m}: {'OK' if info_lookup.get(m, {}).get('validity', 0.0) == 1.0 else 'FAIL'}"
        for m in method_order
    ]
    fig.suptitle(
        f"Waveforms — {dataset_name} sample {rep_idx} | true={true_label} | pred={initial_pred} | "
        + " | ".join(status_parts),
        fontsize=11, y=0.995,
    )

    axes[0].plot(x, label="original", color="black", linewidth=1.8, alpha=0.75)
    axes[0].set_title(f"Original (true={true_label}, pred={initial_pred})")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    for i, method_name in enumerate(method_order, start=1):
        ax = axes[i]
        cf = rep_payload.get(method_name)
        info = info_lookup.get(method_name, {})
        worked = info.get("validity", 0.0) == 1.0
        pred_cf = info.get("pred", None)

        ax.plot(x, label="original", color="black", linewidth=1.0, linestyle="--", alpha=0.25)
        if cf is None:
            ax.text(0.5, 0.5, f"{method_name}: no counterfactual",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            cf_line = _to_channel_first(cf)[0]
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
    parser = argparse.ArgumentParser(
        description="Compare IMFACT vs Native Guide vs Wachter on any UCR/UEA time-series dataset."
    )
    parser.add_argument("--dataset", required=True,
                        help="UCR/UEA dataset name (e.g. GunPoint, FaultDetectionA, FruitFlies)")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of correctly classified test samples to evaluate (default: 50)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: results/<dataset_lower>_compare/)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-wachter", action="store_true",
                        help="Exclude Wachter from the comparison (recommended for long series > 1000 pts)")
    parser.add_argument("--max-search-samples", type=int, default=500,
                        help="Max dataset size passed to native_guide/wachter to avoid OOM (default: 500)")
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Training epochs if no model checkpoint is found (default: 30)")
    parser.add_argument("--model-file", type=str, default=None,
                        help="Path to model checkpoint (default: models/simple_cnn_<dataset>_...pth)")
    parser.add_argument("--n-background", type=int, default=512,
                        help="Background samples for UMAP projection (default: 512)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset
    dataset_lower = dataset_name.lower()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, "results", f"{dataset_lower}_compare")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {dataset_name} | Device: {device} | Repo root: {REPO_ROOT}")

    _, dataset_train = _load_dataset(dataset_name, "train")
    _, dataset_test = _load_dataset(dataset_name, "test")

    output_classes = dataset_train.y_shape[1]
    series_length = int(dataset_train.X_shape[2])
    print(f"Classes: {output_classes} | Series length: {series_length} | "
          f"Train: {len(dataset_train)} | Test: {len(dataset_test)}")

    model_path = args.model_file or os.path.join(
        REPO_ROOT, "models", f"simple_cnn_{dataset_lower}_{output_classes}_len{series_length}.pth"
    )
    model = SimpleCNN(output_channels=output_classes, input_length=series_length).to(device)
    dataloader_train, _ = _load_dataset(dataset_name, "train", shuffle=True)
    dataloader_test_dl, _ = _load_dataset(dataset_name, "test", shuffle=False)
    model = _load_or_train(model_path, model, dataloader_train, dataloader_test_dl, device, args.train_epochs)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for sample, label in dataset_test:
            scores = _predict(model, np.asarray(sample, dtype=np.float32), device)
            y_true.append(_to_class_index(label))
            y_pred.append(int(np.argmax(scores)))
    test_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Macro F1: {test_f1:.4f}")

    if series_length > 1000 and not args.exclude_wachter:
        print(
            f"WARNING: series length {series_length} > 1000. "
            "Wachter gradient descent may run out of memory. "
            "Pass --exclude-wachter to skip it."
        )

    method_order = ["imfact_variance_nun3", "native_guide"]
    if not args.exclude_wachter:
        method_order.append("wachter")
    print(f"Methods: {method_order}")

    reference_count = min(256, len(dataset_train))
    reference_data = np.stack(
        [np.asarray(dataset_train[i][0], dtype=np.float32) for i in range(reference_count)], axis=0
    )

    selected_indices = _select_correct_indices(model, dataset_test, args.n_samples, device)
    print(f"Evaluating {len(selected_indices)} correctly classified samples ...")

    search_dataset = _SubsetView(dataset_test, args.max_search_samples)
    if len(search_dataset) < len(dataset_test):
        print(f"Search dataset capped at {len(search_dataset)} samples for native_guide/wachter")

    results_df, all_cf_outputs = evaluate(
        selected_indices, dataset_test, model, device,
        reference_data, method_order, search_dataset,
    )

    summary_df = build_summary(results_df)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    results_df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"\nSaved CSVs to {out_dir}")

    plot_bar_metrics(summary_df, dataset_name, os.path.join(out_dir, "bar_metrics.png"))
    plot_umap(
        results_df, all_cf_outputs, selected_indices, dataset_test,
        dataset_name, method_order, os.path.join(out_dir, "umap_projection.png"),
        n_background=args.n_background,
    )
    plot_waveforms(
        results_df, all_cf_outputs, selected_indices,
        dataset_name, method_order, os.path.join(out_dir, "waveforms.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
