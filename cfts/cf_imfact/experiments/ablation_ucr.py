"""IMFACT ablation study on any UCR/UEA time-series dataset.

Runs two ablation suites on correctly classified test samples:
  1) IMF selection method ablation (single-guide: distance, fingerprint, variance, extremes, maxmin, coarse_to_fine)
  2) Multi-NUN hyperparameter ablation (n_nuns and switching policy)

For a configurable subset of samples, per-sample line plots and UMAP projections are saved.

Usage:
    python ablation_ucr.py --dataset GunPoint
    python ablation_ucr.py --dataset FaultDetectionA --max-samples 20 --multi-nun-counts 2,3,5
    python ablation_ucr.py --dataset ECG200 --out-prefix ./results/ecg200_ablation/ecg200
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14})
import numpy as np
import umap
import torch
import torch.nn as nn
from aeon.datasets import load_classification
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _find_repo_root(start_dir: str) -> str:
    current = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(current, "pyproject.toml")) and os.path.isdir(os.path.join(current, "cfts")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError("Could not locate repository root from script location.")


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")

for _p in (REPO_ROOT, EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from examples.base.model import SimpleCNN
from examples.base.data import TimeSeriesDataset, collate_sparse
from cfts.cf_imfact.imfact import _psd, imfact_cf
from cfts.metrics import (
    autocorrelation_preservation,
    dtw_distance,
    evaluate_keane_metrics,
    feature_range_validity,
    normalized_distance,
    percentage_changed_points,
    prediction_change,
    temporal_consistency,
)


BASE_METHOD_CONFIGS: Dict[str, Dict[str, object]] = {
    "distance": {"method": "distance"},
    "fingerprint": {"method": "fingerprint"},
    "variance": {"method": "variance"},
    "extremes": {"method": "extremes"},
    "maxmin": {"method": "maxmin"},
    "coarse_to_fine": {"method": "coarse_to_fine"},
}


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


def _load_dataset(dataset_name: str, split: str, batch_size: int = 256, shuffle: bool = True):
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
# Ablation config builders
# ---------------------------------------------------------------------------

def _build_method_ablation_configs() -> Dict[str, Dict[str, object]]:
    return dict(BASE_METHOD_CONFIGS)


def _build_nun_ablation_configs(multi_nun_counts: Sequence[int]) -> Dict[str, Dict[str, object]]:
    nun_configs: Dict[str, Dict[str, object]] = {
        "distance_n1_cycle": {"method": "distance", "n_nuns": 1, "nun_switch": "cycle"},
    }
    for count in multi_nun_counts:
        nun_configs[f"multi_nun_cycle_n{count}"] = {
            "method": "distance", "n_nuns": count, "nun_switch": "cycle",
        }
        nun_configs[f"multi_nun_closest_n{count}"] = {
            "method": "distance", "n_nuns": count, "nun_switch": "closest_psd",
        }
    return nun_configs


def _parse_multi_nun_counts(raw_counts: str) -> List[int]:
    counts: List[int] = []
    for token in raw_counts.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 1:
            raise ValueError("All multi-NUN counts must be integers greater than 1.")
        if value not in counts:
            counts.append(value)
    return counts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _train_epoch(model: nn.Module, dataloader, criterion, optimizer, device):
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


def _validate(model: nn.Module, dataloader, criterion, device):
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


def _load_or_train(model_file: str, model: nn.Module, dataloader_train, dataloader_test, device, epochs: int):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    if os.path.exists(model_file):
        print(f"Loading model: {model_file}")
        model.load_state_dict(torch.load(model_file, map_location=device))
        return model

    print(f"No checkpoint at {model_file}; training for {epochs} epochs ...")
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
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    torch.save(model.state_dict(), model_file)
    print(f"Model saved: {model_file}")
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
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _to_class_index(label) -> int:
    arr = np.asarray(label)
    return int(arr) if arr.ndim == 0 else int(np.argmax(arr))


def _predict(model: nn.Module, sample: np.ndarray, device) -> np.ndarray:
    tensor = torch.tensor(np.asarray(sample, dtype=np.float32), device=device)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 1, -1)
    elif tensor.ndim == 2:
        if tensor.shape[0] > tensor.shape[1]:
            tensor = tensor.T
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        return model(tensor).cpu().numpy().reshape(-1)


def _format_prediction(prediction) -> Tuple[int, float]:
    pred_np = np.asarray(prediction).reshape(-1)
    if np.any(pred_np < 0) or not np.isclose(np.sum(pred_np), 1.0, atol=0.1):
        exp = np.exp(pred_np - np.max(pred_np))
        pred_np = exp / np.sum(exp)
    return int(np.argmax(pred_np)), float(np.max(pred_np))


def _select_correct_samples(model: nn.Module, dataset, max_samples: int, device) -> List[int]:
    selected = []
    for idx in range(len(dataset)):
        sample, label = dataset[idx]
        pred = _predict(model, np.asarray(sample, dtype=np.float32), device)
        if int(np.argmax(pred)) == _to_class_index(label):
            selected.append(idx)
        if len(selected) >= max_samples:
            break
    return selected


def _select_native_guide(sample: np.ndarray, dataset, guide_class: int | None) -> Tuple[np.ndarray, int]:
    sample_cl = _to_channel_first(np.asarray(sample, dtype=np.float32))
    sample_psd = _psd(sample_cl[0])

    candidates = []
    candidate_labels = []
    for idx in range(len(dataset)):
        cand_x, cand_y = dataset[idx]
        cand_label = _to_class_index(cand_y)
        if guide_class is not None and cand_label != int(guide_class):
            continue
        cand_x = _to_channel_first(np.asarray(cand_x, dtype=np.float32))
        candidates.append(cand_x)
        candidate_labels.append(cand_label)

    if not candidates:
        raise ValueError(f"No native guide found for guide_class={guide_class}")

    js_dists = [jensenshannon(sample_psd, _psd(c[0])) for c in candidates]
    best_idx = int(np.argmin(js_dists))
    return candidates[best_idx], candidate_labels[best_idx]


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def _evaluate_method(
    method: str,
    method_configs: Dict[str, Dict[str, object]],
    sample: np.ndarray,
    dataset,
    reference_data: np.ndarray,
    model: nn.Module,
    target_class: int,
    device,
) -> Dict[str, object]:
    start_time = time.time()
    method_config = method_configs[method]
    try:
        cf, pred = imfact_cf(
            sample, dataset, model,
            method=method_config["method"],
            guide_class=target_class,
            step=0.05, max_iter=200, max_imfs=10,
            n_nuns=method_config.get("n_nuns", 1),
            nun_switch=method_config.get("nun_switch", "cycle"),
            verbose=False,
        )
        elapsed = time.time() - start_time
    except Exception as exc:
        return {
            "method": method,
            "elapsed": time.time() - start_time,
            "error": f"{type(exc).__name__}: {exc}",
        }

    pred_class, confidence = _format_prediction(pred)
    success = pred_class == target_class

    sample_flat = np.asarray(sample, dtype=np.float32).reshape(-1)
    cf_flat = np.asarray(cf, dtype=np.float32).reshape(-1)
    l2_dist = float(np.linalg.norm(cf_flat - sample_flat))
    pct_changed = float(100.0 * percentage_changed_points(sample, cf))
    norm_dist = float(normalized_distance(sample_flat, cf_flat))
    temp_cons = float(temporal_consistency(cf))
    range_val = float(feature_range_validity(cf, reference_data))
    autocorr = float(autocorrelation_preservation(sample, cf))

    def _model_fn(x: np.ndarray) -> np.ndarray:
        return _predict(model, np.asarray(x, dtype=np.float32), device)

    keane = evaluate_keane_metrics(
        original_ts_list=np.asarray(sample, dtype=np.float32),
        counterfactual_ts_list=np.asarray(cf, dtype=np.float32),
        model=_model_fn,
        target_classes=int(target_class),
    )

    s_ch = _to_channel_first(np.asarray(sample, dtype=np.float32))
    c_ch = _to_channel_first(np.asarray(cf, dtype=np.float32))
    dtw_dist = float(dtw_distance(s_ch, c_ch))
    validity = float(prediction_change(s_ch, c_ch, _model_fn, target_class=target_class))
    sparsity = 1.0 - pct_changed / 100.0

    return {
        "method": method, "elapsed": elapsed,
        "pred_class": pred_class, "confidence": confidence,
        "l2_distance": l2_dist, "pct_changed": pct_changed,
        "normalized_distance": norm_dist, "temporal_consistency": temp_cons,
        "range_validity": range_val, "autocorr_preservation": autocorr,
        "dtw_distance": dtw_dist, "validity": validity, "sparsity": sparsity,
        "keane_validity": float(keane["validity"]),
        "keane_proximity": float(keane["proximity"]),
        "keane_compactness": float(keane["compactness"]),
        "sample": np.asarray(sample, dtype=np.float32),
        "counterfactual": np.asarray(cf, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summarize_results(
    methods: Sequence[str],
    results: Dict[str, List[Dict[str, object]]],
) -> List[Dict[str, object]]:
    summary_rows = []
    for method in methods:
        method_rows = results[method]
        successes = [r for r in method_rows if r.get("validity", 0.0) == 1.0]
        keane_rows = [r for r in method_rows if "keane_validity" in r]
        total = len(method_rows)

        def _avg(key, rows=successes):
            return float(np.mean([r[key] for r in rows])) if rows else 0.0

        summary_rows.append({
            "method": method,
            "validity_rate": 100.0 * float(np.mean([r.get("validity", 0.0) for r in method_rows])) if method_rows else 0.0,
            "avg_confidence": _avg("confidence"),
            "avg_l2_distance": _avg("l2_distance"),
            "avg_pct_changed": _avg("pct_changed"),
            "avg_normalized_distance": _avg("normalized_distance"),
            "avg_temporal_consistency": _avg("temporal_consistency"),
            "avg_range_validity": _avg("range_validity"),
            "avg_autocorr_preservation": _avg("autocorr_preservation"),
            "avg_dtw_distance": _avg("dtw_distance"),
            "avg_validity": _avg("validity"),
            "avg_sparsity": _avg("sparsity"),
            "avg_keane_validity": float(np.mean([r["keane_validity"] for r in keane_rows])) if keane_rows else 0.0,
            "avg_keane_proximity": float(np.mean([r["keane_proximity"] for r in keane_rows])) if keane_rows else 0.0,
            "avg_keane_compactness": float(np.mean([r["keane_compactness"] for r in keane_rows])) if keane_rows else 0.0,
            "avg_time": float(np.mean([r["elapsed"] for r in method_rows])) if method_rows else 0.0,
        })
    return summary_rows


def _print_summary(title: str, methods: Sequence[str], summary_rows: Sequence[Dict[str, object]]):
    w = max(13, max(len(m) for m in methods) + 2)
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(
        f"{'Method':<{w}} {'Validity':<9} {'Conf':<7} {'L2':<9} {'%Chg':<7} "
        f"{'NormD':<8} {'TmpC':<7} {'RngV':<7} {'ACorr':<7} "
        f"{'DTW':<9} {'Valid':<7} {'Spar':<7} "
        f"{'K.Val':<7} {'K.Prx':<8} {'K.Cmp':<8} {'Time':<7}"
    )
    print("-" * 80)
    for row in summary_rows:
        print(
            f"{row['method']:<{w}} {row['validity_rate']:<9.1f} {row['avg_confidence']:<7.4f} "
            f"{row['avg_l2_distance']:<9.2f} {row['avg_pct_changed']:<7.2f} "
            f"{row['avg_normalized_distance']:<8.4f} {row['avg_temporal_consistency']:<7.4f} "
            f"{row['avg_range_validity']:<7.4f} {row['avg_autocorr_preservation']:<7.4f} "
            f"{row['avg_dtw_distance']:<9.2f} {row['avg_validity']:<7.4f} {row['avg_sparsity']:<7.4f} "
            f"{row['avg_keane_validity']:<7.4f} {row['avg_keane_proximity']:<8.4f} "
            f"{row['avg_keane_compactness']:<8.4f} {row['avg_time']:<7.4f}"
        )


def _save_summary_csv(csv_path: str, summary_rows: Sequence[Dict[str, object]]):
    fieldnames = [
        "method", "validity_rate", "avg_confidence",
        "avg_l2_distance", "avg_pct_changed", "avg_normalized_distance",
        "avg_temporal_consistency", "avg_range_validity", "avg_autocorr_preservation",
        "avg_dtw_distance", "avg_validity", "avg_sparsity",
        "avg_keane_validity", "avg_keane_proximity", "avg_keane_compactness",
        "avg_time",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Saved summary CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _mark_best_barh(ax, values, higher_is_better=True):
    if not values:
        return
    best_idx = int(np.argmax(values) if higher_is_better else np.argmin(values))
    best_bar = ax.patches[best_idx]
    best_bar.set_edgecolor("gold")
    best_bar.set_linewidth(2.5)
    best_bar.set_hatch("//")
    x_pos = best_bar.get_width()
    y_pos = best_bar.get_y() + best_bar.get_height() / 2
    ax.annotate("BEST", xy=(x_pos, y_pos), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontweight="bold", color="gold")


def _method_color(method: str) -> str:
    palette = {
        "distance": "darkorange", "fingerprint": "sienna", "variance": "crimson",
        "extremes": "purple", "maxmin": "teal", "coarse_to_fine": "black",
    }
    if method in palette:
        return palette[method]
    if method.startswith("multi_nun_cycle"):
        return "goldenrod"
    if method.startswith("multi_nun_closest"):
        return "royalblue"
    return "darkorange"


def _save_summary_plot(
    plot_path: str,
    summary_rows: Sequence[Dict[str, object]],
    title: str,
):
    names = [r["method"] for r in summary_rows]
    validity_rates = [r["validity_rate"] for r in summary_rows]
    confidences = [r["avg_confidence"] for r in summary_rows]
    l2_distances = [r["avg_l2_distance"] for r in summary_rows]
    pct_changed = [r["avg_pct_changed"] for r in summary_rows]
    norm_distances = [r["avg_normalized_distance"] for r in summary_rows]
    range_validities = [r["avg_range_validity"] for r in summary_rows]
    keane_validities = [r["avg_keane_validity"] for r in summary_rows]
    keane_proximities = [r["avg_keane_proximity"] for r in summary_rows]
    keane_compactnesses = [r["avg_keane_compactness"] for r in summary_rows]
    dtw_distances = [r["avg_dtw_distance"] for r in summary_rows]
    validities = [r["avg_validity"] for r in summary_rows]
    sparsities = [r["avg_sparsity"] for r in summary_rows]

    fig_height = max(20, 0.85 * len(names) + 16)
    fig, axes = plt.subplots(4, 3, figsize=(18, fig_height))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0, 0].barh(names, validity_rates, color="steelblue")
    axes[0, 0].set_title("Validity Rate (%)")
    axes[0, 0].set_xlim([0, 105])
    for idx, v in enumerate(validity_rates):
        axes[0, 0].text(v + 1, idx, f"{v:.1f}%", va="center")
    _mark_best_barh(axes[0, 0], validity_rates, higher_is_better=True)

    axes[0, 1].barh(names, confidences, color="seagreen")
    axes[0, 1].set_title("Average Confidence")
    axes[0, 1].set_xlim(left=0)
    _mark_best_barh(axes[0, 1], confidences, higher_is_better=True)

    axes[0, 2].barh(names, keane_validities, color="forestgreen")
    axes[0, 2].set_title("Keane Validity")
    axes[0, 2].set_xlim(left=0, right=1.05)
    _mark_best_barh(axes[0, 2], keane_validities, higher_is_better=True)

    axes[1, 0].barh(names, l2_distances, color="darkorange")
    axes[1, 0].set_title("Average L2 Distance")
    axes[1, 0].set_xlim(left=0)
    _mark_best_barh(axes[1, 0], l2_distances, higher_is_better=False)

    axes[1, 1].barh(names, pct_changed, color="slateblue")
    axes[1, 1].set_title("Percentage Changed (%)")
    axes[1, 1].set_xlim(left=0)
    _mark_best_barh(axes[1, 1], pct_changed, higher_is_better=False)

    axes[1, 2].barh(names, keane_proximities, color="sienna")
    axes[1, 2].set_title("Keane Proximity")
    axes[1, 2].set_xlim(left=0)
    _mark_best_barh(axes[1, 2], keane_proximities, higher_is_better=False)

    axes[2, 0].barh(names, norm_distances, color="darkviolet")
    axes[2, 0].set_title("Normalized Distance")
    axes[2, 0].set_xlim(left=0)
    _mark_best_barh(axes[2, 0], norm_distances, higher_is_better=False)

    axes[2, 1].barh(names, range_validities, color="teal")
    axes[2, 1].set_title("Range Validity")
    axes[2, 1].set_xlim(left=0, right=1.05)
    _mark_best_barh(axes[2, 1], range_validities, higher_is_better=True)

    axes[2, 2].barh(names, keane_compactnesses, color="seagreen")
    axes[2, 2].set_title("Keane Compactness")
    axes[2, 2].set_xlim(left=0, right=1.05)
    _mark_best_barh(axes[2, 2], keane_compactnesses, higher_is_better=True)

    axes[3, 0].barh(names, dtw_distances, color="royalblue")
    axes[3, 0].set_title("Average DTW Distance")
    axes[3, 0].set_xlim(left=0)
    _mark_best_barh(axes[3, 0], dtw_distances, higher_is_better=False)

    axes[3, 1].barh(names, validities, color="crimson")
    axes[3, 1].set_title("Validity (prediction_change)")
    axes[3, 1].set_xlim(left=0, right=1.05)
    _mark_best_barh(axes[3, 1], validities, higher_is_better=True)

    axes[3, 2].barh(names, sparsities, color="goldenrod")
    axes[3, 2].set_title("Sparsity (higher better)")
    axes[3, 2].set_xlim(left=0, right=1.05)
    _mark_best_barh(axes[3, 2], sparsities, higher_is_better=True)

    for ax in axes.flat:
        ax.grid(True, axis="x", alpha=0.25)

    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")


def _save_sample_line_plot(
    sample: np.ndarray,
    label,
    original_class: int,
    target_class: int,
    outcomes: List[Dict[str, object]],
    methods: Sequence[str],
    output_prefix: str,
    sample_idx: int,
    dataset_name: str,
):
    sample_cf = _to_channel_first(sample)
    n_rows = 1 + len(methods)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.1 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    true_class = _to_class_index(label)
    fig.suptitle(
        f"IMFACT Ablation — {dataset_name} | sample={sample_idx} | "
        f"original={original_class} | target={target_class}",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(sample_cf.shape[1])
    axes[0].plot(x, sample_cf[0], color="steelblue", linewidth=1.4)
    axes[0].set_title(f"Original — true class: {true_class}", fontsize=10)
    axes[0].grid(True, alpha=0.25)

    for ax, outcome in zip(axes[1:], outcomes):
        method = str(outcome["method"])
        if outcome.get("validity", 0.0) == 1.0:
            cf = _to_channel_first(outcome["counterfactual"])
            ax.plot(x, sample_cf[0], color="steelblue", linestyle="--", linewidth=1.2, label="original")
            ax.plot(x, cf[0], color="darkorange", linewidth=1.3, label="counterfactual")
            ax.set_title(f"{method} [valid] pred={outcome['pred_class']}", fontsize=10)
            ax.legend(loc="upper right", fontsize="small")
        else:
            ax.plot(x, sample_cf[0], color="steelblue", linestyle="--", linewidth=1.2)
            ax.set_title(f"{method} [invalid]", fontsize=10)
        ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    path = f"{output_prefix}_sample{sample_idx}_lineplot.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved line plot: {path}")


def _save_sample_umap_plot(
    projection_data: np.ndarray,
    projection_labels: np.ndarray,
    sample: np.ndarray,
    native_guide: np.ndarray,
    outcomes: List[Dict[str, object]],
    output_prefix: str,
    sample_idx: int,
    original_class: int,
    target_class: int,
    dataset_name: str,
):
    bg = np.asarray(projection_data, dtype=np.float32)
    if bg.ndim == 2:
        bg = bg[:, None, :]
    class_ids = np.asarray([_to_class_index(lbl) for lbl in projection_labels], dtype=np.int64)

    sample_cf = _to_channel_first(sample)
    guide_cf = _to_channel_first(native_guide)

    successful = [o for o in outcomes if o.get("validity", 0.0) == 1.0]
    bg_flat = bg.reshape(bg.shape[0], -1)
    n_neighbors = min(15, max(2, bg_flat.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.15,
                        metric="euclidean", random_state=42)
    bg_emb = reducer.fit_transform(bg_flat)
    orig_emb = reducer.transform(sample_cf.reshape(1, -1))[0]
    guide_emb = reducer.transform(guide_cf.reshape(1, -1))[0]
    cf_embs = np.asarray(
        [reducer.transform(_to_channel_first(o["counterfactual"]).reshape(1, -1))[0] for o in successful],
        dtype=np.float32,
    ) if successful else np.empty((0, 2), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_classes = sorted(np.unique(class_ids).tolist())
    colors_bg = plt.cm.tab10(np.linspace(0, 1, max(len(unique_classes), 1)))
    for i, cls in enumerate(unique_classes):
        mask = class_ids == cls
        ax.scatter(bg_emb[mask, 0], bg_emb[mask, 1], s=10,
                   c=[colors_bg[i]], alpha=0.35, label=f"class {cls}")
    ax.scatter(orig_emb[0], orig_emb[1], s=140, c="steelblue", marker="o",
               edgecolors="white", linewidths=1.4, label="original")
    ax.scatter(guide_emb[0], guide_emb[1], s=140, c="darkgreen", marker="^",
               edgecolors="white", linewidths=1.4, label="native guide")

    for idx, outcome in enumerate(successful):
        color = _method_color(str(outcome["method"]))
        ax.scatter(cf_embs[idx, 0], cf_embs[idx, 1], s=130, c=color, marker="X",
                   edgecolors="white", linewidths=1.1, label=f"{outcome['method']} cf")
        ax.plot([orig_emb[0], cf_embs[idx, 0]], [orig_emb[1], cf_embs[idx, 1]],
                color=color, alpha=0.35, linewidth=1.0)

    ax.set_title(f"UMAP — {dataset_name} sample {sample_idx} | {original_class} → {target_class}", fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize="small", ncols=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = f"{output_prefix}_sample{sample_idx}_umap.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved UMAP plot: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="IMFACT ablation study on any UCR/UEA time-series dataset."
    )
    parser.add_argument("--dataset", required=True,
                        help="UCR/UEA dataset name (e.g. GunPoint, FaultDetectionA)")
    parser.add_argument("--max-samples", type=int, default=8,
                        help="Number of correctly classified test samples to evaluate (default: 8)")
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Training epochs if no checkpoint exists (default: 30)")
    parser.add_argument("--model-file", type=str, default=None,
                        help="Path to model checkpoint (default: models/simple_cnn_<dataset>_...pth)")
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Output file prefix, e.g. results/gunpoint_ablation/gunpoint "
                             "(default: results/<dataset_lower>_ablation/<dataset_lower>)")
    parser.add_argument("--max-plot-samples", type=int, default=1,
                        help="Max samples for which per-sample plots are saved (default: 1, 0=disabled)")
    parser.add_argument("--multi-nun-counts", type=str, default="2,3,5",
                        help="Comma-separated n_nuns values for multi-NUN ablation (default: 2,3,5)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor applied to each time series (default: 1)")
    return parser.parse_args()


class _DownsampledDataset:
    def __init__(self, base, stride: int):
        self._base = base
        self._stride = stride

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        x, y = self._base[idx]
        return np.asarray(x, dtype=np.float32)[..., :: self._stride], y

    def __getattr__(self, name):
        return getattr(self._base, name)


def main():
    args = parse_args()
    dataset_name = args.dataset
    dataset_lower = dataset_name.lower()

    multi_nun_counts = _parse_multi_nun_counts(args.multi_nun_counts)
    method_ablation_configs = _build_method_ablation_configs()
    nun_ablation_configs = _build_nun_ablation_configs(multi_nun_counts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"IMFACT ABLATION — {dataset_name}")
    print("=" * 80)
    print(f"Device: {device}")

    _, dataset_train = _load_dataset(dataset_name, "train")
    _, dataset_test = _load_dataset(dataset_name, "test")

    if args.downsample > 1:
        dataset_train = _DownsampledDataset(dataset_train, args.downsample)
        dataset_test = _DownsampledDataset(dataset_test, args.downsample)

    dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False)

    output_classes = dataset_train.y_shape[1]
    raw_length = dataset_train.X_shape[2]
    input_length = (raw_length + args.downsample - 1) // args.downsample if args.downsample > 1 else raw_length
    print(f"Classes: {output_classes} | Series length: {input_length} | "
          f"Train: {len(dataset_train)} | Test: {len(dataset_test)}")

    model_file = args.model_file or os.path.join(
        REPO_ROOT, "models", f"simple_cnn_{dataset_lower}_{output_classes}_len{input_length}.pth"
    )
    model = SimpleCNN(output_channels=output_classes, input_length=input_length).to(device)
    model = _load_or_train(model_file, model, dataloader_train, dataloader_test, device, args.train_epochs)
    model.eval()

    reference_count = min(256, len(dataset_train))
    reference_data = np.stack(
        [np.asarray(dataset_train[i][0], dtype=np.float32) for i in range(reference_count)], axis=0
    )
    projection_data = np.stack(
        [np.asarray(dataset_test[i][0], dtype=np.float32) for i in range(len(dataset_test))], axis=0
    )
    projection_labels = np.asarray([dataset_test[i][1] for i in range(len(dataset_test))])

    print("\nSelecting correctly classified test samples ...")
    selected_indices = _select_correct_samples(model, dataset_test, args.max_samples, device)
    print(f"Selected {len(selected_indices)} samples")
    if not selected_indices:
        raise RuntimeError("No correctly classified samples found.")

    if args.out_prefix is None:
        out_dir = os.path.join(SCRIPT_DIR, "results", f"{dataset_lower}_ablation")
        os.makedirs(out_dir, exist_ok=True)
        out_prefix = os.path.join(out_dir, dataset_lower)
    else:
        out_prefix = args.out_prefix
        os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    suite_configs = {
        "method_ablation": method_ablation_configs,
        "nun_ablation": nun_ablation_configs,
    }
    suite_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        suite: {m: [] for m in cfg.keys()} for suite, cfg in suite_configs.items()
    }

    for sample_rank, sample_idx in enumerate(selected_indices):
        sample, label = dataset_test[sample_idx]
        sample = np.asarray(sample, dtype=np.float32)
        sample_scores = _predict(model, sample, device)
        original_class = int(np.argmax(sample_scores))
        sorted_classes = np.argsort(sample_scores)[::-1]
        target_class = next(
            (int(c) for c in sorted_classes if int(c) != original_class),
            (original_class + 1) % output_classes,
        )

        print(f"\nSample {sample_idx}: original={original_class} → target={target_class}")
        native_guide, _ = _select_native_guide(sample, dataset_test, target_class)

        for suite_name, configs in suite_configs.items():
            methods = list(configs.keys())
            w = max(13, max(len(m) for m in methods) + 2)
            sample_outcomes: List[Dict[str, object]] = []

            print(f"  [{suite_name}]")
            for method in methods:
                outcome = _evaluate_method(
                    method, configs, sample, dataset_test, reference_data, model, target_class, device
                )
                suite_results[suite_name][method].append(outcome)
                sample_outcomes.append(outcome)
                valid = outcome.get("validity", 0.0) == 1.0
                status = "valid" if valid else "invalid"
                if valid:
                    print(
                        f"    {method:<{w}} {status:<7} pred={outcome['pred_class']} "
                        f"conf={outcome['confidence']:.4f} l2={outcome['l2_distance']:.2f} "
                        f"pct={outcome['pct_changed']:.1f}% time={outcome['elapsed']:.3f}s"
                    )
                else:
                    err = outcome.get("error", "")
                    print(f"    {method:<{w}} {status:<7} {err}")

            if args.max_plot_samples > 0 and sample_rank < args.max_plot_samples:
                suite_prefix = f"{out_prefix}_{suite_name}"
                _save_sample_line_plot(
                    sample, label, original_class, target_class,
                    sample_outcomes, methods, suite_prefix, sample_idx, dataset_name,
                )
                _save_sample_umap_plot(
                    projection_data, projection_labels, sample, native_guide,
                    sample_outcomes, suite_prefix, sample_idx,
                    original_class, target_class, dataset_name,
                )

    for suite_name, configs in suite_configs.items():
        methods = list(configs.keys())
        summary_rows = _summarize_results(methods, suite_results[suite_name])
        _print_summary(f"IMFACT {suite_name.upper()} — {dataset_name}", methods, summary_rows)
        _save_summary_csv(f"{out_prefix}_{suite_name}.csv", summary_rows)
        _save_summary_plot(
            f"{out_prefix}_{suite_name}.png",
            summary_rows,
            f"IMFACT {suite_name.replace('_', ' ').title()} on {dataset_name}",
        )


if __name__ == "__main__":
    main()
