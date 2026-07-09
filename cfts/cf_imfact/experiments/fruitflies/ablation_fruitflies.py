"""Ablation study for IMFACT counterfactual methods on FruitFlies.

This script runs two separate ablations on a small set of correctly classified
FruitFlies test samples:

1) IMF selection method ablation (single-guide methods only)
2) Multi-NUN hyperparameter ablation (``n_nuns`` and switching policy)

For each sample, the native guide is drawn from the selected target class so
the IMF edit path is always conditioned on the intended destination class.

For a configurable subset of evaluated samples, the script saves a line-plot
figure that shows the original query sample together with each generated
counterfactual.

For the same subset, it saves a UMAP projection of the reference dataset
together with the original query, the native guide, and the successful
counterfactuals.

It mirrors the style of the other repository evaluation scripts while staying
lightweight enough to run from the `cfts/cf_imfact` package directory.
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
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder

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

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from examples.base.model import SimpleCNN
from examples.base.data import TimeSeriesDataset, collate_sparse
from cfts.cf_imfact.imfact import _psd, imfact_cf
from scipy.spatial.distance import jensenshannon


DATASET_NAME = "FruitFlies"
CLASS_NAMES: Dict[int, str] = {}


def _class_name(class_id: int) -> str:
    return CLASS_NAMES.get(int(class_id), f"class_{int(class_id)}")


def _ensure_fruitflies_local_files() -> str:
    data_root = os.path.join(REPO_ROOT, "data", "UCR")
    dataset_dir = os.path.join(data_root, DATASET_NAME)
    train_file = os.path.join(dataset_dir, f"{DATASET_NAME}_TRAIN.ts")
    test_file = os.path.join(dataset_dir, f"{DATASET_NAME}_TEST.ts")

    if os.path.exists(train_file) and os.path.exists(test_file):
        return data_root

    os.makedirs(dataset_dir, exist_ok=True)
    archive_path = os.path.join(data_root, f"{DATASET_NAME}.zip")
    if not os.path.exists(archive_path):
        dataset_url = f"https://timeseriesclassification.com/aeon-toolkit/{DATASET_NAME}.zip"
        print(f"Downloading {DATASET_NAME} archive from {dataset_url}")
        urllib.request.urlretrieve(dataset_url, archive_path)

    print(f"Extracting {DATASET_NAME} TRAIN/TEST .ts files with system unzip")
    subprocess.run(
        [
            "unzip",
            "-j",
            "-o",
            archive_path,
            f"{DATASET_NAME}_TRAIN.ts",
            f"{DATASET_NAME}_TEST.ts",
            "-d",
            dataset_dir,
        ],
        check=True,
    )

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        raise FileNotFoundError(f"Failed to prepare local {DATASET_NAME} .ts files in {dataset_dir}")

    return data_root


def _get_ucr_uea_dataloader_resilient(dataset_name: str, split: str, batch_size: int = 256, shuffle: bool = True):
    extract_path = os.path.join(REPO_ROOT, "data", "UCR")
    try:
        X, y = load_classification(name=dataset_name, split=split, extract_path=extract_path)
    except NotImplementedError as exc:
        if dataset_name != DATASET_NAME:
            raise
        print(f"Zip extraction fallback triggered for {dataset_name}: {exc}")
        extract_path = _ensure_fruitflies_local_files()
        X, y = load_classification(name=dataset_name, split=split, extract_path=extract_path)

    encoder = OneHotEncoder(categories="auto", sparse_output=False)
    y = encoder.fit_transform(np.expand_dims(y, axis=-1))
    dataset = TimeSeriesDataset(X=X, y=y, name=dataset_name, mapping=encoder.categories_)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_sparse)
    return dataloader, dataset


BASE_METHOD_CONFIGS = {
    "distance": {"method": "distance"},
    "fingerprint": {"method": "fingerprint"},
    "variance": {"method": "variance"},
    "extremes": {"method": "extremes"},
    "maxmin": {"method": "maxmin"},
    "coarse_to_fine": {"method": "coarse_to_fine"},
}


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
    ax.annotate(
        "BEST",
        xy=(x_pos, y_pos),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontweight="bold",
        color="gold",
    )


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


def _build_method_ablation_configs() -> Dict[str, Dict[str, object]]:
    return dict(BASE_METHOD_CONFIGS)


def _build_nun_ablation_configs(multi_nun_counts: Sequence[int]) -> Dict[str, Dict[str, object]]:
    nun_configs = {
        "distance_n1_cycle": {
            "method": "distance",
            "n_nuns": 1,
            "nun_switch": "cycle",
        }
    }
    for count in multi_nun_counts:
        nun_configs[f"multi_nun_cycle_n{count}"] = {
            "method": "distance",
            "n_nuns": count,
            "nun_switch": "cycle",
        }
        nun_configs[f"multi_nun_closest_n{count}"] = {
            "method": "distance",
            "n_nuns": count,
            "nun_switch": "closest_psd",
        }
    return nun_configs


def _method_color(method: str) -> str:
    if method == "distance":
        return "darkorange"
    if method == "fingerprint":
        return "sienna"
    if method == "variance":
        return "crimson"
    if method == "extremes":
        return "purple"
    if method == "maxmin":
        return "teal"
    if method == "coarse_to_fine":
        return "black"
    if method.startswith("multi_nun_cycle"):
        return "goldenrod"
    if method.startswith("multi_nun_closest"):
        return "royalblue"
    return "darkorange"


class DownsampledDataset:
    """Proxy dataset that downsamples each sample on access."""

    def __init__(self, base, stride: int):
        self._base = base
        self._stride = stride

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        x, y = self._base[idx]
        x_ds = np.asarray(x, dtype=np.float32)[..., :: self._stride]
        return x_ds, y

    def __getattr__(self, name):
        return getattr(self._base, name)


def _to_class_index(label) -> int:
    arr = np.asarray(label)
    if arr.ndim == 0:
        return int(arr)
    return int(np.argmax(arr))


def _format_prediction(prediction) -> Tuple[int, float]:
    pred_np = np.asarray(prediction).reshape(-1)
    if np.any(pred_np < 0) or not np.isclose(np.sum(pred_np), 1.0, atol=0.1):
        exp = np.exp(pred_np - np.max(pred_np))
        pred_np = exp / np.sum(exp)
    return int(np.argmax(pred_np)), float(np.max(pred_np))


def _predict(model: nn.Module, sample: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 1, -1)
    elif tensor.ndim == 2:
        if tensor.shape[0] > tensor.shape[1]:
            tensor = tensor.T
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        return model(tensor).cpu().numpy().reshape(-1)


def _select_correct_samples(model: nn.Module, dataset, max_samples: int, device: torch.device):
    selected = []
    for idx in range(len(dataset)):
        sample, label = dataset[idx]
        pred = _predict(model, np.asarray(sample, dtype=np.float32), device)
        pred_class = int(np.argmax(pred))
        true_class = _to_class_index(label)
        if pred_class == true_class:
            selected.append(idx)
        if len(selected) >= max_samples:
            break
    return selected


def _select_native_guide(sample: np.ndarray, dataset, guide_class: int | None) -> Tuple[np.ndarray, int]:
    sample_cl = np.asarray(sample, dtype=np.float32)
    if sample_cl.ndim == 1:
        sample_cl = sample_cl.reshape(1, -1)
    elif sample_cl.ndim == 2 and sample_cl.shape[0] > sample_cl.shape[1]:
        sample_cl = sample_cl.T

    sample_psd = _psd(sample_cl[0])

    candidates = []
    candidate_labels = []
    for idx in range(len(dataset)):
        cand_x, cand_y = dataset[idx]
        cand_label = _to_class_index(cand_y)
        if guide_class is not None and cand_label != int(guide_class):
            continue
        cand_x = np.asarray(cand_x, dtype=np.float32)
        if cand_x.ndim == 1:
            cand_x = cand_x.reshape(1, -1)
        elif cand_x.ndim == 2 and cand_x.shape[0] > cand_x.shape[1]:
            cand_x = cand_x.T
        candidates.append(cand_x)
        candidate_labels.append(cand_label)

    if not candidates:
        raise ValueError(f"No native guide found for guide_class={guide_class}")

    js_dists = [jensenshannon(sample_psd, _psd(candidate[0])) for candidate in candidates]
    best_idx = int(np.argmin(js_dists))
    return candidates[best_idx], candidate_labels[best_idx]


def _train_model(model: nn.Module, dataloader, criterion, optimizer, device: torch.device):
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


def _validate_model(model: nn.Module, dataloader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels in dataloader:
        inputs = inputs.reshape(inputs.shape[0], 1, -1).float().to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            preds = model(inputs)
            loss = criterion(preds, labels.argmax(dim=-1))
        running_loss += float(loss.item())
        all_preds.extend(preds.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.argmax(dim=-1).cpu().numpy())

    if not all_labels:
        return running_loss, 0.0

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    accuracy = float(np.mean(all_preds == all_labels))
    return running_loss / max(1, len(dataloader)), accuracy


def _load_or_train_model(model_file: str, model: nn.Module, dataloader_train, dataloader_test, device: torch.device, epochs: int):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    if os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
        return model

    print(f"No saved model at {model_file}; training a fresh model.")
    best_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        train_loss = _train_model(model, dataloader_train, criterion, optimizer, device)
        val_loss, val_score = _validate_model(model, dataloader_test, criterion, device)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} - train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | score: {val_score:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")
    return model


def _evaluate_method(
    method: str,
    method_configs: Dict[str, Dict[str, object]],
    sample: np.ndarray,
    dataset,
    reference_data: np.ndarray,
    model: nn.Module,
    target_class: int,
    device: torch.device,
):
    start_time = time.time()
    method_config = method_configs[method]
    try:
        cf, pred = imfact_cf(
            sample,
            dataset,
            model,
            method=method_config["method"],
            # Keep the guide anchored to the chosen destination class.
            guide_class=target_class,
            step=0.05,
            max_iter=200,
            max_imfs=10,
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
    l2_distance = float(np.linalg.norm(cf_flat - sample_flat))
    pct_changed = float(100.0 * percentage_changed_points(sample, cf))
    norm_distance = float(normalized_distance(sample_flat, cf_flat))
    temp_consistency = float(temporal_consistency(cf))

    range_validity = float(feature_range_validity(cf, reference_data))
    autocorr = float(autocorrelation_preservation(sample, cf))

    def _keane_model_predict(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return _predict(model, x, device)

    keane_metrics = evaluate_keane_metrics(
        original_ts_list=np.asarray(sample, dtype=np.float32),
        counterfactual_ts_list=np.asarray(cf, dtype=np.float32),
        model=_keane_model_predict,
        target_classes=int(target_class),
    )

    s_ch = _to_channel_first(np.asarray(sample, dtype=np.float32))
    c_ch = _to_channel_first(np.asarray(cf, dtype=np.float32))
    dtw_dist = float(dtw_distance(s_ch, c_ch))
    validity = float(prediction_change(s_ch, c_ch, _keane_model_predict, target_class=target_class))
    sparsity = 1.0 - pct_changed / 100.0

    return {
        "method": method,
        "elapsed": elapsed,
        "pred_class": pred_class,
        "confidence": confidence,
        "l2_distance": l2_distance,
        "pct_changed": pct_changed,
        "normalized_distance": norm_distance,
        "temporal_consistency": temp_consistency,
        "range_validity": range_validity,
        "autocorr_preservation": autocorr,
        "dtw_distance": dtw_dist,
        "validity": validity,
        "sparsity": sparsity,
        "keane_validity": float(keane_metrics["validity"]),
        "keane_proximity": float(keane_metrics["proximity"]),
        "keane_compactness": float(keane_metrics["compactness"]),
        "sample": np.asarray(sample, dtype=np.float32),
        "counterfactual": np.asarray(cf, dtype=np.float32),
    }


def _to_channel_first(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            return arr
        return arr.T
    raise ValueError(f"Unsupported array shape for plotting: {arr.shape}")


def _save_sample_line_plot(
    sample: np.ndarray,
    label,
    original_class: int,
    target_class: int,
    outcomes: List[Dict[str, object]],
    methods: Sequence[str],
    output_prefix: str,
    sample_idx: int,
):
    sample_cf = _to_channel_first(sample)
    n_rows = 1 + len(methods)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.1 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    true_class = _to_class_index(label)
    fig.suptitle(
        f"IMFACT Counterfactual Line Plots - {DATASET_NAME} | sample={sample_idx} | original={original_class} "
        f"({_class_name(original_class)}) | target={target_class} ({_class_name(target_class)})",
        fontsize=13,
        fontweight="bold",
    )

    x = np.arange(sample_cf.shape[1])
    axes[0].plot(x, sample_cf[0], color="steelblue", linewidth=1.4)
    axes[0].set_title(
        f"Original query sample - true: {true_class} ({_class_name(true_class)})",
        fontsize=10,
    )
    axes[0].grid(True, alpha=0.25)

    for ax, outcome in zip(axes[1:], outcomes):
        method = str(outcome["method"])
        if outcome.get("validity", 0.0) == 1.0:
            cf = _to_channel_first(outcome["counterfactual"])
            ax.plot(x, sample_cf[0], color="steelblue", linestyle="--", linewidth=1.2, label="original")
            ax.plot(x, cf[0], color="darkorange", linewidth=1.3, label="counterfactual")
            ax.set_title(
                f"{method} [valid] - pred: {outcome['pred_class']} ({_class_name(int(outcome['pred_class']))})",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize="small")
        else:
            ax.plot(x, sample_cf[0], color="steelblue", linestyle="--", linewidth=1.2, label="original")
            ax.set_title(f"{method} [invalid]", fontsize=10)
        ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    line_plot_path = f"{output_prefix}_sample{sample_idx}_lineplot.png"
    plt.savefig(line_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved line plot to {line_plot_path}")


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
):
    background_cf = np.asarray(projection_data, dtype=np.float32)
    if background_cf.ndim == 2:
        background_cf = background_cf[:, None, :]
    class_ids = np.asarray([_to_class_index(label) for label in projection_labels], dtype=np.int64)

    sample_cf = _to_channel_first(sample)
    guide_cf = _to_channel_first(native_guide)

    successful_outcomes = [outcome for outcome in outcomes if outcome.get("validity", 0.0) == 1.0]
    background_flat = background_cf.reshape(background_cf.shape[0], -1)
    n_neighbors = min(15, max(2, background_flat.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.15, metric="euclidean", random_state=42)
    background_emb = reducer.fit_transform(background_flat)

    original_emb = reducer.transform(sample_cf.reshape(1, -1))[0]
    guide_emb = reducer.transform(guide_cf.reshape(1, -1))[0]
    cf_embs = []
    for outcome in successful_outcomes:
        cf = _to_channel_first(outcome["counterfactual"])
        cf_embs.append(reducer.transform(cf.reshape(1, -1))[0])
    cf_embs = np.asarray(cf_embs, dtype=np.float32) if cf_embs else np.empty((0, 2), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 8))
    class_palette = {
        0: "#4E79A7",
        1: "#E15759",
        2: "#59A14F",
    }
    for class_id in sorted(np.unique(class_ids)):
        mask = class_ids == class_id
        ax.scatter(
            background_emb[mask, 0],
            background_emb[mask, 1],
            s=10,
            c=class_palette.get(int(class_id), "lightgray"),
            alpha=0.35,
            label=f"dataset {int(class_id)} ({_class_name(int(class_id))})",
        )
    ax.scatter(original_emb[0], original_emb[1], s=140, c="steelblue", marker="o", edgecolors="white", linewidths=1.4, label="original")
    ax.scatter(guide_emb[0], guide_emb[1], s=140, c="darkgreen", marker="^", edgecolors="white", linewidths=1.4, label="native guide")

    for idx, outcome in enumerate(successful_outcomes):
        method = str(outcome["method"])
        color = _method_color(method)
        ax.scatter(cf_embs[idx, 0], cf_embs[idx, 1], s=130, c=color, marker="X", edgecolors="white", linewidths=1.1, label=f"{method} cf")
        ax.plot([original_emb[0], cf_embs[idx, 0]], [original_emb[1], cf_embs[idx, 1]], color=color, alpha=0.35, linewidth=1.0)

    ax.set_title(
        f"UMAP projection - sample {sample_idx} | original {original_class} -> target {target_class}",
        fontsize=12,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize="small", ncols=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    umap_plot_path = f"{output_prefix}_sample{sample_idx}_umap.png"
    plt.savefig(umap_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved UMAP plot to {umap_plot_path}")


def _summarize_results(methods: Sequence[str], results: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, float | str]]:
    summary_rows: List[Dict[str, float | str]] = []
    for method in methods:
        method_rows = results[method]
        successes = [row for row in method_rows if row.get("validity", 0.0) == 1.0]
        keane_rows = [row for row in method_rows if "keane_validity" in row]
        total = len(method_rows)
        validity_rate = 100.0 * float(np.mean([row.get("validity", 0.0) for row in method_rows])) if method_rows else 0.0
        avg_conf = float(np.mean([row["confidence"] for row in successes])) if successes else 0.0
        avg_l2 = float(np.mean([row["l2_distance"] for row in successes])) if successes else 0.0
        avg_pct_changed = float(np.mean([row["pct_changed"] for row in successes])) if successes else 0.0
        avg_norm_distance = float(np.mean([row["normalized_distance"] for row in successes])) if successes else 0.0
        avg_temp_consistency = float(np.mean([row["temporal_consistency"] for row in successes])) if successes else 0.0
        avg_range_validity = float(np.mean([row["range_validity"] for row in successes])) if successes else 0.0
        avg_autocorr = float(np.mean([row["autocorr_preservation"] for row in successes])) if successes else 0.0
        avg_dtw = float(np.mean([row["dtw_distance"] for row in successes])) if successes else 0.0
        avg_validity = float(np.mean([row["validity"] for row in successes])) if successes else 0.0
        avg_sparsity = float(np.mean([row["sparsity"] for row in successes])) if successes else 0.0
        avg_keane_validity = float(np.mean([row["keane_validity"] for row in keane_rows])) if keane_rows else 0.0
        avg_keane_proximity = float(np.mean([row["keane_proximity"] for row in keane_rows])) if keane_rows else 0.0
        avg_keane_compactness = float(np.mean([row["keane_compactness"] for row in keane_rows])) if keane_rows else 0.0
        avg_time = float(np.mean([row["elapsed"] for row in method_rows])) if method_rows else 0.0
        summary_rows.append(
            {
                "method": method,
                "validity_rate": validity_rate,
                "avg_confidence": avg_conf,
                "avg_l2_distance": avg_l2,
                "avg_pct_changed": avg_pct_changed,
                "avg_normalized_distance": avg_norm_distance,
                "avg_temporal_consistency": avg_temp_consistency,
                "avg_range_validity": avg_range_validity,
                "avg_autocorr_preservation": avg_autocorr,
                "avg_dtw_distance": avg_dtw,
                "avg_validity": avg_validity,
                "avg_sparsity": avg_sparsity,
                "avg_keane_validity": avg_keane_validity,
                "avg_keane_proximity": avg_keane_proximity,
                "avg_keane_compactness": avg_keane_compactness,
                "avg_time": avg_time,
            }
        )
    return summary_rows


def _print_summary(title: str, methods: Sequence[str], summary_rows: Sequence[Dict[str, float | str]]):
    method_col_width = max(13, max(len(name) for name in methods) + 2)
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(
        f"{'Method':<{method_col_width}} {'Validity':<10} {'Confidence':<12} {'L2 Distance':<12} "
        f"{'%Changed':<10} {'Norm Dist':<10} {'Temp Cons':<10} {'Range Val':<10} {'AutoCorr':<10} "
        f"{'K.Valid':<8} {'K.Prox':<9} {'K.Comp':<9} {'Time (s)':<10}"
    )
    print("-" * 80)
    for row in summary_rows:
        print(
            f"{row['method']:<{method_col_width}} {row['validity_rate']:<10.1f}% {row['avg_confidence']:<12.4f} "
            f"{row['avg_l2_distance']:<12.2f} {row['avg_pct_changed']:<10.2f} {row['avg_normalized_distance']:<10.4f} "
            f"{row['avg_temporal_consistency']:<10.4f} {row['avg_range_validity']:<10.4f} {row['avg_autocorr_preservation']:<10.4f} "
            f"{row['avg_keane_validity']:<8.4f} {row['avg_keane_proximity']:<9.4f} {row['avg_keane_compactness']:<9.4f} {row['avg_time']:<10.4f}"
        )


def _save_summary_csv(csv_path: str, summary_rows: Sequence[Dict[str, float | str]]):
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "validity_rate",
                "avg_confidence",
                "avg_l2_distance",
                "avg_pct_changed",
                "avg_normalized_distance",
                "avg_temporal_consistency",
                "avg_range_validity",
                "avg_autocorr_preservation",
                "avg_dtw_distance",
                "avg_validity",
                "avg_sparsity",
                "avg_keane_validity",
                "avg_keane_proximity",
                "avg_keane_compactness",
                "avg_time",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\nSaved summary CSV to {csv_path}")


def _save_summary_plot(plot_path: str, summary_rows: Sequence[Dict[str, float | str]], title: str):
    names = [row["method"] for row in summary_rows]
    validity_rates = [row["validity_rate"] for row in summary_rows]
    confidences = [row["avg_confidence"] for row in summary_rows]
    distances = [row["avg_l2_distance"] for row in summary_rows]
    pct_changed = [row["avg_pct_changed"] for row in summary_rows]
    norm_distance = [row["avg_normalized_distance"] for row in summary_rows]
    range_validity = [row["avg_range_validity"] for row in summary_rows]
    keane_validity = [row["avg_keane_validity"] for row in summary_rows]
    keane_proximity = [row["avg_keane_proximity"] for row in summary_rows]
    keane_compactness = [row["avg_keane_compactness"] for row in summary_rows]
    dtw_distances = [row["avg_dtw_distance"] for row in summary_rows]
    validities = [row["avg_validity"] for row in summary_rows]
    sparsities = [row["avg_sparsity"] for row in summary_rows]

    fig_height = max(20, 0.85 * len(names) + 16)
    fig, axes = plt.subplots(4, 3, figsize=(18, fig_height))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0, 0].barh(names, validity_rates, color="steelblue")
    axes[0, 0].set_title("Validity Rate (%)")
    axes[0, 0].set_xlim([0, 105])
    for idx, value in enumerate(validity_rates):
        axes[0, 0].text(value + 1, idx, f"{value:.1f}%", va="center")
    _mark_best_barh(axes[0, 0], validity_rates, higher_is_better=True)

    axes[0, 1].barh(names, confidences, color="seagreen")
    axes[0, 1].set_title("Average Confidence")
    axes[0, 1].set_xlim(left=0)
    _mark_best_barh(axes[0, 1], confidences, higher_is_better=True)

    axes[0, 2].barh(names, keane_validity, color="forestgreen")
    axes[0, 2].set_title("Keane Validity")
    axes[0, 2].set_xlim(left=0, right=1.05)
    _mark_best_barh(axes[0, 2], keane_validity, higher_is_better=True)

    axes[1, 0].barh(names, distances, color="darkorange")
    axes[1, 0].set_title("Average L2 Distance")
    axes[1, 0].set_xlim(left=0)
    _mark_best_barh(axes[1, 0], distances, higher_is_better=False)

    axes[1, 1].barh(names, pct_changed, color="slateblue")
    axes[1, 1].set_title("Percentage Changed (%)")
    axes[1, 1].set_xlim(left=0)
    _mark_best_barh(axes[1, 1], pct_changed, higher_is_better=False)

    axes[1, 2].barh(names, keane_proximity, color="sienna")
    axes[1, 2].set_title("Keane Proximity")
    axes[1, 2].set_xlim(left=0)
    _mark_best_barh(axes[1, 2], keane_proximity, higher_is_better=False)

    axes[2, 0].barh(names, norm_distance, color="darkviolet")
    axes[2, 0].set_title("Normalized Distance")
    axes[2, 0].set_xlim(left=0)
    _mark_best_barh(axes[2, 0], norm_distance, higher_is_better=False)

    axes[2, 1].barh(names, range_validity, color="teal")
    axes[2, 1].set_title("Range Validity")
    axes[2, 1].set_xlim(left=0, right=1.05)
    _mark_best_barh(axes[2, 1], range_validity, higher_is_better=True)

    axes[2, 2].barh(names, keane_compactness, color="seagreen")
    axes[2, 2].set_title("Keane Compactness")
    axes[2, 2].set_xlim(left=0, right=1.05)
    _mark_best_barh(axes[2, 2], keane_compactness, higher_is_better=True)

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
    print(f"Saved plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=f"IMFACT ablation on {DATASET_NAME}.")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor applied to each time series.")
    parser.add_argument("--max-samples", type=int, default=8, help="Number of correctly classified test samples to evaluate.")
    parser.add_argument("--train-epochs", type=int, default=30, help="Training epochs used if no checkpoint exists.")
    parser.add_argument("--model-file", type=str, default=None, help="Optional checkpoint path override.")
    parser.add_argument("--output-prefix", type=str, default=None, help="Optional prefix for CSV/plot outputs.")
    parser.add_argument(
        "--max-plot-samples",
        type=int,
        default=1,
        help=(
            "Maximum number of samples per ablation suite for which per-sample line/UMAP plots are saved. "
            "Set to 0 to disable per-sample image generation."
        ),
    )
    parser.add_argument(
        "--multi-nun-counts",
        type=str,
        default="2,3,5",
        help="Comma-separated n_nuns values to include for the multi-NUN ablations.",
    )
    args = parser.parse_args()

    multi_nun_counts = _parse_multi_nun_counts(args.multi_nun_counts)
    method_ablation_configs = _build_method_ablation_configs()
    nun_ablation_configs = _build_nun_ablation_configs(multi_nun_counts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"IMFACT ABLATION - {DATASET_NAME}")
    print("=" * 80)
    print(f"Device: {device}")

    print("\nLoading dataset...")
    _, dataset_train = _get_ucr_uea_dataloader_resilient(dataset_name=DATASET_NAME, split="train")
    _, dataset_test = _get_ucr_uea_dataloader_resilient(dataset_name=DATASET_NAME, split="test")

    if args.downsample > 1:
        dataset_train = DownsampledDataset(dataset_train, args.downsample)
        dataset_test = DownsampledDataset(dataset_test, args.downsample)

    dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False)
    reference_count = min(256, len(dataset_train))
    reference_data = np.stack(
        [np.asarray(dataset_train[i][0], dtype=np.float32) for i in range(reference_count)],
        axis=0,
    )
    projection_data = np.stack(
        [np.asarray(dataset_test[i][0], dtype=np.float32) for i in range(len(dataset_test))],
        axis=0,
    )
    projection_labels = np.asarray([dataset_test[i][1] for i in range(len(dataset_test))])

    output_classes = dataset_train.y_shape[1]
    raw_length = dataset_train.X_shape[2]
    input_length = (raw_length + args.downsample - 1) // args.downsample if args.downsample > 1 else raw_length

    print(f"Classes: {output_classes}")
    print(f"Raw series length: {raw_length}")
    print(f"Effective series length: {input_length}")

    model = SimpleCNN(output_channels=output_classes, input_length=input_length).to(device)
    if args.model_file is None:
        model_file = os.path.join(REPO_ROOT, "models", f"simple_cnn_fruitflies_{output_classes}_len{input_length}.pth")
    else:
        model_file = args.model_file

    model = _load_or_train_model(model_file, model, dataloader_train, dataloader_test, device, args.train_epochs)
    model.eval()

    print("\nSelecting correctly classified test samples...")
    selected_indices = _select_correct_samples(model, dataset_test, args.max_samples, device)
    print(f"Selected {len(selected_indices)} correctly classified samples")
    if not selected_indices:
        raise RuntimeError("No correctly classified samples were found for ablation.")

    output_prefix = args.output_prefix or os.path.join(SCRIPT_DIR, "fruitflies_ablations")
    if args.output_prefix is None:
        os.makedirs(output_prefix, exist_ok=True)

    suite_configs = {
        "method_ablation": method_ablation_configs,
        "nun_ablation": nun_ablation_configs,
    }
    suite_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        suite_name: {method: [] for method in configs.keys()} for suite_name, configs in suite_configs.items()
    }

    for sample_rank, sample_idx in enumerate(selected_indices):
        sample, label = dataset_test[sample_idx]
        sample = np.asarray(sample, dtype=np.float32)
        sample_scores = _predict(model, sample, device)
        original_class = int(np.argmax(sample_scores))
        sorted_classes = np.argsort(sample_scores)[::-1]
        target_class = next((int(c) for c in sorted_classes if int(c) != original_class), (original_class + 1) % output_classes)

        print(f"\nSample {sample_idx}: original={original_class} ({_class_name(original_class)}), target={target_class} ({_class_name(target_class)})")

        native_guide, native_guide_class = _select_native_guide(sample, dataset_test, target_class)

        for suite_name, configs in suite_configs.items():
            methods = list(configs.keys())
            method_col_width = max(13, max(len(name) for name in methods) + 2)
            sample_outcomes = []

            print(f"  [{suite_name}]")
            for method in methods:
                outcome = _evaluate_method(method, configs, sample, dataset_test, reference_data, model, target_class, device)
                suite_results[suite_name][method].append(outcome)
                sample_outcomes.append(outcome)
                valid = outcome.get("validity", 0.0) == 1.0
                status = "valid" if valid else "invalid"
                if valid:
                    print(
                        f"    {method:<{method_col_width}} {status:<7} pred={outcome['pred_class']} conf={outcome['confidence']:.4f} "
                        f"l2={outcome['l2_distance']:.2f} pct={outcome['pct_changed']:.2f}% time={outcome['elapsed']:.3f}s"
                    )
                else:
                    error_text = outcome.get("error")
                    if error_text:
                        print(f"    {method:<{method_col_width}} {status:<7} {error_text}")
                    else:
                        print(f"    {method:<{method_col_width}} {status:<7} pred={outcome.get('pred_class')} target={target_class}")

            if args.max_plot_samples > 0 and sample_rank < args.max_plot_samples:
                suite_prefix = f"{output_prefix}_{suite_name}"
                _save_sample_line_plot(sample, label, original_class, target_class, sample_outcomes, methods, suite_prefix, sample_idx)
                _save_sample_umap_plot(
                    projection_data,
                    projection_labels,
                    sample,
                    native_guide,
                    sample_outcomes,
                    suite_prefix,
                    sample_idx,
                    original_class,
                    target_class,
                )

    for suite_name, configs in suite_configs.items():
        methods = list(configs.keys())
        summary_rows = _summarize_results(methods, suite_results[suite_name])
        _print_summary(f"IMFACT {suite_name.upper()} SUMMARY - {DATASET_NAME}", methods, summary_rows)
        csv_path = f"{output_prefix}_{suite_name}.csv"
        plot_path = f"{output_prefix}_{suite_name}.png"
        _save_summary_csv(csv_path, summary_rows)
        _save_summary_plot(plot_path, summary_rows, f"IMFACT {suite_name.replace('_', ' ').title()} on {DATASET_NAME}")


if __name__ == "__main__":
    main()