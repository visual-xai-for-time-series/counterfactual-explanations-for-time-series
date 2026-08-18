"""CoDec strategy ablation on any UCR/UEA time-series dataset.

Structured like `cfts/cf_imfact/experiments/compare_ucr.py`, but instead of
comparing CoDec against other CF methods, it compares CoDec **configurations
against itself** - this is the workplan's Phase 6 "Decomposition-heuristic
experiments" (`CoDec_workplan.md` §3): does the favored strategy stack
(composite reference selection + Hungarian matching) actually win over the
IMFACT-equivalent baseline (single NUN + naive index matching) and a
mismatched decomposition, on real UCR/UEA data?

Four named configs (`CODEC_CONFIGS` below), each isolating one design choice
from `CoDec_presentation.pdf`'s "Optional: Beyond a Single NUN" / "Module
Spotlight: Matching" slides:

  imfact_baseline       - EMD + single NUN + index matching   (regression baseline)
  hungarian_matching     - EMD + single NUN + Hungarian matching (isolates matching)
  composite_reference    - EMD + composite reference + Hungarian (the favored stack)
  fourier_decomposition  - Fourier bands + composite + Hungarian (isolates decomposition)

Usage:
    python compare_ucr.py --dataset GunPoint
    python compare_ucr.py --dataset FordA --n-samples 50 --max-search-samples 300
    python compare_ucr.py --dataset CBF --configs imfact_baseline,composite_reference
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
import pandas as pd
import torch
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

from base.data import TimeSeriesDataset, collate_sparse  # noqa: E402
from base.model import SimpleCNN  # noqa: E402
from cfts.cf_codec.codec import codec_cf  # noqa: E402
from cfts.metrics import evaluate_counterfactual  # noqa: E402

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({"font.size": 15})

# ---------------------------------------------------------------------------
# The four configs under comparison - see module docstring.
# ---------------------------------------------------------------------------

CODEC_CONFIGS: dict[str, dict] = {
    "imfact_baseline": dict(
        decomposition="emd", reference_selection="nun", matching="index", perturbation="replace",
    ),
    "hungarian_matching": dict(
        decomposition="emd", reference_selection="nun", matching="hungarian",
        cost_fn="dominant_frequency", perturbation="replace",
    ),
    "composite_reference": dict(
        decomposition="emd", reference_selection="composite", matching="hungarian",
        cost_fn="dominant_frequency", perturbation="replace",
    ),
    "fourier_decomposition": dict(
        decomposition="fourier", reference_selection="composite", matching="hungarian",
        cost_fn="dominant_frequency", perturbation="replace",
    ),
}

CONFIG_COLORS = {
    "imfact_baseline": "#8d99ae",
    "hungarian_matching": "#457b9d",
    "composite_reference": "#e63946",
    "fourier_decomposition": "#2a9d8f",
}


# ---------------------------------------------------------------------------
# Dataset
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
        ["unzip", "-j", "-o", archive_path, f"{dataset_name}_TRAIN.ts", f"{dataset_name}_TEST.ts", "-d", dataset_dir],
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
# Model - same tiny training loop as cf_imfact/experiments/compare_ucr.py
# ---------------------------------------------------------------------------

def _train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.reshape(inputs.shape[0], 1, -1).float().to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels.argmax(dim=-1))
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


def _select_correct_indices(model, dataset, max_count: int, device, seed: int = 13) -> list[int]:
    """Randomly (but reproducibly, via `seed`) picks `max_count` correctly classified samples."""
    correct = []
    for idx in range(len(dataset)):
        sample, label = dataset[idx]
        scores = _predict(model, np.asarray(sample, dtype=np.float32), device)
        if int(np.argmax(scores)) == _to_class_index(label):
            correct.append(idx)
    rng = np.random.RandomState(seed)
    if len(correct) > max_count:
        correct = rng.choice(correct, size=max_count, replace=False).tolist()
    return sorted(correct)


def _infer_target_class(scores: np.ndarray) -> int:
    return int(np.argsort(scores)[::-1][1])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    selected_indices: list[int],
    dataset_test,
    model,
    device,
    reference_data: np.ndarray,
    config_order: list[str],
    max_search_samples: int,
    k: int,
    max_iter: int,
) -> pd.DataFrame:
    model_fn = _model_wrapper(model, device)
    records = []

    for i, idx in enumerate(selected_indices):
        print(f"  [{i + 1}/{len(selected_indices)}] sample {idx}", flush=True)
        sample, label = dataset_test[idx]
        sample = np.asarray(sample, dtype=np.float32)
        scores_orig = _predict(model, sample, device)
        pred_orig = int(np.argmax(scores_orig))
        true_label = _to_class_index(label)
        target_class = _infer_target_class(scores_orig)

        for config_name in config_order:
            base = {
                "sample_idx": idx, "config": config_name,
                "true_label": true_label, "pred_orig": pred_orig, "target_class": target_class,
                "pred_cf": None, "l2_norm": np.nan, "normalized_distance": np.nan,
                "sparsity_pct": np.nan, "n_components_changed": np.nan, "range_validity": np.nan,
                "autocorr": np.nan, "temporal_consistency": np.nan, "confidence": np.nan,
                "validity": 0.0, "elapsed": np.nan, "n_iter": np.nan, "error": None,
            }
            start_time = time.time()
            try:
                cf, pred_cf_scores, result = codec_cf(
                    sample=sample, model=model, target_class=target_class, dataset=dataset_test,
                    max_samples=max_search_samples, k=k, max_iter=max_iter,
                    return_result=True, verbose=False, **CODEC_CONFIGS[config_name],
                )
            except Exception as exc:
                elapsed = time.time() - start_time
                records.append({**base, "elapsed": elapsed, "error": f"{type(exc).__name__}: {exc}"})
                continue
            elapsed = time.time() - start_time

            cf = np.asarray(cf, dtype=np.float32)
            _scores = np.asarray(pred_cf_scores).reshape(-1)
            pred_cf = int(np.argmax(_scores))
            _exp = np.exp(_scores - _scores.max())
            confidence = float(_exp[pred_cf] / _exp.sum())

            s_cf, c_cf = _to_channel_first(sample), _to_channel_first(cf)
            _m = evaluate_counterfactual(
                s_cf, c_cf, model=model_fn, target_class=int(target_class), reference_data=reference_data,
            )

            records.append({
                **base,
                "elapsed": elapsed,
                "n_iter": len(result.history),
                "pred_cf": pred_cf,
                "l2_norm": _m["l2_distance"],
                "normalized_distance": _m["normalized_distance"],
                "sparsity_pct": _m["sparsity"],
                "n_components_changed": result.sparsity,  # workplan §5: sparsity BY COMPONENT, not raw points
                "range_validity": _m.get("range_validity", np.nan),
                "autocorr": _m["autocorr_preservation"],
                "temporal_consistency": _m["temporal_consistency"],
                "confidence": confidence,
                "validity": _m["validity"],
            })

    return pd.DataFrame(records)


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        results_df.groupby("config", dropna=False)
        .agg(
            n_total=("sample_idx", "count"),
            validity_rate=("validity", "mean"),
            l2_norm_mean=("l2_norm", "mean"),
            normalized_distance_mean=("normalized_distance", "mean"),
            n_components_changed_mean=("n_components_changed", "mean"),
            range_validity_mean=("range_validity", "mean"),
            autocorr_mean=("autocorr", "mean"),
            temporal_consistency_mean=("temporal_consistency", "mean"),
            confidence_mean=("confidence", "mean"),
            elapsed_mean=("elapsed", "mean"),
            n_errors=("error", lambda s: s.notna().sum()),
        )
        .reset_index()
    )
    order = [c for c in CODEC_CONFIGS if c in agg["config"].values]
    agg["config"] = pd.Categorical(agg["config"], categories=order, ordered=True)
    return agg.sort_values("config").reset_index(drop=True)


def plot_bar_metrics(summary_df: pd.DataFrame, dataset_name: str, out_path: str) -> None:
    configs = summary_df["config"].astype(str).tolist()
    colors = [CONFIG_COLORS.get(c, "#999999") for c in configs]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(f"{dataset_name} - CoDec strategy ablation", fontsize=18, y=1.05)

    axes[0].bar(configs, summary_df["validity_rate"], color=colors)
    axes[0].set_title("Validity ↑")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(configs, summary_df["l2_norm_mean"], color=colors)
    axes[1].set_title("Proximity ↓ (L2)")

    axes[2].bar(configs, summary_df["n_components_changed_mean"], color=colors)
    axes[2].set_title("Sparsity ↓ (components changed)")

    axes[3].bar(configs, summary_df["elapsed_mean"], color=colors)
    axes[3].set_title("Time ↓ (s)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Compare CoDec strategy configurations on a UCR/UEA dataset.")
    parser.add_argument("--dataset", required=True, help="UCR/UEA dataset name (e.g. CBF, ECG5000, FordA)")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Number of correctly classified test samples to evaluate (default: 30)")
    parser.add_argument("--configs", type=str, default=None,
                        help=f"Comma-separated subset of {list(CODEC_CONFIGS)} (default: all)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: results/<dataset_lower>_codec_ablation/)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sample selection (default: 13)")
    parser.add_argument("--k", type=int, default=5, help="Reference pool size passed to codec_cf (default: 5)")
    parser.add_argument("--max-iter", type=int, default=25, help="max_iter passed to codec_cf (default: 25)")
    parser.add_argument("--max-search-samples", type=int, default=500,
                        help="Dataset cap passed to codec_cf's max_samples, to bound runtime (default: 500)")
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Training epochs if no model checkpoint is found (default: 30)")
    parser.add_argument("--model-file", type=str, default=None,
                        help="Path to model checkpoint (default: models/simple_cnn_<dataset>_...pth)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset
    dataset_lower = dataset_name.lower()
    config_order = args.configs.split(",") if args.configs else list(CODEC_CONFIGS)
    for c in config_order:
        if c not in CODEC_CONFIGS:
            raise ValueError(f"Unknown config '{c}'. Available: {list(CODEC_CONFIGS)}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, "results", f"{dataset_lower}_codec_ablation")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {dataset_name} | Device: {device} | Repo root: {REPO_ROOT}")
    print(f"Configs: {config_order}")

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
    for sample, label in dataset_test:
        scores = _predict(model, np.asarray(sample, dtype=np.float32), device)
        y_true.append(_to_class_index(label))
        y_pred.append(int(np.argmax(scores)))
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")

    reference_count = min(256, len(dataset_train))
    reference_data = np.stack(
        [np.asarray(dataset_train[i][0], dtype=np.float32) for i in range(reference_count)], axis=0
    )

    selected_indices = _select_correct_indices(model, dataset_test, args.n_samples, device, seed=args.seed)
    print(f"Evaluating {len(selected_indices)} correctly classified samples x {len(config_order)} configs "
          f"= {len(selected_indices) * len(config_order)} CoDec runs ...")

    results_df = evaluate(
        selected_indices, dataset_test, model, device, reference_data, config_order,
        max_search_samples=args.max_search_samples, k=args.k, max_iter=args.max_iter,
    )

    summary_df = build_summary(results_df)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    results_df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"\nSaved CSVs to {out_dir}")

    plot_bar_metrics(summary_df, dataset_name, os.path.join(out_dir, "bar_metrics.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
