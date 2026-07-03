"""Compare IMFACT vs Native Guide vs Wachter on FaultDetectionA.

Evaluates three counterfactual methods on correctly-classified FaultDetectionA
test samples and saves metric summary, bar charts, a UMAP projection, and
signal-space waveform plots to an output directory.

Usage:
    python compare_imfact_guide_wachter.py [--n-samples 50] [--out-dir ./output]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.metrics import f1_score

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

from base.data import get_UCR_UEA_dataloader
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
)

plt.style.use("seaborn-v0_8-darkgrid")

METHOD_COLORS = {
    "imfact_variance_nun3": "#e63946",
    "native_guide": "#1d3557",
    "wachter": "#ff7f11",
}
BAR_COLORS = ["#2a9d8f", "#457b9d", "#e76f51"]


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

def run_imfact_variance(sample, dataset, model, target_class):
    return imfact_cf(
        sample=sample,
        dataset=dataset,
        model=model,
        method="variance",
        guide_class=target_class,
        step=0.05,
        max_iter=200,
        max_imfs=10,
        n_nuns=3,
        nun_switch="cycle",
        verbose=False,
    )


def run_native_guide(sample, dataset, model, target_class):
    return native_guide_uni_cf(sample=sample, dataset=dataset, model=model)


def run_wachter(sample, dataset, model, target_class):
    return wachter_gradient_cf(
        sample=sample,
        dataset=dataset,
        model=model,
        target=target_class,
        max_cfs=400,
        distance="euclidean",
        verbose=False,
    )


RUNNERS = {
    "imfact_variance_nun3": run_imfact_variance,
    "native_guide": run_native_guide,
    "wachter": run_wachter,
}

METHOD_ORDER = ["imfact_variance_nun3", "native_guide", "wachter"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(selected_indices, dataset_test, model, device, reference_data) -> tuple[pd.DataFrame, dict]:
    model_for_metrics = model_wrapper_factory(model, device)
    records = []
    all_cf_outputs: dict = {}

    for idx in selected_indices:
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
                "success": False,
                "l2_norm": np.nan,
                "dtw_proximity": np.nan,
                "normalized_distance": np.nan,
                "sparsity": np.nan,
                "range_validity": np.nan,
                "autocorr": np.nan,
                "keane_validity": np.nan,
                "keane_proximity": np.nan,
                "keane_compactness": np.nan,
                "validity": 0.0,
                "error": None,
            }

            try:
                cf, pred_cf_scores = RUNNERS[method_name](sample, dataset_test, model, target_class)
            except Exception as exc:
                records.append({**base, "error": f"{type(exc).__name__}: {exc}"})
                continue

            if cf is None or pred_cf_scores is None:
                records.append({**base, "error": "Method returned None"})
                continue

            cf = np.asarray(cf, dtype=np.float32)
            pred_cf = int(np.argmax(np.asarray(pred_cf_scores).reshape(-1)))
            success = pred_cf == target_class

            s_cf = to_channel_first(sample)
            c_cf = to_channel_first(cf)

            keane = evaluate_keane_metrics(
                original_ts_list=s_cf,
                counterfactual_ts_list=c_cf,
                model=model_for_metrics,
                target_classes=int(target_class),
            )

            records.append({
                **base,
                "pred_cf": pred_cf,
                "success": bool(success),
                "l2_norm": float(l2_distance(s_cf, c_cf)),
                "dtw_proximity": float(dtw_distance(s_cf, c_cf)),
                "normalized_distance": float(normalized_distance(s_cf.reshape(-1), c_cf.reshape(-1))),
                "sparsity": float(1.0 - percentage_changed_points(s_cf, c_cf)),
                "range_validity": float(feature_range_validity(c_cf, reference_data)),
                "autocorr": float(autocorrelation_preservation(s_cf, c_cf)),
                "keane_validity": float(keane["validity"]),
                "keane_proximity": float(keane["proximity"]),
                "keane_compactness": float(keane["compactness"]),
                "validity": float(prediction_change(s_cf, c_cf, model_for_metrics, target_class=target_class)),
            })

            all_cf_outputs[idx][method_name] = cf

    return pd.DataFrame(records), all_cf_outputs


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    successful_df = results_df[results_df["success"]].copy()
    agg_all = (
        results_df.groupby("method", dropna=False)
        .agg(
            n_total=("sample_idx", "count"),
            n_successful=("success", "sum"),
            success_rate=("success", "mean"),
            validity_mean=("validity", "mean"),
        )
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
            keane_validity_mean=("keane_validity", "mean"),
            keane_proximity_mean=("keane_proximity", "mean"),
            keane_compactness_mean=("keane_compactness", "mean"),
        )
        .reset_index()
    )
    summary = agg_all.merge(agg_suc, on="method", how="left")
    summary["n_successful"] = summary["n_successful"].fillna(0).astype(int)
    summary["success_rate"] = 100.0 * summary["success_rate"]
    summary["validity_mean"] = summary["validity_mean"].fillna(0.0)
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
    dtw_score = 1.0 / (1.0 + plot_df["dtw_proximity_mean"])
    norm_dist_score = 1.0 / (1.0 + plot_df["normalized_distance_mean"])

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    _bar(axes[0, 0], plot_df["validity_mean"], "Validity (higher better)")
    _bar(axes[0, 1], plot_df["sparsity_mean"], "Sparsity (higher better)")
    _bar(axes[0, 2], plot_df["range_validity_mean"], "Range Validity (higher better)")
    _bar(axes[1, 0], plot_df["autocorr_mean"], "Autocorr Preservation (higher better)")
    _bar(axes[1, 1], plot_df["keane_validity_mean"], "Keane Validity (higher better)")
    _bar(axes[1, 2], plot_df["keane_compactness_mean"], "Keane Compactness (higher better)")
    _bar(axes[2, 0], l2_score, "L2 Proximity Score (higher better)")
    _bar(axes[2, 1], dtw_score, "DTW Proximity Score (higher better)")
    _bar(axes[2, 2], norm_dist_score, "Normalised Distance Score (higher better)")

    fig.suptitle("FaultDetectionA — IMFACT vs Native Guide vs Wachter", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart: {out_path}")


def plot_umap(results_df: pd.DataFrame, all_cf_outputs: dict, selected_indices: list, dataset_test, out_path: str) -> None:
    background_data = np.stack(
        [to_channel_first(dataset_test[i][0]).reshape(-1) for i in range(len(dataset_test))],
        axis=0,
    )
    background_labels = np.array([to_class_index(dataset_test[i][1]) for i in range(len(dataset_test))])

    sample_success_counts = (
        results_df[results_df["sample_idx"].isin(selected_indices)]
        .groupby("sample_idx")["success"]
        .sum()
        .sort_values(ascending=False)
    )
    full_success = [int(i) for i, c in sample_success_counts.items() if int(c) == len(METHOD_ORDER)]
    rep_idx = full_success[0] if full_success else int(sample_success_counts.index[0])

    rep_payload = all_cf_outputs[rep_idx]
    rep_sample = to_channel_first(rep_payload["sample"]).reshape(1, -1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(background_data) - 1)),
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )
    background_emb = reducer.fit_transform(background_data)
    sample_emb = reducer.transform(rep_sample)[0]

    sample_results = results_df[results_df["sample_idx"] == rep_idx]
    metric_lookup = {
        row["method"]: {
            "l2": float(row["l2_norm"]),
            "dtw": float(row["dtw_proximity"]),
            "success": bool(row["success"]),
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
        dtw_val = m.get("dtw", np.nan)
        worked = m.get("success", False)
        pred_cf = m.get("pred", None)

        annotation_lines.append(
            f"{method_name}: pred={pred_cf}, L2={l2_val:.3f}, DTW={dtw_val:.3f}, {'OK' if worked else 'FAIL'}"
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
    ax.set_title(f"UMAP projection — FaultDetectionA sample {rep_idx}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP plot: {out_path}")


def plot_waveforms(results_df: pd.DataFrame, all_cf_outputs: dict, selected_indices: list, out_path: str) -> None:
    sample_success_counts = (
        results_df[results_df["sample_idx"].isin(selected_indices)]
        .groupby("sample_idx")["success"]
        .sum()
        .sort_values(ascending=False)
    )
    full_success = [int(i) for i, c in sample_success_counts.items() if int(c) == len(METHOD_ORDER)]
    rep_idx = full_success[0] if full_success else int(sample_success_counts.index[0])

    rep_payload = all_cf_outputs[rep_idx]
    x = to_channel_first(rep_payload["sample"])[0]
    x_axis = np.arange(len(x))

    sample_results = results_df[results_df["sample_idx"] == rep_idx]
    info_lookup = {
        row["method"]: {"success": bool(row["success"]), "pred": int(row["pred_cf"]) if not pd.isna(row["pred_cf"]) else None}
        for _, row in sample_results.iterrows()
    }
    orig_row = sample_results.iloc[0]
    true_label = int(orig_row["true_label"])
    initial_pred = int(orig_row["pred_orig"])

    n_rows = 1 + len(METHOD_ORDER)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.8 * n_rows), sharex=True)

    status_parts = [
        f"{m}: {'OK' if info_lookup.get(m, {}).get('success', False) else 'FAIL'}" for m in METHOD_ORDER
    ]
    fig.suptitle(
        f"Waveforms for FaultDetectionA sample {rep_idx} | true={true_label} | pred={initial_pred} | " + " | ".join(status_parts),
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
        worked = info.get("success", False)
        pred_cf = info.get("pred", None)

        ax.plot(x, label="original", color="black", linewidth=1.0, linestyle="--", alpha=0.25)
        if cf is None:
            ax.text(0.5, 0.5, f"{method_name}: no counterfactual", ha="center", va="center", transform=ax.transAxes)
        else:
            cf_line = to_channel_first(cf)[0]
            ax.plot(cf_line, label=method_name, linewidth=1.6, alpha=0.95, color="#1f77b4")
            ax.fill_between(x_axis, x, cf_line, color="#1f77b4", alpha=0.18, linewidth=0)
            diff_abs = np.abs(cf_line - x)
            top_idx = np.argsort(diff_abs)[-3:] if diff_abs.size >= 3 else np.arange(diff_abs.size)
            ax.scatter(x_axis[top_idx], cf_line[top_idx], s=18, color="#d62728", alpha=0.9, zorder=4)

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
    parser = argparse.ArgumentParser(description="Compare IMFACT vs Native Guide vs Wachter on FaultDetectionA")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of correctly classified test samples to evaluate")
    parser.add_argument("--out-dir", type=str, default=os.path.join(SCRIPT_DIR, "results"), help="Directory for saved outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Repo root: {REPO_ROOT}")

    _, dataset_train = get_UCR_UEA_dataloader(dataset_name="FaultDetectionA", split="train")
    _, dataset_test = get_UCR_UEA_dataloader(dataset_name="FaultDetectionA", split="test")

    output_classes = dataset_train.y_shape[1]
    dataset_length = int(dataset_train.X_shape[2])

    models_dir = os.path.join(REPO_ROOT, "models")
    candidates = [
        os.path.join(models_dir, "simple_cnn_faultdetectiona_3_len5120.pth"),
        os.path.join(models_dir, "simple_cnn_faultdetectiona_3.pth"),
    ]

    model = None
    for model_path in candidates:
        if not os.path.exists(model_path):
            continue
        state = torch.load(model_path, map_location=device)
        try:
            candidate = SimpleCNN(output_channels=output_classes, input_length=dataset_length).to(device)
            candidate.load_state_dict(state)
            model = candidate
            print(f"Loaded model: {model_path}")
            break
        except RuntimeError as exc:
            print(f"Skipping {model_path}: {exc}")

    if model is None:
        raise FileNotFoundError("No compatible FaultDetectionA checkpoint found.")

    model.eval()

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

    results_df, all_cf_outputs = evaluate(selected_indices, dataset_test, model, device, reference_data)

    summary_df = build_summary(results_df)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    csv_path = os.path.join(args.out_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV: {csv_path}")

    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    plot_bar_metrics(summary_df, os.path.join(args.out_dir, "bar_metrics.png"))
    plot_umap(results_df, all_cf_outputs, selected_indices, dataset_test, os.path.join(args.out_dir, "umap_projection.png"))
    plot_waveforms(results_df, all_cf_outputs, selected_indices, os.path.join(args.out_dir, "waveforms.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
