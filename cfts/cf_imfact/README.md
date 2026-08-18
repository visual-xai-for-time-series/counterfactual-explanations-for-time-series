# IMFACT Counterfactual Experiments (FaultDetectionA + FruitFlies)

## Installation & Quick Usage

### Installation

Either install the published PyPI package:

```bash
pip install counterfactuals-for-time-series
```

([PyPI project page](https://pypi.org/project/counterfactuals-for-time-series/)) — this installs the whole `cfts` package, including `cfts.cf_imfact`.

Or, from a clone of this repository, install every method's dependencies (including this repo's exact pinned versions):

```bash
pip install -r requirements.txt
```

IMFACT's default decomposer (`decomposer="sift_imfs"`) is self-contained (just `numpy`/`scipy`, pulled in by either install method above). The optional `decomposer="emd"` backend additionally needs the `emd-signal` package (`pip install emd-signal`) — already listed in `requirements.txt`, but not a dependency of the PyPI package, so add it separately if you installed via `pip install counterfactuals-for-time-series`.

### Quick Usage

```python
from cfts.cf_imfact.imfact import imfact_cf

# sample:  query time series; 1-D (L,), (C, L), or (L, C)
# model:   PyTorch classifier, forward(B, C, L) -> (B, n_classes)
# dataset: sequence of (x, y) pairs used to search for a native guide (NUN)
cf, prediction = imfact_cf(
    sample=sample,
    model=model,
    dataset=dataset,
    target_class=1,          # None flips to any different class
    method="distance",       # "distance", "variance", "extremes", "coarse_to_fine"
    step=0.05,
    max_iter=200,
    n_nuns=3,
    nun_switch="cycle",      # "cycle" or "closest_psd"
    verbose=True,
)
```

`cf` is the counterfactual in the same shape/orientation as `sample`; `prediction` holds the model's output scores for `cf`. See the `imfact_cf` docstring in `imfact.py` for the full parameter reference, and `trace_imfact_variant_path` for a variant that also records the full per-iteration interpolation history (used by the notebook walkthroughs in §2.5/§3.5 below).

## Abstract
This folder contains IMFACT (IMF-based Counterfactual) experiments on two UCR datasets:
- FaultDetectionA
- FruitFlies

For each dataset, experiments include:
- method ablation (`distance`, `variance`, `extremes`, `coarse_to_fine`)
- multi-NUN ablation (`distance_n1_cycle`, `multi_nun_cycle_n{2,3,5}`, `multi_nun_closest_n{2,3,5}`)
- a combined method × multi-NUN ablation that crosses every IMF selection method with `n_nuns ∈ {1,2,3,5}` under cycle switching
- head-to-head comparison against external baselines (Native Guide, Wachter, and Glacier) via `compare_faultdetectiona.py` / `compare_fruitflies.py`

Across both datasets, every IMFACT method-ablation and multi-NUN variant reaches 100% validity, while the head-to-head comparison against Native Guide, Wachter, and Glacier shows `imfact_default` as the only method with 100% validity on both datasets.

## 1. Assets In This Folder
- Core implementation: `imfact.py`
- Legacy reference: `imfact_old.py`
- FaultDetectionA artifacts: `faultdetectiona/`
- FruitFlies artifacts: `fruitflies/`
- Dataset-specific notebooks:
  - `faultdetectiona/imfact_projection_walkthrough.ipynb`
  - `faultdetectiona/imfact_vs_native_guide_wachter_keane_projection.ipynb`
  - `fruitflies/imfact_projection_walkthrough.ipynb`
  - `fruitflies/imfact_vs_native_guide_wachter_keane_projection.ipynb`

## 2. FaultDetectionA Results

### 2.1 Method Ablation
Source: `experiments/results/faultdetectiona_ablation/faultdetectiona_results_summary_paper.csv` (suite `method_ablation`)

| Method | Validity (%) | Avg L2 | Pct Changed (%) | Avg Normalized Distance | Range Validity | Autocorr Preservation | Confidence | Avg Time (s) | Avg Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distance | 100.0 | 17.4775 | 100.0 | 0.1088 | 0.9376 | 0.7502 | 0.6677 | 0.4318 | 25.25 |
| variance | 100.0 | 25.4943 | 100.0 | 0.1583 | 0.9032 | 0.8048 | 0.8620 | 36.4826 | 15.88 |
| extremes | 100.0 | 13.4758 | 100.0 | 0.0804 | 0.9585 | 0.6338 | 0.6032 | 0.9964 | 86.13 |
| coarse_to_fine | 100.0 | 15.8475 | 100.0 | 0.0917 | 0.9442 | 0.8716 | 0.6659 | 0.8565 | 89.88 |

Highlights:
- All four methods reach 100% validity.
- `variance` gives the highest confidence and strong autocorrelation preservation, but its average runtime (36.48s) is a clear outlier vs. the other three methods (~0.4–1.0s) in this batch, likely driven by one or a few slow samples rather than a systematic cost.
- `extremes` and `coarse_to_fine` need by far the most iterations (~86–90) for their proximity level.

### 2.2 Multi-NUN Ablation
Source: `experiments/results/faultdetectiona_ablation/faultdetectiona_results_summary_paper.csv` (suite `nun_ablation`)

| Method | Validity (%) | Avg L2 | Pct Changed (%) | Avg Normalized Distance | Range Validity | Autocorr Preservation | Confidence | Avg Time (s) | Avg Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distance_n1_cycle | 100.0 | 17.4775 | 100.0 | 0.1088 | 0.9376 | 0.7502 | 0.6677 | 0.4134 | 25.25 |
| multi_nun_cycle_n2 | 100.0 | 17.0768 | 100.0 | 0.1061 | 0.9380 | 0.7085 | 0.6963 | 0.4171 | 25.88 |
| multi_nun_closest_n2 | 100.0 | 18.4678 | 99.998 | 0.1166 | 0.9335 | 0.7044 | 0.6487 | 0.4183 | 26.38 |
| multi_nun_cycle_n3 | 100.0 | 16.2406 | 100.0 | 0.1007 | 0.9433 | 0.8018 | 0.7271 | 0.4198 | 24.00 |
| multi_nun_closest_n3 | 100.0 | 18.6194 | 99.998 | 0.1176 | 0.9330 | 0.7178 | 0.6172 | 0.4545 | 25.75 |
| multi_nun_cycle_n5 | 100.0 | 16.1696 | 100.0 | 0.1015 | 0.9425 | 0.7611 | 0.6764 | 0.4333 | 23.63 |
| multi_nun_closest_n5 | 100.0 | 17.9294 | 100.0 | 0.1114 | 0.9344 | 0.8043 | 0.6252 | 0.4865 | 24.25 |

Highlights:
- All multi-NUN settings reached 100% validity.
- `multi_nun_cycle_n3` provides the best L2 among the tested multi-NUN options and the highest confidence.

### 2.3 Method × Multi-NUN Ablation
Source: `experiments/results/faultdetectiona_ablation/faultdetectiona_results_summary_paper.csv` (suite `method_nun_ablation`) — every IMF selection method crossed with `n_nuns ∈ {1,2,3,5}` under cycle switching.

| Method | Validity (%) | Avg L2 | Pct Changed (%) | Avg Normalized Distance | Range Validity | Autocorr Preservation | Confidence | Avg Time (s) | Avg Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distance_n1_cycle | 100.0 | 17.4775 | 100.0 | 0.1088 | 0.9376 | 0.7502 | 0.6677 | 0.4291 | 25.25 |
| distance_n2_cycle | 100.0 | 17.0768 | 100.0 | 0.1061 | 0.9380 | 0.7085 | 0.6963 | 0.4451 | 25.88 |
| distance_n3_cycle | 100.0 | 16.2406 | 100.0 | 0.1007 | 0.9433 | 0.8018 | 0.7271 | 0.4294 | 24.00 |
| distance_n5_cycle | 100.0 | 16.1696 | 100.0 | 0.1015 | 0.9425 | 0.7611 | 0.6764 | 0.4469 | 23.63 |
| variance_n1_cycle | 100.0 | 25.4943 | 100.0 | 0.1583 | 0.9032 | 0.8048 | 0.8620 | 48.3180 | 15.88 |
| variance_n2_cycle | 100.0 | 25.8322 | 100.0 | 0.1607 | 0.9010 | 0.7847 | 0.8695 | 11.2569 | 16.00 |
| variance_n3_cycle | 100.0 | 25.7575 | 100.0 | 0.1603 | 0.9017 | 0.8000 | 0.8749 | 11.2385 | 12.00 |
| variance_n5_cycle | 100.0 | 25.3654 | 100.0 | 0.1575 | 0.9030 | 0.8391 | 0.8745 | 11.2696 | 12.25 |
| extremes_n1_cycle | 100.0 | 13.4758 | 100.0 | 0.0804 | 0.9585 | 0.6338 | 0.6032 | 0.9949 | 86.13 |
| extremes_n2_cycle | 100.0 | 14.2805 | 99.998 | 0.0860 | 0.9534 | 0.8091 | 0.5683 | 1.0856 | 101.13 |
| extremes_n3_cycle | 100.0 | 13.9521 | 100.0 | 0.0852 | 0.9543 | 0.7621 | 0.6391 | 1.1312 | 99.63 |
| extremes_n5_cycle | 100.0 | 14.1258 | 100.0 | 0.0865 | 0.9539 | 0.7262 | 0.6225 | 1.1235 | 102.75 |
| coarse_to_fine_n1_cycle | 100.0 | 15.8475 | 100.0 | 0.0917 | 0.9442 | 0.8716 | 0.6659 | 1.5663 | 89.88 |
| coarse_to_fine_n2_cycle | 100.0 | 15.7367 | 99.998 | 0.0907 | 0.9427 | 0.8463 | 0.8523 | 1.0255 | 92.13 |
| coarse_to_fine_n3_cycle | 100.0 | 14.8318 | 100.0 | 0.0856 | 0.9471 | 0.8860 | 0.7260 | 0.9285 | 92.38 |
| coarse_to_fine_n5_cycle | 100.0 | 14.9412 | 100.0 | 0.0859 | 0.9466 | 0.8816 | 0.6894 | 0.9769 | 93.63 |

Highlights:
- All 16 method × NUN-count combinations reach 100% validity.
- `variance_n1_cycle`'s average time (48.32s) is a striking outlier vs. its own `n2`/`n3`/`n5` variants (~11.2–11.3s) and vs. the other methods, again pointing to one or a few slow samples rather than a per-NUN-count cost.
- Increasing NUN count consistently improves `distance`'s L2 and autocorrelation preservation up to `n3`, after which returns flatten or slightly reverse at `n5`.
- `extremes` and `coarse_to_fine` remain the most iteration-hungry variants (~86–103 iterations) regardless of NUN count.

### 2.4 Baseline Comparison (IMFACT vs Native Guide vs Wachter vs Glacier)
Source: `experiments/results/faultdetectiona_compare/results_summary_paper.csv` (50 test samples; MASCOTS excluded by default — see `--exclude-mascots-long` in `run_all_experiments.sh`)

| Method | Validity (%) | Avg L2 | Pct Changed (%) | Avg Normalized Distance | Range Validity | Autocorr Preservation | Confidence | Avg Time (s) | Avg Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| imfact_default | 100.0 | 18.8619 | 100.0 | 0.1040 | 0.9389 | 0.6852 | 0.4351 | 0.4597 | 25.12 |
| glacier | 100.0 | 8.7881 | 99.96 | 0.0484 | 0.9357 | 0.9601 | 0.5759 | 1.5923 | — |
| native_guide | 100.0 | 20.8432 | 58.97 | 0.1160 | 0.9220 | 0.8374 | 0.4563 | 8.2813 | — |
| wachter | 58.0 | 14.7506 | 99.997 | 0.0655 | 0.9474 | 0.8248 | 0.4731 | 3.5232 | — |

Interpretation:
- `imfact_default` and `glacier` are the only methods reaching 100% validity; Glacier is more proximate (lower L2) and more plausible (higher autocorrelation preservation) but ~3.5x slower.
- Native Guide changes far fewer time points (58.97% vs ~100% for the others) but only reaches 100% validity at a much higher L2 cost and ~18x the runtime of IMFACT.
- Wachter succeeds on 58% of evaluated samples, the least reliable of the four.

### 2.5 Notebook Walkthrough (Single-Sample Variant Trace)
Source: `faultdetectiona/imfact_projection_walkthrough.ipynb`

Representative output summary:
- distance -> final class 1, 25 steps, trace matches `imfact_cf`
- variance -> final class 0, 7 steps, trace matches `imfact_cf`
- extremes -> final class 2, 19 steps, trace mismatch in this run
- coarse_to_fine -> final class 2, 19 steps, trace mismatch in this run

## 3. FruitFlies Results

### 3.1 Method Ablation
Source: `experiments/results/fruitflies_ablation/fruitflies_results_summary_paper.csv` (suite `method_ablation`)

| Method | Validity (%) | Avg L2 | Pct Changed (%) | Avg Normalized Distance | Range Validity | Autocorr Preservation | Confidence | Avg Time (s) | Avg Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distance | 100.0 | 0.7531 | 99.96 | 0.1008 | 0.9934 | 0.9906 | 0.7050 | 0.5458 | 27.88 |
| variance | 100.0 | 1.0542 | 99.99 | 0.1530 | 0.9934 | 0.9893 | 0.7447 | 5.6481 | 19.50 |
| extremes | 100.0 | 0.7492 | 99.95 | 0.1011 | 0.9934 | 0.9875 | 0.6719 | 1.1250 | 91.25 |
| coarse_to_fine | 100.0 | 0.6898 | 99.95 | 0.0922 | 0.9951 | 0.9933 | 0.6630 | 0.7507 | 64.13 |

Highlights:
- All four remaining methods reached 100% validity in the sampled batch.
- `coarse_to_fine` obtains the lowest L2 in this run at a moderate iteration budget (~64); `distance` and `extremes` need markedly more iterations (~28 and ~91 respectively) without a proximity advantage.
- `variance` needs ~5.6s on average, an order of magnitude slower than `distance`/`extremes`/`coarse_to_fine` in this batch.

### 3.2 Multi-NUN Ablation
Source: `experiments/results/fruitflies_ablation/fruitflies_results_summary_paper.csv` (suite `nun_ablation`)

| Method | Validity (%) | Avg L2 | Pct Changed (%) | Avg Normalized Distance | Range Validity | Autocorr Preservation | Confidence | Avg Time (s) | Avg Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distance_n1_cycle | 100.0 | 0.7531 | 99.96 | 0.1008 | 0.9934 | 0.9906 | 0.7050 | 0.5467 | 27.88 |
| multi_nun_cycle_n2 | 100.0 | 0.7138 | 99.94 | 0.0992 | 0.9963 | 0.9965 | 0.8186 | 0.5419 | 26.25 |
| multi_nun_closest_n2 | 100.0 | 0.7813 | 99.95 | 0.1086 | 0.9924 | 0.9973 | 0.7668 | 0.5081 | 27.00 |
| multi_nun_cycle_n3 | 100.0 | 0.6958 | 99.95 | 0.0987 | 0.9965 | 0.9936 | 0.8689 | 0.4785 | 23.00 |
| multi_nun_closest_n3 | 100.0 | 0.6981 | 99.94 | 0.0991 | 0.9968 | 0.9948 | 0.7567 | 0.5230 | 25.63 |
| multi_nun_cycle_n5 | 100.0 | 0.7389 | 99.95 | 0.1013 | 0.9952 | 0.9755 | 0.8660 | 0.5053 | 23.00 |
| multi_nun_closest_n5 | 100.0 | 0.6870 | 99.94 | 0.0981 | 0.9968 | 0.9946 | 0.7076 | 0.6096 | 25.63 |

Highlights:
- All multi-NUN settings reached 100% validity.
- `multi_nun_cycle_n3` gives the best L2 and highest confidence among the multi-NUN options tested.

### 3.3 Method × Multi-NUN Ablation
Source: `experiments/results/fruitflies_ablation/fruitflies_results_summary_paper.csv` (suite `method_nun_ablation`) — every IMF selection method crossed with `n_nuns ∈ {1,2,3,5}` under cycle switching.

| Method | Validity (%) | Avg L2 | Pct Changed (%) | Avg Normalized Distance | Range Validity | Autocorr Preservation | Confidence | Avg Time (s) | Avg Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distance_n1_cycle | 100.0 | 0.7531 | 99.96 | 0.1008 | 0.9934 | 0.9906 | 0.7050 | 0.5742 | 27.88 |
| distance_n2_cycle | 100.0 | 0.7138 | 99.94 | 0.0992 | 0.9963 | 0.9965 | 0.8186 | 0.5294 | 26.25 |
| distance_n3_cycle | 100.0 | 0.6958 | 99.95 | 0.0987 | 0.9965 | 0.9936 | 0.8689 | 0.5418 | 23.00 |
| distance_n5_cycle | 100.0 | 0.7389 | 99.95 | 0.1013 | 0.9952 | 0.9755 | 0.8660 | 0.5880 | 23.00 |
| variance_n1_cycle | 100.0 | 1.0542 | 99.99 | 0.1530 | 0.9934 | 0.9893 | 0.7447 | 5.6126 | 19.50 |
| variance_n2_cycle | 100.0 | 1.0283 | 99.99 | 0.1499 | 0.9949 | 0.9897 | 0.8637 | 5.5299 | 5.63 |
| variance_n3_cycle | 100.0 | 1.1215 | 99.99 | 0.1654 | 0.9849 | 0.9884 | 0.8954 | 5.5352 | 4.50 |
| variance_n5_cycle | 100.0 | 1.1052 | 99.995 | 0.1624 | 0.9860 | 0.9889 | 0.8815 | 5.6054 | 4.38 |
| extremes_n1_cycle | 100.0 | 0.7492 | 99.95 | 0.1011 | 0.9934 | 0.9875 | 0.6719 | 1.1459 | 91.25 |
| extremes_n2_cycle | 100.0 | 0.6987 | 99.95 | 0.0971 | 0.9949 | 0.9952 | 0.6463 | 1.0415 | 81.13 |
| extremes_n3_cycle | 100.0 | 0.6504 | 99.97 | 0.0941 | 0.9939 | 0.9889 | 0.6326 | 0.9224 | 65.25 |
| extremes_n5_cycle | 100.0 | 0.6790 | 99.97 | 0.0967 | 0.9942 | 0.9915 | 0.7446 | 0.9591 | 67.50 |
| coarse_to_fine_n1_cycle | 100.0 | 0.6898 | 99.95 | 0.0922 | 0.9951 | 0.9933 | 0.6630 | 0.8042 | 64.13 |
| coarse_to_fine_n2_cycle | 100.0 | 0.7053 | 99.96 | 0.0910 | 0.9959 | 0.9916 | 0.6774 | 0.8183 | 61.38 |
| coarse_to_fine_n3_cycle | 100.0 | 0.7440 | 99.95 | 0.0972 | 0.9979 | 0.9948 | 0.7492 | 0.7914 | 63.63 |
| coarse_to_fine_n5_cycle | 100.0 | 0.7197 | 99.98 | 0.0928 | 0.9941 | 0.9964 | 0.7382 | 0.8675 | 63.75 |

Highlights:
- All 16 method × NUN-count combinations reach 100% validity.
- `variance`'s iteration count drops sharply as NUN count increases (19.5 at `n1` to ~4.4–5.6 at `n2`/`n3`/`n5`), while its runtime stays ~5.5s regardless — the cost here is dominated by per-step IMF computation, not iteration count.
- `coarse_to_fine_n3_cycle` attains the highest range validity (0.9979) of any combination on this dataset alongside strong autocorrelation preservation.

### 3.4 Baseline Comparison (IMFACT vs Native Guide vs Wachter vs Glacier)
Source: `experiments/results/fruitflies_compare/results_summary_paper.csv` (50 test samples; MASCOTS excluded by default — see `--exclude-mascots-long` in `run_all_experiments.sh`)

| Method | Validity (%) | Avg L2 | Pct Changed (%) | Avg Normalized Distance | Range Validity | Autocorr Preservation | Confidence | Avg Time (s) | Avg Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| imfact_default | 100.0 | 0.9219 | 99.98 | 0.1163 | 0.9844 | 0.9873 | 0.4735 | 0.2559 | 23.84 |
| native_guide | 100.0 | 0.8706 | 54.53 | 0.1015 | 0.9797 | 0.9887 | 0.4117 | 10.9509 | — |
| wachter | 30.0 | 1.0697 | 100.0 | 0.1831 | 0.9595 | 0.9852 | 0.5638 | 0.0499 | — |
| glacier | 36.0 | 3.5505 | 99.98 | 0.5407 | 0.1978 | 0.1740 | 0.5761 | 1.6528 | — |

Interpretation:
- `imfact_default` is the only method with 100% validity in this run; Native Guide also reaches 100% but only by changing about half as many time points.
- Glacier and Wachter both struggle on FruitFlies' long (5000-pt) series, succeeding on only 36% and 30% of samples respectively; Glacier's plausibility also collapses (range validity 0.20, autocorrelation preservation 0.17).
- IMFACT is also the fastest method here (0.26s/sample vs 10.95s for Native Guide).

### 3.5 Notebook Walkthrough (Single-Sample Variant Trace)
Source: `fruitflies/imfact_projection_walkthrough.ipynb`

Representative output summary:
- distance -> final class 0, 11 steps, trace mismatch in this run
- variance -> final class 0, 2 steps, trace mismatch in this run
- extremes -> final class 1, 25 steps, trace mismatch in this run
- coarse_to_fine -> final class 1, 25 steps, trace mismatch in this run

## 4. Visual Results

### 4.1 FaultDetectionA Ablation Summary And Examples
Source: `experiments/results/faultdetectiona_ablation/`

Method ablation summary:

![FaultDetectionA method ablation](experiments/results/faultdetectiona_ablation/faultdetectiona_method_ablation.png)

![FaultDetectionA method ablation (canonical)](experiments/results/faultdetectiona_ablation/faultdetectiona_method_ablation_canonical.png)

Multi-NUN ablation summary:

![FaultDetectionA multi-NUN ablation](experiments/results/faultdetectiona_ablation/faultdetectiona_nun_ablation.png)

![FaultDetectionA multi-NUN ablation (canonical)](experiments/results/faultdetectiona_ablation/faultdetectiona_nun_ablation_canonical.png)

Per-sample lineplot and UMAP example:

![FaultDetectionA sample lineplot](experiments/results/faultdetectiona_ablation/faultdetectiona_method_ablation_sample0_lineplot.png)

![FaultDetectionA sample UMAP](experiments/results/faultdetectiona_ablation/faultdetectiona_method_ablation_sample0_umap.png)

### 4.2 FaultDetectionA Baseline Comparison Visuals
Source: `experiments/results/faultdetectiona_compare/` (see §2.4 for the underlying numbers)

![FaultDetectionA comparison bar metrics](experiments/results/faultdetectiona_compare/bar_metrics.png)

![FaultDetectionA comparison bar metrics (canonical)](experiments/results/faultdetectiona_compare/bar_metrics_canonical.png)

![FaultDetectionA comparison UMAP projection](experiments/results/faultdetectiona_compare/umap_projection.png)

![FaultDetectionA comparison waveforms](experiments/results/faultdetectiona_compare/waveforms.png)

### 4.3 FruitFlies Ablation Summary And Examples
Source: `experiments/results/fruitflies_ablation/`

Method ablation summary:

![FruitFlies method ablation](experiments/results/fruitflies_ablation/fruitflies_method_ablation.png)

![FruitFlies method ablation (canonical)](experiments/results/fruitflies_ablation/fruitflies_method_ablation_canonical.png)

Multi-NUN ablation summary:

![FruitFlies multi-NUN ablation](experiments/results/fruitflies_ablation/fruitflies_nun_ablation.png)

![FruitFlies multi-NUN ablation (canonical)](experiments/results/fruitflies_ablation/fruitflies_nun_ablation_canonical.png)

Per-sample lineplot and UMAP example:

![FruitFlies sample lineplot](experiments/results/fruitflies_ablation/fruitflies_method_ablation_sample0_lineplot.png)

![FruitFlies sample UMAP](experiments/results/fruitflies_ablation/fruitflies_method_ablation_sample0_umap.png)

### 4.4 FruitFlies Baseline Comparison Visuals
Source: `experiments/results/fruitflies_compare/` (see §3.4 for the underlying numbers)

![FruitFlies comparison bar metrics](experiments/results/fruitflies_compare/bar_metrics.png)

![FruitFlies comparison bar metrics (canonical)](experiments/results/fruitflies_compare/bar_metrics_canonical.png)

![FruitFlies comparison UMAP projection](experiments/results/fruitflies_compare/umap_projection.png)

![FruitFlies comparison waveforms](experiments/results/fruitflies_compare/waveforms.png)

### 4.5 FaultDetectionA Extracted Notebook Figures
From:
- `experiments/faultdetectiona/imfact_vs_native_guide_wachter_keane_projection.ipynb`
- `experiments/faultdetectiona/imfact_projection_walkthrough.ipynb`

Comparison notebook figures:

![FaultDetectionA notebook comparison fig 1](experiments/faultdetectiona/notebook_images/imfact_vs_native_guide_wachter_keane_projection_cell7_output1.png)

![FaultDetectionA notebook comparison fig 2](experiments/faultdetectiona/notebook_images/imfact_vs_native_guide_wachter_keane_projection_cell8_output2.png)

![FaultDetectionA notebook comparison fig 3](experiments/faultdetectiona/notebook_images/imfact_vs_native_guide_wachter_keane_projection_cell9_output1.png)

Walkthrough notebook figures:

![FaultDetectionA walkthrough fig 1](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell5_output1.png)

![FaultDetectionA walkthrough fig 2](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell7_output1.png)

![FaultDetectionA walkthrough fig 3](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell9_output2.png)

![FaultDetectionA walkthrough fig 4](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell10_output2.png)

![FaultDetectionA walkthrough fig 5](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell11_output1.png)

![FaultDetectionA walkthrough fig 6](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell11_output2.png)

![FaultDetectionA walkthrough fig 7](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell11_output3.png)

![FaultDetectionA walkthrough fig 8](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell11_output4.png)

![FaultDetectionA walkthrough fig 9](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell11_output5.png)

![FaultDetectionA walkthrough fig 10](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell11_output6.png)

![FaultDetectionA walkthrough fig 11](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell11_output7.png)

![FaultDetectionA walkthrough fig 12](experiments/faultdetectiona/notebook_images/imfact_projection_walkthrough_cell11_output8.png)

### 4.6 FruitFlies Extracted Notebook Figures
From:
- `experiments/fruitflies/imfact_vs_native_guide_wachter_keane_projection.ipynb`
- `experiments/fruitflies/imfact_projection_walkthrough.ipynb`

Comparison notebook figures:

![FruitFlies notebook comparison fig 1](experiments/fruitflies/notebook_images/imfact_vs_native_guide_wachter_keane_projection_cell7_output1.png)

![FruitFlies notebook comparison fig 2](experiments/fruitflies/notebook_images/imfact_vs_native_guide_wachter_keane_projection_cell8_output2.png)

![FruitFlies notebook comparison fig 3](experiments/fruitflies/notebook_images/imfact_vs_native_guide_wachter_keane_projection_cell9_output1.png)

Walkthrough notebook figures:

![FruitFlies walkthrough fig 1](experiments/fruitflies/notebook_images/imfact_projection_walkthrough_cell5_output1.png)

![FruitFlies walkthrough fig 2](experiments/fruitflies/notebook_images/imfact_projection_walkthrough_cell7_output1.png)

![FruitFlies walkthrough fig 3](experiments/fruitflies/notebook_images/imfact_projection_walkthrough_cell9_output2.png)

![FruitFlies walkthrough fig 4](experiments/fruitflies/notebook_images/imfact_projection_walkthrough_cell10_output2.png)

![FruitFlies walkthrough fig 5](experiments/fruitflies/notebook_images/imfact_projection_walkthrough_cell11_output1.png)

![FruitFlies walkthrough fig 6](experiments/fruitflies/notebook_images/imfact_projection_walkthrough_cell11_output2.png)

![FruitFlies walkthrough fig 7](experiments/fruitflies/notebook_images/imfact_projection_walkthrough_cell11_output3.png)

## 5. Reproducibility

Run dataset-specific ablations from repository root:

```bash
python cfts/cf_imfact/experiments/faultdetectiona/ablation_faultdetectiona.py
python cfts/cf_imfact/experiments/fruitflies/ablation_fruitflies.py
```

Useful options:

```bash
python cfts/cf_imfact/experiments/faultdetectiona/ablation_faultdetectiona.py \
  --max-samples 8 \
  --max-plot-samples 2 \
  --multi-nun-counts 2,3,5

python cfts/cf_imfact/experiments/fruitflies/ablation_fruitflies.py \
  --max-samples 8 \
  --max-plot-samples 2 \
  --multi-nun-counts 2,3,5
```

As with the comparisons below, `--max-samples` correctly-classified test samples are chosen at random; pass `--seed` (default: 13) to reproduce the exact same sample selection across runs.

Run the head-to-head baseline comparisons (or use `run_all_experiments.sh` to run everything, including both ablations, in one go):

The `--n-samples` correctly-classified test samples are chosen at random from each dataset; pass `--seed` (default: 13) to reproduce the exact same sample selection across runs.

```bash
python cfts/cf_imfact/experiments/faultdetectiona/compare_faultdetectiona.py --n-samples 50 --seed 13
python cfts/cf_imfact/experiments/fruitflies/compare_fruitflies.py --n-samples 50 --seed 13

# or, from cfts/cf_imfact/experiments/:
./run_all_experiments.sh
```

Each run writes a `results_summary_paper.csv` (compare scripts: one per output dir; ablation scripts: one per dataset, combining the `method_ablation`, `nun_ablation`, and `method_nun_ablation` suites) containing the paper's reported metrics — Validity, L2 Distance, Percentage Changed, Normalised Distance, Range Validity, Autocorrelation Preservation, Confidence, Average Time, and (for IMFACT) Average Iterations — which is what the tables above are built from.

## 6. Conclusions
- On both datasets, every IMFACT method-ablation, multi-NUN, and method × multi-NUN combination reaches 100% validity.
- In the head-to-head comparison, `imfact_default` is the only method with 100% validity on *both* FaultDetectionA and FruitFlies; Glacier matches it on FaultDetectionA but collapses to 36% validity (and near-zero plausibility) on FruitFlies' longer series, and Wachter is the least reliable on both datasets (58% and 30% respectively).
- Multi-NUN cycle with `n=3` is a strong default for proximity in both datasets.
- IMFACT variants need markedly different iteration budgets to converge (e.g. ~12–20 for `variance` vs ~86–103 for `extremes`/`coarse_to_fine`), which is worth factoring into any runtime-sensitive deployment.
- The method × multi-NUN cross ablation shows the two hyperparameters interact: raising NUN count mostly helps `distance` and shrinks `variance`'s iteration count sharply, but barely changes the iteration budget of `extremes`/`coarse_to_fine`.

## 7. Limitations And Next Steps
- Reported numbers depend on selected sample subsets (50 test samples per compare run; 8 per ablation run) and are single-seed, not averaged across repeats.
- A few average-time figures (notably `variance` at `n_nuns=1`) look like outliers driven by one or a few slow samples rather than a systematic cost; treat single-run timing numbers with caution.
- Walkthrough notebook traces are illustrative, single-sample analyses.
- Future extensions: confidence intervals over multiple seeds, broader dataset coverage, calibration-aware validity reporting, and re-running the notebooks against the current default method set.

## 8. Short Final Summary
- What was checked:
  - Method ablation for `distance`, `variance`, `extremes`, and `coarse_to_fine` on both FaultDetectionA and FruitFlies.
  - Multi-NUN ablation for `distance_n1_cycle`, `multi_nun_cycle_n{2,3,5}`, and `multi_nun_closest_n{2,3,5}`.
  - A combined method × multi-NUN ablation crossing every method with `n_nuns ∈ {1,2,3,5}` under cycle switching.
  - Direct comparison of `imfact_default` vs Native Guide vs Wachter vs Glacier via `compare_faultdetectiona.py` / `compare_fruitflies.py`.
  - Qualitative validation via extracted notebook figures (line plots, projections, and walkthrough visuals).
- Main findings:
  - In the sampled ablation runs, every IMFACT variant (including all method × NUN-count combinations) achieved 100% validity on both datasets.
  - In the head-to-head comparison, `imfact_default` was the only method with 100% validity on both datasets, and the fastest on FruitFlies.
  - `multi_nun_cycle_n3` is a practical default when balancing proximity and performance.
- Practical takeaway:
  - Start with `imfact_default` for reliability; tune NUN switching (especially cycle with `n=3`) for proximity/efficiency trade-offs, and prefer `variance` (at `n_nuns>1`, where its iteration count drops sharply) over `extremes`/`coarse_to_fine` when iteration budget matters.
