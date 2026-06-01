# EMD Counterfactual Experiments (FaultDetectionA + FruitFlies)

## Abstract
This folder contains empirical mode decomposition (EMD)-based counterfactual experiments on two UCR datasets:
- FaultDetectionA
- FruitFlies

For each dataset, experiments include:
- method ablation (`distance`, `fingerprint`, `variance`, `extremes`, `maxmin`, `coarse_to_fine`)
- multi-NUN ablation (`distance_n1_cycle`, `multi_nun_cycle_n{2,3,5}`, `multi_nun_closest_n{2,3,5}`)
- notebook comparison against external baselines (Native Guide and Wachter)

Across both datasets, the ablation runs show 100% success in the configured sampled evaluation, while notebook comparisons indicate that `emd_variance_nun3` is the most reliable class-flipping method among the compared approaches.

## 1. Assets In This Folder
- Core implementation: `emd.py`
- Legacy reference: `emd_old.py`
- FaultDetectionA artifacts: `faultdetectiona/`
- FruitFlies artifacts: `fruitflies/`
- Dataset-specific notebooks:
  - `faultdetectiona/emd_projection_walkthrough.ipynb`
  - `faultdetectiona/emd_vs_native_guide_wachter_keane_projection.ipynb`
  - `fruitflies/emd_projection_walkthrough.ipynb`
  - `fruitflies/emd_vs_native_guide_wachter_keane_projection.ipynb`

## 2. FaultDetectionA Results

### 2.1 Method Ablation
Source: `faultdetectiona/emd_faultdetectiona_ablation_method_ablation.csv`

| Method | Success Rate (%) | Avg L2 | Avg Normalized Distance | Range Validity | Autocorr Preservation | Avg Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| distance | 100.0 | 17.1531 | 0.1048 | 0.9396 | 0.7575 | 0.7593 |
| fingerprint | 100.0 | 17.1531 | 0.1048 | 0.9396 | 0.7575 | 0.7251 |
| variance | 100.0 | 9.7292 | 0.0570 | 0.9755 | 0.9538 | 1.6563 |
| extremes | 100.0 | 14.8026 | 0.0881 | 0.9515 | 0.7155 | 2.1818 |
| maxmin | 100.0 | 14.8026 | 0.0881 | 0.9515 | 0.7155 | 2.1408 |
| coarse_to_fine | 100.0 | 14.9443 | 0.0867 | 0.9491 | 0.8680 | 1.9372 |

Highlights:
- All methods reached 100% success in this ablation setting.
- `variance` is best on proximity/plausibility-oriented metrics (lowest L2, highest range validity and autocorrelation preservation).
- `fingerprint` is the fastest in this suite.

### 2.2 Multi-NUN Ablation
Source: `faultdetectiona/emd_faultdetectiona_ablation_nun_ablation.csv`

| Method | Success Rate (%) | Avg L2 | Avg Normalized Distance | Range Validity | Autocorr Preservation | Avg Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| distance_n1_cycle | 100.0 | 17.1531 | 0.1048 | 0.9396 | 0.7575 | 0.6834 |
| multi_nun_cycle_n2 | 100.0 | 15.8192 | 0.0967 | 0.9455 | 0.6855 | 0.7361 |
| multi_nun_closest_n2 | 100.0 | 17.4754 | 0.1070 | 0.9389 | 0.7009 | 0.7766 |
| multi_nun_cycle_n3 | 100.0 | 15.6591 | 0.0978 | 0.9467 | 0.7261 | 0.8398 |
| multi_nun_closest_n3 | 100.0 | 18.0238 | 0.1108 | 0.9373 | 0.7143 | 0.8649 |
| multi_nun_cycle_n5 | 100.0 | 16.2817 | 0.1013 | 0.9414 | 0.7081 | 0.8634 |
| multi_nun_closest_n5 | 100.0 | 17.9374 | 0.1106 | 0.9354 | 0.8303 | 0.8137 |

Highlights:
- All multi-NUN settings reached 100% success.
- `multi_nun_cycle_n3` provides the best L2 among the tested multi-NUN options.

### 2.3 Notebook Baseline Comparison (EMD vs Native Guide vs Wachter)
Source: `faultdetectiona/emd_vs_native_guide_wachter_keane_projection.ipynb`

| Method | n_total | n_successful | Success Rate (%) | Validity Mean | L2 Mean | DTW Mean | Sparsity Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| emd_variance_nun3 | 50 | 50 | 100.0 | 1.00 | 13.5032 | 9.6027 | 0.001223 |
| native_guide | 50 | 43 | 86.0 | 0.00 | 0.0000 | 0.0000 | 1.000000 |
| wachter | 50 | 19 | 38.0 | 0.56 | 18.1588 | 13.1216 | 0.000021 |

Interpretation:
- `emd_variance_nun3` is strongest in reliability (100% success) and validity.
- Native Guide is much less successful at targeted flips in this setup.
- Wachter succeeds on fewer than half of evaluated samples here.

### 2.4 Notebook Walkthrough (Single-Sample Variant Trace)
Source: `faultdetectiona/emd_projection_walkthrough.ipynb`

Representative output summary:
- distance -> final class 1, 25 steps, trace matches `emd_cf`
- fingerprint -> final class 1, 25 steps, trace matches `emd_cf`
- variance -> final class 0, 7 steps, trace matches `emd_cf`
- extremes -> final class 2, 19 steps, trace mismatch in this run
- maxmin -> final class 2, 19 steps, trace mismatch in this run
- coarse_to_fine -> final class 2, 19 steps, trace mismatch in this run

## 3. FruitFlies Results

### 3.1 Method Ablation
Source: `fruitflies/emd_fruitflies_ablation_method_ablation.csv`

| Method | Success Rate (%) | Avg L2 | Avg Normalized Distance | Range Validity | Autocorr Preservation | Avg Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| distance | 100.0 | 0.6415 | 0.1388 | 0.9980 | 0.9649 | 3.9824 |
| fingerprint | 100.0 | 0.6415 | 0.1388 | 0.9980 | 0.9649 | 3.4944 |
| variance | 100.0 | 0.6311 | 0.1357 | 0.9999 | 0.9811 | 8.9224 |
| extremes | 100.0 | 0.5490 | 0.1187 | 0.9934 | 0.8239 | 4.4325 |
| maxmin | 100.0 | 0.5490 | 0.1187 | 0.9934 | 0.8239 | 3.9926 |
| coarse_to_fine | 100.0 | 0.6086 | 0.1255 | 0.9988 | 0.9663 | 3.8394 |

Highlights:
- All methods reached 100% success in the sampled batch.
- `extremes` and `maxmin` obtain the lowest L2 in this run.
- `variance` has the strongest range-validity/autocorrelation profile, but at higher runtime.

### 3.2 Multi-NUN Ablation
Source: `fruitflies/emd_fruitflies_ablation_nun_ablation.csv`

| Method | Success Rate (%) | Avg L2 | Avg Normalized Distance | Range Validity | Autocorr Preservation | Avg Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| distance_n1_cycle | 100.0 | 0.6415 | 0.1388 | 0.9980 | 0.9649 | 4.0881 |
| multi_nun_cycle_n2 | 100.0 | 0.6230 | 0.1324 | 0.9979 | 0.9733 | 3.5734 |
| multi_nun_closest_n2 | 100.0 | 0.6716 | 0.1433 | 0.9979 | 0.9760 | 3.9147 |
| multi_nun_cycle_n3 | 100.0 | 0.5705 | 0.1266 | 0.9964 | 0.9640 | 3.6937 |
| multi_nun_closest_n3 | 100.0 | 0.6669 | 0.1429 | 0.9979 | 0.9741 | 3.7737 |
| multi_nun_cycle_n5 | 100.0 | 0.6354 | 0.1526 | 0.9824 | 0.9609 | 3.6791 |
| multi_nun_closest_n5 | 100.0 | 0.7342 | 0.1637 | 0.9880 | 0.9744 | 3.7163 |

Highlights:
- All settings reached 100% success.
- `multi_nun_cycle_n3` provides the best L2 among tested multi-NUN options.

### 3.3 Notebook Baseline Comparison (EMD vs Native Guide vs Wachter)
Source: `fruitflies/emd_vs_native_guide_wachter_keane_projection.ipynb`

| Method | n_total | n_successful | Success Rate (%) | Validity Mean | L2 Mean | DTW Mean | Sparsity Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| emd_variance_nun3 | 50 | 50 | 100.0 | 1.00 | 0.6701 | 0.5021 | 0.000320 |
| native_guide | 50 | 0 | 0.0 | 0.08 | NaN | NaN | NaN |
| wachter | 50 | 29 | 58.0 | 0.86 | 0.7441 | 0.5228 | 0.000069 |

Interpretation:
- `emd_variance_nun3` is the only method with 100% success and perfect validity in this notebook run.
- Native Guide did not produce successful target-class flips in this 50-sample setting.
- Wachter improves over Native Guide on success, but remains below EMD.

### 3.4 Notebook Walkthrough (Single-Sample Variant Trace)
Source: `fruitflies/emd_projection_walkthrough.ipynb`

Representative output summary:
- distance -> final class 0, 11 steps, trace mismatch in this run
- fingerprint -> final class 0, 11 steps, trace mismatch in this run
- variance -> final class 0, 2 steps, trace mismatch in this run
- extremes -> final class 1, 25 steps, trace mismatch in this run
- maxmin -> final class 1, 25 steps, trace mismatch in this run
- coarse_to_fine -> final class 1, 25 steps, trace mismatch in this run

## 4. Visual Results

### 4.1 FaultDetectionA Summary And Examples
Method ablation summary:

![FaultDetectionA method ablation](faultdetectiona/emd_faultdetectiona_ablation_method_ablation.png)

Multi-NUN ablation summary:

![FaultDetectionA multi-NUN ablation](faultdetectiona/emd_faultdetectiona_ablation_nun_ablation.png)

Per-sample lineplot and UMAP example:

![FaultDetectionA sample lineplot](faultdetectiona/emd_faultdetectiona_ablation_method_ablation_sample0_lineplot.png)

![FaultDetectionA sample UMAP](faultdetectiona/emd_faultdetectiona_ablation_method_ablation_sample0_umap.png)

### 4.2 FruitFlies Summary And Examples
Method ablation summary:

![FruitFlies method ablation](fruitflies/emd_fruitflies_ablation_method_ablation.png)

Multi-NUN ablation summary:

![FruitFlies multi-NUN ablation](fruitflies/emd_fruitflies_ablation_nun_ablation.png)

Per-sample lineplot and UMAP example:

![FruitFlies sample lineplot](fruitflies/emd_fruitflies_ablation_method_ablation_sample0_lineplot.png)

![FruitFlies sample UMAP](fruitflies/emd_fruitflies_ablation_method_ablation_sample0_umap.png)

### 4.3 FaultDetectionA Extracted Notebook Figures
From:
- `faultdetectiona/emd_vs_native_guide_wachter_keane_projection.ipynb`
- `faultdetectiona/emd_projection_walkthrough.ipynb`

Comparison notebook figures:

![FaultDetectionA notebook comparison fig 1](faultdetectiona/notebook_images/emd_vs_native_guide_wachter_keane_projection_cell7_output1.png)

![FaultDetectionA notebook comparison fig 2](faultdetectiona/notebook_images/emd_vs_native_guide_wachter_keane_projection_cell8_output2.png)

![FaultDetectionA notebook comparison fig 3](faultdetectiona/notebook_images/emd_vs_native_guide_wachter_keane_projection_cell9_output1.png)

Walkthrough notebook figures:

![FaultDetectionA walkthrough fig 1](faultdetectiona/notebook_images/emd_projection_walkthrough_cell5_output1.png)

![FaultDetectionA walkthrough fig 2](faultdetectiona/notebook_images/emd_projection_walkthrough_cell7_output1.png)

![FaultDetectionA walkthrough fig 3](faultdetectiona/notebook_images/emd_projection_walkthrough_cell9_output2.png)

![FaultDetectionA walkthrough fig 4](faultdetectiona/notebook_images/emd_projection_walkthrough_cell10_output2.png)

![FaultDetectionA walkthrough fig 5](faultdetectiona/notebook_images/emd_projection_walkthrough_cell11_output1.png)

![FaultDetectionA walkthrough fig 6](faultdetectiona/notebook_images/emd_projection_walkthrough_cell11_output2.png)

![FaultDetectionA walkthrough fig 7](faultdetectiona/notebook_images/emd_projection_walkthrough_cell11_output3.png)

![FaultDetectionA walkthrough fig 8](faultdetectiona/notebook_images/emd_projection_walkthrough_cell11_output4.png)

![FaultDetectionA walkthrough fig 9](faultdetectiona/notebook_images/emd_projection_walkthrough_cell11_output5.png)

![FaultDetectionA walkthrough fig 10](faultdetectiona/notebook_images/emd_projection_walkthrough_cell11_output6.png)

![FaultDetectionA walkthrough fig 11](faultdetectiona/notebook_images/emd_projection_walkthrough_cell11_output7.png)

![FaultDetectionA walkthrough fig 12](faultdetectiona/notebook_images/emd_projection_walkthrough_cell11_output8.png)

### 4.4 FruitFlies Extracted Notebook Figures
From:
- `fruitflies/emd_vs_native_guide_wachter_keane_projection.ipynb`
- `fruitflies/emd_projection_walkthrough.ipynb`

Comparison notebook figures:

![FruitFlies notebook comparison fig 1](fruitflies/notebook_images/emd_vs_native_guide_wachter_keane_projection_cell7_output1.png)

![FruitFlies notebook comparison fig 2](fruitflies/notebook_images/emd_vs_native_guide_wachter_keane_projection_cell8_output2.png)

![FruitFlies notebook comparison fig 3](fruitflies/notebook_images/emd_vs_native_guide_wachter_keane_projection_cell9_output1.png)

Walkthrough notebook figures:

![FruitFlies walkthrough fig 1](fruitflies/notebook_images/emd_projection_walkthrough_cell5_output1.png)

![FruitFlies walkthrough fig 2](fruitflies/notebook_images/emd_projection_walkthrough_cell7_output1.png)

![FruitFlies walkthrough fig 3](fruitflies/notebook_images/emd_projection_walkthrough_cell9_output2.png)

![FruitFlies walkthrough fig 4](fruitflies/notebook_images/emd_projection_walkthrough_cell10_output2.png)

![FruitFlies walkthrough fig 5](fruitflies/notebook_images/emd_projection_walkthrough_cell11_output1.png)

![FruitFlies walkthrough fig 6](fruitflies/notebook_images/emd_projection_walkthrough_cell11_output2.png)

![FruitFlies walkthrough fig 7](fruitflies/notebook_images/emd_projection_walkthrough_cell11_output3.png)

## 5. Reproducibility

Run dataset-specific ablations from repository root:

```bash
python cfts/cf_emd/faultdetectiona/ablation_faultdetectiona.py
python cfts/cf_emd/fruitflies/ablation_fruitflies.py
```

Useful options:

```bash
python cfts/cf_emd/faultdetectiona/ablation_faultdetectiona.py \
  --max-samples 8 \
  --max-plot-samples 2 \
  --multi-nun-counts 2,3,5

python cfts/cf_emd/fruitflies/ablation_fruitflies.py \
  --max-samples 8 \
  --max-plot-samples 2 \
  --multi-nun-counts 2,3,5
```

## 6. Conclusions
- On both datasets, EMD ablations are robust in the evaluated sampled setup (100% success across listed settings).
- In notebook baseline comparisons, `emd_variance_nun3` consistently outperforms Native Guide and Wachter on reliability.
- Multi-NUN cycle with `n=3` is a strong default for proximity in both datasets.

## 7. Limitations And Next Steps
- Reported numbers depend on selected sample subsets and notebook run configurations.
- Walkthrough notebook traces are illustrative, single-sample analyses.
- Future extensions: confidence intervals over multiple seeds, broader dataset coverage, and calibration-aware validity reporting.

## 8. Short Final Summary
- What was checked:
  - Method ablation for `distance`, `fingerprint`, `variance`, `extremes`, `maxmin`, and `coarse_to_fine` on both FaultDetectionA and FruitFlies.
  - Multi-NUN ablation for `distance_n1_cycle`, `multi_nun_cycle_n{2,3,5}`, and `multi_nun_closest_n{2,3,5}`.
  - Direct notebook baseline comparisons of `emd_variance_nun3` vs Native Guide vs Wachter.
  - Qualitative validation via extracted notebook figures (line plots, projections, and walkthrough visuals).
- Main findings:
  - In the sampled ablation runs, EMD variants achieved 100% success on both datasets.
  - In baseline notebooks, `emd_variance_nun3` was the most reliable method for class-flip success and validity.
  - `multi_nun_cycle_n3` is a practical default when balancing proximity and performance.
- Practical takeaway:
  - Start with `emd_variance_nun3` for reliability; tune NUN switching (especially cycle with `n=3`) for proximity/efficiency trade-offs.
