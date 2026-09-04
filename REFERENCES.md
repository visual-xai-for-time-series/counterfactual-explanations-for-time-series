# References for Counterfactual Explanation Methods

This document provides comprehensive references for all counterfactual explanation algorithms implemented in this library, along with a quick usage example.

## Table of Contents

- [Quick Usage Example](#quick-usage-example)
- [Method References](#method-references)
  - [Optimization-Based Methods](#optimization-based-methods)
  - [Evolutionary Methods](#evolutionary-methods)
  - [Instance-Based Methods](#instance-based-methods)
  - [Latent Space Methods](#latent-space-methods)
  - [Segment-Based Methods](#segment-based-methods)
  - [Hybrid Methods](#hybrid-methods)
- [Evaluation Metrics References](#evaluation-metrics-references)
- [Related Surveys and Reviews](#related-surveys-and-reviews)

---

## Quick Usage Example

Every counterfactual-generating function in this library shares the same
leading argument order: `(sample, model, target_class=None, dataset=None,
...algorithm-specific parameters...)`. `target_class` and `dataset` are always
keyword-friendly (pass them by name); a handful of methods need a raw
`(X_train, y_train)` pair or a `reference_data` array instead of a `dataset`
object where noted in their own section below.

```python
import numpy as np
import torch
from cfts.cf_wachter.wachter import wachter_genetic_cf
from cfts.cf_native_guide.native_guide import native_guide_uni_cf
from cfts.cf_comte.comte import comte_cf
from cfts.cf_tsevo.tsevo import tsevo_cf
from cfts.metrics import l2_distance, prediction_change

# Load your model and data
# model = ... (trained PyTorch model)
# sample = ... (time series to explain)
# dataset = ... (dataset object)

# Every method in this library follows the same argument order:
#   (sample, model, target_class=None, dataset=None, ...algorithm-specific...)
cf_wachter, pred_wachter = wachter_genetic_cf(
    sample, model, target_class=1, step_size=0.1, max_steps=1000
)

cf_native, pred_native = native_guide_uni_cf(
    sample, model, target_class=1, dataset=dataset
)

cf_comte, pred_comte = comte_cf(
    sample, model, target_class=1, dataset=dataset
)

cf_tsevo, pred_tsevo = tsevo_cf(
    sample, model, target_class=1, dataset=dataset,
    population_size=50, generations=100
)

# Evaluate counterfactual quality
validity = prediction_change(model, sample, cf_wachter, target_class=1)
proximity = l2_distance(sample, cf_wachter)

print(f"Validity: {validity}, Proximity: {proximity}")
```

---

## Method References

### Optimization-Based Methods

#### 1. Wachter et al. (2017)
**Implementation:** `cfts/cf_wachter/wachter.py`

**Description:** Classic counterfactual explanation method using gradient-based optimization or genetic algorithms to find minimal perturbations that change the model's prediction.

**Key Features:**
- **Gradient-based optimization**: Uses model gradients for efficient counterfactual generation
- **Genetic algorithm variant**: Evolutionary approach for complex search spaces
- **Proximity-focused**: Minimizes distance to original while achieving target prediction

**Reference:**
```bibtex
@article{wachter2017counterfactual,
  title={Counterfactual explanations without opening the black box: Automated decisions and the GDPR},
  author={Wachter, Sandra and Mittelstadt, Brent and Russell, Chris},
  journal={Harvard Journal of Law \& Technology},
  volume={31},
  pages={841--887},
  year={2017}
}
```

**Links:**
- Paper: [Harvard Journal](https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf)
- Book Chapter: [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/counterfactual.html)

**Usage Example:**
```python
from cfts.cf_wachter.wachter import wachter_genetic_cf, wachter_gradient_cf

# Genetic algorithm variant
cf, prediction = wachter_genetic_cf(
    sample=sample,
    model=model,
    target_class=1,
    step_size=0.1,
    max_steps=1000
)

# Gradient-based variant (needs a dataset to seed the candidate, unless full_random=True)
cf, prediction = wachter_gradient_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    max_cfs=500
)
```

---

#### 2. COMTE - Counterfactual Explanations for Multivariate Time Series (2021)
**Implementation:** `cfts/cf_comte/comte.py`

**Description:** Specialized counterfactual method for multivariate time series that optimizes each channel independently when beneficial, incorporating smoothness and sparsity constraints.

**Key Features:**
- **Multivariate support**: Handles multi-channel time series effectively
- **Feature-wise optimization**: Optimizes each channel independently when beneficial
- **Regularization**: Incorporates smoothness and sparsity constraints

**Reference:**
```bibtex
@inproceedings{ates2021counterfactual,
  title={Counterfactual Explanations for Multivariate Time Series},
  author={Ates, Emre and Aksar, Burak and Leung, Vitus J and Coskun, Ayse K},
  booktitle={2021 International Conference on Applied Artificial Intelligence (ICAPAI)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```

**Links:**
- Repository: [https://github.com/peaclab/CoMTE](https://github.com/peaclab/CoMTE)

**Usage Example:**
```python
from cfts.cf_comte.comte import comte_cf, comte_cf_gradient

# Feature-wise (per-channel) distractor-swap search
cf, prediction = comte_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    n_segments=10,
)

# Gradient-based variant
cf, prediction = comte_cf_gradient(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    learning_rate=0.1,
    max_iterations=3000,
)
```

---

#### 3. TSCF - Time Series CounterFactuals (Custom)
**Implementation:** `cfts/cf_tscf/tscf.py`

**Description:** Gradient-based optimization with temporal smoothness constraints for generating realistic counterfactual explanations.

**Note:** This is a custom implementation combining standard counterfactual generation techniques with time series-specific regularization.

**Usage Example:**
```python
from cfts.cf_tscf import tscf_cf

cf, prediction = tscf_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    lambda_l1=0.01,
    lambda_l2=0.01,
    lambda_smooth=0.001,
    learning_rate=0.1,
    max_iterations=2000
)
```

---

#### 4. TS-Tweaking - Locally and Globally Explainable Time Series Tweaking (2020)
**Implementation:** `cfts/cf_ts_tweaking/ts_tweaking.py`

**Description:** Three complementary algorithms for generating counterfactual explanations by tweaking time series segments. The global method (τ_NN) uses k-NN with k-means clustering to find the minimal transformation that changes the prediction; the local irreversible method (τ_SF) uses a shapelet forest to push subsequences past decision thresholds; the local reversible method (τ_SF-R) constrains modifications within the threshold sphere for conservative changes.

**Key Features:**
- **Three variants**: Global (τ_NN), Local Irreversible (τ_SF), and Local Reversible (τ_SF-R)
- **k-NN with k-means clustering**: Finds optimal target instances via nearest-neighbor search
- **Shapelet-based approach**: Identifies discriminative subsequences driving the prediction
- **Reversible option**: Conservative modifications that remain within shapelet distance bounds
- **Interpretable changes**: Segment-level tweaks produce easily understandable explanations

**Reference:**
```bibtex
@article{karlsson2020locally,
  title={Locally and globally explainable time series tweaking},
  author={Karlsson, Isak and Rebane, Jonathan and Papapetrou, Panagiotis and Gionis, Aristides},
  journal={Knowledge and Information Systems},
  volume={62},
  pages={1671--1700},
  year={2020},
  publisher={Springer}
}
```

**Links:**
- Paper: [Springer KAIS](https://link.springer.com/article/10.1007/s10115-019-01389-4)
- ArXiv: [arXiv:1809.05183](https://arxiv.org/abs/1809.05183)

**Usage Example:**
```python
from cfts.cf_ts_tweaking.ts_tweaking import (
    ts_tweaking_knn_cf,
    ts_tweaking_irreversible_cf,
    ts_tweaking_reversible_cf,
)

# Global tweaking via k-NN with k-means
cf, prediction = ts_tweaking_knn_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    k=5,
    n_clusters=5,
    alpha_steps=20
)

# Local irreversible tweaking (shapelet-based)
cf, prediction = ts_tweaking_irreversible_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset
)

# Local reversible tweaking (shapelet-based, conservative)
cf, prediction = ts_tweaking_reversible_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset
)
```

---

#### FFT-CF - Fourier Transform Counterfactual Explanations
**Implementation:** `cfts/cf_fft_cf/fft_cf.py`

**Description:** Frequency-based counterfactual generation using Fast Fourier Transform (FFT) to decompose time series into frequency components, then iteratively modifying frequency coefficients (amplitude and/or phase) to find counterfactual explanations that change the model's prediction while maintaining temporal structure and realism.

**Key Features:**
- **Frequency domain manipulation**: Modifies amplitude and/or phase of frequency components
- **Temporal coherence**: Preserves overall temporal patterns through frequency domain operations
- **Selective bands**: Can focus on specific frequency bands (low/high/mid frequencies)
- **Dual strategies**: Greedy search variant and gradient-based optimization variant
- **Efficient for long series**: FFT complexity is O(n log n)

**Reference:**
```bibtex
@inproceedings{delaney2021instance,
  title={Instance-Based Counterfactual Explanations for Time Series Classification},
  author={Delaney, Eoin and Greene, Derek and Keane, Mark T},
  booktitle={International Conference on Case-Based Reasoning},
  pages={32--47},
  year={2021},
  organization={Springer},
  note={Discusses frequency domain manipulations for counterfactuals}
}
```

**Links:**
- Related Repository: [https://github.com/e-delaney/Instance-Based_CFE_TSC](https://github.com/e-delaney/Instance-Based_CFE_TSC)
- FFT Documentation: [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)

**Usage Example:**
```python
from cfts.cf_fft_cf import fft_cf, fft_gradient_cf

# Greedy search variant with amplitude modification
cf, prediction = fft_cf(
    sample=sample,
    model=model,
    target_class=1,
    frequency_bands="all",  # "all", "low", "high", "mid"
    modification_strategy="amplitude",  # "amplitude", "phase", "both"
    step_size=0.05,
    lambda_proximity=0.1,
    max_iterations=1000
)

# Gradient-based optimization variant
cf, prediction = fft_gradient_cf(
    sample=sample,
    model=model,
    target_class=1,
    learning_rate=0.01,
    lambda_proximity=0.1,
    lambda_smoothness=0.05,
    max_iterations=500
)
```

---

#### TopGrad-CF - Gradient-Guided Counterfactual Explanations for Time Series Classification (2026)
**Implementation:** `cfts/cf_topgrad/topgrad_cf.py`

**Description:** Starts from a Nearest-Unlike-Neighbour prototype (as in Native Guide) and optimises toward it with Adam, but at every step keeps only the top `top_k_frac` fraction of the loss gradient by magnitude and zeroes out the rest before the optimiser step — the "TopGrad" masking that restricts each update to the handful of time steps the model is currently most sensitive to. Runs in two phases: a coarse sweep across orders of magnitude of the proximity weight to bracket a workable value, then a refinement stage that additionally restricts updates to a single contiguous "prominent segment" (the window with the largest `|sample - prototype|` gap) and grows that window until the counterfactual is confidently valid. Ported from the authors' TensorFlow-1/Alibi reference implementation to plain PyTorch, with documented, deliberate deviations (see the module docstring) confirmed by cloning and actually running the reference end-to-end in `cfts/cf_topgrad/topgrad_coffee_comparison.ipynb` — most notably, the reference computes but never backpropagates its own classifier loss, leaving prototype proximity as the only force pulling the candidate toward the target class; this port adds it back in.

**Key Features:**
- **Prototype-guided**: seeds the search from a Nearest-Unlike-Neighbour, exactly as in Native Guide
- **TopGrad masking**: each optimiser step only updates the top-k% highest-magnitude gradient positions, leaving the rest of the series untouched
- **Two-phase search**: an exponential lambda-bracketing sweep followed by a growing-segment refinement stage that progressively widens the region allowed to change
- **Gradient-guided objective**: classifier loss, L1/L2 proximity to both the original sample and the prototype, and a smoothness penalty, all optimised jointly with Adam
- **Reproduction-verified**: deviations from the authors' TensorFlow-1 reference are documented and were confirmed by running that reference code directly, not just by reading it

**Reference:**
```bibtex
@inproceedings{hosseinzadeh2026topgrad,
  title={TopGrad-CF: Gradient-Guided Counterfactual Explanations for Time Series Classification},
  author={Hosseinzadeh, Pouya and others},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  year={2026},
  publisher={Springer},
  doi={10.1007/978-3-032-31933-3_35}
}
```

**Links:**
- Paper: [DOI:10.1007/978-3-032-31933-3_35](https://link.springer.com/chapter/10.1007/978-3-032-31933-3_35)
- Repository: [https://github.com/pouyahosseinzadeh/TopGrad-CF](https://github.com/pouyahosseinzadeh/TopGrad-CF)

**Usage Example:**
```python
from cfts.cf_topgrad.topgrad_cf import topgrad_cf

cf, prediction = topgrad_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,       # required — used for the NUN prototype search
    top_k_frac=0.03,       # fraction of gradient positions updated per step
    max_iter=200,
    seed=42,
)
```

---

#### Soft-DTW-CFE - Towards Plausibility in Time Series Counterfactual Explanations (2026)
**Implementation:** `cfts/cf_soft_dtw_cfe/soft_dtw_cfe.py`

**Description:** Optimises a counterfactual directly in input space with Adam, under a four-term loss balancing validity, proximity, sparsity, and a differentiable Soft-DTW plausibility term that pulls the candidate toward its `k` nearest target-class training series — encouraging a realistic temporal *shape* (via DTW's elastic alignment) rather than only matching values timestep-by-timestep, as an L1/L2 proximity term alone would. Soft-DTW replaces the path-selecting `min` in the DTW recursion with a soft-min, `-gamma * log(sum(exp(-r / gamma)))`, making the whole alignment cost differentiable and usable as a genuine gradient-descent loss term rather than only a post-hoc distance metric.

**Key Features:**
- **Soft-DTW as a loss, not just a metric**: a custom autograd `Function` implementing the Soft-DTW forward/backward recursions (ported from `github.com/Sleepwalking/pytorch-softdtw`, as vendored by the paper's own repository), so its gradient participates directly in every Adam step alongside the other three loss terms
- **Target-class neighbor bank**: builds a per-class bank from the training set and selects the `k_neighbors` series closest to the query *from the target class* by Soft-DTW distance, fixed for the whole optimisation run
- **Four-term objective**: `L_CF = L_prox + L_sparse + lambda * (L_valid + L_DTW)`, with a hinge (`max(0, tau - p(target_class|x_cf))`) or cross-entropy validity term
- **Reproduction-verified**: ported line-by-line from the authors' own `dtw.py` / `soft_dtw_loss.py` / `solver.py`, and confirmed to reproduce the official `CounterfactualSolver` bit-for-bit (`max|cf_official - cf_ours| = 0.0`) given an identical classifier, training pool, and hyperparameters — see the comparison notebook below

**Reference:**
```bibtex
@inproceedings{kostrzewa2026softdtwcfe,
  title={Towards Plausibility in Time Series Counterfactual Explanations},
  author={Kostrzewa, Marcin and Galus, Krzysztof and Zi\k{e}ba, Maciej},
  booktitle={Asian Conference on Intelligent Information and Database Systems (ACIIDS)},
  year={2026},
  eprint={2603.08349},
  archivePrefix={arXiv}
}
```

**Links:**
- Paper: [arXiv:2603.08349](https://arxiv.org/abs/2603.08349)
- Repository: [https://github.com/genwro-ai/soft-dtw-counterfactual-explanations](https://github.com/genwro-ai/soft-dtw-counterfactual-explanations)
- Docs: [https://genwro-ai.github.io/soft-dtw-counterfactual-explanations/](https://genwro-ai.github.io/soft-dtw-counterfactual-explanations/)
- Comparison notebook: `cfts/cf_soft_dtw_cfe/soft_dtw_cfe_forda_comparison.ipynb` — runs the authors' own `CounterfactualSolver` (cloned from their repository) against the exact same classifier, training pool, and hyperparameters as this port. Because the optimisation is fully deterministic (no sampling or genetic operators, unlike DiffCF's diffusion sampling or CONFETTI's genetic search elsewhere in this library), the two implementations' outputs match to floating-point precision on every sample tested.

**Usage Example:**
```python
from cfts.cf_soft_dtw_cfe.soft_dtw_cfe import soft_dtw_cfe_cf

cf, prediction = soft_dtw_cfe_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,     # required — builds the target-class Soft-DTW neighbor bank
    steps=500,
    k_neighbors=5,
    lambda_validity=10.0,
)
```

---

### Evolutionary Methods

#### 5. MOC/DANDL - Multi-Objective Counterfactuals (2020)
**Implementation:** `cfts/cf_dandl/dandl.py`

**Description:** Multi-objective evolutionary approach using genetic algorithms to find Pareto-optimal counterfactuals that balance validity, proximity, and sparsity.

**Key Features:**
- **Pareto optimization**: Balances multiple objectives (validity, proximity, sparsity)
- **Evolutionary algorithm**: Uses genetic operations for diverse solutions
- **Multiple solutions**: Returns a set of counterfactuals on the Pareto frontier

**Reference:**
```bibtex
@article{dandl2020multi,
  title={Multi-objective counterfactual explanations},
  author={Dandl, Susanne and Molnar, Christoph and Binder, Martin and Bischl, Bernd},
  journal={arXiv preprint arXiv:2004.11165},
  year={2020}
}
```

**Links:**
- Paper: [arXiv:2004.11165](https://arxiv.org/abs/2004.11165)
- Repository: [https://github.com/susanne-207/moc](https://github.com/susanne-207/moc)

**Usage Example:**
```python
from cfts.cf_dandl.dandl import moc_cf, moc_cf_diverse

# Single counterfactual
cf, prediction = moc_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    population_size=100,
    generations=200,
    mutation_rate=0.1
)

# Multiple diverse Pareto-optimal counterfactuals
cfs, preds, metrics = moc_cf_diverse(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    n_counterfactuals=5
)
```

---

#### 6. TSEvo - Time Series Evolutionary Counterfactuals (2022)
**Implementation:** `cfts/cf_tsevo/tsevo.py`

**Description:** Evolutionary counterfactual explanations using NSGA-II multi-objective optimization with reference set mutation, crossover, Gaussian mutation, and segment-based swapping.

**Key Features:**
- **NSGA-II algorithm**: Industry-standard multi-objective evolutionary optimizer
- **Pareto optimization**: Simultaneously optimizes validity, proximity, and sparsity
- **Reference set mutation**: Leverages target class examples for realistic counterfactuals
- **Multiple operators**: Crossover, Gaussian mutation, and segment-based swapping

**Reference:**
```bibtex
@inproceedings{hollig2022tsevo,
  title={TSEvo: Evolutionary counterfactual explanations for time series classification},
  author={H{\"o}llig, Jacqueline and Kulbach, Cedric and Thoma, Steffen},
  booktitle={2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)},
  pages={29--36},
  year={2022},
  organization={IEEE}
}
```

**Links:**
- Repository: [https://github.com/fzi-forschungszentrum-informatik/TSInterpret](https://github.com/fzi-forschungszentrum-informatik/TSInterpret)

**Usage Example:**
```python
from cfts.cf_tsevo import tsevo_cf

cf, prediction = tsevo_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    population_size=50,
    generations=100,
    crossover_rate=0.7,
    mutation_rate=0.3
)
```

---

#### 7. Multi-SpaCE - Multi-Objective Subsequence-based Sparse Counterfactuals (2024)
**Implementation:** `cfts/cf_multispace/multispace.py`

**Description:** Multi-objective counterfactual generation using feature importance for guided initialization, subsequence optimization, and evolutionary search for diverse solutions.

**Key Features:**
- **Feature importance**: Uses attribution methods for guided initialization
- **Subsequence optimization**: Modifies meaningful temporal segments
- **Multi-objective fitness**: Balances validity, sparsity, and plausibility
- **Evolutionary approach**: Population-based search for diverse solutions

**Reference:**
```bibtex
@article{refoyo2024multi,
  title={Multi-SpaCE: Multi-Objective Subsequence-based Sparse Counterfactual Explanations for Multivariate Time Series Classification},
  author={Refoyo, Mario and Luengo, David},
  journal={arXiv preprint arXiv:2501.04009},
  year={2024}
}
```

**Links:**
- Repository: [https://github.com/MarioRefoyo/Multi-SpaCE](https://github.com/MarioRefoyo/Multi-SpaCE)

**Usage Example:**
```python
from cfts.cf_multispace.multispace import multispace_cf

cf, prediction = multispace_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    population_size=100,
    grouped_iter=75,
    pruning_iter=25
)
```

---

#### 8. Sub-SpaCE - Subsequence-based Sparse Counterfactuals (2023)
**Implementation:** `cfts/cf_subspace/subspace.py`

**Description:** Evolutionary algorithm with subsequence-based representations to generate sparse and interpretable counterfactuals for time series classification.

**Reference:**
```bibtex
@article{refoyo2023subspece,
  title={Sub-SpaCE: Subsequence-based Sparse Counterfactual Explanations for Time Series Classification},
  author={Refoyo, Mario and Luengo, David},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```

**Links:**
- Repository: [https://github.com/MarioRefoyo/Sub-SpaCE](https://github.com/MarioRefoyo/Sub-SpaCE)

**Usage Example:**
```python
from cfts.cf_subspace.subspace import subspace_cf

cf, prediction = subspace_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    population_size=100,
    max_iter=100
)
```

---

#### 9. CONFETTI - COuNterfactual Explanations For Time Series (2026)
**Implementation:** `cfts/cf_confetti/confetti.py` (+ `cfts/cf_confetti/confetti_bridge.py` to run the official package)

**Description:** Combines Nearest Unlike Neighbor (NUN) search with subsequence replacement and multi-objective evolutionary optimization to produce sparse, realistic, confidence-increasing counterfactuals for multivariate time series. This repository ships two from-scratch implementations — `confetti_genetic_cf` (lightweight single-objective genetic search) and `confetti_nsga_cf` (closer to the official mechanism: confidence-gated NUN search, a contiguous replacement window found via binary search, and genuine multi-objective NSGA-II optimization) — plus `confetti_package_cf` / `confetti_bridge.py`, which drive the official Rust-accelerated `confetti-ts` package (NSGA-III) out-of-process, since it requires Python >= 3.12 and can't share an interpreter with the rest of this repository.

**Key Features:**
- **Confidence-gated NUN search**: candidates must clear a minimum confidence threshold `theta` in their own predicted class
- **Contiguous window search**: binary search over the replacement window size before optimizing within it
- **Multi-objective optimization**: confidence, sparsity, and proximity objectives (NSGA-III in the official package; NSGA-II reimplemented here in pure NumPy)
- **Model-agnostic**: works with PyTorch, Keras, or scikit-learn classifiers
- **Official-package bridge**: `confetti_bridge.py` runs `confetti-ts` in a separate Python >= 3.12 subprocess/venv and exchanges data via `.npy` files, so results can be directly compared against this repo's reimplementations

**Reference:**
```bibtex
@inproceedings{cetina2026counterfactual,
  title={Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification},
  author={Cetina, Alan Gabriel Paredes and Benguessoum, Kaouther and Lourenco, Raoni and Kubler, Sylvain},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={21},
  pages={17393--17400},
  year={2026}
}
```

**Links:**
- Paper (arXiv preprint): [arXiv:2511.13237](https://arxiv.org/html/2511.13237v2)
- Repository: [https://github.com/serval-uni-lu/confetti](https://github.com/serval-uni-lu/confetti)
- PyPI package: [confetti-ts](https://pypi.org/project/confetti-ts/)

**Usage Example:**
```python
from cfts.cf_confetti.confetti import confetti_genetic_cf, confetti_nsga_cf

# Lightweight genetic-search reimplementation
cf, prediction = confetti_genetic_cf(
    sample=sample,
    model=model,
    reference_data=reference_data,
    theta=0.51,
    population_size=50,
    max_iterations=100,
)

# Closer NSGA-II reimplementation of the official mechanism
cf, prediction = confetti_nsga_cf(
    sample=sample,
    model=model,
    dataset=dataset,
    target_class=1,
    theta=0.51,
    alpha=0.5,
    population_size=60,
    max_generations=40,
)
```

---

#### 10. FastPACE - Fast PlAnning of Counterfactual Explanations for Time Series Classification (2026)
**Implementation:** `cfts/cf_fastpace/fastpace.py`

**Description:** Casts counterfactual generation as an episodic Markov Decision Process (MDP) over NUN-replacement masks and solves it with hierarchical, block-based Cross-Entropy Method (CEM) planning (Model Predictive Control). Every trajectory starts at the Nearest Unlike Neighbor — already valid by construction — and is progressively refined back toward the query, guaranteeing validity by design.

**Key Features:**
- **MDP formulation**: state is a binary replacement mask between query and NUN; actions flip mask entries
- **Model Predictive Control**: plans a finite-horizon action sequence with the Cross-Entropy Method at every step, executes only the first action, then replans
- **Coarse-to-fine block granularity**: actions operate on contiguous time-step blocks combined with clusters of similarly-behaving channels, refined across granularity levels
- **Guaranteed validity**: since trajectories start at the valid NUN, the last valid mask found along the way is always returned
- **Plausibility term**: Increase-in-Outlier-Score (IOS) from a dedicated reconstruction autoencoder discourages out-of-distribution counterfactuals
- **Shared objective family**: reuses Sub-SpaCE/Multi-SpaCE's weighted combination of adversarial, sparsity, contiguity, and plausibility terms

**Reference:**
```bibtex
@article{refoyo2026fastpace,
  title={FastPACE: Fast PlAnning of Counterfactual Explanations for Time Series Classification},
  author={Refoyo, Mario and Boleas, Yago and Luengo, David},
  journal={Data Mining and Knowledge Discovery},
  year={2026},
  publisher={Springer},
  doi={10.1007/s10618-026-01242-7}
}
```

**Links:**
- Paper: [DOI:10.1007/s10618-026-01242-7](https://doi.org/10.1007/s10618-026-01242-7)
- Preprint: [Research Square](https://doi.org/10.21203/rs.3.rs-8611408/v1)
- Repository: [https://github.com/MarioRefoyo/FastPACE](https://github.com/MarioRefoyo/FastPACE)

**Usage Example:**
```python
from cfts.cf_fastpace import fastpace_cf, train_plausibility_autoencoder

cf, prediction = fastpace_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    horizon=3,
    cem_iterations=3,
    elite_fraction=0.1,
    alpha=0.1,   # adversarial weight
    beta=0.3,    # sparsity weight
    eta=0.4,     # contiguity weight
    lam=0.2,     # plausibility weight
    max_reference_samples=300,
    verbose=True,
)

# Reuse a pre-trained plausibility autoencoder across many samples instead of
# retraining it inside every fastpace_cf call
autoencoder, ae_max_error = train_plausibility_autoencoder(train_ts, device, epochs=20)
cf, prediction = fastpace_cf(
    sample=sample, model=model, target_class=1, dataset=dataset,
    autoencoder=autoencoder, ae_max_error=ae_max_error,
)
```

---

### Instance-Based Methods

#### 11. Native Guide (2021)
**Implementation:** `cfts/cf_native_guide/native_guide.py`

**Description:** Instance-based counterfactual generation using nearest neighbor search and gradient attribution (GradientShap) to preserve important temporal patterns.

**Key Features:**
- **Instance-based approach**: Leverages similar examples from training data
- **Gradient attribution**: Uses Captum's GradientShap for feature importance
- **Temporal awareness**: Preserves important temporal patterns

**Reference:**
```bibtex
@inproceedings{delaney2021instance,
  title={Instance-based counterfactual explanations for time series classification},
  author={Delaney, Eoin and Greene, Derek and Keane, Mark T},
  booktitle={International Conference on Case-Based Reasoning},
  pages={32--47},
  year={2021},
  organization={Springer}
}
```

**Links:**
- Repository: [https://github.com/e-delaney/Instance-Based_CFE_TSC](https://github.com/e-delaney/Instance-Based_CFE_TSC)

**Usage Example:**
```python
from cfts.cf_native_guide.native_guide import native_guide_uni_cf

# Works for both univariate and multivariate time series -- shape is
# inferred from `sample` (accepts 1-D, (C, L), or (L, C))
cf, prediction = native_guide_uni_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset
)
```

---

#### 12. CELS & M-CELS - Counterfactual Explanations via Learned Saliency (2023-2024)
**Implementation:** `cfts/cf_cels/cels.py`

**Description:** Learns saliency maps to identify important time steps and generates counterfactuals through nearest unlike neighbor replacement. Supports both univariate (CELS) and multivariate (M-CELS) time series with automatic selection via `cels_auto`.

**Key Features:**
- **Learned saliency maps**: Identifies important time steps contributing to predictions
- **Nearest unlike neighbor (NUN)**: Finds target class instances for replacement
- **Optimization-based learning**: Balances validity, sparsity, and temporal coherence
- **`cels_auto` wrapper**: Automatically selects between CELS/M-CELS based on dimensionality
- **High sparsity**: Modifies only salient time steps for minimal perturbations
- **Temporal regularization**: Ensures smooth, contiguous explanations

**CELS Algorithm (Univariate):**
1. **Nearest Unlike Neighbor**: Find nearest training instance of target class z'
2. **Initialize Saliency**: Random uniform θ ∈ [0,1]^T
3. **Loss Function**: L = λ·L_Max + L_Budget + L_TReg where:
   - L_Max = 1 - P(y'^z' | x'): Maximize target class probability
   - L_Budget = (1/T)·Σ θ_t: Minimize saliency values for sparsity
   - L_TReg = (1/T)·Σ (θ_t - θ_{t+1})²: Temporal coherence
4. **Optimize**: Learn saliency θ via gradient descent with early stopping
5. **Generate CF**: x' = x ⊙ (1-θ) + nun ⊙ θ (element-wise replacement)

**M-CELS Algorithm (Multivariate):**
1. **Nearest Unlike Neighbor**: Find NUN from target class across all dimensions
2. **Initialize Saliency**: Random θ ∈ [0,1]^{D×T}
3. **Loss Function**: L = λ·L_MMax + L_MBudget + L_MTReg where:
   - L_MMax = 1 - P(y'^z' | x'): Validity loss
   - L_MBudget = (1/D)·Σ_d [(1/T)·Σ_t θ_{t,d}]: Average sparsity across dimensions
   - L_MTReg = (1/D)·Σ_d [(1/T)·Σ_t (θ_{t,d} - θ_{t+1,d})²]: Temporal smoothness per dimension
4. **Optimize**: Learn multi-dimensional saliency via Adam optimizer
5. **Generate CF**: x' = x ⊙ (1-θ) + nun ⊙ θ
6. **Validate**: Check if CF is within distribution using Isolation Forest

**References:**
```bibtex
@inproceedings{li2023cels,
  title={CELS: Counterfactual Explanations for Time Series Data via Learned Saliency Maps},
  author={Li, Peiyu and Bahri, Omar and Filali, Soukaina and Hamdi, Shah Muhammad},
  booktitle={2023 IEEE International Conference on Big Data (BigData)},
  pages={718--727},
  year={2023},
  organization={IEEE}
}

@article{li2024mcels,
  title={M-CELS: Counterfactual Explanation for Multivariate Time Series Data Guided by Learned Saliency Maps},
  author={Li, Peiyu and Bahri, Omar and Boubrahimi, Soukaina Filali and Hamdi, Shah Muhammad},
  journal={arXiv preprint arXiv:2411.02649},
  year={2024}
}
```

**Links:**
- CELS Paper: [IEEE BigData 2023](https://ieeexplore.ieee.org/document/10386229)
- M-CELS Paper: [arXiv:2411.02649](https://arxiv.org/abs/2411.02649)
- M-CELS HTML: [arXiv HTML](https://arxiv.org/html/2411.02649v1)
- Repository: [https://github.com/Healthpy/cfe_tsc_pos](https://github.com/Healthpy/cfe_tsc_pos)

**Usage Example:**
```python
from cfts.cf_cels.cels import cels_generate, m_cels_generate, cels_auto

# Univariate CELS
cf, prediction = cels_generate(
    sample=sample,
    model=model,
    X_train=X_train,
    y_train=y_train,
    target_class=1,
    learning_rate=0.01,
    max_iter=100,
    lambda_valid=1.0,
    lambda_budget=0.1,
    lambda_tv=0.1
)

# Multivariate M-CELS
cf, prediction = m_cels_generate(
    sample=sample,
    model=model,
    X_train=X_train,
    y_train=y_train,
    target_class=1,
    learning_rate=0.01,
    max_iter=100,
    lambda_valid=1.0,
    lambda_sparsity=0.1,
    lambda_smoothness=0.1
)

# cels_auto: automatically picks CELS vs M-CELS based on input dimensionality
cf, prediction = cels_auto(
    sample=sample,
    model=model,
    X_train=X_train,
    y_train=y_train,
    target_class=1
)
```

---

#### 13. AB-CF - Attention-Based Counterfactual Explanation (2023)
**Implementation:** `cfts/cf_ab_cf/ab_cf.py`

**Description:** Uses Shannon entropy-based attention mechanism to identify and replace high-uncertainty subsequences with segments from nearest unlike neighbors (NUN), creating sparse and interpretable counterfactual explanations for multivariate time series classification.

**Key Features:**
- **Shannon entropy attention**: Measures uncertainty of model predictions on subsequences to identify critical segments
- **Sliding window segmentation**: Divides time series into overlapping windows with configurable size and stride
- **Nearest unlike neighbor (NUN)**: Retrieves similar instances from target class using distance-based search (KNN)
- **Selective segment replacement**: Replaces only high-entropy segments, maintaining sparsity and interpretability
- **Multivariate support**: Handles multi-channel time series effectively
- **Early stopping**: Validates counterfactual after each segment replacement for efficiency

**Algorithm:**

1. **Compute original prediction and target class**: Get model probabilities for input time series, select target class (second most likely if not specified)
2. **Sliding window segmentation**: Extract subsequences using sliding window with configurable size (default 10% of time series length) and stride
3. **Entropy calculation**: For each subsequence, compute Shannon entropy H(p) = -Σ p_i log(p_i) of model prediction probabilities
4. **Segment ranking**: Sort subsequences by entropy in descending order, select top-k high-uncertainty segments (default k=10)
5. **NUN retrieval**: Find nearest unlike neighbor from target class using K-nearest neighbors with distance metric (Euclidean/DTW)
6. **Sequential replacement**: For each high-entropy segment (in order of decreasing entropy):
   - Replace segment in original time series with corresponding segment from NUN
   - Compute counterfactual prediction
   - If prediction matches target class, return valid counterfactual
   - Otherwise, continue to next segment
7. **Return result**: Return counterfactual if valid, None otherwise

**Reference:**
```bibtex
@inproceedings{li2023attention,
  title={Attention-Based Counterfactual Explanation for Multivariate Time Series},
  author={Li, Peiyu and Bahri, Omar and Boubrahimi, Souka{\"\i}na Filali and Hamdi, Shah Muhammad},
  booktitle={International Conference on Big Data Analytics and Knowledge Discovery},
  pages={287--293},
  year={2023},
  organization={Springer}
}
```

**Links:**
- Paper: [Springer DaWaK 2023](https://link.springer.com/chapter/10.1007/978-3-031-39831-5_26)
- Repository (Original): [https://github.com/Luckilyeee/AB-CF](https://github.com/Luckilyeee/AB-CF)
- Repository (Reference): [https://github.com/Healthpy/cfe_tsc_pos](https://github.com/Healthpy/cfe_tsc_pos)

**Usage Example:**
```python
from cfts.cf_ab_cf.ab_cf import ab_cf_generate

# Generate AB-CF counterfactual
cf, cf_label = ab_cf_generate(
    sample=sample,
    model=model,
    X_train=X_train,
    y_train=y_train,
    target_class=1,
    n_segments=10,  # number of top-entropy segments to replace
    window_size_ratio=0.1,  # window size as ratio of time series length
    verbose=True
)

# Automatic target selection (second most likely class)
cf, cf_label = ab_cf_generate(
    sample=sample,
    model=model,
    X_train=X_train,
    y_train=y_train,
    n_segments=15,  # try more segments for difficult cases
    window_size_ratio=0.05  # smaller windows for finer granularity
)
```

---

#### 14. IMFACT - Counterfactual Explanations for Time Series via Intrinsic Mode Function Substitution (2026)
**Implementation:** `cfts/cf_imfact/imfact.py`

**Description:** Counterfactual method that decomposes both the query and a native-guide (NUN) reference into Intrinsic Mode Functions (IMFs) via Empirical Mode Decomposition, then iteratively interpolates the query toward the guide in IMF/frequency space rather than raw amplitude space. This avoids the temporal-structure damage that perturbing raw feature space can cause. Several IMF-selection strategies control which modes are nudged first (by PSD distance, class-level variance, extremes, or a coarse-to-fine unlocking schedule), and multiple native guides can be cycled through during the search.

**Key Features:**
- **EMD/sifting decomposition**: extracts Intrinsic Mode Functions per channel with Rilling (2003) boundary padding and B-spline envelopes
- **Four IMF-weighting strategies**: `distance` (Jensen-Shannon divergence between interpolated/target IMF power spectra), `variance` (class-level PSD variance), `extremes` (only the most/least distant IMFs), `coarse_to_fine` (progressively unlocks IMFs)
- **Multi-guide support**: `n_nuns > 1` cycles through several native guides (`cycle` or `closest_psd` switching)
- **Frequency-domain interpolation**: modifies IMF composition rather than raw amplitudes, aiming to preserve temporal/spectral structure
- **Trace utility**: `trace_imfact_variant_path` records the per-iteration interpolation path for a single variant

**Reference:**
```bibtex
@inproceedings{schlegel2026imfact,
  title={IMFACT: Counterfactual Explanations for Time Series via Intrinsic Mode Function Substitution},
  author={Schlegel, Udo and Rakuschek, Julian and Seidl, Thomas and Holzinger, Andreas and Schreck, Tobias and Del Ser, Javier},
  booktitle={XKDD Workshop, ECML-PKDD 2026},
  year={2026}
}
```

**Links:**
- Paper: [arXiv:2608.04777](https://arxiv.org/abs/2608.04777)
- Venue: XKDD Workshop at ECML-PKDD 2026

**Usage Example:**
```python
from cfts.cf_imfact.imfact import imfact_cf

cf, prediction = imfact_cf(
    sample=sample,
    model=model,
    dataset=dataset,
    target_class=1,
    method="distance",       # "distance", "variance", "extremes", "coarse_to_fine"
    step=0.05,
    max_iter=200,
    n_nuns=3,
    nun_switch="cycle",
    verbose=True,
)
```

---

### Latent Space Methods

#### 15. GLACIER - Guided Locally Constrained Counterfactuals (2024)
**Implementation:** `cfts/cf_glacier/glacier.py`

**Description:** Advanced counterfactual generation with enhanced realism constraints, similarity preservation, and robust optimization for complex time series patterns using latent space representations.

**Key Features:**
- **Latent space optimization**: Searches in compressed latent representation for efficient counterfactual generation
- **Realism focus**: Incorporates domain-specific constraints
- **Similarity preservation**: Maintains statistical properties of original data
- **Robust optimization**: Handles noisy and complex time series patterns

**Reference:**
```bibtex
@article{wang2024glacier,
  title={Glacier: Guided locally constrained counterfactual explanations for time series classification},
  author={Wang, Zhendong and Samsten, Isak and Miliou, Ioanna and Mochaourab, Rami and Papapetrou, Panagiotis},
  journal={Machine Learning},
  year={2024},
  publisher={Springer}
}
```

**Links:**
- Repository: [https://github.com/zhendong3wang/learning-time-series-counterfactuals](https://github.com/zhendong3wang/learning-time-series-counterfactuals)

**Usage Example:**
```python
from cfts.cf_glacier.glacier import glacier_cf

cf, prediction = glacier_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    lambda_sparse=0.1,
    lambda_proximity=1.0,
    max_iterations=2000
)
```

---

#### 16. CGM - Conditional Generative Models for Counterfactuals (2021)
**Implementation:** `cfts/cf_cgm/cgm.py`

**Description:** Uses conditional generative models (e.g., conditional VAE/GAN) to generate sparse, in-distribution counterfactual explanations. The approach generates counterfactuals by conditioning a generative model on the desired target prediction, allowing batches of counterfactuals to be generated with a single forward pass.

**Key Features:**
- **Conditional generation**: Conditions on target class for direct generation
- **In-distribution guarantee**: Generates counterfactuals within learned data manifold
- **Batch generation**: Efficient generation of multiple counterfactuals
- **Latent space optimization**: Searches in compressed latent representation
- **Sparsity regularization**: Maintains minimal perturbations
- **VAE/GAN architecture**: Supports multiple conditional generative architectures

**Algorithm:**
1. Train conditional VAE/GAN on training dataset
2. For counterfactual generation:
   - Encode input x into latent space z ~ q(z|x, y_orig)
   - Optimize z to maximize p(y_target|x') while minimizing ||z - z_orig||
   - Decode optimized latent z' conditioned on target class: x' ~ p(x|z', y_target)
3. Return counterfactual x' that is in-distribution and achieves target prediction

**Reference:**
```bibtex
@article{vanlooveren2021conditional,
  title={Conditional Generative Models for Counterfactual Explanations},
  author={Van Looveren, Arnaud and Klaise, Janis and Vacanti, Giovanni and Cobb, Oliver},
  journal={arXiv preprint arXiv:2101.10123},
  year={2021}
}
```

**Links:**
- Paper: [arXiv:2101.10123](https://arxiv.org/abs/2101.10123)
- Repository: [https://github.com/SeldonIO/alibi](https://github.com/SeldonIO/alibi)

**Usage Example:**
```python
from cfts.cf_cgm.cgm import cgm_generate, ConditionalVAE, train_conditional_vae

# Simplest path: cgm_generate trains the conditional VAE on-the-fly from `dataset`
cf, prediction = cgm_generate(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    latent_dim=16,
    lr=0.01,
    max_iter=500,
    lambda_proximity=0.5,
    lambda_sparsity=0.01
)

# Or pre-train once and reuse the VAE across many samples
cvae = ConditionalVAE(input_dim=input_dim, num_classes=num_classes, latent_dim=16)
cvae = train_conditional_vae(cvae, dataset, num_classes, num_epochs=50)
cf, prediction = cgm_generate(
    sample=sample, model=model, target_class=1,
    cvae=cvae, train_vae=False
)
```

---

#### 17. CounTS - Counterfactual Time Series (2023)
**Implementation:** `cfts/cf_counts/counts.py`

**Description:** Self-interpretable time series prediction model with counterfactual explanations. Unlike post-hoc methods, CounTS is built on a structural causal model (SCM) that performs counterfactual reasoning through abduction, action, and prediction steps for causally plausible explanations.

**Key Features:**
- **Self-interpretable**: Built-in interpretability via structural causal model
- **Causal reasoning**: Three-step counterfactual inference (abduction-action-prediction)
- **Variational Bayesian**: Uses VAE framework for latent factor estimation
- **Actionable interventions**: Supports do-interventions on time series or latent factors
- **Plausibility**: Generates causally plausible counterfactual outcomes
- **LSTM encoder-decoder**: Handles temporal dependencies effectively

**Algorithm (Pearl's Three-Step Counterfactual Framework):**
1. **Abduction**: Estimate posterior distribution of latent factors given observation
   - q_φ(z | x, y) via LSTM encoder
2. **Action**: Apply do-intervention to time series or underlying factors
   - Modify specific time steps or latent dimensions
3. **Prediction**: Generate counterfactual outcome based on modified factors
   - p_θ(x' | z') via LSTM decoder
   - Predict outcome: y' = f(x')

**Reference:**
```bibtex
@article{gat2023counts,
  title={Self-Interpretable Time Series Prediction with Counterfactual Explanations},
  author={Gat, Itai and Malkiel, Idan and Schwartz, Idan and Wolf, Lior},
  journal={arXiv preprint arXiv:2306.06024},
  year={2023}
}
```

**Links:**
- Paper: [arXiv:2306.06024](https://arxiv.org/abs/2306.06024)
- HTML: [arXiv HTML](https://arxiv.org/html/2306.06024v1)

**Usage Example:**
```python
from cfts.cf_counts.counts import counts_cf_with_pretrained_model, CounTSModel, train_counts_model, counts_generate_counterfactual

# Simplest path: trains a CounTS model on-the-fly from `dataset`
cf, prediction = counts_cf_with_pretrained_model(
    sample=sample,
    model=model,       # the classifier being explained (used for the original prediction)
    target_class=1,
    dataset=dataset,
    latent_dim=16,
    hidden_dim=64,
    train_epochs=50
)

# Or pre-train once and reuse the CounTS model across many samples
counts_model = CounTSModel(
    input_dim=input_dim, hidden_dim=64, latent_dim=16,
    num_classes=num_classes, seq_len=seq_len
)
counts_model = train_counts_model(counts_model, dataset, num_epochs=50)
cf, prediction = counts_generate_counterfactual(
    sample=sample, counts_model=counts_model, target_class=1
)
```

---

#### 18. Latent-CF - Latent Space Counterfactuals (2020)
**Implementation:** `cfts/cf_latent_cf/latent_cf.py`

**Description:** Simple autoencoder-based approach that projects time series into latent space, optimizes in latent space, then projects back to original space for improved efficiency and interpretability. This method uses gradient descent in the latent space of an autoencoder to generate counterfactuals that are more in-distribution, sparse, and computationally efficient.

**Key Features:**
- **Latent space optimization**: Searches in the learned latent representation for more realistic counterfactuals
- **In-distribution guarantee**: Constrains search to the learned data manifold
- **Computational efficiency**: Faster than complex feature-space methods while maintaining quality
- **Sparsity**: Produces sparse changes by operating in compressed latent space

**Reference:**
```bibtex
@article{balasubramanian2020latent,
  title={Latent-CF: A Simple Baseline for Reverse Counterfactual Explanations},
  author={Balasubramanian, Rachana and Sharpe, Samuel and Barr, Brian and Wittenbach, Jason and Bruss, C Bayan},
  journal={arXiv preprint arXiv:2012.09301},
  year={2020}
}
```

**Links:**
- Paper: [https://arxiv.org/abs/2012.09301](https://arxiv.org/abs/2012.09301)
- HTML: [https://ar5iv.labs.arxiv.org/html/2012.09301](https://ar5iv.labs.arxiv.org/html/2012.09301)

**Usage Example:**
```python
from cfts.cf_latent_cf.latent_cf import latent_cf_generate

cf, prediction = latent_cf_generate(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,   # required unless a pre-trained `autoencoder` is passed
    latent_dim=8,
    lr=0.01,
    max_iter=1000,
    autoencoder=None  # or provide a pre-trained autoencoder to skip training
)
```

---

#### 19. LASTS - Local Agnostic Subsequence-based Time Series Explainer (2020)
**Implementation:** `cfts/cf_lasts/lasts.py`

**Description:** Comprehensive explainability method that provides factual and counterfactual subsequence-based rules, exemplar and counterexemplar time series, and shapelet-based decision tree explanations. LASTS uses an autoencoder to project time series into latent space, generates a neighborhood using genetic algorithms, trains a shapelet-based decision tree surrogate model, and extracts interpretable rules and exemplar/counterexemplar instances.

**Key Features:**
- **Genetic algorithm**: Generates neighborhood in latent space through evolutionary operations
- **Shapelet-based rules**: Extracts factual (why classified as X) and counterfactual (how to change to Y) rules
- **Exemplars & counterexemplars**: Provides concrete examples of similar time series with same/different labels
- **Surrogate model**: Local decision tree for interpretable explanations
- **Comprehensive explanation**: Combines multiple explanation types (rules, examples, importance)

**Algorithm Steps:**
1. Encode instance to latent space using autoencoder
2. Generate neighborhood using genetic algorithm with mutation/crossover
3. Decode neighborhood back to time series space
4. Train shapelet-based decision tree surrogate on neighborhood
5. Extract factual/counterfactual rules and exemplar/counterexemplar instances
6. Find closest counterfactual for actionable recommendations

**Reference:**
```bibtex
@inproceedings{guidotti2020lasts,
  title={Explaining Any Time Series Classifier},
  author={Guidotti, Riccardo and Monreale, Anna and Spinnato, Francesco and Pedreschi, Dino and Giannotti, Fosca},
  booktitle={2020 IEEE Second International Conference on Cognitive Machine Intelligence (CogMI)},
  pages={167--176},
  year={2020},
  organization={IEEE}
}
```

**Links:**
- Paper: IEEE CogMI 2020
- Repository: [https://github.com/fspinna/LASTS_explainer](https://github.com/fspinna/LASTS_explainer)
- Blog Post: [https://sobigdata.eu/blog/explaining-any-time-series-classifier](https://sobigdata.eu/blog/explaining-any-time-series-classifier)

**Usage Example:**
```python
from cfts.cf_lasts.lasts import lasts_cf, LASTS

# Simple counterfactual generation
cf, prediction = lasts_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    latent_dim=32,
    n_samples=500,
    n_iterations=100,
    train_ae_epochs=50,
    verbose=True
)

# Full explanation with rules and exemplars
lasts_explainer = LASTS(model, autoencoder=pretrained_ae)
explanation = lasts_explainer.explain(
    sample,
    dataset=dataset,
    latent_dim=32,
    n_samples=500,
    n_iterations=100,
    binarize_labels=True,
    verbose=True
)

# Access explanation components
print(f"Original class: {explanation['original_class']}")
print(f"Factual rule: {explanation['rules']['factual']}")
print(f"Counterfactual rule: {explanation['rules']['counterfactual']}")
print(f"Exemplars: {explanation['exemplars'].shape}")
print(f"Counterexemplars: {explanation['counterexemplars'].shape}")
print(f"Closest counterfactual: {explanation['closest_counterfactual']}")
```

---

### Segment-Based Methods

#### 20. SETS - Shapelet-Based Counterfactual Explanations (2022)
**Implementation:** `cfts/cf_sets/sets.py`  
**Paper:** "Shapelet-Based Counterfactual Explanations for Multivariate Time Series"  
**Authors:** Omar Bahri, Soukaina Filali Boubrahimi, Shah Muhammad Hamdi  
**Conference:** ACM SIGKDD Workshop on Mining and Learning from Time Series (KDD-MiLeTS 2022)  
**arXiv:** https://arxiv.org/abs/2208.10462  
**Reference Implementation:** https://github.com/fzi-forschungszentrum-informatik/TSInterpret

**Description:** SETS is a shapelet-based counterfactual explanation method that leverages discriminative shapelets (subsequences) to generate interpretable counterfactuals by identifying which class-specific patterns need to be removed or introduced to change the prediction.

**Key Features:**
- **Shapelet extraction**: Discovers class-discriminative subsequences from training data
- **Location detection**: Identifies where shapelets occur in the instance to explain
- **Two-phase modification**: Removes original-class shapelets and introduces target-class shapelets
- **Amplitude scaling**: Scales shapelets to match local statistics for realistic modifications
- **Multivariate support**: Handles multivariate time series with dimension-specific shapelets

**Method:**
1. Extract discriminative shapelets per class from training data
2. Find locations where original-class shapelets occur in the instance
3. Replace original-class shapelets with corresponding segments from target-class nearest neighbor
4. Insert target-class shapelets at important locations (high variance regions)
5. Validate modifications successfully change prediction to target class

**Usage Example:**
```python
from cfts.cf_sets.sets import sets_cf, sets_explain

# Basic counterfactual generation
cf, prediction = sets_cf(
    sample=ts_sample,
    model=trained_model,
    target_class=1,
    dataset=train_dataset,
    n_shapelets_per_class=5,
    shapelet_lengths=[5, 10, 20],
    threshold=0.5,
    verbose=True
)

# Detailed explanation with shapelet information
explanation = sets_explain(
    sample=ts_sample,
    model=trained_model,
    target_class=1,
    dataset=train_dataset
)
```

**Reference:**
```bibtex
@inproceedings{bahri2022sets,
  title={Shapelet-Based Counterfactual Explanations for Multivariate Time Series},
  author={Bahri, Omar and Boubrahimi, Soukaina Filali and Hamdi, Shah Muhammad},
  booktitle={Proceedings of the 8th ACM SIGKDD Workshop on Mining and Learning from Time Series},
  year={2022},
  url={https://arxiv.org/abs/2208.10462}
}
```

---

#### 21. SG-CF - Shapelet-Guided Counterfactual Explanations (2022)
**Implementation:** `cfts/cf_sg_cf/sg_cf.py`  
**Paper:** "SG-CF: Shapelet-Guided Counterfactual Explanation for Time Series Classification"  
**Authors:** Peiyu Li, Omar Bahri, Soukaina Filali Boubrahimi, Shah Muhammad Hamdi  
**Conference:** 2022 IEEE International Conference on Big Data (Big Data)  
**DOI:** 10.1109/bigdata55660.2022.10020866  
**GitHub:** https://github.com/Luckilyeee/SG-CF

**Description:** SG-CF extends the Wachter counterfactual framework with shapelet-based guidance to generate interpretable counterfactuals. It uses discriminative shapelets to identify critical temporal patterns and focuses modifications within shapelet regions through gradient masking.

**Key Features:**
- **Shapelet extraction**: Discovers class-discriminative subsequences from training data using k-means
- **Gradient masking**: Focuses gradient updates within shapelet regions for concentrated modifications
- **Prominent segment detection**: Identifies most important regions based on gradient magnitude
- **Progressive segment expansion**: Gradually increases modification region size
- **Lambda bisection**: Adaptively balances proximity and validity through lambda tuning
- **Multi-objective optimization**: Balances validity, proximity, sparsity, and contiguity

**Usage Example:**
```python
from cfts.cf_sg_cf.sg_cf import sg_cf, sg_cf_explain

# Basic counterfactual generation
cf, prediction = sg_cf(
    sample=ts_sample,
    model=trained_model,
    target_class=1,
    dataset=train_dataset,
    max_iter=1000,
    max_lambda_steps=10,
    lambda_init=0.1,
    learning_rate=0.1,
    segment_rate_init=0.05,
    target_proba=0.95,
    verbose=True
)

# Detailed explanation with shapelet information
explanation = sg_cf_explain(
    sample=ts_sample,
    model=trained_model,
    target_class=1,
    dataset=train_dataset,
    verbose=True
)

# Access explanation details
print(f"Original class: {explanation['original_class']}")
print(f"Target class: {explanation['target_class']}")
print(f"Shapelets used: {explanation['n_target_shapelets']}")
print(f"Distance: {explanation['distance']:.4f}")
print(f"Success: {explanation['success']}")
```

**Reference:**
```bibtex
@inproceedings{li2022sg,
  title={SG-CF: Shapelet-Guided Counterfactual Explanation for Time Series Classification},
  author={Li, Peiyu and Bahri, Omar and Boubrahimi, Souka{\"\i}na Filali and Hamdi, Shah Muhammad},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={1564--1569},
  year={2022},
  organization={IEEE},
  doi={10.1109/bigdata55660.2022.10020866}
}
```

---

#### 22. DisCOX - Discord-based Counterfactual Explanations (2024)
**Implementation:** `cfts/cf_discox/discox.py`  
**Paper:** "Discord-based counterfactual explanations for time series classification"  
**Authors:** Omar Bahri, Peiyu Li, Soukaina Filali Boubrahimi, Shah Muhammad Hamdi  
**Journal:** Data Mining and Knowledge Discovery, Springer (2024)  
**DOI:** 10.1007/s10618-024-01028-9

**Description:** DisCOX identifies and modifies discordant subsequences (the most anomalous patterns) in time series to generate interpretable counterfactual explanations. The method leverages matrix profile analysis to find discord regions and replaces them with patterns from the target class.

**Key Features:**
- **Matrix profile analysis**: Computes matrix profile to identify discord subsequences
- **Discord discovery**: Finds top-k most anomalous (discordant) subsequences
- **Prototype-based replacement**: Replaces discord regions with corresponding patterns from target class
- **Multi-strategy modification**: Supports prototype replacement, amplification, attenuation, and inversion
- **Amplitude scaling**: Scales replacement regions to match local statistics
- **Interpretability**: Modifications focus on anomalous regions, making changes more understandable

**Usage Example:**
```python
from cfts.cf_discox import discox_cf, discox_explain

# Basic counterfactual generation
cf, prediction = discox_cf(
    sample=ts_sample,
    model=trained_model,
    target_class=1,
    dataset=train_dataset,
    window_size=20,  # or None for automatic (10% of series length)
    max_iterations=100,
    verbose=True
)

# Detailed explanation with discord information
explanation = discox_explain(
    sample=ts_sample,
    model=trained_model,
    target_class=1,
    dataset=train_dataset,
    window_size=20,
    verbose=True
)

# Access explanation details
print(f"Original class: {explanation['original_class']}")
print(f"Target class: {explanation['target_class']}")
print(f"Discord info: {explanation['discord_info']}")
print(f"Number of discords: {explanation['n_discords_found']}")
print(f"Success: {explanation['success']}")
```

**Reference:**
```bibtex
@article{bahri2024discox,
  title={Discord-based counterfactual explanations for time series classification},
  author={Bahri, Omar and Li, Peiyu and Boubrahimi, Soukaina Filali and Hamdi, Shah Muhammad},
  journal={Data Mining and Knowledge Discovery},
  year={2024},
  publisher={Springer},
  doi={10.1007/s10618-024-01028-9}
}
```

**Related Work on Discords:**
```bibtex
@inproceedings{keogh2005hot,
  title={Hot sax: Efficiently finding the most unusual time series subsequence},
  author={Keogh, Eamonn and Lin, Jessica and Fu, Ada},
  booktitle={Fifth IEEE International Conference on Data Mining (ICDM)},
  pages={8--pp},
  year={2005},
  organization={IEEE}
}

@article{yeh2016matrix,
  title={Matrix profile I: All pairs similarity joins for time series},
  author={Yeh, Chin-Chia Michael and Zhu, Yan and Ulanova, Liudmila and Begum, Nurjahan and Ding, Yifei and Dau, Hoang Anh and Silva, Diego Furtado and Mueen, Abdullah and Keogh, Eamonn},
  booktitle={2016 IEEE 16th International Conference on Data Mining (ICDM)},
  pages={1317--1322},
  year={2016},
  organization={IEEE}
}
```

---

#### 23. CFWoT - Counterfactual Explanations Without Training Datasets (2024)
**Implementation:** `cfts/cf_cfwot/cfwot.py`

**Description:** Reinforcement learning-based counterfactual explanation method for both static and multivariate time-series data. CFWoT operates without requiring training datasets and is model-agnostic, supporting both differentiable and non-differentiable models.

**Key Features:**
- **No training dataset required**: Operates without access to training data
- **Model-agnostic**: Works with any predictive model (differentiable or non-differentiable)
- **Multivariate support**: Handles multivariate time-series and static data
- **Mixed feature types**: Supports continuous and discrete features
- **User preferences**: Allows feature feasibility weights and constraints
- **Causal constraints**: Supports actionable features and causal relationships
- **Policy-based approach**: Uses reinforcement learning with policy network

**Algorithm:**
CFWoT uses a policy network that outputs distributions for action selection:
- **a_time**: Which time step to intervene on
- **a_feat**: Which feature to modify
- **a_stre**: The strength/value of the intervention

The policy is trained via policy gradient methods (REINFORCE) to maximize reward based on:
- Validity: Achieving target class prediction
- Proximity: Minimizing distance from original instance
- Sparsity: Minimizing number of feature modifications
- Feasibility: Respecting user-defined constraints

**Reference:**
```bibtex
@article{sun2024cfwot,
  title={Counterfactual Explanations for Multivariate Time-Series without Training Datasets},
  author={Sun, Xiangqian and Aoki, Ryota and Wilson, Kevin H},
  journal={arXiv preprint arXiv:2405.18563},
  year={2024}
}
```

**Links:**
- Paper: [arXiv:2405.18563](https://arxiv.org/abs/2405.18563)
- HTML: [arXiv HTML](https://arxiv.org/html/2405.18563v1)

**Usage Example:**
```python
from cfts.cf_cfwot.cfwot import cfwot

cf, prediction = cfwot(
    sample=sample,           # shape (K, D) for time series, or (D,) for static
    model=model,
    target_class=1,
    D_act=None,               # actionable feature indices (default: all)
    W_fsib=None,               # optional per-feature feasibility weights
    lambda_pxmt=0.001,        # proximity weight in the reward
    M_E=100,                  # max episodes
    M_T=100,                  # max interventions per episode
    gamma=0.99,               # RL discount factor
    lr=0.0001,
    verbose=True
)
```

---

#### 24. TS-CEM - Contrastive Explanation Method for Time Series (2020)
**Implementation:** `cfts/cf_cem/cem.py`

**Description:** Applies the Contrastive Explanation Method (CEM) to time series classification, finding Pertinent Negatives (PN) that change the model's prediction or Pertinent Positives (PP) that preserve it. Optimization is performed via FISTA with L1/L2 regularization and an optional autoencoder reconstruction loss.

**Key Features:**
- **Pertinent Negatives (PN)**: Minimal additions to the input that flip the model's prediction
- **Pertinent Positives (PP)**: Minimal subset of the input that preserves the prediction
- **FISTA optimization**: Fast iterative shrinkage-thresholding for sparse solutions
- **Autoencoder support**: Optional reconstruction loss encourages in-distribution counterfactuals
- **Binary search**: Automatically tunes the confidence penalty constant *c*

**References:**
```bibtex
@inproceedings{labaien2020contrastive,
  title={Contrastive Explanations for a Deep Learning Model on Time-Series Data},
  author={Labaien, Jokin and Zugasti, Ekhi and De Carlos, Xabier},
  booktitle={Big Data Analytics and Knowledge Discovery (DaWaK 2020)},
  series={Lecture Notes in Computer Science},
  volume={12393},
  pages={190--204},
  year={2020},
  publisher={Springer}
}

@inproceedings{dhurandhar2018explanations,
  title={Explanations based on the missing: Towards contrastive explanations with pertinent negatives},
  author={Dhurandhar, Amit and Chen, Pin-Yu and Luss, Ronny and Tu, Chun-Chen and Ting, Paishun and Shanmugam, Karthikeyan and Das, Payel},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={31},
  year={2018}
}
```

**Links:**
- Paper: [Springer DaWaK 2020](https://doi.org/10.1007/978-3-030-59065-9_19)
- Original CEM (IBM): [https://github.com/IBM/Contrastive-Explanation-Method](https://github.com/IBM/Contrastive-Explanation-Method)

**Usage Example:**
```python
from cfts.cf_cem.cem import cem_cf

# Pertinent Negative (PN) – find minimal addition that flips prediction
cf, prediction = cem_cf(
    sample=sample,
    model=model,
    mode='PN',
    autoencoder=None,
    kappa=0.5,
    beta=0.1,
    gamma=0.2,
    c_init=10.0,
    c_steps=5,
    max_iterations=500,
    learning_rate=1e-2
)

# Pertinent Positive (PP) – find minimal subset that preserves prediction
cf, prediction = cem_cf(
    sample=sample,
    model=model,
    mode='PP',
    autoencoder=None,
    kappa=0.5,
    beta=0.1
)
```

---

#### 25. MASCOTS - Model-Agnostic Symbolic COunterfactual explanations for Time Series (2025)
**Implementation:** `cfts/cf_mascots/mascots.py`

**Description:** Builds a SAX-based bag-of-receptive-fields (BoRF) symbolic surrogate over the training data, then generates counterfactuals through importance-guided "word swaps" of symbolic subsequences until the surrogate (and underlying model) predictions flip to the target class. Feature importance can come from the surrogate's own linear coefficients or from SHAP values.

**Key Features:**
- **Symbolic representation**: SAX/BoRF bag-of-words encoding of subsequences at multiple window sizes
- **Importance-guided swaps**: replaces high-importance symbolic words with alternatives, sampling among the top-k candidates at random for diversity
- **Multiple restarts**: several independent search restarts for more diverse/robust solutions
- **Pluggable swap distributions**: scalar swaps by default, or Gaussian-informed swaps when `gpytorch` is available
- **Model-agnostic**: only requires prediction/probability functions, so it wraps any classifier

**Reference:**
```bibtex
@article{pludowski2025mascots,
  title={MASCOTS: Model-Agnostic Symbolic COunterfactual explanations for Time Series},
  author={P{\l}udowski, Dawid and Spinnato, Francesco and Wilczy{\'n}ski, Piotr and Kotowski, Krzysztof and Ntagiou, Evridiki V and Guidotti, Riccardo and Biecek, Przemys{\l}aw},
  journal={arXiv preprint arXiv:2503.22389},
  year={2025}
}
```

**Links:**
- Paper: [arXiv:2503.22389](https://arxiv.org/abs/2503.22389)
- Repository: [https://github.com/DawidPludowski/borf](https://github.com/DawidPludowski/borf)

**Usage Example:**
```python
from cfts.cf_mascots import mascots_cf

cf, scores = mascots_cf(
    sample=sample,
    model=model,
    dataset=dataset,          # sequence of (x, y) pairs used to build the BoRF surrogate
    target_class=1,
    max_iter=100,
    swap_method="scalar",     # "scalar" or "gaussian" (requires gpytorch)
    n_restarts=3,
    C=0.1,
    select_top_k=5,
    attribution_name="coef",  # "coef" or "shap"
    verbose=True,
)
```

---

### Hybrid Methods

#### 26. SPARCE - Generating SPARse Counterfactual Explanations (2022)
**Implementation:** `cfts/cf_sparce/sparce.py`

**Description:** GAN-based architecture to generate sparse counterfactual explanations for multivariate time series. The generator creates residuals (modifications) that are added to the input query to produce counterfactuals. The approach regularizes the loss with adversarial, classification, similarity, sparsity, and smoothness (jerk) losses.

**Key Features:**
- **GAN architecture**: Uses generator-discriminator framework for realistic counterfactuals
- **Residual generation**: Generates modifications rather than entire sequences
- **Multi-objective optimization**: Balances adversarial, classification, similarity, sparsity, and smoothness
- **Bidirectional LSTM**: Handles temporal dependencies in both directions
- **Sparsity emphasis**: L0 norm encourages sparse modifications
- **Smoothness constraint**: Jerk loss ensures smooth trajectory changes
- **Multivariate support**: Designed for multi-channel time series

**Loss Function Components:**
- **L_adv**: Adversarial loss (discriminator-based)
- **L_class**: Classification loss (target class prediction)
- **L_sim**: Similarity loss (L1 norm between query and counterfactual)
- **L_sparse**: Sparsity loss (L0 norm encouraging sparse modifications)
- **L_jerk**: Jerk loss (smoothness of trajectory changes)

**Total Loss:** L = λ_adv·L_adv + λ_class·L_class + λ_sim·L_sim + λ_sparse·L_sparse + λ_jerk·L_jerk

**Reference:**
```bibtex
@article{lang2022sparce,
  title={Generating Sparse Counterfactual Explanations For Multivariate Time Series},
  author={Lang, Jana and Giese, Martin and Ilg, Winfried and Otte, Sebastian},
  journal={arXiv preprint arXiv:2206.00931},
  year={2022}
}
```

**Links:**
- Paper: [arXiv:2206.00931](https://arxiv.org/abs/2206.00931)
- Repository: [https://github.com/janalang/SPARCE](https://github.com/janalang/SPARCE)

**Usage Example:**
```python
from cfts.cf_sparce.sparce import sparce_gan_cf, sparce_gradient_cf

# GAN variant: trains its own generator/discriminator per call
cf, prediction = sparce_gan_cf(
    sample=sample,
    model=model,
    target_class=1,
    lambda_adv=1.0,
    lambda_cls=1.0,
    lambda_sim=1.0,
    lambda_sparse=1.0,
    lambda_jerk=1.0,
    num_epochs=50
)

# Lighter-weight gradient-optimization variant (no GAN training)
cf, prediction = sparce_gradient_cf(
    sample=sample,
    model=model,
    target_class=1,
    lambda_cls=1.0,
    lambda_sim=1.0,
    lambda_sparse=1.0,
    lambda_jerk=1.0,
    max_iter=100
)
```

---

#### 27. CFE4MTS - Plausible Conditional Generation-based Counterfactual Explanations for Multivariate Time Series Classification (2025)
**Implementation:** `cfts/cf_cfe4mts/cfe4mts.py`

**Description:** A conditional, class-generative GAN-style method: a "central noiser" network is trained once per dataset to predict an additive perturbation δ = N(X, y_target) given a query and a one-hot target class, while a "central discriminator" (single-direction LSTM) judges whether (X + δ, y_target) pairs are plausible. Once trained, generating a counterfactual for any (query, target class) pair is a single forward pass — no per-sample optimization. It is a multivariate, class-conditional extension of CFE4SITS.

**Key Features:**
- **Amortized inference**: one trained noiser generates counterfactuals for any query/target-class pair in a single forward pass
- **Adversarial + classification + distance losses**: `L_noiser = λ_gen·L_gen + λ_clas·L_cla + λ_dist·L_dist` balances realism, validity, and sparsity/contiguity
- **Circular-distance sparsity term**: `L_dist` concentrates perturbation mass around a single contiguous window per channel via a circular (mod T) distance to the peak-perturbation time step
- **Random target sampling during training**: targets are sampled uniformly at random per training instance so the noiser generalizes across all class pairs
- **Fit/generate split**: `cfe4mts_fit` trains once, `cfe4mts_generate` produces cheap repeated counterfactuals; `cfe4mts_cf` offers a one-shot fit+generate call

**Reference:**
```bibtex
@inproceedings{sevellec2025cfe4mts,
  title={Plausible Conditional Generation-based Counterfactual Explanations for Multivariate Time Series Classification},
  author={Sevellec, Paul and Fromont, Elisa and Gaudel, Romain and Roze, Laurence and Sammarco, Matteo},
  booktitle={European Conference on Artificial Intelligence (ECAI)},
  year={2025}
}
```

**Links:**
- Paper: [HAL preprint](https://hal.science/hal-04928456v2/file/m1254.pdf)
- Repository: [https://github.com/PaulSevellec/CFE4MTS](https://github.com/PaulSevellec/CFE4MTS)

**Usage Example:**
```python
from cfts.cf_cfe4mts.cfe4mts import cfe4mts_cf, cfe4mts_fit, cfe4mts_generate

# One-shot: train a fresh noiser and generate immediately (single sample)
cf, prediction = cfe4mts_cf(
    sample=sample,
    dataset=dataset,
    model=model,
    target_class=1,
    epochs=100,
    lambda_gen=1.0,
    lambda_clas=10.0,
    lambda_dist=0.01,
)

# Preferred when explaining many samples: fit once, generate cheaply per sample
fitted = cfe4mts_fit(dataset, model, epochs=100, lambda_clas=10.0, lambda_dist=0.01)
cf, prediction = cfe4mts_generate(fitted, sample, model, target_class=1)
```

---

#### 28. Time-CF - Shapelet-based Model-agnostic Counterfactual Local Explanations
**Implementation:** `cfts/cf_time_cf/time_cf.py`

**Description:** Time-CF leverages shapelets and TimeGAN to provide counterfactual explanations for arbitrary time series classifiers. The method extracts discriminative shapelet candidates using Random Shapelet Transform (RST), trains TimeGAN on instances from other classes (not the to-be-explained class), generates synthetic instances, and replaces shapelet regions in the original instance with synthetic shapelets. The counterfactual with minimum Hamming distance that flips the prediction is returned.

**Algorithm Steps:**
1. Extract shapelet candidates using Random Shapelet Transform
2. Sort shapelets by information gain and select top N discriminative shapelets
3. Train TimeGAN on instances from OTHER classes (not the to-be-explained class)
4. Generate M synthetic instances using TimeGAN
5. For each shapelet candidate, find its position in the original instance
6. Crop the same time interval from each generated fake instance to get fake shapelets
7. Replace shapelet regions in the original instance with synthetic shapelets
8. Test if replacement creates valid counterfactual (flips prediction)
9. Return counterfactual with minimum Hamming distance

**Reference:**
```bibtex
@article{huang2024timecf,
  title={Shapelet-based Model-agnostic Counterfactual Local Explanations for Time Series Classification},
  author={Huang, Qi and Chen, Wei and B{\"a}ck, Thomas and van Stein, Niki},
  journal={arXiv preprint arXiv:2402.01343},
  year={2024}
}
```

**Links:**
- Paper: https://arxiv.org/abs/2402.01343

**Usage Example:**
```python
from cfts.cf_time_cf.time_cf import time_cf_generate

cf, prediction = time_cf_generate(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    n_shapelets=10,
    M=32,
    timegan_epochs=100
)
```

---

#### 29. TeRCE - Temporal Rule-Based Counterfactual Explanations (2022)
**Implementation:** `cfts/cf_terce/terce.py`

**Description:** TeRCE generates counterfactual explanations by mining class-specific temporal rules using discriminative shapelet pairs, then systematically removing original class rules and introducing target class rules through nearest unlike neighbor (NUN) replacement with min-max normalization for scale adaptation.

**Key Features:**
- **Temporal rule discovery**: Mines discriminative shapelet pairs as temporal rules using RuleTransform
- **Class-specific rules**: Identifies exclusive rules that occur only in specific classes (>90% purity)
- **Two-stage strategy**: First removes original class rules, then introduces target class rules
- **Nearest unlike neighbor**: Finds similar instances from target class for pattern replacement
- **Min-max normalization**: Adapts shapelet scales to match local time series statistics
- **Combinatorial search**: Tries combinations of rules when single rules insufficient
- **Heatmap localization**: Uses class-specific heatmaps to determine shapelet placement locations
- **Multivariate support**: Naturally handles multi-dimensional time series through paired shapelets

**Algorithm:**

1. **Rule Mining Phase** (offline):
   - Mine discriminative shapelet pairs using Contracted Shapelet Transform (CST)
   - For each shapelet pair (sh₁, sh₂): compute Fisher score based on co-occurrence patterns
   - Select top-k rules ranked by discriminative power
   - Identify class-specific rules: rules occurring exclusively (>90%) in one class
   - Build class-specific heatmaps: aggregate shapelet locations across training instances

2. **Counterfactual Generation** (online):
   - **Step 1 - Original Rule Removal**:
     - Identify which class-specific rules of original class occur in query instance
     - Find nearest unlike neighbor (NUN) from target class
     - For each original rule occurrence:
       - Extract NUN segments at corresponding shapelet locations
       - Apply min-max normalization: map NUN segment to query's local scale
       - Replace query segments with normalized NUN segments
     - Check if prediction changed to target class
   
   - **Step 2 - Target Rule Introduction**:
     - If prediction not changed, introduce target class rules
     - For each target class rule:
       - Determine placement location using class-specific heatmap (center of distribution)
       - Extract rule's shapelet pair from training data
       - Apply min-max normalization to match query's local scale at placement location
       - Insert normalized shapelets at computed positions
       - Check if prediction changed to target class
   
   - **Step 3 - Combinatorial Search**:
     - If single rules insufficient, try combinations of target rules
     - Use iterative combinatorial search through rule subsets
     - Return first valid counterfactual found

3. **Min-max Normalization**: For shapelet s and segment t:
   - If range(s) ≠ 0: t' = (max(t) - min(t)) × (s - min(s)) / (max(s) - min(s)) + min(t)
   - If range(s) = 0: t' = (max(t) + min(t)) / 2 × ones(len(s))

**Reference:**
```bibtex
@inproceedings{bahri2022terce,
  title={Temporal Rule-Based Counterfactual Explanations for Multivariate Time Series},
  author={Bahri, Omar and Li, Peiyu and Boubrahimi, Soukaina Filali and Hamdi, Shah Muhammad},
  booktitle={2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)},
  pages={1244--1249},
  year={2022},
  organization={IEEE},
  doi={10.1109/ICMLA55696.2022.00200}
}
```

**Links:**
- Paper: [IEEE ICMLA 2022](https://ieeexplore.ieee.org/document/10069254)
- Repository: [https://github.com/omarbahri/TeRCE](https://github.com/omarbahri/TeRCE)
- RuleTransform: [https://github.com/omarbahri/RuleTransform](https://github.com/omarbahri/RuleTransform)

**Usage Example:**
```python
from cfts.cf_terce.terce import terce_generate

# Generate TeRCE counterfactual (simplified version using gradient saliency)
cf, cf_label = terce_generate(
    sample=sample,
    model=model,
    X_train=X_train,
    y_train=y_train,
    target_class=1,
    n_regions=5,  # number of important regions to replace
    window_size_ratio=0.1,  # size of regions as ratio of time series length
    verbose=True
)

# Automatic target selection
cf, cf_label = terce_generate(
    sample=sample,
    model=model,
    X_train=X_train,
    y_train=y_train,
    n_regions=10,  # more regions for complex cases
    window_size_ratio=0.05  # smaller regions for finer control
)
```

**Note:** The implementation in this library is a simplified version that uses gradient-based saliency to identify important regions instead of mining explicit temporal rules with RuleTransform. For the full TeRCE algorithm with shapelet-based rule mining, see the original repository.

---

#### 30. MG-CF - Motif-Guided Counterfactual Explanations
**Implementation:** `cfts/cf_mg_cf/mg_cf.py`

**Description:** MG-CF uses shapelet transform to extract discriminative motifs (subsequences) from training data and generates counterfactuals by replacing the corresponding motif region in the original instance with the motif from the target class. This is a simple yet effective model-agnostic method that produces sparse and contiguous explanations.

**Algorithm Steps:**
1. Extract discriminative motifs using Shapelet Transform for each class
2. Sort motifs by information gain and select best motif per class  
3. For a query instance, identify the target class motif region
4. Replace that region with the target class motif to create counterfactual
5. Verify if the counterfactual flips the prediction

**Reference:**
```bibtex
@inproceedings{li2022motif,
  title={Motif-guided time series counterfactual explanations},
  author={Li, Peiyu and Boubrahimi, Souka{\"i}na Filali and Hamdi, Shah Muhammad},
  booktitle={International Conference on Pattern Recognition},
  pages={203--215},
  year={2022},
  organization={Springer}
}
```

**Links:**
- Paper: https://arxiv.org/abs/2211.04411
- arXiv: 2211.04411v3
- GitHub: https://github.com/Luckilyeee/motif_guided_cf

**Usage Example:**
```python
from cfts.cf_mg_cf import mg_cf_generate

cf, prediction = mg_cf_generate(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
    n_shapelets=100,
    lengths_ratio=[0.3, 0.5, 0.7]
)
```

---

#### 31. TimeX - Encoding Time-Series Explanations (2023)
**Implementation:** `cfts/cf_timex/timex.py`

**Description:** Time series explainer that learns interpretable surrogate models through self-supervised model behavior consistency, generating saliency-based explanations.

**Reference:**
```bibtex
@article{mujkanovic2023timex,
  title={TimeX: Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency},
  author={Mujkanovic, Felix and Dosen{\'c}ovi{\'c}, Vanja and Vranješ, Marinela and Petkovi{\'c}, Matej and Schiele, Bernt and Frintrop, Simone},
  journal={arXiv preprint arXiv:2306.02109},
  year={2023}
}
```

**Links:**
- Paper: [arXiv:2306.02109](https://arxiv.org/abs/2306.02109)
- Repository: [https://github.com/mims-harvard/TimeX](https://github.com/mims-harvard/TimeX)

**Usage Example:**
```python
from cfts.cf_timex.timex import timex_explanation

# Note: Requires pre-trained TimeX model
saliency, prediction = timex_explanation(
    sample=sample,
    model=model,
    timex_model=pretrained_timex_model,
    return_saliency=True
)
```

**Naming collision — the counterfactual "TimeX" used in `example_metrics_evaluation.py`:**
`example_metrics_evaluation.py` evaluates a *different* method also called "TimeX", implemented in `cfts/cf_timex/timex_cf.py::timex_cf`. That one is a Wachter-style gradient optimiser with an added DTW class-prototype term, matching the "TimeX" method from the TS-Counterfactual-Explanation-Bake-off benchmark (companion code to "Counterfactual Explanation Bake-off: A Review and Experimental Evaluation for Time Series Classification", Machine Learning Journal 2026) — an unrelated paper that happens to share the name. It requires no pre-training (it optimises per query, like the rest of this repository's Wachter-family methods) and produces an actual counterfactual time series, not a saliency map. See `timex_cf.py`'s module docstring for the full naming-collision discussion.

```python
from cfts.cf_timex.timex_cf import timex_cf

cf, prediction = timex_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=dataset,
)
```

---

#### 32. TimeXcf++ - Unified Information Bottleneck Framework for Time Series Explanations (2026)
**Implementation:** `cfts/cf_timex_plus_plus/timex_plus_plus.py`

**Status: implements the paper's own counterfactual path (TimeXcf++), not a hand-retargeted explainer.** arXiv:2608.25897 unifies the ICML2024 TimeX++ architecture into two explicit paths sharing one bottleneck extractor g_phi: an attribution path (TimeXa++, label-consistency against the query's own class) and a counterfactual path (TimeXcf++, label-consistency against a target class Y'', with its own generator/reference structure). `timexplusplus_cf` implements the latter directly: a straight-through-Bernoulli-sampled bottleneck mask M gates a learned perturbation E (plus a target-class-conditioned noise term used only while training), anchored by a structural loss that forces everything outside M back to the original query. See the module's docstring for the full pipeline, the paper details it leaves unspecified (generator architecture, L_KL closed form, alpha/beta/lambda_con numeric defaults), and `timexplusplus_fit` / `timexplusplus_generate` for reusing one trained extractor/generator set across several queries that share a target class.

**Description:** Unifies attribution and counterfactual time-series explanation under one information-bottleneck objective, sharing a stochastic bottleneck extractor between "preserved information yields attribution explanations" and "controlled information removal produces stable counterfactual explanations."

**Key Features:**
- **Unified information bottleneck framework**: one extractor, two label-consistency targets (original class for attribution, target class for counterfactual)
- **Bottleneck extractor g_phi**: transformer encoder producing a stochastic per-(timestep, channel) mask distribution pi
- **Straight-through Bernoulli mask sampling**: M = STE(Bern(pi)), gradients bypass the discrete sampling op
- **Counterfactual generators psi_cf / psi_n**: psi_cf predicts the edit E applied within M; psi_n folds in a target-class reference instance as training-only noise
- **Hard causal anchor**: a structural loss forces the counterfactual to equal the original query outside M, confining edits to the sparse bottleneck
- **Label consistency**: JS-divergence term, computed against the target class for the counterfactual path

**Algorithm (counterfactual path):**
1. **Bottleneck extraction**: transformer g_phi maps input X to stochastic selection probabilities pi
2. **Mask sampling**: M = STE(Bern(pi)) (straight-through estimator)
3. **Perturbation**: E = psi_cf(X, M); training-only noise epsilon = psi_n(X, M, X_ref) from a target-class reference X_ref
4. **Counterfactual instance**: X̃_cf = X + M⊙E + epsilon (training) / X + M⊙E (inference)
5. **Compactness**: minimizes KL(Bernoulli(pi) || Bernoulli(r)) + continuity penalty, r=0.1 by default for this path (paper's own value, stricter than the attribution path's r=0.5)
6. **Structural/bound loss**: ‖(1-M)⊙(X̃_cf - X)‖² anchors the unedited region to the original query
7. **Overall Objective**: ℒ = ℒ_LC(target_class, ·) + α·ℒ_M + β·(ℒ_KL + ℒ_bound)

**Reference:**
```bibtex
@article{zheng2026unifiedib,
  title={Towards A Unified Information Bottleneck Framework for Time Series Explanations},
  author={Zheng, Xu and Liu, Zichuan and Chen, Zhuomin and Akewar, Mayur and Bhimani, Janki and Liu, Jason and Sha, Mo and Ni, Jingchao and Cheng, Wei and Luo, Dongsheng},
  journal={arXiv preprint arXiv:2608.25897},
  year={2026}
}
@inproceedings{liu2024timexplusplus,
  title={TimeX++: Learning Time-Series Explanations with Information Bottleneck},
  author={Liu, Zichuan and Wang, Tianchun and Shi, Jimeng and Xu, Zheng and Chen, Zhuomin and Song, Lei and Dong, Wenqian and Obeysekera, Jayantha and Shirani, Farhad and Luo, Dongsheng},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```

**Links:**
- Paper (counterfactual path, TimeXcf++): [arXiv:2608.25897](https://arxiv.org/abs/2608.25897)
- Paper (shared extractor architecture, TimeX++): [arXiv:2405.09308](https://arxiv.org/abs/2405.09308)
- Repository (TimeX++ ICML2024 release): [https://github.com/zichuan-liu/TimeXplusplus](https://github.com/zichuan-liu/TimeXplusplus)

**Usage Example:**
```python
from cfts.cf_timex_plus_plus.timex_plus_plus import timexplusplus_cf

# Trains a fresh bottleneck extractor + perturbation/noise generators on
# `dataset`, targeting target_class, then generates a counterfactual for
# `sample` with one forward pass. alpha/beta default to values retuned for
# reliable validity (see the module docstring); r=0.1 is the paper's own
# stated default for this (counterfactual) path.
cf, prediction = timexplusplus_cf(
    sample=sample,
    model=model,
    target_class=1,
    dataset=training_data,
    alpha=0.5,  # Compactness weight
    beta=0.25,  # Distribution/structural consistency weight
    r=0.1,      # Mask sparsity parameter (paper's counterfactual-path default)
    epochs=80,
)

# Pass return_mask=True for a 3-tuple (cf, prediction, mask): the sparse
# bottleneck mask M the counterfactual edit was confined to.
cf, prediction, mask = timexplusplus_cf(
    sample=sample, model=model, target_class=1, dataset=training_data,
    return_mask=True,
)
```

---

#### 33. CoDec - Counterfactual Decomposition (2026, in progress)
**Implementation:** `cfts/cf_codec/`

**Status: prototype / work in progress.** No standalone publication yet - CoDec generalizes IMFACT (#14) into a modular framework and is being validated ahead of a September 2026 planning meeting; see `cfts/cf_codec/CoDec_workplan.md` and `CoDec_presentation.pdf` for the design docs this implementation follows.

**Description:** Generalizes IMFACT's single-decomposition (EMD), single-reference (NUN), greedy index-matched pipeline into a framework where reference selection, decomposition, matching, and perturbation are each independently swappable. Given a query series and a fitted black-box classifier, it decomposes the query into components, substitutes one or more components with a matched donor's components, and reconstructs - widening the substituted-component set or advancing to the next reference on failure until the classifier flips or the search budget is exhausted. Kept classifier-agnostic by design (no gradients).

**Key Features:**
- **Swappable decomposition** (`cfts/cf_codec/decompositions.py`), one per row of the "Choosing a Decomposition" heuristic table: `"emd"` (wraps `emd.sift.sift`, reproduces IMFACT's IMF baseline), `"wavelet"` (multi-level DWT via PyWavelets), `"fourier"` (trend + STFT frequency bands), `"stl"` (trend/seasonal/residual, ACF-estimated period, falls back to `"fourier"` when none is found), `"eigen"` (Singular Spectrum Analysis - trajectory-matrix SVD + diagonal averaging, applied per channel), `"shapelet"` (localized high-local-variance windows, masked to zero elsewhere), `"changepoint"` (piecewise-constant regime segmentation via `ruptures`), `"quantile"` (robust rolling-median trend + quantile-thresholded spike/noise split). Every strategy reconstructs exactly or near-exactly by construction.
- **Swappable reference selection** (`references.py`): `"nun"` (exact IMFACT nearest-unlike-neighbor) or `"composite"` (favored - per-component donor stitched from the `k` nearest candidates, so different components can come from different donor series)
- **Swappable matching** (`matching.py`): `"hungarian"` (favored - optimal cross-series component assignment via `scipy.optimize.linear_sum_assignment`, cost pluggable across dominant-frequency / energy / spectral-similarity) or `"index"` (naive positional fallback)
- **Swappable perturbation** (`perturbation.py`): `"replace"` (direct substitution, IMFACT baseline) or `"interpolate"` (gradual blend toward the donor component)
- **`CoDecPipeline`**: the search loop as a standalone, framework-agnostic object (any `predict_fn`, not just PyTorch) that `codec_cf` adapts to this repository's shared `<name>_cf` contract
- **Sparsity by components, not raw time points**: `CoDecResult.sparsity` counts substituted components, matching the workplan's explicit reviewer-driven metric correction

**Scope note:** this pass implements the algorithm (all eight decomposers, two reference selectors, two matchers, two perturbers, + search loop); the workplan's Phase 5/6 evaluation harness (full 128 UCR + 30 UEA archive runner, baseline-method wrappers, plausibility/robustness metrics) is intentionally not included - a smaller per-dataset ablation lives in `cfts/cf_codec/experiments/compare_ucr.py` instead. See the module docstring in `cfts/cf_codec/codec.py` for the full scope notes.

**Reference:**
```bibtex
@misc{schlegel2026codec,
  title={CoDec: Counterfactual Decomposition - A Modular Framework for Decomposition-Based Counterfactual Explanations of Time Series Classifiers},
  author={Schlegel, Udo},
  year={2026},
  note={Work in progress, extends IMFACT (schlegel2026imfact)}
}
```

**Usage Example:**
```python
from cfts.cf_codec.codec import codec_cf

cf, prediction = codec_cf(
    sample=sample,
    model=model,
    dataset=dataset,
    target_class=1,
    decomposition="emd",              # "emd" | "wavelet" | "fourier" | "stl" | "eigen" | "shapelet" | "changepoint" | "quantile"
    reference_selection="composite",   # "nun" | "composite"
    matching="hungarian",              # "hungarian" | "index"
    cost_fn="dominant_frequency",      # "dominant_frequency" | "energy" | "spectral_similarity"
    perturbation="replace",            # "replace" | "interpolate"
    k=5,
    max_iter=20,
    verbose=True,
)

# Full search trace (validity, sparsity-by-components, substituted indices, history)
cf, prediction, result = codec_cf(sample, model, dataset=dataset, target_class=1, return_result=True)
print(result.valid, result.sparsity, result.substituted_components)
```

---

#### 34. DiffCF - Generating Realistic Time-Series Counterfactuals via Diffusion-Guided Sampling (2026)
**Implementation:** `cfts/cf_diffcf/diffcf.py`

**Description:** Trains an unconditional denoising diffusion model (a 1-D UNet epsilon predictor) on the training distribution, then turns a query series into a counterfactual with an SDEdit-style guided reverse process: partially noise the query up to a chosen timestep instead of starting from pure noise, then DDIM-denoise it back to t=0 while nudging each step's denoised estimate with the gradient of a combined classification / proximity / smoothness objective. Retries with more noise and a stronger classification weight if the class doesn't flip.

**Key Features:**
- **Diffusion backbone**: a compact 1-D UNet (`UNet1D`) with sinusoidal time embeddings, trained with an MSE + total-variation denoising loss (`GaussianDiffusion`)
- **SDEdit-style partial noising**: starts the reverse process from a noised version of the real query (`start_ratio` of the schedule) rather than pure noise, so the sampler repairs rather than invents
- **Classifier guidance on x0**: at each DDIM step, nudges the denoised estimate with a weighted, unit-normalized combination of a classification gradient (log p(target|x0)), a proximity gradient (L1 to the original), and a smoothness gradient (second-derivative penalty)
- **Guidance stabilization**: optional Gaussian-blurred gradient (`grad_smooth`) or multi-augmentation-averaged gradient (`aug_avg`) to reduce noisy per-step updates
- **Retry loop**: on failure to flip the class, retries with a later `start_ratio` (more freedom to change) and a multiplicatively larger classification weight, up to `max_retries` times
- **Bring-your-own-backbone**: `diffusion_model=`/`diffusion=`/`norm_stats=` accept a pre-trained `UNet1D`/`GaussianDiffusion` (e.g. from `train_diffcf_diffusion`, or the official repo's own classes directly — their forward/constructor signatures match), so a backbone can be trained once and reused, or shared with the official implementation for direct comparison

**Reference:**
```bibtex
@misc{li2026diffcf,
  title={Generating Realistic Time-Series Counterfactuals via Diffusion-Guided Sampling},
  author={Li, Peiyu},
  year={2026},
  note={Accepted at ECML PKDD 2026. Author confirmed via the repository's LICENSE and commit history; a full author list and formal proceedings entry were not independently verifiable at time of writing.},
  howpublished={\url{https://github.com/Luckilyeee/DiffCF}}
}
```

**Links:**
- Repository: [https://github.com/Luckilyeee/DiffCF](https://github.com/Luckilyeee/DiffCF/tree/main)
- Comparison notebook: `cfts/cf_diffcf/diffcf_forda_comparison.ipynb` — trains one diffusion backbone with the official, unmodified `UNet1D`/`GaussianDiffusion` classes and feeds those exact weights into both the official `generate_counterfactual` and this repo's `diffcf_cf`, isolating the comparison to the sampling-loop port itself (plus a third run of `diffcf_cf` with its own default backbone, for typical usage). Also documents a one-line dead-import workaround needed to make the official repo importable at all (`src/cf/guidance.py` imports a `TemporalConeProjector` from a file that doesn't exist in the repo; it's never referenced anywhere, so the workaround doesn't affect what actually runs).

**Usage Example:**
```python
from cfts.cf_diffcf.diffcf import diffcf_cf, train_diffcf_diffusion

# Simplest path: trains its own (small, fast) diffusion model on `dataset`
cf, scores = diffcf_cf(
    sample=sample,
    model=model,
    dataset=dataset,
    diffusion_epochs=300,
    ddim_steps=100,
    max_retries=3,
    verbose=True,
)

# Pre-train once and reuse the backbone across many samples
unet, diffusion, norm_stats = train_diffcf_diffusion(dataset, epochs=500)
cf, scores = diffcf_cf(
    sample=sample, model=model, target_class=1,
    diffusion_model=unet, diffusion=diffusion, norm_stats=norm_stats,
)
```

---

## Evaluation Metrics References

### Keane et al. (2021) Metrics Framework
**Implementation:** `cfts/metrics/keane.py`

**Description:** Comprehensive metrics for evaluating counterfactual quality including validity, proximity, and compactness.

**Reference:**
```bibtex
@article{keane2021good,
  title={If only we had better counterfactual explanations: Five key deficits to rectify in the evaluation of counterfactual XAI techniques},
  author={Keane, Mark T and Kenny, Eoin M and Delaney, Eoin and Smyth, Barry},
  booktitle={Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21)},
  pages={4466--4474},
  year={2021}
}
```

### Distance Metrics
**Implementations:** `cfts/metrics/proximity.py`

- **L2 Distance (Euclidean):** Standard Euclidean distance measure
- **L1 Distance (Manhattan):** Sum of absolute differences
- **DTW (Dynamic Time Warping):** Temporal alignment-aware distance
- **Fréchet Distance:** Similarity measure that considers ordering

**DTW Reference:**
```bibtex
@article{berndt1994using,
  title={Using dynamic time warping to find patterns in time series},
  author={Berndt, Donald J and Clifford, James},
  booktitle={KDD Workshop},
  volume={10},
  number={16},
  pages={359--370},
  year={1994}
}
```

### Sparsity Metrics
**Implementation:** `cfts/metrics/sparsity.py`

**Reference:**
```bibtex
@article{laugel2019dangers,
  title={Dangers of post-hoc interpretability: Unjustified counterfactual explanations},
  author={Laugel, Thibault and Lesot, Marie-Jeanne and Marsala, Christophe and Renard, Xavier and Detyniecki, Marcin},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)},
  pages={2801--2807},
  year={2019}
}
```

### Diversity Metrics
**Implementation:** `cfts/metrics/diversity.py`

**Reference:**
```bibtex
@article{mothilal2020explaining,
  title={Explaining machine learning classifiers through diverse counterfactual explanations},
  author={Mothilal, Ramaravind K and Sharma, Amit and Tan, Chenhao},
  booktitle={Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency},
  pages={607--617},
  year={2020}
}
```

---

## Related Surveys and Reviews

### Counterfactual Explanations - General

```bibtex
@article{guidotti2022counterfactual,
  title={Counterfactual explanations and how to find them: literature review and benchmarking},
  author={Guidotti, Riccardo},
  journal={Data Mining and Knowledge Discovery},
  pages={1--55},
  year={2022},
  publisher={Springer}
}
```

```bibtex
@article{verma2020counterfactual,
  title={Counterfactual explanations for machine learning: A review},
  author={Verma, Sahil and Dickerson, John and Hines, Keegan},
  journal={arXiv preprint arXiv:2010.10596},
  year={2020}
}
```

### Interpretable Machine Learning

```bibtex
@book{molnar2020interpretable,
  title={Interpretable machine learning: A guide for making black box models explainable},
  author={Molnar, Christoph},
  year={2020},
  url={https://christophm.github.io/interpretable-ml-book/}
}
```

### Time Series Classification

```bibtex
@article{fawaz2019deep,
  title={Deep learning for time series classification: a review},
  author={Fawaz, Hassan Ismail and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal={Data Mining and Knowledge Discovery},
  volume={33},
  number={4},
  pages={917--963},
  year={2019},
  publisher={Springer}
}
```

### Explainable AI for Time Series

```bibtex
@article{theissler2022explainable,
  title={Explainable AI for time series classification: A review, taxonomy and research directions},
  author={Theissler, Andreas and Spinnato, Francesco and Schlegel, Udo and Guidotti, Riccardo},
  journal={IEEE Access},
  volume={10},
  pages={100700--100747},
  year={2022},
  publisher={IEEE}
}
```

```bibtex
@article{schlegel2021towards,
  title={Towards a rigorous evaluation of XAI methods on time series},
  author={Schlegel, Udo and Arnout, Hiba and El-Assady, Mennatallah and Oelke, Daniela and Keim, Daniel A},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={4197--4201},
  year={2019},
  organization={IEEE}
}
```

---

## Citation

If you use this library in your research, please cite one of the following sources (the second one is preferred):

```bibtex
@software{cfts-us-2025,
  author = {Schlegel, Udo},
  title = {Counterfactual Explanation Algorithms for Time Series Models},
  url = {https://github.com/visual-xai-for-time-series/counterfactual-explanations-for-time-series},
  year = {2025}
}
```

```bibtex
@inproceedings{schlegel_what-if_2026,
  title={What-If Explanations Over Time: Counterfactuals for Time Series Classification},
  author={Schlegel, Udo and Seidl, Thomas},
  booktitle={World Conference on Explainable Artificial Intelligence (XAI)},
  year={2026}
}
```

---

## Notes

- **Custom Implementations:** Some methods marked as "Custom" are implementations inspired by general counterfactual techniques adapted specifically for time series data, or are composite approaches combining multiple established techniques.
- **Repository Links:** Where available, links to original repositories and papers are provided for reference.
- **Method Categories:** Methods are organized by their primary approach, though some may employ multiple techniques.
- **Ongoing Research:** This field is rapidly evolving. Check the original papers and repositories for the most recent developments.

---

## Additional Resources

- **UCR Time Series Archive:** [https://www.cs.ucr.edu/~eamonn/time_series_data_2018/](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
- **Counterfactual Explanation Tutorial:** [https://christophm.github.io/interpretable-ml-book/counterfactual.html](https://christophm.github.io/interpretable-ml-book/counterfactual.html)
- **Time Series Classification Website:** [https://timeseriesclassification.com/](https://timeseriesclassification.com/)

---

**Last Updated:** August 2026
