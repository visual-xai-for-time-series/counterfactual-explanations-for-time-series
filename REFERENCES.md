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

```python
import numpy as np
import torch
from cfts.cf_wachter import wachter_genetic_cf
from cfts.cf_native_guide import native_guide_uni_cf
from cfts.cf_comte import comte_generate
from cfts.cf_tsevo import tsevo_cf
from cfts.metrics import l2_distance, prediction_change

# Load your model and data
# model = ... (trained PyTorch model)
# sample = ... (time series to explain)
# dataset = ... (dataset object)

# Generate counterfactuals using different methods
cf_wachter, pred_wachter = wachter_genetic_cf(
    sample, model, step_size=0.1, max_iterations=1000
)

cf_native, pred_native = native_guide_uni_cf(
    sample, dataset, model, k=5
)

cf_comte, pred_comte = comte_generate(
    sample, model, target_class=1, lambda_val=0.1
)

cf_tsevo, pred_tsevo = tsevo_cf(
    sample, dataset, model, target_class=1, 
    pop_size=50, n_generations=100
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
from cfts.cf_wachter import wachter_genetic_cf, wachter_gradient_cf

# Genetic algorithm variant
cf, prediction = wachter_genetic_cf(
    sample=sample,
    model=model,
    step_size=0.1,
    max_iterations=1000,
    target_class=1
)

# Gradient-based variant
cf, prediction = wachter_gradient_cf(
    sample=sample,
    model=model,
    learning_rate=0.01,
    max_iterations=500
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
from cfts.cf_comte import comte_generate

cf, prediction = comte_generate(
    sample=sample,
    model=model,
    target_class=1,
    lambda_val=0.1,
    max_iterations=1000,
    learning_rate=0.01
)
```

---

#### 3. GLACIER - Guided Locally Constrained Counterfactuals (2024)
**Implementation:** `cfts/cf_glacier/glacier.py`

**Description:** Advanced counterfactual generation with enhanced realism constraints, similarity preservation, and robust optimization for complex time series patterns.

**Key Features:**
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
from cfts.cf_glacier import glacier_cf

cf, prediction = glacier_cf(
    sample=sample,
    dataset=dataset,
    model=model,
    target_class=1,
    lambda_l1=0.01,
    lambda_l2=0.1,
    max_iterations=2000
)
```

---

#### 4. TSCF - Time Series CounterFactuals (Custom)
**Implementation:** `cfts/cf_tscf/tscf.py`

**Description:** Gradient-based optimization with temporal smoothness constraints for generating realistic counterfactual explanations.

**Note:** This is a custom implementation combining standard counterfactual generation techniques with time series-specific regularization.

**Usage Example:**
```python
from cfts.cf_tscf import tscf_cf

cf, prediction = tscf_cf(
    sample=sample,
    dataset=dataset,
    model=model,
    target_class=1,
    lambda_l1=0.01,
    lambda_l2=0.01,
    lambda_smooth=0.001,
    learning_rate=0.1,
    max_iterations=2000
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
from cfts.cf_dandl import dandl_generate

# Returns multiple Pareto-optimal counterfactuals
counterfactuals = dandl_generate(
    sample=sample,
    model=model,
    target_class=1,
    pop_size=100,
    n_generations=200,
    mutation_rate=0.1
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
    dataset=dataset,
    model=model,
    target_class=1,
    pop_size=50,
    n_generations=100,
    crossover_prob=0.7,
    mutation_prob=0.3
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
from cfts.cf_multispace import multispace_cf

cf, prediction = multispace_cf(
    sample=sample,
    dataset=dataset,
    model=model,
    target_class=1,
    pop_size=100,
    n_generations=150,
    use_saliency=True
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
from cfts.cf_subspace import subspace_cf

cf, prediction = subspace_cf(
    sample=sample,
    dataset=dataset,
    model=model,
    target_class=1,
    pop_size=100,
    n_generations=150,
    window_sizes=[5, 10, 20]
)
```

---

### Instance-Based Methods

#### 9. Native Guide (2021)
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
from cfts.cf_native_guide import native_guide_uni_cf, native_guide_multi_cf

# Univariate time series
cf, prediction = native_guide_uni_cf(
    sample=sample,
    dataset=dataset,
    model=model,
    k=5,  # number of nearest neighbors
    target_class=1
)

# Multivariate time series
cf, prediction = native_guide_multi_cf(
    sample=sample,
    dataset=dataset,
    model=model,
    k=5
)
```

---

#### 10. CELS & M-CELS - Counterfactual Explanations via Learned Saliency (2023-2024)
**Implementation:** `cfts/cf_cels/cels.py`

**Description:** Learns saliency maps to identify important time steps and generates counterfactuals through nearest unlike neighbor replacement. Supports both univariate (CELS) and multivariate (M-CELS) time series with automatic selection via AutoCELS.

**Key Features:**
- **Learned saliency maps**: Identifies important time steps contributing to predictions
- **Nearest unlike neighbor (NUN)**: Finds target class instances for replacement
- **Optimization-based learning**: Balances validity, sparsity, and temporal coherence
- **AutoCELS wrapper**: Automatically selects between CELS/M-CELS based on dimensionality
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
from cfts.cf_cels import cels_generate, mcels_generate

# Univariate CELS
cf, prediction = cels_generate(
    sample=sample,
    model=model,
    X_train=X_train,
    y_train=y_train,
    target=1,
    learning_rate=0.01,
    max_iter=100,
    lambda_valid=1.0,
    lambda_budget=0.1,
    lambda_tv=0.1
)

# Multivariate M-CELS
cf, prediction = mcels_generate(
    sample=sample,
    model=model,
    X_train=X_train,
    y_train=y_train,
    target=1,
    learning_rate=0.01,
    max_iter=100,
    lambda_valid=1.0,
    lambda_sparsity=0.1,
    lambda_smoothness=0.1
)

# AutoCELS (automatic selection)
from cfts.cf_cels import AutoCELS

explainer = AutoCELS(model)
cf = explainer.generate(sample, target_class, X_train, y_train)
```

---

#### 11. AB-CF - Attention-Based Counterfactual Explanation (2023)
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
from cfts.cf_ab_cf import ab_cf_generate

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

### Latent Space Methods

#### 12. CGM - Conditional Generative Models for Counterfactuals (2021)
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
from cfts.cf_cgm import cgm_generate

# Train conditional VAE on dataset
from cfts.cf_cgm import ConditionalVAE, train_conditional_vae

cvae = train_conditional_vae(
    X_train=X_train,
    y_train=y_train,
    input_dim=input_dim,
    num_classes=num_classes,
    latent_dim=16,
    epochs=50
)

# Generate counterfactual
cf, prediction = cgm_generate(
    sample=sample,
    model=model,
    conditional_vae=cvae,
    target_class=1,
    learning_rate=0.01,
    max_iterations=500,
    lambda_proximity=0.1,
    lambda_sparsity=0.01
)
```

---

#### 13. CounTS - Counterfactual Time Series (2023)
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
from cfts.cf_counts import counts_generate, CounTS

# Train CounTS model
counts_model = CounTS(
    input_dim=input_dim,
    hidden_dim=64,
    latent_dim=16,
    num_classes=num_classes
)

# Train on dataset
train_counts_model(counts_model, X_train, y_train, epochs=50)

# Generate counterfactual via intervention
cf, prediction = counts_generate(
    sample=sample,
    counts_model=counts_model,
    target_class=1,
    intervention_type='latent',  # 'latent' or 'time'
    intervention_strength=0.5,
    max_iterations=100
)
```

---

#### 14. Latent-CF - Latent Space Counterfactuals (2020)
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
from cfts.cf_latent_cf import latent_cf_generate

cf, prediction = latent_cf_generate(
    sample=sample,
    model=model,
    latent_dim=16,
    target_class=1,
    learning_rate=0.01,
    max_iterations=1000,
    pretrained_autoencoder=None  # or provide pre-trained model
)
```

---

#### 15. LASTS - Local Agnostic Subsequence-based Time Series Explainer (2020)
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
from cfts.cf_lasts import lasts_cf, LASTS

# Simple counterfactual generation
cf, prediction = lasts_cf(
    sample=sample,
    dataset=dataset,
    model=model,
    target_class=1,
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

#### 16. SETS - Shapelet-Based Counterfactual Explanations (2022)
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
from cfts.cf_sets import sets_cf, sets_explain

# Basic counterfactual generation
cf, prediction = sets_cf(
    sample=ts_sample,
    dataset=train_dataset,
    model=trained_model,
    target_class=1,
    n_shapelets_per_class=5,
    shapelet_lengths=[5, 10, 20],
    threshold=0.5,
    verbose=True
)

# Detailed explanation with shapelet information
explanation = sets_explain(
    sample=ts_sample,
    dataset=train_dataset,
    model=trained_model,
    target_class=1
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

#### 17. SG-CF - Shapelet-Guided Counterfactual Explanations (2022)
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

```

**Usage Example:**
```python
from cfts.cf_sg_cf import sg_cf, sg_cf_explain

# Basic counterfactual generation
cf, prediction = sg_cf(
    sample=ts_sample,
    dataset=train_dataset,
    model=trained_model,
    target_class=1,
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
    dataset=train_dataset,
    model=trained_model,
    target_class=1,
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

#### 18. DisCOX - Discord-based Counterfactual Explanations (2024)
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
    dataset=train_dataset,
    model=trained_model,
    target_class=1,
    window_size=20,  # or None for automatic (10% of series length)
    k_discords=3,
    modification_strategy='prototype',  # 'prototype', 'amplify', 'invert'
    blend_factor=0.3,
    verbose=True
)

# Detailed explanation with discord information
explanation = discox_explain(
    sample=ts_sample,
    dataset=train_dataset,
    model=trained_model,
    target_class=1,
    window_size=20,
    k_discords=5,
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

#### 19. CFWoT - Counterfactual Explanations Without Training Datasets (2024)
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
from cfts import cfwot

cf, prediction = cfwot(
    sample=sample,
    model=model,
    target_class=1,
    K=100,  # number of time steps
    D=5,    # number of features
    D_C=5,  # number of continuous features
    D_D=0,  # number of discrete features
    num_episodes=100,
    learning_rate=0.001,
    gamma=0.99,  # discount factor
    feasibility_weights=None,  # optional feature weights
    verbose=True
)
```

---

### Frequency-Domain Methods

#### 20. FFT-CF - Fourier Transform Counterfactual Explanations
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
    dataset=dataset,
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
    dataset=dataset,
    model=model,
    target_class=1,
    learning_rate=0.01,
    lambda_proximity=0.1,
    lambda_smoothness=0.05,
    max_iterations=500
)
```

---

### Hybrid Methods

#### 21. SPARCE - Generating SPARse Counterfactual Explanations (2022)
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
from cfts.cf_sparce import sparce_generate, train_sparce

# Train SPARCE GAN on dataset
generator, discriminator = train_sparce(
    X_train=X_train,
    y_train=y_train,
    input_dim=input_dim,
    hidden_dim=64,
    layer_dim=2,
    epochs=100,
    batch_size=32
)

# Generate sparse counterfactual
cf, prediction = sparce_generate(
    sample=sample,
    model=model,
    generator=generator,
    discriminator=discriminator,
    target_class=1,
    lambda_adv=0.1,
    lambda_class=1.0,
    lambda_sim=0.5,
    lambda_sparse=0.1,
    lambda_jerk=0.05,
    max_iterations=500
)
```

---

#### 22. Time-CF - Shapelet-based Model-agnostic Counterfactual Local Explanations
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
from cfts.cf_time_cf import time_cf_generate

cf, prediction = time_cf_generate(
    sample=sample,
    dataset=dataset,
    model=model,
    target_class=1,
    n_shapelets=10,
    M=32,
    timegan_epochs=100
)
```

---

#### 23. TeRCE - Temporal Rule-Based Counterfactual Explanations (2022)
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
from cfts.cf_terce import terce_generate

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

#### 24. MG-CF - Motif-Guided Counterfactual Explanations
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
    dataset=dataset,
    model=model,
    target_class=1,
    n_shapelets=100,
    lengths_ratio=[0.3, 0.5, 0.7]
)
```

---

#### 25. TimeX - Encoding Time-Series Explanations (2023)
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
from cfts.cf_timex import timex_explanation

# Note: Requires pre-trained TimeX model
saliency, prediction = timex_explanation(
    sample=sample,
    model=model,
    timex_model=pretrained_timex_model,
    return_saliency=True
)
```

---

#### 26. TimeX++ - Learning Time-Series Explanations with Information Bottleneck (2024)
**Implementation:** `cfts/cf_timex_plus_plus/timex_plus_plus.py`

**Description:** Improved time series explainer based on information bottleneck principle that generates in-distributed and label-preserving explanation instances. Addresses distribution shift and signaling issues in applying IB to time series explainability.

**Key Features:**
- **Information bottleneck framework**: Modified IB objective for time series explainability
- **Explanation extractor**: Transformer encoder-decoder architecture producing stochastic masks
- **Explanation conditioner**: MLP-based network generating in-distribution instances
- **Label consistency**: Preserves predictions through JS divergence minimization
- **Avoids OOD issues**: Generates explanation-embedded instances within original distribution
- **Theoretical foundation**: Addresses signaling problem and compactness issues in IB

**Algorithm:**
1. **Explanation Extraction**: Transformer-based encoder-decoder maps input X to stochastic mask π
2. **Compactness Quantifier**: Minimizes KL divergence D_KL(P(M|X)||Q(M)) + connective loss
3. **Reference Instance**: Generate X̃_r via Gaussian padding on masked input
4. **Conditioner**: MLP Ψ_θ maps [M, X] to explanation-embedded instance X̃
5. **Informativeness**: Minimize label consistency loss (JS divergence) + distribution shift loss (KL) + reference distance loss
6. **Overall Objective**: ℒ = ℒ_LC + α·ℒ_M + β·(ℒ_KL + ℒ_dr)

**Reference:**
```bibtex
@inproceedings{liu2024timexplusplus,
  title={TimeX++: Learning Time-Series Explanations with Information Bottleneck},
  author={Liu, Zichuan and Wang, Tianchun and Shi, Jimeng and Xu, Zheng and Chen, Zhuomin and Song, Lei and Dong, Wenqian and Obeysekera, Jayantha and Shirani, Farhad and Luo, Dongsheng},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```

**Links:**
- Paper: [arXiv:2405.09308](https://arxiv.org/abs/2405.09308)
- HTML Paper: [arXiv HTML](https://arxiv.org/html/2405.09308v1)
- Repository: [https://github.com/zichuan-liu/TimeXplusplus](https://github.com/zichuan-liu/TimeXplusplus)

**Usage Example:**
```python
from cfts.cf_timex_plus_plus import timexplusplus_explanation

# Note: Requires training TimeX++ explanation extractor and conditioner
saliency_mask, embedded_instance, prediction = timexplusplus_explanation(
    sample=sample,
    model=model,
    dataset=training_data,
    alpha=2.0,  # Compactness weight
    beta=1.0,   # Distribution consistency weight
    r=0.5,      # Mask sparsity parameter
    epochs=50
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

If you use this library in your research, please cite:

```bibtex
@software{cfts-us-2025,
  author = {Schlegel, Udo},
  title = {Counterfactual Explanation Algorithms for Time Series Models},
  url = {https://github.com/visual-xai-for-time-series/counterfactual-explanations-for-time-series},
  year = {2025}
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

**Last Updated:** January 2025
