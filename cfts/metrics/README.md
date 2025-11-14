# Counterfactual Metrics for Time Series

This directory contains a comprehensive collection of metrics for evaluating counterfactual explanations for time series data. The metrics are organized into six main categories that cover different aspects of counterfactual quality.

## Metric Categories

### 1. Validity Metrics (`validity.py`)
Evaluate whether counterfactuals achieve the desired prediction changes:

- **`prediction_change`**: Measures if the counterfactual changes the model's prediction to the target class
- **`class_probability_confidence`**: Evaluates the model's confidence in the counterfactual prediction  
- **`decision_boundary_distance`**: Measures how far the counterfactual crosses the decision boundary

### 2. Proximity Metrics (`proximity.py`)
Measure similarity between original and counterfactual time series:

- **`l2_distance`**: Euclidean distance between time series
- **`manhattan_distance`**: L1/Manhattan distance between time series
- **`dtw_distance`**: Dynamic Time Warping distance (requires `dtaidistance` package)
- **`frechet_distance`**: Discrete Fréchet distance considering temporal ordering
- **`normalized_distance`**: Distance normalized by the time series range
- **`mahalanobis_distance`**: Mahalanobis distance accounting for feature correlations and scales

### 3. Sparsity Metrics (`sparsity.py`) 
Evaluate how minimal the changes are:

- **`l0_norm`**: Number of changed time points
- **`percentage_changed_points`**: Fraction of modified timestamps
- **`segment_based_sparsity`**: Number of continuous segments modified
- **`feature_sparsity`**: Sparsity at the feature level (for multivariate series)
- **`temporal_sparsity_profile`**: Binary profile showing when changes occur
- **`gini_sparsity_coefficient`**: Gini coefficient of change magnitudes

### 4. Realism Metrics (`realism.py`)
Assess whether counterfactuals are realistic and plausible:

- **`domain_constraint_violations`**: Counts violations of domain-specific constraints
- **`statistical_similarity`**: Statistical similarity to original data distribution
- **`temporal_consistency`**: Measures smooth transitions and realistic patterns
- **`feature_range_validity`**: Checks if values stay within realistic ranges
- **`autocorrelation_preservation`**: Evaluates preservation of temporal dependencies
- **`spectral_similarity`**: Compares frequency domain characteristics

### 5. Diversity Metrics (`diversity.py`)
Evaluate diversity when multiple counterfactuals are generated:

- **`pairwise_distance`**: Average distance between generated counterfactuals
- **`coverage_metric`**: How well counterfactuals span the feature space
- **`novelty_metric`**: Uniqueness compared to training data
- **`diversity_index`**: Shannon diversity index of counterfactuals
- **`intra_cluster_distance`**: Diversity relative to the original instance
- **`feature_diversity`**: Diversity for each feature/time point separately

### 6. Stability Metrics (`stability.py`)
Assess robustness and consistency of counterfactual generation:

- **`algorithmic_stability`**: Consistency across multiple algorithm runs
- **`input_perturbation_robustness`**: Sensitivity to small input changes
- **`model_robustness`**: Performance across different model architectures
- **`hyperparameter_sensitivity`**: Sensitivity to hyperparameter changes
- **`convergence_stability`**: Stability of optimization convergence

## Usage

### Basic Usage

```python
from cfts.metrics import l2_distance, prediction_change, temporal_consistency

# Calculate individual metrics
distance = l2_distance(original_ts, counterfactual_ts)
validity = prediction_change(original_ts, counterfactual_ts, model, target_class=1)
consistency = temporal_consistency(counterfactual_ts)
```

### Using Mahalanobis Distance

```python
from cfts.metrics import mahalanobis_distance
import numpy as np

# Without reference data (equivalent to Euclidean distance)
dist = mahalanobis_distance(original_ts, counterfactual_ts)

# With reference data (accounts for feature correlations and scales)
reference_data = np.array([...])  # Your training/reference data
dist = mahalanobis_distance(original_ts, counterfactual_ts, reference_data)
```

### Comprehensive Evaluation

```python
from cfts.metrics import CounterfactualEvaluator

# Initialize evaluator with reference data
evaluator = CounterfactualEvaluator(reference_data=training_data)

# Evaluate a single counterfactual
results = evaluator.evaluate_single(
    original_ts=original_ts,
    counterfactual_ts=counterfactual_ts, 
    model=trained_model,
    target_class=1
)

# Evaluate multiple counterfactuals for diversity
diversity_results = evaluator.evaluate_multiple(
    original_ts=original_ts,
    counterfactuals=[cf1, cf2, cf3],
    model=trained_model,
    target_class=1
)
```

### Algorithm Benchmarking

```python
from cfts.metrics import benchmark_algorithms

# Define test cases
test_cases = [(ts1, target1), (ts2, target2), ...]

# Benchmark multiple algorithms
results = benchmark_algorithms(
    algorithms={
        'Algorithm_A': algorithm_a_function,
        'Algorithm_B': algorithm_b_function
    },
    test_cases=test_cases,
    model=trained_model,
    reference_data=training_data
)
```

## Metric Suites

Pre-defined metric suites for different evaluation scenarios:

```python
from cfts.metrics import create_metric_suite

# Different evaluation scenarios
basic_metrics = create_metric_suite('basic')          # Essential metrics
comprehensive = create_metric_suite('comprehensive')  # All metrics
research_suite = create_metric_suite('research')      # Research-focused
production_suite = create_metric_suite('production')  # Production-ready
```

## Dependencies

Required packages:
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`

Optional packages:
- `dtaidistance` (for DTW distance)

## Installation

The metrics are part of the `cfts` package. Install dependencies:

```bash
pip install numpy scipy scikit-learn pandas
# Optional: pip install dtaidistance
```

## Examples

See `example_usage.py` for a complete demonstration of all metrics with synthetic data.

## Metric Interpretation

### Validity Metrics
- **Higher is better**: `prediction_change`, `class_probability_confidence`
- **Contextual**: `decision_boundary_distance` (depends on application)

### Proximity Metrics  
- **Lower is better**: All distance metrics (closer to original is preferred)

### Sparsity Metrics
- **Lower is better**: All sparsity metrics (fewer changes is preferred)

### Realism Metrics
- **Higher is better**: `temporal_consistency`, `feature_range_validity`, `autocorrelation_preservation`
- **Contextual**: `statistical_similarity` (depends on test used)
- **Lower is better**: `domain_constraint_violations`

### Diversity Metrics
- **Higher is better**: All diversity metrics (more diverse counterfactuals preferred)

### Stability Metrics
- **Higher is better**: All stability metrics (more robust algorithms preferred)
