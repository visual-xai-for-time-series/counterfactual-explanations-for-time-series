# Counterfactual Explanation Algorithms for Time Series Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive collection of counterfactual explanation algorithms for time series classification with PyTorch implementations. This library provides state-of-the-art methods for generating and evaluating counterfactual explanations, helping to understand and interpret deep learning models for time series data.

## Table of Contents

- [Quick Start](#quick-start)
- [Implemented Algorithms](#implemented-algorithms)
- [Comprehensive Evaluation Metrics](#comprehensive-evaluation-metrics)
- [Examples and Datasets](#examples-and-datasets)
- [Visualization Examples](#visualization-examples)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [License](#license)
- [References and Citations](#references-and-citations)
- [Disclaimer](#disclaimer)
- [Acknowledgments](#acknowledgments)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all examples and evaluations
cd examples
python run_all.py

# Run individual examples
python example_forda.py          # FordA dataset
python example_vibration.py      # Synthetic vibration data
python example_arabicdigits.py   # Multi-channel Arabic digits
python metrics_evaluation_example.py  # Comprehensive metrics
```

## Implemented Algorithms

### 1. **Wachter et al.** (`cf_wachter`)
Classic counterfactual explanation method with two optimization variants.

**Key Features:**
- **Gradient-based optimization**: Uses model gradients for efficient counterfactual generation
- **Genetic algorithm variant**: Evolutionary approach for complex search spaces
- **Proximity-focused**: Minimizes distance to original while achieving target prediction

**Implementation:** `cfts/cf_wachter/`  
**Reference:** [Wachter et al. (2017)](https://christophm.github.io/interpretable-ml-book/counterfactual.html)
```bibtex
@article{wachter2017counterfactual,
  title={Counterfactual explanations without opening the black box: Automated decisions and the GDPR},
  author={Wachter, Sandra and Mittelstadt, Brent and Russell, Chris},
  journal={Harvard Journal of Law \& Technology},
  volume={31}, pages={841}, year={2017}
}
```

### 2. **Native Guide** (`cf_native_guide`)
Instance-based counterfactual generation using nearest neighbor search and gradient attribution.

**Key Features:**
- **Instance-based approach**: Leverages similar examples from training data
- **Gradient attribution**: Uses Captum's GradientShap for feature importance
- **Temporal awareness**: Preserves important temporal patterns

**Implementation:** `cfts/cf_native_guide/`  
**Reference:** [Delaney et al. (2021)](https://github.com/e-delaney/Instance-Based_CFE_TSC)
```bibtex
@inproceedings{delaney2021instance,
  title={Instance-based counterfactual explanations for time series classification},
  author={Delaney, Eoin and Greene, Derek and Keane, Mark T},
  booktitle={International Conference on Case-Based Reasoning},
  pages={32--47}, year={2021}, publisher={Springer}
}
```

### 3. **COMTE** (`cf_comte`)
Counterfactual explanations specifically designed for multivariate time series.

**Key Features:**
- **Multivariate support**: Handles multi-channel time series effectively
- **Feature-wise optimization**: Optimizes each channel independently when beneficial
- **Regularization**: Incorporates smoothness and sparsity constraints

**Implementation:** `cfts/cf_comte/`  
**Reference:** [Ates et al. (2021)](https://github.com/peaclab/CoMTE)
```bibtex
@inproceedings{ates2021counterfactual,
  title={Counterfactual Explanations for Multivariate Time Series},
  author={Ates, Emre and Aksar, Burak and Leung, Vitus J and Coskun, Ayse K},
  booktitle={2021 International Conference on Applied Artificial Intelligence (ICAPAI)},
  pages={1--8}, year={2021}, organization={IEEE}
}
```

### 4. **SETS** (`cf_sets`)
Scalable Explanation for Time Series using segment-based modifications.

**Key Features:**
- **Segment-based approach**: Modifies coherent temporal segments
- **Interpretability focus**: Generates more human-interpretable explanations
- **Scalable design**: Efficient for long time series

**Implementation:** `cfts/cf_sets/`

### 5. **MOC - Multi-Objective Counterfactuals** (`cf_dandl`)
Multi-objective evolutionary approach finding Pareto-optimal counterfactuals.

**Key Features:**
- **Pareto optimization**: Balances multiple objectives (validity, proximity, sparsity)
- **Evolutionary algorithm**: Uses genetic operations for diverse solutions
- **Multiple solutions**: Returns a set of counterfactuals on the Pareto frontier

**Implementation:** `cfts/cf_dandl/`
**Reference:** [Dandl et al. (2021)](https://arxiv.org/abs/2004.11165)
```bibtex
@article{dandl2020multi,
  title={Multi-objective counterfactual explanations},
  author={Dandl, Susanne and Molnar, Christoph and Binder, Martin and Bischl, Bernd},
  journal={arXiv preprint arXiv:2004.11165},
  year={2020}
}
```

### 6. **GLACIER** (`cf_glacier`)
Advanced counterfactual generation with enhanced realism constraints.

**Key Features:**
- **Realism focus**: Incorporates domain-specific constraints
- **Similarity preservation**: Maintains statistical properties of original data
- **Robust optimization**: Handles noisy and complex time series patterns

**Implementation:** `cfts/cf_glacier/`
**Reference:** [Wang et al. (2021)](https://github.com/zhendong3wang/learning-time-series-counterfactuals)
```bibtex
@article{glacier2024,
  title={Glacier: guided locally constrained counterfactual explanations for time series classification},
  author = {Wang, Zhendong and Samsten, Isak and Miliou, Ioanna and Mochaourab, Rami and Papapetrou, Panagiotis},
  journal={Machine Learning},
  year={2024}
}
```

### 7. **Multi-SpaCE** (`cf_multispace`)
Multi-objective counterfactual generation with subsequence-based optimization.

**Key Features:**
- **Feature importance**: Uses attribution methods for guided initialization
- **Subsequence optimization**: Modifies meaningful temporal segments
- **Multi-objective fitness**: Balances validity, sparsity, and plausibility
- **Evolutionary approach**: Population-based search for diverse solutions

**Implementation:** `cfts/cf_multispace/`
**Reference:** [Refoyo and Luengo (2025)](https://github.com/MarioRefoyo/Multi-SpaCE)
```bibtex
@article{refoyo2024multi,
  title={Multi-SpaCE: Multi-Objective Subsequence-based Sparse Counterfactual Explanations for Multivariate Time Series Classification},
  author={Refoyo, Mario and Luengo, David},
  journal={arXiv preprint arXiv:2501.04009},
  year={2024}
}
```

### 8. **TSEvo** (`cf_tsevo`)
Evolutionary counterfactual explanations using NSGA-II multi-objective optimization.

**Key Features:**
- **NSGA-II algorithm**: Industry-standard multi-objective evolutionary optimizer
- **Pareto optimization**: Simultaneously optimizes validity, proximity, and sparsity
- **Reference set mutation**: Leverages target class examples for realistic counterfactuals
- **Multiple operators**: Crossover, Gaussian mutation, and segment-based swapping

**Implementation:** `cfts/cf_tsevo/`
**Reference:** [Höllig et al. (2022)](https://github.com/fzi-forschungszentrum-informatik/TSInterpret)
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

## Comprehensive Evaluation Metrics

The library includes a complete suite of metrics for evaluating counterfactual quality across six key dimensions:

### **Validity Metrics** (`cfts/metrics/validity.py`)
- `prediction_change`: Verifies target class prediction is achieved
- `class_probability_confidence`: Measures prediction confidence
- `decision_boundary_distance`: Distance from decision boundary

### **Proximity Metrics** (`cfts/metrics/proximity.py`)
- `l2_distance`: Euclidean distance between time series
- `manhattan_distance`: L1 distance measure
- `dtw_distance`: Dynamic Time Warping distance
- `frechet_distance`: Temporal ordering-aware distance
- `normalized_distance`: Scale-invariant distance

### **Sparsity Metrics** (`cfts/metrics/sparsity.py`)
- `l0_norm`: Number of modified time points
- `percentage_changed_points`: Fraction of changes
- `segment_based_sparsity`: Continuous segment modifications
- `gini_sparsity_coefficient`: Distribution of change magnitudes

### **Realism Metrics** (`cfts/metrics/realism.py`)
- `domain_constraint_violations`: Domain-specific rule violations
- `statistical_similarity`: Distribution similarity to original data
- `temporal_consistency`: Temporal pattern preservation
- `autocorrelation_preservation`: Time dependency maintenance
- `spectral_similarity`: Frequency domain characteristics

### **Diversity Metrics** (`cfts/metrics/diversity.py`)
- `pairwise_distance`: Diversity between multiple counterfactuals
- `coverage_metric`: Feature space coverage
- `novelty_metric`: Uniqueness compared to training data
- `diversity_index`: Shannon diversity index

### **Stability Metrics** (`cfts/metrics/stability.py`)
- `algorithmic_stability`: Consistency across runs
- `input_stability`: Robustness to input perturbations
- `hyperparameter_sensitivity`: Parameter stability analysis

## Examples and Datasets

### Available Examples
- **`example_forda.py`**: FordA automotive dataset (UCR Archive)
- **`example_vibration.py`**: Synthetic vibration fault detection
- **`example_arabicdigits.py`**: Multi-channel spoken Arabic digits
- **`metrics_evaluation_example.py`**: Comprehensive metrics demonstration

### Supported Datasets
- **UCR Time Series Archive**: Automatic download and preprocessing
- **Synthetic Data**: Built-in generators for controlled experiments
- **Custom Datasets**: Easy integration with custom time series data

### Pre-trained Models
- `simple_cnn_2.pth`: Binary classification (FordA)
- `cnn_multi_arabicdigits_10ch.pth`: Multi-class, multi-channel
- `tiny1dcnn_vibration_6.pth`: Vibration fault classification

## Visualization Examples

The library generates publication-ready visualizations:

![FordA Counterfactual Explanations](counterfactuals_forda.png)
*Individual counterfactuals and overlay comparisons for FordA dataset*

![Arabic Digits Counterfactual Explanations](counterfactuals_arabic_digits.png)
*Multi-channel counterfactual analysis for Arabic digits*

![Vibration Counterfactual Explanations](counterfactuals_vibration.png)
*Synthetic vibration data counterfactual explanations*

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib
- Scikit-learn
- Captum (for attribution methods)
- Optional: dtaidistance (for DTW distance metric)

### Setup
```bash
# Clone the repository
git clone https://github.com/visual-xai-for-time-series/counterfactual-explanations-for-time-series.git
cd counterfactual-explanations-for-time-series

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
import numpy as np
from cfts.cf_wachter import wachter_genetic_cf
from cfts.cf_native_guide import native_guide_uni_cf
from cfts.metrics import l2_distance, prediction_change

# Generate counterfactual
cf, prediction = wachter_genetic_cf(sample, model, step_size=0.1)

# Evaluate quality
proximity = l2_distance(sample, cf)
validity = prediction_change(model, sample, cf, target_class=1)
```

### Advanced Example
```python
from cfts.metrics import CounterfactualEvaluator, benchmark_algorithms

# Comprehensive evaluation
evaluator = CounterfactualEvaluator()
results = benchmark_algorithms(
    algorithms=['wachter', 'native_guide', 'comte', 'sets'],
    samples=test_samples,
    model=trained_model,
    dataset=dataset
)
```

### Running All Examples
```bash
# Execute complete pipeline
python examples/run_all.py
```

## Key Features

- **Research-Ready**: Implementations of state-of-the-art algorithms
- **Comprehensive Metrics**: Six categories of evaluation measures
- **Rich Visualizations**: Publication-quality plots and comparisons
- **Easy Integration**: Simple API for custom models and datasets
- **Efficient**: Optimized implementations with GPU support
- **Well-Documented**: Extensive examples and documentation
- **Reproducible**: Seed control and deterministic results
- **Robust**: Error handling and input validation

## Project Structure

```
counterfactual-explanations-for-time-series/
├── cfts/                          # Main library
│   ├── cf_wachter/               # Wachter et al. implementation
│   ├── cf_native_guide/          # Native Guide implementation
│   ├── cf_comte/                 # COMTE implementation
│   ├── cf_sets/                  # SETS implementation
│   ├── cf_dandl/                 # MOC implementation
│   ├── cf_glacier/               # GLACIER implementation
│   ├── cf_multispace/            # Multi-SpaCE implementation
│   ├── cf_tsevo/                 # TSEvo implementation
│   └── metrics/                  # Evaluation metrics
│       ├── validity.py           # Validity metrics
│       ├── proximity.py          # Proximity metrics
│       ├── sparsity.py           # Sparsity metrics
│       ├── realism.py            # Realism metrics
│       ├── diversity.py          # Diversity metrics
│       └── stability.py          # Stability metrics
├── examples/                      # Usage examples
│   ├── example_forda.py          # FordA dataset example
│   ├── example_vibration.py      # Vibration data example
│   ├── example_arabicdigits.py   # Arabic digits example
│   ├── metrics_evaluation_example.py  # Metrics demo
│   └── run_all.py               # Execute all examples
├── models/                       # Pre-trained models
└── requirements.txt              # Dependencies
```

## License

Released under MIT License. See the LICENSE file for details.

## References and Citations

### Core Library Citation
```bibtex
@software{cfts-us-2025,
  author = {Schlegel, Udo},
  title = {Counterfactual Explanation Algorithms for Time Series Models},
  url = {https://github.com/visual-xai-for-time-series/counterfactual-explanations-for-time-series},
  version = {0.1.0},
  year = {2025}
}
```

### Related Work and Surveys

- **Guidotti, R. (2022)**. "Counterfactual explanations and how to find them: literature review and benchmarking." *Data Mining and Knowledge Discovery*, 1-55.

- **Verma, S., et al. (2020)**. "Counterfactual explanations for machine learning: A review." *arXiv preprint arXiv:2010.10596*.

- **Molnar, C. (2020)**. "Interpretable machine learning: A guide for making black box models explainable." *christophm.github.io/interpretable-ml-book/*

---

## Disclaimer

**AI-Assisted Development**: Please note that portions of this codebase have been generated or enhanced with the assistance of AI coding tools. While we have thoroughly tested and validated all implementations, users are encouraged to review the code and verify its correctness for their specific use cases.

---

## Acknowledgments

This library builds upon numerous research contributions in explainable AI and counterfactual explanations. We thank all researchers and developers who have contributed to this field, particularly the authors of the implemented algorithms.

**Special thanks to:**
- The UCR Time Series Classification Archive for providing standard datasets
- The PyTorch and scikit-learn communities for excellent tools
- All contributors and users who help improve this library

