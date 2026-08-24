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
# Option 1: Install the released package from PyPI
# https://pypi.org/project/counterfactuals-for-time-series/
pip install counterfactuals-for-time-series

# Option 2: Install dependencies to run this repo's examples from source
# (the examples/, pre-trained models, and datasets ship only with the source checkout)
pip install -r requirements.txt

# Run all examples and evaluations
cd examples
python run_all.py

# Run individual examples
python example_univariate.py                  # FordA dataset
python example_univariate_ecg.py              # ECG200 dataset
python example_univariate_faultdetectiona.py  # FaultDetectionA dataset
python example_multivariate.py                # Multi-channel Arabic digits
python example_metrics_evaluation.py          # Comprehensive metrics
```

## Implemented Algorithms

This library implements **35 state-of-the-art counterfactual explanation methods** for time series classification, organized into:

- **Optimization-Based Methods**: Wachter, COMTE, TSCF, TS-Tweaking, FFT-CF, TopGrad-CF
- **Evolutionary Methods**: MOC/DANDL, TSEvo, Multi-SpaCE, Sub-SpaCE, CONFETTI, FastPACE
- **Instance-Based Methods**: Native Guide, CELS/M-CELS, AB-CF, IMFACT
- **Latent Space Methods**: CGM, CounTS, Latent-CF, LASTS, GLACIER
- **Segment-Based Methods**: SETS, SG-CF, DisCOX, CFWoT, TS-CEM, MASCOTS
- **Hybrid Methods**: SPARCE, Time-CF, TeRCE, MG-CF, TimeX, TimeX++, CFE4MTS, DiffCF

📚 **For detailed descriptions, key features, academic references, and code examples for each method, see [REFERENCES.md](REFERENCES.md)**

> **In progress:** CoDec generalizes IMFACT's decomposition/reference/perturbation pipeline into independently swappable strategies and is being validated ahead of a September 2026 planning meeting. It has no standalone implementation yet — see the CoDec entry in [REFERENCES.md](REFERENCES.md) for the design docs.

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
- **`example_univariate.py`**: FordA automotive fault detection dataset (UCR Archive)
- **`example_univariate_ecg.py`**: ECG200 electrocardiogram dataset (UCR Archive)
- **`example_univariate_faultdetectiona.py`**: FaultDetectionA electromechanical drive dataset (3-class)
- **`example_multivariate.py`**: Multi-channel spoken Arabic digits (13 channels)
- **`example_metrics_evaluation.py`**: Comprehensive metrics evaluation across 53 algorithm variants — the base set plus additional variants of several methods (8 GLACIER bake-off variants, InfoCELS, NG-DBA, COMTE-Distractor, COMTE-Advanced-Gradient, Multi-SpaCE-Canonical, SPARCE-GAN, CGM-Simple) — including TimeX and TimeX++ (their own counterfactual reinterpretations — trained inline, per call, like every other trained method here; see `example_metrics_evaluation.py`'s module docstring for why, since both names collide with unrelated saliency/attribution papers that need external pre-trained-explainer infrastructure this repository doesn't provide). The combined metrics PNG is stamped with the dataset, model, and sample counts used.

### Supported Datasets
- **UCR Time Series Archive**: Automatic download and preprocessing
- **Synthetic Data**: Built-in generators for controlled experiments
- **Custom Datasets**: Easy integration with custom time series data

### Pre-trained Models
- `simple_cnn_forda_2.pth`: Binary classification (FordA)
- `simple_cnn_ecg200_2.pth`: Binary classification (ECG200)
- `simple_cnn_faultdetectiona_3.pth`: 3-class classification (FaultDetectionA)
- `cnn_multi_arabicdigits_10ch.pth`: 10-class multi-channel (Arabic Digits)

## Visualization Examples

The library generates publication-ready visualizations:

![FordA Counterfactual Explanations](counterfactuals_forda.png)
*Individual counterfactuals and overlay comparisons for FordA dataset*

![ECG200 Counterfactual Explanations](counterfactuals_ecg200.png)
*Individual counterfactuals and overlay comparisons for the ECG200 dataset*

![FaultDetectionA Counterfactual Explanations](counterfactuals_faultdetectiona.png)
*Individual counterfactuals and overlay comparisons for the 3-class FaultDetectionA dataset, including failed/skipped algorithms*

![Arabic Digits Counterfactual Explanations](counterfactuals_arabic_digits.png)
*Multi-channel counterfactual analysis for Arabic digits*

![Combined Metrics Comparison](metrics_combined.png)
*Top-10 ranked algorithms across the full metrics suite, Keane et al. (2021) metrics (Validity, Proximity, Compactness), and the `evaluate.py` metric suite, generated by `example_metrics_evaluation.py`. The complete, unranked results for every algorithm are written to [`metrics_full_results.csv`](metrics_full_results.csv).*

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib
- Scikit-learn
- Captum (for attribution methods)
- Optional: dtaidistance (for DTW distance metric)

### Setup

**Option 1 — Install the released package from [PyPI](https://pypi.org/project/counterfactuals-for-time-series/):**
```bash
pip install counterfactuals-for-time-series
```
This installs the `cfts` package (`import cfts`) with its core dependencies. The `examples/` scripts, pre-trained models, and sample datasets are not part of the PyPI distribution — use Option 2 for those.

**Option 2 — Install from source (needed to run the bundled examples):**
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
│   ├── cf__abstract/             # Shared abstract base class & CFMethod contract
│   ├── cf_ab_cf/                 # AB-CF implementation
│   ├── cf_cels/                  # CELS implementation
│   ├── cf_cem/                   # TS-CEM implementation
│   ├── cf_cfe4mts/               # CFE4MTS implementation
│   ├── cf_cfwot/                 # CFWoT implementation
│   ├── cf_cgm/                   # CGM implementation
│   ├── cf_comte/                 # COMTE implementation
│   ├── cf_confetti/              # CONFETTI implementation
│   ├── cf_counts/                # CoUNTS implementation
│   ├── cf_dandl/                 # MOC implementation
│   ├── cf_discox/                # DisCOX implementation
│   ├── cf_fastpace/              # FastPACE implementation
│   ├── cf_fft_cf/                # FFT-CF implementation
│   ├── cf_glacier/               # GLACIER implementation
│   ├── cf_imfact/                # IMFACT implementation
│   ├── cf_lasts/                 # LASTS implementation
│   ├── cf_latent_cf/             # Latent CF implementation
│   ├── cf_mascots/               # MASCOTS implementation
│   ├── cf_mg_cf/                 # MG-CF implementation
│   ├── cf_multispace/            # Multi-SpaCE implementation
│   ├── cf_native_guide/          # Native Guide implementation
│   ├── cf_sets/                  # SETS implementation
│   ├── cf_sg_cf/                 # SG-CF implementation
│   ├── cf_sparce/                # SpArCE implementation
│   ├── cf_subspace/              # Sub-SpaCE implementation
│   ├── cf_terce/                 # TERCE implementation
│   ├── cf_time_cf/               # Time-CF implementation
│   ├── cf_timex/                 # TimeX implementation
│   ├── cf_timex_plus_plus/       # TimeX++ implementation
│   ├── cf_topgrad/               # TopGrad-CF implementation
│   ├── cf_ts_tweaking/           # TS-Tweaking implementation
│   ├── cf_tscf/                  # TSCF implementation
│   ├── cf_tsevo/                 # TSEvo implementation
│   ├── cf_wachter/               # Wachter et al. implementation
│   └── metrics/                  # Evaluation metrics
│       ├── validity.py           # Validity metrics
│       ├── proximity.py          # Proximity metrics
│       ├── sparsity.py           # Sparsity metrics
│       ├── realism.py            # Realism metrics
│       ├── diversity.py          # Diversity metrics
│       └── stability.py          # Stability metrics
├── examples/                      # Usage examples
│   ├── example_univariate.py                  # FordA dataset example
│   ├── example_univariate_ecg.py              # ECG200 dataset example
│   ├── example_univariate_faultdetectiona.py  # FaultDetectionA example
│   ├── example_multivariate.py                # Arabic digits example
│   ├── example_metrics_evaluation.py          # Comprehensive metrics demo
│   └── run_all.py                             # Execute all examples
├── models/                       # Pre-trained models
└── requirements.txt              # Dependencies
```

## License

Released under MIT License. See the LICENSE file for details.

## References and Citations

If you use the library, please cite one of the following sources (the second one is preferred).

### Core Library Citation
```bibtex
@software{cfts-us-2025,
  author = {Schlegel, Udo},
  title = {Counterfactual Explanation Algorithms for Time Series Models},
  url = {https://github.com/visual-xai-for-time-series/counterfactual-explanations-for-time-series},
  year = {2025}
}
```

### Reference Paper Citation
```bibtex
@inproceedings{schlegel_what-if_2026,
  title={What-If Explanations Over Time: Counterfactuals for Time Series Classification},
  author={Schlegel, Udo and Seidl, Thomas},
  booktitle={World Conference on Explainable Artificial Intelligence (XAI)},
  year={2026}
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
