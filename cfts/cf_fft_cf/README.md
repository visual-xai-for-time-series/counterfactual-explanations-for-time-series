# FFT-CF: Fourier Transform-Based Counterfactual Explanations

This module implements 9 variants of FFT-based counterfactual generation for time series classification.

## Quick Start

```python
from cfts.cf_fft_cf import fft_nn_cf

# Generate counterfactual
cf, pred = fft_nn_cf(
    sample=your_time_series,
    dataset=your_dataset,
    model=your_model,
    target_class=1,
    k=5
)
```

## Available Variants

| Variant | Speed | Quality | Best For |
|---------|-------|---------|----------|
| `fft_freq_distance_cf` | ⚡⚡⚡ | ⭐⭐⭐ | Fastest option (0.011s) |
| `fft_nn_cf` | ⚡⚡ | ⭐⭐⭐ | Baseline, reliable |
| `fft_wavelet_cf` | ⚡⚡ | ⭐⭐⭐ | Multi-resolution |
| `fft_adaptive_cf` | ⚡⚡ | ⭐⭐⭐ | Interpretability |
| `fft_smart_blend_cf` | ⚡⚡ | ⭐⭐⭐⭐ | Optimal blending |
| `fft_iterative_cf` | ⚡ | ⭐⭐⭐⭐⭐ | Best quality (15.94 dist) |
| `fft_hybrid_cf` | ⚡ | ⭐⭐⭐ | Adaptive amp/phase |
| `fft_cf` | ⚡ | ⭐⭐ | Original greedy |
| `fft_gradient_cf` | ⚡ | ⭐⭐ | Pure optimization |

## Test Results

Run comprehensive comparison:
```bash
cd examples
python fft_cf_variants_comparison.py
```

**Performance on FordA dataset:**
- Success Rate: 8/9 variants (88.9%)
- Average Time: 0.117s
- Average Distance: 16.46
- Average Confidence: 0.5559

## Examples

### Fast and Reliable
```python
from cfts.cf_fft_cf import fft_freq_distance_cf

cf, pred = fft_freq_distance_cf(
    sample, dataset, model, target_class=1,
    k=5, freq_weight_strategy="energy"
)
# Fastest: 0.011s, dist=16.11
```

### Best Quality
```python
from cfts.cf_fft_cf import fft_iterative_cf

cf, pred = fft_iterative_cf(
    sample, dataset, model, target_class=1,
    k=5, refine_iterations=50
)
# Best distance: 15.94, time=0.672s
```

### Multi-Resolution
```python
from cfts.cf_fft_cf import fft_wavelet_cf

cf, pred = fft_wavelet_cf(
    sample, dataset, model, target_class=1,
    k=5, wavelet="db4", level=3
)
# Requires: pip install PyWavelets
```

### Interpretable
```python
from cfts.cf_fft_cf import fft_adaptive_cf

cf, pred = fft_adaptive_cf(
    sample, dataset, model, target_class=1,
    k=5, use_saliency=True
)
# Shows which frequencies matter most
```

## Testing

Quick integration test:
```bash
cd examples
python test_fft_cf_integration.py
```

Full evaluation suite:
```bash
python example_univariate.py       # Univariate test
python example_multivariate.py     # Multivariate test
python metrics_evaluation_example.py  # Benchmark
```

## Documentation

See [FFT_CF_VARIANTS_SUMMARY.md](../FFT_CF_VARIANTS_SUMMARY.md) for:
- Detailed algorithm descriptions
- Performance analysis
- Usage recommendations
- Implementation details

## Dependencies

**Required:**
- numpy
- torch
- pytorch (for model inference)

**Optional:**
- PyWavelets (for `fft_wavelet_cf`)

## Reference

```bibtex
@inproceedings{delaney2021instance,
  title={Instance-Based Counterfactual Explanations for Time Series Classification},
  author={Delaney, Eoin and Greene, Derek and Keane, Mark T},
  booktitle={International Conference on Case-Based Reasoning},
  pages={32--47},
  year={2021},
  organization={Springer}
}
```

## Integration Status

✅ Integrated into:
- `examples/example_univariate.py`
- `examples/example_multivariate.py`
- `examples/metrics_evaluation_example.py`

✅ Benchmarked:
- FordA (univariate): 100% success
- SpokenArabicDigits (multivariate): 100% success
- Keane et al. (2021) metrics: Ranked 8/20 (score 0.852)

## Recommendations

| Use Case | Recommended Variant | Why |
|----------|-------------------|-----|
| Production (speed) | `fft_freq_distance_cf` | 0.011s, reliable |
| Production (quality) | `fft_iterative_cf` | Best distance |
| Research | `fft_wavelet_cf` | Novel, theoretically interesting |
| Interpretability | `fft_adaptive_cf` | Shows frequency importance |
| Prototyping | `fft_nn_cf` | Simple, well-tested baseline |

## Known Issues

- `fft_gradient_cf`: May not converge for some samples
- `fft_hybrid_cf`: Occasional failures (87.5% success rate)
- `fft_wavelet_cf`: Requires PyWavelets package

## Performance Tips

1. **Speed**: Use `fft_freq_distance_cf` with `k=3`
2. **Quality**: Use `fft_iterative_cf` with `refine_iterations=100`
3. **Balance**: Use `fft_nn_cf` with `k=5` (default)
4. **Memory**: Reduce `k` if dataset is very large

## Files

```
cfts/cf_fft_cf/
├── __init__.py              # Exports all 9 variants
├── fft_cf.py                # Implementation (1873 lines)
└── README.md                # This file

examples/
├── fft_cf_variants_comparison.py   # Comprehensive comparison
├── test_fft_cf_integration.py      # Quick integration test
└── example_univariate.py           # Basic usage example
```

## Metrics

Evaluated using Keane et al. (2021) framework:
- **Validity**: 1.0 (100% valid counterfactuals)
- **Proximity**: 11.44 (L2 distance to original)
- **Compactness**: 21.16% (percentage of time series modified)
- **Overall Score**: 0.852 (ranked 8/20 algorithms)

## License

See [LICENSE](../../LICENSE) in repository root.
