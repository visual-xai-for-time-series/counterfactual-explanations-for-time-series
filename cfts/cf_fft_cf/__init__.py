"""
FFT-CF: Frequency-based Counterfactual Explanations for Time Series

This module provides counterfactual explanation generation using Fast Fourier Transform (FFT)
to manipulate time series in the frequency domain.

Available variants:
- fft_cf: Original greedy search approach
- fft_gradient_cf: Gradient-based optimization in frequency domain
- fft_nn_cf: Nearest neighbor blending (recommended default)
- fft_adaptive_cf: Adaptive frequency band selection with saliency
- fft_iterative_cf: Iterative refinement with local optimization
- fft_smart_blend_cf: Smart blend ratio selection (binary/golden search)
- fft_freq_distance_cf: Frequency-domain distance for neighbor selection
- fft_wavelet_cf: Wavelet transform for multi-resolution analysis
- fft_hybrid_cf: Hybrid amplitude-phase modification strategy
- fft_progressive_cf: Progressive neighbor switching optimization
- fft_confidence_threshold_cf: Confidence-threshold early stopping for sparsity
- fft_hybrid_enhanced_cf: Enhanced hybrid with NN fallback mechanism
- fft_band_optimizer_cf: Multi-band frequency optimization with saliency
"""

from .fft_cf import (
    fft_cf, 
    fft_gradient_cf, 
    fft_nn_cf,
    fft_adaptive_cf,
    fft_iterative_cf,
    fft_smart_blend_cf,
    fft_freq_distance_cf,
    fft_wavelet_cf,
    fft_hybrid_cf,
    fft_progressive_cf,
    fft_confidence_threshold_cf,
    fft_hybrid_enhanced_cf,
    fft_band_optimizer_cf
)

__all__ = [
    'fft_cf', 
    'fft_gradient_cf', 
    'fft_nn_cf',
    'fft_adaptive_cf',
    'fft_iterative_cf',
    'fft_smart_blend_cf',
    'fft_freq_distance_cf',
    'fft_wavelet_cf',
    'fft_hybrid_cf',
    'fft_progressive_cf',
    'fft_confidence_threshold_cf',
    'fft_hybrid_enhanced_cf',
    'fft_band_optimizer_cf'
]
