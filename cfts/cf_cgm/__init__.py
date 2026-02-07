"""
CGM: Conditional Generative Models for Counterfactual Explanations

This module implements the Conditional Generative Models (CGM) approach
for generating counterfactual explanations for time series data.

Paper: Van Looveren, A., Klaise, J., Vacanti, G., & Cobb, O. (2021).
       "Conditional Generative Models for Counterfactual Explanations"
       arXiv:2101.10123

Main functions:
- cgm_generate: Generate counterfactuals using Conditional VAE
- cgm_generate_simple: Simple variant without VAE
- ConditionalVAE: Conditional Variational Autoencoder model
- train_conditional_vae: Train the Conditional VAE
"""

from .cgm import (
    cgm_generate,
    cgm_generate_simple,
    ConditionalVAE,
    ConditionalEncoder,
    ConditionalDecoder,
    train_conditional_vae,
    detach_to_numpy,
    numpy_to_torch
)

__all__ = [
    'cgm_generate',
    'cgm_generate_simple',
    'ConditionalVAE',
    'ConditionalEncoder',
    'ConditionalDecoder',
    'train_conditional_vae',
    'detach_to_numpy',
    'numpy_to_torch'
]
