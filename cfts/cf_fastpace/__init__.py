"""
FastPACE: Fast PlAnning of Counterfactual Explanations for Time Series Classification

Paper: Refoyo, M., Boleas, Y., & Luengo, D. (2026).
       "FastPACE: Fast PlAnning of Counterfactual Explanations for Time Series
       Classification." Data Mining and Knowledge Discovery.
       https://doi.org/10.1007/s10618-026-01242-7

This module casts counterfactual generation as an episodic Markov Decision Process
over NUN-replacement masks and solves it with hierarchical, block-based Cross-Entropy
Method planning, guaranteeing validity by design.
"""

from .fastpace import fastpace_cf, fastpace_batch_cf, train_plausibility_autoencoder

__all__ = ['fastpace_cf', 'fastpace_batch_cf', 'train_plausibility_autoencoder']
