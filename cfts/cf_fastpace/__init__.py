"""
FastPACE: Fast PlAnning of Counterfactual Explanations

This module implements the FastPACE algorithm for generating counterfactual explanations
using a planning-based approach with feasible interventions and plausibility constraints.
"""

from .fastpace import fastpace_cf, fastpace_batch_cf

__all__ = ['fastpace_cf', 'fastpace_batch_cf']
