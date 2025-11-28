"""
TSCF: Time Series CounterFactuals

This module implements the TSCF algorithm for generating counterfactual explanations
using gradient-based optimization with temporal smoothness constraints.
"""

from .tscf import tscf_cf, tscf_batch_cf

__all__ = ['tscf_cf', 'tscf_batch_cf']
