"""
TSEvo: Evolutionary Counterfactual Explanations for Time Series

This module implements the TSEvo algorithm for generating counterfactual explanations
using multi-objective evolutionary optimization (NSGA-II).
"""

from .tsevo import tsevo_cf

__all__ = ['tsevo_cf']
