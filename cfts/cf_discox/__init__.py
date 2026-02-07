"""
DisCOX: Discord-based Counterfactual Explanations for Time Series

This module implements the DisCOX method for generating counterfactual explanations.
DisCOX identifies and modifies the most discordant (unusual) subsequences in time series
to create meaningful counterfactuals.

Main functions:
- discox_cf: Generate a single counterfactual explanation
- discox_explain: Generate counterfactual with detailed explanation
"""

from .discox import discox_cf, discox_explain

__all__ = ["discox_cf", "discox_explain"]
