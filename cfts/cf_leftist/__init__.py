"""
LEFTIST: Learning-Free Saliency-Based Counterfactual Explanations for Time Series

This module implements the LEFTIST algorithm for generating counterfactual explanations
using gradient-based saliency to identify important time segments that should be modified.
"""

from .leftist import leftist_cf, leftist_multi_cf

__all__ = ['leftist_cf', 'leftist_multi_cf']
