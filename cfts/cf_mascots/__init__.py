"""
MASCOTS: Model-Agnostic Symbolic COunterfactual explanations for Time Series.

Implements the MASCOTS algorithm (Płudowski et al., arXiv:2503.22389) using a
SAX bag-of-receptive-fields surrogate and importance-guided word-swap search.

Main function:
- mascots_cf: Generate a single counterfactual explanation.
"""

from .mascots import mascots_cf

__all__ = ["mascots_cf"]
