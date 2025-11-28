"""
Original LEFTIST explanation method wrapper.

This module provides a wrapper around the TSInterpret LEFTIST implementation
for feature attribution and explanation of time series classifications.

To use this module, you need to install TSInterpret:
    pip install tsinterpret

For more information, see:
https://github.com/fzi-forschungszentrum-informatik/TSInterpret
"""

from .leftist_wrapper import leftist_explain, leftist_explain_batch

__all__ = ['leftist_explain', 'leftist_explain_batch']
