"""
Counterfactual Explanation Algorithms for Time Series (cfts)

This package provides implementations such as:
- Wachter-style counterfactuals (cf_wachter)
- Native-Guide counterfactuals (cf_native_guide)
- Comte counterfactuals (cf_comte)
- DANDL counterfactuals (cf_dandl)
- SETS counterfactuals (cf_sets)
Utility helpers under `utils`.
"""

from . import cf_wachter, cf_native_guide, cf_comte, cf_dandl, cf_sets

__all__ = ["cf_wachter", "cf_native_guide", "cf_comte", "cf_dandl", "cf_sets"]

__version__ = "0.1.0"
