"""
Counterfactual Explanation Algorithms for Time Series (cfts)

This package provides implementations of various counterfactual explanation algorithms for time series data.

Available Algorithms:
- Wachter-style counterfactuals (cf_wachter)
- Native-Guide counterfactuals (cf_native_guide)
- Comte counterfactuals (cf_comte)
- DANDL counterfactuals (cf_dandl)
- SETS counterfactuals (cf_sets)
- GLACIER counterfactuals (cf_glacier)
- MultiSpace counterfactuals (cf_multispace)
- TSEvo counterfactuals (cf_tsevo)
- LASTS counterfactuals (cf_lasts)
- TSCF counterfactuals (cf_tscf)

Additional modules:
- metrics: Evaluation metrics for counterfactual explanations
"""

from . import (
    cf_wachter,
    cf_native_guide,
    cf_comte,
    cf_dandl,
    cf_sets,
    cf_glacier,
    cf_multispace,
    cf_tsevo,
    cf_lasts,
    cf_tscf,
    metrics,
)

__all__ = [
    "cf_wachter",
    "cf_native_guide",
    "cf_comte",
    "cf_dandl",
    "cf_sets",
    "cf_glacier",
    "cf_multispace",
    "cf_tsevo",
    "cf_lasts",
    "cf_tscf",
    "metrics",
]

# List of all available counterfactual algorithms
COUNTERFACTUAL_ALGORITHMS = [
    "cf_wachter",
    "cf_native_guide",
    "cf_comte",
    "cf_dandl",
    "cf_sets",
    "cf_glacier",
    "cf_multispace",
    "cf_tsevo",
    "cf_lasts",
    "cf_tscf",
]

__version__ = "0.1.2"
