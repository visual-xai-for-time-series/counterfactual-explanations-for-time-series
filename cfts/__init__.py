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
- LEFTIST counterfactuals (cf_leftist)

Additional modules:
- metrics: Evaluation metrics for counterfactual explanations
- cf_leftist_original: Wrapper for original LEFTIST explanation method (requires tsinterpret)
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
    cf_leftist,
    metrics,
)

# Optional: original LEFTIST explanation method (requires tsinterpret)
try:
    from . import cf_leftist_original
    _has_leftist_original = True
except ImportError:
    _has_leftist_original = False

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
    "cf_leftist",
    "metrics",
]

if _has_leftist_original:
    __all__.append("cf_leftist_original")

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
    "cf_leftist",
]

__version__ = "0.1.1"
