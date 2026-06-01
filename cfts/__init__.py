"""
Counterfactual Explanation Algorithms for Time Series (cfts)

This package provides implementations of various counterfactual explanation algorithms for time series data.

Available Algorithms:
- Abstract counterfactuals (cf__abstract)
- AB-CF counterfactuals (cf_ab_cf)
- CELS counterfactuals (cf_cels)
- CEM counterfactuals (cf_cem)
- CFWOT counterfactuals (cf_cfwot)
- CGM counterfactuals (cf_cgm)
- Comte counterfactuals (cf_comte)
- Confetti counterfactuals (cf_confetti)
- CoUNTS counterfactuals (cf_counts)
- DANDL counterfactuals (cf_dandl)
- DisCOX counterfactuals (cf_discox)
- FASTPACE counterfactuals (cf_fastpace)
- FFT-CF counterfactuals (cf_fft_cf)
- GLACIER counterfactuals (cf_glacier)
- LASTS counterfactuals (cf_lasts)
- Latent-CF counterfactuals (cf_latent_cf)
- MG-CF counterfactuals (cf_mg_cf)
- MultiSpace counterfactuals (cf_multispace)
- Native-Guide counterfactuals (cf_native_guide)
- SETS counterfactuals (cf_sets)
- SG-CF counterfactuals (cf_sg_cf)
- SPARCE counterfactuals (cf_sparce)
- Subspace counterfactuals (cf_subspace)
- TERCE counterfactuals (cf_terce)
- Time-CF counterfactuals (cf_time_cf)
- TIMEX counterfactuals (cf_timex)
- TIMEX++ counterfactuals (cf_timex_plus_plus)
- TS-Tweaking counterfactuals (cf_ts_tweaking)
- TSCF counterfactuals (cf_tscf)
- TSEvo counterfactuals (cf_tsevo)
- Wachter-style counterfactuals (cf_wachter)

Additional modules:
- metrics: Evaluation metrics for counterfactual explanations
"""

from . import (
    cf__abstract,
    cf_ab_cf,
    cf_cels,
    cf_cem,
    cf_cfwot,
    cf_cgm,
    cf_comte,
    cf_confetti,
    cf_counts,
    cf_dandl,
    cf_discox,
    cf_fastpace,
    cf_fft_cf,
    cf_glacier,
    cf_lasts,
    cf_latent_cf,
    cf_mg_cf,
    cf_multispace,
    cf_native_guide,
    cf_sets,
    cf_sg_cf,
    cf_sparce,
    cf_subspace,
    cf_terce,
    cf_time_cf,
    cf_timex,
    cf_timex_plus_plus,
    cf_ts_tweaking,
    cf_tscf,
    cf_tsevo,
    cf_wachter,
    metrics,
)

__all__ = [
    "cf__abstract",
    "cf_ab_cf",
    "cf_cels",
    "cf_cem",
    "cf_cfwot",
    "cf_cgm",
    "cf_comte",
    "cf_confetti",
    "cf_counts",
    "cf_dandl",
    "cf_discox",
    "cf_fastpace",
    "cf_fft_cf",
    "cf_glacier",
    "cf_lasts",
    "cf_latent_cf",
    "cf_mg_cf",
    "cf_multispace",
    "cf_native_guide",
    "cf_sets",
    "cf_sg_cf",
    "cf_sparce",
    "cf_subspace",
    "cf_terce",
    "cf_time_cf",
    "cf_timex",
    "cf_timex_plus_plus",
    "cf_ts_tweaking",
    "cf_tscf",
    "cf_tsevo",
    "cf_wachter",
    "metrics",
]

# List of all available counterfactual algorithms
COUNTERFACTUAL_ALGORITHMS = [
    "cf__abstract",
    "cf_ab_cf",
    "cf_cels",
    "cf_cem",
    "cf_cfwot",
    "cf_cgm",
    "cf_comte",
    "cf_confetti",
    "cf_counts",
    "cf_dandl",
    "cf_discox",
    "cf_fastpace",
    "cf_fft_cf",
    "cf_glacier",
    "cf_lasts",
    "cf_latent_cf",
    "cf_mg_cf",
    "cf_multispace",
    "cf_native_guide",
    "cf_sets",
    "cf_sg_cf",
    "cf_sparce",
    "cf_subspace",
    "cf_terce",
    "cf_time_cf",
    "cf_timex",
    "cf_timex_plus_plus",
    "cf_ts_tweaking",
    "cf_tscf",
    "cf_tsevo",
    "cf_wachter",
]

__version__ = "0.1.6"
