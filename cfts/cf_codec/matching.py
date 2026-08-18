"""
cf_codec.matching - cross-series component alignment.

CoDec's "Matching" stage (``CoDec_presentation.pdf`` slide 8, "Module
Spotlight: Matching"). Decompositions can produce different numbers of
components on different series, and component indices aren't reliably
aligned across series - a key IMFACT reviewer criticism the workplan calls
out explicitly. A :class:`Matcher` builds a one-to-one correspondence between
a query series' components and a reference series' components.

Implemented here
-----------------
- :class:`IndexMatcher` - naive fallback, pairs ``query[i]`` with ``ref[i]``.
- :class:`HungarianMatcher` (favored, workplan §4 checklist item 1) - builds a
  cost matrix ``C[i, j]`` from a pluggable cost function and solves it with
  ``scipy.optimize.linear_sum_assignment``. Handles ``len(query) != len(ref)``
  by padding the smaller side with high-cost dummy rows/columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import linear_sum_assignment


class Matcher(ABC):
    @abstractmethod
    def match(
        self,
        query_components: np.ndarray,
        ref_components: np.ndarray,
        return_costs: bool = False,
    ):
        """Return a list of ``(query_idx, ref_idx)`` pairs - a one-to-one
        assignment covering every query component. When ``return_costs`` is
        ``True``, also return a parallel list of per-pair costs."""
        raise NotImplementedError


class IndexMatcher(Matcher):
    """Naive fallback: pairs ``query[i]`` with ``ref[i]``. Query components
    beyond ``len(ref)`` are paired with a zero (silent) reference component,
    matching how IMFACT implicitly treated a missing IMF."""

    def match(self, query_components, ref_components, return_costs: bool = False):
        n_q, n_r = len(query_components), len(ref_components)
        pairs = [(i, min(i, n_r - 1)) for i in range(n_q)] if n_r else [(i, -1) for i in range(n_q)]
        if not return_costs:
            return pairs
        costs = []
        for qi, ri in pairs:
            ref_c = ref_components[ri] if ri >= 0 else np.zeros_like(query_components[qi])
            costs.append(float(np.linalg.norm(query_components[qi] - ref_c)))
        return pairs, costs


# ---------------------------------------------------------------------------
# Cost functions for HungarianMatcher - each reduces a component (1-D array)
# to a small feature vector; cost is the Euclidean distance between features.
# ---------------------------------------------------------------------------

def _dominant_frequency(c: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(c))
    if spec.sum() < 1e-12:
        return np.array([0.0])
    freqs = np.fft.rfftfreq(len(c))
    return np.array([float(freqs[np.argmax(spec)])])


def _energy(c: np.ndarray) -> np.ndarray:
    return np.array([float(np.mean(c ** 2))])


def _spectral(c: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(c))
    total = spec.sum()
    if total < 1e-12:
        return spec * 0.0
    return spec / total


COST_FEATURES = {
    "dominant_frequency": _dominant_frequency,
    "energy": _energy,
    "spectral_similarity": _spectral,
}


class HungarianMatcher(Matcher):
    """Optimal-assignment matcher (workplan §4 checklist item 1, favored).

    Builds ``C[i, j] = ||feature(query_i) - feature(ref_j)||`` for a
    configurable ``cost_fn`` (``"dominant_frequency"``, ``"energy"``, or
    ``"spectral_similarity"``; see :data:`COST_FEATURES`, or pass any
    ``callable(component) -> np.ndarray`` directly) and solves it with
    ``scipy.optimize.linear_sum_assignment``. When the two sides have
    different component counts, the smaller side is padded with dummy rows
    or columns at a fixed high cost so every query component still gets an
    assignment.
    """

    def __init__(self, cost_fn: str | callable = "dominant_frequency", dummy_cost: float = 1e6):
        self.cost_fn = COST_FEATURES[cost_fn] if isinstance(cost_fn, str) else cost_fn
        self.dummy_cost = dummy_cost

    def _cost_matrix(self, query_components, ref_components) -> np.ndarray:
        # Feature vectors can vary in length across components with
        # different lengths (rare, but Fourier components across mismatched
        # series lengths could differ); pad to a common width.
        q_feats = [np.atleast_1d(self.cost_fn(c)) for c in query_components]
        r_feats = [np.atleast_1d(self.cost_fn(c)) for c in ref_components]
        width = max((f.shape[0] for f in q_feats + r_feats), default=1)
        q_feats = [np.pad(f, (0, width - f.shape[0])) for f in q_feats]
        r_feats = [np.pad(f, (0, width - f.shape[0])) for f in r_feats]
        C = np.zeros((len(q_feats), len(r_feats)))
        for i, qf in enumerate(q_feats):
            for j, rf in enumerate(r_feats):
                C[i, j] = np.linalg.norm(qf - rf)
        return C

    def match(self, query_components, ref_components, return_costs: bool = False):
        n_q, n_r = len(query_components), len(ref_components)
        if n_q == 0 or n_r == 0:
            pairs = [(i, -1) for i in range(n_q)]
            return (pairs, [self.dummy_cost] * n_q) if return_costs else pairs

        C = self._cost_matrix(query_components, ref_components)
        n = max(n_q, n_r)
        C_padded = np.full((n, n), self.dummy_cost)
        C_padded[:n_q, :n_r] = C
        row_idx, col_idx = linear_sum_assignment(C_padded)

        pairs, costs = [], []
        for r, c in sorted(zip(row_idx, col_idx)):
            if r >= n_q:
                continue  # dummy row: no query component to assign
            ri = c if c < n_r else -1  # dummy column: ref has no match either
            pairs.append((int(r), int(ri)))
            costs.append(float(C_padded[r, c]))
        return (pairs, costs) if return_costs else pairs


MATCHERS: dict[str, type[Matcher]] = {
    "index": IndexMatcher,
    "hungarian": HungarianMatcher,
}


def make_matcher(name: str, **kwargs) -> Matcher:
    """Instantiate a registered :class:`Matcher` by name (see :data:`MATCHERS`)."""
    if name not in MATCHERS:
        raise ValueError(f"Unknown matching method '{name}'. Available: {sorted(MATCHERS)}")
    return MATCHERS[name](**kwargs)
