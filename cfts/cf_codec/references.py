"""
cf_codec.references - reference/donor selection strategies.

CoDec's "Reference Selection" stage (``CoDec_presentation.pdf`` slides 6 & 8).
Generalizes IMFACT's single nearest-unlike-neighbor (NUN) lookup into a
swappable strategy that returns either one whole donor series (NUN) or a
per-component donor list stitched from several candidates (Composite).

Implemented here
-----------------
- :class:`NUNReferenceSelector` - exact IMFACT behaviour: nearest unlike
  neighbor by raw Euclidean distance. The regression baseline (workplan
  Phase 2 item 1).
- :class:`CompositeReferenceSelector` (favored, workplan §4 checklist item 2)
  - pools the ``pool_k`` nearest target-class candidates, then for each of
  the query's decomposed components picks whichever pooled candidate's
  matched component minimizes the configured :class:`Matcher`'s cost. No
  single donor series - a stitched, arbitrary-per-slot NUN (slide 8).

Barycenter and typicality-anchored selectors (workplan Phase 2 items 3-4) and
the generative selector (explicitly flagged as later-stage / do-not-build)
are left as future extension points - register a new
:class:`ReferenceSelector` subclass in :data:`REFERENCE_SELECTORS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cfts.cf_codec.decompositions import Decomposer
from cfts.cf_codec.matching import Matcher


class ReferenceSelector(ABC):
    @abstractmethod
    def select(self, x: np.ndarray, candidates: np.ndarray, k: int = 1) -> np.ndarray:
        """Return the top-``k`` rows of ``candidates`` (shape ``(N, L)`` for a
        single channel), ranked by this strategy, as an array of shape
        ``(k, L)``."""
        raise NotImplementedError


class NUNReferenceSelector(ReferenceSelector):
    """Nearest-Unlike-Neighbor by raw Euclidean distance - the IMFACT baseline."""

    def select(self, x: np.ndarray, candidates: np.ndarray, k: int = 1) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        cands = np.asarray(candidates, dtype=np.float64)
        dists = np.linalg.norm(cands.reshape(len(cands), -1) - x.reshape(1, -1), axis=1)
        order = np.argsort(dists)[: min(k, len(cands))]
        return candidates[order]


class CompositeReferenceSelector(ReferenceSelector):
    """Per-component donor selection (favored - ``CoDec_presentation.pdf``
    slide 8, "Beyond a Single NUN"). ``select`` still returns the coarse
    ``pool_k``-nearest whole series (so this class satisfies the same
    :class:`ReferenceSelector` contract as :class:`NUNReferenceSelector`);
    the actual composite logic lives in :meth:`select_components`, which the
    pipeline (``codec.py``) calls once a :class:`Decomposer` is available -
    this selector needs one to score components, same as the workplan notes
    ("requires Matcher to already exist").
    """

    def __init__(self, decomposer: Decomposer, matcher: Matcher, pool_k: int = 5):
        self.decomposer = decomposer
        self.matcher = matcher
        self.pool_k = pool_k
        self._nun = NUNReferenceSelector()

    def select(self, x: np.ndarray, candidates: np.ndarray, k: int = 1) -> np.ndarray:
        return self._nun.select(x, candidates, k=k)

    def select_components(self, query_components: np.ndarray, candidate_pool: np.ndarray):
        """For each query component, search ``candidate_pool``'s (already a
        whole-series pool, e.g. from :meth:`select`) decomposed components
        for the cheapest donor under ``self.matcher``.

        Parameters
        ----------
        query_components:
            The query channel's components, shape ``(n_components, L)``.
        candidate_pool:
            Whole candidate series for this channel, shape ``(pool_k, L)``.

        Returns
        -------
        donor_components : np.ndarray, shape ``(n_components, L)``
            Per-slot donor components - may come from different pool members.
        donor_source : np.ndarray, shape ``(n_components,)``
            Index into ``candidate_pool`` that donated each component slot
            (``-1`` if no candidate provided a match for that slot).
        donor_cost : np.ndarray, shape ``(n_components,)``
            The matcher cost of the winning donor per slot (``inf`` for
            unmatched slots) - used by the pipeline to order components
            cheapest/most-similar first when widening the substituted set.
        """
        n_comp = len(query_components)
        donor_components = np.zeros_like(query_components)
        donor_source = np.full(n_comp, -1, dtype=int)
        donor_cost = np.full(n_comp, np.inf)

        for src_i, cand in enumerate(candidate_pool):
            cand_components = self.decomposer.decompose(cand)
            pairs, costs = self.matcher.match(query_components, cand_components, return_costs=True)
            for (qi, ri), cost in zip(pairs, costs):
                if ri < 0 or cost >= donor_cost[qi]:
                    continue
                donor_cost[qi] = cost
                donor_components[qi] = cand_components[ri]
                donor_source[qi] = src_i

        # Slots no candidate matched (shouldn't normally happen with a
        # non-empty pool) fall back to the query's own component, i.e. no-op.
        unmatched = donor_source < 0
        donor_components[unmatched] = query_components[unmatched]
        return donor_components, donor_source, donor_cost


REFERENCE_SELECTORS: dict[str, type[ReferenceSelector]] = {
    "nun": NUNReferenceSelector,
    "composite": CompositeReferenceSelector,
}


def make_reference_selector(name: str, **kwargs) -> ReferenceSelector:
    """Instantiate a registered :class:`ReferenceSelector` by name (see
    :data:`REFERENCE_SELECTORS`)."""
    if name not in REFERENCE_SELECTORS:
        raise ValueError(f"Unknown reference_selection strategy '{name}'. Available: {sorted(REFERENCE_SELECTORS)}")
    return REFERENCE_SELECTORS[name](**kwargs)
